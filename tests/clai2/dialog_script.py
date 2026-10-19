"""Scripted `Dialogs`, so settings-menu actions and the helpers they call run without a terminal or a browser."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic_clai2.plugins.menus import Option

Kind = Literal['choose', 'text', 'confirm', 'wait', 'view']


@dataclass(frozen=True, kw_only=True)
class Pick:
    """Answer `choose` with the option whose value is `value`."""

    value: str


@dataclass(frozen=True, kw_only=True)
class Type:
    """Answer `text` with `text`, which the dialog's `check` must accept, as the terminal only accepts such text."""

    text: str


@dataclass(frozen=True, kw_only=True)
class Confirm:
    """Answer `confirm` with `yes` or `no`."""

    yes: bool


@dataclass(frozen=True, kw_only=True)
class Finish:
    """Let `wait`'s task run until it returns or raises."""


@dataclass(frozen=True, kw_only=True)
class Back:
    """Back out, as Esc does: `choose` and `text` return `None`, `confirm` `False`.

    `wait` runs its task until it first shows something, or ends, then cancels it.
    """


Answer = Pick | Type | Confirm | Finish | Back

_KINDS: dict[type[Answer], Kind] = {Pick: 'choose', Type: 'text', Confirm: 'confirm', Finish: 'wait'}


@dataclass(frozen=True, kw_only=True)
class Asked:
    """One dialog that was shown, with what it offered. Fields a kind of dialog does not have keep their defaults."""

    kind: Kind
    title: str
    description: str = ''
    options: tuple[Option, ...] = ()
    current: str | None = None
    label: str = ''
    placeholder: str = ''
    initial: str = ''
    secret: bool = False
    multiline: bool = False
    yes: str = ''
    no: str = ''
    shown: list[str] = field(default_factory=list[str])
    """Each text `wait`'s task passed to `show`, in order."""
    text: str = ''
    """What `view` showed."""


class ScriptedDialogs:
    """`Dialogs` that give scripted answers in order and record each dialog in `asked`.

    Each answer must suit the dialog it meets: `Pick` a `choose`, `Type` a `text`, `Confirm` a `confirm`,
    `Finish` a `wait`, and `Back` any of them. A mismatch, an option that was not offered, text the dialog's
    `check` rejects, or running out of answers fails the test with `AssertionError`. `view` asks nothing, so it
    takes no answer: it is recorded and closes at once.

        dialogs = ScriptedDialogs([Pick(value='enter'), Type(text='secret'), Confirm(yes=True)])
        await choose_key(name='DEMO', label='Demo token', dialogs=dialogs)
        assert [asked.kind for asked in dialogs.asked] == ['choose', 'text', 'confirm']
    """

    def __init__(self, answers: Sequence[Answer]) -> None:
        self._answers = list(answers)
        self.asked: list[Asked] = []

    @property
    def unused(self) -> list[Answer]:
        """The answers no dialog has taken yet."""
        return list(self._answers)

    def _next(self, asked: Asked) -> Answer:
        self.asked.append(asked)
        assert self._answers, f'no answer left for {asked.kind} {asked.title!r}'
        answer = self._answers.pop(0)
        assert isinstance(answer, Back) or _KINDS[type(answer)] == asked.kind, (
            f'{answer!r} cannot answer {asked.kind} {asked.title!r}'
        )
        return answer

    async def choose(
        self, *, title: str, options: Sequence[Option], current: str | None = None, description: str = ''
    ) -> str | None:
        answer = self._next(
            Asked(kind='choose', title=title, options=tuple(options), current=current, description=description)
        )
        if not isinstance(answer, Pick):
            return None
        assert answer.value in [option.value for option in options], f'{answer.value!r} is not offered in {title!r}'
        return answer.value

    async def text(
        self,
        *,
        title: str,
        label: str = 'Value',
        placeholder: str = '',
        initial: str = '',
        secret: bool = False,
        check: Callable[[str], str | None] | None = None,
        description: str = '',
        multiline: bool = False,
    ) -> str | None:
        answer = self._next(
            Asked(
                kind='text',
                title=title,
                description=description,
                label=label,
                placeholder=placeholder,
                initial=initial,
                secret=secret,
                multiline=multiline,
            )
        )
        if not isinstance(answer, Type):
            return None
        text = answer.text.strip()
        problem = check(text) if check is not None and text else None
        assert problem is None, f'{title!r} would not accept {text!r}: {problem}'
        return text

    async def view(self, *, title: str, text: str) -> None:
        self.asked.append(Asked(kind='view', title=title, text=text))

    async def confirm(self, *, title: str, yes: str, no: str, description: str = '') -> bool:
        answer = self._next(Asked(kind='confirm', title=title, yes=yes, no=no, description=description))
        return isinstance(answer, Confirm) and answer.yes

    async def wait(
        self, *, title: str, description: str, task: Callable[[Callable[[str], None]], Awaitable[object]]
    ) -> bool:
        asked = Asked(kind='wait', title=title, description=description)
        answer = self._next(asked)
        showed = asyncio.Event()

        def show(text: str) -> None:
            asked.shown.append(text)
            showed.set()

        if isinstance(answer, Finish):
            await task(show)
            return True
        running = asyncio.ensure_future(task(show))
        backing_out = asyncio.ensure_future(showed.wait())
        try:
            await asyncio.wait({running, backing_out}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            running.cancel()
            backing_out.cancel()
            await asyncio.gather(running, backing_out, return_exceptions=True)
        if running.cancelled():
            return False
        running.result()
        return True
