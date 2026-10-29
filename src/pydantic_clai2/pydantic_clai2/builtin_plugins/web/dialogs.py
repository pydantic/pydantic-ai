"""`Dialogs` in the browser: each one a message to every connected browser, answered with `POST /api/dialogs/ID`."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from itertools import count

from pydantic_clai2.builtin_plugins.web.protocol import (
    Choice,
    Dialog,
    DialogClosed,
    DialogOpened,
    DialogShown,
    ServerMessage,
)
from pydantic_clai2.plugins.menus import Option


@dataclass(kw_only=True)
class _Open:
    dialog: Dialog
    answered: asyncio.Future[str | None]
    check: Callable[[str], str | None] | None = None


class BrowserDialogs:
    """`Dialogs` shown in the browser, for a settings menu opened there.

    A dialog stays open until a browser answers or cancels it, so one that connects later, or
    reloads, still sees it. What a browser types into a secret dialog goes to the asker alone.
    """

    def __init__(self, *, publish: Callable[[ServerMessage], None]) -> None:
        self._publish = publish
        self._open: dict[str, _Open] = {}
        self._ids = count(1)

    @property
    def dialogs(self) -> list[Dialog]:
        """The open dialogs, oldest first."""
        return [entry.dialog for entry in self._open.values()]

    def answer(self, dialog_id: str, value: str | None) -> None:
        """Answer an open dialog, or cancel it with `None`.

        Raises `LookupError` when it is not open, and `ValueError` for an answer it cannot take: a
        value `choose` does not offer, or text `check` rejects, which leaves the dialog open.
        """
        entry = self._open.get(dialog_id)
        if entry is None or entry.answered.done():
            raise LookupError('That dialog is no longer open.')
        dialog = entry.dialog
        if value is not None:
            if dialog.kind == 'choose' and value not in {option.value for option in dialog.options}:
                raise ValueError(f'{value!r} is not one of the choices.')
            if dialog.kind == 'text':
                value = value.strip()
                if value and entry.check is not None and (problem := entry.check(value)) is not None:
                    raise ValueError(problem)
        entry.answered.set_result(value)

    def cancel_all(self) -> None:
        """Back out of every open dialog, as when the web UI stops."""
        for entry in self._open.values():
            if not entry.answered.done():
                entry.answered.set_result(None)

    async def _ask(self, dialog: Dialog, *, check: Callable[[str], str | None] | None = None) -> str | None:
        entry = _Open(dialog=dialog, answered=asyncio.get_running_loop().create_future(), check=check)
        self._open[dialog.id] = entry
        self._publish(DialogOpened(dialog=dialog))
        try:
            return await entry.answered
        finally:
            self._close(dialog.id)

    def _close(self, dialog_id: str) -> None:
        del self._open[dialog_id]
        self._publish(DialogClosed(id=dialog_id))

    def _id(self) -> str:
        return f'dialog{next(self._ids)}'

    async def choose(
        self, *, title: str, options: Sequence[Option], current: str | None = None, description: str = ''
    ) -> str | None:
        choices = [Choice(value=option.value, label=option.label, description=option.description) for option in options]
        dialog = Dialog(
            id=self._id(), kind='choose', title=title, description=description, options=choices, current=current
        )
        return await self._ask(dialog)

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
        secret = secret and not multiline
        dialog = Dialog(
            id=self._id(),
            kind='text',
            title=title,
            description=description,
            label=label,
            placeholder=placeholder,
            initial='' if secret else initial,
            secret=secret,
            multiline=multiline,
        )
        return await self._ask(dialog, check=check)

    async def view(self, *, title: str, text: str) -> None:
        """Open until a browser answers it, whatever the answer."""
        await self._ask(Dialog(id=self._id(), kind='view', title=title, text=text))

    async def confirm(self, *, title: str, yes: str, no: str, description: str = '') -> bool:
        dialog = Dialog(id=self._id(), kind='confirm', title=title, description=description, yes=yes, no=no)
        return await self._ask(dialog) == 'yes'

    async def wait(
        self, *, title: str, description: str, task: Callable[[Callable[[str], None]], Awaitable[object]]
    ) -> bool:
        """Any answer cancels the task; the dialog closes by itself when the task ends."""
        dialog = Dialog(id=self._id(), kind='wait', title=title, description=description)
        entry = _Open(dialog=dialog, answered=asyncio.get_running_loop().create_future())

        def show(text: str) -> None:
            entry.dialog = replace(entry.dialog, shown=text)
            self._publish(DialogShown(id=dialog.id, text=text))

        self._open[dialog.id] = entry
        self._publish(DialogOpened(dialog=dialog))
        running = asyncio.ensure_future(task(show))
        try:
            await asyncio.wait({running, entry.answered}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)
            self._close(dialog.id)
        if running.cancelled():
            return False
        running.result()
        return True
