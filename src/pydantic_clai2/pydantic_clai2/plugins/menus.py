"""A plugin's settings menu, declared once and shown by the terminal or by the web UI.

`Plugin.settings_menu` returns a `SettingsMenu`: a `FieldSource` whose rows each edit one value,
and `actions` for rows that need more than a value, such as picking a `/keys` entry or signing in.
An action asks the user through `Dialogs`, which the terminal shows as full-screen widgets and the
web UI as dialogs in the page, so one declaration serves both.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol


class ChoicePreview(Protocol):
    """Render a sample of `choice` for the terminal choice picker's right-hand panel, `width` cells wide."""

    def __call__(self, choice: str, /, *, width: int) -> str: ...


@dataclass(frozen=True, kw_only=True)
class FieldRow:
    """One editable field as the menu sees it."""

    key: str
    description: str
    default: str
    choices: tuple[str, ...] = ()
    label: str = ''
    choice_labels: Mapping[str, str] = field(default_factory=dict[str, str])
    allow_custom: bool = True
    secret: bool = False
    note: str = ''
    """Where the value comes from when not from the user; shown muted after the value."""
    preview: ChoicePreview | None = None
    """Renders a sample of each choice in the terminal choice picker."""
    button: str = ''
    """What the web UI's button for a row with an action says, such as `View…`.

    When empty, `Change…` for a row with a value or a secret, else `Open…`.
    """

    def display(self, value: str) -> str:
        """Label a choice without changing its stored or validated value."""
        return self.choice_labels.get(value, value)


class FieldSource(Protocol):
    """Where the rows come from and where edits go. Every method is synchronous and may block, so callers run it off the event loop."""

    @property
    def title(self) -> str:
        """Menu title."""
        ...

    def rows(self) -> Sequence[FieldRow]:
        """Every editable field, in display order."""
        ...

    def current(self, row: FieldRow) -> str:
        """The active value as the user would type it; the typed editor opens holding it unless it is the row's `default`."""
        ...

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Why `text` is not valid for the row, or `None` when it is."""
        ...

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save and apply a non-empty value; return the message to show."""
        ...

    def reset(self, row: FieldRow) -> str:
        """Forget the override; return the message to show."""
        ...


@dataclass(frozen=True, kw_only=True)
class Option:
    """One answer `Dialogs.choose` offers."""

    value: str
    """What `choose` returns when this option is picked."""
    label: str
    description: str = ''
    """More about the option, beside the list in the terminal and under the label in the browser."""


class Dialogs(Protocol):
    """Ask the user things, wherever they opened the menu: the terminal, or the web UI.

    Every method waits for the answer. Backing out (Esc, or closing the dialog) returns `None` or
    `False`, never raises.
    """

    async def choose(
        self, *, title: str, options: Sequence[Option], current: str | None = None, description: str = ''
    ) -> str | None:
        """The picked option's `value`, or `None` when the user backed out. `current` is marked and starts selected."""
        ...

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
        """The typed text, stripped, or `None` when the user backed out.

        `secret` masks it. `check` says why non-empty text is not acceptable, or returns `None`;
        the dialog stays open until it accepts the text or the text is empty. `description` says
        more above the input, such as where to find the value, with a link to open. `multiline`
        takes text with line breaks, such as instructions, in a text editor; it is never masked.
        Without it, an `initial` of several lines is still edited over several lines where the
        surface can, as the web UI does.
        """
        ...

    async def confirm(self, *, title: str, yes: str, no: str, description: str = '') -> bool:
        """Whether the user picked `yes`; `no`, which starts selected, and backing out are `False`."""
        ...

    async def view(self, *, title: str, text: str) -> None:
        """Show `text`, read-only and scrollable, until the user closes it."""
        ...

    async def wait(
        self, *, title: str, description: str, task: Callable[[Callable[[str], None]], Awaitable[object]]
    ) -> bool:
        """Run `task` behind a waiting screen the user can cancel; whether it finished.

        `task` gets a `show` function whose latest text the screen shows, such as a sign-in link
        or code. The screen closes when `task` returns; cancelling it cancels `task`. `task`'s
        errors propagate.
        """
        ...


class CloseMenu(tuple[str, ...]):
    """The messages an action returns, as `CloseMenu(['Saved.'])`, when its menu should close once the action is done.

    For an action that starts what the menu is for. The terminal closes the menu the action's row is in and
    shows the messages; leaving a menu another opened returns to that one, as Esc does. The web UI shows the
    messages, as after any action. Being a sequence of the messages, it is what any other caller expects.
    """

    __slots__ = ()


Action = Callable[[Dialogs], Awaitable[Sequence[str]]]
"""What a row with more than a value to edit does when chosen: ask through `Dialogs`, save, and return messages to show.

Return them as a `CloseMenu` to close the menu as well.
"""


@dataclass(frozen=True, kw_only=True)
class SettingsMenu:
    """A plugin's settings menu: rows that edit a value each, and actions for rows that need more."""

    source: FieldSource
    actions: Mapping[str, Action] = field(default_factory=dict[str, Action])
    """Row key to what choosing that row does, instead of editing its value."""
    menus: 'Mapping[str, Callable[[], SettingsMenu]]' = field(default_factory=dict[str, 'Callable[[], SettingsMenu]'])
    """Row key to the menu choosing that row opens, built when it opens; leaving it returns to this one.

    Read as this menu opens, so a row an action adds meanwhile opens its menu once this one opens again.
    """
    searchable: bool = True
    """Whether typing filters the terminal list. A short menu may prefer single-key shortcuts instead."""
