"""`/update`'s menu: the running CLAI and whether a newer one is out, an update that asks first, and `updates.channel`.

Bare `/update` opens it, in the terminal and in the web UI, which shows it as Settings > About.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace

from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.cli.self_update import Updates
from pydantic_clai2.plugins.menus import CloseMenu, Dialogs, FieldRow, SettingsMenu
from pydantic_clai2.ui.menus.set_menu import SettingsSource

VERSION = 'version'
CHECK_WAIT = 1.0
"""Seconds the menu waits for the update check before it first shows: it cannot redraw once the answer comes."""


class AboutSource:
    """The version row, read-only, then `/set`'s `updates.channel` row."""

    title = 'About CLAI'

    def __init__(self, context: CommandContext, updates: Updates) -> None:
        """The channel is edited through `context`, as `/set updates.channel` does."""
        self._settings = SettingsSource(context, keys=('updates.channel',))
        self._updates = updates

    def version(self) -> str:
        """The running CLAI's version, with the commit a `main` install was built from."""
        current = self._updates.current
        return current.version if current.commit is None else f'{current.version} ({current.commit[:9]})'

    def status(self) -> str:
        """Whether a newer CLAI is out on the update channel, checking for one in the background first."""
        found = self._updates.check(wait=CHECK_WAIT)
        if found is None:
            return 'Checking for a newer CLAI…'
        if found.error is not None:
            return f'Could not check for a newer CLAI: {found.error}'
        if found.update is None:
            return (
                f'clai2 is already on the latest version {self._updates.current.label} on the {found.channel} channel.'
            )
        return f'a new version {found.update.label} of clai2 is available on the {found.channel} channel.'

    def rows(self) -> list[FieldRow]:
        """The version, whose note says whether a newer one is out and whose action updates, then the channel."""
        version = FieldRow(
            key=VERSION,
            label='Version',
            description='The CLAI running now. Update installs the newest one on the update channel, as /update now does.',
            default=self.version(),
            note=self.status(),
            button='Update…',
        )
        return [version, *(replace(row, label='Update channel') for row in self._settings.rows())]

    def current(self, row: FieldRow) -> str:
        """The version, or the channel."""
        return self.version() if row.key == VERSION else self._settings.current(row)

    def problem(self, row: FieldRow, text: str) -> str | None:
        """The version takes no value; the channel is checked as `/set` checks it."""
        if row.key == VERSION:
            return 'The version is read-only; Update installs a newer one.'
        return self._settings.problem(row, text)

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save the channel, as `/set updates.channel` does."""
        if row.key == VERSION:
            raise ValueError(self.problem(row, raw))
        return self._settings.apply(row, raw)

    def reset(self, row: FieldRow) -> str:
        """Reset the channel to `stable`."""
        if row.key == VERSION:
            return 'The version has nothing to reset.'
        return self._settings.reset(row)


def update_menu(
    context: CommandContext,
    updates: Updates,
    *,
    run: Callable[[str], None],
    refusal: Callable[[], str | None] = lambda: None,
    raise_refusal: bool = False,
) -> SettingsMenu:
    """The menu, whose version row asks before entering `/update now` with `run`, as typing it would, and closing.

    `refusal` says why `/update` would be refused now, for example while other sessions run turns, so the menu
    says so instead of claiming it runs. With `raise_refusal`, the action raises that reason as `ValueError`
    instead, so a remote UI such as the web UI's About shows it as an error rather than as a saved row.
    """

    def refused(reason: str) -> Sequence[str]:
        if raise_refusal:
            raise ValueError(reason)
        return [reason]

    async def update(dialogs: Dialogs) -> Sequence[str]:
        if (reason := refusal()) is not None:
            return refused(reason)
        channel = context.settings.update_channel
        confirmed = await dialogs.confirm(
            title='Update CLAI?',
            yes='Update',
            no='Cancel',
            description=(
                f'Installs the newest CLAI from the {channel} channel with uv tool install --force '
                '--compile-bytecode, as /update now does. A running turn finishes first.'
            ),
        )
        if not confirmed:
            return []
        # Another session may have started a turn while the dialog was open.
        if (reason := refusal()) is not None:
            return refused(reason)
        run('/update now')
        return CloseMenu([f'Running /update now on the {channel} channel.'])

    return SettingsMenu(source=AboutSource(context, updates), actions={VERSION: update}, searchable=False)
