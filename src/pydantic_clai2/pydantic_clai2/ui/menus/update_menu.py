"""`/update`'s declared menu: the running CLAI, an update that asks first, and the `updates.channel` setting.

The terminal's bare `/update` still installs at once; the web UI shows this menu, as Settings > About.
"""

from collections.abc import Callable
from dataclasses import replace

from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.cli.self_update import Updates
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, SettingsMenu
from pydantic_clai2.ui.menus.set_menu import SettingsSource

VERSION = 'version'


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

    def rows(self) -> list[FieldRow]:
        """The version, whose action updates, then the channel."""
        version = FieldRow(
            key=VERSION,
            label='Version',
            description='The CLAI running now. Update installs the newest one on the update channel, as /update does.',
            default=self.version(),
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
) -> SettingsMenu:
    """The menu, whose version row asks before entering `/update` with `run`, as typing it would.

    `refusal` says why `/update` would be refused now, for example while other sessions run turns, so the menu
    says so instead of claiming it runs.
    """

    async def update(dialogs: Dialogs) -> list[str]:
        if (refused := refusal()) is not None:
            return [refused]
        channel = context.settings.update_channel
        confirmed = await dialogs.confirm(
            title='Update CLAI?',
            yes='Update',
            no='Cancel',
            description=(
                f'Installs the newest CLAI from the {channel} channel with uv tool install --force '
                '--compile-bytecode, as /update typed in the terminal does. A running turn finishes first.'
            ),
        )
        if not confirmed:
            return []
        # Another session may have started a turn while the dialog was open.
        if (refused := refusal()) is not None:
            return [refused]
        run('/update')
        return [f'Running /update on the {channel} channel.']

    return SettingsMenu(source=AboutSource(context, updates), actions={VERSION: update}, searchable=False)
