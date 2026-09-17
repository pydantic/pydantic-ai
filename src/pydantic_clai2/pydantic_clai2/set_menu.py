"""The `/set` menu: CLAI's own settings, edited through the shared field editor."""

from collections.abc import Callable

from pydantic import ValidationError
from pydantic_ai.models import known_model_names

from .command_context import CommandContext
from .config import SETTING_FIELDS, Settings
from .field_menu import FieldMenu, FieldRow, first_error, run_flow, shown
from .menu_worker import run_worker


class SettingsSource:
    """Rows from the `Settings` model; edits through `CommandContext`, same as `/set KEY VALUE`."""

    title = 'Settings'

    def __init__(self, context: CommandContext) -> None:
        """Edits are validated, saved, and applied by `context`."""
        self._context = context

    def rows(self) -> list[FieldRow]:
        """Every `/set` key with its description, default, and fixed choices if it has any."""
        rows: list[FieldRow] = []
        for key, field in SETTING_FIELDS.items():
            info = Settings.model_fields[field]
            if info.annotation is bool:
                choices: tuple[str, ...] = ('true', 'false')
            elif key == 'model':
                choices = tuple(known_model_names())
            else:
                choices = ()
            rows.append(
                FieldRow(
                    key=key,
                    description=info.description or '',
                    default=shown(info.default),
                    choices=choices,
                    note='project' if self._context.from_project(key) else '',
                )
            )
        return rows

    def current(self, row: FieldRow) -> str:
        """The active value as the user would type it."""
        return shown(self._context.settings.model_dump()[SETTING_FIELDS[row.key]])

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Why `text` is not a valid value for the row, or `None` if it is."""
        try:
            self._context.validate(row.key, text)
        except ValidationError as exc:
            return first_error(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save and apply. Returns the message to show."""
        try:
            return self._context.set_setting([row.key, raw])
        except ValidationError as exc:
            return f'{row.key}: {first_error(exc)}'

    def reset(self, row: FieldRow) -> str:
        """Forget the override and apply the default."""
        return self._context.reset_setting(row.key)


async def open_settings_menu(context: CommandContext, *, run: Callable[[FieldMenu], list[str]] | None = None) -> str:
    """Show the menu in a thread; edits save and apply as they happen."""
    messages = await run_worker(lambda: (run or run_flow)(FieldMenu(SettingsSource(context))))
    return '\n'.join(messages) or 'No changes.'
