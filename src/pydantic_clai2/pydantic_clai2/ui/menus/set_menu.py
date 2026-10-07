"""The `/set` menu: CLAI's own settings, edited through the shared field editor."""

from collections.abc import Callable

from pydantic_ai.models import known_model_names
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config import SETTING_FIELDS, TOOL_CALL_DISPLAYS, UPDATE_CHANNELS, Settings
from pydantic_clai2.config.api_keys import set_api_key
from pydantic_clai2.ui.menus.field_menu import ChoicePreview, FieldMenu, FieldRow, first_error, run_flow, shown
from pydantic_clai2.ui.menus.menu_worker import run_worker
from pydantic_clai2.ui.menus.tool_calls_preview import tool_calls_preview
from pydantic_clai2.ui.rendering.spinners import BUILTIN_SPINNERS
from pydantic_clai2.ui.rendering.theme import names


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
            preview: ChoicePreview | None = None
            if info.annotation is bool:
                choices: tuple[str, ...] = ('true', 'false')
            elif key == 'display.theme':
                choices = names()
            elif key == 'display.spinner':
                choices = tuple(BUILTIN_SPINNERS)
            elif key == 'model':
                choices = tuple(known_model_names())
            elif key == 'display.tool_calls':
                choices = TOOL_CALL_DISPLAYS
                preview = tool_calls_preview
            elif key == 'updates.channel':
                choices = UPDATE_CHANNELS
            else:
                choices = ()
            rows.append(
                FieldRow(
                    key=key,
                    description=info.description or '',
                    default=shown(info.default),
                    choices=choices,
                    note='project' if self._context.from_project(key) else '',
                    preview=preview,
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
        except ValueError as exc:
            return first_error(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save and apply. Returns the message to show."""
        try:
            return self._context.set_setting([row.key, raw])
        except ValueError as exc:
            return f'{row.key}: {first_error(exc)}'

    def reset(self, row: FieldRow) -> str:
        """Forget the override and apply the default."""
        return self._context.reset_setting(row.key)


async def open_settings_menu(context: CommandContext, *, run: Callable[[FieldMenu], list[str]] | None = None) -> str:
    """Show the menu in a thread; edits save and apply as they happen."""
    messages = await run_worker(lambda: (run or run_flow)(FieldMenu(SettingsSource(context))))
    return '\n'.join(messages) or 'No changes.'


async def set_command(context: CommandContext, args: list[str]) -> str:
    """Route private credential entry separately from ordinary settings."""
    if args and args[0] == 'api_key':
        return await set_api_key(args=args[1:])
    return context.set_setting(args) if args else await open_settings_menu(context)
