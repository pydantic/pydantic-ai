"""The `/set` menu: Pax's own settings, edited through the shared field editor."""

from collections.abc import Callable, Collection

from pydantic_ai.models import known_model_names
from pydantic_ai_pax.cli.command_context import CommandContext
from pydantic_ai_pax.config import SETTING_FIELDS, TOOL_CALL_DISPLAYS, UPDATE_CHANNELS, Settings
from pydantic_ai_pax.config.api_keys import set_api_key
from pydantic_ai_pax.config.input_bindings import BUSY_DELIVERIES, KEYBOARD_LABELS, QUESTION_ANSWERERS, QUESTION_SUBMITS
from pydantic_ai_pax.plugins.menus import SettingsMenu
from pydantic_ai_pax.ui.menus.dialogs import run_settings_menu
from pydantic_ai_pax.ui.menus.field_menu import ChoicePreview, FieldMenu, FieldRow, first_error, shown
from pydantic_ai_pax.ui.menus.footer_menu import FOOTER_SETTING, footer_settings_menu
from pydantic_ai_pax.ui.menus.menu_worker import run_worker
from pydantic_ai_pax.ui.menus.tool_calls_preview import tool_calls_preview
from pydantic_ai_pax.ui.rendering.spinners import BUILTIN_SPINNERS
from pydantic_ai_pax.ui.rendering.theme import names


class SettingsSource:
    """Rows from the `Settings` model; edits through `CommandContext`, same as `/set KEY VALUE`."""

    def __init__(self, context: CommandContext, *, keys: Collection[str] = (), title: str = 'Settings') -> None:
        """Edits are validated, saved, and applied by `context`; `keys` narrows the rows to those settings."""
        self._context = context
        self._keys = keys
        self.title = title

    def rows(self) -> list[FieldRow]:
        """Every `/set` key with its description, default, and fixed choices if it has any.

        `run.instructions` is left to `/system_prompt`: a one-line input would lose its line breaks.
        """
        rows: list[FieldRow] = []
        defaults = Settings().model_dump(mode='json')
        for key, field in SETTING_FIELDS.items():
            if key == 'run.instructions' or (self._keys and key not in self._keys):
                continue
            info = Settings.model_fields[field]
            preview: ChoicePreview | None = None
            match key:
                case _ if info.annotation is bool:
                    choices: tuple[str, ...] = ('true', 'false')
                case 'display.theme':
                    choices = names()
                case 'display.spinner':
                    choices = tuple(BUILTIN_SPINNERS)
                case 'model':
                    choices = tuple(known_model_names())
                case 'display.tool_calls':
                    choices = TOOL_CALL_DISPLAYS
                    preview = tool_calls_preview
                case 'updates.channel':
                    choices = UPDATE_CHANNELS
                case 'input.keyboard_labels':
                    choices = KEYBOARD_LABELS
                case 'input.busy_delivery':
                    choices = BUSY_DELIVERIES
                case 'input.question_submit':
                    choices = QUESTION_SUBMITS
                case 'input.questions':
                    choices = QUESTION_ANSWERERS
                case _:
                    choices = ()
            rows.append(
                FieldRow(
                    key=key,
                    description=info.description or '',
                    default=shown(defaults[field]),
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


def settings_menu(context: CommandContext, *, keys: Collection[str] = (), title: str = 'Settings') -> SettingsMenu:
    """`/set`'s rows as a declared menu, which the web UI shows; `keys` narrows it, as `/theme` and `/spinner` do."""
    return SettingsMenu(
        source=SettingsSource(context, keys=keys, title=title),
        menus={FOOTER_SETTING: lambda: footer_settings_menu(context)},
    )


async def open_settings_menu(context: CommandContext, *, run: Callable[[FieldMenu], list[str]] | None = None) -> str:
    """Show the menu in a thread; edits save and apply as they happen."""
    messages = (
        await run_worker(lambda: run(FieldMenu(SettingsSource(context))))
        if run is not None
        else await run_settings_menu(settings_menu(context))
    )
    return '\n'.join(messages) or 'No changes.'


async def set_command(context: CommandContext, args: list[str]) -> str:
    """Route private credential entry separately from ordinary settings."""
    if args and args[0] == 'api_key':
        return await set_api_key(args=args[1:])
    return context.set_setting(args) if args else await open_settings_menu(context)
