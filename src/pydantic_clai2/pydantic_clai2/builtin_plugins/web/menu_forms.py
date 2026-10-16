"""Settings menus and forms as the page shows them: a plugin's form, a menu's rows, and an edit through a menu."""

from anyio import to_thread
from pydantic.errors import PydanticInvalidForJsonSchema

from pydantic_clai2.builtin_plugins.web.protocol import Choice, MenuField, MenuInfo, PluginForm
from pydantic_clai2.plugins import PluginInfo
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, SettingsMenu


def plugin_form(info: PluginInfo) -> PluginForm | None:
    """A settings form for a plugin that is on and takes settings, unless its model has no JSON Schema."""
    if info.settings_model is None or info.settings is None:
        return None
    try:
        schema = info.settings_model.model_json_schema(by_alias=True)
    except PydanticInvalidForJsonSchema:  # A type JSON Schema cannot express; its own menu or `/plugins add` still can.
        return None
    return PluginForm(json_schema=schema, values=info.settings.model_dump(mode='json', by_alias=True))


def menu_info(menu: SettingsMenu) -> MenuInfo:
    """Every row of `menu` with its current value; reads the menu's source, so call it off the loop."""
    source = menu.source
    return MenuInfo(
        title=source.title,
        fields=[
            MenuField(
                key=row.key,
                label=row.label or row.key,
                description=row.description,
                value=source.current(row),
                default=row.default,
                choices=[Choice(value=choice, label=row.display(choice)) for choice in row.choices],
                allow_custom=row.allow_custom,
                secret=row.secret,
                note=row.note,
                action=row.key in menu.actions,
            )
            for row in source.rows()
        ],
    )


def _row(menu: SettingsMenu, key: str) -> FieldRow:
    row = next((row for row in menu.source.rows() if row.key == key), None)
    if row is None:
        raise LookupError(f'{menu.source.title} has no setting {key}.')
    return row


async def set_row(menu: SettingsMenu, *, key: str, value: str | None) -> list[str]:
    """Set a row as its menu does: empty text resets it, and a value the row rejects raises `ValueError`."""
    row = await to_thread.run_sync(_row, menu, key)
    text = (value or '').strip()
    if not text:
        return [await to_thread.run_sync(menu.source.reset, row)]
    if (problem := await to_thread.run_sync(menu.source.problem, row, text)) is not None:
        raise ValueError(problem)
    return [await to_thread.run_sync(menu.source.apply, row, text)]


async def run_action(menu: SettingsMenu, *, key: str, dialogs: Dialogs, owner: str) -> list[str]:
    """Run a row's action, asking through `dialogs`; `owner`, the plugin or `/command`, names the menu in errors."""
    action = menu.actions.get(key)
    if action is None:
        raise LookupError(f'{owner} has no action {key}.')
    return list(await action(dialogs))
