"""`/model`: select, add, delete, or configure models."""

from dataclasses import dataclass
from enum import Enum
from textwrap import fill

from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu, MenuResult

from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.commands import set_completions
from pydantic_clai2.ui.menus.field_menu import TERMINAL, Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.rendering._rendering import markdown_style


class ModelPickerAction(Enum):
    """Actions distinct from saved model names."""

    ADD = 'add'


@dataclass(frozen=True, kw_only=True)
class DeleteModel:
    """Request confirmation before deleting a saved model."""

    name: str


MODEL_SUBCOMMANDS = ('add', 'settings')
"""`/model` subcommands; model names normally start with `PROVIDER:`, so none is called `add` or `settings`."""

_USAGE = 'Usage: /model [NAME] | /model add [NAME] | /model settings [NAME]'


def model_completions(context: CommandContext, args: list[str]) -> list[str]:
    """Read the saved list on each completion so changes appear immediately."""
    if len(args) <= 1:
        return [*MODEL_SUBCOMMANDS, *context.store.models()]
    if len(args) == 2 and args[0] == 'add':
        return list(set_completions(['model', args[1]], plugin_models=context.plugin_models()))
    if len(args) == 2 and args[0] == 'settings':
        return context.store.models()
    return []


def _protected_models(context: CommandContext) -> dict[str, str]:
    """Explain why the current model and saved default cannot be deleted."""
    protected: dict[str, str] = {}
    saved_default = context.store.overrides().get('model')
    if isinstance(saved_default, str):
        protected[saved_default] = 'This is your saved default model. Choose another with /set model first.'
    if context.settings.model is not None:
        protected[context.settings.model] = 'Select another model before deleting the current model.'
    return protected


def build_model_picker(context: CommandContext, *, message: str = '') -> Menu:
    """List saved models with routes to add, select, and delete models."""
    names = context.store.models()
    current = context.settings.model
    protected = _protected_models(context)
    items: list[MenuItem] = []
    for name in names:
        status = ' (current)' if name == current else ' (saved default)' if name in protected else ''
        items.append(MenuItem(f'{name}{status}', value=name))
    items.append(MenuItem('Add a model...', value=ModelPickerAction.ADD))
    if message:
        items.append(MenuItem(message, disabled=True))

    def delete(menu: Menu, item: MenuItem) -> MenuResult | None:
        if not item.disabled and isinstance(item.value, str):
            return MenuResult(item=MenuItem(item.label, value=DeleteModel(name=item.value)))
        return None

    def preview(item: MenuItem) -> str:
        if item.disabled:
            return item.label
        if isinstance(item.value, str):
            deletion = (
                protected.get(item.value) or 'Ctrl+D or Delete removes this model and its settings after confirmation.'
            )
            return f'{item.value}\n\nEnter selects this model\nfor the next prompt.\n\n{fill(deletion, width=35)}'
        return 'Browse providers to add\nand select a model.'

    return (
        MenuBuilder('Select model')
        .style(markdown_style())
        .items(items)
        .searchable()
        .initial_index(names.index(current) if current in names else 0)
        .preview(preview)
        .on_key('ctrl-d', delete)
        .on_key('delete', delete)
        .footer_hint('type to filter - Enter select - Ctrl+D/Del delete - Esc close')
        .key_source(menu_key)
        .build()
    )


def _run_model_picker(context: CommandContext, *, runners: Runners) -> tuple[MenuResult, list[str]]:
    messages: list[str] = []
    message = ''
    while True:
        result = runners.run_list(build_model_picker(context, message=message))
        if result.cancelled or result.item is None or not isinstance(result.item.value, DeleteModel):
            return result, messages
        name = result.item.value.name
        if reason := _protected_models(context).get(name):
            message = reason
            continue
        confirmation = runners.run_choice(
            MenuBuilder(f'Delete {name}?')
            .style(markdown_style())
            .items([MenuItem('Keep model', value=False), MenuItem('Delete model', value=True)])
            .preview(
                lambda item: (
                    'Remove this saved model and its\nper-model settings.\n\n'
                    'Provider credentials are kept.\nThis cannot be undone.'
                )
            )
            .footer_hint('Enter select - Esc keep model')
            .key_source(menu_key)
            .build()
        )
        if not confirmation.cancelled and confirmation.item is not None and confirmation.item.value is True:
            if not context.store.remove_model(name=name):
                message = f'Kept {name}: it became the saved default.'
                continue
            message = f'Deleted {name}.'
            messages.append(message)


async def model_command(context: CommandContext, args: list[str], *, runners: Runners = TERMINAL) -> str:
    """Select a model by name or through the picker, or run the `add` and `settings` subcommands.

    A name not yet in the saved list is added; an unknown model fails on the next request.
    """
    if args[:1] == ['settings']:
        from pydantic_clai2.ui.menus.model_menu import model_settings_command

        return await model_settings_command(context, args[1:], runners=runners)
    if args == ['add']:
        from pydantic_clai2.ui.menus.model_menu import open_add_model_menu

        return await open_add_model_menu(context, runners=runners)
    if args[:1] == ['add']:
        args = args[1:]
    if len(args) > 1:
        raise ValueError(_USAGE)
    messages: list[str] = []
    if args:
        name = args[0]
    else:
        result, messages = await run_worker(lambda: _run_model_picker(context, runners=runners))
        if result.cancelled or result.item is None:
            return '\n'.join(messages) or 'No changes.'
        if result.item.value is ModelPickerAction.ADD:
            from pydantic_clai2.ui.menus.model_menu import open_add_model_menu

            messages.append(await open_add_model_menu(context, runners=runners))
            return '\n'.join(messages)
        if not isinstance(result.item.value, str):
            return '\n'.join(messages) or 'No changes.'
        name = result.item.value
    messages.append(context.set_setting(['model', name]))
    return '\n'.join(messages)
