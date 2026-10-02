"""Select or delete previously added models without browsing providers."""

from dataclasses import dataclass
from enum import Enum

from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu, MenuResult

from pydantic_clai2.cli.command_context import CommandContext
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


def model_completions(context: CommandContext, args: list[str]) -> list[str]:
    """Read the saved list on each completion so changes appear immediately."""
    return context.store.models() if len(args) <= 1 else []


def build_model_picker(context: CommandContext, *, message: str = '') -> Menu:
    """List saved models with routes to add, select, and delete models."""
    names = context.store.models()
    current = context.settings.model
    saved_default = context.store.overrides().get('model')
    items: list[MenuItem] = []
    for name in names:
        status = ' (current)' if name == current else ' (saved default)' if name == saved_default else ''
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
        if item.value is ModelPickerAction.ADD:
            return 'Browse providers to add\nand select a model.'
        if item.value == current:
            deletion = 'Select another model before\ndeleting the current model.'
        elif item.value == saved_default:
            deletion = 'This is your saved default model.\nChoose another with /set model first.'
        else:
            deletion = 'Ctrl+D or Delete removes this model\nand its settings after confirmation.'
        return f'{item.value}\n\nEnter selects this model\nfor the next prompt.\n\n{deletion}'

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
        if name == context.settings.model:
            message = 'Select another model before deleting the current model.'
            continue
        if name == context.store.overrides().get('model'):
            message = 'This is your saved default model. Choose another with /set model first.'
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
            context.store.remove_model(name=name)
            message = f'Deleted {name}.'
            messages.append(message)


async def model_command(context: CommandContext, args: list[str], *, runners: Runners = TERMINAL) -> str:
    """Select an existing model by name or through the picker."""
    if len(args) > 1:
        raise ValueError('Usage: /model [NAME]')
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
    if name not in context.store.models():
        raise ValueError(f'Model not added: {name}. Use /add_model {name} first.')
    messages.append(context.set_setting(['model', name]))
    return '\n'.join(messages)
