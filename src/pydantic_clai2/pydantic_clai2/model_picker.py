"""Select a previously added model without browsing providers or editing settings."""

from enum import Enum

from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu  # pyright: ignore[reportMissingTypeStubs]

from ._rendering import markdown_style
from .command_context import CommandContext
from .field_menu import TERMINAL, Runners
from .menu_worker import menu_key, run_worker


class ModelPickerAction(Enum):
    """Actions distinct from saved model names."""

    ADD = 'add'


def model_completions(context: CommandContext, args: list[str]) -> list[str]:
    """Read the saved list on each completion so additions appear immediately."""
    return context.store.models() if len(args) <= 1 else []


def build_model_picker(context: CommandContext) -> Menu:
    """List saved models with a route to add and select another model."""
    names = context.store.models()
    current = context.settings.model
    items = [MenuItem(f'{name}{" (current)" if name == current else ""}', value=name) for name in names]
    return (
        MenuBuilder('Select model')
        .style(markdown_style())
        .items([*items, MenuItem('Add a model...', value=ModelPickerAction.ADD)])
        .searchable()
        .initial_index(names.index(current) if current in names else 0)
        .preview(
            lambda item: (
                'Browse providers to add and select a model.'
                if item.value is ModelPickerAction.ADD
                else f'{item.value}\n\nEnter selects this model for the next prompt.'
            )
        )
        .footer_hint('type to filter - Enter select - Esc close')
        .key_source(menu_key)
        .build()
    )


async def model_command(context: CommandContext, args: list[str], *, runners: Runners = TERMINAL) -> str:
    """Select an existing model by name or through the picker."""
    if len(args) > 1:
        raise ValueError('Usage: /model [NAME]')
    if args:
        name = args[0]
    else:
        result = await run_worker(lambda: runners.run_list(build_model_picker(context)))
        if result.cancelled or result.item is None:
            return 'No changes.'
        if result.item.value is ModelPickerAction.ADD:
            from .model_menu import open_add_model_menu  # noqa: PLC0415

            return await open_add_model_menu(context, runners=runners)
        if not isinstance(result.item.value, str):
            return 'No changes.'
        name = result.item.value
    if name not in context.store.models():
        raise ValueError(f'Model not added: {name}. Use /add_model {name} first.')
    return context.set_setting(['model', name])
