"""`/model`: select, add, delete, or configure models, and create, edit, rename, or delete fallback chains."""

from dataclasses import dataclass
from enum import Enum
from textwrap import fill

from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu, MenuResult

from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.commands import set_completions
from pydantic_clai2.models.chains import chain_name
from pydantic_clai2.ui.menus.chain_menu import ChainChange, chain_details, edit_chain, rename_chain
from pydantic_clai2.ui.menus.field_menu import TERMINAL, Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.menus.slash_search import KeyHandler
from pydantic_clai2.ui.rendering._rendering import markdown_style


class ModelPickerAction(Enum):
    """Actions distinct from saved model names."""

    ADD = 'add'
    NEW_CHAIN = 'new chain'


@dataclass(frozen=True, kw_only=True)
class DeleteModel:
    """Request confirmation before deleting a saved model."""

    name: str


@dataclass(frozen=True, kw_only=True)
class EditChain:
    """`Ctrl+E` on a chain: change its models and their order."""

    name: str
    """The chain's `NAME`, without `chain:`."""


@dataclass(frozen=True, kw_only=True)
class RenameChain:
    """`Ctrl+R` on a chain: give it another name."""

    name: str
    """The chain's `NAME`, without `chain:`."""


MODEL_SUBCOMMANDS = ('add', 'settings', 'chains')
"""`/model` subcommands; model names normally start with `PROVIDER:`, so none is called like one."""

_USAGE = 'Usage: /model [NAME] | /model add [NAME] | /model settings [NAME] | /model chains'
_HINT = 'type to filter - Enter select - Ctrl+E edit chain - Ctrl+R rename chain - Ctrl+D/Del delete - Esc close'


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


def build_model_picker(context: CommandContext, *, message: str = '', focus: object = None) -> Menu:
    """List saved models, then fallback chains, with routes to add, select, edit, and delete them.

    The cursor starts on `focus` when it is a row's value, else on the current model.
    """
    names = context.store.models()
    chains = context.store.chains()
    current = context.settings.model
    protected = _protected_models(context)

    def row(name: str) -> MenuItem:
        status = ' (current)' if name == current else ' (saved default)' if name in protected else ''
        return MenuItem(f'{name}{status}', value=name)

    items = [row(name) for name in names if chain_name(name) is None]
    items.append(MenuItem('Add a model...', value=ModelPickerAction.ADD))
    items.append(MenuItem('Fallback chains', disabled=True))
    items += [row(name) for name in names if chain_name(name) is not None]
    items.append(MenuItem('New fallback chain...', value=ModelPickerAction.NEW_CHAIN))
    if message:
        items.append(MenuItem(message, disabled=True))

    def delete(menu: Menu, item: MenuItem) -> MenuResult | None:
        if not item.disabled and isinstance(item.value, str):
            return MenuResult(item=MenuItem(item.label, value=DeleteModel(name=item.value)))
        return None

    def on_chain(action: type[EditChain] | type[RenameChain]) -> KeyHandler:
        def handler(menu: Menu, item: MenuItem) -> MenuResult | None:
            name = chain_name(item.value) if isinstance(item.value, str) else None
            return None if name is None else MenuResult(item=MenuItem(item.label, value=action(name=name)))

        return handler

    def preview(item: MenuItem) -> str:
        if item.disabled:
            return item.label
        if isinstance(item.value, str):
            deletion = (
                protected.get(item.value) or 'Ctrl+D or Delete removes this model and its settings after confirmation.'
            )
            if (chain := chain_name(item.value)) is not None:
                lines = [item.value, '', *chain_details(chains.get(chain)), '', 'Enter selects this chain.']
                lines += ['Ctrl+E changes its models,', 'Ctrl+R renames it.', '', fill(deletion, width=35)]
                return '\n'.join(lines)
            return f'{item.value}\n\nEnter selects this model\nfor the next prompt.\n\n{fill(deletion, width=35)}'
        if item.value is ModelPickerAction.NEW_CHAIN:
            return 'Create a fallback chain:\nmodels tried in order, each\ntaking over when the one\nbefore it fails.'
        return 'Browse providers to add\nand select a model.'

    # Headings and the status row have no value, so neither `None` target can land on them.
    rows = [(index, item.value) for index, item in enumerate(items) if not item.disabled]
    initial = next((index for target in (focus, current) for index, value in rows if value == target), 0)
    return (
        MenuBuilder('Select model')
        .style(markdown_style())
        .items(items)
        .searchable()
        .initial_index(initial)
        .preview(preview)
        .on_key('ctrl-d', delete)
        .on_key('delete', delete)
        .on_key('ctrl-e', on_chain(EditChain))
        .on_key('ctrl-r', on_chain(RenameChain))
        .footer_hint(_HINT)
        .key_source(menu_key)
        .build()
    )


def _chain_action(context: CommandContext, value: object, runners: Runners) -> ChainChange | None:
    """Run the chain editor a picker row asked for; raises `ValueError` with why it changed nothing."""
    if isinstance(value, RenameChain):
        if f'chain:{value.name}' in _protected_models(context):
            raise ValueError(f'Select another model before renaming chain:{value.name}.')
        return rename_chain(context, value.name, runners)
    return edit_chain(context, value.name if isinstance(value, EditChain) else None, runners)


def _run_model_picker(
    context: CommandContext, *, runners: Runners, focus: object = None
) -> tuple[MenuResult, list[str]]:
    messages: list[str] = []
    message = ''
    while True:
        result = runners.run_list(build_model_picker(context, message=message, focus=focus))
        value = None if result.cancelled or result.item is None else result.item.value
        if value is ModelPickerAction.NEW_CHAIN or isinstance(value, EditChain | RenameChain):
            try:
                change = _chain_action(context, value, runners)
            except ValueError as exc:
                message = str(exc)
                continue
            message = '' if change is None else change.message
            if change is not None:
                messages.append(message)
                focus = f'chain:{change.name}'
            continue
        if not isinstance(value, DeleteModel):
            return result, messages
        name = value.name
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
    """Select a model by name or through the picker, or run the `add`, `settings`, and `chains` subcommands.

    A name not yet in the saved list is added; an unknown model fails on the next request.
    """
    if args[:1] == ['settings']:
        from pydantic_clai2.ui.menus.model_menu import model_settings_command

        return await model_settings_command(context, args[1:], runners=runners)
    if args == ['add']:
        from pydantic_clai2.ui.menus.model_menu import open_add_model_menu

        return await open_add_model_menu(context, runners=runners)
    focus: object = None
    if args[:1] == ['chains']:
        # Opens the picker on its fallback chains: the first one, or the row that creates one.
        if args[1:]:
            raise ValueError('Usage: /model chains. Create, edit, rename, and delete fallback chains in its picker.')
        args = []
        chains = [name for name in context.store.models() if chain_name(name) is not None]
        focus = chains[0] if chains else ModelPickerAction.NEW_CHAIN
    if args[:1] == ['add']:
        args = args[1:]
    if len(args) > 1:
        raise ValueError(_USAGE)
    messages: list[str] = []
    if args:
        name = args[0]
    else:
        result, messages = await run_worker(lambda: _run_model_picker(context, runners=runners, focus=focus))
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
