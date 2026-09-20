"""The `/add_model` menu: pick the model for the next prompt, or edit one model's settings."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, get_args, get_origin

from pydantic import JsonValue, TypeAdapter, ValidationError
from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu, MenuResult  # pyright: ignore[reportMissingTypeStubs]

from . import openrouter, vllm
from ._rendering import markdown_style
from .command_context import CommandContext
from .custom_params import CustomParamsMenu
from .field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow, shown
from .menu_worker import menu_key, run_worker
from .model_catalog import CatalogModel, catalog
from .model_options import model_options, validate_model_options
from .model_settings import ModelSettingsForm, model_defaults
from .settings_store import SettingsStore

_HINT = 'type to filter - Enter add and use model - Ctrl+S settings - Esc close'
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


@dataclass(frozen=True)
class _EditSettings:
    """What the `Ctrl+S` key hands back to the loop instead of a model name."""

    model: str


class ModelSettingsSource:
    """One model's overrides, from `ModelSettingsForm`, saved in the store."""

    def __init__(self, store: SettingsStore, model: str) -> None:
        """Edits are saved under `model` and validated as a whole form each time."""
        self._store = store
        self.model = model

    @property
    def title(self) -> str:
        """Menu title naming the model."""
        return f'Settings - {self.model}'

    def rows(self) -> list[FieldRow]:
        """Every form field with its description and any fixed choices."""
        rows: list[FieldRow] = []
        options = model_options(model=self.model)
        saved = self._store.model_settings(self.model)
        for key, info in ModelSettingsForm.model_fields.items():
            if key not in options and key not in saved:
                continue
            rows.append(
                FieldRow(
                    key=key,
                    label=_setting_label(key),
                    allow_custom=False,
                    description=info.description or '',
                    default=shown(model_defaults(model=self.model).get(key)),
                    choices=options.get(key, ()) or _choices(info.annotation),
                )
            )
        return rows

    def current(self, row: FieldRow) -> str:
        """The effective value, without persisting inherited defaults."""
        values = {**model_defaults(model=self.model), **self._store.model_settings(self.model)}
        return shown(values.get(row.key))

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Why `text` is not valid for the row, or `None` if it is."""
        try:
            self._validated(row, text)
        except (ValidationError, ValueError) as exc:
            return first_error(exc) if isinstance(exc, ValidationError) else str(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        """Validate the whole form with this change, then save it."""
        try:
            form = self._validated(row, raw)
        except (ValidationError, ValueError) as exc:
            return f'{row.key}: {first_error(exc) if isinstance(exc, ValidationError) else str(exc)}'
        saved = {
            key: value
            for key, value in self._store.model_settings(self.model).items()
            if key not in ModelSettingsForm.model_fields
        }
        self._store.save_model_settings(self.model, {**saved, **form.model_dump(exclude_none=True)})
        return f'Saved {row.key} for {self.model}. Applies when this model is selected.'

    def reset(self, row: FieldRow) -> str:
        """Drop one override."""
        saved = self._store.model_settings(self.model)
        saved.pop(row.key, None)
        if row.key == 'anthropic_thinking_mode':
            saved.pop('anthropic_thinking_budget', None)
            saved.pop('anthropic_thinking_display', None)
            saved.pop('anthropic_preserved_thinking', None)
        self._store.save_model_settings(self.model, saved)
        return f'Reset {row.key} for {self.model}.'

    def _validated(self, row: FieldRow, text: str) -> ModelSettingsForm:
        try:
            value = _JSON.validate_json(text)
        except ValidationError:
            value = text
        saved = {
            key: value
            for key, value in self._store.model_settings(self.model).items()
            if key in ModelSettingsForm.model_fields
        }
        form = ModelSettingsForm.model_validate({**saved, row.key: value})
        validate_model_options(model=self.model, form=form)
        return form


def _setting_label(key: str) -> str:
    labels = {
        'max_tokens': 'Max Output Tokens',
        'top_p': 'Top-P (Nucleus Sampling)',
        'openai_text_verbosity': 'Verbosity',
        'anthropic_thinking_mode': 'Extended Thinking',
        'anthropic_thinking_budget': 'Thinking Budget',
        'anthropic_effort': 'Effort',
        'anthropic_thinking_display': 'Thinking Display',
        'anthropic_preserved_thinking': 'Preserved Thinking',
        'anthropic_interleaved_thinking': 'Interleaved Thinking',
        'glm_thinking': 'Thinking (GLM)',
        'glm_clear_thinking': 'Clear Thinking (GLM)',
        'glm_reasoning_effort': 'Reasoning Effort (GLM)',
    }
    return labels.get(key, key.removeprefix('openai_').removeprefix('anthropic_').replace('_', ' ').title())


def _choices(annotation: object) -> tuple[str, ...]:
    members = get_args(annotation) if get_origin(annotation) is not None else (annotation,)
    choices: list[str] = []
    for member in members:
        if member is bool:
            choices += ['true', 'false']
        elif get_origin(member) is Literal:
            choices += [str(literal) for literal in get_args(member)]
    return tuple(choices)


class ModelMenu:
    """The model list with a details pane; Enter picks, `Ctrl+S` opens that model's settings."""

    def __init__(self, context: CommandContext, *, provider: str | None = None) -> None:
        """The current model is always listed, even when no source knows it."""
        self._context = context
        self.models = [
            model
            for model in catalog(include=[context.settings.model or ''])
            if provider is None or model.name.partition(':')[0] == provider
        ]

    @property
    def current(self) -> str | None:
        """The model `/set model` holds right now."""
        return self._context.settings.model

    def items(self) -> list[MenuItem]:
        """One row per model, marking the current one."""
        return [
            MenuItem(
                f'{model.name}{" (current)" if model.name == self.current else ""}',
                value=model.name,
            )
            for model in self.models
        ]

    def details(self, item: MenuItem) -> str:
        """The right-hand panel: provider, context window, prices, saved overrides."""
        model = self.model_for(item.value)
        if model is None:
            return ''
        overrides = self._context.store.model_settings(model.name)
        lines = [
            model.label,
            '',
            f'provider  {model.provider}',
            f'context   {_tokens(model.context_window) or "unknown"}',
            f'prices    {model.prices or "unknown"}',
            f'settings  {", ".join(f"{key}={shown(value)}" for key, value in overrides.items()) or "none"}',
        ]
        return '\n'.join(lines)

    def build(self, initial: int = 0) -> Menu:
        """The model list, opened on the current model."""
        return (
            MenuBuilder('Models')
            .style(markdown_style())
            .items(self.items())
            .searchable()
            .initial_index(min(initial, max(len(self.models) - 1, 0)))
            .preview(self.details)
            .on_key('ctrl-s', self.settings_marker)
            .footer_hint(_HINT)
            .key_source(menu_key)
            .build()
        )

    def settings_marker(self, menu: object, item: MenuItem) -> MenuResult:
        """Ctrl+S: hand the model back to the loop tagged for its settings editor."""
        return MenuResult(item=MenuItem(item.label, value=_EditSettings(str(item.value))))

    def choose(self, name: str) -> str:
        """Make `name` the model for the next prompt."""
        return self._context.set_setting(['model', name])

    def model_for(self, name: object) -> CatalogModel | None:
        """Look a model up by its qualified name."""
        return next((model for model in self.models if model.name == name), None)

    def index_of(self, name: str | None) -> int:
        """Where a model sits in the list, or 0."""
        return next((index for index, model in enumerate(self.models) if model.name == name), 0)

    def providers(self) -> list[str]:
        """Unique provider prefixes from the merged catalog."""
        return sorted({model.name.partition(':')[0] for model in self.models} | {'openrouter', 'vllm'})

    def build_providers(self) -> Menu:
        """Choose a provider before browsing its models."""
        providers = self.providers()
        current = (self.current or '').partition(':')[0]
        return (
            MenuBuilder('Providers')
            .style(markdown_style())
            .items([MenuItem(provider, value=provider) for provider in providers])
            .searchable()
            .initial_index(providers.index(current) if current in providers else 0)
            .footer_hint('type to filter - Enter browse models - Esc close')
            .key_source(menu_key)
            .build()
        )

    def for_provider(self, provider: str) -> 'ModelMenu':
        """Browse one provider without changing the active model."""
        return ModelMenu(self._context, provider=provider)

    def edit_settings(self, *, name: str, runners: Runners) -> list[str]:
        """Run the same settings flow as the direct slash command."""
        return run_model_settings(store=self._context.store, model=name, runners=runners)


def _tokens(count: int | None) -> str:
    return f'{count:,} tokens' if count is not None else ''


def run_model_flow(menu: ModelMenu, runners: Runners = TERMINAL, *, connect_provider: bool = False) -> list[str]:
    """Show the list; Enter picks and closes, `Ctrl+S` edits settings and returns to the list."""
    messages: list[str] = []
    while True:
        selection = runners.run_list(menu.build_providers())
        if selection.cancelled or selection.item is None or not isinstance(selection.item.value, str):
            return messages
        if selection.item.value in ('openrouter', 'vllm') and connect_provider:
            raise _ConnectProvider(messages, provider=selection.item.value)
        provider_menu = menu.for_provider(selection.item.value)
        if _run_provider(provider_menu, runners, messages):
            return messages


def _run_provider(menu: ModelMenu, runners: Runners, messages: list[str]) -> bool:
    cursor = menu.index_of(menu.current)
    while True:
        result = runners.run_list(menu.build(cursor))
        if result.cancelled or result.item is None:
            return False
        value = result.item.value
        if isinstance(value, _EditSettings):
            cursor = menu.index_of(value.model)
            messages += menu.edit_settings(name=value.model, runners=runners)
            continue
        if isinstance(value, str):
            messages.append(menu.choose(value))
        return True


async def open_add_model_menu(context: CommandContext, *, run: Callable[[ModelMenu], list[str]] | None = None) -> str:
    """Show the menu in a thread; the pick and any settings edits apply to the next prompt."""

    def flow(menu: ModelMenu) -> list[str]:
        return run_model_flow(menu, connect_provider=True)

    accumulated: list[str] = []
    while True:
        try:
            messages = await run_worker(lambda: (run or flow)(ModelMenu(context)))
        except _ConnectProvider as request:
            accumulated.extend(request.messages)
            connector = openrouter.connect if request.provider == 'openrouter' else vllm.connect
            result = await connector(context, [])
            if result == 'Connection cancelled.':
                continue
            return '\n'.join([*accumulated, result])
        return '\n'.join([*accumulated, *messages]) or 'No changes.'


class _ConnectProvider(Exception):
    """Release the menu worker before prompting or awaiting provider discovery."""

    def __init__(self, messages: list[str], *, provider: str) -> None:
        self.provider = provider
        self.messages = messages
        super().__init__()


async def model_settings_command(context: CommandContext, args: list[str], *, runners: Runners = TERMINAL) -> str:
    """Pick a saved model to edit, or open a named one, without switching models."""
    if len(args) > 1:
        raise ValueError('Usage: /model_settings [NAME]')
    if not args:
        messages = await run_worker(lambda: run_model_settings_picker(context=context, runners=runners))
        return '\n'.join(messages) or 'No changes.'
    name = args[0]
    if name not in context.store.models():
        raise ValueError(f'Model not added: {name}. Use /add_model {name} first.')
    messages = await run_worker(lambda: run_model_settings(store=context.store, model=name, runners=runners))
    return '\n'.join(messages) or 'No changes.'


def run_model_settings(*, store: SettingsStore, model: str, runners: Runners) -> list[str]:
    """Both entry points use the same field editor and custom-params submenu."""
    custom = CustomParamsMenu(store=store, model=model)
    return run_flow(
        FieldMenu(ModelSettingsSource(store, model), searchable=False),
        runners,
        submenus={'custom_params': lambda: custom.run(runners=runners)},
    )


def build_model_settings_picker(*, context: CommandContext, current: str | None) -> Menu:
    """Choose a model to configure without changing the active run model."""
    names = context.store.models()

    return (
        MenuBuilder('Model Settings - Select a Model')
        .style(markdown_style())
        .items(
            [MenuItem(name, value=name) for name in names]
            or [MenuItem('No models added. Use /add_model first.', disabled=True)]
        )
        .searchable()
        .list_width(40)
        .initial_index(names.index(current) if current in names else 0)
        .preview(lambda item: model_settings_summary(store=context.store, model=str(item.value)))
        .footer_hint('type filter - Enter configure - Esc exit')
        .key_source(menu_key)
        .build()
    )


def model_settings_summary(*, store: SettingsStore, model: str) -> str:
    """Preview effective settings without mutating the model or its saved overrides."""
    values = {**model_defaults(model=model), **store.model_settings(model)}
    lines = [model, '', 'Configured settings:' if values else 'No custom settings (model defaults).']
    lines.extend(f'{_setting_label(key)}: {shown(value)}' for key, value in values.items())
    return '\n'.join(lines)


def run_model_settings_picker(*, context: CommandContext, runners: Runners) -> list[str]:
    """Return from a model's editor to the picker, preserving its selection."""
    current = context.settings.model
    messages: list[str] = []
    while True:
        result = runners.run_list(build_model_settings_picker(context=context, current=current))
        if result.cancelled or result.item is None or not isinstance(result.item.value, str):
            return messages
        current = result.item.value
        if current not in context.store.models():
            return messages
        messages.extend(run_model_settings(store=context.store, model=current, runners=runners))
