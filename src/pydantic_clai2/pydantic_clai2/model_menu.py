"""The `/model` menu: pick the model for the next prompt, or edit one model's settings."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, get_args, get_origin

from pydantic import JsonValue, TypeAdapter, ValidationError
from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu, MenuResult  # pyright: ignore[reportMissingTypeStubs]

from . import openrouter, vllm
from ._rendering import markdown_style
from .command_context import CommandContext
from .field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow, shown
from .menu_worker import menu_key, run_worker
from .model_catalog import CatalogModel, catalog
from .model_settings import ModelSettingsForm
from .settings_store import SettingsStore

_HINT = 'type to filter - Enter use this model - Ctrl+S settings - Esc close'
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
        return f'Settings for {self.model}'

    def rows(self) -> list[FieldRow]:
        """Every form field with its description and any fixed choices."""
        rows: list[FieldRow] = []
        for key, info in ModelSettingsForm.model_fields.items():
            rows.append(
                FieldRow(
                    key=key, description=info.description or '', default='(not set)', choices=_choices(info.annotation)
                )
            )
        return rows

    def current(self, row: FieldRow) -> str:
        """The saved value as the user would type it."""
        return shown(self._store.model_settings(self.model).get(row.key))

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Why `text` is not valid for the row, or `None` if it is."""
        try:
            self._validated(row, text)
        except ValidationError as exc:
            return first_error(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        """Validate the whole form with this change, then save it."""
        try:
            form = self._validated(row, raw)
        except ValidationError as exc:
            return f'{row.key}: {first_error(exc)}'
        self._store.save_model_settings(self.model, form.model_dump(exclude_none=True))
        return f'Saved {row.key} for {self.model}. Applies when this model is selected.'

    def reset(self, row: FieldRow) -> str:
        """Drop one override."""
        saved = self._store.model_settings(self.model)
        saved.pop(row.key, None)
        self._store.save_model_settings(self.model, saved)
        return f'Reset {row.key} for {self.model}.'

    def _validated(self, row: FieldRow, text: str) -> ModelSettingsForm:
        try:
            value = _JSON.validate_json(text)
        except ValidationError:
            value = text
        return ModelSettingsForm.model_validate({**self._store.model_settings(self.model), row.key: value})


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

    def settings_menu(self, name: str) -> FieldMenu:
        """The field editor for one model's overrides."""
        return FieldMenu(ModelSettingsSource(self._context.store, name))


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
            messages += run_flow(menu.settings_menu(value.model), runners)
            continue
        if isinstance(value, str):
            messages.append(menu.choose(value))
        return True


async def open_model_menu(context: CommandContext, *, run: Callable[[ModelMenu], list[str]] | None = None) -> str:
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
