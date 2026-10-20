"""The `/model add` menu: pick the model for the next prompt, or edit one model's settings."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from typing import Literal, get_args, get_origin

from anyio import to_thread
from pydantic import JsonValue, TypeAdapter, ValidationError
from rich.console import Console
from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu, MenuResult

from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models import github_copilot, openrouter, vllm
from pydantic_clai2.models.model_catalog import CatalogModel, catalog, github_copilot_models
from pydantic_clai2.models.model_options import model_options, validate_model_options
from pydantic_clai2.models.model_settings import ModelSettingsForm, model_defaults
from pydantic_clai2.models.profiles import provider_of
from pydantic_clai2.plugins.menus import Dialogs, Option, SettingsMenu
from pydantic_clai2.ui.menus.custom_params import edit_custom_params
from pydantic_clai2.ui.menus.dialogs import run_settings_menu
from pydantic_clai2.ui.menus.field_menu import TERMINAL, FieldRow, Runners, first_error, shown
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.rendering._rendering import markdown_style

_HINT = 'type to filter - Enter add and use model - Ctrl+S settings - Esc close'
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_CONNECTS = ('github-copilot', 'openrouter', 'vllm')
"""Providers `/model add` connects to first: a sign-in, or an address to discover models at."""


@dataclass(frozen=True)
class _EditSettings:
    """What the `Ctrl+S` key hands back to the loop instead of a model name."""

    model: str


class ModelSettingsSource:
    """One model's overrides, from `ModelSettingsForm`, saved in the store."""

    def __init__(self, store: SettingsStore, model: str, *, settings_as: str | None = None) -> None:
        """Edits are saved under `model`, offering and validating the controls of `settings_as` (default `model`)."""
        self._store = store
        self.model = model
        self._settings_as = settings_as or model

    @property
    def title(self) -> str:
        """Menu title naming the model."""
        return f'Settings - {self.model}'

    def rows(self) -> list[FieldRow]:
        """Every form field with its description and any fixed choices."""
        rows: list[FieldRow] = []
        options = model_options(model=self._settings_as)
        saved = self._store.model_settings(self.model)
        for key, info in ModelSettingsForm.model_fields.items():
            if key not in options and key not in saved:
                continue
            codex_tier = key == 'service_tier' and provider_of(self._settings_as) == 'openai-codex'
            rows.append(
                FieldRow(
                    key=key,
                    label='Service Tier / Fast Mode' if codex_tier else setting_label(key),
                    allow_custom=False,
                    description=(
                        'Fast mode uses more ChatGPT credits.\n'
                        'Requests priority processing on supported Codex models.\n'
                        'Availability depends on your model and account.\n'
                        'Standard turns fast mode off.\n'
                        'Reasoning effort is unchanged.\n'
                        'Custom service_tier parameters override this setting.'
                        if codex_tier
                        else info.description or ''
                    ),
                    choice_labels={'priority': 'Fast (priority)', 'default': 'Standard (default)'}
                    if codex_tier
                    else {},
                    default=shown(model_defaults(model=self._settings_as).get(key)),
                    choices=options.get(key, ()) or _choices(info.annotation),
                )
            )
        return rows

    def current(self, row: FieldRow) -> str:
        """The effective value, without persisting inherited defaults."""
        values = {**model_defaults(model=self._settings_as), **self._store.model_settings(self.model)}
        return shown(values.get(row.key))

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Why `text` is not valid for the row, or `None` if it is."""
        try:
            self._validated(row, text)
        except (ValidationError, ValueError) as exc:
            return first_error(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        """Validate the whole form with this change, then save it."""
        try:
            form = self._validated(row, raw)
        except (ValidationError, ValueError) as exc:
            return f'{row.key}: {first_error(exc)}'
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
        validate_model_options(model=self._settings_as, form=form)
        return form


def setting_label(key: str) -> str:
    """How the settings menu names a `ModelSettingsForm` field."""
    labels = {
        'max_tokens': 'Max Output Tokens',
        'top_p': 'Top-P (Nucleus Sampling)',
        'cache': 'Prompt Caching',
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

    def __init__(
        self, context: CommandContext, *, provider: str | None = None, discovered: Iterable[CatalogModel] = ()
    ) -> None:
        """Keep the current model unless authenticated discovery excludes it."""
        self._context = context
        self._discovered = tuple(discovered)
        self.models = [
            model
            for model in catalog(
                include=[context.settings.model or '', *context.plugin_models()], discovered=self._discovered
            )
            if provider is None or model.provider == provider
        ]

    @property
    def current(self) -> str | None:
        """The model `/set model` holds right now."""
        return self._context.settings.model

    @property
    def store(self) -> SettingsStore:
        """Where models, their settings, and accounts are saved."""
        return self._context.store

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
        return sorted({model.provider for model in self.models} | set(_CONNECTS))

    def build_providers(self) -> Menu:
        """Choose a provider before browsing its models."""
        providers = self.providers()
        current = provider_of(self.current or '')
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
        return ModelMenu(self._context, provider=provider, discovered=self._discovered)

    def settings_menu(self, name: str) -> SettingsMenu:
        """`name`'s settings, as `/model settings NAME` shows them."""
        return model_settings_menu(self._context.store, name, settings_as=self._context.settings_model(name))


def _tokens(count: int | None) -> str:
    return f'{count:,} tokens' if count is not None else ''


def run_model_flow(menu: ModelMenu, runners: Runners = TERMINAL, *, connect_provider: bool = False) -> list[str]:
    """Show the providers, then one's models; Enter picks and closes.

    `Ctrl+S` on a model raises `_OpenSettings` (and a provider that connects first `_ConnectProvider`),
    so the caller closes this worker before that menu opens, as `open_add_model_menu` does.
    """
    messages: list[str] = []
    while True:
        selection = runners.run_list(menu.build_providers())
        if selection.cancelled or selection.item is None or not isinstance(selection.item.value, str):
            return messages
        if selection.item.value in _CONNECTS and connect_provider:
            raise _ConnectProvider(messages, provider=selection.item.value)
        provider_menu = menu.for_provider(selection.item.value)
        if _run_provider(provider_menu, runners, messages):
            return messages


def _run_provider(menu: ModelMenu, runners: Runners, messages: list[str], *, focus: str | None = None) -> bool:
    """Whether a model was picked; the list opens on `focus`, else the current model."""
    while True:
        result = runners.run_list(menu.build(menu.index_of(focus or menu.current)))
        if result.cancelled or result.item is None:
            return False
        value = result.item.value
        if isinstance(value, _EditSettings):
            raise _OpenSettings(menu=menu, model=value.model)
        if isinstance(value, str):
            from pydantic_clai2.ui.menus.accounts_menu import choose_account

            name = choose_account(menu.store, value, runners)
            if name is None:
                continue
            messages.append(menu.choose(name))
        return True


async def open_add_model_menu(
    context: CommandContext,
    *,
    console: Console,
    run: Callable[[ModelMenu], list[str]] | None = None,
    runners: Runners = TERMINAL,
) -> str:
    """Show the menu in a thread; the pick and any settings edits apply to the next prompt.

    A provider sign-in prints through `console`, the shell's, so its URL stays in the transcript.
    """

    def flow(menu: ModelMenu) -> list[str]:
        return run_model_flow(menu, runners, connect_provider=True)

    accumulated: list[str] = []
    while True:
        try:
            messages = await run_worker(lambda: (run or flow)(ModelMenu(context)))
        except _OpenSettings as request:
            accumulated += await run_settings_menu(request.menu.settings_menu(request.model), runners)
            if await _browse(request.menu, runners, accumulated, focus=request.model):
                return '\n'.join(accumulated)
            continue
        except _ConnectProvider as request:
            accumulated.extend(request.messages)
            if request.provider == 'github-copilot':
                await github_copilot.ensure_login(console=console)
                models = await github_copilot_models()
                provider_menu = ModelMenu(context, provider='github-copilot', discovered=models)
                if await _browse(provider_menu, runners, accumulated):
                    return '\n'.join(accumulated) or 'No changes.'
                continue
            if request.provider == 'openrouter':
                result = await openrouter.connect(context, [], console=console)
            else:
                result = await vllm.connect(context, [])
            if result == 'Connection cancelled.':
                continue
            return '\n'.join([*accumulated, result])
        return '\n'.join([*accumulated, *messages]) or 'No changes.'


async def _browse(menu: ModelMenu, runners: Runners, messages: list[str], *, focus: str | None = None) -> bool:
    """One provider's models until one is picked (`True`) or the list closes; `Ctrl+S` edits a model between."""
    while True:
        try:
            return await run_worker(partial(_run_provider, menu, runners, messages, focus=focus))
        except _OpenSettings as request:
            messages += await run_settings_menu(menu.settings_menu(request.model), runners)
            focus = request.model


class _ConnectProvider(Exception):
    """Release the menu worker before prompting or awaiting provider discovery."""

    def __init__(self, messages: list[str], *, provider: str) -> None:
        self.provider = provider
        self.messages = messages
        super().__init__()


class _OpenSettings(Exception):
    """Release the menu worker before a model's settings menu opens, as `/plugins` does for a plugin's."""

    def __init__(self, *, menu: ModelMenu, model: str) -> None:
        self.menu = menu
        self.model = model
        super().__init__()


async def add_model(context: CommandContext, dialogs: Dialogs) -> list[str]:
    """`/model add` through `Dialogs`: a provider, one of its catalog models, and its account, then select it.

    Providers that connect first (GitHub Copilot, OpenRouter, vLLM) sign in or ask for an address in the
    terminal, so they are left to `/model add` there.
    """
    from pydantic_clai2.ui.menus.accounts_menu import account_options

    menu = await to_thread.run_sync(ModelMenu, context)
    providers = [provider for provider in menu.providers() if provider not in _CONNECTS]
    provider = provider_of(menu.current or '')
    while True:
        picked = await dialogs.choose(
            title='Add a model: provider',
            options=[Option(value=name, label=name) for name in providers],
            current=provider if provider in providers else None,
            description='Connect GitHub Copilot, OpenRouter, or vLLM with /model add in the terminal.',
        )
        if picked is None:
            return []
        provider = picked
        name = await dialogs.choose(
            title=f'Add a model: {provider}',
            options=[
                Option(
                    value=model.name,
                    label=model.name,
                    description=' · '.join(fact for fact in (model.label, _tokens(model.context_window)) if fact),
                )
                for model in menu.models
                if model.provider == provider
            ],
            current=menu.current,
        )
        if name is None:
            continue
        accounts = await to_thread.run_sync(account_options, context.store, name)
        if accounts:
            name = await dialogs.choose(title=f'Which {provider} account?', options=accounts)
            if name is None:
                continue
        return [await to_thread.run_sync(menu.choose, name)]


async def model_settings_command(context: CommandContext, args: list[str], *, runners: Runners = TERMINAL) -> str:
    """Pick a saved model to edit, or open a named one, without switching models."""
    if len(args) > 1:
        raise ValueError('Usage: /model settings [NAME]')
    if not args:
        messages = await _pick_model_settings(context, runners)
        return '\n'.join(messages) or 'No changes.'
    name = args[0]
    if name not in context.store.models():
        raise ValueError(f'Model not added: {name}. Use /model add {name} first.')
    messages = await run_settings_menu(
        model_settings_menu(context.store, name, settings_as=context.settings_model(name)), runners
    )
    return '\n'.join(messages) or 'No changes.'


def model_settings_menu(store: SettingsStore, model: str, *, settings_as: str | None = None) -> SettingsMenu:
    """One model's settings, as `/model settings NAME`, the `/model add` list's `Ctrl+S`, and the web UI show them.

    Edits save under `model`, offering and validating the controls of `settings_as` (default `model`).
    """
    return SettingsMenu(
        source=ModelSettingsSource(store, model, settings_as=settings_as),
        actions={'custom_params': partial(edit_custom_params, store=store, model=model)},
        searchable=False,
    )


def build_model_settings_picker(*, context: CommandContext, current: str | None) -> Menu:
    """Choose a model to configure without changing the active run model."""
    names = context.store.models()

    return (
        MenuBuilder('Model Settings - Select a Model')
        .style(markdown_style())
        .items(
            [MenuItem(name, value=name) for name in names]
            or [MenuItem('No models added. Use /model add first.', disabled=True)]
        )
        .searchable()
        .list_width(40)
        .initial_index(names.index(current) if current in names else 0)
        .preview(
            lambda item: model_settings_summary(
                store=context.store, model=str(item.value), settings_as=context.settings_model(str(item.value))
            )
        )
        .footer_hint('type filter - Enter configure - Esc exit')
        .key_source(menu_key)
        .build()
    )


def model_settings_summary(*, store: SettingsStore, model: str, settings_as: str | None = None) -> str:
    """Preview effective settings without mutating the model or its saved overrides."""
    values = {**model_defaults(model=settings_as or model), **store.model_settings(model)}
    # Never empty: every model has at least CLAI's prompt caching default.
    lines = [model, '', 'Configured settings:']
    lines.extend(f'{setting_label(key)}: {shown(value)}' for key, value in values.items())
    return '\n'.join(lines)


async def _pick_model_settings(context: CommandContext, runners: Runners) -> list[str]:
    """Return from a model's editor to the picker, preserving its selection."""
    current = context.settings.model
    messages: list[str] = []
    while True:
        result = await run_worker(
            lambda: runners.run_list(build_model_settings_picker(context=context, current=current))
        )
        if result.cancelled or result.item is None or not isinstance(result.item.value, str):
            return messages
        current = result.item.value
        if current not in context.store.models():
            return messages
        messages += await run_settings_menu(
            model_settings_menu(context.store, current, settings_as=context.settings_model(current)), runners
        )
