"""The `/model` menu, the catalog behind it, and per-model settings."""

from pathlib import Path

import pytest
from menu_script import Script, make_context, pick, typed
from pydantic import JsonValue, ValidationError
from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import Session
from pydantic_clai2.field_menu import FieldMenu
from pydantic_clai2.model_catalog import catalog, genai_prices_models, runnable_providers
from pydantic_clai2.model_menu import ModelMenu, ModelSettingsSource, open_model_menu, run_model_flow
from pydantic_clai2.model_settings import ModelSettingsForm, model_settings_from_json
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def test_catalog_merges_sources_and_only_lists_runnable_providers() -> None:
    providers = runnable_providers()
    assert {'openai', 'anthropic', 'google', 'openai-codex'} <= providers
    priced = genai_prices_models()
    assert priced and all(model.provider in providers for model in priced)
    assert all(':' not in model.name.partition(':')[2] for model in priced)
    priced_names = {model.name for model in priced}
    merged = catalog(include=['openai-codex:gpt-6-astra', ''])
    names = [model.name for model in merged]
    assert names == sorted(set(names))
    assert priced_names <= set(names)
    astra = next(model for model in merged if model.name == 'openai-codex:gpt-6-astra')
    assert astra.provider == 'openai-codex' and astra.context_window is None and astra.prices is None
    sonnet = next(model for model in merged if model.name.startswith('anthropic:claude') and model.prices)
    assert sonnet.context_window and 'MTok' in (sonnet.prices or '')


def test_settings_form_validates_and_converts() -> None:
    assert model_settings_from_json({}).to_model_settings() is None
    form = model_settings_from_json({'max_tokens': 100, 'thinking': 'high', 'parallel_tool_calls': False})
    assert form.to_model_settings() == {'max_tokens': 100, 'thinking': 'high', 'parallel_tool_calls': False}
    everything: dict[str, JsonValue] = {
        'max_tokens': 1,
        'temperature': 0.5,
        'top_p': 0.9,
        'top_k': 4,
        'seed': 2,
        'timeout': 30.0,
        'presence_penalty': 0.1,
        'frequency_penalty': -0.1,
        'parallel_tool_calls': True,
        'thinking': True,
        'service_tier': 'flex',
    }
    assert model_settings_from_json(everything).to_model_settings() == everything
    with pytest.raises(ValidationError):
        ModelSettingsForm(max_tokens=0)
    with pytest.raises(ValidationError):
        model_settings_from_json({'nope': 1})


def test_store_round_trips_model_settings(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    assert store.model_settings('x') == {}
    store.save_model_settings('x', {'max_tokens': 5})
    assert SettingsStore(store.path).model_settings('x') == {'max_tokens': 5}
    store.save_model_settings('x', {})
    assert store.model_settings('x') == {}


def test_model_settings_source(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    source = ModelSettingsSource(store, 'openai:gpt-5')
    menu = FieldMenu(source)
    keys = [row.key for row in menu.rows]
    assert keys[:3] == ['max_tokens', 'temperature', 'top_p']
    thinking = menu.row_for('thinking')
    assert thinking is not None and thinking.choices == ('true', 'false', 'minimal', 'low', 'medium', 'high', 'xhigh')
    tier = menu.row_for('service_tier')
    assert tier is not None and tier.choices == ('auto', 'default', 'flex', 'priority')
    max_tokens = menu.rows[0]
    assert max_tokens.choices == () and source.title == 'Settings for openai:gpt-5'
    assert source.problem(max_tokens, '10') is None
    assert source.problem(max_tokens, '0') == 'Input should be greater than 0'
    assert source.problem(max_tokens, 'ten') is not None
    assert source.apply(max_tokens, '10') == 'Saved max_tokens for openai:gpt-5. Applies when this model is selected.'
    assert source.apply(thinking, 'high').startswith('Saved thinking')
    assert source.apply(max_tokens, '-1') == 'max_tokens: Input should be greater than 0'
    assert store.model_settings('openai:gpt-5') == {'max_tokens': 10, 'thinking': 'high'}
    assert source.current(max_tokens) == '10' and source.current(thinking) == 'high'
    assert source.reset(max_tokens) == 'Reset max_tokens for openai:gpt-5.'
    assert store.model_settings('openai:gpt-5') == {'thinking': 'high'}
    assert 'current  high' in menu.details(MenuItem('thinking', value='thinking'))


def test_provider_catalog_and_back_navigation(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    menu = ModelMenu(context)
    assert menu.providers() == sorted({model.name.partition(':')[0] for model in menu.models} | {'openrouter', 'vllm'})
    codex = menu.for_provider('openai-codex')
    assert all(model.provider == 'openai-codex' for model in codex.models)
    assert {f'openai-codex:gpt-5.6-{suffix}' for suffix in ('luna', 'terra', 'sol')} <= {
        model.name for model in codex.models
    }
    assert menu.build_providers().highlighted == MenuItem('openai-codex', value='openai-codex')
    script = Script(
        lists=[pick('anthropic'), MenuResult(cancelled=True), pick('openai-codex'), pick('openai-codex:gpt-5.6-luna')],
        choices=[],
        texts=[],
    )
    assert run_model_flow(menu, script.runners) == ['Saved model. Applied.']
    assert context.settings.model == 'openai-codex:gpt-5.6-luna'
    assert run_model_flow(menu, Script(lists=[pick('openai-codex'), pick(0)], choices=[], texts=[]).runners) == []


def test_settings_shortcut_does_not_consume_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)
    menu = ModelMenu(context)
    keys = iter(['s', 'ctrl-s'])
    monkeypatch.setattr('pydantic_clai2.model_menu.menu_key', lambda: next(keys))
    widget = menu.build()
    result = widget.run()
    assert widget.highlighted is not None
    assert 's' in str(widget.highlighted.value).lower()
    assert result == menu.settings_marker(widget, widget.highlighted)
    with pytest.raises(StopIteration):
        next(keys)


def test_model_menu_rows_details_and_flow(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    menu = ModelMenu(context)
    assert menu.current == 'openai-codex:gpt-6-astra'
    items = menu.items()
    current = next(item for item in items if item.value == menu.current)
    assert current.label == f'{menu.current} (current)'
    assert all(item.label == item.value for item in items if item.value != menu.current)
    assert menu.index_of(menu.current) == items.index(current)
    assert menu.index_of('nope') == 0
    details = menu.details(current)
    assert 'provider  openai-codex' in details and 'context   unknown' in details and 'settings  none' in details
    priced = next(item for item in items if str(item.value).startswith('anthropic:claude'))
    assert 'tokens' in menu.details(priced) and 'MTok' in menu.details(priced)
    assert menu.details(MenuItem('stray', value='nope')) == ''
    assert menu.build() is not None
    marker = menu.settings_marker(object(), priced)
    script = Script(
        lists=[pick('anthropic'), marker, pick('max_tokens'), MenuResult(cancelled=True), pick(priced.value)],
        choices=[],
        texts=[typed('42')],
    )
    messages = run_model_flow(menu, script.runners)
    assert messages == [
        f'Saved max_tokens for {priced.value}. Applies when this model is selected.',
        'Saved model. Applied.',
    ]
    assert context.settings.model == priced.value
    assert applied == ['model']
    assert 'settings  max_tokens=42' in menu.details(priced)
    assert run_model_flow(menu, Script(lists=[MenuResult(cancelled=True)], choices=[], texts=[]).runners) == []
    assert run_model_flow(menu, Script(lists=[pick(0)], choices=[], texts=[]).runners) == []


async def test_open_model_menu_and_settings_reach_the_run(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    assert await open_model_menu(context, run=lambda menu: []) == 'No changes.'
    assert await open_model_menu(context, run=lambda menu: [menu.choose('test')]) == 'Saved model. Applied.'
    context.store.save_model_settings('test', {'max_tokens': 3, 'seed': 7})
    assert context.model_settings('test') == {'max_tokens': 3, 'seed': 7}
    assert context.model_settings('other') is None
    seen: list[ModelSettings | None] = []
    hooks = Hooks[None]()

    @hooks.on.before_model_request
    async def capture(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        seen.append(request_context.model_settings)
        return request_context

    session = Session(Agent(TestModel(custom_output_text='ok')), deps=None, plugins=[hooks])
    session.model_settings = context.model_settings('test')
    assert (await session.prompt('hi')).output == 'ok'
    assert seen == [{'max_tokens': 3, 'seed': 7}]
