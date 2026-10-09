"""The `/model` commands and menus, the catalog behind it, and per-model settings."""

import sys
from io import StringIO
from pathlib import Path

import pytest
from pydantic import JsonValue, ValidationError
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.completion import CompleteEvent, Document
from termflow.tui.menu import Menu, MenuResult

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.models import infer_model
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_clai2 import Session
from pydantic_clai2._app import create_shell
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models.model_catalog import catalog, genai_prices_models, runnable_providers
from pydantic_clai2.models.model_settings import ModelSettingsForm, model_settings_from_json
from pydantic_clai2.ui.menus.field_menu import FieldMenu, Runners
from pydantic_clai2.ui.menus.model_menu import ModelMenu, ModelSettingsSource, open_add_model_menu, run_model_flow
from pydantic_clai2.ui.menus.model_picker import (
    DeleteModel,
    ModelPickerAction,
    build_model_picker,
    model_command,
    model_completions,
)
from pydantic_clai2.ui.menus.set_menu import SettingsSource
from tests.clai2.menu_script import Script, make_context, pick, typed


def test_catalog_merges_sources_and_only_lists_runnable_providers() -> None:
    providers = runnable_providers()
    assert {'openai', 'anthropic', 'google', 'openai-codex', 'github-copilot'} <= providers
    priced = genai_prices_models()
    assert priced and all(model.provider in providers for model in priced)
    assert all(':' not in model.name.partition(':')[2] for model in priced)
    priced_names = {model.name for model in priced}
    merged = catalog(include=['openai-codex:gpt-6-astra', ''])
    names = [model.name for model in merged]
    assert names == sorted(set(names))
    assert priced_names <= set(names)
    assert {'openai-codex:gpt-6.1-sol', 'openai-codex:gpt-6-sol', 'openai-codex:gpt-6-luna'} <= set(names)
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
    assert model_settings_from_json({'nope': 1}).to_model_settings() is None
    with pytest.raises(ValidationError):
        ModelSettingsForm.model_validate({'nope': 1})


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
    assert keys[:3] == ['max_tokens', 'thinking', 'service_tier']
    assert not {'temperature', 'top_p', 'seed', 'timeout', 'top_k'} & set(keys)
    thinking = menu.row_for('thinking')
    assert thinking is not None and thinking.choices == ('true', 'false', 'minimal', 'low', 'medium', 'high', 'xhigh')
    tier = menu.row_for('service_tier')
    assert tier is not None and tier.choices == ('auto', 'default', 'flex', 'priority')
    max_tokens = menu.rows[0]
    assert max_tokens.choices == () and source.title == 'Settings - openai:gpt-5'
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
    assert menu.providers() == sorted(
        {model.name.partition(':')[0] for model in menu.models} | {'github-copilot', 'openrouter', 'vllm'}
    )
    codex = menu.for_provider('openai-codex')
    assert all(model.provider == 'openai-codex' for model in codex.models)
    assert {f'openai-codex:gpt-5.6-{suffix}' for suffix in ('luna', 'terra', 'sol')} <= {
        model.name for model in codex.models
    }
    assert menu.build_providers().highlighted == MenuItem('openai-codex', value='openai-codex')
    profiled, _ = make_context(tmp_path / 'profiled')
    profiled.settings = Settings(model='openai@work:gpt-5')
    profiled_menu = ModelMenu(profiled)
    assert profiled_menu.build_providers().highlighted == MenuItem('openai', value='openai')
    # Another account's model is listed under its provider, not as a provider of its own.
    assert 'openai@work' not in profiled_menu.providers()
    assert 'openai@work:gpt-5' in [model.name for model in profiled_menu.for_provider('openai').models]
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
    monkeypatch.setattr('pydantic_clai2.ui.menus.model_menu.menu_key', lambda: next(keys))
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


async def test_open_add_model_menu_and_settings_reach_the_run(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    assert await open_add_model_menu(context, run=lambda menu: []) == 'No changes.'
    assert await open_add_model_menu(context, run=lambda menu: [menu.choose('test')]) == 'Saved model. Applied.'
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


def test_live_model_overrides_follow_saves_and_keep_the_last_valid_ones(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    live = context.live_model_overrides('test')
    assert live(None) == {}
    context.store.save_model_settings('test', {'seed': 7})
    assert live(None) == {'seed': 7}
    context.store.save_model_settings('test', {'temperature': 'hot'})
    assert live(None) == {'seed': 7}
    with pytest.raises(ValidationError):
        context.live_model_overrides('test')


def test_added_models_persist_independently_of_settings(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    original = context.settings.model
    assert original is not None
    assert context.store.models() == [original]
    ModelMenu(context).choose('test')
    context.store.add_model(name='test')
    context.store.save_model_settings('test', {'seed': 1})
    context.store.save_model_settings('test', {})
    assert SettingsStore(context.store.path).models() == sorted([original, 'test'])
    assert context.store.load().model == 'test'


async def test_saved_model_picker_and_completion(tmp_path: Path) -> None:

    context, applied = make_context(tmp_path)
    original = context.settings.model
    assert original is not None
    context.store.add_model(name='test')
    assert model_completions(context, ['']) == ['add', 'settings', 'chains', *sorted([original, 'test'])]
    assert model_completions(context, ['settings', '']) == sorted([original, 'test'])
    assert 'openai-codex:' in model_completions(context, ['add', ''])
    assert model_completions(context, ['test', 'extra']) == []
    widget = build_model_picker(context)
    assert widget.highlighted == MenuItem(f'{original} (current)', value=original)
    script = Script(lists=[pick('test')], choices=[], texts=[])
    assert await model_command(context, [], runners=script.runners) == 'Saved model. Applied.'
    assert context.settings.model == 'test' and applied == ['model']
    assert await model_command(context, [original]) == 'Saved model. Applied.'
    with pytest.raises(ValueError, match=r'Usage: /model \[NAME\] \| /model add \[NAME\]'):
        await model_command(context, ['test', 'extra'])
    with pytest.raises(ValueError, match='Usage: /model'):
        await model_command(context, ['add', 'test', 'extra'])


@pytest.mark.parametrize('args', [['unsaved:model'], ['add', 'unsaved:model']])
async def test_model_name_not_yet_saved_is_added_and_selected(tmp_path: Path, args: list[str]) -> None:
    context, applied = make_context(tmp_path)
    assert await model_command(context, args) == 'Saved model. Applied.'
    assert context.settings.model == 'unsaved:model' and applied == ['model']
    reopened = SettingsStore(context.store.path)
    assert 'unsaved:model' in reopened.models() and reopened.load().model == 'unsaved:model'


async def test_model_add_subcommand_opens_add_flow(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    script = Script(lists=[pick('anthropic'), pick('anthropic:claude-sonnet-4-5')], choices=[], texts=[])
    assert await model_command(context, ['add'], runners=script.runners) == 'Saved model. Applied.'
    assert context.settings.model == 'anthropic:claude-sonnet-4-5' and applied == ['model']


async def test_model_settings_subcommand(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    model = context.settings.model
    assert model is not None
    direct = Script(lists=[pick('max_tokens'), MenuResult(cancelled=True)], choices=[], texts=[typed('42')])
    assert await model_command(context, ['settings', model], runners=direct.runners) == (
        f'Saved max_tokens for {model}. Applies when this model is selected.'
    )
    assert context.store.model_settings(model) == {'max_tokens': 42}
    picked = Script(lists=[MenuResult(cancelled=True)], choices=[], texts=[])
    assert await model_command(context, ['settings'], runners=picked.runners) == 'No changes.'
    with pytest.raises(ValueError, match=r'Model not added: unsaved:model. Use /model add unsaved:model first.'):
        await model_command(context, ['settings', 'unsaved:model'])
    with pytest.raises(ValueError, match=r'Usage: /model settings \[NAME\]'):
        await model_command(context, ['settings', model, 'extra'])
    assert context.settings.model == model and applied == []


@pytest.mark.parametrize('result', [MenuResult(cancelled=True), MenuResult(), pick(0)])
async def test_saved_model_picker_cancel(tmp_path: Path, result: MenuResult) -> None:

    context, applied = make_context(tmp_path)
    script = Script(lists=[result], choices=[], texts=[])
    assert await model_command(context, [], runners=script.runners) == 'No changes.'
    assert applied == []


def test_empty_model_picker(tmp_path: Path) -> None:

    context = CommandContext(
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'empty.db'),
        clear_history=lambda: None,
        apply_setting=lambda key, settings: None,
    )
    assert model_completions(context, []) == ['add', 'settings', 'chains']
    widget = build_model_picker(context)
    assert widget.highlighted is not None
    assert not widget.highlighted.disabled
    assert widget.highlighted.value is ModelPickerAction.ADD


@pytest.mark.parametrize('initial_model', [None, 'test'])
async def test_select_new_model_from_picker(tmp_path: Path, initial_model: str | None) -> None:
    applied: list[str] = []
    context = CommandContext(
        settings=Settings(model=initial_model),
        store=SettingsStore(tmp_path / 'config.db'),
        clear_history=lambda: None,
        apply_setting=lambda key, settings: applied.append(key),
    )
    assert context.store.models() == ([initial_model] if initial_model else [])
    script = Script(
        lists=[pick(ModelPickerAction.ADD), pick('anthropic'), pick('anthropic:claude-sonnet-4-5')],
        choices=[],
        texts=[],
    )
    assert await model_command(context, [], runners=script.runners) == 'Saved model. Applied.'
    assert context.settings.model == 'anthropic:claude-sonnet-4-5'
    assert SettingsStore(context.store.path).load().model == context.settings.model
    assert context.settings.model in model_completions(context, [])
    assert applied == ['model']


async def test_cancel_adding_from_picker(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    original = context.settings.model
    script = Script(lists=[pick(ModelPickerAction.ADD), MenuResult(cancelled=True)], choices=[], texts=[])
    assert await model_command(context, [], runners=script.runners) == 'No changes.'
    assert context.settings.model == original
    assert applied == []


@pytest.mark.parametrize('saved_default', [None, 'test', 'unused:model'])
def test_remove_model_persists_without_changing_other_preferences(tmp_path: Path, saved_default: str | None) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    if saved_default is not None:
        store.set('model', saved_default)
    store.set('display.thinking', False)
    settings: dict[str, JsonValue] = {'max_tokens': 5, 'custom_params': {'extra_body.key': 'value'}}
    for name in ['test', 'unused:model']:
        store.add_model(name=name)
        store.save_model_settings(name, settings)

    removable = saved_default != 'unused:model'
    assert store.remove_model(name='unused:model') is removable
    assert store.remove_model(name='unused:model') is removable
    reopened = SettingsStore(store.path)
    assert reopened.models() == (['test'] if removable else ['test', 'unused:model'])
    assert reopened.model_settings('unused:model') == ({} if removable else settings)
    assert reopened.model_settings('test') == settings
    expected: dict[str, JsonValue] = {'display.thinking': False}
    if saved_default is not None:
        expected['model'] = saved_default
    assert reopened.overrides() == expected
    reopened.add_model(name='unused:model')
    assert reopened.model_settings('unused:model') == ({} if removable else settings)


@pytest.mark.parametrize('delete_key', ['ctrl-d', 'delete'])
async def test_delete_model_through_picker_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], delete_key: str
) -> None:
    context, applied = make_context(tmp_path)
    original = context.settings.model
    assert original is not None
    context.store.set('model', original)
    context.store.add_model(name='unused:model')
    context.store.save_model_settings('unused:model', {'max_tokens': 5})
    pressed = iter([*'unused:', delete_key, 'down', 'enter', 'escape'])
    monkeypatch.setattr('pydantic_clai2.ui.menus.model_picker.menu_key', lambda: next(pressed))

    assert await model_command(context, []) == 'Deleted unused:model.'
    assert context.settings.model == context.store.load().model == original
    assert applied == []
    assert SettingsStore(context.store.path).models() == [original]
    assert context.store.model_settings('unused:model') == {}
    assert model_completions(context, ['settings', '']) == [original]
    assert 'Deleted unused:model.' in capsys.readouterr().out


def test_model_picker_search_still_accepts_d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)
    context.store.add_model(name='unused:model')
    pressed = iter([*'unused:', 'enter'])
    monkeypatch.setattr('pydantic_clai2.ui.menus.model_picker.menu_key', lambda: next(pressed))
    result = build_model_picker(context).run()
    assert result.item is not None and result.item.value == 'unused:model'


@pytest.mark.parametrize('delete_key', ['ctrl-d', 'delete'])
def test_cannot_delete_picker_status_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delete_key: str) -> None:
    context, _ = make_context(tmp_path)
    pressed = iter(['end', delete_key, 'escape'])
    monkeypatch.setattr('pydantic_clai2.ui.menus.model_picker.menu_key', lambda: next(pressed))
    assert build_model_picker(context, message='Deleted unused:model.').run().cancelled


@pytest.mark.parametrize('confirmation', [MenuResult(cancelled=True), MenuResult(), pick(False)])
async def test_cancel_deleting_model(tmp_path: Path, confirmation: MenuResult) -> None:
    context, applied = make_context(tmp_path)
    context.store.add_model(name='unused:model')
    context.store.save_model_settings('unused:model', {'max_tokens': 5})
    script = Script(
        lists=[pick(DeleteModel(name='unused:model')), MenuResult(cancelled=True)], choices=[confirmation], texts=[]
    )
    assert await model_command(context, [], runners=script.runners) == 'No changes.'
    assert 'unused:model' in context.store.models()
    assert context.store.model_settings('unused:model') == {'max_tokens': 5}
    assert applied == []
    assert script.opened == ['list', 'choice', 'list']


async def test_cannot_delete_current_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    context, applied = make_context(tmp_path)
    original = context.settings.model
    assert original is not None
    context.store.save_model_settings(original, {'max_tokens': 5})
    pressed = iter(['ctrl-d', 'escape'])
    monkeypatch.setattr('pydantic_clai2.ui.menus.model_picker.menu_key', lambda: next(pressed))
    assert await model_command(context, []) == 'No changes.'
    assert context.store.models() == [original]
    assert context.store.model_settings(original) == {'max_tokens': 5}
    assert context.settings.model == original and applied == []
    assert 'Select another model' in capsys.readouterr().out


async def test_saved_default_changed_during_delete_confirmation(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    context.store.add_model(name='unused:model')
    context.store.save_model_settings('unused:model', {'max_tokens': 5})
    script = Script(lists=[pick(DeleteModel(name='unused:model')), MenuResult(cancelled=True)], choices=[], texts=[])

    def confirm_after_default_changes(menu: Menu) -> MenuResult:
        SettingsStore(context.store.path).set('model', 'unused:model')
        return pick(True)

    runners = Runners(run_list=script.run_list, run_choice=confirm_after_default_changes, run_text=script.run_text)
    assert await model_command(context, [], runners=runners) == 'No changes.'
    assert 'unused:model' in context.store.models()
    assert context.store.model_settings('unused:model') == {'max_tokens': 5}
    assert context.store.load().model == 'unused:model' and applied == []


@pytest.mark.parametrize('current', [None, 'test'])
async def test_cannot_delete_saved_default_model(
    tmp_path: Path, current: str | None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.set('model', 'saved:default')
    store.save_model_settings('saved:default', {'max_tokens': 5})
    context = CommandContext(
        settings=Settings(model=current),
        store=store,
        clear_history=lambda: None,
        apply_setting=lambda key, settings: None,
    )
    pressed = iter(['home', 'ctrl-d', 'escape'])
    monkeypatch.setattr('pydantic_clai2.ui.menus.model_picker.menu_key', lambda: next(pressed))
    assert await model_command(context, []) == 'No changes.'
    output = capsys.readouterr().out
    assert 'saved:default (saved default)' in output
    assert 'This is your saved default model.' in output
    assert '/set model' in output
    assert 'Delete saved:default?' not in output
    assert 'saved:default' in store.models()
    assert store.model_settings('saved:default') == {'max_tokens': 5}
    assert store.load().model == 'saved:default'
    assert context.settings.model == current


@pytest.mark.parametrize('keys', [['enter'], ['escape'], ['ctrl-c']])
async def test_delete_confirmation_keeps_model_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keys: list[str]
) -> None:
    context, applied = make_context(tmp_path)
    context.store.add_model(name='unused:model')
    pressed = iter([*'unused:', 'ctrl-d', *keys, 'escape'])
    monkeypatch.setattr('pydantic_clai2.ui.menus.model_picker.menu_key', lambda: next(pressed))
    assert await model_command(context, []) == 'No changes.'
    assert 'unused:model' in context.store.models() and applied == []


@pytest.mark.parametrize('ending', [MenuResult(cancelled=True), MenuResult(), pick(0), pick('test')])
async def test_delete_then_leave_or_select_model(tmp_path: Path, ending: MenuResult) -> None:
    context, applied = make_context(tmp_path)
    context.store.add_model(name='unused:model')
    context.store.add_model(name='test')
    script = Script(lists=[pick(DeleteModel(name='unused:model')), ending], choices=[pick(True)], texts=[])
    selected = ending.item is not None and ending.item.value == 'test'
    expected = 'Deleted unused:model.' + ('\nSaved model. Applied.' if selected else '')
    assert await model_command(context, [], runners=script.runners) == expected
    assert 'unused:model' not in context.store.models()
    assert applied == (['model'] if selected else [])


async def test_delete_last_model_then_add_from_picker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, applied = make_context(tmp_path)
    context.settings = Settings(model=None)
    for name in context.store.models():
        context.store.remove_model(name=name)
    context.store.add_model(name='unused:model')
    pressed = iter(['ctrl-d', 'down', 'enter', 'ctrl-d', 'escape'])
    monkeypatch.setattr('pydantic_clai2.ui.menus.model_picker.menu_key', lambda: next(pressed))
    assert await model_command(context, []) == 'Deleted unused:model.'
    assert context.store.models() == [] and applied == []
    assert build_model_picker(context).highlighted == MenuItem('Add a model...', value=ModelPickerAction.ADD)
    script = Script(lists=[pick(ModelPickerAction.ADD), pick('test'), pick('test')], choices=[], texts=[])
    assert await model_command(context, [], runners=script.runners) == 'Saved model. Applied.'
    assert context.store.models() == ['test'] and applied == ['model']


async def test_select_model_whose_provider_extra_is_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip('typesafe_sdk', reason='needs the `typesafe` extra')
    monkeypatch.setenv('TYPESAFE_API_KEY', 'test-key')
    context, applied = make_context(tmp_path)
    context.store.add_model(name='typesafe:jev-latest')
    assert await model_command(context, ['typesafe:jev-latest']) == 'Saved model. Applied.'
    assert applied == ['model']
    assert infer_model('typesafe:jev-latest').system == 'typesafe'


@pytest.mark.parametrize(
    ('model', 'sdk', 'expected'),
    [
        pytest.param(
            'typesafe:jev-latest',
            'typesafe_sdk',
            [
                'Cannot use typesafe:jev-latest: the `typesafe` extra is not installed in the Python CLAI2 runs on',
                'pip install "pydantic-ai-slim[typesafe]"',
                'uv run --package pydantic-clai2 --extra typesafe clai2',
            ],
            id='clai2-extra',
        ),
        pytest.param(
            'groq:llama-3.3-70b-versatile',
            'groq',
            ['Cannot use groq:llama-3.3-70b-versatile with the Python CLAI2 runs on', 'pydantic-ai-slim[groq]'],
            id='slim-extra',
        ),
    ],
)
async def test_select_model_whose_provider_sdk_is_missing_fails_before_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, sdk: str, expected: list[str]
) -> None:
    monkeypatch.setitem(sys.modules, sdk, None)
    monkeypatch.delitem(sys.modules, f'pydantic_ai.providers.{model.partition(":")[0]}', raising=False)
    context, applied = make_context(tmp_path)
    original = context.settings.model
    context.store.add_model(name=model)
    with pytest.raises(ValueError) as exc_info:
        await model_command(context, [model])
    message = str(exc_info.value)
    assert all(part in message for part in expected), message
    assert sys.executable in message
    assert context.settings.model == original and context.store.load().model == original and applied == []
    model_row = FieldMenu(SettingsSource(context)).rows[0]
    assert model_row.key == 'model'
    assert SettingsSource(context).problem(model_row, model) == message
    assert SettingsSource(context).apply(model_row, model) == f'model: {message}'


async def test_shell_registers_model_subcommands_and_deprecated_aliases(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=Settings(model='test'),
        store=store,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    commands = shell.commands

    def completions(text: str) -> list[str]:
        return [c.text for c in commands.get_completions(Document(text), CompleteEvent())]

    help_text = await commands.execute_async('/help')
    assert '/add_model: Deprecated: use /model add' in help_text
    assert '/model_settings: Deprecated: use /model settings' in help_text
    assert '/chain: Alias of /model chains' in help_text
    assert completions('/model ') == ['add', 'settings', 'chains', 'test']
    assert completions('/model settings ') == completions('/model_settings ') == ['test']
    assert 'openai-codex:' in completions('/model add ') and completions('/model add ') == completions('/add_model ')
    mid_turn = ('/model', '/model add', '/model settings', '/model chains', '/add_model', '/model_settings', '/chain')
    assert all(commands.runs_during_turn(text) for text in mid_turn)
    assert not commands.runs_during_turn('/model add test')
    assert await commands.execute_async('/model unsaved:one') == 'Saved model. Applied.'
    assert await commands.execute_async('/add_model unsaved:two') == 'Saved model. Applied.'
    assert store.models() == ['test', 'unsaved:one', 'unsaved:two'] and store.load().model == 'unsaved:two'
    with pytest.raises(ValueError, match='Use /model add unsaved:three first'):
        await commands.execute_async('/model_settings unsaved:three')
