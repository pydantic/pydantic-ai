"""Codex speeds: CLAI defaults `openai-codex` models to ultrafast, and `/fast` and the settings menu pick another."""

import json
from io import StringIO
from pathlib import Path

import httpx2 as httpx
import pytest
from pydantic import JsonValue, TypeAdapter
from rich.console import Console
from termflow.tui.completion import CompleteEvent, Document
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexProvider
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell
from pydantic_clai2.ui.menus.field_menu import FieldMenu, shown
from pydantic_clai2.ui.menus.model_menu import ModelSettingsSource, model_settings_command
from tests.clai2.menu_script import Script, make_context, pick


def _shell(store: SettingsStore, model: str | None, agent_model: str = 'test') -> SessionRuntime[None, str]:
    return create_shell(
        Agent(TestModel(model_name=agent_model)),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=Settings(model=model),
        store=store,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )


@pytest.mark.parametrize(
    ('name', 'default'),
    [('gpt-6-astra', 'ultrafast'), ('gpt-6.2-sol', 'ultrafast'), ('gpt-5.6-luna', None), ('gpt-5.4', None)],
)
def test_codex_speed_row(tmp_path: Path, name: str, default: str | None) -> None:
    """Ultrafast is the default speed only of the Codex models that offer it."""
    context, _ = make_context(tmp_path)
    model = f'openai-codex:{name}'
    source = ModelSettingsSource(context.store, model)
    row = FieldMenu(source).row_for('codex_speed')
    assert row is not None
    assert row.label == 'Speed'
    assert row.choices == ('standard', 'fast', 'ultrafast')
    assert 'more ChatGPT credits' in row.description
    assert source.current(row) == row.default == shown(default)
    assert (context.model_settings(model) or {}).get('openai_service_tier') == default
    assert source.apply(row, 'fast').startswith('Saved')
    assert (context.model_settings(model) or {}).get('openai_service_tier') == 'priority'
    source.reset(row)
    assert context.store.model_settings(model) == {}
    assert source.current(row) == shown(default)


@pytest.mark.parametrize('provider', ['openai', 'openai-chat', 'openai-responses', 'openrouter'])
def test_other_providers_have_no_codex_speed(tmp_path: Path, provider: str) -> None:
    context, _ = make_context(tmp_path)
    model = f'{provider}:gpt-6-astra'
    menu = FieldMenu(ModelSettingsSource(context.store, model))
    assert menu.row_for('codex_speed') is None
    tier = menu.row_for('service_tier')
    assert tier is not None and tier.label == 'Service Tier' and not tier.choice_labels
    assert 'openai_service_tier' not in (context.model_settings(model) or {})


async def test_model_settings_flow_saves_speed_without_selecting_model(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    original = context.settings.model
    model = 'openai-codex:gpt-6-astra'
    script = Script(lists=[pick('codex_speed'), MenuResult(cancelled=True)], choices=[pick('fast')], texts=[])
    assert 'Saved codex_speed' in await model_settings_command(context, [model], runners=script.runners)
    assert context.store.model_settings(model) == {'codex_speed': 'fast'}
    assert context.settings.model == original and not applied


@pytest.mark.parametrize(
    ('model', 'saved', 'command', 'tier'),
    [
        pytest.param('gpt-6-astra', {}, None, 'ultrafast', id='default'),
        pytest.param('gpt-5.6-luna', {}, None, 'default', id='default-without-ultrafast'),
        pytest.param('gpt-5.6-luna', {}, '/fast', 'priority', id='toggle-without-ultrafast'),
        pytest.param('gpt-6-astra', {}, '/fast off', 'default', id='off'),
        pytest.param('gpt-6-astra', {}, '/fast on', 'priority', id='on'),
        pytest.param('gpt-6-astra', {'codex_speed': 'standard'}, '/fast ultrafast', 'ultrafast', id='ultrafast'),
        pytest.param('gpt-6-astra', {'service_tier': 'priority'}, None, 'priority', id='saved-tier'),
    ],
)
async def test_speed_reaches_codex_request(
    tmp_path: Path, model: str, saved: dict[str, JsonValue], command: str | None, tier: str
) -> None:
    context, _ = make_context(tmp_path)
    name = f'openai-codex:{model}'
    context.store.save_model_settings(name, saved)
    if command is not None:
        await _shell(context.store, name).commands.execute_async(command)
    bodies: list[dict[str, JsonValue]] = []
    adapter = TypeAdapter(dict[str, JsonValue])

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(adapter.validate_json(request.content))
        response: dict[str, JsonValue] = {
            'id': 'resp_test',
            'created_at': 1,
            'model': 'gpt-6-astra',
            'status': 'completed',
            'output': [],
        }
        events: list[dict[str, JsonValue]] = [
            {'type': 'response.created', 'sequence_number': 0, 'response': response},
            {
                'type': 'response.output_item.added',
                'sequence_number': 1,
                'output_index': 0,
                'item': {'type': 'message', 'id': 'msg_test', 'role': 'assistant', 'content': []},
            },
            {
                'type': 'response.output_text.delta',
                'sequence_number': 2,
                'item_id': 'msg_test',
                'output_index': 0,
                'content_index': 0,
                'delta': 'done',
            },
            {'type': 'response.completed', 'sequence_number': 3, 'response': response},
        ]
        return httpx.Response(
            200,
            headers={'content-type': 'text/event-stream'},
            text=''.join(f'data: {json.dumps(event)}\n\n' for event in events),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = OpenAICodexProvider(
            credentials=OpenAICodexCredentials(access_token='test', refresh_token='test', account_id='test'),
            http_client=client,
        )
        with override_allow_model_requests(True):
            result = await Agent(OpenAICodexModel(model, provider=provider)).run(
                'hello', model_settings=context.model_settings(name)
            )
    assert result.output == 'done'
    assert len(bodies) == 1
    assert bodies[0]['service_tier'] == tier
    assert bodies[0]['stream'] is True and bodies[0]['store'] is False
    reasoning = bodies[0]['reasoning']
    assert isinstance(reasoning, dict) and reasoning['effort'] == 'medium'


@pytest.mark.parametrize('model', ['openai-codex:gpt-6-astra', 'openai-codex:xyz', 'openai:gpt-6-astra', 'test', None])
async def test_fast_command_tracks_active_model(tmp_path: Path, model: str | None) -> None:
    """Exercise the shell registry without credentials or model requests."""
    shell = _shell(SettingsStore(tmp_path / 'config.db'), model)
    available = model is not None and model.startswith('openai-codex:')
    assert ('/fast:' in await shell.commands.execute_async('/help')) is available
    assert ('fast' in [c.text for c in shell.commands.get_completions(Document('/fa'), CompleteEvent())]) is available
    assert [c.text for c in shell.commands.get_completions(Document('/fast '), CompleteEvent())] == (
        ['on', 'off', 'ultrafast'] if available else []
    )
    assert list(shell.commands.get_completions(Document('/fast on '), CompleteEvent())) == []
    assert not shell.commands.runs_during_turn('/fast')
    if not available:
        with pytest.raises(ValueError, match='Unknown command /fast'):
            await shell.commands.execute_async('/fast')
    for command in ('/set model openai-codex:xyz', '/model add openai-codex:xyz', '/model openai-codex:xyz'):
        await shell.commands.execute_async(command)
        assert '/fast:' in await shell.commands.execute_async('/help')
        assert 'Codex speed fast' in await shell.commands.execute_async('/fast on')
        await shell.commands.execute_async('/set model test')
        assert '/fast:' not in await shell.commands.execute_async('/help')
        with pytest.raises(ValueError, match='Unknown command /fast'):
            await shell.commands.execute_async('/fast off')


_TIERS = {'standard': 'default', 'fast': 'priority', 'ultrafast': 'ultrafast'}


async def test_fast_toggle_preserves_other_preferences(tmp_path: Path) -> None:
    model = 'openai-codex:xyz'
    saved: dict[str, JsonValue] = {'openai_reasoning_effort': 'high', 'future_setting': {'value': 1}}
    store = SettingsStore(tmp_path / 'config.db')
    store.save_model_settings(model, saved)
    shell = _shell(store, None, agent_model=model)
    for command, speed in [
        ('/fast', 'fast'),
        ('/fast', 'standard'),
        ('/fast ultrafast', 'ultrafast'),
        ('/fast', 'standard'),
        ('/fast on', 'fast'),
        ('/fast off', 'standard'),
        ('/fast ultrafast', 'ultrafast'),
    ]:
        result = await shell.commands.execute_async(command)
        assert f'Codex speed {speed} for {model}' in result
        assert ('more ChatGPT credits' in result) is (speed != 'standard')
        assert SettingsStore(store.path).model_settings(model) == {
            **saved,
            'codex_speed': speed,
            'service_tier': 'default' if speed == 'standard' else 'priority',
        }
        assert (shell.context.model_settings(model) or {}).get('openai_service_tier') == _TIERS[speed]
    for command in ('/fast yes', '/fast on off'):
        with pytest.raises(ValueError, match='Usage: /fast'):
            await shell.commands.execute_async(command)
        assert store.model_settings(model) == {**saved, 'codex_speed': 'ultrafast', 'service_tier': 'priority'}
    store.save_model_settings(model, {**saved, 'custom_params': {'service_tier': 'flex'}})
    with pytest.raises(ValueError, match='Custom service_tier overrides the Codex speed'):
        await shell.commands.execute_async('/fast')
    assert store.model_settings(model) == {**saved, 'custom_params': {'service_tier': 'flex'}}


async def test_fast_toggles_ultrafast_where_offered(tmp_path: Path) -> None:
    model = 'openai-codex:gpt-6-astra'
    store = SettingsStore(tmp_path / 'config.db')
    shell = _shell(store, model)
    assert 'Codex speed standard' in await shell.commands.execute_async('/fast')
    assert 'Codex speed ultrafast' in await shell.commands.execute_async('/fast')
    assert store.model_settings(model) == {'codex_speed': 'ultrafast', 'service_tier': 'priority'}


async def test_fast_toggle_turns_a_saved_priority_tier_off(tmp_path: Path) -> None:
    """A `service_tier` that `/fast on` saved before Codex speeds counts as on."""
    model = 'openai-codex:gpt-6-astra'
    store = SettingsStore(tmp_path / 'config.db')
    store.save_model_settings(model, {'service_tier': 'priority'})
    result = await _shell(store, model).commands.execute_async('/fast')
    assert 'Codex speed standard' in result
    assert store.model_settings(model) == {'service_tier': 'default', 'codex_speed': 'standard'}
