"""Codex speed choices retain the existing service-tier storage and request contract."""

import json
from pathlib import Path

import httpx2 as httpx
import pytest
from menu_script import Script, make_context, pick
from pydantic import JsonValue, TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexProvider
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2.field_menu import FieldMenu
from pydantic_clai2.model_menu import ModelSettingsSource, model_settings_command


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize('name', ['gpt-6-astra', 'gpt-5.6-luna', 'gpt-5.4'])
def test_codex_speed_labels_do_not_change_values(tmp_path: Path, name: str) -> None:
    context, _ = make_context(tmp_path)
    model = f'openai-codex:{name}'
    source = ModelSettingsSource(context.store, model)
    menu = FieldMenu(source)
    row = menu.row_for('service_tier')
    assert row is not None
    assert row.label == 'Service Tier / Fast Mode'
    assert row.choices == ('auto', 'default', 'flex', 'priority')
    assert 'more ChatGPT credits' in row.description
    assert 'Reasoning effort' in row.description
    assert source.apply(row, 'priority').startswith('Saved')
    selected = menu.build_choices(row).highlighted
    assert selected is not None
    assert selected.value == 'priority' and selected.label == 'Fast (priority) (current)'
    assert 'current  Fast (priority)' in menu.details(MenuItem('', value=row.key))
    assert any('Fast (priority)' in item.label for item in menu.items())
    assert source.current(row) == 'priority'
    settings = context.model_settings(model)
    assert settings is not None and settings.get('service_tier') == 'priority'
    assert source.apply(row, 'default').startswith('Saved')
    assert menu.build_choices(row).highlighted == MenuItem('Standard (default) (current)', value='default')
    assert source.problem(row, 'fast') is not None  # Display labels are not new stored values.
    source.reset(row)
    assert context.store.model_settings(model) == {}


@pytest.mark.parametrize('provider', ['openai', 'openai-chat', 'openai-responses', 'openrouter'])
def test_other_providers_keep_original_tier_labels(tmp_path: Path, provider: str) -> None:
    context, _ = make_context(tmp_path)
    source = ModelSettingsSource(context.store, f'{provider}:gpt-6-astra')
    menu = FieldMenu(source)
    row = menu.row_for('service_tier')
    assert row is not None
    assert row.label == 'Service Tier' and not row.choice_labels
    assert source.apply(row, 'priority').startswith('Saved')
    assert menu.build_choices(row).highlighted == MenuItem('priority (current)', value='priority')


async def test_model_settings_flow_saves_speed_without_selecting_model(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    original = context.settings.model
    model = 'openai-codex:gpt-6-astra'
    script = Script(lists=[pick('service_tier'), MenuResult(cancelled=True)], choices=[pick('priority')], texts=[])
    assert 'Saved service_tier' in await model_settings_command(context, [model], runners=script.runners)
    assert context.store.model_settings(model) == {'service_tier': 'priority'}
    assert context.settings.model == original and not applied


@pytest.mark.parametrize('tier', ['priority', 'default'])
async def test_saved_speed_reaches_codex_request(tmp_path: Path, tier: str) -> None:
    context, _ = make_context(tmp_path)
    name = 'openai-codex:gpt-6-astra'
    source = ModelSettingsSource(context.store, name)
    row = FieldMenu(source).row_for('service_tier')
    assert row is not None
    source.apply(row, tier)
    bodies: list[dict[str, JsonValue]] = []
    adapter = TypeAdapter(dict[str, JsonValue])

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith('/responses')
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
            result = await Agent(OpenAIResponsesModel('gpt-6-astra', provider=provider)).run(
                'hello', model_settings=context.model_settings(name)
            )
    assert result.output == 'done'
    assert len(bodies) == 1
    assert bodies[0]['service_tier'] == tier
    assert bodies[0]['stream'] is True and bodies[0]['store'] is False
    reasoning = bodies[0]['reasoning']
    assert isinstance(reasoning, dict) and reasoning['effort'] == 'medium'
