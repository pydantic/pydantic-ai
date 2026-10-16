r"""Create the ElevenLabs Agents a re-recording of the ElevenLabs Agents cassettes needs.

ElevenLabs Agents wraps a *hosted agent*, so the tests in `tests/realtime/test_elevenlabs_agents_ws.py` and the
`elevenlabs-agents` row of `tests/realtime/test_parity_ws.py` record against agents configured on an ElevenLabs
workspace. This script creates them, every one named `pydantic-ai-tests-*`, and prints the environment the tests read
their ids from:

    uv run --env-file .env python scripts/setup_elevenlabs_agents_test_workspace.py > .elevenlabs-agents-test.env
    uv run --env-file .env --env-file .elevenlabs-agents-test.env pytest --record-mode=rewrite \
        tests/realtime/test_elevenlabs_agents_ws.py -k 'not tool_sync'

The `elevenlabs_agents_tool_sync='sync'` tests list every client tool in the workspace, so record them one at a time
on a workspace holding nothing but these agents, in this order (each `-k` one of the `tool_sync` tests):

1. `creates`, against the toolless `sync-create` agent;
2. `python scripts/setup_elevenlabs_agents_test_workspace.py orphan`, which deletes the `sync-create` agent so the tool
   it got becomes an unattached orphan;
3. `adopts`, against the toolless `sync-adopt` agent (it adopts that orphan);
4. `updates`, against `sync-update`, whose `get_weather` carries a stale description;
5. `detaches`, against `sync-detach`, which holds `get_weather`, an extra client tool, and a webhook tool.

`python scripts/setup_elevenlabs_agents_test_workspace.py update` brings existing agents in line with this script,
and `python scripts/setup_elevenlabs_agents_test_workspace.py cleanup` deletes every `pydantic-ai-tests-*` agent.
Every agent is on 16 kHz PCM in and out, permits the per-conversation overrides the tests use, and sends the client
events the adapter reads.
"""

from __future__ import annotations as _annotations

import os
import sys
from typing import Any

import httpx

BASE_URL = os.environ.get('ELEVENLABS_BASE_URL', 'https://api.elevenlabs.io')
PREFIX = 'pydantic-ai-tests-'
TTS_MODEL = 'eleven_flash_v2'

WEATHER_TOOL: dict[str, Any] = {
    'type': 'client',
    'name': 'get_weather',
    'description': 'Look up the current weather for a city.',
    'expects_response': True,
    'parameters': {
        'type': 'object',
        'required': ['city'],
        'properties': {'city': {'type': 'string', 'description': 'City name'}},
    },
}

CLIENT_EVENTS = [
    'audio',
    'interruption',
    'agent_response',
    'user_transcript',
    'agent_response_correction',
    'agent_tool_response',
    'context_usage',
    'mcp_tool_call',
]

OVERRIDES: dict[str, Any] = {
    'conversation_config_override': {
        'agent': {'first_message': True, 'language': True, 'prompt': {'prompt': True, 'llm': True, 'tool_ids': True}},
        'conversation': {'text_only': True},
        'tts': {'voice_id': True, 'model_id': True, 'stability': True, 'speed': True, 'similarity_boost': True},
    },
    'custom_llm_extra_body': True,
}


def _client() -> httpx.Client:
    return httpx.Client(base_url=BASE_URL, headers={'xi-api-key': os.environ['ELEVENLABS_API_KEY']}, timeout=30)


def create_tool(client: httpx.Client, config: dict[str, Any]) -> str:
    """Create a workspace tool, returning its id."""
    response = client.post('/v1/convai/tools', json={'tool_config': config})
    response.raise_for_status()
    return response.json()['id']


def create_agent(client: httpx.Client, name: str, *, tool_ids: list[str] | None = None) -> str:
    """Create a `pydantic-ai-tests-<name>` agent, returning its id."""
    conversation_config: dict[str, Any] = {
        'agent': {
            'first_message': '',
            'language': 'en',
            'prompt': {'prompt': 'You are a helpful assistant.', 'llm': 'gemini-2.5-flash', 'tool_ids': tool_ids or []},
        },
        'asr': {'user_input_audio_format': 'pcm_16000'},
        # A TTS model without expressive tags, which would otherwise show up in the reply transcripts ("[happy] Hi.").
        'tts': {'agent_output_audio_format': 'pcm_16000', 'model_id': TTS_MODEL},
        # Long enough that the silence a microphone streams after a short reply doesn't prompt an "are you there?".
        'turn': {'turn_timeout': 30},
        'conversation': {'client_events': CLIENT_EVENTS},
    }
    response = client.post(
        '/v1/convai/agents/create',
        json={
            'name': PREFIX + name,
            'conversation_config': conversation_config,
            'platform_settings': {'overrides': OVERRIDES},
        },
    )
    response.raise_for_status()
    return response.json()['agent_id']


def setup() -> dict[str, str]:
    """Create the agents (and their tools), returning the environment the tests read their ids from."""
    with _client() as client:
        weather = create_tool(client, WEATHER_TOOL)
        stale = create_tool(client, {**WEATHER_TOOL, 'description': 'A stale description.'})
        extra = create_tool(
            client,
            {
                **WEATHER_TOOL,
                'name': 'get_time',
                'description': 'Get the time.',
                'parameters': {'type': 'object', 'properties': {}},
            },
        )
        webhook = create_tool(
            client,
            {
                'type': 'webhook',
                'name': 'crm_lookup',
                'description': 'Look a customer up in the CRM.',
                'api_schema': {'url': 'https://example.com/crm', 'method': 'GET'},
            },
        )
        return {
            'ELEVENLABS_TEST_AGENT_ID': create_agent(client, 'weather', tool_ids=[weather]),
            'ELEVENLABS_TEST_TOOLLESS_AGENT_ID': create_agent(client, 'toolless'),
            'ELEVENLABS_TEST_SYNC_CREATE_AGENT_ID': create_agent(client, 'sync-create'),
            'ELEVENLABS_TEST_SYNC_ADOPT_AGENT_ID': create_agent(client, 'sync-adopt'),
            'ELEVENLABS_TEST_SYNC_UPDATE_AGENT_ID': create_agent(client, 'sync-update', tool_ids=[stale]),
            'ELEVENLABS_TEST_SYNC_DETACH_AGENT_ID': create_agent(
                client, 'sync-detach', tool_ids=[weather, extra, webhook]
            ),
        }


def _named(client: httpx.Client, path: str, key: str) -> list[dict[str, Any]]:
    response = client.get(path, params={'page_size': 100})
    response.raise_for_status()
    return response.json()[key]


def orphan() -> None:
    """Delete the `sync-create` agent, leaving the tool its recording created unattached."""
    with _client() as client:
        for agent in _named(client, '/v1/convai/agents', 'agents'):
            if agent['name'] == PREFIX + 'sync-create':
                client.delete(f'/v1/convai/agents/{agent["agent_id"]}').raise_for_status()


def update() -> None:
    """Bring the existing `pydantic-ai-tests-*` agents' TTS and client events in line with `setup()`."""
    with _client() as client:
        for agent in _named(client, '/v1/convai/agents', 'agents'):
            if agent['name'].startswith(PREFIX):
                client.patch(
                    f'/v1/convai/agents/{agent["agent_id"]}',
                    json={
                        'conversation_config': {
                            'tts': {'model_id': TTS_MODEL},
                            'conversation': {'client_events': CLIENT_EVENTS},
                        }
                    },
                ).raise_for_status()


def cleanup() -> None:
    """Delete every `pydantic-ai-tests-*` agent.

    The tools the agents used are left: a tool's name is its function name, so nothing marks it as this script's.
    Delete them in the dashboard once no agent uses them.
    """
    with _client() as client:
        for agent in _named(client, '/v1/convai/agents', 'agents'):
            if agent['name'].startswith(PREFIX):
                client.delete(f'/v1/convai/agents/{agent["agent_id"]}').raise_for_status()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else 'setup'
    if command == 'orphan':
        orphan()
    elif command == 'update':
        update()
    elif command == 'cleanup':
        cleanup()
    else:
        for name, value in setup().items():
            print(f'{name}={value}')
