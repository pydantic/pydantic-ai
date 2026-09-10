from __future__ import annotations

import json
import os

import pytest

from pydantic_ai import Agent, Tool, ToolReturn
from pydantic_ai.usage import UsageLimits

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import
from .conftest import json_objects

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.models.moonshotai import MoonshotAIModel
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


async def test_live_dynamic_tool_loading(allow_model_requests: None, request_capture: RequestCapture):
    """Kimi accepts a revealed definition, calls the tool, and retains it in the next request."""
    client = AsyncOpenAI(
        api_key=os.getenv('MOONSHOTAI_API_KEY', 'mock-api-key'),
        base_url='https://api.moonshot.cn/v1',
        http_client=request_capture.http_client(timeout=30),
        max_retries=0,
    )
    model = MoonshotAIModel(
        'kimi-k3',
        provider=MoonshotAIProvider(openai_client=client),
        settings={'max_tokens': 1024, 'thinking': 'low'},
    )
    calls: list[str] = []

    def weather() -> str:
        """Get the current weather."""
        calls.append('weather')
        return 'sunny'

    def load_weather() -> ToolReturn:
        """Load the weather tool."""
        calls.append('load_weather')
        return ToolReturn(return_value='Weather tool loaded.', tools=['weather'])

    agent = Agent(
        model,
        tools=[load_weather, Tool(weather, defer_loading=True)],
        instructions='Call load_weather, then call weather exactly once. Reply with only the weather result.',
        retries=0,
    )
    async with agent.run_stream(
        'Get the weather.', usage_limits=UsageLimits(request_limit=3, tool_calls_limit=2)
    ) as result:
        assert (await result.get_output()).strip().lower() == 'sunny'
        assert result.usage.requests == 3
    assert calls == ['load_weather', 'weather']

    initial, revealed, final = request_capture.bodies('/chat/completions')
    assert initial['tools'] == revealed['tools'] == final['tools']
    for tool in json_objects(initial['tools']):
        function = tool['function']
        assert isinstance(function, dict)
        assert function['name'] != 'weather'

    initial_messages, revealed_messages, final_messages = (
        json_objects(body['messages']) for body in (initial, revealed, final)
    )
    assert json.dumps(revealed_messages[: len(initial_messages)]) == json.dumps(initial_messages)
    assert json.dumps(final_messages[: len(revealed_messages)]) == json.dumps(revealed_messages)
    additions = [message for message in revealed_messages if 'tools' in message]
    assert additions == snapshot(
        [
            {
                'role': 'system',
                'tools': [
                    {
                        'type': 'function',
                        'function': {
                            'name': 'weather',
                            'description': 'Get the current weather.',
                            'parameters': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                        },
                    }
                ],
            }
        ]
    )
    assert [message for message in final_messages if 'tools' in message] == additions
