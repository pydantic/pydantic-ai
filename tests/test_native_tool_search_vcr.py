"""VCR tests for native provider tool search APIs (raw SDK, not pydantic-ai).

These tests call the OpenAI Responses API and Anthropic Beta API directly
to verify that native tool search works. They serve as integration references
for future pydantic-ai support of native tool search (#4566, #4167).
"""

from __future__ import annotations

import os

import openai
import pytest

from .conftest import try_import

with try_import() as anthropic_available:
    import anthropic

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
]

MOCK_API_KEYS: dict[str, str] = {
    'OPENAI_API_KEY': 'mock-api-key',
    'ANTHROPIC_API_KEY': 'mock-api-key',
}

_TOOLS = [
    {
        'name': 'get_weather',
        'description': 'Get the current weather for a city.',
        'parameters': {
            'type': 'object',
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
            'additionalProperties': False,
        },
    },
    {
        'name': 'get_exchange_rate',
        'description': 'Look up the current exchange rate between two currencies.',
        'parameters': {
            'type': 'object',
            'properties': {
                'from_currency': {'type': 'string'},
                'to_currency': {'type': 'string'},
            },
            'required': ['from_currency', 'to_currency'],
            'additionalProperties': False,
        },
    },
    {
        'name': 'stock_lookup',
        'description': 'Look up stock price by ticker symbol.',
        'parameters': {
            'type': 'object',
            'properties': {'symbol': {'type': 'string'}},
            'required': ['symbol'],
            'additionalProperties': False,
        },
    },
    {
        'name': 'mortgage_calculator',
        'description': 'Calculate monthly mortgage payment for a home loan.',
        'parameters': {
            'type': 'object',
            'properties': {
                'principal': {'type': 'number'},
                'rate': {'type': 'number'},
                'years': {'type': 'integer'},
            },
            'required': ['principal', 'rate', 'years'],
            'additionalProperties': False,
        },
    },
]


@pytest.fixture(autouse=True)
def _mock_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, default in MOCK_API_KEYS.items():
        if not os.getenv(key):  # pragma: no branch
            monkeypatch.setenv(key, default)


async def test_openai_native_tool_search(allow_model_requests: None) -> None:
    """Test OpenAI Responses API with defer_loading + tool_search.

    Uses a namespace with defer_loading=True tools and a tool_search entry.
    The model should discover and call get_exchange_rate via native search.
    See: https://developers.openai.com/api/docs/guides/tools-tool-search
    """
    client = openai.AsyncOpenAI()

    namespace_tools: list[dict[str, object]] = [
        {
            'type': 'function',
            'name': _TOOLS[0]['name'],
            'description': _TOOLS[0]['description'],
            'parameters': _TOOLS[0]['parameters'],
        },
    ]
    for tool_def in _TOOLS[1:]:
        namespace_tools.append(
            {
                'type': 'function',
                'name': tool_def['name'],
                'description': tool_def['description'],
                'parameters': tool_def['parameters'],
                'defer_loading': True,
            }
        )

    response = await client.responses.create(
        model='gpt-5.4',
        input='What is the current exchange rate from USD to EUR?',
        tools=[
            {
                'type': 'namespace',
                'name': 'tools',
                'description': 'Available tools for weather, finance, and calculations.',
                'tools': namespace_tools,
            },
            {'type': 'tool_search'},
        ],
        parallel_tool_calls=False,
    )

    tool_calls = [item for item in response.output if item.type == 'function_call']

    assert any(tc.name == 'get_exchange_rate' for tc in tool_calls)


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_native_tool_search(allow_model_requests: None) -> None:
    """Test Anthropic Beta API with tool_search_tool_bm25.

    Sends tools with a tool_search_tool_bm25 entry and deferred tools.
    The model should discover and call get_exchange_rate via native search.
    """
    client = anthropic.AsyncAnthropic()

    tools: list[dict[str, object]] = [
        {
            'type': 'custom',
            'name': _TOOLS[0]['name'],
            'description': _TOOLS[0]['description'],
            'input_schema': _TOOLS[0]['parameters'],
        },
    ]
    for tool_def in _TOOLS[1:]:
        tools.append(
            {
                'type': 'custom',
                'name': tool_def['name'],
                'description': tool_def['description'],
                'input_schema': tool_def['parameters'],
                'defer_loading': True,
            }
        )
    tools.append(
        {
            'type': 'tool_search_tool_bm25_20251119',
            'name': 'tool_search_tool_bm25',
        }
    )

    response = await client.beta.messages.create(
        model='claude-sonnet-4-5',
        max_tokens=1024,
        tools=tools,
        messages=[{'role': 'user', 'content': 'What is the current exchange rate from USD to EUR?'}],
    )

    tool_use_blocks = [block for block in response.content if block.type == 'tool_use']
    server_tool_blocks = [block for block in response.content if block.type == 'server_tool_use']

    assert len(server_tool_blocks) >= 1
    assert any(block.name == 'get_exchange_rate' for block in tool_use_blocks)
