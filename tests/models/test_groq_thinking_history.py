"""How `GroqModel` sends `ThinkingPart`s from message history back to the model.

A mock transport reads the request body actually sent: the cassette matchers ignore the body, so a
recording would keep passing if the reasoning moved back into the message content.
"""

from __future__ import annotations as _annotations

import json
from typing import Any

import httpx
import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='groq not installed')


HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content='What is the weather in Paris?')]),
    ModelResponse(
        parts=[
            ThinkingPart(content='I should look up the weather.'),
            ToolCallPart(tool_name='get_weather', args='{"city": "Paris"}', tool_call_id='call_1'),
        ]
    ),
    ModelRequest(parts=[ToolReturnPart(tool_name='get_weather', content='Sunny', tool_call_id='call_1')]),
    ModelResponse(
        parts=[
            ThinkingPart(content='The tool says sunny.'),
            ThinkingPart(content='Keep it short.'),
            TextPart(content='It is sunny in Paris.'),
        ]
    ),
]

ASSISTANT_MESSAGES: dict[str, list[dict[str, Any]]] = {
    # gpt-oss gets its reasoning back in the field Groq returns it in, never as `<think>` content:
    # replayed as content, it teaches the model to write those tags into its replies.
    'openai/gpt-oss-120b': snapshot(
        [
            {
                'role': 'assistant',
                'reasoning': 'I should look up the weather.',
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'function',
                        'function': {'name': 'get_weather', 'arguments': '{"city":"Paris"}'},
                    }
                ],
            },
            {
                'role': 'assistant',
                'reasoning': """\
The tool says sunny.

Keep it short.\
""",
                'content': 'It is sunny in Paris.',
            },
        ]
    ),
    # Other models get it back as text in their thinking tags.
    'qwen/qwen3-32b': snapshot(
        [
            {
                'role': 'assistant',
                'content': """\
<think>
I should look up the weather.
</think>\
""",
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'function',
                        'function': {'name': 'get_weather', 'arguments': '{"city":"Paris"}'},
                    }
                ],
            },
            {
                'role': 'assistant',
                'content': """\
<think>
The tool says sunny.
</think>

<think>
Keep it short.
</think>

It is sunny in Paris.\
""",
            },
        ]
    ),
}


async def assistant_messages_sent(model_name: str, history: list[ModelMessage]) -> list[dict[str, Any]]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                'id': '123',
                'object': 'chat.completion',
                'created': 1704067200,
                'model': model_name,
                'choices': [
                    {'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Cloudy.'}}
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        model = GroqModel(model_name, provider=GroqProvider(api_key='test-key', http_client=http_client))
        await Agent(model).run('And tomorrow?', message_history=history)

    return [message for message in bodies[0]['messages'] if message['role'] == 'assistant']


@pytest.mark.parametrize('model_name', list(ASSISTANT_MESSAGES))
async def test_thinking_in_message_history(allow_model_requests: None, model_name: str):
    assert await assistant_messages_sent(model_name, HISTORY) == ASSISTANT_MESSAGES[model_name]


async def test_empty_thinking_not_sent_in_reasoning_field(allow_model_requests: None):
    """Encrypted reasoning from another provider has no content, so it adds nothing to the `reasoning` field."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='What is the weather in Paris?')]),
        ModelResponse(
            parts=[
                ThinkingPart(content='', signature='encrypted', provider_name='openai'),
                ThinkingPart(content='Keep it short.'),
                ThinkingPart(content='', signature='encrypted', provider_name='openai'),
                TextPart(content='It is sunny in Paris.'),
            ]
        ),
        ModelRequest(parts=[UserPromptPart(content='Thanks.')]),
        ModelResponse(
            parts=[
                ThinkingPart(content='', signature='encrypted', provider_name='openai'),
                TextPart(content='You are welcome.'),
            ]
        ),
    ]
    assert await assistant_messages_sent('openai/gpt-oss-120b', history) == snapshot(
        [
            {'role': 'assistant', 'reasoning': 'Keep it short.', 'content': 'It is sunny in Paris.'},
            {'role': 'assistant', 'content': 'You are welcome.'},
        ]
    )
