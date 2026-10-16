from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart

from .._inline_snapshot import snapshot
from ..conftest import try_import
from .conftest import RequestCapture, json_objects

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return 'rainy'  # pragma: no cover


async def test_moonshotai_field_mode_thinking_backfill(
    allow_model_requests: None, moonshotai_api_key: str, request_capture: RequestCapture
):
    """MoonshotAI accepts the empty `reasoning_content` a thinking request sends on tool calls no step thought on.

    `MoonshotAIProvider` uses `'field'` mode. The history holds no `ThinkingPart`, and its tool calls sit
    before and after the last user message, as a step that didn't think leaves them. With `thinking=True`,
    both go out with `reasoning_content` empty; the run completing is MoonshotAI accepting the request.
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather in Paris? Use the get_weather tool.')]),
        ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id='call_paris')]),
        ModelRequest(
            parts=[
                ToolReturnPart('get_weather', 'sunny', tool_call_id='call_paris'),
                UserPromptPart('And in London? Use the tool again, then answer for both in one short sentence.'),
            ]
        ),
        ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'London'}, tool_call_id='call_london')]),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'rainy', tool_call_id='call_london')]),
    ]
    provider = MoonshotAIProvider(api_key=moonshotai_api_key, http_client=request_capture.client)
    agent = Agent(OpenAIChatModel('kimi-k3', provider=provider), tools=[get_weather])

    result = await agent.run(message_history=history, model_settings=OpenAIChatModelSettings(thinking=True))

    assert result.output == snapshot("The weather in Paris is sunny and in London it's rainy.")
    messages = request_capture.body('/chat/completions')['messages']
    assert [
        message.get('reasoning_content') for message in json_objects(messages) if message['role'] == 'assistant'
    ] == ['', '']
