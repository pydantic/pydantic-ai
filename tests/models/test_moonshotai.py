from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent

from .._inline_snapshot import snapshot
from ..conftest import try_import
from .conftest import RequestCapture, get_weather, json_objects, weather_tool_calls_without_thinking

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]


async def test_moonshotai_field_mode_thinking_backfill(
    allow_model_requests: None, moonshotai_api_key: str, request_capture: RequestCapture
):
    """MoonshotAI accepts the empty `reasoning_content` a thinking request sends on tool calls no step thought on.

    `MoonshotAIProvider` uses `'field'` mode. The history holds no `ThinkingPart`, and its tool calls sit
    before and after the last user message, as a step that didn't think leaves them. With `thinking=True`,
    both go out with `reasoning_content` empty; the run completing is MoonshotAI accepting the request.
    """
    provider = MoonshotAIProvider(api_key=moonshotai_api_key, http_client=request_capture.client)
    agent = Agent(OpenAIChatModel('kimi-k3', provider=provider), tools=[get_weather])

    result = await agent.run(
        message_history=weather_tool_calls_without_thinking(), model_settings=OpenAIChatModelSettings(thinking=True)
    )

    assert result.output == snapshot('Paris is sunny, while London is rainy.')
    messages = request_capture.body('/chat/completions')['messages']
    assert [
        message.get('reasoning_content') for message in json_objects(messages) if message['role'] == 'assistant'
    ] == ['', '']
