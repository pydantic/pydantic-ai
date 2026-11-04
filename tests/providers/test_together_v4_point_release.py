from __future__ import annotations as _annotations

import json
from typing import Any

import httpx2
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.settings import ModelSettings

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.together import TogetherProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def _final_result(answer: str) -> str:
    return answer


_COMPLETION = {
    'id': 'chatcmpl-1',
    'object': 'chat.completion',
    'created': 1700000000,
    'model': 'deepseek-v4.1-flash',
    'choices': [
        {
            'index': 0,
            'finish_reason': 'stop',
            'message': {'role': 'assistant', 'content': 'Hello'},
        }
    ],
    'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
}

_TOOL_CALL_COMPLETION = {
    'id': 'chatcmpl-2',
    'object': 'chat.completion',
    'created': 1700000000,
    'model': 'deepseek-v4.1-flash',
    'choices': [
        {
            'index': 0,
            'finish_reason': 'tool_calls',
            'message': {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'function',
                        'function': {'name': 'final_result', 'arguments': '{"answer": "ok"}'},
                    }
                ],
            },
        }
    ],
    'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
}


async def _together_chat_body(model_name: str, model_settings: ModelSettings | None = None) -> dict[str, Any]:
    """Run one Together chat request over a mock transport and return the captured request body."""
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_COMPLETION)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        provider = TogetherProvider(api_key='test', http_client=http_client)
        model = OpenAIChatModel(model_name, provider=provider)
        agent = Agent(model, model_settings=model_settings)
        await agent.run('hi')
    return bodies[0]


@pytest.mark.anyio
async def test_together_deepseek_v4_1_forced_tool_choice_falls_back_to_auto(allow_model_requests: None):
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_TOOL_CALL_COMPLETION)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        provider = TogetherProvider(api_key='test', http_client=http_client)
        model = OpenAIChatModel('deepseek-ai/DeepSeek-V4.1-Flash', provider=provider)
        agent = Agent(model, output_type=_final_result)
        result = await agent.run('hi')
    assert result.output == 'ok'
    # `supports_forced_tool_choice=False` downgrades the resolved output-tool forcing
    # to an unforced request instead of a per-function tool_choice.
    assert bodies[0]['tool_choice'] == 'auto'

    # Explicit forcing still fails loudly instead of being silently downgraded.
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        provider = TogetherProvider(api_key='test', http_client=http_client)
        model = OpenAIChatModel('deepseek-ai/DeepSeek-V4.1-Flash', provider=provider)
        agent = Agent(model, model_settings=ModelSettings(tool_choice='required'))

        @agent.tool_plain
        def get_answer() -> str:
            return 'ok'

        with pytest.raises(UserError):
            await agent.run('hi')
    assert len(bodies) == 1  # the second run sent nothing


@pytest.mark.anyio
async def test_together_deepseek_v4_1_thinking_reaches_wire(allow_model_requests: None):
    body_v4_1_max = await _together_chat_body('deepseek-ai/DeepSeek-V4.1-Flash', ModelSettings(thinking='max'))
    body_v4_max = await _together_chat_body('deepseek-ai/DeepSeek-V4-Flash', ModelSettings(thinking='max'))
    assert body_v4_1_max.pop('model') == 'deepseek-ai/DeepSeek-V4.1-Flash'
    assert body_v4_max.pop('model') == 'deepseek-ai/DeepSeek-V4-Flash'
    assert body_v4_1_max['reasoning_effort'] == 'xhigh'
    assert body_v4_1_max == body_v4_max

    body_v4_1_off = await _together_chat_body('deepseek-ai/DeepSeek-V4.1-Flash', ModelSettings(thinking=False))
    body_v4_off = await _together_chat_body('deepseek-ai/DeepSeek-V4-Flash', ModelSettings(thinking=False))
    assert body_v4_1_off.pop('model') == 'deepseek-ai/DeepSeek-V4.1-Flash'
    assert body_v4_off.pop('model') == 'deepseek-ai/DeepSeek-V4-Flash'
    assert body_v4_1_off['reasoning_effort'] == 'none'
    assert body_v4_1_off == body_v4_off
