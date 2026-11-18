"""MiniMax requests on the OpenAI-compatible routes that resolve `minimax_model_profile` through the family map.

Not VCR tests: there are no MiniMax recordings, and the claims are about the request the code builds, which
cassette matching doesn't check. A `MockTransport` serves synthetic responses, and `request_capture` records the
requests.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx2
import pytest
from pydantic import BaseModel, JsonValue

from pydantic_ai import Agent, NativeOutput, ToolReturnPart, UserError
from pydantic_ai.settings import ModelSettings

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture, try_import
from .conftest import json_objects

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.litellm import LiteLLMProvider
    from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

    from .mock_openai import completion_message

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

BASE_URL = 'https://gateway.example/v1'


def gateway_client(request_capture: RequestCapture, *messages: ChatCompletionMessage) -> httpx2.AsyncClient:
    """A client that answers each request with the next of `messages`, and records it in `request_capture`."""
    responses = iter(messages)

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=completion_message(next(responses)).model_dump(mode='json'))

    return httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        event_hooks={'request': list(request_capture.client.event_hooks['request'])},
    )


@dataclass(frozen=True)
class ThinkingCase:
    id: str
    model_name: str
    thinking: bool | None
    expected: dict[str, JsonValue]


THINKING_CASES = [
    ThinkingCase('m2.7-default', 'MiniMaxAI/MiniMax-M2.7', None, snapshot({})),
    ThinkingCase('m2.7-on', 'MiniMaxAI/MiniMax-M2.7', True, snapshot({'reasoning_effort': 'medium'})),
    # The M2 family and M3.1-Flash reject `reasoning_effort: 'none'` with a 400, per the probes in
    # https://github.com/pydantic/pydantic-ai/issues/12077, so turning thinking off sends nothing and the model
    # keeps reasoning.
    ThinkingCase('m2.7-off', 'MiniMaxAI/MiniMax-M2.7', False, snapshot({})),
    ThinkingCase('m3.1-flash-off', 'minimax/minimax-m3.1-flash', False, snapshot({})),
    ThinkingCase('m3-default', 'MiniMaxAI/MiniMax-M3', None, snapshot({})),
    # M3 is hybrid, and accepted `none` in the same probes. Which switch each gateway takes is
    # https://github.com/pydantic/pydantic-ai/issues/12065.
    ThinkingCase('m3-off', 'MiniMaxAI/MiniMax-M3', False, snapshot({'reasoning_effort': 'none'})),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in THINKING_CASES])
async def test_minimax_thinking_wire_shape(
    allow_model_requests: None, request_capture: RequestCapture, case: ThinkingCase
):
    async with gateway_client(request_capture, ChatCompletionMessage(content='hello', role='assistant')) as client:
        model = OpenAIChatModel(
            case.model_name,
            provider=OpenAICompatibleProvider('gateway', base_url=BASE_URL, api_key='test', http_client=client),
        )
        settings: ModelSettings = {} if case.thinking is None else {'thinking': case.thinking}
        await Agent(model).run('hello', model_settings=settings)

    body = request_capture.body('/chat/completions')
    assert {key: value for key, value in body.items() if key == 'reasoning_effort'} == case.expected


class Item(BaseModel):
    qty: int


class ToolArgs(BaseModel):
    item: Item


async def test_minimax_inlines_ref_tool_schema(allow_model_requests: None, request_capture: RequestCapture):
    """MiniMax stringifies nested tool arguments whose schema is a `$ref` into `$defs`, so the schema goes out inlined.

    See https://github.com/pydantic/pydantic-ai/issues/12077.
    """
    tool_call = ChatCompletionMessageFunctionToolCall(
        id='call_nested',
        function=Function(name='nested_tool', arguments='{"item": {"qty": 7}}'),
        type='function',
    )
    async with gateway_client(
        request_capture,
        ChatCompletionMessage(content=None, role='assistant', tool_calls=[tool_call]),
        ChatCompletionMessage(content='done', role='assistant'),
    ) as client:
        model = OpenAIChatModel(
            'MiniMaxAI/MiniMax-M2.7',
            provider=OpenAICompatibleProvider('gateway', base_url=BASE_URL, api_key='test', http_client=client),
        )
        agent = Agent(model)

        @agent.tool_plain
        def nested_tool(args: ToolArgs) -> str:
            return str(args.item.qty)

        result = await agent.run('hello')

    tools = json_objects(request_capture.body('/chat/completions')['tools'])
    assert tools[0]['function'] == snapshot(
        {
            'name': 'nested_tool',
            'description': '',
            'parameters': {
                'properties': {
                    'item': {
                        'properties': {'qty': {'type': 'integer'}},
                        'required': ['qty'],
                        'title': 'Item',
                        'type': 'object',
                    }
                },
                'required': ['item'],
                'title': 'ToolArgs',
                'type': 'object',
            },
            'strict': True,
        }
    )
    tool_returns = [
        part.content for message in result.new_messages() for part in message.parts if isinstance(part, ToolReturnPart)
    ]
    assert tool_returns == ['7']


class Answer(BaseModel):
    text: str


@pytest.mark.parametrize('provider_name', ['openai-compatible', 'litellm'])
async def test_minimax_native_output_unsupported(
    allow_model_requests: None, request_capture: RequestCapture, provider_name: str
):
    """MiniMax documents no `response_format` on its OpenAI-compatible API, so `NativeOutput` raises before a request.

    See https://platform.minimax.io/docs/api-reference/text-openai-api.
    """
    async with gateway_client(request_capture) as client:
        provider = (
            OpenAICompatibleProvider('gateway', base_url=BASE_URL, api_key='test', http_client=client)
            if provider_name == 'openai-compatible'
            else LiteLLMProvider(api_base=BASE_URL, api_key='test', http_client=client)
        )
        agent = Agent(OpenAIChatModel('MiniMaxAI/MiniMax-M2.7', provider=provider), output_type=NativeOutput(Answer))
        with pytest.raises(UserError, match='Native structured output is not supported by this model'):
            await agent.run('hello')

    assert request_capture.bodies() == []
