"""The Pydantic AI Gateway's reported cost on `RequestUsage.cost`.

The recorded `test_gateway_provider_with_*` cases in `test_gateway.py` cover the cost names deployed gateways
serve. These cases cover the newer names (`usage.logfire_ai_gateway` and the `logfire-ai-gateway-cost` header) and
response-cache hits, which no recorded gateway serves yet, by stubbing the HTTP layer underneath the real provider
SDK: the SDK's own parsing decides whether the gateway's cost survives.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, UsageLimitExceeded
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage, UsageLimits

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from botocore.awsrequest import AWSResponse
    from botocore.compat import HTTPHeaders

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.bedrock import BedrockProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='Providers not installed')


def _mock_httpx_client(body: dict[str, Any], headers: dict[str, str] | None = None) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=body, headers=headers))
    )


def _mock_sse_client(events: list[dict[str, Any]], *, named: bool) -> httpx2.AsyncClient:
    content = ''.join(
        (f'event: {event["type"]}\n' if named else '') + f'data: {json.dumps(event)}\n\n' for event in events
    )
    if not named:
        content += 'data: [DONE]\n\n'
    return httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(200, content=content.encode(), headers={'content-type': 'text/event-stream'})
        )
    )


GATEWAY_COST_BLOCKS = {'logfire_ai_gateway': {'cost': '0.5'}, 'pydantic_ai_gateway': {'cost_estimate': '0.25'}}


async def test_gateway_reported_cost_openai(allow_model_requests: None):
    body = {
        'id': 'chatcmpl-1',
        'object': 'chat.completion',
        'created': 1704067200,
        'model': 'gpt-4o-2024-08-06',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'hi'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 1000, 'completion_tokens': 1000, 'total_tokens': 2000, **GATEWAY_COST_BLOCKS},
    }
    provider = OpenAIProvider(api_key='mock-api-key', http_client=_mock_httpx_client(body))
    result = await Agent(OpenAIChatModel('gpt-4o', provider=provider)).run('hello')
    assert result.usage.cost == snapshot(Decimal('0.5'))


async def test_gateway_reported_cost_anthropic(allow_model_requests: None):
    body = {
        'id': 'msg_1',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-5',
        'content': [{'type': 'text', 'text': 'hi'}],
        'stop_reason': 'end_turn',
        'usage': {'input_tokens': 1000, 'output_tokens': 1000, **GATEWAY_COST_BLOCKS},
    }
    provider = AnthropicProvider(api_key='mock-api-key', http_client=_mock_httpx_client(body))
    # Without `max_tokens` the request is streamed behind the scenes, and the stub answers with a non-streamed body.
    agent = Agent(AnthropicModel('claude-sonnet-4-5', provider=provider), model_settings={'max_tokens': 100})
    result = await agent.run('hello')
    assert result.usage.cost == snapshot(Decimal('0.5'))


async def test_gateway_reported_cost_openai_stream(allow_model_requests: None):
    chunk = {'id': 'chatcmpl-1', 'object': 'chat.completion.chunk', 'created': 1704067200, 'model': 'gpt-4o-2024-08-06'}
    events: list[dict[str, Any]] = [
        {**chunk, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'hi'}, 'finish_reason': 'stop'}]},
        {
            **chunk,
            'choices': [],
            'usage': {'prompt_tokens': 1000, 'completion_tokens': 1000, 'total_tokens': 2000, **GATEWAY_COST_BLOCKS},
        },
    ]
    provider = OpenAIProvider(api_key='mock-api-key', http_client=_mock_sse_client(events, named=False))
    async with Agent(OpenAIChatModel('gpt-4o', provider=provider)).run_stream('hello') as result:
        await result.get_output()
    assert result.usage.cost == snapshot(Decimal('0.5'))


async def test_gateway_reported_cost_anthropic_stream(allow_model_requests: None):
    message: dict[str, Any] = {
        'id': 'msg_1',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-5',
        'content': [],
        'stop_reason': None,
        'usage': {'input_tokens': 1000, 'output_tokens': 1},
    }
    events: list[dict[str, Any]] = [
        {'type': 'message_start', 'message': message},
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'hi'}},
        {'type': 'content_block_stop', 'index': 0},
        {
            'type': 'message_delta',
            'delta': {'stop_reason': 'end_turn'},
            'usage': {'output_tokens': 1000, **GATEWAY_COST_BLOCKS},
        },
        {'type': 'message_stop'},
    ]
    provider = AnthropicProvider(api_key='mock-api-key', http_client=_mock_sse_client(events, named=True))
    async with Agent(AnthropicModel('claude-sonnet-4-5', provider=provider)).run_stream('hello') as result:
        await result.get_output()
    assert result.usage.cost == snapshot(Decimal('0.5'))


@pytest.mark.parametrize(
    'model_name, headers, expected_cost',
    [
        # The header wins over the genai-prices estimate.
        pytest.param(
            'gemini-2.5-flash', {'logfire-ai-gateway-cost': '0.00013875USD'}, Decimal('0.00013875'), id='cost'
        ),
        # A response-cache hit is free.
        pytest.param('gemini-2.5-flash', {'logfire-ai-gateway-cost': '0USD'}, Decimal('0'), id='cache-hit'),
        pytest.param('custom-model', {'logfire-ai-gateway-cost': '0.0012USD'}, Decimal('0.0012'), id='unpriced-model'),
        # Without the header, the genai-prices estimate applies.
        pytest.param('gemini-2.5-flash', {}, Decimal('0.0028'), id='no-header'),
        pytest.param('custom-model', {'logfire-ai-gateway-cost': 'abcUSD'}, None, id='malformed'),
    ],
)
async def test_gateway_header_cost_google(
    allow_model_requests: None, model_name: str, headers: dict[str, str], expected_cost: Decimal | None
):
    body = {
        'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'hi'}]}, 'finishReason': 'STOP'}],
        # google-genai drops the body's cost blocks.
        'usageMetadata': {
            'promptTokenCount': 1000,
            'candidatesTokenCount': 1000,
            'totalTokenCount': 2000,
            **GATEWAY_COST_BLOCKS,
        },
        'modelVersion': model_name,
    }
    provider = GoogleProvider(api_key='mock-api-key', http_client=_mock_httpx_client(body, headers))
    result = await Agent(GoogleModel(model_name, provider=provider)).run('hello')
    assert result.usage.cost == expected_cost


@pytest.mark.parametrize(
    'model_name, headers, expected_cost',
    [
        pytest.param(
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            {'logfire-ai-gateway-cost': '0.00013875USD'},
            Decimal('0.00013875'),
            id='cost',
        ),
        pytest.param(
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            {'logfire-ai-gateway-cost': '0USD'},
            Decimal('0'),
            id='cache-hit',
        ),
    ],
)
async def test_gateway_header_cost_bedrock(
    allow_model_requests: None, model_name: str, headers: dict[str, str], expected_cost: Decimal
):
    body = {
        'output': {'message': {'role': 'assistant', 'content': [{'text': 'hi'}]}},
        'stopReason': 'end_turn',
        'usage': {'inputTokens': 1000, 'outputTokens': 1000, 'totalTokens': 2000},
        'metrics': {'latencyMs': 10},
    }

    class _Raw:
        def stream(self, *_args: Any, **_kwargs: Any):
            yield json.dumps(body).encode()

    def send(request: Any, **_kwargs: Any) -> AWSResponse:
        response_headers = HTTPHeaders()
        for name, value in {'content-type': 'application/json', **headers}.items():
            response_headers[name] = value
        return AWSResponse(request.url, 200, response_headers, _Raw())

    provider = BedrockProvider(region_name='us-east-1', aws_access_key_id='x', aws_secret_access_key='x')
    # botocore sends a response returned from `before-send` instead of the request; the stubs type handlers as `-> None`.
    provider.client.meta.events.register('before-send.bedrock-runtime.Converse', send)  # pyright: ignore[reportArgumentType]
    agent = Agent(BedrockConverseModel(model_name, provider=provider), model_settings={'max_tokens': 100})
    result = await agent.run('hello')
    assert result.usage.cost == expected_cost


def test_gateway_reported_cost_parsing():
    def extract(**gateway_blocks: object) -> RequestUsage:
        return RequestUsage.extract(
            {'model': 'gpt-5', 'usage': {'prompt_tokens': 2, 'completion_tokens': 1, **gateway_blocks}},
            provider='openai',
            provider_url='https://api.openai.com/v1',
            provider_fallback='openai',
            api_flavor='chat',
        )

    assert extract(**GATEWAY_COST_BLOCKS).cost == snapshot(Decimal('0.5'))
    assert extract(pydantic_ai_gateway={'cost_estimate': '0.00012625'}).cost == snapshot(Decimal('0.00012625'))
    # Older gateways sent a JSON number.
    assert extract(logfire_ai_gateway={'cost': 0.5}).cost == snapshot(Decimal('0.5'))
    # A response-cache hit is a real zero cost, not a missing one.
    assert extract(logfire_ai_gateway={'cost': '0'}).cost == snapshot(Decimal('0'))
    # An unusable current block falls back to the deprecated one.
    assert extract(
        logfire_ai_gateway={'cost': 'not-a-number'}, pydantic_ai_gateway={'cost_estimate': '0.25'}
    ).cost == snapshot(Decimal('0.25'))
    assert extract(logfire_ai_gateway='not-a-mapping', pydantic_ai_gateway={'cost_estimate': '0.25'}).cost == snapshot(
        Decimal('0.25')
    )
    # NaN would break `cost_limit` comparisons and a negative cost would reduce the run total.
    for bad_cost in ('NaN', '-0.5', True, 'not-a-number'):
        assert extract(pydantic_ai_gateway={'cost_estimate': bad_cost}).cost is None
    assert extract(pydantic_ai_gateway={}).cost is None
    assert extract().input_tokens == 2


async def test_gateway_reported_cost_counts_towards_cost_limit():
    """A reported cost counts even when genai-prices can't price the model."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart('4')],
            usage=RequestUsage.extract(
                {'model': 'custom-model', 'usage': {'logfire_ai_gateway': {'cost': '0.125'}}},
                provider='unknown-custom-provider',
                provider_url='',
                provider_fallback='',
            ),
        )

    agent = Agent(FunctionModel(respond))
    result = await agent.run('What is 2+2?')
    assert result.usage.cost == snapshot(Decimal('0.125'))
    with pytest.raises(UsageLimitExceeded, match='cost_limit'):
        await agent.run('What is 3+3?', usage=result.usage, usage_limits=UsageLimits(cost_limit=Decimal('0.2')))
