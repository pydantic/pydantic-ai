"""The Pydantic AI Gateway's reported cost lands on `RequestUsage.cost` and wins over the genai-prices estimate.

The gateway reports its cost in the response body (`usage.logfire_ai_gateway.cost`, falling back to the deprecated
`usage.pydantic_ai_gateway.cost_estimate`) and in the `logfire-ai-gateway-cost` response header. The deprecated
`pydantic-ai-gateway-price-estimate` header is ignored because its rounding can turn small positive charges into zero.
OpenAI and Anthropic read the body; google-genai and boto3 drop unknown body keys, so Google and Bedrock read the
canonical header.

Each case stubs the HTTP layer underneath the real provider SDK rather than replaying a cassette: the SDK's own
parsing decides whether the body block survives, which is the behavior under test, and the current names aren't
yet served by a released gateway to record against. Every case uses a model genai-prices can price, so a cost that
differs from the estimate proves the reported cost took precedence.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.models import Model

from ._inline_snapshot import snapshot
from .conftest import try_import

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as bedrock_available:
    from botocore.awsrequest import AWSResponse
    from botocore.compat import HTTPHeaders

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

NEW_BLOCK = {'logfire_ai_gateway': {'cost': '0.5'}}
OLD_BLOCK = {'pydantic_ai_gateway': {'cost_estimate': '0.25'}}
# A gateway response-cache hit reports a zero cost, which must not fall back to the estimate.
CACHE_HIT_BLOCKS = {'logfire_ai_gateway': {'cost': '0'}, 'pydantic_ai_gateway': {'cost_estimate': '0'}}

NEW_HEADER = {'logfire-ai-gateway-cost': '0.5USD'}
OLD_HEADER = {'pydantic-ai-gateway-price-estimate': '0.25USD'}
CACHE_HIT_HEADERS = {'logfire-ai-gateway-cost': '0USD', 'pydantic-ai-gateway-price-estimate': '0.25USD'}


def _httpx_client(body: dict[str, Any], headers: dict[str, str]) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=body, headers=headers))
    )


def _sse_client(events: list[dict[str, Any]], *, named: bool) -> httpx2.AsyncClient:
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


def openai_model(usage_extra: dict[str, Any], headers: dict[str, str]) -> Model:
    body = {
        'id': 'chatcmpl-1',
        'object': 'chat.completion',
        'created': 1704067200,
        'model': 'gpt-4o-2024-08-06',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'hi'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 1000, 'completion_tokens': 1000, 'total_tokens': 2000, **usage_extra},
    }
    provider = OpenAIProvider(api_key='mock-api-key', http_client=_httpx_client(body, headers))
    return OpenAIChatModel('gpt-4o', provider=provider)


def anthropic_model(usage_extra: dict[str, Any], headers: dict[str, str]) -> Model:
    body = {
        'id': 'msg_1',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-5',
        'content': [{'type': 'text', 'text': 'hi'}],
        'stop_reason': 'end_turn',
        'usage': {'input_tokens': 1000, 'output_tokens': 1000, **usage_extra},
    }
    provider = AnthropicProvider(api_key='mock-api-key', http_client=_httpx_client(body, headers))
    return AnthropicModel('claude-sonnet-4-5', provider=provider)


def google_model(usage_extra: dict[str, Any], headers: dict[str, str]) -> Model:
    body = {
        'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'hi'}]}, 'finishReason': 'STOP'}],
        'usageMetadata': {
            'promptTokenCount': 1000,
            'candidatesTokenCount': 1000,
            'totalTokenCount': 2000,
            **usage_extra,
        },
        'modelVersion': 'gemini-2.5-flash',
    }
    provider = GoogleProvider(api_key='mock-api-key', http_client=_httpx_client(body, headers))
    return GoogleModel('gemini-2.5-flash', provider=provider)


def bedrock_model(usage_extra: dict[str, Any], headers: dict[str, str]) -> Model:
    body = {
        'output': {'message': {'role': 'assistant', 'content': [{'text': 'hi'}]}},
        'stopReason': 'end_turn',
        'usage': {'inputTokens': 1000, 'outputTokens': 1000, 'totalTokens': 2000, **usage_extra},
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
    return BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=provider)


@dataclass(frozen=True)
class Case:
    id: str
    model: Callable[[dict[str, Any], dict[str, str]], Model]
    expected_cost: Decimal
    usage_extra: dict[str, Any] = field(default_factory=dict[str, Any])
    headers: dict[str, str] = field(default_factory=dict[str, str])
    marks: tuple[pytest.MarkDecorator, ...] = ()


OPENAI = (pytest.mark.skipif(not openai_available(), reason='openai not installed'),)
ANTHROPIC = (pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed'),)
GOOGLE = (pytest.mark.skipif(not google_available(), reason='google-genai not installed'),)
BEDROCK = (pytest.mark.skipif(not bedrock_available(), reason='boto3 not installed'),)

CASES = [
    Case(
        id='openai-new-block',
        model=openai_model,
        usage_extra={**NEW_BLOCK, **OLD_BLOCK},
        expected_cost=Decimal('0.5'),
        marks=OPENAI,
    ),
    Case(id='openai-old-block', model=openai_model, usage_extra=OLD_BLOCK, expected_cost=Decimal('0.25'), marks=OPENAI),
    Case(
        id='openai-cache-hit',
        model=openai_model,
        usage_extra=CACHE_HIT_BLOCKS,
        expected_cost=Decimal('0'),
        marks=OPENAI,
    ),
    Case(id='openai-no-block', model=openai_model, expected_cost=snapshot(Decimal('0.0125')), marks=OPENAI),
    Case(
        id='anthropic-new-block',
        model=anthropic_model,
        usage_extra={**NEW_BLOCK, **OLD_BLOCK},
        expected_cost=Decimal('0.5'),
        marks=ANTHROPIC,
    ),
    Case(
        id='anthropic-old-block',
        model=anthropic_model,
        usage_extra=OLD_BLOCK,
        expected_cost=Decimal('0.25'),
        marks=ANTHROPIC,
    ),
    Case(
        id='anthropic-cache-hit',
        model=anthropic_model,
        usage_extra=CACHE_HIT_BLOCKS,
        expected_cost=Decimal('0'),
        marks=ANTHROPIC,
    ),
    Case(
        id='google-new-header',
        model=google_model,
        headers={**NEW_HEADER, **OLD_HEADER},
        expected_cost=Decimal('0.5'),
        marks=GOOGLE,
    ),
    Case(
        id='google-old-header-ignored',
        model=google_model,
        headers=OLD_HEADER,
        expected_cost=snapshot(Decimal('0.0028')),
        marks=GOOGLE,
    ),
    Case(
        id='google-cache-hit',
        model=google_model,
        headers=CACHE_HIT_HEADERS,
        expected_cost=Decimal('0'),
        marks=GOOGLE,
    ),
    # google-genai drops the body block, so without a header the estimate is used.
    Case(
        id='google-body-only',
        model=google_model,
        usage_extra=NEW_BLOCK,
        expected_cost=snapshot(Decimal('0.0028')),
        marks=GOOGLE,
    ),
    # An invalid canonical header does not make the deprecated header authoritative; a value without `USD` is ignored.
    Case(
        id='google-malformed-header',
        model=google_model,
        headers={'logfire-ai-gateway-cost': 'abcUSD', 'pydantic-ai-gateway-price-estimate': '0.25USD'},
        expected_cost=snapshot(Decimal('0.0028')),
        marks=GOOGLE,
    ),
    Case(
        id='bedrock-new-header',
        model=bedrock_model,
        headers={**NEW_HEADER, **OLD_HEADER},
        expected_cost=Decimal('0.5'),
        marks=BEDROCK,
    ),
    Case(
        id='bedrock-old-header-ignored',
        model=bedrock_model,
        headers=OLD_HEADER,
        expected_cost=snapshot(Decimal('0.0198')),
        marks=BEDROCK,
    ),
    Case(
        id='bedrock-cache-hit',
        model=bedrock_model,
        headers=CACHE_HIT_HEADERS,
        expected_cost=Decimal('0'),
        marks=BEDROCK,
    ),
    # boto3 drops the body block, so without a header the estimate is used.
    Case(
        id='bedrock-body-only',
        model=bedrock_model,
        usage_extra=NEW_BLOCK,
        expected_cost=snapshot(Decimal('0.0198')),
        marks=BEDROCK,
    ),
    # An invalid canonical header does not make the deprecated header authoritative.
    Case(
        id='bedrock-negative-header',
        model=bedrock_model,
        headers={'logfire-ai-gateway-cost': '-1USD', 'pydantic-ai-gateway-price-estimate': '0.25USD'},
        expected_cost=snapshot(Decimal('0.0198')),
        marks=BEDROCK,
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
async def test_gateway_reported_cost(allow_model_requests: None, case: Case):
    agent = Agent(case.model(case.usage_extra, case.headers))
    # Without `max_tokens`, Anthropic and Claude-on-Bedrock requests are streamed behind the scenes, and the stubs
    # answer with a non-streamed body.
    result = await agent.run('hello', model_settings={'max_tokens': 100})
    assert result.usage.cost == case.expected_cost


def openai_streamed_model(usage_extra: dict[str, Any]) -> Model:
    chunk = {'id': 'chatcmpl-1', 'object': 'chat.completion.chunk', 'created': 1704067200, 'model': 'gpt-4o-2024-08-06'}
    events: list[dict[str, Any]] = [
        {**chunk, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'hi'}, 'finish_reason': None}]},
        {**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]},
        {
            **chunk,
            'choices': [],
            'usage': {'prompt_tokens': 1000, 'completion_tokens': 1000, 'total_tokens': 2000, **usage_extra},
        },
    ]
    provider = OpenAIProvider(api_key='mock-api-key', http_client=_sse_client(events, named=False))
    return OpenAIChatModel('gpt-4o', provider=provider)


def anthropic_streamed_model(usage_extra: dict[str, Any]) -> Model:
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
            'usage': {'output_tokens': 1000, **usage_extra},
        },
        {'type': 'message_stop'},
    ]
    provider = AnthropicProvider(api_key='mock-api-key', http_client=_sse_client(events, named=True))
    return AnthropicModel('claude-sonnet-4-5', provider=provider)


@dataclass(frozen=True)
class StreamedCase:
    id: str
    model: Callable[[dict[str, Any]], Model]
    usage_extra: dict[str, Any]
    expected_cost: Decimal
    marks: tuple[pytest.MarkDecorator, ...] = ()


# The gateway also reports its cost in the final usage event of an OpenAI or Anthropic stream.
STREAMED_CASES = [
    StreamedCase(
        id='openai-new-block',
        model=openai_streamed_model,
        usage_extra={**NEW_BLOCK, **OLD_BLOCK},
        expected_cost=Decimal('0.5'),
        marks=OPENAI,
    ),
    StreamedCase(
        id='openai-old-block',
        model=openai_streamed_model,
        usage_extra=OLD_BLOCK,
        expected_cost=Decimal('0.25'),
        marks=OPENAI,
    ),
    StreamedCase(
        id='anthropic-new-block',
        model=anthropic_streamed_model,
        usage_extra={**NEW_BLOCK, **OLD_BLOCK},
        expected_cost=Decimal('0.5'),
        marks=ANTHROPIC,
    ),
    StreamedCase(
        id='anthropic-cache-hit',
        model=anthropic_streamed_model,
        usage_extra=CACHE_HIT_BLOCKS,
        expected_cost=Decimal('0'),
        marks=ANTHROPIC,
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in STREAMED_CASES])
async def test_gateway_reported_cost_streamed(allow_model_requests: None, case: StreamedCase):
    agent = Agent(case.model(case.usage_extra))
    async with agent.run_stream('hello') as result:
        await result.get_output()
    assert result.usage.cost == case.expected_cost
