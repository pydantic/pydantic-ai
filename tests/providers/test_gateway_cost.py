"""The Pydantic AI Gateway's reported cost on `RequestUsage.cost`.

The recorded `test_gateway_provider_with_*` cases in `test_gateway.py` cover the cost names deployed gateways
serve. These cases cover the newer names (`usage.logfire_ai_gateway` and the `logfire-ai-gateway-cost` header), the
cost in Google and Bedrock streams, and response-cache hits, which no recorded gateway serves yet, by stubbing the
HTTP layer underneath the real provider SDK: the SDK's own parsing decides whether the gateway's cost survives.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, ModelResponsePart, TextPart, UsageLimitExceeded
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage, UsageLimits

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from botocore.awsrequest import AWSResponse
    from botocore.compat import HTTPHeaders
    from logfire.testing import CaptureLogfire

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


def _mock_sse_client(events: list[dict[str, Any]], *, named: bool = False, done: bool = False) -> httpx2.AsyncClient:
    content = ''.join(
        (f'event: {event["type"]}\n' if named else '') + f'data: {json.dumps(event)}\n\n' for event in events
    )
    if done:
        content += 'data: [DONE]\n\n'
    return httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(200, content=content.encode(), headers={'content-type': 'text/event-stream'})
        )
    )


def _mock_bedrock_provider(operation: str, content: bytes, headers: dict[str, str]) -> BedrockProvider:
    class _Raw:
        def stream(self, *_args: Any, **_kwargs: Any):
            yield content

        def close(self) -> None:
            pass

    def send(request: Any, **_kwargs: Any) -> AWSResponse:
        response_headers = HTTPHeaders()
        for name, value in headers.items():
            response_headers[name] = value
        return AWSResponse(request.url, 200, response_headers, _Raw())

    provider = BedrockProvider(region_name='us-east-1', aws_access_key_id='x', aws_secret_access_key='x')
    # botocore sends a response returned from `before-send` instead of the request; the stubs type handlers as `-> None`.
    provider.client.meta.events.register(f'before-send.bedrock-runtime.{operation}', send)  # pyright: ignore[reportArgumentType]
    return provider


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
    provider = OpenAIProvider(api_key='mock-api-key', http_client=_mock_sse_client(events, done=True))
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


def _request_span_cost(capfire: CaptureLogfire) -> float:
    [span] = [span for span in capfire.exporter.exported_spans_as_dict() if span['name'].startswith('chat ')]
    return span['attributes']['operation.cost']


@dataclass(frozen=True)
class GoogleStreamCase:
    id: str
    final_part: dict[str, Any]
    expected_cost: Decimal
    expected_parts: list[ModelResponsePart]


GOOGLE_STREAM_CASES = [
    # The gateway adds an empty text part to carry the cost, which doesn't become a part of its own.
    GoogleStreamCase(
        id='empty-part',
        final_part={'text': '', 'partMetadata': GATEWAY_COST_BLOCKS},
        expected_cost=Decimal('0.5'),
        expected_parts=snapshot([TextPart(content='hi')]),
    ),
    # Or it puts the cost on the model's own last part, which can carry a thought signature.
    GoogleStreamCase(
        id='thought-signature',
        final_part={'text': '!', 'thoughtSignature': 'c2lnbmF0dXJl', 'partMetadata': GATEWAY_COST_BLOCKS},
        expected_cost=Decimal('0.5'),
        expected_parts=snapshot(
            [TextPart(content='hi!', provider_name='google', provider_details={'thought_signature': 'c2lnbmF0dXJl'})]
        ),
    ),
    # A response-cache hit is free.
    GoogleStreamCase(
        id='cache-hit',
        final_part={'text': '', 'partMetadata': {'logfire_ai_gateway': {'cost': '0'}}},
        expected_cost=Decimal('0'),
        expected_parts=snapshot([TextPart(content='hi')]),
    ),
    # Without a reported cost, the genai-prices estimate applies.
    GoogleStreamCase(
        id='no-cost',
        final_part={'text': '!'},
        expected_cost=Decimal('0.0028'),
        expected_parts=snapshot([TextPart(content='hi!')]),
    ),
]


@pytest.mark.parametrize('case', GOOGLE_STREAM_CASES, ids=lambda case: case.id)
async def test_gateway_reported_cost_google_stream(
    allow_model_requests: None, capfire: CaptureLogfire, case: GoogleStreamCase
):
    """The gateway reports a stream's cost in the `part_metadata` of the final chunk's last part."""
    chunk = {'modelVersion': 'gemini-2.5-flash', 'responseId': 'resp-1'}
    events: list[dict[str, Any]] = [
        {**chunk, 'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'hi'}]}}]},
        {
            **chunk,
            'candidates': [{'content': {'role': 'model', 'parts': [case.final_part]}, 'finishReason': 'STOP'}],
            'usageMetadata': {'promptTokenCount': 1000, 'candidatesTokenCount': 1000, 'totalTokenCount': 2000},
        },
    ]
    provider = GoogleProvider(api_key='mock-api-key', http_client=_mock_sse_client(events))
    agent = Agent(GoogleModel('gemini-2.5-flash', provider=provider), capabilities=[Instrumentation()])
    async with agent.run_stream('hello') as result:
        await result.get_output()
    assert result.usage.cost == case.expected_cost
    assert result.all_messages()[-1].parts == case.expected_parts
    assert _request_span_cost(capfire) == float(case.expected_cost)


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
    provider = _mock_bedrock_provider(
        'Converse', json.dumps(body).encode(), {'content-type': 'application/json', **headers}
    )
    agent = Agent(BedrockConverseModel(model_name, provider=provider), model_settings={'max_tokens': 100})
    result = await agent.run('hello')
    assert result.usage.cost == expected_cost


def _bedrock_stream_event(event_type: str, payload: dict[str, Any]) -> bytes:
    """Encode one event in the AWS event stream format that `ConverseStream` answers with."""
    headers = b''.join(
        bytes([len(name)]) + name.encode() + b'\x07' + struct.pack('>H', len(value)) + value.encode()
        for name, value in {
            ':event-type': event_type,
            ':content-type': 'application/json',
            ':message-type': 'event',
        }.items()
    )
    body = json.dumps(payload).encode()
    prelude = struct.pack('>II', 16 + len(headers) + len(body), len(headers))
    message = prelude + struct.pack('>I', zlib.crc32(prelude)) + headers + body
    return message + struct.pack('>I', zlib.crc32(message))


@pytest.mark.parametrize(
    'message_stop, expected_cost',
    [
        pytest.param(
            {'stopReason': 'end_turn', 'additionalModelResponseFields': GATEWAY_COST_BLOCKS}, Decimal('0.5'), id='cost'
        ),
        # A response-cache hit is free.
        pytest.param(
            {'stopReason': 'end_turn', 'additionalModelResponseFields': {'logfire_ai_gateway': {'cost': '0'}}},
            Decimal('0'),
            id='cache-hit',
        ),
        # Without a reported cost, the genai-prices estimate applies.
        pytest.param({'stopReason': 'end_turn'}, Decimal('0.0198'), id='no-cost'),
    ],
)
async def test_gateway_reported_cost_bedrock_stream(
    allow_model_requests: None, capfire: CaptureLogfire, message_stop: dict[str, Any], expected_cost: Decimal
):
    """The gateway reports a stream's cost in `messageStop`, before the `metadata` event that carries the tokens."""
    content = b''.join(
        _bedrock_stream_event(event_type, payload)
        for event_type, payload in [
            ('messageStart', {'role': 'assistant'}),
            ('contentBlockDelta', {'contentBlockIndex': 0, 'delta': {'text': 'hi'}}),
            ('contentBlockStop', {'contentBlockIndex': 0}),
            ('messageStop', message_stop),
            (
                'metadata',
                {
                    'usage': {'inputTokens': 1000, 'outputTokens': 1000, 'totalTokens': 2000},
                    'metrics': {'latencyMs': 10},
                },
            ),
        ]
    )
    provider = _mock_bedrock_provider('ConverseStream', content, {'content-type': 'application/vnd.amazon.eventstream'})
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=provider)
    agent = Agent(model, capabilities=[Instrumentation()])
    async with agent.run_stream('hello') as result:
        assert await result.get_output() == 'hi'
    assert result.usage.cost == expected_cost
    assert _request_span_cost(capfire) == float(expected_cost)


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
