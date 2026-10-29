"""Tests for Gemini long decode, which google-genai continues across requests itself.

A long-decode model stops a request at its per-request output cap (32K tokens, 64K streamed) and returns a
continuation token while `max_tokens` has budget left. google-genai (`automatic_continuation`, on by default) then
sends the same request again with the token, inside one `generate_content` or `generate_content_stream` call, and
merges the requests' responses into one with summed usage.

`test_recorded_long_decode` records a live long decode, unary and streamed. The other tests script the requests on
an `httpx2.MockTransport` in the recorded shapes, for what a recording can't pin:

- `test_one_request_is_unchanged`: a response that isn't continued, in the shapes of the continued ones so only how it
  ends differs, including a cap reached without a token, which a live request can't be made to return.
- `test_continued_requests_are_one_response` and `test_streamed_continuation_events`: a decode small enough to pin
  every event and each request's usage.
- `test_per_request_input_limit_reads_the_first_request`: a third request.
- `test_failed_continuation_request_raises` and the `test_stream_drop_*` tests: a request that fails or drops.
"""

from __future__ import annotations as _annotations

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from decimal import Decimal

import httpx2
import pytest

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    FinalResultEvent,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, UsageLimits

from ..._inline_snapshot import snapshot
from ...conftest import IsDatetime, IsStr, RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')

_MODEL_NAME = 'gemini-4-argon'


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode()


def _usage(prompt: int, output: int) -> dict[str, int]:
    """Gemini's `usageMetadata`.

    A stream reports a provisional prompt count on every chunk, and the final one on each request's finishing chunk:
    74, then 110, in the stream recording of `test_recorded_long_decode`.
    """
    return {'promptTokenCount': prompt, 'candidatesTokenCount': output}


def _chunk(
    *parts: dict[str, str],
    response_id: str,
    usage: dict[str, int],
    finish_reason: str | None = None,
    continuation_token: bytes | None = None,
    model_version: str = _MODEL_NAME,
) -> dict[str, object]:
    candidate: dict[str, object] = {'content': {'role': 'model', 'parts': list(parts)}, 'index': 0}
    if finish_reason is not None:
        candidate['finishReason'] = finish_reason
    if continuation_token is not None:
        candidate['continuationToken'] = _b64(continuation_token)
    return {'candidates': [candidate], 'usageMetadata': usage, 'modelVersion': model_version, 'responseId': response_id}


def _sse(*chunks: dict[str, object]) -> httpx2.Response:
    body = ''.join(f'data: {json.dumps(chunk)}\r\n\r\n' for chunk in chunks)
    return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=body.encode())


def _model(
    responses: list[httpx2.Response],
    requests: list[dict[str, object]],
    *,
    model_name: str = _MODEL_NAME,
    vertex: bool = False,
) -> GoogleModel:
    """A model whose requests are answered with `responses` in order, and recorded in `requests`."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return responses.pop(0)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    if vertex:
        return GoogleModel(model_name, provider=GoogleCloudProvider(api_key='test', http_client=client))
    return GoogleModel(model_name, provider=GoogleProvider(api_key='test', http_client=client))


async def _run(
    agent: Agent[None, str],
    stream: bool,
    *,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    prompt: str = 'Count.',
) -> tuple[str, ModelResponse]:
    """Run `agent` on `prompt`, streamed or not, returning its output and its last response."""
    if stream:
        async with agent.run_stream(prompt, model_settings=model_settings, usage_limits=usage_limits) as result:
            output = await result.get_output()
        response = result.all_messages()[-1]
    else:
        run_result = await agent.run(prompt, model_settings=model_settings, usage_limits=usage_limits)
        output = run_result.output
        response = run_result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    return output, response


@dataclass(frozen=True)
class _OneRequestCase:
    id: str
    expected: ModelResponse
    finish_reason: str = 'STOP'
    continuation_token: bytes | None = None
    model_name: str = _MODEL_NAME
    vertex: bool = False


_ONE_REQUEST_CASES = [
    _OneRequestCase(
        id='gemini-api',
        model_name='gemini-3.5-flash',
        expected=snapshot(
            ModelResponse(
                parts=[
                    TextPart(
                        content='1\n2\n3\n',
                        provider_name='google',
                        provider_details={'thought_signature': _b64(b'signature')},
                    )
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=6, cost=Decimal('0.000069')),
                model_name='gemini-3.5-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='request-1',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ),
    ),
    _OneRequestCase(
        id='google-cloud',
        model_name='gemini-3.5-flash',
        vertex=True,
        expected=snapshot(
            ModelResponse(
                parts=[
                    TextPart(
                        content='1\n2\n3\n',
                        provider_name='google-cloud',
                        provider_details={'thought_signature': _b64(b'signature')},
                    )
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=6, cost=Decimal('0.000069')),
                model_name='gemini-3.5-flash',
                timestamp=IsDatetime(),
                provider_name='google-cloud',
                provider_url='https://aiplatform.googleapis.com/',
                canonical_system='google-cloud',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='request-1',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ),
    ),
    # `max_tokens` is spent: the response is truncated, and google-genai does not continue it.
    _OneRequestCase(
        id='max-tokens-with-token',
        finish_reason='MAX_TOKENS',
        continuation_token=b'token-1',
        expected=snapshot(
            ModelResponse(
                parts=[
                    TextPart(
                        content='1\n2\n3\n',
                        provider_name='google',
                        provider_details={'thought_signature': _b64(b'signature')},
                    )
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=6, cost=Decimal('0.00008')),
                model_name='gemini-4-argon',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={'finish_reason': 'MAX_TOKENS'},
                provider_response_id='request-1',
                finish_reason='length',
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ),
    ),
    # The per-request cap was reached, but there is no token to continue from, so the response is truncated.
    _OneRequestCase(
        id='continuation-without-token',
        finish_reason='CONTINUATION',
        expected=snapshot(
            ModelResponse(
                parts=[
                    TextPart(
                        content='1\n2\n3\n',
                        provider_name='google',
                        provider_details={'thought_signature': _b64(b'signature')},
                    )
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=6, cost=Decimal('0.00008')),
                model_name='gemini-4-argon',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={'finish_reason': 'CONTINUATION'},
                provider_response_id='request-1',
                finish_reason='length',
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ),
    ),
]


def _one_request(case: _OneRequestCase, stream: bool) -> list[httpx2.Response]:
    """A response from one request. Streamed, only its finishing chunk carries a signature and the final prompt count."""

    def finishing(text: str) -> dict[str, object]:
        return _chunk(
            {'text': text, 'thoughtSignature': _b64(b'signature')},
            response_id='request-1',
            usage=_usage(10, 6),
            finish_reason=case.finish_reason,
            continuation_token=case.continuation_token,
            model_version=case.model_name,
        )

    if stream:
        return [
            _sse(
                _chunk({'text': '1\n2\n'}, response_id='request-1', usage=_usage(8, 4), model_version=case.model_name),
                _chunk({'text': '3\n'}, response_id='request-1', usage=_usage(8, 6), model_version=case.model_name),
                finishing(''),
            )
        ]
    return [httpx2.Response(200, json=finishing('1\n2\n3\n'))]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in _ONE_REQUEST_CASES])
async def test_one_request_is_unchanged(case: _OneRequestCase, stream: bool, allow_model_requests: None):
    """A response google-genai does not continue is sent and read as before the SDK continued long decode.

    One request goes out, with no continuation field in its body, and the response carries no continuation token
    and no `metadata`. That covers every model that never returns a token, which is every Gemini model before long
    decode. A truncated long decode reads as `'length'`.
    """
    requests: list[dict[str, object]] = []
    model = _model(_one_request(case, stream), requests, model_name=case.model_name, vertex=case.vertex)

    output, response = await _run(Agent(model), stream, model_settings={'max_tokens': 100_000})

    assert output == '1\n2\n3\n'
    assert requests == snapshot(
        [
            {
                'contents': [{'parts': [{'text': 'Count.'}], 'role': 'user'}],
                'generationConfig': {'maxOutputTokens': 100000, 'responseModalities': ['TEXT']},
            }
        ]
    )
    assert response == case.expected


def _two_requests(stream: bool) -> list[httpx2.Response]:
    """A long decode continued once: the first request writes 6 tokens and a signature, the second 2 more.

    Streamed, the first request checkpoints a token mid-stream and finishes with another. The second request's prompt
    re-counts the first request's output.
    """
    if stream:
        return [
            _sse(
                _chunk({'text': '1\n2\n'}, response_id='request-1', usage=_usage(8, 4)),
                _chunk(
                    {'text': '3\n'},
                    response_id='request-1',
                    usage=_usage(8, 6),
                    continuation_token=b'checkpoint-1',
                ),
                _chunk(
                    {'text': '', 'thoughtSignature': _b64(b'signature')},
                    response_id='request-1',
                    usage=_usage(10, 6),
                    finish_reason='CONTINUATION',
                    continuation_token=b'token-1',
                ),
            ),
            _sse(
                _chunk({'text': '4\n5'}, response_id='request-2', usage=_usage(14, 2)),
                _chunk({'text': ''}, response_id='request-2', usage=_usage(16, 2), finish_reason='STOP'),
            ),
        ]
    return [
        httpx2.Response(
            200,
            json=_chunk(
                {'text': '1\n2\n3\n', 'thoughtSignature': _b64(b'signature')},
                response_id='request-1',
                usage=_usage(10, 6),
                finish_reason='CONTINUATION',
                continuation_token=b'token-1',
            ),
        ),
        httpx2.Response(
            200, json=_chunk({'text': '4\n5'}, response_id='request-2', usage=_usage(16, 2), finish_reason='STOP')
        ),
    ]


@pytest.mark.parametrize('stream', [False, True])
async def test_continued_requests_are_one_response(stream: bool, allow_model_requests: None):
    """The requests of a long decode come back as one response, with the same parts streamed or not.

    The second request is the first plus the token, as Gemini requires. Each request's text is its own part: the
    first carries its signature, and a streamed response must not append the second request's text to that part.
    Usage is summed, which is what Google bills; the first request's input tokens are kept for context-size checks.
    """
    requests: list[dict[str, object]] = []
    agent = Agent(_model(_two_requests(stream), requests))

    output, response = await _run(agent, stream, model_settings={'max_tokens': 100_000})

    assert output == '1\n2\n3\n4\n5'
    first, second = requests
    assert second.pop('continuationToken') == _b64(b'token-1')
    assert second == first
    assert response == snapshot(
        ModelResponse(
            parts=[
                TextPart(
                    content='1\n2\n3\n',
                    provider_name='google',
                    provider_details={'thought_signature': _b64(b'signature')},
                ),
                TextPart(content='4\n5'),
            ],
            usage=RequestUsage(input_tokens=26, output_tokens=8, cost=Decimal('0.000132')),
            model_name='gemini-4-argon',
            timestamp=IsDatetime(),
            provider_name='google',
            provider_url='https://generativelanguage.googleapis.com/',
            canonical_system='google',
            provider_details={'finish_reason': 'STOP'},
            provider_response_id='request-2',
            finish_reason='stop',
            run_id=IsStr(),
            conversation_id=IsStr(),
            metadata={'__pydantic_ai__': {'context_input_tokens': 10}},
        )
    )


async def test_streamed_continuation_events(allow_model_requests: None):
    """A streamed long decode is one stream of events, and its usage only grows.

    google-genai hides the first request's `CONTINUATION` finish reason and adds the usage of finished requests to
    every later chunk, so the usage after the last event is the sum of the requests', with nothing counted twice.
    """
    model = _model(_two_requests(stream=True), [])
    events: list[ModelResponseStreamEvent] = []
    usages: list[tuple[int, int]] = []

    async with model.request_stream(
        [ModelRequest.user_text_prompt('Count.')], {'max_tokens': 100_000}, ModelRequestParameters()
    ) as streamed:
        async for event in streamed:
            events.append(event)
            usages.append((streamed.usage.input_tokens, streamed.usage.output_tokens))

    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=TextPart(content='1\n2\n'),
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='3\n')),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta='', provider_name='google', provider_details={'thought_signature': _b64(b'signature')}
                ),
            ),
            PartEndEvent(
                index=0,
                part=TextPart(
                    content='1\n2\n3\n',
                    provider_name='google',
                    provider_details={'thought_signature': _b64(b'signature')},
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=1,
                part=TextPart(content='4\n5'),
                previous_part_kind='text',
            ),
            PartEndEvent(
                index=1,
                part=TextPart(content='4\n5'),
            ),
        ]
    )
    assert usages == snapshot([(8, 4), (8, 4), (8, 6), (10, 6), (24, 8), (24, 8), (26, 8)])


def _three_requests(stream: bool) -> list[httpx2.Response]:
    """A long decode continued twice, each request's prompt re-counting the output so far: 10, 16 and 18 tokens."""
    hops = [('1\n2\n3\n', 10, 6), ('4\n', 16, 2), ('5', 18, 1)]
    responses: list[httpx2.Response] = []
    for number, (text, prompt, output) in enumerate(hops, start=1):
        last = number == len(hops)
        finishing = _chunk(
            {'text': '' if stream else text},
            response_id=f'request-{number}',
            usage=_usage(prompt, output),
            finish_reason='STOP' if last else 'CONTINUATION',
            continuation_token=None if last else f'token-{number}'.encode(),
        )
        if stream:
            responses.append(
                _sse(
                    _chunk({'text': text}, response_id=f'request-{number}', usage=_usage(prompt - 2, output)), finishing
                )
            )
        else:
            responses.append(httpx2.Response(200, json=finishing))
    return responses


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize(
    'limit,error',
    [
        pytest.param(20, None, id='within'),
        pytest.param(9, 'request_input_tokens=10', id='exceeded'),
    ],
)
async def test_per_request_input_limit_reads_the_first_request(
    limit: int, error: str | None, stream: bool, allow_model_requests: None
):
    """`per_request_input_tokens_limit` caps a long decode's context, not the input tokens it is billed for.

    The requests' summed input (44 tokens) re-counts earlier output, so the limit is checked against the first
    request's (10). A stream checks it after every event, including those of the later requests, whose summed input so
    far already counts the earlier ones.
    """
    agent = Agent(_model(_three_requests(stream), []), model_settings={'max_tokens': 100_000})
    usage_limits = UsageLimits(per_request_input_tokens_limit=limit)

    if error is None:
        output, response = await _run(agent, stream, usage_limits=usage_limits)
        assert output == '1\n2\n3\n4\n5'
        assert (response.usage.input_tokens, response.usage.output_tokens) == (44, 9)
    else:
        with pytest.raises(UsageLimitExceeded, match=error):
            await _run(agent, stream, usage_limits=usage_limits)


@pytest.mark.parametrize('stream', [False, True])
async def test_failed_continuation_request_raises(stream: bool, allow_model_requests: None):
    """A continuation request that fails fails the model request, as google-genai does not retry it.

    Not streamed, the output of the requests before it is lost, though billed. Streamed, the run keeps what was
    streamed as an interrupted response carrying the finished requests' usage.
    """
    requests: list[dict[str, object]] = []
    error = {'error': {'code': 503, 'message': 'The model is overloaded.', 'status': 'UNAVAILABLE'}}
    responses = [*_two_requests(stream)[:1], httpx2.Response(503, json=error)]
    agent = Agent(_model(responses, requests), model_settings={'max_tokens': 100_000})

    with capture_run_messages() as messages, pytest.raises(ModelHTTPError) as exc_info:
        await _run(agent, stream)

    assert exc_info.value.status_code == 503
    assert len(requests) == 2
    if stream:
        assert messages[1:] == snapshot(
            [
                ModelResponse(
                    parts=[
                        TextPart(
                            content='1\n2\n3\n',
                            provider_name='google',
                            provider_details={'thought_signature': _b64(b'signature')},
                        )
                    ],
                    usage=RequestUsage(input_tokens=10, output_tokens=6, cost=Decimal('0.00008')),
                    model_name='gemini-4-argon',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_response_id='request-1',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                    state='interrupted',
                )
            ]
        )
    else:
        assert messages[1:] == []


class _DroppedStream(httpx2.AsyncByteStream):
    """A streamed response that sends `chunks` and then loses its connection."""

    def __init__(self, *chunks: dict[str, object]):
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield f'data: {json.dumps(chunk)}\r\n\r\n'.encode()
        raise httpx2.ReadError('Connection lost.')


def _dropped_after(*chunks: dict[str, object]) -> list[httpx2.Response]:
    """A stream that drops after `chunks`, and the request google-genai resumes it with."""
    resumed = _sse(
        _chunk({'text': '2\n3\n'}, response_id='request-2', usage=_usage(10, 4)),
        _chunk({'text': ''}, response_id='request-2', usage=_usage(12, 4), finish_reason='STOP'),
    )
    return [
        httpx2.Response(200, headers={'content-type': 'text/event-stream'}, stream=_DroppedStream(*chunks)),
        resumed,
    ]


_CHECKPOINT = _chunk({'text': '1\n'}, response_id='request-1', usage=_usage(8, 2), continuation_token=b'checkpoint')


@pytest.mark.parametrize(
    'after_checkpoint',
    [
        pytest.param([], id='nothing'),
        pytest.param(
            [{'candidates': [{'index': 0}], 'usageMetadata': _usage(8, 2), 'responseId': 'request-1'}],
            id='no-output',
        ),
    ],
)
async def test_stream_drop_resumes_from_the_last_checkpoint(
    after_checkpoint: list[dict[str, object]], allow_model_requests: None
):
    """google-genai resumes a stream that drops with no output after its last continuation token from that token.

    Live Argon streams carry a token on every chunk from the first checkpoint on, so this is how they resume.
    """
    requests: list[dict[str, object]] = []
    agent = Agent(
        _model(_dropped_after(_CHECKPOINT, *after_checkpoint), requests), model_settings={'max_tokens': 100_000}
    )

    output, _ = await _run(agent, stream=True)

    assert output == '1\n2\n3\n'
    first, second = requests
    assert second.pop('continuationToken') == _b64(b'checkpoint')
    assert second == first


async def test_stream_drop_that_would_repeat_output_raises(allow_model_requests: None):
    """A stream that drops after output no continuation token covers fails instead of resuming.

    google-genai resumes it from the last token, and the resumed request repeats what was streamed after it. The run
    keeps what was streamed as an interrupted response, whose usage counts the resumed request's first chunk.
    """
    uncheckpointed = _chunk({'text': '2\n'}, response_id='request-1', usage=_usage(8, 4))
    agent = Agent(_model(_dropped_after(_CHECKPOINT, uncheckpointed), []), model_settings={'max_tokens': 100_000})

    with capture_run_messages() as messages, pytest.raises(ModelAPIError, match='would repeat that output'):
        await _run(agent, stream=True)

    assert messages[1:] == snapshot(
        [
            ModelResponse(
                parts=[TextPart(content='1\n2\n')],
                usage=RequestUsage(input_tokens=18, output_tokens=8, cost=Decimal('0.000116')),
                model_name='gemini-4-argon',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_response_id='request-1',
                run_id=IsStr(),
                conversation_id=IsStr(),
                metadata={'__pydantic_ai__': {'context_input_tokens': 8}},
                state='interrupted',
            )
        ]
    )


@dataclass(frozen=True)
class _RecordedCase:
    id: str
    stream: bool
    lines: int
    expected: ModelResponse


# 7,500 lines reach the 32K unary cap once, and 14,000 the 64K streamed one, at Argon's lowest thinking level.
_RECORDED_CASES = [
    _RecordedCase(
        id='unary',
        stream=False,
        lines=7_500,
        expected=snapshot(
            ModelResponse(
                parts=[],
                usage=RequestUsage(
                    details={'text_prompt_tokens': 32986, 'text_candidates_tokens': 36396},
                    input_tokens=32986,
                    input_text_tokens=32986,
                    output_text_tokens=36396,
                    output_tokens=36396,
                    cost=Decimal('0.429932'),
                ),
                model_name='gemini-4-argon',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='Y5DOauOoK-O7qtsP89GUkQc',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
                metadata={'__pydantic_ai__': {'context_input_tokens': 109}},
            )
        ),
    ),
    _RecordedCase(
        id='stream',
        stream=True,
        lines=14_000,
        expected=snapshot(
            ModelResponse(
                parts=[],
                usage=RequestUsage(
                    details={'text_prompt_tokens': 65756, 'text_candidates_tokens': 72897},
                    input_tokens=65756,
                    input_text_tokens=65756,
                    output_text_tokens=72897,
                    output_tokens=72897,
                    cost=Decimal('0.860482'),
                ),
                model_name='gemini-4-argon',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='45LOaoDJFZrQz7IP8LeaqQw',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
                metadata={'__pydantic_ai__': {'context_input_tokens': 110}},
            )
        ),
    ),
]


@pytest.mark.vcr
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in _RECORDED_CASES])
async def test_recorded_long_decode(
    case: _RecordedCase, allow_model_requests: None, gemini_api_key: str, request_capture: RequestCapture
):
    """A live long decode reaches the per-request cap once, and google-genai continues it in a second request.

    The second request is the first plus the token. The response has one part per request, the first carrying its
    signature, summed usage, and the first request's input tokens.
    """
    provider = GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=20 * 60))
    model_settings: ModelSettings = {'max_tokens': 100_000, 'thinking': 'low'}
    agent = Agent(
        GoogleModel(_MODEL_NAME, provider=provider),
        instructions=(
            'Very long outputs are supported and expected here: they are continued automatically across requests. '
            'Never refuse, shorten or summarize because of length. Follow the requested format exactly.'
        ),
        model_settings=model_settings,
    )

    output, response = await _run(
        agent,
        case.stream,
        prompt=(
            f'Print the integers from 1 to {case.lines} in ascending order, one per line. '
            'Output only the numbers: no code, no commentary, no ellipses, no markdown, no skipped numbers.'
        ),
    )

    assert output.split() == [str(number) for number in range(1, case.lines + 1)]
    first, second = request_capture.bodies()
    assert second.pop('continuationToken')
    assert second == first
    assert [(part.part_kind, sorted(part.provider_details or {})) for part in response.parts] == [
        ('text', ['thought_signature']),
        ('text', []),
    ]
    assert replace(response, parts=[]) == case.expected
