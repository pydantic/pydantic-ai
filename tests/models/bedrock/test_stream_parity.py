"""Every recorded Bedrock `ConverseStream` response builds the same `ModelResponse` as the equivalent `Converse` response.

`BedrockStreamedResponse` must build the same response from the stream's events as `BedrockConverseModel._process_response`
does from the complete `Converse` output. botocore has no `ConverseStream` accumulator, so this folds each recorded
stream's events into the `Converse` output shape (AWS documents `ConverseStream` as the streaming form of `Converse`)
and replays each recording through both, and replays it again re-chunked, as `tests/models/stream_parity.py`
describes.
"""

from __future__ import annotations as _annotations

import binascii
import itertools
import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote

import anyio.to_thread
import pytest
from cassetter import Cassette, HttpInteraction, RecordMode
from pydantic import JsonValue

from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models import ModelRequestParameters

from ...conftest import try_import
from ..stream_parity import (
    JsonObject,
    Strategy,
    StreamParityDriver,
    assert_rechunked_streams_match,
    claim_of,
    comparable_response,
    cut,
    json_object,
    merge_deltas,
    split_deltas,
    with_field,
)

with try_import() as imports_successful:
    import boto3
    from botocore.awsrequest import AWSPreparedRequest, AWSResponse
    from botocore.compat import HTTPHeaders
    from botocore.eventstream import EventStreamBuffer

    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime.type_defs import ConverseResponseTypeDef

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='boto3 not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]

_TESTS_DIR = Path(__file__).parents[2]
_CLAIM = claim_of('BedrockConverseModel')


# Both tests read the streams, so the cassettes are scanned once.
@cache
def _recorded_streams() -> dict[str, HttpInteraction]:
    streams: dict[str, HttpInteraction] = {}
    for path in sorted(_TESTS_DIR.rglob('*.yaml')):
        if _CLAIM.endpoint not in path.read_text():
            continue
        # cassetter reads both its own format and the vcrpy one older cassettes were recorded in.
        cassette = Cassette(path, record_mode=RecordMode.NONE)
        cassette.load()
        for index, interaction in enumerate(cassette.interactions):
            # An error response has no stream to compare.
            if interaction.request.uri.endswith(_CLAIM.endpoint) and interaction.response.status == 200:
                streams[f'{path.relative_to(_TESTS_DIR)}#{index}'] = interaction
    return streams


class _RecordedBody:
    """The raw HTTP response botocore reads an event stream from."""

    def __init__(self, body: bytes):
        self.body = body

    def stream(self, *_: Any, **__: Any) -> Iterator[bytes]:
        yield self.body

    def close(self) -> None:
        pass


@dataclass
class _Replay:
    """A real `bedrock-runtime` client whose `ConverseStream` calls get `interaction`'s recorded response.

    The response streams `body` in place of the recorded body, when it's set.
    """

    client: BedrockRuntimeClient
    interaction: HttpInteraction | None = None
    body: bytes | None = None

    def send(self, request: AWSPreparedRequest, **_: Any) -> AWSResponse:
        assert self.interaction is not None
        response = self.interaction.response
        headers = HTTPHeaders.from_dict({key: value[0] for key, value in response.headers.items()})
        body = _body(self.interaction) if self.body is None else self.body
        return AWSResponse(request.url, response.status, headers, _RecordedBody(body))


@dataclass(frozen=True)
class _Recording:
    """A recorded `ConverseStream` interaction, and the client that replays it."""

    interaction: HttpInteraction
    replay: _Replay


def _body(interaction: HttpInteraction) -> bytes:
    body = interaction.response.body.content
    assert isinstance(body, bytes)
    return body


@pytest.fixture
def replay() -> Iterator[_Replay]:
    client: BedrockRuntimeClient = boto3.client(  # pyright: ignore[reportUnknownMemberType]
        'bedrock-runtime', region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test'
    )
    replay = _Replay(client)
    # A `before-send` handler's response replaces the HTTP request; the stubs type handlers as returning `None`.
    client.meta.events.register('before-send.bedrock-runtime.ConverseStream', replay.send)  # pyright: ignore[reportArgumentType]
    yield replay
    client.close()


def _fold(events: list[dict[str, Any]], *, keeps_whitespace_blocks: bool) -> dict[str, Any]:  # noqa: C901
    """Fold `ConverseStream` events into the `Converse` response they stream."""
    blocks: dict[int, dict[str, Any]] = {}
    tool_inputs: dict[int, str] = {}
    citations: dict[int, list[dict[str, Any]]] = {}
    response: dict[str, Any] = {}
    for event in events:
        if start := event.get('contentBlockStart'):
            index, block = start['contentBlockIndex'], start['start']
            if 'toolUse' in block:
                blocks[index] = {'toolUse': dict(block['toolUse'])}
                tool_inputs[index] = ''
            elif 'toolResult' in block:  # pragma: no branch
                blocks[index] = {'toolResult': {**block['toolResult'], 'content': []}}
        elif delta_event := event.get('contentBlockDelta'):
            index, delta = delta_event['contentBlockIndex'], delta_event['delta']
            block = blocks.setdefault(index, {})
            if 'text' in delta:
                block['text'] = block.get('text', '') + delta['text']
            if 'citation' in delta:
                citations.setdefault(index, []).append(delta['citation'])
            if 'toolUse' in delta:
                tool_inputs[index] += delta['toolUse'].get('input', '')
            if 'toolResult' in delta:
                block['toolResult']['content'].extend(delta['toolResult'])
            if reasoning := delta.get('reasoningContent'):
                target = block.setdefault('reasoningContent', {})
                if 'redactedContent' in reasoning:
                    target['redactedContent'] = target.get('redactedContent', b'') + reasoning['redactedContent']
                else:
                    reasoning_text = target.setdefault('reasoningText', {'text': ''})
                    reasoning_text['text'] += reasoning.get('text', '')
                    if 'signature' in reasoning:
                        reasoning_text['signature'] = reasoning_text.get('signature', '') + reasoning['signature']
        elif message_stop := event.get('messageStop'):
            response['stopReason'] = message_stop['stopReason']
        elif metadata := event.get('metadata'):
            response |= {key: metadata[key] for key in ('usage', 'metrics', 'trace') if key in metadata}
    for index, tool_input in tool_inputs.items():
        blocks[index]['toolUse']['input'] = json.loads(tool_input) if tool_input else {}
    # `Converse` returns a cited text block as `citationsContent`, holding the text and its citations.
    for index, block_citations in citations.items():
        text = blocks[index].pop('text', '')
        blocks[index]['citationsContent'] = {'content': [{'text': text}], 'citations': block_citations}
    # `Converse` leaves out the whitespace-only text blocks that `ConverseStream` sends, like Qwen3's `''` and `'\n\n'`
    # (live-verified on Qwen3 models; see `test_bedrock_qwen_stream_whitespace_text_blocks`). It keeps Claude's: the
    # `' '` between two cited blocks comes back from `Converse` too (recorded in `test_citations.py` before Claude
    # requests were streamed).
    content = [
        block
        for _, block in sorted(blocks.items())
        if keeps_whitespace_blocks or 'text' not in block or block['text'].strip()
    ]
    return {'output': {'message': {'role': 'assistant', 'content': content}}, **response}


def _model_name(interaction: HttpInteraction) -> str:
    return unquote(interaction.request.uri.split('/model/')[1].removesuffix(_CLAIM.endpoint))


async def _replay(recording: _Recording, body: bytes) -> ModelResponse:
    model_name = _model_name(recording.interaction)
    model = BedrockConverseModel(model_name, provider=BedrockProvider(bedrock_client=recording.replay.client))
    recording.replay.interaction, recording.replay.body = recording.interaction, body
    async with model.request_stream([ModelRequest.user_text_prompt('')], None, ModelRequestParameters()) as streamed:
        async for _ in streamed:
            pass
    return streamed.get()


async def _streamed_and_complete(
    interaction: HttpInteraction, replay: _Replay
) -> tuple[dict[str, object], dict[str, object]]:
    model_name = _model_name(interaction)
    model = BedrockConverseModel(model_name, provider=BedrockProvider(bedrock_client=replay.client))
    replay.interaction, replay.body = interaction, None

    def recorded_response() -> dict[str, Any]:
        output = replay.client.converse_stream(modelId=model_name, messages=[])
        # The request ID comes from the HTTP headers, which a stream and a complete response share.
        events = [dict(event) for event in output['stream']]
        return _fold(events, keeps_whitespace_blocks='anthropic' in model_name) | {
            'ResponseMetadata': output['ResponseMetadata']
        }

    response = cast('ConverseResponseTypeDef', await anyio.to_thread.run_sync(recorded_response))
    complete = await model._process_response(response)  # pyright: ignore[reportPrivateUsage]
    streamed = await _replay(_Recording(interaction, replay), _body(interaction))
    return comparable_response(streamed), comparable_response(complete)


async def test_recorded_streams_match_the_complete_response(replay: _Replay) -> None:
    streams = _recorded_streams()
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(streams) > 10

    streamed: dict[str, dict[str, object]] = {}
    complete: dict[str, dict[str, object]] = {}
    for name, interaction in streams.items():
        streamed[name], complete[name] = await _streamed_and_complete(interaction, replay)
    assert streamed == complete


_EventStreamMessage = tuple[dict[str, str], JsonObject]
"""An event stream message's string headers, like `:event-type`, and its JSON payload."""


def _decode(body: bytes) -> list[_EventStreamMessage]:
    buffer = EventStreamBuffer()
    buffer.add_data(body)
    messages: list[_EventStreamMessage] = []
    for message in buffer:
        # The stubs type `EventStreamMessage.headers` as an `int`; its response dict carries the parsed headers.
        recorded_headers: dict[str, object] = message.to_response_dict()['headers']
        headers: dict[str, str] = {}
        for name, value in recorded_headers.items():
            assert isinstance(value, str), (name, value)
            headers[name] = value
        payload: JsonValue = json.loads(message.payload)
        assert isinstance(payload, dict)
        messages.append((headers, payload))
    return messages


def _encode(messages: list[_EventStreamMessage]) -> bytes:
    """The event stream that carries `messages`, framed as AWS's `vnd.amazon.eventstream` with string headers."""
    encoded = b''
    for headers, payload in messages:
        header_bytes = b''.join(
            bytes([len(name.encode())])
            + name.encode()
            + b'\x07'
            + struct.pack('>H', len(value.encode()))
            + value.encode()
            for name, value in headers.items()
        )
        payload_bytes = json.dumps(payload).encode()
        prelude = struct.pack('>II', 16 + len(header_bytes) + len(payload_bytes), len(header_bytes))
        message = prelude + struct.pack('>I', binascii.crc32(prelude)) + header_bytes + payload_bytes
        encoded += message + struct.pack('>I', binascii.crc32(message))
    return encoded


def _delta_text(payload: JsonObject) -> tuple[str, str | None, str] | None:
    """Where a `contentBlockDelta` carries text, thinking or tool input: the delta's key, its nested key and the text."""
    delta = json_object(payload.get('delta'))
    for key, nested in (('text', None), ('toolUse', 'input'), ('reasoningContent', 'text')):
        value = delta.get(key) if nested is None else json_object(delta.get(key)).get(nested)
        if (
            isinstance(value, str)
            and (nested is None or set(json_object(delta.get(key))) == {nested})
            and len(delta) == 1
        ):
            return key, nested, value
    return None


def _with_delta_text(payload: JsonObject, key: str, nested: str | None, text: str) -> JsonObject:
    delta = json_object(payload.get('delta'))
    value = text if nested is None else with_field(json_object(delta.get(key)), nested, text)
    return with_field(payload, 'delta', with_field(delta, key, value))


def _split(message: _EventStreamMessage, rng: Random) -> tuple[_EventStreamMessage, _EventStreamMessage] | None:
    headers, payload = message
    if headers.get(':event-type') != 'contentBlockDelta' or (found := _delta_text(payload)) is None:
        return None
    key, nested, text = found
    if (pieces := cut(text, rng)) is None:
        return None
    return (headers, _with_delta_text(payload, key, nested, pieces[0])), (
        headers,
        _with_delta_text(payload, key, nested, pieces[1]),
    )


def _merge(first: _EventStreamMessage, second: _EventStreamMessage) -> _EventStreamMessage | None:
    (first_headers, first_payload), (second_headers, second_payload) = first, second
    if (
        first_headers.get(':event-type') != 'contentBlockDelta'
        or second_headers.get(':event-type') != 'contentBlockDelta'
        or first_payload.get('contentBlockIndex') != second_payload.get('contentBlockIndex')
        or (first_found := _delta_text(first_payload)) is None
        or (second_found := _delta_text(second_payload)) is None
        or first_found[:2] != second_found[:2]
    ):
        return None
    key, nested, text = first_found
    return first_headers, _with_delta_text(first_payload, key, nested, text + second_found[2])


def _rechunk(body: bytes, strategy: Strategy, rng: Random) -> bytes | None:
    messages = _decode(body)
    if strategy == 'split-delta':
        rechunked = split_deltas(messages, rng, _split)
    else:
        assert strategy == 'merge-deltas', strategy
        rechunked = merge_deltas(messages, rng, _merge)
    return None if rechunked is None else _encode(rechunked)


def _recorded_body(recording: _Recording) -> bytes:
    return _body(recording.interaction)


DRIVER = StreamParityDriver(wire=_recorded_body, replay=_replay, rechunk=_rechunk)


@pytest.fixture(scope='module')
def recorded_streams() -> dict[str, HttpInteraction]:
    return _recorded_streams()


# Splits the replays across tests so that no single test carries all of them against the per-test budget.
_STRIDE = 2


# A full sweep over every recording under many seeds stays out of the suite on purpose; see `stream_parity.py`.
@pytest.mark.parametrize('offset', range(_STRIDE))
async def test_rechunked_streams_match_the_recorded_stream(
    offset: int, recorded_streams: dict[str, HttpInteraction], replay: _Replay
) -> None:
    recordings = itertools.islice(recorded_streams.items(), offset, None, _STRIDE)
    await assert_rechunked_streams_match(
        DRIVER, ((name, _Recording(interaction, replay)) for name, interaction in recordings)
    )
