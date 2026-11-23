"""Every recorded streamed xAI response builds the same `ModelResponse` as the response the SDK accumulates from it.

`XaiModel` streams with `chat.stream()`, which yields each chunk together with the SDK's running aggregate of the
chunks so far, so `XaiStreamedResponse` must build the same response as `XaiModel._process_response` does for the
final aggregate. This replays each recorded stream through the SDK's own `Chat.stream()` and builds both.

`XaiStreamedResponse` reads usage, the response ID and the finish reason from the running aggregate, so those are
partly compared against themselves; the parts it builds from the chunk deltas are compared independently.

Each recorded stream is also replayed re-chunked, as `tests/models/stream_parity.py` describes.
"""

from __future__ import annotations as _annotations

import dataclasses
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path
from random import Random
from typing import Literal

import pytest

from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestParameters

from ...conftest import try_import
from ..stream_parity import (
    Strategy,
    StreamParityDriver,
    assert_rechunked_streams_match,
    comparable_response,
    cut,
    merge_deltas,
    split_deltas,
)
from ..xai_proto_cassettes import StreamInteraction, XaiProtoCassette

with try_import() as imports_successful:
    from xai_sdk import AsyncClient
    from xai_sdk.aio.chat import Chat
    from xai_sdk.proto import chat_pb2

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed')

_TESTS_DIR = Path(__file__).parents[2]


# Both tests read the streams, so the cassettes are scanned once.
@cache
def _recorded_streams() -> dict[str, StreamInteraction]:
    streams: dict[str, StreamInteraction] = {}
    for path in sorted(_TESTS_DIR.rglob('*.xai.yaml')):
        for index, interaction in enumerate(XaiProtoCassette.load(path).interactions):
            if isinstance(interaction, StreamInteraction):
                streams[f'{path.relative_to(_TESTS_DIR)}#{index}'] = interaction
    return streams


@dataclasses.dataclass
class _RecordedChatStub:
    """Stands in for the gRPC `ChatStub`, answering `Chat.stream()` with the recorded chunks."""

    chunks_raw: list[bytes]

    async def GetCompletionChunk(
        self, request: chat_pb2.GetCompletionsRequest
    ) -> AsyncIterator[chat_pb2.GetChatCompletionChunk]:
        for chunk_raw in self.chunks_raw:
            yield chat_pb2.GetChatCompletionChunk.FromString(chunk_raw)


def _chat(interaction: StreamInteraction, chunks_raw: list[bytes] | None = None) -> Chat:
    """A chat whose stream answers `interaction`'s request with its recorded chunks, or with `chunks_raw`."""
    chat = Chat(_RecordedChatStub(interaction.chunks_raw if chunks_raw is None else chunks_raw), None, None)  # pyright: ignore[reportArgumentType]
    # The SDK aggregates the outputs differently when the request enables server-side tools.
    chat.proto.CopyFrom(chat_pb2.GetCompletionsRequest.FromString(interaction.request_raw))
    return chat


async def _replay(interaction: StreamInteraction, chunks_raw: list[bytes]) -> ModelResponse:
    chat = _chat(interaction, chunks_raw)
    model = XaiModel(chat.proto.model, provider=XaiProvider(xai_client=AsyncClient(api_key='test')))
    streamed = await model._process_streamed_response(chat.stream(), ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    async for _ in streamed:
        pass
    return streamed.get()


async def _streamed_and_complete(interaction: StreamInteraction) -> tuple[dict[str, object], dict[str, object]]:
    model = XaiModel(_chat(interaction).proto.model, provider=XaiProvider(xai_client=AsyncClient(api_key='test')))

    aggregated = None
    async for aggregated, _ in _chat(interaction).stream():
        pass
    assert aggregated is not None
    complete = model._process_response(aggregated)  # pyright: ignore[reportPrivateUsage]
    return comparable_response(await _replay(interaction, interaction.chunks_raw)), comparable_response(complete)


async def test_recorded_streams_match_the_complete_response() -> None:
    streams = _recorded_streams()
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(streams) >= 7

    streamed: dict[str, dict[str, object]] = {}
    complete: dict[str, dict[str, object]] = {}
    for name, interaction in streams.items():
        streamed[name], complete[name] = await _streamed_and_complete(interaction)
    assert streamed == complete


_DeltaField = Literal['content', 'reasoning_content']


def _delta_text(output: chat_pb2.CompletionOutputChunk) -> tuple[_DeltaField, str] | None:
    """The one text field an assistant output's delta carries, if it carries only that and its role."""
    delta = output.delta
    if (
        delta.role != chat_pb2.MessageRole.ROLE_ASSISTANT
        or delta.tool_calls
        or delta.encrypted_content
        or delta.citations
    ):
        return None
    if delta.content and not delta.reasoning_content:
        return 'content', delta.content
    if delta.reasoning_content and not delta.content:
        return 'reasoning_content', delta.reasoning_content
    return None


def _with_delta_text(chunk: chat_pb2.GetChatCompletionChunk, field: _DeltaField, text: str) -> bytes:
    rechunked = chat_pb2.GetChatCompletionChunk()
    rechunked.CopyFrom(chunk)
    if field == 'content':
        rechunked.outputs[0].delta.content = text
    else:
        rechunked.outputs[0].delta.reasoning_content = text
    return rechunked.SerializeToString()


def _split(chunk_raw: bytes, rng: Random) -> tuple[bytes, bytes] | None:
    """A chunk cut in two at a seeded offset of its one output's text or thinking.

    The first piece carries only that head; the second carries the rest of the chunk, so its finish reason, logprobs
    and citations arrive once, where the stream ended them.
    """
    chunk = chat_pb2.GetChatCompletionChunk.FromString(chunk_raw)
    if len(chunk.outputs) != 1 or (found := _delta_text(chunk.outputs[0])) is None:
        return None
    field, text = found
    if (pieces := cut(text, rng)) is None:
        return None
    head = chat_pb2.GetChatCompletionChunk()
    head.CopyFrom(chunk)
    del head.citations[:]
    head.ClearField('output_files')
    head.ClearField('debug_output')
    output = head.outputs[0]
    output.ClearField('finish_reason')
    output.ClearField('logprobs')
    return _with_delta_text(head, field, pieces[0]), _with_delta_text(chunk, field, pieces[1])


def _merge(first_raw: bytes, second_raw: bytes) -> bytes | None:
    """The one chunk that carries two adjacent chunks' text or thinking, if the first carries nothing else."""
    first = chat_pb2.GetChatCompletionChunk.FromString(first_raw)
    second = chat_pb2.GetChatCompletionChunk.FromString(second_raw)
    if (
        len(first.outputs) != 1
        or len(second.outputs) != 1
        or first.outputs[0].index != second.outputs[0].index
        or first.outputs[0].finish_reason
        or first.outputs[0].HasField('logprobs')
        or first.citations
        or first.output_files
        or first.HasField('debug_output')
        or (first_found := _delta_text(first.outputs[0])) is None
        or (second_found := _delta_text(second.outputs[0])) is None
        or first_found[0] != second_found[0]
    ):
        return None
    return _with_delta_text(second, first_found[0], first_found[1] + second_found[1])


def _rechunk(chunks_raw: list[bytes], strategy: Strategy, rng: Random) -> list[bytes] | None:
    if strategy == 'split-delta':
        return split_deltas(chunks_raw, rng, _split)
    assert strategy == 'merge-deltas', strategy
    return merge_deltas(chunks_raw, rng, _merge)


def _recorded_chunks(interaction: StreamInteraction) -> list[bytes]:
    return interaction.chunks_raw


DRIVER = StreamParityDriver(wire=_recorded_chunks, replay=_replay, rechunk=_rechunk)


@pytest.fixture(scope='module')
def recorded_streams() -> dict[str, StreamInteraction]:
    return _recorded_streams()


# A full sweep over every recording under many seeds stays out of the suite on purpose; see `stream_parity.py`.
async def test_rechunked_streams_match_the_recorded_stream(recorded_streams: dict[str, StreamInteraction]) -> None:
    await assert_rechunked_streams_match(DRIVER, recorded_streams.items())
