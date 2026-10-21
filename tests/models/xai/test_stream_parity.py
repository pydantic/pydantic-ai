"""Every recorded streamed xAI response builds the same `ModelResponse` as the response the SDK accumulates from it.

`XaiModel` streams with `chat.stream()`, which yields each chunk together with the SDK's running aggregate of the
chunks so far, so `XaiStreamedResponse` must build the same response as `XaiModel._process_response` does for the
final aggregate. This replays each recorded stream through the SDK's own `Chat.stream()` and builds both.

`XaiStreamedResponse` reads usage, the response ID and the finish reason from the running aggregate, so those are
partly compared against themselves; the parts it builds from the chunk deltas are compared independently.
"""

from __future__ import annotations as _annotations

import dataclasses
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai.messages import BaseToolCallPart, ModelResponse, ModelResponsePart
from pydantic_ai.models import ModelRequestParameters

from ...conftest import try_import
from ..xai_proto_cassettes import StreamInteraction, XaiProtoCassette

with try_import() as imports_successful:
    from xai_sdk import AsyncClient
    from xai_sdk.aio.chat import Chat
    from xai_sdk.proto import chat_pb2

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed')

_TESTS_DIR = Path(__file__).parents[2]


def _recorded_streams() -> dict[str, StreamInteraction]:
    streams: dict[str, StreamInteraction] = {}
    for path in sorted(_TESTS_DIR.rglob('*.xai.yaml')):
        for index, interaction in enumerate(XaiProtoCassette.load(path).interactions):
            if isinstance(interaction, StreamInteraction):
                streams[f'{path.relative_to(_TESTS_DIR)}#{index}'] = interaction
    return streams


def _comparable(response: ModelResponse) -> dict[str, Any]:
    fields = {field.name: getattr(response, field.name) for field in dataclasses.fields(response)}
    del fields['timestamp']
    # A return part stamps the time it was built.
    fields['parts'] = [
        {'type': type(part).__name__, **dataclasses.asdict(part), 'timestamp': None} | _comparable_args(part)
        for part in response.parts
    ]
    return fields


def _comparable_args(part: ModelResponsePart) -> dict[str, Any]:
    # Streamed tool call args are the JSON string the deltas built, where a complete response has a dict.
    return {'args': part.args_as_dict()} if isinstance(part, BaseToolCallPart) else {}


@dataclasses.dataclass
class _RecordedChatStub:
    """Stands in for the gRPC `ChatStub`, answering `Chat.stream()` with the recorded chunks."""

    chunks_raw: list[bytes]

    async def GetCompletionChunk(
        self, request: chat_pb2.GetCompletionsRequest
    ) -> AsyncIterator[chat_pb2.GetChatCompletionChunk]:
        for chunk_raw in self.chunks_raw:
            yield chat_pb2.GetChatCompletionChunk.FromString(chunk_raw)


def _chat(interaction: StreamInteraction) -> Chat:
    chat = Chat(_RecordedChatStub(interaction.chunks_raw), None, None)  # pyright: ignore[reportArgumentType]
    # The SDK aggregates the outputs differently when the request enables server-side tools.
    chat.proto.CopyFrom(chat_pb2.GetCompletionsRequest.FromString(interaction.request_raw))
    return chat


async def _streamed_and_complete(interaction: StreamInteraction) -> tuple[dict[str, Any], dict[str, Any]]:
    model = XaiModel(_chat(interaction).proto.model, provider=XaiProvider(xai_client=AsyncClient(api_key='test')))

    aggregated = None
    async for aggregated, _ in _chat(interaction).stream():
        pass
    assert aggregated is not None
    complete = model._process_response(aggregated)  # pyright: ignore[reportPrivateUsage]

    streamed = await model._process_streamed_response(  # pyright: ignore[reportPrivateUsage]
        _chat(interaction).stream(), ModelRequestParameters()
    )
    async for _ in streamed:
        pass
    return _comparable(streamed.get()), _comparable(complete)


async def test_recorded_streams_match_the_complete_response() -> None:
    streams = _recorded_streams()
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(streams) >= 7

    streamed: dict[str, dict[str, Any]] = {}
    complete: dict[str, dict[str, Any]] = {}
    for name, interaction in streams.items():
        streamed[name], complete[name] = await _streamed_and_complete(interaction)
    assert streamed == complete
