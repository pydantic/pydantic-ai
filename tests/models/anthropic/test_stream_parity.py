"""Every recorded streamed Anthropic response builds the same `ModelResponse` as the message the SDK accumulates from it.

`run()` streams a request whose `max_tokens` is above the SDK's non-streaming limit, which is the default for current
models, so `AnthropicStreamedResponse` must build the same response as `AnthropicModel._process_response` does for a
complete `BetaMessage`. This replays each recorded stream through both, and replays it again re-chunked, as
`tests/models/stream_parity.py` describes.
"""

from __future__ import annotations as _annotations

import itertools
import json
from random import Random
from typing import Any

import httpx2
import pytest

from pydantic_ai import _utils
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestParameters

from ...cassette_utils import recorded_interactions
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
    parse_sse,
    rechunk_sse,
    with_field,
)

with try_import() as imports_successful:
    from anthropic import NOT_GIVEN, AsyncAnthropic, AsyncStream
    from anthropic.lib.streaming import BetaAsyncMessageStream
    from anthropic.types.beta import BetaRawMessageStreamEvent, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')

_CLAIM = claim_of('AnthropicModel')


@pytest.fixture(scope='module')
def recorded_streams() -> dict[str, dict[str, Any]]:
    streams: dict[str, dict[str, Any]] = {}
    for name, interaction in recorded_interactions(_CLAIM.endpoint, _CLAIM.marker):
        content = interaction['response']['body'].get('content')
        if isinstance(content, str) and _CLAIM.marker in content:
            streams[name] = interaction
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(streams) > 200
    return streams


def _events(interaction: dict[str, Any]) -> str:
    return interaction['response']['body']['content']


def _client(events: str) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key='test',
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(
                lambda _: httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=events.encode())
            )
        ),
    )


async def _open_stream(client: AsyncAnthropic, model_name: str) -> AsyncStream[BetaRawMessageStreamEvent]:
    return await client.beta.messages.create(model=model_name, max_tokens=1, messages=[], stream=True)


def _enable_server_tools(monkeypatch: pytest.MonkeyPatch, model: AnthropicModel, events: str) -> None:
    # Both sides drop a server tool call whose tool the request didn't enable, which depends on request parameters the
    # recording doesn't keep, so every server tool the stream calls counts as enabled.
    sse_events, _ = parse_sse(events)
    blocks = [json_object((event.data or {}).get('content_block')) for event in sse_events]
    names = frozenset(
        name
        for block in blocks
        if block.get('type') == 'server_tool_use' and isinstance(name := block.get('name'), str)
    )
    monkeypatch.setattr(model, '_get_enabled_server_tool_names', lambda *_: names)  # pyright: ignore[reportUnknownArgumentType, reportUnknownLambdaType]


async def _replay(interaction: dict[str, Any], events: str) -> ModelResponse:
    model_name = interaction['request']['body']['content']['model']
    client = _client(events)
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))
    with pytest.MonkeyPatch.context() as monkeypatch:
        _enable_server_tools(monkeypatch, model, events)
        streamed = await model._process_streamed_response(  # pyright: ignore[reportPrivateUsage]
            _utils.PeekableAsyncStream(await _open_stream(client, model_name)), ModelRequestParameters(), {}
        )
    async for _ in streamed:
        pass
    return streamed.get()


async def _streamed_and_complete(
    interaction: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], dict[str, object]]:
    model_name = interaction['request']['body']['content']['model']
    events = _events(interaction)
    client = _client(events)
    model = AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))

    message = await BetaAsyncMessageStream(
        await _open_stream(client, model_name), output_format=NOT_GIVEN
    ).get_final_message()
    # The SDK's accumulator copies only the usage fields it knows from `message_delta`, so it drops the cost that
    # Pydantic AI Gateway reports there. The streamed response reads it, as `_process_response` does on a complete
    # message from the gateway, so the reference gets it back from the final `message_delta`.
    delta_usages: list[dict[str, object]] = [
        json.loads(line.removeprefix('data: '))['usage']
        for line in events.splitlines()
        if line.startswith('data: ') and '"type":"message_delta"' in line.replace(' ', '')
    ]
    if gateway_usage := {
        key: value
        for key, value in (delta_usages[-1] if delta_usages else {}).items()
        if key in ('logfire_ai_gateway', 'pydantic_ai_gateway')
    }:
        message.usage = BetaUsage.model_validate({**message.usage.model_dump(), **gateway_usage})
    _enable_server_tools(monkeypatch, model, events)
    complete = model._process_response(message, ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage]
    return comparable_response(await _replay(interaction, events)), comparable_response(complete)


# Splits the replays across tests so that no single test carries all of them against the per-test budget.
_STRIDE = 8


# One worker runs every offset, so the cassettes are scanned once.
@pytest.mark.xdist_group(name='anthropic_stream_parity')
@pytest.mark.parametrize('offset', range(_STRIDE))
async def test_recorded_streams_match_the_complete_message(
    offset: int, recorded_streams: dict[str, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    streamed: dict[str, dict[str, object]] = {}
    complete: dict[str, dict[str, object]] = {}
    for name, interaction in itertools.islice(recorded_streams.items(), offset, None, _STRIDE):
        streamed[name], complete[name] = await _streamed_and_complete(interaction, monkeypatch)
    assert streamed == complete


# The text, thinking and tool input deltas, by the field of the delta that carries them.
_DELTA_FIELDS = {'text_delta': 'text', 'thinking_delta': 'thinking', 'input_json_delta': 'partial_json'}


def _split(data: JsonObject, rng: Random) -> tuple[JsonObject, JsonObject] | None:
    delta = json_object(data.get('delta'))
    key = _DELTA_FIELDS.get(str(delta.get('type')))
    if data.get('type') != 'content_block_delta' or key is None:
        return None
    text = delta.get(key)
    if not isinstance(text, str) or (pieces := cut(text, rng)) is None:
        return None
    first, second = pieces
    return with_field(data, 'delta', with_field(delta, key, first)), with_field(
        data, 'delta', with_field(delta, key, second)
    )


def _merge(first: JsonObject, second: JsonObject) -> JsonObject | None:
    first_delta, second_delta = json_object(first.get('delta')), json_object(second.get('delta'))
    key = _DELTA_FIELDS.get(str(first_delta.get('type')))
    if (
        key is None
        or first.get('type') != 'content_block_delta'
        or second.get('type') != 'content_block_delta'
        or first.get('index') != second.get('index')
        or first_delta.get('type') != second_delta.get('type')
    ):
        return None
    first_text, second_text = first_delta.get(key), second_delta.get(key)
    if not isinstance(first_text, str) or not isinstance(second_text, str):
        return None
    return with_field(first, 'delta', with_field(first_delta, key, first_text + second_text))


def _rechunk(events: str, strategy: Strategy, rng: Random) -> str | None:
    return rechunk_sse(events, strategy, rng, split=_split, merge=_merge)


DRIVER = StreamParityDriver(wire=_events, replay=_replay, rechunk=_rechunk)


# A full sweep over every recording under many seeds stays out of the suite on purpose; see `stream_parity.py`.
@pytest.mark.xdist_group(name='anthropic_stream_parity')
@pytest.mark.parametrize('offset', range(_STRIDE))
async def test_rechunked_streams_match_the_recorded_stream(
    offset: int, recorded_streams: dict[str, dict[str, Any]]
) -> None:
    await assert_rechunked_streams_match(DRIVER, itertools.islice(recorded_streams.items(), offset, None, _STRIDE))
