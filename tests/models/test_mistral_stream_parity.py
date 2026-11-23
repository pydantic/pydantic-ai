"""Every recorded complete Mistral response builds the same `ModelResponse` when it's streamed.

The Mistral SDK has no stream accumulator to build a complete response from a recorded stream, so this goes the other
way: it renders each recorded complete `ChatCompletionResponse` as the stream that sends it, as `render_chat_stream`
in `tests/models/stream_parity.py` describes, and replays that stream through `MistralStreamedResponse` and the
complete response through `MistralModel._process_response`. It also replays each rendered stream re-chunked, with a
cut tool call continuing in the fragment Mistral sends: the call's index, an empty name and no ID.
"""

from __future__ import annotations as _annotations

import functools
import itertools
import json
from random import Random
from typing import Any

import httpx2
import pytest
from pydantic import JsonValue

from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestParameters

from ..cassette_utils import recorded_interactions, recorded_request_body
from ..conftest import try_import
from .stream_parity import (
    JsonObject,
    Strategy,
    StreamParityDriver,
    assert_rechunked_streams_match,
    chat_message_deltas,
    claim_of,
    claimant,
    comparable_response,
    json_object,
    json_objects,
    merge_chat_chunks,
    rechunk_sse,
    render_chat_stream,
    split_chat_chunk,
)

with try_import() as imports_successful:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='mistralai not installed')

_CLAIM = claim_of('MistralModel')


def _completion(interaction: dict[str, Any]) -> JsonObject:
    return json_object(recorded_request_body(interaction['response']))


@pytest.fixture(scope='module')
def recorded_completions() -> dict[str, dict[str, Any]]:
    completions: dict[str, dict[str, Any]] = {}
    for name, interaction in recorded_interactions(_CLAIM.endpoint, _CLAIM.host):
        uri: str = interaction['request']['uri']
        completion = _completion(interaction)
        if (
            uri.endswith(_CLAIM.endpoint)
            and completion.get('object') == 'chat.completion'
            and claimant(uri, json.dumps(completion)) == 'MistralModel'
        ):
            completions[name] = interaction
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(completions) > 50
    return completions


def _model(interaction: dict[str, Any], response: httpx2.Response) -> MistralModel:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _: response))
    return MistralModel(
        recorded_request_body(interaction['request'])['model'],
        provider=MistralProvider(api_key='test', http_client=client),
    )


def _rendered_stream(interaction: dict[str, Any]) -> str:
    completion = _completion(interaction)
    [choice] = json_objects(completion.get('choices'))
    return render_chat_stream(
        completion,
        chunk_fields=('id', 'created', 'model'),
        deltas=chat_message_deltas(json_object(choice.get('message')), ('content',)),
        final={'usage': completion.get('usage')},
    )


_MESSAGES: list[Any] = [{'role': 'user', 'content': 'hello'}]


async def _replay(interaction: dict[str, Any], events: str) -> ModelResponse:
    model = _model(
        interaction, httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=events.encode())
    )
    streamed = await model._process_streamed_response(  # pyright: ignore[reportPrivateUsage]
        await model.client.chat.stream_async(model=model.model_name, messages=_MESSAGES), ModelRequestParameters()
    )
    async for _ in streamed:
        pass
    return streamed.get()


async def test_recorded_completions_match_their_rendered_streams(
    recorded_completions: dict[str, dict[str, Any]],
) -> None:
    streamed: dict[str, dict[str, object]] = {}
    complete: dict[str, dict[str, object]] = {}
    for name, interaction in recorded_completions.items():
        streamed[name] = comparable_response(await _replay(interaction, _rendered_stream(interaction)))
        model = _model(interaction, httpx2.Response(200, json=_completion(interaction)))
        response = await model.client.chat.complete_async(model=model.model_name, messages=_MESSAGES)
        complete[name] = comparable_response(model._process_response(response))  # pyright: ignore[reportPrivateUsage]
    assert streamed == complete


def _continuation(call: JsonObject, arguments: str) -> JsonValue:
    return {'type': 'function', 'function': {'name': '', 'arguments': arguments}, 'index': call.get('index')}


def _rechunk(events: str, strategy: Strategy, rng: Random) -> str | None:
    return rechunk_sse(
        events,
        strategy,
        rng,
        split=functools.partial(split_chat_chunk, continuation=_continuation),
        merge=merge_chat_chunks,
    )


# A rendered stream sends each field whole, so it has no adjacent deltas of one field to merge.
DRIVER = StreamParityDriver(wire=_rendered_stream, replay=_replay, rechunk=_rechunk, strategies=('split-delta',))


# Splits the replays across tests so that no single test carries all of them against the per-test budget.
_STRIDE = 2


# A full sweep over every recording under many seeds stays out of the suite on purpose; see `stream_parity.py`.
@pytest.mark.parametrize('offset', range(_STRIDE))
async def test_rechunked_streams_match_the_rendered_stream(
    offset: int, recorded_completions: dict[str, dict[str, Any]]
) -> None:
    await assert_rechunked_streams_match(DRIVER, itertools.islice(recorded_completions.items(), offset, None, _STRIDE))
