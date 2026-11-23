"""Every recorded complete Hugging Face response builds the same `ModelResponse` when it's streamed.

`huggingface_hub` has no stream accumulator to build a complete response from a recorded stream, so this goes the
other way: it renders each recorded complete `ChatCompletionOutput` as the stream that sends it, as
`render_chat_stream` in `tests/models/stream_parity.py` describes, and replays that stream through
`HuggingFaceStreamedResponse` and the complete response through `HuggingFaceModel._process_response`. It also replays
each rendered stream re-chunked.

The chunks are parsed with the SDK's own types, as `AsyncInferenceClient` parses each `data:` line, rather than sent
through the client, which would first look up the model's inference provider on the Hub.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator
from random import Random
from typing import Any

import pytest

from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestParameters

from ..cassette_utils import recorded_interactions, recorded_request_body
from ..conftest import try_import
from ..stream_contract import Known
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
    parse_sse,
    rechunk_sse,
    render_chat_stream,
    split_chat_chunk,
)

with try_import() as imports_successful:
    from huggingface_hub import AsyncInferenceClient, ChatCompletionOutput, ChatCompletionStreamOutput

    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='huggingface_hub not installed')

_CLAIM = claim_of('HuggingFaceModel')


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
            and claimant(uri, json.dumps(completion)) == 'HuggingFaceModel'
        ):
            completions[name] = interaction
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(completions) > 10
    return completions


def _model(interaction: dict[str, Any]) -> HuggingFaceModel:
    return HuggingFaceModel(
        recorded_request_body(interaction['request'])['model'],
        provider=HuggingFaceProvider(
            hf_client=AsyncInferenceClient(
                base_url=interaction['request']['uri'].removesuffix(_CLAIM.endpoint), token='hf_test'
            )
        ),
    )


def _rendered_stream(interaction: dict[str, Any]) -> str:
    completion = _completion(interaction)
    [choice] = json_objects(completion.get('choices'))
    return render_chat_stream(
        completion,
        chunk_fields=('id', 'created', 'model', 'system_fingerprint'),
        deltas=chat_message_deltas(json_object(choice.get('message')), ('reasoning', 'content')),
        final={'usage': completion.get('usage')},
    )


async def _chunks(events: str) -> AsyncIterator[ChatCompletionStreamOutput]:
    for event in parse_sse(events)[0]:
        if event.data is not None:
            # The SDK types the payload it parses as a `dict` of unknown keys and values.
            yield ChatCompletionStreamOutput.parse_obj_as_instance(event.data)  # pyright: ignore[reportUnknownMemberType]


async def _replay(interaction: dict[str, Any], events: str) -> ModelResponse:
    streamed = await _model(interaction)._process_streamed_response(  # pyright: ignore[reportPrivateUsage]
        _chunks(events), ModelRequestParameters()
    )
    async for _ in streamed:
        pass
    return streamed.get()


_KNOWN: dict[str, Known] = {
    'models/cassettes/test_huggingface/test_max_completion_tokens.yaml#0': Known(
        'A response cut off inside `<think>` streams the thinking as a `ThinkingPart`, where '
        "`split_content_into_text_and_thinking` makes an unclosed thinking tag's text a `TextPart`",
        tracked_by='to open: unclosed-thinking-tag-streams-as-thinking',
        repro='tests/models/test_huggingface_stream_parity.py::test_recorded_completions_match_their_rendered_streams',
    ),
}


async def test_recorded_completions_match_their_rendered_streams(
    recorded_completions: dict[str, dict[str, Any]],
) -> None:
    streamed: dict[str, dict[str, object]] = {}
    complete: dict[str, dict[str, object]] = {}
    for name, interaction in recorded_completions.items():
        streamed[name] = comparable_response(await _replay(interaction, _rendered_stream(interaction)))
        completion = ChatCompletionOutput.parse_obj_as_instance(_completion(interaction))  # pyright: ignore[reportUnknownMemberType]
        complete[name] = comparable_response(_model(interaction)._process_response(completion))  # pyright: ignore[reportPrivateUsage]
    for name in _KNOWN:
        assert streamed.pop(name) != complete.pop(name), f'{name} streams what it completes: remove it from `_KNOWN`'
    assert streamed == complete


def _rechunk(events: str, strategy: Strategy, rng: Random) -> str | None:
    return rechunk_sse(events, strategy, rng, split=split_chat_chunk, merge=merge_chat_chunks)


# A rendered stream sends each field whole, so it has no adjacent deltas of one field to merge.
DRIVER = StreamParityDriver(wire=_rendered_stream, replay=_replay, rechunk=_rechunk, strategies=('split-delta',))


# A full sweep over every recording under many seeds stays out of the suite on purpose; see `stream_parity.py`.
async def test_rechunked_streams_match_the_rendered_stream(recorded_completions: dict[str, dict[str, Any]]) -> None:
    await assert_rechunked_streams_match(DRIVER, recorded_completions.items())
