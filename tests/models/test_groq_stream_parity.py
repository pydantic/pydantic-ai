"""Every recorded complete Groq response builds the same `ModelResponse` when it's streamed.

The Groq SDK has no stream accumulator to build a complete response from a recorded stream, so this goes the other
way: it renders each recorded complete `ChatCompletion` as the stream that sends it, as `render_chat_stream` in
`tests/models/stream_parity.py` describes, and replays that stream through `GroqStreamedResponse` and the complete
response through `GroqModel._process_response`. It also replays each rendered stream re-chunked.
"""

from __future__ import annotations as _annotations

import dataclasses
import itertools
import json
from random import Random
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

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
    generated_tool_call_id,
    json_object,
    json_objects,
    merge_chat_chunks,
    rechunk_sse,
    render_chat_stream,
    split_chat_chunk,
)

with try_import() as imports_successful:
    from groq import AsyncGroq

    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='groq not installed')

_CLAIM = claim_of('GroqModel')


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
            and claimant(uri, json.dumps(completion)) == 'GroqModel'
        ):
            completions[name] = interaction
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(completions) > 50
    return completions


def _model(interaction: dict[str, Any], client: AsyncGroq) -> GroqModel:
    return GroqModel(recorded_request_body(interaction['request'])['model'], provider=GroqProvider(groq_client=client))


def _client(response: httpx.Response) -> AsyncGroq:
    return AsyncGroq(api_key='test', http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)))


def _rendered_stream(interaction: dict[str, Any]) -> str:
    completion = _completion(interaction)
    [choice] = json_objects(completion.get('choices'))
    message = json_object(choice.get('message'))
    deltas: list[JsonObject] = []
    for delta in chat_message_deltas(message, ('reasoning', 'executed_tools', 'content')):
        # Groq streams each executed tool twice: when it's called, without its output, and when it returns.
        if tools := json_objects(delta.get('executed_tools')):
            called: list[JsonValue] = [
                {key: value for key, value in tool.items() if key not in ('output', 'search_results')} for tool in tools
            ]
            deltas.append({'executed_tools': called})
        deltas.append(delta)
    # Groq reports a stream's usage in `x_groq`, with the request ID.
    final: JsonObject = {'x_groq': {**json_object(completion.get('x_groq')), 'usage': completion.get('usage')}}
    return render_chat_stream(
        completion, chunk_fields=('id', 'created', 'model', 'system_fingerprint'), deltas=deltas, final=final
    )


async def _replay(interaction: dict[str, Any], events: str) -> ModelResponse:
    client = _client(httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=events.encode()))
    model = _model(interaction, client)
    streamed = await model._process_streamed_response(  # pyright: ignore[reportPrivateUsage]
        await client.chat.completions.create(model=model.model_name, messages=[], stream=True),
        ModelRequestParameters(),
    )
    async for _ in streamed:
        pass
    return streamed.get()


def _comparable(response: ModelResponse) -> dict[str, object]:
    # A complete response reports the service tier, which none of Groq's recorded streams carry.
    details = {key: value for key, value in (response.provider_details or {}).items() if key != 'service_tier'}
    return comparable_response(dataclasses.replace(response, provider_details=details), generated_tool_call_id)


async def test_recorded_completions_match_their_rendered_streams(
    recorded_completions: dict[str, dict[str, Any]],
) -> None:
    streamed: dict[str, dict[str, object]] = {}
    complete: dict[str, dict[str, object]] = {}
    for name, interaction in recorded_completions.items():
        streamed[name] = _comparable(await _replay(interaction, _rendered_stream(interaction)))
        client = _client(httpx.Response(200, json=_completion(interaction)))
        model = _model(interaction, client)
        completion = await client.chat.completions.create(model=model.model_name, messages=[])
        complete[name] = _comparable(model._process_response(completion))  # pyright: ignore[reportPrivateUsage]
    assert streamed == complete


def _rechunk(events: str, strategy: Strategy, rng: Random) -> str | None:
    return rechunk_sse(events, strategy, rng, split=split_chat_chunk, merge=merge_chat_chunks)


# A rendered stream sends each field whole, so it has no adjacent deltas of one field to merge.
DRIVER = StreamParityDriver(
    wire=_rendered_stream, replay=_replay, rechunk=_rechunk, strategies=('split-delta',), comparable=_comparable
)


_KNOWN: dict[str, Known] = {
    'cassettes/test_thinking_wire_contract/test_reasoning_wire_contract[groq-qwen3-effort-setting].yaml#0'
    ' split-delta seed=0': Known(
        '`ignore_streamed_leading_whitespace` drops a first text delta that is only whitespace, so the `\\n\\n` after '
        '`</think>` is dropped when a chunk sends it alone, and kept, as in the complete response, when it comes with text',
        tracked_by='to open: streamed-leading-whitespace-depends-on-chunks',
        repro='tests/models/test_groq_stream_parity.py::test_rechunked_streams_match_the_rendered_stream',
    ),
}


# Splits the replays across tests so that no single test carries all of them against the per-test budget.
_STRIDE = 2


# A full sweep over every recording under many seeds stays out of the suite on purpose; see `stream_parity.py`.
@pytest.mark.parametrize('offset', range(_STRIDE))
async def test_rechunked_streams_match_the_rendered_stream(
    offset: int, recorded_completions: dict[str, dict[str, Any]]
) -> None:
    await assert_rechunked_streams_match(
        DRIVER, itertools.islice(recorded_completions.items(), offset, None, _STRIDE), _KNOWN
    )


def test_known_rechunkings_name_recordings(recorded_completions: dict[str, dict[str, Any]]) -> None:
    """Each test above sees only a share of the recordings, so this checks a renamed one can't leave `_KNOWN` stale."""
    assert {key.rsplit(' ', 2)[0] for key in _KNOWN} <= recorded_completions.keys()
