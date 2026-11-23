"""Every recorded complete Cohere response builds the same `ModelResponse` when it's streamed.

The Cohere SDK has no stream accumulator to build a complete response from a recorded stream, so this goes the other
way: it renders each recorded complete `V2ChatResponse` as the `/v2/chat` event stream that sends it, and replays
that stream through `CohereStreamedResponse` and the complete response through `CohereModel.request`. It also
replays each rendered stream re-chunked, as `tests/models/stream_parity.py` describes.

The stream sends the tool plan, then each content block between its `content-start` and `content-end`, then each tool
call as a `tool-call-start` that names it and a `tool-call-delta` with its arguments, then the finish reason and
usage in `message-end`, as the recorded streams do. The citations are left out: neither side reads them.
"""

from __future__ import annotations as _annotations

import json
from random import Random
from typing import Any

import httpx
import pytest
from pydantic import JsonValue

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.models import ModelRequestParameters

from ..cassette_utils import recorded_interactions, recorded_request_body
from ..conftest import try_import
from .stream_parity import (
    JsonObject,
    Strategy,
    StreamParityDriver,
    assert_rechunked_streams_match,
    claim_of,
    claimant,
    comparable_response,
    cut,
    json_object,
    json_objects,
    rechunk_sse,
)

with try_import() as imports_successful:
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='cohere not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]

_CLAIM = claim_of('CohereModel')


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
            and isinstance(completion.get('message'), dict)
            and claimant(uri, json.dumps(completion)) == 'CohereModel'
        ):
            completions[name] = interaction
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(completions) > 10
    return completions


def _model(interaction: dict[str, Any], response: httpx.Response) -> CohereModel:
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response))
    return CohereModel(
        recorded_request_body(interaction['request'])['model'],
        provider=CohereProvider(api_key='test', http_client=client),
    )


def _event(data: JsonObject) -> str:
    return f'event: {data["type"]}\ndata: {json.dumps(data)}\n\n'


def _rendered_stream(interaction: dict[str, Any]) -> str:
    completion = _completion(interaction)
    message = json_object(completion.get('message'))
    start: JsonValue = {'role': 'assistant', 'content': [], 'tool_plan': '', 'tool_calls': [], 'citations': []}
    events: list[JsonObject] = [{'id': completion.get('id'), 'type': 'message-start', 'delta': {'message': start}}]
    if tool_plan := message.get('tool_plan'):
        events.append({'type': 'tool-plan-delta', 'delta': {'message': {'tool_plan': tool_plan}}})
    for index, content in enumerate(json_objects(message.get('content'))):
        kind = content.get('type')
        assert isinstance(kind, str), content
        events.extend(
            [
                {'type': 'content-start', 'index': index, 'delta': {'message': {'content': {'type': kind, kind: ''}}}},
                {'type': 'content-delta', 'index': index, 'delta': {'message': {'content': {kind: content.get(kind)}}}},
                {'type': 'content-end', 'index': index},
            ]
        )
    for index, call in enumerate(json_objects(message.get('tool_calls'))):
        function = json_object(call.get('function'))
        named: JsonValue = {**call, 'function': {'name': function.get('name'), 'arguments': ''}}
        arguments: JsonValue = {'function': {'arguments': function.get('arguments')}}
        events.extend(
            [
                {'type': 'tool-call-start', 'index': index, 'delta': {'message': {'tool_calls': named}}},
                {'type': 'tool-call-delta', 'index': index, 'delta': {'message': {'tool_calls': arguments}}},
                {'type': 'tool-call-end', 'index': index},
            ]
        )
    end: JsonValue = {'finish_reason': completion.get('finish_reason'), 'usage': completion.get('usage')}
    events.append({'type': 'message-end', 'delta': end})
    return ''.join(_event(event) for event in events) + 'data: [DONE]\n\n'


_MESSAGES: list[ModelMessage] = [ModelRequest.user_text_prompt('hello')]


async def _replay(interaction: dict[str, Any], events: str) -> ModelResponse:
    model = _model(
        interaction, httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=events.encode())
    )
    async with model.request_stream(_MESSAGES, None, ModelRequestParameters()) as streamed:
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
        model = _model(interaction, httpx.Response(200, json=_completion(interaction)))
        complete[name] = comparable_response(await model.request(_MESSAGES, None, ModelRequestParameters()))
    assert streamed == complete


def _delta_text(event: JsonObject) -> tuple[list[str], str] | None:
    """The path to the text, thinking, tool plan or tool arguments a delta event carries, and that string."""
    message = json_object(json_object(event.get('delta')).get('message'))
    if event.get('type') == 'tool-plan-delta':
        path = ['tool_plan']
    elif event.get('type') == 'tool-call-delta':
        path = ['tool_calls', 'function', 'arguments']
    elif event.get('type') == 'content-delta':
        path = ['content', next(iter(json_object(message.get('content'))), '')]
    else:
        return None
    value: JsonValue = message
    for key in path:
        value = json_object(value).get(key)
    return (path, value) if isinstance(value, str) else None


def _with_delta_text(event: JsonObject, path: list[str], text: str) -> JsonObject:
    value: JsonValue = text
    for key in reversed(path):
        value = {key: value}
    rebuilt: JsonObject = {**event, 'delta': {'message': value}}
    return rebuilt


def _split(event: JsonObject, rng: Random) -> tuple[JsonObject, JsonObject] | None:
    if (found := _delta_text(event)) is None or (pieces := cut(found[1], rng)) is None:
        return None
    return _with_delta_text(event, found[0], pieces[0]), _with_delta_text(event, found[0], pieces[1])


def _merge(first: JsonObject, second: JsonObject) -> JsonObject | None:
    first_found, second_found = _delta_text(first), _delta_text(second)
    if (
        first_found is None
        or second_found is None
        or first.get('type') != second.get('type')
        or first.get('index') != second.get('index')
        or first_found[0] != second_found[0]
    ):
        return None
    return _with_delta_text(first, first_found[0], first_found[1] + second_found[1])


def _rechunk(events: str, strategy: Strategy, rng: Random) -> str | None:
    return rechunk_sse(events, strategy, rng, split=_split, merge=_merge)


# A rendered stream sends each block whole, so it has no adjacent deltas of one block to merge.
DRIVER = StreamParityDriver(wire=_rendered_stream, replay=_replay, rechunk=_rechunk, strategies=('split-delta',))


# A full sweep over every recording under many seeds stays out of the suite on purpose; see `stream_parity.py`.
async def test_rechunked_streams_match_the_rendered_stream(recorded_completions: dict[str, dict[str, Any]]) -> None:
    await assert_rechunked_streams_match(DRIVER, recorded_completions.items())
