"""Every recorded streamed Gemini response builds the same `ModelResponse` as the complete response its chunks add up to.

`GeminiStreamedResponse` and `GoogleModel._process_response` each turn Gemini's wire format into a `ModelResponse`, so
this replays each recorded stream through both and asserts they agree.

google-genai has no stream accumulator (`Chat.send_message_stream` just appends every chunk's `Content` to the
history), so `_merge_chunks` builds the complete response from the chunks by the rules Gemini itself follows:

- Parts are the chunks' parts in order, with consecutive text parts of the same kind (thought or not) joined, unless
  both carry a `thought_signature`: the thought signature docs say never to merge two signed parts, and a joined part
  keeps its one signature. Empty unsigned text parts carry nothing and are dropped.
- Every other field is the last value a chunk sent, since `usage_metadata` is cumulative and `finish_reason`,
  `grounding_metadata` and friends arrive on the final chunk, except `create_time` and the HTTP response, which come
  from the first chunk.

The joining rule is what the non-streamed responses in the thought signature cassettes look like, and every stream
that signs a part in the middle of its text matches under it, which is independent evidence that the merge is sound.

Each recorded stream is also replayed re-chunked, as `tests/models/stream_parity.py` describes.
"""

from __future__ import annotations as _annotations

import itertools
import re
from collections.abc import AsyncIterator
from functools import cache
from random import Random
from typing import Any

import httpx2
import pytest
from pydantic import JsonValue

from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import AbstractNativeTool, CodeExecutionTool, FileSearchTool, WebFetchTool, WebSearchTool

from ...cassette_utils import recorded_interactions, recorded_request_body
from ...conftest import try_import
from ..stream_parity import (
    JsonObject,
    Strategy,
    StreamParityDriver,
    assert_rechunked_streams_match,
    claim_of,
    comparable_response,
    cut,
    generated_tool_call_id,
    json_object,
    json_objects,
    rechunk_sse,
    with_field,
)

with try_import() as imports_successful:
    from google.genai.types import Candidate, Content, GenerateContentResponse, Part

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')

_MODEL_NAME_PATTERN = re.compile(r'/models/([^/:]+):streamGenerateContent')
_CLAIM = claim_of('GoogleModel')


# Both tests read the streams, so the cassettes are scanned once.
@cache
def _recorded_streams() -> dict[str, dict[str, Any]]:
    streams: dict[str, dict[str, Any]] = {}
    for name, interaction in recorded_interactions(_CLAIM.endpoint):
        if _CLAIM.endpoint in interaction['request']['uri']:
            streams[name] = interaction
    return streams


def _native_tools(request_body: dict[str, Any]) -> list[AbstractNativeTool]:
    # The streamed processor reads the enabled native tools (file search runs as `executable_code` on Gemini 2.5).
    tools: list[AbstractNativeTool] = []
    request_tools: list[dict[str, Any]] = request_body.get('tools') or []
    for tool in request_tools:
        if 'fileSearch' in tool:
            tools.append(FileSearchTool(file_store_ids=tool['fileSearch']['file_search_store_names']))
        elif 'googleSearch' in tool:
            tools.append(WebSearchTool())
        elif 'codeExecution' in tool:
            tools.append(CodeExecutionTool())
        elif 'urlContext' in tool:
            tools.append(WebFetchTool())
    return tools


def _merge_parts(parts: list[Part]) -> list[Part]:
    merged: list[Part] = []
    for part in parts:
        if part.text == '' and not part.thought_signature:
            continue
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.text is not None
            and part.text is not None
            and bool(previous.thought) == bool(part.thought)
            and not (previous.thought_signature and part.thought_signature)
        ):
            merged[-1] = previous.model_copy(
                update={
                    'text': previous.text + part.text,
                    'thought_signature': previous.thought_signature or part.thought_signature,
                }
            )
        else:
            merged.append(part)
    return merged


def _merge_chunks(chunks: list[GenerateContentResponse]) -> GenerateContentResponse:
    parts: list[Part] = []
    candidate_fields: dict[str, Any] = {}
    response_fields: dict[str, Any] = {}
    for chunk in chunks:
        for key in ('usage_metadata', 'model_version', 'response_id', 'prompt_feedback'):
            if (value := getattr(chunk, key)) is not None:
                response_fields[key] = value
        for key in ('create_time', 'sdk_http_response'):
            if (value := getattr(chunk, key)) is not None:
                response_fields.setdefault(key, value)
        candidate = chunk.candidates[0] if chunk.candidates else Candidate()
        parts.extend(candidate.content.parts or [] if candidate.content else [])
        for key in (
            'finish_reason',
            'safety_ratings',
            'grounding_metadata',
            'url_context_metadata',
            'avg_logprobs',
            'logprobs_result',
        ):
            if (value := getattr(candidate, key)) is not None:
                candidate_fields[key] = value
    merged = Candidate(content=Content(role='model', parts=_merge_parts(parts)), **candidate_fields)
    return GenerateContentResponse(candidates=[merged] if parts or candidate_fields else None, **response_fields)


def _comparable(response: ModelResponse) -> dict[str, object]:
    # Gemini doesn't id code execution calls or the native tool calls rebuilt from grounding metadata.
    return comparable_response(response, generated_tool_call_id)


def _events(interaction: dict[str, Any]) -> str:
    # vcrpy recorded the raw text as `string`; cassetter writes a typed `{type, content}` mapping.
    recorded_body = interaction['response']['body']
    return recorded_body['string'] if 'string' in recorded_body else recorded_body['content']


def _model(interaction: dict[str, Any], events: str) -> GoogleModel:
    model_name = _MODEL_NAME_PATTERN.search(interaction['request']['uri'])
    assert model_name is not None
    headers = {key: values[0] for key, values in interaction['response']['headers'].items()}
    headers.pop('content-encoding', None)
    headers.pop('content-length', None)
    provider = GoogleProvider(
        api_key='test',
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(lambda _: httpx2.Response(200, headers=headers, content=events.encode()))
        ),
    )
    return GoogleModel(model_name.group(1), provider=provider)


def _model_request_parameters(interaction: dict[str, Any]) -> ModelRequestParameters:
    return ModelRequestParameters(native_tools=_native_tools(recorded_request_body(interaction['request'])))


async def _open_stream(model: GoogleModel) -> AsyncIterator[GenerateContentResponse]:
    # One recorded response is compared, so a long decode's continuation token must not send another request.
    return await model.client.aio.models.generate_content_stream(
        model=model.model_name, contents='', config={'automatic_continuation': False}
    )


async def _replay(interaction: dict[str, Any], events: str) -> ModelResponse:
    model = _model(interaction, events)
    streamed = await model._process_streamed_response(  # pyright: ignore[reportPrivateUsage]
        await _open_stream(model), _model_request_parameters(interaction)
    )
    async for _ in streamed:
        pass
    return streamed.get()


async def _streamed_and_complete(interaction: dict[str, Any]) -> tuple[dict[str, object], dict[str, object]]:
    events = _events(interaction)
    model = _model(interaction, events)
    chunks = [chunk async for chunk in await _open_stream(model)]
    complete = model._process_response(_merge_chunks(chunks), _model_request_parameters(interaction))  # pyright: ignore[reportPrivateUsage]
    return _comparable(await _replay(interaction, events)), _comparable(complete)


async def test_recorded_streams_match_the_complete_response() -> None:
    streams = _recorded_streams()
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(streams) >= 20

    streamed: dict[str, dict[str, object]] = {}
    complete: dict[str, dict[str, object]] = {}
    for name, interaction in streams.items():
        streamed[name], complete[name] = await _streamed_and_complete(interaction)
    assert streamed == complete


# The response's fields that hold for every chunk of it, which each piece of a cut chunk carries.
_RESPONSE_IDENTITY = ('modelVersion', 'responseId', 'createTime')


def _split(chunk: JsonObject, rng: Random) -> tuple[JsonObject, JsonObject] | None:
    """A chunk cut in two at a seeded offset of one of its unsigned text parts.

    The first piece carries the parts before the cut and the head of the cut part, and the URL context metadata that
    Gemini sends before the text; the second carries the rest of the chunk, so its usage, finish reason, grounding and
    continuation token still arrive with the parts they come after. A signed part isn't cut: which piece would carry
    its signature is up to Gemini.
    """
    candidates = json_objects(chunk.get('candidates'))
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    content = json_object(candidate.get('content'))
    parts = json_objects(content.get('parts'))
    texts = [index for index, part in enumerate(parts) if set(part) <= {'text', 'thought'}]
    if not texts:
        return None
    index = rng.choice(texts)
    text = parts[index]['text']
    if not isinstance(text, str) or (pieces := cut(text, rng)) is None:
        return None
    head_candidate: JsonObject = {
        key: value for key, value in candidate.items() if key in ('index', 'urlContextMetadata')
    }
    head_candidate['content'] = with_field(
        content, 'parts', [*parts[:index], with_field(parts[index], 'text', pieces[0])]
    )
    head: JsonObject = {key: value for key, value in chunk.items() if key in _RESPONSE_IDENTITY}
    head['candidates'] = [head_candidate]
    tail_candidate = {key: value for key, value in candidate.items() if key != 'urlContextMetadata'}
    tail_candidate['content'] = with_field(
        content, 'parts', [with_field(parts[index], 'text', pieces[1]), *parts[index + 1 :]]
    )
    return head, with_field(chunk, 'candidates', [tail_candidate])


def _merge(first: JsonObject, second: JsonObject) -> JsonObject | None:
    """The one chunk that carries two adjacent chunks' parts, if the first carries nothing the second doesn't repeat.

    The second's usage replaces the first's, as Gemini's usage is the running total.
    """
    first_candidates, second_candidates = json_objects(first.get('candidates')), json_objects(second.get('candidates'))
    if len(first_candidates) != 1 or len(second_candidates) != 1:
        return None
    (first_candidate,), (second_candidate,) = first_candidates, second_candidates
    if any(
        value != second.get(key) for key, value in first.items() if key not in ('candidates', 'usageMetadata')
    ) or any(value != second_candidate.get(key) for key, value in first_candidate.items() if key != 'content'):
        return None
    first_content, second_content = (
        json_object(first_candidate.get('content')),
        json_object(second_candidate.get('content')),
    )
    parts: list[JsonValue] = [*json_objects(first_content.get('parts')), *json_objects(second_content.get('parts'))]
    if not parts:
        return None
    return with_field(
        second, 'candidates', [with_field(second_candidate, 'content', with_field(second_content, 'parts', parts))]
    )


def _rechunk(events: str, strategy: Strategy, rng: Random) -> str | None:
    return rechunk_sse(events, strategy, rng, split=_split, merge=_merge)


DRIVER = StreamParityDriver(wire=_events, replay=_replay, rechunk=_rechunk, comparable=_comparable)


@pytest.fixture(scope='module')
def recorded_streams() -> dict[str, dict[str, Any]]:
    return _recorded_streams()


# Splits the replays across tests so that no single test carries all of them against the per-test budget.
_STRIDE = 2


# A full sweep over every recording under many seeds stays out of the suite on purpose; see `stream_parity.py`.
@pytest.mark.parametrize('offset', range(_STRIDE))
async def test_rechunked_streams_match_the_recorded_stream(
    offset: int, recorded_streams: dict[str, dict[str, Any]]
) -> None:
    await assert_rechunked_streams_match(DRIVER, itertools.islice(recorded_streams.items(), offset, None, _STRIDE))
