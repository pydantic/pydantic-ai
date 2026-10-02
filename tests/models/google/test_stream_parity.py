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
"""

from __future__ import annotations as _annotations

import dataclasses
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml

from pydantic_ai.messages import BaseToolCallPart, ModelResponse, ModelResponsePart, NativeToolReturnPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import AbstractNativeTool, CodeExecutionTool, FileSearchTool, WebFetchTool, WebSearchTool

from ...conftest import try_import

with try_import() as imports_successful:
    from google.genai.types import Candidate, Content, GenerateContentResponse, Part

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')

_TESTS_DIR = Path(__file__).parents[2]
_MODEL_NAME_PATTERN = re.compile(r'/models/([^/:]+):streamGenerateContent')


def _recorded_streams() -> dict[str, dict[str, Any]]:
    streams: dict[str, dict[str, Any]] = {}
    for path in sorted(_TESTS_DIR.rglob('*.yaml')):
        text = path.read_text()
        if ':streamGenerateContent' not in text:
            continue
        for index, interaction in enumerate(yaml.safe_load(text)['interactions']):
            if ':streamGenerateContent' in interaction['request']['uri']:
                streams[f'{path.relative_to(_TESTS_DIR)}#{index}'] = interaction
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


def _comparable(response: ModelResponse) -> dict[str, Any]:
    fields = {field.name: getattr(response, field.name) for field in dataclasses.fields(response)}
    del fields['timestamp']
    # A return part stamps the time it was built.
    fields['parts'] = [
        {'type': type(part).__name__, **dataclasses.asdict(part), 'timestamp': None} | _comparable_tool_fields(part)
        for part in response.parts
    ]
    return fields


def _comparable_tool_fields(part: ModelResponsePart) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if isinstance(part, BaseToolCallPart | NativeToolReturnPart) and part.tool_call_id.startswith('pyd_ai_'):
        # Gemini doesn't id code execution calls or the native tool calls rebuilt from grounding metadata, so
        # each side generates its own.
        fields['tool_call_id'] = '<generated>'
    if isinstance(part, BaseToolCallPart):
        # A streamed tool call's args can be the JSON string its deltas built, where a complete response has a dict.
        fields['args'] = part.args_as_dict()
    return fields


async def _streamed_and_complete(interaction: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model_name = _MODEL_NAME_PATTERN.search(interaction['request']['uri'])
    assert model_name is not None
    body = interaction['response']['body']['string'].encode()
    headers = {key: values[0] for key, values in interaction['response']['headers'].items()}
    headers.pop('content-encoding', None)
    headers.pop('content-length', None)
    provider = GoogleProvider(
        api_key='test',
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(lambda _: httpx2.Response(200, headers=headers, content=body))
        ),
    )
    model = GoogleModel(model_name.group(1), provider=provider)
    model_request_parameters = ModelRequestParameters(native_tools=_native_tools(interaction['request']['parsed_body']))

    async def open_stream() -> AsyncIterator[GenerateContentResponse]:
        return await provider.client.aio.models.generate_content_stream(model=model.model_name, contents='')

    chunks = [chunk async for chunk in await open_stream()]
    complete = model._process_response(_merge_chunks(chunks), model_request_parameters)  # pyright: ignore[reportPrivateUsage]

    streamed = await model._process_streamed_response(await open_stream(), model_request_parameters)  # pyright: ignore[reportPrivateUsage]
    async for _ in streamed:
        pass
    return _comparable(streamed.get()), _comparable(complete)


async def test_recorded_streams_match_the_complete_response() -> None:
    streams = _recorded_streams()
    # Guards against a cassette format change silently leaving nothing to compare.
    assert len(streams) >= 20

    streamed: dict[str, dict[str, Any]] = {}
    complete: dict[str, dict[str, Any]] = {}
    for name, interaction in streams.items():
        streamed[name], complete[name] = await _streamed_and_complete(interaction)
    assert streamed == complete
