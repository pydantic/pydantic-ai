"""Tests for Google native tools serialization in `_content_model_response`.

Covers the message-history echo path that round-trips `NativeToolCallPart` /
`NativeToolReturnPart` between the API and the application:

- pre-Gemini-3 models drop server-side native parts (the API would reject them);
- `pyd_ai_`-synthesized `tool_call_id`s are dropped on every model;
- `CodeExecutionTool` uses `executable_code` / `code_execution_result` parts and is
  preserved regardless of the tool-combination capability.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai.messages import (
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartStartEvent,
    TextPart,
)
from pydantic_ai.native_tools import (
    CodeExecutionTool,
    FileSearchTool,
    WebSearchTool,
)
from pydantic_ai.usage import RequestUsage

from ...conftest import IsDatetime, try_import

with try_import() as imports_successful:
    from google.genai.types import GenerateContentResponse, GroundingMetadata, Part, ToolType

    from pydantic_ai import _utils
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.google import (
        GeminiStreamedResponse,
        _content_model_response,  # pyright: ignore[reportPrivateUsage]
        _process_response_from_parts,  # pyright: ignore[reportPrivateUsage]
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')


def test_content_model_response_pre_gemini_3_drops_native_tool_parts():
    response = ModelResponse(
        parts=[
            NativeToolCallPart(
                tool_name=WebSearchTool.kind,
                provider_name='google-gla',
                tool_call_id='web_search_call',
                args={'query': 'foo'},
            ),
            NativeToolReturnPart(
                tool_name=WebSearchTool.kind,
                provider_name='google-gla',
                tool_call_id='web_search_call',
                content={'result': 'ok'},
            ),
            NativeToolCallPart(
                tool_name=FileSearchTool.kind,
                provider_name='google-gla',
                tool_call_id='file_search_call',
                args={'query': 'bar'},
            ),
            NativeToolReturnPart(
                tool_name=FileSearchTool.kind,
                provider_name='google-gla',
                tool_call_id='file_search_call',
                content={'result': 'ok'},
            ),
            TextPart(content='hello'),
        ],
        provider_name='google-gla',
    )

    assert _content_model_response(response, frozenset({'google-gla'})) == snapshot(
        {'role': 'model', 'parts': [{'text': 'hello'}]}
    )
    assert _content_model_response(response, frozenset({'google-gla'}), supports_tool_combination=True) == snapshot(
        {
            'role': 'model',
            'parts': [
                {
                    'tool_call': {
                        'id': 'web_search_call',
                        'tool_type': ToolType.GOOGLE_SEARCH_WEB,
                        'args': {'query': 'foo'},
                    }
                },
                {
                    'tool_response': {
                        'id': 'web_search_call',
                        'tool_type': ToolType.GOOGLE_SEARCH_WEB,
                        'response': {'result': 'ok'},
                    }
                },
                {
                    'tool_call': {
                        'id': 'file_search_call',
                        'tool_type': ToolType.FILE_SEARCH,
                        'args': {'query': 'bar'},
                    }
                },
                {
                    'tool_response': {
                        'id': 'file_search_call',
                        'tool_type': ToolType.FILE_SEARCH,
                        'response': {'result': 'ok'},
                    }
                },
                {'text': 'hello'},
            ],
        }
    )

    native_only = ModelResponse(
        parts=[
            NativeToolCallPart(
                tool_name=WebSearchTool.kind,
                provider_name='google-gla',
                tool_call_id='web_search_call',
                args={'query': 'foo'},
            ),
            NativeToolReturnPart(
                tool_name=WebSearchTool.kind,
                provider_name='google-gla',
                tool_call_id='web_search_call',
                content={'result': 'ok'},
            ),
        ],
        provider_name='google-gla',
    )
    assert _content_model_response(native_only, frozenset({'google-gla'})) is None


def test_content_model_response_drops_pyd_ai_synthesized_native_tool_ids():
    """`pyd_ai_`-prefixed `tool_call_id`s come from `grounding_metadata` reconstruction in older versions
    of pydantic-ai (or from streaming chunks before native `tool_call`/`tool_response` parts landed).
    The Gemini API rejects unknown ids, so message histories built that way must drop those parts even
    on Gemini 3+, regardless of the model profile.
    """
    response = ModelResponse(
        parts=[
            NativeToolCallPart(
                tool_name=WebSearchTool.kind,
                provider_name='google-gla',
                tool_call_id='pyd_ai_legacy_synthesized',
                args={'queries': ['foo']},
            ),
            NativeToolReturnPart(
                tool_name=WebSearchTool.kind,
                provider_name='google-gla',
                tool_call_id='pyd_ai_legacy_synthesized',
                content=[{'web': {'uri': 'http://example.com'}}],
            ),
            TextPart(content='hello'),
        ],
        provider_name='google-gla',
    )
    assert _content_model_response(response, frozenset({'google-gla'}), supports_tool_combination=True) == snapshot(
        {'role': 'model', 'parts': [{'text': 'hello'}]}
    )


@pytest.mark.parametrize('supports_tool_combination', [False, True])
def test_content_model_response_pre_gemini_3_preserves_code_execution(supports_tool_combination: bool):
    response = ModelResponse(
        parts=[
            NativeToolCallPart(
                tool_name=CodeExecutionTool.kind,
                provider_name='google-gla',
                tool_call_id='code_exec_call',
                args={'language': 'PYTHON', 'code': 'print(1)'},
            ),
            NativeToolReturnPart(
                tool_name=CodeExecutionTool.kind,
                provider_name='google-gla',
                tool_call_id='code_exec_call',
                content={'outcome': 'OUTCOME_OK', 'output': '1\n'},
            ),
        ],
        provider_name='google-gla',
    )

    assert _content_model_response(
        response, frozenset({'google-gla'}), supports_tool_combination=supports_tool_combination
    ) == snapshot(
        {
            'role': 'model',
            'parts': [
                {'executable_code': {'language': 'PYTHON', 'code': 'print(1)'}},
                {'code_execution_result': {'outcome': 'OUTCOME_OK', 'output': '1\n'}},
            ],
        }
    )


# On Gemini 3+, File Search runs as a server-side tool and the API returns explicit `tool_call` /
# `tool_response` parts, but leaves the `tool_response` content empty and delivers the retrieved contexts
# (including each document's `custom_metadata`, e.g. a `source_url`) in `grounding_metadata` instead. These
# tests pin that the empty `NativeToolReturnPart` is filled from the grounding metadata. They are unit tests
# rather than VCR because the cassette matcher is body-insensitive (a changed internal payload could still
# replay green), and the cross-chunk assembly is asserted at the event level, which a VCR test can't reach.
# The streaming tests place the grounding on a later chunk than the empty `tool_response`, reproducing the
# confirmed wire order the fix depends on.

_FILE_SEARCH_GROUNDING_METADATA: dict[str, Any] = {
    'grounding_chunks': [
        {
            'retrieved_context': {
                'text': 'Paris is the capital of France.',
                'title': 'paris.txt',
                'custom_metadata': [{'key': 'source_url', 'string_value': 'https://example.com/paris-facts'}],
                'file_search_store': 'fileSearchStores/test-store',
            }
        }
    ]
}


def test_file_search_grounding_fills_empty_tool_response():
    """The empty file_search `tool_response` is filled from `grounding_metadata`, keeping the explicit parts."""
    response = _process_response_from_parts(
        parts=[
            Part.model_validate({'tool_call': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH', 'args': {}}}),
            Part.model_validate({'tool_response': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH'}}),
            Part.model_validate({'text': 'Paris is the capital of France.'}),
        ],
        grounding_metadata=GroundingMetadata.model_validate(_FILE_SEARCH_GROUNDING_METADATA),
        model_name='gemini-3.5-flash',
        provider_name='google-gla',
        provider_url='https://generativelanguage.googleapis.com/',
        usage=RequestUsage(),
        provider_response_id='response-id',
    )

    assert response.parts == snapshot(
        [
            NativeToolCallPart(
                tool_name='file_search',
                args={},
                tool_call_id='file_search_call',
                provider_name='google-gla',
            ),
            NativeToolReturnPart(
                tool_name='file_search',
                content=[
                    {
                        'text': 'Paris is the capital of France.',
                        'title': 'paris.txt',
                        'custom_metadata': [{'key': 'source_url', 'string_value': 'https://example.com/paris-facts'}],
                        'file_search_store': 'fileSearchStores/test-store',
                    }
                ],
                tool_call_id='file_search_call',
                timestamp=IsDatetime(),
                provider_name='google-gla',
            ),
            TextPart(content='Paris is the capital of France.'),
        ]
    )


def test_file_search_populated_tool_response_not_overwritten():
    """A file_search `tool_response` that already carries content is kept as-is, not clobbered by grounding."""
    response = _process_response_from_parts(
        parts=[
            Part.model_validate({'tool_call': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH', 'args': {}}}),
            Part.model_validate(
                {'tool_response': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH', 'response': {'kept': 'value'}}}
            ),
        ],
        grounding_metadata=GroundingMetadata.model_validate(_FILE_SEARCH_GROUNDING_METADATA),
        model_name='gemini-3.5-flash',
        provider_name='google-gla',
        provider_url='https://generativelanguage.googleapis.com/',
        usage=RequestUsage(),
        provider_response_id='response-id',
    )

    _, file_search_return = response.parts
    assert isinstance(file_search_return, NativeToolReturnPart)
    assert file_search_return.content == {'kept': 'value'}


def _stream_chunk(parts: list[dict[str, Any]], grounding: dict[str, Any] | None = None) -> GenerateContentResponse:
    candidate: dict[str, Any] = {'content': {'role': 'model', 'parts': parts}}
    if grounding is not None:
        candidate['grounding_metadata'] = grounding
    return GenerateContentResponse.model_validate({'candidates': [candidate]})


async def _drive_stream(
    chunks: list[GenerateContentResponse],
) -> tuple[list[ModelResponseStreamEvent], list[ModelResponsePart]]:
    async def stream() -> AsyncIterator[GenerateContentResponse]:
        for chunk in chunks:
            yield chunk

    streamed = GeminiStreamedResponse(
        model_request_parameters=ModelRequestParameters(),
        _model_name='gemini-3.5-flash',
        _response=_utils.PeekableAsyncStream(stream()),
        _provider_name='google-gla',
        _provider_url='https://generativelanguage.googleapis.com/',
    )
    events = [event async for event in streamed]
    return events, list(streamed.get().parts)


def _file_search_return_start_parts(events: list[ModelResponseStreamEvent]) -> list[NativeToolReturnPart]:
    return [
        e.part
        for e in events
        if isinstance(e, PartStartEvent)
        and isinstance(e.part, NativeToolReturnPart)
        and e.part.tool_name == 'file_search'
    ]


@pytest.mark.anyio
async def test_file_search_grounding_fills_empty_tool_response_streaming():
    """Streaming: grounding arrives several chunks after the empty `tool_response`, so the part is filled then."""
    events, parts = await _drive_stream(
        [
            _stream_chunk([{'tool_call': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH', 'args': {}}}]),
            _stream_chunk([{'tool_response': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH'}}]),
            _stream_chunk([{'text': 'Paris is the '}]),
            _stream_chunk([{'text': 'capital of France.'}]),
            _stream_chunk([{'text': ''}], grounding=_FILE_SEARCH_GROUNDING_METADATA),
        ]
    )

    # The empty `tool_response` is filled in place, not duplicated by a second grounding-derived part, and
    # the part order matches the non-streaming path: the call and its (filled) return precede the grounded text.
    assert parts == snapshot(
        [
            NativeToolCallPart(
                tool_name='file_search',
                args={},
                tool_call_id='file_search_call',
                provider_name='google-gla',
            ),
            NativeToolReturnPart(
                tool_name='file_search',
                content=[
                    {
                        'text': 'Paris is the capital of France.',
                        'title': 'paris.txt',
                        'custom_metadata': [{'key': 'source_url', 'string_value': 'https://example.com/paris-facts'}],
                        'file_search_store': 'fileSearchStores/test-store',
                    }
                ],
                tool_call_id='file_search_call',
                timestamp=IsDatetime(),
                provider_name='google-gla',
            ),
            TextPart(content='Paris is the capital of France.'),
        ]
    )

    # Exactly one `PartStartEvent` for the file_search return: the empty placeholder's event is deferred until
    # it is filled, so streaming consumers see a single populated result rather than an empty one then a duplicate.
    starts = _file_search_return_start_parts(events)
    assert len(starts) == 1
    assert starts[0].content is not None


@pytest.mark.anyio
async def test_file_search_multiple_calls_all_filled_streaming():
    """Every reserved file_search return is filled from the aggregate grounding, matching the non-streaming path."""
    events, parts = await _drive_stream(
        [
            _stream_chunk([{'tool_call': {'id': 'call_1', 'tool_type': 'FILE_SEARCH', 'args': {}}}]),
            _stream_chunk([{'tool_response': {'id': 'call_1', 'tool_type': 'FILE_SEARCH'}}]),
            _stream_chunk([{'tool_call': {'id': 'call_2', 'tool_type': 'FILE_SEARCH', 'args': {}}}]),
            _stream_chunk([{'tool_response': {'id': 'call_2', 'tool_type': 'FILE_SEARCH'}}]),
            _stream_chunk([{'text': 'Paris.'}], grounding=_FILE_SEARCH_GROUNDING_METADATA),
        ]
    )

    returns = [p for p in parts if isinstance(p, NativeToolReturnPart) and p.tool_name == 'file_search']
    assert [(r.tool_call_id, r.content) for r in returns] == snapshot(
        [
            (
                'call_1',
                [
                    {
                        'text': 'Paris is the capital of France.',
                        'title': 'paris.txt',
                        'custom_metadata': [{'key': 'source_url', 'string_value': 'https://example.com/paris-facts'}],
                        'file_search_store': 'fileSearchStores/test-store',
                    }
                ],
            ),
            (
                'call_2',
                [
                    {
                        'text': 'Paris is the capital of France.',
                        'title': 'paris.txt',
                        'custom_metadata': [{'key': 'source_url', 'string_value': 'https://example.com/paris-facts'}],
                        'file_search_store': 'fileSearchStores/test-store',
                    }
                ],
            ),
        ]
    )
    assert len(_file_search_return_start_parts(events)) == 2


@pytest.mark.anyio
async def test_file_search_grounding_absent_leaves_empty_content_streaming():
    """If grounding never arrives, the reserved return keeps its empty content and its deferred event is
    flushed at the end of the stream, so event consumers still see every part present in the final response."""
    events, parts = await _drive_stream(
        [
            _stream_chunk([{'tool_call': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH', 'args': {}}}]),
            _stream_chunk([{'tool_response': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH'}}]),
            _stream_chunk([{'text': 'Paris is the capital of France.'}]),
        ]
    )

    file_search_returns = [p for p in parts if isinstance(p, NativeToolReturnPart) and p.tool_name == 'file_search']
    assert len(file_search_returns) == 1
    assert file_search_returns[0].content is None
    # The deferred `PartStartEvent` is flushed at the end of the stream, empty, exactly once.
    starts = _file_search_return_start_parts(events)
    assert len(starts) == 1
    assert starts[0].content is None
    assert isinstance(events[-1], PartStartEvent) and events[-1].part is starts[0]
