"""Tests for Google native tool part handling.

Two related areas:

- The message-history echo path (`_content_model_response`) that round-trips
  `NativeToolCallPart` / `NativeToolReturnPart` between the API and the application:
    - pre-Gemini-3 models drop server-side native parts (the API would reject them);
    - `pyd_ai_`-synthesized `tool_call_id`s are dropped on every model;
    - `CodeExecutionTool` uses `executable_code` / `code_execution_result` parts and is
      preserved regardless of the tool-combination capability.
- Response assembly (`_process_response_from_parts`) and streaming
  (`GeminiStreamedResponse`) filling an empty Gemini 3+ `file_search` `tool_response`
  from `grounding_metadata`, including the streaming cross-chunk deferral.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Sequence
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

from ...conftest import try_import

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


# On Gemini 3+ File Search runs server-side: the API returns explicit `tool_call`/`tool_response` parts but
# leaves the response empty, delivering the retrieved contexts (incl. each doc's `custom_metadata`, e.g.
# `source_url`) in `grounding_metadata`. These pin that the empty `NativeToolReturnPart` is filled from it.
# Unit, not VCR: the cassette matcher is body-insensitive, and the streaming cross-chunk assembly is asserted
# at the event level, which VCR can't reach.

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


def _process_response(parts: list[dict[str, Any]], *, grounding: dict[str, Any]) -> ModelResponse:
    return _process_response_from_parts(
        parts=[Part.model_validate(p) for p in parts],
        grounding_metadata=GroundingMetadata.model_validate(grounding),
        model_name='gemini-3.5-flash',
        provider_name='google-gla',
        provider_url='https://generativelanguage.googleapis.com/',
        usage=RequestUsage(),
        provider_response_id='response-id',
    )


def test_file_search_grounding_fills_empty_tool_response():
    """The empty file_search `tool_response` is filled from `grounding_metadata`, incl. each doc's source_url."""
    response = _process_response(
        [
            {'tool_call': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH', 'args': {}}},
            {'tool_response': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH'}},
        ],
        grounding=_FILE_SEARCH_GROUNDING_METADATA,
    )

    _, file_search_return = response.parts
    assert isinstance(file_search_return, NativeToolReturnPart)
    assert file_search_return.content == snapshot(
        [
            {
                'text': 'Paris is the capital of France.',
                'title': 'paris.txt',
                'custom_metadata': [{'key': 'source_url', 'string_value': 'https://example.com/paris-facts'}],
                'file_search_store': 'fileSearchStores/test-store',
            }
        ]
    )


def test_file_search_populated_tool_response_not_overwritten():
    """A file_search `tool_response` that already carries content is kept as-is, not clobbered by grounding."""
    response = _process_response(
        [
            {'tool_call': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH', 'args': {}}},
            {'tool_response': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH', 'response': {'kept': 'value'}}},
        ],
        grounding=_FILE_SEARCH_GROUNDING_METADATA,
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
    chunks: list[GenerateContentResponse], model_request_parameters: ModelRequestParameters | None = None
) -> tuple[list[ModelResponseStreamEvent], list[ModelResponsePart]]:
    async def stream() -> AsyncIterator[GenerateContentResponse]:
        for chunk in chunks:
            yield chunk

    streamed = GeminiStreamedResponse(
        model_request_parameters=model_request_parameters or ModelRequestParameters(),
        _model_name='gemini-3.5-flash',
        _response=_utils.PeekableAsyncStream(stream()),
        _provider_name='google-gla',
        _model_id_namespace='google',
        _provider_url='https://generativelanguage.googleapis.com/',
    )
    events = [event async for event in streamed]
    return events, list(streamed.get().parts)


def _file_search_returns(parts: list[ModelResponsePart]) -> list[NativeToolReturnPart]:
    return [p for p in parts if isinstance(p, NativeToolReturnPart) and p.tool_name == 'file_search']


def _file_search_return_start_parts(events: list[ModelResponseStreamEvent]) -> list[NativeToolReturnPart]:
    return _file_search_returns([e.part for e in events if isinstance(e, PartStartEvent)])


async def test_file_search_grounding_fills_empty_tool_response_streaming():
    """Streaming: grounding arrives several chunks after the empty `tool_response`, which is then filled in
    place — a single `PartStartEvent` (no empty-then-filled duplicate), ordered ahead of the grounded text.

    Content shape is pinned by the non-streaming test and end-to-end by the VCR test; here we only assert the
    streaming-specific mechanics.
    """
    events, parts = await _drive_stream(
        [
            _stream_chunk([{'tool_call': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH', 'args': {}}}]),
            _stream_chunk([{'tool_response': {'id': 'file_search_call', 'tool_type': 'FILE_SEARCH'}}]),
            _stream_chunk([{'text': 'Paris is the '}]),
            _stream_chunk([{'text': 'capital of France.'}]),
            _stream_chunk([{'text': ''}], grounding=_FILE_SEARCH_GROUNDING_METADATA),
        ]
    )

    call, file_search_return, text = parts
    assert isinstance(call, NativeToolCallPart)
    assert isinstance(file_search_return, NativeToolReturnPart) and file_search_return.content is not None
    assert isinstance(text, TextPart)
    assert len(_file_search_return_start_parts(events)) == 1


async def test_file_search_multiple_calls_all_filled_streaming():
    """Every reserved file_search return is filled from the aggregate grounding, not just the last."""
    events, parts = await _drive_stream(
        [
            _stream_chunk([{'tool_call': {'id': 'call_1', 'tool_type': 'FILE_SEARCH', 'args': {}}}]),
            _stream_chunk([{'tool_response': {'id': 'call_1', 'tool_type': 'FILE_SEARCH'}}]),
            _stream_chunk([{'tool_call': {'id': 'call_2', 'tool_type': 'FILE_SEARCH', 'args': {}}}]),
            _stream_chunk([{'tool_response': {'id': 'call_2', 'tool_type': 'FILE_SEARCH'}}]),
            _stream_chunk([{'text': 'Paris.'}], grounding=_FILE_SEARCH_GROUNDING_METADATA),
        ]
    )

    returns = _file_search_returns(parts)
    assert [r.tool_call_id for r in returns] == ['call_1', 'call_2']
    assert all(r.content is not None for r in returns)
    assert len(_file_search_return_start_parts(events)) == 2


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

    returns = _file_search_returns(parts)
    assert len(returns) == 1 and returns[0].content is None
    # The reserved return's deferred `PartStartEvent` is still flushed (empty, exactly once), so event
    # consumers see every part present in the final response.
    starts = _file_search_return_start_parts(events)
    assert len(starts) == 1 and starts[0].content is None


# Without server-side tool invocation parts (Gemini 2.5, and Google Cloud on every model), the native tool parts
# are rebuilt from the grounding metadata, which Gemini only sends with the final chunk, after the text it grounds.


async def test_file_search_grounding_without_executable_code_comes_after_text():
    """Streamed and complete responses both put the file search rebuilt from grounding metadata after the text."""
    chunks = [_stream_chunk([{'text': 'Paris is the '}]), _stream_chunk([{'text': 'capital.'}])]
    chunks.append(_stream_chunk([{'text': ''}], grounding=_FILE_SEARCH_GROUNDING_METADATA))
    _, streamed_parts = await _drive_stream(chunks)
    complete = _process_response([{'text': 'Paris is the capital.'}], grounding=_FILE_SEARCH_GROUNDING_METADATA)

    for parts in (streamed_parts, complete.parts):
        text, call, file_search_return = parts
        assert isinstance(text, TextPart)
        assert isinstance(call, NativeToolCallPart) and call.tool_name == 'file_search' and call.args == {}
        assert isinstance(file_search_return, NativeToolReturnPart)
        assert file_search_return.tool_call_id == call.tool_call_id and file_search_return.content


def test_file_search_executable_code_without_retrieved_contexts_has_no_return():
    response = _process_response_from_parts(
        parts=[
            Part.model_validate({'executable_code': {'code': 'print(file_search.query(query="capital"))'}}),
            Part.model_validate({'text': 'I found nothing.'}),
        ],
        grounding_metadata=None,
        model_name='gemini-2.5-pro',
        provider_name='google-gla',
        provider_url='https://generativelanguage.googleapis.com/',
        usage=RequestUsage(),
        provider_response_id='response-id',
        file_search_enabled=True,
    )

    call, text = response.parts
    assert isinstance(call, NativeToolCallPart) and call.tool_name == 'file_search'
    assert call.args == {'query': 'capital'}
    assert isinstance(text, TextPart)


async def test_file_search_executable_code_calls_all_get_returns():
    """Each `file_search.query()` call gets the retrieved contexts, which the grounding metadata doesn't attribute."""
    parts = [
        {'executable_code': {'code': 'print(file_search.query(query="capital"))'}},
        {'executable_code': {'code': 'print(file_search.query(query="landmarks"))'}},
        {'text': 'Paris.'},
    ]
    _, streamed_parts = await _drive_stream(
        [_stream_chunk(parts[:2]), _stream_chunk(parts[2:], grounding=_FILE_SEARCH_GROUNDING_METADATA)],
        ModelRequestParameters(native_tools=[FileSearchTool(file_store_ids=['fileSearchStores/test-store'])]),
    )
    complete = _process_response_from_parts(
        parts=[Part.model_validate(p) for p in parts],
        grounding_metadata=GroundingMetadata.model_validate(_FILE_SEARCH_GROUNDING_METADATA),
        model_name='gemini-2.5-pro',
        provider_name='google-gla',
        provider_url='https://generativelanguage.googleapis.com/',
        usage=RequestUsage(),
        provider_response_id='response-id',
        file_search_enabled=True,
    )

    for response_parts in (streamed_parts, complete.parts):
        first_call, second_call, text, first_return, second_return = response_parts
        assert isinstance(first_call, NativeToolCallPart) and isinstance(second_call, NativeToolCallPart)
        assert isinstance(text, TextPart)
        assert isinstance(first_return, NativeToolReturnPart) and isinstance(second_return, NativeToolReturnPart)
        assert [first_return.tool_call_id, second_return.tool_call_id] == [
            first_call.tool_call_id,
            second_call.tool_call_id,
        ]
        assert first_return.content == second_return.content


# On Gemini 3, a web search `tool_response` only carries `search_suggestions`: the sources arrive in the response's
# grounding metadata, which doesn't say which of several searches found them.

_WEB_SEARCH_GROUNDING_METADATA: dict[str, Any] = {
    'web_search_queries': ['2026 World Cup winner', 'population of Iceland'],
    'grounding_chunks': [
        {'web': {'uri': 'https://example.com/world-cup', 'title': 'fifa.com'}},
        {'web': {'uri': 'https://example.com/iceland', 'title': 'worldometers.info'}},
    ],
}

_WEB_SEARCH_PARTS: list[dict[str, Any]] = [
    {'tool_call': {'id': 'call_1', 'tool_type': 'GOOGLE_SEARCH_WEB', 'args': {'queries': ['2026 World Cup winner']}}},
    {'tool_response': {'id': 'call_1', 'tool_type': 'GOOGLE_SEARCH_WEB', 'response': {'search_suggestions': '<a>'}}},
    {'tool_call': {'id': 'call_2', 'tool_type': 'GOOGLE_SEARCH_WEB', 'args': {'queries': ['population of Iceland']}}},
    {'tool_response': {'id': 'call_2', 'tool_type': 'GOOGLE_SEARCH_WEB', 'response': {'search_suggestions': '<b>'}}},
    {'tool_call': {'id': 'call_3', 'tool_type': 'URL_CONTEXT', 'args': {'urls': ['https://example.com/iceland']}}},
    {'tool_response': {'id': 'call_3', 'tool_type': 'URL_CONTEXT', 'response': {'status': 'ok'}}},
    {'text': 'Spain won, and Iceland has 404,000 people.'},
]


def _web_search_returns(parts: Sequence[ModelResponsePart]) -> list[NativeToolReturnPart]:
    return [p for p in parts if isinstance(p, NativeToolReturnPart) and p.tool_name == 'web_search']


async def test_web_search_sources_go_on_the_last_web_search_return():
    sources = [
        {'domain': None, 'title': 'fifa.com', 'uri': 'https://example.com/world-cup'},
        {'domain': None, 'title': 'worldometers.info', 'uri': 'https://example.com/iceland'},
    ]
    events, streamed_parts = await _drive_stream(
        [_stream_chunk([part]) for part in _WEB_SEARCH_PARTS[:-1]]
        + [_stream_chunk([_WEB_SEARCH_PARTS[-1]], grounding=_WEB_SEARCH_GROUNDING_METADATA)]
    )
    complete = _process_response(_WEB_SEARCH_PARTS, grounding=_WEB_SEARCH_GROUNDING_METADATA)

    for parts in (streamed_parts, complete.parts):
        assert [part.content for part in _web_search_returns(parts)] == [
            {'search_suggestions': '<a>'},
            {'search_suggestions': '<b>', 'sources': sources},
        ]
    # Each return starts once, complete: the first when the next search makes it not the last, the last at the end.
    assert [
        part.content for part in _web_search_returns([e.part for e in events if isinstance(e, PartStartEvent)])
    ] == [
        {'search_suggestions': '<a>'},
        {'search_suggestions': '<b>', 'sources': sources},
    ]


def test_content_model_response_leaves_web_search_sources_out_of_the_tool_response():
    """The sources came from the grounding metadata, not Gemini's `tool_response`, so they aren't sent back."""
    complete = _process_response(_WEB_SEARCH_PARTS, grounding=_WEB_SEARCH_GROUNDING_METADATA)

    assert _content_model_response(complete, frozenset({'google-gla'}), supports_tool_combination=True) == snapshot(
        {
            'role': 'model',
            'parts': [
                {
                    'tool_call': {
                        'id': 'call_1',
                        'tool_type': ToolType.GOOGLE_SEARCH_WEB,
                        'args': {'queries': ['2026 World Cup winner']},
                    }
                },
                {
                    'tool_response': {
                        'id': 'call_1',
                        'tool_type': ToolType.GOOGLE_SEARCH_WEB,
                        'response': {'search_suggestions': '<a>'},
                    }
                },
                {
                    'tool_call': {
                        'id': 'call_2',
                        'tool_type': ToolType.GOOGLE_SEARCH_WEB,
                        'args': {'queries': ['population of Iceland']},
                    }
                },
                {
                    'tool_response': {
                        'id': 'call_2',
                        'tool_type': ToolType.GOOGLE_SEARCH_WEB,
                        'response': {'search_suggestions': '<b>'},
                    }
                },
                {
                    'tool_call': {
                        'id': 'call_3',
                        'tool_type': ToolType.URL_CONTEXT,
                        'args': {'urls': ['https://example.com/iceland']},
                    }
                },
                {'tool_response': {'id': 'call_3', 'tool_type': ToolType.URL_CONTEXT, 'response': {'status': 'ok'}}},
                {'text': 'Spain won, and Iceland has 404,000 people.'},
            ],
        }
    )
