"""Tests for Google native tool parts on both sides of the API boundary.

Request side, `_content_model_response` — the message-history echo path that
round-trips `NativeToolCallPart` / `NativeToolReturnPart` between the API and
the application:

- pre-Gemini-3 models drop server-side native parts (the API would reject them);
- `pyd_ai_`-synthesized `tool_call_id`s are dropped on every model;
- `CodeExecutionTool` uses `executable_code` / `code_execution_result` parts and is
  preserved regardless of the tool-combination capability.

Response side, `_process_response_from_parts` / `GeminiStreamedResponse` — on Gemini 3+
the explicit native `tool_response` doesn't carry the grounded results: file_search
leaves it empty and web_search holds only a `search_suggestions` widget, with the real
contexts/source URLs delivered in `grounding_metadata.grounding_chunks`. These tests pin
that grounding chunks populate the `NativeToolReturnPart` content in both the streamed
and non-streamed paths, including the streaming cross-chunk deferral.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import models
from pydantic_ai._run_context import RunContext
from pydantic_ai._utils import PeekableAsyncStream
from pydantic_ai.capabilities import CombinedCapability
from pydantic_ai.messages import (
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import (
    CodeExecutionTool,
    FileSearchTool,
    WebFetchTool,
    WebSearchTool,
)
from pydantic_ai.usage import RequestUsage, RunUsage

from ...conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from google.genai.types import (
        Candidate,
        Content,
        FinishReason as GoogleFinishReason,
        GenerateContentResponse,
        GroundingMetadata,
        Part,
        ToolType,
    )

    from pydantic_ai._output import TextOutputProcessor, TextOutputSchema
    from pydantic_ai.models.google import (
        GeminiStreamedResponse,
        _content_model_response,  # pyright: ignore[reportPrivateUsage]
        _process_response_from_parts,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.result import AgentStream

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
]


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


def test_content_model_response_echoes_web_search_raw_tool_response():
    """When the web-search return part carries the raw API response in `provider_details`
    (because its `content` was replaced with grounding sources), the echo sends the raw
    response back to the API, not the source list.
    """
    response = ModelResponse(
        parts=[
            NativeToolCallPart(
                tool_name=WebSearchTool.kind,
                provider_name='google-gla',
                tool_call_id='web_search_call',
                args={'queries': ['foo']},
            ),
            NativeToolReturnPart(
                tool_name=WebSearchTool.kind,
                provider_name='google-gla',
                tool_call_id='web_search_call',
                content=[{'domain': None, 'title': 'Pydantic AI', 'uri': 'https://ai.pydantic.dev/'}],
                provider_details={'raw_tool_response': {'search_suggestions': '<style>chip</style>'}},
            ),
            TextPart(content='hello'),
        ],
        provider_name='google-gla',
    )

    assert _content_model_response(response, frozenset({'google-gla'}), supports_tool_combination=True) == snapshot(
        {
            'role': 'model',
            'parts': [
                {
                    'tool_call': {
                        'id': 'web_search_call',
                        'tool_type': ToolType.GOOGLE_SEARCH_WEB,
                        'args': {'queries': ['foo']},
                    }
                },
                {
                    'tool_response': {
                        'id': 'web_search_call',
                        'tool_type': ToolType.GOOGLE_SEARCH_WEB,
                        'response': {'search_suggestions': '<style>chip</style>'},
                    }
                },
                {'text': 'hello'},
            ],
        }
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


def _generate_stream_response(
    response_id: str,
    *,
    parts: list[dict[str, Any]] | None = None,
    grounding_metadata: dict[str, Any] | None = None,
    finish_reason: GoogleFinishReason | None = None,
) -> GenerateContentResponse:
    candidate: dict[str, Any] = {'content': {'role': 'model', 'parts': parts or []}}
    if grounding_metadata is not None:
        candidate['groundingMetadata'] = grounding_metadata
    if finish_reason is not None:
        candidate['finishReason'] = finish_reason

    return GenerateContentResponse.model_validate(
        {
            'responseId': response_id,
            'modelVersion': 'gemini-test',
            'usageMetadata': {
                'promptTokenCount': 0,
                'candidatesTokenCount': 0,
            },
            'candidates': [candidate],
        }
    )


def _gemini_streamed_response_from_chunks(chunks: list[GenerateContentResponse]) -> GeminiStreamedResponse:
    async def response_iterator() -> AsyncIterator[GenerateContentResponse]:
        for chunk in chunks:
            yield chunk

    return GeminiStreamedResponse(
        model_request_parameters=ModelRequestParameters(native_tools=[WebSearchTool()]),
        _model_name='gemini-test',
        _response=cast(Any, PeekableAsyncStream(response_iterator())),
        _timestamp=IsDatetime(),
        _provider_name='google',
        _provider_url='https://generativelanguage.googleapis.com/',
    )


_EXPLICIT_WEB_SEARCH_PARTS: list[dict[str, Any]] = [
    {
        'toolCall': {
            'id': 'web-search-call-1',
            'toolType': 'GOOGLE_SEARCH_WEB',
            'args': {'queries': ['explicit query']},
        }
    },
    {
        'toolResponse': {
            'id': 'web-search-call-1',
            'toolType': 'GOOGLE_SEARCH_WEB',
            'response': {'search_suggestions': '<style>chip</style>'},
        }
    },
]

_GROUNDING_METADATA: dict[str, Any] = {
    'webSearchQueries': ['metadata query'],
    'groundingChunks': [{'web': {'uri': 'https://metadata.example/', 'title': 'Metadata source'}}],
}

_EXPLICIT_WEB_FETCH_PARTS: list[dict[str, Any]] = [
    {
        'toolCall': {
            'id': 'web-fetch-call-1',
            'toolType': 'URL_CONTEXT',
            'args': {'urls': ['https://ai.pydantic.dev/']},
        }
    },
    {
        'toolResponse': {
            'id': 'web-fetch-call-1',
            'toolType': 'URL_CONTEXT',
            'response': {'result': 'fetched'},
        }
    },
]


async def test_gemini_streamed_response_appends_web_search_grounding_metadata_after_text():
    streamed_response = _gemini_streamed_response_from_chunks(
        [
            _generate_stream_response('stream-1', parts=[{'text': 'Pydantic AI '}]),
            _generate_stream_response('stream-2', parts=[{'text': 'supports agents.'}]),
            _generate_stream_response(
                'stream-3',
                grounding_metadata={
                    'webSearchQueries': ['Pydantic AI docs'],
                    'groundingChunks': [
                        {'web': {'uri': 'https://ai.pydantic.dev/', 'title': 'Pydantic AI'}},
                        {
                            'web': {
                                'uri': 'https://docs.pydantic.dev/latest/',
                                'title': 'Pydantic documentation',
                            }
                        },
                    ],
                },
                finish_reason=GoogleFinishReason.STOP,
            ),
        ],
    )

    events = [event async for event in streamed_response]
    response = streamed_response.get()

    assert events[0] == snapshot(PartStartEvent(index=0, part=TextPart(content='Pydantic AI ')))
    assert events[2] == snapshot(PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='supports agents.')))
    assert response.text == snapshot('Pydantic AI supports agents.')
    assert response.parts == snapshot(
        [
            TextPart(content='Pydantic AI supports agents.'),
            NativeToolCallPart(
                tool_name='web_search',
                args={'queries': ['Pydantic AI docs']},
                tool_call_id=IsStr(),
                provider_name='google',
            ),
            NativeToolReturnPart(
                tool_name='web_search',
                content=[
                    {'domain': None, 'title': 'Pydantic AI', 'uri': 'https://ai.pydantic.dev/'},
                    {
                        'domain': None,
                        'title': 'Pydantic documentation',
                        'uri': 'https://docs.pydantic.dev/latest/',
                    },
                ],
                tool_call_id=IsStr(),
                timestamp=IsDatetime(),
                provider_name='google',
            ),
        ]
    )
    assert response.native_tool_calls == snapshot(
        [
            (
                NativeToolCallPart(
                    tool_name='web_search',
                    args={'queries': ['Pydantic AI docs']},
                    tool_call_id=IsStr(),
                    provider_name='google',
                ),
                NativeToolReturnPart(
                    tool_name='web_search',
                    content=[
                        {'domain': None, 'title': 'Pydantic AI', 'uri': 'https://ai.pydantic.dev/'},
                        {
                            'domain': None,
                            'title': 'Pydantic documentation',
                            'uri': 'https://docs.pydantic.dev/latest/',
                        },
                    ],
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='google',
                ),
            )
        ]
    )


def _text_output_agent_stream(streamed_response: GeminiStreamedResponse) -> AgentStream[None, str]:
    return AgentStream(
        _raw_stream_response=streamed_response,
        _output_schema=TextOutputSchema[str](
            text_processor=TextOutputProcessor(),
            allows_deferred_tools=False,
            allows_image=False,
            allows_none=False,
        ),
        _model_request_parameters=ModelRequestParameters(),
        _output_validators=[],
        _run_ctx=RunContext(deps=None, model=models.infer_model('test'), usage=RunUsage()),
        _usage_limits=None,
        _tool_manager=cast(Any, None),
        _root_capability=CombinedCapability([]),
    )


async def test_gemini_streamed_response_web_search_grounding_metadata_does_not_trail_text_output():
    """`stream_text` must not append a dangling separator when the web-search pair trails all text."""
    streamed_response = _gemini_streamed_response_from_chunks(
        [
            _generate_stream_response('stream-1', parts=[{'text': 'Pydantic AI '}]),
            _generate_stream_response('stream-2', parts=[{'text': 'supports agents.'}]),
            _generate_stream_response(
                'stream-3',
                grounding_metadata={
                    'webSearchQueries': ['Pydantic AI docs'],
                    'groundingChunks': [{'web': {'uri': 'https://ai.pydantic.dev/', 'title': 'Pydantic AI'}}],
                },
                finish_reason=GoogleFinishReason.STOP,
            ),
        ],
    )

    stream = _text_output_agent_stream(streamed_response)
    deltas = [delta async for delta in stream.stream_text(delta=True, debounce_by=None)]

    assert deltas == snapshot(['Pydantic AI ', 'supports agents.'])
    assert ''.join(deltas) == stream.response.text


async def test_gemini_streamed_response_web_search_sources_are_extractable():
    streamed_response = _gemini_streamed_response_from_chunks(
        [
            _generate_stream_response('stream-1', parts=[{'text': 'Use the docs.'}]),
            _generate_stream_response(
                'stream-2',
                grounding_metadata={
                    'webSearchQueries': ['Pydantic AI source extraction'],
                    'groundingChunks': [
                        {'web': {'uri': 'https://ai.pydantic.dev/native-tools/', 'title': 'Native tools'}}
                    ],
                },
                finish_reason=GoogleFinishReason.STOP,
            ),
        ],
    )

    async for _ in streamed_response:
        pass

    sources: list[tuple[str, str]] = []
    for call_part, return_part in streamed_response.get().native_tool_calls:
        assert call_part.tool_name == WebSearchTool.kind

        content = return_part.content
        assert isinstance(content, list)
        for source in cast('list[dict[str, object]]', content):
            uri = source['uri']
            title = source['title']
            assert isinstance(uri, str)
            assert isinstance(title, str)
            sources.append((uri, title))

    assert sources == snapshot([('https://ai.pydantic.dev/native-tools/', 'Native tools')])


@pytest.mark.parametrize(
    'grounding_with_explicit_parts',
    [
        pytest.param(True, id='grounding-metadata-with-explicit-parts'),
        pytest.param(False, id='grounding-metadata-in-later-chunk'),
    ],
)
async def test_gemini_streamed_response_uses_grounding_sources_with_explicit_web_search_parts(
    grounding_with_explicit_parts: bool,
):
    if grounding_with_explicit_parts:
        chunks = [
            _generate_stream_response('stream-1', parts=[{'text': 'Explicit source.'}]),
            _generate_stream_response(
                'stream-2',
                parts=_EXPLICIT_WEB_SEARCH_PARTS,
                grounding_metadata=_GROUNDING_METADATA,
                finish_reason=GoogleFinishReason.STOP,
            ),
        ]
    else:
        chunks = [
            _generate_stream_response('stream-1', parts=[{'text': 'Explicit source.'}]),
            _generate_stream_response('stream-2', parts=_EXPLICIT_WEB_SEARCH_PARTS),
            _generate_stream_response(
                'stream-3',
                grounding_metadata=_GROUNDING_METADATA,
                finish_reason=GoogleFinishReason.STOP,
            ),
        ]

    streamed_response = _gemini_streamed_response_from_chunks(chunks)

    async for _ in streamed_response:
        pass

    response = streamed_response.get()
    assert response.parts == snapshot(
        [
            TextPart(content='Explicit source.'),
            NativeToolCallPart(
                tool_name='web_search',
                args={'queries': ['explicit query']},
                tool_call_id='web-search-call-1',
                provider_name='google',
            ),
            NativeToolReturnPart(
                tool_name='web_search',
                content=[{'domain': None, 'title': 'Metadata source', 'uri': 'https://metadata.example/'}],
                tool_call_id='web-search-call-1',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_details={'raw_tool_response': {'search_suggestions': '<style>chip</style>'}},
            ),
        ]
    )
    assert response.native_tool_calls == snapshot(
        [
            (
                NativeToolCallPart(
                    tool_name='web_search',
                    args={'queries': ['explicit query']},
                    tool_call_id='web-search-call-1',
                    provider_name='google',
                ),
                NativeToolReturnPart(
                    tool_name='web_search',
                    content=[{'domain': None, 'title': 'Metadata source', 'uri': 'https://metadata.example/'}],
                    tool_call_id='web-search-call-1',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_details={'raw_tool_response': {'search_suggestions': '<style>chip</style>'}},
                ),
            )
        ]
    )


async def test_gemini_response_uses_grounding_sources_with_explicit_web_search_parts():
    raw_response = _generate_stream_response(
        'response-1',
        parts=_EXPLICIT_WEB_SEARCH_PARTS,
        grounding_metadata=_GROUNDING_METADATA,
        finish_reason=GoogleFinishReason.STOP,
    )
    candidates = cast('list[Candidate]', raw_response.candidates)
    candidate = candidates[0]
    content = cast(Content, candidate.content)
    parts = cast('list[Part]', content.parts)

    response = _process_response_from_parts(
        parts,
        cast(Any, candidate.grounding_metadata),
        'gemini-3-flash-preview',
        'google',
        'https://generativelanguage.googleapis.com/',
        RequestUsage(),
        raw_response.response_id,
    )

    assert response.parts == snapshot(
        [
            NativeToolCallPart(
                tool_name='web_search',
                args={'queries': ['explicit query']},
                tool_call_id='web-search-call-1',
                provider_name='google',
            ),
            NativeToolReturnPart(
                tool_name='web_search',
                content=[{'domain': None, 'title': 'Metadata source', 'uri': 'https://metadata.example/'}],
                tool_call_id='web-search-call-1',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_details={'raw_tool_response': {'search_suggestions': '<style>chip</style>'}},
            ),
        ]
    )


async def test_gemini_streamed_response_with_explicit_web_fetch_parts():
    """Non-web-search native tool parts stream through without the web-search-specific
    vendor-id and grounding-source handling.
    """
    streamed_response = _gemini_streamed_response_from_chunks(
        [
            _generate_stream_response('stream-1', parts=[{'text': 'Fetched.'}]),
            _generate_stream_response(
                'stream-2',
                parts=_EXPLICIT_WEB_FETCH_PARTS,
                finish_reason=GoogleFinishReason.STOP,
            ),
        ],
    )

    async for _ in streamed_response:
        pass

    response = streamed_response.get()
    assert response.parts == snapshot(
        [
            TextPart(content='Fetched.'),
            NativeToolCallPart(
                tool_name=WebFetchTool.kind,
                args={'urls': ['https://ai.pydantic.dev/']},
                tool_call_id='web-fetch-call-1',
                provider_name='google',
            ),
            NativeToolReturnPart(
                tool_name=WebFetchTool.kind,
                content={'result': 'fetched'},
                tool_call_id='web-fetch-call-1',
                timestamp=IsDatetime(),
                provider_name='google',
            ),
        ]
    )


async def test_gemini_response_with_explicit_web_fetch_parts():
    raw_response = _generate_stream_response(
        'response-1',
        parts=_EXPLICIT_WEB_FETCH_PARTS,
        finish_reason=GoogleFinishReason.STOP,
    )
    candidates = cast('list[Candidate]', raw_response.candidates)
    candidate = candidates[0]
    content = cast(Content, candidate.content)
    parts = cast('list[Part]', content.parts)

    response = _process_response_from_parts(
        parts,
        cast(Any, candidate.grounding_metadata),
        'gemini-3-flash-preview',
        'google',
        'https://generativelanguage.googleapis.com/',
        RequestUsage(),
        raw_response.response_id,
    )

    assert response.parts == snapshot(
        [
            NativeToolCallPart(
                tool_name=WebFetchTool.kind,
                args={'urls': ['https://ai.pydantic.dev/']},
                tool_call_id='web-fetch-call-1',
                provider_name='google',
            ),
            NativeToolReturnPart(
                tool_name=WebFetchTool.kind,
                content={'result': 'fetched'},
                tool_call_id='web-fetch-call-1',
                timestamp=IsDatetime(),
                provider_name='google',
            ),
        ]
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
    chunks: list[GenerateContentResponse],
) -> tuple[list[ModelResponseStreamEvent], list[ModelResponsePart]]:
    async def stream() -> AsyncIterator[GenerateContentResponse]:
        for chunk in chunks:
            yield chunk

    streamed = GeminiStreamedResponse(
        model_request_parameters=ModelRequestParameters(),
        _model_name='gemini-3.5-flash',
        _response=PeekableAsyncStream(stream()),
        _provider_name='google-gla',
        _provider_url='https://generativelanguage.googleapis.com/',
    )
    events = [event async for event in streamed]
    return events, list(streamed.get().parts)


def _file_search_returns(parts: list[ModelResponsePart]) -> list[NativeToolReturnPart]:
    return [p for p in parts if isinstance(p, NativeToolReturnPart) and p.tool_name == 'file_search']


def _file_search_return_start_parts(events: list[ModelResponseStreamEvent]) -> list[NativeToolReturnPart]:
    return _file_search_returns([e.part for e in events if isinstance(e, PartStartEvent)])


@pytest.mark.anyio
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


@pytest.mark.anyio
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

    returns = _file_search_returns(parts)
    assert len(returns) == 1 and returns[0].content is None
    # The reserved return's deferred `PartStartEvent` is still flushed (empty, exactly once), so event
    # consumers see every part present in the final response.
    starts = _file_search_return_start_parts(events)
    assert len(starts) == 1 and starts[0].content is None
