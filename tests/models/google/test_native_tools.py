"""Tests for Google native tool parts on both sides of the API boundary.

Request side, `_content_model_response` — the message-history echo path that
round-trips `NativeToolCallPart` / `NativeToolReturnPart` between the API and
the application:

- pre-Gemini-3 models drop server-side native parts (the API would reject them);
- `pyd_ai_`-synthesized `tool_call_id`s are dropped on every model;
- `CodeExecutionTool` uses `executable_code` / `code_execution_result` parts and is
  preserved regardless of the tool-combination capability.

Response side, `_process_response_from_parts` / `GeminiStreamedResponse` — web-search
source URLs live in `grounding_metadata.grounding_chunks`, not in the explicit
web-search `tool_response` (which holds only a `search_suggestions` widget on
Gemini 3+), so grounding chunks populate the `NativeToolReturnPart` content in both
the streamed and non-streamed paths.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai._utils import PeekableAsyncStream
from pydantic_ai.messages import (
    ModelResponse,
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
    WebSearchTool,
)
from pydantic_ai.usage import RequestUsage

from ...conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from google.genai.types import (
        Candidate,
        Content,
        FinishReason as GoogleFinishReason,
        GenerateContentResponse,
        Part,
        ToolType,
    )

    from pydantic_ai.models.google import (
        GeminiStreamedResponse,
        _content_model_response,  # pyright: ignore[reportPrivateUsage]
        _process_response_from_parts,  # pyright: ignore[reportPrivateUsage]
    )

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
        if call_part.tool_name != WebSearchTool.kind:
            continue

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
