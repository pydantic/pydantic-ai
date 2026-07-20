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

These unit tests pin request and response payload shapes that VCR matching does not
validate, including defensive variants that the live API does not reliably produce.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import _utils, models
from pydantic_ai._run_context import RunContext
from pydantic_ai._utils import PeekableAsyncStream
from pydantic_ai.capabilities import CombinedCapability
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
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
        ToolResponse,
        ToolType,
    )

    from pydantic_ai._output import TextOutputProcessor, TextOutputSchema
    from pydantic_ai.models.google import (
        _TOOL_TYPE_TO_NATIVE_TOOL_NAME,  # pyright: ignore[reportPrivateUsage]
        GeminiStreamedResponse,
        _content_model_response,  # pyright: ignore[reportPrivateUsage]
        _fill_native_tool_return_from_grounding,  # pyright: ignore[reportPrivateUsage]
        _map_tool_response,  # pyright: ignore[reportPrivateUsage]
        _native_tool_return_part_dict,  # pyright: ignore[reportPrivateUsage]
        _process_response_from_parts,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.result import AgentStream

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
]

_TOOL_TYPES = list(_TOOL_TYPE_TO_NATIVE_TOOL_NAME) if imports_successful() else []


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


def test_content_model_response_echoes_none_web_search_raw_tool_response_after_round_trip():
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
                provider_details={'raw_tool_response': None},
            ),
        ],
        provider_name='google-gla',
    )
    [round_tripped] = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json([response]))
    assert isinstance(round_tripped, ModelResponse)

    assert _content_model_response(
        round_tripped, frozenset({'google-gla'}), supports_tool_combination=True
    ) == snapshot(
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
                        'response': None,
                    }
                },
            ],
        }
    )


def test_content_model_response_normalizes_invalid_raw_tool_response():
    response = ModelResponse(
        parts=[
            NativeToolReturnPart(
                tool_name=WebSearchTool.kind,
                provider_name='google-gla',
                tool_call_id='web_search_call',
                content=[],
                provider_details={'raw_tool_response': 'invalid history value'},
            ),
        ],
        provider_name='google-gla',
    )

    mapped = _content_model_response(response, frozenset({'google-gla'}), supports_tool_combination=True)
    assert mapped is not None
    parts = mapped.get('parts')
    assert parts is not None
    assert parts == snapshot(
        [
            {
                'tool_response': {
                    'id': 'web_search_call',
                    'tool_type': ToolType.GOOGLE_SEARCH_WEB,
                    'response': {'result': 'invalid history value'},
                }
            }
        ]
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


# On Gemini 3+ Web Search runs server-side: the API returns explicit `tool_call`/`tool_response` parts, but
# the response holds only a `search_suggestions` HTML widget — the source URLs arrive in `grounding_metadata`
# (in streaming, chunks later). Pre-Gemini-3 models send no explicit parts at all, so the pair is
# reconstructed from the metadata after the text. These pin both paths at the part/event level.
# Unit, not VCR: the cassette matcher is body-insensitive, and the streaming cross-chunk assembly is
# asserted at the event level, which VCR can't reach (the structured-output VCR tests pin the explicit-parts
# paths end to end on a real Gemini 3 recording).


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


async def test_gemini_streamed_response_grounding_metadata_keeps_partial_text_output():
    streamed_response = _gemini_streamed_response_from_chunks(
        [
            _generate_stream_response('stream-1', parts=[{'text': 'Answer.'}]),
            _generate_stream_response(
                'stream-2', grounding_metadata=_GROUNDING_METADATA, finish_reason=GoogleFinishReason.STOP
            ),
        ]
    )

    outputs = [output async for output in _text_output_agent_stream(streamed_response).stream_output(debounce_by=None)]

    assert outputs
    assert set(outputs) == {'Answer.'}


async def test_gemini_streamed_response_web_search_grounding_metadata_reconstructed_once():
    """Grounding metadata repeated on a trailing chunk must not emit a duplicate web-search pair."""
    streamed_response = _gemini_streamed_response_from_chunks(
        [
            _generate_stream_response('stream-1', parts=[{'text': 'Answer.'}]),
            _generate_stream_response('stream-2', grounding_metadata=_GROUNDING_METADATA),
            _generate_stream_response(
                'stream-3',
                grounding_metadata=_GROUNDING_METADATA,
                finish_reason=GoogleFinishReason.STOP,
            ),
        ],
    )

    async for _ in streamed_response:
        pass

    response = streamed_response.get()
    assert [type(part).__name__ for part in response.parts] == snapshot(
        ['TextPart', 'NativeToolCallPart', 'NativeToolReturnPart']
    )


@pytest.mark.parametrize('explicit_parts', [False, True])
async def test_gemini_streamed_response_accumulates_web_search_grounding_chunks(explicit_parts: bool):
    chunks = [_generate_stream_response('stream-1', parts=[{'text': 'Answer.'}])]
    chunks.extend(
        [
            _generate_stream_response(
                'stream-2',
                parts=_EXPLICIT_WEB_SEARCH_PARTS if explicit_parts else None,
                grounding_metadata={
                    'webSearchQueries': ['metadata query'],
                    'groundingChunks': [
                        {'web': {'uri': 'https://first.example/', 'title': 'First source'}},
                    ],
                },
            ),
            _generate_stream_response(
                'stream-3',
                grounding_metadata={
                    'groundingChunks': [
                        {'web': {'uri': 'https://second.example/', 'title': 'Second source'}},
                    ],
                },
                finish_reason=GoogleFinishReason.STOP,
            ),
        ]
    )
    streamed_response = _gemini_streamed_response_from_chunks(chunks)

    async for _ in streamed_response:
        pass

    [tool_return] = [
        part
        for part in streamed_response.get().parts
        if isinstance(part, NativeToolReturnPart) and part.tool_name == WebSearchTool.kind
    ]
    assert tool_return.content == snapshot(
        [
            {'domain': None, 'title': 'First source', 'uri': 'https://first.example/'},
            {'domain': None, 'title': 'Second source', 'uri': 'https://second.example/'},
        ]
    )


async def test_gemini_streamed_response_preserves_equal_grounding_chunks():
    grounding_chunk = {'web': {'uri': 'https://same.example/', 'title': 'Same source'}}
    streamed_response = _gemini_streamed_response_from_chunks(
        [
            _generate_stream_response(
                'stream-1',
                grounding_metadata={
                    'webSearchQueries': ['metadata query'],
                    'groundingChunks': [grounding_chunk],
                },
            ),
            _generate_stream_response(
                'stream-2',
                grounding_metadata={'groundingChunks': [grounding_chunk]},
                finish_reason=GoogleFinishReason.STOP,
            ),
        ]
    )

    async for _ in streamed_response:
        pass

    [tool_return] = [
        part
        for part in streamed_response.get().parts
        if isinstance(part, NativeToolReturnPart) and part.tool_name == WebSearchTool.kind
    ]
    assert tool_return.content == snapshot(
        [
            {'domain': None, 'title': 'Same source', 'uri': 'https://same.example/'},
            {'domain': None, 'title': 'Same source', 'uri': 'https://same.example/'},
        ]
    )


async def test_gemini_streamed_response_preserves_explicit_web_search_return_without_grounding():
    streamed_response = _gemini_streamed_response_from_chunks(
        [
            _generate_stream_response('stream-1', parts=[{'text': 'Answer.'}]),
            _generate_stream_response(
                'stream-2', parts=_EXPLICIT_WEB_SEARCH_PARTS, finish_reason=GoogleFinishReason.STOP
            ),
        ]
    )

    events = [event async for event in streamed_response]

    [tool_return] = [
        part
        for part in streamed_response.get().parts
        if isinstance(part, NativeToolReturnPart) and part.tool_name == WebSearchTool.kind
    ]
    assert tool_return.content == {'search_suggestions': '<style>chip</style>'}
    assert (
        len(
            [
                event
                for event in events
                if isinstance(event, PartStartEvent)
                and isinstance(event.part, NativeToolReturnPart)
                and event.part.tool_name == WebSearchTool.kind
            ]
        )
        == 1
    )


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

    events = [event async for event in streamed_response]

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
    # The reserved return emits a single deferred `PartStartEvent`, already grounded — never the
    # ungrounded widget followed by a filled duplicate.
    return_starts = [
        e.part for e in events if isinstance(e, PartStartEvent) and isinstance(e.part, NativeToolReturnPart)
    ]
    assert len(return_starts) == 1
    assert return_starts[0].content == [
        {'domain': None, 'title': 'Metadata source', 'uri': 'https://metadata.example/'}
    ]


def test_fill_web_search_return_preserves_raw_tool_response_once_filled():
    """The fill replaces `content` but never touches the raw response `_map_tool_response` preserved,
    so a repeated fill can't lose it either. Unit, not VCR: pins the private mapping pipeline.
    """
    part = _map_tool_response(
        ToolResponse.model_validate(
            {
                'id': 'web-search-call-1',
                'tool_type': 'GOOGLE_SEARCH_WEB',
                'response': {'search_suggestions': '<style>chip</style>'},
            }
        ),
        'google',
    )
    grounding = GroundingMetadata.model_validate(_GROUNDING_METADATA)

    _fill_native_tool_return_from_grounding(part, grounding)
    _fill_native_tool_return_from_grounding(part, grounding)

    assert part.provider_details == {'raw_tool_response': {'search_suggestions': '<style>chip</style>'}}
    assert part.content == [{'domain': None, 'title': 'Metadata source', 'uri': 'https://metadata.example/'}]


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


async def test_gemini_streamed_response_preserves_trailing_web_search_metadata_marker_after_round_trip():
    streamed_response = _gemini_streamed_response_from_chunks(
        [
            _generate_stream_response('stream-1', parts=[{'text': 'Answer.'}]),
            _generate_stream_response(
                'stream-2', grounding_metadata=_GROUNDING_METADATA, finish_reason=GoogleFinishReason.STOP
            ),
        ]
    )
    async for _ in streamed_response:
        pass

    response = streamed_response.get()
    response.timestamp = _utils.now_utc()
    [round_tripped] = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json([response]))
    assert isinstance(round_tripped, ModelResponse)
    assert _utils.is_trailing_provider_metadata_native_tool_call(round_tripped, 1)


async def test_gemini_streamed_response_with_explicit_web_fetch_parts():
    """Native tool parts that don't await grounding stream through immediately, without
    the reserve-and-fill deferral used for web_search and file_search returns.
    """
    streamed_response = _gemini_streamed_response_from_chunks(
        [
            _generate_stream_response('stream-1', parts=[{'text': 'Fetched.'}]),
            _generate_stream_response('stream-2', parts=_EXPLICIT_WEB_FETCH_PARTS),
            _generate_stream_response(
                'stream-3',
                parts=[{'text': 'After.'}],
                finish_reason=GoogleFinishReason.STOP,
            ),
        ],
    )

    events = [event async for event in streamed_response]

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
            TextPart(content='After.'),
        ]
    )
    # The return's `PartStartEvent` is emitted as it arrives, ahead of the following text — it is
    # not reserved and flushed at end of stream like a grounding-awaiting return would be.
    assert [type(e.part).__name__ for e in events if isinstance(e, PartStartEvent)] == snapshot(
        ['TextPart', 'NativeToolCallPart', 'NativeToolReturnPart', 'TextPart']
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


@pytest.mark.parametrize('stream', [False, True])
async def test_gemini_response_reconstructs_web_search_with_explicit_web_fetch_parts(stream: bool):
    raw_response = _generate_stream_response(
        'response-1',
        parts=_EXPLICIT_WEB_FETCH_PARTS,
        grounding_metadata=_GROUNDING_METADATA,
        finish_reason=GoogleFinishReason.STOP,
    )

    if stream:
        streamed_response = _gemini_streamed_response_from_chunks([raw_response])
        async for _ in streamed_response:
            pass
        response = streamed_response.get()
    else:
        candidates = cast('list[Candidate]', raw_response.candidates)
        candidate = candidates[0]
        content = cast(Content, candidate.content)
        response = _process_response_from_parts(
            cast('list[Part]', content.parts),
            cast(Any, candidate.grounding_metadata),
            'gemini-3-flash-preview',
            'google',
            'https://generativelanguage.googleapis.com/',
            RequestUsage(),
            raw_response.response_id,
        )

    assert [part.tool_name for part in response.parts if isinstance(part, NativeToolCallPart)] == [
        'web_fetch',
        'web_search',
    ]
    [web_search_return] = [
        part
        for part in response.parts
        if isinstance(part, NativeToolReturnPart) and part.tool_name == WebSearchTool.kind
    ]
    assert web_search_return.content == snapshot(
        [{'domain': None, 'title': 'Metadata source', 'uri': 'https://metadata.example/'}]
    )
    web_search_call_index = next(
        index
        for index, part in enumerate(response.parts)
        if isinstance(part, NativeToolCallPart) and part.tool_name == WebSearchTool.kind
    )
    assert _utils.is_trailing_provider_metadata_native_tool_call(response, web_search_call_index)


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


def test_web_search_none_tool_response_round_trips_unchanged():
    response = _process_response(
        [
            {'tool_call': {'id': 'web_search_call', 'tool_type': 'GOOGLE_SEARCH_WEB', 'args': {}}},
            {'tool_response': {'id': 'web_search_call', 'tool_type': 'GOOGLE_SEARCH_WEB'}},
        ],
        grounding={},
    )

    _, web_search_return = response.parts
    assert isinstance(web_search_return, NativeToolReturnPart)
    assert web_search_return.provider_details == {'raw_tool_response': None}
    mapped = _content_model_response(response, frozenset({'google-gla'}), supports_tool_combination=True)
    assert mapped is not None
    assert cast(Any, mapped)['parts'][1]['tool_response']['response'] is None


@pytest.mark.parametrize('tool_type', _TOOL_TYPES)
@pytest.mark.parametrize('response', [None, {'payload': 'from-google'}], ids=['empty', 'populated'])
def test_raw_tool_response_is_preserved_for_web_search_only(tool_type: ToolType, response: dict[str, Any] | None):
    """`raw_tool_response` wins over `content` on the echo, so only web_search may set it.

    web_search is the only tool whose `content` grounding *replaces*, so it alone needs the payload it
    replaced kept for the echo. Every other tool is grounded into `content` itself, so preserving a raw
    response for it echoes that stale value and silently drops the grounded data.

    Parametrized off `_TOOL_TYPE_TO_NATIVE_TOOL_NAME` so a newly mapped tool type must make this choice
    deliberately, and over an empty response because that is the shape that regressed file_search: it
    arrives empty and is filled later, so keying the preserve off emptiness pins the wrong value.
    A VCR test would not pin this private mapping rule or the exact follow-up request payload.
    """
    item = _map_tool_response(
        ToolResponse.model_validate({'id': 'c1', 'tool_type': tool_type, 'response': response}), 'google-gla'
    )

    preserves_raw = 'raw_tool_response' in (item.provider_details or {})
    assert preserves_raw == (item.tool_name == WebSearchTool.kind)


@pytest.mark.parametrize('tool_type', _TOOL_TYPES)
def test_grounded_content_survives_the_echo_for_non_web_search_tools(tool_type: ToolType):
    """Whatever grounding recovers into `content` must reach Gemini on the follow-up request.

    The echo is the only consumer of a filled return, so a tool that recovers content but echoes
    something that no longer carries it has silently lost the data (see the file_search regression,
    where the contexts were filled but `response: None` went out on the wire).
    A VCR test would not pin this private mapping rule or the exact follow-up request payload.
    """
    item = _map_tool_response(ToolResponse.model_validate({'id': 'c1', 'tool_type': tool_type}), 'google-gla')
    if item.tool_name == WebSearchTool.kind:
        pytest.skip('web_search deliberately echoes the raw response its content replaced')

    item.content = ['recovered-from-grounding']
    echoed = _native_tool_return_part_dict(item, frozenset({'google-gla'}), None, supports_tool_combination=True)

    assert echoed is not None
    tool_response = echoed.get('tool_response')
    assert tool_response is not None
    assert tool_response.get('response') == snapshot({'result': ['recovered-from-grounding']})


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
    # Only web_search preserves a raw response: file_search is filled into `content` itself.
    assert file_search_return.provider_details is None
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
    # The contexts recovered from grounding must survive the echo, otherwise a follow-up request
    # drops what #6216 recovered.
    mapped = _content_model_response(response, frozenset({'google-gla'}), supports_tool_combination=True)
    assert mapped is not None
    assert cast(Any, mapped)['parts'][1]['tool_response']['response'] == snapshot(
        {
            'result': [
                {
                    'text': 'Paris is the capital of France.',
                    'title': 'paris.txt',
                    'custom_metadata': [{'key': 'source_url', 'string_value': 'https://example.com/paris-facts'}],
                    'file_search_store': 'fileSearchStores/test-store',
                }
            ]
        }
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
async def test_file_search_metadata_reconstructed_with_explicit_web_fetch_streaming():
    """A metadata-only file search is not suppressed by an explicit native tool of another type."""
    raw_parts = [*_EXPLICIT_WEB_FETCH_PARTS, {'text': 'Answer.'}]
    streamed_response = _gemini_streamed_response_from_chunks(
        [_stream_chunk(raw_parts, grounding=_FILE_SEARCH_GROUNDING_METADATA)]
    )
    async for _ in streamed_response:
        pass
    streamed_model_response = streamed_response.get()
    response = _process_response(raw_parts, grounding=_FILE_SEARCH_GROUNDING_METADATA)
    expected_file_search_content = snapshot(
        [
            {
                'text': 'Paris is the capital of France.',
                'title': 'paris.txt',
                'custom_metadata': [{'key': 'source_url', 'string_value': 'https://example.com/paris-facts'}],
                'file_search_store': 'fileSearchStores/test-store',
            }
        ]
    )

    for parts in (streamed_model_response.parts, response.parts):
        assert [part.tool_name for part in parts if isinstance(part, NativeToolCallPart)] == [
            WebFetchTool.kind,
            FileSearchTool.kind,
        ]
        [file_search_return] = _file_search_returns(list(parts))
        assert file_search_return.content == expected_file_search_content

    file_search_call_index = next(
        index
        for index, part in enumerate(streamed_model_response.parts)
        if isinstance(part, NativeToolCallPart) and part.tool_name == FileSearchTool.kind
    )
    assert _utils.is_trailing_provider_metadata_native_tool_call(streamed_model_response, file_search_call_index)


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
    # A streamed file_search return is filled from grounding like the non-streamed one, so it echoes
    # its recovered contexts rather than a preserved raw response.
    assert file_search_return.provider_details is None


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
