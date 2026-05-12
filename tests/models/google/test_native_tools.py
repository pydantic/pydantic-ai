"""Tests for Google native tools serialization in `_content_model_response`.

Covers the message-history echo path that round-trips `NativeToolCallPart` /
`NativeToolReturnPart` between the API and the application:

- pre-Gemini-3 models drop server-side native parts (the API would reject them);
- `pyd_ai_`-synthesized `tool_call_id`s are dropped on every model;
- `CodeExecutionTool` uses `executable_code` / `code_execution_result` parts and is
  preserved regardless of the tool-combination capability.
"""

from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.messages import (
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
)
from pydantic_ai.native_tools import (
    CodeExecutionTool,
    FileSearchTool,
    WebSearchTool,
)

from ...conftest import try_import

with try_import() as imports_successful:
    from google.genai.types import ToolType

    from pydantic_ai.models.google import (
        _content_model_response,  # pyright: ignore[reportPrivateUsage]
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

    assert _content_model_response(response, 'google-gla') == snapshot({'role': 'model', 'parts': [{'text': 'hello'}]})
    assert _content_model_response(response, 'google-gla', supports_tool_combination=True) == snapshot(
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
    assert _content_model_response(native_only, 'google-gla') is None


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
    assert _content_model_response(response, 'google-gla', supports_tool_combination=True) == snapshot(
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
        response, 'google-gla', supports_tool_combination=supports_tool_combination
    ) == snapshot(
        {
            'role': 'model',
            'parts': [
                {'executable_code': {'language': 'PYTHON', 'code': 'print(1)'}},
                {'code_execution_result': {'outcome': 'OUTCOME_OK', 'output': '1\n'}},
            ],
        }
    )
