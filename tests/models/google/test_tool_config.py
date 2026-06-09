"""Tests for `GoogleModel`'s `tool_config` construction.

A `function_calling_config` only governs *function* tools. The Gemini Developer API
(`GoogleProvider(api_key=...)`, the `generativelanguage.googleapis.com` backend) rejects one
that has no `function_declarations` to apply to — ``400 INVALID_ARGUMENT: Function calling
config is set without function_declarations`` — which is what happens when only native tools
(e.g. web search) are configured. Vertex AI tolerates the empty config, so the regression only
bit Gemini Developer API users.

The no-network tests assert the request shape directly: that is the actual regression guard,
because a VCR *replay* test cannot catch a malformed request (it replays a recorded response and
never re-validates against the API — which is why the pre-existing `test_google_model_web_search_tool`
VCR test stayed green through the regression). The VCR test below proves the fix end-to-end: its
cassette can only be recorded against the live API once the request is accepted.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.tools import ToolDefinition

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `NativeToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `NativeToolReturnPart` instead.:DeprecationWarning'
    ),
]


@dataclass(frozen=True)
class Case:
    id: str
    model: str
    request_parameters: ModelRequestParameters
    expected_tool_config: dict[str, Any] | None
    marks: tuple[pytest.MarkDecorator, ...] = ()


CASES = [
    Case(
        id='native-only-pre-gemini-3-omits-config',
        model='gemini-2.5-pro',
        request_parameters=ModelRequestParameters(native_tools=[WebSearchTool()]),
        expected_tool_config=None,
    ),
    Case(
        id='native-only-gemini-3-keeps-only-server-side-flag',
        model='gemini-3-flash-preview',
        request_parameters=ModelRequestParameters(native_tools=[WebSearchTool()]),
        expected_tool_config={'include_server_side_tool_invocations': True},
    ),
    Case(
        id='function-tool-keeps-config',
        model='gemini-2.5-pro',
        request_parameters=ModelRequestParameters(function_tools=[ToolDefinition(name='get_weather')]),
        expected_tool_config={'function_calling_config': {'mode': 'AUTO'}},
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
async def test_tool_config_set_only_when_function_tools_present(allow_model_requests: None, case: Case):
    m = GoogleModel(case.model, provider=GoogleProvider(api_key='test-key'))

    _, config = await m._build_content_and_config(  # pyright: ignore[reportPrivateUsage]
        messages=[ModelRequest(parts=[UserPromptPart(content='Hello')])],
        model_settings={},
        model_request_parameters=case.request_parameters,
    )

    assert cast(dict[str, Any], config).get('tool_config') == case.expected_tool_config


@pytest.mark.vcr()
async def test_native_tool_only_web_search_completes(allow_model_requests: None, gemini_api_key: str):
    """A native-tool-only request must reach the live API and return an answer.

    On the buggy code this 400s before any response, so the cassette could not have been recorded.
    Pinned to a Gemini 2.5 model on purpose: that is the line the empty `function_calling_config`
    actually breaks. Gemini 3+ sets `include_server_side_tool_invocations` and the API tolerates the
    empty config there, so a Gemini 3 model would pass with or without the fix and prove nothing.
    """
    m = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(m, capabilities=[NativeTool(WebSearchTool())])

    result = await agent.run('What is the weather in San Francisco right now?')

    assert result.output
