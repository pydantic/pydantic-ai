"""Lock-in tests for the `builtin_tools`/`builtin=`/`Builtin*` deprecations from card 35.

Each test exercises one deprecated entry point and asserts the
[`PydanticAIDeprecationWarning`][pydantic_ai._warnings.PydanticAIDeprecationWarning] message,
so the surface stays stable until removal in v2.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import NativeTool, WebSearch
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import (
    AbstractNativeTool,
    CodeExecutionTool,
    MCPServerTool,
    WebSearchTool,
)

from .conftest import try_import

with try_import() as mcp_imports:
    from pydantic_ai.capabilities import MCP

pytestmark = pytest.mark.anyio


# --- Module-level shim deprecations: `pydantic_ai.builtin_tools.*` ---


def test_builtin_tools_module_abstract_builtin_tool_renamed():
    """`from pydantic_ai.builtin_tools import AbstractBuiltinTool` warns and resolves."""
    import pydantic_ai.builtin_tools as bt

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.builtin_tools\.AbstractBuiltinTool` is deprecated, '
        r'use `pydantic_ai\.native_tools\.AbstractNativeTool`',
    ):
        cls = bt.AbstractBuiltinTool
    assert cls is AbstractNativeTool


def test_builtin_tools_module_web_search_tool_path_moved():
    """`from pydantic_ai.builtin_tools import WebSearchTool` warns about path move only."""
    import pydantic_ai.builtin_tools as bt

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'Importing `WebSearchTool` from `pydantic_ai\.builtin_tools` is deprecated, '
        r'import it from `pydantic_ai\.native_tools`',
    ):
        cls = bt.WebSearchTool
    assert cls is WebSearchTool


def test_builtin_tools_module_builtin_tool_types_renamed():
    """`from pydantic_ai.builtin_tools import BUILTIN_TOOL_TYPES` warns about rename to `NATIVE_TOOL_TYPES`."""
    import pydantic_ai.builtin_tools as bt
    from pydantic_ai.native_tools import NATIVE_TOOL_TYPES

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.builtin_tools\.BUILTIN_TOOL_TYPES` is deprecated, '
        r'use `pydantic_ai\.native_tools\.NATIVE_TOOL_TYPES`',
    ):
        registry = bt.BUILTIN_TOOL_TYPES
    assert registry is NATIVE_TOOL_TYPES


def test_builtin_tools_module_supported_builtin_tools_renamed():
    """`from pydantic_ai.builtin_tools import SUPPORTED_BUILTIN_TOOLS` warns about rename."""
    import pydantic_ai.builtin_tools as bt
    from pydantic_ai.native_tools import SUPPORTED_NATIVE_TOOLS

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.builtin_tools\.SUPPORTED_BUILTIN_TOOLS` is deprecated, '
        r'use `pydantic_ai\.native_tools\.SUPPORTED_NATIVE_TOOLS`',
    ):
        registry = bt.SUPPORTED_BUILTIN_TOOLS
    assert registry is SUPPORTED_NATIVE_TOOLS


def test_builtin_tools_module_deprecated_builtin_tools_renamed():
    """`from pydantic_ai.builtin_tools import DEPRECATED_BUILTIN_TOOLS` warns about rename."""
    import pydantic_ai.builtin_tools as bt
    from pydantic_ai.native_tools import DEPRECATED_NATIVE_TOOLS

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.builtin_tools\.DEPRECATED_BUILTIN_TOOLS` is deprecated, '
        r'use `pydantic_ai\.native_tools\.DEPRECATED_NATIVE_TOOLS`',
    ):
        registry = bt.DEPRECATED_BUILTIN_TOOLS
    assert registry is DEPRECATED_NATIVE_TOOLS


def test_builtin_tools_module_unknown_attribute_raises():
    """Unknown attributes raise `AttributeError`, not a deprecation warning."""
    import pydantic_ai.builtin_tools as bt

    with pytest.raises(AttributeError, match=r'has no attribute'):
        bt.DefinitelyDoesNotExist  # pyright: ignore[reportAttributeAccessIssue]


# --- `pydantic_ai.messages` `__getattr__` deprecations ---


def test_messages_builtin_tool_call_part_renamed():
    """`from pydantic_ai.messages import BuiltinToolCallPart` warns and resolves."""
    import pydantic_ai.messages as messages

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.messages\.BuiltinToolCallPart` is deprecated, '
        r'use `pydantic_ai\.messages\.NativeToolCallPart`',
    ):
        cls = messages.BuiltinToolCallPart
    assert cls is messages.NativeToolCallPart


def test_messages_builtin_tool_return_part_renamed():
    """`from pydantic_ai.messages import BuiltinToolReturnPart` warns and resolves."""
    import pydantic_ai.messages as messages

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.messages\.BuiltinToolReturnPart` is deprecated, '
        r'use `pydantic_ai\.messages\.NativeToolReturnPart`',
    ):
        cls = messages.BuiltinToolReturnPart
    assert cls is messages.NativeToolReturnPart


# --- `pydantic_ai.tools` `__getattr__` deprecations ---


def test_tools_builtin_tool_func_renamed():
    """`from pydantic_ai.tools import BuiltinToolFunc` warns and resolves."""
    import pydantic_ai.tools as tools

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.tools\.BuiltinToolFunc` is deprecated, '
        r'use `pydantic_ai\.tools\.NativeToolFunc`',
    ):
        alias = tools.BuiltinToolFunc
    assert alias is tools.NativeToolFunc


def test_tools_agent_builtin_tool_renamed():
    """`from pydantic_ai.tools import AgentBuiltinTool` warns and resolves."""
    import pydantic_ai.tools as tools

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.tools\.AgentBuiltinTool` is deprecated, '
        r'use `pydantic_ai\.tools\.AgentNativeTool`',
    ):
        alias = tools.AgentBuiltinTool
    assert alias is tools.AgentNativeTool


# --- `pydantic_ai.capabilities` `__getattr__` deprecations ---


def test_capabilities_builtin_tool_renamed():
    """`from pydantic_ai.capabilities import BuiltinTool` warns and resolves to `NativeTool`."""
    import pydantic_ai.capabilities as capabilities

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.capabilities\.BuiltinTool` is deprecated, '
        r'use `pydantic_ai\.capabilities\.NativeTool`',
    ):
        cls = capabilities.BuiltinTool
    assert cls is capabilities.NativeTool


def test_capabilities_builtin_or_local_tool_renamed():
    """`from pydantic_ai.capabilities import BuiltinOrLocalTool` warns and resolves to `NativeOrLocalTool`."""
    import pydantic_ai.capabilities as capabilities

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.capabilities\.BuiltinOrLocalTool` is deprecated, '
        r'use `pydantic_ai\.capabilities\.NativeOrLocalTool`',
    ):
        cls = capabilities.BuiltinOrLocalTool
    assert cls is capabilities.NativeOrLocalTool


# --- Property aliases ---


def test_model_response_builtin_tool_calls_property_deprecated():
    """`ModelResponse.builtin_tool_calls` warns and returns `native_tool_calls`."""
    from pydantic_ai.messages import ModelResponse

    response = ModelResponse(parts=[], model_name='test')
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ModelResponse\.builtin_tool_calls` is deprecated, '
        r'use `ModelResponse\.native_tool_calls`',
    ):
        result = response.builtin_tool_calls
    assert result == response.native_tool_calls


def test_model_request_parameters_builtin_tools_property_deprecated():
    """`ModelRequestParameters.builtin_tools` warns and returns `native_tools`."""
    params = ModelRequestParameters(native_tools=[WebSearchTool()])
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ModelRequestParameters\.builtin_tools` is deprecated, '
        r'use `ModelRequestParameters\.native_tools`',
    ):
        result = params.builtin_tools
    assert result == params.native_tools


def test_native_or_local_tool_builtin_attr_alias_deprecated():
    """Reading `cap.builtin` warns and returns `cap.native`."""
    cap = WebSearch()
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`WebSearch\.builtin` is deprecated, use `\.native`',
    ):
        result = cap.builtin  # pyright: ignore[reportDeprecated]
    assert result is cap.native


# --- Constructor kwarg aliases on `NativeOrLocalTool` and subclasses ---


def test_native_or_local_tool_builtin_kwarg_deprecated():
    """`NativeOrLocalTool(builtin=...)` warns and routes to `native=`."""
    from pydantic_ai.capabilities.native_or_local import NativeOrLocalTool

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`NativeOrLocalTool\(builtin=\.\.\.\)` is deprecated, use `native=`',
    ):
        cap = NativeOrLocalTool(builtin=WebSearchTool())
    assert isinstance(cap.native, WebSearchTool)


def test_web_search_builtin_kwarg_deprecated():
    """`WebSearch(builtin=...)` warns and routes to `native=`."""
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`WebSearch\(builtin=\.\.\.\)` is deprecated, use `native=`',
    ):
        cap = WebSearch(builtin=False)
    assert cap.native is False


@pytest.mark.skipif(not mcp_imports(), reason='mcp not installed')
def test_mcp_builtin_kwarg_deprecated():
    """`MCP(builtin=...)` warns and routes to `native=`."""
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`MCP\(builtin=\.\.\.\)` is deprecated, use `native=`',
    ):
        cap = MCP(
            builtin=MCPServerTool(id='deepwiki', url='https://mcp.deepwiki.com/mcp'),
            url='https://mcp.deepwiki.com/mcp',
            id='deepwiki',
        )
    assert isinstance(cap.native, MCPServerTool)


def test_model_request_parameters_builtin_tools_constructor_deprecated():
    """`ModelRequestParameters(builtin_tools=...)` warns and routes to `native_tools=`."""
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ModelRequestParameters\(builtin_tools=\.\.\.\)` is deprecated, use `native_tools=`',
    ):
        params = ModelRequestParameters(builtin_tools=[WebSearchTool()])  # pyright: ignore[reportCallIssue]
    assert len(params.native_tools) == 1
    assert isinstance(params.native_tools[0], WebSearchTool)


# --- Pydantic deserialization alias on `ModelRequestParameters` ---


def test_model_request_parameters_builtin_tools_validation_alias_silent():
    """`TypeAdapter(ModelRequestParameters).validate_python({'builtin_tools': [...]})` resolves silently.

    The Pydantic field validation alias keeps round-trip compatibility for serialized
    legacy payloads without surfacing a deprecation warning per item.
    """
    import warnings

    adapter = TypeAdapter(ModelRequestParameters)
    payload = {'builtin_tools': [{'kind': 'web_search'}]}

    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticAIDeprecationWarning)
        params = adapter.validate_python(payload)

    assert len(params.native_tools) == 1
    assert isinstance(params.native_tools[0], WebSearchTool)


# --- `Agent` constructor and per-call kwarg deprecations ---


def test_agent_builtin_tools_constructor_deprecated():
    """`Agent(model, builtin_tools=[...])` warns and registers as a native tool capability."""
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`Agent\(builtin_tools=\.\.\.\)` is deprecated, '
        r'use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        agent = Agent(TestModel(), builtin_tools=[WebSearchTool()])  # pyright: ignore[reportDeprecated]

    assert len(agent._cap_native_tools) == 1  # pyright: ignore[reportPrivateUsage]


async def test_agent_run_builtin_tools_kwarg_deprecated():
    """`agent.run(..., builtin_tools=[...])` warns and forwards to `native_tools=`."""
    agent = Agent(TestModel())

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`builtin_tools=` is deprecated, use `native_tools=`',
    ):
        with pytest.raises(UserError, match='TestModel does not support built-in tools'):
            await agent.run('hi', builtin_tools=[WebSearchTool()])  # pyright: ignore[reportCallIssue]


def test_agent_run_sync_builtin_tools_kwarg_deprecated():
    """`agent.run_sync(..., builtin_tools=[...])` warns and forwards to `native_tools=`."""
    agent = Agent(TestModel())

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`builtin_tools=` is deprecated, use `native_tools=`',
    ):
        with pytest.raises(UserError, match='TestModel does not support built-in tools'):
            agent.run_sync('hi', builtin_tools=[WebSearchTool()])  # pyright: ignore[reportCallIssue]


async def test_agent_run_stream_builtin_tools_kwarg_deprecated():
    """`agent.run_stream(..., builtin_tools=[...])` warns and forwards to `native_tools=`."""
    agent = Agent(TestModel())

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`builtin_tools=` is deprecated, use `native_tools=`',
    ):
        with pytest.raises(UserError, match='TestModel does not support built-in tools'):
            async with agent.run_stream('hi', builtin_tools=[WebSearchTool()]):  # pyright: ignore[reportCallIssue]
                ...  # pragma: no cover


async def test_agent_iter_builtin_tools_kwarg_deprecated():
    """`agent.iter(..., builtin_tools=[...])` warns and forwards to `native_tools=`."""
    agent = Agent(TestModel())

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`builtin_tools=` is deprecated, use `native_tools=`',
    ):
        with pytest.raises(UserError, match='TestModel does not support built-in tools'):
            async with agent.iter('hi', builtin_tools=[WebSearchTool()]) as agent_run:  # pyright: ignore[reportCallIssue]
                async for _ in agent_run:  # pragma: no cover
                    pass


def test_agent_override_builtin_tools_kwarg_deprecated():
    """`agent.override(builtin_tools=[...])` warns and forwards to `native_tools=`."""
    model = TestModel()
    agent = Agent(model, capabilities=[NativeTool(WebSearchTool())])

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`builtin_tools=` is deprecated, use `native_tools=`',
    ):
        with (
            agent.override(builtin_tools=[CodeExecutionTool()]),  # pyright: ignore[reportCallIssue]
            pytest.raises(UserError, match='TestModel does not support built-in tools'),
        ):
            agent.run_sync('hi')

    assert model.last_model_request_parameters is not None
    assert model.last_model_request_parameters.native_tools == [CodeExecutionTool()]
