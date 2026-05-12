"""Lock-in tests for the `builtin_tools`/`builtin=`/`Builtin*` deprecations from card 35.

Each test exercises one deprecated entry point and asserts the
[`PydanticAIDeprecationWarning`][pydantic_ai._warnings.PydanticAIDeprecationWarning] message,
so the surface stays stable until removal in v2.
"""

from __future__ import annotations

from typing import Any, cast

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
        bt.DefinitelyDoesNotExist


# --- Top-level `pydantic_ai` `__getattr__` deprecations ---


def test_top_level_builtin_tool_call_part_renamed():
    """`from pydantic_ai import BuiltinToolCallPart` warns and resolves to `NativeToolCallPart`."""
    import pydantic_ai

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.BuiltinToolCallPart` is deprecated, use `pydantic_ai\.NativeToolCallPart`',
    ):
        cls = pydantic_ai.BuiltinToolCallPart
    assert cls is pydantic_ai.NativeToolCallPart


def test_top_level_builtin_tool_return_part_renamed():
    """`from pydantic_ai import BuiltinToolReturnPart` warns and resolves to `NativeToolReturnPart`."""
    import pydantic_ai

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.BuiltinToolReturnPart` is deprecated, use `pydantic_ai\.NativeToolReturnPart`',
    ):
        cls = pydantic_ai.BuiltinToolReturnPart
    assert cls is pydantic_ai.NativeToolReturnPart


def test_top_level_agent_builtin_tool_renamed():
    """`from pydantic_ai import AgentBuiltinTool` warns and resolves to `AgentNativeTool`."""
    import pydantic_ai

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`pydantic_ai\.AgentBuiltinTool` is deprecated, use `pydantic_ai\.AgentNativeTool`',
    ):
        alias = pydantic_ai.AgentBuiltinTool
    assert alias is pydantic_ai.AgentNativeTool


def test_top_level_unknown_attribute_raises():
    """Unknown attributes on `pydantic_ai` raise `AttributeError`, not a deprecation warning."""
    import pydantic_ai

    with pytest.raises(AttributeError, match=r'has no attribute'):
        pydantic_ai.DefinitelyDoesNotExist


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
        result = cap.builtin
    assert result is cap.native


# --- Constructor kwarg aliases on `NativeOrLocalTool` and subclasses ---


def test_native_or_local_tool_builtin_kwarg_deprecated():
    """`NativeOrLocalTool(builtin=...)` warns and routes to `native=`."""
    from pydantic_ai.capabilities.native_or_local import NativeOrLocalTool

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`NativeOrLocalTool\(builtin=\.\.\.\)` is deprecated, use `native=`',
    ):
        cap = NativeOrLocalTool(builtin=WebSearchTool())  # pyright: ignore[reportCallIssue]
    assert isinstance(cap.native, WebSearchTool)


def test_web_search_builtin_kwarg_deprecated():
    """`WebSearch(builtin=...)` warns and routes to `native=`."""
    # `local=False` avoids invoking `_default_local()` (DuckDuckGo lookup),
    # which would emit unrelated warnings in slim CI environments.
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`WebSearch\(builtin=\.\.\.\)` is deprecated, use `native=`',
    ):
        cap = WebSearch(builtin=WebSearchTool(), local=False)  # pyright: ignore[reportCallIssue]
    assert isinstance(cap.native, WebSearchTool)


@pytest.mark.skipif(not mcp_imports(), reason='mcp not installed')
def test_mcp_builtin_kwarg_deprecated():
    """`MCP(builtin=...)` warns and routes to `native=`."""
    # `local=False` avoids invoking `_default_local()` (imports `pydantic_ai.mcp`),
    # which would fail when the `mcp` extra isn't installed.
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`MCP\(builtin=\.\.\.\)` is deprecated, use `native=`',
    ):
        cap = MCP(
            builtin=MCPServerTool(id='deepwiki', url='https://mcp.deepwiki.com/mcp'),  # pyright: ignore[reportCallIssue]
            url='https://mcp.deepwiki.com/mcp',
            id='deepwiki',
            local=False,
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
        agent = Agent(TestModel(), builtin_tools=[WebSearchTool()])  # pyright: ignore[reportCallIssue]

    assert len(agent._cap_native_tools) == 1  # pyright: ignore[reportPrivateUsage]


async def test_agent_run_builtin_tools_kwarg_deprecated():
    """`agent.run(..., builtin_tools=[...])` warns and routes through `capabilities=[NativeTool(...)]`."""
    agent = Agent(TestModel())

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`agent\.run\(builtin_tools=\.\.\.\)` is deprecated, '
        r'use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        with pytest.raises(UserError, match='TestModel does not support built-in tools'):
            await agent.run('hi', builtin_tools=[WebSearchTool()])  # pyright: ignore[reportCallIssue]


def test_agent_run_sync_builtin_tools_kwarg_deprecated():
    """`agent.run_sync(..., builtin_tools=[...])` warns and routes through `capabilities=[NativeTool(...)]`."""
    agent = Agent(TestModel())

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`agent\.run_sync\(builtin_tools=\.\.\.\)` is deprecated, '
        r'use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        with pytest.raises(UserError, match='TestModel does not support built-in tools'):
            agent.run_sync('hi', builtin_tools=[WebSearchTool()])  # pyright: ignore[reportCallIssue]


async def test_agent_run_stream_builtin_tools_kwarg_deprecated():
    """`agent.run_stream(..., builtin_tools=[...])` warns and routes through `capabilities=[NativeTool(...)]`."""
    agent = Agent(TestModel())

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`agent\.run_stream\(builtin_tools=\.\.\.\)` is deprecated, '
        r'use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        with pytest.raises(UserError, match='TestModel does not support built-in tools'):
            async with agent.run_stream('hi', builtin_tools=[WebSearchTool()]):  # pyright: ignore[reportCallIssue]
                ...  # pragma: no cover


async def test_agent_iter_builtin_tools_kwarg_deprecated():
    """`agent.iter(..., builtin_tools=[...])` warns and routes through `capabilities=[NativeTool(...)]`."""
    agent = Agent(TestModel())

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`agent\.iter\(builtin_tools=\.\.\.\)` is deprecated, '
        r'use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        with pytest.raises(UserError, match='TestModel does not support built-in tools'):
            async with agent.iter('hi', builtin_tools=[WebSearchTool()]) as agent_run:  # pyright: ignore[reportCallIssue,reportUnknownVariableType]
                async for _ in agent_run:  # pyright: ignore[reportUnknownVariableType]
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
            agent.override(builtin_tools=[CodeExecutionTool()]),
            pytest.raises(UserError, match='TestModel does not support built-in tools'),
        ):
            agent.run_sync('hi')

    assert model.last_model_request_parameters is not None
    assert model.last_model_request_parameters.native_tools == [CodeExecutionTool()]


def test_agent_override_native_tools_wins_when_both_passed():
    """`agent.override(native_tools=..., builtin_tools=...)` warns and the explicit `native_tools=` wins."""
    model = TestModel()
    agent = Agent(model, capabilities=[NativeTool(WebSearchTool())])

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`builtin_tools=` is deprecated, use `native_tools=`',
    ):
        with (
            agent.override(
                native_tools=[CodeExecutionTool()],
                builtin_tools=[WebSearchTool()],
            ),
            pytest.raises(UserError, match='TestModel does not support built-in tools'),
        ):
            agent.run_sync('hi')

    # The explicit `native_tools=` wins; the legacy `builtin_tools=` is discarded.
    assert model.last_model_request_parameters is not None
    assert model.last_model_request_parameters.native_tools == [CodeExecutionTool()]


# --- Class-method constructors: `Agent.from_spec` / `Agent.from_file` ---


def test_agent_from_spec_builtin_tools_kwarg_deprecated():
    """`Agent.from_spec(spec, builtin_tools=[...])` warns and registers as a native tool capability."""
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`Agent\.from_spec\(builtin_tools=\.\.\.\)` is deprecated, '
        r'use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        agent = Agent.from_spec(  # pyright: ignore[reportCallIssue,reportUnknownVariableType]
            {'model': 'test'},
            builtin_tools=[WebSearchTool()],
        )

    native_tools: list[Any] = list(agent._cap_native_tools)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert len(native_tools) == 1
    assert isinstance(native_tools[0], WebSearchTool)


def test_agent_from_file_builtin_tools_kwarg_deprecated(tmp_path: Any):
    """`Agent.from_file(path, builtin_tools=[...])` warns and registers as a native tool capability."""
    spec_path = tmp_path / 'agent.yaml'
    spec_path.write_text('model: test\n')

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`Agent\.from_file\(builtin_tools=\.\.\.\)` is deprecated, '
        r'use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        agent = Agent.from_file(  # pyright: ignore[reportCallIssue,reportUnknownVariableType]
            spec_path,
            builtin_tools=[WebSearchTool()],
        )

    native_tools: list[Any] = list(agent._cap_native_tools)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    assert len(native_tools) == 1
    assert isinstance(native_tools[0], WebSearchTool)


# --- Additional streaming entry-point deprecations ---


def test_agent_run_stream_sync_builtin_tools_kwarg_deprecated():
    """`agent.run_stream_sync(..., builtin_tools=[...])` warns and routes through `capabilities=[NativeTool(...)]`."""
    agent = Agent(TestModel())

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`agent\.run_stream_sync\(builtin_tools=\.\.\.\)` is deprecated, '
        r'use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        with pytest.raises(UserError, match='TestModel does not support built-in tools'):
            agent.run_stream_sync('hi', builtin_tools=[WebSearchTool()])  # pyright: ignore[reportCallIssue]


async def test_agent_run_stream_events_builtin_tools_kwarg_deprecated():
    """`agent.run_stream_events(..., builtin_tools=[...])` warns and routes through `capabilities=[NativeTool(...)]`."""
    agent = Agent(TestModel())

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`agent\.run_stream_events\(builtin_tools=\.\.\.\)` is deprecated, '
        r'use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        with pytest.raises(UserError, match='TestModel does not support built-in tools'):
            async with agent.run_stream_events('hi', builtin_tools=[WebSearchTool()]) as stream:  # pyright: ignore[reportCallIssue,reportUnknownVariableType]
                async for _ in stream:  # pyright: ignore[reportUnknownVariableType]
                    pass


async def test_wrapper_agent_iter_builtin_tools_kwarg_deprecated():
    """`WrapperAgent(agent).iter(..., builtin_tools=[...])` warns and routes through `capabilities=[NativeTool(...)]`."""
    from pydantic_ai.agent import WrapperAgent

    agent = Agent(TestModel())
    wrapped = WrapperAgent(agent)

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`agent\.iter\(builtin_tools=\.\.\.\)` is deprecated, '
        r'use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        with pytest.raises(UserError, match='TestModel does not support built-in tools'):
            async with wrapped.iter('hi', builtin_tools=[WebSearchTool()]) as agent_run:  # pyright: ignore[reportCallIssue,reportUnknownVariableType]
                async for _ in agent_run:  # pyright: ignore[reportUnknownVariableType]
                    pass


# --- Override path where BOTH `native_tools=` and `builtin_tools=` are passed ---


def test_native_or_local_tool_native_wins_when_both_kwargs_passed():
    """`WebSearch(native=..., builtin=...)` warns and the explicit `native=` wins; the legacy `builtin=` is discarded."""
    explicit_native = WebSearchTool()
    legacy_builtin = WebSearchTool()

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`WebSearch\(builtin=\.\.\.\)` is deprecated, use `native=`',
    ):
        cap = WebSearch(
            native=explicit_native,
            builtin=legacy_builtin,  # pyright: ignore[reportCallIssue]
            local=False,
        )
    assert cap.native is explicit_native


def test_native_or_local_tool_getattr_unknown_attribute_raises():
    """Accessing an unknown attribute on a `NativeOrLocalTool` raises `AttributeError` (not a deprecation warning)."""
    cap = WebSearch(local=False)
    with pytest.raises(AttributeError, match='definitely_not_a_real_attr'):
        cap.definitely_not_a_real_attr


# --- `ModelRequestParameters` constructor: both kwargs passed ---


def test_model_request_parameters_native_tools_wins_when_both_kwargs_passed():
    """`ModelRequestParameters(native_tools=..., builtin_tools=...)` warns and the explicit `native_tools=` wins."""
    explicit = WebSearchTool()
    legacy = CodeExecutionTool()

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ModelRequestParameters\(builtin_tools=\.\.\.\)` is deprecated, use `native_tools=`',
    ):
        params = ModelRequestParameters(
            native_tools=[explicit],
            builtin_tools=[legacy],  # pyright: ignore[reportCallIssue]
        )

    assert params.native_tools == [explicit]


# --- `UIAdapter` deprecations ---


async def test_ui_adapter_run_stream_native_capabilities_and_builtin_tools_kwarg():
    """`UIAdapter.run_stream_native(capabilities=[...], builtin_tools=[...])` merges both into the run.

    Exercises both the explicit `capabilities=` branch and the deprecated `builtin_tools=` branch
    inside `run_stream_native` (`run_capabilities.extend(capabilities)` and
    `run_capabilities.extend(extra_capabilities)`).
    """
    starlette = pytest.importorskip('starlette')
    del starlette

    from pydantic_ai.capabilities import ReinjectSystemPrompt
    from pydantic_ai.messages import ModelRequest

    from .test_ui import DummyUIAdapter, DummyUIRunInput

    agent = Agent(model=TestModel())
    adapter = DummyUIAdapter(agent, DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Hello')]))

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`UIAdapter\.run_stream_native\(builtin_tools=\.\.\.\)` is deprecated, '
        r'use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        with pytest.raises(UserError, match='TestModel does not support built-in tools'):
            async for _ in adapter.run_stream_native(
                capabilities=[ReinjectSystemPrompt()],
                builtin_tools=[WebSearchTool()],
            ):
                pass  # pragma: no cover


async def test_ui_adapter_dispatch_request_builtin_tools_kwarg_deprecated():
    """`UIAdapter.dispatch_request(..., builtin_tools=[...])` forwards the legacy kwarg through `run_stream_native`."""
    starlette = pytest.importorskip('starlette')
    del starlette

    from starlette.requests import Request as _StarletteRequest
    from starlette.responses import StreamingResponse as _StreamingResponse

    from pydantic_ai.messages import ModelRequest

    from .test_ui import DummyUIAdapter, DummyUIRunInput

    agent = Agent(model=TestModel())
    request_body = DummyUIRunInput(messages=[ModelRequest.user_text_prompt('Hello')])

    async def receive() -> dict[str, Any]:
        return {'type': 'http.request', 'body': request_body.model_dump_json().encode('utf-8')}

    starlette_request = _StarletteRequest(
        scope={
            'type': 'http',
            'method': 'POST',
            'headers': [(b'content-type', b'application/json')],
        },
        receive=receive,
    )

    # The deprecation warning fires synchronously inside `dispatch_request` when it
    # forwards `builtin_tools=` through to `run_stream_native`. Reaching it confirms the
    # legacy kwarg was extracted from `**kwargs` before forwarding to `from_request`.
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`UIAdapter\.run_stream_native\(builtin_tools=\.\.\.\)` is deprecated, '
        r'use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        response = await DummyUIAdapter.dispatch_request(
            starlette_request,
            agent=agent,
            builtin_tools=[WebSearchTool()],
        )
    assert isinstance(response, _StreamingResponse)


def test_ag_ui_app_builtin_tools_kwarg_routed_to_capabilities(monkeypatch: pytest.MonkeyPatch):
    """`AGUIApp(builtin_tools=[...])` warns AND forwards the legacy tools to `dispatch_request`.

    Lock-in for a Devin-flagged bug where the helper return value was previously
    discarded, silently dropping any deprecated `builtin_tools=` before the request
    reached the agent.
    """
    pytest.importorskip('starlette')
    pytest.importorskip('pydantic_ai.ui.ag_ui')
    from pydantic_ai.ui.ag_ui import AGUIAdapter
    from pydantic_ai.ui.ag_ui.app import AGUIApp

    captured: dict[str, object] = {}

    async def fake_dispatch_request(*_args: object, **kwargs: object) -> None:
        captured.update(kwargs)
        return None

    monkeypatch.setattr(AGUIAdapter, 'dispatch_request', fake_dispatch_request)

    agent = Agent(TestModel())
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`AGUIApp\(builtin_tools=\.\.\.\)` is deprecated, use `capabilities=\[NativeTool\(\.\.\.\)\]`',
    ):
        app = AGUIApp(agent, builtin_tools=[WebSearchTool()])

    # Find the registered POST handler and invoke it so dispatch_request runs.
    from starlette.routing import Route

    route = next(r for r in app.routes if isinstance(r, Route) and r.path == '/')
    import asyncio

    asyncio.run(route.endpoint(object()))

    capabilities = captured.get('capabilities')
    assert isinstance(capabilities, list)
    capabilities_list = cast(list[NativeTool[Any]], capabilities)
    assert any(isinstance(cap, NativeTool) and isinstance(cap.tool, WebSearchTool) for cap in capabilities_list)


# --- Audit-driven deprecations (post-rename version-policy fixes) ---


def test_tool_definition_prefer_builtin_constructor_deprecated():
    """`ToolDefinition(prefer_builtin=...)` warns and routes to `prefer_native=`."""
    from pydantic_ai.tools import ToolDefinition

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ToolDefinition\(prefer_builtin=\.\.\.\)` is deprecated, use `prefer_native=`',
    ):
        td = ToolDefinition(name='foo', prefer_builtin='web_search')  # pyright: ignore[reportCallIssue]
    assert td.prefer_native == 'web_search'


def test_tool_definition_prefer_builtin_attribute_deprecated():
    """Reading `ToolDefinition.prefer_builtin` warns and returns `prefer_native`."""
    from pydantic_ai.tools import ToolDefinition

    td = ToolDefinition(name='foo', prefer_native='web_search')
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ToolDefinition\.prefer_builtin` is deprecated, use `ToolDefinition\.prefer_native`',
    ):
        result = td.prefer_builtin
    assert result == td.prefer_native == 'web_search'


def test_tool_definition_prefer_native_wins_when_both_kwargs_passed():
    """`ToolDefinition(prefer_native=..., prefer_builtin=...)` warns and the explicit `prefer_native=` wins."""
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ToolDefinition\(prefer_builtin=\.\.\.\)` is deprecated, use `prefer_native=`',
    ):
        from pydantic_ai.tools import ToolDefinition

        td = ToolDefinition(
            name='foo',
            prefer_native='web_search',
            prefer_builtin='code_execution',  # pyright: ignore[reportCallIssue]
        )
    assert td.prefer_native == 'web_search'


def test_model_profile_supported_builtin_tools_constructor_deprecated():
    """`ModelProfile(supported_builtin_tools=...)` warns and routes to `supported_native_tools=`."""
    from pydantic_ai.profiles import ModelProfile

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ModelProfile\(supported_builtin_tools=\.\.\.\)` is deprecated, '
        r'use `supported_native_tools=`',
    ):
        profile = ModelProfile(supported_builtin_tools=frozenset({WebSearchTool}))  # pyright: ignore[reportCallIssue]
    assert profile.supported_native_tools == frozenset({WebSearchTool})


def test_model_profile_supported_builtin_tools_attribute_deprecated():
    """Reading `ModelProfile.supported_builtin_tools` warns and returns `supported_native_tools`."""
    from pydantic_ai.profiles import ModelProfile

    profile = ModelProfile(supported_native_tools=frozenset({WebSearchTool}))
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ModelProfile\.supported_builtin_tools` is deprecated, use `\.supported_native_tools`',
    ):
        result = profile.supported_builtin_tools
    assert result == profile.supported_native_tools == frozenset({WebSearchTool})


def test_model_profile_supported_native_tools_wins_when_both_kwargs_passed():
    """`ModelProfile(supported_native_tools=..., supported_builtin_tools=...)` warns and the explicit `supported_native_tools=` wins."""
    from pydantic_ai.profiles import ModelProfile

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ModelProfile\(supported_builtin_tools=\.\.\.\)` is deprecated, '
        r'use `supported_native_tools=`',
    ):
        profile = ModelProfile(
            supported_native_tools=frozenset({WebSearchTool}),
            supported_builtin_tools=frozenset({CodeExecutionTool}),  # pyright: ignore[reportCallIssue]
        )
    assert profile.supported_native_tools == frozenset({WebSearchTool})


def test_model_profile_subclass_supported_builtin_tools_constructor_deprecated():
    """`OpenAIModelProfile(supported_builtin_tools=...)` propagates the deprecated alias through subclasses."""
    from pydantic_ai.profiles.openai import OpenAIModelProfile

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`OpenAIModelProfile\(supported_builtin_tools=\.\.\.\)` is deprecated, '
        r'use `supported_native_tools=`',
    ):
        profile = OpenAIModelProfile(supported_builtin_tools=frozenset({WebSearchTool}))  # pyright: ignore[reportCallIssue]
    assert profile.supported_native_tools == frozenset({WebSearchTool})


def test_model_subclass_supported_builtin_tools_override_still_used():
    """`Model` subclass overriding only the deprecated classmethod still has its tools picked up."""
    from pydantic_ai.models import Model

    # Subclass creation should warn that the override uses the deprecated name.
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'overrides `supported_builtin_tools\(\)`, which is deprecated — '
        r'override `supported_native_tools\(\)` instead',
    ):

        class _LegacyModel(Model[Any]):
            @classmethod
            def supported_builtin_tools(cls) -> frozenset[type[AbstractNativeTool]]:
                return frozenset({WebSearchTool})

            @property
            def system(self) -> str:
                return 'test'  # pragma: no cover

            @property
            def model_name(self) -> str:
                return 'test'  # pragma: no cover

            async def request(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError  # pragma: no cover

    # Framework lookup via the new name reaches the user's legacy override.
    assert _LegacyModel.supported_native_tools() == frozenset({WebSearchTool})
    # And the legacy classmethod direct call still works.
    assert _LegacyModel.supported_builtin_tools() == frozenset({WebSearchTool})


def test_model_supported_builtin_tools_classmethod_deprecated_on_base():
    """`Model.supported_builtin_tools()` warns and dispatches to `supported_native_tools()` for unmodified subclasses."""
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`Model\.supported_builtin_tools\(\)` is deprecated, '
        r'use `Model\.supported_native_tools\(\)`',
    ):
        result = TestModel.supported_builtin_tools()
    assert result == TestModel.supported_native_tools()


def test_google_profile_supports_native_output_with_builtin_tools_constructor_deprecated():
    """`GoogleModelProfile(google_supports_native_output_with_builtin_tools=...)` warns and routes to the renamed field."""
    from pydantic_ai.profiles.google import GoogleModelProfile

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`GoogleModelProfile\(google_supports_native_output_with_builtin_tools=\.\.\.\)` is deprecated, '
        r'use `google_supports_native_output_with_native_tools=`',
    ):
        profile = GoogleModelProfile(google_supports_native_output_with_builtin_tools=True)  # pyright: ignore[reportCallIssue]
    assert profile.google_supports_native_output_with_native_tools is True


def test_google_profile_supports_native_output_with_builtin_tools_attribute_deprecated():
    """Reading `GoogleModelProfile.google_supports_native_output_with_builtin_tools` warns and returns the renamed field."""
    from pydantic_ai.profiles.google import GoogleModelProfile

    profile = GoogleModelProfile(google_supports_native_output_with_native_tools=True)
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`GoogleModelProfile\.google_supports_native_output_with_builtin_tools` is deprecated, '
        r'use `\.google_supports_native_output_with_native_tools`',
    ):
        result = profile.google_supports_native_output_with_builtin_tools
    assert result is profile.google_supports_native_output_with_native_tools is True


def test_openai_responses_settings_openai_builtin_tools_key_deprecated():
    """The deprecated `openai_builtin_tools` settings key warns and its tools reach the resolved native-tool list."""
    from pydantic_ai.models.openai import (  # pyright: ignore[reportPrivateUsage]
        OpenAIResponsesModelSettings,
        _resolve_openai_native_tools_setting,
    )

    legacy_tool: dict[str, Any] = {'type': 'web_search_preview'}
    settings_with_legacy = cast(OpenAIResponsesModelSettings, {'openai_builtin_tools': [legacy_tool]})

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`OpenAIResponsesModelSettings\(\{"openai_builtin_tools": \[\.\.\.\]\}\)` is deprecated, '
        r'use `openai_native_tools`',
    ):
        resolved = _resolve_openai_native_tools_setting(settings_with_legacy)
    assert list(resolved) == [legacy_tool]


def test_openai_responses_settings_openai_native_tools_key_wins_when_both_passed():
    """When both legacy and new keys are present, the explicit `openai_native_tools` wins and no warning fires."""
    import warnings

    from pydantic_ai.models.openai import (  # pyright: ignore[reportPrivateUsage]
        OpenAIResponsesModelSettings,
        _resolve_openai_native_tools_setting,
    )

    new_tool: dict[str, Any] = {'type': 'web_search'}
    legacy_tool: dict[str, Any] = {'type': 'web_search_preview'}
    settings_with_both = cast(
        OpenAIResponsesModelSettings,
        {'openai_native_tools': [new_tool], 'openai_builtin_tools': [legacy_tool]},
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticAIDeprecationWarning)
        resolved = _resolve_openai_native_tools_setting(settings_with_both)
    assert list(resolved) == [new_tool]


def test_abstract_capability_get_builtin_tools_override_still_used():
    """`AbstractCapability` subclass overriding only `get_builtin_tools()` keeps tools picked up by `get_native_tools()`."""
    from pydantic_ai.capabilities.abstract import AbstractCapability
    from pydantic_ai.tools import AgentNativeTool

    tool = WebSearchTool()

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'overrides `get_builtin_tools\(\)`, which is deprecated — '
        r'override `get_native_tools\(\)` instead',
    ):

        class _LegacyCap(AbstractCapability[Any]):
            def get_builtin_tools(self) -> list[AgentNativeTool[Any]]:
                return [tool]

    cap = _LegacyCap()
    # Framework lookup via the new name returns the legacy override's tools.
    assert list(cap.get_native_tools()) == [tool]


def test_abstract_capability_get_builtin_tools_base_method_deprecated():
    """Calling `AbstractCapability.get_builtin_tools()` on a non-legacy subclass warns and dispatches to `get_native_tools()`."""
    from pydantic_ai.capabilities.abstract import AbstractCapability
    from pydantic_ai.tools import AgentNativeTool

    class _ModernCap(AbstractCapability[Any]):
        def get_native_tools(self) -> list[AgentNativeTool[Any]]:
            return [WebSearchTool()]

    cap = _ModernCap()
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`AbstractCapability\.get_builtin_tools\(\)` is deprecated, use `get_native_tools\(\)`',
    ):
        legacy_result = list(cap.get_builtin_tools())
    assert legacy_result == list(cap.get_native_tools())


def test_image_generation_subagent_tool_builtin_tool_constructor_deprecated():
    """`ImageGenerationSubagentTool(builtin_tool=...)` warns and routes to `native_tool=`."""
    from pydantic_ai.common_tools.image_generation import ImageGenerationSubagentTool
    from pydantic_ai.native_tools import ImageGenerationTool

    legacy_tool = ImageGenerationTool()

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ImageGenerationSubagentTool\(builtin_tool=\.\.\.\)` is deprecated, use `native_tool=`',
    ):
        subagent = ImageGenerationSubagentTool(
            model='openai-responses:gpt-5',
            builtin_tool=legacy_tool,  # pyright: ignore[reportCallIssue]
        )
    assert subagent.native_tool is legacy_tool


def test_image_generation_subagent_tool_builtin_tool_attribute_deprecated():
    """Reading `ImageGenerationSubagentTool.builtin_tool` warns and returns `native_tool`."""
    from pydantic_ai.common_tools.image_generation import ImageGenerationSubagentTool
    from pydantic_ai.native_tools import ImageGenerationTool

    native_tool = ImageGenerationTool()
    subagent = ImageGenerationSubagentTool(model='openai-responses:gpt-5', native_tool=native_tool)
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ImageGenerationSubagentTool\.builtin_tool` is deprecated, use `\.native_tool`',
    ):
        result = subagent.builtin_tool
    assert result is native_tool


def test_image_generation_subagent_tool_native_tool_wins_when_both_kwargs_passed():
    """When both kwargs are passed, the explicit `native_tool=` wins and the legacy `builtin_tool=` is dropped."""
    from pydantic_ai.common_tools.image_generation import ImageGenerationSubagentTool
    from pydantic_ai.native_tools import ImageGenerationTool

    explicit = ImageGenerationTool()
    legacy = ImageGenerationTool()

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`ImageGenerationSubagentTool\(builtin_tool=\.\.\.\)` is deprecated, use `native_tool=`',
    ):
        subagent = ImageGenerationSubagentTool(
            model='openai-responses:gpt-5',
            native_tool=explicit,
            builtin_tool=legacy,  # pyright: ignore[reportCallIssue]
        )
    assert subagent.native_tool is explicit


def test_image_generation_tool_factory_builtin_tool_kwarg_deprecated():
    """`image_generation_tool(builtin_tool=...)` warns and forwards the legacy kwarg to the subagent."""
    from pydantic_ai.common_tools.image_generation import image_generation_tool
    from pydantic_ai.native_tools import ImageGenerationTool

    legacy_tool = ImageGenerationTool()

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`image_generation_tool\(builtin_tool=\.\.\.\)` is deprecated, use `native_tool=`',
    ):
        tool = image_generation_tool(
            'openai-responses:gpt-5',
            builtin_tool=legacy_tool,
        )

    # The legacy value reaches the subagent's `native_tool` field via the factory.
    subagent_self = tool.function.__self__  # pyright: ignore[reportFunctionMemberAccess]
    assert subagent_self.native_tool is legacy_tool


# --- Profile-subclass and spec-loader deprecations (post-audit catch-ups) ---


def test_bedrock_profile_supported_builtin_tools_constructor_deprecated():
    """`BedrockModelProfile(supported_builtin_tools=...)` warns and routes to `supported_native_tools=`.

    Lock-in for the audit gap where `BedrockModelProfile` (in `providers/bedrock.py`)
    was missed when the prior fix wired the deprecated kwarg through each known profile
    subclass via explicit `install_deprecated_kwarg_alias` calls.
    """
    from pydantic_ai.providers.bedrock import BedrockModelProfile

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`BedrockModelProfile\(supported_builtin_tools=\.\.\.\)` is deprecated, '
        r'use `supported_native_tools=`',
    ):
        profile = BedrockModelProfile(supported_builtin_tools=frozenset({WebSearchTool}))  # pyright: ignore[reportCallIssue]
    assert profile.supported_native_tools == frozenset({WebSearchTool})


def test_user_subclass_modelprofile_supported_builtin_tools_constructor_deprecated():
    """User-defined `@dataclass(kw_only=True)` subclasses of `ModelProfile` still accept the legacy kwarg.

    Lock-in for the lazy-install pattern in `ModelProfile.__new__`: each subclass's
    dataclass-generated `__init__` is wrapped on first instantiation, so user-defined
    subclasses get the alias without needing an explicit `install_deprecated_kwarg_alias`
    call.
    """
    from dataclasses import dataclass

    from pydantic_ai.profiles import ModelProfile

    @dataclass(kw_only=True)
    class _UserProfile(ModelProfile):
        pass

    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`_UserProfile\(supported_builtin_tools=\.\.\.\)` is deprecated, '
        r'use `supported_native_tools=`',
    ):
        profile = _UserProfile(supported_builtin_tools=frozenset({WebSearchTool}))  # pyright: ignore[reportCallIssue]
    assert WebSearchTool in profile.supported_native_tools


def test_agent_from_spec_with_builtin_tool_capability_key_deprecated():
    """`Agent.from_spec({'capabilities': [{'BuiltinTool': ...}]})` warns and resolves to `NativeTool`.

    Lock-in for the spec-loader legacy alias map: the renamed capability still works in
    YAML/JSON specs, with a `PydanticAIDeprecationWarning` pointing users to `NativeTool`.
    """
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r"capability name 'BuiltinTool' is deprecated, use 'NativeTool'",
    ):
        agent = Agent.from_spec(
            {'name': 'a', 'model': 'test', 'capabilities': [{'BuiltinTool': {'kind': 'web_search'}}]}
        )

    native_tools = list(agent._cap_native_tools)  # pyright: ignore[reportPrivateUsage]
    assert len(native_tools) == 1
    assert isinstance(native_tools[0], WebSearchTool)


def test_agent_from_spec_with_builtin_or_local_tool_capability_key_deprecated():
    """`Agent.from_spec({'capabilities': [{'BuiltinOrLocalTool': ...}]})` warns about the rename.

    `NativeOrLocalTool` is a base class not registered in `CAPABILITY_TYPES`, so resolution
    still fails with the usual "valid choices" error — but the deprecation warning fires
    first so the user knows the name was renamed.
    """
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r"capability name 'BuiltinOrLocalTool' is deprecated, use 'NativeOrLocalTool'",
    ):
        with pytest.raises(ValueError, match=r"'NativeOrLocalTool' is not in the provided"):
            Agent.from_spec(
                {
                    'name': 'a',
                    'model': 'test',
                    'capabilities': [{'BuiltinOrLocalTool': {'native': {'kind': 'web_search'}}}],
                }
            )
