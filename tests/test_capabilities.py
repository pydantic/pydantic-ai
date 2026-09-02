from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace
from traceback import extract_tb
from typing import Any, cast

import anyio
import pytest
from pydantic import BaseModel

from pydantic_ai import _agent_graph
from pydantic_ai._run_context import RunContext
from pydantic_ai._utils import Some
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.agent import Agent
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities import (
    Capability,
    CapabilityOrdering,
    DynamicCapability,
    PrefixTools,
    ResolveModelId,
    Thinking,
    ToolSearch,
    Toolset,
    WebFetch,
    WebSearch,
    WrapperCapability,
)
from pydantic_ai.capabilities._dynamic import ResolvedDynamicCapability
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    AgentNode,
    NodeResult,
    WrapModelRequestHandler,
    WrapNodeRunHandler,
)
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.capabilities.hooks import Hooks, HookTimeoutError
from pydantic_ai.exceptions import (
    AgentRunError,
    ModelRetry,
    UserError,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    KnownModelName,
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    ModelResolutionContext,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.output import OutputContext, ToolOutput
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings as _ModelSettings
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.toolsets._capability_owned import (
    resolve_capability_id,
    tool_defs_from_pre_definition_load_returns,
)
from pydantic_ai.toolsets._deferred_capability_loader import (
    LOAD_CAPABILITY_TOOL_NAME,
)
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_graph import End

from ._inline_snapshot import snapshot
from .capability_models import (
    CustomCapability,
    LoggingCapability,
    ToolsetFuncCapability,
    build_run_context as _build_run_context,
    make_text_response,
    registered_capability_context as _registered_capability_context,
    simple_model_function,
    simple_stream_function,
    tool_calling_model,
)
from .conftest import IsDatetime, IsStr, iter_message_parts, message

_SEARCH_TOOLS_NAME = ToolSearch.function_tool_name

pytestmark = [
    pytest.mark.anyio,
]


async def test_runtime_capability_contributions_applied():
    """Run-time `capabilities=` contributions (tools, instructions, etc.) must be applied.

    Regression guard: the `source_cap` selection previously only checked for `override()`
    or spec capabilities, so tool contributions from a capability passed only via
    `Agent.run(capabilities=[...])` were silently dropped.
    """
    agent = Agent(TestModel())
    result = await agent.run('Greet Alice', capabilities=[ToolsetFuncCapability()])

    tool_calls = list(iter_message_parts(result.all_messages(), ModelResponse, ToolCallPart))
    assert [c.tool_name for c in tool_calls] == ['greet']


async def test_capability_returning_toolset_func_combined():
    """Test that a ToolsetFunc capability works alongside other capabilities via CombinedCapability."""
    agent = Agent(
        TestModel(),
        instructions='You are a helpful greeter.',
        capabilities=[
            ToolsetFuncCapability(),
        ],
    )
    result = await agent.run('Greet Bob')

    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith('Hello, ')


def test_abstract_capability_get_model_settings_default():
    """AbstractCapability.get_model_settings() returns None by default."""

    @dataclass
    class PlainCap(AbstractCapability):
        pass

    cap = PlainCap()
    assert cap.get_model_settings() is None
    assert cap.get_description() is None


async def test_abstract_capability_description_field_is_optional_in_deferred_catalog() -> None:
    """Deferred capability catalog entries can include a description but do not require one."""

    @dataclass
    class AccountSecurityRunbook(AbstractCapability):
        id: str | None = 'account-security'
        description: str | None = 'Use for suspicious logins, account takeover, or session revocation.'
        defer_loading: bool = True

    @dataclass
    class RefundsRunbook(AbstractCapability):
        id: str | None = 'refunds'
        defer_loading: bool = True

    def model_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[AccountSecurityRunbook(), RefundsRunbook()])
    result = await agent.run('hi')
    request = next(message for message in result.all_messages() if isinstance(message, ModelRequest))

    assert request.instructions == snapshot(
        """\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- account-security: Use for suspicious logins, account takeover, or session revocation.
- refunds\
"""
    )


# --- for_run tests ---


def test_resolve_capability_id_scans_run_context_capabilities() -> None:
    @dataclass
    class SimpleCap(AbstractCapability):
        pass

    target = SimpleCap()
    other = SimpleCap()
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        capabilities={'other': other, 'target': target},
    )

    assert resolve_capability_id(ctx, target) == 'target'


async def test_capability_for_run_default_returns_self():
    """Default for_run returns self."""

    @dataclass
    class SimpleCap(AbstractCapability):
        pass

    cap = SimpleCap()
    ctx = _build_run_context()
    assert await cap.for_run(ctx) is cap


async def test_run_context_available_tool_names_empty_before_tool_manager_is_ready() -> None:
    """Early capability hooks can ask for available tool names before the tool manager is populated."""
    seen_available_tool_names: list[set[str]] = []
    seen_tools: list[dict[str, ToolDefinition]] = []

    @dataclass
    class AvailableToolsCap(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            seen_available_tool_names.append(ctx.available_tool_names)
            seen_tools.append(ctx.tools)

    agent = Agent(TestModel(), capabilities=[AvailableToolsCap()])
    await agent.run('hello')

    assert seen_available_tool_names == [set()]
    # The `tools` empty-guard mirrors `available_tool_names`: no tool manager yet → empty dict.
    assert seen_tools == [{}]


def test_run_context_available_tool_names_includes_discovered_before_tool_manager() -> None:
    ctx = _build_run_context()
    ctx.discovered_tool_names = {'discovered_tool'}

    assert ctx.tools == {}
    assert ctx.available_tool_names == {'discovered_tool'}
    assert ctx.is_tool_available('discovered_tool')
    assert not ctx.is_tool_available('unknown_tool')


def test_run_context_is_tool_available_falls_back_while_tools_unresolved() -> None:
    """Mid-`get_tools` the manager exists but its tool set is `None`; the name form must take
    the same history fallback as `available_tool_names` instead of reporting `False`."""
    ctx = _build_run_context()
    ctx.tool_manager = ToolManager(FunctionToolset())
    ctx.discovered_tool_names = {'discovered_tool'}

    assert ctx.tool_manager.tools is None
    assert ctx.available_tool_names == {'discovered_tool'}
    assert ctx.is_tool_available('discovered_tool')
    assert not ctx.is_tool_available('unknown_tool')


async def test_run_context_available_tool_names_unions_discovered_current_tools() -> None:
    """Available tool names are always-visible current tools plus revealed corpus tools.

    `loaded_capability_tool` counts as revealed on the strength of its capability's load alone:
    `is_gated_by_deferred_capability` keeps every tool of a deferred capability out of the search
    corpus, so the load is the only thing that can ever disclose it, and requiring a separate reveal
    marker would strand it for good once history processing dropped one. `pending_tool` is the
    contrast — search-gated but unowned, so it still has to be searched for.
    """
    toolset = FunctionToolset()

    @toolset.tool_plain
    def always_tool() -> str:  # pragma: no cover
        return 'always'

    @toolset.tool_plain(defer_loading=True)
    def discovered_tool() -> str:  # pragma: no cover
        return 'discovered'

    @toolset.tool_plain(defer_loading=True)
    def pending_tool() -> str:  # pragma: no cover
        return 'pending'

    @toolset.tool_plain(defer_loading=True)
    def loaded_capability_tool() -> str:  # pragma: no cover
        return 'loaded'

    ctx = _build_run_context()
    ctx.capabilities = {
        'loaded_capability': Capability(id='loaded_capability', defer_loading=True),
    }
    ctx.discovered_tool_names = {'discovered_tool', 'removed_tool'}
    ctx.loaded_capability_ids = {'loaded_capability'}
    tools = await toolset.get_tools(ctx)
    tools['discovered_tool'] = replace(
        tools['discovered_tool'],
        tool_def=replace(tools['discovered_tool'].tool_def, with_native=ToolSearchTool.kind),
    )
    tools['pending_tool'] = replace(
        tools['pending_tool'],
        tool_def=replace(tools['pending_tool'].tool_def, with_native=ToolSearchTool.kind, defer_loading=True),
    )
    tools['loaded_capability_tool'] = replace(
        tools['loaded_capability_tool'],
        tool_def=replace(
            tools['loaded_capability_tool'].tool_def,
            with_native=ToolSearchTool.kind,
            capability_id='loaded_capability',
        ),
    )
    tool_manager = ToolManager(toolset=toolset, ctx=ctx, tools=tools)
    ctx.tool_manager = tool_manager

    assert ctx.available_tool_names == {'always_tool', 'discovered_tool', 'loaded_capability_tool'}


async def test_run_context_is_tool_available() -> None:
    """Exercise the predicate directly across every reveal path and both argument forms.

    Covers always-visible, history-revealed, still-hidden, and unknown-name
    outcomes for both the `str` and `ToolDefinition` forms; the end-to-end fold and stale-resume
    scenarios are covered by the integration tests below.
    """
    toolset = FunctionToolset()

    @toolset.tool_plain
    def plain_tool() -> str:  # pragma: no cover
        return 'plain'

    @toolset.tool_plain(defer_loading=True)
    def discovered_tool() -> str:  # pragma: no cover
        return 'discovered'

    @toolset.tool_plain(defer_loading=True)
    def pending_tool() -> str:  # pragma: no cover
        return 'pending'

    @toolset.tool_plain(defer_loading=True)
    def loaded_tool() -> str:  # pragma: no cover
        return 'loaded'

    @toolset.tool_plain(defer_loading=True)
    def unloaded_tool() -> str:  # pragma: no cover
        return 'unloaded'

    ctx = _build_run_context()
    ctx.capabilities = {
        'loaded': Capability(id='loaded', defer_loading=True),
        'unloaded': Capability(id='unloaded', defer_loading=True),
    }
    ctx.loaded_capability_ids = {'loaded'}
    ctx.discovered_tool_names = {'discovered_tool', 'loaded_tool'}
    tools = await toolset.get_tools(ctx)
    for name in ('discovered_tool', 'pending_tool', 'loaded_tool', 'unloaded_tool'):
        tools[name] = replace(
            tools[name],
            tool_def=replace(tools[name].tool_def, with_native=ToolSearchTool.kind),
        )
    tools['loaded_tool'] = replace(
        tools['loaded_tool'],
        tool_def=replace(tools['loaded_tool'].tool_def, capability_id='loaded'),
    )
    tools['unloaded_tool'] = replace(
        tools['unloaded_tool'],
        tool_def=replace(tools['unloaded_tool'].tool_def, capability_id='unloaded'),
    )
    ctx.tool_manager = ToolManager(toolset=toolset, ctx=ctx, tools=tools)

    assert ctx.is_tool_available('plain_tool')
    assert ctx.is_tool_available(tools['plain_tool'].tool_def)
    assert ctx.is_tool_available('discovered_tool')
    assert ctx.is_tool_available(tools['loaded_tool'].tool_def)
    assert not ctx.is_tool_available('pending_tool')
    assert not ctx.is_tool_available(tools['unloaded_tool'].tool_def)
    assert not ctx.is_tool_available('unknown_tool')


def test_stale_loaded_eager_capability_is_not_revealed() -> None:
    ctx = _build_run_context()
    ctx.capabilities = {'refunds': Capability(id='refunds')}
    ctx.loaded_capability_ids = {'refunds'}
    tool_def = ToolDefinition(
        name='lookup_refund',
        description='Look up a refund.',
        parameters_json_schema={'type': 'object', 'properties': {}},
        capability_id='refunds',
    )

    assert ctx.is_tool_available(tool_def)
    assert tool_defs_from_pre_definition_load_returns(ctx, [tool_def]) == {}


async def test_is_tool_available_definition_survives_aggregator_fold() -> None:
    """A caller-held definition stays available after an aggregator removes it from resolved tools."""
    capability_tools = FunctionToolset()

    @capability_tools.tool_plain
    def lookup_refund() -> str:  # pragma: no cover
        return 'refund available'

    @dataclass
    class FoldingToolset(WrapperToolset[Any]):
        availability: list[bool] = field(default_factory=list[bool])

        async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
            tools = await self.wrapped.get_tools(ctx)
            available = ctx.is_tool_available(tools['lookup_refund'].tool_def)
            self.availability.append(available)
            if available:
                tools = {name: value for name, value in tools.items() if name != 'lookup_refund'}
            return tools

    folding_toolset: FoldingToolset | None = None

    @dataclass
    class FoldAvailableTools(AbstractCapability[Any]):
        def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any]:
            nonlocal folding_toolset
            folding_toolset = FoldingToolset(toolset)
            return folding_toolset

    refunds = Capability[object](
        id='refunds', description='Refund tools.', toolsets=[capability_tools], defer_loading=True
    )

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )
        if not any(part.tool_name == 'ping' for part in tool_returns):
            return ModelResponse(parts=[ToolCallPart(tool_name='ping', args={}, tool_call_id='ping')])
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds, FoldAvailableTools()])

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    result = await agent.run('Load refunds, then ping.')

    assert result.output == 'done'
    assert folding_toolset is not None
    assert folding_toolset.availability == [False, True, True]


async def test_stale_loaded_eager_capability_tool_stays_hidden() -> None:
    """Resumed loaded state does not reveal a tool owned by a capability that is now eager."""
    toolset = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def searchable_tool() -> str:  # pragma: no cover
        return 'found'

    capability = Capability[object](id='x', toolsets=[toolset])
    visibility: list[tuple[bool, set[str]]] = []

    @dataclass
    class CaptureVisibility(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            visibility.append((ctx.is_tool_available('searchable_tool'), ctx.available_tool_names))
            return request_context

    revealed_names: list[set[str]] = []

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        revealed_names.append(info.model_request_parameters.revealed_tool_names)
        return make_text_response('done')

    history = [
        ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'x'}, tool_call_id='load-x')]),
        ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='load-x')]),
    ]
    agent = Agent(FunctionModel(model_fn), capabilities=[capability, CaptureVisibility()])
    await agent.run('Resume.', message_history=history)
    discovered_history = [
        *history,
        ModelResponse(parts=[ToolSearchCallPart(args={'queries': ['searchable']}, tool_call_id='search-searchable')]),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'searchable_tool'}]},
                    tool_call_id='search-searchable',
                )
            ]
        ),
    ]
    await agent.run('Resume after discovery.', message_history=discovered_history)

    [(is_available, available_names), (is_discovered, discovered_names)] = visibility
    assert not is_available
    assert 'searchable_tool' not in available_names
    assert is_discovered
    assert 'searchable_tool' in discovered_names
    assert revealed_names == [set(), {'searchable_tool'}]


_DEFERRED_HOOK_NAMES = {
    'prepare_output_tools',
    'wrap_run_event_stream',
    'on_model_request_error',
    'on_tool_validate_error',
    'on_tool_execute_error',
    'before_output_validate',
    'after_output_validate',
    'wrap_output_validate',
    'on_output_validate_error',
    'on_output_process_error',
    'handle_deferred_tool_calls',
}


@dataclass
class _FailIfDispatchedDeferredCap(AbstractCapability):
    id: str | None = 'deferred'
    defer_loading: bool = True

    def __getattribute__(self, name: str) -> Any:
        if name in _DEFERRED_HOOK_NAMES:  # pragma: no cover
            raise AssertionError(f'unloaded capability hook should be skipped: {name}')
        return super().__getattribute__(name)


@dataclass
class _NoopCap(AbstractCapability):
    pass


@dataclass
class _NodeModelHookCap(AbstractCapability[Any]):
    log: list[str] = field(default_factory=lambda: [])

    async def wrap_node_run(
        self, ctx: RunContext[Any], *, node: AgentNode[Any], handler: WrapNodeRunHandler[Any]
    ) -> NodeResult[Any]:
        self.log.append('wrap_node_run')
        return await handler(node)

    async def on_node_run_error(
        self, ctx: RunContext[Any], *, node: AgentNode[Any], error: Exception
    ) -> NodeResult[Any]:
        self.log.append('on_node_run_error')
        raise error

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        self.log.append('wrap_model_request')
        return await handler(request_context)

    async def on_model_request_error(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        self.log.append('on_model_request_error')
        raise error


async def test_default_node_and_model_hooks_remain_directly_callable() -> None:
    ctx = _build_run_context()
    node = _agent_graph.UserPromptNode[Any, Any](user_prompt='test')
    request_context = ModelRequestContext(
        model=TestModel(),
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    response = ModelResponse(parts=[])

    async def node_handler(node: AgentNode[Any]) -> NodeResult[Any]:
        return node

    async def model_handler(_request_context: ModelRequestContext) -> ModelResponse:
        return response

    for capability in (_NoopCap(), Hooks()):
        assert await capability.wrap_node_run(ctx, node=node, handler=node_handler) is node
        assert (
            await capability.wrap_model_request(ctx, request_context=request_context, handler=model_handler) is response
        )

    error = RuntimeError('provider failure')
    with pytest.raises(RuntimeError, match='provider failure') as node_exc_info:
        await _NoopCap().on_node_run_error(ctx, node=node, error=error)
    assert node_exc_info.value is error

    with pytest.raises(RuntimeError, match='provider failure') as model_exc_info:
        await _NoopCap().on_model_request_error(ctx, request_context=request_context, error=error)
    assert model_exc_info.value is error


async def test_inherited_noop_capability_hooks_are_absent_from_traceback() -> None:
    before_run_called = False

    async def before_run(_ctx: RunContext[Any]) -> None:
        nonlocal before_run_called
        before_run_called = True

    async def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise RuntimeError('provider failure')

    agent = Agent(
        FunctionModel(fail),
        capabilities=[_NoopCap(), WrapperCapability(wrapped=_NoopCap()), Hooks(before_run=before_run)],
    )

    with pytest.raises(RuntimeError, match='provider failure') as exc_info:
        await agent.run('test')

    frames = extract_tb(exc_info.value.__traceback__)
    noop_hook_names = {'wrap_node_run', 'on_node_run_error', 'wrap_model_request', 'on_model_request_error'}
    assert before_run_called
    assert not any(
        frame.filename.endswith(
            (
                'pydantic_ai/capabilities/abstract.py',
                'pydantic_ai/capabilities/combined.py',
                'pydantic_ai/capabilities/hooks.py',
                'pydantic_ai/capabilities/wrapper.py',
            )
        )
        and frame.name in noop_hook_names
        for frame in frames
    )


async def test_implemented_nested_capability_hooks_are_preserved() -> None:
    async def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise RuntimeError('provider failure')

    capability = _NodeModelHookCap()
    wrapped = WrapperCapability(wrapped=CombinedCapability([capability]))

    with pytest.raises(RuntimeError, match='provider failure'):
        await Agent(FunctionModel(fail), capabilities=[wrapped]).run('test')

    assert capability.log == [
        'wrap_node_run',
        'wrap_node_run',
        'wrap_model_request',
        'on_model_request_error',
        'on_node_run_error',
    ]


async def test_unloaded_deferred_error_hooks_are_skipped() -> None:
    async def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise RuntimeError('provider failure')

    capability = _NodeModelHookCap(id='deferred', defer_loading=True)

    with pytest.raises(RuntimeError, match='provider failure'):
        await Agent(FunctionModel(fail), capabilities=[capability]).run('test')

    assert capability.log == []


async def test_hooks_subclass_overrides_are_not_skipped() -> None:
    """A Hooks subclass that overrides wrap/error methods still runs them.

    Registry-only `_has_*` checks would skip the override and call the model handler
    directly.
    """

    class RecoveringHooks(Hooks):
        async def wrap_model_request(
            self,
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            handler: WrapModelRequestHandler,
        ) -> ModelResponse:
            try:
                return await handler(request_context)
            except RuntimeError:
                return ModelResponse(parts=[TextPart(content='hooks-wrapped-recovery')])

    class RecoveringErrorHooks(Hooks):
        async def on_model_request_error(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='hooks-error-recovery')])

    async def fail(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise RuntimeError('provider failure')

    wrapped = await Agent(FunctionModel(fail), capabilities=[RecoveringHooks()]).run('test')
    assert wrapped.output == 'hooks-wrapped-recovery'

    recovered = await Agent(FunctionModel(fail), capabilities=[RecoveringErrorHooks()]).run('test')
    assert recovered.output == 'hooks-error-recovery'


def _output_context() -> OutputContext:
    return OutputContext(mode='text', output_type=str, object_def=None, has_function=False)


async def _empty_event_stream() -> AsyncIterator[AgentStreamEvent]:
    if False:  # pragma: no cover
        yield cast(AgentStreamEvent, None)


async def _validate_output(output: str | dict[str, Any]) -> Any:
    return output


async def test_combined_capability_skips_unloaded_deferred_forward_hooks() -> None:
    """Forward-order hook dispatch skips unloaded deferred capabilities."""
    combined = CombinedCapability([_FailIfDispatchedDeferredCap(), _NoopCap()])
    ctx = _build_run_context()
    output_context = _output_context()
    tool_def = ToolDefinition(name='tool')

    assert await combined.prepare_output_tools(ctx, [tool_def]) == [tool_def]
    assert await combined.before_output_validate(ctx, output_context=output_context, output='raw') == 'raw'
    assert (
        await combined.handle_deferred_tool_calls(
            ctx, requests=DeferredToolRequests(calls=[ToolCallPart('tool', {}, tool_call_id='deferred-call')])
        )
        is None
    )


async def test_combined_capability_skips_unloaded_deferred_reverse_hooks() -> None:
    """Reverse-order hook dispatch skips unloaded deferred capabilities."""
    combined = CombinedCapability([_NoopCap(), _FailIfDispatchedDeferredCap()])
    ctx = _build_run_context()
    output_context = _output_context()
    tool_def = ToolDefinition(name='tool')
    call = ToolCallPart('tool', {}, tool_call_id='tool-call')
    request_context = ModelRequestContext(
        model=TestModel(),
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    assert [event async for event in combined.wrap_run_event_stream(ctx, stream=_empty_event_stream())] == []
    assert await combined.after_output_validate(ctx, output_context=output_context, output='parsed') == 'parsed'
    assert (
        await combined.wrap_output_validate(ctx, output_context=output_context, output='raw', handler=_validate_output)
        == 'raw'
    )

    with pytest.raises(RuntimeError, match='model'):
        await combined.on_model_request_error(ctx, request_context=request_context, error=RuntimeError('model'))
    with pytest.raises(ModelRetry, match='tool validate'):
        await combined.on_tool_validate_error(
            ctx, call=call, tool_def=tool_def, args={}, error=ModelRetry('tool validate')
        )
    with pytest.raises(RuntimeError, match='tool execute'):
        await combined.on_tool_execute_error(
            ctx, call=call, tool_def=tool_def, args={}, error=RuntimeError('tool execute')
        )
    with pytest.raises(ModelRetry, match='output validate'):
        await combined.on_output_validate_error(
            ctx, output_context=output_context, output='raw', error=ModelRetry('output validate')
        )
    with pytest.raises(RuntimeError, match='output process'):
        await combined.on_output_process_error(
            ctx, output_context=output_context, output='parsed', error=RuntimeError('output process')
        )


async def test_combined_capability_for_run_propagates():
    """CombinedCapability propagates for_run to children."""

    @dataclass
    class SimpleCap(AbstractCapability):
        label: str = ''

    cap1 = SimpleCap(label='a')
    cap2 = SimpleCap(label='b')
    combined = CombinedCapability([cap1, cap2])
    ctx = _build_run_context()

    # No child changes → returns self
    result = await combined.for_run(ctx)
    assert result is combined


async def test_combined_capability_for_run_returns_new_when_child_changes():
    """CombinedCapability returns new instance when a child's for_run returns different."""

    @dataclass
    class PerRunCap(AbstractCapability):
        run_id: int = 0

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return PerRunCap(run_id=self.run_id + 1)

    @dataclass
    class StaticCap(AbstractCapability):
        pass

    static_cap = StaticCap()
    per_run_cap = PerRunCap()
    combined = CombinedCapability([static_cap, per_run_cap])
    ctx = _build_run_context()

    result = await combined.for_run(ctx)
    assert result is not combined
    assert isinstance(result, CombinedCapability)
    assert result.capabilities[0] is static_cap  # unchanged
    new_per_run = result.capabilities[1]
    assert isinstance(new_per_run, PerRunCap)
    assert new_per_run.run_id == 1


async def test_combined_capability_for_run_cancels_siblings_on_failure():
    """When one child's for_run fails, siblings are cancelled instead of leaking as orphan tasks."""
    sibling_completed = False

    @dataclass
    class FailingCap(AbstractCapability):
        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            raise RuntimeError('boom')

    @dataclass
    class SlowCap(AbstractCapability):
        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            nonlocal sibling_completed
            await anyio.sleep(0.1)
            sibling_completed = True  # pragma: no cover
            return self  # pragma: no cover

    combined = CombinedCapability([FailingCap(), SlowCap()])
    ctx = _build_run_context()

    with pytest.raises(RuntimeError, match='boom'):
        await combined.for_run(ctx)

    await anyio.sleep(0.2)
    assert sibling_completed is False


def test_apply_single_capability():
    """AbstractCapability.apply() visits just the capability itself."""

    @dataclass
    class MyCap(AbstractCapability):
        pass

    cap = MyCap()
    visited: list[AbstractCapability] = []
    cap.apply(visited.append)
    assert visited == [cap]


def test_apply_combined_capability():
    """CombinedCapability.apply() recursively visits all leaf capabilities."""

    @dataclass
    class CapA(AbstractCapability):
        pass

    @dataclass
    class CapB(AbstractCapability):
        pass

    cap_a = CapA()
    cap_b = CapB()
    combined = CombinedCapability([cap_a, cap_b])

    visited: list[AbstractCapability] = []
    combined.apply(visited.append)
    assert visited == [cap_a, cap_b]


def test_apply_nested_combined_capability():
    """CombinedCapability.apply() flattens nested CombinedCapabilities."""

    @dataclass
    class CapA(AbstractCapability):
        pass

    @dataclass
    class CapB(AbstractCapability):
        pass

    @dataclass
    class CapC(AbstractCapability):
        pass

    cap_a = CapA()
    cap_b = CapB()
    cap_c = CapC()
    inner = CombinedCapability([cap_a, cap_b])
    outer = CombinedCapability([inner, cap_c])

    visited: list[AbstractCapability] = []
    outer.apply(visited.append)
    assert visited == [cap_a, cap_b, cap_c]


def test_apply_wrapper_capability():
    """WrapperCapability.apply() visits the wrapper registered for the wrapped capability."""
    inner = Thinking()
    wrapper = WrapperCapability(wrapped=inner)

    visited: list[AbstractCapability] = []
    wrapper.apply(visited.append)
    assert visited == [wrapper]


def test_apply_wrapper_over_combined_capability():
    """WrapperCapability.apply() also visits children when the wrapped capability is a container."""

    @dataclass
    class CapA(AbstractCapability):
        pass

    @dataclass
    class CapB(AbstractCapability):
        pass

    cap_a = CapA()
    cap_b = CapB()
    wrapper = WrapperCapability(wrapped=CombinedCapability([cap_a, cap_b]))

    visited: list[AbstractCapability] = []
    wrapper.apply(visited.append)
    assert visited == [wrapper, cap_a, cap_b]


async def test_wrapper_over_combined_capability_registers_child_tool_owners():
    """Child-owned toolsets still resolve capability ids when a wrapper contains a CombinedCapability."""
    toolset_a = FunctionToolset()

    @toolset_a.tool_plain
    def tool_a() -> str:
        return 'a'  # pragma: no cover

    toolset_b = FunctionToolset()

    @toolset_b.tool_plain
    def tool_b() -> str:
        return 'b'  # pragma: no cover

    wrapper = WrapperCapability(
        wrapped=CombinedCapability(
            [
                Toolset(toolset_a, id='a'),
                Toolset(toolset_b, id='b'),
            ]
        )
    )
    seen_capability_ids: list[str] = []

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for tool in info.function_tools:
            assert tool.capability_id is not None
            seen_capability_ids.append(tool.capability_id)
        return ModelResponse(parts=[TextPart(','.join(sorted(tool.name for tool in info.function_tools)))])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    result = await agent.run('list tools')

    assert result.output == 'tool_a,tool_b'
    assert sorted(seen_capability_ids) == ['a', 'b']


def test_apply_prefix_tools():
    """PrefixTools.apply() visits the wrapper registered for the wrapped capability."""
    thinking = Thinking()
    prefixed = PrefixTools(wrapped=thinking, prefix='ns')

    visited: list[AbstractCapability] = []
    prefixed.apply(visited.append)
    assert visited == [prefixed]


def test_apply_finds_capability_by_type():
    """Realistic usage: use apply() to check if a specific capability type is present."""
    thinking = Thinking()
    web_search = WebSearch(local='duckduckgo')
    combined = CombinedCapability([thinking, web_search])

    visited: list[AbstractCapability] = []
    combined.apply(visited.append)

    assert any(isinstance(c, Thinking) for c in visited)
    assert any(isinstance(c, WebSearch) for c in visited)
    assert not any(isinstance(c, WebFetch) for c in visited)


def test_apply_finds_wrapped_capability_by_type():
    """apply() registers wrappers themselves because wrapper behavior affects the loaded capability."""
    thinking = Thinking()
    prefixed = PrefixTools(wrapped=thinking, prefix='ns')
    combined = CombinedCapability([prefixed, WebSearch(local='duckduckgo')])

    visited: list[AbstractCapability] = []
    combined.apply(visited.append)

    assert not any(isinstance(c, Thinking) for c in visited)
    assert any(isinstance(c, WebSearch) for c in visited)
    assert any(isinstance(c, PrefixTools) for c in visited)


def test_apply_empty_combined():
    """CombinedCapability with no children visits nothing."""
    combined = CombinedCapability([])
    visited: list[AbstractCapability] = []
    combined.apply(visited.append)
    assert visited == []


async def test_for_run_with_different_toolset():
    """When for_run returns a capability with a different get_toolset(), the per-run toolset is used."""
    toolset_a = FunctionToolset(id='a')

    @toolset_a.tool_plain
    def tool_a() -> str:
        return 'a'  # pragma: no cover

    toolset_b = FunctionToolset(id='b')

    @toolset_b.tool_plain
    def tool_b() -> str:
        return 'b'  # pragma: no cover

    @dataclass
    class SwitchingCap(AbstractCapability):
        use_b: bool = False

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return SwitchingCap(use_b=True)

        def get_toolset(self) -> AbstractToolset:
            return toolset_b if self.use_b else toolset_a

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Check which tools are available
        tool_names = [t.name for t in info.function_tools]
        return ModelResponse(parts=[TextPart(f'tools: {",".join(sorted(tool_names))}')])

    agent = Agent(FunctionModel(respond), capabilities=[SwitchingCap()])

    # At run time, for_run switches to toolset_b
    result = await agent.run('Hello')
    assert 'tool_b' in result.output


async def test_for_run_with_different_instructions():
    """When for_run returns a capability with different get_instructions(), per-run instructions are used."""

    @dataclass
    class DynamicInstructionsCap(AbstractCapability):
        run_instructions: str = 'init-time'

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return DynamicInstructionsCap(run_instructions='per-run')

        def get_instructions(self) -> str:
            return self.run_instructions

    captured_messages: list[ModelMessage] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_messages.extend(messages)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[DynamicInstructionsCap()])
    await agent.run('Hello')

    # The per-run instructions should appear in the request's instructions field
    instructions_found = [
        msg.instructions for msg in captured_messages if isinstance(msg, ModelRequest) and msg.instructions
    ]
    assert any('per-run' in i for i in instructions_found), (
        f'Expected per-run instructions in messages, got: {captured_messages}'
    )


async def test_for_run_receives_populated_run_context():
    """`for_run` hooks receive a `RunContext` with run_id, conversation_id, and resolved metadata."""

    captured: dict[str, Any] = {}

    class CapturingCap(AbstractCapability):
        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            captured['run_id'] = ctx.run_id
            captured['conversation_id'] = ctx.conversation_id
            captured['metadata'] = ctx.metadata
            captured['instrumentation_version'] = ctx.instrumentation_version
            return self

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    def metadata_factory(ctx: RunContext) -> dict[str, Any]:
        # Factory should be able to read run_id/conversation_id from the early ctx.
        return {'run_id_seen': ctx.run_id, 'conversation_id_seen': ctx.conversation_id}

    agent = Agent(FunctionModel(respond), capabilities=[CapturingCap()])

    await agent.run('Hello', conversation_id='conv-123', metadata=metadata_factory)

    assert captured['run_id'] is not None
    assert captured['conversation_id'] == 'conv-123'
    assert captured['metadata'] == {'run_id_seen': captured['run_id'], 'conversation_id_seen': 'conv-123'}
    assert captured['instrumentation_version'] is not None


async def test_concurrent_runs_capability_isolation():
    """Multiple concurrent runs don't share state on stateful capabilities."""

    @dataclass
    class CountingCap(AbstractCapability):
        request_count: int = 0

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return CountingCap()

        async def before_model_request(
            self,
            ctx: RunContext,
            request_context: ModelRequestContext,
        ) -> ModelRequestContext:
            self.request_count += 1
            assert self.request_count == 1, f'Expected 1, got {self.request_count} — state leaked between runs!'
            return request_context

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Done')])

    agent = Agent(FunctionModel(respond), capabilities=[CountingCap()])

    # Run two concurrent runs — each should get its own CountingCap with count=0
    results = await asyncio.gather(agent.run('A'), agent.run('B'))
    assert results[0].output == 'Done'
    assert results[1].output == 'Done'


@pytest.mark.parametrize(
    'forced_choice',
    [
        pytest.param('required', id='required'),
        pytest.param(['get_weather'], id='list'),
    ],
)
async def test_capability_can_inject_forcing_tool_choice_per_step(forced_choice: Any):
    """A capability returning a callable from get_model_settings() may inject `tool_choice='required'`
    or `list[str]` per step without tripping the agent.run baseline validator.

    Forces the tool on step 1, then steps aside so the agent can produce a final response.
    """

    class ForceFirstStep(AbstractCapability):
        def get_model_settings(self) -> Any:
            def settings(ctx: RunContext) -> _ModelSettings:
                tool_called = any(
                    isinstance(part, ToolReturnPart) and part.tool_name == 'get_weather'
                    for message in ctx.messages
                    if isinstance(message, ModelRequest)
                    for part in message.parts
                )
                if tool_called:
                    return _ModelSettings()
                return _ModelSettings(tool_choice=forced_choice)

            return settings

    seen_tool_choices: list[Any] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tool_choices.append((info.model_settings or {}).get('tool_choice'))
        if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
            return ModelResponse(parts=[TextPart(content='sunny')])
        return ModelResponse(parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Paris'})])

    agent = Agent(FunctionModel(respond), capabilities=[ForceFirstStep()])

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'Weather in {city}: sunny'

    result = await agent.run('Weather in Paris?')

    assert result.output == 'sunny'
    assert seen_tool_choices == [forced_choice, None]


# --- Hooks capability tests ---


class TestHooksCapability:
    """Tests for the Hooks decorator-based capability."""

    async def test_decorator_registration(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def log_request(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('before_model_request')
            return request_context

        @hooks.on.after_model_request
        async def log_response(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            call_log.append('after_model_request')
            return response

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['before_model_request', 'after_model_request']

    async def test_constructor_form(self):
        call_log: list[str] = []

        async def log_request(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('before_model_request')
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[Hooks(before_model_request=log_request)])
        await agent.run('hello')
        assert call_log == ['before_model_request']

    async def test_multiple_hooks_same_event(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def first(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('first')
            return request_context

        @hooks.on.before_model_request
        async def second(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('second')
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['first', 'second']

    async def test_tool_names_filtering(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_tool_execute(tools=['target_tool'])
        async def filtered(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
        ) -> dict[str, Any]:
            call_log.append(f'filtered:{call.tool_name}')
            return args

        @hooks.on.after_tool_execute
        async def unfiltered(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any], result: Any
        ) -> Any:
            call_log.append(f'unfiltered:{call.tool_name}')
            return result

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def target_tool() -> str:
            return 'result'

        await agent.run('call tool')
        assert 'filtered:target_tool' in call_log
        assert 'unfiltered:target_tool' in call_log

    async def test_wrap_model_request(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.model_request
        async def wrap(ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: Any) -> ModelResponse:
            call_log.append('wrap_start')
            result = await handler(request_context)
            call_log.append('wrap_end')
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['wrap_start', 'wrap_end']

    async def test_wrap_run(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.run
        async def wrap(ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
            call_log.append('wrap_run_start')
            result = await handler()
            call_log.append('wrap_run_end')
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['wrap_run_start', 'wrap_run_end']

    async def test_on_error_recovery(self):
        hooks = Hooks()

        @hooks.on.model_request_error
        async def recover(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='recovered')])

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == 'recovered'

    async def test_sync_function_auto_wrapping(self):
        hooks = Hooks()
        call_log: list[str] = []
        hook_thread_ids: list[int] = []

        @hooks.on.before_model_request
        def sync_hook(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('sync_hook')
            hook_thread_ids.append(threading.get_ident())
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['sync_hook']
        # The sync hook runs in a thread, so it can't block the event loop.
        assert hook_thread_ids[0] != threading.get_ident()

    async def test_sync_function_returning_awaitable(self):
        hooks = Hooks()
        call_log: list[str] = []

        async def log_request(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('log_request')
            return request_context

        @hooks.on.before_model_request
        def sync_hook(ctx: RunContext[Any], request_context: ModelRequestContext) -> Awaitable[ModelRequestContext]:
            return log_request(ctx, request_context)

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['log_request']

    async def test_timeout(self):
        hooks = Hooks()

        @hooks.on.before_model_request(timeout=0.01)
        async def slow_hook(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            await asyncio.sleep(10)
            return request_context  # pragma: no cover

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        with pytest.raises(HookTimeoutError) as exc_info:
            await agent.run('hello')
        assert exc_info.value.hook_name == 'before_model_request'
        assert exc_info.value.func_name == 'slow_hook'
        assert exc_info.value.timeout == 0.01
        assert isinstance(exc_info.value, AgentRunError)
        assert isinstance(exc_info.value, TimeoutError)

    async def test_timeout_sync_hook(self):
        """A sync hook runs in a worker thread, which is abandoned when its deadline expires."""
        hooks = Hooks()

        @hooks.on.before_model_request(timeout=0.01)
        def slow_sync_hook(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            time.sleep(0.1)
            # The abandoned thread runs to completion, so this line is covered; only its result is discarded.
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        with pytest.raises(HookTimeoutError) as exc_info:
            await agent.run('hello')
        assert exc_info.value.hook_name == 'before_model_request'
        assert exc_info.value.func_name == 'slow_sync_hook'

    async def test_has_wrap_node_run(self):
        hooks = Hooks()
        with pytest.warns(PydanticAIDeprecationWarning, match=r'`has_wrap_node_run`.*`wrap_node_run`'):
            assert hooks.has_wrap_node_run is False  # type: ignore[reportDeprecated]

        nodes_seen: list[str] = []

        @hooks.on.node_run
        async def wrap(ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
            nodes_seen.append(type(node).__name__)
            return await handler(node)

        with pytest.warns(PydanticAIDeprecationWarning, match=r'`has_wrap_node_run`.*`wrap_node_run`'):
            assert hooks.has_wrap_node_run is True  # type: ignore[reportDeprecated]

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert len(nodes_seen) > 0

    async def test_composition_with_other_capabilities(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def hooks_before(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('hooks_before')
            return request_context

        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks, cap])
        await agent.run('hello')
        assert 'hooks_before' in call_log
        assert 'before_model_request' in cap.log

    async def test_before_run(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_run
        async def on_start(ctx: RunContext[Any]) -> None:
            call_log.append('before_run')

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['before_run']

    async def test_after_run(self):
        hooks = Hooks()
        outputs: list[str] = []

        @hooks.on.after_run
        async def on_end(ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            outputs.append(result.output)
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        result = await agent.run('hello')
        assert outputs == [result.output]

    async def test_repr(self):
        hooks = Hooks()
        assert repr(hooks) == 'Hooks({})'

        @hooks.on.before_model_request
        async def hook(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            return request_context

        assert repr(hooks) == "Hooks({'before_model_request': 1})"

        # Verify the registered hook actually works
        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')

    async def test_on_model_request_error_reraise(self):
        """Error hooks that re-raise propagate the error to the caller."""

        hooks = Hooks()

        @hooks.on.model_request_error
        async def log_and_reraise(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            raise error

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')

    async def test_on_run_error_reraise(self):
        """on_run_error hooks that re-raise propagate the error."""

        hooks = Hooks()

        @hooks.on.run_error
        async def log_and_reraise(ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            raise error

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')

    async def test_on_run_error_recovery(self):
        hooks = Hooks()

        @hooks.on.run_error
        async def recover(ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            return AgentRunResult(output='recovered from run error')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == 'recovered from run error'

    async def test_on_run_error_chaining(self):
        hooks = Hooks()

        @hooks.on.run_error
        async def first_handler(ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            raise ValueError('transformed by first')

        @hooks.on.run_error
        async def second_handler(ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            return AgentRunResult(output=f'caught: {error}')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('original error')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run('hello')
        assert 'transformed by first' in result.output

    async def test_error_hook_chaining(self):
        hooks = Hooks()

        @hooks.on.model_request_error
        async def first(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            raise ValueError('transformed')

        @hooks.on.model_request_error
        async def second(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content=f'recovered: {error}')])

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('original')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run('hello')
        assert 'transformed' in result.output

    async def test_wrap_run_event_stream(self):
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.run_event_stream
        async def observe_stream(
            ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterable[AgentStreamEvent]:
            async for event in stream:
                events_seen.append(type(event).__name__)
                yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert len(events_seen) > 0

    async def test_hooks_with_streaming_run(self):
        """Hooks capability used during a streaming run exercises the default wrap_run_event_stream path."""

        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def log_request(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('before_model_request')
            return request_context

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert 'before_model_request' in call_log

    async def test_node_run_hooks(self):
        """Exercise before_node_run, after_node_run, and node_run (wrap) via .on namespace."""
        hooks = Hooks()
        nodes_seen: list[str] = []

        @hooks.on.before_node_run
        async def before(ctx: RunContext[Any], *, node: Any) -> Any:
            nodes_seen.append(f'before:{type(node).__name__}')
            return node

        @hooks.on.after_node_run
        async def after(ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
            nodes_seen.append(f'after:{type(node).__name__}')
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert any('before:' in n for n in nodes_seen)
        assert any('after:' in n for n in nodes_seen)

    async def test_node_run_error_hook(self):
        """on.node_run_error fires when a node fails."""
        hooks = Hooks()
        error_log: list[str] = []

        @hooks.on.node_run_error
        async def handle(ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
            error_log.append(f'error:{type(error).__name__}')
            raise error

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('node exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        with pytest.raises(RuntimeError, match='node exploded'):
            await agent.run('hello')
        assert any('error:RuntimeError' in e for e in error_log)

    async def test_on_event_hook(self):
        """on.event fires for each stream event and can modify events."""
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.event
        async def observe(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent:
            events_seen.append(type(event).__name__)
            return event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert len(events_seen) > 0

    async def test_on_event_hook_fires_in_run(self):
        """on.event fires in run() even without an event_stream_handler."""
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.event
        async def observe(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent:
            events_seen.append(type(event).__name__)
            return event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        result = await agent.run('hello')
        assert result.output is not None
        assert 'PartStartEvent' in events_seen

    async def test_wrap_run_event_stream_fires_in_run(self):
        """on.run_event_stream fires in run() even without an event_stream_handler."""
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.run_event_stream
        async def observe_stream(
            ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterable[AgentStreamEvent]:
            async for event in stream:
                events_seen.append(type(event).__name__)
                yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        result = await agent.run('hello')
        assert result.output is not None
        assert 'PartStartEvent' in events_seen

    async def test_on_event_with_run_event_stream(self):
        """on.event and on.run_event_stream can be used together."""
        hooks = Hooks()
        event_log: list[str] = []
        stream_log: list[str] = []

        @hooks.on.event
        async def per_event(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent:
            event_log.append(type(event).__name__)
            return event

        @hooks.on.run_event_stream
        async def wrap_stream(
            ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterable[AgentStreamEvent]:
            stream_log.append('started')
            async for event in stream:
                yield event
            stream_log.append('finished')

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert len(event_log) > 0
        assert stream_log == ['started', 'finished']

    async def test_prepare_tools_hook(self):
        """on.prepare_tools filters tool definitions."""
        hooks = Hooks()

        @hooks.on.prepare_tools
        async def hide_tools(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [td for td in tool_defs if not td.name.startswith('hidden_')]

        tool_called = False

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def visible_tool() -> str:
            nonlocal tool_called
            tool_called = True
            return 'visible'

        @agent.tool_plain
        def hidden_tool() -> str:
            return 'hidden'  # pragma: no cover

        await agent.run('call tool')
        assert tool_called

    async def test_prepare_output_tools_hook(self):
        """`on.prepare_output_tools` filters output tool definitions — model only sees the
        non-filtered ones."""
        hooks = Hooks()

        @hooks.on.prepare_output_tools
        async def hide_secret(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [td for td in tool_defs if td.name != 'secret_output']

        seen_output_tools: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            seen_output_tools.extend(td.name for td in info.output_tools)
            # Call the only remaining (non-filtered) output tool
            return ModelResponse(parts=[ToolCallPart('public_output', {'value': 'ok'})])

        class SecretOutput(BaseModel):
            value: str

        class PublicOutput(BaseModel):
            value: str

        agent = Agent(
            FunctionModel(model_fn),
            output_type=[
                ToolOutput(SecretOutput, name='secret_output'),
                ToolOutput(PublicOutput, name='public_output'),
            ],
            capabilities=[hooks],
        )
        result = await agent.run('hello')
        assert isinstance(result.output, PublicOutput)
        assert seen_output_tools == ['public_output']

    async def test_tool_validate_hooks(self):
        """Exercise before/after/wrap tool_validate and on_tool_validate_error."""
        hooks = Hooks()
        validate_log: list[str] = []

        @hooks.on.before_tool_validate
        async def before_validate(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any
        ) -> Any:
            validate_log.append('before_validate')
            return args

        @hooks.on.after_tool_validate
        async def after_validate(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
        ) -> dict[str, Any]:
            validate_log.append('after_validate')
            return args

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'

        await agent.run('call tool')
        assert 'before_validate' in validate_log
        assert 'after_validate' in validate_log

    async def test_wrap_tool_validate_hook(self):
        """Exercise on.tool_validate (wrap) via decorator."""
        hooks = Hooks()
        wrap_log: list[str] = []

        @hooks.on.tool_validate
        async def wrap_validate(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, handler: Any
        ) -> dict[str, Any]:
            wrap_log.append('wrap_start')
            result = await handler(args)
            wrap_log.append('wrap_end')
            return result

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'

        await agent.run('call tool')
        assert wrap_log == ['wrap_start', 'wrap_end']

    async def test_tool_validate_error_hook(self):
        """on.tool_validate_error can recover from validation failures."""
        hooks = Hooks()

        @hooks.on.tool_validate_error
        async def recover_validate(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
        ) -> dict[str, Any]:
            return {'name': 'recovered'}

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                tool = info.function_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"wrong": 1}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[hooks])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f'hello {name}'

        result = await agent.run('greet someone')
        assert 'hello recovered' in result.output

    async def test_wrap_tool_execute_hook(self):
        """Exercise on.tool_execute (wrap) via decorator."""
        hooks = Hooks()
        wrap_log: list[str] = []

        @hooks.on.tool_execute
        async def wrap_exec(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any], handler: Any
        ) -> Any:
            wrap_log.append('exec_start')
            result = await handler(args)
            wrap_log.append('exec_end')
            return result

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'

        await agent.run('call tool')
        assert wrap_log == ['exec_start', 'exec_end']

    async def test_tool_execute_error_hook(self):
        """on.tool_execute_error can recover from tool execution failures."""
        hooks = Hooks()

        @hooks.on.tool_execute_error
        async def recover_exec(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            error: Exception,
        ) -> Any:
            return 'fallback result'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        result = await agent.run('call tool')
        assert 'fallback result' in result.output

    async def test_tool_validate_error_reraise(self):
        """on.tool_validate_error that re-raises propagates the error."""
        hooks = Hooks()

        @hooks.on.tool_validate_error
        async def reraise(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
        ) -> dict[str, Any]:
            raise error

        call_count = 0

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                tool = info.function_tools[0]
                if call_count <= 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"wrong": 1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"name": "ok"}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[hooks])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f'hello {name}'

        await agent.run('greet someone')

    async def test_tool_execute_error_reraise(self):
        """on.tool_execute_error that re-raises propagates the error."""
        hooks = Hooks()

        @hooks.on.tool_execute_error
        async def reraise(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            error: Exception,
        ) -> Any:
            raise error

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        with pytest.raises(ValueError, match='tool failed'):
            await agent.run('call tool')

    async def test_get_serialization_name(self):
        assert Hooks.get_serialization_name() is None

    async def test_default_on_tool_execute_error_reraises(self):
        """The default on_tool_execute_error just re-raises, exercised with a minimal capability."""

        @dataclass
        class MinimalCap(AbstractCapability[Any]):
            """Capability that doesn't override error hooks."""

            def get_instructions(self):
                return 'Be helpful.'

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[MinimalCap()])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        with pytest.raises(ValueError, match='tool failed'):
            await agent.run('call the tool')


# --- Context var propagation tests ---

_test_cv: contextvars.ContextVar[str] = contextvars.ContextVar('_test_cv')


class TestContextVarPropagation:
    """Context vars set in wrap_run propagate to all hooks in the outer task."""

    async def test_wrap_run_contextvar_visible_in_node_hooks(self):
        """A capability that sets a contextvar in wrap_run should have it
        visible in another capability's node-level hooks via agent.run()."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('from-wrap-run')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(('before_node_run', _test_cv.get(None)))
                return node

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                self.seen.append(('wrap_node_run', _test_cv.get(None)))
                return await handler(node)

            async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
                self.seen.append(('after_node_run', _test_cv.get(None)))
                return result

            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                self.seen.append(('after_run', _test_cv.get(None)))
                return result

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[Setter(), reader])
        await agent.run('hello')

        for hook_name, value in reader.seen:
            assert value == 'from-wrap-run', f'{hook_name} did not see contextvar'

    async def test_wrap_run_contextvar_visible_via_iter_next(self):
        """Context vars set in wrap_run are visible when using agent.iter() + next()."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('from-iter')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(('before_node_run', _test_cv.get(None)))
                return node

            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                self.seen.append(('after_run', _test_cv.get(None)))
                return result

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[Setter(), reader])

        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)

        for hook_name, value in reader.seen:
            assert value == 'from-iter', f'{hook_name} did not see contextvar'

    async def test_contextvar_cleaned_up_after_run(self):
        """Context vars set in wrap_run are restored after the run completes."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('temporary')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        agent = Agent(TestModel(), capabilities=[Setter()])
        assert _test_cv.get(None) is None

        await agent.run('hello')

        # After the run, the contextvar should be cleaned up
        assert _test_cv.get(None) is None

    async def test_contextvar_cleaned_up_on_early_iter_exit(self):
        """Context vars are restored even when the caller exits iter() early."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('early-exit')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        agent = Agent(TestModel(), capabilities=[Setter()])
        assert _test_cv.get(None) is None

        async with agent.iter('hello') as agent_run:
            # Exit immediately without driving any nodes
            _ = agent_run.next_node

        # Context var must be cleaned up even though we abandoned the run
        assert _test_cv.get(None) is None

    async def test_before_run_contextvar_propagates(self):
        """Context vars set in before_run (not wrap_run) also propagate."""

        @dataclass
        class Setter(AbstractCapability):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                _test_cv.set('from-before-run')

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(('before_node_run', _test_cv.get(None)))
                return node

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[Setter(), reader])
        await agent.run('hello')

        for hook_name, value in reader.seen:
            assert value == 'from-before-run', f'{hook_name} did not see contextvar'

    async def test_sync_before_run_hook_contextvar_does_not_propagate(self):
        """Context vars set in a sync `before_run` hook do not propagate."""
        hooks = Hooks()

        @hooks.on.before_run
        def set_contextvar(ctx: RunContext[Any]) -> None:
            _test_cv.set('from-sync-hook')

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(('before_node_run', _test_cv.get(None)))
                return node

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[hooks, reader])
        await agent.run('hello')

        # Documented consequence of sync hooks running in a thread pool: the write lands in
        # the worker thread's copied context, so neither the run nor the caller ever sees it.
        assert reader.seen
        for hook_name, value in reader.seen:
            assert value is None, f'{hook_name} unexpectedly saw contextvar'
        assert _test_cv.get(None) is None

    async def test_contextvar_visible_in_on_run_error(self):
        """Context vars set in wrap_run are visible in on_run_error."""

        @dataclass
        class SetterWithRecovery(AbstractCapability):
            seen_in_error: str | None = None

            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('error-path')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

            async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
                self.seen_in_error = _test_cv.get(None)
                return AgentRunResult(output='recovered')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        cap = SetterWithRecovery()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        result = await agent.run('hello')

        assert result.output == 'recovered'
        assert cap.seen_in_error == 'error-path'


# --- WrapperCapability and PrefixTools tests ---


async def test_prefix_tools_prefixes_wrapped_capability_tools():
    """PrefixTools prefixes only the wrapped capability's tools, not other agent tools."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def inner_tool() -> str:
        return 'inner'  # pragma: no cover

    cap = PrefixTools(wrapped=Toolset(toolset), prefix='ns')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(','.join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[cap])

    @agent.tool_plain
    def outer_tool() -> str:
        return 'outer'  # pragma: no cover

    result = await agent.run('list tools')
    # inner_tool should be prefixed, outer_tool should not
    assert result.output == 'ns_inner_tool,outer_tool'


async def test_prefix_tools_from_spec():
    """PrefixTools from spec supports both dict-form and bare-name nested capabilities."""

    # Dict form (kwargs): nested capability with arguments
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {
                    'PrefixTools': {
                        'prefix': 'search',
                        'capability': {'NativeTool': {'kind': 'web_search'}},
                    }
                },
            ],
        },
    )
    assert agent.model is not None

    # Bare name form with custom_capability_types forwarded through contextvar
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {
                    'PrefixTools': {
                        'prefix': 'custom',
                        'capability': 'CustomCapability',
                    }
                },
            ],
        },
        custom_capability_types=[CustomCapability],
    )
    assert agent.model is not None


async def test_prefix_tools_from_spec_direct():
    """PrefixTools.from_spec works outside Agent.from_spec (no contextvar), using default registry."""
    cap = PrefixTools.from_spec(prefix='ws', capability={'WebSearch': {'local': 'duckduckgo'}})  # pyright: ignore[reportArgumentType]
    assert isinstance(cap, PrefixTools)
    assert cap.prefix == 'ws'


async def test_prefix_tools_returns_none_when_no_toolset():
    """PrefixTools.get_toolset() returns None if the wrapped capability has no toolset."""
    cap = PrefixTools(wrapped=CustomCapability(), prefix='ns')
    assert cap.get_toolset() is None


async def test_prefix_tools_with_callable_toolset():
    """PrefixTools handles a wrapped capability that returns a callable toolset."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def dynamic_tool() -> str:
        return 'dynamic'  # pragma: no cover

    def toolset_func(ctx: RunContext) -> FunctionToolset:
        return toolset

    cap = PrefixTools(wrapped=Toolset(toolset_func), prefix='dyn')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(','.join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[cap])
    result = await agent.run('list tools')
    assert result.output == 'dyn_dynamic_tool'


async def test_prefix_tools_inherits_wrapped_metadata_for_registration():
    """A wrapper with no id of its own delegates identity to the capability it wraps.

    This is what lets a wrapper sit over a deferred capability without losing its deferral or
    its place in the load catalog: the wrapper registers under the wrapped capability's id and
    keeps `defer_loading` and `description`.
    """
    toolset = FunctionToolset()
    wrapped = Toolset(
        toolset,
        id='leaf-tools',
        description='Leaf tool bundle.',
        defer_loading=True,
    )
    cap = PrefixTools(wrapped=wrapped, prefix='leaf')

    visited: list[AbstractCapability] = []
    cap.apply(visited.append)
    capability_map, available_ids = await _registered_capability_context(cap)

    assert cap.id == 'leaf-tools'
    assert cap.defer_loading is True
    assert cap.get_description() == 'Leaf tool bundle.'
    assert capability_map == {'leaf-tools': cap}
    # Deferred and not yet loaded, so it is registered but not available this turn.
    assert 'leaf-tools' not in available_ids
    assert visited == [cap]


async def test_prefix_tools_can_override_metadata():
    """A wrapper with explicit metadata becomes its own registered capability."""
    wrapped = Toolset(FunctionToolset(), id='leaf-tools', description='Leaf tool bundle.', defer_loading=True)
    cap = PrefixTools(
        wrapped=wrapped,
        prefix='leaf',
        id='prefixed-leaf-tools',
        description='Prefixed leaf tools.',
        defer_loading=False,
    )

    visited: list[AbstractCapability] = []
    cap.apply(visited.append)
    capability_map, available_ids = await _registered_capability_context(cap)

    assert cap.id == 'prefixed-leaf-tools'
    assert cap.description == 'Prefixed leaf tools.'
    assert capability_map == {'prefixed-leaf-tools': cap}
    assert 'prefixed-leaf-tools' in available_ids
    assert cap.defer_loading is False
    assert visited == [cap]


async def test_prefix_tools_registration_inherits_or_overrides_wrapper_metadata():
    """A wrapper inherits the wrapped capability's identity, unless it sets its own id."""

    github = Capability[object](
        id='github',
        description='GitHub MCP server.',
        defer_loading=True,
    )

    # No id of its own: inherit the wrapped capability's id, deferral, and description, so the
    # deferred capability still shows up in the load catalog under its own id.
    prefixed = PrefixTools(github, prefix='github')

    registered, available_ids = await _registered_capability_context(prefixed)

    assert registered['github'] is prefixed
    assert 'github' not in available_ids
    assert prefixed.id == 'github'
    assert prefixed.defer_loading is True
    assert prefixed.get_description() == 'GitHub MCP server.'

    # An explicit id makes the wrapper its own capability: it no longer inherits the wrapped
    # capability's id or deferral, though it still falls back to its description.
    explicit_id = PrefixTools(github, prefix='github', id='github_prefixed')
    registered, available_ids = await _registered_capability_context(explicit_id)

    assert registered['github_prefixed'] is explicit_id
    assert 'github_prefixed' in available_ids
    assert explicit_id.defer_loading is False
    assert explicit_id.get_description() == 'GitHub MCP server.'


async def test_wrapper_over_deferred_capability_preserves_deferral_end_to_end() -> None:
    """Wrapping a deferred capability keeps it deferred through a full run.

    Regression guard for metadata delegation: a wrapper with no id of its own must surface the
    wrapped deferred capability in the load catalog and reveal its (prefixed) tools after
    `load_capability`, rather than silently becoming an always-available capability.
    """
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    refunds = Capability[object](
        id='refunds',
        description='Refund policy tools.',
        toolsets=[toolset],
        defer_loading=True,
    )
    wrapped = PrefixTools(refunds, prefix='refunds')

    first_request_instructions: list[str | None] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))

        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            first_request = message(messages, ModelRequest)
            first_request_instructions.append(first_request.instructions)
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )

        if not any(part.tool_name == 'refunds_lookup_refund_policy' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='refunds_lookup_refund_policy',
                        args={'order_id': 'order-1'},
                        tool_call_id='lookup',
                    )
                ]
            )

        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[wrapped])
    result = await agent.run('Can I get a refund?')

    assert result.output == 'done'
    # The deferred capability is surfaced in the catalog under the wrapped capability's id.
    assert first_request_instructions == [
        "The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:\n"
        '- refunds: Refund policy tools.'
    ]


async def test_prefix_tools_explicit_defer_loading_overrides_anonymous_wrapped() -> None:
    """`PrefixTools(..., id='github', defer_loading=True)` over an anonymous wrapped
    capability registers as deferred under the wrapper's own id, not the wrapped's."""
    explicit_deferred = PrefixTools(
        Capability[object](),
        prefix='github',
        id='github',
        defer_loading=True,
    )

    registered, available_ids = await _registered_capability_context(explicit_deferred)

    assert registered['github'] is explicit_deferred
    assert 'github' not in available_ids
    assert explicit_deferred.defer_loading is True


async def test_prefix_tools_can_be_deferred():
    """A deferred PrefixTools wrapper keeps its prefixed tools deferred until load."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        return f'{order_id}: refund allowed'

    cap = PrefixTools(
        wrapped=Toolset(
            toolset,
        ),
        prefix='billing',
        id='refunds',
        description='Refund policy tools.',
        defer_loading=True,
    )
    seen_tool_state: list[list[tuple[str, bool]]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tool_state.append([(t.name, bool(t.defer_loading)) for t in info.function_tools])
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))

        if not any(isinstance(part, LoadCapabilityReturnPart) for message in messages for part in message.parts):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'refunds'},
                        tool_call_id='load-refunds',
                    )
                ]
            )

        if not any(part.tool_name == 'billing_lookup_refund_policy' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='billing_lookup_refund_policy',
                        args={'order_id': 'order-123'},
                        tool_call_id='lookup-refund',
                    )
                ]
            )

        refund_result = next(part.content for part in tool_returns if part.tool_name == 'billing_lookup_refund_policy')
        return make_text_response(f'done: {refund_result}')

    agent = Agent(FunctionModel(model_fn), capabilities=[cap])
    result = await agent.run('Can I get a refund?')

    assert result.output == 'done: order-123: refund allowed'
    assert seen_tool_state == snapshot(
        [
            [('load_capability', False)],
            [('load_capability', False), ('billing_lookup_refund_policy', True)],
            [('load_capability', False), ('billing_lookup_refund_policy', True)],
        ]
    )


async def test_prefix_tools_convenience_method():
    """AbstractCapability.prefix_tools() returns a PrefixTools wrapping self."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def inner_tool() -> str:
        return 'inner'  # pragma: no cover

    cap = Toolset(toolset).prefix_tools('ns')
    assert isinstance(cap, PrefixTools)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(','.join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[cap])
    result = await agent.run('list tools')
    assert result.output == 'ns_inner_tool'


async def test_wrapper_capability_delegates_hooks():
    """WrapperCapability delegates lifecycle hooks to the wrapped capability."""
    hook_calls: list[str] = []

    @dataclass
    class HookCap(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            hook_calls.append('before_run')

        async def after_run(self, ctx: RunContext, *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            hook_calls.append('after_run')
            return result

    wrapper = WrapperCapability(wrapped=HookCap())

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    await agent.run('Hello')

    assert 'before_run' in hook_calls
    assert 'after_run' in hook_calls


def test_wrapper_capability_for_agent_replaces():
    """WrapperCapability.for_agent replaces wrapped when its for_agent rebinds.

    Some capabilities (e.g. `TemporalDurability`) snapshot agent state in `for_agent`
    and return a new instance. The wrapper must propagate that.
    """

    @dataclass
    class RebindCap(AbstractCapability[None]):
        bound_to: str = ''

        def for_agent(self, agent: AbstractAgent[None, Any]) -> AbstractCapability[None]:
            return RebindCap(bound_to=agent.name or '')

    inner = RebindCap()
    wrapper = WrapperCapability(wrapped=inner)

    agent = Agent(FunctionModel(_resolve_dummy_model_fn), name='wrapper_for_agent_test')
    bound = wrapper.for_agent(agent)
    assert isinstance(bound, WrapperCapability)
    assert bound is not wrapper
    assert bound.wrapped is not inner
    assert cast(RebindCap, bound.wrapped).bound_to == 'wrapper_for_agent_test'


async def test_wrapper_capability_for_run_replaces():
    """WrapperCapability.for_run replaces wrapped when it changes."""
    toolset_a = FunctionToolset(id='a')

    @toolset_a.tool_plain
    def tool_a() -> str:
        return 'a'  # pragma: no cover

    toolset_b = FunctionToolset(id='b')

    @toolset_b.tool_plain
    def tool_b() -> str:
        return 'b'  # pragma: no cover

    @dataclass
    class SwitchCap(AbstractCapability):
        use_b: bool = False

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return SwitchCap(use_b=True)

        def get_toolset(self) -> AbstractToolset:
            return toolset_b if self.use_b else toolset_a

    wrapper = WrapperCapability(wrapped=SwitchCap())

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(','.join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    result = await agent.run('Hello')
    # for_run switches to toolset_b
    assert 'tool_b' in result.output


async def test_wrapper_capability_for_run_preserves_explicit_metadata() -> None:
    """WrapperCapability.for_run preserves explicit wrapper metadata."""

    @dataclass
    class SwitchCap(AbstractCapability):
        name: str = 'before'

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return SwitchCap(name='after')

    wrapper = WrapperCapability(
        wrapped=SwitchCap(),
        id='explicit-wrapper',
        description='Explicit wrapper metadata.',
        defer_loading=False,
    )

    result = await wrapper.for_run(_build_run_context())

    assert result is not wrapper
    assert isinstance(result, WrapperCapability)
    assert result.id == 'explicit-wrapper'
    assert result.description == 'Explicit wrapper metadata.'
    assert result.defer_loading is False
    assert isinstance(result.wrapped, SwitchCap)
    assert result.wrapped.name == 'after'


async def test_wrapper_capability_has_wrap_node_run():
    """WrapperCapability.has_wrap_node_run delegates to the wrapped capability."""
    plain = CustomCapability()
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`has_wrap_node_run`.*`wrap_node_run`'):
        assert WrapperCapability(wrapped=plain).has_wrap_node_run is False  # type: ignore[reportDeprecated]

    @dataclass
    class NodeRunCap(AbstractCapability):
        async def wrap_node_run(self, ctx: RunContext, *, node: Any, handler: Any) -> Any:
            return await handler(node)  # pragma: no cover

    with pytest.warns(PydanticAIDeprecationWarning, match=r'`has_wrap_node_run`.*`wrap_node_run`'):
        assert WrapperCapability(wrapped=NodeRunCap()).has_wrap_node_run is True  # type: ignore[reportDeprecated]


async def test_combined_capability_has_wrap_node_run():
    """CombinedCapability.has_wrap_node_run reports whether any child overrides the hook.

    Nothing in the library branches on this anymore — the bare-iteration warning it used to gate
    is gone now that `async for node in agent_run` fires node hooks — but it stays available for
    capability authors introspecting a chain, alongside `has_wrap_run_event_stream`.
    """

    @dataclass
    class NodeRunCap(AbstractCapability):
        async def wrap_node_run(self, ctx: RunContext, *, node: Any, handler: Any) -> Any:
            return await handler(node)  # pragma: no cover

    with pytest.warns(PydanticAIDeprecationWarning, match=r'`has_wrap_node_run`.*`wrap_node_run`'):
        assert CombinedCapability([CustomCapability()]).has_wrap_node_run is False  # type: ignore[reportDeprecated]
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`has_wrap_node_run`.*`wrap_node_run`'):
        assert CombinedCapability([CustomCapability(), NodeRunCap()]).has_wrap_node_run is True  # type: ignore[reportDeprecated]


async def test_wrapper_capability_delegates_resolve_model_id():
    """WrapperCapability delegates `resolve_model_id` (and `has_resolve_model_id`) to the wrapped capability."""
    resolved = TestModel()

    @dataclass
    class ResolverCap(AbstractCapability[Any]):
        async def resolve_model_id(self, ctx: ModelResolutionContext[Any], *, model_id: str) -> Any:
            return resolved if model_id == 'magic' else None

    wrapper = WrapperCapability(wrapped=ResolverCap())
    assert wrapper.has_resolve_model_id is True

    agent = Agent('test', capabilities=[wrapper])
    resolution_ctx = ModelResolutionContext[Any](agent=agent, deps=None)
    assert await wrapper.resolve_model_id(resolution_ctx, model_id='magic') is resolved
    assert await wrapper.resolve_model_id(resolution_ctx, model_id='other') is None

    # Wrapping a capability without `resolve_model_id` is a no-op.
    plain_wrapper = WrapperCapability(wrapped=CustomCapability())
    assert plain_wrapper.has_resolve_model_id is False
    assert await plain_wrapper.resolve_model_id(resolution_ctx, model_id='any') is None


async def test_wrapper_capability_delegates_model_request_hooks():
    """WrapperCapability delegates before/after model request hooks."""
    hook_calls: list[str] = []

    @dataclass
    class ModelRequestHookCap(AbstractCapability):
        async def before_model_request(
            self, ctx: RunContext, request_context: ModelRequestContext
        ) -> ModelRequestContext:
            hook_calls.append('before_model_request')
            return request_context

        async def after_model_request(
            self, ctx: RunContext, *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            hook_calls.append('after_model_request')
            return response

    wrapper = WrapperCapability(wrapped=ModelRequestHookCap())

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    await agent.run('Hello')

    assert 'before_model_request' in hook_calls
    assert 'after_model_request' in hook_calls


async def test_prefix_tools_tool_call_strips_prefix():
    """PrefixTools correctly strips the prefix when calling the underlying tool."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def greet(name: str) -> str:
        return f'hello {name}'

    cap = PrefixTools(wrapped=Toolset(toolset), prefix='ns')

    call_count = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart('ns_greet', {'name': 'world'})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[cap])
    result = await agent.run('greet world')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='greet world', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='ns_greet',
                        args={'name': 'world'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=52, output_tokens=5),
                model_name='function:respond:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ns_greet',
                        content='hello world',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=54, output_tokens=6),
                model_name='function:respond:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_wrapper_capability_get_serialization_name():
    """WrapperCapability.get_serialization_name returns None (abstract base)."""
    assert WrapperCapability.get_serialization_name() is None


async def test_wrapper_capability_delegates_on_run_error():
    """WrapperCapability delegates on_run_error to the wrapped capability."""

    @dataclass
    class RecoverCap(AbstractCapability[Any]):
        async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            return AgentRunResult(output='recovered')

    def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('model exploded')

    agent = Agent(FunctionModel(failing_model), capabilities=[WrapperCapability(wrapped=RecoverCap())])
    result = await agent.run('hello')
    assert result.output == 'recovered'


async def test_wrapper_capability_delegates_on_node_run_error():
    """WrapperCapability delegates on_node_run_error to the wrapped capability."""
    from pydantic_ai.result import FinalResult
    from pydantic_graph import End

    @dataclass
    class NodeRecoverCap(AbstractCapability[Any]):
        async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
            return End(FinalResult(output='node recovered'))

    def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('model exploded')

    agent = Agent(FunctionModel(failing_model), capabilities=[WrapperCapability(wrapped=NodeRecoverCap())])
    async with agent.iter('hello') as agent_run:
        node = agent_run.next_node
        while not isinstance(node, End):
            node = await agent_run.next(node)
    assert isinstance(node, End)
    assert node.data.output == 'node recovered'


async def test_wrapper_capability_delegates_wrap_run_event_stream():
    """WrapperCapability delegates wrap_run_event_stream to the wrapped capability."""
    observed_events: list[AgentStreamEvent] = []

    @dataclass
    class StreamObserverCap(AbstractCapability[Any]):
        async def wrap_run_event_stream(
            self,
            ctx: RunContext[Any],
            *,
            stream: AsyncIterable[AgentStreamEvent],
        ) -> AsyncIterable[AgentStreamEvent]:
            async for event in stream:
                observed_events.append(event)
                yield event

    agent = Agent(
        FunctionModel(simple_model_function, stream_function=simple_stream_function),
        capabilities=[WrapperCapability(wrapped=StreamObserverCap())],
    )

    async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in stream:
            pass

    await agent.run('hello', event_stream_handler=handler)
    assert len(observed_events) > 0


async def test_wrapper_capability_delegates_on_model_request_error():
    """WrapperCapability delegates on_model_request_error to the wrapped capability."""

    @dataclass
    class ModelErrorRecoverCap(AbstractCapability[Any]):
        async def on_model_request_error(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='recovered from model error')])

    def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('model request failed')

    agent = Agent(FunctionModel(failing_model), capabilities=[WrapperCapability(wrapped=ModelErrorRecoverCap())])
    result = await agent.run('hello')
    assert result.output == 'recovered from model error'


async def test_wrapper_capability_delegates_on_tool_validate_error():
    """WrapperCapability delegates on_tool_validate_error to the wrapped capability."""

    @dataclass
    class ValidateErrorCap(AbstractCapability[Any]):
        async def on_tool_validate_error(
            self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
        ) -> dict[str, Any]:
            # Recover by providing valid args
            return {'x': 1}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    return ModelResponse(parts=[TextPart(content='done')])
        if info.function_tools:
            return ModelResponse(parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='invalid json!!')])
        return ModelResponse(parts=[TextPart(content='no tools')])  # pragma: no cover

    agent = Agent(FunctionModel(model_fn), capabilities=[WrapperCapability(wrapped=ValidateErrorCap())])

    @agent.tool_plain
    def my_tool(x: int) -> str:
        return f'result: {x}'

    result = await agent.run('call tool')
    assert result.output == 'done'


async def test_wrapper_capability_delegates_on_tool_execute_error():
    """WrapperCapability delegates on_tool_execute_error to the wrapped capability."""

    @dataclass
    class ExecuteErrorCap(AbstractCapability[Any]):
        async def on_tool_execute_error(
            self,
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            error: Exception,
        ) -> Any:
            return 'recovered tool result'

    agent = Agent(
        FunctionModel(tool_calling_model),
        capabilities=[WrapperCapability(wrapped=ExecuteErrorCap())],
    )

    @agent.tool_plain
    def my_tool() -> str:
        raise ValueError('tool failed')

    result = await agent.run('call tool')
    assert result.output == 'final response'


# --- Capability ordering tests ---


@dataclass
class OutermostCap(AbstractCapability[Any]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')


@dataclass
class InnermostCap(AbstractCapability[Any]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')


@dataclass
class PlainCapA(AbstractCapability[Any]):
    pass


@dataclass
class PlainCapB(AbstractCapability[Any]):
    pass


@dataclass
class WrapsACap(AbstractCapability[Any]):
    """Must wrap around PlainCapA."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=[PlainCapA])


@dataclass
class RequiresOutermostCap(AbstractCapability[Any]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(requires=[OutermostCap])


def _cap_names(combined: CombinedCapability) -> list[str]:
    return [type(c).__name__ for c in combined.capabilities]


def test_ordering_outermost():
    """Capability declaring 'outermost' ends up at index 0."""
    combined = CombinedCapability([PlainCapA(), OutermostCap(), PlainCapB()])
    assert _cap_names(combined) == ['OutermostCap', 'PlainCapA', 'PlainCapB']


def test_ordering_innermost():
    """Capability declaring 'innermost' ends up last."""
    combined = CombinedCapability([InnermostCap(), PlainCapA(), PlainCapB()])
    assert _cap_names(combined) == ['PlainCapA', 'PlainCapB', 'InnermostCap']


def test_ordering_both_outermost_and_innermost():
    """Both outermost and innermost present."""
    combined = CombinedCapability([PlainCapA(), InnermostCap(), OutermostCap()])
    assert combined.capabilities[0].__class__ is OutermostCap
    assert combined.capabilities[-1].__class__ is InnermostCap


def test_ordering_multiple_outermost_tier():
    """Multiple outermost capabilities form a tier; original order breaks ties."""

    @dataclass
    class OutermostCap2(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='outermost')

    combined = CombinedCapability([PlainCapA(), OutermostCap2(), OutermostCap()])
    # Both outermost caps before PlainCapA; original order (OutermostCap2 before OutermostCap) preserved
    assert _cap_names(combined) == ['OutermostCap2', 'OutermostCap', 'PlainCapA']


def test_ordering_multiple_innermost_tier():
    """Multiple innermost capabilities form a tier; original order breaks ties."""

    @dataclass
    class InnermostCap2(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='innermost')

    combined = CombinedCapability([InnermostCap(), InnermostCap2(), PlainCapA()])
    # PlainCapA first, then both innermost in original order
    assert _cap_names(combined) == ['PlainCapA', 'InnermostCap', 'InnermostCap2']


def test_ordering_outermost_tier_with_wraps():
    """wraps/wrapped_by refines order within the outermost tier."""

    @dataclass
    class OuterA(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='outermost')

    @dataclass
    class OuterB(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='outermost', wraps=[OuterA])

    # OuterB listed after OuterA, but wraps=[OuterA] overrides tiebreaker
    combined = CombinedCapability([OuterA(), PlainCapA(), OuterB()])
    assert _cap_names(combined) == ['OuterB', 'OuterA', 'PlainCapA']


def test_ordering_wraps():
    """Explicit 'wraps' edge is respected."""
    combined = CombinedCapability([PlainCapA(), WrapsACap()])
    assert _cap_names(combined) == ['WrapsACap', 'PlainCapA']


def test_ordering_wrapped_by():
    """Explicit 'wrapped_by' edge is respected."""

    @dataclass
    class WrappedByACap(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(wrapped_by=[PlainCapA])

    combined = CombinedCapability([WrappedByACap(), PlainCapA()])
    assert _cap_names(combined) == ['PlainCapA', 'WrappedByACap']


def test_innermost_binds_after_capability_toolsets():
    """`innermost` capabilities bind after other capabilities' toolsets are extracted.

    Durability capabilities (the `innermost` tier) wrap `agent.toolsets` in their `for_agent`,
    so `Agent.__init__` binds them in a second phase, after toolsets contributed by other
    capabilities (e.g. `Capability(tools=...)`) have been extracted and are visible on the
    agent. Binding everything in one phase would leave those toolsets invisible to durability
    and running unwrapped (non-deterministically) inside durable workflows.
    """
    seen_tool_names: set[str] = set()

    @dataclass
    class RecordingInnermostCap(AbstractCapability[Any]):
        def for_agent(self, agent: AbstractAgent[Any, Any]) -> RecordingInnermostCap:
            for toolset in agent.toolsets:
                toolset.apply(
                    lambda leaf: seen_tool_names.update(leaf.tools) if isinstance(leaf, FunctionToolset) else None
                )
            # Return a bound copy, like durability capabilities do.
            return replace(self)

        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='innermost')

    def greet() -> str:
        return 'hi'  # pragma: no cover

    original = RecordingInnermostCap()
    agent = Agent('test', capabilities=[Capability(tools=[greet]), original])
    assert seen_tool_names == {'greet'}
    # The bound copy replaced the original in the agent's capability chain.
    assert not any(cap is original for cap in agent.root_capability.capabilities)
    assert any(isinstance(cap, RecordingInnermostCap) for cap in agent.root_capability.capabilities)


def test_combined_capability_for_agent_binds_children():
    """`CombinedCapability.for_agent` rebinds children that return new bound instances."""

    @dataclass
    class BindingCap(AbstractCapability[Any]):
        bound: bool = False

        def for_agent(self, agent: AbstractAgent[Any, Any]) -> BindingCap:
            return replace(self, bound=True)

    combined = CombinedCapability([BindingCap(), PlainCapA()])
    agent = Agent('test')
    bound = combined.for_agent(agent)
    assert bound is not combined
    assert isinstance(bound.capabilities[0], BindingCap)
    assert bound.capabilities[0].bound is True


def test_ordering_requires_present():
    """No error when required capability is present."""
    combined = CombinedCapability([RequiresOutermostCap(), OutermostCap()])
    assert len(combined.capabilities) == 2


def test_ordering_requires_missing():
    with pytest.raises(UserError, match='`RequiresOutermostCap` requires `OutermostCap`'):
        CombinedCapability([RequiresOutermostCap(), PlainCapA()])


def test_ordering_preserves_user_order():
    """Capabilities without constraints keep their relative order."""
    a, b = PlainCapB(), PlainCapA()
    combined = CombinedCapability([a, b])
    assert list(combined.capabilities) == [a, b]


def test_ordering_nested_combined():
    """Leaves of a nested `CombinedCapability` participate as siblings in the outer sort.

    `CombinedCapability` auto-flattens nested instances so each leaf is sorted
    independently rather than as a group. Here `OutermostCap` (inside `inner`)
    sorts to the front; its former sibling `PlainCapB` is unconstrained.
    """
    inner = CombinedCapability([PlainCapB(), OutermostCap()])
    combined = CombinedCapability([PlainCapA(), inner])
    # `inner` is splatted; `OutermostCap` sorts first.
    assert [type(c) for c in combined.capabilities] == [OutermostCap, PlainCapA, PlainCapB]


def test_ordering_nested_combined_no_constraints():
    """A nested `CombinedCapability` with no ordering leaves is splatted as flat siblings."""
    inner = CombinedCapability([PlainCapA(), PlainCapB()])
    combined = CombinedCapability([inner, OutermostCap()])
    # `OutermostCap` first; `inner`'s leaves follow as flat siblings in their original order.
    assert [type(c) for c in combined.capabilities] == [OutermostCap, PlainCapA, PlainCapB]


def test_ordering_nested_combined_wraps_without_position():
    """A `wraps` constraint on a leaf inside a nested `CombinedCapability` applies to that leaf only."""
    inner = CombinedCapability([PlainCapB(), WrapsACap()])
    combined = CombinedCapability([PlainCapA(), inner])
    # `WrapsACap` is splatted and sorts before `PlainCapA`; `PlainCapB` is unconstrained
    # and keeps its insertion order (it sits between PlainCapA and WrapsACap in the
    # post-flatten input list, so the topo sort surfaces it first as ready-without-deps).
    assert [type(c) for c in combined.capabilities] == [PlainCapB, WrapsACap, PlainCapA]


def test_ordering_single_capability():
    """Single capability in CombinedCapability is unchanged."""
    cap = OutermostCap()
    combined = CombinedCapability([cap])
    assert list(combined.capabilities) == [cap]


def test_ordering_no_constraints_noop():
    """When no capability declares ordering, list is unchanged."""
    a, b = PlainCapA(), PlainCapB()
    combined = CombinedCapability([a, b])
    assert list(combined.capabilities) == [a, b]


def test_ordering_cycle_detection():
    @dataclass
    class CycleA(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(wraps=[CycleB])

    @dataclass
    class CycleB(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(wraps=[CycleA])

    with pytest.raises(UserError, match='Circular ordering constraints'):
        CombinedCapability([CycleA(), CycleB()])


def test_ordering_mixed_positions_in_nested():
    """Mixed positions in a nested `CombinedCapability` work — leaves are splatted into the outer sort."""
    inner = CombinedCapability([OutermostCap(), InnermostCap()])
    combined = CombinedCapability([inner, PlainCapA()])
    # `OutermostCap` first (outermost tier), `PlainCapA` middle, `InnermostCap` last (innermost tier).
    assert [type(c) for c in combined.capabilities] == [OutermostCap, PlainCapA, InnermostCap]


def test_ordering_conflicting_positions_in_custom_nested_capability():
    """A custom capability tree cannot collapse outermost and innermost leaves into one ordered group."""

    @dataclass
    class NestedCapabilityGroup(AbstractCapability[Any]):
        leaves: tuple[AbstractCapability[Any], ...]

        def apply(self, visitor: Callable[[AbstractCapability[Any]], None]) -> None:
            for leaf in self.leaves:
                leaf.apply(visitor)

    nested = NestedCapabilityGroup((OutermostCap(), InnermostCap()))

    with pytest.raises(UserError, match='Conflicting positions among nested leaves'):
        CombinedCapability([nested, PlainCapA()])


def test_ordering_hooks_ordering_parameter():
    """Hooks with ordering= are sorted according to those constraints."""
    hooks = Hooks(ordering=CapabilityOrdering(position='outermost'))
    combined = CombinedCapability([PlainCapA(), hooks, PlainCapB()])
    assert combined.capabilities[0] is hooks


def test_ordering_hooks_ordering_wraps():
    """Hooks with ordering wraps= are placed before the referenced type."""
    hooks = Hooks(ordering=CapabilityOrdering(wraps=[PlainCapA]))
    combined = CombinedCapability([PlainCapA(), hooks])
    assert combined.capabilities[0] is hooks


def test_ordering_hooks_ordering_wrapped_by():
    """Hooks with ordering wrapped_by= are placed after the referenced type."""
    hooks = Hooks(ordering=CapabilityOrdering(wrapped_by=[PlainCapA]))
    combined = CombinedCapability([hooks, PlainCapA()])
    assert combined.capabilities[0].__class__ is PlainCapA
    assert combined.capabilities[1] is hooks


def test_ordering_hooks_no_ordering():
    """Hooks without ordering= preserve their list position."""
    hooks = Hooks()
    combined = CombinedCapability([PlainCapA(), hooks, PlainCapB()])
    assert combined.capabilities[1] is hooks


def test_ordering_hooks_ordering_requires():
    """Hooks with ordering requires= validates that the required type is present."""
    hooks = Hooks(ordering=CapabilityOrdering(requires=[OutermostCap]))
    with pytest.raises(UserError, match='`Hooks` requires `OutermostCap`'):
        CombinedCapability([hooks, PlainCapA()])


def test_ordering_wraps_instance_ref():
    """wraps= with an instance ref only constrains the specific instance, not all instances of that type."""
    target = PlainCapA()
    other_a = PlainCapA()

    @dataclass
    class WrapsInstance(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(wraps=[target])

    # Arrange so that instance ref vs type ref produces a distinguishable result:
    # - Instance ref wraps=[target] → only target must come after WrapsInstance
    # - A type ref wraps=[PlainCapA] would constrain both other_a and target
    combined = CombinedCapability([other_a, target, WrapsInstance()])
    # other_a stays before WrapsInstance (no constraint), WrapsInstance before target
    assert combined.capabilities[0] is other_a
    assert combined.capabilities[1].__class__ is WrapsInstance
    assert combined.capabilities[2] is target


def test_ordering_wrapped_by_instance_ref():
    """wrapped_by= can reference a specific capability instance."""
    wrapper = PlainCapA()

    @dataclass
    class WrappedByInstance(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(wrapped_by=[wrapper])

    combined = CombinedCapability([WrappedByInstance(), wrapper])
    assert combined.capabilities[0] is wrapper
    assert combined.capabilities[1].__class__ is WrappedByInstance


def test_ordering_hooks_wraps_instance():
    """Hooks can order relative to a specific capability instance via wraps=."""
    target = PlainCapA()
    hooks = Hooks(ordering=CapabilityOrdering(wraps=[target]))
    combined = CombinedCapability([target, hooks])
    assert combined.capabilities[0] is hooks
    assert combined.capabilities[1] is target


def test_ordering_hooks_wrapped_by_instance():
    """Hooks can order relative to a specific capability instance via wrapped_by=."""
    outer = PlainCapA()
    hooks = Hooks(ordering=CapabilityOrdering(wrapped_by=[outer]))
    combined = CombinedCapability([hooks, outer])
    assert combined.capabilities[0] is outer
    assert combined.capabilities[1] is hooks


def test_ordering_instance_ref_not_present():
    """Instance ref in wraps= that isn't in the list has no effect (no edge added)."""
    absent = PlainCapA()
    hooks = Hooks(ordering=CapabilityOrdering(wraps=[absent]))
    # absent is NOT in the capabilities list — the wraps ref should be a no-op
    combined = CombinedCapability([PlainCapB(), hooks])
    # Order preserved since the instance ref doesn't match anything
    assert combined.capabilities[0].__class__ is PlainCapB
    assert combined.capabilities[1] is hooks


def test_ordering_mixed_type_and_instance_refs():
    """wraps= can mix type refs and instance refs."""
    target_instance = PlainCapB()

    @dataclass
    class MixedRefs(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(wraps=[PlainCapA, target_instance])

    combined = CombinedCapability([PlainCapA(), target_instance, MixedRefs()])
    assert combined.capabilities[0].__class__ is MixedRefs


async def test_ordering_survives_dynamic_capability_resolution():
    """A factory-returned capability's ordering constraints survive the per-run wrapper.

    `CombinedCapability.for_run` re-sorts the replaced capabilities, so the
    `ResolvedDynamicCapability` wrapper must delegate `get_ordering` to the resolved
    capability for its `outermost`/`innermost`/`wraps` declarations to be honored.
    """

    def factory(ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return OutermostCap()

    combined = CombinedCapability([PlainCapA(), DynamicCapability(factory)])
    # At construction, the unresolved wrapper has no ordering of its own.
    assert _cap_names(combined) == ['PlainCapA', 'DynamicCapability']

    ctx = _build_run_context()
    ctx.agent = Agent(TestModel())
    run_capability = await combined.for_run(ctx)
    assert isinstance(run_capability, CombinedCapability)
    assert _cap_names(run_capability) == ['ResolvedDynamicCapability', 'PlainCapA']
    assert isinstance(run_capability.capabilities[0], ResolvedDynamicCapability)
    assert isinstance(run_capability.capabilities[0].wrapped, OutermostCap)


async def test_runtime_capability_with_mixed_position_root():
    """Per-run capabilities can be added to an agent whose root mixes outermost and innermost.

    `Agent.iter()` builds the effective capability by merging per-run capabilities into the
    agent's `_root_capability`. If `_root_capability` is a `CombinedCapability` whose leaves
    span tiers (e.g. an outermost-tier cap and an innermost-tier cap), wrapping it in another
    `CombinedCapability` used to trigger "Conflicting positions in nested CombinedCapability"
    because the outer sort tried to compute a single effective ordering for the inner group.
    The fix splats the root container so each leaf participates as a sibling in the outer
    ordering pass.
    """
    agent = Agent(TestModel(), capabilities=[OutermostCap(), InnermostCap()])
    result = await agent.run('hi', capabilities=[Hooks()])
    assert result.output == snapshot('success (no tool calls)')


# --- resolve_model_id hook tests ---


def _resolve_dummy_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='ok')])


@dataclass
class _StringResolver(AbstractCapability[Any]):
    """Test capability that maps known strings to a fixed FunctionModel."""

    target: FunctionModel

    async def resolve_model_id(self, ctx: ModelResolutionContext[Any], *, model_id: Any) -> Any:
        if model_id == 'magic-model':
            return self.target
        return None


@dataclass
class _PassThroughResolver(AbstractCapability[Any]):
    """Test capability that always defers, recording what it saw."""

    seen: list[Any] = field(default_factory=list[Any])
    seen_deps: list[Any] = field(default_factory=list[Any])

    async def resolve_model_id(self, ctx: ModelResolutionContext[Any], *, model_id: Any) -> Any:
        self.seen.append(model_id)
        self.seen_deps.append(ctx.deps)
        return None


async def test_resolve_model_id_maps_string_to_model() -> None:
    """A capability's resolve_model_id maps a runtime string to a Model instance."""
    target = FunctionModel(_resolve_dummy_model_fn, model_name='resolved')
    agent = Agent(name='resolve_test', capabilities=[_StringResolver(target=target)])

    result = await agent.run('hi', model='magic-model')
    assert result.output == 'ok'


async def test_resolve_model_id_returns_none_falls_back_to_infer_model() -> None:
    """When all capabilities defer, _get_model uses the default infer_model path."""
    cap = _PassThroughResolver()
    agent = Agent(name='resolve_pass', capabilities=[cap], defer_model_check=True)

    # 'test' is the special string that infer_model maps to TestModel.
    result = await agent.run('hi', model='test')
    assert result.output is not None
    assert cap.seen == ['test']


async def test_resolve_model_id_returns_none_for_unknown_string() -> None:
    """A resolver that doesn't recognize the string returns None so the next layer can try."""
    target = FunctionModel(_resolve_dummy_model_fn, model_name='resolved')
    cap = _StringResolver(target=target)
    resolution_ctx = ModelResolutionContext(agent=cast(Any, None), deps=None)
    assert await cap.resolve_model_id(resolution_ctx, model_id='different-string') is None


async def test_resolve_model_id_first_non_none_wins() -> None:
    """When two capabilities declare resolve_model_id, the first one in the list wins.

    Composition is first-non-None-wins (not each-layer-wraps): only one capability
    can claim a given string. Per-request *wrapping* of a resolved Model lives in
    `before_model_request`, not here.
    """
    first_target = FunctionModel(_resolve_dummy_model_fn, model_name='first')
    second_target = FunctionModel(_resolve_dummy_model_fn, model_name='second')

    first = _StringResolver(target=first_target)
    second = _StringResolver(target=second_target)
    combined = CombinedCapability([first, second])

    agent = Agent(name='resolve_layered', capabilities=[first, second], defer_model_check=True)
    result = await combined.resolve_model_id(ModelResolutionContext(agent=agent, deps=None), model_id='magic-model')
    assert result is first_target


def test_resolve_model_id_skipped_for_model_instance() -> None:
    """The hook is never called when the user passes a Model instance directly."""
    cap = _PassThroughResolver()
    target = FunctionModel(_resolve_dummy_model_fn, model_name='direct')
    agent = Agent(target, name='resolve_skip_instance', capabilities=[cap])

    # No string ever flows through; cap.seen should stay empty.
    assert agent.model is target
    assert cap.seen == []


async def test_resolve_model_id_invoked_on_override() -> None:
    """`agent.override(model=string)` routes the string through resolve_model_id."""
    target = FunctionModel(_resolve_dummy_model_fn, model_name='override-resolved')
    cap = _StringResolver(target=target)

    initial_model = FunctionModel(_resolve_dummy_model_fn, model_name='initial')
    agent = Agent(initial_model, name='resolve_override', capabilities=[cap])

    with agent.override(model='magic-model'):
        result = await agent.run('hi')
    assert result.output == 'ok'


async def test_resolve_model_id_invoked_on_agent_default_string() -> None:
    """`Agent(model='string', capabilities=[cap])` routes the default through resolve_model_id at run setup.

    Capabilities with `resolve_model_id` need a shot at the default model string just
    like they do for runtime overrides. The hook is deps-aware and only fires at run
    setup, so the agent keeps the raw string at construction (like `defer_model_check`)
    and resolution happens per run — under different deps, potentially to different models.
    """
    target = FunctionModel(_resolve_dummy_model_fn, model_name='default-resolved')
    cap = _StringResolver(target=target)

    agent = Agent('magic-model', name='resolve_default_string', capabilities=[cap])

    # The default stays a string at construction; the hook can't run without deps.
    assert agent.model == 'magic-model'

    result = await agent.run('hi')
    assert result.output == 'ok'

    # No memoization: the raw string is kept so per-run resolution keeps firing.
    assert agent.model == 'magic-model'


async def test_resolve_model_id_receives_deps() -> None:
    """The hook receives the run's deps on `ctx.deps`, so resolution can be run-dependent."""
    cap = _PassThroughResolver()
    agent = Agent(name='resolve_deps', deps_type=str, capabilities=[cap], defer_model_check=True)

    await agent.run('hi', model='test', deps='user-credential')
    assert cap.seen == ['test']
    assert cap.seen_deps == ['user-credential']


async def test_override_model_string_deferral_considers_override_capabilities() -> None:
    """`override(model=str)`'s defer-vs-eager choice consults the effective root capability.

    Neither the spec capability nor the agent chain implements `resolve_model_id` here, so
    the string resolves eagerly via `infer_model` — checked against the spec-supplied root
    when set in the same call, and against an already-active root override when nested.
    """
    agent = Agent(name='override_deferral_effective_root')

    with agent.override(spec={'capabilities': [{'IncludeToolReturnSchemas': {}}]}, model='test'):
        result = await agent.run('hi')
        assert result.output is not None

    with agent.override(spec={'capabilities': [{'IncludeToolReturnSchemas': {}}]}):
        with agent.override(model='test'):
            result = await agent.run('hi')
            assert result.output is not None


async def test_resolve_model_id_uses_override_root_capability() -> None:
    """A root-capability override (as set by `override(spec=...)`) owns model-string resolution.

    Not a public-API test: no built-in spec-constructible capability implements
    `resolve_model_id` yet, so this drives the `_override_root_capability` contextvar —
    the exact seam `override(spec=...)` sets when a spec replaces the root — directly.
    Pins that resolution honors the effective (replaced) root, and that the resolved
    model doesn't get memoized onto `agent.model` past the override's scope.
    """
    chain_target = FunctionModel(_resolve_dummy_model_fn, model_name='agent-chain')
    override_target = FunctionModel(_resolve_dummy_model_fn, model_name='override-root')

    agent = Agent('magic-model', name='resolve_override_root', capabilities=[_StringResolver(target=chain_target)])

    override_root = CombinedCapability[Any]([_StringResolver(target=override_target)])
    token = agent._override_root_capability.set(Some(override_root))  # pyright: ignore[reportPrivateUsage]
    try:
        resolved = await agent._resolve_model_selection(  # pyright: ignore[reportPrivateUsage]
            agent._pick_raw_model(None),  # pyright: ignore[reportPrivateUsage]
            capability=agent._effective_root_capability(),  # pyright: ignore[reportPrivateUsage]
            deps=None,
        )
        assert resolved is override_target
        # No memoization under an override: the raw string default survives.
        assert agent.model == 'magic-model'
    finally:
        agent._override_root_capability.reset(token)  # pyright: ignore[reportPrivateUsage]

    resolved = await agent._resolve_model_selection(  # pyright: ignore[reportPrivateUsage]
        agent._pick_raw_model(None),  # pyright: ignore[reportPrivateUsage]
        capability=agent._effective_root_capability(),  # pyright: ignore[reportPrivateUsage]
        deps=None,
    )
    assert resolved is chain_target


async def test_resolve_model_id_alias_unusable_outside_run() -> None:
    """A capability-owned alias default resolves during runs, and says so clearly outside one.

    Sync entry points like `set_mcp_sampling_model` can't invoke the async, deps-aware
    hook, so an alias only a capability can resolve raises an explanation asking for a
    concrete model rather than attempting deps-blind resolution.
    """
    target = FunctionModel(_resolve_dummy_model_fn, model_name='aliased')

    def resolver(ctx: ModelResolutionContext[Any], model_id: str) -> FunctionModel | None:
        return target if model_id == 'alias' else None

    agent = Agent('alias', name='alias_outside_run', capabilities=[ResolveModelId(resolver)])
    with pytest.raises(UserError, match='requires run dependencies and cannot be used for MCP sampling'):
        agent.set_mcp_sampling_model()

    # Inside a run, the alias resolves through the hook as usual.
    result = await agent.run('hi')
    assert result.output == 'ok'


# --- ResolveModelId capability tests ---


async def test_resolve_model_id_capability_sync_resolver() -> None:
    """`ResolveModelId` wraps a sync resolver function that maps strings to models using deps."""
    target = FunctionModel(_resolve_dummy_model_fn, model_name='sync-resolved')
    seen_deps: list[Any] = []

    def resolver(ctx: ModelResolutionContext[str], model_id: str) -> FunctionModel | None:
        seen_deps.append(ctx.deps)
        return target if model_id == 'alias' else None

    agent = Agent('alias', name='resolve_cap_sync', deps_type=str, capabilities=[ResolveModelId(resolver)])
    result = await agent.run('hi', deps='credential')
    assert result.output == 'ok'
    assert seen_deps == ['credential']


async def test_resolve_model_id_capability_async_resolver() -> None:
    """`ResolveModelId` also accepts an async resolver function."""
    target = FunctionModel(_resolve_dummy_model_fn, model_name='async-resolved')

    async def resolver(ctx: ModelResolutionContext[Any], model_id: str) -> FunctionModel | None:
        return target if model_id == 'alias' else None

    agent = Agent(name='resolve_cap_async', capabilities=[ResolveModelId(resolver)])
    result = await agent.run('hi', model='alias')
    assert result.output == 'ok'


async def test_resolve_model_id_capability_sync_resolver_returning_coroutine() -> None:
    """A plain-`def` resolver returning a coroutine is awaited, not mistaken for the resolved model.

    `ModelIdResolver` permits a sync function whose return value is an `Awaitable[Model | None]`;
    the hook must await that coroutine to obtain the model rather than returning the coroutine itself.
    """
    target = FunctionModel(_resolve_dummy_model_fn, model_name='coroutine-resolved')

    async def _resolve(model_id: str) -> FunctionModel | None:
        return target if model_id == 'alias' else None

    def resolver(ctx: ModelResolutionContext[Any], model_id: str) -> Awaitable[FunctionModel | None]:
        return _resolve(model_id)

    agent = Agent(name='resolve_cap_sync_coroutine', capabilities=[ResolveModelId(resolver)])
    result = await agent.run('hi', model='alias')
    assert result.output == 'ok'


async def test_resolve_model_id_capability_defers_to_infer_model() -> None:
    """A `ResolveModelId` resolver returning None falls back to the default `infer_model` flow."""

    def resolver(ctx: ModelResolutionContext[Any], model_id: str) -> None:
        return None

    agent = Agent(name='resolve_cap_defer', capabilities=[ResolveModelId(resolver)])
    # 'test' is the special string that infer_model maps to TestModel.
    result = await agent.run('hi', model='test')
    assert result.output is not None


# --- Agent-bound capabilities ---


@dataclass
class _AgentBoundCapability(AbstractCapability[Any]):
    bound_name: str | None = None
    for_agent_calls: int = 0

    def for_agent(self, agent: AbstractAgent[Any, Any]) -> _AgentBoundCapability:
        return replace(self, bound_name=agent.name, for_agent_calls=self.for_agent_calls + 1)

    def get_instructions(self) -> str:
        return f'Bound to {self.bound_name}.'


async def test_for_agent_returns_bound_copy() -> None:
    capability = _AgentBoundCapability()

    first = Agent(TestModel(), name='first', capabilities=[capability])
    second = Agent(TestModel(), name='second', capabilities=[capability])

    first_bound = next(cap for cap in first.root_capability.capabilities if isinstance(cap, _AgentBoundCapability))
    second_bound = next(cap for cap in second.root_capability.capabilities if isinstance(cap, _AgentBoundCapability))
    assert capability.bound_name is None
    assert first_bound is not capability
    assert second_bound is not capability
    assert first_bound.bound_name == 'first'
    assert second_bound.bound_name == 'second'
    assert first_bound.for_agent_calls == second_bound.for_agent_calls == 1

    result = await first.run('hello')
    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Bound to first.'


def test_wrapper_for_agent_replaces_wrapped_capability() -> None:
    capability = _AgentBoundCapability()
    wrapper = WrapperCapability(capability)

    agent = Agent(TestModel(), name='wrapped', capabilities=[wrapper])

    bound_wrapper = next(cap for cap in agent.root_capability.capabilities if isinstance(cap, WrapperCapability))
    assert bound_wrapper is not wrapper
    assert cast(_AgentBoundCapability, bound_wrapper.wrapped).bound_name == 'wrapped'


def test_wrapper_for_agent_preserves_identity_without_replacement() -> None:
    """Identity preservation is an internal binding contract that a request cassette cannot observe."""
    wrapper = WrapperCapability[Any](AbstractCapability[Any]())
    agent = Agent(TestModel())

    assert wrapper.for_agent(agent) is wrapper


async def test_for_agent_composes_with_model_selection_and_resolution() -> None:
    selected_model = TestModel(custom_output_text='selected')

    @dataclass
    class BoundModelCapability(AbstractCapability[Any]):
        model_id: str | None = None

        def for_agent(self, agent: AbstractAgent[Any, Any]) -> BoundModelCapability:
            return replace(self, model_id=f'bound:{agent.name}')

        def get_model(self) -> str | None:
            return self.model_id

        async def resolve_model_id(
            self,
            ctx: ModelResolutionContext[Any],
            *,
            model_id: KnownModelName | str,
        ) -> Model | None:
            assert ctx.agent.name == 'selector'
            return selected_model if model_id == self.model_id else None

    agent = Agent(name='selector', capabilities=[BoundModelCapability()])
    result = await agent.run('hello')
    assert result.output == 'selected'


async def test_for_agent_can_introduce_model_id_resolution() -> None:
    selected_model = TestModel(custom_output_text='selected')

    @dataclass
    class BoundResolver(AbstractCapability[Any]):
        async def resolve_model_id(
            self,
            ctx: ModelResolutionContext[Any],
            *,
            model_id: KnownModelName | str,
        ) -> Model | None:
            return selected_model if model_id == 'custom-model' else None

    @dataclass
    class BindingCapability(AbstractCapability[Any]):
        def for_agent(self, agent: AbstractAgent[Any, Any]) -> AbstractCapability[Any]:
            assert agent.model == 'custom-model'
            return BoundResolver()

    agent = Agent('custom-model', capabilities=[BindingCapability()])
    assert (await agent.run('hello')).output == 'selected'


async def test_for_agent_can_introduce_resolution_for_known_model_id() -> None:
    selected_model = TestModel(custom_output_text='selected')

    @dataclass
    class BoundResolver(AbstractCapability[Any]):
        async def resolve_model_id(
            self,
            ctx: ModelResolutionContext[Any],
            *,
            model_id: KnownModelName | str,
        ) -> Model | None:
            return selected_model if model_id == 'test' else None

    @dataclass
    class BindingCapability(AbstractCapability[Any]):
        def for_agent(self, agent: AbstractAgent[Any, Any]) -> AbstractCapability[Any]:
            assert agent.model == 'test'
            return BoundResolver()

    agent = Agent('test', capabilities=[BindingCapability()])
    assert agent.model == 'test'
    assert (await agent.run('hello')).output == 'selected'


def test_for_agent_without_resolver_preserves_unknown_model_error() -> None:
    with pytest.raises(UserError, match='Unknown model: custom-model'):
        Agent('custom-model', capabilities=[_AgentBoundCapability()])


async def test_for_agent_binds_per_run_capabilities() -> None:
    capability = _AgentBoundCapability()
    agent = Agent(TestModel(), name='runner')

    result = await agent.run('hello', capabilities=[capability])

    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Bound to runner.'
    assert capability.for_agent_calls == 0


async def test_per_run_binding_can_supply_bootstrap_model_and_resolver() -> None:
    """Run binding precedes bootstrap selection and resolution, an ordering contract cassettes cannot isolate."""
    selected_model = TestModel(custom_output_text='run-bound')

    @dataclass
    class BoundRunModel(AbstractCapability[Any]):
        def get_model(self) -> str:
            return 'run-bound-id'

        async def resolve_model_id(
            self,
            ctx: ModelResolutionContext[Any],
            *,
            model_id: KnownModelName | str,
        ) -> Model | None:
            return selected_model if model_id == 'run-bound-id' else None

    @dataclass
    class BindAtRun(AbstractCapability[Any]):
        def for_agent(self, agent: AbstractAgent[Any, Any]) -> AbstractCapability[Any]:
            return BoundRunModel()

    agent = Agent(None)
    result = await agent.run('hello', capabilities=[BindAtRun()])

    assert result.output == 'run-bound'


# --- Dynamic capabilities ---


@dataclass
class _RecordingCapability(AbstractCapability[Any]):
    """Test capability that records every hook firing and contributes instructions."""

    label: str
    fired: list[str] = field(default_factory=list[str])

    def get_instructions(self) -> str:
        return f'Label is {self.label}.'

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.fired.append(f'{self.label}:before_run')

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.fired.append(f'{self.label}:before_model_request')
        return request_context


async def test_dynamic_capability_factory_called_with_run_context() -> None:
    """The factory receives the run's `RunContext` (with deps) once per run."""
    seen: list[Any] = []

    def factory(ctx: RunContext[str]) -> AbstractCapability[Any] | None:
        seen.append(ctx.deps)
        return _RecordingCapability(label=ctx.deps)

    agent = Agent(TestModel(), deps_type=str, capabilities=[factory])
    await agent.run('hi', deps='admin')
    await agent.run('hi', deps='guest')
    assert seen == ['admin', 'guest']


async def test_dynamic_capability_factory_result_is_bound_to_agent() -> None:
    """A factory's standalone result is agent-bound before its run binding; a cassette cannot observe hook order."""

    def factory(ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return _AgentBoundCapability()

    agent = Agent(TestModel(), name='dynamic', capabilities=[factory])
    result = await agent.run('hi')

    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Bound to dynamic.'


async def test_for_run_result_is_not_bound_again() -> None:
    """A specialized run-bound result skips agent binding; a provider cassette cannot observe that distinction."""

    @dataclass
    class BuildsRunCapability(AbstractCapability[Any]):
        async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
            return _AgentBoundCapability()

    agent = Agent(TestModel(), name='static', capabilities=[BuildsRunCapability()])
    result = await agent.run('hi')

    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Bound to None.'


async def test_dynamic_capability_async_factory() -> None:
    """Async factories are awaited."""
    calls = 0

    async def factory(ctx: RunContext) -> AbstractCapability[Any]:
        nonlocal calls
        calls += 1
        return _RecordingCapability(label='async')

    agent = Agent(TestModel(), capabilities=[factory])
    await agent.run('hi')
    assert calls == 1


async def test_dynamic_capability_returning_none_contributes_nothing() -> None:
    """A factory returning None is a no-op for the run."""

    def factory(ctx: RunContext) -> AbstractCapability[Any] | None:
        return None

    agent = Agent(TestModel(), capabilities=[factory])
    result = await agent.run('hi')
    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions is None

    dynamic = DynamicCapability(factory)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    assert await dynamic.for_run(ctx) is dynamic

    # Direct toolset-factory call (unit-style): the standalone fallback — a context without the
    # run's capability registry, as inside a durable unit — re-resolves the factory, and an async
    # factory returning `None` still contributes nothing.
    async def async_none_factory(ctx: RunContext[Any]) -> AbstractCapability[Any] | None:
        return None

    async_dynamic = DynamicCapability(async_none_factory)
    resolved = async_dynamic.get_toolset().toolset_func(ctx)
    assert inspect.isawaitable(resolved)
    assert await resolved is None


def test_dynamic_capability_toolset_is_cached_and_inherits_id() -> None:
    dynamic = DynamicCapability(lambda ctx: None, id='x')
    toolset = dynamic.get_toolset()

    assert toolset.id == 'x'
    assert dynamic.get_toolset() is toolset


async def test_dynamic_capability_contributes_instructions_per_run() -> None:
    """Resolved capability's instructions flow through to the model request."""

    def factory(ctx: RunContext[str]) -> AbstractCapability[Any] | None:
        if ctx.deps == 'admin':
            return _RecordingCapability(label='admin')
        return None

    agent = Agent(TestModel(), deps_type=str, capabilities=[factory])

    admin_result = await agent.run('hi', deps='admin')
    admin_request = next(m for m in admin_result.all_messages() if isinstance(m, ModelRequest))
    assert admin_request.instructions == 'Label is admin.'

    guest_result = await agent.run('hi', deps='guest')
    guest_request = next(m for m in guest_result.all_messages() if isinstance(m, ModelRequest))
    assert guest_request.instructions is None


async def test_dynamic_capability_contributes_toolset() -> None:
    """The resolved toolset is exposed once while instructions and settings still apply."""
    calls = 0
    toolset = FunctionToolset()

    @toolset.tool_plain
    def special() -> str:
        return 'used'

    @dataclass
    class ToolCap(AbstractCapability):
        def get_instructions(self) -> str:
            return 'Use the special tool.'

        def get_model_settings(self) -> _ModelSettings:
            return _ModelSettings(temperature=0.25)

        def get_toolset(self) -> AbstractToolset[Any]:
            return toolset

    def factory(ctx: RunContext[bool]) -> AbstractCapability[Any] | None:
        nonlocal calls
        calls += 1
        return ToolCap() if ctx.deps else None

    seen_tools: list[str] = []
    seen_temperatures: list[float | None] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.append(','.join(sorted(t.name for t in info.function_tools)))
        seen_temperatures.append(info.model_settings.get('temperature') if info.model_settings else None)
        # On the first request call the tool if it's available; on the follow-up
        # request after the tool return, finish.
        already_called = any(
            isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts
        )
        if not already_called and any(t.name == 'special' for t in info.function_tools):
            return ModelResponse(parts=[ToolCallPart('special')])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), deps_type=bool, capabilities=[factory])

    with_tool = await agent.run('hi', deps=True)
    tool_returns = [
        p.content
        for m in with_tool.all_messages()
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ]
    assert tool_returns == ['used']
    first_request = next(m for m in with_tool.all_messages() if isinstance(m, ModelRequest))
    assert first_request.instructions == 'Use the special tool.'

    await agent.run('hi', deps=False)
    assert seen_tools == ['special', 'special', '']
    assert seen_temperatures == [0.25, 0.25, None]
    assert calls == 2


async def test_dynamic_capability_contributes_toolset_function() -> None:
    """A resolved capability may contribute a toolset *function*; it's evaluated with the run context."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def func_tool() -> str:
        # The tool listing is what's asserted.
        return 'from func'  # pragma: no cover

    @dataclass
    class AsyncToolFuncCap(AbstractCapability):
        def get_toolset(self):
            async def toolset_func(ctx: RunContext[Any]) -> AbstractToolset[Any] | None:
                return toolset if ctx.deps else None

            return toolset_func

    @dataclass
    class SyncToolFuncCap(AbstractCapability):
        def get_toolset(self):
            def toolset_func(ctx: RunContext[Any]) -> AbstractToolset[Any] | None:
                return toolset

            return toolset_func

    seen_tools: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.append(','.join(sorted(t.name for t in info.function_tools)))
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(respond),
        deps_type=bool,
        capabilities=[DynamicCapability(lambda ctx: AsyncToolFuncCap())],
    )
    await agent.run('hi', deps=True)
    await agent.run('hi', deps=False)

    sync_agent = Agent(
        FunctionModel(respond),
        deps_type=bool,
        capabilities=[DynamicCapability(lambda ctx: SyncToolFuncCap())],
    )
    await sync_agent.run('hi', deps=True)
    assert seen_tools == ['func_tool', '', 'func_tool']


async def test_dynamic_capability_instructions_and_tools_share_resolved_state() -> None:
    """Instructions and tools observe the *same* resolved capability instance per run.

    The factory allocates fresh state on every call, so if the contributed toolset were
    resolved through a second factory invocation, the tool would see different state than
    the instructions.
    """
    resolution_count = 0

    @dataclass
    class StatefulCap(AbstractCapability):
        token: str = ''

        def get_instructions(self) -> str:
            return f'Token is {self.token}.'

        def get_toolset(self):
            toolset = FunctionToolset()

            @toolset.tool_plain
            def read_token() -> str:
                return self.token

            return toolset

    def factory(ctx: RunContext[Any]) -> AbstractCapability[Any]:
        nonlocal resolution_count
        resolution_count += 1
        return StatefulCap(token=f'run-{resolution_count}')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not tool_returns:
            return ModelResponse(parts=[ToolCallPart(tool_name='read_token', args={}, tool_call_id='read')])
        return make_text_response(str(tool_returns[0].content))

    agent = Agent(FunctionModel(respond), capabilities=[factory])
    result = await agent.run('hi')
    first_request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert first_request.instructions == 'Token is run-1.'
    assert result.output == 'run-1'
    assert resolution_count == 1


async def test_dynamic_capability_returning_deferred_capability() -> None:
    """A factory-returned deferred capability keeps its tools hidden until `load_capability`."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def hidden_tool() -> str:
        return 'now visible'

    def factory(ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return Capability(
            id='skills',
            description='Deferred skills.',
            toolsets=[toolset],
            defer_loading=True,
        )

    seen_defer_flags: list[bool] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if hidden_def := next((t for t in info.function_tools if t.name == 'hidden_tool'), None):
            # Authored deferral remains stable after the capability is loaded.
            seen_defer_flags.append(hidden_def.defer_loading)
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'skills'}, tool_call_id='load')]
            )
        if not any(part.tool_name == 'hidden_tool' for part in tool_returns):
            return ModelResponse(parts=[ToolCallPart(tool_name='hidden_tool', args={}, tool_call_id='use')])
        return make_text_response('done')

    agent = Agent(FunctionModel(respond), capabilities=[factory])
    result = await agent.run('hi')
    assert result.output == 'done'
    assert seen_defer_flags == [True, True]


async def test_dynamic_capability_hooks_fire() -> None:
    """Hooks contributed by the resolved capability fire during the run."""
    cap = _RecordingCapability(label='dyn')

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return cap

    agent = Agent(TestModel(), capabilities=[factory])
    await agent.run('hi')
    assert 'dyn:before_run' in cap.fired
    assert 'dyn:before_model_request' in cap.fired


async def test_dynamic_capability_factory_called_once_per_run_not_per_step() -> None:
    """The factory is called once at for_run, not on every model request."""
    calls = 0

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        nonlocal calls
        calls += 1
        return _RecordingCapability(label='once')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Two-step run: first a tool call, then a final text response.
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('echo', {'text': 'hi'})])
        return ModelResponse(parts=[TextPart('done')])

    toolset = FunctionToolset()

    @toolset.tool_plain
    def echo(text: str) -> str:
        return text

    agent = Agent(FunctionModel(respond), toolsets=[toolset], capabilities=[factory])
    await agent.run('hi')
    assert calls == 1


async def test_dynamic_capability_returning_combined() -> None:
    """A factory may return a CombinedCapability; all child contributions flow through."""
    fired: list[str] = []

    @dataclass
    class A(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            fired.append('A')

    @dataclass
    class B(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            fired.append('B')

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return CombinedCapability([A(), B()])

    agent = Agent(TestModel(), capabilities=[factory])
    await agent.run('hi')
    assert fired == ['A', 'B']


async def test_dynamic_deferred_capability_returned_from_custom_init_requires_stable_id() -> None:
    """Deferred capability validation also catches custom init objects returned at run time."""

    @dataclass(init=False)
    class CustomInitDeferredCap(AbstractCapability):
        def __init__(self) -> None:
            self.defer_loading = True

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return CustomInitDeferredCap()

    agent = Agent(FunctionModel(lambda _messages, _info: make_text_response('done')), capabilities=[factory])

    with pytest.raises(UserError, match='stable explicit `id` values'):
        await agent.run('hi')


async def test_dynamic_deferred_capability_uses_resolved_capability_for_loaded_tools() -> None:
    """A loaded dynamic deferred capability exposes tools from the resolved capability."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed'

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return Capability[object](
            id='dynamic-refunds',
            description='Refund policy tools.',
            toolsets=[toolset],
            defer_loading=True,
        )

    seen_tool_state: list[list[tuple[str, bool]]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tool_state.append([(t.name, bool(t.defer_loading)) for t in info.function_tools])
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))

        if not any(
            isinstance(part, LoadCapabilityReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'dynamic-refunds'},
                        tool_call_id='load-dynamic-refunds',
                    )
                ]
            )

        if not any(part.tool_name == 'lookup_refund_policy' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_refund_policy',
                        args={'order_id': 'order-123'},
                        tool_call_id='lookup-refund',
                    )
                ]
            )

        refund_result = next(part.content for part in tool_returns if part.tool_name == 'lookup_refund_policy')
        return make_text_response(f'done: {refund_result}')

    agent = Agent(FunctionModel(model_fn), capabilities=[factory])
    result = await agent.run('Can I get a refund?')

    assert result.output == 'done: order-123: refund allowed'
    assert seen_tool_state == snapshot(
        [
            [('load_capability', False)],
            [('load_capability', False), ('lookup_refund_policy', True)],
            [('load_capability', False), ('lookup_refund_policy', True)],
        ]
    )


async def test_dynamic_capability_in_run_call() -> None:
    """`agent.run(capabilities=[factory])` accepts callables as well."""
    calls = 0

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        nonlocal calls
        calls += 1
        return _RecordingCapability(label='run-time')

    agent = Agent(TestModel())
    result = await agent.run('hi', capabilities=[factory])
    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Label is run-time.'
    assert calls == 1


async def test_dynamic_capability_composes_with_static() -> None:
    """Static and dynamic capabilities both contribute, in order."""
    fired: list[str] = []

    @dataclass
    class Static(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            fired.append('static')

    @dataclass
    class Dynamic(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            fired.append('dynamic')

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return Dynamic()

    agent = Agent(TestModel(), capabilities=[Static(), factory])
    await agent.run('hi')
    assert fired == ['static', 'dynamic']


async def test_dynamic_capability_per_run_isolation() -> None:
    """Concurrent runs see independent factory calls and resolved capabilities."""
    seen_deps: list[str] = []

    def factory(ctx: RunContext[str]) -> AbstractCapability[Any]:
        seen_deps.append(ctx.deps)
        return _RecordingCapability(label=ctx.deps)

    agent = Agent(TestModel(), deps_type=str, capabilities=[factory])
    results = await asyncio.gather(*(agent.run('hi', deps=f'user-{i}') for i in range(5)))

    assert sorted(seen_deps) == ['user-0', 'user-1', 'user-2', 'user-3', 'user-4']
    for i, result in enumerate(results):
        request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
        assert request.instructions == f'Label is user-{i}.'


async def test_dynamic_capability_wraps_func_in_constructor() -> None:
    """Constructor wraps a bare function into a `DynamicCapability`, and the factory runs at run time."""

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return _RecordingCapability(label='x')

    agent = Agent(TestModel(), capabilities=[factory])

    result = await agent.run('hi')
    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Label is x.'


def test_dynamic_capability_rejects_wrapper_fields() -> None:
    """`defer_loading` on the wrapper would otherwise be silently ignored — reject at construction."""

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return _RecordingCapability(label='x')  # pragma: no cover

    with pytest.raises(UserError, match='not supported on `DynamicCapability`'):
        DynamicCapability(factory, defer_loading=True)


# endregion


async def test_combined_capability_subclass_custom_init_for_run() -> None:
    """`CombinedCapability` subclasses with a custom `__init__` don't crash in `for_run` when a child returns a fresh instance.

    Regression test for #6674: `dataclasses.replace` reconstructed through the subclass
    `__init__`, which does not accept the `capabilities` kwarg.
    """

    @dataclass
    class PerRunLeaf(AbstractCapability[Any]):
        n: int = 0

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return PerRunLeaf(n=self.n + 1)

        def get_instructions(self) -> str:
            return f'leaf {self.n}'

    class CombinedSubclass(CombinedCapability[Any]):
        """Bundle a leaf behind a friendly constructor without exposing `capabilities`."""

        def __init__(self, *, size: int = 3) -> None:
            self.post_init_calls = 0
            super().__init__(capabilities=[PerRunLeaf(n=size)])

        def __post_init__(self) -> None:
            self.post_init_calls += 1
            super().__post_init__()

    combined = CombinedSubclass(size=5)
    ctx = _build_run_context()

    result = await combined.for_run(ctx)

    assert isinstance(result, CombinedSubclass)
    assert result is not combined
    assert result.post_init_calls == 1
    leaf = result.capabilities[0]
    assert isinstance(leaf, PerRunLeaf)
    assert leaf.n == 6
    # Exercising `get_instructions` also covers the leaf's instruction emission.
    assert leaf.get_instructions() == 'leaf 6'


def test_combined_capability_subclass_custom_init_for_agent() -> None:
    """`CombinedCapability` subclasses with a custom `__init__` don't crash in `for_agent` when a child returns a fresh instance.

    Regression test for #6674.
    """

    @dataclass
    class BindingLeaf(AbstractCapability[Any]):
        bound: bool = False

        def for_agent(self, agent: AbstractAgent[Any, Any]) -> AbstractCapability[Any]:
            return replace(self, bound=True)

    class CombinedSubclass(CombinedCapability[Any]):
        def __init__(self) -> None:
            super().__init__(capabilities=[BindingLeaf()])

    combined = CombinedSubclass()
    agent = Agent('test')

    bound = combined.for_agent(agent)

    assert isinstance(bound, CombinedSubclass)
    assert bound is not combined
    bound_leaf = bound.capabilities[0]
    assert isinstance(bound_leaf, BindingLeaf)
    assert bound_leaf.bound is True


async def test_wrapper_capability_subclass_custom_init_rebinds_wrapped() -> None:
    """`WrapperCapability` subclasses with a custom `__init__` survive both binding paths.

    Same `dataclasses.replace`-through-subclass-`__init__` defect as #6674, on the sibling
    container: `WrapperCapability` rebuilt itself with `replace(self, wrapped=...)`, which the
    subclass constructor can't accept. Driven through `Agent` because — unlike
    `CombinedCapability`, whose `__post_init__` splats a nested subclass away — a wrapper
    reaches both `for_agent` (agent construction) and `for_run` (per-run) intact.
    """

    @dataclass
    class PerRunLeaf(AbstractCapability[Any]):
        n: int = 0
        bound: bool = False

        def for_agent(self, agent: AbstractAgent[Any, Any]) -> AbstractCapability[Any]:
            return replace(self, bound=True)

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return replace(self, n=self.n + 1)

        def get_instructions(self) -> str:
            return f'leaf {self.n}'

    class WrapperSubclass(WrapperCapability[Any]):
        """Bundle a leaf behind a friendly constructor without exposing `wrapped`."""

        def __init__(self, *, size: int = 3) -> None:
            self.post_init_calls = 0
            super().__init__(wrapped=PerRunLeaf(n=size))

        def __post_init__(self) -> None:
            self.post_init_calls += 1
            super().__post_init__()

    agent = Agent('test', capabilities=[WrapperSubclass(size=5)])
    result = await agent.run('hi')

    # `for_agent` bound the leaf at construction, then `for_run` incremented it for this run,
    # and the wrapper delegated the resulting instructions through both rebuilds.
    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions == 'leaf 6'
    wrapper = next(cap for cap in agent.root_capability.capabilities if isinstance(cap, WrapperSubclass))
    assert wrapper.post_init_calls == 1


async def test_wrapper_capability_subclass_custom_init_preserves_type_and_id() -> None:
    """Rebuilding a `WrapperCapability` keeps the subclass type and re-resolves the adopted `id`.

    Pins transparent-wrapper identity re-resolution: a wrapper without an explicit `id` adopts
    the wrapped capability's `id`, which is only known after `for_run` has produced the new
    wrapped instance.
    """

    @dataclass
    class IdentifiedLeaf(AbstractCapability[Any]):
        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return IdentifiedLeaf(id='resolved-at-run-time')

    class WrapperSubclass(WrapperCapability[Any]):
        def __init__(self, *, size: int = 3) -> None:
            super().__init__(wrapped=IdentifiedLeaf())
            self.size = size

    wrapper = WrapperSubclass(size=5)
    assert wrapper.id is None

    rebuilt = await wrapper.for_run(_build_run_context())

    assert isinstance(rebuilt, WrapperSubclass)
    assert rebuilt is not wrapper
    assert rebuilt.size == 5, 'subclass-only attributes must survive the rebuild'
    assert rebuilt.id == 'resolved-at-run-time'
    assert wrapper.id is None, 'the original must not be mutated'


async def test_wrapper_capability_subclass_derived_state_contract() -> None:
    """Pins the documented rebind contract for subclass state.

    A rebind shallow-copies the wrapper without re-running `__init__`/`__post_init__`, so
    values derived from `wrapped` must be computed on access to stay fresh — an eager cache
    made at construction is carried over verbatim and reflects the pre-rebind wrapped.
    """

    @dataclass
    class PerRunLeaf(AbstractCapability[Any]):
        n: int = 0

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return PerRunLeaf(n=self.n + 1)

    class SummarizingWrapper(WrapperCapability[Any]):
        def __init__(self, leaf: PerRunLeaf) -> None:
            super().__init__(wrapped=leaf)
            self.cached_summary = self.summary

        @property
        def summary(self) -> str:
            assert isinstance(self.wrapped, PerRunLeaf)
            return f'wrapping leaf {self.wrapped.n}'

    wrapper = SummarizingWrapper(PerRunLeaf(n=1))
    rebound = await wrapper.for_run(_build_run_context())

    assert isinstance(rebound, SummarizingWrapper)
    assert rebound.summary == 'wrapping leaf 2', 'computed-on-access state re-derives from the new wrapped'
    assert rebound.cached_summary == 'wrapping leaf 1', 'eagerly cached state is carried over verbatim'
    assert wrapper.summary == 'wrapping leaf 1', 'the original must not be mutated'


async def test_tool_return_cannot_reveal_capability_owned_tools_without_loading() -> None:
    """A bare-name reveal of a capability tool would skip the capability's hooks and instructions.

    `load_capability` activates the whole bundle; `ToolReturn.tools` naming a capability-owned tool
    while its capability is unloaded is rejected so the tool can never become callable with its
    capability's `before_tool_validate`/`before_tool_execute` hooks and instructions inactive.
    """
    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain(name='capability_tool')
    def capability_tool() -> str:  # pragma: no cover
        return 'refund'

    refunds = Capability[object](id='refunds', toolsets=[refunds_toolset], defer_loading=True)

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if not list(iter_message_parts(messages, ModelRequest, ToolReturnPart)):
            return ModelResponse(parts=[ToolCallPart(tool_name='reveal_it', args={}, tool_call_id='reveal')])
        return make_text_response('done')  # pragma: no cover - the run raises before a second model call

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])

    @agent.tool_plain
    def reveal_it() -> ToolReturn[str]:
        return ToolReturn(return_value='revealed', tools=['capability_tool'])

    with pytest.raises(UserError, match=r"belongs to capability 'refunds', which must be loaded"):
        await agent.run('Reveal the capability tool directly.')


async def test_tool_return_can_reveal_capability_owned_tools_once_loaded() -> None:
    """After `load_capability`, naming a capability tool in `ToolReturn.tools` is a legal no-op-ish reveal."""
    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain(name='capability_tool')
    def capability_tool() -> str:  # pragma: no cover
        return 'refund'

    refunds = Capability[object](id='refunds', toolsets=[refunds_toolset], defer_loading=True)

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not returns:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='l1')]
            )
        if not any(part.tool_name == 'reveal_it' for part in returns):
            return ModelResponse(parts=[ToolCallPart(tool_name='reveal_it', args={}, tool_call_id='r1')])
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])

    @agent.tool_plain
    def reveal_it() -> ToolReturn[str]:
        return ToolReturn(return_value='revealed', tools=['capability_tool'])

    result = await agent.run('Load, then reveal by name.')
    assert result.output == 'done'
