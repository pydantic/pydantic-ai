from __future__ import annotations

import asyncio
import re
import sys
import uuid
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest
from pydantic_core import PydanticSerializationError

from pydantic_ai import (
    AbstractToolset,
    Agent,
    AgentStreamEvent,
    BinaryImage,
    DocumentUrl,
    ExternalToolset,
    FunctionToolset,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    MultiModalContent,
    PartStartEvent,
    RequestUsage,
    RunContext,
    RunUsage,
    TextPart,
    Tool,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    ToolsetTool,
    UserPromptPart,
)
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import (
    MCP,
    Capability,
)
from pydantic_ai.exceptions import (
    ApprovalRequired,
    UserError,
)
from pydantic_ai.models import (
    ModelRequestContext,
    ModelRequestParameters,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.external import TOOL_SCHEMA_VALIDATOR

from ..._inline_snapshot import snapshot

try:
    from temporalio import workflow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.client import Client, WorkflowHistory
    from temporalio.common import RetryPolicy
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.testing import ActivityEnvironment
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
    from temporalio.workflow import ActivityCancellationType, ActivityConfig

    from pydantic_ai.durable_exec._toolset import (
        CallToolResult,
        unwrap_tool_call_result,
        wrap_tool_call_result,
    )
    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        TemporalAgent,  # pyright: ignore[reportDeprecated]
        TemporalDurability,
    )
    from pydantic_ai.durable_exec.temporal._durability import (
        _CancelParams,  # pyright: ignore[reportPrivateUsage]
        _EventStreamHandlerParams,  # pyright: ignore[reportPrivateUsage]
        _RequestParams,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.durable_exec.temporal._dynamic_toolset import temporalize_dynamic_toolset
    from pydantic_ai.durable_exec.temporal._function_toolset import TemporalFunctionToolset
    from pydantic_ai.durable_exec.temporal._mcp_toolset import TemporalMCPToolset
    from pydantic_ai.durable_exec.temporal._run_context import TemporalRunContext
    from pydantic_ai.durable_exec.temporal._toolset import (
        CallToolParams,
        GetToolsParams,
        TemporalWrapperToolset,
        heartbeating,
        resolve_tool_activity_config,
        toolset_temporal_activities,
    )
    from pydantic_ai.durable_exec.temporal._transports import (
        _CompactMessagesParams,  # pyright: ignore[reportPrivateUsage]
    )

except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)


# The 3.14 durable-exec CI leg takes this skip; every other leg falls through. `lax` rather than
# plain because which of the two arms a run measures depends on its Python version.
if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

try:
    import logfire  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError:  # pragma: lax no cover
    pytest.skip('logfire not installed', allow_module_level=True)

try:
    from fastmcp.client.transports import StdioTransport

    from pydantic_ai.mcp import MCPToolset
except ImportError:  # pragma: lax no cover
    pytest.skip('mcp not installed', allow_module_level=True)

try:
    from pydantic_ai.models.openai import OpenAIChatModel  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError:  # pragma: lax no cover
    pytest.skip('openai not installed', allow_module_level=True)


with workflow.unsafe.imports_passed_through():
    # Workaround for a race condition when running `logfire.info` inside an activity with attributes to serialize and pandas importable:
    # AttributeError: partially initialized module 'pandas' has no attribute '_pandas_parser_CAPI' (most likely due to a circular import)
    try:
        import pandas  # pyright: ignore[reportUnusedImport] # noqa: F401
    except ImportError:  # pragma: lax no cover
        pass

    # https://github.com/temporalio/sdk-python/blob/3244f8bffebee05e0e7efefb1240a75039903dda/tests/test_client.py#L112C1-L113C1
    from mcp.client.session import ClientSession
    from mcp.types import ClientRequest

    from ..._inline_snapshot import snapshot

    # Loads `vcr`, which Temporal doesn't like without passing through the import
    from ...conftest import IsDatetime, IsStr, message

    # `_shared` loads the same sandbox-sensitive modules, so import it passed-through as well.
    from ._shared import (
        BASE_ACTIVITY_CONFIG,
        TASK_QUEUE,
        ComplexAgentWorkflow,
        Deps,
        DynamicToolsetDeps,
        complex_agent,
        complex_temporal_agent,
        dynamic_toolset_temporal_agent,
        model,
        payload_limit_detail,
        simple_temporal_agent,
        workflow_raises,
    )


# `TemporalAgent` is deprecated in favor of `capabilities=[TemporalDurability(...)]`.
# These tests exercise the wrapper-agent path on purpose; suppress the warning here
# rather than globally in `pyproject.toml`. The `pytestmark` entry below covers warnings
# emitted *inside* test functions; the `filterwarnings` call below covers warnings emitted
# at module import time (e.g. module-level construction of `TemporalAgent`).
warnings.filterwarnings('ignore', message='`TemporalAgent` is deprecated', category=PydanticAIDeprecationWarning)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.xdist_group(name='temporal-toolsets'),
    pytest.mark.filterwarnings(
        'ignore:`TemporalAgent` is deprecated:pydantic_ai._warnings.PydanticAIDeprecationWarning'
    ),
]


async def test_mcp_tools_cached_across_activities(allow_model_requests: None, client: Client):
    """Verify that MCP tool caching reduces server round-trips across activities.

    The complex agent makes 3 model requests, each preceded by a get_tools activity.
    With the run-scoped tool-defs cache, only the first get_tools activity actually runs
    (opening an MCP connection and calling `tools/list`). Subsequent get_tools calls return
    the run-cached tool definitions without scheduling an activity at all.
    """

    original_send_request = ClientSession.send_request
    methods_called: list[str] = []

    async def tracking_send_request(self_: ClientSession, request: ClientRequest, *args: Any, **kwargs: Any) -> Any:
        methods_called.append(request.root.method)
        return await original_send_request(self_, request, *args, **kwargs)

    with patch.object(ClientSession, 'send_request', tracking_send_request):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[ComplexAgentWorkflow],
            plugins=[AgentPlugin(complex_temporal_agent)],
        ):
            coro = client.execute_workflow(
                ComplexAgentWorkflow.run,
                args=[
                    'Tell me: the capital of the country; the weather there; the product name',
                    Deps(country='Mexico'),
                ],
                id=f'{ComplexAgentWorkflow.__name__}_cache_test',
                task_queue=TASK_QUEUE,
            )
            output = await coro
        assert output is not None

    # 3 get_tools calls are made, but only 1 results in an actual tools/list MCP request
    assert methods_called.count('tools/list') == 1
    # call_tool should still make a request each time (not cached)
    assert methods_called.count('tools/call') == 1


def _call_mcp_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Two model steps: call an MCP tool on the first request, return text on the second.

    Two model requests means `get_tools` is invoked twice on the MCP toolset within one run,
    so the run-scoped cache (and the activity it does or doesn't schedule each step) is exercised.
    """
    tool_returned = any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
    if tool_returned:
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('get_weather_forecast', {'location': 'Mexico City'})])


# A holder lets the replay step swap in a freshly-constructed (cold-process) instance,
# reproducing the worker-restart scenario from #5875.
mcp_replay_holder: dict[str, TemporalAgent[None, str]] = {}  # pyright: ignore[reportDeprecated]


def _make_mcp_replay_agent(cache_tools: bool = True) -> TemporalAgent[None, str]:  # pyright: ignore[reportDeprecated]
    agent = Agent(
        FunctionModel(_call_mcp_then_finish),
        name='mcp_replay_agent',
        toolsets=[
            MCPToolset(
                StdioTransport(command='python', args=['-m', 'tests.mcp_server']),
                id='mcp',
                init_timeout=20,
                cache_tools=cache_tools,
            )
        ],
    )
    return TemporalAgent(agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


mcp_replay_holder['agent'] = _make_mcp_replay_agent()


@workflow.defn
class MCPReplayWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await mcp_replay_holder['agent'].run(prompt)
        return result.output


def _scheduled_get_tools_count(history: WorkflowHistory) -> int:
    return sum(
        1
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
        and event.activity_task_scheduled_event_attributes.activity_type.name.endswith('__get_tools')
    )


async def test_temporal_mcp_get_tools_replay_deterministic(allow_model_requests: None, client: Client):
    """#5875 regression: `get_tools` activity scheduling must be replay-deterministic.

    The tool-defs cache must not let shared-process cache warmth decide whether a workflow
    emits a `get_tools` activity command — otherwise a history recorded on a warm worker fails
    replay on a cold one (and vice versa) with `TMPRL1100`. Each run must independently record
    exactly one `get_tools` activity: the #4331 within-run win (N calls collapse to one activity)
    without leaking cache state across the replay boundary.
    """
    warm = _make_mcp_replay_agent()
    mcp_replay_holder['agent'] = warm

    histories: list[WorkflowHistory] = []
    # Unsandboxed so the module-level instance (and its cache) is shared across both runs,
    # exactly as a long-running worker process shares it in production — the condition under
    # which #5875 records a warm run with no `get_tools` event.
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MCPReplayWorkflow],
        activities=warm.temporal_activities,
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        for i in range(2):
            wf_id = f'{MCPReplayWorkflow.__name__}_{i}'
            await client.execute_workflow(MCPReplayWorkflow.run, args=['hello'], id=wf_id, task_queue=TASK_QUEUE)
            histories.append(await client.get_workflow_handle(wf_id).fetch_history())
    h1, h2 = histories

    # Within a run, the run-scoped cache collapses the per-step `get_tools` calls to one activity...
    assert _scheduled_get_tools_count(h1) == 1
    # ...and each run records it independently — run 2 does not inherit run 1's warm process cache.
    assert _scheduled_get_tools_count(h2) == 1

    def replayer() -> Replayer:
        return Replayer(
            workflows=[MCPReplayWorkflow],
            workflow_runner=UnsandboxedWorkflowRunner(),
            data_converter=pydantic_data_converter,
        )

    try:
        # Direction 1: cold-recorded history (run 1) replayed after the process cache warmed
        # (the same-process sticky-cache-eviction trigger). Holder still points at the warm instance.
        await replayer().replay_workflow(h1)

        # Direction 2: warm-recorded history (run 2) replayed on a freshly-constructed cold instance
        # (the worker-restart trigger).
        mcp_replay_holder['agent'] = _make_mcp_replay_agent()
        await replayer().replay_workflow(h2)
    finally:
        mcp_replay_holder['agent'] = warm


async def test_temporal_mcp_get_tools_not_cached_when_disabled(allow_model_requests: None, client: Client):
    """With `cache_tools=False`, `get_tools` is scheduled for every model request (no run cache).

    The complementary case to the run-scoped cache: each of the two model requests records its own
    `get_tools` activity, so disabling the cache stays replay-deterministic by always scheduling.
    """
    agent = _make_mcp_replay_agent(cache_tools=False)
    mcp_replay_holder['agent'] = agent
    try:
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[MCPReplayWorkflow],
            activities=agent.temporal_activities,
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            wf_id = f'{MCPReplayWorkflow.__name__}_no_cache'
            await client.execute_workflow(MCPReplayWorkflow.run, args=['hello'], id=wf_id, task_queue=TASK_QUEUE)
            history = await client.get_workflow_handle(wf_id).fetch_history()
        assert _scheduled_get_tools_count(history) == 2
    finally:
        mcp_replay_holder['agent'] = _make_mcp_replay_agent()


async def test_old_temporalize_toolset_func_compat():
    """Old 6-arg temporalize_toolset_func implementations still work."""
    from pydantic_ai.durable_exec.temporal._toolset import temporalize_toolset

    def old_style_func(
        toolset: Any, prefix: Any, config: Any, tool_config: Any, deps_type: Any, run_context_type: Any
    ) -> Any:
        return temporalize_toolset(toolset, prefix, config, tool_config, deps_type, run_context_type)

    TemporalAgent(  # pyright: ignore[reportDeprecated]
        Agent(model=model, name='old_compat_agent'),
        activity_config=BASE_ACTIVITY_CONFIG,
        temporalize_toolset_func=old_style_func,  # pyright: ignore[reportArgumentType]
    )


async def test_toolset_without_id():
    with pytest.raises(
        UserError,
        match=re.escape(
            "Toolsets that are 'leaves' (i.e. those that implement their own tool listing and calling) need to have a unique `id` in order to be used with Temporal. The ID will be used to identify the toolset's activities within the workflow."
        ),
    ):
        TemporalAgent(Agent(model=model, name='test_agent', toolsets=[FunctionToolset()]))  # pyright: ignore[reportDeprecated]


async def test_capability_contributed_toolset_id_from_capability():
    """A capability's `id` flows to its contributed leaf toolset, so combining a capability with a
    function toolset or MCP server can be used under Temporal instead of tripping the
    'leaves need a unique id' error at construction.

    This isn't a VCR test: it inspects the constructed toolset tree and registered Temporal activity
    names during local agent construction, before any model or MCP request, so there's no network
    round-trip to record.

    Regression for https://github.com/pydantic/pydantic-ai/issues/6334.
    """

    def add(x: int) -> int:
        return x + 1  # pragma: no cover

    agent = Agent(
        model,
        name='capability_agent',
        capabilities=[
            Capability(id='billing', tools=[add]),
            MCP(url='https://mcp.example.com/api', id='docs'),
        ],
    )
    # Previously raised `UserError` because the contributed leaf toolsets had `id=None`.
    temporal_agent = TemporalAgent(agent)  # pyright: ignore[reportDeprecated]

    # Each contributed leaf toolset is registered as activities named after the capability id, so the
    # function toolset and the MCP server can be driven durably.
    activity_names = {
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in temporal_agent.temporal_activities
    }
    assert 'agent__capability_agent__toolset__billing__call_tool' in activity_names
    assert 'agent__capability_agent__mcp_server__docs__get_tools' in activity_names


async def test_deferred_capability_contributed_toolset_id_from_capability():
    """A deferred capability (`defer_loading=True`) still stamps its `id` on the contributed leaf
    toolset, so the derived id survives the deferred-loading wrapper and the toolset is registered as
    durable activities. Deferred capabilities require an explicit `id`.

    This isn't a VCR test: it inspects deferred toolset ids and registered Temporal activity names
    during local agent construction, before any model or MCP request, so there's no network round-trip
    to record.

    Regression for https://github.com/pydantic/pydantic-ai/issues/6334.
    """

    def add(x: int) -> int:
        return x + 1  # pragma: no cover

    agent = Agent(
        model,
        name='deferred_capability_agent',
        capabilities=[
            Capability(id='billing', tools=[add], defer_loading=True),
            MCP(url='https://mcp.example.com/api', id='docs', defer_loading=True),
        ],
    )
    temporal_agent = TemporalAgent(agent)  # pyright: ignore[reportDeprecated]

    activity_names = {
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in temporal_agent.temporal_activities
    }
    assert 'agent__deferred_capability_agent__toolset__billing__call_tool' in activity_names
    assert 'agent__deferred_capability_agent__mcp_server__docs__get_tools' in activity_names


@workflow.defn
class DynamicToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: DynamicToolsetDeps) -> str:
        result = await dynamic_toolset_temporal_agent.run(prompt, deps=deps)
        return result.output


async def test_dynamic_toolset_in_workflow(client: Client):
    """Test that @agent.toolset works correctly in a Temporal workflow."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DynamicToolsetAgentWorkflow],
        plugins=[AgentPlugin(dynamic_toolset_temporal_agent)],
    ):
        output = await client.execute_workflow(
            DynamicToolsetAgentWorkflow.run,
            args=['Get the weather for London', DynamicToolsetDeps(user_name='Alice')],
            id='test_dynamic_toolset_workflow',
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('{"get_dynamic_weather":"Weather in a for Alice: sunny."}')


async def test_dynamic_toolset_outside_workflow():
    """Test that the dynamic toolset agent works correctly outside of a workflow."""
    result = await dynamic_toolset_temporal_agent.run(
        'Get the weather for Paris', deps=DynamicToolsetDeps(user_name='Bob')
    )
    assert result.output == snapshot('{"get_dynamic_weather":"Weather in a for Bob: sunny."}')


# --- DynamicToolset.get_instructions test (issue #5282) ---
# A dynamic toolset whose resolved toolset implements `get_instructions()` must contribute those
# instructions under `TemporalAgent`, resolved inside an activity like `get_tools`.


def _echo_instructions(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    request = message(messages, ModelRequest, index=-1)
    return ModelResponse(parts=[TextPart(request.instructions or '<no instructions>')])


dynamic_instructions_agent = Agent(FunctionModel(_echo_instructions), name='dynamic_instructions_agent')


@dynamic_instructions_agent.toolset(id='dynamic_instruction_toolset', per_run_step=False)
def dynamic_instruction_toolset(ctx: RunContext[object]) -> AbstractToolset[object]:
    # A toolset that only contributes instructions, no tools.
    return FunctionToolset(instructions='SENTINEL_INSTRUCTION_FROM_DYNAMIC_TOOLSET', id='instruction-only-toolset')


dynamic_instructions_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    dynamic_instructions_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class DynamicInstructionsAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await dynamic_instructions_temporal_agent.run(prompt)
        return result.output


async def test_dynamic_toolset_instructions_in_workflow(allow_model_requests: None, client: Client):
    """A dynamic toolset's `get_instructions()` reaches the model under `TemporalAgent` (issue #5282).

    The model echoes the request's instructions back as its output, so the sentinel in the output
    proves the resolved dynamic toolset's instructions were collected via the new activity.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DynamicInstructionsAgentWorkflow],
        plugins=[AgentPlugin(dynamic_instructions_temporal_agent)],
    ):
        output = await client.execute_workflow(
            DynamicInstructionsAgentWorkflow.run,
            args=['hello'],
            id='test_dynamic_toolset_instructions_workflow',
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('SENTINEL_INSTRUCTION_FROM_DYNAMIC_TOOLSET')


def test_dynamic_toolset_temporal_activities():
    """The temporalized dynamic toolset collects instructions inside `get_tools`, so it has no separate `get_instructions` activity."""
    activity_names = {
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in dynamic_instructions_temporal_agent.temporal_activities
    }
    prefix = 'agent__dynamic_instructions_agent__dynamic_toolset__dynamic_instruction_toolset'
    assert {f'{prefix}__get_tools', f'{prefix}__call_tool'} <= activity_names
    assert f'{prefix}__get_instructions' not in activity_names


async def test_temporal_wrapper_toolset_extension_surface():
    """`TemporalWrapperToolset` stays the base for custom `temporalize_toolset_func` toolsets.

    No in-core toolset subclasses it anymore (the factories build the shared durable toolsets),
    but it remains public for the deprecated `TemporalAgent`'s `temporalize_toolset_func`
    extension point, so its surface is pinned here the way a custom subclass would use it.
    """

    def sentinel_activity() -> None: ...  # pragma: no cover

    class _CustomTemporalToolset(TemporalWrapperToolset[None]):
        @property
        def temporal_activities(self) -> list[Callable[..., Any]]:
            return [sentinel_activity]

    toolset = _CustomTemporalToolset(FunctionToolset[None](id='custom_wrapped'))
    assert toolset.id == 'custom_wrapped'
    assert toolset_temporal_activities(toolset) == [sentinel_activity]

    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    assert await toolset.for_run_step(ctx) is toolset

    # Outside a workflow the wrapper enters/exits its wrapped toolset; inside one, both are no-ops.
    async with toolset:
        pass
    with patch('pydantic_ai.durable_exec.temporal._toolset.workflow.in_workflow', return_value=True):
        assert await toolset.__aenter__() is toolset
        assert await toolset.__aexit__(None, None, None) is None

    async def return_value() -> str:
        return 'value'

    wrapped_result = await toolset._wrap_call_tool_result(return_value())  # pyright: ignore[reportPrivateUsage]
    assert toolset._unwrap_call_tool_result(wrapped_result) == 'value'  # pyright: ignore[reportPrivateUsage]


async def test_temporal_dynamic_toolset_rejects_activity_opt_out():
    """`metadata={'temporal': False}` / config `False` is rejected for dynamic-toolset tools.

    Running such a tool inline would resolve the dynamic toolset and call the tool in
    workflow code, where I/O and thread dispatch are forbidden.
    """
    durable = temporalize_dynamic_toolset(
        DynamicToolset(lambda ctx: None, id='dyn_opt_out'),
        activity_name_prefix='agent__dyn_opt_out',
        activity_config={},
        tool_activity_config={'boom': False},
        deps_type=type(None),
    )
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    tool = ToolsetTool(
        toolset=durable, tool_def=ToolDefinition(name='boom'), max_retries=1, args_validator=TOOL_SCHEMA_VALIDATOR
    )
    with pytest.raises(UserError, match='activity disabled'):
        await durable.call_tool('boom', {}, ctx, tool)


async def test_temporalize_dynamic_toolset_runs_args_validator_in_activity() -> None:
    from pydantic_ai.durable_exec._toolset import get_dynamic_tools

    validated: list[int] = []

    async def tool(value: int) -> int: ...  # pragma: no branch

    def validator(ctx: RunContext[None], value: int) -> None:
        validated.append(value)

    dynamic = DynamicToolset(
        lambda ctx: FunctionToolset([Tool(tool, args_validator=validator)]), id='legacy-validation'
    )
    durable = temporalize_dynamic_toolset(
        dynamic,
        activity_name_prefix='agent__legacy_validation',
        activity_config={},
        tool_activity_config={},
        deps_type=type(None),
    )
    durable._in_durable_context = lambda: True  # pyright: ignore[reportPrivateUsage]
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    activity_calls = 0

    async def run_activity(*, activity: Callable[..., Any], args: Sequence[Any], **config: Any) -> Any:
        nonlocal activity_calls
        activity_calls += 1
        if activity_calls == 1:
            return await get_dynamic_tools(dynamic, ctx)
        return await ActivityEnvironment().run(activity, *args)

    with (
        patch('pydantic_ai.durable_exec.temporal._dynamic_toolset.workflow.in_workflow', return_value=True),
        patch('pydantic_ai.durable_exec.temporal._dynamic_toolset.execute_activity', run_activity),
    ):
        run_toolset = await durable.for_run(ctx)
        tools = await run_toolset.get_tools(ctx)
        assert tools
        resolved = next(iter(tools.values()))
        assert resolved.args_validator_func is not None
        await resolved.args_validator_func(ctx, value=1)

    assert validated == [1]


# --- DynamicToolset instructions refresh across run steps (issue #5282 follow-up) ---
# The per-run instructions cache is written by `get_tools` and read by `get_instructions` each
# step; this guards against it serving a stale step-1 value on a later step.


def _echo_instructions_after_tool_call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # First request: call a tool to force a second model-request step.
    # Second request (carrying the tool return): echo the instructions, which by then must
    # reflect the current step — proving the cache is repopulated by `get_tools` each step.
    request = message(messages, ModelRequest, index=-1)
    if any(isinstance(part, ToolReturnPart) for part in request.parts):
        return ModelResponse(parts=[TextPart(request.instructions or '<no instructions>')])
    return ModelResponse(parts=[ToolCallPart('noop', {})])


multi_step_instructions_agent = Agent(
    FunctionModel(_echo_instructions_after_tool_call), name='multi_step_instructions_agent'
)


@multi_step_instructions_agent.toolset(id='multi_step_instruction_toolset')
def multi_step_instruction_toolset(ctx: RunContext[object]) -> AbstractToolset[object]:
    # Instructions encode the run step, so a stale step-1 cached value read at step 2 would
    # surface as the wrong sentinel in the model output.
    toolset = FunctionToolset[object](
        instructions=f'INSTRUCTIONS_FOR_STEP_{ctx.run_step}', id='step-instruction-toolset'
    )

    @toolset.tool_plain
    def noop() -> str:
        return 'noop'

    return toolset


multi_step_instructions_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    multi_step_instructions_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class MultiStepInstructionsAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await multi_step_instructions_temporal_agent.run(prompt)
        return result.output


async def test_dynamic_toolset_instructions_refresh_across_steps_in_workflow(
    allow_model_requests: None, client: Client
):
    """A dynamic toolset's instructions are refreshed each run step under `TemporalAgent` (issue #5282).

    The toolset encodes the run step in its instructions; the model calls a tool on the first request to
    force a second step, then echoes the instructions on the second request. The output being the step-2
    sentinel (not the step-1 one) proves `get_tools` repopulates the per-run instructions cache each step
    rather than serving a stale step-1 value.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiStepInstructionsAgentWorkflow],
        plugins=[AgentPlugin(multi_step_instructions_temporal_agent)],
    ):
        output = await client.execute_workflow(
            MultiStepInstructionsAgentWorkflow.run,
            args=['hello'],
            id='test_dynamic_toolset_instructions_refresh_workflow',
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('INSTRUCTIONS_FOR_STEP_2')


# --- DynamicToolset instructions replay determinism (issue #5282) ---
# The per-run instructions cache lives on a `for_run` copy of the wrapper rather than on the
# process-shared, module-level instance. A history recorded on one worker must replay on a
# freshly-constructed (cold) one, proving the `for_run` override reconstructs identically and
# introduces no `TMPRL1100` nondeterminism.

# A holder lets the replay step swap in a freshly-constructed (cold-process) instance.
dynamic_instructions_replay_holder: dict[str, TemporalAgent[object, str]] = {}  # pyright: ignore[reportDeprecated]


def _make_dynamic_instructions_replay_agent() -> TemporalAgent[object, str]:  # pyright: ignore[reportDeprecated]
    agent = Agent(FunctionModel(_echo_instructions_after_tool_call), name='dynamic_instructions_replay_agent')

    @agent.toolset(id='replay_instruction_toolset')
    def _replay_toolset(ctx: RunContext[object]) -> AbstractToolset[object]:
        toolset = FunctionToolset[object](
            instructions=f'INSTRUCTIONS_FOR_STEP_{ctx.run_step}', id='step-instruction-toolset'
        )

        @toolset.tool_plain
        def noop() -> str:
            return 'noop'

        return toolset

    return TemporalAgent(agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


dynamic_instructions_replay_holder['agent'] = _make_dynamic_instructions_replay_agent()


@workflow.defn
class DynamicInstructionsReplayWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await dynamic_instructions_replay_holder['agent'].run(prompt)
        return result.output


async def test_dynamic_toolset_instructions_replay_deterministic(allow_model_requests: None, client: Client):
    """The per-run `for_run` instructions cache must be replay-deterministic (issue #5282).

    Instructions resolved by `get_tools` are held on a per-run `for_run` copy of the wrapper, not
    on the module-level instance. This records a two-step workflow (instructions differ per step)
    and replays its history on a freshly-constructed cold instance — the worker-restart scenario —
    asserting no nondeterminism, so the `for_run` copy is reconstructed identically on replay.
    """
    warm = _make_dynamic_instructions_replay_agent()
    dynamic_instructions_replay_holder['agent'] = warm

    # Unsandboxed so the module-level instance is shared across the run exactly as a long-running
    # worker process shares it in production.
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DynamicInstructionsReplayWorkflow],
        activities=warm.temporal_activities,
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        wf_id = DynamicInstructionsReplayWorkflow.__name__
        output = await client.execute_workflow(
            DynamicInstructionsReplayWorkflow.run, args=['hello'], id=wf_id, task_queue=TASK_QUEUE
        )
        assert output == snapshot('INSTRUCTIONS_FOR_STEP_2')
        history = await client.get_workflow_handle(wf_id).fetch_history()

    # Warm-recorded history replayed on a freshly-constructed cold instance (worker-restart trigger).
    dynamic_instructions_replay_holder['agent'] = _make_dynamic_instructions_replay_agent()
    try:
        await Replayer(
            workflows=[DynamicInstructionsReplayWorkflow],
            workflow_runner=UnsandboxedWorkflowRunner(),
            data_converter=pydantic_data_converter,
        ).replay_workflow(history)
    finally:
        dynamic_instructions_replay_holder['agent'] = warm


# --- MCP-based DynamicToolset test ---
# Tests that @agent.toolset returning an MCPToolset works with Temporal workflows.
# Uses an HTTP-based MCP server rather than subprocess-based since the subprocess transports
# don't play nicely with Temporal's sandbox.


mcptoolset_dynamic_toolset_agent = Agent(model, name='mcptoolset_dynamic_toolset_agent')


@mcptoolset_dynamic_toolset_agent.toolset(id='mcptoolset_dynamic')
def my_mcptoolset_dynamic_toolset(ctx: RunContext) -> MCPToolset:
    """Dynamic toolset that returns an `MCPToolset` — exercises lifecycle + `TemporalMCPToolset`."""
    return MCPToolset('https://mcp.deepwiki.com/mcp')


mcptoolset_dynamic_toolset_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    mcptoolset_dynamic_toolset_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class MCPToolsetDynamicToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await mcptoolset_dynamic_toolset_temporal_agent.run(prompt)
        return result.output


async def test_mcptoolset_dynamic_toolset_in_workflow(allow_model_requests: None, client: Client):
    """`@agent.toolset` returning an `MCPToolset` works in a Temporal workflow.

    Verifies the `MCPToolset`/`TemporalMCPToolset` pair handles `DynamicToolset` lifecycle
    (entering/exiting the context manager around each activity invocation).
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MCPToolsetDynamicToolsetAgentWorkflow],
        plugins=[AgentPlugin(mcptoolset_dynamic_toolset_temporal_agent)],
    ):
        output = await client.execute_workflow(
            MCPToolsetDynamicToolsetAgentWorkflow.run,
            args=['Can you tell me about the pydantic/pydantic-ai repo? Keep it short.'],
            id='test_mcptoolset_dynamic_toolset_workflow',
            task_queue=TASK_QUEUE,
        )
        assert 'pydantic' in output.lower() or 'agent' in output.lower()


@workflow.defn
class SimpleAgentWorkflowWithRunToolsets:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt, toolsets=[FunctionToolset()])
        return result.output  # pragma: no cover


async def test_temporal_agent_run_in_workflow_with_executing_toolsets(allow_model_requests: None, client: Client):
    # Executing toolsets (here a `FunctionToolset`) can't be added per-run because their activities must
    # be registered with the worker before the workflow runs.
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithRunToolsets],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'FunctionToolset cannot be passed to `run(toolsets=...)` at runtime with Temporal, because '
                'toolsets that execute their own tools or resolve dynamically must be registered for durable '
                'execution when the agent is constructed. Pass them to the agent constructor instead. '
                'Non-executing toolsets like `ExternalToolset` can be passed at runtime. Async tools that '
                "don't need durable wrapping can opt out with metadata={'temporal': False} to be allowed at runtime."
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithRunToolsets.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunToolsets.__name__,
                task_queue=TASK_QUEUE,
            )


def request_runtime_external_tool(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('external', {'query': 'runtime'}, tool_call_id='call-1')])


runtime_external_agent = Agent(
    FunctionModel(request_runtime_external_tool),
    name='runtime_external_toolset_agent',
    output_type=[str, DeferredToolRequests],
)

runtime_external_temporal_agent = TemporalAgent(runtime_external_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


runtime_external_toolset = ExternalToolset(
    tool_defs=[
        ToolDefinition(
            name='external',
            parameters_json_schema={
                'type': 'object',
                'properties': {'query': {'type': 'string'}},
                'required': ['query'],
            },
        )
    ],
    id='external',
)


@workflow.defn
class RuntimeExternalToolsetWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str | DeferredToolRequests]:
        return await runtime_external_temporal_agent.run(prompt, toolsets=[runtime_external_toolset])


async def test_temporal_agent_run_in_workflow_with_runtime_external_toolset(allow_model_requests: None, client: Client):
    # Non-executing toolsets like `ExternalToolset` need no durable wrapping, so they can be added per-run.
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RuntimeExternalToolsetWorkflow],
        plugins=[AgentPlugin(runtime_external_temporal_agent)],
    ):
        result = await client.execute_workflow(
            RuntimeExternalToolsetWorkflow.run,
            args=['Call the runtime external tool.'],
            id=RuntimeExternalToolsetWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert result.output == DeferredToolRequests(
            calls=[ToolCallPart('external', {'query': 'runtime'}, tool_call_id='call-1')]
        )


@workflow.defn
class SimpleAgentWorkflowWithOverrideToolsets:
    @workflow.run
    async def run(self, prompt: str) -> None:
        with simple_temporal_agent.override(toolsets=[FunctionToolset()]):
            pass


async def test_temporal_agent_override_toolsets_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideToolsets],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Toolsets cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithOverrideToolsets.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithOverrideToolsets.__name__,
                task_queue=TASK_QUEUE,
            )


async def test_temporal_agent_mcp_server_activity_disabled(client: Client):
    with pytest.raises(
        UserError,
        match=re.escape(
            "Temporal activity config for MCP tool 'get_product_name' has been explicitly set to `False` (activity disabled), "
            'but MCP tools require the use of IO and so cannot be run outside of an activity.'
        ),
    ):
        TemporalAgent(  # pyright: ignore[reportDeprecated]
            complex_agent,
            tool_activity_config={
                'mcp': {
                    'get_product_name': False,
                },
            },
        )


def return_mcp_instructions(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(agent_info.instructions or '')])


# Exercises the `TemporalMCPToolset` wrapper's `get_instructions` activity path.
mcptoolset_instructions_agent = Agent(
    FunctionModel(return_mcp_instructions),
    name='mcptoolset_instructions_agent',
    toolsets=[
        MCPToolset(
            StdioTransport(command='python', args=['-m', 'tests.mcp_server']),
            include_instructions=True,
            id='mcp',
        )
    ],
)


mcptoolset_instructions_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    mcptoolset_instructions_agent, activity_config=BASE_ACTIVITY_CONFIG
)


@workflow.defn
class MCPToolsetInstructionsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await mcptoolset_instructions_temporal_agent.run(prompt)
        return result.output


async def test_temporal_mcptoolset_instructions_propagate(client: Client):
    """`MCPToolset` instructions propagate through the `TemporalMCPToolset` wrapper."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MCPToolsetInstructionsWorkflow],
        plugins=[AgentPlugin(mcptoolset_instructions_temporal_agent)],
    ):
        output = await client.execute_workflow(
            MCPToolsetInstructionsWorkflow.run,
            args=['Use MCP instructions'],
            id=MCPToolsetInstructionsWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('Be a helpful assistant.')


def test_temporalize_mcptoolset_dispatches_to_temporalmcptoolset():
    """`temporalize_toolset` wraps `MCPToolset` in `TemporalMCPToolset`."""
    toolset = MCPToolset('https://example.com/mcp', id='test_dispatch')
    agent = Agent(model=model, name='dispatch_agent', toolsets=[toolset])
    temporal = TemporalAgent(agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]
    wrapped = next(ts for ts in temporal.toolsets if isinstance(ts, TemporalMCPToolset))
    assert wrapped.wrapped is toolset


async def _call_oversized_image_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('get_oversized_image', {})])
    return ModelResponse(parts=[TextPart('done')])  # pragma: no cover


oversized_tool_return_agent = Agent(
    FunctionModel(_call_oversized_image_tool, model_name='oversized-image-model'),
    name='oversized_tool_return_agent',
    deps_type=type(None),
    # Deliberately no `retry_policy`: Temporal's default is unlimited attempts, and half of what this
    # test pins is that an over-limit payload is non-retryable, so the run fails instead of hanging.
    capabilities=[TemporalDurability(activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=60)))],
)


@oversized_tool_return_agent.tool_plain
def get_oversized_image() -> BinaryImage:
    # Under Temporal's 2MB blob limit as raw bytes, over it once base64-encoded into the activity
    # payload — which is exactly why the usable budget is ~1.5MB rather than the nominal 2MB.
    return BinaryImage(data=b'\x00' * 1_600_000, media_type='image/png')


@workflow.defn
class OversizedToolReturnWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await oversized_tool_return_agent.run(prompt)
        return result.output  # pragma: no cover


async def test_oversized_tool_return_payload(client: Client):
    """A tool returning binary content over Temporal's payload limit points at the cause (#7110).

    Without the guard the run gets Temporal's own `[TMPRL1103] ... Size: N bytes, Limit: M bytes`,
    which names neither the tool, the image, nor Pydantic AI — and because Temporal treats an
    over-limit payload as retryable, the default policy resends it forever and the workflow never
    fails at all. The `execution_timeout` is what turns a regression of that second half into a test
    failure instead of a hang.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[OversizedToolReturnWorkflow],
        plugins=[AgentPlugin(oversized_tool_return_agent)],
    ):
        with workflow_raises(
            UserError,
            f"Tool 'get_oversized_image' returned a result too large for Temporal. {payload_limit_detail(2133494)}. Binary content like an image is base64-encoded into the activity payload, so if that is the cause, the raw-byte budget is about three quarters of the limit — roughly 1.5MB at the 2MB default. Return a reference instead of the value itself, like a URL or a key your application resolves later. To keep large payloads out of the workflow history without changing what your tools or models return, configure Temporal external storage (or a claim-check `payload_codec`) on your `DataConverter` — `PydanticAIPlugin` preserves it, and it covers every payload in both directions. See https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/#large-payloads",
        ):
            await client.execute_workflow(
                OversizedToolReturnWorkflow.run,
                args=['Get the image.'],
                id=OversizedToolReturnWorkflow.__name__,
                task_queue=TASK_QUEUE,
                execution_timeout=timedelta(seconds=30),
            )


@dataclass
class MetadataSidecar:
    label: str


async def test_tool_metadata_crosses_activity_boundary_as_json():
    """`metadata` is untyped, so its values arrive inside an activity as their JSON shapes.

    Not a workflow test: both halves are properties of the activity payloads themselves, and
    running them through the converter `PydanticAIPlugin` installs pins them directly. Observing
    the inbound half through the public API would take a tool call whose activity consumes the
    round-tripped `tool_def` rather than re-resolving its own.
    """
    # One value per Python type whose JSON shape differs from the original.
    metadata: dict[str, Any] = {
        'set': {'a'},
        'tuple': (1, 2),
        'dataclass': MetadataSidecar(label='x'),
        'bytes': b'\x01',
        'int_keys': {1: 'one'},
    }
    params = CallToolParams(
        name='analyze',
        tool_args={},
        serialized_run_context={},
        tool_def=ToolDefinition(name='analyze', metadata=metadata),
    )
    [decoded_params] = await pydantic_data_converter.decode(
        await pydantic_data_converter.encode([params]), [CallToolParams]
    )
    assert isinstance(decoded_params, CallToolParams)
    assert decoded_params.tool_def == snapshot(
        ToolDefinition(
            name='analyze',
            metadata={
                'set': ['a'],
                'tuple': [1, 2],
                'dataclass': {'label': 'x'},
                'bytes': '\x01',
                'int_keys': {'1': 'one'},
            },
        )
    )

    # And the same for `metadata` coming back out of an activity on a control-flow exception.
    async def require_approval() -> None:
        raise ApprovalRequired(metadata=metadata)

    [decoded_result] = await pydantic_data_converter.decode(
        await pydantic_data_converter.encode([await wrap_tool_call_result(require_approval())]),
        # The activity's declared return type is this discriminated union, which Temporal resolves
        # through a `TypeAdapter`; its `type_hints` parameter is annotated as `list[type]`.
        [cast('type', CallToolResult)],
    )
    with pytest.raises(ApprovalRequired) as exc_info:
        unwrap_tool_call_result(decoded_result)
    assert exc_info.value.metadata == snapshot(
        {'set': ['a'], 'tuple': [1, 2], 'dataclass': {'label': 'x'}, 'bytes': '\x01', 'int_keys': {'1': 'one'}}
    )

    # Only UTF-8-decodable bytes make it across at all; arbitrary binary needs base64 encoding.
    binary_params = CallToolParams(
        name='analyze',
        tool_args={},
        serialized_run_context={},
        tool_def=ToolDefinition(name='analyze', metadata={'bytes': b'\xff'}),
    )
    with pytest.raises(PydanticSerializationError, match='invalid utf-8 sequence'):
        await pydantic_data_converter.encode([binary_params])


def _tool_return_metadata_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('analyze_data', {})])
    else:
        return ModelResponse(parts=[TextPart('done')])


_tool_return_metadata_agent = Agent(
    FunctionModel(_tool_return_metadata_model),
    name='tool_return_metadata_agent',
)


@_tool_return_metadata_agent.tool_plain
def analyze_data() -> ToolReturn:
    return ToolReturn(
        return_value='analysis result',
        content='extra content for model',
        metadata={'key': 'value', 'count': 42},
    )


_tool_return_metadata_temporal_agent = TemporalAgent(_tool_return_metadata_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class ToolReturnMetadataWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[ModelMessage]:
        result = await _tool_return_metadata_temporal_agent.run(prompt)
        return result.all_messages()


async def test_tool_return_metadata_survives_temporal(allow_model_requests: None, client: Client):
    """ToolReturn metadata and content survive Temporal serialization.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/4676
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ToolReturnMetadataWorkflow],
        plugins=[AgentPlugin(_tool_return_metadata_temporal_agent)],
    ):
        messages = await client.execute_workflow(
            ToolReturnMetadataWorkflow.run,
            args=['analyze'],
            id=ToolReturnMetadataWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='analyze', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='analyze_data', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:_tool_return_metadata_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='analyze_data',
                        content='analysis result',
                        tool_call_id=IsStr(),
                        metadata={'key': 'value', 'count': 42},
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(content='extra content for model', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=57, output_tokens=3),
                model_name='function:_tool_return_metadata_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


mcptoolset_agent = Agent(
    model,
    name='mcptoolset_agent',
    toolsets=[MCPToolset('https://mcp.deepwiki.com/mcp', id='deepwiki')],
)


mcptoolset_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    mcptoolset_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class MCPToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await mcptoolset_temporal_agent.run(prompt)
        return result.output


async def test_mcptoolset_in_temporal_workflow(allow_model_requests: None, client: Client):
    """`MCPToolset` works in a Temporal workflow — parallel to `test_fastmcp_toolset`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MCPToolsetAgentWorkflow],
        plugins=[AgentPlugin(mcptoolset_temporal_agent)],
    ):
        output = await client.execute_workflow(
            MCPToolsetAgentWorkflow.run,
            args=['Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short'],
            id=MCPToolsetAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert 'pydantic' in output.lower() or 'agent' in output.lower()


_mcp_task_agent = Agent(
    TestModel(call_tools=['required_task_tool', 'optional_task_tool']),
    name='mcp_task_temporal_agent',
    toolsets=[
        MCPToolset(
            StdioTransport(command='python', args=['-m', 'tests.mcp_task_server']),
            id='mcp_tasks',
            init_timeout=20,
            prefer_tasks=False,
        )
    ],
)

_mcp_task_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    _mcp_task_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class MCPTaskSupportWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _mcp_task_temporal_agent.run(prompt)).output


async def test_temporal_mcptoolset_preserves_task_routing(client: Client):
    """Effective task routing in `ToolDefinition.metadata` survives Temporal activities."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MCPTaskSupportWorkflow],
        plugins=[AgentPlugin(_mcp_task_temporal_agent)],
    ):
        output = await client.execute_workflow(
            MCPTaskSupportWorkflow.run,
            args=['Call both tools'],
            id=MCPTaskSupportWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    assert output == '{"required_task_tool":"required_completed","optional_task_tool":"optional_sync"}'


nested_multimodal_tool_return_agent = Agent(TestModel(), name='nested_multimodal_tool_return_agent')


@nested_multimodal_tool_return_agent.tool
def get_nested_multimodal_content(ctx: RunContext) -> dict[str, str | MultiModalContent]:
    """Return multimodal content nested inside a mapping."""
    return {
        'caption': 'see attached',
        'attachment': BinaryImage(data=b'\x89PNG', media_type='image/png'),
        'source': DocumentUrl(url='https://example.com/doc/12345', media_type='application/pdf'),
    }


nested_multimodal_tool_return_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    nested_multimodal_tool_return_agent, activity_config=BASE_ACTIVITY_CONFIG
)


@workflow.defn
class NestedMultiModalToolReturnWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[ModelMessage]:
        result = await nested_multimodal_tool_return_temporal_agent.run(prompt)
        return result.all_messages()


async def test_nested_multimodal_tool_return_survives_temporal(client: Client):
    """Nested multimodal values in tool returns survive the Temporal activity boundary."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[NestedMultiModalToolReturnWorkflow],
        plugins=[AgentPlugin(nested_multimodal_tool_return_temporal_agent)],
    ):
        messages = await client.execute_workflow(
            NestedMultiModalToolReturnWorkflow.run,
            args=['inspect attachment'],
            id='test_nested_multimodal_tool_return',
            task_queue=TASK_QUEUE,
        )

    tool_return = next(
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'get_nested_multimodal_content'
    )
    tool_return_content_obj = tool_return.content
    assert isinstance(tool_return_content_obj, dict)
    tool_return_content = cast(dict[str, object], tool_return_content_obj)
    assert tool_return_content['caption'] == 'see attached'

    attachment = tool_return_content['attachment']
    assert isinstance(attachment, BinaryImage)
    assert attachment.media_type == 'image/png'
    assert attachment.data == b'\x89PNG'

    source = tool_return_content['source']
    assert isinstance(source, DocumentUrl)
    assert source.media_type == 'application/pdf'
    assert source.url == 'https://example.com/doc/12345'


def test_resolve_tool_activity_config_reads_metadata():
    """Tool metadata takes priority while defaults and caller-owned retry policies stay intact."""
    configured_retry_policy = RetryPolicy(maximum_attempts=3, non_retryable_error_types=['CustomError'])
    metadata_config = ActivityConfig(
        start_to_close_timeout=timedelta(seconds=120), retry_policy=configured_retry_policy
    )

    fn_toolset = FunctionToolset[None](id='resolve_meta_toolset')

    async def fn_tool() -> str:
        # Registered with the toolset; the test only resolves metadata.
        return 'ok'  # pragma: no cover

    fn_toolset.add_function(fn_tool, metadata={'temporal': metadata_config})
    tool_def = ToolDefinition(name='fn_tool', metadata={'temporal': metadata_config})
    tool = ToolsetTool[None](
        toolset=fn_toolset,
        tool_def=tool_def,
        max_retries=0,
        args_validator=None,  # pyright: ignore[reportArgumentType]
    )

    # Metadata wins over the per-tool dict.
    resolved = resolve_tool_activity_config(tool, 'fn_tool', {'fn_tool': ActivityConfig(summary='from_dict')})
    assert resolved is not metadata_config
    assert resolved is not False
    assert metadata_config.get('retry_policy') is configured_retry_policy
    assert configured_retry_policy.non_retryable_error_types == ['CustomError']
    retry_policy = resolved.get('retry_policy')
    assert retry_policy is not None
    assert retry_policy.non_retryable_error_types == [
        'CustomError',
        'UserError',
        'PydanticUserError',
        'UnexpectedModelBehavior',
        'FallbackExceptionGroup',
        'PayloadsTooLarge',
        'PayloadSizeError',
    ]

    inherited_retry_policy = RetryPolicy(maximum_attempts=7)
    resolved_without_override = resolve_tool_activity_config(None, 'fn_tool', {})
    assert resolved_without_override is not False
    assert resolved_without_override == {}
    assert ActivityConfig(retry_policy=inherited_retry_policy) | resolved_without_override == {
        'retry_policy': inherited_retry_policy
    }

    # `False` in metadata also wins.
    tool.tool_def.metadata = {'temporal': False}
    assert resolve_tool_activity_config(tool, 'fn_tool', {}) is False

    # Invalid metadata (e.g. a string from a misuse like `metadata={'temporal': '5s'}`)
    # raises `UserError` instead of silently passing the wrong shape to Temporal.
    tool.tool_def.metadata = {'temporal': '5s'}
    with pytest.raises(UserError, match=r"Tool 'fn_tool' has invalid 'temporal' metadata"):
        resolve_tool_activity_config(tool, 'fn_tool', {})


def test_resolve_tool_activity_config_restores_round_tripped_types():
    """A config that came back from an activity as JSON is validated into Temporal's own types.

    A `DynamicToolset`'s tools are discovered inside the get-tools activity, so their
    `ToolDefinition.metadata` returns to the workflow as JSON: `timedelta(minutes=5)` as `'PT5M'`,
    a `RetryPolicy` as a dict, an `ActivityCancellationType` as an int. `workflow.execute_activity`
    rejects those, failing the workflow *task*, which Temporal retries forever.
    """
    fn_toolset = FunctionToolset[None](id='round_trip_toolset')
    tool = ToolsetTool[None](
        toolset=fn_toolset,
        tool_def=ToolDefinition(
            name='slow',
            metadata={
                'temporal': {
                    'start_to_close_timeout': 'PT5M',
                    'heartbeat_timeout': 'PT30S',
                    'cancellation_type': 0,
                    'retry_policy': {'initial_interval': 'PT1S', 'maximum_attempts': 2},
                }
            },
        ),
        max_retries=0,
        args_validator=None,  # pyright: ignore[reportArgumentType]
    )

    resolved = resolve_tool_activity_config(tool, 'slow', {})
    assert resolved is not False
    assert resolved.get('start_to_close_timeout') == timedelta(minutes=5)
    assert resolved.get('heartbeat_timeout') == timedelta(seconds=30)
    assert resolved.get('cancellation_type') == ActivityCancellationType.TRY_CANCEL
    retry_policy = resolved.get('retry_policy')
    assert retry_policy is not None
    assert retry_policy.initial_interval == timedelta(seconds=1)
    assert retry_policy.maximum_attempts == 2
    assert retry_policy.non_retryable_error_types == [
        'UserError',
        'PydanticUserError',
        'UnexpectedModelBehavior',
        'FallbackExceptionGroup',
        'PayloadsTooLarge',
        'PayloadSizeError',
    ]


def test_resolve_tool_activity_config_rejects_unusable_config():
    """What validation can't restore fails the workflow with a `UserError` instead of livelocking it.

    `UserError` is in `workflow_failure_exception_types`, so it terminates the workflow; anything
    else `workflow.execute_activity` chokes on is a workflow-task failure Temporal retries forever.
    """
    fn_toolset = FunctionToolset[None](id='unusable_config_toolset')
    tool = ToolsetTool[None](
        toolset=fn_toolset,
        tool_def=ToolDefinition(name='slow', metadata={'temporal': {'start_to_close_timeout': 'five minutes'}}),
        max_retries=0,
        args_validator=None,  # pyright: ignore[reportArgumentType]
    )
    with pytest.raises(UserError, match=r"Tool 'slow' has an invalid Temporal `ActivityConfig`"):
        resolve_tool_activity_config(tool, 'slow', {})

    # A misspelled key is reported rather than dropped: `execute_activity` would reject it too,
    # but as a workflow-task failure.
    tool.tool_def.metadata = {'temporal': {'start_to_close_timout': timedelta(minutes=5)}}
    with pytest.raises(UserError, match=r'Extra inputs are not permitted'):
        resolve_tool_activity_config(tool, 'slow', {})


@pytest.mark.parametrize(
    'content',
    [
        {'kind': 'tool-return', 'value': 1},
        {'kind': 'tool-return', 'return_value': 'user-data'},
    ],
)
async def test_tool_return_content_with_framework_kind_round_trips(content: dict[str, Any]) -> None:
    """User mappings with framework-like `kind` keys round-trip as ordinary tool content."""

    async def return_content() -> dict[str, Any]:
        return content

    wrapped = await wrap_tool_call_result(return_content())
    assert wrapped.kind == 'tool_content_result'
    payloads = await pydantic_data_converter.encode([wrapped])
    decoded = await pydantic_data_converter.decode(payloads, [CallToolResult])  # pyright: ignore[reportArgumentType]
    assert unwrap_tool_call_result(decoded[0]) == content


async def test_structured_tool_return_round_trips() -> None:
    """Temporal serialization preserves every field of an explicit structured `ToolReturn`."""

    async def return_structured() -> ToolReturn:
        return ToolReturn('result', content='extra', metadata={'source': 'test'})

    wrapped = await wrap_tool_call_result(return_structured())
    assert wrapped.kind == 'tool_return'
    payloads = await pydantic_data_converter.encode([wrapped])
    decoded = await pydantic_data_converter.decode(payloads, [CallToolResult])  # pyright: ignore[reportArgumentType]
    assert unwrap_tool_call_result(decoded[0]) == ToolReturn('result', content='extra', metadata={'source': 'test'})


async def test_ordinary_tool_return_keeps_legacy_wire_shape() -> None:
    """Ordinary return values retain the legacy `tool_return` wire discriminator."""

    async def return_content() -> str:
        return 'result'

    wrapped = await wrap_tool_call_result(return_content())

    assert wrapped.kind == 'tool_return'


async def test_legacy_structured_tool_return_payload_decodes() -> None:
    """Temporal still decodes structured tool returns recorded with the legacy payload shape."""
    payloads = await pydantic_data_converter.encode(
        [{'result': {'return_value': 'legacy', 'kind': 'tool-return'}, 'kind': 'tool_return'}]
    )
    decoded = await pydantic_data_converter.decode(payloads, [CallToolResult])  # pyright: ignore[reportArgumentType]
    assert unwrap_tool_call_result(decoded[0]) == ToolReturn('legacy')


# --- Heartbeat supervision ---
# Unit tests on the internal `heartbeating` helper: a `beat()` crash requires simulating an
# SDK failure that no workflow-level test can trigger, and the exception-precedence contract
# (request error wins; beat crash surfaces after a successful request) is exactly the kind of
# internal invariant a VCR/workflow test would silently miss.


async def test_heartbeating_beats_and_stops(monkeypatch: pytest.MonkeyPatch):
    """Heartbeats fire on the derived cadence while the body runs and stop cleanly after."""
    beats: list[None] = []
    monkeypatch.setattr('temporalio.activity.info', lambda: SimpleNamespace(heartbeat_timeout=timedelta(seconds=0.02)))
    monkeypatch.setattr('temporalio.activity.heartbeat', lambda: beats.append(None))

    async with heartbeating():
        await asyncio.sleep(0.05)

    assert beats  # at least the immediate first beat, then every ~10ms
    count_after_exit = len(beats)
    await asyncio.sleep(0.05)
    assert len(beats) == count_after_exit  # the beater was cancelled on exit


async def test_heartbeating_beat_crash_surfaces_after_body(monkeypatch: pytest.MonkeyPatch):
    """A `beat()` crash fails the activity loudly instead of silently running unheartbeated."""

    def broken_heartbeat() -> None:
        raise RuntimeError('heartbeat exploded')

    monkeypatch.setattr('temporalio.activity.info', lambda: SimpleNamespace(heartbeat_timeout=None))
    monkeypatch.setattr('temporalio.activity.heartbeat', broken_heartbeat)

    with pytest.raises(RuntimeError, match='heartbeat exploded'):
        async with heartbeating():
            await asyncio.sleep(0.01)


async def test_heartbeating_body_error_wins_over_beat_crash(monkeypatch: pytest.MonkeyPatch):
    """An exception from the wrapped request is never replaced by a heartbeat failure."""

    def broken_heartbeat() -> None:
        raise RuntimeError('heartbeat exploded')

    monkeypatch.setattr('temporalio.activity.info', lambda: SimpleNamespace(heartbeat_timeout=None))
    monkeypatch.setattr('temporalio.activity.heartbeat', broken_heartbeat)

    with pytest.raises(ValueError, match='request failed'):
        async with heartbeating():
            await asyncio.sleep(0.01)
            raise ValueError('request failed')


# --- Every registered activity heartbeats ---


async def heartbeat_probe_tool() -> str:
    """A tool that yields to the event loop, giving the heartbeat task a chance to run."""
    await asyncio.sleep(0.01)
    return 'probe tool ran'


async def heartbeat_probe_agent_tool() -> str:
    """The same, for the agent's own implicit toolset, which registers its own activity."""
    await asyncio.sleep(0.01)
    return 'probe agent tool ran'


async def _heartbeat_probe_args_validator(ctx: RunContext[None]) -> None:
    await asyncio.sleep(0.01)


_heartbeat_function_toolset = FunctionToolset[None](
    tools=[Tool(heartbeat_probe_tool, args_validator=_heartbeat_probe_args_validator)], id='hb_tools'
)

_heartbeat_mcp_toolset = MCPToolset(
    StdioTransport(command='python', args=['-m', 'tests.mcp_server']),
    id='hb_mcp',
    init_timeout=20,
    # Without this, the test's own `get_tools()` warms the cache and the `get_tools` activity
    # returns without ever awaiting the server, leaving no window for a heartbeat to be observed.
    cache_tools=False,
)


async def _heartbeat_dynamic_toolset(ctx: RunContext[None]) -> AbstractToolset[None]:
    await asyncio.sleep(0.01)
    return FunctionToolset[None](
        tools=[Tool(heartbeat_probe_tool, args_validator=_heartbeat_probe_args_validator)], id='hb_dynamic_inner'
    )


async def _heartbeat_event_stream_handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in stream:
        await asyncio.sleep(0.01)


async def _heartbeat_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    await asyncio.sleep(0.01)
    return ModelResponse(parts=[TextPart('probe model response')])


async def _heartbeat_stream_model_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    await asyncio.sleep(0.01)
    yield 'probe model response'


class _HeartbeatProbeModel(FunctionModel):
    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        await asyncio.sleep(0.01)

    async def compact_messages(
        self, request_context: ModelRequestContext, *, instructions: str | None = None
    ) -> ModelResponse:
        await asyncio.sleep(0.01)
        return ModelResponse(parts=[TextPart('compacted')])


_heartbeat_agent = Agent(
    _HeartbeatProbeModel(_heartbeat_model_fn, stream_function=_heartbeat_stream_model_fn),
    name='heartbeat_probe_agent',
    deps_type=type(None),
    tools=[Tool(heartbeat_probe_agent_tool, args_validator=_heartbeat_probe_args_validator)],
    toolsets=[
        _heartbeat_function_toolset,
        _heartbeat_mcp_toolset,
        DynamicToolset(_heartbeat_dynamic_toolset, id='hb_dynamic'),
    ],
    capabilities=[TemporalDurability(event_stream_handler=_heartbeat_event_stream_handler)],
)


async def _heartbeats_during_activity(activity_fn: Callable[..., Any], args: Sequence[Any]) -> list[tuple[Any, ...]]:
    """Run an activity body inside an activity context, recording the heartbeats it emits."""
    beats: list[tuple[Any, ...]] = []
    env = ActivityEnvironment()
    env.info = replace(env.info, heartbeat_timeout=timedelta(seconds=0.02))
    env.on_heartbeat = lambda *details: beats.append(details)
    await env.run(activity_fn, *args)
    return beats


async def test_every_registered_activity_heartbeats(allow_model_requests: None):
    """Every activity Pydantic AI registers beats while it runs, not just the model ones (#6914).

    Heartbeats have no observable effect unless a `heartbeat_timeout` is configured and the
    activity outlives it, so a workflow-level test can only cover one activity kind at a time,
    and only slowly (see the test below for the user-visible consequence). Running each
    registered body in an `ActivityEnvironment` pins the property for all of them at once, and
    the exhaustiveness assertion means a newly registered activity has to be listed here — and
    so wrapped in `heartbeating()` — deliberately.
    """
    durability = TemporalDurability.from_agent(_heartbeat_agent)
    assert durability is not None

    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), run_id='hb-run')
    serialized_run_context = TemporalRunContext.serialize_run_context(ctx)
    request_params = _RequestParams(
        messages=[ModelRequest(parts=[UserPromptPart('hello')])],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        serialized_run_context=serialized_run_context,
    )
    get_tools_params = GetToolsParams(serialized_run_context=serialized_run_context)

    async with _heartbeat_mcp_toolset:
        agent_toolset = durability._toolsets_by_id['<agent>']  # pyright: ignore[reportPrivateUsage]
        agent_tool_def = (await agent_toolset.get_tools(ctx))['heartbeat_probe_agent_tool'].tool_def
        function_tool_def = (await _heartbeat_function_toolset.get_tools(ctx))['heartbeat_probe_tool'].tool_def
        mcp_tool_def = (await _heartbeat_mcp_toolset.get_tools(ctx))['get_none'].tool_def

        prefix = 'agent__heartbeat_probe_agent'
        args_by_activity_name: dict[str, list[Any]] = {
            f'{prefix}__toolset__<agent>__call_tool': [
                CallToolParams(
                    name='heartbeat_probe_agent_tool',
                    tool_args={},
                    serialized_run_context=serialized_run_context,
                    tool_def=agent_tool_def,
                ),
                None,
            ],
            f'{prefix}__toolset__<agent>__validate_args': [
                CallToolParams(
                    name='heartbeat_probe_agent_tool',
                    tool_args={},
                    serialized_run_context=serialized_run_context,
                    tool_def=agent_tool_def,
                ),
                None,
            ],
            f'{prefix}__model_request': [request_params, None],
            f'{prefix}__model_request_stream': [request_params, None],
            f'{prefix}__model_compact_messages': [
                _CompactMessagesParams(
                    messages=request_params.messages,
                    model_settings=None,
                    model_request_parameters=request_params.model_request_parameters,
                    streaming=False,
                    instructions='Keep decisions',
                    serialized_run_context=serialized_run_context,
                ),
                None,
            ],
            f'{prefix}__model_cancel_suspended_response': [
                _CancelParams(
                    response=ModelResponse(parts=[TextPart('suspended')]),
                    serialized_run_context=serialized_run_context,
                ),
                None,
            ],
            f'{prefix}__event_stream_handler': [
                _EventStreamHandlerParams(
                    event=PartStartEvent(index=0, part=TextPart('probe')),
                    serialized_run_context=serialized_run_context,
                ),
                None,
            ],
            f'{prefix}__toolset__hb_tools__call_tool': [
                CallToolParams(
                    name='heartbeat_probe_tool',
                    tool_args={},
                    serialized_run_context=serialized_run_context,
                    tool_def=function_tool_def,
                ),
                None,
            ],
            f'{prefix}__toolset__hb_tools__validate_args': [
                CallToolParams(
                    name='heartbeat_probe_tool',
                    tool_args={},
                    serialized_run_context=serialized_run_context,
                    tool_def=function_tool_def,
                ),
                None,
            ],
            f'{prefix}__mcp_server__hb_mcp__get_tools': [get_tools_params, None],
            f'{prefix}__mcp_server__hb_mcp__get_instructions': [get_tools_params, None],
            f'{prefix}__mcp_server__hb_mcp__call_tool': [
                CallToolParams(
                    name='get_none',
                    tool_args={},
                    serialized_run_context=serialized_run_context,
                    tool_def=mcp_tool_def,
                ),
                None,
            ],
            f'{prefix}__dynamic_toolset__hb_dynamic__get_tools': [get_tools_params, None],
            f'{prefix}__dynamic_toolset__hb_dynamic__call_tool': [
                CallToolParams(
                    name='heartbeat_probe_tool',
                    tool_args={},
                    serialized_run_context=serialized_run_context,
                    tool_def=function_tool_def,
                ),
                None,
            ],
            f'{prefix}__dynamic_toolset__hb_dynamic__validate_args': [
                CallToolParams(
                    name='heartbeat_probe_tool',
                    tool_args={},
                    serialized_run_context=serialized_run_context,
                    tool_def=function_tool_def,
                ),
                None,
            ],
        }

        activities_by_name: dict[str, Callable[..., Any]] = {}
        for activity_fn in durability.temporal_activities:
            activity_name = ActivityDefinition.must_from_callable(activity_fn).name  # pyright: ignore[reportUnknownMemberType]
            assert activity_name is not None
            activities_by_name[activity_name] = activity_fn
        assert activities_by_name.keys() == args_by_activity_name.keys()

        for name, activity_fn in activities_by_name.items():
            beats = await _heartbeats_during_activity(activity_fn, args_by_activity_name[name])
            assert beats, f'activity {name!r} ran without heartbeating'


def test_tool_activities_get_no_default_heartbeat_timeout():
    """Only model activities get a default `heartbeat_timeout`; tool activities deliberately don't.

    A `heartbeat_timeout` fails the attempt as soon as the beats stop, and a CPU-bound tool can
    occupy the event loop and starve the heartbeat task — so defaulting one would kill tools that
    run indefinitely today. Users who want one set it themselves.
    """
    agent = Agent(
        TestModel(),
        name='heartbeat_default_agent',
        deps_type=type(None),
        toolsets=[FunctionToolset[None](tools=[heartbeat_probe_tool], id='hb_default_tools')],
        capabilities=[TemporalDurability()],
    )
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None

    assert bound._model_activity_config.get('heartbeat_timeout') == timedelta(seconds=30)  # pyright: ignore[reportPrivateUsage]
    assert 'heartbeat_timeout' not in bound.activity_config

    toolset_wrapper = bound._toolsets_by_id['hb_default_tools']  # pyright: ignore[reportPrivateUsage]
    assert isinstance(toolset_wrapper, TemporalFunctionToolset)
    assert toolset_wrapper.durable_config is not None
    assert 'heartbeat_timeout' not in toolset_wrapper.durable_config


async def slow_heartbeat_tool() -> str:
    """Outlive the `heartbeat_timeout` the agent below configures for all of its activities."""
    await asyncio.sleep(2)
    return 'slow tool finished'


def _slow_tool_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(str(part.content))])
    return ModelResponse(parts=[ToolCallPart('slow_heartbeat_tool', {})])


_slow_tool_agent = Agent(
    FunctionModel(_slow_tool_model_fn),
    name='slow_tool_agent',
    deps_type=type(None),
    toolsets=[FunctionToolset[None](tools=[slow_heartbeat_tool], id='slow_tools')],
    capabilities=[
        TemporalDurability(
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=30),
                heartbeat_timeout=timedelta(seconds=1),
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
        )
    ],
)


@workflow.defn
class SlowToolHeartbeatWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await _slow_tool_agent.run('call the slow tool')).output


async def test_tool_outliving_configured_heartbeat_timeout_survives(client: Client):
    """A tool that runs longer than the `heartbeat_timeout` its user set completes (#6914).

    Setting a `heartbeat_timeout` — on the base config here, but `toolset_activity_config` and
    per-tool metadata reach the same activity — used to arm a kill switch: the tool activity
    never beat, so the server failed the attempt the moment the timeout elapsed.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SlowToolHeartbeatWorkflow],
        plugins=[AgentPlugin(_slow_tool_agent)],
    ):
        output = await client.execute_workflow(
            SlowToolHeartbeatWorkflow.run,
            id=f'{SlowToolHeartbeatWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )

    assert output == snapshot('slow tool finished')
