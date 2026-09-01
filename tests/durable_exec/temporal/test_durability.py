from __future__ import annotations

import asyncio
import re
import sys
import uuid
import warnings
from collections.abc import AsyncIterable
from dataclasses import dataclass, replace
from datetime import timedelta
from decimal import Decimal
from typing import Any, cast
from unittest.mock import patch

import pytest

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    BinaryContent,
    BinaryImage,
    CodeExecutionTool,
    DocumentUrl,
    ExternalToolset,
    FunctionToolset,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    MultiModalContent,
    PartDeltaEvent,
    PartStartEvent,
    RequestUsage,
    RetryPromptPart,
    RunContext,
    RunUsage,
    TextPart,
    TextPartDelta,
    Tool,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    ToolsetTool,
    UserContent,
    UserPromptPart,
    WebSearchTool,
    WebSearchUserLocation,
)
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import (
    Capability,
    DynamicCapability,
    Instrumentation,
    NativeTool,
    ProcessEventStream,
    ResolveModelId,
    Toolset,
    WrapperCapability,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.durable_exec._operation import ToolsetCallToolId
from pydantic_ai.exceptions import (
    ModelRetry,
    SkipModelRequest,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import UploadedFile
from pydantic_ai.models import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    ModelResolutionContext,
    infer_model,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.external import TOOL_SCHEMA_VALIDATOR
from pydantic_ai.usage import UsageLimits
from pydantic_graph import GraphBuilder, StepContext

from ..._inline_snapshot import snapshot
from ...continuation_utils import ScriptedContinuationModel, StreamSegment, scripted_response

try:
    from temporalio import activity, workflow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.client import Client, WorkflowFailureError
    from temporalio.common import RetryPolicy
    from temporalio.worker import UnsandboxedWorkflowRunner, Worker
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec._toolset import unwrap_tool_call_result
    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        TemporalAgent,  # pyright: ignore[reportDeprecated]
        TemporalDurability,
    )
    from pydantic_ai.durable_exec.temporal._function_toolset import (
        TemporalFunctionToolset,
        temporalize_function_toolset,
    )
    from pydantic_ai.durable_exec.temporal._run_context import TemporalRunContext
    from pydantic_ai.durable_exec.temporal._toolset import CallToolParams

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
    from logfire.testing import CaptureLogfire
except ImportError:  # pragma: lax no cover
    pytest.skip('logfire not installed', allow_module_level=True)

try:
    from fastmcp.client.transports import StdioTransport

    from pydantic_ai.mcp import MCPToolset
except ImportError:  # pragma: lax no cover
    pytest.skip('mcp not installed', allow_module_level=True)

try:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
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

    from ..._inline_snapshot import snapshot

    # Loads `vcr`, which Temporal doesn't like without passing through the import
    from ...conftest import IsDatetime, IsInt, IsList, IsStr

    # `_shared` loads the same sandbox-sensitive modules, so import it passed-through as well.
    from ._shared import (
        BASE_ACTIVITY_CONFIG,
        TASK_QUEUE,
        Answer,
        BasicSpan,
        Deps,
        DynamicToolsetDeps,
        Response,
        StreamDurableAgentWorkflow,
        _durability_fn_model,  # pyright: ignore[reportPrivateUsage]
        _durability_model_fn,  # pyright: ignore[reportPrivateUsage]
        _durability_reveal_tool,  # pyright: ignore[reportPrivateUsage]
        _scheduled_activity_count,  # pyright: ignore[reportPrivateUsage]
        _select_builtin_tool,  # pyright: ignore[reportPrivateUsage]
        _stream_durable_agent,  # pyright: ignore[reportPrivateUsage]
        _stream_events_collected,  # pyright: ignore[reportPrivateUsage]
        _stream_fn_model,  # pyright: ignore[reportPrivateUsage]
        _stream_model_events_in_activity,  # pyright: ignore[reportPrivateUsage]
        _workflow_failure_cause,  # pyright: ignore[reportPrivateUsage]
        code_execution_builtin_model,
        event_stream_handler,
        get_country,
        get_weather,
        http_client,
        model,
        web_search_builtin_model,
        web_search_model,
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
    pytest.mark.xdist_group(name='temporal-durability'),
    pytest.mark.filterwarnings(
        'ignore:`TemporalAgent` is deprecated:pydantic_ai._warnings.PydanticAIDeprecationWarning'
    ),
]


simple_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)

simple_durable_agent = Agent(_durability_fn_model, name='durability_simple_agent', capabilities=[simple_durability])


@workflow.defn
class SimpleDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_durable_agent.run(prompt)
        return result.output


@workflow.defn
class RunSyncDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return simple_durable_agent.run_sync(prompt).output


async def test_durability_run_sync_in_workflow_fails_the_workflow(client: Client):
    """`agent.run_sync()` inside a workflow fails the workflow with a clear error instead of hanging.

    Temporal's workflow event loop leaves `run_until_complete()` (and `is_closed()`) unimplemented, so
    before this was detected up front, `run_sync()` raised the bare `NotImplementedError` `asyncio`'s
    abstract loop raises. That type isn't among the plugin's `workflow_failure_exception_types`, so it
    failed the workflow *task*, which Temporal retries forever -- the caller hung instead of seeing an
    error. `UserError` is in that list, so the failure now reaches the caller.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RunSyncDurableAgentWorkflow],
        plugins=[AgentPlugin(simple_durable_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                RunSyncDurableAgentWorkflow.run,
                args=['What is the capital of Mexico?'],
                id=RunSyncDurableAgentWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )

    assert 'does not implement `run_until_complete()`' in str(exc_info.value.cause)
    assert '`await agent.run()` rather than `agent.run_sync()`' in str(exc_info.value.cause)


_sync_graph_builder = GraphBuilder(name='run_sync_graph', input_type=str, output_type=str)


@_sync_graph_builder.step
async def _echo_step(ctx: StepContext[None, None, str]) -> str:
    return ctx.inputs  # pragma: no cover


_sync_graph_builder.add(
    _sync_graph_builder.edge_from(_sync_graph_builder.start_node).to(_echo_step),
    _sync_graph_builder.edge_from(_echo_step).to(_sync_graph_builder.end_node),
)

_sync_graph = _sync_graph_builder.build()


@workflow.defn
class GraphRunSyncWorkflow:
    @workflow.run
    async def run(self) -> str:
        return _sync_graph.run_sync(inputs='hello')


async def test_durability_graph_run_sync_in_workflow_fails_the_workflow(client: Client):
    """`Graph.run_sync()` inside a workflow fails the workflow too, not just the workflow task.

    `pydantic_graph`'s sync entry points raise `UnsupportedEventLoopError` directly rather than going
    through the `pydantic_ai` wrapper that converts it to `UserError`, so the plugin has to recognize
    that type as well; otherwise this path keeps hanging with a good message nobody ever sees.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[GraphRunSyncWorkflow],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                GraphRunSyncWorkflow.run,
                id=GraphRunSyncWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )

    assert 'does not implement `run_until_complete()`' in str(exc_info.value.cause)


async def test_durability_simple_agent_run_in_workflow(client: Client):
    """TemporalDurability routes model requests through activities."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleDurableAgentWorkflow],
        plugins=[AgentPlugin(simple_durable_agent)],
    ):
        output = await client.execute_workflow(
            SimpleDurableAgentWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=SimpleDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == 'Echo: What is the capital of Mexico?'


# --- Durability with tools ---


def _tool_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Model function that calls `get_country` tool then returns the result."""
    # Check if we already have a tool result
    for msg in reversed(messages):
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(content=f'The country is: {part.content}')])

    # First call: invoke the tool
    if info.function_tools:
        return ModelResponse(parts=[ToolCallPart(tool_name='get_country', args='{}')])

    return ModelResponse(parts=[TextPart(content='no tools')])  # pragma: no cover


durability_country_toolset = FunctionToolset[Deps](tools=[get_country], id='durability_country')


_tool_fn_model = FunctionModel(_tool_model_fn)


complex_durability = TemporalDurability[Deps](deps_type=Deps, activity_config=BASE_ACTIVITY_CONFIG)

complex_durable_agent = Agent(
    _tool_fn_model,
    deps_type=Deps,
    toolsets=[durability_country_toolset],
    capabilities=[complex_durability],
    name='durability_complex_agent',
)


@workflow.defn
class ComplexDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: Deps) -> str:
        result = await complex_durable_agent.run(prompt, deps=deps)
        return result.output


async def test_durability_agent_with_tools_in_workflow(client: Client):
    """TemporalDurability wraps toolsets and routes tool calls through activities."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ComplexDurableAgentWorkflow],
        plugins=[AgentPlugin(complex_durable_agent)],
    ):
        output = await client.execute_workflow(
            ComplexDurableAgentWorkflow.run,
            args=['What country?', Deps(country='France')],
            id=ComplexDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == 'The country is: France'


# --- Durability outside workflow (transparent passthrough) ---


async def test_durability_outside_workflow_is_transparent():
    """TemporalDurability is a no-op outside a workflow — calls pass through to the real model."""
    result = await simple_durable_agent.run('Hello')
    assert result.output == 'Echo: Hello'


# --- Durability wrap_run disables threads ---


_threads_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)

_threads_agent = Agent(_durability_fn_model, name='sync_tool_test', capabilities=[_threads_durability])


@workflow.defn
class ThreadsDurableWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _threads_agent.run(prompt)
        return result.output


async def test_durability_wrap_run_disables_threads(client: Client):
    """wrap_run disables threads when inside a Temporal workflow."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ThreadsDurableWorkflow],
        plugins=[AgentPlugin(_threads_agent)],
    ):
        output = await client.execute_workflow(
            ThreadsDurableWorkflow.run,
            args=['test'],
            id='ThreadsDurableWorkflow',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Echo: test'


# --- Durability validation ---


def test_durability_requires_agent_name():
    """TemporalDurability raises UserError when agent has no name."""
    durability = TemporalDurability()
    with pytest.raises(UserError, match='unique `name`'):
        Agent(_durability_fn_model, capabilities=[durability])


def test_durability_explicit_name_overrides_agent_name_and_supports_unnamed_agent():
    named_agent = Agent(_durability_fn_model, name='agent-name', capabilities=[TemporalDurability(name='custom')])
    bound = TemporalDurability.from_agent(named_agent)
    assert bound is not None
    assert bound.name == 'custom'
    activity_names = [
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in bound.temporal_activities
    ]
    assert all(name is not None and name.startswith('agent__custom__') for name in activity_names)

    unnamed_agent = Agent(_durability_fn_model, capabilities=[TemporalDurability(name='unnamed-custom')])
    unnamed_bound = TemporalDurability.from_agent(unnamed_agent)
    assert unnamed_bound is not None
    assert unnamed_bound.name == 'unnamed-custom'


def test_durability_requires_model():
    """TemporalDurability raises UserError when the agent has no model at all."""
    durability = TemporalDurability()
    with pytest.raises(UserError, match='needs to have a `model`'):
        Agent(name='test', capabilities=[durability])


def test_durability_rejects_default_model_key():
    """TemporalDurability raises UserError when 'default' is used in the models dict."""
    with pytest.raises(UserError, match="'default' is reserved"):
        Agent(
            _durability_fn_model,
            name='test',
            capabilities=[TemporalDurability(models={'default': _durability_fn_model})],
        )


def test_durability_from_agent_rejects_duplicates():
    agent = Agent(
        _durability_fn_model,
        name='duplicate_durability',
        capabilities=[TemporalDurability(), TemporalDurability()],
    )

    with pytest.raises(
        UserError,
        match=r'Multiple TemporalDurability capabilities are attached to this agent; attach at most one\.',
    ):
        TemporalDurability.from_agent(agent)


def test_durability_rejects_construction_inside_workflow(monkeypatch: pytest.MonkeyPatch):
    """`TemporalDurability.for_agent` rejects construction inside a workflow.

    Activities have to be registered with the worker before the workflow runs, so
    `for_agent` (which discovers and registers activities) must run at module level
    or in worker setup code — not inside `@workflow.run`.
    """
    from temporalio import workflow as _wf

    monkeypatch.setattr(_wf, 'in_workflow', lambda: True)
    with pytest.raises(UserError, match=r'must be constructed outside of a Temporal workflow'):
        Agent(_durability_fn_model, name='test', capabilities=[TemporalDurability()])


def test_durability_image_output_rejected():
    """TemporalDurability rejects image output rather than letting it fail on payload size."""
    agent = Agent(_durability_fn_model, name='test', capabilities=[TemporalDurability()])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    with pytest.raises(UserError) as exc_info:
        bound._validate_model_request_parameters(  # pyright: ignore[reportPrivateUsage]
            ModelRequestParameters(allow_image_output=True),
        )
    assert str(exc_info.value) == snapshot(
        'Image output is not supported with Temporal because the image would ride the activity payload, '
        'which is capped by the server blob-size limit (2MB by default, leaving about 1.5MB of raw image '
        'bytes once base64-encoded).'
    )


# --- Model registry ---


def test_durability_find_model_id_by_identity():
    """_find_model_id matches models by identity."""
    m1 = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content='hi')]))
    m2 = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content='hi')]))
    agent = Agent(m1, name='test', capabilities=[TemporalDurability(models={'alt': m2})])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    assert bound._find_model_id(m1) is None  # default → None  # pyright: ignore[reportPrivateUsage]
    assert bound._find_model_id(m2) == 'alt'  # pyright: ignore[reportPrivateUsage]


def test_durability_find_model_id_prefers_registered_wrapper_identity():
    """Temporal preserves a registered wrapper's alias before considering its inner model."""
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content='bare')]))
    wrapped = WrapperModel(model)
    agent = Agent(model, name='test', capabilities=[TemporalDurability(models={'wrapped': wrapped})])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    assert bound._find_model_id(wrapped) == 'wrapped'  # pyright: ignore[reportPrivateUsage]
    # An unregistered wrapper (e.g. a user-built `InstrumentedModel`) around a registered
    # wrapper peels off to the shallowest registered match instead of collapsing to the default.
    assert bound._find_model_id(WrapperModel(wrapped)) == 'wrapped'  # pyright: ignore[reportPrivateUsage]
    # An unregistered wrapper around the bare default still takes the default's fast path.
    assert bound._find_model_id(WrapperModel(model)) is None  # pyright: ignore[reportPrivateUsage]


def test_durability_find_model_id_does_not_unwrap_registered_wrappers():
    """A registered wrapper's identity holds at its registered depth.

    Its bare inner model must not inherit the wrapper's alias — the activity would rebuild the
    wrapper and add behavior the request never had — so the bare model counts as unregistered.
    """
    default = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content='default')]))
    inner = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(content='inner')]))
    agent = Agent(default, name='test', capabilities=[TemporalDurability(models={'wrapped_alt': WrapperModel(inner)})])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    with pytest.raises(UserError, match='was not registered with `TemporalDurability`'):
        bound._find_model_id(inner)  # pyright: ignore[reportPrivateUsage]


def test_durability_temporal_activities():
    """temporal_activities returns all registered activities after for_agent."""
    agent = Agent(_durability_fn_model, name='test', capabilities=[TemporalDurability()])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    # 4 base activities + call/validation activities for the agent's <agent> FunctionToolset
    assert len(bound.temporal_activities) == 6


def test_durability_temporal_activities_with_toolsets():
    """temporal_activities includes toolset activities for agent's toolsets."""
    agent = Agent(
        _durability_fn_model,
        name='test',
        toolsets=[FunctionToolset(id='test_toolset')],
        capabilities=[TemporalDurability()],
    )
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    # 4 base activities + call/validation activities for both function toolsets
    assert len(bound.temporal_activities) == 8


def test_durability_duplicate_toolset_id_rejected():
    """Two distinct toolsets under one `id` are rejected at binding time.

    The registry maps `id` → activity wrapper, so a duplicate would silently replace the
    first entry and route both toolsets' calls through the last one's activities.
    """
    with pytest.raises(UserError, match="Two toolsets have the same `id` 'dup'"):
        Agent(
            _durability_fn_model,
            name='durability_dup_toolset',
            toolsets=[FunctionToolset(id='dup'), FunctionToolset(id='dup')],
            capabilities=[TemporalDurability()],
        )


def test_durability_same_toolset_instance_reused():
    """The same toolset instance appearing twice maps to one wrapper, not an `id` conflict.

    Its activities must register with the worker exactly once — Temporal rejects duplicate
    activity names at worker start.
    """
    ts = FunctionToolset[Any](id='shared_fn')
    agent = Agent(
        _durability_fn_model,
        name='durability_shared_toolset',
        toolsets=[ts, ts],
        capabilities=[TemporalDurability()],
    )
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    # 4 base activities + call/validation activities for <agent> and the shared toolset (once)
    assert len(bound.temporal_activities) == 8


def test_durability_activity_config_not_mutated():
    """The capability normalizes the retry policy on copies of the caller's config.

    A `RetryPolicy` (or `ActivityConfig`) shared with other Temporal activities must not
    gain the capability's non-retryable error types, and constructing multiple capabilities
    from the same config must not accumulate duplicate entries.
    """
    retry_policy = RetryPolicy(non_retryable_error_types=['MyError'])
    config = ActivityConfig(start_to_close_timeout=timedelta(seconds=60), retry_policy=retry_policy)

    durability = TemporalDurability(activity_config=config)
    TemporalDurability(activity_config=config)

    assert retry_policy.non_retryable_error_types == ['MyError']
    assert config.get('retry_policy') is retry_policy
    normalized = durability.activity_config.get('retry_policy')
    assert normalized is not None
    assert normalized is not retry_policy
    assert normalized.non_retryable_error_types == [
        'MyError',
        'UserError',
        'PydanticUserError',
        'UnexpectedModelBehavior',
        'FallbackExceptionGroup',
        'PayloadsTooLarge',
        'PayloadSizeError',
    ]


def test_durability_custom_retry_policy_keeps_non_retryable_errors():
    """A caller-supplied `retry_policy` must not drop the framework's non-retryable errors.

    A `retry_policy` in `model_activity_config` or a per-toolset config would otherwise
    replace the normalized base policy wholesale, letting a `UserError` or a
    continuation-ceiling `UnexpectedModelBehavior` retry the whole (paid) segment.
    """
    toolset = FunctionToolset[None](id='my_toolset')

    async def my_tool() -> str:
        return 'ok'  # pragma: no cover

    toolset.add_function(my_tool)

    durability = TemporalDurability(
        model_activity_config=ActivityConfig(retry_policy=RetryPolicy(non_retryable_error_types=['ModelError'])),
        toolset_activity_config={
            'my_toolset': ActivityConfig(retry_policy=RetryPolicy(non_retryable_error_types=['ToolError'])),
        },
    )
    agent = Agent(
        _durability_fn_model,
        name='custom_retry_agent',
        deps_type=type(None),
        toolsets=[toolset],
        capabilities=[durability],
    )
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None

    model_retry = bound._model_activity_config.get('retry_policy')  # pyright: ignore[reportPrivateUsage]
    assert model_retry is not None
    assert model_retry.non_retryable_error_types == [
        'ModelError',
        'UserError',
        'PydanticUserError',
        'UnexpectedModelBehavior',
        'FallbackExceptionGroup',
        'PayloadsTooLarge',
        'PayloadSizeError',
    ]

    toolset_wrapper = bound._toolsets_by_id['my_toolset']  # pyright: ignore[reportPrivateUsage]
    assert isinstance(toolset_wrapper, TemporalFunctionToolset)
    assert toolset_wrapper.durable_config is not None
    toolset_retry = toolset_wrapper.durable_config.get('retry_policy')
    assert toolset_retry is not None
    assert toolset_retry.non_retryable_error_types == [
        'ToolError',
        'UserError',
        'PydanticUserError',
        'UnexpectedModelBehavior',
        'FallbackExceptionGroup',
        'PayloadsTooLarge',
        'PayloadSizeError',
    ]


def test_durability_event_stream_handler_activity_config_keeps_non_retryable_errors() -> None:
    durability = TemporalDurability(
        activity_config=ActivityConfig(summary='base'),
        event_stream_handler_activity_config=ActivityConfig(
            summary='handle stream event',
            retry_policy=RetryPolicy(non_retryable_error_types=['HandlerError']),
        ),
    )
    config = durability._event_stream_handler_activity_config  # pyright: ignore[reportPrivateUsage]
    assert config.get('summary') == 'handle stream event'
    retry_policy = config.get('retry_policy')
    assert retry_policy is not None
    assert retry_policy.non_retryable_error_types == [
        'HandlerError',
        'UserError',
        'PydanticUserError',
        'UnexpectedModelBehavior',
        'FallbackExceptionGroup',
        'PayloadsTooLarge',
        'PayloadSizeError',
    ]


@pytest.mark.parametrize(
    'kwargs,expected',
    [
        pytest.param(
            {'activity_config': {'timeout': timedelta(seconds=1)}},
            'Invalid Temporal `ActivityConfig` in `activity_config`',
            id='activity_config',
        ),
        pytest.param(
            {'model_activity_config': {'start_to_close': timedelta(seconds=1)}},
            'Invalid Temporal `ActivityConfig` in `model_activity_config`',
            id='model_activity_config',
        ),
        pytest.param(
            {'event_stream_handler_activity_config': {'summry': 'oops', 'task_q': 'oops'}},
            'Invalid Temporal `ActivityConfig` in `event_stream_handler_activity_config`',
            id='event_stream_handler_activity_config',
        ),
        pytest.param(
            {'toolset_activity_config': {'my_toolset': {'my_tool': False}}},
            "Invalid Temporal `ActivityConfig` in `toolset_activity_config['my_toolset']`",
            id='toolset_activity_config',
        ),
        pytest.param(
            {'model_activity_config': {'start_to_close_timeout': 'five minutes'}},
            'Invalid Temporal `ActivityConfig` in `model_activity_config`',
            id='unusable-value',
        ),
    ],
)
def test_durability_rejects_unknown_activity_config_keys(kwargs: dict[str, Any], expected: str):
    """An `ActivityConfig` key Temporal doesn't know fails at construction, not mid-workflow.

    `ActivityConfig` is a `total=False` `TypedDict`, so an unknown key survives construction and
    would only fail when it's splatted into `workflow.start_activity()` inside workflow code —
    where the resulting `TypeError` isn't a `workflow_failure_exception_types` member and so fails
    the workflow *task*, which Temporal retries forever. The last case is the shape reported in
    #6917: a per-tool map (which belongs in tool `metadata`) passed as a toolset's config.
    """
    with pytest.raises(UserError, match=re.escape(expected)):
        TemporalDurability(**kwargs)


def test_durability_coerces_activity_config_values():
    """Validation keeps the coerced config, not the caller's raw one.

    A config that round-tripped through JSON carries `'PT5M'` where Temporal wants a `timedelta`.
    That validates fine, so only *keeping* the coerced result stops the raw string from reaching
    `workflow.start_activity()` and wedging the workflow task — the same failure an unknown key
    causes, just via a value.
    """
    durability = TemporalDurability(
        activity_config={'start_to_close_timeout': 'PT5M'},  # pyright: ignore[reportArgumentType]
        toolset_activity_config={'my_toolset': {'schedule_to_close_timeout': 'PT9M'}},  # pyright: ignore[reportArgumentType]
    )

    assert durability.activity_config.get('start_to_close_timeout') == timedelta(minutes=5)
    assert durability._model_activity_config.get('start_to_close_timeout') == timedelta(minutes=5)  # pyright: ignore[reportPrivateUsage]
    toolset_config = durability._toolset_activity_config['my_toolset']  # pyright: ignore[reportPrivateUsage]
    assert toolset_config.get('schedule_to_close_timeout') == timedelta(minutes=9)


def test_durability_shared_instance_across_agents():
    """Same TemporalDurability instance can be reused across multiple agents.

    for_agent returns a new bound copy; the original stays pristine.
    """
    durability = TemporalDurability()
    a1 = Agent(_durability_fn_model, name='a1', capabilities=[durability])
    a2 = Agent(_durability_fn_model, name='a2', capabilities=[durability])
    # Original is unbound
    assert durability.name == ''
    assert durability.temporal_activities == []
    # Each agent has its own bound copy
    b1 = TemporalDurability.from_agent(a1)
    b2 = TemporalDurability.from_agent(a2)
    assert b1 is not None and b2 is not None
    assert b1 is not b2
    assert b1.name == 'a1'
    assert b2.name == 'a2'


# --- _find_model_id rejects unregistered models ---


_rt_primary_model = FunctionModel(_durability_model_fn, model_name='primary')

_rt_alt_model = FunctionModel(
    lambda messages, info: ModelResponse(parts=[TextPart(content='alt-response')]),
    model_name='alt',
)

_rt_durability = TemporalDurability(models={'alt': _rt_alt_model}, activity_config=BASE_ACTIVITY_CONFIG)

_rt_agent = Agent(_rt_primary_model, name='runtime_model_test', capabilities=[_rt_durability])


@workflow.defn
class RuntimeModelWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _rt_agent.run(prompt, model=_rt_alt_model)
        return result.output


async def test_durability_runtime_registered_model_is_used(client: Client):
    """agent.run(model=registered_model) routes through the registered model's activity."""
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[RuntimeModelWorkflow], plugins=[AgentPlugin(_rt_agent)]
    ):
        output = await client.execute_workflow(
            RuntimeModelWorkflow.run,
            args=['ignored'],
            id='RuntimeModelWorkflow',
            task_queue=TASK_QUEUE,
        )
    assert output == 'alt-response'


async def test_durability_resolve_model_id_uses_models_registry():
    """resolve_model_id maps a registered model-id string to its registered Model instance."""
    primary = FunctionModel(_durability_model_fn, model_name='primary')
    alt = FunctionModel(_durability_model_fn, model_name='alt')

    durability = TemporalDurability(models={'alt': alt}, activity_config=BASE_ACTIVITY_CONFIG)
    agent = Agent(primary, name='resolve_registry_test', capabilities=[durability])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    resolution_ctx = ModelResolutionContext[Any](agent=agent, deps=None)

    # String matches a registered model → returns that exact instance.
    assert await bound.resolve_model_id(resolution_ctx, model_id='alt') is alt

    # String not in registry → defer (None) so the default `infer_model` flow — or a
    # user's `ResolveModelId` capability — handles it, and so an exception raised by a
    # user resolver is never masked by this capability's backstop.
    assert await bound.resolve_model_id(resolution_ctx, model_id='test') is None


async def test_durability_default_string_registered_in_models_becomes_default():
    """A `models=` key equal to the agent's raw default model string supplies the default instance.

    The user explicitly mapped that string to an instance, so binding uses it as `'default'`
    (rather than building an orphaned one via `infer_model`), and run-time resolution of the
    default string returns the same instance — keeping the identity match that gives the
    default the `model_id=None` fast path across the activity boundary.
    """
    custom = FunctionModel(_durability_model_fn, model_name='custom-default')
    durability = TemporalDurability(models={'test': custom}, activity_config=BASE_ACTIVITY_CONFIG)
    agent = Agent('test', name='default_collision_test', capabilities=[durability])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None

    assert await bound.resolve_model_id(ModelResolutionContext(agent=agent, deps=None), model_id='test') is custom
    assert bound._find_model_id(custom) is None  # identity-matches 'default'  # pyright: ignore[reportPrivateUsage]


async def test_durability_default_string_not_in_models_defers_to_resolution_chain():
    """A plain string default isn't resolved at bind time — it defers to run-time resolution.

    Building the default eagerly here could construct the wrong provider — with its
    authentication/configuration side effects — before a sibling `ResolveModelId` gets to
    reinterpret the string, so no `'default'` is registered and the raw string re-resolves
    through the capability chain (or `infer_model`) on the worker.
    """
    durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)
    agent = Agent('test', name='default_defers_test', capabilities=[durability])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None

    # No concrete default was built at bind time, so the registry is empty and resolving the
    # default string defers (`None`) to the chain / `infer_model` rather than a pre-built instance.
    assert bound._models_by_id == {}  # pyright: ignore[reportPrivateUsage]
    assert await bound.resolve_model_id(ModelResolutionContext(agent=agent, deps=None), model_id='test') is None


# --- Deps-aware model resolution via the `ResolveModelId` capability ---


def _tenant_resolver(ctx: ModelResolutionContext[str], model_id: str) -> FunctionModel | None:
    """Resolve the 'tenant-model' alias to a model built from the run's deps.

    Matches the alias exactly: the run's original model-id string (not the resolved
    model's `'function:tenant-model'`) is what crosses the durable boundary, so the
    worker-side re-resolution sees the same string the caller wrote.
    """
    if model_id != 'tenant-model':
        return None
    tenant = ctx.deps

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=f'tenant:{tenant}')])

    return FunctionModel(fn, model_name='tenant-model')


_tenant_agent = Agent(
    _rt_primary_model,
    name='tenant_resolver_test',
    deps_type=str,
    capabilities=[ResolveModelId(_tenant_resolver), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class TenantModelWorkflow:
    @workflow.run
    async def run(self, tenant: str) -> str:
        result = await _tenant_agent.run('hi', model='tenant-model', deps=tenant)
        # A string the resolver doesn't recognize defers to the default `infer_model` flow.
        fallthrough = await _tenant_agent.run('hi', model='test', deps=tenant)
        return f'{result.output} | {fallthrough.output}'


async def test_durability_resolve_model_id_capability_is_deps_aware(client: Client):
    """A deps-aware `ResolveModelId` resolver rebuilds the model with the run's deps inside the activity.

    The response content is produced by the model *inside* the model-request activity, so it
    proves the activity re-ran the capability chain with the deserialized deps — not just that
    the workflow-side resolution saw them.

    The resolver is deliberately *synchronous*: workflow-side resolution runs before
    `TemporalDurability.wrap_run`'s `disable_threads()` guard is active, so this also pins
    that `ResolveModelId` invokes sync resolvers inline rather than via a thread executor
    (which is unavailable inside the deterministic workflow sandbox and would hang).
    """
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[TenantModelWorkflow], plugins=[AgentPlugin(_tenant_agent)]
    ):
        for tenant in ('acme', 'globex'):
            output = await client.execute_workflow(
                TenantModelWorkflow.run,
                args=[tenant],
                id=f'TenantModelWorkflow-{tenant}',
                task_queue=TASK_QUEUE,
            )
            assert output == f'tenant:{tenant} | success (no tool calls)'


_alias_default_agent = Agent(
    'tenant-model',
    name='alias_default_test',
    deps_type=str,
    capabilities=[ResolveModelId(_tenant_resolver), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class AliasDefaultWorkflow:
    @workflow.run
    async def run(self, tenant: str) -> str:
        result = await _alias_default_agent.run('hi', deps=tenant)
        return result.output


async def test_durability_alias_default_model(client: Client):
    """An agent whose *default* model is an alias only a `ResolveModelId` capability can resolve.

    `infer_model` can't build `'tenant-model'`, so binding registers no concrete default;
    every request carries the raw alias string across the activity boundary and the
    worker-side chain re-resolves it with the run's deps.
    """
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[AliasDefaultWorkflow], plugins=[AgentPlugin(_alias_default_agent)]
    ):
        output = await client.execute_workflow(
            AliasDefaultWorkflow.run,
            args=['acme'],
            id='AliasDefaultWorkflow',
            task_queue=TASK_QUEUE,
        )
    assert output == 'tenant:acme'


# --- Outer capability swaps `request_context.model` inside a workflow ---


# The swapped-in model never runs — the request is rejected before it is dispatched — so this
# reuses the shared durability model function rather than defining an unreachable one.
_swap_target_registered = FunctionModel(_durability_model_fn)


class _SwapModelCapability(AbstractCapability[Any]):
    """Outer capability that swaps the request's model to a fresh, unregistered instance."""

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        request_context.model = FunctionModel(_durability_model_fn)
        return request_context


_swap_model_durability = TemporalDurability(
    # A *different* instance is registered under the same `model_id`: registration is matched by
    # identity, so the swapped-in instance is still unregistered.
    models={_swap_target_registered.model_id: _swap_target_registered},
    activity_config=BASE_ACTIVITY_CONFIG,
)

_swap_model_agent = Agent(
    _durability_fn_model,
    name='durability_swap_model_agent',
    capabilities=[_SwapModelCapability(), _swap_model_durability],
)


@workflow.defn
class SwapModelWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _swap_model_agent.run(prompt)
        return result.output  # pragma: no cover


async def test_durability_outer_capability_model_swap_rejected(client: Client):
    """A model swapped in by an outer capability's `before_model_request` is rejected too.

    Managed-style capabilities sit outside the durability capability and may replace
    `request_context.model` with a freshly-built instance the registry has never seen. Another
    instance registered under the same `model_id` doesn't make it registered — registration is
    matched by identity, and rebuilding from a `model_id` is exactly the assumption this rejects.
    Such a capability should supply its model through `resolve_model_id` (from a string) instead.
    """
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[SwapModelWorkflow], plugins=[AgentPlugin(_swap_model_agent)]
    ):
        with workflow_raises(
            UserError,
            snapshot(
                "The model instance 'function:function:_durability_model_fn:' was not registered with `TemporalDurability`, so it cannot be used inside a workflow. A `Model` instance cannot be serialized across the activity boundary, and rebuilding it from its `model_id` would build a different model — the same model name on the provider the worker environment implies — so the request would go to another endpoint with other credentials. Register the instance in `models=` on `TemporalDurability` and reference it by key (or pass the registered instance), or pass a model-name string and build the instance from it with a `ResolveModelId` capability."
            ),
        ):
            await client.execute_workflow(
                SwapModelWorkflow.run,
                args=['ignored'],
                id='SwapModelWorkflow',
                task_queue=TASK_QUEUE,
            )


# --- Unregistered `Model` instances are rejected ---


# A per-tenant endpoint and API key: rebuilding this from `'openai:gpt-5.6-sol'` on the worker
# would quietly send the request to `api.openai.com` with the ambient key instead.
_tenant_endpoint_model = OpenAIChatModel(
    'gpt-5.6-sol',
    provider=OpenAIProvider(api_key='tenant-key', base_url='https://tenant.example.com/v1', http_client=http_client),
)


_unregistered_instance_agent = Agent(
    _rt_primary_model,
    name='durability_unregistered_instance',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class UnregisteredModelInstanceWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _unregistered_instance_agent.run(prompt, model=_tenant_endpoint_model)
        return result.output  # pragma: no cover


async def test_durability_unregistered_model_instance_errors(client: Client):
    """An unregistered `Model` instance is rejected in the workflow, before any activity runs.

    A `Model` can't be serialized into an activity, and rebuilding this one from its `model_id`
    would build the same model name on the default provider — dropping the tenant's `base_url` and
    API key, so the request would silently go to `api.openai.com` with the worker's credentials.
    Registering the instance in `models=`, or passing a string a `ResolveModelId` capability builds
    on the worker, are the two supported paths.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[UnregisteredModelInstanceWorkflow],
        plugins=[AgentPlugin(_unregistered_instance_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                "The model instance 'openai:gpt-5.6-sol' was not registered with `TemporalDurability`, so it cannot be used inside a workflow. A `Model` instance cannot be serialized across the activity boundary, and rebuilding it from its `model_id` would build a different model — the same model name on the provider the worker environment implies — so the request would go to another endpoint with other credentials. Register the instance in `models=` on `TemporalDurability` and reference it by key (or pass the registered instance), or pass a model-name string and build the instance from it with a `ResolveModelId` capability."
            ),
        ):
            await client.execute_workflow(
                UnregisteredModelInstanceWorkflow.run,
                args=['ignored'],
                id='UnregisteredModelInstanceWorkflow',
                task_queue=TASK_QUEUE,
            )


# --- Runtime capability validation ---


async def test_durability_validates_only_resolved_runtime_capability_layers():
    """Temporal accepts resolved and safe per-run layers but rejects per-run dynamic layers."""

    @dataclass
    class _BaseOne(AbstractCapability[None]):
        pass

    @dataclass
    class _BaseTwo(AbstractCapability[None]):
        pass

    @dataclass
    class _ExtraOne(AbstractCapability[None]):
        pass

    @dataclass
    class _ExtraTwo(AbstractCapability[None]):
        pass

    @dataclass
    class _SkipRequest(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            raise SkipModelRequest(ModelResponse(parts=[TextPart(content='skipped')]))

    def base_factory(ctx: RunContext[None]) -> AbstractCapability[None]:
        return CombinedCapability([_BaseOne(), _BaseTwo(), _SkipRequest()])

    def extra_factory(ctx: RunContext[None]) -> AbstractCapability[None]:
        return CombinedCapability([_ExtraOne(), _ExtraTwo()])

    agent = Agent(
        TestModel(),
        name='runtime_capability_layers',
        deps_type=type(None),
        capabilities=[base_factory, WrapperCapability(wrapped=TemporalDurability())],
    )

    with patch('pydantic_ai.durable_exec.temporal._durability.workflow.in_workflow', return_value=True):
        result = await agent.run('hello', capabilities=[Instrumentation(InstrumentationSettings())])
        assert result.output == 'skipped'

        with pytest.raises(UserError, match='Capabilities added per-run inside a Temporal workflow'):
            await agent.run('hello', capabilities=[extra_factory])


# --- get_serialization_name returns None ---


def test_durability_get_serialization_name():
    """TemporalDurability.get_serialization_name() returns None."""
    assert TemporalDurability.get_serialization_name() is None


def test_durability_plugin_requires_durability_capability():
    """`AgentPlugin` raises a clear error when the agent has no `TemporalDurability`."""
    plain_agent = Agent(_durability_fn_model, name='no_cap_agent')
    with pytest.raises(UserError, match='no `TemporalDurability` capability'):
        AgentPlugin(plain_agent)


# --- Toolset without ID raises UserError ---


def test_durability_unwrapped_toolset_without_id_is_allowed():
    """An unwrapped leaf toolset doesn't need an ID because it isn't registered as an activity."""
    durability = TemporalDurability()
    agent = Agent(
        _durability_fn_model,
        name='no_id_test',
        toolsets=[ExternalToolset(tool_defs=[ToolDefinition(name='ext_tool')])],
        capabilities=[durability],
    )
    assert TemporalDurability.from_agent(agent) is not None


# --- temporalize returning non-TemporalWrapperToolset (passthrough / unwrapped leaf) ---


def test_durability_non_temporal_wrapper_toolset_not_in_registry():
    """When temporalize returns a non-TemporalWrapperToolset, it's not added to the registry."""
    agent = Agent(
        _durability_fn_model,
        name='external_ts_test',
        toolsets=[ExternalToolset(tool_defs=[ToolDefinition(name='ext_tool')], id='ext')],
        capabilities=[TemporalDurability()],
    )
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None
    # ExternalToolset is not wrapped into a TemporalWrapperToolset by the default
    # temporalize_toolset, so 'ext' should not appear in _toolsets_by_id.
    assert 'ext' not in bound._toolsets_by_id  # pyright: ignore[reportPrivateUsage]
    # The agent's built-in <agent> FunctionToolset IS wrapped.
    assert '<agent>' in bound._toolsets_by_id  # pyright: ignore[reportPrivateUsage]


# --- get_wrapper_toolset returns None when no temporal toolsets ---


def test_durability_get_wrapper_toolset_returns_none():
    """get_wrapper_toolset returns None when `_toolsets_by_id` is empty."""
    # An unbound capability has an empty registry — `for_agent` is what populates it.
    durability = TemporalDurability()
    assert len(durability._toolsets_by_id) == 0  # pyright: ignore[reportPrivateUsage]

    dummy_toolset = FunctionToolset[object](id='dummy')
    assert durability.get_wrapper_toolset(dummy_toolset) is None


# --- get_wrapper_toolset swap returns unchanged toolset ---


def test_durability_get_wrapper_toolset_swap_unchanged():
    """get_wrapper_toolset's swap returns a toolset unchanged if its ID is not in the registry."""
    agent = Agent(_durability_fn_model, name='swap_test', capabilities=[TemporalDurability()])
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None

    # Create a new toolset not registered with this durability
    unregistered_toolset = FunctionToolset(id='unregistered')
    result = bound.get_wrapper_toolset(unregistered_toolset)
    # The toolset should be returned as-is since its ID is not in the registry
    assert result is unregistered_toolset


# --- `run_sync()` / `run_stream()` / `run_stream_events()` inside a workflow ---
# The deprecated `TemporalAgent` wrapper rejects all three inside a workflow (see
# `test_temporal_agent_run_sync_in_workflow` and friends). The `TemporalDurability`
# capability has no such guards, so these tests pin what the capability actually does:
# the two streaming entry points work, and `run_sync()` does not.
# `test_temporal_durability_buffers_caller_streams` already covers the single-step text
# happy path for both streaming methods; these add the durability `event_stream_handler`
# under `run_stream()` (completing the handler matrix alongside `run()` and `iter()`) and
# a multi-step tool-calling run under `run_stream_events()`.


_run_stream_handler_events: list[tuple[str, bool]] = []


async def _run_stream_durability_handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        _run_stream_handler_events.append((type(event).__name__, activity.in_activity()))


_run_stream_durable_agent = Agent(
    _stream_fn_model,
    name='durability_run_stream_agent',
    capabilities=[
        TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG, event_stream_handler=_run_stream_durability_handler)
    ],
)


@workflow.defn
class RunStreamDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> tuple[str, list[str]]:
        async with _run_stream_durable_agent.run_stream(prompt) as result:
            deltas = [delta async for delta in result.stream_text(delta=True)]
            return await result.get_output(), deltas


async def test_durability_run_stream_in_workflow(client: Client) -> None:
    """`agent.run_stream()` works inside a workflow under the `TemporalDurability` capability.

    The model streams inside the request-stream activity — the capability's handler sees the model
    events with `activity.in_activity()` true — and the workflow-side `StreamedRunResult` is fed by
    the events the activity captured off the live stream, so it stays deterministic across replays.
    The single text delta is not a durability artifact: `run_stream()` consumes events up to the
    `FinalResultEvent` before yielding, so `stream_text(delta=True)` returns the same one chunk for
    this model outside a workflow.
    """
    _run_stream_handler_events.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RunStreamDurableAgentWorkflow],
        plugins=[AgentPlugin(_run_stream_durable_agent)],
    ):
        output, deltas = await client.execute_workflow(
            RunStreamDurableAgentWorkflow.run,
            args=['Hello'],
            id=RunStreamDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    assert output == snapshot('Streamed response')
    assert deltas == snapshot(['Streamed response'])
    assert _run_stream_handler_events == snapshot(
        [
            ('PartStartEvent', True),
            ('FinalResultEvent', True),
            ('PartDeltaEvent', True),
            ('PartDeltaEvent', True),
            ('PartEndEvent', True),
        ]
    )


_run_stream_events_durable_agent = Agent(
    TestModel(custom_output_text='Streamed events output'),
    name='durability_run_stream_events_agent',
    tools=[_durability_reveal_tool],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class RunStreamEventsDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[str]:
        async with _run_stream_events_durable_agent.run_stream_events(prompt) as stream:
            return [type(event).__name__ async for event in stream]


async def test_durability_run_stream_events_in_workflow(client: Client) -> None:
    """`agent.run_stream_events()` works inside a workflow under the `TemporalDurability` capability.

    Model events are replayed workflow-side after each model-request activity completes, so the
    workflow sees the full event stream (including tool call/result events) and the final
    `AgentRunResultEvent`.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RunStreamEventsDurableAgentWorkflow],
        plugins=[AgentPlugin(_run_stream_events_durable_agent)],
    ):
        events = await client.execute_workflow(
            RunStreamEventsDurableAgentWorkflow.run,
            args=['Hello'],
            id=RunStreamEventsDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    assert events == snapshot(
        [
            'PartStartEvent',
            'PartEndEvent',
            'FunctionToolCallEvent',
            'FunctionToolResultEvent',
            'ToolAvailabilityDeltaEvent',
            'PartStartEvent',
            'FinalResultEvent',
            'PartDeltaEvent',
            'PartDeltaEvent',
            'PartDeltaEvent',
            'PartEndEvent',
            'AgentRunResultEvent',
        ]
    )


async def test_durability_streaming_in_workflow(client: Client):
    """`ProcessEventStream` routes model requests through a streaming activity."""
    _stream_events_collected.clear()
    _stream_model_events_in_activity.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[StreamDurableAgentWorkflow],
        plugins=[AgentPlugin(_stream_durable_agent)],
    ):
        output, model_events_in_activity = await client.execute_workflow(
            StreamDurableAgentWorkflow.run,
            args=['Hello streaming'],
            id=StreamDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        # The non-streaming FunctionModel function is NOT used for the streaming activity;
        # instead, request_stream_activity uses the stream_function path.
        # The final response is assembled from the streamed chunks.
        assert output == 'Streamed response'
        assert model_events_in_activity
        assert not any(model_events_in_activity)


# --- ProcessEventStream capability fires workflow-side ---

_process_events_collected: list[AgentStreamEvent] = []

_process_model_events_in_activity: list[bool] = []


async def _process_event_stream_handler(
    ctx: RunContext[object],
    stream: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in stream:
        if isinstance(event, (PartStartEvent, PartDeltaEvent)):
            _process_model_events_in_activity.append(activity.in_activity())
        _process_events_collected.append(event)


_process_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)

_process_durable_agent = Agent(
    _stream_fn_model,
    name='durability_process_agent',
    capabilities=[
        ProcessEventStream(_process_event_stream_handler),
        _process_durability,
    ],
)


@workflow.defn
class ProcessStreamDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> tuple[str, list[bool], list[str]]:
        result = await _process_durable_agent.run(prompt)
        text_chunks = [
            event.delta.content_delta
            for event in _process_events_collected
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta)
        ]
        return result.output, _process_model_events_in_activity, text_chunks


def test_durability_tool_metadata_disables_activity():
    """Tool metadata={'temporal': False} disables activity wrapping for that tool."""

    async def slow_tool() -> str:
        # Registered with the toolset; the test only verifies wrapping.
        return 'slow'  # pragma: no cover

    toolset = FunctionToolset[object](id='meta_toolset')
    toolset.add_function(slow_tool, metadata={'temporal': False})

    agent = Agent(
        _durability_fn_model,
        name='meta_disable_test',
        toolsets=[toolset],
        capabilities=[TemporalDurability()],
    )
    bound = TemporalDurability.from_agent(agent)
    assert bound is not None

    # Should have wrapped the toolset (capability discovered it at for_agent time);
    # the per-tool skip is applied at call time via resolve_tool_activity_config.
    assert 'meta_toolset' in bound._toolsets_by_id  # pyright: ignore[reportPrivateUsage]


async def test_durability_resolves_supported_and_rejected_tool_activity_opt_outs():
    """Capability-owned config preserves every legacy `metadata={'temporal': False}` outcome."""

    async def async_tool() -> str: ...  # pragma: no branch

    def sync_tool() -> str: ...  # pragma: no branch

    toolset = FunctionToolset[None](id='opt_out_tools')
    toolset.add_function(async_tool, metadata={'temporal': False})
    toolset.add_function(sync_tool, metadata={'temporal': False})
    agent = Agent(
        TestModel(),
        name='opt_outs',
        deps_type=type(None),
        toolsets=[toolset],
        capabilities=[TemporalDurability()],
    )
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    tools = await toolset.get_tools(ctx)

    assert (
        durability._resolve_temporal_tool_config(  # pyright: ignore[reportPrivateUsage]
            ToolsetCallToolId('function', toolset_id='opt_out_tools'), tools['async_tool'], 'async_tool'
        )
        is False
    )
    with pytest.raises(UserError, match='non-async tools are run in threads'):
        durability._resolve_temporal_tool_config(  # pyright: ignore[reportPrivateUsage]
            ToolsetCallToolId('function', toolset_id='opt_out_tools'), tools['sync_tool'], 'sync_tool'
        )

    with pytest.raises(UserError, match='dynamic-toolset tools cannot run inside the workflow'):
        durability._resolve_temporal_tool_config(  # pyright: ignore[reportPrivateUsage]
            ToolsetCallToolId('dynamic', toolset_id='dynamic_opt_out'), tools['async_tool'], 'async_tool'
        )

    mcp_toolset = MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='mcp_opt_out')
    mcp_tool = ToolsetTool(
        toolset=mcp_toolset,
        tool_def=ToolDefinition(name='mcp_tool', metadata={'temporal': False}),
        max_retries=1,
        args_validator=TOOL_SCHEMA_VALIDATOR,
    )
    with pytest.raises(UserError, match='MCP tools require the use of IO'):
        durability._resolve_temporal_tool_config(  # pyright: ignore[reportPrivateUsage]
            ToolsetCallToolId('mcp', toolset_id='mcp_opt_out'), mcp_tool, 'mcp_tool'
        )


async def test_durability_mcp_instructions_use_operation_activity_summary(monkeypatch: pytest.MonkeyPatch):
    """The common MCP instructions operation retains Temporal's legacy activity summary."""
    mcp_toolset = MCPToolset(
        StdioTransport(command='python', args=['-m', 'tests.mcp_server']),
        id='instructions_summary',
        include_instructions=True,
    )
    agent = Agent(
        TestModel(),
        name='instructions_summary',
        toolsets=[mcp_toolset],
        capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
    )
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    wrapped = durability._toolsets_by_id['instructions_summary']  # pyright: ignore[reportPrivateUsage]
    dispatched_summaries: list[str] = []

    async def execute_activity(*args: Any, **config: Any) -> str:
        dispatched_summaries.append(config['summary'])
        return 'server instructions'

    monkeypatch.setattr('pydantic_ai.durable_exec.temporal._operation_backend.execute_activity', execute_activity)
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    with patch('pydantic_ai.durable_exec.temporal._durability.workflow.in_workflow', return_value=True):
        assert await wrapped.get_instructions(ctx) == 'server instructions'
    assert dispatched_summaries == ['get instructions: instructions_summary']


async def test_durability_process_event_stream_fires_workflow_side(client: Client):
    """ProcessEventStream sees the real captured events replayed in the workflow."""
    _process_events_collected.clear()
    _process_model_events_in_activity.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ProcessStreamDurableAgentWorkflow],
        plugins=[AgentPlugin(_process_durable_agent)],
    ):
        output, model_events_in_activity, text_chunks = await client.execute_workflow(
            ProcessStreamDurableAgentWorkflow.run,
            args=['Hello'],
            id=ProcessStreamDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == 'Streamed response'
        assert model_events_in_activity
        assert not any(model_events_in_activity)

    assert text_chunks == ['ed ', 'response']


# ==========================================
# TemporalDurability capability — parity with TemporalAgent wrapper tests
# ==========================================
#
# Each test below is the capability-path equivalent of a `TemporalAgent`-based
# test earlier in this file. They assert the same behaviors but use
# `Agent(..., capabilities=[TemporalDurability(...)])` and `AgentPlugin`
# instead of wrapping the agent.


# --- Complex agent: full Logfire span tree ---

complex_durability_for_logfire = TemporalDurability[Deps](
    deps_type=Deps,
    event_stream_handler=event_stream_handler,
    activity_config=BASE_ACTIVITY_CONFIG,
    model_activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=90)),
    toolset_activity_config={
        'durability_complex_country': ActivityConfig(start_to_close_timeout=timedelta(seconds=120)),
    },
)

complex_durable_logfire_agent = Agent(
    model,
    deps_type=Deps,
    output_type=Response,
    capabilities=[complex_durability_for_logfire],
    toolsets=[
        FunctionToolset[Deps](tools=[get_country], id='durability_complex_country'),
        MCPToolset(
            StdioTransport(command='python', args=['-m', 'tests.mcp_server']),
            id='durability_complex_mcp',
            init_timeout=20,
        ),
        ExternalToolset(tool_defs=[ToolDefinition(name='external')], id='durability_complex_external'),
    ],
    tools=[get_weather],
    name='durability_complex_agent_logfire',
)


@workflow.defn
class ComplexDurableAgentLogfireWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: Deps) -> Response:
        result = await complex_durable_logfire_agent.run(prompt, deps=deps)
        return result.output


async def test_durability_complex_agent_logfire_span_tree(
    allow_model_requests: None, client_with_logfire: Client, capfire: CaptureLogfire
):
    """Capability-path equivalent of `test_complex_agent_run_in_workflow`.

    Asserts the Logfire span tree shape — span names will use
    `agent__durability_complex_agent_logfire__*` instead of `agent__complex_agent__*`,
    but the structure should otherwise match. Run with `--inline-snapshot=create`
    to populate the expected value on first run; needs a fresh VCR cassette under
    the new test name (record in CI / locally with `--record-mode=once`).
    """
    async with Worker(
        client_with_logfire,
        task_queue=TASK_QUEUE,
        workflows=[ComplexDurableAgentLogfireWorkflow],
        plugins=[AgentPlugin(complex_durable_logfire_agent)],
    ):
        output = await client_with_logfire.execute_workflow(
            ComplexDurableAgentLogfireWorkflow.run,
            args=[
                'Tell me: the capital of the country; the weather there; the product name',
                Deps(country='Mexico'),
            ],
            id=ComplexDurableAgentLogfireWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            Response(
                answers=[
                    Answer(label='Capital of the country', answer='Mexico City'),
                    Answer(label='Weather in the capital', answer='Sunny'),
                    Answer(label='Product Name', answer='Pydantic AI'),
                ]
            )
        )
    exporter = capfire.exporter

    spans = exporter.exported_spans_as_dict()
    basic_spans_by_id = {
        span['context']['span_id']: BasicSpan(
            parent_id=span['parent']['span_id'] if span['parent'] else None,
            content=attributes.get('event') or attributes['logfire.msg'],
        )
        for span in spans
        if (attributes := span.get('attributes'))
    }
    root_span = None
    for basic_span in basic_spans_by_id.values():
        if basic_span.parent_id is None:
            root_span = basic_span
        else:
            parent_id = basic_span.parent_id
            parent_span = basic_spans_by_id[parent_id]
            parent_span.children.append(basic_span)

    def _normalize_json_spans(span: BasicSpan) -> None:
        """Normalize non-deterministic tool_call_ids in JSON event spans."""
        import json

        for child in span.children:
            if child.content.startswith('{'):
                try:
                    data = json.loads(child.content)
                    _strip_volatile_fields(data)
                    child.content = json.dumps(data)
                except json.JSONDecodeError:
                    pass
            _normalize_json_spans(child)

    def _strip_volatile_fields(obj: dict[str, Any]) -> None:
        for k, v in obj.items():
            if k in ('tool_call_id', 'timestamp'):
                obj[k] = None
            elif isinstance(v, dict):
                _strip_volatile_fields(cast(dict[str, Any], v))

    assert root_span is not None
    _normalize_json_spans(root_span)

    assert root_span == snapshot(
        BasicSpan(
            content='StartWorkflow:ComplexDurableAgentLogfireWorkflow',
            children=[
                BasicSpan(content='RunWorkflow:ComplexDurableAgentLogfireWorkflow'),
                BasicSpan(
                    content='durability_complex_agent_logfire run',
                    children=IsList(
                        BasicSpan(
                            content='StartActivity:agent__durability_complex_agent_logfire__mcp_server__durability_complex_mcp__get_tools',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__durability_complex_agent_logfire__mcp_server__durability_complex_mcp__get_tools',
                                    children=[BasicSpan(content='tools/list')],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__durability_complex_agent_logfire__model_request_stream',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__durability_complex_agent_logfire__model_request_stream',
                                            children=[
                                                BasicSpan(content='ctx.run_step=1'),
                                                BasicSpan(
                                                    content='{"index": 0, "part": {"tool_name": "get_country", "args": "", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "previous_part_kind": null, "event_kind": "part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "{}", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "part": {"tool_name": "get_country", "args": "{}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "next_part_kind": "tool-call", "event_kind": "part_end"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 1, "part": {"tool_name": "get_product_name", "args": "", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "previous_part_kind": "tool-call", "event_kind": "part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 1, "delta": {"tool_name_delta": null, "args_delta": "{}", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 1, "part": {"tool_name": "get_product_name", "args": "{}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "next_part_kind": null, "event_kind": "part_end"}'
                                                ),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_country", "args": "{}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "function_tool_call"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_product_name", "args": "{}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "function_tool_call"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='running tool: get_country',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__durability_complex_agent_logfire__toolset__durability_complex_country__call_tool',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__durability_complex_agent_logfire__toolset__durability_complex_country__call_tool'
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_country", "content": "Mexico", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "content": null, "event_kind": "function_tool_result"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='running tool: get_product_name',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__durability_complex_agent_logfire__mcp_server__durability_complex_mcp__call_tool',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__durability_complex_agent_logfire__mcp_server__durability_complex_mcp__call_tool',
                                            children=[BasicSpan(content='tools/call get_product_name')],
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_product_name", "content": "Pydantic AI", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "content": null, "event_kind": "function_tool_result"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__durability_complex_agent_logfire__model_request_stream',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__durability_complex_agent_logfire__model_request_stream',
                                            children=[
                                                BasicSpan(content='ctx.run_step=2'),
                                                BasicSpan(
                                                    content='{"index": 0, "part": {"tool_name": "get_weather", "args": "", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "previous_part_kind": null, "event_kind": "part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "{\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "city", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "Mexico", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " City", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\"}", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "part": {"tool_name": "get_weather", "args": "{\\"city\\":\\"Mexico City\\"}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "next_part_kind": null, "event_kind": "part_end"}'
                                                ),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_weather", "args": "{\\"city\\":\\"Mexico City\\"}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "function_tool_call"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='running tool: get_weather',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__durability_complex_agent_logfire__toolset__<agent>__call_tool',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__durability_complex_agent_logfire__toolset__<agent>__call_tool'
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=2'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_weather", "content": "sunny", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "content": null, "event_kind": "function_tool_result"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__durability_complex_agent_logfire__model_request_stream',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__durability_complex_agent_logfire__model_request_stream',
                                            children=[
                                                BasicSpan(content='ctx.run_step=3'),
                                                BasicSpan(
                                                    content='{"index": 0, "part": {"tool_name": "final_result", "args": "", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "previous_part_kind": null, "event_kind": "part_start"}'
                                                ),
                                                BasicSpan(
                                                    content='{"tool_name": "final_result", "tool_call_id": null, "event_kind": "final_result"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "{\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "answers", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":[", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "{\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "label", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "Capital", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " of", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " the", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " country", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\",\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "answer", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "Mexico", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " City", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\"},{\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "label", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "Weather", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " in", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " the", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " capital", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\",\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "answer", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "Sunny", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\"},{\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "label", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "Product", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " Name", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\",\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "answer", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\":\\"", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "P", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "yd", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "antic", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": " AI", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "\\"}", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "delta": {"tool_name_delta": null, "args_delta": "]}", "tool_call_id": null, "provider_name": null, "provider_details": null, "part_delta_kind": "tool_call"}, "event_kind": "part_delta"}'
                                                ),
                                                BasicSpan(
                                                    content='{"index": 0, "part": {"tool_name": "final_result", "args": "{\\"answers\\":[{\\"label\\":\\"Capital of the country\\",\\"answer\\":\\"Mexico City\\"},{\\"label\\":\\"Weather in the capital\\",\\"answer\\":\\"Sunny\\"},{\\"label\\":\\"Product Name\\",\\"answer\\":\\"Pydantic AI\\"}]}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "next_part_kind": null, "event_kind": "part_end"}'
                                                ),
                                            ],
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=3'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "final_result", "args": "{\\"answers\\":[{\\"label\\":\\"Capital of the country\\",\\"answer\\":\\"Mexico City\\"},{\\"label\\":\\"Weather in the capital\\",\\"answer\\":\\"Sunny\\"},{\\"label\\":\\"Product Name\\",\\"answer\\":\\"Pydantic AI\\"}]}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "output_tool_call"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__durability_complex_agent_logfire__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=3'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "final_result", "content": "Final result processed.", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "event_kind": "output_tool_result"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        check_order=False,
                    ),
                ),
                BasicSpan(content='CompleteWorkflow:ComplexDurableAgentLogfireWorkflow'),
            ],
        )
    )


# --- Model retry ---


_durability_model_retry_agent = Agent(model, name='durability_model_retry_agent', capabilities=[TemporalDurability()])


@_durability_model_retry_agent.tool_plain
def durability_get_weather_in_city(city: str) -> str:
    if city != 'Mexico City':
        raise ModelRetry('Did you mean Mexico City?')
    return 'sunny'


@workflow.defn
class DurabilityModelRetryWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str]:
        result = await _durability_model_retry_agent.run(prompt)
        return result


async def test_durability_agent_with_model_retry(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_temporal_agent_with_model_retry`.

    Needs a fresh VCR cassette (different test name from the wrapper test);
    record locally with `--record-mode=once`.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityModelRetryWorkflow],
        plugins=[AgentPlugin(_durability_model_retry_agent)],
    ):
        wf = await client.start_workflow(
            DurabilityModelRetryWorkflow.run,
            args=['What is the weather in CDMX?'],
            id=DurabilityModelRetryWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        result = await wf.result()
        assert result.output == snapshot('The weather in Mexico City is currently sunny.')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='What is the weather in CDMX?', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='durability_get_weather_in_city',
                            args='{"city":"CDMX"}',
                            tool_call_id='call_TtLEMpCeAhnG48btCDrw8lhl',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=48,
                        output_tokens=20,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                        cost=Decimal('0.00032'),
                        output_reasoning_tokens=0,
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls', 'timestamp': '2026-05-08T21:37:16Z'},
                    provider_response_id='chatcmpl-DdNAiT49qrYrZOaeeAd39RynAa1g7',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Did you mean Mexico City?',
                            tool_name='durability_get_weather_in_city',
                            tool_call_id='call_TtLEMpCeAhnG48btCDrw8lhl',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='durability_get_weather_in_city',
                            args='{"city":"Mexico City"}',
                            tool_call_id='call_d8k0Vk8dw6eWKFWF8Dj0rCL6',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=93,
                        output_tokens=20,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                        cost=Decimal('0.0004325'),
                        output_reasoning_tokens=0,
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls', 'timestamp': '2026-05-08T21:37:17Z'},
                    provider_response_id='chatcmpl-DdNAjt5pJt1nYbeCdbHGbo4ntTKy8',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='durability_get_weather_in_city',
                            content='sunny',
                            tool_call_id='call_d8k0Vk8dw6eWKFWF8Dj0rCL6',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='The weather in Mexico City is currently sunny.')],
                    usage=RequestUsage(
                        input_tokens=127,
                        output_tokens=10,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                        cost=Decimal('0.0004175'),
                        output_reasoning_tokens=0,
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'stop', 'timestamp': '2026-05-08T21:37:18Z'},
                    provider_response_id='chatcmpl-DdNAkzvAFU1knSut20EiutyMs7PZy',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


# --- Multi-model selection by ID ---

_durability_model_1 = TestModel(custom_output_text='Response from model 1')

_durability_model_2 = TestModel(custom_output_text='Response from model 2')

_durability_model_3 = TestModel(custom_output_text='Response from model 3')


_durability_multi_model_agent = Agent(
    _durability_model_1,
    name='durability_multi_model_agent',
    capabilities=[
        TemporalDurability(
            models={
                'model_2': _durability_model_2,
                'model_3': _durability_model_3,
            },
            activity_config=BASE_ACTIVITY_CONFIG,
        )
    ],
)


@workflow.defn
class DurabilityMultiModelWorkflow:
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await _durability_multi_model_agent.run(prompt, model=model_id)
        return result.output


async def test_durability_multi_model_selection_in_workflow(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_temporal_agent_multi_model_selection_in_workflow`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityMultiModelWorkflow],
        plugins=[AgentPlugin(_durability_multi_model_agent)],
    ):
        # Default model (no model arg)
        output = await client.execute_workflow(
            DurabilityMultiModelWorkflow.run,
            args=['Hello', None],
            id='DurabilityMultiModelWorkflow_default',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 1'

        # Selecting registered second model by ID
        output = await client.execute_workflow(
            DurabilityMultiModelWorkflow.run,
            args=['Hello', 'model_2'],
            id='DurabilityMultiModelWorkflow_model2',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 2'

        # Selecting registered third model by ID
        output = await client.execute_workflow(
            DurabilityMultiModelWorkflow.run,
            args=['Hello', 'model_3'],
            id='DurabilityMultiModelWorkflow_model3',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 3'


# --- Model selection by instance ---

_durability_model_instance_map = {
    'default_instance': _durability_model_1,
    'model_2_instance': _durability_model_2,
}


@workflow.defn
class DurabilityMultiModelInstanceWorkflow:
    @workflow.run
    async def run(self, prompt: str, instance_key: str) -> str:
        model_instance = _durability_model_instance_map[instance_key]
        result = await _durability_multi_model_agent.run(prompt, model=model_instance)
        return result.output


@pytest.mark.parametrize(
    ('instance_key', 'expected_output'),
    [
        pytest.param('default_instance', 'Response from model 1', id='default_instance'),
        pytest.param('model_2_instance', 'Response from model 2', id='registered_instance'),
    ],
)
async def test_durability_model_selection_by_instance(
    allow_model_requests: None, client: Client, instance_key: str, expected_output: str
):
    """Capability-path equivalent of `test_temporal_agent_model_selection_by_instance`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityMultiModelInstanceWorkflow],
        plugins=[AgentPlugin(_durability_multi_model_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityMultiModelInstanceWorkflow.run,
            args=['Hello', instance_key],
            id=f'DurabilityMultiModelInstanceWorkflow_{instance_key}',
            task_queue=TASK_QUEUE,
        )
        assert output == expected_output


# --- Web search builtin tool ---

_durability_web_search_agent = Agent(
    web_search_model,
    name='durability_web_search_agent',
    capabilities=[
        NativeTool(WebSearchTool(user_location=WebSearchUserLocation(city='Mexico City', country='MX'))),
        TemporalDurability(
            activity_config=BASE_ACTIVITY_CONFIG,
            model_activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=300)),
        ),
    ],
)


@workflow.defn
class DurabilityWebSearchAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _durability_web_search_agent.run(prompt)
        return result.output


async def test_durability_web_search_in_workflow(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_web_search_agent_run_in_workflow`.

    Needs a fresh VCR cassette (different test name from the wrapper test);
    record in CI / locally with `--record-mode=once`.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityWebSearchAgentWorkflow],
        plugins=[AgentPlugin(_durability_web_search_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityWebSearchAgentWorkflow.run,
            args=['In one sentence, what is the top news story in my country today?'],
            id=DurabilityWebSearchAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            "Mexico's central bank cut its benchmark interest rate by 25 basis points to 6.50%--effective today, May 8, 2026--signaling the end of its rate‐cut cycle. ([banxico.org.mx](https://www.banxico.org.mx/publicaciones-y-prensa/anuncios-de-las-decisiones-de-politica-monetaria/%7B8A05C722-0A97-4527-2166-0CE802CE6838%7D.pdf?utm_source=openai))"
        )


# --- Dynamic builtin tools select-by-model ---

_durability_builtin_tool_agent = Agent(
    web_search_builtin_model,
    name='durability_builtin_tool_dynamic_agent',
    capabilities=[
        NativeTool(_select_builtin_tool),
        TemporalDurability(
            models={'code': code_execution_builtin_model},
            activity_config=BASE_ACTIVITY_CONFIG,
        ),
    ],
)


@workflow.defn
class DurabilityBuiltinToolWorkflow:
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await _durability_builtin_tool_agent.run(prompt, model=model_id)
        return result.output


async def test_durability_dynamic_builtin_tools_select_by_model(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_temporal_dynamic_builtin_tools_select_by_model`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityBuiltinToolWorkflow],
        plugins=[AgentPlugin(_durability_builtin_tool_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityBuiltinToolWorkflow.run,
            args=['Hello', None],
            id='DurabilityBuiltinToolWorkflow_default',
            task_queue=TASK_QUEUE,
        )
        assert output == 'search model'
        assert isinstance(web_search_builtin_model.last_model_request_parameters, ModelRequestParameters)
        assert web_search_builtin_model.last_model_request_parameters.native_tools
        assert isinstance(web_search_builtin_model.last_model_request_parameters.native_tools[0], WebSearchTool)

        output = await client.execute_workflow(
            DurabilityBuiltinToolWorkflow.run,
            args=['Hello', 'code'],
            id='DurabilityBuiltinToolWorkflow_code',
            task_queue=TASK_QUEUE,
        )
        assert output == 'code model'
        assert isinstance(code_execution_builtin_model.last_model_request_parameters, ModelRequestParameters)
        assert code_execution_builtin_model.last_model_request_parameters.native_tools
        assert isinstance(
            code_execution_builtin_model.last_model_request_parameters.native_tools[0],
            CodeExecutionTool,
        )


# --- @agent.toolset returning an MCP toolset ---

_durability_mcp_dynamic_toolset_agent = Agent(
    model,
    name='durability_mcp_dynamic_toolset_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@_durability_mcp_dynamic_toolset_agent.toolset(id='durability_mcp_toolset')
def _durability_my_mcp_dynamic_toolset(ctx: RunContext[object]) -> MCPToolset[object]:
    # Exercised only by the skipped test below.
    return MCPToolset('https://mcp.deepwiki.com/mcp')  # pragma: no cover


@workflow.defn
class DurabilityMCPDynamicToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        # This body runs only under the skipped test below.
        result = await _durability_mcp_dynamic_toolset_agent.run(prompt)  # pragma: no cover
        return result.output  # pragma: no cover


@pytest.mark.skip(
    reason=(
        'Pending: replays of this MCP toolset workflow trip the Temporal sandbox with '
        '`Module certifi was imported after initial workflow load`. Issue tracked.'
    )
)
async def test_durability_mcp_dynamic_toolset_in_workflow(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_mcp_dynamic_toolset_in_workflow`.

    Needs a fresh VCR cassette (different test name from the wrapper test);
    record in CI / locally with `--record-mode=once`.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityMCPDynamicToolsetAgentWorkflow],
        plugins=[AgentPlugin(_durability_mcp_dynamic_toolset_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityMCPDynamicToolsetAgentWorkflow.run,
            args=['Can you tell me about the pydantic/pydantic-ai repo? Keep it short.'],
            id='test_durability_mcp_dynamic_toolset_workflow',
            task_queue=TASK_QUEUE,
        )
        # The deepwiki MCP server should return info about the pydantic-ai repo
        assert 'pydantic' in output.lower() or 'agent' in output.lower()


# --- MCPToolset over HTTP ---

_durability_mcptoolset_agent = Agent(
    model,
    name='durability_mcptoolset_agent',
    toolsets=[MCPToolset('https://mcp.deepwiki.com/mcp', id='durability_deepwiki')],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class DurabilityMCPToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        # This body runs only under the skipped test below.
        result = await _durability_mcptoolset_agent.run(prompt)  # pragma: no cover
        return result.output  # pragma: no cover


@pytest.mark.skip(
    reason=(
        'Pending: replays of this MCP toolset workflow trip the Temporal sandbox with '
        '`Module certifi was imported after initial workflow load`. Issue tracked.'
    )
)
async def test_durability_mcptoolset_in_workflow(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_mcptoolset_in_temporal_workflow`.

    Needs a fresh VCR cassette (different test name from the wrapper test);
    record in CI / locally with `--record-mode=once`.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityMCPToolsetAgentWorkflow],
        plugins=[AgentPlugin(_durability_mcptoolset_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityMCPToolsetAgentWorkflow.run,
            args=['Can you tell me more about the pydantic/pydantic-ai repo? Keep your answer short'],
            id=DurabilityMCPToolsetAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot()


# --- @agent.toolset returning a FunctionToolset ---

_durability_dynamic_toolset_agent = Agent(
    TestModel(),
    name='durability_dynamic_toolset_agent',
    deps_type=DynamicToolsetDeps,
    capabilities=[
        TemporalDurability[DynamicToolsetDeps](deps_type=DynamicToolsetDeps, activity_config=BASE_ACTIVITY_CONFIG)
    ],
)


@_durability_dynamic_toolset_agent.toolset(id='durability_my_dynamic_tools')
def _durability_my_dynamic_toolset(ctx: RunContext[DynamicToolsetDeps]) -> FunctionToolset[DynamicToolsetDeps]:
    toolset = FunctionToolset[DynamicToolsetDeps](id='durability_dynamic_weather')

    @toolset.tool_plain
    def get_dynamic_weather(location: str) -> str:
        """Get the weather for a location."""
        user = ctx.deps.user_name
        return f'Weather in {location} for {user}: sunny.'

    return toolset


@workflow.defn
class DurabilityDynamicToolsetAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: DynamicToolsetDeps) -> str:
        result = await _durability_dynamic_toolset_agent.run(prompt, deps=deps)
        return result.output


async def test_durability_dynamic_toolset_in_workflow(client: Client):
    """Capability-path equivalent of `test_dynamic_toolset_in_workflow`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityDynamicToolsetAgentWorkflow],
        plugins=[AgentPlugin(_durability_dynamic_toolset_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityDynamicToolsetAgentWorkflow.run,
            args=['Get the weather for London', DynamicToolsetDeps(user_name='Alice')],
            id='test_durability_dynamic_toolset_workflow',
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('{"get_dynamic_weather":"Weather in a for Alice: sunny."}')


def _dynamic_activity_config_toolset(ctx: RunContext[Any]) -> FunctionToolset[Any]:
    toolset = FunctionToolset[Any](id='dynamic_activity_config_inner')

    @toolset.tool_plain(metadata={'temporal': ActivityConfig(start_to_close_timeout=timedelta(seconds=30))})
    def timed_tool() -> str:
        assert activity.in_activity()
        return 'timed result'

    return toolset


# Passed at construction time so the durability capability actually wraps it (see #6902).
_dynamic_activity_config_agent = Agent(
    TestModel(),
    name='dynamic_activity_config_agent',
    toolsets=[DynamicToolset(_dynamic_activity_config_toolset, id='dynamic_activity_config')],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class DynamicToolActivityConfigWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await _dynamic_activity_config_agent.run('Call the tool')).output


async def test_durability_dynamic_tool_timedelta_activity_config_survives_round_trip(client: Client):
    """A `timedelta` in a dynamic tool's `ActivityConfig` metadata reaches `execute_activity` intact.

    The tool is discovered inside the get-tools activity, so its metadata comes back to the
    workflow as JSON and the `timedelta` arrives as the string `'PT5M'`. Handing that to
    `workflow.execute_activity` raises inside protobuf's `Duration.FromTimedelta`, which is a
    workflow-*task* failure that Temporal retries forever — hence the short `execution_timeout`,
    so a regression fails the test instead of hanging it.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DynamicToolActivityConfigWorkflow],
        plugins=[AgentPlugin(_dynamic_activity_config_agent)],
    ):
        output = await client.execute_workflow(
            DynamicToolActivityConfigWorkflow.run,
            id='test_dynamic_tool_activity_config',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
    assert output == snapshot('{"timed_tool":"timed result"}')


@dataclass
class _TemporalDynamicToolCapability(AbstractCapability[Any]):
    def get_toolset(self) -> FunctionToolset[Any]:
        toolset = FunctionToolset[Any]()

        @toolset.tool_plain
        def dynamic_capability_tool() -> str:
            assert activity.in_activity()
            return 'called in activity'

        return toolset


def _temporal_dynamic_capability_factory(ctx: RunContext[Any]) -> AbstractCapability[Any]:
    return _TemporalDynamicToolCapability()


_temporal_dynamic_capability_agent = Agent(
    TestModel(),
    name='temporal_dynamic_capability_agent',
    capabilities=[
        DynamicCapability(_temporal_dynamic_capability_factory, id='dyn'),
        TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class TemporalDynamicCapabilityWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await _temporal_dynamic_capability_agent.run('Call the tool')).output


async def test_durability_dynamic_capability_tool_runs_in_activity(client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[TemporalDynamicCapabilityWorkflow],
        plugins=[AgentPlugin(_temporal_dynamic_capability_agent)],
    ):
        output = await client.execute_workflow(
            TemporalDynamicCapabilityWorkflow.run,
            id='test_temporal_dynamic_capability',
            task_queue=TASK_QUEUE,
        )
    assert output == '{"dynamic_capability_tool":"called in activity"}'


def test_durability_dynamic_capability_requires_id() -> None:
    with pytest.raises(UserError, match=r"DynamicCapability\(\.\.\., id='user-tools'\)"):
        Agent(
            TestModel(),
            name='idless_dynamic_capability',
            capabilities=[
                DynamicCapability(_temporal_dynamic_capability_factory),
                TemporalDurability(),
            ],
        )


async def test_durability_dynamic_capability_transparent_outside_workflow():
    """Outside a workflow, dynamic-capability tools resolve and run inline, not via activities.

    The durable wrapper's `for_run` must hand the run the *resolved* dynamic toolset:
    delegating to the unresolved construction-time factory would silently contribute no tools.
    """
    in_activity_flags: list[bool] = []

    def dynamic_tool() -> str:
        in_activity_flags.append(activity.in_activity())
        return 'inline result'

    def factory(ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return Capability(tools=[dynamic_tool])

    agent = Agent(
        TestModel(),
        name='temporal_dynamic_capability_outside',
        capabilities=[
            DynamicCapability(factory, id='dyn_outside'),
            TemporalDurability(),
        ],
    )

    result = await agent.run('Call the tool')
    assert result.output == '{"dynamic_tool":"inline result"}'
    assert in_activity_flags == [False]


# --- ToolReturn metadata round-trip ---


def _durability_tool_return_metadata_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('durability_analyze_data', {})])
    else:
        return ModelResponse(parts=[TextPart('done')])


_durability_tool_return_metadata_agent = Agent(
    FunctionModel(_durability_tool_return_metadata_model),
    name='durability_tool_return_metadata_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@_durability_tool_return_metadata_agent.tool_plain
def durability_analyze_data() -> ToolReturn:
    return ToolReturn(
        return_value='analysis result',
        content='extra content for model',
        metadata={'key': 'value', 'count': 42},
    )


@workflow.defn
class DurabilityToolReturnMetadataWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[ModelMessage]:
        result = await _durability_tool_return_metadata_agent.run(prompt)
        return result.all_messages()


async def test_durability_tool_return_metadata_survives(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_tool_return_metadata_survives_temporal`.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/4676 — `ToolReturn`
    `metadata` and `content` survive Temporal serialization on the capability path too.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityToolReturnMetadataWorkflow],
        plugins=[AgentPlugin(_durability_tool_return_metadata_agent)],
    ):
        messages = await client.execute_workflow(
            DurabilityToolReturnMetadataWorkflow.run,
            args=['analyze'],
            id=DurabilityToolReturnMetadataWorkflow.__name__,
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
                parts=[ToolCallPart(tool_name='durability_analyze_data', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt()),
                model_name='function:_durability_tool_return_metadata_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='durability_analyze_data',
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
                usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt()),
                model_name='function:_durability_tool_return_metadata_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


# --- Deferred tool reveal round-trip ---


def _durability_reveal_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    tool_names = {tool.name for tool in info.function_tools}
    responses = sum(isinstance(message, ModelResponse) for message in messages)
    if responses == 0:
        assert 'durability_refund' not in tool_names
        return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'billing'}, tool_call_id='load')])
    if responses == 1:
        assert 'durability_refund' in tool_names
        return ModelResponse(parts=[ToolCallPart('durability_refund', {}, tool_call_id='refund')])
    if responses == 2:
        assert 'durability_hidden' not in tool_names
        return ModelResponse(parts=[ToolCallPart('durability_opener', {}, tool_call_id='open')])
    if responses == 3:
        assert 'durability_hidden' in tool_names
        return ModelResponse(parts=[ToolCallPart('durability_hidden', {}, tool_call_id='hidden')])
    return ModelResponse(parts=[TextPart('done')])


_durability_billing = Capability[None](id='billing', defer_loading=True)


@_durability_billing.tool
def durability_refund(ctx: RunContext[None]) -> str:
    # The always-visible check exercises the availability snapshot carried across the activity
    # boundary: `durability_opener` is never revealed, so the `discovered_tool_names` fallback
    # alone would answer False for it inside the activity.
    return (
        f'refund available: {ctx.is_tool_available("durability_refund")}, '
        f'opener available: {ctx.is_tool_available("durability_opener")}'
    )


_durability_reveal_agent = Agent(
    FunctionModel(_durability_reveal_model),
    name='durability_reveal_agent',
    deps_type=type(None),
    capabilities=[_durability_billing, TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@_durability_reveal_agent.tool
def durability_opener(ctx: RunContext[None]) -> ToolReturn[str]:
    return ToolReturn(
        return_value='opened',
        tools=['durability_hidden'],
    )


@_durability_reveal_agent.tool_plain(defer_loading=True)
def durability_hidden() -> str:
    return 'secret'


@workflow.defn
class DurabilityRevealWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[ModelMessage]:
        result = await _durability_reveal_agent.run(prompt)
        return result.all_messages()


async def test_durability_tool_reveals_survive_workflow_and_activity(allow_model_requests: None, client: Client):
    """Capability and activity-authored reveals both become durable history facts."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityRevealWorkflow],
        plugins=[AgentPlugin(_durability_reveal_agent)],
    ):
        messages = await client.execute_workflow(
            DurabilityRevealWorkflow.run,
            args=['refund and open'],
            id=DurabilityRevealWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    deltas = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    assert [(part.tools_added, part.tool_call_id) for part in deltas] == [
        (['durability_refund'], 'load'),
        (['durability_hidden'], 'open'),
    ]
    returns = {
        part.tool_name: part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    assert returns['durability_refund'] == 'refund available: True, opener available: True'
    assert returns['durability_opener'] == 'opened'


# A fallback model cannot exercise Temporal's re-preparation seam: `FallbackModel.request()`
# prepares the history separately for every inner model, so the required mutation would still pass.
# Use raw model IDs across workflow executions instead, so only the worker-side concrete model can
# project the serialized reveal history.
def _cross_model_reveal_secondary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    parts = [part for message in messages for part in message.parts]
    assert not any(isinstance(part, ToolAvailabilityDeltaPart) for part in parts)
    assert any(
        isinstance(part, UserPromptPart)
        and part.content == '<system>The following tool(s) are now available: `cross_model_refund`</system>'
        for part in parts
    )
    assert 'cross_model_refund' in {tool.name for tool in info.function_tools}
    if not any(isinstance(part, ToolReturnPart) and part.tool_name == 'cross_model_refund' for part in parts):
        return ModelResponse(parts=[ToolCallPart('cross_model_refund', {}, tool_call_id='refund')])
    return ModelResponse(parts=[TextPart('refund complete')])


def _cross_model_reveal_primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    deltas = [part for message in messages for part in message.parts if isinstance(part, ToolAvailabilityDeltaPart)]
    if not deltas:
        return ModelResponse(
            parts=[ToolCallPart('load_capability', {'id': 'cross-model-billing'}, tool_call_id='load')]
        )
    assert [(part.tools_added, part.tool_call_id) for part in deltas] == [(['cross_model_refund'], 'load')]
    return ModelResponse(parts=[TextPart('capability loaded')], usage=RequestUsage(input_tokens=1, output_tokens=1))


def _infer_cross_model(model_id: Any, **kwargs: Any) -> Model:
    if model := _cross_model_reveal_models.get(str(model_id)):
        return model
    return infer_model(model_id, **kwargs)


_cross_model_reveal_models = {
    'openai:cross-model-secondary': FunctionModel(
        _cross_model_reveal_secondary,
        model_name='cross-model-secondary',
        profile=ModelProfile(),
    ),
    'anthropic:cross-model-primary': FunctionModel(
        _cross_model_reveal_primary,
        model_name='cross-model-primary',
        profile=ModelProfile(tool_addition_mode='by_reference', tool_deferral_mode='standalone'),
    ),
}


_cross_model_billing = Capability[None](id='cross-model-billing', defer_loading=True)


@_cross_model_billing.tool
def cross_model_refund(ctx: RunContext[None]) -> str:
    return f'refund available in activity: {ctx.is_tool_available("cross_model_refund")}'


_cross_model_reveal_base_agent = Agent(
    _cross_model_reveal_models['openai:cross-model-secondary'],
    name='cross_model_reveal_agent',
    deps_type=type(None),
    capabilities=[
        _cross_model_billing,
    ],
)

_cross_model_reveal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    _cross_model_reveal_base_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


@dataclass
class CrossModelRevealResult:
    output: str
    messages: list[ModelMessage]


@workflow.defn
class CrossModelRevealWorkflow:
    @workflow.run
    async def run(
        self, prompt: str, model_id: str, message_history: list[ModelMessage] | None
    ) -> CrossModelRevealResult:
        result = await _cross_model_reveal_agent.run(prompt, model=model_id, message_history=message_history)
        return CrossModelRevealResult(output=result.output, messages=result.all_messages())


async def test_durability_reprepares_reveal_history_for_different_model(client: Client):
    """A serialized reveal is projected onto a different model's channel in a later workflow.

    Raw model IDs keep message preparation out of the workflow. The channel-bearing primary
    authors the reveal; the channel-less secondary receives an announcement, then calls the
    newly available tool inside an activity.
    """
    with patch(
        'pydantic_ai.durable_exec.temporal._model.models.infer_model',
        side_effect=_infer_cross_model,
    ):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[CrossModelRevealWorkflow],
            plugins=[AgentPlugin(_cross_model_reveal_agent)],
        ):
            first = await client.execute_workflow(
                CrossModelRevealWorkflow.run,
                args=['load refund capability', 'anthropic:cross-model-primary', None],
                id=f'{CrossModelRevealWorkflow.__name__}-primary',
                task_queue=TASK_QUEUE,
            )
            second = await client.execute_workflow(
                CrossModelRevealWorkflow.run,
                args=['issue refund', 'openai:cross-model-secondary', first.messages],
                id=f'{CrossModelRevealWorkflow.__name__}-secondary',
                task_queue=TASK_QUEUE,
            )

    assert first.output == 'capability loaded'
    assert second.output == 'refund complete'
    tool_return = next(
        part.content
        for message in second.messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'cross_model_refund'
    )
    assert tool_return == 'refund available in activity: True'


# --- Passing image (BinaryImage) input through to a workflow ---

_durability_multimodal_agent = Agent(
    TestModel(),
    name='durability_multimodal_content_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@_durability_multimodal_agent.tool
def _durability_get_multimodal_content(ctx: RunContext[object]) -> list[str | MultiModalContent]:
    """Return a list with text, BinaryContent, and DocumentUrl."""
    return [
        'test',
        BinaryImage(data=b'\x89PNG', media_type='image/png'),
        DocumentUrl(url='https://example.com/doc/12345', media_type='application/pdf'),
    ]


@workflow.defn
class DurabilityMultiModalContentWorkflow:
    @workflow.run
    async def run(self, prompt: list[UserContent]) -> list[ModelMessage]:
        result = await _durability_multimodal_agent.run(prompt)
        return result.all_messages()


async def test_durability_passing_image_to_run(client: Client):
    """Capability-path equivalent of `test_multimodal_content_serialization_in_workflow` — image input.

    Verifies BinaryImage / DocumentUrl survive Temporal serialization both as workflow
    input and as tool return values when running on the TemporalDurability capability path.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityMultiModalContentWorkflow],
        plugins=[AgentPlugin(_durability_multimodal_agent)],
    ):
        prompt: list[str | MultiModalContent] = [
            'Process these files and call the tool',
            BinaryImage(data=b'\x89PNG', media_type='image/png'),
            DocumentUrl(url='https://example.com/doc/12345', media_type='application/pdf'),
        ]
        messages = await client.execute_workflow(
            DurabilityMultiModalContentWorkflow.run,
            args=[prompt],
            id='test_durability_passing_image_to_run',
            task_queue=TASK_QUEUE,
        )

    # media_type is preserved through serialization for both BinaryContent and DocumentUrl.
    media_types: list[tuple[str, str]] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                for content in part.content:
                    if isinstance(content, (BinaryContent, DocumentUrl)):
                        media_types.append((type(content).__name__, content.media_type))
            elif isinstance(part, ToolReturnPart):
                for content in part.content_items():
                    if isinstance(content, (BinaryContent, DocumentUrl)):
                        media_types.append((type(content).__name__, content.media_type))
    # The image `BinaryContent` round-trips as `BinaryImage`: narrowing is applied during
    # validation on the way back across the activity boundary.
    assert media_types == [
        ('BinaryImage', 'image/png'),
        ('DocumentUrl', 'application/pdf'),
        ('BinaryImage', 'image/png'),
        ('DocumentUrl', 'application/pdf'),
    ]


# --- UploadedFile output round-trip ---

_durability_uploaded_file_agent = Agent(
    TestModel(
        custom_output_args={
            'file_id': 'file-abc123',
            'provider_name': 'openai',
            'media_type': 'image/png',
            'identifier': 'file-1',
        }
    ),
    name='durability_uploaded_file_agent',
    output_type=UploadedFile,
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class DurabilityUploadedFileAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> UploadedFile:
        result = await _durability_uploaded_file_agent.run(prompt)
        return result.output


async def test_durability_uploaded_file_serialization_preserves_media_type(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_uploaded_file_serialization_preserves_media_type`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityUploadedFileAgentWorkflow],
        plugins=[AgentPlugin(_durability_uploaded_file_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityUploadedFileAgentWorkflow.run,
            args=['Return a file reference'],
            id=DurabilityUploadedFileAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            UploadedFile(file_id='file-abc123', provider_name='openai', _media_type='image/png', _identifier='file-1')
        )


# --- Toolsets at runtime ---


def _runtime_tool_model(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('runtime_tool', {}, tool_call_id='call-1')])


_runtime_tool_agent = Agent(
    FunctionModel(_runtime_tool_model),
    name='runtime_tool_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


async def _opted_out_runtime_tool() -> str:
    return 'tool-result'


# Rejected before any tool runs.
async def _not_opted_out_runtime_tool() -> str:  # pragma: no cover
    return 'other-result'


@workflow.defn
class DurabilityOptedOutRuntimeFunctionToolsetWorkflow:
    @workflow.run
    async def run(self, partially_opted_out: bool) -> str:
        toolset = FunctionToolset(id='runtime')
        toolset.add_function(_opted_out_runtime_tool, name='runtime_tool', metadata={'temporal': False})
        if partially_opted_out:
            toolset.add_function(_not_opted_out_runtime_tool)
        return (await _runtime_tool_agent.run('use the tool', toolsets=[toolset])).output


async def test_durability_runtime_function_toolset_opt_out(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityOptedOutRuntimeFunctionToolsetWorkflow],
        plugins=[AgentPlugin(_runtime_tool_agent)],
    ):
        assert (
            await client.execute_workflow(
                DurabilityOptedOutRuntimeFunctionToolsetWorkflow.run,
                args=[False],
                id=f'{DurabilityOptedOutRuntimeFunctionToolsetWorkflow.__name__}-full',
                task_queue=TASK_QUEUE,
            )
            == 'done'
        )
        with workflow_raises(
            UserError,
            snapshot(
                "FunctionToolset 'runtime' cannot be passed to `run(toolsets=...)` at runtime with Temporal, because "
                'toolsets that execute their own tools or resolve dynamically must be registered for durable '
                'execution when the agent is constructed. Pass them to the agent constructor instead. '
                'Non-executing toolsets like `ExternalToolset` can be passed at runtime. Async tools that '
                "don't need durable wrapping can opt out with metadata={'temporal': False} to be allowed at runtime."
            ),
        ):
            await client.execute_workflow(
                DurabilityOptedOutRuntimeFunctionToolsetWorkflow.run,
                args=[True],
                id=f'{DurabilityOptedOutRuntimeFunctionToolsetWorkflow.__name__}-partial',
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class DurabilityRuntimeFunctionToolsetWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_durable_agent.run(prompt, toolsets=[FunctionToolset()])
        return result.output  # pragma: no cover


async def test_durability_rejects_runtime_executing_toolsets_in_workflow(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_temporal_agent_run_in_workflow_with_executing_toolsets`.

    Executing toolsets can't be added per-run inside a workflow because their activities must
    be registered with the worker before the workflow runs.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityRuntimeFunctionToolsetWorkflow],
        plugins=[AgentPlugin(simple_durable_agent)],
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
                DurabilityRuntimeFunctionToolsetWorkflow.run,
                args=['What is the capital of Mexico?'],
                id=DurabilityRuntimeFunctionToolsetWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )


async def test_durability_allows_runtime_toolsets_outside_workflow(allow_model_requests: None):
    """Outside a workflow the capability is transparent, so per-run executing toolsets are fine."""

    def call_then_answer(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('runtime_tool', {}, tool_call_id='call-1')])

    def runtime_tool() -> str:
        return 'tool-result'

    agent = Agent(
        FunctionModel(call_then_answer),
        name='durability_runtime_outside_workflow',
        capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
    )
    result = await agent.run(
        'Call the runtime tool.', toolsets=[FunctionToolset(tools=[runtime_tool], id='runtime_fn')]
    )
    assert result.output == 'done'


def _durability_request_external_tool(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('external', {'query': 'runtime'}, tool_call_id='call-1')])


_durability_runtime_external_agent = Agent(
    FunctionModel(_durability_request_external_tool),
    name='durability_runtime_external_agent',
    output_type=[str, DeferredToolRequests],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


_durability_runtime_external_toolset = ExternalToolset(
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
class DurabilityRuntimeExternalToolsetWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> DeferredToolRequests | str:
        result = await _durability_runtime_external_agent.run(prompt, toolsets=[_durability_runtime_external_toolset])
        return result.output


async def test_durability_run_in_workflow_with_runtime_external_toolset(allow_model_requests: None, client: Client):
    """Capability-path equivalent of `test_temporal_agent_run_in_workflow_with_runtime_external_toolset`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityRuntimeExternalToolsetWorkflow],
        plugins=[AgentPlugin(_durability_runtime_external_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityRuntimeExternalToolsetWorkflow.run,
            args=['Call the runtime external tool.'],
            id=DurabilityRuntimeExternalToolsetWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == DeferredToolRequests(
            calls=[ToolCallPart('external', {'query': 'runtime'}, tool_call_id='call-1')]
        )


# --- Capability-contributed toolsets ---


def _durability_call_where_am_i(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(str(part.content))])
    return ModelResponse(parts=[ToolCallPart('where_am_i', {}, tool_call_id='call-1')])


def where_am_i() -> str:
    return 'activity' if activity.in_activity() else 'workflow'


_durability_cap_toolset_agent = Agent(
    FunctionModel(_durability_call_where_am_i),
    name='durability_cap_toolset_agent',
    capabilities=[
        Toolset(FunctionToolset([where_am_i], id='cap_tools')),
        TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class DurabilityCapabilityToolsetWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _durability_cap_toolset_agent.run(prompt)
        return result.output


async def test_durability_temporalizes_capability_contributed_toolsets(allow_model_requests: None, client: Client):
    """Toolsets contributed by other capabilities run as Temporal activities.

    Durability capabilities are in the `innermost` ordering tier, so `Agent.__init__` binds
    them only after every other capability's contributed toolsets have been extracted into
    `agent.toolsets`. Without that two-phase binding, the `Toolset(...)` capability's tools
    would be invisible to `for_agent` and run unwrapped (non-deterministically) inside the
    workflow instead of in an activity.
    """
    durability = TemporalDurability.from_agent(_durability_cap_toolset_agent)
    assert durability is not None
    assert 'agent__durability_cap_toolset_agent__toolset__cap_tools__call_tool' in [
        ActivityDefinition.must_from_callable(act).name  # pyright: ignore[reportUnknownMemberType]
        for act in durability.temporal_activities
    ]

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityCapabilityToolsetWorkflow],
        plugins=[AgentPlugin(_durability_cap_toolset_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityCapabilityToolsetWorkflow.run,
            args=['Where does the tool run?'],
            id=DurabilityCapabilityToolsetWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == 'activity'


_continuation_model = ScriptedContinuationModel()

_continuation_agent = Agent(
    _continuation_model,
    name='durability_continuation_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class DurabilityContinuationWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str]:
        return await _continuation_agent.run(prompt)


@workflow.defn
class DurabilityContinuationResumeWorkflow:
    @workflow.run
    async def run(self, messages: list[ModelMessage]) -> AgentRunResult[str]:
        return await _continuation_agent.run(message_history=messages)


@workflow.defn
class DurabilityContinuationUsageLimitWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str]:
        return await _continuation_agent.run(prompt, usage_limits=UsageLimits(total_tokens_limit=20))


async def test_durability_continuation_chain_in_workflow(client: Client):
    """A suspended → complete chain resolves across per-segment activities as one merged response.

    Usage is counted once (a continuation isn't a separate request step), and the workflow
    history shows one scheduled activity for each segment.
    """
    _continuation_model.reset(
        responses=[
            scripted_response(
                texts=['The answer '],
                state='suspended',
                provider_response_id='cont1',
                input_tokens=5,
                output_tokens=2,
            ),
            scripted_response(texts=['is 42.'], provider_response_id='cont2', input_tokens=3, output_tokens=4),
        ]
    )
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationWorkflow],
        plugins=[AgentPlugin(_continuation_agent)],
    ):
        wf = await client.start_workflow(
            DurabilityContinuationWorkflow.run,
            args=['go'],
            id='DurabilityContinuationWorkflow_chain',
            task_queue=TASK_QUEUE,
        )
        result = await wf.result()
        history = await wf.fetch_history()

    assert result.output == 'The answer is 42.'
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'complete'
    assert [part.content for part in response.parts if isinstance(part, TextPart)] == ['The answer ', 'is 42.']
    usage = result.usage
    assert usage.requests == 1
    assert usage.input_tokens == 8
    assert usage.output_tokens == 6
    # Both segments ran in their own durable boundary.
    assert _continuation_model.request_calls == 2
    assert _scheduled_activity_count(history) == 2


class _DelayedContinuationModel(ScriptedContinuationModel):
    def continuation_delay(self, response: ModelResponse) -> float | None:
        return 0.2


_continuation_delay_model = _DelayedContinuationModel()

_continuation_delay_agent = Agent(
    _continuation_delay_model,
    name='durability_continuation_delay_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class DurabilityContinuationDelayWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str]:
        return await _continuation_delay_agent.run(prompt)


async def test_durability_continuation_delay_uses_durable_timer(client: Client):
    """The wait before re-polling a suspended segment burns a durable Temporal timer.

    `TemporalDurability` registers `workflow.sleep` as the agent-graph sleep, so a model's
    `continuation_delay` (forwarded through the per-segment wrapper to the real workflow-side
    model) shows up in workflow history as a timer that survives replays, rather than
    consuming activity wall-clock time.
    """
    _continuation_delay_model.reset(
        responses=[
            scripted_response(
                texts=['The answer '],
                state='suspended',
                provider_response_id='cont1',
                input_tokens=5,
                output_tokens=2,
            ),
            scripted_response(texts=['is 42.'], provider_response_id='cont2', input_tokens=3, output_tokens=4),
        ]
    )
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationDelayWorkflow],
        plugins=[AgentPlugin(_continuation_delay_agent)],
    ):
        wf = await client.start_workflow(
            DurabilityContinuationDelayWorkflow.run,
            args=['go'],
            id='DurabilityContinuationDelayWorkflow_timer',
            task_queue=TASK_QUEUE,
        )
        result = await wf.result()
        history = await wf.fetch_history()

    assert result.output == 'The answer is 42.'
    assert _continuation_delay_model.request_calls == 2
    assert _scheduled_activity_count(history) == 2
    assert any(event.HasField('timer_started_event_attributes') for event in history.events)


async def test_durability_continuation_resume_from_history(client: Client):
    """A `message_history` ending in a suspended response resumes inside the activity.

    The suspended tail crosses the activity boundary as the last request message and seeds
    the continuation loop there, so the run completes the paused turn instead of starting a
    fresh generation.
    """
    _continuation_model.reset(
        responses=[scripted_response(texts=['is 42.'], provider_response_id='cont2', input_tokens=3, output_tokens=4)]
    )
    history_messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='go')]),
        scripted_response(
            texts=['The answer '], state='suspended', provider_response_id='cont1', input_tokens=5, output_tokens=2
        ),
    ]
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationResumeWorkflow],
        plugins=[AgentPlugin(_continuation_agent)],
    ):
        wf = await client.start_workflow(
            DurabilityContinuationResumeWorkflow.run,
            args=[history_messages],
            id='DurabilityContinuationWorkflow_resume',
            task_queue=TASK_QUEUE,
        )
        result = await wf.result()
        history = await wf.fetch_history()

    assert result.output == 'The answer is 42.'
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'complete'
    assert [part.content for part in response.parts if isinstance(part, TextPart)] == ['The answer ', 'is 42.']
    usage = result.usage
    assert usage.requests == 1
    assert usage.input_tokens == 8
    assert usage.output_tokens == 6
    # The continuation request ran inside the boundary — the seed wasn't re-generated.
    assert _continuation_model.request_calls == 1
    assert _scheduled_activity_count(history) == 1


async def test_durability_continuation_error_cancels_job_inside_activity(client: Client):
    """A request failure mid-chain cancels the suspended server-side job inside the activity.

    The cancel-on-error policy runs on the real model inside the durable boundary — the
    workflow side never sees the live suspended response.
    """
    _continuation_model.reset(
        responses=[
            scripted_response(
                texts=['The answer '],
                state='suspended',
                provider_response_id='cont1',
                input_tokens=5,
                output_tokens=2,
            ),
            RuntimeError('provider blew up'),
        ]
    )
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationWorkflow],
        plugins=[AgentPlugin(_continuation_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                DurabilityContinuationWorkflow.run,
                args=['go'],
                id='DurabilityContinuationWorkflow_cancel_on_error',
                task_queue=TASK_QUEUE,
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == 'RuntimeError'
    assert cause.message == 'provider blew up'
    assert _continuation_model.request_calls == 2
    assert len(_continuation_model.cancelled) == 1
    assert _continuation_model.cancelled[0].provider_response_id == 'cont1'


async def test_durability_continuation_usage_limit_checked_inside_activity(client: Client):
    """Token limits are enforced mid-chain inside the activity, cancelling the live job.

    `usage`/`usage_limits` cross the activity boundary on the serialized run context (a
    custom `TemporalRunContext` subclass must keep including them), so a runaway
    continuation fails fast without waiting for the workflow-side commit.
    """
    _continuation_model.reset(
        responses=[
            scripted_response(
                texts=['The answer '],
                state='suspended',
                provider_response_id='cont1',
                input_tokens=5,
                output_tokens=2,
            ),
            scripted_response(
                texts=['keeps going '],
                state='suspended',
                provider_response_id='cont2',
                input_tokens=100,
                output_tokens=50,
            ),
        ]
    )
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationUsageLimitWorkflow],
        plugins=[AgentPlugin(_continuation_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                DurabilityContinuationUsageLimitWorkflow.run,
                args=['go'],
                id='DurabilityContinuationWorkflow_usage_limit',
                task_queue=TASK_QUEUE,
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == UsageLimitExceeded.__name__
    assert 'total_tokens_limit' in cause.message
    assert _continuation_model.request_calls == 2
    # The over-budget merge was still suspended, so the live job was cancelled before raising.
    assert len(_continuation_model.cancelled) == 1
    assert _continuation_model.cancelled[0].provider_response_id == 'cont2'


_continuation_ceiling_model = ScriptedContinuationModel()

_continuation_ceiling_agent = Agent(
    _continuation_ceiling_model,
    name='durability_continuation_ceiling_agent',
    capabilities=[
        TemporalDurability(
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=60),
                # More than one attempt allowed, to prove `UnexpectedModelBehavior` is
                # non-retryable rather than merely running out of attempts.
                retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(milliseconds=10)),
            )
        )
    ],
)


@workflow.defn
class DurabilityContinuationCeilingWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str]:
        return await _continuation_ceiling_agent.run(prompt)


async def test_durability_continuation_ceiling_surfaces_unexpected_model_behavior(client: Client):
    """Exceeding the continuation ceiling fails the workflow without activity retries.

    `UnexpectedModelBehavior` is in the activity retry policy's non-retryable error types:
    re-running the whole chain wouldn't fix a model that never leaves `'suspended'`, it
    would only re-incur its cost. The single-attempt call count proves no retry happened.
    """
    _continuation_ceiling_model.reset(
        responses=[
            scripted_response(
                texts=[f'segment {i} '],
                state='suspended',
                provider_response_id=f'cont{i}',
                input_tokens=1,
                output_tokens=1,
            )
            for i in range(1, 12)
        ]
    )
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationCeilingWorkflow],
        plugins=[AgentPlugin(_continuation_ceiling_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                DurabilityContinuationCeilingWorkflow.run,
                args=['go'],
                id=DurabilityContinuationCeilingWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == UnexpectedModelBehavior.__name__
    assert cause.message == snapshot("Model response 'cont11' was suspended more than the maximum of 10 times")
    # 1 initial + 10 continuation requests, from a single activity attempt (no retries).
    assert _continuation_ceiling_model.request_calls == 11
    # Giving up on a still-suspended job cancels it inside the activity so it doesn't leak.
    assert len(_continuation_ceiling_model.cancelled) == 1


# --- Streaming continuation chains inside the activity ---

_continuation_stream_model = ScriptedContinuationModel()

_continuation_stream_events: list[AgentStreamEvent] = []


async def _continuation_event_stream_handler(
    ctx: RunContext[object],
    stream: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in stream:
        _continuation_stream_events.append(event)


_continuation_stream_agent = Agent(
    _continuation_stream_model,
    name='durability_continuation_stream_agent',
    capabilities=[
        ProcessEventStream(_continuation_event_stream_handler),
        TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class DurabilityContinuationStreamWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> tuple[AgentRunResult[str], list[tuple[str, int]]]:
        result = await _continuation_stream_agent.run(prompt)
        return result, _text_part_indices(_continuation_stream_events)


@workflow.defn
class DurabilityContinuationStreamResumeWorkflow:
    @workflow.run
    async def run(self, messages: list[ModelMessage]) -> tuple[AgentRunResult[str], list[tuple[str, int]]]:
        result = await _continuation_stream_agent.run(message_history=messages)
        return result, _text_part_indices(_continuation_stream_events)


def _text_part_indices(events: list[AgentStreamEvent]) -> list[tuple[str, int]]:
    return [
        (type(event).__name__, event.index) for event in events if isinstance(event, (PartStartEvent, PartDeltaEvent))
    ]


async def test_durability_streaming_continuation_chain_in_workflow(client: Client):
    """A streamed suspended → complete chain is stitched across per-segment activities.

    `ProcessEventStream` receives each captured segment in workflow code, and the
    final response merges both segments' text with usage summed once.
    """
    _continuation_stream_model.reset(
        segments=[
            StreamSegment(
                texts=['The answer '],
                state='suspended',
                provider_response_id='cont1',
                input_tokens=5,
                output_tokens=2,
            ),
            StreamSegment(
                texts=['is 42.'], state='complete', provider_response_id='cont2', input_tokens=3, output_tokens=4
            ),
        ]
    )
    _continuation_stream_events.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationStreamWorkflow],
        plugins=[AgentPlugin(_continuation_stream_agent)],
    ):
        wf = await client.start_workflow(
            DurabilityContinuationStreamWorkflow.run,
            args=['go'],
            id='DurabilityContinuationStreamWorkflow_chain',
            task_queue=TASK_QUEUE,
        )
        result, indices = await wf.result()
        history = await wf.fetch_history()

    assert result.output == 'The answer is 42.'
    usage = result.usage
    assert usage.requests == 1
    assert usage.input_tokens == 8
    assert usage.output_tokens == 6
    assert indices == snapshot(
        [('PartStartEvent', 0), ('PartDeltaEvent', 0), ('PartStartEvent', 1), ('PartDeltaEvent', 1)]
    )
    assert _continuation_stream_model.request_stream_calls == 2
    assert _scheduled_activity_count(history) == 2


async def test_durability_streaming_continuation_resume_from_history(client: Client):
    """A streamed resume passes the suspended history tail to the first activity.

    The suspended tail seeds the workflow-side composite and the final output merges both texts.
    """
    _continuation_stream_model.reset(
        segments=[
            StreamSegment(
                texts=['is 42.'], state='complete', provider_response_id='cont2', input_tokens=3, output_tokens=4
            ),
        ]
    )
    _continuation_stream_events.clear()
    history_messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='go')]),
        scripted_response(
            texts=['The answer '], state='suspended', provider_response_id='cont1', input_tokens=5, output_tokens=2
        ),
    ]
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityContinuationStreamResumeWorkflow],
        plugins=[AgentPlugin(_continuation_stream_agent)],
    ):
        wf = await client.start_workflow(
            DurabilityContinuationStreamResumeWorkflow.run,
            args=[history_messages],
            id='DurabilityContinuationStreamWorkflow_resume',
            task_queue=TASK_QUEUE,
        )
        result, indices = await wf.result()

    assert result.output == 'The answer is 42.'
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'complete'
    assert [part.content for part in response.parts if isinstance(part, TextPart)] == ['The answer ', 'is 42.']
    assert indices == snapshot(
        [
            ('PartStartEvent', 1),
            ('PartDeltaEvent', 1),
        ]
    )
    assert _continuation_stream_model.request_stream_calls == 1


# --- A static toolset's `prepare` function runs only in workflow code ---
#
# The tool-call activity rebuilds the tool from the `ToolDefinition` the workflow prepared (like
# the MCP path does) instead of listing the toolset's tools again, so `prepare` never runs a
# second time against the activity's limited `RunContext`, and the definition the model saw is
# the one the activity enforces. These tests use `UnsandboxedWorkflowRunner` so workflow-side
# and activity-side calls land on the same module state.

_prepare_run_steps: list[int] = []

_prepared_descriptions: list[str | None] = []


async def _prepare_sleepy_tool(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
    """Set a timeout on the first call only, so a second call would change the tool's behavior."""
    _prepare_run_steps.append(ctx.run_step)
    return replace(
        tool_def,
        description=f'prepared {len(_prepare_run_steps)}',
        timeout=0.01 if len(_prepare_run_steps) == 1 else None,
    )


async def _sleepy_tool() -> str:
    await asyncio.sleep(0.5)
    return 'slept'  # pragma: no cover


def _prepare_tool_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    _prepared_descriptions.append(info.function_tools[0].description)
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('sleepy_tool', {})])
    return ModelResponse(parts=[TextPart('done')])


_prepare_agent = Agent(
    FunctionModel(_prepare_tool_model),
    name='durability_prepare_agent',
    toolsets=[
        FunctionToolset[object](
            tools=[Tool(_sleepy_tool, name='sleepy_tool', prepare=_prepare_sleepy_tool)], id='prepare_ts'
        )
    ],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class DurabilityPrepareWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[ModelMessage]:
        return (await _prepare_agent.run(prompt)).all_messages()


async def test_durability_static_tool_prepare_runs_only_in_workflow(client: Client):
    """`prepare` runs once per model step in workflow code, and the activity honours its `tool_def`.

    Only the first `prepare` call sets `timeout=0.01`, so re-preparing inside the activity would
    silently drop the timeout that the tool definition the model saw carried.
    """
    _prepare_run_steps.clear()
    _prepared_descriptions.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityPrepareWorkflow],
        plugins=[AgentPlugin(_prepare_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        messages = await client.execute_workflow(
            DurabilityPrepareWorkflow.run,
            args=['go'],
            id=f'{DurabilityPrepareWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )

    # One call per model step, both in workflow code; none inside the tool-call activity.
    assert _prepare_run_steps == snapshot([1, 2])
    assert _prepared_descriptions == snapshot(['prepared 1', 'prepared 2'])
    # The `timeout=0.01` from the workflow-side call is what the activity enforced.
    retry_prompts = [
        part.content for message in messages for part in message.parts if isinstance(part, RetryPromptPart)
    ]
    assert retry_prompts == snapshot(['Timed out after 0.01 seconds.'])


async def victim_tool() -> str:
    return 'victim'  # pragma: no cover


_removal_toolset = FunctionToolset[object]([victim_tool], id='removal_ts')


def _removal_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # Runs inside the model activity, after the workflow listed this step's tools: dropping the
    # tool now leaves the workflow calling a tool the activity can no longer resolve.
    _removal_toolset.tools.pop('victim_tool')
    return ModelResponse(parts=[ToolCallPart('victim_tool', {})])


_removal_agent = Agent(
    FunctionModel(_removal_model),
    name='durability_removal_agent',
    toolsets=[_removal_toolset],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class DurabilityRemovedToolWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _removal_agent.run(prompt)).output


async def test_durability_removed_tool_still_raises_user_error(client: Client):
    """A tool that's really gone from the toolset still fails with the tool-removal error."""
    try:
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[DurabilityRemovedToolWorkflow],
            plugins=[AgentPlugin(_removal_agent)],
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            with pytest.raises(WorkflowFailureError) as exc_info:
                await client.execute_workflow(
                    DurabilityRemovedToolWorkflow.run,
                    args=['go'],
                    id=f'{DurabilityRemovedToolWorkflow.__name__}-{uuid.uuid4()}',
                    task_queue=TASK_QUEUE,
                )
    finally:
        _removal_toolset.add_function(victim_tool)

    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == UserError.__name__
    assert cause.message == snapshot(
        "Tool 'victim_tool' not found in toolset 'removal_ts'. "
        'Removing or renaming tools during an agent run is not supported with Temporal.'
    )


async def test_durability_call_tool_activity_without_tool_def_re_prepares_tool():
    """A tool-call activity scheduled without a `tool_def` still runs, by preparing the tool itself.

    Unit test: the workflow side always sends the prepared `tool_def` now, so only an activity
    scheduled by a worker predating that field can arrive without one — no workflow run can
    produce this payload, but a rolling upgrade can.
    """
    prepare_run_steps: list[int] = []

    async def prepare_legacy_tool(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition:
        prepare_run_steps.append(ctx.run_step)
        return tool_def

    async def legacy_tool() -> str:
        return 'legacy'

    toolset = FunctionToolset[None](
        tools=[Tool(legacy_tool, name='legacy_tool', prepare=prepare_legacy_tool)], id='legacy_ts'
    )
    durable_toolset = temporalize_function_toolset(
        toolset,
        activity_name_prefix='test__legacy_call_tool_params',
        activity_config=BASE_ACTIVITY_CONFIG,
        tool_activity_config={},
        deps_type=type(None),
    )
    (call_tool_activity,) = durable_toolset.durable_registrations

    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run-123', run_step=3)
    result = await call_tool_activity(
        CallToolParams(
            name='legacy_tool',
            tool_args={},
            serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
            tool_def=None,
        ),
        None,
    )

    assert unwrap_tool_call_result(result) == 'legacy'
    assert prepare_run_steps == [3]

    with pytest.raises(
        UserError,
        match=re.escape(
            "Tool 'missing' not found in toolset 'legacy_ts'. Removing or renaming tools during an agent run"
        ),
    ):
        await call_tool_activity(
            CallToolParams(
                name='missing',
                tool_args={},
                serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
                tool_def=None,
            ),
            None,
        )


async def test_durability_common_call_activity_accepts_legacy_params_without_tool_def():
    """The common operation activity keeps accepting payloads scheduled before `tool_def` existed."""

    async def legacy_tool() -> str:
        return 'legacy'

    toolset = FunctionToolset[None]([legacy_tool], id='common_legacy_ts')
    agent = Agent(
        TestModel(),
        name='common_legacy_params',
        deps_type=type(None),
        toolsets=[toolset],
        capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
    )
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    registrations_by_name = {
        ActivityDefinition.must_from_callable(registration).name: registration  # pyright: ignore[reportUnknownMemberType]
        for registration in durability.temporal_activities
    }
    call_tool_activity = registrations_by_name['agent__common_legacy_params__toolset__common_legacy_ts__call_tool']
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run-123', run_step=3)

    result = await call_tool_activity(
        CallToolParams(
            name='legacy_tool',
            tool_args={},
            serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
            tool_def=None,
        ),
        None,
    )
    assert unwrap_tool_call_result(result) == 'legacy'

    with pytest.raises(
        UserError,
        match=re.escape(
            "Tool 'missing' not found in toolset 'common_legacy_ts'. Removing or renaming tools during an agent run"
        ),
    ):
        await call_tool_activity(
            CallToolParams(
                name='missing',
                tool_args={},
                serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
                tool_def=None,
            ),
            None,
        )


_renamed_tool_names: list[list[str]] = []


async def registered_tool() -> str:
    return 'the registered function ran'


async def _prepare_renamed_tool(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
    """Expose the tool to the model under a name the toolset doesn't hold it under."""
    return replace(tool_def, name='exposed_tool')


def _renaming_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    _renamed_tool_names.append([tool_def.name for tool_def in info.function_tools])
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(str(part.content))])
    return ModelResponse(parts=[ToolCallPart('exposed_tool', {})])


_renaming_agent = Agent(
    FunctionModel(_renaming_model),
    name='durability_renaming_agent',
    toolsets=[
        FunctionToolset[object](
            tools=[Tool(registered_tool, name='registered_tool', prepare=_prepare_renamed_tool)], id='renaming_ts'
        )
    ],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class DurabilityRenamedToolWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _renaming_agent.run(prompt)).output


async def test_durability_prepare_renamed_tool_runs_in_activity(client: Client):
    """A `prepare` function that renames its tool still resolves to that tool inside the activity.

    The activity looks the tool up by the name the toolset holds it under, which the workflow sends
    alongside the prepared `tool_def`; looking it up by the model-visible name would raise the
    tool-removal error for a tool that is still right there. Runs sandboxed, so the workflow's
    `prepare` result reaches the activity over the wire rather than through shared module state.
    """
    _renamed_tool_names.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DurabilityRenamedToolWorkflow],
        plugins=[AgentPlugin(_renaming_agent)],
    ):
        output = await client.execute_workflow(
            DurabilityRenamedToolWorkflow.run,
            args=['go'],
            id=f'{DurabilityRenamedToolWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )

    assert output == snapshot('the registered function ran')
    # The model only ever saw the renamed tool, in both steps.
    assert _renamed_tool_names == snapshot([['exposed_tool'], ['exposed_tool']])
