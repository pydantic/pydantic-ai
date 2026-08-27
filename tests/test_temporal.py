from __future__ import annotations

import asyncio
import inspect
import os
import re
import sys
import uuid
import warnings
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Callable, Generator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from datetime import timedelta
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal, cast
from unittest.mock import Mock, patch

import anyio
import httpx
import httpx2
import pytest
from packaging.version import Version
from pydantic import BaseModel, TypeAdapter
from pydantic_core import PydanticSerializationError

from pydantic_ai import (
    AbstractToolset,
    Agent,
    AgentRunResultEvent,
    AgentStreamEvent,
    BinaryContent,
    BinaryImage,
    CancellationToken,
    CodeExecutionTool,
    DocumentUrl,
    ExternalToolset,
    FilePart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    FunctionToolset,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    MultiModalContent,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RequestUsage,
    RetryPromptPart,
    RunContext,
    RunUsage,
    SystemPromptPart,
    TextContent,
    TextPart,
    TextPartDelta,
    Tool,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturn,
    ToolReturnPart,
    ToolsetTool,
    UserContent,
    UserPromptPart,
    WebSearchTool,
    WebSearchUserLocation,
)
from pydantic_ai._run_context import AnchoredEvidence
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities import (
    MCP,
    Capability,
    DynamicCapability,
    Instrumentation,
    NativeTool,
    ProcessEventStream,
    ProcessHistory,
    ResolveModelId,
    Toolset,
    WrapperCapability,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    RunCancelled,
    SkipModelRequest,
    ToolFailed,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import UploadedFile
from pydantic_ai.models import (
    CompletedStreamedResponse,
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    ModelResolutionContext,
    infer_model,
    infer_model_profile,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from pydantic_ai.profiles import DEFAULT_PROFILE, ModelProfile
from pydantic_ai.realtime import (
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeSession,
)
from pydantic_ai.realtime.codec import RealtimeConnection
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.external import TOOL_SCHEMA_VALIDATOR
from pydantic_ai.usage import UsageLimits
from pydantic_graph import GraphBuilder, StepContext
from pydantic_graph.join import reduce_list_append

from ._inline_snapshot import snapshot
from .continuation_utils import ScriptedContinuationModel, StreamSegment, scripted_response

try:
    import temporalio.api.common.v1
    from temporalio import activity, workflow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.client import Client, WorkflowFailureError, WorkflowHistory
    from temporalio.common import RetryPolicy
    from temporalio.contrib.opentelemetry import TracingInterceptor
    from temporalio.contrib.pydantic import PydanticPayloadConverter, pydantic_data_converter
    from temporalio.converter import (
        DataConverter,
        DefaultPayloadConverter,
        ExternalStorage,
        PayloadCodec,
        StorageDriver,
    )
    from temporalio.exceptions import ApplicationError, CancelledError as TemporalCancelledError
    from temporalio.testing import ActivityEnvironment, WorkflowEnvironment
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner
    from temporalio.workflow import ActivityCancellationType, ActivityConfig

    from pydantic_ai.durable_exec._toolset import (
        CallToolResult,
        unwrap_tool_call_result,
        wrap_tool_call_result,
    )
    from pydantic_ai.durable_exec._utils import StreamedActivityResult
    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        LogfirePlugin,
        PydanticAIPayloadConverter,
        PydanticAIPlugin,
        PydanticAIWorkflow,
        TemporalAgent,  # pyright: ignore[reportDeprecated]
        TemporalDurability,
        _logfire as temporal_logfire,  # pyright: ignore[reportPrivateUsage]
        _payload_converter as temporal_payload_converter,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.durable_exec.temporal._activity_execution import (
        execute_activity as execute_temporal_activity,
    )
    from pydantic_ai.durable_exec.temporal._durability import (
        _CancelParams,  # pyright: ignore[reportPrivateUsage]
        _EventStreamHandlerParams,  # pyright: ignore[reportPrivateUsage]
        _RequestParams,  # pyright: ignore[reportPrivateUsage]
        _StreamedActivityPayload,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.durable_exec.temporal._dynamic_toolset import temporalize_dynamic_toolset
    from pydantic_ai.durable_exec.temporal._function_toolset import (
        TemporalFunctionToolset,
        temporalize_function_toolset,
    )
    from pydantic_ai.durable_exec.temporal._mcp_toolset import TemporalMCPToolset
    from pydantic_ai.durable_exec.temporal._model import (
        TemporalModel,
        _CancelParams as _ModelCancelParams,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.durable_exec.temporal._run_context import TemporalRunContext, deserialize_run_context
    from pydantic_ai.durable_exec.temporal._toolset import (
        CallToolParams,
        GetToolsParams,
        TemporalWrapperToolset,
        heartbeating,
        resolve_tool_activity_config,
        toolset_temporal_activities,
    )

    from .temporal_sandbox_workflow import PydanticAIPluginSandboxWorkflow
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)


if sys.version_info >= (3, 14):
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

try:
    import logfire
    from logfire import Logfire
    from logfire._internal.config import LogfireConfig
    from logfire._internal.tracer import _ProxyTracer  # pyright: ignore[reportPrivateUsage]
    from logfire.testing import CaptureLogfire
    from opentelemetry.trace import ProxyTracer
except ImportError:  # pragma: lax no cover
    pytest.skip('logfire not installed', allow_module_level=True)

try:
    from fastmcp.client.transports import StdioTransport

    from pydantic_ai.mcp import MCPToolset
except ImportError:  # pragma: lax no cover
    pytest.skip('mcp not installed', allow_module_level=True)

try:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
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
    from mcp.client.session import ClientSession
    from mcp.types import ClientRequest

    from ._inline_snapshot import snapshot

    # Loads `vcr`, which Temporal doesn't like without passing through the import
    from .conftest import IsDatetime, IsInt, IsList, IsStr, message, try_import

with try_import() as anthropic_imports_successful:
    import anthropic

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

# `TemporalAgent` is deprecated in favor of `capabilities=[TemporalDurability(...)]`.
# These tests exercise the wrapper-agent path on purpose; suppress the warning here
# rather than globally in `pyproject.toml`. The `pytestmark` entry below covers warnings
# emitted *inside* test functions; the `filterwarnings` call below covers warnings emitted
# at module import time (e.g. module-level construction of `TemporalAgent`).
warnings.filterwarnings('ignore', message='`TemporalAgent` is deprecated', category=PydanticAIDeprecationWarning)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.xdist_group(name='temporal'),
    pytest.mark.filterwarnings(
        'ignore:`TemporalAgent` is deprecated:pydantic_ai._warnings.PydanticAIDeprecationWarning'
    ),
]


@pytest.fixture
def blockbuster_enabled() -> bool:
    """Disable detection for Temporal's synchronous worker and integration setup.

    It performs module/config introspection above Pydantic AI plugin frames; BlockBuster changes
    its error handling and makes these tests unusably slow. Rebenchmark after
    https://github.com/cbornet/blockbuster/pull/61 is released, but retain this opt-out until the
    synchronous-introspection false positives are isolated too.
    """
    return False


# We need to use a custom cached HTTP client here as the default one created for OpenAIProvider will be closed automatically
# at the end of each test, but we need this one to live longer.
http_client = httpx2.AsyncClient()


# Scoped to `session` rather than `module`: the `http_client` and the module-level agents that
# capture it are constructed at import time, so they must outlive a single module entry. This is a
# sync fixture so it doesn't force AnyIO to reuse a session-level event loop for all Temporal async
# fixtures; the `temporal_env` teardown can make that loop unusable for later tests.
@pytest.fixture(autouse=True, scope='session')
def close_cached_httpx_client() -> Iterator[None]:
    try:
        yield
    finally:
        asyncio.run(http_client.aclose())


# `LogfirePlugin` calls `logfire.instrument_pydantic_ai()`, so we need to make sure this doesn't bleed into other tests.
@pytest.fixture(autouse=True, scope='module')
def uninstrument_pydantic_ai() -> Iterator[None]:
    try:
        yield
    finally:
        Agent.instrument_all(False)


@contextmanager
def workflow_raises(exc_type: type[Exception], exc_message: str) -> Generator[None]:
    """Helper for asserting that a Temporal workflow fails with the expected error."""
    with pytest.raises(WorkflowFailureError) as exc_info:
        yield
    assert isinstance(exc_info.value.__cause__, ApplicationError)
    assert exc_info.value.__cause__.type == exc_type.__name__
    assert exc_info.value.__cause__.message == exc_message


TEMPORAL_PORT = 7243
TASK_QUEUE = 'pydantic-ai-agent-task-queue'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)


def _kill_leaked_temporal_server(port: int) -> None:
    """Kill any `temporal-sdk-python-*` dev server still bound to `port`.

    A previous test run that crashed mid-fixture leaves the embedded Temporal
    dev server listening on `port`, which makes the next run fail to bind. The
    leak persists across pytest invocations, so detect-and-kill at fixture entry
    keeps local iterations smooth without requiring a manual `kill` between runs.
    Best-effort: failures here don't propagate, the fixture's own bind attempt
    will surface a real port conflict downstream.
    """
    import signal
    import subprocess

    try:
        result = subprocess.run(
            ['ss', '-tlnpH', f'sport = :{port}'],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):  # pragma: lax no cover
        # No `ss` on this platform, or it was unresponsive.
        return

    # The body fires only on a real leak, so it's covered on some runs and not on others.
    for line in result.stdout.splitlines():  # pragma: lax no cover
        if 'temporal-sdk-py' not in line:
            continue
        match = re.search(r'pid=(\d+)', line)
        if not match:
            continue
        pid = int(match.group(1))
        try:
            os.kill(pid, signal.SIGTERM)
        except (PermissionError, ProcessLookupError):
            pass


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    _kill_leaked_temporal_server(TEMPORAL_PORT)
    # `start_local` downloads the dev-server binary to the system temp dir by default, which is empty on
    # every CI run, so a CDN hiccup used to fail the entire suite at setup (#5399). Download to a stable
    # per-user cache dir instead so CI can restore it via `actions/cache` and local runs reuse it across
    # reboots. Resolved here rather than at module level: the workflow sandbox re-imports this module and
    # restricts `Path.home()` access.
    download_dest_dir = Path.home() / '.cache' / 'temporal-dev-server'
    download_dest_dir.mkdir(parents=True, exist_ok=True)
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        ui=True,
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
        download_dest_dir=str(download_dest_dir),
    ) as env:
        yield env


@pytest.fixture
async def client(temporal_env: WorkflowEnvironment) -> Client:
    return await Client.connect(
        f'localhost:{TEMPORAL_PORT}',
        plugins=[PydanticAIPlugin()],
    )


@pytest.fixture
async def client_with_logfire(temporal_env: WorkflowEnvironment) -> Client:
    return await Client.connect(
        f'localhost:{TEMPORAL_PORT}',
        plugins=[PydanticAIPlugin(), LogfirePlugin()],
    )


# Can't use the `openai_api_key` fixture here because the workflow needs to be defined at the top level of the file.
model = OpenAIChatModel(
    'gpt-4o',
    provider=OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'),
        http_client=http_client,
    ),
)

simple_agent = Agent(model, name='simple_agent')

# This needs to be done before the `TemporalAgent` is bound to the workflow.
simple_temporal_agent = TemporalAgent(simple_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class SimpleAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt)
        return result.output


async def test_simple_agent_run_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflow],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        output = await client.execute_workflow(
            SimpleAgentWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=SimpleAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')


_cancellation_activity_started: asyncio.Event | None = None
_cancellation_activity_cancel_absorbed = False


async def _cancellation_stream_model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    global _cancellation_activity_cancel_absorbed

    assert _cancellation_activity_started is not None
    _cancellation_activity_started.set()
    try:
        while True:
            activity.heartbeat()
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        _cancellation_activity_cancel_absorbed = True
        yield 'completed despite activity cancellation'


async def _cancellation_event_stream_handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
    try:
        async for _ in stream:
            pass
    except asyncio.CancelledError:
        pass


_cancellation_agent = Agent(
    FunctionModel(stream_function=_cancellation_stream_model),
    name='cancellation_backstop_agent',
    deps_type=type(None),
    capabilities=[
        TemporalDurability(
            event_stream_handler=_cancellation_event_stream_handler,
            model_activity_config=ActivityConfig(
                cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
                heartbeat_timeout=timedelta(seconds=1),
            ),
        )
    ],
)


@workflow.defn
class CancellationBackstopWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _cancellation_agent.run(prompt)).output


@activity.defn
async def _slow_cancellable_activity() -> str:
    await asyncio.sleep(1)
    return 'completed slowly'


@workflow.defn
class AnyioScopeActivityCancellationWorkflow:
    @workflow.run
    async def run(self) -> str:
        async def run_activity() -> None:
            await execute_temporal_activity(
                _slow_cancellable_activity,
                args=[],
                start_to_close_timeout=timedelta(seconds=5),
                retry_policy=RetryPolicy(maximum_attempts=1),
                cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            )

        async def run_in_task_group() -> None:
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(run_activity)

        try:
            await asyncio.wait_for(run_in_task_group(), timeout=0.1)
        except asyncio.TimeoutError:
            return 'timed out cleanly'
        return 'completed'  # pragma: no cover


async def test_anyio_scope_cancel_of_activity_await_does_not_wedge(client: Client) -> None:
    """Exercise the precise anyio/Temporal interaction that cannot be timed reliably through the agent API.

    Agent-level activity awaits use the same executor, and the test below covers the public path.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AnyioScopeActivityCancellationWorkflow],
        activities=[_slow_cancellable_activity],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            AnyioScopeActivityCancellationWorkflow.run,
            id=f'{AnyioScopeActivityCancellationWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        assert await handle.result() == 'timed out cleanly'
        history = await handle.fetch_history()

    assert not [event for event in history.events if 'WORKFLOW_TASK_FAILED' in str(event.event_type)]


@workflow.defn
class WaitForNonStreamingAgentTimeoutWorkflow:
    @workflow.run
    async def run(self) -> str:
        try:
            result = await asyncio.wait_for(_wait_for_nonstreaming_agent.run('say hi'), timeout=0.5)
        except asyncio.TimeoutError:
            return 'clean-timeout'
        return f'unexpected-success:{result.output}'  # pragma: no cover


async def _slow_nonstreaming_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    await asyncio.sleep(10)
    return ModelResponse(parts=[TextPart('done')])  # pragma: no cover


_wait_for_nonstreaming_agent = Agent(
    FunctionModel(_slow_nonstreaming_model, model_name='slow-model'),
    name='wait_for_nonstreaming_agent',
    deps_type=type(None),
    capabilities=[TemporalDurability()],
)


async def test_wait_for_nonstreaming_agent_timeout_does_not_livelock(client: Client) -> None:
    """The exact MRE shape from #6883 (trigger A): a non-streaming model request as an activity,
    the workflow body bounding `agent.run()` with `asyncio.wait_for`. Must end in a clean
    `TimeoutError`, not a deadlock-detector livelock."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WaitForNonStreamingAgentTimeoutWorkflow],
        plugins=[AgentPlugin(_wait_for_nonstreaming_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        result = await client.execute_workflow(
            WaitForNonStreamingAgentTimeoutWorkflow.run,
            id=f'{WaitForNonStreamingAgentTimeoutWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )

    assert result == 'clean-timeout'


async def _wait_for_timeout_stream_model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    while True:
        activity.heartbeat()
        await asyncio.sleep(0.01)
        yield ''


async def _consume_wait_for_timeout_events(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in stream:
        pass


_wait_for_timeout_agent = Agent(
    FunctionModel(stream_function=_wait_for_timeout_stream_model),
    name='wait_for_timeout_agent',
    deps_type=type(None),
    capabilities=[
        TemporalDurability(
            event_stream_handler=_consume_wait_for_timeout_events,
            model_activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=10),
                heartbeat_timeout=timedelta(seconds=1),
                retry_policy=RetryPolicy(maximum_attempts=1),
                cancellation_type=ActivityCancellationType.WAIT_CANCELLATION_COMPLETED,
            ),
        )
    ],
)


@workflow.defn
class WaitForAgentTimeoutWorkflow:
    @workflow.run
    async def run(self) -> str:
        try:
            await asyncio.wait_for(_wait_for_timeout_agent.run('go slowly'), timeout=0.5)
        except asyncio.TimeoutError:
            return 'timed out cleanly'
        return 'completed'  # pragma: no cover


async def test_wait_for_agent_timeout_in_workflow_does_not_livelock(client: Client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WaitForAgentTimeoutWorkflow],
        plugins=[AgentPlugin(_wait_for_timeout_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        result = await client.execute_workflow(
            WaitForAgentTimeoutWorkflow.run,
            id=f'{WaitForAgentTimeoutWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )

    assert result == 'timed out cleanly'


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason='the cancellation backstop needs `Task.cancelling()` (Python 3.11+); on 3.10 the absorbed cancel legitimately completes',
)
async def test_temporal_cancellation_backstop_survives_absorbed_activity_cancel(client: Client) -> None:
    """A cancelled workflow cannot complete after its streaming model activity absorbs cancellation."""
    global _cancellation_activity_cancel_absorbed, _cancellation_activity_started

    _cancellation_activity_started = asyncio.Event()
    _cancellation_activity_cancel_absorbed = False
    workflow_id = f'{CancellationBackstopWorkflow.__name__}-{uuid.uuid4()}'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CancellationBackstopWorkflow],
        plugins=[AgentPlugin(_cancellation_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            CancellationBackstopWorkflow.run,
            args=['cancel me'],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        await _cancellation_activity_started.wait()
        await handle.cancel()

        with pytest.raises(WorkflowFailureError) as exc_info:
            await handle.result()
        assert isinstance(exc_info.value.__cause__, TemporalCancelledError)
        assert _cancellation_activity_cancel_absorbed

        history = await handle.fetch_history()

    await Replayer(
        workflows=[CancellationBackstopWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(history)


async def _migration_event_stream_handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in stream:
        pass


async def _migration_tool() -> str:
    return 'tool result'


_migration_agent_name = 'temporal_agent_migration'
# A tool call makes the recorded history include graph-level `__event_stream_handler`
# activities and a tool-call activity, so replay verifies the workflow-side event
# dispatch sequence — not just the model activities.
_legacy_migration_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    Agent(
        TestModel(custom_output_text='migrated'),
        name=_migration_agent_name,
        deps_type=type(None),
        tools=[_migration_tool],
    ),
    activity_config=BASE_ACTIVITY_CONFIG,
    event_stream_handler=_migration_event_stream_handler,
)
_capability_migration_agent = Agent(
    TestModel(custom_output_text='migrated'),
    name=_migration_agent_name,
    deps_type=type(None),
    tools=[_migration_tool],
    capabilities=[
        TemporalDurability(
            activity_config=BASE_ACTIVITY_CONFIG,
            event_stream_handler=_migration_event_stream_handler,
        )
    ],
)


async def test_temporal_agent_rejects_cancellation_token() -> None:
    """The wrapper agent rejects `cancellation_token` up front: a token is same-process state
    that cannot cross the durable execution boundary."""
    with pytest.raises(UserError, match='cannot cross the durable execution boundary'):
        await _legacy_migration_agent.run('hello', cancellation_token=CancellationToken())


_migration_agent: AbstractAgent[None, str] = _legacy_migration_agent


@workflow.defn
class TemporalAgentMigrationWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _migration_agent.run(prompt)).output


async def test_temporal_agent_history_replays_after_migrating_to_durability(client: Client) -> None:
    """A recorded wrapper-agent workflow must replay with the capability implementation.

    This is an engine-level replay test rather than a provider VCR test: the compatibility
    contract is the Temporal activity payload and result schema, independent of the provider.
    """
    global _migration_agent

    _migration_agent = _legacy_migration_agent
    workflow_id = f'{TemporalAgentMigrationWorkflow.__name__}-{uuid.uuid4()}'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[TemporalAgentMigrationWorkflow],
        activities=_legacy_migration_agent.temporal_activities,
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        output = await client.execute_workflow(
            TemporalAgentMigrationWorkflow.run,
            args=['hello'],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    assert output == 'migrated'

    _migration_agent = _capability_migration_agent
    try:
        await Replayer(
            workflows=[TemporalAgentMigrationWorkflow],
            workflow_runner=UnsandboxedWorkflowRunner(),
            data_converter=pydantic_data_converter,
        ).replay_workflow(history)
    finally:
        _migration_agent = _legacy_migration_agent


def test_temporal_agent_construction_warns_deprecated() -> None:
    """The `TemporalAgent` deprecation fires at runtime; the module-level filters only suppress it."""
    with pytest.warns(PydanticAIDeprecationWarning, match='`TemporalAgent` is deprecated'):
        TemporalAgent(Agent(TestModel(), name='temporal_agent_deprecation_probe'))  # pyright: ignore[reportDeprecated]


async def test_temporal_durability_accepts_legacy_cancel_activity_payload() -> None:
    """Temporal decodes old cancel payloads and resolves registered and inferred models."""
    response = ModelResponse(parts=[TextPart(content='cancel')], model_name='test')
    params = TypeAdapter(_CancelParams).validate_python({'response': response, 'model_id': None})
    assert params == _CancelParams(response=response)
    assert params.serialized_run_context is None

    cancelled: list[tuple[str, ModelResponse]] = []

    class RecordingModel(TestModel):
        def __init__(self, name: str):
            super().__init__()
            self.name = name

        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append((self.name, response))

    registered_model = RecordingModel('registered')
    inferred_model = RecordingModel('inferred')
    agent = Agent(
        TestModel(),
        name='legacy_cancel_payload',
        capabilities=[TemporalDurability(models={'registered': registered_model})],
    )
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    signature = inspect.signature(durability.cancel_suspended_response_activity)
    assert signature.parameters['deps'].default is None

    await durability.cancel_suspended_response_activity(_CancelParams(response, model_id='registered'))
    with patch('pydantic_ai.durable_exec.temporal._durability.infer_model', return_value=inferred_model):
        await durability.cancel_suspended_response_activity(_CancelParams(response, model_id='unregistered'))

    assert cancelled == [('registered', response), ('inferred', response)]


class Deps(BaseModel):
    country: str


async def event_stream_handler(
    ctx: RunContext[Deps],
    stream: AsyncIterable[AgentStreamEvent],
):
    logfire.info(f'{ctx.run_step=}')
    async for event in stream:
        logfire.info('event', event=event)


async def get_country(ctx: RunContext[Deps]) -> str:
    return ctx.deps.country


class WeatherArgs(BaseModel):
    city: str


def get_weather(args: WeatherArgs) -> str:
    if args.city == 'Mexico City':
        return 'sunny'
    else:
        return 'unknown'  # pragma: no cover


@dataclass
class Answer:
    label: str
    answer: str


@dataclass
class Response:
    answers: list[Answer]


complex_agent = Agent(
    model,
    deps_type=Deps,
    output_type=Response,
    toolsets=[
        FunctionToolset[Deps](tools=[get_country], id='country'),
        MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='mcp', init_timeout=20),
        ExternalToolset(tool_defs=[ToolDefinition(name='external')], id='external'),
    ],
    tools=[get_weather],
    name='complex_agent',
)

# This needs to be done before the `TemporalAgent` is bound to the workflow.
complex_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    complex_agent,
    event_stream_handler=event_stream_handler,
    activity_config=BASE_ACTIVITY_CONFIG,
    model_activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=90)),
    toolset_activity_config={
        'country': ActivityConfig(start_to_close_timeout=timedelta(seconds=120)),
    },
    tool_activity_config={
        'country': {
            'get_country': False,
        },
        'mcp': {
            'get_product_name': ActivityConfig(start_to_close_timeout=timedelta(seconds=150)),
        },
        '<agent>': {
            'get_weather': ActivityConfig(start_to_close_timeout=timedelta(seconds=180)),
        },
    },
)


@workflow.defn
class ComplexAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, deps: Deps) -> Response:
        result = await complex_temporal_agent.run(prompt, deps=deps)
        return result.output


@dataclass
class BasicSpan:
    content: str
    children: list[BasicSpan] = field(default_factory=list['BasicSpan'])
    parent_id: int | None = field(repr=False, compare=False, default=None)


async def test_complex_agent_run_in_workflow(
    allow_model_requests: None, client_with_logfire: Client, capfire: CaptureLogfire
):
    async with Worker(
        client_with_logfire,
        task_queue=TASK_QUEUE,
        workflows=[ComplexAgentWorkflow],
        plugins=[AgentPlugin(complex_temporal_agent)],
    ):
        output = await client_with_logfire.execute_workflow(
            ComplexAgentWorkflow.run,
            args=[
                'Tell me: the capital of the country; the weather there; the product name',
                Deps(country='Mexico'),
            ],
            id=ComplexAgentWorkflow.__name__,
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
            content='StartWorkflow:ComplexAgentWorkflow',
            children=[
                BasicSpan(content='RunWorkflow:ComplexAgentWorkflow'),
                BasicSpan(
                    content='complex_agent run',
                    children=[
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__mcp_server__mcp__get_tools',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__mcp_server__mcp__get_tools',
                                    children=[BasicSpan(content='tools/list')],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='chat gpt-4o',
                            children=[
                                BasicSpan(
                                    content='StartActivity:agent__complex_agent__model_request_stream',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__complex_agent__model_request_stream',
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
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
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
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=1'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "get_product_name", "args": "{}", "tool_call_id": null, "tool_kind": null, "id": null, "provider_name": null, "provider_details": null, "part_kind": "tool-call"}, "args_valid": true, "event_kind": "function_tool_call"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(content='running tool: get_country'),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
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
                                    content='StartActivity:agent__complex_agent__mcp_server__mcp__call_tool',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__complex_agent__mcp_server__mcp__call_tool',
                                            children=[BasicSpan(content='tools/call get_product_name')],
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
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
                                    content='StartActivity:agent__complex_agent__model_request_stream',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__complex_agent__model_request_stream',
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
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
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
                                    content='StartActivity:agent__complex_agent__toolset__<agent>__call_tool',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__complex_agent__toolset__<agent>__call_tool'
                                        )
                                    ],
                                )
                            ],
                        ),
                        BasicSpan(
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
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
                                    content='StartActivity:agent__complex_agent__model_request_stream',
                                    children=[
                                        BasicSpan(
                                            content='RunActivity:agent__complex_agent__model_request_stream',
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
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
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
                            content='StartActivity:agent__complex_agent__event_stream_handler',
                            children=[
                                BasicSpan(
                                    content='RunActivity:agent__complex_agent__event_stream_handler',
                                    children=[
                                        BasicSpan(content='ctx.run_step=3'),
                                        BasicSpan(
                                            content='{"part": {"tool_name": "final_result", "content": "Final result processed.", "tool_call_id": null, "tool_kind": null, "metadata": null, "timestamp": null, "outcome": "success", "part_kind": "tool-return"}, "event_kind": "output_tool_result"}'
                                        ),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
                BasicSpan(content='CompleteWorkflow:ComplexAgentWorkflow'),
            ],
        )
    )


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


async def test_complex_agent_run(allow_model_requests: None):
    events: list[AgentStreamEvent] = []

    async def event_stream_handler(
        ctx: RunContext[Deps],
        stream: AsyncIterable[AgentStreamEvent],
    ):
        async for event in stream:
            events.append(event)

    with complex_temporal_agent.override(deps=Deps(country='Mexico')):
        result = await complex_temporal_agent.run(
            'Tell me: the capital of the country; the weather there; the product name',
            deps=Deps(country='The Netherlands'),
            event_stream_handler=event_stream_handler,
        )
    assert result.output == snapshot(
        Response(
            answers=[
                Answer(label='Capital', answer='The capital of Mexico is Mexico City.'),
                Answer(label='Weather', answer='The weather in Mexico City is currently sunny.'),
                Answer(label='Product Name', answer='The product name is Pydantic AI.'),
            ]
        )
    )
    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='get_country', args='', tool_call_id='call_q2UyBRP7eXNTzAoR8lEhjc9Z'),
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='{}', tool_call_id='call_q2UyBRP7eXNTzAoR8lEhjc9Z')
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='get_country', args='{}', tool_call_id='call_q2UyBRP7eXNTzAoR8lEhjc9Z'),
                next_part_kind='tool-call',
            ),
            PartStartEvent(
                index=1,
                part=ToolCallPart(tool_name='get_product_name', args='', tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5'),
                previous_part_kind='tool-call',
            ),
            PartDeltaEvent(
                index=1, delta=ToolCallPartDelta(args_delta='{}', tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5')
            ),
            PartEndEvent(
                index=1,
                part=ToolCallPart(
                    tool_name='get_product_name', args='{}', tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5'
                ),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(tool_name='get_country', args='{}', tool_call_id='call_q2UyBRP7eXNTzAoR8lEhjc9Z'),
                args_valid=True,
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='get_product_name', args='{}', tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5'
                ),
                args_valid=True,
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='get_country',
                    content='Mexico',
                    tool_call_id='call_q2UyBRP7eXNTzAoR8lEhjc9Z',
                    timestamp=IsDatetime(),
                )
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='get_product_name',
                    content='Pydantic AI',
                    tool_call_id='call_b51ijcpFkDiTQG1bQzsrmtW5',
                    timestamp=IsDatetime(),
                )
            ),
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='get_weather', args='', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv'),
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='{"', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='city', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='Mexico', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' City', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='"}', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv')
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(
                    tool_name='get_weather', args='{"city":"Mexico City"}', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv'
                ),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='get_weather', args='{"city":"Mexico City"}', tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv'
                ),
                args_valid=True,
            ),
            FunctionToolResultEvent(
                part=ToolReturnPart(
                    tool_name='get_weather',
                    content='sunny',
                    tool_call_id='call_LwxJUB9KppVyogRRLQsamRJv',
                    timestamp=IsDatetime(),
                )
            ),
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='final_result', args='', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn'),
            ),
            FinalResultEvent(tool_name='final_result', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn'),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='{"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='answers', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":[', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='{"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='label', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='Capital', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='","', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='answer', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='The', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' capital', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' of', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' Mexico', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' is', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' Mexico', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' City', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='."', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='},{"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='label', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='Weather', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='","', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='answer', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='The', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' weather', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' in', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' Mexico', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' City', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' is', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' currently', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' sunny', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='."', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='},{"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='label', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='Product', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' Name', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='","', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='answer', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='":"', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='The', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' product', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' name', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' is', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' P', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='yd', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='antic', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=' AI', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='."', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta='}', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartDeltaEvent(
                index=0, delta=ToolCallPartDelta(args_delta=']}', tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn')
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(
                    tool_name='final_result',
                    args='{"answers":[{"label":"Capital","answer":"The capital of Mexico is Mexico City."},{"label":"Weather","answer":"The weather in Mexico City is currently sunny."},{"label":"Product Name","answer":"The product name is Pydantic AI."}]}',
                    tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn',
                ),
            ),
            OutputToolCallEvent(
                part=ToolCallPart(
                    tool_name='final_result',
                    args='{"answers":[{"label":"Capital","answer":"The capital of Mexico is Mexico City."},{"label":"Weather","answer":"The weather in Mexico City is currently sunny."},{"label":"Product Name","answer":"The product name is Pydantic AI."}]}',
                    tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn',
                ),
                args_valid=True,
            ),
            OutputToolResultEvent(
                part=ToolReturnPart(
                    tool_name='final_result',
                    content='Final result processed.',
                    tool_call_id='call_CCGIWaMeYWmxOQ91orkmTvzn',
                    timestamp=IsDatetime(),
                )
            ),
        ]
    )


async def test_multiple_agents(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflow, ComplexAgentWorkflow],
        plugins=[AgentPlugin(simple_temporal_agent), AgentPlugin(complex_temporal_agent)],
    ):
        output = await client.execute_workflow(
            SimpleAgentWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=SimpleAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')

        output = await client.execute_workflow(
            ComplexAgentWorkflow.run,
            args=[
                'Tell me: the capital of the country; the weather there; the product name',
                Deps(country='Mexico'),
            ],
            id=ComplexAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            Response(
                answers=[
                    Answer(label='Capital of the Country', answer='Mexico City'),
                    Answer(label='Weather in Mexico City', answer='Sunny'),
                    Answer(label='Product Name', answer='Pydantic AI'),
                ]
            )
        )


async def test_agent_name_collision(allow_model_requests: None, client: Client):
    with pytest.raises(ValueError, match='More than one activity named agent__simple_agent__event_stream_handler'):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[SimpleAgentWorkflow],
            plugins=[AgentPlugin(simple_temporal_agent), AgentPlugin(simple_temporal_agent)],
        ):
            pass


async def test_agent_without_name():
    with pytest.raises(
        UserError,
        match=re.escape(
            "An agent needs to have a unique `name` in order to be used with Temporal. The name will be used to identify the agent's activities within the workflow."
        ),
    ):
        TemporalAgent(Agent())  # pyright: ignore[reportDeprecated]


async def test_agent_without_model():
    with pytest.raises(
        UserError,
        match=re.escape(
            "The wrapped agent's `model` or the TemporalAgent's `models` parameter must provide at least one Model instance to be used with Temporal. Models cannot be set at agent run time."
        ),
    ):
        TemporalAgent(Agent(name='test_agent'))  # pyright: ignore[reportDeprecated]


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


# --- DynamicToolset / @agent.toolset tests ---


@dataclass
class DynamicToolsetDeps:
    user_name: str


dynamic_toolset_agent = Agent(TestModel(), name='dynamic_toolset_agent', deps_type=DynamicToolsetDeps)


@dynamic_toolset_agent.toolset(id='my_dynamic_tools')
def my_dynamic_toolset(ctx: RunContext[DynamicToolsetDeps]) -> FunctionToolset[DynamicToolsetDeps]:
    toolset = FunctionToolset[DynamicToolsetDeps](id='dynamic_weather')

    @toolset.tool_plain
    def get_dynamic_weather(location: str) -> str:
        """Get the weather for a location."""
        user = ctx.deps.user_name
        return f'Weather in {location} for {user}: sunny.'

    return toolset


dynamic_toolset_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    dynamic_toolset_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
)


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


# Regression test for the workflow-sandbox passthrough list (`_workflow_runner` in
# `durable_exec/temporal/__init__.py`). A `gateway/` model named by string is constructed lazily via
# `infer_model` *inside* the workflow, so the provider's SDK is imported and its client built under
# the `SandboxedWorkflowRunner`. Provider SDKs touch the filesystem/env at construction time, which
# the sandbox forbids unless the SDK module is passed through. Every other test builds its model at
# module scope (outside the sandbox), so this seam was previously uncovered. Construction-only (no
# model request) keeps it deterministic.
@workflow.defn
class ConstructModelInWorkflow:
    @workflow.run
    async def run(self, model_name: str) -> str:
        # We assert only that construction succeeds — no request is made.
        return type(infer_model(model_name)).__name__


@pytest.mark.parametrize(
    ('model_name', 'expected_model_class'),
    [
        # Only `gateway/` providers exercise the sandbox: they import their SDK lazily inside
        # `gateway_provider()`, so the import and client construction run *inside* the workflow. Direct
        # providers (e.g. `anthropic:`) import their SDK at module level, which rides Temporal's
        # transitive passthrough of `pydantic_ai` and never trips — so they give no regression coverage.
        #
        # The reported regression: `gateway/anthropic:` in-workflow tripped the `anthropic` SDK's
        # `Path.home()` access.
        pytest.param('gateway/anthropic:claude-sonnet-4-6', 'AnthropicModel', id='gateway-anthropic'),
        # Canary: OpenAI needs no passthrough today; turns red here (not in a user's workflow) if a
        # future SDK release makes a restricted call (e.g. reads `~/...`) during construction.
        pytest.param('gateway/openai-chat:gpt-5', 'OpenAIChatModel', id='gateway-openai'),
        # Positive coverage of the `google.auth` (+`certifi`) passthrough: `google-genai` lazily
        # imports `google.auth` during construction, which the sandbox flags without it.
        pytest.param('gateway/google-cloud:gemini-2.5-pro', 'GoogleModel', id='gateway-google'),
    ],
)
async def test_model_construction_in_workflow_passes_sandbox(
    model_name: str,
    expected_model_class: str,
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
):
    # Dummy credentials suffice since no request is made. The gateway key must encode a region
    # (`pylf_v<n>_<region>_...`) so the base URL can be inferred.
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_API_KEY', 'pylf_v1_us_0123456789abcdef')

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ConstructModelInWorkflow],
        # A sandbox violation surfaces as a workflow *task* failure, which Temporal retries forever
        # by default — so a regression would hang rather than fail. Promote any in-workflow exception
        # (e.g. `RestrictedWorkflowAccessError`) to a workflow failure so it surfaces immediately.
        workflow_failure_exception_types=[Exception],
    ):
        # Without the SDK passed through this fails with a `WorkflowFailureError`: under the suite's
        # warnings-as-errors, Temporal's "imported after initial workflow load" becomes a hard error;
        # in production the SDK's restricted `Path.home()`/env access raises `RestrictedWorkflowAccessError`.
        result = await client.execute_workflow(
            ConstructModelInWorkflow.run,
            args=[model_name],
            id=f'construct_model_{re.sub(r"[^a-zA-Z0-9]", "_", model_name)}',
            task_queue=TASK_QUEUE,
        )
    assert result == expected_model_class


# Regression test for the `httpx2` stack passthrough entries in `_workflow_runner`.
# `ModelResponse.cost()` lazily imports genai-prices on first call; inside a workflow that trips the
# sandbox unless those modules are passed through (see #6215).
@workflow.defn
class CalculateCostInWorkflow:
    @workflow.run
    async def run(self) -> float:
        response = ModelResponse(
            parts=[TextPart('ok')],
            usage=RequestUsage(input_tokens=100, output_tokens=10),
            model_name='claude-sonnet-4-5',
            provider_name='anthropic',
        )
        return float(response.cost().total_price)


async def test_response_cost_in_workflow_passes_sandbox(client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CalculateCostInWorkflow],
        workflow_failure_exception_types=[Exception],
    ):
        result = await client.execute_workflow(
            CalculateCostInWorkflow.run,
            id='calculate_cost_in_workflow',
            task_queue=TASK_QUEUE,
        )
    assert result > 0


async def test_temporal_agent():
    assert isinstance(complex_temporal_agent.model, TemporalModel)
    assert complex_temporal_agent.model.wrapped == complex_agent.model

    toolsets = complex_temporal_agent.toolsets
    assert len(toolsets) == 5

    # Empty function toolset for the agent's own tools
    assert isinstance(toolsets[0], FunctionToolset)
    assert toolsets[0].id == '<agent>'
    assert toolsets[0].tools == {}

    # Wrapped function toolset for the agent's own tools
    assert isinstance(toolsets[1], TemporalFunctionToolset)
    assert toolsets[1].id == '<agent>'
    assert isinstance(toolsets[1].wrapped, FunctionToolset)
    assert toolsets[1].wrapped.tools.keys() == {'get_weather'}

    # Wrapped 'country' toolset
    assert isinstance(toolsets[2], TemporalFunctionToolset)
    assert toolsets[2].id == 'country'
    assert toolsets[2].wrapped == complex_agent.toolsets[1]
    assert isinstance(toolsets[2].wrapped, FunctionToolset)
    assert toolsets[2].wrapped.tools.keys() == {'get_country'}

    # Wrapped 'mcp' MCP server
    assert isinstance(toolsets[3], TemporalMCPToolset)
    assert toolsets[3].id == 'mcp'
    assert toolsets[3].wrapped == complex_agent.toolsets[2]

    # Unwrapped 'external' toolset
    assert isinstance(toolsets[4], ExternalToolset)
    assert toolsets[4].id == 'external'
    assert toolsets[4] == complex_agent.toolsets[3]

    assert [
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in complex_temporal_agent.temporal_activities
    ] == snapshot(
        [
            'agent__complex_agent__event_stream_handler',
            'agent__complex_agent__model_request',
            'agent__complex_agent__model_request_stream',
            'agent__complex_agent__model_cancel_suspended_response',
            'agent__complex_agent__toolset__<agent>__call_tool',
            'agent__complex_agent__toolset__country__call_tool',
            'agent__complex_agent__mcp_server__mcp__get_instructions',
            'agent__complex_agent__mcp_server__mcp__get_tools',
            'agent__complex_agent__mcp_server__mcp__call_tool',
        ]
    )


def test_temporal_model_request_activities_capture_deps_type():
    """Both model-request activities must capture the real `deps_type` as the `deps` argument type.

    `temporalio`'s `@activity.defn` freezes a function's type hints into `arg_types` at decoration time for
    payload conversion, so `deps`'s annotation has to be set before decorating. If it's set afterwards (as the
    non-streaming activity used to do), the patch is cosmetic and the activity deserializes `deps` as a raw
    dict instead of the declared deps type.
    """
    model = dynamic_toolset_temporal_agent.model
    assert isinstance(model, TemporalModel)

    # `arg_types[1]` is the `deps` argument's captured type, which drives Temporal's payload conversion.
    deps_type = DynamicToolsetDeps | None
    request_arg_types = ActivityDefinition.must_from_callable(model.request_activity).arg_types  # pyright: ignore[reportUnknownMemberType]
    stream_arg_types = ActivityDefinition.must_from_callable(model.request_stream_activity).arg_types  # pyright: ignore[reportUnknownMemberType]
    assert request_arg_types is not None and request_arg_types[1] == deps_type
    assert stream_arg_types is not None and stream_arg_types[1] == deps_type


def test_temporal_wrapper_visit_and_replace():
    """Temporal wrapper toolsets should not be replaced by visit_and_replace."""
    from pydantic_ai.durable_exec.temporal._function_toolset import TemporalFunctionToolset

    toolsets = complex_temporal_agent._toolsets  # pyright: ignore[reportPrivateUsage]
    temporal_function_toolsets = [ts for ts in toolsets if isinstance(ts, TemporalFunctionToolset)]
    assert len(temporal_function_toolsets) >= 1

    temporal_function_toolset = temporal_function_toolsets[0]

    # visit_and_replace should return self for temporal wrappers
    result = temporal_function_toolset.visit_and_replace(lambda t: FunctionToolset(id='replaced'))
    assert result is temporal_function_toolset


async def test_temporal_agent_run(allow_model_requests: None):
    result = await simple_temporal_agent.run('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


def test_temporal_agent_run_sync(allow_model_requests: None):
    result = simple_temporal_agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is Mexico City.')


async def test_temporal_agent_run_stream(allow_model_requests: None):
    async with simple_temporal_agent.run_stream('What is the capital of Mexico?') as result:
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            [
                'The',
                'The capital',
                'The capital of',
                'The capital of Mexico',
                'The capital of Mexico is',
                'The capital of Mexico is Mexico',
                'The capital of Mexico is Mexico City',
                'The capital of Mexico is Mexico City.',
            ]
        )


async def test_temporal_agent_run_stream_events(allow_model_requests: None):
    async with simple_temporal_agent.run_stream_events('What is the capital of Mexico?') as event_stream:
        events = [event async for event in event_stream]
    assert events == snapshot(
        [
            PartStartEvent(index=0, part=TextPart(content='The')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' capital')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' of')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' Mexico')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' is')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' Mexico')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' City')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='.')),
            PartEndEvent(index=0, part=TextPart(content='The capital of Mexico is Mexico City.')),
            AgentRunResultEvent(result=AgentRunResult(output='The capital of Mexico is Mexico City.')),
        ]
    )


async def test_temporal_agent_iter(allow_model_requests: None):
    output: list[str] = []
    async with simple_temporal_agent.iter('What is the capital of Mexico?') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for chunk in stream.stream_text(debounce_by=None):
                        output.append(chunk)
    assert output == snapshot(
        [
            'The',
            'The capital',
            'The capital of',
            'The capital of Mexico',
            'The capital of Mexico is',
            'The capital of Mexico is Mexico',
            'The capital of Mexico is Mexico City',
            'The capital of Mexico is Mexico City.',
        ]
    )


@workflow.defn
class SimpleAgentWorkflowWithRunSync:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = simple_temporal_agent.run_sync(prompt)
        return result.output  # pragma: no cover


async def test_temporal_agent_run_sync_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithRunSync],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot('`agent.run_sync()` cannot be used inside a Temporal workflow. Use `await agent.run()` instead.'),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithRunSync.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunSync.__name__,
                task_queue=TASK_QUEUE,
            )


def drop_first_message(msgs: list[ModelMessage]) -> list[ModelMessage]:
    return msgs[1:] if len(msgs) > 1 else msgs


agent_with_sync_history_processor = Agent(
    model, name='agent_with_sync_history_processor', capabilities=[ProcessHistory(drop_first_message)]
)
temporal_agent_with_sync_history_processor = TemporalAgent(  # pyright: ignore[reportDeprecated]
    agent_with_sync_history_processor, activity_config=BASE_ACTIVITY_CONFIG
)


@workflow.defn
class AgentWithSyncHistoryProcessorWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_agent_with_sync_history_processor.run(prompt)
        return result.output


async def test_temporal_agent_with_sync_history_processor(allow_model_requests: None, client: Client):
    """Test that sync history processors work inside Temporal workflows.

    This validates that the _disable_threads ContextVar is properly set
    by TemporalAgent._temporal_overrides(), allowing sync history processors to
    execute without triggering NotImplementedError from anyio.to_thread.run_sync.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWithSyncHistoryProcessorWorkflow],
        plugins=[AgentPlugin(temporal_agent_with_sync_history_processor)],
    ):
        output = await client.execute_workflow(
            AgentWithSyncHistoryProcessorWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=AgentWithSyncHistoryProcessorWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')


agent_with_sync_instructions = Agent(model, name='agent_with_sync_instructions')


@agent_with_sync_instructions.instructions
def sync_instructions_fn() -> str:
    return 'You are a helpful assistant.'


temporal_agent_with_sync_instructions = TemporalAgent(  # pyright: ignore[reportDeprecated]
    agent_with_sync_instructions, activity_config=BASE_ACTIVITY_CONFIG
)


@workflow.defn
class AgentWithSyncInstructionsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_agent_with_sync_instructions.run(prompt)
        return result.output


async def test_temporal_agent_with_sync_instructions(allow_model_requests: None, client: Client):
    """Test that sync instructions functions work inside Temporal workflows.

    This validates that the _disable_threads ContextVar is properly set
    by TemporalAgent._temporal_overrides(), allowing sync instructions functions to
    execute without triggering NotImplementedError from anyio.to_thread.run_sync.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWithSyncInstructionsWorkflow],
        plugins=[AgentPlugin(temporal_agent_with_sync_instructions)],
    ):
        output = await client.execute_workflow(
            AgentWithSyncInstructionsWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=AgentWithSyncInstructionsWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')


@workflow.defn
class SimpleAgentWorkflowWithRunStream:
    @workflow.run
    async def run(self, prompt: str) -> str:
        async with simple_temporal_agent.run_stream(prompt) as result:
            pass
        return await result.get_output()  # pragma: no cover


async def test_temporal_agent_run_stream_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithRunStream],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                '`agent.run_stream()` cannot be used inside a Temporal workflow. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithRunStream.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunStream.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class SimpleAgentWorkflowWithRunStreamEvents:
    @workflow.run
    async def run(self, prompt: str) -> list[AgentStreamEvent | AgentRunResultEvent]:
        async with simple_temporal_agent.run_stream_events(prompt) as event_stream:
            return [event async for event in event_stream]  # pragma: no cover


async def test_temporal_agent_run_stream_events_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithRunStreamEvents],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                '`agent.run_stream_events()` cannot be used inside a Temporal workflow. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithRunStreamEvents.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunStreamEvents.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class SimpleAgentWorkflowWithIter:
    @workflow.run
    async def run(self, prompt: str) -> str:
        async with simple_temporal_agent.iter(prompt) as run:
            async for _ in run:
                pass
        return 'done'  # pragma: no cover


async def test_temporal_agent_iter_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithIter],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                '`agent.iter()` cannot be used inside a Temporal workflow. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithIter.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithIter.__name__,
                task_queue=TASK_QUEUE,
            )


async def test_temporal_agent_realtime_session_in_workflow():
    # A realtime session opens a long-lived, non-deterministic connection, so it can't run inside a
    # workflow; the guard trips before the model is ever connected.
    with patch.object(workflow, 'in_workflow', return_value=True):
        with pytest.raises(UserError, match='cannot be used inside a Temporal workflow'):
            async with simple_temporal_agent.realtime(cast('Any', object())).session():
                pass  # pragma: no cover


async def test_temporal_agent_realtime_signaling_in_workflow():
    # Browser-call signaling issues a live provider request, so it is guarded like a session: the two
    # helpers reach the agent through `_resolve_realtime_session`, which the wrapper guards too.
    with patch.object(workflow, 'in_workflow', return_value=True):
        realtime = simple_temporal_agent.realtime(cast('Any', object()))
        with pytest.raises(UserError, match='cannot be used inside a Temporal workflow'):
            await realtime.answer_webrtc_offer('v=0')
        with pytest.raises(UserError, match='cannot be used inside a Temporal workflow'):
            await realtime.create_client_secret()


class _FakeRealtimeConnection(RealtimeConnection):
    async def send(self, content: Any) -> None: ...  # pragma: no cover

    async def __aiter__(self) -> AsyncIterator[Any]:
        return
        yield  # pragma: no cover


class _FakeRealtimeModel(RealtimeModel):
    @property
    def model_name(self) -> str:
        return 'fake-realtime'

    @property
    def system(self) -> str:
        return 'fake'

    @property
    def profile(self) -> RealtimeModelProfile:
        return RealtimeModelProfile(
            supports_image_input=True,
            supports_manual_turn_control=True,
            supports_interruption=True,
            supports_output_truncation=True,
            supports_session_seeding=True,
            supported_native_tools=frozenset(),
        )

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[_FakeRealtimeConnection]:
        yield _FakeRealtimeConnection()


async def test_temporal_agent_realtime_session_outside_workflow():
    # Outside a workflow, the session is delegated to the wrapped agent.
    async with simple_temporal_agent.realtime(_FakeRealtimeModel()).session() as session:
        assert isinstance(session, RealtimeSession)
        assert [event async for event in session] == []


async def simple_event_stream_handler(
    ctx: RunContext,
    stream: AsyncIterable[AgentStreamEvent],
):
    pass


@workflow.defn
class SimpleAgentWorkflowWithEventStreamHandler:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt, event_stream_handler=simple_event_stream_handler)
        return result.output  # pragma: no cover


async def test_temporal_agent_run_in_workflow_with_event_stream_handler(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithEventStreamHandler],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Event stream handler cannot be set at agent run time inside a Temporal workflow, it must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithEventStreamHandler.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithEventStreamHandler.__name__,
                task_queue=TASK_QUEUE,
            )


# Unregistered model instance for testing error case
unregistered_model = OpenAIChatModel(
    'gpt-4o-mini',
    provider=OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'),
        http_client=http_client,
    ),
)


@workflow.defn
class SimpleAgentWorkflowWithRunModel:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt, model=unregistered_model)
        return result.output  # pragma: no cover


async def test_temporal_agent_run_in_workflow_with_model(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithRunModel],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Arbitrary model instances cannot be used at runtime inside a Temporal workflow. Register the model via `models` or reference a registered model by id.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithRunModel.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithRunModel.__name__,
                task_queue=TASK_QUEUE,
            )


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
class SimpleAgentWorkflowWithOverrideModel:
    @workflow.run
    async def run(self, prompt: str) -> None:
        with simple_temporal_agent.override(model=model):
            pass


async def test_temporal_agent_override_model_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideModel],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Model cannot be contextually overridden inside a Temporal workflow, it must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithOverrideModel.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithOverrideModel.__name__,
                task_queue=TASK_QUEUE,
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


@workflow.defn
class SimpleAgentWorkflowWithOverrideTools:
    @workflow.run
    async def run(self, prompt: str) -> None:
        with simple_temporal_agent.override(tools=[get_weather]):
            pass


async def test_temporal_agent_override_tools_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideTools],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Tools cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithOverrideTools.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithOverrideTools.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class SimpleAgentWorkflowWithOverrideBuiltinTools:
    @workflow.run
    async def run(self, prompt: str) -> None:
        with simple_temporal_agent.override(native_tools=[WebSearchTool()]):
            pass


async def test_temporal_agent_override_builtin_tools_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideBuiltinTools],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Native tools cannot be contextually overridden inside a Temporal workflow, they must be set at agent creation time.'
            ),
        ):
            await client.execute_workflow(
                SimpleAgentWorkflowWithOverrideBuiltinTools.run,
                args=['What is the capital of Mexico?'],
                id=SimpleAgentWorkflowWithOverrideBuiltinTools.__name__,
                task_queue=TASK_QUEUE,
            )


@workflow.defn
class SimpleAgentWorkflowWithOverrideDeps:
    @workflow.run
    async def run(self, prompt: str) -> str:
        with simple_temporal_agent.override(deps=None):
            result = await simple_temporal_agent.run(prompt)
            return result.output


async def test_temporal_agent_override_deps_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SimpleAgentWorkflowWithOverrideDeps],
        plugins=[AgentPlugin(simple_temporal_agent)],
    ):
        output = await client.execute_workflow(
            SimpleAgentWorkflowWithOverrideDeps.run,
            args=['What is the capital of Mexico?'],
            id=SimpleAgentWorkflowWithOverrideDeps.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')


agent_with_sync_tool = Agent(model, name='agent_with_sync_tool', tools=[get_weather])

# This needs to be done before the `TemporalAgent` is bound to the workflow.
temporal_agent_with_sync_tool_activity_disabled = TemporalAgent(  # pyright: ignore[reportDeprecated]
    agent_with_sync_tool,
    activity_config=BASE_ACTIVITY_CONFIG,
    tool_activity_config={
        '<agent>': {
            'get_weather': False,
        },
    },
)


@workflow.defn
class AgentWorkflowWithSyncToolActivityDisabled:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await temporal_agent_with_sync_tool_activity_disabled.run(prompt)
        return result.output  # pragma: no cover


async def test_temporal_agent_sync_tool_activity_disabled(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AgentWorkflowWithSyncToolActivityDisabled],
        plugins=[AgentPlugin(temporal_agent_with_sync_tool_activity_disabled)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                "Temporal activity config for tool 'get_weather' has been explicitly set to `False` (activity disabled), but non-async tools are run in threads which are not supported outside of an activity. Make the tool function async instead."
            ),
        ):
            await client.execute_workflow(
                AgentWorkflowWithSyncToolActivityDisabled.run,
                args=['What is the weather in Mexico City?'],
                id=AgentWorkflowWithSyncToolActivityDisabled.__name__,
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


@workflow.defn
class DirectStreamWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        messages: list[ModelMessage] = [ModelRequest.user_text_prompt(prompt)]
        async with model_request_stream(complex_temporal_agent.model, messages) as stream:
            async for _ in stream:
                pass
        return 'done'  # pragma: no cover


async def test_temporal_model_stream_direct(client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DirectStreamWorkflow],
        plugins=[AgentPlugin(complex_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'A Temporal model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            ),
        ):
            await client.execute_workflow(
                DirectStreamWorkflow.run,
                args=['What is the capital of Mexico?'],
                id=DirectStreamWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )


unserializable_deps_agent = Agent(model, name='unserializable_deps_agent', deps_type=Model)


@unserializable_deps_agent.tool
async def get_model_name(ctx: RunContext[Model]) -> str:
    return ctx.deps.model_name  # pragma: no cover


# This needs to be done before the `TemporalAgent` is bound to the workflow.
unserializable_deps_temporal_agent = TemporalAgent(unserializable_deps_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class UnserializableDepsAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await unserializable_deps_temporal_agent.run(prompt, deps=unserializable_deps_temporal_agent.model)
        return result.output  # pragma: no cover


async def test_temporal_agent_with_unserializable_deps_type(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[UnserializableDepsAgentWorkflow],
        plugins=[AgentPlugin(unserializable_deps_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                "A value passed to a Temporal activity failed to be serialized (Unable to serialize unknown type: <class 'pydantic_ai.providers.openai.OpenAIProvider'>). Temporal requires all values that are passed to activities to be serializable using Pydantic's `TypeAdapter`. Besides `deps`, this includes `model_settings`, the `RunContext` `metadata` and `tool_call_metadata`, and tool `metadata`."
            ),
        ):
            await client.execute_workflow(
                UnserializableDepsAgentWorkflow.run,
                args=['What is the model name?'],
                id=UnserializableDepsAgentWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )


async def test_logfire_plugin(client: Client):
    def setup_logfire(send_to_logfire: bool = True, metrics: Literal[False] | None = None) -> Logfire:
        instance = logfire.configure(local=True, metrics=metrics)
        instance.config.token = 'test'
        instance.config.send_to_logfire = send_to_logfire
        return instance

    plugin = LogfirePlugin(setup_logfire)

    config = client.config()
    config['plugins'] = [plugin]
    new_client = Client(**config)

    interceptor = new_client.config(active_config=True)['interceptors'][0]
    assert isinstance(interceptor, TracingInterceptor)
    if isinstance(interceptor.tracer, ProxyTracer):
        assert interceptor.tracer._instrumenting_module_name == 'temporalio'  # pyright: ignore[reportPrivateUsage] # pragma: lax no cover
    elif isinstance(interceptor.tracer, _ProxyTracer):
        assert interceptor.tracer.instrumenting_module_name == 'temporalio'  # pragma: lax no cover
    else:
        assert False, f'Unexpected tracer type: {type(interceptor.tracer)}'  # pragma: no cover

    with patch.object(
        temporal_logfire, 'OpenTelemetryConfig', wraps=temporal_logfire.OpenTelemetryConfig
    ) as open_telemetry_config:
        new_client = await Client.connect(client.service_client.config.target_host, plugins=[plugin])
    assert new_client.service_client.config.runtime is not None
    assert open_telemetry_config.call_args.kwargs['metric_periodicity'] == timedelta(seconds=60)

    plugin = LogfirePlugin(setup_logfire, metric_periodicity=timedelta(minutes=5))
    with patch.object(
        temporal_logfire, 'OpenTelemetryConfig', wraps=temporal_logfire.OpenTelemetryConfig
    ) as open_telemetry_config:
        await Client.connect(client.service_client.config.target_host, plugins=[plugin])
    assert open_telemetry_config.call_args.kwargs['metric_periodicity'] == timedelta(minutes=5)

    plugin = LogfirePlugin(setup_logfire, metrics=False)
    custom_runtime = temporal_logfire.Runtime(telemetry=temporal_logfire.TelemetryConfig())
    new_client = await Client.connect(
        client.service_client.config.target_host, plugins=[plugin], runtime=custom_runtime
    )
    assert new_client.service_client.config.runtime is custom_runtime

    plugin = LogfirePlugin(lambda: setup_logfire(send_to_logfire=False))
    new_client = await Client.connect(client.service_client.config.target_host, plugins=[plugin])
    assert new_client.service_client.config.runtime is None

    plugin = LogfirePlugin(lambda: setup_logfire(metrics=False))
    new_client = await Client.connect(client.service_client.config.target_host, plugins=[plugin])
    assert new_client.service_client.config.runtime is None


@pytest.mark.parametrize('already_configured', [True, False])
async def test_logfire_plugin_default_setup(client: Client, monkeypatch: pytest.MonkeyPatch, already_configured: bool):
    """The default setup only calls `logfire.configure()` when Logfire isn't configured yet.

    `logfire.configure()` is a reset rather than an additive call: it re-derives every unspecified
    argument from the environment and shuts down the existing tracer provider. Calling it on every
    `Client.connect()` silently discarded a host's own scrubbing patterns, additional span processors,
    console settings, and service name. Pydantic AI is instrumented either way.

    `logfire.DEFAULT_LOGFIRE_INSTANCE` is swapped for a stand-in so the assertions don't depend on
    (or disturb) whatever configuration the rest of the test session has installed globally.
    """
    instance = (
        logfire.configure(local=True, send_to_logfire=False) if already_configured else Logfire(config=LogfireConfig())
    )
    assert instance.config._initialized is already_configured  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(logfire, 'DEFAULT_LOGFIRE_INSTANCE', instance)

    configure_calls: list[dict[str, Any]] = []
    instrumented: list[Logfire] = []

    def configure(**kwargs: Any) -> Logfire:
        configure_calls.append(kwargs)
        return instance

    def instrument_pydantic_ai(self: Logfire, *args: Any, **kwargs: Any) -> None:
        instrumented.append(self)

    monkeypatch.setattr(logfire, 'configure', configure)
    monkeypatch.setattr(Logfire, 'instrument_pydantic_ai', instrument_pydantic_ai)

    await Client.connect(client.service_client.config.target_host, plugins=[LogfirePlugin()])

    assert configure_calls == ([] if already_configured else [{}])
    assert instrumented == [instance]


@pytest.mark.parametrize('already_instrumented', [True, False])
def test_logfire_plugin_default_setup_preserves_instrumentation(
    monkeypatch: pytest.MonkeyPatch, already_instrumented: bool
):
    """The default setup leaves a host's own Pydantic AI instrumentation settings alone.

    `instrument_pydantic_ai()` replaces rather than merges `Agent._instrument_default`, so calling it
    unconditionally turned a deliberate `include_content=False` back on, putting prompts, completions
    and tool call results on exported spans. A host that hasn't instrumented is still instrumented.

    As in `test_logfire_plugin_default_setup` above, `logfire.DEFAULT_LOGFIRE_INSTANCE`, `configure`
    and `instrument_pydantic_ai` are swapped for stand-ins so the assertions neither depend on nor
    disturb whatever configuration the rest of the test session has installed globally.
    """
    instance = Logfire(config=LogfireConfig())
    monkeypatch.setattr(logfire, 'DEFAULT_LOGFIRE_INSTANCE', instance)

    instrumented: list[Logfire] = []

    def configure(**kwargs: Any) -> Logfire:
        return instance

    def instrument_pydantic_ai(self: Logfire, *args: Any, **kwargs: Any) -> None:
        instrumented.append(self)

    monkeypatch.setattr(logfire, 'configure', configure)
    monkeypatch.setattr(Logfire, 'instrument_pydantic_ai', instrument_pydantic_ai)

    settings = InstrumentationSettings(include_content=False, include_binary_content=False)
    monkeypatch.setattr(Agent, '_instrument_default', settings if already_instrumented else False)

    temporal_logfire._default_setup_logfire()  # pyright: ignore[reportPrivateUsage]

    # With a stand-in in place, whether the plugin instruments at all is the observable: the stand-in
    # deliberately doesn't assign `_instrument_default`, so asserting on it here would prove nothing.
    assert instrumented == ([] if already_instrumented else [instance])
    if already_instrumented:
        assert Agent._instrument_default is settings  # pyright: ignore[reportPrivateUsage]


hitl_agent = Agent(
    model,
    name='hitl_agent',
    output_type=[str, DeferredToolRequests],
    instructions='Just call tools without asking for confirmation.',
)


@hitl_agent.tool
async def create_file(ctx: RunContext, path: str) -> None:
    raise CallDeferred


@hitl_agent.tool
async def delete_file(ctx: RunContext, path: str) -> bool:
    if not ctx.tool_call_approved:
        raise ApprovalRequired
    return True


hitl_temporal_agent = TemporalAgent(hitl_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class HitlAgentWorkflow:
    def __init__(self):
        self._status: Literal['running', 'waiting_for_results', 'done'] = 'running'
        self._deferred_tool_requests: DeferredToolRequests | None = None
        self._deferred_tool_results: DeferredToolResults | None = None

    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str | DeferredToolRequests]:
        messages: list[ModelMessage] = [ModelRequest.user_text_prompt(prompt)]
        while True:
            result = await hitl_temporal_agent.run(
                message_history=messages, deferred_tool_results=self._deferred_tool_results
            )
            messages = result.all_messages()

            if isinstance(result.output, DeferredToolRequests):
                self._deferred_tool_requests = result.output
                self._deferred_tool_results = None
                self._status = 'waiting_for_results'

                await workflow.wait_condition(lambda: self._deferred_tool_results is not None)
                self._status = 'running'
            else:
                self._status = 'done'
                return result

    @workflow.query
    def get_status(self) -> Literal['running', 'waiting_for_results', 'done']:
        return self._status

    @workflow.query
    def get_deferred_tool_requests(self) -> DeferredToolRequests | None:
        return self._deferred_tool_requests

    @workflow.signal
    def set_deferred_tool_results(self, results: DeferredToolResults) -> None:
        self._status = 'running'
        self._deferred_tool_requests = None
        self._deferred_tool_results = results


async def test_temporal_agent_with_hitl_tool(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[HitlAgentWorkflow],
        plugins=[AgentPlugin(hitl_temporal_agent)],
    ):
        workflow = await client.start_workflow(
            HitlAgentWorkflow.run,
            args=['Delete the file `.env` and create `test.txt`'],
            id=HitlAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        while True:
            await asyncio.sleep(1)
            status = await workflow.query(HitlAgentWorkflow.get_status)
            if status == 'done':
                break
            elif status == 'waiting_for_results':  # pragma: no branch
                deferred_tool_requests = await workflow.query(HitlAgentWorkflow.get_deferred_tool_requests)
                assert deferred_tool_requests is not None

                results = DeferredToolResults()
                # Approve all calls
                for tool_call in deferred_tool_requests.approvals:
                    results.approvals[tool_call.tool_call_id] = True

                for tool_call in deferred_tool_requests.calls:
                    results.calls[tool_call.tool_call_id] = 'Success'

                await workflow.signal(HitlAgentWorkflow.set_deferred_tool_results, results)

        result = await workflow.result()
        assert result.output == snapshot(
            'The file `.env` has been deleted and `test.txt` has been created successfully.'
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Delete the file `.env` and create `test.txt`',
                            timestamp=IsDatetime(),
                        )
                    ],
                    # NOTE in other tests we check timestamp=IsNow(tz=timezone.utc)
                    # but temporal tests fail when we use IsNow
                    timestamp=IsDatetime(),
                    instructions='Just call tools without asking for confirmation.',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='delete_file',
                            args='{"path": ".env"}',
                            tool_call_id='call_jYdIdRZHxZTn5bWCq5jlMrJi',
                        ),
                        ToolCallPart(
                            tool_name='create_file',
                            args='{"path": "test.txt"}',
                            tool_call_id='call_TmlTVWQbzrXCZ4jNsCVNbNqu',
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=71,
                        output_tokens=46,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                        cost=Decimal('0.0006375'),
                        output_reasoning_tokens=0,
                    ),
                    model_name=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls', 'timestamp': '2025-08-28T22:11:03Z'},
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='delete_file',
                            content=True,
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        ),
                        ToolReturnPart(
                            tool_name='create_file',
                            content='Success',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        ),
                    ],
                    timestamp=IsDatetime(),
                    instructions='Just call tools without asking for confirmation.',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The file `.env` has been deleted and `test.txt` has been created successfully.'
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=133,
                        output_tokens=19,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                        cost=Decimal('0.0005225'),
                        output_reasoning_tokens=0,
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'stop', 'timestamp': '2025-08-28T22:11:06Z'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


model_retry_agent = Agent(model, name='model_retry_agent')


@model_retry_agent.tool_plain
def get_weather_in_city(city: str) -> str:
    if city != 'Mexico City':
        raise ModelRetry('Did you mean Mexico City?')
    return 'sunny'


model_retry_temporal_agent = TemporalAgent(model_retry_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class ModelRetryWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> AgentRunResult[str]:
        result = await model_retry_temporal_agent.run(prompt)
        return result


async def test_temporal_agent_with_model_retry(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ModelRetryWorkflow],
        plugins=[AgentPlugin(model_retry_temporal_agent)],
    ):
        workflow = await client.start_workflow(
            ModelRetryWorkflow.run,
            args=['What is the weather in CDMX?'],
            id=ModelRetryWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        result = await workflow.result()
        assert result.output == snapshot('The weather in Mexico City is currently sunny.')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the weather in CDMX?',
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
                            tool_name='get_weather_in_city',
                            args='{"city":"CDMX"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=47,
                        output_tokens=17,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                        cost=Decimal('0.0002875'),
                        output_reasoning_tokens=0,
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls', 'timestamp': '2025-08-28T23:19:50Z'},
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Did you mean Mexico City?',
                            tool_name='get_weather_in_city',
                            tool_call_id=IsStr(),
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
                            tool_name='get_weather_in_city',
                            args='{"city":"Mexico City"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=87,
                        output_tokens=17,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                        cost=Decimal('0.0003875'),
                        output_reasoning_tokens=0,
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'tool_calls', 'timestamp': '2025-08-28T23:19:51Z'},
                    provider_response_id=IsStr(),
                    finish_reason='tool_call',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather_in_city',
                            content='sunny',
                            tool_call_id=IsStr(),
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
                        input_tokens=116,
                        output_tokens=10,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                        cost=Decimal('0.00039'),
                        output_reasoning_tokens=0,
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'stop', 'timestamp': '2025-08-28T23:19:52Z'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


tool_failed_agent = Agent(TestModel(call_tools=['failing_tool']), name='tool_failed_agent')


@tool_failed_agent.tool_plain
def failing_tool() -> str:
    raise ToolFailed('Disk full')


tool_failed_temporal_agent = TemporalAgent(tool_failed_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class ToolFailedWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[tuple[str, Any, str]]:
        result = await tool_failed_temporal_agent.run(prompt)
        return [
            (part.tool_name, part.content, part.outcome)
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]


async def test_temporal_agent_with_tool_failed(client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ToolFailedWorkflow],
        plugins=[AgentPlugin(tool_failed_temporal_agent)],
    ):
        tool_returns = await client.execute_workflow(
            ToolFailedWorkflow.run,
            args=['Call the failing tool'],
            id=ToolFailedWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    assert tool_returns == [('failing_tool', 'Disk full', 'failed')]


class CustomModelSettings(ModelSettings, total=False):
    custom_setting: str


def return_settings(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(str(agent_info.model_settings))])


model_settings = CustomModelSettings(max_tokens=123, custom_setting='custom_value')
return_settings_model = FunctionModel(return_settings, settings=model_settings)

settings_agent = Agent(return_settings_model, name='settings_agent')

# This needs to be done before the `TemporalAgent` is bound to the workflow.
settings_temporal_agent = TemporalAgent(settings_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class SettingsAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await settings_temporal_agent.run(prompt)
        return result.output


async def test_custom_model_settings(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SettingsAgentWorkflow],
        plugins=[AgentPlugin(settings_temporal_agent)],
    ):
        output = await client.execute_workflow(
            SettingsAgentWorkflow.run,
            args=['Give me those settings'],
            id=SettingsAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot("{'max_tokens': 123, 'custom_setting': 'custom_value'}")


# `httpx.Timeout` is a documented `ModelSettings.timeout` value, but it isn't serializable by
# Pydantic, so it fails when the model request activity is scheduled — the error must not blame `deps`.
timeout_settings_agent = Agent(
    FunctionModel(return_settings, settings=ModelSettings(timeout=httpx.Timeout(10.0))),
    name='timeout_settings_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class UnserializableModelSettingsWorkflow:
    @workflow.run
    async def run(self) -> str:
        result = await timeout_settings_agent.run('Give me those settings')
        return result.output  # pragma: no cover


async def test_unserializable_model_settings(client: Client):
    """An unserializable `model_settings` value fails the workflow with an accurate `UserError`.

    The expected type name is built from `httpx.Timeout` itself because importing `google-genai`
    replaces it with a subclass of its own, so the name depends on what the session imported.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[UnserializableModelSettingsWorkflow],
        plugins=[AgentPlugin(timeout_settings_agent)],
    ):
        with workflow_raises(
            UserError,
            f'A value passed to a Temporal activity failed to be serialized '
            f'(Unable to serialize unknown type: {httpx.Timeout!r}). '
            "Temporal requires all values that are passed to activities to be serializable using Pydantic's "
            '`TypeAdapter`. Besides `deps`, this includes `model_settings`, the `RunContext` `metadata` and '
            '`tool_call_metadata`, and tool `metadata`.',
        ):
            await client.execute_workflow(
                UnserializableModelSettingsWorkflow.run,
                id=UnserializableModelSettingsWorkflow.__name__,
                task_queue=TASK_QUEUE,
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


image_agent = Agent(model, name='image_agent', output_type=BinaryImage)

# This needs to be done before the `TemporalAgent` is bound to the workflow.
image_temporal_agent = TemporalAgent(image_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class ImageAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> BinaryImage:
        result = await image_temporal_agent.run(prompt)
        return result.output  # pragma: no cover


async def test_image_agent(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ImageAgentWorkflow],
        plugins=[AgentPlugin(image_temporal_agent)],
    ):
        with workflow_raises(
            UserError,
            snapshot(
                'Image output is not supported with Temporal because the image would ride the activity payload, '
                'which is capped by the server blob-size limit (2MB by default, leaving about 1.5MB of raw image '
                'bytes once base64-encoded).'
            ),
        ):
            await client.execute_workflow(
                ImageAgentWorkflow.run,
                args=['Generate an image of an axolotl.'],
                id=ImageAgentWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )


def payload_limit_detail(size: int) -> str:
    """Temporal's own sentence inside the guard's message, which differs across the range we support.

    `temporalio` 1.31 moved the payload-size check out of the Python SDK and into Temporal's Rust core,
    which reports the breach without the byte counts the SDK's own check appended. Both shapes are inside
    the `>=1.24` range the `temporal` extra declares, so which one to expect is read off the installed
    SDK rather than pinned.
    """
    exceeded = '[TMPRL1103] Attempted to upload payloads with size that exceeded the error limit'
    if Version(version('temporalio')) >= Version('1.31'):
        return exceeded
    return f'{exceeded}. Size: {size} bytes, Limit: 2097152 bytes'


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


async def _respond_with_oversized_image(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # A native image-generation tool puts the image on the response like this, so it rides the
    # model-request activity payload rather than a tool-call one.
    return ModelResponse(
        parts=[
            TextPart('here is your image'),
            FilePart(content=BinaryImage(data=b'\x00' * 1_600_000, media_type='image/png')),
        ]
    )


oversized_model_response_agent = Agent(
    FunctionModel(_respond_with_oversized_image, model_name='oversized-response-model'),
    name='oversized_model_response_agent',
    deps_type=type(None),
    capabilities=[TemporalDurability(activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=60)))],
)


@workflow.defn
class OversizedModelResponseWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await oversized_model_response_agent.run(prompt)
        return result.output  # pragma: no cover


async def test_oversized_model_response_payload(client: Client):
    """A model response carrying binary content over Temporal's payload limit points at the cause (#7110).

    The `allow_image_output` guard doesn't cover this: it fires on the agent's `output_type`, while a
    native image-generation tool returns the image as a `FilePart` on the model response instead.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[OversizedModelResponseWorkflow],
        plugins=[AgentPlugin(oversized_model_response_agent)],
    ):
        with workflow_raises(
            UserError,
            f"The response from model 'function:oversized-response-model' is too large for Temporal. {payload_limit_detail(2134150)}. Binary content like an image is base64-encoded into the activity payload, so if that is the cause, the raw-byte budget is about three quarters of the limit — roughly 1.5MB at the 2MB default. A generated image is the usual cause, so ask the model for a smaller one through the model settings; a streamed segment can also overflow on its buffered events alone. To keep large payloads out of the workflow history without changing what your tools or models return, configure Temporal external storage (or a claim-check `payload_codec`) on your `DataConverter` — `PydanticAIPlugin` preserves it, and it covers every payload in both directions. See https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/#large-payloads",
        ):
            await client.execute_workflow(
                OversizedModelResponseWorkflow.run,
                args=['Draw me something.'],
                id=OversizedModelResponseWorkflow.__name__,
                task_queue=TASK_QUEUE,
                execution_timeout=timedelta(seconds=30),
            )


# ============================================================================
# DocumentUrl Serialization Test - Verifies that DocumentUrl with custom
# media_type is properly serialized through Temporal activities
# ============================================================================

document_url_agent = Agent(
    TestModel(custom_output_args={'url': 'https://example.com/doc/12345', 'media_type': 'application/pdf'}),
    name='document_url_agent',
    output_type=DocumentUrl,
)

document_url_temporal_agent = TemporalAgent(document_url_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class DocumentUrlAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> DocumentUrl:
        result = await document_url_temporal_agent.run(prompt)
        return result.output


async def test_document_url_serialization_preserves_media_type(allow_model_requests: None, client: Client):
    """Test that `DocumentUrl` with custom `media_type` is preserved through Temporal serialization.

    This is a regression test for https://github.com/pydantic/pydantic-ai/issues/3949
    where `DocumentUrl.media_type` (a computed field) was lost during Temporal activity
    serialization because the backing field `_media_type` was excluded from serialization.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DocumentUrlAgentWorkflow],
        plugins=[AgentPlugin(document_url_temporal_agent)],
    ):
        output = await client.execute_workflow(
            DocumentUrlAgentWorkflow.run,
            args=['Return a document'],
            id=DocumentUrlAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            DocumentUrl(url='https://example.com/doc/12345', _media_type='application/pdf', _identifier='eb8998')
        )


# ============================================================================
# UploadedFile Serialization Test - Verifies that UploadedFile with custom
# media_type is properly serialized through Temporal activities
# ============================================================================

uploaded_file_agent = Agent(
    TestModel(
        custom_output_args={
            'file_id': 'file-abc123',
            'provider_name': 'openai',
            'media_type': 'image/png',
            'identifier': 'file-1',
        }
    ),
    name='uploaded_file_agent',
    output_type=UploadedFile,
)

uploaded_file_temporal_agent = TemporalAgent(uploaded_file_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class UploadedFileAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> UploadedFile:
        result = await uploaded_file_temporal_agent.run(prompt)
        return result.output


async def test_uploaded_file_serialization_preserves_media_type(allow_model_requests: None, client: Client):
    """Test that `UploadedFile` with custom `media_type` is preserved through Temporal serialization."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[UploadedFileAgentWorkflow],
        plugins=[AgentPlugin(uploaded_file_temporal_agent)],
    ):
        output = await client.execute_workflow(
            UploadedFileAgentWorkflow.run,
            args=['Return a file reference'],
            id=UploadedFileAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            UploadedFile(file_id='file-abc123', provider_name='openai', _media_type='image/png', _identifier='file-1')
        )


# Can't use the `openai_api_key` fixture here because the workflow needs to be defined at the top level of the file.
web_search_model = OpenAIResponsesModel(
    'gpt-5',
    provider=OpenAIProvider(
        api_key=os.getenv('OPENAI_API_KEY', 'mock-api-key'),
        http_client=http_client,
    ),
)

web_search_agent = Agent(
    web_search_model,
    name='web_search_agent',
    capabilities=[NativeTool(WebSearchTool(user_location=WebSearchUserLocation(city='Mexico City', country='MX')))],
)

# This needs to be done before the `TemporalAgent` is bound to the workflow.
web_search_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    web_search_agent,
    activity_config=BASE_ACTIVITY_CONFIG,
    model_activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=300)),
)


@workflow.defn
class WebSearchAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await web_search_temporal_agent.run(prompt)
        return result.output


async def test_web_search_agent_run_in_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WebSearchAgentWorkflow],
        plugins=[AgentPlugin(web_search_temporal_agent)],
    ):
        output = await client.execute_workflow(
            WebSearchAgentWorkflow.run,
            args=['In one sentence, what is the top news story in my country today?'],
            id=WebSearchAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot(
            'Severe floods and landslides across Veracruz, Hidalgo, and Puebla have cut off hundreds of communities and left dozens dead and many missing, prompting a major federal emergency response. ([apnews.com](https://apnews.com/article/5d036e18057361281e984b44402d3b1b?utm_source=openai))'
        )


def test_temporal_run_context_preserves_run_id():
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id='run-123',
    )

    serialized = TemporalRunContext.serialize_run_context(ctx)
    assert serialized['run_id'] == 'run-123'

    reconstructed = TemporalRunContext.deserialize_run_context(serialized, deps=None)
    assert reconstructed.run_id == 'run-123'


run_id_test_agent = Agent(TestModel(custom_output_text='ok'), name='run_id_test_agent')
run_id_temporal_agent = TemporalAgent(run_id_test_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class RunIdAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str, run_id: str) -> list[str]:
        result = await run_id_temporal_agent.run(prompt, run_id=run_id)
        return [result.run_id, *[m.run_id or '<unset>' for m in result.all_messages()]]


async def test_temporal_agent_explicit_run_id(client: Client):
    """A pre-minted `run_id=` survives Temporal activity serialization and stamps all new messages."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RunIdAgentWorkflow],
        plugins=[AgentPlugin(run_id_temporal_agent)],
    ):
        output = await client.execute_workflow(
            RunIdAgentWorkflow.run,
            args=['Hello', 'run-from-temporal'],
            id=RunIdAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == ['run-from-temporal', 'run-from-temporal', 'run-from-temporal']


def test_temporal_run_context_serializes_metadata():
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id='run-123',
        metadata={'env': 'prod'},
    )

    serialized = TemporalRunContext.serialize_run_context(ctx)
    assert serialized['metadata'] == {'env': 'prod'}

    reconstructed = TemporalRunContext.deserialize_run_context(serialized, deps=None)
    assert reconstructed.metadata == {'env': 'prod'}


def test_temporal_run_context_excludes_agent():
    """agent is not serialized but defaults to None after deserialization."""
    agent = Agent('test', name='test_agent')
    ctx = RunContext(
        deps=None,
        agent=agent,
        model=TestModel(),
        usage=RunUsage(),
        run_id='run-123',
    )

    serialized = TemporalRunContext.serialize_run_context(ctx)
    assert 'agent' not in serialized

    # Without agent — e.g. when _agent was never set on a temporal wrapper
    reconstructed = deserialize_run_context(TemporalRunContext, serialized, deps=None, agent=None)
    assert reconstructed.agent is None

    # With agent — as used by TemporalAgent's wrappers
    reconstructed = deserialize_run_context(TemporalRunContext, serialized, deps=None, agent=agent)
    assert reconstructed.agent is agent
    assert agent.name == 'test_agent'


def test_temporal_run_context_enqueue_raises_inside_activity():
    """`ctx.enqueue()` inside a Temporal activity raises the shared durable explanation.

    `pending_messages` isn't serialized across the activity boundary, so any code running
    activity-side (a tool, a `process_tool_call` hook, an `event_stream_handler`) is in a
    durable unit whose result is replayed without re-running it; an enqueue would be dropped.
    """
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run-123')
    serialized = TemporalRunContext.serialize_run_context(ctx)
    reconstructed = deserialize_run_context(TemporalRunContext, serialized, deps=None, agent=None)

    with pytest.raises(UserError, match='enqueued messages would be dropped'):
        reconstructed.enqueue('later')
    # An empty enqueue stays a no-op, matching a normal run.
    assert reconstructed.enqueue() is None


def test_temporal_run_context_serializes_usage():
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(
            requests=2,
            tool_calls=1,
            input_tokens=123,
            output_tokens=456,
            details={'foo': 1},
            future_tokens=7,
            label='original',
            zero_tokens=0,
        ),
        run_id='run-123',
    )

    serialized = TemporalRunContext.serialize_run_context(ctx)
    assert serialized['usage'] == ctx.usage

    reconstructed = TemporalRunContext.deserialize_run_context(serialized, deps=None)
    assert reconstructed.usage == ctx.usage


def test_temporal_run_context_serializes_usage_limits():
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        usage_limits=UsageLimits(request_limit=7, total_tokens_limit=1000),
        run_id='run-123',
    )

    serialized = TemporalRunContext.serialize_run_context(ctx)
    assert serialized['usage_limits'] == ctx.usage_limits

    reconstructed = TemporalRunContext.deserialize_run_context(serialized, deps=None)
    assert reconstructed.usage_limits == ctx.usage_limits


async def test_temporal_run_context_preserves_anchored_evidence():
    """Provider-exact evidence is computed workflow-side and survives the untyped activity payload."""
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        _anchored_evidence=AnchoredEvidence(
            discovered_tool_names=frozenset({'deferred_tool'}),
            loaded_capability_ids=frozenset({'deferred_capability'}),
        ),
    )

    wire = await _serialized_run_context_across_the_wire(ctx)
    reconstructed = TemporalRunContext.deserialize_run_context(wire, deps=None)

    assert reconstructed._anchored_evidence == AnchoredEvidence(  # pyright: ignore[reportPrivateUsage]
        discovered_tool_names=frozenset({'deferred_tool'}),
        loaded_capability_ids=frozenset({'deferred_capability'}),
    )


async def test_temporal_run_context_without_anchored_evidence_still_answers_availability():
    """A payload that predates the field keeps answering, with the history-derived window.

    `serialize_run_context` is a documented override point, so a subclass written against an
    earlier version returns a dict without it. Guarding it like the other omitted fields would
    turn that into a `UserError` from a tool that only asked whether it may run.
    """
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    wire = await _serialized_run_context_across_the_wire(ctx)
    older_payload = {name: value for name, value in wire.items() if name != '_anchored_evidence'}

    reconstructed = TemporalRunContext.deserialize_run_context(older_payload, deps=None)

    assert reconstructed._anchored_evidence == AnchoredEvidence()  # pyright: ignore[reportPrivateUsage]
    assert reconstructed.is_tool_available(ToolDefinition(name='hidden', defer_loading=True)) is False


@pytest.mark.parametrize('carried', [True, False, None])
async def test_temporal_run_context_accepts_the_legacy_capability_loaded_key(carried: bool | None):
    """A payload written under the old field name still lands on `capability_active`.

    An activity can be dispatched by one worker version and replayed by another, and
    `serialize_run_context` is a documented override point, so both a mid-deployment payload and a
    subclass written against the old name reach here. Without the mapping the value would sit in
    `__dict__` under a name nothing reads, and the guard would report `capability_active` as a
    field that never crossed the boundary.

    `None` is the case that matters most and is easiest to miss: it is what every activity dispatched
    *outside* capability dispatch carries, so a mapping keyed on the value rather than on the key's
    presence breaks the common path while passing for `True`.
    """
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), capability_active=carried)
    wire = await _serialized_run_context_across_the_wire(ctx)
    renamed = {'capability_active': 'capability_loaded', 'active_capability_ids': 'available_capability_ids'}
    legacy_payload = {renamed.get(name, name): value for name, value in wire.items()}
    assert 'capability_loaded' in legacy_payload
    assert 'available_capability_ids' in legacy_payload

    reconstructed = TemporalRunContext.deserialize_run_context(legacy_payload, deps=None)

    assert reconstructed.capability_active is carried
    # The second rename rides the same mapping: without it the guard would report
    # `active_capability_ids` as a field that never crossed the boundary.
    assert reconstructed.active_capability_ids == set()


def test_temporal_run_context_serialization_is_exhaustive():
    """Every `RunContext` field must be consciously categorized for Temporal serialization.

    Guards against silent drift: when a `RunContext` field is added, this test fails until
    the author either includes it in `TemporalRunContext.serialize_run_context` or lists it
    in `intentionally_unserialized` below with a reason. Without that decision a new field
    silently becomes unavailable inside a Temporal activity (the `__getattribute__` guard
    raises `UserError` on access), which is how the deferred-capability fields were missed.
    """
    # Fields deliberately NOT carried across the activity boundary, each with its reason.
    intentionally_unserialized = {
        'deps',  # passed separately to deserialize_run_context
        'agent',  # reattached after deserialize by deserialize_run_context
        'model',  # live Model instance, not serializable
        'tracer',  # live tracer, not serializable
        'tool_manager',  # live ToolManager, not serializable (documented on the field)
        'capabilities',  # live capability objects (toolsets/hooks/callables), not serializable
        'root_capability',  # live capability chain, not serializable; reattached from the bound agent by deserialize_run_context
        'pending_messages',  # live run queue, meaningless outside the running agent; replaced by an EnqueueGuard
        'messages',  # full history would be duplicated into every activity payload, against Temporal's 2MB limit
        'prompt',  # multi-modal BinaryContent would ride in every payload, against Temporal's 2MB limit; text-only subclasses can opt in
        'validation_context',  # arbitrary user object with no serialization contract
        'model_settings',  # only set for model requests, which receive it as their own typed activity param
        '_mcp_tool_defs_cache',  # run-local cache read/written in workflow code; never needed inside an activity
        '_event_stream_buffer',  # run-local event buffer drained in workflow code; a public emit surface for activities is a follow-up
        'realtime_session',  # live RealtimeSession, not serializable; realtime sessions don't run inside Temporal activities
        '_cancellation',  # runtime-only controller holding a live asyncio task reference; cannot cross the activity boundary
    }
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    serialized = set(TemporalRunContext.serialize_run_context(ctx))
    all_fields = set(RunContext.__dataclass_fields__)

    overlap = serialized & intentionally_unserialized
    assert not overlap, f'Fields both serialized and excluded: {overlap}'

    uncategorized = all_fields - (serialized | intentionally_unserialized)
    assert not uncategorized, (
        f'Uncategorized `RunContext` fields: {uncategorized}. Add each to '
        '`TemporalRunContext.serialize_run_context` or to `intentionally_unserialized` (with a reason).'
    )


async def _serialized_run_context_across_the_wire(ctx: RunContext[Any]) -> dict[str, Any]:
    """Serialize a run context and put it through Temporal's Pydantic data converter.

    The run context reaches an activity inside `CallToolParams.serialized_run_context`, which is
    `Any`-typed so `TemporalRunContext` subclasses can add their own fields. The converter has no
    type to decode against, so it hands back plain JSON — which is what makes rehydration in
    `TemporalRunContext.__init__` load-bearing rather than decoration.
    """
    params = CallToolParams(
        name='tool', tool_args={}, serialized_run_context=TemporalRunContext.serialize_run_context(ctx), tool_def=None
    )
    payloads = await pydantic_data_converter.encode([params])
    (decoded,) = await pydantic_data_converter.decode(payloads, [CallToolParams])
    return cast('dict[str, Any]', decoded.serialized_run_context)


async def test_temporal_run_context_rehydrates_containers():
    """Sets and usage arrive inside an activity as the objects they were.

    Everything structured degrades on the untyped hop: before rehydration `discovered_tool_names`
    and `loaded_capability_ids` arrived as `list`s, so `available_tool_names` raised
    `TypeError: unsupported operand type(s) for |: 'set' and 'list'` and
    `loaded_capability_ids.add(...)` raised `AttributeError: 'list' object has no attribute 'add'`.
    """
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(input_tokens=3),
        usage_limits=UsageLimits(request_limit=7),
        run_id='run-123',
        conversation_id='conv-123',
        discovered_tool_names={'searched_tool'},
        loaded_capability_ids={'deferred_capability'},
        trace_include_content=True,
        instrumentation_version=4,
    )

    wire = await _serialized_run_context_across_the_wire(ctx)
    # What the activity actually receives: sets as lists and models as dicts.
    assert wire['discovered_tool_names'] == ['searched_tool']
    assert isinstance(wire['usage'], dict)
    assert 'prompt' not in wire

    reconstructed = TemporalRunContext.deserialize_run_context(wire, deps=None)
    assert reconstructed.discovered_tool_names == {'searched_tool'}
    assert reconstructed.loaded_capability_ids == {'deferred_capability'}
    # Mutating the loaded-capability set is what the `load_capability` tool body does in-step.
    reconstructed.loaded_capability_ids.add('loaded_in_activity')
    assert reconstructed.loaded_capability_ids == {'deferred_capability', 'loaded_in_activity'}
    # `tool_manager` is `None` inside an activity, so this is the documented fallback path.
    assert reconstructed.tool_manager is None
    assert reconstructed.available_tool_names == {'searched_tool'}
    assert reconstructed.usage == ctx.usage
    assert reconstructed.usage_limits == ctx.usage_limits
    assert reconstructed.conversation_id == 'conv-123'
    assert reconstructed.trace_include_content is True
    assert reconstructed.instrumentation_version == 4


async def test_temporal_run_context_omitted_field_raises_instead_of_defaulting():
    """An omitted field raises rather than reading as the `RunContext` dataclass default.

    Fields with plain defaults live on the class, so `super().__getattribute__` used to find them:
    reads of `model_settings` and `validation_context` returned `None` inside an activity,
    indistinguishable from a run that really had none.
    """
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run-123')
    reconstructed = deserialize_run_context(
        TemporalRunContext, await _serialized_run_context_across_the_wire(ctx), deps=None, agent=None
    )

    with pytest.raises(UserError) as exc_info:
        _ = reconstructed.model_settings
    assert str(exc_info.value) == snapshot(
        "'model_settings' is not available on 'TemporalRunContext' inside a Temporal activity. To make the attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it as the `run_context_type` argument to `TemporalDurability`."
    )
    for name in ('prompt', 'messages', 'validation_context', 'model', 'tracer', 'capabilities'):
        with pytest.raises(UserError, match=f'{name!r} is not available'):
            getattr(reconstructed, name)

    # The framework re-attaches these, so they read as `None` rather than raising: `agent` and
    # `root_capability` come from the worker's agent instance, `tool_manager` is documented as
    # unavailable and keeps `available_tool_names` working.
    assert reconstructed.agent is None
    assert reconstructed.root_capability is None
    assert reconstructed.tool_manager is None
    assert reconstructed.available_tool_names == set()
    # An attribute that isn't a `RunContext` field at all keeps raising plain `AttributeError`.
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(reconstructed, 'not_a_field')


async def test_is_tool_available_answers_for_a_capability_owned_tool_inside_an_activity():
    """The definition form must answer, not raise, for a tool a capability contributed.

    `is_tool_available` consults `active_capability_ids` for any tool carrying a
    `capability_id`, and the `capabilities` registry deliberately doesn't cross the boundary. The
    docs send toolset authors to the definition form precisely because it works inside `get_tools`,
    which under Temporal runs in an activity — so the ids travel as a snapshot.
    """
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id='run-123',
        capabilities={'guarded': Capability[Any](id='guarded', description='Guarded.', defer_loading=True)},
        loaded_capability_ids={'guarded'},
        discovered_tool_names={'secret_op'},
    )
    reconstructed = deserialize_run_context(
        TemporalRunContext, await _serialized_run_context_across_the_wire(ctx), deps=None, agent=None
    )

    assert reconstructed.active_capability_ids == {'guarded'}
    loaded = ToolDefinition(name='secret_op', defer_loading=True, capability_id='guarded')
    assert reconstructed.is_tool_available(loaded) is True

    unloaded = ToolDefinition(name='other_op', defer_loading=True, capability_id='not_loaded')
    assert reconstructed.is_tool_available(unloaded) is False


async def test_loaded_capability_tool_without_a_reveal_marker_answers_inside_an_activity():
    """The on-demand set travels too, so a stripped reveal marker doesn't flip the answer.

    A deferred capability's load is itself the reveal for its own tools, and telling that apart
    from a capability since reconfigured always-on needs the *configured* set, which lives in the
    `capabilities` registry and cannot cross the boundary. Without the snapshot this degrades to
    the discovery check and answers `False` inside an activity while the workflow says `True` --
    for a tool no search can ever surface, so nothing could restore the marker.
    """
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id='run-123',
        capabilities={'guarded': Capability[Any](id='guarded', description='Guarded.', defer_loading=True)},
        loaded_capability_ids={'guarded'},
        # No `discovered_tool_names`: the reveal marker is gone, as a history processor can leave it.
    )
    tool_def = ToolDefinition(name='secret_op', defer_loading=True, capability_id='guarded')
    assert ctx.is_tool_available(tool_def) is True

    reconstructed = deserialize_run_context(
        TemporalRunContext, await _serialized_run_context_across_the_wire(ctx), deps=None, agent=None
    )
    assert reconstructed.is_tool_available(tool_def) is True

    # The registry itself still doesn't cross — only the ids it resolves to.
    with pytest.raises(UserError, match="'capabilities' is not available"):
        _ = reconstructed.capabilities


class LegacyFieldsRunContext(TemporalRunContext[Any]):
    """A user subclass with its own field set."""

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        return {
            'run_id': ctx.run_id,
            'usage': ctx.usage,
            'usage_limits': ctx.usage_limits,
            'discovered_tool_names': ctx.discovered_tool_names,
            'custom': 'from-subclass',
        }


async def test_temporal_run_context_subclass_with_its_own_field_set():
    """A subclass that overrides `serialize_run_context` keeps working, errors and all.

    Carrying more fields by default must not require subclasses to be updated: the fields the
    subclass includes (including its own extra ones) are available, and the ones it leaves out
    raise the error that points at `serialize_run_context`.
    """
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(input_tokens=3),
        prompt='hello',
        run_id='run-123',
        conversation_id='conv-123',
        discovered_tool_names={'searched_tool'},
    )
    params = CallToolParams(
        name='tool',
        tool_args={},
        serialized_run_context=LegacyFieldsRunContext.serialize_run_context(ctx),
        tool_def=None,
    )
    payloads = await pydantic_data_converter.encode([params])
    (decoded,) = await pydantic_data_converter.decode(payloads, [CallToolParams])
    reconstructed = LegacyFieldsRunContext.deserialize_run_context(decoded.serialized_run_context, deps=None)

    assert reconstructed.run_id == 'run-123'
    assert reconstructed.usage == ctx.usage
    assert reconstructed.discovered_tool_names == {'searched_tool'}
    assert reconstructed.available_tool_names == {'searched_tool'}
    # No capability snapshot in this subclass's field set, so the property falls back to the base
    # one, which reads the registry — and that is guarded, so it raises rather than quietly
    # reporting no capabilities are active.
    with pytest.raises(UserError, match="'capabilities' is not available"):
        _ = reconstructed.active_capability_ids
    # Same for the on-demand set that `is_tool_available` consults: an older subclass doesn't carry
    # it either, so the base property reads the guarded registry and raises rather than reporting an
    # empty set, which would silently answer "no capability is deferred" for every tool.
    with pytest.raises(UserError, match="'capabilities' is not available"):
        _ = reconstructed._deferred_capability_ids  # pyright: ignore[reportPrivateUsage]
    assert reconstructed.__dict__['custom'] == 'from-subclass'
    for name in ('prompt', 'conversation_id', 'instrumentation_version'):
        with pytest.raises(UserError, match=f'{name!r} is not available on {LegacyFieldsRunContext.__name__!r}'):
            getattr(reconstructed, name)


def _run_context_fields_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('report_run_context', {})])
    else:
        return ModelResponse(parts=[TextPart('done')])


_run_context_fields_agent = Agent(
    FunctionModel(_run_context_fields_model),
    name='run_context_fields_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@_run_context_fields_agent.tool
def report_run_context(ctx: RunContext) -> dict[str, Any]:
    """Report what a tool running inside an activity sees on its run context."""
    try:
        prompt = repr(ctx.prompt)
    except UserError as e:
        prompt = str(e)
    try:
        messages = repr(ctx.messages)
    except UserError as e:
        messages = str(e)
    return {
        'prompt': prompt,
        'conversation_id': ctx.conversation_id,
        'discovered_tool_names_type': type(ctx.discovered_tool_names).__name__,
        'available_tool_names': sorted(ctx.available_tool_names),
        'instrumentation_version': ctx.instrumentation_version,
        'messages': messages,
    }


@workflow.defn
class RunContextFieldsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> dict[str, Any]:
        result = await _run_context_fields_agent.run(prompt)
        report = next(
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        )
        return {'report': report, 'conversation_id': result.conversation_id}


async def test_run_context_fields_in_temporal_activity(client: Client):
    """A tool inside an activity correlates to the conversation and lists tools.

    `conversation_id` is carried, and `available_tool_names` works because
    `discovered_tool_names` is rehydrated as a set. `prompt` and `messages` are not carried, so
    reading either raises the actionable error.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[RunContextFieldsWorkflow],
        plugins=[AgentPlugin(_run_context_fields_agent)],
    ):
        output = await client.execute_workflow(
            RunContextFieldsWorkflow.run,
            args=['What did I ask?'],
            id=RunContextFieldsWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    # The tool saw the run's real conversation id, not a `None` default.
    assert output['report']['conversation_id'] == output['conversation_id']
    # `available_tool_names` is the `discovered_tool_names` fallback here (no tool search in this
    # run, so empty), but it returns rather than raising `TypeError` on a `set | list`.
    assert output['report'] == snapshot(
        {
            'prompt': "'prompt' is not available on 'TemporalRunContext' inside a Temporal activity. To make the attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it as the `run_context_type` argument to `TemporalDurability`.",
            'conversation_id': IsStr(),
            'discovered_tool_names_type': 'set',
            'available_tool_names': ['report_run_context'],
            'instrumentation_version': 5,
            'messages': "'messages' is not available on 'TemporalRunContext' inside a Temporal activity. To make the attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it as the `run_context_type` argument to `TemporalDurability`.",
        }
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


# ============================================================================
# ctx.agent in Temporal activities
# ============================================================================


def _ctx_agent_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('get_agent_name', {})])
    else:
        return ModelResponse(parts=[TextPart('done')])


_ctx_agent_test_agent = Agent(
    FunctionModel(_ctx_agent_model),
    name='ctx_agent_test',
)


@_ctx_agent_test_agent.tool
def get_agent_name(ctx: RunContext) -> str:
    return (ctx.agent.name or 'unnamed') if ctx.agent else 'unknown'


_ctx_agent_temporal_agent = TemporalAgent(_ctx_agent_test_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class CtxAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[ModelMessage]:
        result = await _ctx_agent_temporal_agent.run(prompt)
        return result.all_messages()


async def test_ctx_agent_in_temporal_activity(allow_model_requests: None, client: Client):
    """ctx.agent is available inside Temporal activities, giving access to agent properties like name."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CtxAgentWorkflow],
        plugins=[AgentPlugin(_ctx_agent_temporal_agent)],
    ):
        messages = await client.execute_workflow(
            CtxAgentWorkflow.run,
            args=['test'],
            id=CtxAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='test', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_agent_name', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:_ctx_agent_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_agent_name',
                        content='ctx_agent_test',
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
                usage=RequestUsage(input_tokens=52, output_tokens=3),
                model_name='function:_ctx_agent_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


# ============================================================================
# Beta Graph API Tests - Tests for running pydantic-graph beta API in Temporal
# ============================================================================


@dataclass
class GraphState:
    """State for the graph execution test."""

    values: list[int] = field(default_factory=list[int])


# Create a graph with parallel execution using the beta API
graph_builder = GraphBuilder(
    name='parallel_test_graph',
    state_type=GraphState,
    input_type=int,
    output_type=list[int],
)


@graph_builder.step
async def source(ctx: StepContext[GraphState, None, int]) -> int:
    """Source step that passes through the input value."""
    return ctx.inputs


@graph_builder.step
async def multiply_by_two(ctx: StepContext[GraphState, None, int]) -> int:
    """Multiply input by 2."""
    return ctx.inputs * 2


@graph_builder.step
async def multiply_by_three(ctx: StepContext[GraphState, None, int]) -> int:
    """Multiply input by 3."""
    return ctx.inputs * 3


@graph_builder.step
async def multiply_by_four(ctx: StepContext[GraphState, None, int]) -> int:
    """Multiply input by 4."""
    return ctx.inputs * 4


# Create a join to collect results
result_collector = graph_builder.join(reduce_list_append, initial_factory=list[int])

# Build the graph with parallel edges (broadcast pattern)
graph_builder.add(
    graph_builder.edge_from(graph_builder.start_node).to(source),
    # Broadcast: send value to all three parallel steps
    graph_builder.edge_from(source).to(multiply_by_two, multiply_by_three, multiply_by_four),
    # Collect all results
    graph_builder.edge_from(multiply_by_two, multiply_by_three, multiply_by_four).to(result_collector),
    graph_builder.edge_from(result_collector).to(graph_builder.end_node),
)

parallel_test_graph = graph_builder.build()


@workflow.defn
class ParallelGraphWorkflow:
    """Workflow that executes a graph with parallel task execution."""

    @workflow.run
    async def run(self, input_value: int) -> list[int]:
        """Run the parallel graph workflow.

        Args:
            input_value: The input number to process

        Returns:
            List of results from parallel execution
        """
        result = await parallel_test_graph.run(
            state=GraphState(),
            inputs=input_value,
        )
        return result


async def test_beta_graph_parallel_execution_in_workflow(client: Client):
    """Test that beta graph API with parallel execution works in Temporal workflows.

    This test verifies the fix for the bug where parallel task execution in graphs
    wasn't working properly with Temporal workflows due to GraphTask/GraphTaskRequest
    serialization issues.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ParallelGraphWorkflow],
    ):
        output = await client.execute_workflow(
            ParallelGraphWorkflow.run,
            args=[10],
            id=ParallelGraphWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        # Results can be in any order due to parallel execution
        # 10 * 2 = 20, 10 * 3 = 30, 10 * 4 = 40
        assert sorted(output) == [20, 30, 40]


@workflow.defn
class WorkflowWithAgents(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [simple_temporal_agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt)
        return result.output


@workflow.defn
class WorkflowWithAgentsWithoutPydanticAIWorkflow:
    __pydantic_ai_agents__ = [simple_temporal_agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await simple_temporal_agent.run(prompt)
        return result.output


async def test_passing_agents_through_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WorkflowWithAgents],
    ):
        output = await client.execute_workflow(
            WorkflowWithAgents.run,
            args=['What is the capital of Mexico?'],
            id=WorkflowWithAgents.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')


async def test_passing_agents_through_workflow_without_pydantic_ai_workflow(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WorkflowWithAgentsWithoutPydanticAIWorkflow],
    ):
        output = await client.execute_workflow(
            WorkflowWithAgentsWithoutPydanticAIWorkflow.run,
            args=['What is the capital of Mexico?'],
            id=WorkflowWithAgentsWithoutPydanticAIWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == snapshot('The capital of Mexico is Mexico City.')


# Multi-Model Support Tests

# Module-level test models for multi-model selection test
test_model_selection_1 = TestModel(custom_output_text='Response from model 1')
test_model_selection_2 = TestModel(custom_output_text='Response from model 2')
test_model_selection_3 = TestModel(custom_output_text='Response from model 3')

# Module-level test models for error test
test_model_error_1 = TestModel()
test_model_error_2 = TestModel()
test_model_error_unregistered = TestModel()

# Module-level temporal agents
agent_selection = Agent(test_model_selection_1, name='multi_model_workflow_test')
multi_model_selection_test_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    agent_selection,
    name='multi_model_workflow_test',
    models={
        'model_2': test_model_selection_2,
        'model_3': test_model_selection_3,
    },
    activity_config=BASE_ACTIVITY_CONFIG,
)

agent_error = Agent(test_model_error_1, name='error_test')
multi_model_error_test_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    agent_error,
    name='error_test',
    models={'other': test_model_error_2},
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class MultiModelWorkflow:
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await multi_model_selection_test_agent.run(prompt, model=model_id)
        return result.output


class _BuiltinToolModel(TestModel):
    SUPPORTED_TOOLS: frozenset[type[AbstractNativeTool]] = frozenset()

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        return cls.SUPPORTED_TOOLS

    def _request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        # Override to skip TestModel._request's builtin tools rejection
        return ModelResponse(parts=[TextPart(self.custom_output_text or '')], model_name=self.model_name)


class _WebSearchOnlyModel(_BuiltinToolModel):
    SUPPORTED_TOOLS = frozenset({WebSearchTool})


class _CodeExecutionOnlyModel(_BuiltinToolModel):
    SUPPORTED_TOOLS = frozenset({CodeExecutionTool})


def _select_builtin_tool(ctx: RunContext[Any]) -> AbstractNativeTool:
    # `RunContext.model` is an `AbstractModel`; narrow to a request-response model to read its profile.
    ctx_model = ctx.model
    assert isinstance(ctx_model, Model)
    model = cast('Model[Any]', ctx_model)
    if WebSearchTool in model.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS):
        return WebSearchTool()
    return CodeExecutionTool()


web_search_builtin_model = _WebSearchOnlyModel(custom_output_text='search model', model_name='web-search')
code_execution_builtin_model = _CodeExecutionOnlyModel(custom_output_text='code model', model_name='code-exec')

builtin_tool_agent = Agent(
    web_search_builtin_model,
    name='builtin_tool_dynamic_agent',
    capabilities=[NativeTool(_select_builtin_tool)],
)

builtin_tool_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    builtin_tool_agent,
    name='builtin_tool_dynamic_agent',
    models={'code': code_execution_builtin_model},
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class BuiltinToolWorkflow:
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await builtin_tool_temporal_agent.run(prompt, model=model_id)
        return result.output


# Model that does NOT support any builtin tools (used as default)
no_builtin_support_model = _BuiltinToolModel(custom_output_text='no builtin support', model_name='no-builtin-test')

# Model that DOES support WebSearchTool (registered as alternate model)
web_search_builtin_override_model = _WebSearchOnlyModel(
    custom_output_text='web search response',
    model_name='web-search-override',
)

# Agent initialized with model that doesn't support builtins, but has builtin tools configured
builtins_in_workflow_agent = Agent(
    no_builtin_support_model,
    capabilities=[NativeTool(WebSearchTool()), Instrumentation(settings=InstrumentationSettings())],
    name='builtins_in_workflow',
)

# TemporalAgent registers an alternate model that DOES support builtins
builtins_in_workflow_temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    builtins_in_workflow_agent,
    name='builtins_in_workflow',
    models={'web_search': web_search_builtin_override_model},
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class BuiltinsInWorkflow(PydanticAIWorkflow):
    @workflow.run
    async def run(self, prompt: str, model_id: str | None = None) -> str:
        result = await builtins_in_workflow_temporal_agent.run(prompt, model=model_id)
        return result.output


@workflow.defn
class MultiModelWorkflowUnregistered:
    @workflow.run
    async def run(self, prompt: str) -> str:
        # Try to use an unregistered model
        result = await multi_model_error_test_agent.run(prompt, model=test_model_error_unregistered)
        return result.output  # pragma: no cover


async def test_temporal_agent_multi_model_reserved_id():
    """Test that reserved model IDs raise helpful errors."""
    test_model1 = TestModel()
    test_model2 = TestModel()

    agent = Agent(test_model1, name='reserved_id_test')
    with pytest.raises(UserError, match="Model ID 'default' is reserved"):
        TemporalAgent(  # pyright: ignore[reportDeprecated]
            agent,
            name='reserved_id_test',
            models={'default': test_model2},
        )


async def test_temporal_agent_multi_model_selection_in_workflow(allow_model_requests: None, client: Client):
    """Test selecting different models in a workflow using the model parameter."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModelWorkflow],
        plugins=[AgentPlugin(multi_model_selection_test_agent)],
    ):
        # Test using default model (model_id=None)
        output = await client.execute_workflow(
            MultiModelWorkflow.run,
            args=['Hello', None],
            id='MultiModelWorkflow_default',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 1'

        # Test selecting second model by ID
        output = await client.execute_workflow(
            MultiModelWorkflow.run,
            args=['Hello', 'model_2'],
            id='MultiModelWorkflow_model2',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 2'

        # Test selecting third model by ID
        output = await client.execute_workflow(
            MultiModelWorkflow.run,
            args=['Hello', 'model_3'],
            id='MultiModelWorkflow_model3',
            task_queue=TASK_QUEUE,
        )
        assert output == 'Response from model 3'


async def test_temporal_dynamic_builtin_tools_select_by_model(allow_model_requests: None, client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BuiltinToolWorkflow],
        plugins=[AgentPlugin(builtin_tool_temporal_agent)],
    ):
        output = await client.execute_workflow(
            BuiltinToolWorkflow.run,
            args=['Hello', None],
            id='BuiltinToolWorkflow_default',
            task_queue=TASK_QUEUE,
        )
        assert output == 'search model'
        assert isinstance(web_search_builtin_model.last_model_request_parameters, ModelRequestParameters)
        assert web_search_builtin_model.last_model_request_parameters.native_tools
        assert isinstance(web_search_builtin_model.last_model_request_parameters.native_tools[0], WebSearchTool)

        output = await client.execute_workflow(
            BuiltinToolWorkflow.run,
            args=['Hello', 'code'],
            id='BuiltinToolWorkflow_code',
            task_queue=TASK_QUEUE,
        )
        assert output == 'code model'
        assert isinstance(code_execution_builtin_model.last_model_request_parameters, ModelRequestParameters)
        assert code_execution_builtin_model.last_model_request_parameters.native_tools
        assert isinstance(
            code_execution_builtin_model.last_model_request_parameters.native_tools[0],
            CodeExecutionTool,
        )


async def test_builtins_in_workflow_with_runtime_model_override(allow_model_requests: None, client: Client):
    """Test that builtin tools work when agent is initialized with a non-supporting model
    but run with a model that does support builtins."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BuiltinsInWorkflow],
        plugins=[AgentPlugin(builtins_in_workflow_temporal_agent)],
    ):
        # Run with the model that supports WebSearchTool
        result = await client.execute_workflow(
            BuiltinsInWorkflow.run,
            args=['search for something', 'web_search'],
            id='BuiltinsInWorkflow',
            task_queue=TASK_QUEUE,
        )
        assert result == 'web search response'

    # Verify the web search model received the WebSearchTool in its request parameters
    assert isinstance(web_search_builtin_override_model.last_model_request_parameters, ModelRequestParameters)
    assert web_search_builtin_override_model.last_model_request_parameters.native_tools
    assert isinstance(
        web_search_builtin_override_model.last_model_request_parameters.native_tools[0],
        WebSearchTool,
    )


async def test_temporal_agent_multi_model_unregistered_error(allow_model_requests: None, client: Client):
    """Test that using an unregistered model raises a helpful error."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModelWorkflowUnregistered],
        plugins=[AgentPlugin(multi_model_error_test_agent)],
    ):
        with workflow_raises(
            UserError,
            'Arbitrary model instances cannot be used at runtime inside a Temporal workflow. Register the model via `models` or reference a registered model by id.',
        ):
            await client.execute_workflow(
                MultiModelWorkflowUnregistered.run,
                args=['Hello'],
                id='MultiModelWorkflowUnregistered',
                task_queue=TASK_QUEUE,
            )


async def test_temporal_agent_multi_model_outside_workflow():
    """Test that multi-model agents work outside workflows (using wrapped agent behavior).

    Outside a workflow, a TemporalAgent should behave like a regular Agent.
    This includes supporting model selection by registered ID or instance.
    """
    test_model1 = TestModel(custom_output_text='Model 1 response')
    test_model2 = TestModel(custom_output_text='Model 2 response')
    test_model_unregistered = TestModel(custom_output_text='Unregistered model response')

    agent = Agent(test_model1, name='outside_workflow_test')
    temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
        agent,
        name='outside_workflow_test',
        models={'secondary': test_model2},
    )

    # Outside workflow, should use default model
    result = await temporal_agent.run('Hello')
    assert result.output == 'Model 1 response'

    # Outside workflow, passing a registered model ID should also work
    result = await temporal_agent.run('Hello', model='secondary')
    assert result.output == 'Model 2 response'

    # Passing a registered model instance should also work
    result = await temporal_agent.run('Hello', model=test_model2)
    assert result.output == 'Model 2 response'

    # Passing an unregistered model instance should also work outside workflow
    result = await temporal_agent.run('Hello', model=test_model_unregistered)
    assert result.output == 'Unregistered model response'


async def test_temporal_agent_without_default_model():
    """Test that a TemporalAgent can be created without a default model if models is provided.

    When no model is provided to run(), the first registered model should be used.
    """
    test_model1 = TestModel(custom_output_text='Model 1 response')
    test_model2 = TestModel(custom_output_text='Model 2 response')

    # Agent without a model
    agent = Agent(name='no_default_model_test')
    temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
        agent,
        name='no_default_model_test',
        models={
            'primary': test_model1,
            'secondary': test_model2,
        },
    )

    # Without a model, should use the first registered model
    result = await temporal_agent.run('Hello')
    assert result.output == 'Model 1 response'

    # Outside workflow, can use registered models by id
    result = await temporal_agent.run('Hello', model='primary')
    assert result.output == 'Model 1 response'

    result = await temporal_agent.run('Hello', model='secondary')
    assert result.output == 'Model 2 response'


# Workflow for testing passing model instances (can't be workflow args, so map by key)
_model_instance_map = {
    'default_instance': test_model_selection_1,
    'model_2_instance': test_model_selection_2,
}


@workflow.defn
class MultiModelWorkflowInstance:
    @workflow.run
    async def run(self, prompt: str, instance_key: str) -> str:
        model_instance = _model_instance_map[instance_key]
        result = await multi_model_selection_test_agent.run(prompt, model=model_instance)
        return result.output


@pytest.mark.parametrize(
    ('model_id', 'expected_output'),
    [
        pytest.param('default', 'Response from model 1', id='default_explicit'),
    ],
)
async def test_temporal_agent_model_selection_by_id(
    allow_model_requests: None, client: Client, model_id: str, expected_output: str
):
    """Test model selection by passing model ID strings."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModelWorkflow],
        plugins=[AgentPlugin(multi_model_selection_test_agent)],
    ):
        output = await client.execute_workflow(
            MultiModelWorkflow.run,
            args=['Hello', model_id],
            id=f'MultiModelWorkflow_{model_id}',
            task_queue=TASK_QUEUE,
        )
        assert output == expected_output


@pytest.mark.parametrize(
    ('instance_key', 'expected_output'),
    [
        pytest.param('default_instance', 'Response from model 1', id='default_instance'),
        pytest.param('model_2_instance', 'Response from model 2', id='registered_instance'),
    ],
)
async def test_temporal_agent_model_selection_by_instance(
    allow_model_requests: None, client: Client, instance_key: str, expected_output: str
):
    """Test model selection by passing model instances."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModelWorkflowInstance],
        plugins=[AgentPlugin(multi_model_selection_test_agent)],
    ):
        output = await client.execute_workflow(
            MultiModelWorkflowInstance.run,
            args=['Hello', instance_key],
            id=f'MultiModelWorkflowInstance_{instance_key}',
            task_queue=TASK_QUEUE,
        )
        assert output == expected_output


def test_temporal_model_profile_for_raw_strings():
    """Test TemporalModel infers model_name, system, and profile from raw strings without constructing providers."""

    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__profile_inference',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # Without using_model, properties come from default
    assert temporal_model.profile == default_model.profile
    assert temporal_model.model_name == default_model.model_name
    assert temporal_model.system == default_model.system

    # With raw string, all properties are inferred correctly
    with temporal_model.using_model('openai:gpt-5'):
        assert temporal_model.model_name == 'gpt-5'
        assert temporal_model.system == 'openai'
        assert temporal_model.profile == infer_model_profile('openai:gpt-5')

    # Anthropic profile inference includes WebSearchTool support
    with temporal_model.using_model('anthropic:claude-sonnet-4-5'):
        assert temporal_model.model_name == 'claude-sonnet-4-5'
        assert temporal_model.system == 'anthropic'
        assert temporal_model.profile == infer_model_profile('anthropic:claude-sonnet-4-5')

    # Registered models work correctly for all properties
    alt_model = TestModel(custom_output_text='alt', model_name='alt-model')
    temporal_model_with_registry = TemporalModel(
        default_model,
        activity_name_prefix='test__profile_registry',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
        models={'alt': alt_model},
    )
    with temporal_model_with_registry.using_model('alt'):
        assert temporal_model_with_registry.model_name == 'alt-model'
        assert temporal_model_with_registry.system == alt_model.system
        assert temporal_model_with_registry.profile == alt_model.profile


class DefaultHostModel(TestModel):
    @property
    def base_url(self) -> str:
        return 'https://default.example.com:1111/v1'


class AltHostModel(TestModel):
    @property
    def base_url(self) -> str:
        return 'https://alt.example.com:2222/v1'


def test_temporal_model_base_url_follows_active_model():
    """`base_url` resolves through `using_model()` like the other identity properties.

    Without this it would report the wrapped default's URL, so a request span would name the active
    model in `gen_ai.request.model` while pointing `server.address` at a different model's host.
    """
    temporal_model = TemporalModel(
        DefaultHostModel(model_name='default-model'),
        activity_name_prefix='test__base_url',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
        models={'alt': AltHostModel(model_name='alt-model')},
    )

    assert temporal_model.base_url == snapshot('https://default.example.com:1111/v1')

    with temporal_model.using_model('alt'):
        assert temporal_model.base_url == snapshot('https://alt.example.com:2222/v1')

    with temporal_model.using_model('openai:gpt-5'):
        assert temporal_model.base_url is None


def test_temporal_model_model_id_follows_active_model():
    """`model_id` resolves through `using_model()` rather than reporting the wrapped default's.

    `WrapperModel` forwards `model_id` so a wrapped `FallbackModel` keeps its own composed ID, which
    would otherwise pin this to the default model. The ID names the activity a request runs under, so
    a swapped-in model has to be the one it reports.
    """
    temporal_model = TemporalModel(
        TestModel(model_name='default-model'),
        activity_name_prefix='test__model_id',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
        models={'alt': FallbackModel(TestModel(model_name='alt-model'), TestModel(model_name='spare-model'))},
    )

    assert temporal_model.model_id == snapshot('test:default-model')

    with temporal_model.using_model('alt'):
        assert temporal_model.model_id == snapshot('fallback:test:alt-model,test:spare-model')

    with temporal_model.using_model('openai:gpt-5'):
        assert temporal_model.model_id == snapshot('openai:gpt-5')

    with temporal_model.using_model('gpt-5'):
        assert temporal_model.model_id == snapshot('test:gpt-5')


async def test_temporal_model_request_outside_workflow():
    """Test that TemporalModel.request() falls back to wrapped model outside a workflow.

    When TemporalModel.request() is called directly (not through TemporalAgent.run())
    and not inside a Temporal workflow, it should delegate to the wrapped model's request method.
    """
    test_model = TestModel(custom_output_text='Direct model response')

    temporal_model = TemporalModel(
        test_model,
        activity_name_prefix='test__direct_request',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # Call request() directly - outside a workflow, this should fall back to super().request()
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
    response = await temporal_model.request(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            native_tools=[],
            output_mode='text',
            allow_text_output=True,
            output_tools=[],
            output_object=None,
        ),
    )

    # Verify response comes from the wrapped TestModel
    assert any(isinstance(part, TextPart) and part.content == 'Direct model response' for part in response.parts)


async def test_temporal_model_cancel_suspended_response_outside_workflow():
    """`TemporalModel.cancel_suspended_response()` falls back to the wrapped model outside a workflow.

    Inside a workflow it runs the provider teardown in the `model_cancel_suspended_response` activity
    (registered in `temporal_activities`) so the raw HTTP call never runs in the workflow sandbox;
    outside a workflow it delegates straight to the wrapped model.
    """
    cancelled: list[ModelResponse] = []

    class RecordingModel(TestModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append(response)

    temporal_model = TemporalModel(
        RecordingModel(),
        activity_name_prefix='test__direct_cancel',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # The cancel activity is registered alongside the request activities.
    assert [
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in temporal_model.temporal_activities
    ] == snapshot(
        [
            'test__direct_cancel__model_request',
            'test__direct_cancel__model_request_stream',
            'test__direct_cancel__model_cancel_suspended_response',
        ]
    )

    response = ModelResponse(parts=[TextPart('paused')], state='suspended')
    await temporal_model.cancel_suspended_response(response)
    assert cancelled == [response]


@dataclass
class CancelTenantDeps:
    tenant_id: str


factory_cancel_calls: list[tuple[CancelTenantDeps, str]] = []
factory_cancelled_responses: list[ModelResponse] = []


def cancel_provider_factory(ctx: RunContext[object], provider_name: str) -> Any:
    assert isinstance(ctx.deps, CancelTenantDeps)
    factory_cancel_calls.append((ctx.deps, provider_name))
    return object()


class FactoryCancelRecordingModel(TestModel):
    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        factory_cancelled_responses.append(response)


factory_cancel_temporal_model = TemporalModel(
    TestModel(),
    activity_name_prefix='test__factory_cancel',
    activity_config=BASE_ACTIVITY_CONFIG,
    deps_type=CancelTenantDeps,
    provider_factory=cancel_provider_factory,
)


@workflow.defn
class FactoryCancelWorkflow:
    @workflow.run
    async def run(self, response: ModelResponse) -> None:
        deps = CancelTenantDeps(tenant_id='tenant-a')
        ctx = RunContext[CancelTenantDeps](deps=deps, model=TestModel(), usage=RunUsage(), run_id='factory-cancel')
        await execute_temporal_activity(
            activity=factory_cancel_temporal_model.cancel_suspended_response_activity,
            args=[
                _ModelCancelParams(
                    response=response,
                    model_id='runtime-provider:model',
                    serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
                    deps=deps,
                )
            ],
            **BASE_ACTIVITY_CONFIG,
        )


async def test_temporal_model_cancel_suspended_response_uses_provider_factory(client: Client) -> None:
    """A real worker preserves structured deps used to select the cancellation client."""
    factory_cancel_calls.clear()
    factory_cancelled_responses.clear()
    factory_model = FactoryCancelRecordingModel()

    def infer_runtime_model(model_id: str, provider_factory: Callable[[str], object] | None = None):
        assert provider_factory is not None
        provider_factory('runtime-provider')
        return factory_model

    response = ModelResponse(parts=[TextPart('paused')], state='suspended')
    with patch('pydantic_ai.durable_exec.temporal._model.models.infer_model', side_effect=infer_runtime_model):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[FactoryCancelWorkflow],
            activities=factory_cancel_temporal_model.temporal_activities,
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            await client.execute_workflow(
                FactoryCancelWorkflow.run,
                args=[response],
                id=f'{FactoryCancelWorkflow.__name__}-{uuid.uuid4()}',
                task_queue=TASK_QUEUE,
            )

    assert factory_cancel_calls == [(CancelTenantDeps(tenant_id='tenant-a'), 'runtime-provider')]
    assert factory_cancelled_responses == [response]


async def test_temporal_model_cancel_suspended_response_accepts_legacy_payload() -> None:
    """An old cancel payload keeps the environment-inference behavior."""
    response = ModelResponse(parts=[TextPart('paused')], state='suspended')
    params = TypeAdapter(_ModelCancelParams).validate_python(
        {'response': response, 'model_id': 'runtime-provider:model'}
    )
    assert params.serialized_run_context is None
    assert params.deps is None

    cancelled: list[ModelResponse] = []

    class RecordingModel(TestModel):
        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            cancelled.append(response)

    environment_model = RecordingModel()

    temporal_model = TemporalModel(
        TestModel(),
        activity_name_prefix='test__legacy_cancel',
        activity_config=BASE_ACTIVITY_CONFIG,
        deps_type=str,
        provider_factory=Mock(),
    )
    with patch('pydantic_ai.durable_exec.temporal._model.models.infer_model', return_value=environment_model) as infer:
        await ActivityEnvironment().run(temporal_model.cancel_suspended_response_activity, params)

    infer.assert_called_once_with('runtime-provider:model')
    assert cancelled == [response]


# Module-level so the `@workflow.defn` below can bind to it (mirrors `simple_temporal_agent`). The
# activity records into this list; since activities always run outside the workflow sandbox in the
# worker process, the workflow can dispatch the teardown while the assertion still observes it here.
model_cancel_calls: list[ModelResponse] = []


class CancelRecordingModel(TestModel):
    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        model_cancel_calls.append(response)


cancel_temporal_model = TemporalModel(
    CancelRecordingModel(),
    activity_name_prefix='cancel_suspended',
    activity_config=BASE_ACTIVITY_CONFIG,
    deps_type=type(None),
)


@workflow.defn
class CancelSuspendedResponseWorkflow:
    @workflow.run
    async def run(self, response: ModelResponse) -> None:
        # In-workflow, `cancel_suspended_response` must dispatch the provider teardown to the
        # `model_cancel_suspended_response` activity rather than make the raw HTTP call in the sandbox.
        await cancel_temporal_model.cancel_suspended_response(response)


replay_legacy_cancel_payload = True


@workflow.defn
class CancelSuspendedResponseReplayWorkflow:
    @workflow.run
    async def run(self, response: ModelResponse) -> None:
        params = _ModelCancelParams(response=response)
        if not replay_legacy_cancel_payload:
            ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), run_id='cancel-replay')
            params.serialized_run_context = TemporalRunContext.serialize_run_context(ctx)
            params.deps = ctx.deps
        await execute_temporal_activity(
            activity=cancel_temporal_model.cancel_suspended_response_activity,
            args=[params],
            **BASE_ACTIVITY_CONFIG,
        )


async def test_temporal_model_cancel_suspended_response_in_workflow(client: Client):
    """Inside a workflow, `cancel_suspended_response` tears the server-side job down via an activity.

    Counterpart to `test_temporal_model_cancel_suspended_response_outside_workflow`: it drives the
    in-workflow override -> `workflow.execute_activity` -> activity-body path end to end, proving the
    wrapped model's cancel actually runs and that the `ModelResponse` argument survives serialization
    across both the workflow and activity boundaries.
    """
    model_cancel_calls.clear()
    response = ModelResponse(parts=[TextPart('paused')], state='suspended')
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CancelSuspendedResponseWorkflow],
        activities=cancel_temporal_model.temporal_activities,
    ):
        await client.execute_workflow(
            CancelSuspendedResponseWorkflow.run,
            args=[response],
            id=CancelSuspendedResponseWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    # The teardown ran in the activity worker against the wrapped model, with the response faithfully
    # round-tripped through both serialization boundaries.
    assert model_cancel_calls == [response]


async def test_temporal_model_cancel_suspended_response_replays_legacy_history(client: Client) -> None:
    """A history recorded with the old cancel payload replays with the one-argument command intact."""
    global replay_legacy_cancel_payload

    replay_legacy_cancel_payload = True
    response = ModelResponse(parts=[TextPart('paused')], state='suspended')
    workflow_id = f'{CancelSuspendedResponseReplayWorkflow.__name__}-{uuid.uuid4()}'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CancelSuspendedResponseReplayWorkflow],
        activities=cancel_temporal_model.temporal_activities,
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        await client.execute_workflow(
            CancelSuspendedResponseReplayWorkflow.run,
            args=[response],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    replay_legacy_cancel_payload = False
    try:
        await Replayer(
            workflows=[CancelSuspendedResponseReplayWorkflow],
            workflow_runner=UnsandboxedWorkflowRunner(),
            data_converter=pydantic_data_converter,
        ).replay_workflow(history)
    finally:
        replay_legacy_cancel_payload = True


async def test_temporal_model_request_stream_outside_workflow():
    """Test that TemporalModel.request_stream() falls back to wrapped model outside a workflow.

    When TemporalModel.request_stream() is called directly (not through TemporalAgent.run())
    and not inside a Temporal workflow, it should delegate to the wrapped model's request_stream method.
    """
    test_model = TestModel(custom_output_text='Direct stream response')

    temporal_model = TemporalModel(
        test_model,
        activity_name_prefix='test__direct_stream',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # Call request_stream() directly - outside a workflow, this should fall back to super().request_stream()
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('Hello')]
    async with temporal_model.request_stream(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            native_tools=[],
            output_mode='text',
            allow_text_output=True,
            output_tools=[],
            output_object=None,
        ),
    ) as stream:
        # Consume the stream
        async for _ in stream:
            pass

        # Get the final response
        response = stream.get()

    # Verify response comes from the wrapped TestModel
    assert any(isinstance(part, TextPart) and part.content == 'Direct stream response' for part in response.parts)


class CustomPydanticPayloadConverter(PydanticPayloadConverter):
    """A custom payload converter that inherits from PydanticPayloadConverter."""

    pass


class CustomPayloadConverter(DefaultPayloadConverter):
    """A custom payload converter that does not inherit from PydanticPayloadConverter."""

    pass


class MockPayloadCodec(PayloadCodec):
    """A mock payload codec for testing (simulates encryption codec)."""

    async def encode(
        self, payloads: Sequence[temporalio.api.common.v1.Payload]
    ) -> list[temporalio.api.common.v1.Payload]:  # pragma: no cover
        return list(payloads)

    async def decode(
        self, payloads: Sequence[temporalio.api.common.v1.Payload]
    ) -> list[temporalio.api.common.v1.Payload]:  # pragma: no cover
        return list(payloads)


async def test_pydantic_ai_payload_converter_builds_type_adapter_once() -> None:
    """Repeated decoding reuses one adapter instead of rebuilding it for every payload."""
    temporal_payload_converter._type_adapter.cache_clear()  # pyright: ignore[reportPrivateUsage]
    converter = DataConverter(payload_converter_class=PydanticAIPayloadConverter)
    payloads = await converter.encode(['result'])

    with patch.object(
        temporal_payload_converter, 'TypeAdapter', wraps=temporal_payload_converter.TypeAdapter
    ) as type_adapter:
        for _ in range(5):
            assert await converter.decode(payloads, [str]) == ['result']

    assert type_adapter.call_count == 1


async def test_pydantic_ai_payload_converter_reuses_more_than_128_type_adapters() -> None:
    """Cyclic access over 129 distinct hints does not rebuild adapters after warmup."""
    temporal_payload_converter._type_adapter.cache_clear()  # pyright: ignore[reportPrivateUsage]
    hints = [type(f'Result{i}', (BaseModel,), {'__annotations__': {'v': int}}) for i in range(129)]

    for hint in hints:
        temporal_payload_converter._type_adapter(hint)  # pyright: ignore[reportPrivateUsage]

    misses_after_warmup = temporal_payload_converter._type_adapter.cache_info().misses  # pyright: ignore[reportPrivateUsage]
    for _ in range(3):
        for hint in hints:
            temporal_payload_converter._type_adapter(hint)  # pyright: ignore[reportPrivateUsage]

    assert temporal_payload_converter._type_adapter.cache_info().misses == misses_after_warmup  # pyright: ignore[reportPrivateUsage]


async def test_pydantic_ai_payload_converter_separates_type_hints() -> None:
    """Different hints use distinct adapters and preserve their respective output types."""
    temporal_payload_converter._type_adapter.cache_clear()  # pyright: ignore[reportPrivateUsage]
    converter = DataConverter(payload_converter_class=PydanticAIPayloadConverter)
    str_payloads = await converter.encode(['1'])
    int_payloads = await converter.encode([1])

    with patch.object(
        temporal_payload_converter, 'TypeAdapter', wraps=temporal_payload_converter.TypeAdapter
    ) as type_adapter:
        assert await converter.decode(str_payloads, [str]) == ['1']
        assert await converter.decode(int_payloads, [int]) == [1]

    assert type_adapter.call_count == 2


async def test_pydantic_ai_payload_converter_accepts_unhashable_type_hint() -> None:
    """Unhashable Pydantic-compatible hints are built uncached rather than rejected."""
    converter = DataConverter(payload_converter_class=PydanticAIPayloadConverter)
    payloads = await converter.encode([1])
    unhashable_hint = Annotated[int, []]

    with patch.object(
        temporal_payload_converter, 'TypeAdapter', wraps=temporal_payload_converter.TypeAdapter
    ) as type_adapter:
        assert await converter.decode(payloads, [unhashable_hint]) == [1]  # pyright: ignore[reportArgumentType]
        assert await converter.decode(payloads, [unhashable_hint]) == [1]  # pyright: ignore[reportArgumentType]

    assert type_adapter.call_count == 2


@pytest.mark.parametrize(
    'value',
    [
        {'metadata': {'reason': 'review'}, 'kind': 'approval_required'},
        {'metadata': {'reason': 'later'}, 'kind': 'call_deferred'},
        {'message': 'retry this', 'kind': 'model_retry'},
        {'result': 'result', 'kind': 'tool_return'},
        {'result': {'kind': 'tool-return', 'value': 1}, 'kind': 'tool_content_result'},
        {'message': 'failed', 'kind': 'tool_failed'},
    ],
)
async def test_pydantic_ai_payload_converter_matches_stock_for_call_tool_result(value: dict[str, Any]) -> None:
    """Every `CallToolResult` variant round-trips identically through stock and memoized converters."""
    stock_payloads = await pydantic_data_converter.encode([value])
    stock_result = await pydantic_data_converter.decode(stock_payloads, [CallToolResult])  # pyright: ignore[reportArgumentType]

    converter = DataConverter(payload_converter_class=PydanticAIPayloadConverter)
    memoized_payloads = await converter.encode([value])
    memoized_result = await converter.decode(memoized_payloads, [CallToolResult])  # pyright: ignore[reportArgumentType]

    assert memoized_payloads == stock_payloads
    assert memoized_result == stock_result


def test_pydantic_ai_plugin_no_converter_uses_memoizing_converter() -> None:
    """When no converter is provided, `PydanticAIPlugin` uses its memoizing converter."""
    plugin = PydanticAIPlugin()
    # Create a minimal config without data_converter
    config: dict[str, Any] = {}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter


def test_pydantic_ai_plugin_passes_pydantic_monty_through_sandbox() -> None:
    runner = SandboxedWorkflowRunner()
    config: dict[str, Any] = {'workflow_runner': runner}

    result = PydanticAIPlugin().configure_worker(config)  # type: ignore[arg-type]

    assert 'workflow_runner' in result
    configured_runner = result['workflow_runner']
    assert isinstance(configured_runner, SandboxedWorkflowRunner)
    assert 'pydantic_monty' in configured_runner.restrictions.passthrough_modules


async def test_pydantic_ai_plugin_runs_workflow_in_sandbox(temporal_env: WorkflowEnvironment) -> None:
    client = await Client.connect(f'localhost:{TEMPORAL_PORT}')
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[PydanticAIPluginSandboxWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=SandboxedWorkflowRunner(),
    ):
        result = await client.execute_workflow(
            PydanticAIPluginSandboxWorkflow.run,
            id=f'{PydanticAIPluginSandboxWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )

    assert result == 'sandboxed'


def test_pydantic_ai_plugin_with_stock_pydantic_payload_converter_upgraded() -> None:
    """The exact stock `PydanticPayloadConverter` is upgraded to the memoizing converter."""
    plugin = PydanticAIPlugin()
    codec = MockPayloadCodec()
    converter = DataConverter(payload_converter_class=PydanticPayloadConverter, payload_codec=codec)
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is not converter
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter
    assert result['data_converter'].payload_codec is codec
    assert result['data_converter'].failure_converter_class is converter.failure_converter_class


def test_pydantic_ai_plugin_with_custom_pydantic_subclass_unchanged() -> None:
    """When converter uses a subclass of PydanticPayloadConverter, return it unchanged (no warning)."""
    plugin = PydanticAIPlugin()
    converter = DataConverter(payload_converter_class=CustomPydanticPayloadConverter)
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is converter
    assert result['data_converter'].payload_converter_class is CustomPydanticPayloadConverter


def test_pydantic_ai_plugin_with_default_payload_converter_replaced() -> None:
    """When converter uses DefaultPayloadConverter, replace payload_converter_class without warning."""
    plugin = PydanticAIPlugin()
    converter = DataConverter(payload_converter_class=DefaultPayloadConverter)
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is not converter
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter


def test_pydantic_ai_plugin_preserves_custom_payload_codec() -> None:
    """When converter has a custom payload_codec, preserve it while replacing payload_converter_class."""
    plugin = PydanticAIPlugin()
    codec = MockPayloadCodec()
    converter = DataConverter(
        payload_converter_class=DefaultPayloadConverter,
        payload_codec=codec,
    )
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'] is not converter
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter
    assert result['data_converter'].payload_codec is codec
    assert result['data_converter'].failure_converter_class is converter.failure_converter_class


def test_pydantic_ai_plugin_preserves_external_storage() -> None:
    """A user's Temporal external storage config survives the payload converter swap.

    The Temporal docs point large-payload users at `external_storage`, so this has to keep working.
    """

    class MockStorageDriver(StorageDriver):
        def name(self) -> str:
            return 'mock'

        async def store(self, context: Any, payloads: Any) -> Any:
            raise NotImplementedError

        async def retrieve(self, context: Any, claims: Any) -> Any:
            raise NotImplementedError

    external_storage = ExternalStorage(drivers=[MockStorageDriver()])
    plugin = PydanticAIPlugin()
    converter = DataConverter(
        payload_converter_class=DefaultPayloadConverter,
        external_storage=external_storage,
    )
    config: dict[str, Any] = {'data_converter': converter}
    result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter
    assert result['data_converter'].external_storage is external_storage


def test_pydantic_ai_plugin_with_non_pydantic_converter_warns() -> None:
    """When converter uses a non-Pydantic payload converter, warn and replace."""
    plugin = PydanticAIPlugin()
    converter = DataConverter(payload_converter_class=CustomPayloadConverter)
    config: dict[str, Any] = {'data_converter': converter}
    with pytest.warns(
        UserWarning,
        match='A non-Pydantic Temporal payload converter was used which has been replaced with '
        '`PydanticAIPayloadConverter`',
    ):
        result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter


def test_pydantic_ai_plugin_with_non_pydantic_converter_preserves_codec() -> None:
    """When converter uses a non-Pydantic payload converter with custom codec, warn but preserve codec."""
    plugin = PydanticAIPlugin()
    codec = MockPayloadCodec()
    converter = DataConverter(
        payload_converter_class=CustomPayloadConverter,
        payload_codec=codec,
    )
    config: dict[str, Any] = {'data_converter': converter}
    with pytest.warns(UserWarning):
        result = plugin.configure_client(config)  # type: ignore[arg-type]
    assert result['data_converter'].payload_converter_class is PydanticAIPayloadConverter
    assert result['data_converter'].payload_codec is codec


def test_temporal_model_profile_with_no_provider_prefix() -> None:
    """Test TemporalModel uses DEFAULT_PROFILE when model string has no inferable provider."""

    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__no_provider_prefix',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # A model string without a provider prefix that can't be inferred returns DEFAULT_PROFILE
    with temporal_model.using_model('some-random-model'):
        assert temporal_model.profile is DEFAULT_PROFILE


def test_temporal_model_profile_with_unknown_provider() -> None:
    """Test TemporalModel uses DEFAULT_PROFILE when provider is unknown."""

    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__unknown_provider',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    # An unknown provider should return DEFAULT_PROFILE
    with temporal_model.using_model('unknown-provider:some-model'):
        assert temporal_model.profile is DEFAULT_PROFILE


@pytest.mark.parametrize(
    'model_id',
    [
        'openai:gpt-5',
        'gateway/openai:gpt-5',
    ],
)
def test_temporal_model_prepare_request_with_unregistered_model_string(model_id: str) -> None:
    """Test prepare_request uses inferred profile for unregistered model strings.

    Verifies that the OpenAI json_schema_transformer is applied to function tool
    schemas (adding additionalProperties: false) when using an OpenAI model string,
    both directly and via gateway/.
    """
    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__prepare_request_unregistered',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    tool_def = ToolDefinition(
        name='my_tool',
        description='A test tool',
        parameters_json_schema={
            'type': 'object',
            'properties': {'x': {'type': 'integer'}},
            'required': ['x'],
        },
    )

    model_request_params = ModelRequestParameters(
        function_tools=[tool_def],
        native_tools=[],
        output_mode='text',
        allow_text_output=True,
        output_tools=[],
        output_object=None,
    )

    # With an unregistered model string, prepare_request should use the inferred
    # profile's json_schema_transformer (OpenAI adds additionalProperties: false)
    with temporal_model.using_model(model_id):
        _, params = temporal_model.prepare_request(None, model_request_params)
        assert params.output_mode == 'text'
        assert len(params.function_tools) == 1
        assert params.function_tools[0].parameters_json_schema['additionalProperties'] is False


def test_temporal_model_prepare_messages_with_unregistered_model_string() -> None:
    """`prepare_messages` defers preparation for unregistered model strings.

    The temporal wrapper has no concrete `Model` instance to delegate to in the workflow,
    so the activity performs the single authoritative pass after resolving it.
    """
    default_model = TestModel(custom_output_text='default')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__prepare_messages_unregistered',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
    )

    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hi')])]
    with temporal_model.using_model('openai:gpt-5'):
        prepared = temporal_model.prepare_messages(messages)
    assert prepared == messages


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_temporal_model_runtime_provider_prepares_messages_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistered model string is prepared only after its concrete profile is known."""

    def provider_factory(_ctx: RunContext[object], _provider_name: str) -> AnthropicProvider:
        return AnthropicProvider(api_key='test-api-key')

    temporal_model = TemporalModel(
        TestModel(),
        activity_name_prefix='test__runtime_provider_prepare_messages_once',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=object,
        provider_factory=provider_factory,
    )
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart('leading'), UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('answer')]),
        ModelRequest(parts=[SystemPromptPart('mid'), UserPromptPart('second')]),
    ]

    def infer_unsupported_profile(_model_id: str) -> ModelProfile:
        return DEFAULT_PROFILE

    monkeypatch.setattr('pydantic_ai.durable_exec.temporal._model.infer_model_profile', infer_unsupported_profile)
    with temporal_model.using_model('anthropic:claude-opus-5'):
        prepared_messages = temporal_model.prepare_messages(messages)

    received_messages: list[list[ModelMessage]] = []

    async def request(
        _model: AnthropicModel,
        activity_messages: list[ModelMessage],
        _model_settings: ModelSettings | None,
        _model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        received_messages.append(activity_messages)
        return ModelResponse(parts=[TextPart('done')])

    monkeypatch.setattr(AnthropicModel, 'request', request)
    deps = object()
    ctx = RunContext[object](deps=deps, model=TestModel(), usage=RunUsage(), run_id='runtime-provider')
    params = _RequestParams(
        messages=prepared_messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
        model_id='anthropic:claude-opus-5',
    )
    await ActivityEnvironment().run(temporal_model.request_activity, params, deps)

    assert received_messages == [messages]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_temporal_model_runtime_provider_reprepares_messages(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """The activity applies the concrete transport profile before sending serialized history."""
    foundry_client = anthropic.AsyncAnthropicFoundry(
        resource='test-resource',
        api_key='test-api-key',
    )

    def provider_factory(_ctx: RunContext[object], _provider_name: str) -> AnthropicProvider:
        return AnthropicProvider(anthropic_client=foundry_client)

    async def event_stream_handler(
        _ctx: RunContext[object], _streamed_response: AsyncIterable[AgentStreamEvent]
    ) -> None:
        pass

    temporal_model = TemporalModel(
        TestModel(),
        activity_name_prefix=f'test__runtime_provider_reprepare_{stream}',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=object,
        provider_factory=provider_factory,
        event_stream_handler=event_stream_handler,
    )
    messages: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart('leading'), UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('answer')]),
        ModelRequest(parts=[SystemPromptPart('mid'), UserPromptPart('second')]),
    ]
    with temporal_model.using_model('anthropic:claude-opus-5'):
        prepared_messages = temporal_model.prepare_messages(messages)
    assert prepared_messages == messages

    rendered_requests: list[dict[str, Any]] = []

    async def render(
        model: AnthropicModel,
        activity_messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        assert model_settings is None
        anthropic_settings: AnthropicModelSettings = {}
        system_prompt, anthropic_messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
            activity_messages,
            model_request_parameters,
            anthropic_settings,
        )
        rendered_requests.append({'system': system_prompt, 'messages': anthropic_messages})
        return ModelResponse(parts=[TextPart('done')])

    if stream:

        @asynccontextmanager
        async def request_stream(
            model: AnthropicModel,
            activity_messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[object] | None = None,
        ) -> AsyncGenerator[CompletedStreamedResponse]:
            del run_context
            response = await render(model, activity_messages, model_settings, model_request_parameters)
            yield CompletedStreamedResponse(response, model_request_parameters=model_request_parameters)

        monkeypatch.setattr(AnthropicModel, 'request_stream', request_stream)
    else:
        monkeypatch.setattr(AnthropicModel, 'request', render)

    deps = object()
    ctx = RunContext[object](deps=deps, model=TestModel(), usage=RunUsage(), run_id='runtime-provider')
    params = _RequestParams(
        messages=prepared_messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
        model_id='anthropic:claude-opus-5',
    )
    if stream:
        await ActivityEnvironment().run(
            temporal_model.request_stream_activity,
            params,
            deps,  # pyright: ignore[reportArgumentType]
        )
    else:
        await ActivityEnvironment().run(temporal_model.request_activity, params, deps)

    assert rendered_requests == snapshot(
        [
            {
                'system': 'leading',
                'messages': [
                    {'role': 'user', 'content': [{'text': 'first', 'type': 'text'}]},
                    {'role': 'assistant', 'content': [{'text': 'answer', 'type': 'text'}]},
                    {
                        'role': 'user',
                        'content': [
                            {'text': '<system>mid</system>', 'type': 'text'},
                            {'text': 'second', 'type': 'text'},
                        ],
                    },
                ],
            }
        ]
    )


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_temporal_model_runtime_provider_preserves_unmodified_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The activity forwards history unchanged when the concrete model has nothing to rewrite."""

    def provider_factory(_ctx: RunContext[object], _provider_name: str) -> AnthropicProvider:
        return AnthropicProvider(api_key='test-api-key')

    temporal_model = TemporalModel(
        TestModel(),
        activity_name_prefix='test__runtime_provider_preserve_messages',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=object,
        provider_factory=provider_factory,
    )
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('hello')])]
    received_messages: list[list[ModelMessage]] = []

    async def request(
        _model: AnthropicModel,
        activity_messages: list[ModelMessage],
        _model_settings: ModelSettings | None,
        _model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        received_messages.append(activity_messages)
        return ModelResponse(parts=[TextPart('done')])

    monkeypatch.setattr(AnthropicModel, 'request', request)

    deps = object()
    ctx = RunContext[object](deps=deps, model=TestModel(), usage=RunUsage(), run_id='runtime-provider')
    params = _RequestParams(
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
        model_id='anthropic:claude-opus-5',
    )
    await ActivityEnvironment().run(temporal_model.request_activity, params, deps)
    assert received_messages
    assert received_messages[0] is messages


def test_temporal_model_customize_request_parameters_with_registered_model() -> None:
    """Test customize_request_parameters delegates to the currently active registered model."""

    class _CustomizingTestModel(TestModel):
        def customize_request_parameters(
            self, model_request_parameters: ModelRequestParameters
        ) -> ModelRequestParameters:
            return ModelRequestParameters(output_mode='tool', allow_text_output=False)

    default_model = TestModel(custom_output_text='default')
    alternate_model = _CustomizingTestModel(custom_output_text='alternate')
    temporal_model = TemporalModel(
        default_model,
        activity_name_prefix='test__customize_registered',
        activity_config={'start_to_close_timeout': timedelta(seconds=60)},
        deps_type=type(None),
        models={'alternate': alternate_model},
    )

    with temporal_model.using_model('alternate'):
        customized = temporal_model.customize_request_parameters(ModelRequestParameters())

    assert customized.output_mode == 'tool'
    assert customized.allow_text_output is False


# Tests for BinaryContent and DocumentUrl serialization in Temporal
# This is a regression test for #3702 (BinaryContent) and verifies that FileUrl
# instances (like DocumentUrl) with explicit media_type are properly preserved.


multimodal_content_agent = Agent(TestModel(), name='multimodal_content_agent')


@multimodal_content_agent.tool
def get_multimodal_content(ctx: RunContext) -> list[str | MultiModalContent]:
    """Return a list with text, BinaryContent, and DocumentUrl."""
    return [
        'test',
        BinaryImage(data=b'\x89PNG', media_type='image/png'),
        # URL doesn't hint at media type, so media_type must be specified explicitly
        DocumentUrl(url='https://example.com/doc/12345', media_type='application/pdf'),
    ]


multimodal_content_temporal_agent = TemporalAgent(multimodal_content_agent, activity_config=BASE_ACTIVITY_CONFIG)  # pyright: ignore[reportDeprecated]


@workflow.defn
class MultiModalContentWorkflow:
    @workflow.run
    async def run(self, prompt: list[UserContent]) -> list[ModelMessage]:
        result = await multimodal_content_temporal_agent.run(prompt)
        return result.all_messages()


async def test_multimodal_content_serialization_in_workflow(client: Client):
    """Test that BinaryContent and DocumentUrl survive Temporal serialization.

    This tests both:
    1. Passing BinaryContent and DocumentUrl as input to agent.run (workflow→activity)
    2. Returning BinaryContent and DocumentUrl from a tool (activity→workflow)

    BinaryContent is serialized with base64 encoding. DocumentUrl requires explicit
    media_type since it cannot be inferred from the URL.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModalContentWorkflow],
        plugins=[AgentPlugin(multimodal_content_temporal_agent)],
    ):
        # Pass both BinaryContent and DocumentUrl as input
        prompt: list[str | MultiModalContent] = [
            'Process these files and call the tool',
            BinaryImage(data=b'\x89PNG', media_type='image/png'),
            DocumentUrl(url='https://example.com/doc/12345', media_type='application/pdf'),
        ]
        messages = await client.execute_workflow(
            MultiModalContentWorkflow.run,
            args=[prompt],
            id='test_multimodal_content_serialization',
            task_queue=TASK_QUEUE,
        )
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content=[
                                'Process these files and call the tool',
                                BinaryImage(data=b'\x89PNG', media_type='image/png', identifier='4effda'),
                                DocumentUrl(
                                    url='https://example.com/doc/12345',
                                    _media_type='application/pdf',
                                    _identifier='eb8998',
                                ),
                            ],
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
                            tool_name='get_multimodal_content',
                            args={},
                            tool_call_id='pyd_ai_tool_call_id__get_multimodal_content',
                        )
                    ],
                    usage=RequestUsage(input_tokens=61, output_tokens=2),
                    model_name='test',
                    timestamp=IsDatetime(),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_multimodal_content',
                            content=[
                                'test',
                                BinaryImage(data=b'\x89PNG', media_type='image/png', identifier='4effda'),
                                DocumentUrl(
                                    url='https://example.com/doc/12345',
                                    _media_type='application/pdf',
                                    _identifier='eb8998',
                                ),
                            ],
                            tool_call_id='pyd_ai_tool_call_id__get_multimodal_content',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='{"get_multimodal_content":["test",{"data":"iVBORw==","media_type":"image/png","vendor_metadata":null,"kind":"binary","identifier":"4effda"},{"url":"https://example.com/doc/12345","force_download":false,"vendor_metadata":null,"kind":"document-url","media_type":"application/pdf","identifier":"eb8998"}]}'
                        )
                    ],
                    usage=RequestUsage(input_tokens=62, output_tokens=34),
                    model_name='test',
                    timestamp=IsDatetime(),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

        # Explicitly verify that media_type is preserved through serialization for both
        # BinaryContent and DocumentUrl. This is important because _media_type has compare=False
        # on DocumentUrl, so the snapshot comparison doesn't actually verify it. The media_type
        # cannot be inferred from the URL, so if serialization loses it, accessing media_type
        # would raise an error.
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
        # Should have 4 items: 2 from user input, 2 from tool return.
        # The image `BinaryContent` round-trips as `BinaryImage`: narrowing is applied during
        # `MultiModalContent` validation, so it now survives the Temporal serialization boundary too.
        assert media_types == [
            ('BinaryImage', 'image/png'),
            ('DocumentUrl', 'application/pdf'),
            ('BinaryImage', 'image/png'),
            ('DocumentUrl', 'application/pdf'),
        ]


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


async def test_text_content_serialization_in_workflow(client: Client):
    """Test that TextContent is properly serialized in Temporal."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MultiModalContentWorkflow],
        plugins=[AgentPlugin(multimodal_content_temporal_agent)],
    ):
        prompt = [
            'This is a text content test',
            TextContent(content='This should be preserved as TextContent', metadata={'preserved': True}),
        ]
        messages = await client.execute_workflow(
            MultiModalContentWorkflow.run,
            args=[prompt],
            id='test_text_content_serialization',
            task_queue=TASK_QUEUE,
        )
        assert messages[0] == snapshot(
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'This is a text content test',
                            TextContent(
                                content='This should be preserved as TextContent', metadata={'preserved': True}
                            ),
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        )


# ==========================================
# TemporalDurability capability tests
# ==========================================


def _durability_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Simple model function for durability tests that echoes the last user prompt."""
    # The first message always carries the prompt and its first part is always the `UserPromptPart`, so none branch.
    for msg in reversed(messages):  # pragma: no branch
        for part in msg.parts:  # pragma: no branch
            if isinstance(part, UserPromptPart):  # pragma: no branch
                return ModelResponse(parts=[TextPart(content=f'Echo: {part.content}')])
    return ModelResponse(parts=[TextPart(content='no prompt')])  # pragma: no cover


_durability_fn_model = FunctionModel(_durability_model_fn)

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
    # 3 base activities (request, request_stream, cancel) + 1 for the agent's <agent> FunctionToolset
    assert len(bound.temporal_activities) == 4


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
    # 3 base activities + 1 for <agent> FunctionToolset + 1 for test_toolset
    assert len(bound.temporal_activities) == 5


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
    # 3 base activities + 1 for <agent> FunctionToolset + 1 (not 2) for the shared toolset
    assert len(bound.temporal_activities) == 5


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


def test_temporal_agent_retry_policy_non_retryable_errors():
    """The deprecated wrapper normalizes its base activity retry policy with `with_non_retryable_errors`."""
    temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
        Agent(TestModel(), name='retry_policy_probe_agent'),
    )

    retry_policy = temporal_agent.activity_config.get('retry_policy')
    assert retry_policy is not None
    assert retry_policy.non_retryable_error_types == [
        'UserError',
        'PydanticUserError',
        'UnexpectedModelBehavior',
        'FallbackExceptionGroup',
        'PayloadsTooLarge',
        'PayloadSizeError',
    ]


def test_temporal_agent_custom_retry_policy_keeps_non_retryable_errors():
    """A caller-supplied `retry_policy` in a merged config must not drop the non-retryable errors.

    `TemporalAgent`'s `model_activity_config` (and per-toolset configs) merge over the normalized
    base config, and a `retry_policy` in the override replaces the base policy wholesale — without
    re-normalization an oversized payload would retry the whole (paid) model request forever.
    """
    toolset = FunctionToolset[None](id='merge_probe_toolset')

    async def my_tool() -> str:
        return 'ok'  # pragma: no cover

    toolset.add_function(my_tool)

    temporal_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
        Agent(TestModel(), name='retry_policy_merge_probe_agent', deps_type=type(None), toolsets=[toolset]),
        model_activity_config=ActivityConfig(retry_policy=RetryPolicy(non_retryable_error_types=['ModelError'])),
        toolset_activity_config={
            'merge_probe_toolset': ActivityConfig(retry_policy=RetryPolicy(non_retryable_error_types=['ToolError'])),
        },
    )

    model_retry = temporal_agent._temporal_model.activity_config.get('retry_policy')  # pyright: ignore[reportPrivateUsage]
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


_pydantic_ai_agents_durable = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)
_pydantic_ai_agents_agent = Agent(
    _durability_fn_model,
    name='pydantic_ai_agents_attr_test',
    capabilities=[_pydantic_ai_agents_durable],
)


@workflow.defn
class _BareAgentWorkflowViaAttribute:
    __pydantic_ai_agents__ = [_pydantic_ai_agents_agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _pydantic_ai_agents_agent.run(prompt)
        return result.output


async def test_pydantic_ai_plugin_discovers_bare_agent_with_durability(client: Client):
    """`PydanticAIPlugin` registers activities from a bare `AbstractAgent` listed in `__pydantic_ai_agents__`."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[_BareAgentWorkflowViaAttribute],
    ):
        output = await client.execute_workflow(
            _BareAgentWorkflowViaAttribute.run,
            args=['Discovered'],
            id=_BareAgentWorkflowViaAttribute.__name__,
            task_queue=TASK_QUEUE,
        )
        assert output == 'Echo: Discovered'


_missing_cap_agent = Agent(_durability_fn_model, name='no_cap_in_attr')


@workflow.defn
class _MissingCapWorkflow:
    __pydantic_ai_agents__ = [_missing_cap_agent]

    # `configure_worker` rejects before this can execute.
    @workflow.run
    async def run(self, prompt: str) -> str:  # pragma: no cover
        result = await _missing_cap_agent.run(prompt)
        return result.output


async def test_pydantic_ai_plugin_rejects_bare_agent_without_durability(client: Client):
    """`PydanticAIPlugin` raises a clear error when an agent in `__pydantic_ai_agents__` lacks `TemporalDurability`."""
    with pytest.raises(UserError, match='no `TemporalDurability` capability'):
        async with Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[_MissingCapWorkflow],
        ):
            # The error is raised before reaching here.
            pass  # pragma: no cover


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


# --- Streaming in workflow (event_stream_handler) ---


async def _stream_model_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'Stream'
    yield 'ed '
    yield 'response'


_stream_fn_model = FunctionModel(_durability_model_fn, stream_function=_stream_model_fn)

_stream_events_collected: list[AgentStreamEvent] = []
_stream_model_events_in_activity: list[bool] = []


async def _durability_event_stream_handler(
    ctx: RunContext[object],
    stream: AsyncIterable[AgentStreamEvent],
) -> None:
    async for event in stream:
        if isinstance(event, (PartStartEvent, PartDeltaEvent)):
            _stream_model_events_in_activity.append(activity.in_activity())
        _stream_events_collected.append(event)


_stream_durability = TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)
_stream_durable_agent = Agent(
    _stream_fn_model,
    name='durability_stream_agent',
    capabilities=[ProcessEventStream(_durability_event_stream_handler), _stream_durability],
)


@workflow.defn
class StreamDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> tuple[str, list[bool]]:
        result = await _stream_durable_agent.run(prompt)
        return result.output, _stream_model_events_in_activity


_durability_handler_events: list[tuple[AgentStreamEvent, bool]] = []


async def _durability_handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        _durability_handler_events.append((event, activity.in_activity()))


async def _durability_handler_tool() -> str:
    return 'handled'


async def _durability_reveal_tool() -> ToolReturn[str]:
    return ToolReturn(return_value='handled', tools=['hidden_tool'])


_handler_durability = TemporalDurability(
    activity_config=BASE_ACTIVITY_CONFIG,
    event_stream_handler=_durability_handler,
)
_handler_durable_agent = Agent(
    TestModel(),
    name='durability_handler_agent',
    tools=[_durability_handler_tool],
    capabilities=[_handler_durability],
)


@workflow.defn
class HandlerDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _handler_durable_agent.run(prompt)
        return result.output


async def test_temporal_durability_event_stream_handler(client: Client) -> None:
    _durability_handler_events.clear()
    bound = TemporalDurability.from_agent(_handler_durable_agent)
    assert bound is not None
    activity_names = [
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in bound.temporal_activities
    ]
    assert 'agent__durability_handler_agent__event_stream_handler' in activity_names

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[HandlerDurableAgentWorkflow],
        plugins=[AgentPlugin(_handler_durable_agent)],
    ):
        await client.execute_workflow(
            HandlerDurableAgentWorkflow.run,
            args=['Hello'],
            id=HandlerDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    events = [event for event, _ in _durability_handler_events]
    assert events
    assert all(in_activity for _, in_activity in _durability_handler_events)
    assert sum(isinstance(event, FunctionToolCallEvent) for event in events) == 1
    assert sum(isinstance(event, FunctionToolResultEvent) for event in events) == 1
    assert any(isinstance(event, PartStartEvent) for event in events)
    assert any(isinstance(event, FinalResultEvent) for event in events)


_iter_handler_events: list[tuple[AgentStreamEvent, bool]] = []


async def _iter_handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        _iter_handler_events.append((event, activity.in_activity()))


_iter_handler_durability = TemporalDurability(
    activity_config=BASE_ACTIVITY_CONFIG,
    event_stream_handler=_iter_handler,
)
_iter_handler_durable_agent = Agent(
    TestModel(),
    name='durability_iter_handler_agent',
    tools=[_durability_handler_tool],
    capabilities=[_iter_handler_durability],
)


@workflow.defn
class IterHandlerDurableAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        async with _iter_handler_durable_agent.iter(prompt) as agent_run:
            async for _node in agent_run:
                pass
        assert agent_run.result is not None
        return str(agent_run.result.output)


async def test_temporal_durability_iter_in_workflow_event_stream_handler(client: Client) -> None:
    """`agent.iter()` inside a workflow delivers events to the durability capability's handler.

    Only the deprecated `TemporalAgent` wrapper blocks `iter()` inside a workflow; the
    `TemporalDurability` capability allows it, and used to skip the handler entirely because
    `wrap_run_event_stream` was applied by `run()`/`run_stream()` rather than by the node stream
    primitives. Delivery stays inside the model-request activity, matching the `run()` path.
    """
    _iter_handler_events.clear()

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[IterHandlerDurableAgentWorkflow],
        plugins=[AgentPlugin(_iter_handler_durable_agent)],
    ):
        await client.execute_workflow(
            IterHandlerDurableAgentWorkflow.run,
            args=['Hello'],
            id=IterHandlerDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    events = [event for event, _ in _iter_handler_events]
    assert events
    assert all(in_activity for _, in_activity in _iter_handler_events)
    assert sum(isinstance(event, FunctionToolCallEvent) for event in events) == 1
    assert sum(isinstance(event, FunctionToolResultEvent) for event in events) == 1
    assert any(isinstance(event, PartStartEvent) for event in events)
    assert any(isinstance(event, FinalResultEvent) for event in events)


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


async def test_temporal_durability_event_stream_handler_outside_workflow() -> None:
    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    durability = TemporalDurability(event_stream_handler=handler)
    agent = Agent(TestModel(custom_output_text='done'), name='outside_handler', capabilities=[durability])
    await agent.run('Hello')
    assert any(isinstance(event, PartStartEvent) for event in events)


def test_temporal_durability_without_handler_does_not_wrap_event_stream() -> None:
    durability = TemporalDurability()
    assert durability.has_wrap_run_event_stream is False


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


async def test_stream_activity_payload_decodes_both_recorded_shapes() -> None:
    """The stream-activity result union decodes both recorded wire shapes unambiguously.

    A `TemporalDurability` history (v2.14+) records a `StreamedActivityResult`; a legacy
    `TemporalAgent` history recorded the bare `ModelResponse`. Replay of either kind of
    in-flight workflow decodes the recorded payload through `_StreamedActivityPayload`.
    """
    response = {'parts': [{'content': 'streamed', 'part_kind': 'text'}], 'kind': 'response'}
    event = {'index': 0, 'part': {'content': 'streamed', 'part_kind': 'text'}, 'event_kind': 'part_start'}
    payloads = await pydantic_data_converter.encode([{'response': response, 'events': [event]}, response])

    hints = [_StreamedActivityPayload, _StreamedActivityPayload]
    current_shape, legacy_shape = await pydantic_data_converter.decode(payloads, hints)  # pyright: ignore[reportArgumentType]

    assert isinstance(current_shape, StreamedActivityResult)
    assert current_shape.response.parts == [TextPart(content='streamed')]
    assert current_shape.events == [PartStartEvent(index=0, part=TextPart(content='streamed'))]
    assert isinstance(legacy_shape, ModelResponse)
    assert legacy_shape.parts == [TextPart(content='streamed')]


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


_buffered_stream_agent = Agent(
    TestModel(custom_output_text='hello world'),
    name='durability_buffered_streams',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class BufferedStreamDurableAgentWorkflow:
    @workflow.run
    async def run(self) -> tuple[list[str], str, list[str]]:
        async with _buffered_stream_agent.run_stream('Hello') as stream:
            chunks = [chunk async for chunk in stream.stream_text(debounce_by=None)]
            output = await stream.get_output()

        async with _buffered_stream_agent.run_stream_events('Hello') as event_stream:
            events = [event async for event in event_stream]
        deltas = [
            event.delta.content_delta
            for event in events
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta)
        ]
        return chunks, output, deltas


async def test_temporal_durability_buffers_caller_streams(client: Client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[BufferedStreamDurableAgentWorkflow],
        plugins=[AgentPlugin(_buffered_stream_agent)],
    ):
        result = await client.execute_workflow(
            BufferedStreamDurableAgentWorkflow.run,
            id=BufferedStreamDurableAgentWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    assert result == (['hello ', 'hello world'], 'hello world', ['hello ', 'world'])


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
                'FunctionToolset cannot be passed to `run(toolsets=...)` at runtime with Temporal, because '
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


# --- Continuation chains (suspended → complete) run one activity per segment ---
#
# When a model suspends a turn (Anthropic `pause_turn`, OpenAI background mode), the
# continuation loop in the innermost `model_request`/`model_request_stream` helpers runs
# workflow-side under `TemporalDurability`, dispatching each segment through its own
# model-request activity, so a failed segment retries alone and the suspended response is
# checkpointed in workflow history between segments. These tests use a scripted model (no
# cassettes: `FunctionModel` can't emit suspended streaming segments, and VCR matchers
# wouldn't pin the chain shape).


def _workflow_failure_cause(exc: WorkflowFailureError) -> ApplicationError:
    """The innermost `ApplicationError` of a workflow failure (walking through `ActivityError`)."""
    cause: BaseException | None = exc.__cause__
    while cause is not None and not isinstance(cause, ApplicationError):
        cause = cause.__cause__
    assert isinstance(cause, ApplicationError), f'expected ApplicationError in cause chain of {exc!r}'
    return cause


def _scheduled_activity_count(history: WorkflowHistory) -> int:
    return len([e for e in history.events if e.HasField('activity_task_scheduled_event_attributes')])


_workflow_cancel_agent = Agent(
    TestModel(custom_output_text='finished'),
    name='workflow_cancel_agent',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class WorkflowCancelAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        try:
            async with _workflow_cancel_agent.iter(prompt) as agent_run:
                async for node in agent_run:
                    if Agent.is_call_tools_node(node):
                        agent_run.cancel()
        except RunCancelled as exc:
            return f'cancelled:{bool(exc.all_messages())}'
        return 'completed'  # pragma: no cover


async def test_workflow_agent_run_cancel_is_application_outcome_and_replays(client: Client) -> None:
    """Workflow-side first-party cancellation completes normally and remains replay-deterministic."""
    workflow_id = f'{WorkflowCancelAgentWorkflow.__name__}-{uuid.uuid4()}'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[WorkflowCancelAgentWorkflow],
        plugins=[AgentPlugin(_workflow_cancel_agent)],
    ):
        output = await client.execute_workflow(
            WorkflowCancelAgentWorkflow.run,
            args=['cancel after the first model response'],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    assert output == 'cancelled:True'
    await Replayer(
        workflows=[WorkflowCancelAgentWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(history)


def _cancel_from_activity(ctx: RunContext[None]) -> str:
    ctx.cancel()
    return 'cancelled'  # pragma: no cover


_activity_cancel_agent = Agent(
    TestModel(call_tools=['_cancel_from_activity']),
    name='activity_cancel_agent',
    deps_type=type(None),
    tools=[_cancel_from_activity],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class ActivityCancelAgentWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _activity_cancel_agent.run(prompt)).output


async def test_run_context_cancel_in_activity_surfaces_user_error(client: Client) -> None:
    """An activity cannot cancel its workflow-side run and fails clearly instead of hanging."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ActivityCancelAgentWorkflow],
        plugins=[AgentPlugin(_activity_cancel_agent)],
    ):
        with pytest.raises(WorkflowFailureError) as exc_info:
            await client.execute_workflow(
                ActivityCancelAgentWorkflow.run,
                args=['call the cancellation tool'],
                id=f'{ActivityCancelAgentWorkflow.__name__}-{uuid.uuid4()}',
                task_queue=TASK_QUEUE,
            )

    cause = _workflow_failure_cause(exc_info.value)
    assert cause.type == UserError.__name__
    assert cause.message == snapshot(
        '`cancel` is only available during an agent run (from tools, event stream handlers, or capability hooks) '
        'in the same process as the run itself. This `RunContext` has no run to cancel.'
    )


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


_heartbeat_function_toolset = FunctionToolset[None](tools=[heartbeat_probe_tool], id='hb_tools')
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
    return FunctionToolset[None](tools=[heartbeat_probe_tool], id='hb_dynamic_inner')


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


_heartbeat_agent = Agent(
    _HeartbeatProbeModel(_heartbeat_model_fn, stream_function=_heartbeat_stream_model_fn),
    name='heartbeat_probe_agent',
    deps_type=type(None),
    tools=[heartbeat_probe_agent_tool],
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
            f'{prefix}__model_request': [request_params, None],
            f'{prefix}__model_request_stream': [request_params, None],
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


# --- Usage mutated inside an activity ---


def _usage_delegation_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call the `delegate` tool once, then finish."""
    for msg in reversed(messages):
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(
                    parts=[TextPart(content=f'Delegate said: {part.content}')],
                    usage=RequestUsage(input_tokens=5, output_tokens=1),
                )
    return ModelResponse(
        parts=[ToolCallPart(tool_name='delegate', args='{}')],
        usage=RequestUsage(input_tokens=5, output_tokens=1),
    )


_usage_delegate_agent = Agent(
    FunctionModel(
        lambda messages, info: ModelResponse(
            parts=[TextPart(content='delegated')],
            usage=RequestUsage(input_tokens=100, output_tokens=10),
        )
    ),
    name='usage_delegate_agent',
)

usage_delegation_agent = Agent(
    FunctionModel(_usage_delegation_model_fn),
    name='usage_delegation_agent',
    deps_type=type(None),
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@usage_delegation_agent.tool
async def delegate(ctx: RunContext[None]) -> str:
    """Delegate to another agent, passing the parent run's usage as the docs recommend."""
    result = await _usage_delegate_agent.run('delegate this', usage=ctx.usage)
    return result.output


@workflow.defn
class UsageDelegationWorkflow:
    @workflow.run
    async def run(self) -> RunUsage:
        result = await usage_delegation_agent.run('delegate please')
        return result.usage


async def test_delegate_agent_usage_is_not_merged_back_from_activity(client: Client):
    """Pins the documented Temporal limitation: `ctx.usage` mutations inside an activity are lost.

    A tool running inside an activity gets a deserialized copy of the run's `RunUsage`, so the
    usage a delegate agent accrues through `usage=ctx.usage` never reaches the workflow-side run:
    the delegate's 100 input tokens, 10 output tokens, and its request are missing from the
    workflow result, while the same agent run in-process (below) counts them.

    See https://github.com/pydantic/pydantic-ai/issues/6886.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[UsageDelegationWorkflow],
        plugins=[AgentPlugin(usage_delegation_agent)],
    ):
        workflow_usage = await client.execute_workflow(
            UsageDelegationWorkflow.run,
            id=UsageDelegationWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )
    assert workflow_usage == snapshot(RunUsage(input_tokens=10, output_tokens=2, requests=2, tool_calls=1))

    in_process_result = await usage_delegation_agent.run('delegate please')
    assert in_process_result.usage == snapshot(RunUsage(requests=3, input_tokens=110, output_tokens=12, tool_calls=1))


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
