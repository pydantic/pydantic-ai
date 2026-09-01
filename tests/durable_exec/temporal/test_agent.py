from __future__ import annotations

import asyncio
import inspect
import os
import re
import sys
import uuid
import warnings
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Any, Literal, cast
from unittest.mock import patch

import anyio
import httpx
import pytest
from pydantic import TypeAdapter

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    AgentStreamEvent,
    CancellationToken,
    ExternalToolset,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    FunctionToolset,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettings,
    OutputToolCallEvent,
    OutputToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RequestUsage,
    RetryPromptPart,
    RunContext,
    RunUsage,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai._run_context import AnchoredEvidence, get_current_run_context
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities import (
    Capability,
    ProcessHistory,
)
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    RunCancelled,
    ToolFailed,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime import (
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeSession,
)
from pydantic_ai.realtime.codec import RealtimeConnection
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition
from pydantic_ai.usage import UsageLimits

from ..._inline_snapshot import snapshot
from ...continuation_utils import ScriptedContinuationModel, scripted_response
from ...model_lifecycle_utils import LifecycleTrackingModel

try:
    from temporalio import activity, workflow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.client import Client, WorkflowFailureError
    from temporalio.common import RetryPolicy
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.exceptions import CancelledError as TemporalCancelledError
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
    from temporalio.workflow import ActivityCancellationType, ActivityConfig

    from pydantic_ai.durable_exec._utils import StreamedActivityResult
    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIWorkflow,
        TemporalAgent,  # pyright: ignore[reportDeprecated]
        TemporalDurability,
    )
    from pydantic_ai.durable_exec.temporal._activity_execution import (
        execute_activity as execute_temporal_activity,
    )
    from pydantic_ai.durable_exec.temporal._durability import (
        _CancelParams,  # pyright: ignore[reportPrivateUsage]
        _StreamedActivityPayload,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.durable_exec.temporal._function_toolset import TemporalFunctionToolset
    from pydantic_ai.durable_exec.temporal._mcp_toolset import TemporalMCPToolset
    from pydantic_ai.durable_exec.temporal._model import TemporalModel
    from pydantic_ai.durable_exec.temporal._run_context import TemporalRunContext, deserialize_run_context
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
    from pydantic_ai.mcp import MCPToolset  # noqa: F401  # pyright: ignore[reportUnusedImport]
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
    from ...conftest import IsDatetime, IsStr

    # `_shared` loads the same sandbox-sensitive modules, so import it passed-through as well.
    from ._shared import (
        BASE_ACTIVITY_CONFIG,
        TASK_QUEUE,
        Answer,
        BasicSpan,
        ComplexAgentWorkflow,
        Deps,
        GraphState,
        Response,
        _workflow_failure_cause,  # pyright: ignore[reportPrivateUsage]
        complex_agent,
        complex_temporal_agent,
        get_weather,
        http_client,
        model,
        model_settings,
        parallel_test_graph,
        simple_temporal_agent,
        workflow_activity_raises,
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
    pytest.mark.xdist_group(name='temporal-agent'),
    pytest.mark.filterwarnings(
        'ignore:`TemporalAgent` is deprecated:pydantic_ai._warnings.PydanticAIDeprecationWarning'
    ),
]


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


async def test_temporal_operation_backend_registers_novel_id_generically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.durable_exec._operation import DurableOperation, NoCacheIdentity, TypedResultCodec
    from pydantic_ai.durable_exec.temporal._operation_backend import (
        TemporalOperationBackend,
        TemporalParameterTransport,
    )

    @dataclass(frozen=True)
    class NovelOperationId:
        name: str

    @dataclass(frozen=True)
    class Params:
        value: str

    @dataclass(frozen=True)
    class WireParams:
        value: str

    class Transport(TemporalParameterTransport[Params, tuple[WireParams, None]]):
        wire_type = WireParams
        result_type = str

        def dump(self, params: Params) -> tuple[WireParams, None]:
            return WireParams(params.value), None

        def load(self, payload: tuple[WireParams, None], *, runtime: object) -> Params:
            return Params(payload[0].value)

    async def handler(params: Params) -> str:
        return f'handled:{params.value}'

    async def execute_registered_activity(
        activity: Callable[..., object], *, args: Sequence[object], **config: object
    ) -> object:
        return await cast(Any, activity)(*args)

    monkeypatch.setattr(
        'pydantic_ai.durable_exec.temporal._operation_backend.execute_activity',
        execute_registered_activity,
    )
    operation = DurableOperation(
        operation_id=cast(Any, NovelOperationId('novel')),
        handler=handler,
        parameter_transport=Transport(),
        cache_identity=NoCacheIdentity[Params](),
        result_codec=TypedResultCodec[str](str, mode='identity'),
        config_role='capability',
    )
    backend = TemporalOperationBackend(
        agent_name='novel',
        deps_type=type(None),
        model_config={},
        event_config={},
        tool_config={},
        resolve_tool_config=lambda operation_id, tool, tool_name: {},
    )
    bound, registrations = backend.register(operation, name='novel.generic', config={})

    assert await bound(Params('input')) == 'handled:input'
    assert registrations == (cast(Any, bound).registration,)


async def test_temporal_durability_accepts_legacy_cancel_activity_payload() -> None:
    """Temporal decodes old cancel payloads and manages only inferred models."""
    response = ModelResponse(parts=[TextPart(content='cancel')], model_name='test')
    params = TypeAdapter(_CancelParams).validate_python({'response': response, 'model_id': None})
    assert params == _CancelParams(response=response)
    assert params.serialized_run_context is None

    class RecordingModel(LifecycleTrackingModel):
        def __init__(self, name: str, events: list[str], *, fail: bool = False):
            super().__init__(events, fail=fail)
            self.name = name

        async def cancel_suspended_response(self, response: ModelResponse) -> None:
            self.events.append(f'cancel:{self.name}')
            if self.fail:
                raise RuntimeError('cancel failed')

    default_events: list[str] = []
    registered_events: list[str] = []
    default_model = RecordingModel('default', default_events)
    registered_model = RecordingModel('registered', registered_events)
    agent = Agent(
        default_model,
        name='legacy_cancel_payload',
        capabilities=[TemporalDurability(models={'registered': registered_model})],
    )
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    signature = inspect.signature(durability.cancel_suspended_response_activity)
    assert signature.parameters['deps'].default is None

    await durability.cancel_suspended_response_activity(_CancelParams(response=response, model_id='registered'))
    await durability.cancel_suspended_response_activity(_CancelParams(response=response))
    assert registered_events == ['cancel:registered']
    assert default_events == ['cancel:default']

    inferred_events: list[str] = []
    inferred_model = RecordingModel('inferred', inferred_events)
    with patch('pydantic_ai.durable_exec.temporal._durability.infer_model', return_value=inferred_model):
        await durability.cancel_suspended_response_activity(_CancelParams(response=response, model_id='unregistered'))
    assert inferred_events == ['enter', 'cancel:inferred', 'exit:none']

    failing_events: list[str] = []
    failing_model = RecordingModel('failing', failing_events, fail=True)
    with (
        patch('pydantic_ai.durable_exec.temporal._durability.infer_model', return_value=failing_model),
        pytest.raises(RuntimeError, match='cancel failed'),
    ):
        await durability.cancel_suspended_response_activity(_CancelParams(response=response, model_id='failing'))
    assert failing_events == ['enter', 'cancel:failing', 'exit:RuntimeError']


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


def return_settings(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(str(agent_info.model_settings))])


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


_durability_handler_events: list[tuple[AgentStreamEvent, bool]] = []


async def _durability_handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        _durability_handler_events.append((event, activity.in_activity()))


async def _durability_handler_tool() -> str:
    return 'handled'


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


_ENQUEUE_GUARD_ERROR = (
    '`ctx.enqueue()` is not supported inside a durable activity: the durable runtime replays '
    "the activity's recorded result without re-running your code, so the enqueued messages "
    'would be dropped. Enqueue messages from workflow-level code instead.'
)
_enqueue_handler_boundaries: set[str] = set()
_enqueue_cancellation_rejected = False


async def _enqueue_guard_handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        with pytest.raises(UserError, match='enqueued messages would be dropped'):
            ctx.enqueue('later')
        boundary = 'model' if isinstance(event, (PartStartEvent, PartDeltaEvent)) else 'agent'
        _enqueue_handler_boundaries.add(boundary)


_enqueue_guard_tool_queue: list[str] = []
_enqueue_guard_model_queue: list[str] = []


async def _enqueue_guard_tool(ctx: RunContext[Deps]) -> str:
    while _enqueue_guard_tool_queue:
        ctx.enqueue(_enqueue_guard_tool_queue.pop())
    return 'done'


def _enqueue_guard_model_request(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
    ctx = get_current_run_context()
    assert ctx is not None
    while _enqueue_guard_model_queue:
        ctx.enqueue(_enqueue_guard_model_queue.pop())
    return ModelResponse(parts=[TextPart('done')])


class _EnqueueOnCancelModel(ScriptedContinuationModel):
    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        global _enqueue_cancellation_rejected
        ctx = get_current_run_context()
        assert ctx is not None
        with pytest.raises(UserError, match='enqueued messages would be dropped'):
            ctx.enqueue('later')
        _enqueue_cancellation_rejected = True


_enqueue_handler_agent = Agent(
    TestModel(),
    name='temporal_handler_enqueue',
    tools=[_durability_handler_tool],
    capabilities=[
        TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG, event_stream_handler=_enqueue_guard_handler)
    ],
)
_enqueue_tool_agent = Agent(
    TestModel(),
    deps_type=Deps,
    name='temporal_tool_enqueue',
    tools=[_enqueue_guard_tool],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)
_enqueue_model_agent = Agent(
    FunctionModel(_enqueue_guard_model_request),
    name='temporal_model_enqueue',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)
_enqueue_cancel_model = _EnqueueOnCancelModel()
_enqueue_cancel_agent = Agent(
    _enqueue_cancel_model,
    name='temporal_cancel_enqueue',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class EnqueueGuardHandlerWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await _enqueue_handler_agent.run('run')).output


@workflow.defn
class EnqueueGuardToolWorkflow:
    @workflow.run
    async def run(self) -> None:
        await _enqueue_tool_agent.run('run', deps=Deps(country='test'))


@workflow.defn
class EnqueueGuardModelWorkflow:
    @workflow.run
    async def run(self) -> None:
        await _enqueue_model_agent.run('run')


@workflow.defn
class EnqueueGuardCancellationWorkflow:
    @workflow.run
    async def run(self) -> None:
        await _enqueue_cancel_agent.run('run', usage_limits=UsageLimits(total_tokens_limit=50))


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


async def test_temporal_event_stream_handler_rejects_enqueue(client: Client) -> None:
    _enqueue_handler_boundaries.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[EnqueueGuardHandlerWorkflow],
        plugins=[AgentPlugin(_enqueue_handler_agent)],
    ):
        await client.execute_workflow(
            EnqueueGuardHandlerWorkflow.run,
            id=EnqueueGuardHandlerWorkflow.__name__,
            task_queue=TASK_QUEUE,
        )

    assert _enqueue_handler_boundaries == {'model', 'agent'}


async def test_temporal_tool_rejects_enqueue(client: Client) -> None:
    _enqueue_guard_tool_queue[:] = ['later']
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[EnqueueGuardToolWorkflow],
        plugins=[AgentPlugin(_enqueue_tool_agent)],
    ):
        with workflow_activity_raises(UserError, _ENQUEUE_GUARD_ERROR):
            await client.execute_workflow(
                EnqueueGuardToolWorkflow.run,
                id=EnqueueGuardToolWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )

    _enqueue_guard_tool_queue[:] = ['later']
    await _enqueue_tool_agent.run('run', deps=Deps(country='test'))
    assert not _enqueue_guard_tool_queue


async def test_temporal_non_streaming_model_request_rejects_enqueue(client: Client) -> None:
    _enqueue_guard_model_queue[:] = ['later']
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[EnqueueGuardModelWorkflow],
        plugins=[AgentPlugin(_enqueue_model_agent)],
    ):
        with workflow_activity_raises(UserError, _ENQUEUE_GUARD_ERROR):
            await client.execute_workflow(
                EnqueueGuardModelWorkflow.run,
                id=EnqueueGuardModelWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )

    _enqueue_guard_model_queue[:] = ['later']
    assert (await _enqueue_model_agent.run('run')).output == 'done'
    assert not _enqueue_guard_model_queue


async def test_temporal_cancellation_rejects_enqueue(client: Client) -> None:
    global _enqueue_cancellation_rejected
    _enqueue_cancellation_rejected = False
    _enqueue_cancel_model.reset(
        responses=[
            scripted_response(
                texts=['still going'], state='suspended', provider_response_id='cont1', input_tokens=10, output_tokens=5
            ),
            scripted_response(
                texts=['over budget'],
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
        workflows=[EnqueueGuardCancellationWorkflow],
        plugins=[AgentPlugin(_enqueue_cancel_agent)],
    ):
        with workflow_raises(
            UsageLimitExceeded,
            (
                'Exceeded the total_tokens_limit of 50 (total_tokens=165). Consider raising the limit, or see the docs '
                'on usage limits for budget-aware patterns: https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits'
            ),
        ):
            await client.execute_workflow(
                EnqueueGuardCancellationWorkflow.run,
                id=EnqueueGuardCancellationWorkflow.__name__,
                task_queue=TASK_QUEUE,
            )
    assert _enqueue_cancellation_rejected


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
