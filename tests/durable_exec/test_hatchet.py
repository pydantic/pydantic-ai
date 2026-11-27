"""`HatchetDurability` against a real Hatchet engine, run embedded in the test process's machine.

One engine and one worker serve the whole module: the worker runs its tasks in a thread of this process, so the
agents, models and tools defined in `_build_app` are the ones the durable tasks and their child tasks run.
"""

from __future__ import annotations

import asyncio
import signal
import threading
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import anyio
import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    AgentStreamEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import ModelFallback, RetryOnModelErrors
from pydantic_ai.capabilities.abstract import AbstractCapability, leaf_capabilities
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    ModelServerError,
    RunCancelled,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    WorkspaceOutputLimitError,
    WorkspaceReadOnlyError,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from .._timeouts import READINESS_WAIT_TIMEOUT
from ..conftest import try_import

with try_import() as imports_successful:
    from hatchet_sdk import ClientConfig, DurableContext, Hatchet, Standalone, Worker, embedded
    from hatchet_sdk.clients.rest.api.durable_tasks_api import DurableTasksApi
    from hatchet_sdk.clients.rest.models.v1_durable_event_log_entry import V1DurableEventLogEntry
    from hatchet_sdk.config import EmbeddedHatchetConfig

    from pydantic_ai.durable_exec.hatchet import HatchetDurability, HatchetEventStream
    from pydantic_ai.durable_exec.hatchet._operation_backend import (
        _model_error_message,  # pyright: ignore[reportPrivateUsage]
        hatchet_task_name,
    )
    from pydantic_ai.durable_exec.hatchet._transports import (
        _RequestWire,  # pyright: ignore[reportPrivateUsage]
        _sort_sets,  # pyright: ignore[reportPrivateUsage]
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='hatchet-sdk not installed'),
    pytest.mark.xdist_group(name='hatchet'),
]


EMBEDDED_ENGINE_VERSION = 'v0.114.2'
"""The engine release the tests run against, pinned so CI doesn't pick up a new release on its own.

Pinning also skips the SDK's lookup of the latest release, which leaks the redirect response it reads
(`ResourceWarning: Implicitly cleaning up <HTTPError 302: 'Found'>` on Python 3.14+).
"""


class Prompt(BaseModel):
    prompt: str


class RunOutput(BaseModel):
    output: str
    run_id: str


@dataclass
class Deps:
    city_prefix: str


@dataclass
class Calls:
    """What the models and tools saw, in the worker that ran them."""

    model_requests: int = 0
    tools: list[str] = field(default_factory=list[str])


def _weather_model(calls: Calls, *, tool_rounds: int = 1) -> FunctionModel:
    """Call `get_weather` in `tool_rounds` rounds, then answer with what it last returned."""

    def tool_returns(messages: list[ModelMessage]) -> int:
        return sum(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.model_requests += 1
        if tool_returns(messages) < tool_rounds:
            return ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Paris'})])
        tool_return = messages[-1].parts[0]
        assert isinstance(tool_return, ToolReturnPart)
        return ModelResponse(parts=[TextPart(f'Answer: {tool_return.content}')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        calls.model_requests += 1
        if tool_returns(messages) < tool_rounds:
            yield {0: DeltaToolCall(name='get_weather', json_args='{"city": "Paris"}')}
            return
        for chunk in ['It is ', 'sunny.']:
            yield chunk

    return FunctionModel(respond, stream_function=stream)


@dataclass
class Scenario:
    """A durable task running an agent, what its model or tools saw, and the tasks a worker needs to serve it."""

    run: Standalone[Prompt, dict[str, Any]]
    calls: list[str]
    workflows: list[Any]


@dataclass
class StreamScenario:
    """A durable task running an agent that publishes its events, and the tasks a worker needs to serve it."""

    run: Standalone[Prompt, dict[str, Any]]
    durability: HatchetDurability[Any]
    workflows: list[Any]


@dataclass
class App:
    hatchet: Hatchet
    calls: Calls
    durability: HatchetDurability[Deps]
    streaming_durability: HatchetDurability[Any]
    failing_durability: HatchetDurability[Any]
    run_weather: Standalone[Prompt, RunOutput]
    run_streaming: Standalone[Prompt, RunOutput]
    run_failing_model: Standalone[Prompt, dict[str, str]]
    run_inline_tool: Standalone[Prompt, dict[str, bool]]
    inline_tool_contexts: list[str]
    after_run_durability: HatchetDurability[Any]
    run_after_run_failing: Standalone[Prompt, dict[str, str]]
    unserializable: Scenario
    late_stream_failure: Scenario
    retrying: Scenario
    cancelling: StreamScenario
    parallel: StreamScenario
    unretryable: Scenario
    retried_run: StreamScenario


def _build_app(hatchet: Hatchet) -> tuple[App, list[Any]]:
    calls = Calls()

    durability = HatchetDurability[Deps](hatchet)
    weather_agent = Agent(_weather_model(calls), name='weather', deps_type=Deps, capabilities=[durability])

    @weather_agent.tool
    async def get_weather(ctx: RunContext[Deps], city: str) -> str:
        calls.tools.append(city)
        return f'sunny in {ctx.deps.city_prefix}{city}'

    @hatchet.durable_task(name='pydantic_ai_test_weather', input_validator=Prompt)
    async def run_weather(input: Prompt, ctx: DurableContext) -> RunOutput:
        result = await weather_agent.run(input.prompt, deps=Deps(city_prefix='downtown '), timeout=60)
        return RunOutput(output=result.output, run_id=result.run_id)

    streaming_durability = HatchetDurability[Any](hatchet, event_stream=HatchetEventStream())
    streaming_agent = Agent(
        _weather_model(Calls(), tool_rounds=2), name='streaming', capabilities=[streaming_durability]
    )

    @streaming_agent.tool_plain(name='get_weather')
    async def get_weather_plain(city: str) -> str:
        return f'sunny in {city}'

    @hatchet.durable_task(name='pydantic_ai_test_streaming', input_validator=Prompt)
    async def run_streaming(input: Prompt, ctx: DurableContext) -> RunOutput:
        result = await streaming_agent.run(input.prompt)
        return RunOutput(output=result.output, run_id=result.run_id)

    def reject(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=400, model_name='rejecting', body={'error': 'bad request'})

    async def reject_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise ModelHTTPError(status_code=400, model_name='rejecting', body={'error': 'bad request'})
        yield  # pragma: no cover

    failing_durability = HatchetDurability[Any](hatchet, event_stream=True)
    failing_agent = Agent(
        FunctionModel(reject, stream_function=reject_stream), name='failing', capabilities=[failing_durability]
    )

    @hatchet.durable_task(name='pydantic_ai_test_failing_model', input_validator=Prompt)
    async def run_failing_model(input: Prompt, ctx: DurableContext) -> dict[str, str]:
        try:
            await failing_agent.run(input.prompt)
        except ModelHTTPError as error:
            return {'type': type(error).__name__, 'status_code': str(error.status_code), 'body': str(error.body)}
        raise AssertionError('the model error was not raised')  # pragma: no cover

    # `durable=False` runs a tool in the durable task itself instead of a child task.
    inline_tool_contexts: list[str] = []
    inline_durability = HatchetDurability[Any](hatchet)
    inline_agent = Agent(TestModel(), name='inline', capabilities=[inline_durability])

    @inline_agent.tool_plain(durable=False)
    async def where_am_i() -> str:
        inline_tool_contexts.append(type(hatchet.current_context).__name__)
        return 'here'

    @hatchet.durable_task(name='pydantic_ai_test_inline_tool', input_validator=Prompt)
    async def run_inline_tool(input: Prompt, ctx: DurableContext) -> dict[str, bool]:
        await inline_agent.run(input.prompt)
        return {'ran': True}

    # A capability whose `after_run` raises fails the run after it produced a result.
    class FailingAfterRun(AbstractCapability[Any]):
        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            raise RuntimeError('after_run failed')

    after_run_durability = HatchetDurability[Any](hatchet, event_stream=True)
    after_run_agent = Agent(
        TestModel(), name='after_run_failing', capabilities=[after_run_durability, FailingAfterRun()]
    )

    @hatchet.durable_task(name='pydantic_ai_test_after_run_failing', input_validator=Prompt)
    async def run_after_run_failing(input: Prompt, ctx: DurableContext) -> dict[str, str]:
        try:
            await after_run_agent.run(input.prompt)
        except RuntimeError as error:
            return {'error': str(error)}
        raise AssertionError('the after_run error was not raised')  # pragma: no cover

    app = App(
        hatchet=hatchet,
        calls=calls,
        durability=durability,
        streaming_durability=streaming_durability,
        failing_durability=failing_durability,
        run_weather=run_weather,
        run_streaming=run_streaming,
        run_failing_model=run_failing_model,
        run_inline_tool=run_inline_tool,
        inline_tool_contexts=inline_tool_contexts,
        after_run_durability=after_run_durability,
        run_after_run_failing=run_after_run_failing,
        unserializable=_build_unserializable(hatchet),
        late_stream_failure=_build_late_stream_failure(hatchet),
        retrying=_build_retrying(hatchet),
        cancelling=_build_cancelling(hatchet),
        parallel=_build_parallel(hatchet),
        unretryable=_build_unretryable(hatchet),
        retried_run=_build_retried_run(hatchet),
    )
    workflows: list[Any] = [
        run_weather,
        run_streaming,
        run_failing_model,
        *durability.workflows,
        *streaming_durability.workflows,
        *failing_durability.workflows,
        run_inline_tool,
        *inline_durability.workflows,
        run_after_run_failing,
        *after_run_durability.workflows,
        *app.unserializable.workflows,
        *app.late_stream_failure.workflows,
        *app.retrying.workflows,
        *app.cancelling.workflows,
        *app.parallel.workflows,
        *app.unretryable.workflows,
        *app.retried_run.workflows,
    ]
    return app, workflows


def _build_unserializable(hatchet: Hatchet) -> Scenario:
    """A tool whose result can't be serialized, in tasks Hatchet would retry."""
    tool_calls: list[str] = []
    durability = HatchetDurability[Any](hatchet, task_config={'retries': 1})
    agent = Agent(TestModel(), name='unserializable', capabilities=[durability])

    @agent.tool_plain
    async def make_object() -> Any:
        tool_calls.append('make_object')
        return object()

    @hatchet.durable_task(name='pydantic_ai_test_unserializable', input_validator=Prompt)
    async def run(input: Prompt, ctx: DurableContext) -> dict[str, Any]:
        try:
            await agent.run(input.prompt)
        except UserError as error:
            return {'error': str(error)}
        raise AssertionError('the serialization error was not raised')  # pragma: no cover

    return Scenario(run=run, calls=tool_calls, workflows=[run, *durability.workflows])


def _build_late_stream_failure(hatchet: Hatchet) -> Scenario:
    """A model whose stream fails after it produced output, with a `ModelFallback` to a backup model."""
    backup_streamed: list[str] = []

    async def fail_partway(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'partial'
        raise ModelAPIError('primary', 'stream broke')

    async def backup(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:  # pragma: no cover
        backup_streamed.append('backup')
        yield 'from backup'

    async def consume_events(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for _event in stream:
            pass

    durability = HatchetDurability[Any](
        hatchet,
        models={'backup': FunctionModel(stream_function=backup, model_name='backup')},
        event_stream_handler=consume_events,
    )
    agent = Agent(
        FunctionModel(stream_function=fail_partway, model_name='primary'),
        name='late_stream_failure',
        capabilities=[ModelFallback('backup'), durability],
    )

    @hatchet.durable_task(name='pydantic_ai_test_late_stream_failure', input_validator=Prompt)
    async def run(input: Prompt, ctx: DurableContext) -> dict[str, Any]:
        usage = RunUsage()
        try:
            output = (await agent.run(input.prompt, usage=usage)).output
        except ModelAPIError as error:
            output = f'raised: {error.message}'
        return {'output': output, 'requests': usage.requests, 'output_tokens': usage.output_tokens}

    return Scenario(run=run, calls=backup_streamed, workflows=[run, *durability.workflows])


def _build_retrying(hatchet: Hatchet) -> Scenario:
    """A model that always fails with a transient error, with `RetryOnModelErrors` and a model task Hatchet retries."""
    model_calls: list[str] = []

    def server_error(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_calls.append('call')
        raise ModelServerError('flaky', 'overloaded')

    durability = HatchetDurability[Any](hatchet, model_task_config={'retries': 1})
    agent = Agent(
        FunctionModel(server_error),
        name='retrying',
        capabilities=[RetryOnModelErrors(max_attempts=2, initial_delay=0), durability],
    )

    @hatchet.durable_task(name='pydantic_ai_test_retrying', input_validator=Prompt)
    async def run(input: Prompt, ctx: DurableContext) -> dict[str, Any]:
        try:
            await agent.run(input.prompt)
        except ModelServerError as error:
            return {'type': type(error).__name__}
        raise AssertionError('the model error was not raised')  # pragma: no cover

    return Scenario(run=run, calls=model_calls, workflows=[run, *durability.workflows])


UNRETRYABLE_ERRORS: dict[str, Exception] = {
    'usage_limit': UsageLimitExceeded('over the limit'),
    'workspace_timeout': WorkspaceTimeoutError('timed out'),
    'workspace_output_limit': WorkspaceOutputLimitError('too much output', limit=1),
    'workspace_read_only': WorkspaceReadOnlyError('read-only'),
    'workspace_unavailable': WorkspaceUnavailableError('gone'),
}


def _build_unretryable(hatchet: Hatchet) -> Scenario:
    """A tool raising the error its prompt names, in tasks Hatchet would retry."""
    tool_calls: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        (request,) = messages
        assert isinstance(request, ModelRequest)
        (prompt,) = [part.content for part in request.parts if isinstance(part, UserPromptPart)]
        return ModelResponse(parts=[ToolCallPart('fail', {'kind': prompt})])

    durability = HatchetDurability[Any](hatchet, task_config={'retries': 1})
    agent = Agent(FunctionModel(respond), name='unretryable', capabilities=[durability])

    @agent.tool_plain
    async def fail(kind: str) -> str:
        tool_calls.append(kind)
        raise UNRETRYABLE_ERRORS[kind]

    @hatchet.durable_task(name='pydantic_ai_test_unretryable', input_validator=Prompt)
    async def run(input: Prompt, ctx: DurableContext) -> dict[str, Any]:
        try:
            await agent.run(input.prompt)
        except Exception as error:
            return {'error': type(error).__name__}
        raise AssertionError('the tool error was not raised')  # pragma: no cover

    return Scenario(run=run, calls=tool_calls, workflows=[run, *durability.workflows])


def _build_retried_run(hatchet: Hatchet) -> StreamScenario:
    """A durable task Hatchet retries, whose first attempt's run fails in a tool running in the durable task."""
    attempts: list[int] = []
    durability = HatchetDurability[Any](hatchet, event_stream=True)
    agent = Agent(TestModel(), name='retried_run', capabilities=[durability])

    @agent.tool(durable=False)
    async def fail_once(ctx: RunContext[Any]) -> str:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise RuntimeError('the first attempt fails')
        return 'ok'

    @hatchet.durable_task(name='pydantic_ai_test_retried_run', input_validator=Prompt, retries=1)
    async def run(input: Prompt, ctx: DurableContext) -> dict[str, Any]:
        result = await agent.run(input.prompt)
        return {'output': result.output}

    return StreamScenario(run=run, durability=durability, workflows=[run, *durability.workflows])


def _build_cancelling(hatchet: Hatchet) -> StreamScenario:
    """A run that cancels itself, as its `timeout=` would, from a tool running in the durable task."""
    durability = HatchetDurability[Any](hatchet, event_stream=True)
    agent = Agent(TestModel(), name='cancelling', capabilities=[durability])

    @agent.tool(durable=False)
    async def cancel_run(ctx: RunContext[Any]) -> str:
        ctx.cancel()
        return 'cancelled'

    @hatchet.durable_task(name='pydantic_ai_test_cancelling', input_validator=Prompt)
    async def run(input: Prompt, ctx: DurableContext) -> dict[str, Any]:
        try:
            await agent.run(input.prompt)
        except RunCancelled as error:
            return {'error': type(error).__name__}
        raise AssertionError('the run was not cancelled')  # pragma: no cover

    return StreamScenario(run=run, durability=durability, workflows=[run, *durability.workflows])


def _build_parallel(hatchet: Hatchet) -> StreamScenario:
    """Two parallel tool calls, the first of which only finishes once the second has."""
    second_done = asyncio.Event()
    durability = HatchetDurability[Any](hatchet, event_stream=True)
    agent = Agent(TestModel(call_tools=['first', 'second']), name='parallel', capabilities=[durability])

    @agent.tool_plain
    async def first() -> str:
        await second_done.wait()
        return 'first'

    @agent.tool_plain
    async def second() -> str:
        second_done.set()
        return 'second'

    @hatchet.durable_task(name='pydantic_ai_test_parallel', input_validator=Prompt)
    async def run(input: Prompt, ctx: DurableContext) -> dict[str, Any]:
        await agent.run(input.prompt)
        return {'ran': True}

    return StreamScenario(run=run, durability=durability, workflows=[run, *durability.workflows])


@pytest.fixture(scope='module')
def app(tmp_path_factory: pytest.TempPathFactory) -> Iterator[App]:
    """An embedded Hatchet engine and a worker serving every durable task and child task in this module."""
    # A data directory of its own, so the tests never share a database with a developer's embedded engine, and
    # durable streams enabled for the tenant the engine creates in it. The engine reads the setting from the
    # environment it inherits.
    data_dir = tmp_path_factory.mktemp('hatchet') / 'postgres'
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv('SERVER_DEFAULT_TENANT_ENTITLEMENT_DURABLE_STREAMS', 'true')
        embedded_config = EmbeddedHatchetConfig(version=EMBEDDED_ENGINE_VERSION, postgres_data_dir=str(data_dir))
        hatchet = Hatchet.from_embedded(ClientConfig(embedded=embedded_config))
    app, workflows = _build_app(hatchet)
    # The worker installs its own signal handlers; put pytest's back once it's done.
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)}
    worker: Worker = hatchet.worker('pydantic-ai-test-worker', workflows=workflows)
    # Wait for the worker to start its action listener process during setup, where its launch belongs to this
    # module's fixture rather than to the first test.
    listener_started = threading.Event()
    start_listener = worker._start_action_listener  # pyright: ignore[reportPrivateUsage]

    def start_action_listener() -> Any:
        process = start_listener()
        listener_started.set()
        return process

    worker._start_action_listener = start_action_listener  # pyright: ignore[reportPrivateUsage]
    thread = threading.Thread(target=worker.start, daemon=True)
    thread.start()
    try:
        assert listener_started.wait(READINESS_WAIT_TIMEOUT), f'Hatchet worker did not start: {worker.status}'
        yield app
    finally:
        _stop_worker(worker, thread)
        sidecars = list(embedded._active_sidecars)  # pyright: ignore[reportPrivateUsage]
        hatchet.stop_embedded()
        # TODO: `EmbeddedSidecar.stop()` leaves the sidecar's stdin pipe open; drop this once a release closes it.
        for sidecar in sidecars:
            if sidecar.process.stdin is not None:
                sidecar.process.stdin.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def _stop_worker(worker: Worker, thread: threading.Thread) -> None:
    """Stop the worker started in `thread`, terminating its processes if a graceful exit doesn't finish."""
    loop = worker._loop  # pyright: ignore[reportPrivateUsage]
    if loop is None or not loop.is_running():
        return
    exiting = asyncio.run_coroutine_threadsafe(worker._exit_gracefully(), loop)  # pyright: ignore[reportPrivateUsage]
    try:
        exiting.result(timeout=10)  # real-wait: how long a graceful exit gets before the worker is stopped by force
    except TimeoutError:
        # `_exit_forcefully` ends with `sys.exit`, so do what it does short of that.
        worker._terminate_processes()  # pyright: ignore[reportPrivateUsage]
        loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=READINESS_WAIT_TIMEOUT)


@pytest.fixture(autouse=True, scope='module')
async def shared_event_loop(anyio_backend: str) -> AsyncIterator[None]:
    """Run every test in this module on one event loop.

    The Hatchet client keeps the gRPC listener that waits for run results on the loop it was first used on, so a
    test on a new loop would wait for a result forever. An active module-scoped async fixture keeps anyio's runner,
    and its loop, for the whole module.
    """
    yield


async def test_agent_run_in_durable_task(app: App):
    """Model requests and tool calls run as child tasks, with `deps` serialized into the tool's task."""
    app.calls.tools.clear()
    output = await app.run_weather.aio_run(Prompt(prompt='What is the weather?'))
    assert output.output == 'Answer: sunny in downtown Paris'
    assert app.calls.tools == ['Paris']


async def test_child_tasks_are_registered(app: App):
    """Every operation is a task with a valid Hatchet name; the agent's own `<agent>` toolset ID is escaped."""
    assert sorted(task.name for task in app.durability.workflows) == [
        'weather__function_toolset__-3c-agent-3e-.call_tool',
        'weather__function_toolset__-3c-agent-3e-.validate_args',
        'weather__model.cancel_suspended_response',
        'weather__model.compact_messages',
        'weather__model.count_tokens',
        'weather__model.request',
        'weather__model.request_stream',
    ]


async def _durable_event_log(hatchet: Hatchet, task_external_id: str) -> list[V1DurableEventLogEntry]:
    """The entries of a durable task's event log, which the SDK doesn't expose outside its REST client."""

    def list_entries() -> list[V1DurableEventLogEntry]:
        with hatchet.runs.client() as client:
            return DurableTasksApi(client).v1_durable_task_event_log_list(
                tenant=hatchet.runs.client_config.tenant_id, durable_task=task_external_id
            )

    return await anyio.to_thread.run_sync(list_entries)


async def test_replayed_durable_task_reuses_child_results(app: App):
    """A durable task reset to after its child tasks replays their results instead of running them again.

    The run has a `timeout=`, so its deadline crosses into every child task's input, which Hatchet checks on replay.
    """
    ref = await app.run_weather.aio_run(Prompt(prompt='What is the weather?'), wait_for_result=False)
    first = await ref.aio_result()
    requests, tools = app.calls.model_requests, list(app.calls.tools)

    details = await app.hatchet.runs.aio_get_details(ref.workflow_run_id)
    (task_run,) = details.task_runs.values()
    log = await _durable_event_log(app.hatchet, task_run.external_id)
    # The run's start time, then its two model requests and tool call.
    assert [entry.kind.value for entry in sorted(log, key=lambda entry: entry.node_id)] == ['MEMO', 'RUN', 'RUN', 'RUN']
    # Branch from after the last of them, so the durable task runs again and replays every one.
    after_last = max(entry.node_id for entry in log) + 1
    await app.hatchet.runs.aio_reset_durable_task(task_run.external_id, node_id=after_last, branch_id=1)
    # A workflow run's result maps each of its tasks to that task's output.
    outputs = await app.hatchet.runs.get_run_ref(ref.workflow_run_id).aio_result()
    replayed = RunOutput.model_validate(outputs['pydantic_ai_test_weather'])

    assert replayed == first
    assert (app.calls.model_requests, app.calls.tools) == (requests, tools)


async def test_model_error_keeps_its_type(app: App):
    """A model error raised in a model request task reaches the durable task as the error the model raised."""
    assert await app.run_failing_model.aio_run(Prompt(prompt='hi')) == {
        'type': 'ModelHTTPError',
        'status_code': '400',
        'body': "{'error': 'bad request'}",
    }


async def test_event_stream(app: App):
    """Tool events and the result are published from the durable task, and model token deltas from the model task."""
    ref = await app.run_streaming.aio_run(Prompt(prompt='What is the weather?'), wait_for_result=False)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        result = await ref.aio_result()
    # The topic keeps the run's events after it ends, so subscribing afterwards reads them all.
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        events = [event async for event in app.streaming_durability.stream_agent_events(ref.workflow_run_id)]

    assert [type(event).__name__ for event in events if not isinstance(event, PartDeltaEvent)] == [
        'PartStartEvent',
        'PartEndEvent',
        'FunctionToolCallEvent',
        'FunctionToolResultEvent',
        'PartStartEvent',
        'PartEndEvent',
        'FunctionToolCallEvent',
        'FunctionToolResultEvent',
        'PartStartEvent',
        'FinalResultEvent',
        'PartEndEvent',
        'AgentRunResultEvent',
    ]
    assert any(isinstance(event, PartDeltaEvent) for event in events)
    terminal = events[-1]
    assert isinstance(terminal, AgentRunResultEvent)
    assert terminal.result.output == RunOutput.model_validate(result).output == 'It is sunny.'


async def test_event_stream_of_failed_run(app: App):
    """A run that fails publishes no result, so its subscription ends early with the failure instead of waiting."""
    ref = await app.run_failing_model.aio_run(Prompt(prompt='hi'), wait_for_result=False)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await ref.aio_result()
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        events = app.failing_durability.stream_agent_events(ref.workflow_run_id)
        assert [event async for event in events] == []
    assert events.result is None
    # Only the type: the stream may reach end users, and the message can carry the provider's error body.
    assert events.error == 'ModelHTTPError'


async def test_event_stream_ends_when_after_run_fails(app: App):
    """A capability's `after_run` failing the run still ends the subscription, though no `on_run_error` runs."""
    ref = await app.run_after_run_failing.aio_run(Prompt(prompt='hi'), wait_for_result=False)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        assert await ref.aio_result() == {'error': 'after_run failed'}
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        events = app.after_run_durability.stream_agent_events(ref.workflow_run_id)
        published = [type(event).__name__ async for event in events]
    assert 'AgentRunResultEvent' not in published
    assert events.error == 'RuntimeError'


async def test_event_stream_of_cancelled_run(app: App):
    """A run that cancels itself, as its `timeout=` does, ends the subscription with `RunCancelled`."""
    ref = await app.cancelling.run.aio_run(Prompt(prompt='hi'), wait_for_result=False)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        assert await ref.aio_result() == {'error': 'RunCancelled'}
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        events = app.cancelling.durability.stream_agent_events(ref.workflow_run_id)
        async for _event in events:
            pass
    assert events.error == 'RunCancelled'


async def test_event_stream_of_retried_run_ends_with_its_result(app: App):
    """A failed attempt that Hatchet retries publishes no error, so subscribers get the retry's result."""
    ref = await app.retried_run.run.aio_run(Prompt(prompt='hi'), wait_for_result=False)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await ref.aio_result()
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        events = app.retried_run.durability.stream_agent_events(ref.workflow_run_id)
        async for _event in events:
            pass
    assert events.error is None
    assert events.result is not None


async def test_parallel_tool_results_are_published_in_call_order(app: App):
    """Parallel tool calls publish their results in call order, which a replay reproduces, not the order they finished in."""
    ref = await app.parallel.run.aio_run(Prompt(prompt='hi'), wait_for_result=False)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await ref.aio_result()
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        events = [event async for event in app.parallel.durability.stream_agent_events(ref.workflow_run_id)]
    results = [event.part.tool_name for event in events if isinstance(event, FunctionToolResultEvent)]
    assert results == ['first', 'second']


def test_strict_deps_cross_the_task_boundary(app: App):
    """`deps` cross as JSON, so a strict model's fields that JSON holds as strings, like a date, are read back."""

    class StrictDeps(BaseModel, strict=True):
        day: date

    agent = Agent(
        TestModel(), name='strict_deps', deps_type=StrictDeps, capabilities=[HatchetDurability[StrictDeps](app.hatchet)]
    )
    durability = next(cap for cap in leaf_capabilities(agent.root_capability) if isinstance(cap, HatchetDurability))
    deps = StrictDeps(day=date(2026, 1, 1))
    assert durability.load_deps(durability.dump_deps(deps)) == deps


def test_request_sets_cross_sorted():
    """A set crosses as a sorted list, as its iteration order changes with the hash seed of the worker replaying it."""
    parameters = ModelRequestParameters(revealed_tool_names={'charlie', 'alpha', 'bravo'})
    payload = _sort_sets(
        _RequestWire, {'model_request_parameters': {'revealed_tool_names': list(parameters.revealed_tool_names)}}
    )
    assert payload == {'model_request_parameters': {'revealed_tool_names': ['alpha', 'bravo', 'charlie']}}


def test_model_error_with_unencodable_body_crosses_as_text():
    """A model error whose body isn't UTF-8 crosses as its message instead of failing the task with an encoding error."""
    error = ModelHTTPError(status_code=400, model_name='rejecting', body=b'\xff')
    assert _model_error_message(error) == str(error)


@pytest.mark.parametrize('kind', list(UNRETRYABLE_ERRORS))
async def test_errors_a_retry_cannot_fix_are_not_retried(app: App, kind: str):
    """A usage limit or workspace error fails a task the same way on every attempt, so it isn't retried."""
    app.unretryable.calls.clear()
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await app.unretryable.run.aio_run(Prompt(prompt=kind))
    assert app.unretryable.calls == [kind]


async def test_engine_without_durable_event_log_is_refused(app: App, monkeypatch: pytest.MonkeyPatch):
    """An engine too old to record child tasks in the event log can't replay a run, so the run refuses to start."""
    app.unserializable.calls.clear()
    monkeypatch.setattr(DurableContext, '_supports_durable_eviction', property(lambda self: False))
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        output = await app.unserializable.run.aio_run(Prompt(prompt='hi'))
    assert 'records no child tasks in the event log' in output['error']
    assert app.unserializable.calls == []


async def test_unserializable_tool_result_is_not_retried(app: App):
    """A tool result that can't be serialized fails the run with a `UserError`, without running the tool again."""
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        output = await app.unserializable.run.aio_run(Prompt(prompt='hi'))
    assert 'failed to be serialized' in output['error']
    assert app.unserializable.calls == ['make_object']


async def test_stream_failing_after_output_does_not_fall_back(app: App):
    """A stream that fails after producing output in its task is raised, not fallen back from, and its usage counts.

    The model request task delivers the output to the event handler before the durable task sees the stream, so the
    error crosses with what the stream produced, as outside durable execution, where only a stream that fails before
    its first event falls back.
    """
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        output = await app.late_stream_failure.run.aio_run(Prompt(prompt='hello'))
    assert output == {'output': 'raised: stream broke', 'requests': 1, 'output_tokens': 1}
    assert app.late_stream_failure.calls == []


async def test_retry_on_model_errors_replaces_task_retries(app: App):
    """With `RetryOnModelErrors`, a failing model is retried as often as the capability says, not times the task's `retries`."""
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        assert await app.retrying.run.aio_run(Prompt(prompt='hi')) == {'type': 'ModelServerError'}
    assert app.retrying.calls == ['call', 'call']


async def test_non_durable_tool_runs_in_the_durable_task(app: App):
    """A tool with `durable=False` runs in the durable task's own code, not in a child task."""
    assert await app.run_inline_tool.aio_run(Prompt(prompt='hi')) == {'ran': True}
    assert app.inline_tool_contexts == ['DurableContext']


async def test_outside_durable_task_is_transparent(app: App):
    """Outside a durable task the agent runs in-process, without spawning tasks."""
    agent = Agent(TestModel(), name='transparent', capabilities=[HatchetDurability(app.hatchet)])
    result = await agent.run('hi')
    assert result.output == 'success (no tool calls)'


def test_hatchet_task_name_escapes_invalid_characters():
    """Hatchet names allow letters, digits, `.`, `-` and `_`; everything else is escaped deterministically."""
    assert hatchet_task_name('my agent__function_toolset__<agent>.call_tool') == (
        'my-20-agent__function_toolset__-3c-agent-3e-.call_tool'
    )


def test_hatchet_task_names_never_collide():
    """`-` introduces an escape and is escaped itself, so different names never become the same task name."""
    names = ['toolset__a/b', 'toolset__a-2f-b', 'toolset__a_x2f_b', 'toolset__a-b']
    assert len({hatchet_task_name(name) for name in names}) == len(names)


def test_invalid_task_config(app: App):
    """An unknown task config key fails when the capability is created, not when the task runs."""
    with pytest.raises(UserError, match=r"`task_config` has unknown Hatchet task config keys: \['timeout'\]"):
        HatchetDurability(app.hatchet, task_config={'timeout': 1})  # pyright: ignore[reportArgumentType]


def test_stream_agent_events_needs_event_stream(app: App):
    with pytest.raises(UserError, match='has no `event_stream`'):
        app.durability.stream_agent_events('run-id')
