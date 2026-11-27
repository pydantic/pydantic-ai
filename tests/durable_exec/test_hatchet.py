"""`HatchetDurability` against a real Hatchet engine, run embedded in the test process's machine.

One engine and one worker serve the whole module: the worker runs its tasks in a thread of this process, so the
agents, models and tools defined in `_build_app` are the ones the durable tasks and their child tasks run.
"""

from __future__ import annotations

import asyncio
import dataclasses
import signal
import threading
from collections.abc import AsyncIterable, AsyncIterator, Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    AgentStreamEvent,
    FinalResultEvent,
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
from pydantic_ai.capabilities import (
    DynamicCapability,
    LocalWorkspace,
    ModelFallback,
    RetryOnModelErrors,
    Toolset,
    durable_operation,
)
from pydantic_ai.capabilities.abstract import AbstractCapability, leaf_capabilities
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    ModelServerError,
    RetryModelRequest,
    RunCancelled,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import DynamicToolset, FunctionToolset
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits
from pydantic_ai.workspaces import (
    WorkspaceOutputLimitError,
    WorkspaceReadOnlyError,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from .._timeouts import READINESS_WAIT_TIMEOUT
from ..conftest import try_import
from ..continuation_utils import ScriptedContinuationModel, scripted_response

with try_import() as imports_successful:
    import grpc
    from fastmcp.client.transports import FastMCPTransport
    from hatchet_sdk import ClientConfig, DurableContext, Hatchet, Standalone, Worker, embedded
    from hatchet_sdk.clients.rest.api.durable_tasks_api import DurableTasksApi
    from hatchet_sdk.clients.rest.models.v1_durable_event_log_entry import V1DurableEventLogEntry
    from hatchet_sdk.config import EmbeddedHatchetConfig
    from hatchet_sdk.exceptions import TaskRunError

    from pydantic_ai.durable_exec._operation import ModelCancelSuspendedResponseParams, ToolsetCallToolParams
    from pydantic_ai.durable_exec.hatchet import (
        HatchetAgentRunEvents,
        HatchetDurability,
        HatchetEventStream,
        HatchetRunContext,
    )
    from pydantic_ai.durable_exec.hatchet._event_stream import task_event_stream_handler
    from pydantic_ai.durable_exec.hatchet._operation_backend import (
        _model_error_message,  # pyright: ignore[reportPrivateUsage]
        _rebuild_error,  # pyright: ignore[reportPrivateUsage]
        _summary_metadata,  # pyright: ignore[reportPrivateUsage]
        hatchet_task_name,
    )
    from pydantic_ai.durable_exec.hatchet._run_context import stable_capability_ids
    from pydantic_ai.durable_exec.hatchet._spec import _serialization_failure  # pyright: ignore[reportPrivateUsage]
    from pydantic_ai.durable_exec.hatchet._transports import (
        HatchetParameterTransports,
        _RequestWire,  # pyright: ignore[reportPrivateUsage]
        _sort_sets,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.mcp import MCPToolset
    from tests.mcp_server import mcp as test_mcp_server

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
class RetriedScenario(StreamScenario):
    """A `StreamScenario` whose tool records each attempt of the durable task that reached it."""

    attempts: list[int]


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
    retried_run: RetriedScenario
    scenarios: dict[str, Scenario]
    streams: dict[str, StreamScenario]


def _build_app(hatchet: Hatchet, workspace_dir: Path) -> tuple[App, list[Any]]:
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

    def reject(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
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
        scenarios=_build_scenarios(hatchet, workspace_dir),
        streams=_build_streams(hatchet),
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
        *(workflow for scenario in app.scenarios.values() for workflow in scenario.workflows),
        *(workflow for scenario in app.streams.values() for workflow in scenario.workflows),
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


def _build_retried_run(hatchet: Hatchet) -> RetriedScenario:
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

    return RetriedScenario(run=run, durability=durability, workflows=[run, *durability.workflows], attempts=attempts)


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


def _durable_run(
    hatchet: Hatchet, name: str, agent: Agent[Any, Any], **run_kwargs: Any
) -> Standalone[Prompt, dict[str, Any]]:
    """A durable task running `agent`, reporting its output or the error it failed with."""

    @hatchet.durable_task(name=f'pydantic_ai_test_{name}', input_validator=Prompt)
    async def run(input: Prompt, ctx: DurableContext) -> dict[str, Any]:
        try:
            result = await agent.run(input.prompt, **run_kwargs)
        except Exception as error:
            return {'error': type(error).__name__, 'message': str(error)}
        return {'output': result.output}

    return run


def _call_then_answer(tool_args: dict[str, dict[str, Any]]) -> FunctionModel:
    """Call each tool with its arguments at once, then answer with what each returned."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart(name, args) for name, args in tool_args.items()])
        return ModelResponse(parts=[TextPart(', '.join(f'{part.tool_name}={part.content}' for part in returns))])

    return FunctionModel(respond)


async def double(x: int) -> int:
    return x * 2


def _doubling_toolset(ctx: RunContext[Any]) -> FunctionToolset[Any]:
    return FunctionToolset([double])


def _scenario(
    hatchet: Hatchet,
    agent: Agent[Any, Any],
    durability: HatchetDurability[Any],
    calls: list[str] | None = None,
    **run_kwargs: Any,
) -> Scenario:
    assert agent.name is not None
    run = _durable_run(hatchet, agent.name, agent, **run_kwargs)
    return Scenario(run=run, calls=[] if calls is None else calls, workflows=[run, *durability.workflows])


def _build_toolsets(hatchet: Hatchet) -> Scenario:
    """Tools of an MCP server and of a `DynamicToolset`, each kind of toolset in tasks of its own."""
    durability = HatchetDurability[Any](hatchet)
    mcp = MCPToolset(
        FastMCPTransport(test_mcp_server), id='mcp', init_timeout=READINESS_WAIT_TIMEOUT, include_instructions=True
    )
    agent = Agent(
        _call_then_answer({'celsius_to_fahrenheit': {'celsius': 100}, 'double': {'x': 21}}),
        name='toolsets',
        toolsets=[mcp, DynamicToolset(_doubling_toolset, id='dynamic')],
        capabilities=[durability],
    )
    return _scenario(hatchet, agent, durability)


def _build_workspace(hatchet: Hatchet, workspace_dir: Path) -> Scenario:
    """A tool that writes and reads a file in the agent's workspace, whose every call is a task."""
    durability = HatchetDurability[Any](hatchet)
    agent = Agent(
        TestModel(call_tools=['round_trip']),
        name='workspace',
        capabilities=[LocalWorkspace(workspace_dir), durability],
    )

    @agent.tool
    async def round_trip(ctx: RunContext[Any]) -> str:
        await ctx.workspace.write_text('note.txt', 'workspace content')
        return await ctx.workspace.read_text('note.txt')

    return _scenario(hatchet, agent, durability)


def _build_capability_operation(hatchet: Hatchet) -> Scenario:
    """A capability whose `@durable_operation` method runs in a task of its own."""
    calls: list[str] = []

    class Counting(AbstractCapability[Any]):
        id = 'counting'

        async def before_run(self, ctx: RunContext[Any]) -> None:
            await self.count(ctx)

        @durable_operation('count')
        async def count(self, ctx: RunContext[Any]) -> None:
            calls.append(type(hatchet.current_context).__name__)

    durability = HatchetDurability[Any](hatchet)
    agent = Agent(TestModel(), name='capability_operation', capabilities=[Counting(), durability])
    return _scenario(hatchet, agent, durability, calls)


def _rerouted(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('from the rerouted model')])


def _build_reroute(hatchet: Hatchet, name: str, target: str | Model) -> Scenario:
    """A capability hook, running as a durable operation, that moves the first attempt to `target`."""

    class Reroute(AbstractCapability[Any]):
        id = 'reroute'

        @durable_operation('prepare')
        async def prepare_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if request_context.attempt == 1:
                raise RetryModelRequest(target)
            return request_context  # pragma: no cover

    durability = HatchetDurability[Any](hatchet, models={'rerouted': FunctionModel(_rerouted, model_name='rerouted')})
    # The agent's own model is never called: the hook moves the first attempt before anything is sent.
    agent = Agent(TestModel(), name=name, capabilities=[Reroute(), durability])
    return _scenario(hatchet, agent, durability)


def _build_event_handler(hatchet: Hatchet) -> Scenario:
    """An `event_stream_handler` beside a published stream, which gets the run's tool events in tasks of their own."""
    calls: list[str] = []

    async def record(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            calls.append(event.event_kind)

    durability = HatchetDurability[Any](hatchet, event_stream=True, event_stream_handler=record)
    agent = Agent(TestModel(call_tools=['ping']), name='event_handler', capabilities=[durability])

    @agent.tool_plain
    async def ping() -> str:
        return 'pong'

    return _scenario(hatchet, agent, durability, calls)


def _build_compaction(hatchet: Hatchet) -> Scenario:
    """A model whose history a capability compacts, with tokens counted before each request, each in a task."""
    calls: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('ping')])
        return ModelResponse(parts=[TextPart(f'saw {len(messages)} messages')])

    class CompactingModel(FunctionModel):
        async def compact_messages(
            self, request_context: ModelRequestContext, *, instructions: str | None = None
        ) -> ModelResponse:
            calls.append('compact')
            return ModelResponse(parts=[TextPart('summary')])

        async def count_tokens(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> RequestUsage:
            calls.append('count')
            return RequestUsage(input_tokens=1)

    class CompactHistory(AbstractCapability[Any]):
        """Compacts every message before the current request, as provider compaction capabilities do."""

        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if len(request_context.messages) < 2:
                return request_context
            history = ModelRequestContext(
                model=request_context.model,
                messages=request_context.messages[:-1],
                model_settings=request_context.model_settings,
                model_request_parameters=request_context.model_request_parameters,
            )
            compacted = await request_context.model.compact_messages(history)
            request_context.messages = [compacted, request_context.messages[-1]]
            return request_context

    durability = HatchetDurability[Any](hatchet)
    agent = Agent(CompactingModel(respond), name='compaction', capabilities=[CompactHistory(), durability])

    @agent.tool_plain
    async def ping() -> str:
        return 'pong'

    limits = UsageLimits(input_tokens_limit=10_000, count_tokens_before_request=True)
    return _scenario(hatchet, agent, durability, calls, usage_limits=limits)


def _build_continuation(hatchet: Hatchet) -> tuple[Scenario, ScriptedContinuationModel]:
    """A model whose suspended response goes over the usage limit, so the run cancels it in a task."""
    model = ScriptedContinuationModel(
        responses=[
            scripted_response(
                texts=['still going '],
                state='suspended',
                provider_response_id='cont1',
                input_tokens=10,
                output_tokens=5,
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
    durability = HatchetDurability[Any](hatchet)
    agent = Agent(model, name='continuation', capabilities=[durability])
    return _scenario(hatchet, agent, durability, usage_limits=UsageLimits(total_tokens_limit=20)), model


def _sync_tool() -> str:
    return 'sync'  # pragma: no cover


def _build_tool_metadata(
    hatchet: Hatchet, name: str, metadata: object, *, sync: bool = False, dynamic: bool = False
) -> Scenario:
    """An agent calling one tool, `tool`, whose `'hatchet'` metadata is `metadata`."""
    calls: list[str] = []

    async def tool() -> str:
        calls.append(type(hatchet.current_context).__name__)
        return 'ran'

    function = _sync_tool if sync else tool
    toolset = FunctionToolset[Any](id='tools')
    toolset.add_function(function, name='tool', metadata={'hatchet': metadata})
    durability = HatchetDurability[Any](hatchet)
    toolsets: list[Any] = [DynamicToolset(lambda ctx: toolset, id='dynamic')] if dynamic else [toolset]
    agent = Agent(TestModel(call_tools=['tool']), name=name, toolsets=toolsets, capabilities=[durability])
    return _scenario(hatchet, agent, durability, calls)


def _build_construct_in_task(hatchet: Hatchet) -> Scenario:
    """A durable task that constructs its agent itself, too late to register the agent's tasks with the worker."""

    @hatchet.durable_task(name='pydantic_ai_test_construct_in_task', input_validator=Prompt)
    async def run(input: Prompt, ctx: DurableContext) -> dict[str, Any]:
        try:
            Agent(TestModel(), name='too_late', capabilities=[HatchetDurability(hatchet)])
        except UserError as error:
            return {'error': type(error).__name__, 'message': str(error)}
        raise AssertionError('the agent was constructed')  # pragma: no cover

    return Scenario(run=run, calls=[], workflows=[run])


def _build_dynamic_retry(hatchet: Hatchet) -> Scenario:
    """A `RetryOnModelErrors` that only a `DynamicCapability` returns for the run."""
    durability = HatchetDurability[Any](hatchet)
    agent = Agent(
        TestModel(),
        name='dynamic_retry',
        capabilities=[DynamicCapability(lambda ctx: RetryOnModelErrors(), id='retry'), durability],
    )
    return _scenario(hatchet, agent, durability)


def _fail_once(calls: list[str], error: Exception) -> FunctionModel:
    """A model whose first request fails with `error`, and whose next one answers."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('call')
        if len(calls) == 1:
            raise error
        return ModelResponse(parts=[TextPart('recovered')])

    return FunctionModel(respond)


def _build_retry_after(hatchet: Hatchet) -> Scenario:
    """A rate-limited model request whose provider asks for a wait, in a model task Hatchet retries."""
    calls: list[str] = []
    durability = HatchetDurability[Any](hatchet, model_task_config={'retries': 1})
    rate_limited = ModelHTTPError(status_code=429, model_name='limited', retry_after=0.01)
    agent = Agent(_fail_once(calls, rate_limited), name='retry_after', capabilities=[durability])
    return _scenario(hatchet, agent, durability, calls)


def _build_durable_sleep(hatchet: Hatchet) -> Scenario:
    """A `RetryOnModelErrors` waiting between attempts, which the durable task does on a durable sleep."""
    calls: list[str] = []
    durability = HatchetDurability[Any](hatchet)
    agent = Agent(
        _fail_once(calls, ModelServerError('flaky', 'overloaded')),
        name='durable_sleep',
        capabilities=[RetryOnModelErrors(max_attempts=2, initial_delay=0.01), durability],
    )
    return _scenario(hatchet, agent, durability, calls)


def _build_anonymous_capability(hatchet: Hatchet) -> Scenario:
    """A tool from a capability without an `id`, which core keys by a name that's random every run."""
    calls: list[str] = []

    async def note() -> str:
        calls.append('note')
        return 'noted'

    durability = HatchetDurability[Any](hatchet)
    agent = Agent(
        TestModel(call_tools=['note']),
        name='anonymous_capability',
        capabilities=[Toolset(FunctionToolset([note], id='notes')), durability],
    )
    return _scenario(hatchet, agent, durability, calls)


def _build_externally_cancelled(hatchet: Hatchet) -> StreamScenario:
    """A run its durable task cancels, as Hatchet does when it evicts the task, followed by another run in the task."""
    scopes: list[anyio.CancelScope] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        prompt = messages[0].parts[0]
        assert isinstance(prompt, UserPromptPart)
        if prompt.content == 'cancel':
            yield {0: DeltaToolCall(name='cancel', json_args='{}')}
            return
        yield 'done'

    durability = HatchetDurability[Any](hatchet, event_stream=True)
    agent = Agent(FunctionModel(stream_function=stream), name='externally_cancelled', capabilities=[durability])

    @agent.tool_plain(durable=False)
    async def cancel() -> str:
        scopes[-1].cancel()
        await anyio.sleep_forever()
        raise AssertionError('the run was not cancelled')  # pragma: no cover

    @hatchet.durable_task(name='pydantic_ai_test_externally_cancelled', input_validator=Prompt)
    async def run(input: Prompt, ctx: DurableContext) -> dict[str, Any]:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await agent.run('cancel')
        result = await agent.run('again')
        return {'cancelled': scope.cancelled_caught, 'output': result.output}

    return StreamScenario(run=run, durability=durability, workflows=[run, *durability.workflows])


def _build_scenarios(hatchet: Hatchet, workspace_dir: Path) -> dict[str, Scenario]:
    continuation, continuation_model = _build_continuation(hatchet)
    CONTINUATION_MODELS.append(continuation_model)
    return {
        'toolsets': _build_toolsets(hatchet),
        'workspace': _build_workspace(hatchet, workspace_dir),
        'capability_operation': _build_capability_operation(hatchet),
        'reroute': _build_reroute(hatchet, 'reroute', 'rerouted'),
        'reroute_to_instance': _build_reroute(
            hatchet, 'reroute_to_instance', FunctionModel(_rerouted, model_name='instance')
        ),
        'event_handler': _build_event_handler(hatchet),
        'compaction': _build_compaction(hatchet),
        'continuation': continuation,
        'construct_in_task': _build_construct_in_task(hatchet),
        'dynamic_retry': _build_dynamic_retry(hatchet),
        'retry_after': _build_retry_after(hatchet),
        'durable_sleep': _build_durable_sleep(hatchet),
        'anonymous_capability': _build_anonymous_capability(hatchet),
        **{
            name: _build_tool_metadata(hatchet, name, metadata, **flags)
            for name, (metadata, flags) in TOOL_METADATA_CASES.items()
        },
    }


CONTINUATION_MODELS: list[ScriptedContinuationModel] = []
"""The continuation scenario's model, which records the responses it was asked to cancel."""

TOOL_METADATA_CASES: dict[str, tuple[object, dict[str, bool]]] = {
    'inline_metadata': (False, {}),
    'run_option_metadata': ({'priority': 'HIGH'}, {}),
    'sync_inline_metadata': (False, {'sync': True}),
    'dynamic_inline_metadata': (False, {'dynamic': True}),
    'invalid_metadata': ('inline', {}),
    'registration_option_metadata': ({'retries': 3}, {}),
}


def _build_unpublishable(hatchet: Hatchet) -> StreamScenario:
    """A capability's `after_run` replacing the output with one the stream can't carry."""

    class Unpublishable(AbstractCapability[Any]):
        async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            return dataclasses.replace(result, output=object())

    durability = HatchetDurability[Any](hatchet, event_stream=True)
    agent = Agent(TestModel(), name='unpublishable', capabilities=[Unpublishable(), durability])
    run = _durable_run(hatchet, 'unpublishable', agent)
    return StreamScenario(run=run, durability=durability, workflows=[run, *durability.workflows])


def _build_filtered(hatchet: Hatchet) -> StreamScenario:
    """A streamed run publishing every event but the model's token deltas."""
    durability = HatchetDurability[Any](
        hatchet, event_stream=HatchetEventStream(events=lambda event: not isinstance(event, PartDeltaEvent))
    )
    agent = Agent(_weather_model(Calls()), name='filtered', capabilities=[durability])

    @agent.tool_plain(name='get_weather')
    async def get_weather(city: str) -> str:
        return f'sunny in {city}'

    run = _durable_run(hatchet, 'filtered', agent)
    return StreamScenario(run=run, durability=durability, workflows=[run, *durability.workflows])


def _build_streams(hatchet: Hatchet) -> dict[str, StreamScenario]:
    return {
        'unpublishable': _build_unpublishable(hatchet),
        'filtered': _build_filtered(hatchet),
        'externally_cancelled': _build_externally_cancelled(hatchet),
    }


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
    app, workflows = _build_app(hatchet, tmp_path_factory.mktemp('workspace'))
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
            if sidecar.process.stdin is not None:  # pragma: no branch
                sidecar.process.stdin.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def _stop_worker(worker: Worker, thread: threading.Thread) -> None:
    """Stop the worker started in `thread`, terminating its processes if a graceful exit doesn't finish."""
    loop = worker._loop  # pyright: ignore[reportPrivateUsage]
    if loop is None or not loop.is_running():  # pragma: no cover
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
    app.retried_run.attempts.clear()
    ref = await app.retried_run.run.aio_run(Prompt(prompt='hi'), wait_for_result=False)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await ref.aio_result()
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        events = app.retried_run.durability.stream_agent_events(ref.workflow_run_id)
        async for _event in events:
            pass
    assert app.retried_run.attempts == [0, 1]
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
    app.unserializable.calls.clear()
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
    """Outside a durable task the agent runs in-process, without spawning tasks, `timeout=` included."""
    agent = Agent(TestModel(), name='transparent', capabilities=[HatchetDurability(app.hatchet)])
    result = await agent.run('hi', timeout=60)
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


async def _run(app: App, name: str) -> dict[str, Any]:
    scenario = app.scenarios[name]
    scenario.calls.clear()
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        return await scenario.run.aio_run(Prompt(prompt='hi'))


async def test_mcp_and_dynamic_toolset_tools_run_in_tasks(app: App):
    """Tools of an MCP server and of a `DynamicToolset` are listed and called from tasks, and their results return."""
    assert await _run(app, 'toolsets') == {'output': 'celsius_to_fahrenheit=212, double=42'}


async def test_workspace_calls_run_in_tasks(app: App):
    """A tool's calls on the agent's workspace cross into tasks and back."""
    assert await _run(app, 'workspace') == {'output': '{"round_trip":"workspace content"}'}


async def test_capability_durable_operation_runs_in_a_task(app: App):
    """A capability's `@durable_operation` method runs in a child task, not in the durable task."""
    assert await _run(app, 'capability_operation') == {'output': 'success (no tool calls)'}
    assert app.scenarios['capability_operation'].calls == ['Context']


async def test_durable_hook_can_reroute_the_request(app: App):
    """A `RetryModelRequest` a hook raises in its task reaches the attempt loop, which moves to the named model."""
    assert await _run(app, 'reroute') == {'output': 'from the rerouted model'}


async def test_durable_hook_cannot_reroute_to_a_model_instance(app: App):
    """A model instance can't cross out of a hook's task, so naming one fails the run with a `UserError`."""
    output = await _run(app, 'reroute_to_instance')
    assert output['error'] == 'UserError'
    assert 'can only name the model to attempt by its name' in output['message']


async def test_event_stream_handler_gets_tool_events(app: App):
    """The `event_stream_handler` gets the events the durable task produces, like tool calls and their results."""
    assert await _run(app, 'event_handler') == {'output': '{"ping":"pong"}'}
    assert {'function_tool_call', 'function_tool_result'} <= set(app.scenarios['event_handler'].calls)


async def test_compaction_and_token_counting_run_in_tasks(app: App):
    """A capability compacting the history and a usage limit counting tokens each call the model from a task."""
    assert await _run(app, 'compaction') == {'output': 'saw 2 messages'}
    assert sorted(app.scenarios['compaction'].calls) == ['compact', 'count', 'count']


async def test_suspended_response_over_the_limit_is_cancelled(app: App):
    """A suspended response that takes the run over its usage limit is cancelled at the provider before the run fails."""
    (model,) = CONTINUATION_MODELS
    model.cancelled.clear()
    output = await _run(app, 'continuation')
    assert output['error'] == 'UsageLimitExceeded'
    assert [response.provider_response_id for response in model.cancelled] == ['cont2']


async def test_agent_constructed_in_a_durable_task_is_refused(app: App):
    """An agent constructed inside a durable task can't register its tasks with the running worker."""
    output = await _run(app, 'construct_in_task')
    assert output['error'] == 'UserError'
    assert 'must be constructed outside of a Hatchet durable task' in output['message']


async def test_retry_on_model_errors_added_for_the_run_is_refused(app: App):
    """A `RetryOnModelErrors` that only a `DynamicCapability` returns fails the run, as the model task would retry too."""
    output = await _run(app, 'dynamic_retry')
    assert output['error'] == 'UserError'
    assert 'must be one of the capabilities the agent is built with' in output['message']


async def test_tool_with_false_metadata_runs_in_the_durable_task(app: App):
    """An async function tool with `'hatchet': False` metadata runs in the durable task, not in a child task."""
    assert await _run(app, 'inline_metadata') == {'output': '{"tool":"ran"}'}
    assert app.scenarios['inline_metadata'].calls == ['DurableContext']


async def test_tool_metadata_can_set_run_options(app: App):
    """A tool's `'hatchet'` metadata can set the options of each run of its task, like its priority."""
    assert await _run(app, 'run_option_metadata') == {'output': '{"tool":"ran"}'}
    assert app.scenarios['run_option_metadata'].calls == ['Context']


@pytest.mark.parametrize(
    'name, message',
    [
        pytest.param('sync_inline_metadata', 'non-async tools run in threads', id='sync-tool'),
        pytest.param('dynamic_inline_metadata', 'only function tools can run outside a task', id='dynamic-tool'),
        pytest.param('invalid_metadata', "invalid `'hatchet'` metadata: expected a dict or `False`", id='not-a-dict'),
        pytest.param(
            'registration_option_metadata', "Hatchet task options ['retries'] are fixed", id='registration-key'
        ),
    ],
)
async def test_invalid_tool_metadata_fails_the_run(app: App, name: str, message: str):
    """Tool `'hatchet'` metadata that Hatchet can't honour fails the run with a `UserError` naming the problem."""
    output = await _run(app, name)
    assert output['error'] == 'UserError'
    assert message in output['message']


async def _stream_run(app: App, name: str) -> tuple[dict[str, Any], HatchetAgentRunEvents[Any], list[Any]]:
    scenario = app.streams[name]
    ref = await scenario.run.aio_run(Prompt(prompt='What is the weather?'), wait_for_result=False)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        output = await ref.aio_result()
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with scenario.durability.stream_agent_events(ref.workflow_run_id) as events:
            published = [event async for event in events]
    return output, events, published


async def test_result_the_stream_cannot_carry_ends_the_stream(app: App):
    """A result that can't be published fails the run and ends the subscription with the error, not a result."""
    output, events, published = await _stream_run(app, 'unpublishable')
    assert output['error'] == 'PydanticSerializationError'
    assert not any(isinstance(event, AgentRunResultEvent) for event in published)
    assert events.error == 'PydanticSerializationError'


async def test_event_stream_publishes_only_selected_events(app: App):
    """`HatchetEventStream(events=...)` selects the model task's events to publish, here all but token deltas."""
    _, events, published = await _stream_run(app, 'filtered')
    assert published
    assert not any(isinstance(event, PartDeltaEvent) for event in published)
    assert isinstance(published[-1], AgentRunResultEvent)
    assert events.result is not None


async def test_event_stream_resumes_after_its_cursor(app: App):
    """A subscription resumed from the cursor of the last event read yields the events after it."""
    scenario = app.streams['filtered']
    ref = await scenario.run.aio_run(Prompt(prompt='What is the weather?'), wait_for_result=False)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await ref.aio_result()
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        everything = [event async for event in scenario.durability.stream_agent_events(ref.workflow_run_id)]
        async with scenario.durability.stream_agent_events(ref.workflow_run_id) as first_read:
            first = await anext(first_read)
        resumed = scenario.durability.stream_agent_events(ref.workflow_run_id, cursor=first_read.cursor)
        rest = [event async for event in resumed]
    assert [first, *rest] == everything


async def test_event_stream_resubscribes_when_the_server_hangs_up(app: App, monkeypatch: pytest.MonkeyPatch):
    """A subscription the server hangs up before the run ends is picked up again after the last event read.

    The server hangs up a subscription after it has been idle for a while; the SDK's subscription is replaced by one
    that hangs up after each event, so the test doesn't wait that long.
    """
    scenario = app.streams['filtered']
    ref = await scenario.run.aio_run(Prompt(prompt='What is the weather?'), wait_for_result=False)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await ref.aio_result()
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        everything = [event async for event in scenario.durability.stream_agent_events(ref.workflow_run_id)]
    subscribe = app.hatchet.streams.aio_subscribe
    subscriptions: list[str | None] = []

    async def hang_up_after_one(topic: str, *, namespace: str = '', cursor: str | None = None) -> AsyncIterator[Any]:
        subscriptions.append(cursor)
        async for item in subscribe(topic, namespace=namespace, cursor=cursor):  # pragma: no branch
            yield item
            return

    monkeypatch.setattr(app.hatchet.streams, 'aio_subscribe', hang_up_after_one)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        resubscribed = [event async for event in scenario.durability.stream_agent_events(ref.workflow_run_id)]
    assert resubscribed == everything
    assert len(subscriptions) == len(everything)


@pytest.mark.parametrize(
    'code, error',
    [
        pytest.param(grpc.StatusCode.PERMISSION_DENIED, 'UserError', id='durable-streams-disabled'),
        pytest.param(grpc.StatusCode.INTERNAL, 'TaskRunError', id='other-error'),
    ],
)
async def test_event_stream_publish_failure_fails_the_run(
    app: App, monkeypatch: pytest.MonkeyPatch, code: grpc.StatusCode, error: str
):
    """A tenant without durable streams fails the run with a `UserError` saying so; another publish error fails it too.

    The model request task publishes the first events, so any other error reaches the durable task as its failure.

    The embedded engine enables durable streams for its only tenant, so the SDK's publish is replaced by one that
    fails as the engine does for a tenant without them.
    """

    async def fail(*args: Any, **kwargs: Any) -> None:
        raise grpc.aio.AioRpcError(code, grpc.aio.Metadata(), grpc.aio.Metadata(), details='durable streams')

    monkeypatch.setattr(app.hatchet.streams, 'aio_publish', fail)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        output = await app.streams['filtered'].run.aio_run(Prompt(prompt='What is the weather?'))
    assert output['error'] == error


@pytest.mark.parametrize('fails', [False, True])
async def test_event_stream_outside_durable_task_publishes_nothing(app: App, fails: bool):
    """Outside a durable task, an agent with `event_stream` streams to its handler in-process, succeeding or failing."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if fails:
            raise ModelHTTPError(status_code=400, model_name='rejecting')
        return ModelResponse(parts=[TextPart('in-process')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        respond(messages, info)
        yield 'in-process'

    async def consume(ctx: RunContext[Any], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _event in events:
            pass

    agent = Agent(
        FunctionModel(respond, stream_function=stream),
        name='outside',
        capabilities=[HatchetDurability(app.hatchet, event_stream=True, event_stream_handler=consume)],
    )
    if fails:
        with pytest.raises(ModelHTTPError):
            await agent.run('hi')
    else:
        assert (await agent.run('hi')).output == 'in-process'


def test_backend_needs_an_agent(app: App):
    """A `HatchetDurability` registers its tasks when attached to an agent, so it has none to run before that."""
    with pytest.raises(UserError, match='must be attached to an agent before its tasks are used'):
        HatchetDurability(app.hatchet).get_durable_operation_backend()


def test_stream_agent_events_needs_one_agent(app: App):
    """The stream to read is the agent's, so the capability must be attached to one agent, or be given a `name`."""
    unattached = HatchetDurability(app.hatchet, event_stream=True)
    with pytest.raises(UserError, match='must be attached to an agent before its events can be streamed'):
        unattached.stream_agent_events('run-id')
    shared = HatchetDurability(app.hatchet, event_stream=True)
    Agent(TestModel(), name='first', capabilities=[shared])
    Agent(TestModel(), name='second', capabilities=[shared])
    with pytest.raises(UserError, match=r"attached to agents \['first', 'second'\]"):
        shared.stream_agent_events('run-id')


def test_stream_agent_events_of_named_capability(app: App):
    """A capability given a `name` reads that agent's stream without being attached to it."""
    assert HatchetDurability(app.hatchet, event_stream=True, name='named').stream_agent_events('run-id').cursor is None


def test_deps_type_given_to_the_capability(app: App):
    """A `deps_type` given to the capability is what `deps` cross a task boundary as."""
    agent = Agent(
        TestModel(),
        name='deps_type',
        deps_type=Deps,
        capabilities=[HatchetDurability[Deps](app.hatchet, deps_type=Deps)],
    )
    durability = next(cap for cap in leaf_capabilities(agent.root_capability) if isinstance(cap, HatchetDurability))
    assert durability.load_deps(durability.dump_deps(Deps(city_prefix='x'))) == Deps(city_prefix='x')


def test_non_retryable_model_errors_are_ignored_with_retry_on_model_errors(app: App):
    """`RetryOnModelErrors` does the retrying of model errors, so a custom `non_retryable_model_errors` is ignored."""
    durability = HatchetDurability(app.hatchet, non_retryable_model_errors=(ModelHTTPError,))
    with pytest.warns(UserWarning, match='`non_retryable_model_errors` is ignored'):
        Agent(TestModel(), name='ignored_errors', capabilities=[RetryOnModelErrors(), durability])


async def test_provider_retry_after_is_waited_before_the_retry(app: App):
    """A model error asking for a wait fails its task with Hatchet's `RetryAfterError`, and the retry recovers."""
    assert await _run(app, 'retry_after') == {'output': 'recovered'}
    assert app.scenarios['retry_after'].calls == ['call', 'call']


async def test_retry_on_model_errors_waits_on_a_durable_sleep(app: App):
    """`RetryOnModelErrors` waits between attempts in the durable task, which a replay doesn't wait again."""
    assert await _run(app, 'durable_sleep') == {'output': 'recovered'}
    assert app.scenarios['durable_sleep'].calls == ['call', 'call']


async def test_tools_of_capability_without_id_replay(app: App):
    """A tool from a capability without an `id` crosses under a key a replay reproduces, so its result is replayed."""
    scenario = app.scenarios['anonymous_capability']
    scenario.calls.clear()
    ref = await scenario.run.aio_run(Prompt(prompt='hi'), wait_for_result=False)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        first = await ref.aio_result()
    details = await app.hatchet.runs.aio_get_details(ref.workflow_run_id)
    (task_run,) = details.task_runs.values()
    log = await _durable_event_log(app.hatchet, task_run.external_id)
    await app.hatchet.runs.aio_reset_durable_task(
        task_run.external_id, node_id=max(entry.node_id for entry in log) + 1, branch_id=1
    )
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        outputs = await app.hatchet.runs.get_run_ref(ref.workflow_run_id).aio_result()
    assert outputs['pydantic_ai_test_anonymous_capability'] == first == {'output': '{"note":"noted"}'}
    assert scenario.calls == ['note']


async def test_run_cancelled_from_outside_publishes_no_outcome(app: App):
    """A run cancelled by its durable task, not by itself, publishes no error: Hatchet may yet resume an evicted task.

    The durable task runs the agent again afterwards, on the same topic, so the subscription ends with that run's
    result, which it would not if the cancelled run had published an error.
    """
    scenario = app.streams['externally_cancelled']
    ref = await scenario.run.aio_run(Prompt(prompt='hi'), wait_for_result=False)
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        assert await ref.aio_result() == {'cancelled': True, 'output': 'done'}
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        events = scenario.durability.stream_agent_events(ref.workflow_run_id)
        async for _event in events:
            pass
    assert events.error is None
    assert events.result is not None
    assert events.result.output == 'done'


def test_one_capability_on_agents_of_one_name_streams_that_name(app: App):
    """Agents of the same name share a topic, so a capability attached to each still knows whose stream to read."""
    shared = HatchetDurability(app.hatchet, event_stream=True)
    Agent(TestModel(), name='twin', capabilities=[shared])
    Agent(TestModel(), name='twin', capabilities=[shared])
    assert shared.stream_agent_events('run-id').cursor is None


def test_context_rebuilt_in_a_task_has_no_capability_keys(app: App):
    """A run context rebuilt inside a task has no capability registry, so reading one says what's missing.

    A unit test: no task dumps the context it was handed into another task, so no run reaches the empty
    replacements through a task boundary.
    """
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), run_id='run')
    rebuilt = HatchetRunContext.deserialize_run_context(HatchetRunContext.serialize_run_context(ctx), deps=None)
    with pytest.raises(UserError, match="is not available on 'HatchetRunContext' inside a Hatchet task"):
        rebuilt.capabilities
    assert stable_capability_ids(rebuilt) == {}


def _bound_durability(app: App, agent_name: str) -> HatchetDurability[Any]:
    agent = Agent(TestModel(), name=agent_name, capabilities=[HatchetDurability(app.hatchet)])
    return next(cap for cap in leaf_capabilities(agent.root_capability) if isinstance(cap, HatchetDurability))


async def test_tool_removed_during_the_run_fails_its_task(app: App):
    """A tool call whose tool the worker's toolset no longer has fails with a `UserError` saying why.

    A unit test: the worker and the durable task share one process here, so their toolsets can't differ in a run.
    """
    transports = HatchetParameterTransports(_bound_durability(app, 'removed_tool'))
    before = FunctionToolset[Any]([double], id='tools')
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    tool = (await before.get_tools(ctx))['double']
    payload = transports.function_call(before).dump(
        ToolsetCallToolParams('double', tool_args={'x': 1}, ctx=ctx, tool=tool)
    )
    with pytest.raises(UserError, match="Tool 'double' not found in toolset 'tools'"):
        transports.function_call(FunctionToolset[Any](id='tools')).load(payload, runtime=None)


def test_cancel_without_run_context_crosses_without_one(app: App):
    """Cancelling a suspended response needs no run context, so one without crosses into its task as is.

    A unit test: a run always cancels with its context, so only a direct call crosses without one.
    """
    transport = HatchetParameterTransports(_bound_durability(app, 'cancel_without_context')).cancel_suspended_response()
    response = ModelResponse(parts=[TextPart('suspended')], provider_response_id='job')
    params = ModelCancelSuspendedResponseParams('model', response=response, run_context=None)
    loaded = transport.load(transport.dump(params), runtime=None)
    assert (loaded.model_id, loaded.response, loaded.run_context) == ('model', response, None)


@pytest.mark.parametrize(
    'message',
    [
        pytest.param('pydantic_ai_error:{"kind": "user_error"', id='cut-short'),
        pytest.param('pydantic_ai_error:{"kind": "unknown"}', id='unknown-kind'),
    ],
)
def test_task_failure_that_cannot_be_rebuilt_crosses_as_itself(message: str):
    """A task failure whose encoded error is cut short or of a kind it doesn't know is raised as Hatchet's own error.

    A unit test: every task encodes its errors in full, with a kind this version rebuilds.
    """
    error = TaskRunError(message, 'NonRetryableError', '', None)
    assert _rebuild_error(error) is None


def test_operation_unknown_to_this_version_has_no_summary_metadata():
    """An operation kind added to core after this backend gets no dashboard metadata rather than failing.

    A unit test: the union of operation IDs has no kind this backend doesn't summarize yet.
    """
    assert _summary_metadata(cast(Any, object()), None) == {}


def test_error_other_than_serialization_is_left_as_is():
    """Only a failure to serialize a task's input or output is replaced by a `UserError` explaining JSON payloads.

    A unit test: the serializers only raise serialization errors.
    """
    error = ValueError('unrelated')
    assert _serialization_failure(error) is error


async def test_model_stream_outside_a_task_passes_through(app: App):
    """Outside a task spawned by a durable task there's no run topic, so the model stream reaches the handler as is.

    A unit test: a durable task always runs its model requests in tasks it spawns.
    """
    received: list[AgentStreamEvent] = []

    async def record(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        received.extend([event async for event in stream])

    async def events() -> AsyncIterator[AgentStreamEvent]:
        yield FinalResultEvent(tool_name=None, tool_call_id=None)

    handler = task_event_stream_handler(app.hatchet, HatchetEventStream(), 'outside', handler=record)
    await handler(RunContext[None](deps=None, model=TestModel(), usage=RunUsage()), events())
    assert received == [FinalResultEvent(tool_name=None, tool_call_id=None)]
