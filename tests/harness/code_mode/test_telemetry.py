"""`CodeMode` says on the `run_code` call's `execute_tool` span why the call failed."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

import anyio.lowlevel
import pytest
from inline_snapshot import snapshot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai._telemetry import RunTelemetry, run_telemetry_scope
from pydantic_ai.exceptions import ModelRetry, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    PartStartEvent,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness.code_mode import CodeMode, CodeModeToolset
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.harness.conftest import Instrumented

# Patched to make a feed fail outside the snippet the way a dead worker or a host-binding bug does.
MONTY_EXECUTOR_RUN = 'pydantic_ai_harness._monty_exec.MontyExecutor.run'


def code_mode_attributes(exporter: InMemorySpanExporter) -> list[tuple[str, dict[str, AttributeValue]]]:
    """Each span with `code_mode.*` attributes, as its name and those attributes, in the order the spans ended."""
    found = [
        (span.name, {key: value for key, value in (span.attributes or {}).items() if key.startswith('code_mode.')})
        for span in exporter.get_finished_spans()
    ]
    return [(name, attributes) for name, attributes in found if attributes]


def run_code_snippets(*snippets: str, before_sending: Callable[[int], None] | None = None) -> FunctionModel:
    """A model that sends each snippet to `run_code` in turn, then answers.

    `before_sending` is called with each snippet's index just before the model sends it.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent = len(messages) // 2
        if sent == len(snippets):
            return ModelResponse(parts=[TextPart('done')])
        if before_sending is not None:
            before_sending(sent)
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': snippets[sent]})])

    return FunctionModel(respond)


async def failure_attributes(
    *snippets: str,
    capability: CodeMode[object] | None = None,
    before_sending: Callable[[int], None] | None = None,
    instrumented: Instrumented,
) -> list[tuple[str, dict[str, AttributeValue]]]:
    instrumentation, exporter = instrumented(False)
    agent = Agent(
        run_code_snippets(*snippets, before_sending=before_sending),
        capabilities=[capability or CodeMode[object](), instrumentation],
    )

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    await agent.run('go')
    return code_mode_attributes(exporter)


class PanicException(BaseException):
    """Named to match the pyo3 panic class `is_sandbox_panic` recognizes."""


@dataclass
class CrashedError(Exception):
    """Stands in for `MontyCrashedError`, which Python cannot construct."""

    timed_out: bool


@dataclass
class DisconnectError(Exception):
    """Stands in for `MontyDisconnectError`, which Python cannot construct."""

    cause: str | None


async def diverged_prefix_attributes(
    original: str, rewritten: str, *, ready: Callable[[], object], sync_code: str, capability: CodeMode[None]
) -> list[tuple[str, dict[str, AttributeValue]]]:
    """Stream `original` to an eager `run_code` call, let its statements run, then dispatch it as `rewritten`.

    `original`'s last fed statement calls `ready`, which must set the event the helper waits on, and
    then fails in that same statement. A second `run_code` call in the response (`sync_code`) is
    never fed eagerly, and its dispatch waits for the first call's in-flight statement, so
    dispatching it before the first call guarantees the failing statement finished first.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('test')
    started = asyncio.Event()

    async def ready_tool() -> object:
        started.set()
        return ready()

    root = capability.get_wrapper_toolset(FunctionToolset[None](tools=[Tool(ready_tool, name='ready')]))
    assert isinstance(root, CodeModeToolset)
    ctx = RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        pending_messages=[],
        tracer=tracer,
    )
    # A real run publishes its telemetry before any tool runs; this hand-built run does it here.
    with run_telemetry_scope(RunTelemetry(tracer=tracer, include_content=ctx.trace_include_content)):
        async with root:
            manager = await ToolManager(toolset=root).for_run_step(ctx)
            ctx.tool_manager = manager
            assert manager.tools is not None
            run_code = manager.tools['run_code']
            toolset = run_code.toolset
            assert isinstance(toolset, CodeModeToolset)

            async def stream() -> AsyncIterator[PartStartEvent]:
                yield PartStartEvent(index=0, part=ToolCallPart('run_code', {'code': original}, tool_call_id='c1'))
                yield PartStartEvent(index=1, part=ToolCallPart('run_code', {'code': sync_code}, tool_call_id='c2'))

            async for _ in capability.wrap_run_event_stream(ctx, stream=stream()):
                pass
            await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
            # Dispatched outside any span, so its own outcome is not recorded.
            sync_ctx = dataclasses.replace(ctx, tool_call_id='c2', tool_name='run_code')
            try:
                await toolset.call_tool('run_code', {'code': sync_code}, sync_ctx, run_code)
            except ModelRetry:
                pass

            exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
            with (
                tracer.start_as_current_span('execute_tool run_code'),
                pytest.raises(ModelRetry, match='no longer matches'),
            ):
                await toolset.call_tool('run_code', {'code': rewritten}, exec_ctx, run_code)
    return code_mode_attributes(exporter)


def run_code_span(*, budget_exhausted: bool = False, **failure: str | bool) -> tuple[str, dict[str, AttributeValue]]:
    """An `execute_tool run_code` span as `code_mode_attributes` lists it, with `failure` as `code_mode.*` keys."""
    attributes = {f'code_mode.{key}': value for key, value in failure.items()}
    return ('execute_tool run_code', {'code_mode.budget_exhausted': budget_exhausted, **attributes})


class TestRunCodeFailureTelemetry:
    async def test_a_successful_call_records_only_the_budget(self, instrumented: Instrumented):
        assert await failure_attributes('await add(a=1, b=2)', instrumented=instrumented) == [run_code_span()]

    @pytest.mark.parametrize(
        ('snippets', 'capability', 'expected'),
        [
            pytest.param(
                # Only the first snippet is type-checked, so a later one's syntax error comes from the parser.
                ('x = 1', 'def broken(:\n    pass'),
                None,
                [run_code_span(), run_code_span(error_kind='syntax', session_reset=False)],
                id='syntax',
            ),
            pytest.param(
                ("await add(a='one', b=2)",),
                None,
                # The reset drops only the type-check stubs: no code has run yet.
                [run_code_span(error_kind='type_check', session_reset=False)],
                id='type_check',
            ),
            pytest.param(
                ('def broken(:\n    pass',),
                None,
                [run_code_span(error_kind='type_check', session_reset=False)],
                id='syntax-in-a-type-checked-snippet',
            ),
            pytest.param(
                ('1 / 0',),
                None,
                [run_code_span(error_kind='runtime', session_reset=False)],
                id='runtime',
            ),
            pytest.param(
                ('x = [0] * 50_000_000',),
                CodeMode[object](resource_limits={'max_memory': 8 * 1024 * 1024}),
                [run_code_span(error_kind='resource_limit', session_reset=True, limit='max_memory')],
                id='max_memory',
            ),
            pytest.param(
                ('x = 0\nwhile True:\n    x += 1',),
                CodeMode[object](resource_limits={'max_duration_seconds': 0.01}),
                # The first snippet ran, so the reset loses `x` even though no earlier snippet ran.
                [run_code_span(error_kind='resource_limit', session_reset=True, limit='max_duration_seconds')],
                id='max_duration_seconds-in-the-first-snippet',
            ),
            pytest.param(
                ('await add(a=1, b=2)',),
                CodeMode[object](resource_limits={'max_suspensions': 1}),
                [run_code_span(error_kind='resource_limit', session_reset=False, limit='max_suspensions')],
                id='max_suspensions',
            ),
            pytest.param(
                ('await add(a=1, b=2)\nawait add(a=3, b=4)',),
                CodeMode[object](max_tool_calls=1),
                [run_code_span(budget_exhausted=True, error_kind='runtime', session_reset=False)],
                id='max_tool_calls',
            ),
            pytest.param(
                ('await add(a=1, b=2)\ntry:\n    await add(a=3, b=4)\nexcept RuntimeError:\n    pass',),
                CodeMode[object](max_tool_calls=1),
                [run_code_span(budget_exhausted=True)],
                id='max_tool_calls-caught',
            ),
        ],
    )
    async def test_a_sandbox_failure_is_recorded_by_kind(
        self,
        snippets: tuple[str, ...],
        capability: CodeMode[object] | None,
        expected: list[tuple[str, dict[str, AttributeValue]]],
        instrumented: Instrumented,
    ):
        assert await failure_attributes(*snippets, capability=capability, instrumented=instrumented) == expected

    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            pytest.param(
                CrashedError(timed_out=False), run_code_span(session_reset=True, error_kind='crashed'), id='crashed'
            ),
            pytest.param(
                CrashedError(timed_out=True),
                run_code_span(session_reset=True, error_kind='resource_limit', limit='max_duration_seconds'),
                id='worker-timed-out',
            ),
            pytest.param(
                DisconnectError(cause='turn_timeout'),
                run_code_span(session_reset=True, error_kind='resource_limit', limit='max_duration_seconds'),
                id='turn-timeout',
            ),
            pytest.param(
                DisconnectError(cause=None),
                run_code_span(session_reset=True, error_kind='disconnected'),
                id='disconnected',
            ),
            pytest.param(
                RuntimeError('host binding failed'), run_code_span(session_reset=True, error_kind='host'), id='host'
            ),
            pytest.param(
                PanicException('sandbox aborted'), run_code_span(session_reset=True, error_kind='panic'), id='panic'
            ),
        ],
    )
    async def test_a_failure_outside_the_snippet_is_recorded_by_kind(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: BaseException,
        expected: tuple[str, dict[str, AttributeValue]],
        instrumented: Instrumented,
    ):
        """The second snippet fails, so the reset discards the state the first one left."""
        monkeypatch.setattr('pydantic_ai_harness._monty_turns.MontyCrashedError', CrashedError)
        monkeypatch.setattr('pydantic_ai_harness._monty_turns.MontyDisconnectError', DisconnectError)

        async def fail(self: object, feed_start: object) -> object:
            raise error

        def fail_the_second_snippet(index: int) -> None:
            if index == 1:
                monkeypatch.setattr(MONTY_EXECUTOR_RUN, fail)

        assert await failure_attributes(
            'x = 1', '1 + 1', before_sending=fail_the_second_snippet, instrumented=instrumented
        ) == [run_code_span(), expected]

    async def test_a_streamed_statement_that_failed_early_is_recorded_on_run_code(self, instrumented: Instrumented):
        """Eager execution runs statements before `run_code`'s span opens; the failure still lands on that span."""
        code = 'boom = 1 // 0\n"unused"'
        args = json.dumps({'code': code})

        async def stream_once(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) > 1:
                yield 'done'
                return
            yield {1: DeltaToolCall(name='run_code')}
            for offset in range(0, len(args), 8):
                yield {1: DeltaToolCall(json_args=args[offset : offset + 8])}
                await anyio.lowlevel.checkpoint()

        instrumentation, exporter = instrumented(False)
        agent = Agent(
            FunctionModel(stream_function=stream_once), capabilities=[CodeMode[object](eager=True), instrumentation]
        )

        await agent.run('go')

        assert code_mode_attributes(exporter) == [run_code_span(error_kind='runtime', session_reset=False)]

    @pytest.mark.parametrize('eager', [False, True], ids=['sequential', 'eager'])
    async def test_a_call_ended_by_a_usage_limit_records_only_the_budget(self, eager: bool, instrumented: Instrumented):
        """Core's tool call limit ends the run inside a nested call, so `run_code` records no failure of its own."""
        code = 'await add(a=1, b=2)\nawait add(a=3, b=4)'
        args = json.dumps({'code': code})

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code})])

        async def stream_once(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            yield {1: DeltaToolCall(name='run_code')}
            for offset in range(0, len(args), 8):
                yield {1: DeltaToolCall(json_args=args[offset : offset + 8])}
                await anyio.lowlevel.checkpoint()

        instrumentation, exporter = instrumented(False)
        agent = Agent(
            FunctionModel(respond, stream_function=stream_once),
            capabilities=[CodeMode[object](eager=eager), instrumentation],
        )

        @agent.tool_plain
        def add(a: int, b: int) -> int:
            return a + b

        with pytest.raises(UsageLimitExceeded):
            await agent.run('go', usage_limits=UsageLimits(tool_calls_limit=2))

        assert code_mode_attributes(exporter) == snapshot(
            [('execute_tool run_code', {'code_mode.budget_exhausted': False})]
        )

    async def test_a_diverged_prefix_records_the_failed_statement_and_the_restart(self):
        """The failure kept the session, but the restart discards the one `x = 1` ran in."""
        assert await diverged_prefix_attributes(
            'x = 1\nboom = await ready() // 0\n"unused"\n',
            'y = 2\ny',
            ready=lambda: 1,
            sync_code='None',
            capability=CodeMode[None](eager=True),
        ) == [run_code_span(error_kind='runtime', session_reset=True)]
