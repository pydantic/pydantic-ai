"""`CodeMode` says on the `run_code` call's `execute_tool` span why the call failed."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_monty import AsyncSnapshot, MontyComplete

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness._monty_exec import MontyExecutor
from pydantic_ai_harness.code_mode import CodeMode
from pydantic_ai_harness.code_mode._eager import EagerCodeModeToolset


def instrumented() -> tuple[Instrumentation, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return Instrumentation(settings=InstrumentationSettings(tracer_provider=provider)), exporter


def code_mode_attributes(exporter: InMemorySpanExporter) -> list[tuple[str, dict[str, Any]]]:
    """Each span with `code_mode.*` attributes, as its name and those attributes, in the order the spans ended."""
    found = [
        (span.name, {key: value for key, value in (span.attributes or {}).items() if key.startswith('code_mode.')})
        for span in exporter.get_finished_spans()
    ]
    return [(name, attributes) for name, attributes in found if attributes]


def run_code_snippets(*snippets: str) -> FunctionModel:
    """A model that sends each snippet to `run_code` in turn, then answers."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent = len(messages) // 2
        if sent == len(snippets):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': snippets[sent]})])

    return FunctionModel(respond)


async def failure_attributes(
    *snippets: str, capability: CodeMode[object] | None = None
) -> list[tuple[str, dict[str, Any]]]:
    instrumentation, exporter = instrumented()
    agent = Agent(run_code_snippets(*snippets), capabilities=[capability or CodeMode[object](), instrumentation])

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


async def diverged_prefix_attributes(original: str, rewritten: str) -> list[tuple[str, dict[str, Any]]]:
    """Stream `original` to an eager `run_code` call, then dispatch it as `rewritten`, which drops the fed prefix."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('test')
    capability = CodeMode[None](eager=True)
    root = capability.get_wrapper_toolset(FunctionToolset[None]())
    assert root is not None
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
    async with root:
        manager = await ToolManager(toolset=root).for_run_step(ctx)
        ctx.tool_manager = manager
        assert manager.tools is not None
        run_code = manager.tools['run_code']
        toolset = run_code.toolset
        assert isinstance(toolset, EagerCodeModeToolset)

        async def observe(*events: PartStartEvent | PartDeltaEvent) -> None:
            async def stream() -> AsyncIterator[PartStartEvent | PartDeltaEvent]:
                for event in events:
                    yield event

            async for _ in capability.wrap_run_event_stream(ctx, stream=stream()):
                pass

        part = ToolCallPart(tool_name='run_code', args={'code': original}, tool_call_id='c1')
        await observe(PartStartEvent(index=0, part=part))
        await asyncio.gather(*toolset.execution.pumps)
        await observe(
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': rewritten}, tool_call_id='c1'))
        )

        exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
        with (
            tracer.start_as_current_span('execute_tool run_code'),
            pytest.raises(ModelRetry, match='no longer matches'),
        ):
            await toolset.call_tool('run_code', {'code': rewritten}, exec_ctx, run_code)
    return code_mode_attributes(exporter)


def run_code_span(*, budget_exhausted: bool = False, **failure: Any) -> tuple[str, dict[str, Any]]:
    """An `execute_tool run_code` span as `code_mode_attributes` lists it, with `failure` as `code_mode.*` keys."""
    attributes = {f'code_mode.{key}': value for key, value in failure.items()}
    return ('execute_tool run_code', {'code_mode.budget_exhausted': budget_exhausted, **attributes})


class TestRunCodeFailureTelemetry:
    async def test_a_successful_call_records_only_the_budget(self):
        assert await failure_attributes('await add(a=1, b=2)') == [run_code_span()]

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
        self, snippets: tuple[str, ...], capability: CodeMode[object] | None, expected: list[tuple[str, dict[str, Any]]]
    ):
        assert await failure_attributes(*snippets, capability=capability) == expected

    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            pytest.param(CrashedError(timed_out=False), {'error_kind': 'crashed'}, id='crashed'),
            pytest.param(
                CrashedError(timed_out=True),
                {'error_kind': 'resource_limit', 'limit': 'max_duration_secs'},
                id='worker-timed-out',
            ),
            pytest.param(
                DisconnectError(cause='turn_timeout'),
                {'error_kind': 'resource_limit', 'limit': 'max_duration_secs'},
                id='turn-timeout',
            ),
            pytest.param(DisconnectError(cause=None), {'error_kind': 'disconnected'}, id='disconnected'),
            pytest.param(RuntimeError('host binding failed'), {'error_kind': 'host'}, id='host'),
            pytest.param(PanicException('sandbox aborted'), {'error_kind': 'panic'}, id='panic'),
        ],
    )
    async def test_a_failure_outside_the_snippet_is_recorded_by_kind(
        self, monkeypatch: pytest.MonkeyPatch, error: BaseException, expected: dict[str, Any]
    ):
        """The second snippet fails, so the reset discards the state the first one left."""
        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset.MontyCrashedError', CrashedError)
        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset.MontyDisconnectError', DisconnectError)
        original_run = MontyExecutor.run
        feeds: list[None] = []

        async def fail_second_feed(
            self: MontyExecutor, feed_start: Callable[[], Awaitable[AsyncSnapshot]]
        ) -> MontyComplete:
            feeds.append(None)
            if len(feeds) == 1:
                return await original_run(self, feed_start)
            raise error

        monkeypatch.setattr(MontyExecutor, 'run', fail_second_feed)

        assert await failure_attributes('x = 1', '1 + 1') == [
            run_code_span(),
            run_code_span(session_reset=True, **expected),
        ]

    async def test_a_failure_before_any_code_ran_discards_no_executed_state(self, monkeypatch: pytest.MonkeyPatch):
        async def fail(self: MontyExecutor, feed_start: Callable[[], Awaitable[AsyncSnapshot]]) -> MontyComplete:
            raise RuntimeError('host binding failed')

        monkeypatch.setattr(MontyExecutor, 'run', fail)

        assert await failure_attributes('1 + 1') == [run_code_span(error_kind='host', session_reset=False)]

    async def test_a_streamed_statement_that_failed_early_is_recorded_on_run_code(self):
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
                await asyncio.sleep(0)

        instrumentation, exporter = instrumented()
        agent = Agent(
            FunctionModel(stream_function=stream_once), capabilities=[CodeMode[object](eager=True), instrumentation]
        )

        await agent.run('go')

        assert code_mode_attributes(exporter) == [run_code_span(error_kind='runtime', session_reset=False)]

    async def test_a_diverged_prefix_records_the_failed_statement_and_the_restart(self):
        """The failure kept the session, but the restart discards the one `x = 1` ran in."""
        assert await diverged_prefix_attributes('x = 1\nboom = 1 // 0\n"unused"\n', 'y = 2\ny') == [
            run_code_span(error_kind='runtime', session_reset=True)
        ]

    async def test_a_diverged_prefix_keeps_the_reset_a_failure_already_made(self, monkeypatch: pytest.MonkeyPatch):
        """The host failure discards the session first, so the restart finds nothing left to discard."""
        original_run = MontyExecutor.run
        feeds: list[None] = []

        async def fail_second_feed(
            self: MontyExecutor, feed_start: Callable[[], Awaitable[AsyncSnapshot]]
        ) -> MontyComplete:
            feeds.append(None)
            if len(feeds) == 1:
                return await original_run(self, feed_start)
            raise RuntimeError('host binding failed')

        monkeypatch.setattr(MontyExecutor, 'run', fail_second_feed)

        assert await diverged_prefix_attributes('x = 1\n1 + 1\n"unused"\n', 'y = 2\ny') == [
            run_code_span(error_kind='host', session_reset=True)
        ]
