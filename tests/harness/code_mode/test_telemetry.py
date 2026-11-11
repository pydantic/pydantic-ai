"""`CodeMode` says on the `run_code` call's `execute_tool` span why the call failed."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest
from inline_snapshot import snapshot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_monty import AsyncSnapshot, MontyComplete

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai_harness._monty_exec import MontyExecutor
from pydantic_ai_harness.code_mode import CodeMode


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


class TestRunCodeFailureTelemetry:
    async def test_a_successful_call_records_nothing(self):
        assert await failure_attributes('await add(a=1, b=2)') == []

    @pytest.mark.parametrize(
        ('snippets', 'capability', 'expected'),
        [
            pytest.param(
                # The first snippet is type-checked, which reports a syntax error as a type error.
                ('x = 1', 'def broken(:\n    pass'),
                None,
                {'code_mode.error_kind': 'syntax', 'code_mode.session_reset': False},
                id='syntax',
            ),
            pytest.param(
                ("await add(a='one', b=2)",),
                None,
                {'code_mode.error_kind': 'type', 'code_mode.session_reset': True},
                id='type',
            ),
            pytest.param(
                ('1 / 0',),
                None,
                {'code_mode.error_kind': 'runtime', 'code_mode.session_reset': False},
                id='runtime',
            ),
            pytest.param(
                ('x = [0] * 50_000_000',),
                CodeMode[object](resource_limits={'max_memory': 8 * 1024 * 1024}),
                {
                    'code_mode.error_kind': 'resource_limit',
                    'code_mode.session_reset': True,
                    'code_mode.limit': 'max_memory',
                },
                id='max_memory',
            ),
            pytest.param(
                ('await add(a=1, b=2)',),
                CodeMode[object](resource_limits={'max_suspensions': 1}),
                {
                    'code_mode.error_kind': 'resource_limit',
                    'code_mode.session_reset': False,
                    'code_mode.limit': 'max_suspensions',
                },
                id='max_suspensions',
            ),
            pytest.param(
                ('await add(a=1, b=2)\nawait add(a=3, b=4)',),
                CodeMode[object](max_tool_calls=1),
                {
                    'code_mode.error_kind': 'resource_limit',
                    'code_mode.session_reset': False,
                    'code_mode.limit': 'max_tool_calls',
                },
                id='max_tool_calls',
            ),
        ],
    )
    async def test_a_sandbox_failure_is_recorded_by_kind(
        self, snippets: tuple[str, ...], capability: CodeMode[object] | None, expected: dict[str, Any]
    ):
        assert await failure_attributes(*snippets, capability=capability) == [('execute_tool run_code', expected)]

    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            pytest.param(
                CrashedError(timed_out=False),
                {'code_mode.error_kind': 'crashed', 'code_mode.session_reset': True},
                id='crashed',
            ),
            pytest.param(
                CrashedError(timed_out=True),
                {
                    'code_mode.error_kind': 'resource_limit',
                    'code_mode.session_reset': True,
                    'code_mode.limit': 'max_duration_secs',
                },
                id='worker-timed-out',
            ),
            pytest.param(
                DisconnectError(cause='turn_timeout'),
                {
                    'code_mode.error_kind': 'resource_limit',
                    'code_mode.session_reset': True,
                    'code_mode.limit': 'max_duration_secs',
                },
                id='turn-timeout',
            ),
            pytest.param(
                DisconnectError(cause=None),
                {'code_mode.error_kind': 'disconnected', 'code_mode.session_reset': True},
                id='disconnected',
            ),
            pytest.param(
                RuntimeError('host binding failed'),
                {'code_mode.error_kind': 'host', 'code_mode.session_reset': True},
                id='host',
            ),
            pytest.param(
                PanicException('sandbox aborted'),
                {'code_mode.error_kind': 'panic', 'code_mode.session_reset': True},
                id='panic',
            ),
        ],
    )
    async def test_a_failure_outside_the_snippet_is_recorded_by_kind(
        self, monkeypatch: pytest.MonkeyPatch, error: BaseException, expected: dict[str, Any]
    ):
        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset.MontyCrashedError', CrashedError)
        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset.MontyDisconnectError', DisconnectError)

        async def fail(self: MontyExecutor, feed_start: Callable[[], Awaitable[AsyncSnapshot]]) -> MontyComplete:
            raise error

        monkeypatch.setattr(MontyExecutor, 'run', fail)

        assert await failure_attributes('1 + 1') == [('execute_tool run_code', expected)]

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

        assert code_mode_attributes(exporter) == snapshot(
            [('execute_tool run_code', {'code_mode.error_kind': 'runtime', 'code_mode.session_reset': False})]
        )
