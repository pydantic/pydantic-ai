"""`DynamicWorkflow` says on the `run_workflow` call's `execute_tool` span how the script ended."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_monty import AsyncSnapshot, MontyComplete

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness._monty_exec import MontyExecutor
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow


def run_workflow_once(code: str) -> FunctionModel:
    """A model that sends `code` to `run_workflow` once, then answers whatever came back."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='run_workflow', args={'code': code})])

    return FunctionModel(respond)


async def workflow_attributes(code: str, **workflow_options: Any) -> list[tuple[str, dict[str, Any]]]:
    """Each span with `dynamic_workflow.*` attributes after running `code`, as its name and those attributes."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    sub = Agent(TestModel(custom_output_text='ok'), name='sub')
    workflow = DynamicWorkflow[object](agents=[sub], **workflow_options)
    instrumentation = Instrumentation(settings=InstrumentationSettings(tracer_provider=provider))

    await Agent(run_workflow_once(code), capabilities=[workflow, instrumentation]).run('go')

    found = [
        (
            span.name,
            {key: value for key, value in (span.attributes or {}).items() if key.startswith('dynamic_workflow.')},
        )
        for span in exporter.get_finished_spans()
    ]
    return [(name, attributes) for name, attributes in found if attributes]


class PanicException(BaseException):
    """Named to match the pyo3 panic class `is_sandbox_panic` recognizes."""


class CrashedError(Exception):
    """Stands in for `MontyCrashedError`, which Python cannot construct."""


class TestDynamicWorkflowTelemetry:
    async def test_a_completed_script_records_only_the_budget(self):
        assert await workflow_attributes("await sub(task='x')") == [
            ('execute_tool run_workflow', {'dynamic_workflow.budget_exhausted': False})
        ]

    @pytest.mark.parametrize(
        ('code', 'workflow_options', 'expected'),
        [
            # The type checker parses the script first, so a syntax error is reported as a type error too.
            pytest.param('await sub(1)', {}, {'dynamic_workflow.error_kind': 'type'}, id='type'),
            pytest.param('1 / 0', {}, {'dynamic_workflow.error_kind': 'runtime'}, id='runtime'),
            pytest.param(
                'x = [0] * 50_000_000',
                {'resource_limits': {'max_memory': 8 * 1024 * 1024}},
                {'dynamic_workflow.error_kind': 'resource_limit', 'dynamic_workflow.limit': 'max_memory'},
                id='max_memory',
            ),
        ],
    )
    async def test_a_failed_script_is_recorded_by_kind(
        self, code: str, workflow_options: dict[str, Any], expected: dict[str, Any]
    ):
        assert await workflow_attributes(code, **workflow_options) == [
            ('execute_tool run_workflow', {**expected, 'dynamic_workflow.budget_exhausted': False})
        ]

    @pytest.mark.parametrize(
        ('error', 'kind'),
        [
            pytest.param(CrashedError(), 'crashed', id='crashed'),
            pytest.param(PanicException('sandbox aborted'), 'panic', id='panic'),
        ],
    )
    async def test_a_sandbox_failure_is_recorded_by_kind(
        self, monkeypatch: pytest.MonkeyPatch, error: BaseException, kind: str
    ):
        monkeypatch.setattr('pydantic_ai_harness.dynamic_workflow._toolset.MontyCrashedError', CrashedError)

        async def fail(self: MontyExecutor, feed_start: Callable[[], Awaitable[AsyncSnapshot]]) -> MontyComplete:
            raise error

        monkeypatch.setattr(MontyExecutor, 'run', fail)

        assert await workflow_attributes('1 + 1') == [
            (
                'execute_tool run_workflow',
                {'dynamic_workflow.error_kind': kind, 'dynamic_workflow.budget_exhausted': False},
            )
        ]

    async def test_a_script_that_fails_on_the_budget_records_both(self):
        assert await workflow_attributes("await sub(task='a')\nawait sub(task='b')", max_agent_calls=1) == [
            (
                'execute_tool run_workflow',
                {'dynamic_workflow.error_kind': 'runtime', 'dynamic_workflow.budget_exhausted': True},
            )
        ]

    async def test_a_script_that_catches_the_budget_error_still_records_it(self):
        code = "await sub(task='a')\ntry:\n    await sub(task='b')\nexcept RuntimeError:\n    pass"

        assert await workflow_attributes(code, max_agent_calls=1) == [
            ('execute_tool run_workflow', {'dynamic_workflow.budget_exhausted': True})
        ]
