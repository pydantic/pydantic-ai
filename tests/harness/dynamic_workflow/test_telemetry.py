"""`DynamicWorkflow` says on the `run_workflow` call's `execute_tool` span how the script ended."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from opentelemetry.util.types import AttributeValue

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow
from tests.harness.conftest import Instrumented

SUB = Agent(TestModel(custom_output_text='ok'), name='sub')


def run_workflow_once(code: str) -> FunctionModel:
    """A model that sends `code` to `run_workflow` once, then answers whatever came back."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='run_workflow', args={'code': code})])

    return FunctionModel(respond)


async def workflow_attributes(
    code: str,
    *,
    instrumented: Instrumented,
    workflow: DynamicWorkflow[object] | None = None,
) -> list[tuple[str, dict[str, AttributeValue]]]:
    """Each span with `dynamic_workflow.*` attributes after running `code`, as its name and those attributes."""
    instrumentation, exporter = instrumented(False)
    workflow = workflow or DynamicWorkflow[object](agents=[SUB])

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


@dataclass
class CrashedError(Exception):
    """Stands in for `MontyCrashedError`, which Python cannot construct."""

    timed_out: bool


class TestDynamicWorkflowTelemetry:
    async def test_a_completed_script_records_only_the_budget(self, instrumented: Instrumented):
        assert await workflow_attributes("await sub(task='x')", instrumented=instrumented) == [
            ('execute_tool run_workflow', {'dynamic_workflow.budget_exhausted': False})
        ]

    @pytest.mark.parametrize(
        ('code', 'workflow', 'expected'),
        [
            pytest.param('await sub(1)', None, {'dynamic_workflow.error_kind': 'type_check'}, id='type_check'),
            pytest.param('1 / 0', None, {'dynamic_workflow.error_kind': 'runtime'}, id='runtime'),
            pytest.param(
                'x = [0] * 50_000_000',
                DynamicWorkflow[object](agents=[SUB], resource_limits={'max_memory': 8 * 1024 * 1024}),
                {'dynamic_workflow.error_kind': 'resource_limit', 'dynamic_workflow.limit': 'max_memory'},
                id='max_memory',
            ),
        ],
    )
    async def test_a_failed_script_is_recorded_by_kind(
        self,
        code: str,
        workflow: DynamicWorkflow[object] | None,
        expected: dict[str, str],
        instrumented: Instrumented,
    ):
        assert await workflow_attributes(code, instrumented=instrumented, workflow=workflow) == [
            ('execute_tool run_workflow', {**expected, 'dynamic_workflow.budget_exhausted': False})
        ]

    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            pytest.param(CrashedError(timed_out=False), {'dynamic_workflow.error_kind': 'crashed'}, id='crashed'),
            # The pool's watchdog is the transport's timer, not `max_duration_seconds`, which is off by default.
            pytest.param(
                CrashedError(timed_out=True), {'dynamic_workflow.error_kind': 'crashed'}, id='worker-timed-out'
            ),
            pytest.param(PanicException('sandbox aborted'), {'dynamic_workflow.error_kind': 'panic'}, id='panic'),
        ],
    )
    async def test_a_sandbox_failure_is_recorded_by_kind(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: BaseException,
        expected: dict[str, str],
        instrumented: Instrumented,
    ):
        monkeypatch.setattr('pydantic_ai_harness.dynamic_workflow._toolset.MontyCrashedError', CrashedError)
        monkeypatch.setattr('pydantic_ai_harness._monty_turns.MontyCrashedError', CrashedError)

        async def fail(self: object, feed_start: object) -> object:
            raise error

        monkeypatch.setattr('pydantic_ai_harness._monty_exec.MontyExecutor.run', fail)

        assert await workflow_attributes('1 + 1', instrumented=instrumented) == [
            ('execute_tool run_workflow', {**expected, 'dynamic_workflow.budget_exhausted': False})
        ]

    async def test_a_script_that_fails_on_the_budget_records_both(self, instrumented: Instrumented):
        assert await workflow_attributes(
            "await sub(task='a')\nawait sub(task='b')",
            instrumented=instrumented,
            workflow=DynamicWorkflow[object](agents=[SUB], max_agent_calls=1),
        ) == [
            (
                'execute_tool run_workflow',
                {'dynamic_workflow.error_kind': 'runtime', 'dynamic_workflow.budget_exhausted': True},
            )
        ]

    async def test_a_script_that_catches_the_budget_error_still_records_it(self, instrumented: Instrumented):
        code = "await sub(task='a')\ntry:\n    await sub(task='b')\nexcept RuntimeError:\n    pass"

        assert await workflow_attributes(
            code, instrumented=instrumented, workflow=DynamicWorkflow[object](agents=[SUB], max_agent_calls=1)
        ) == [('execute_tool run_workflow', {'dynamic_workflow.budget_exhausted': True})]
