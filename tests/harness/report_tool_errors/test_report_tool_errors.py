"""Tests for `ReportToolErrors`, driven through `Agent(..., capabilities=[ReportToolErrors()])`."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, DeferredToolRequests, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    RunCancelled,
    SkipToolExecution,
    ToolFailed,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness import ReportToolErrors, ToolGuardrail
from pydantic_ai_harness.guardrails import GuardrailResult, ToolResultInfo
from pydantic_ai_harness.report_tool_errors import ToolErrorInfo


def _scripted(*responses: Sequence[ModelResponsePart]) -> FunctionModel:
    """A model that sends `responses` in order, then answers `'done'`."""
    steps = iter(responses)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=list(next(steps, [TextPart('done')])))

    return FunctionModel(model)


def _divide(tool_call_id: str, b: int) -> ToolCallPart:
    return ToolCallPart('divide', {'a': 1, 'b': b}, tool_call_id=tool_call_id)


def _agent(model: FunctionModel, *capabilities: AbstractCapability[None]) -> Agent[None, str | DeferredToolRequests]:
    agent = Agent(model, deps_type=type(None), capabilities=list(capabilities), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def divide(a: int, b: int) -> float:
        return a / b

    @agent.tool_plain(retries=1)
    def flaky() -> str:
        raise ModelRetry('try again')

    @agent.tool_plain
    def refuse() -> str:
        raise ToolFailed('not today')

    @agent.tool_plain
    def external() -> str:
        raise CallDeferred

    @agent.tool_plain
    def skipped() -> str:
        raise SkipToolExecution('skipped by the tool')

    @agent.tool_plain
    def sub_agent() -> str:
        raise RunCancelled('stopped itself')

    @agent.tool_plain
    def guarded() -> str:
        raise ApprovalRequired

    @agent.tool_plain
    def misconfigured() -> str:
        raise UserError('missing API key setting')

    @agent.tool_plain
    def delegate() -> str:
        raise UsageLimitExceeded('The whole tree is out of budget')

    return agent


def _tool_results(messages: list[ModelMessage]) -> list[tuple[str, object, str]]:
    return [
        (part.tool_call_id, part.content, part.outcome)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


class TestReportToolErrors:
    def test_without_capability_the_exception_ends_the_run(self) -> None:
        agent = _agent(_scripted([_divide('c1', 0)]))
        with pytest.raises(ZeroDivisionError):
            agent.run_sync('divide')

    def test_exception_is_reported_and_the_run_continues(self) -> None:
        result = _agent(_scripted([_divide('c1', 0)], [_divide('c2', 2)]), ReportToolErrors()).run_sync('divide')
        assert result.output == 'done'
        assert _tool_results(result.all_messages()) == snapshot(
            [('c1', '`divide` raised ZeroDivisionError', 'failed'), ('c2', 0.5, 'success')]
        )

    def test_failing_again_is_reported_again(self) -> None:
        result = _agent(
            _scripted([_divide('c1', 0)], [_divide('c2', 0)], [_divide('c3', 0)]), ReportToolErrors()
        ).run_sync('divide')
        assert [outcome for _, _, outcome in _tool_results(result.all_messages())] == ['failed', 'failed', 'failed']

    def test_format_error_replaces_the_text(self) -> None:
        def hide_internals(ctx: RunContext[None], info: ToolErrorInfo) -> str:
            return f'{info.name}({info.args}) failed ({type(info.error).__name__}) in call {info.tool_call_id}'

        result = _agent(_scripted([_divide('c1', 0)]), ReportToolErrors(format_error=hide_internals)).run_sync('go')
        assert _tool_results(result.all_messages()) == snapshot(
            [('c1', "divide({'a': 1, 'b': 0}) failed (ZeroDivisionError) in call c1", 'failed')]
        )

    def test_exceptions_narrows_what_is_reported(self) -> None:
        report = ReportToolErrors[None](exceptions=(TimeoutError,))
        with pytest.raises(ZeroDivisionError):
            _agent(_scripted([_divide('c1', 0)]), report).run_sync('go')

    @pytest.mark.parametrize(('tool_name', 'error'), [('misconfigured', UserError), ('delegate', UsageLimitExceeded)])
    def test_setup_errors_and_exhausted_budgets_still_end_the_run(self, tool_name: str, error: type[Exception]) -> None:
        agent = _agent(_scripted([ToolCallPart(tool_name, {}, tool_call_id='c1')]), ReportToolErrors())
        with pytest.raises(error):
            agent.run_sync('go')

    def test_the_last_listed_instance_formats_the_error(self) -> None:
        first = ReportToolErrors[None](format_error=lambda ctx, info: 'first')
        last = ReportToolErrors[None](format_error=lambda ctx, info: 'last')
        result = _agent(_scripted([_divide('c1', 0)]), first, last).run_sync('go')
        assert _tool_results(result.all_messages()) == snapshot([('c1', 'last', 'failed')])

    def test_model_retry_still_asks_for_a_retry(self) -> None:
        result = _agent(_scripted([ToolCallPart('flaky', {}, tool_call_id='c1')]), ReportToolErrors()).run_sync('go')
        assert _tool_results(result.all_messages()) == snapshot([('c1', 'try again', 'retried')])

    def test_running_out_of_retries_still_ends_the_run(self) -> None:
        calls = [[ToolCallPart('flaky', {}, tool_call_id=f'c{i}')] for i in range(3)]
        agent = _agent(_scripted(*calls), ReportToolErrors())
        with pytest.raises(UnexpectedModelBehavior, match="Tool 'flaky' exceeded max retries count of 1"):
            agent.run_sync('go')

    def test_tool_failed_keeps_its_message(self) -> None:
        result = _agent(_scripted([ToolCallPart('refuse', {}, tool_call_id='c1')]), ReportToolErrors()).run_sync('go')
        assert _tool_results(result.all_messages()) == snapshot([('c1', 'not today', 'failed')])

    def test_tool_guardrail_result_guard_screens_the_reported_text(self) -> None:
        def redact(ctx: RunContext[None], result: ToolResultInfo) -> GuardrailResult:
            return GuardrailResult.replace(str(result.result).replace('ZeroDivisionError', '[redacted]'))

        result = _agent(
            _scripted([_divide('c1', 0)]), ReportToolErrors(), ToolGuardrail[None](result_guard=redact)
        ).run_sync('go')
        assert _tool_results(result.all_messages()) == snapshot([('c1', '`divide` raised [redacted]', 'failed')])

    def test_error_hooks_are_not_called_for_a_reported_exception(self) -> None:
        seen: list[str] = []

        class RecordErrors(AbstractCapability[None]):
            async def on_tool_execute_error(
                self,
                ctx: RunContext[None],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: ValidatedToolArgs,
                error: Exception,
            ) -> None:
                seen.append(type(error).__name__)
                raise error

        for capabilities in ([RecordErrors(), ReportToolErrors[None]()], [ReportToolErrors[None](), RecordErrors()]):
            result = _agent(_scripted([_divide('c1', 0)]), *capabilities).run_sync('go')
            assert [outcome for _, _, outcome in _tool_results(result.all_messages())] == ['failed']
        assert seen == []

        narrowed = ReportToolErrors[None](exceptions=(TimeoutError,))
        with pytest.raises(ZeroDivisionError):
            _agent(_scripted([_divide('c1', 0)]), narrowed, RecordErrors()).run_sync('go')
        assert seen == ['ZeroDivisionError']

    def test_approval_required_still_defers(self) -> None:
        result = _agent(_scripted([ToolCallPart('guarded', {}, tool_call_id='c1')]), ReportToolErrors()).run_sync('go')
        assert isinstance(result.output, DeferredToolRequests)
        assert [call.tool_call_id for call in result.output.approvals] == ['c1']

    def test_call_deferred_still_defers(self) -> None:
        result = _agent(_scripted([ToolCallPart('external', {}, tool_call_id='c1')]), ReportToolErrors()).run_sync('go')
        assert isinstance(result.output, DeferredToolRequests)
        assert [call.tool_call_id for call in result.output.calls] == ['c1']

    @pytest.mark.parametrize(
        ('tool_name', 'expected'),
        [
            ('skipped', ('c1', 'skipped by the tool', 'success')),
            ('sub_agent', ('c1', 'The sub-agent run was cancelled: stopped itself', 'failed')),
        ],
    )
    def test_skip_and_sub_agent_cancellation_keep_their_results(
        self, tool_name: str, expected: tuple[str, object, str]
    ) -> None:
        result = _agent(_scripted([ToolCallPart(tool_name, {}, tool_call_id='c1')]), ReportToolErrors()).run_sync('go')
        assert _tool_results(result.all_messages()) == [expected]
