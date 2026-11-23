"""Exception transport matrix for the Harness durable boundaries: every boundary times every exception in the universe.

`tests/exception_transport.py` holds the universe, the families the tables are built from, and the verdict types;
`tests/durable_exec/test_exception_transport.py` the core boundaries, including the tool-call result every engine
records a tool's control flow with.

Not VCR tests: no model request goes out. A `FunctionModel` raises the exception, or calls the tool that does, and
the cell checks what the run sees:

- Azure Durable Task: the agent runs in an orchestration on the in-process `InMemoryWorker`, which replays real SDK
  histories. A retry shows as the timer the orchestration schedules before the next attempt.
- CodeMode: a tool called from a `run_code` snippet has to end the run the way the same tool called directly does.

Absurd, pgtask, AWS Lambda and Render have no in-process crossing that shows what their SDK does with a failure, so
their cells are `NotApplicable`, each with its reason.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from dataclasses import dataclass
from datetime import timedelta

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRequestTimeout,
    UndrainedPendingMessagesError,
    UserError,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ..conftest import try_import
from ..exception_transport import (
    CANCELLATION,
    EXCEPTION_UNIVERSE,
    HOOK_CONTROL_FLOW,
    NON_RETRYABLE,
    NON_RETRYABLE_MODEL_ERRORS,
    OTHER_RUN_ERRORS,
    RETRY_AFTER,
    RETRYABLE_MODEL_ERRORS,
    RUN_CANCELLATION,
    TOOL_CONTROL_FLOW,
    USAGE_LIMIT,
    VALIDATION,
    Carried,
    Crossed,
    NotApplicable,
    Red,
    Wrapped,
    error_type_named,
    excluding,
    matrix_params,
    sample,
    verdicts,
)
from ..manifests import Classified, assert_classified

with try_import() as azure_imports_successful:
    from durabletask import task

    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

    from .azure_durable_task.worker import InMemoryWorker, registered_agent

with try_import() as code_mode_imports_successful:
    from pydantic_ai_harness import CodeMode

NON_RETRYABLE_CLASSIFICATION = (
    'https://github.com/pydantic/pydantic-ai/issues/ISSUE-PENDING-non-retryable-classification'
)
CODE_MODE_NESTED_EXCEPTIONS = 'https://github.com/pydantic/pydantic-ai/issues/ISSUE-PENDING-codemode-nested-exceptions'

_TASK_FAILED: type[BaseException] = task.TaskFailedError if azure_imports_successful() else Exception


# --- Azure Durable Task: model and tool activities -----------------------------------------------------------------

_RETRY_POLICY_SECONDS = 1.0
"""The `first_retry_interval` of the retry policy every Azure cell's activities get: the wait before a native retry."""

_AZURE_WRAPS = 'reaches the orchestrator as `TaskFailedError` (azure-durable-task.md)'
_AZURE_RETRIES = f'the retry policy retries every activity failure, and it {_AZURE_WRAPS}'
_USER_ERROR_KEEPS_ITS_TYPE = 'recorded as data and raised again as `UserError`, outside the retry policy'
_AZURE_NON_RETRYABLE = Red(
    Wrapped(_TASK_FAILED, retried=False, reason='a retry cannot fix it'),
    observed='retried: only `UserError` is recorded outside the retry policy (`azure_durable_task._operation_backend`)',
    tracked_by=NON_RETRYABLE_CLASSIFICATION,
)

AZURE_MODEL_ACTIVITY = verdicts(
    (TOOL_CONTROL_FLOW + RUN_CANCELLATION + HOOK_CONTROL_FLOW, NotApplicable('a model request does not raise it')),
    (
        RETRYABLE_MODEL_ERRORS + excluding(NON_RETRYABLE_MODEL_ERRORS, ModelRequestTimeout),
        Wrapped(
            _TASK_FAILED,
            retried=True,
            delay=RETRY_AFTER,
            reason='Azure has no `non_retryable_model_errors`, and retries a model error after its `retry_after`, on '
            f'a durable timer; it {_AZURE_WRAPS}',
        ),
    ),
    (
        (ModelRequestTimeout,),
        Wrapped(_TASK_FAILED, retried=True, delay=_RETRY_POLICY_SECONDS, reason=f'no `retry_after`: {_AZURE_RETRIES}'),
    ),
    ((UserError,), Carried(retried=False, reason=_USER_ERROR_KEEPS_ITS_TYPE)),
    ((UndrainedPendingMessagesError,), Wrapped(UserError, retried=False, reason=_USER_ERROR_KEEPS_ITS_TYPE)),
    (excluding(NON_RETRYABLE, UserError, UndrainedPendingMessagesError) + USAGE_LIMIT, _AZURE_NON_RETRYABLE),
    (
        OTHER_RUN_ERRORS + VALIDATION + CANCELLATION,
        Wrapped(_TASK_FAILED, retried=True, delay=_RETRY_POLICY_SECONDS, reason=_AZURE_RETRIES),
    ),
)

AZURE_TOOL_ACTIVITY = verdicts(
    (
        TOOL_CONTROL_FLOW + RUN_CANCELLATION + VALIDATION,
        NotApplicable(
            'the tool-call result records it, so the activity completes; '
            '`tests/durable_exec/test_exception_transport.py` checks that boundary'
        ),
    ),
    (HOOK_CONTROL_FLOW, NotApplicable('a tool does not raise it')),
    ((UserError,), Carried(retried=False, reason=_USER_ERROR_KEEPS_ITS_TYPE)),
    ((UndrainedPendingMessagesError,), Wrapped(UserError, retried=False, reason=_USER_ERROR_KEEPS_ITS_TYPE)),
    (excluding(NON_RETRYABLE, UserError, UndrainedPendingMessagesError) + USAGE_LIMIT, _AZURE_NON_RETRYABLE),
    (
        OTHER_RUN_ERRORS + RETRYABLE_MODEL_ERRORS + NON_RETRYABLE_MODEL_ERRORS + CANCELLATION,
        Wrapped(
            _TASK_FAILED,
            retried=True,
            delay=_RETRY_POLICY_SECONDS,
            reason=f'only a model activity waits a `retry_after`; {_AZURE_RETRIES}',
        ),
    ),
)


def _cross_azure(error_type: type[BaseException], *, raised_by: str) -> Crossed:
    """Run an agent whose model, or whose tool, raises the sample, in an orchestration on the in-process worker."""
    worker = InMemoryWorker()

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert len(messages) == 1, 'the model or the tool raises, so the run ends at its first step'
        if raised_by == 'model':
            raise sample(error_type)
        return ModelResponse(parts=[ToolCallPart('raise_named', {'name': error_type.__qualname__})])

    policy = task.RetryPolicy(first_retry_interval=timedelta(seconds=_RETRY_POLICY_SECONDS), max_number_of_attempts=2)
    durability = (
        AzureDurableTaskDurability(model_options=policy)
        if raised_by == 'model'
        else AzureDurableTaskDurability(tool_options=policy)
    )
    agent = Agent(FunctionModel(respond), name=f'exception-transport-{raised_by}', capabilities=[durability])

    @agent.tool_plain
    def raise_named(name: str) -> str:
        raise sample(error_type_named(name))

    def orchestrator(context: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, str]:
        async def body() -> str:
            try:
                await agent.run('Hi')
            except BaseException as raised:
                return type(raised).__qualname__
            raise AssertionError('the model or the tool raises')

        return (yield from run_durable(context, body))

    waits: list[float] = []
    with registered_agent(worker, agent):
        worker.add_orchestrator(orchestrator)
        instance_id = worker.schedule(orchestrator)
        worker.run_until_idle()
        while worker.timers:
            [(_, _, timer)] = worker.timers
            waits.append((timer.fireAt.ToDatetime() - worker.now).total_seconds())
            worker.fire_timers()
            worker.run_until_idle()
        state = worker.wait(instance_id)
    state.raise_if_failed()
    [arrived_as] = [error for error in (*EXCEPTION_UNIVERSE, _TASK_FAILED) if error.__qualname__ == state.get_output()]
    return Crossed(arrived_as, retried=bool(waits), delay=waits[0] if waits else None)


azure_installed = pytest.mark.skipif(not azure_imports_successful(), reason='durabletask not installed')


@azure_installed
@pytest.mark.parametrize(('error_type', 'verdict'), matrix_params(AZURE_MODEL_ACTIVITY))
def test_azure_durable_task_model_activity(error_type: type[BaseException], verdict: Carried | Wrapped) -> None:
    assert _cross_azure(error_type, raised_by='model') == verdict.expected(error_type)


@azure_installed
@pytest.mark.parametrize(('error_type', 'verdict'), matrix_params(AZURE_TOOL_ACTIVITY))
def test_azure_durable_task_tool_activity(error_type: type[BaseException], verdict: Carried | Wrapped) -> None:
    assert _cross_azure(error_type, raised_by='tool') == verdict.expected(error_type)


# --- CodeMode: a tool called from a `run_code` snippet --------------------------------------------------------------


@dataclass(frozen=True)
class RunOutcome:
    """How a run that called the tool ended: with the exception it raised, or with the tool result the model saw."""

    raised: type[BaseException] | None = None
    tool_result: str | None = None
    """The `outcome` of the tool result the model got back: from the tool itself, or from `run_code`."""


@dataclass(frozen=True)
class MatchesDirectCall:
    """A nested call ends the run the way the same tool called directly does."""

    reason: str

    def expected(self, direct: RunOutcome) -> RunOutcome:
        return direct


@dataclass(frozen=True)
class Diverges:
    """A nested call ends the run another way, on purpose."""

    outcome: RunOutcome
    reason: str

    def expected(self, direct: RunOutcome) -> RunOutcome:
        return self.outcome


CODE_MODE_NESTED_CALL = verdicts(
    (HOOK_CONTROL_FLOW, NotApplicable('a tool does not raise it')),
    (
        excluding(TOOL_CONTROL_FLOW, CallDeferred, ApprovalRequired) + VALIDATION + CANCELLATION,
        MatchesDirectCall('code-mode.md: a nested `ToolFailed` is a failed result, a nested `ModelRetry` a retry'),
    ),
    (
        (CallDeferred, ApprovalRequired),
        Diverges(
            RunOutcome(tool_result='retried'),
            reason='the sandbox cannot hand an unresolved deferral to the caller, so `run_nested_call` raises a '
            '`UserError` into the snippet, which `run_code` turns into a retry (`_code_tools.run_nested_call`)',
        ),
    ),
    (
        RUN_CANCELLATION
        + NON_RETRYABLE
        + USAGE_LIMIT
        + OTHER_RUN_ERRORS
        + RETRYABLE_MODEL_ERRORS
        + NON_RETRYABLE_MODEL_ERRORS,
        Red(
            MatchesDirectCall('a tool failure does not depend on how the tool was called (#13076)'),
            observed='a `run_code` retry: Monty hands the snippet a plain `Exception`, and only a `ToolFailed` is told '
            'apart from it',
            tracked_by=CODE_MODE_NESTED_EXCEPTIONS,
        ),
    ),
)


async def _run_calling_the_tool(error_type: type[BaseException], *, code_mode: bool) -> RunOutcome:
    tool_results: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            if code_mode:
                return ModelResponse(parts=[ToolCallPart('run_code', {'code': 'await raise_sample()'})])
            return ModelResponse(parts=[ToolCallPart('raise_sample', {})])
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        [result] = [part for part in request.parts if isinstance(part, ToolReturnPart)]
        tool_results.append(result.outcome)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[CodeMode()] if code_mode else [], retries=1)

    @agent.tool_plain
    async def raise_sample() -> str:
        raise sample(error_type)

    try:
        await agent.run('Hi')
    except BaseException as raised:
        return RunOutcome(raised=type(raised))
    [tool_result] = tool_results
    return RunOutcome(tool_result=tool_result)


@pytest.mark.skipif(not code_mode_imports_successful(), reason='pydantic-monty not installed')
@pytest.mark.parametrize(('error_type', 'verdict'), matrix_params(CODE_MODE_NESTED_CALL))
async def test_code_mode_nested_call(error_type: type[BaseException], verdict: MatchesDirectCall | Diverges) -> None:
    direct = await _run_calling_the_tool(error_type, code_mode=False)
    nested = await _run_calling_the_tool(error_type, code_mode=True)
    assert nested == verdict.expected(direct)


# --- Engines with no in-process crossing ---------------------------------------------------------------------------

ABSURD = verdicts(
    (
        EXCEPTION_UNIVERSE,
        NotApplicable(
            'Absurd journals on PostgreSQL (`ABSURD_TEST_DATABASE_URL`, tests/harness/absurd/conftest.py), with no '
            'in-process engine to cross'
        ),
    ),
)
PGTASK = verdicts(
    (
        EXCEPTION_UNIVERSE,
        NotApplicable(
            'pgtask journals on PostgreSQL (`PGTASK_TEST_DATABASE_URL`, tests/harness/pgtask/conftest.py), with no '
            'in-process engine to cross'
        ),
    ),
)
AWS_LAMBDA = verdicts(
    (
        EXCEPTION_UNIVERSE,
        NotApplicable(
            "`FakeDurableContext.step` (tests/harness/aws_lambda/conftest.py) re-raises a failed step's own "
            'exception, where the SDK raises `StepError`, so an in-process cell would not see what crosses'
        ),
    ),
)
RENDER = verdicts(
    (
        EXCEPTION_UNIVERSE,
        NotApplicable(
            "the in-process task context (tests/harness/render/conftest.py) re-raises a failed task's exception, "
            'where the Render SDK raises `TaskRunError`; Render waits no `retry_after` (render-workflows.md)'
        ),
    ),
)

TABLES: dict[str, Mapping[type[BaseException], Classified]] = {
    'AZURE_MODEL_ACTIVITY': AZURE_MODEL_ACTIVITY,
    'AZURE_TOOL_ACTIVITY': AZURE_TOOL_ACTIVITY,
    'CODE_MODE_NESTED_CALL': CODE_MODE_NESTED_CALL,
    'ABSURD': ABSURD,
    'PGTASK': PGTASK,
    'AWS_LAMBDA': AWS_LAMBDA,
    'RENDER': RENDER,
}


@pytest.mark.parametrize('table_name', TABLES)
def test_table_classifies_every_exception(table_name: str) -> None:
    """A new exception class fails until each boundary gives it a verdict with a reason; a removed one, as stale."""
    assert_classified(EXCEPTION_UNIVERSE, TABLES[table_name], complete_walk=True, table_name=table_name)
