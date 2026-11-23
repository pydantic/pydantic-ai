"""Exception transport matrix for the core durable boundaries: every boundary times every exception in the universe.

Each cell raises `sample(error_type)` on the far side of a boundary, in-process, and checks what the near side
sees against the verdict its table pins: the class that arrives, whether the engine attempts the unit again, and
how long it waits when a model error asked for a `retry_after`. `tests/exception_transport.py` holds the universe,
the families the tables are built from, and the verdict types.

Not VCR tests: no model request goes out. Each cell drives the code that converts the exception at a boundary, as
the engine calls it, and emulates the engine's own part:

- Temporal: the activities `TemporalDurability` registers run in `ActivityEnvironment`. The failure goes through
  the `PydanticAIPlugin` data converter, as the worker records it, and the retry decision is the server's: a
  failure is final when it's marked non-retryable or its type is one the retry policy names exactly.
- DBOS: the agent runs in a real DBOS workflow on SQLite, with `DBOS.sleep_async` recording the durable wait
  instead of sleeping.
- Prefect: the `retry_condition_fn` that `PrefectDurability` gives every task decides. Prefect runs tasks in the
  flow's process, so the exception arrives as itself.
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.durable_exec._toolset import CallToolResult, unwrap_tool_call_result, wrap_tool_call_result
from pydantic_ai.durable_exec._workspace import WorkspaceCallError, error_as_data, raise_error
from pydantic_ai.exceptions import (
    FallbackExceptionGroup,
    ModelRequestTimeout,
    RetryModelRequest,
    UndrainedPendingMessagesError,
    UserError,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

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
    assert_families_partition_the_universe,
    error_type_named,
    excluding,
    matrix_params,
    sample,
    sampled_types,
    verdicts,
)
from ..manifests import Classified, assert_classified

with try_import() as temporal_imports_successful:
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]
    from temporalio.api.failure.v1 import Failure
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import ActivityError
    from temporalio.testing import ActivityEnvironment

    from pydantic_ai.durable_exec.temporal import (
        TemporalDurability,
        TemporalRunContext,
        _data_converter,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.durable_exec.temporal._model_errors import rebuilt_model_errors, rebuilt_retry_requests
    from pydantic_ai.durable_exec.temporal._toolset import CallToolParams
    from pydantic_ai.durable_exec.temporal._transports import (
        _CapabilityOperationParams,
        _RequestParams,
    )

with try_import() as prefect_imports_successful:
    from pydantic_ai.durable_exec.prefect import TaskConfig
    from pydantic_ai.durable_exec.prefect._toolset import with_non_retryable_errors

with try_import() as dbos_imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID
    from dbos._error import DBOSMaxStepRetriesExceeded

    from pydantic_ai.durable_exec.dbos import DBOSDurability, StepConfig

ISSUE_14190 = 'https://github.com/pydantic/pydantic-ai/issues/14190'
NON_RETRYABLE_CLASSIFICATION = (
    'https://github.com/pydantic/pydantic-ai/issues/ISSUE-PENDING-non-retryable-classification'
)
DBOS_RETRIES_LOSE_TYPE = 'https://github.com/pydantic/pydantic-ai/issues/ISSUE-PENDING-dbos-exhausted-retries-lose-type'

_ACTIVITY_ERROR: type[BaseException] = ActivityError if temporal_imports_successful() else Exception
_STEP_RETRIES_EXCEEDED: type[BaseException] = DBOSMaxStepRetriesExceeded if dbos_imports_successful() else Exception


def test_families_partition_the_universe() -> None:
    """Every exception class in the universe is in exactly one family, and has a sample."""
    assert_families_partition_the_universe()
    assert sampled_types() == set(EXCEPTION_UNIVERSE)


def test_samples_are_their_class() -> None:
    """A sample is an instance of exactly its class, so a cell tests the class it is named after."""
    assert [type(sample(error_type)) for error_type in EXCEPTION_UNIVERSE] == list(EXCEPTION_UNIVERSE)


# --- The tool-call result: the boundary every engine's tool call goes through -------------------------------------
#
# `wrap_tool_call_result` turns what a tool raised into the `CallToolResult` an engine records, and
# `unwrap_tool_call_result` raises it again on the other side. The payload goes through JSON, as a JSON-journal
# engine records it. `retried` here means the exception escapes the wrapper and fails the unit, leaving the decision
# to the engine.

_FAILS_THE_UNIT = 'an error fails the unit, and the engine classifies it'

TOOL_CALL_RESULT = verdicts(
    (TOOL_CONTROL_FLOW, Carried(retried=False, reason='tool control flow is recorded as the call result')),
    (RUN_CANCELLATION, Carried(retried=False, reason='a cancelled sub-agent run is recorded as the result (#11163)')),
    (VALIDATION, Carried(retried=False, reason='invalid arguments are recorded as the result: a retry prompt')),
    (HOOK_CONTROL_FLOW, Carried(retried=True, reason='hook control flow is not a tool result')),
    (
        NON_RETRYABLE + USAGE_LIMIT + OTHER_RUN_ERRORS + RETRYABLE_MODEL_ERRORS + NON_RETRYABLE_MODEL_ERRORS,
        Carried(retried=True, reason=_FAILS_THE_UNIT),
    ),
    (CANCELLATION, Carried(retried=True, reason='not an `Exception`, so the wrapper lets it through')),
)

_CALL_TOOL_RESULT: TypeAdapter[CallToolResult] = TypeAdapter(CallToolResult)


async def _cross_tool_call_result(error_type: type[BaseException]) -> Crossed:
    async def tool() -> str:
        raise sample(error_type)

    try:
        payload = await wrap_tool_call_result(tool())
    except BaseException as escaped:
        return Crossed(type(escaped), retried=True)
    restored = _CALL_TOOL_RESULT.validate_python(_CALL_TOOL_RESULT.dump_python(payload, mode='json'))
    with pytest.raises(BaseException) as arrived:
        unwrap_tool_call_result(restored)
    return Crossed(type(arrived.value), retried=False)


@pytest.mark.parametrize(('error_type', 'verdict'), matrix_params(TOOL_CALL_RESULT))
async def test_tool_call_result(error_type: type[BaseException], verdict: Carried | Wrapped) -> None:
    assert await _cross_tool_call_result(error_type) == verdict.expected(error_type)


# --- The workspace error table: workspace calls from workflow code -------------------------------------------------
#
# A workspace call made from workflow code runs as the durability capability's workspace operation: `error_as_data`
# records an error the table expects as the call's result, through JSON, and `raise_error` raises it again in
# workflow code. `retried` means the error isn't in the table, so it fails the unit.

_NOT_IN_THE_TABLE = 'not in the workspace error table, so it fails the unit for the engine to retry'
_FIRST_ENTRY = 'crosses as the first entry of `_workspace._EXPECTED_ERRORS` it is an instance of'

WORKSPACE_ERROR_TABLE = verdicts(
    ((UserError,), Carried(retried=False, reason='in the table: recorded as the call result')),
    ((UndrainedPendingMessagesError,), Wrapped(UserError, retried=False, reason=_FIRST_ENTRY)),
    (VALIDATION, Wrapped(ValueError, retried=False, reason=f'a `ValueError`: it {_FIRST_ENTRY}')),
    (
        excluding(NON_RETRYABLE, UserError, UndrainedPendingMessagesError)
        + TOOL_CONTROL_FLOW
        + RUN_CANCELLATION
        + HOOK_CONTROL_FLOW
        + USAGE_LIMIT
        + OTHER_RUN_ERRORS
        + RETRYABLE_MODEL_ERRORS
        + NON_RETRYABLE_MODEL_ERRORS,
        Carried(retried=True, reason=_NOT_IN_THE_TABLE),
    ),
    (
        CANCELLATION,
        NotApplicable('not an `Exception`: `execute_call` catches only `Exception`, so it fails the unit untouched'),
    ),
)

_WORKSPACE_CALL_ERROR: TypeAdapter[WorkspaceCallError] = TypeAdapter(WorkspaceCallError)


@pytest.mark.parametrize(('error_type', 'verdict'), matrix_params(WORKSPACE_ERROR_TABLE))
def test_workspace_error_table(error_type: type[BaseException], verdict: Carried | Wrapped) -> None:
    error = sample(error_type)
    assert isinstance(error, Exception)
    if (data := error_as_data(error)) is None:
        crossed = Crossed(error_type, retried=True)
    else:
        with pytest.raises(BaseException) as arrived:
            raise_error(_WORKSPACE_CALL_ERROR.validate_json(_WORKSPACE_CALL_ERROR.dump_json(data)))
        crossed = Crossed(type(arrived.value), retried=False)
    assert crossed == verdict.expected(error_type)


# --- Prefect: the retry condition every task gets ------------------------------------------------------------------

_PREFECT_RETRIES = 'Prefect classifies only the framework errors it lists, and retries any other failure'

PREFECT_TASK = verdicts(
    (
        TOOL_CONTROL_FLOW + RUN_CANCELLATION + VALIDATION,
        NotApplicable('the tool-call result records it, so the task completes and no retry condition runs'),
    ),
    (
        excluding(NON_RETRYABLE, FallbackExceptionGroup),
        Carried(retried=False, reason='listed in `prefect._toolset._NON_RETRYABLE_ERRORS`'),
    ),
    (
        (FallbackExceptionGroup, *USAGE_LIMIT),
        Red(
            Carried(retried=False, reason='a retry cannot fix it, and Temporal does not retry it'),
            observed='retried: missing from `prefect._toolset._NON_RETRYABLE_ERRORS`',
            tracked_by=NON_RETRYABLE_CLASSIFICATION,
        ),
    ),
    (
        HOOK_CONTROL_FLOW,
        Carried(retried=True, reason=f'a capability task that raises it fails, and {_PREFECT_RETRIES}'),
    ),
    (OTHER_RUN_ERRORS, Carried(retried=True, reason=_PREFECT_RETRIES)),
    (
        RETRYABLE_MODEL_ERRORS + NON_RETRYABLE_MODEL_ERRORS,
        Carried(
            retried=True,
            reason='Prefect has no `non_retryable_model_errors`, and waits no `retry_after` (prefect.md '
            '"Retry Considerations")',
        ),
    ),
    (CANCELLATION, Carried(retried=True, reason='the condition does not classify cancellation')),
)


@pytest.mark.skipif(not prefect_imports_successful(), reason='prefect not installed')
@pytest.mark.parametrize(('error_type', 'verdict'), matrix_params(PREFECT_TASK))
async def test_prefect_task(error_type: type[BaseException], verdict: Carried | Wrapped) -> None:
    condition = with_non_retryable_errors(TaskConfig(retries=1)).get('retry_condition_fn')
    assert condition is not None
    error = sample(error_type)
    state = MagicMock()
    state.result.return_value = error
    decision = condition(MagicMock(), MagicMock(), state)
    retried = await decision if inspect.isawaitable(decision) else decision
    assert Crossed(type(error), retried=retried) == verdict.expected(error_type)


# --- Temporal: the activities `TemporalDurability` registers ------------------------------------------------------

_WRAPPED = 'reaches workflow code as `ActivityError`, with its class name as the cause type (temporal.md)'
_RETRIED = f'Temporal retries it, and it {_WRAPPED}'
_FINAL = f'in `NON_RETRYABLE_ERRORS`, subclasses included (#12105), and it {_WRAPPED}'
_NOT_CANCELLED = f'the server did not cancel the activity: {_RETRIED}'

_USAGE_LIMIT_RETRIED = Red(
    Wrapped(_ACTIVITY_ERROR, retried=False, reason='a limit checked against a usage snapshot fails again on retry'),
    observed='retried: `UsageLimitExceeded` is missing from `NON_RETRYABLE_ERRORS`',
    tracked_by=ISSUE_14190,
)

TEMPORAL_MODEL_ACTIVITY = verdicts(
    (TOOL_CONTROL_FLOW + RUN_CANCELLATION + HOOK_CONTROL_FLOW, NotApplicable('a model request does not raise it')),
    (
        RETRYABLE_MODEL_ERRORS,
        Carried(
            retried=True,
            delay=RETRY_AFTER,
            reason='rebuilt as itself in workflow code (#9580), and retried after its `retry_after` (#12106)',
        ),
    ),
    (
        NON_RETRYABLE_MODEL_ERRORS,
        Carried(retried=False, reason='rebuilt as itself, and in the default `non_retryable_model_errors`'),
    ),
    (NON_RETRYABLE, Wrapped(_ACTIVITY_ERROR, retried=False, reason=_FINAL)),
    (USAGE_LIMIT, _USAGE_LIMIT_RETRIED),
    (OTHER_RUN_ERRORS + VALIDATION, Wrapped(_ACTIVITY_ERROR, retried=True, reason=_RETRIED)),
    (CANCELLATION, Wrapped(_ACTIVITY_ERROR, retried=True, reason=_NOT_CANCELLED)),
)

TEMPORAL_TOOL_ACTIVITY = verdicts(
    (TOOL_CONTROL_FLOW, Carried(retried=False, reason='recorded as the activity result')),
    (RUN_CANCELLATION, Carried(retried=False, reason='recorded as the activity result (#11163)')),
    (VALIDATION, Carried(retried=False, reason='recorded as the activity result')),
    (HOOK_CONTROL_FLOW, NotApplicable('a tool does not raise it')),
    (NON_RETRYABLE, Wrapped(_ACTIVITY_ERROR, retried=False, reason=_FINAL)),
    (USAGE_LIMIT, _USAGE_LIMIT_RETRIED),
    (OTHER_RUN_ERRORS, Wrapped(_ACTIVITY_ERROR, retried=True, reason=_RETRIED)),
    (
        RETRYABLE_MODEL_ERRORS + NON_RETRYABLE_MODEL_ERRORS,
        Wrapped(
            _ACTIVITY_ERROR,
            retried=True,
            reason=f'`non_retryable_model_errors` and `retry_after` apply to model activities only; {_RETRIED}',
        ),
    ),
    (CANCELLATION, Wrapped(_ACTIVITY_ERROR, retried=True, reason=_NOT_CANCELLED)),
)

_UNDOCUMENTED_IN_AN_OPERATION = (
    'only `RetryModelRequest` is documented to cross a capability operation (custom.md "Durable capability '
    f'operations"), so {_RETRIED}'
)

TEMPORAL_CAPABILITY_ACTIVITY = verdicts(
    ((RetryModelRequest,), Carried(retried=False, reason='rebuilt as itself in workflow code')),
    (
        excluding(HOOK_CONTROL_FLOW, RetryModelRequest) + TOOL_CONTROL_FLOW + RUN_CANCELLATION + VALIDATION,
        Wrapped(_ACTIVITY_ERROR, retried=True, reason=_UNDOCUMENTED_IN_AN_OPERATION),
    ),
    (NON_RETRYABLE, Wrapped(_ACTIVITY_ERROR, retried=False, reason=_FINAL)),
    (USAGE_LIMIT, _USAGE_LIMIT_RETRIED),
    (
        OTHER_RUN_ERRORS + RETRYABLE_MODEL_ERRORS + NON_RETRYABLE_MODEL_ERRORS,
        Wrapped(_ACTIVITY_ERROR, retried=True, reason=_RETRIED),
    ),
    (CANCELLATION, Wrapped(_ACTIVITY_ERROR, retried=True, reason=_NOT_CANCELLED)),
)


def _raise_named(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A model that raises the sample of the class its first prompt names."""
    request = messages[0]
    assert isinstance(request, ModelRequest)
    prompt = request.parts[0]
    assert isinstance(prompt, UserPromptPart) and isinstance(prompt.content, str)
    raise sample(error_type_named(prompt.content))


def raise_named(name: str) -> str:
    """Raise the sample of the named class."""
    raise sample(error_type_named(name))


class _RaiseNamed(AbstractCapability[None]):
    id = 'raise_named'

    @durable_operation('raise')
    async def raise_named(self, ctx: RunContext[None], name: str) -> str:
        raise sample(error_type_named(name))


@dataclass
class _TemporalAgent:
    """The activities of one agent's `TemporalDurability`, and the arguments workflow code sends them."""

    activities: dict[str, Callable[..., Awaitable[object]]]
    model_policy: RetryPolicy
    tool_policy: RetryPolicy
    serialized_run_context: dict[str, object]
    tool_def: ToolDefinition

    def model_request(self, name: str) -> tuple[Callable[..., Awaitable[object]], Sequence[object]]:
        params = _RequestParams(
            messages=[ModelRequest(parts=[UserPromptPart(name)])],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
            serialized_run_context=self.serialized_run_context,
        )
        return self.activities['agent__exception_transport__model_request'], [params, None]

    def call_tool(self, name: str) -> tuple[Callable[..., Awaitable[object]], Sequence[object]]:
        params = CallToolParams(
            name='raise_named',
            tool_args={'name': name},
            serialized_run_context=self.serialized_run_context,
            tool_def=self.tool_def,
        )
        return self.activities['agent__exception_transport__toolset__<agent>__call_tool'], [params, None]

    def capability_operation(self, name: str) -> tuple[Callable[..., Awaitable[object]], Sequence[object]]:
        params = _CapabilityOperationParams(
            arguments={'name': name}, serialized_run_context=self.serialized_run_context
        )
        return self.activities['agent__exception_transport__capability__raise_named__raise'], [params, None]


@pytest.fixture(scope='module')
def temporal_agent() -> _TemporalAgent:
    tool = Tool(raise_named)
    agent = Agent(
        FunctionModel(_raise_named),
        name='exception_transport',
        deps_type=type(None),
        tools=[tool],
        capabilities=[_RaiseNamed(), TemporalDurability()],
    )
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    activities: dict[str, Callable[..., Awaitable[object]]] = {}
    for activity in durability.temporal_activities:
        name = ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        assert name is not None
        activities[name] = activity
    model_policy = durability._model_activity_config.get('retry_policy')  # pyright: ignore[reportPrivateUsage]
    tool_policy = durability.activity_config.get('retry_policy')
    assert model_policy is not None and tool_policy is not None
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), run_id='exception-transport')
    return _TemporalAgent(
        activities=activities,
        model_policy=model_policy,
        tool_policy=tool_policy,
        serialized_run_context=TemporalRunContext.serialize_run_context(ctx),
        tool_def=tool.tool_def,
    )


def _temporal_retries(failure: Failure, policy: RetryPolicy) -> bool:
    """Whether the Temporal server attempts an activity that failed with `failure` again, while the policy allows.

    An application failure is final when it's marked non-retryable, or when its type is one the retry policy's
    `non_retryable_error_types` names exactly; any other failure is retried.
    """
    if not failure.HasField('application_failure_info'):
        return True
    info = failure.application_failure_info
    return not info.non_retryable and info.type not in (policy.non_retryable_error_types or ())


async def _cross_temporal_activity(
    activity: Callable[..., Awaitable[object]],
    args: Sequence[object],
    *,
    policy: RetryPolicy,
    rebuilt: Callable[[], AbstractContextManager[None]],
) -> Crossed:
    """Run `activity` as the worker would, and raise what it failed with in workflow code, inside `rebuilt`."""
    environment = ActivityEnvironment()
    environment.info = replace(environment.info, retry_policy=policy)
    try:
        result = await environment.run(activity, *args)
    except BaseException as raised:
        converter = _data_converter(None)
        failure = Failure()
        await converter.encode_failure(raised, failure)
        info = failure.application_failure_info
        delay = info.next_retry_delay.ToTimedelta().total_seconds() if info.HasField('next_retry_delay') else None
        activity_error = ActivityError(
            'Activity task failed',
            scheduled_event_id=1,
            started_event_id=2,
            identity='worker',
            activity_type='exception_transport',
            activity_id='1',
            retry_state=None,
        )
        activity_error.__cause__ = await converter.decode_failure(failure)
        with pytest.raises(BaseException) as arrived:
            with rebuilt():
                raise activity_error
        return Crossed(type(arrived.value), _temporal_retries(failure, policy), delay)
    # Only a tool activity completes here: it records the tool's control flow as its result.
    with pytest.raises(BaseException) as arrived:
        unwrap_tool_call_result(_CALL_TOOL_RESULT.validate_python(result))
    return Crossed(type(arrived.value), retried=False)


temporal_installed = pytest.mark.skipif(not temporal_imports_successful(), reason='temporal not installed')


@temporal_installed
@pytest.mark.parametrize(('error_type', 'verdict'), matrix_params(TEMPORAL_MODEL_ACTIVITY))
async def test_temporal_model_activity(
    temporal_agent: _TemporalAgent, error_type: type[BaseException], verdict: Carried | Wrapped
) -> None:
    activity, args = temporal_agent.model_request(error_type.__qualname__)
    crossed = await _cross_temporal_activity(
        activity, args, policy=temporal_agent.model_policy, rebuilt=rebuilt_model_errors
    )
    assert crossed == verdict.expected(error_type)


@temporal_installed
@pytest.mark.parametrize(('error_type', 'verdict'), matrix_params(TEMPORAL_TOOL_ACTIVITY))
async def test_temporal_tool_activity(
    temporal_agent: _TemporalAgent, error_type: type[BaseException], verdict: Carried | Wrapped
) -> None:
    activity, args = temporal_agent.call_tool(error_type.__qualname__)
    crossed = await _cross_temporal_activity(activity, args, policy=temporal_agent.tool_policy, rebuilt=nullcontext)
    assert crossed == verdict.expected(error_type)


@temporal_installed
@pytest.mark.parametrize(('error_type', 'verdict'), matrix_params(TEMPORAL_CAPABILITY_ACTIVITY))
async def test_temporal_capability_operation_activity(
    temporal_agent: _TemporalAgent, error_type: type[BaseException], verdict: Carried | Wrapped
) -> None:
    activity, args = temporal_agent.capability_operation(error_type.__qualname__)
    crossed = await _cross_temporal_activity(
        activity, args, policy=temporal_agent.tool_policy, rebuilt=rebuilt_retry_requests
    )
    assert crossed == verdict.expected(error_type)


# --- DBOS: the model request step ----------------------------------------------------------------------------------

_RAISED_BY_A_MODEL = (
    NON_RETRYABLE
    + USAGE_LIMIT
    + OTHER_RUN_ERRORS
    + RETRYABLE_MODEL_ERRORS
    + NON_RETRYABLE_MODEL_ERRORS
    + VALIDATION
    + CANCELLATION
)

DBOS_MODEL_STEP = verdicts(
    (TOOL_CONTROL_FLOW + RUN_CANCELLATION + HOOK_CONTROL_FLOW, NotApplicable('a model request does not raise it')),
    (
        _RAISED_BY_A_MODEL,
        Carried(retried=False, reason='the default `StepConfig` allows no retries, and DBOS raises the step error'),
    ),
)

_EXHAUSTED_RETRIES_LOSE_TYPE = (
    'arrives as `DBOSMaxStepRetriesExceeded`, which holds the errors in `.errors` and has no `__cause__`, so hooks '
    'and capabilities that handle model errors never see one'
)

DBOS_MODEL_STEP_WITH_RETRIES = verdicts(
    (TOOL_CONTROL_FLOW + RUN_CANCELLATION + HOOK_CONTROL_FLOW, NotApplicable('a model request does not raise it')),
    (
        NON_RETRYABLE + USAGE_LIMIT,
        Red(
            Carried(
                retried=False,
                reason='a retry cannot fix it. dbos.md "Step Retries" says DBOS cannot leave an error out of its '
                'retries, but `DBOS.step` takes a `should_retry` in dbos 3.4, the version floor',
            ),
            observed=f'retried, as `StepConfig` passes no `should_retry`; then it {_EXHAUSTED_RETRIES_LOSE_TYPE}',
            tracked_by=NON_RETRYABLE_CLASSIFICATION,
        ),
    ),
    (
        RETRYABLE_MODEL_ERRORS + excluding(NON_RETRYABLE_MODEL_ERRORS, ModelRequestTimeout),
        Red(
            Carried(
                retried=True,
                delay=RETRY_AFTER,
                reason='retried after a durable `DBOS.sleep` of its `retry_after` (dbos.md "Step Retries"), then '
                'raised as itself, as an in-process engine raises a model error (#9580)',
            ),
            observed=_EXHAUSTED_RETRIES_LOSE_TYPE,
            tracked_by=DBOS_RETRIES_LOSE_TYPE,
        ),
    ),
    (
        (ModelRequestTimeout, *OTHER_RUN_ERRORS, *VALIDATION),
        Red(
            Carried(retried=True, reason='DBOS retries every step error (dbos.md "Step Retries"), then raises it'),
            observed=_EXHAUSTED_RETRIES_LOSE_TYPE,
            tracked_by=DBOS_RETRIES_LOSE_TYPE,
        ),
    ),
    (CANCELLATION, Carried(retried=False, reason='not an `Exception`: the step retry loop lets it through')),
)


@dataclass
class _DBOSAgents:
    """Agents whose model raises the named sample, run in a DBOS workflow that returns the class it caught."""

    attempts: list[str]
    sleeps: list[float]
    run: Callable[[str, str], Awaitable[type[BaseException]]]


@pytest.fixture(scope='module')
def dbos_agents(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_DBOSAgents]:
    database = tmp_path_factory.mktemp('dbos') / 'exception_transport.sqlite'
    DBOS(config=DBOSConfig(name='exception_transport', system_database_url=f'sqlite:///{database}'))
    DBOS.launch()
    attempts: list[str] = []
    sleeps: list[float] = []

    def counted(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        attempts.append('attempt')
        return _raise_named(messages, info)

    step_retries = StepConfig(retries_allowed=True, max_attempts=2, interval_seconds=0.0)
    agents = {
        'default': Agent(FunctionModel(counted), name='exception_transport', capabilities=[DBOSDurability()]),
        'retries': Agent(
            FunctionModel(counted),
            name='exception_transport_retries',
            capabilities=[DBOSDurability(model_step_config=step_retries)],
        ),
    }

    @DBOS.workflow()
    async def run(config: str, name: str) -> type[BaseException]:
        with pytest.raises(BaseException) as raised:
            await agents[config].run(name)
        return type(raised.value)

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(DBOS, 'sleep_async', record_sleep)
        try:
            yield _DBOSAgents(attempts=attempts, sleeps=sleeps, run=run)
        finally:
            DBOS.destroy()


async def _cross_dbos_model_step(agents: _DBOSAgents, config: str, error_type: type[BaseException]) -> Crossed:
    agents.attempts.clear()
    agents.sleeps.clear()
    with SetWorkflowID(str(uuid.uuid4())):
        arrived_as = await agents.run(config, error_type.__qualname__)
    return Crossed(arrived_as, retried=len(agents.attempts) > 1, delay=agents.sleeps[0] if agents.sleeps else None)


dbos_installed = pytest.mark.skipif(not dbos_imports_successful(), reason='dbos not installed')


@dbos_installed
@pytest.mark.xdist_group(name='dbos')
@pytest.mark.parametrize(('error_type', 'verdict'), matrix_params(DBOS_MODEL_STEP))
async def test_dbos_model_step(dbos_agents: _DBOSAgents, error_type: type[BaseException], verdict: Carried) -> None:
    assert await _cross_dbos_model_step(dbos_agents, 'default', error_type) == verdict.expected(error_type)


@dbos_installed
@pytest.mark.xdist_group(name='dbos')
@pytest.mark.parametrize(('error_type', 'verdict'), matrix_params(DBOS_MODEL_STEP_WITH_RETRIES))
async def test_dbos_model_step_with_retries(
    dbos_agents: _DBOSAgents, error_type: type[BaseException], verdict: Carried
) -> None:
    assert await _cross_dbos_model_step(dbos_agents, 'retries', error_type) == verdict.expected(error_type)


# --- Every table classifies the whole universe ---------------------------------------------------------------------

TABLES: dict[str, Mapping[type[BaseException], Classified]] = {
    'TOOL_CALL_RESULT': TOOL_CALL_RESULT,
    'WORKSPACE_ERROR_TABLE': WORKSPACE_ERROR_TABLE,
    'PREFECT_TASK': PREFECT_TASK,
    'TEMPORAL_MODEL_ACTIVITY': TEMPORAL_MODEL_ACTIVITY,
    'TEMPORAL_TOOL_ACTIVITY': TEMPORAL_TOOL_ACTIVITY,
    'TEMPORAL_CAPABILITY_ACTIVITY': TEMPORAL_CAPABILITY_ACTIVITY,
    'DBOS_MODEL_STEP': DBOS_MODEL_STEP,
    'DBOS_MODEL_STEP_WITH_RETRIES': DBOS_MODEL_STEP_WITH_RETRIES,
}


@pytest.mark.parametrize('table_name', TABLES)
def test_table_classifies_every_exception(table_name: str) -> None:
    """A new exception class fails until each boundary gives it a verdict with a reason; a removed one, as stale."""
    assert_classified(EXCEPTION_UNIVERSE, TABLES[table_name], complete_walk=True, table_name=table_name)
