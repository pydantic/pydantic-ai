"""The exceptions a durable boundary carries, and the verdict each boundary gives each of them.

Shared by the two exception transport matrices: `tests/durable_exec/test_exception_transport.py` holds the core
boundaries, `tests/harness/test_exception_transport.py` the Harness ones. A boundary is crossed in-process, without
an engine server: the test raises `sample(error_type)` on the far side and records what the near side sees. Each
boundary's table gives every member of `EXCEPTION_UNIVERSE` a verdict, built from the families below, and
`assert_classified` checks the table against the universe, so a new public exception class fails every matrix
until it is classified, and a removed one fails as stale.

A cell that is wrong on main is a `Red` cell: its expected verdict runs as a strict expected failure, with the
issue that tracks the fix, and starts failing once the fix lands, so the table is updated then.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TypeVar

import pytest
from _pytest.mark import ParameterSet
from pydantic import ValidationError

import pydantic_ai.exceptions as exceptions_module
from pydantic_ai.exceptions import (
    AgentRunError,
    ApprovalRequired,
    CallDeferred,
    ConcurrencyLimitExceeded,
    ContentFilterError,
    FallbackExceptionGroup,
    IncompleteToolCall,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelNotFoundError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelRequestTimeout,
    ModelRetry,
    ModelServerError,
    ModelTimeoutError,
    ModelUnavailableError,
    OutputTokenLimitExceeded,
    RetryModelRequest,
    RunCancelled,
    RunTimedOut,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    SuspendedResponseExpired,
    ToolFailed,
    UndrainedPendingMessagesError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import ModelResponse, TextPart

from .manifests import assert_classified

PUBLIC_EXCEPTIONS: tuple[type[BaseException], ...] = tuple(
    member
    for name in exceptions_module.__all__
    if isinstance(member := getattr(exceptions_module, name), type)
    and issubclass(member, BaseException)
    and not issubclass(member, Warning)
)
"""The exception classes `pydantic_ai.exceptions` exports. Its warnings are left out: nothing raises them."""

EXCEPTION_UNIVERSE: tuple[type[BaseException], ...] = (*PUBLIC_EXCEPTIONS, asyncio.CancelledError, ValidationError)
"""Every exception class a matrix classifies.

Two classes from outside `pydantic_ai.exceptions` are added because the framework gives them meaning when a tool
raises them: `CancelledError` is a `BaseException` an engine's unit can swallow or leave hanging, and a pydantic
`ValidationError` from a tool body reads as invalid arguments and becomes a retry prompt (#10730).
"""

RETRY_AFTER = 3.0
"""The `retry_after`, in seconds, that every model error sample asks for."""

# The families group the universe by what the framework means by each exception. Every boundary table is built
# from them, and `assert_families_partition_the_universe` checks that they cover the universe exactly once.

TOOL_CONTROL_FLOW: tuple[type[BaseException], ...] = (ModelRetry, ToolFailed, CallDeferred, ApprovalRequired)
"""Raised by a tool to tell the agent loop how the call went: the tool-call result carries them as data."""

RUN_CANCELLATION: tuple[type[BaseException], ...] = (RunCancelled, RunTimedOut)
"""Raised by a sub-agent run a tool awaits: the tool-call result carries them as data, like a failed call (#11163)."""

VALIDATION: tuple[type[BaseException], ...] = (ValidationError,)
"""Raised by a tool's argument validation, or its body: the tool-call result carries it as data."""

HOOK_CONTROL_FLOW: tuple[type[BaseException], ...] = (
    SkipModelRequest,
    RetryModelRequest,
    SkipToolValidation,
    SkipToolExecution,
)
"""Raised by capability hooks, never by a tool or a model request."""

NON_RETRYABLE: tuple[type[BaseException], ...] = (
    UserError,
    UndrainedPendingMessagesError,
    UnexpectedModelBehavior,
    ContentFilterError,
    OutputTokenLimitExceeded,
    IncompleteToolCall,
    FallbackExceptionGroup,
)
"""The framework errors a retry can't fix, subclasses included: Temporal's `NON_RETRYABLE_ERRORS`."""

USAGE_LIMIT: tuple[type[BaseException], ...] = (UsageLimitExceeded,)
"""A limit checked against a usage snapshot fails the same way on every retry (#14190)."""

OTHER_RUN_ERRORS: tuple[type[BaseException], ...] = (AgentRunError, SuspendedResponseExpired, ConcurrencyLimitExceeded)
"""Run errors no engine classifies: each retries them as its policy says."""

RETRYABLE_MODEL_ERRORS: tuple[type[BaseException], ...] = (
    ModelAPIError,
    ModelHTTPError,
    ModelRateLimitError,
    ModelUnavailableError,
    ModelOverloadedError,
    ModelServerError,
    ModelConnectionError,
    ModelTimeoutError,
)
"""The model errors another attempt can fix, which Temporal retries after their `retry_after`."""

NON_RETRYABLE_MODEL_ERRORS: tuple[type[BaseException], ...] = (
    ModelRequestRejectedError,
    ModelContextWindowExceededError,
    ModelAuthenticationError,
    ModelNotFoundError,
    ModelQuotaExceededError,
    ModelRequestTimeout,
)
"""The model errors another attempt at the same request can't fix: Temporal's default `non_retryable_model_errors`."""

CANCELLATION: tuple[type[BaseException], ...] = (asyncio.CancelledError,)
"""A cancelled task: not an `Exception`, so an `except Exception` in a durable unit lets it through."""


@dataclass(frozen=True)
class _Family:
    reason: str


_FAMILIES: tuple[tuple[tuple[type[BaseException], ...], _Family], ...] = (
    (TOOL_CONTROL_FLOW, _Family('tool control flow')),
    (RUN_CANCELLATION, _Family('a sub-agent run cancelled itself')),
    (VALIDATION, _Family('invalid tool arguments')),
    (HOOK_CONTROL_FLOW, _Family('capability hook control flow')),
    (NON_RETRYABLE, _Family('framework errors a retry cannot fix')),
    (USAGE_LIMIT, _Family('a usage limit')),
    (OTHER_RUN_ERRORS, _Family('run errors no engine classifies')),
    (RETRYABLE_MODEL_ERRORS, _Family('model errors another attempt can fix')),
    (NON_RETRYABLE_MODEL_ERRORS, _Family('model errors another attempt cannot fix')),
    (CANCELLATION, _Family('task cancellation')),
)


def assert_families_partition_the_universe() -> None:
    """Fail unless every class in the universe is in exactly one family."""
    table: dict[type[BaseException], _Family] = {}
    for members, family in _FAMILIES:
        for member in members:
            assert member not in table, f'{member.__qualname__} is in two families'
            table[member] = family
    assert_classified(EXCEPTION_UNIVERSE, table, complete_walk=True, table_name='_FAMILIES')


def _model_error(error_type: type[ModelAPIError]) -> Callable[[], BaseException]:
    return lambda: error_type('test-model', f'{error_type.__name__} from the provider', retry_after=RETRY_AFTER)


def _validation_error() -> BaseException:
    return ValidationError.from_exception_data(
        'boom', [{'type': 'int_parsing', 'loc': ('value',), 'input': 'not a number'}]
    )


_SAMPLES: dict[type[BaseException], Callable[[], BaseException]] = {
    ModelRetry: lambda: ModelRetry('Try again.'),
    ToolFailed: lambda: ToolFailed('The record is gone.'),
    CallDeferred: lambda: CallDeferred(metadata={'queue': 'later'}),
    ApprovalRequired: lambda: ApprovalRequired(metadata={'reason': 'review'}),
    RunCancelled: lambda: RunCancelled('The sub-agent cancelled itself.'),
    RunTimedOut: lambda: RunTimedOut('The sub-agent ran out of time.'),
    ValidationError: _validation_error,
    SkipModelRequest: lambda: SkipModelRequest(ModelResponse(parts=[TextPart('skipped')])),
    RetryModelRequest: lambda: RetryModelRequest(),
    SkipToolValidation: lambda: SkipToolValidation({'value': 1}),
    SkipToolExecution: lambda: SkipToolExecution('done'),
    UserError: lambda: UserError('The tool is misconfigured.'),
    UndrainedPendingMessagesError: lambda: UndrainedPendingMessagesError('Messages are still pending.'),
    UnexpectedModelBehavior: lambda: UnexpectedModelBehavior('Unexpected response.'),
    ContentFilterError: lambda: ContentFilterError('Content filter triggered.'),
    OutputTokenLimitExceeded: lambda: OutputTokenLimitExceeded('Output token limit exceeded.'),
    IncompleteToolCall: lambda: IncompleteToolCall('Token limit exceeded while generating a tool call.'),
    FallbackExceptionGroup: lambda: FallbackExceptionGroup(
        'All models from FallbackModel failed', [ModelServerError('test-model', 'Internal error.')]
    ),
    UsageLimitExceeded: lambda: UsageLimitExceeded('The next request would exceed the request_limit of 1'),
    AgentRunError: lambda: AgentRunError('The run failed.'),
    SuspendedResponseExpired: lambda: SuspendedResponseExpired('The background response expired.'),
    ConcurrencyLimitExceeded: lambda: ConcurrencyLimitExceeded('The queue is full.'),
    ModelAPIError: _model_error(ModelAPIError),
    ModelHTTPError: lambda: ModelHTTPError(503, 'test-model', headers={'retry-after': str(int(RETRY_AFTER))}),
    ModelRateLimitError: _model_error(ModelRateLimitError),
    ModelUnavailableError: _model_error(ModelUnavailableError),
    ModelOverloadedError: _model_error(ModelOverloadedError),
    ModelServerError: _model_error(ModelServerError),
    ModelConnectionError: _model_error(ModelConnectionError),
    ModelTimeoutError: _model_error(ModelTimeoutError),
    ModelRequestRejectedError: _model_error(ModelRequestRejectedError),
    ModelContextWindowExceededError: _model_error(ModelContextWindowExceededError),
    ModelAuthenticationError: _model_error(ModelAuthenticationError),
    ModelNotFoundError: _model_error(ModelNotFoundError),
    ModelQuotaExceededError: _model_error(ModelQuotaExceededError),
    ModelRequestTimeout: lambda: ModelRequestTimeout('test-model', 30.0),
    asyncio.CancelledError: lambda: asyncio.CancelledError(),
}


def sample(error_type: type[BaseException]) -> BaseException:
    """A fresh instance of `error_type`, with the fields a boundary has to carry set."""
    return _SAMPLES[error_type]()


def sampled_types() -> set[type[BaseException]]:
    return set(_SAMPLES)


_BY_NAME = {error_type.__qualname__: error_type for error_type in EXCEPTION_UNIVERSE}


def error_type_named(name: str) -> type[BaseException]:
    """The class in the universe with this qualified name, which a test passes across a boundary as a string."""
    return _BY_NAME[name]


@dataclass(frozen=True)
class Crossed:
    """What the near side of a boundary saw for an exception raised on the far side."""

    arrives_as: type[BaseException]
    """The class the near side raises."""
    retried: bool
    """Whether the engine attempts the unit again. On a boundary that belongs to no engine, whether the unit fails
    into the engine's retry policy rather than recording a result."""
    delay: float | None = None
    """How long the engine waits before that attempt, when the test measures it."""


@dataclass(frozen=True)
class Carried:
    """The near side raises the exception as its own class."""

    retried: bool
    reason: str
    delay: float | None = None

    def expected(self, error_type: type[BaseException]) -> Crossed:
        return Crossed(error_type, self.retried, self.delay)


@dataclass(frozen=True)
class Wrapped:
    """The near side raises a documented wrapper instead, which names the exception's class."""

    wrapper: type[BaseException]
    retried: bool
    reason: str
    delay: float | None = None

    def expected(self, error_type: type[BaseException]) -> Crossed:
        return Crossed(self.wrapper, self.retried, self.delay)


@dataclass(frozen=True)
class NotApplicable:
    """The exception can't reach this boundary, or the boundary can't be crossed in-process; no test runs."""

    reason: str


@dataclass(frozen=True)
class Red:
    """A cell that is wrong on main: `expected` runs as a strict expected failure until `tracked_by` is fixed."""

    expected: object
    """The verdict the cell should have, which the test for its boundary takes."""
    observed: str
    """What main does instead."""
    tracked_by: str

    @property
    def reason(self) -> str:
        return f'{self.observed} ({self.tracked_by})'


VerdictT = TypeVar('VerdictT')


def excluding(
    family: tuple[type[BaseException], ...], *members: type[BaseException]
) -> tuple[type[BaseException], ...]:
    """`family` without `members`, which a table gives a verdict of their own."""
    assert all(member in family for member in members)
    return tuple(member for member in family if member not in members)


def verdicts(
    *groups: tuple[Iterable[type[BaseException]], VerdictT],
) -> dict[type[BaseException], VerdictT]:
    """A boundary table: each group's classes get its verdict. A class in two groups fails."""
    table: dict[type[BaseException], VerdictT] = {}
    for members, verdict in groups:
        for member in members:
            assert member not in table, f'{member.__qualname__} has two verdicts'
            table[member] = verdict
    return table


def matrix_params(table: Mapping[type[BaseException], object]) -> list[ParameterSet]:
    """One test case per cell that runs: a `Red` cell's expected verdict as a strict expected failure, and no case for
    a `NotApplicable` one."""
    params: list[ParameterSet] = []
    for error_type, verdict in table.items():
        if isinstance(verdict, NotApplicable):
            continue
        if isinstance(verdict, Red):
            params.append(
                pytest.param(
                    error_type,
                    verdict.expected,
                    id=error_type.__qualname__,
                    marks=pytest.mark.xfail(strict=True, raises=AssertionError, reason=verdict.reason),
                )
            )
        else:
            params.append(pytest.param(error_type, verdict, id=error_type.__qualname__))
    return params
