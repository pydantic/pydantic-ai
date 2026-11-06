"""Request, wall-clock and delegation budgets."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from opentelemetry.trace import Span, Status, StatusCode, get_current_span

from pydantic_ai import AgentRunResult, ModelRequestContext, RunContext, UsageLimitExceeded
from pydantic_ai.capabilities import (
    AbstractCapability,
    ValidatedToolArgs,
    WrapModelRequestHandler,
    WrapRunHandler,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.tools import ToolDefinition

from ._config import RUN_TIME_RESERVE_SECS, request_limit, run_timeout_secs
from ._safe_outputs import safe_output_pending

REQUEST_BUDGET_NOTICE = (
    '## Request budget nearly spent\n\n'
    'This run is within its last tenth of model requests, and reaching the limit '
    'stops it with nothing posted. Stop investigating. Emit the safe output your '
    'task ends with now, from what you have already found.'
)

TIME_BUDGET_NOTICE = (
    '## Time budget nearly spent\n\n'
    f'This run has under {RUN_TIME_RESERVE_SECS // 60} minutes of wall-clock time left, and reaching '
    'the limit stops it with nothing posted. Stop investigating. Emit the safe output your '
    'task ends with now, from what you have already found.'
)

DELEGATE_TOOL_NAME = 'delegate_task'

_run_deadline: float | None = None
_in_delegate: ContextVar[bool] = ContextVar('_in_delegate', default=False)
_wall_clock_caps: ContextVar[tuple[Callable[[], bool], ...]] = ContextVar('_wall_clock_caps', default=())


def _mark_timed_out(span: Span) -> None:
    span.set_attribute('error.type', 'timeout')
    span.set_status(Status(StatusCode.ERROR, 'timeout'))


def _wall_clock_cap_expired() -> bool:
    return any(expired() for expired in _wall_clock_caps.get())


def in_delegate() -> bool:
    """Whether the current run is a delegated sub-agent, which has no safe output of its own."""
    return _in_delegate.get()


def run_seconds_left() -> float | None:
    return None if _run_deadline is None else _run_deadline - time.monotonic()


def time_budget_notice() -> str | None:
    seconds_left = run_seconds_left()
    if seconds_left is None or seconds_left > RUN_TIME_RESERVE_SECS:
        return None
    return TIME_BUDGET_NOTICE


def request_budget_notice(ctx: RunContext[object]) -> str | None:
    limit = request_limit()
    if limit - ctx.usage.requests > limit // 10:
        return None
    return REQUEST_BUDGET_NOTICE


@dataclass
class RequestBudget(AbstractCapability[object]):
    """Stop at the workflow's request cap before the next model call.

    The CLI's own limit is left above it because pydantic-ai checks that limit
    before any hook runs, which would hide a safe output already emitted on the
    last request from `LimitAfterOutputIsSuccess`.
    """

    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        limit = request_limit()
        if ctx.usage.requests >= limit:
            raise UsageLimitExceeded(f'The next request would exceed the request limit of {limit}')
        return request_context


@asynccontextmanager
async def _wall_clock_cap(seconds: float) -> AsyncGenerator[None]:
    cap = asyncio.timeout(seconds)
    token = _wall_clock_caps.set((*_wall_clock_caps.get(), cap.expired))
    try:
        async with cap:
            yield
    except TimeoutError:
        if cap.expired():
            _mark_timed_out(get_current_span())
        raise
    finally:
        _wall_clock_caps.reset(token)


@dataclass
class WallClockBudget(AbstractCapability[object]):
    """End the run two minutes before the job timeout and mark the cut-off spans."""

    seconds: float

    async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        global _run_deadline
        _run_deadline = time.monotonic() + self.seconds
        try:
            async with _wall_clock_cap(self.seconds):
                return await handler()
        finally:
            _run_deadline = None

    async def wrap_model_request(
        self, ctx: RunContext[object], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        try:
            return await handler(request_context)
        except asyncio.CancelledError:
            if _wall_clock_cap_expired():
                _mark_timed_out(get_current_span())
            raise


@dataclass
class RecordRunTotals(AbstractCapability[object]):
    """Record what the whole run spent on its root span, delegates included."""

    async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        try:
            return await handler()
        finally:
            usage = ctx.usage
            get_current_span().set_attributes(
                {
                    'gh_aw.total_usage.requests': usage.requests,
                    'gh_aw.total_usage.input_tokens': usage.input_tokens,
                    'gh_aw.total_usage.output_tokens': usage.output_tokens,
                    'gh_aw.total_usage.cache_read_tokens': usage.cache_read_tokens,
                    'gh_aw.total_usage.cache_write_tokens': usage.cache_write_tokens,
                }
            )


@dataclass
class LimitAfterOutputIsSuccess(AbstractCapability[object]):
    """Keep a run that spent its request budget after publishing a safe output successful.

    The shim behaved this way: the closing turn is lost, but the task is done and
    gh-aw has the output it needs.
    """

    async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        try:
            return await handler()
        except UsageLimitExceeded:
            if safe_output_pending():
                raise
            return AgentRunResult('request limit reached after the safe output was emitted')


@dataclass
class DelegateReserve(AbstractCapability[object]):
    """Cut a delegation off before the run's own time reserve.

    CI Review needs up to five minutes between its last delegate returning and its
    posted review, so a delegate started inside the reserve is refused.
    """

    async def wrap_tool_execute(
        self,
        ctx: RunContext[object],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> object:
        if tool_def.name != DELEGATE_TOOL_NAME:
            return await handler(args)
        token = _in_delegate.set(True)
        try:
            seconds_left = run_seconds_left()
            if seconds_left is None:
                return await handler(args)
            budget = math.floor(seconds_left - RUN_TIME_RESERVE_SECS)
            if budget <= 0:
                return 'error: the time budget is nearly spent; finish the task from what you already have'
            try:
                async with _wall_clock_cap(budget):
                    return await handler(args)
            except TimeoutError:
                return f'error: the delegate timed out after {math.ceil(budget / 60)}min'
        finally:
            _in_delegate.reset(token)


def default_wall_clock_seconds() -> int:
    return run_timeout_secs()
