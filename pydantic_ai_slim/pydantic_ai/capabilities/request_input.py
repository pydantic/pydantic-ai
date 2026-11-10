"""Capability that asks for input on deferred tool calls and continues the agent run with the answer."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import KW_ONLY, dataclass
from datetime import timedelta
from functools import partial

from pydantic import JsonValue, TypeAdapter

from pydantic_ai._run_context import set_current_run_context
from pydantic_ai._utils import await_maybe, is_async_callable, run_in_executor
from pydantic_ai.exceptions import ToolFailed, UserError
from pydantic_ai.external_events import (
    ExternalEventError,
    _check_timeout,  # pyright: ignore[reportPrivateUsage]
    wait_for_external_event,
)
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, DeferredToolResults, RunContext, ToolDenied

from .abstract import AbstractCapability, CapabilityOrdering

_RESULTS_ADAPTER = TypeAdapter(DeferredToolResults)


@dataclass
class RequestInput(AbstractCapability[AgentDepsT]):
    """Ask for input on deferred tool calls, and continue the same agent run with the answer.

    When tools require approval, ask a question or need an external result, the run passes the pending
    calls and a resume token to `ask`. `ask` either answers right away by returning
    [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults], as a terminal prompt, a bot or a
    policy does, or hands the token to whoever will answer, by posting a Slack message, sending an
    email or writing a database row, and returns `None`. The run then waits until code you write
    elsewhere resolves the token with the JSON form of `DeferredToolResults`. Either way, the run
    continues with the approved calls executed, the denied ones reported to the model, and the
    external results filled in.

    The wait is `wait_for_external_event`. In a
    plain run it happens in this process, and your code resolves the token with
    `resolve_external_event`. In a durable run on
    an engine that supports it, currently AWS Lambda durable functions with Pydantic AI Harness's
    `AWSLambdaDurability`, the engine records the answer and suspends the run while it waits, and the
    token is resolved through the engine's API.

    See [Request Input](../capabilities/request-input.md).
    """

    ask: Callable[
        [RunContext[AgentDepsT], DeferredToolRequests, str],
        Awaitable[DeferredToolResults | None] | DeferredToolResults | None,
    ]
    """Answers the pending calls, or hands them and the resume token to whoever will.

    Called with the run context, the pending requests and the token, once per wait. Return the
    answer as `DeferredToolResults`, or `None` to wait until the token is resolved. Can be sync or
    async; a sync function runs in a worker thread, like a sync tool. In a durable run it runs inside a
    durable unit that records its answer, so a replay doesn't call it again, but the engine can retry
    it when it fails: return quickly, and make it idempotent, for example by keying a message or
    database row on the token.
    """

    _: KW_ONLY

    timeout: timedelta | None = None
    """How long to wait for the answer, or `None` to wait without a limit (in a durable run, as long as the engine allows).

    When it elapses, or the token is resolved with a failure instead of an answer, every pending
    approval is denied and every pending external call fails, with a message to the model saying why.
    In process, it also bounds `ask`: an async `ask` still running when it elapses is cancelled.
    """

    allow_non_durable_wait: bool = False
    """Wait in process inside a durable run whose engine doesn't support waiting for external events, instead of raising a `UserError`.

    The wait is then not recorded: when the engine replays the run, `ask` is called again, and when
    the process restarts, the wait and any answer to it are lost. It doesn't work on Temporal, whose
    workflow code can't be woken from outside the workflow.
    """

    def __post_init__(self) -> None:
        _check_timeout(self.timeout)

    def get_ordering(self) -> CapabilityOrdering:
        """Ask last: every handler that can resolve a call without waiting goes first."""
        return CapabilityOrdering(position='innermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults:
        try:
            # Current, so the wait can tell a durable run whose engine can't wait from a plain run.
            with set_current_run_context(ctx):
                payload = await wait_for_external_event(
                    'deferred_tool_calls',
                    ask=partial(self._ask, ctx, requests),
                    timeout=self.timeout,
                    allow_non_durable_wait=self.allow_non_durable_wait,
                )
        except ExternalEventError as error:
            message = f'No decision on this tool call arrived: {error}'
            return requests.build_results(
                approvals={call.tool_call_id: ToolDenied(message) for call in requests.approvals},
                calls={call.tool_call_id: ToolFailed(message) for call in requests.calls},
            )
        try:
            decision = _RESULTS_ADAPTER.validate_python(payload)
            # `build_results` checks `approvals` and `calls` only; without this, a decision could set the
            # `tool_call_metadata` of a call an earlier handler already resolved.
            pending_ids = {call.tool_call_id for call in (*requests.approvals, *requests.calls)}
            for field_name, ids in (('metadata', decision.metadata), ('answered_by', decision.answered_by)):
                if extra_ids := set(ids) - pending_ids:
                    raise ValueError(f'`{field_name}` contains tool call IDs that are not pending: {sorted(extra_ids)}')
            return requests.build_results(
                approvals=decision.approvals,
                calls=decision.calls,
                metadata=decision.metadata,
                answered_by=decision.answered_by,
            )
        # Both a payload that is not `DeferredToolResults` (a `ValidationError`) and one naming calls
        # that are not pending, such as a decision sent for an earlier wait.
        except ValueError as error:
            raise UserError(f'The decision sent for the pending deferred tool calls is not valid: {error}') from error

    async def _ask(self, ctx: RunContext[AgentDepsT], requests: DeferredToolRequests, token: str) -> JsonValue:
        """Call `ask`, and turn an answer into the JSON form a resolver would send, so both reach the run the same way."""
        if is_async_callable(self.ask):
            results = await await_maybe(self.ask(ctx, requests, token))
        else:
            results = await await_maybe(await run_in_executor(self.ask, ctx, requests, token))
        return None if results is None else _RESULTS_ADAPTER.dump_python(results, mode='json')
