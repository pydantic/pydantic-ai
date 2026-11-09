"""Capability that waits for a person's decision on deferred tool calls inside the agent run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import KW_ONLY, dataclass
from datetime import timedelta
from functools import partial

from pydantic import TypeAdapter

from pydantic_ai._run_context import set_current_run_context
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
class AwaitApproval(AbstractCapability[AgentDepsT]):
    """Wait for a person to decide on deferred tool calls, and continue the same agent run with the decision.

    When tools require approval or external execution, the run hands a resume token to `announce`, which
    shows the pending calls and the token to a person, and waits until the token is resolved with the
    decision: the JSON form of [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults]. The run
    then continues with the approved calls executed, the denied ones reported to the model, and the
    external results filled in.

    The wait is [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event]. In a
    plain run it happens in this process, and your code resolves the token with
    [`resolve_external_event`][pydantic_ai.external_events.resolve_external_event]. In a durable run on
    an engine that supports it, currently AWS Lambda durable functions with Pydantic AI Harness's
    `AWSLambdaDurability`, the engine records the wait and suspends the run while it lasts, and the
    token is resolved through the engine's API.

    See [Await Approval](../capabilities/await-approval.md).
    """

    announce: Callable[[RunContext[AgentDepsT], DeferredToolRequests, str], Awaitable[None] | None]
    """Shows the pending calls and the resume token to whoever decides: a Slack message, an email, a database row.

    It does not return the decision: code you write elsewhere resolves the token with it. Called with
    the run context, the pending requests and the token, once per decision; in a durable run, inside a
    durable unit that is not run again on replay. Can be sync or async; a sync function runs in a
    worker thread, like a sync tool.
    """

    _: KW_ONLY

    timeout: timedelta | None = None
    """How long to wait for the decision, or `None` to wait without a limit (in a durable run, as long as the engine allows).

    When it elapses, or the token is resolved with a failure instead of a decision, every pending
    approval is denied and every pending external call fails, with a message to the model saying why.
    """

    def __post_init__(self) -> None:
        _check_timeout(self.timeout)

    def get_ordering(self) -> CapabilityOrdering:
        """Ask last: every handler that can resolve a call without waiting for a person goes first."""
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
                    announce=partial(self.announce, ctx, requests),
                    timeout=self.timeout,
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
            if extra_metadata := set(decision.metadata) - pending_ids:
                raise ValueError(f'`metadata` contains tool call IDs that are not pending: {sorted(extra_metadata)}')
            return requests.build_results(
                approvals=decision.approvals, calls=decision.calls, metadata=decision.metadata
            )
        # Both a payload that is not `DeferredToolResults` (a `ValidationError`) and one naming calls
        # that are not pending, such as a decision sent for an earlier wait.
        except ValueError as error:
            raise UserError(f'The decision sent for the pending deferred tool calls is not valid: {error}') from error
