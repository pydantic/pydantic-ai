"""Capability that waits for a person's decision on deferred tool calls inside a durable agent run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import KW_ONLY, dataclass
from datetime import timedelta

from pydantic import TypeAdapter

from pydantic_ai.durable_exec._external_events import ExternalEventError, wait_for_external_event
from pydantic_ai.exceptions import ToolFailed, UserError
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, DeferredToolResults, RunContext, ToolDenied

from .abstract import AbstractCapability, CapabilityOrdering

_RESULTS_ADAPTER = TypeAdapter(DeferredToolResults)


@dataclass
class DurableApprovals(AbstractCapability[AgentDepsT]):
    """Wait for a person to decide on deferred tool calls without ending the durable agent run.

    When tools require approval or external execution, the run hands a resume token to `announce`
    and suspends until the token is resolved with the decision, then continues with the approved
    calls executed, the denied ones reported to the model, and the external results filled in. The
    decision is the JSON form of [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults].

    Requires a durable agent run on an engine that can wait for external events, currently AWS Lambda
    durable functions with Pydantic AI Harness's `AWSLambdaDurability`. Anywhere else the run fails
    with a `UserError` when a call needs a decision.

    See [Durable Approvals](../capabilities/durable-approvals.md).
    """

    announce: Callable[[RunContext[AgentDepsT], DeferredToolRequests, str], Awaitable[None] | None]
    """Hands the resume token to whoever decides, for example by posting it to Slack or writing a database row.

    Called with the run context, the pending requests and the token, once per decision, inside a
    durable unit that is not run again on replay. Can be sync or async.
    """

    _: KW_ONLY

    timeout: timedelta | None = None
    """How long to wait for the decision before giving up, or `None` to wait as long as the engine allows.

    When it elapses, or the token is resolved with a failure instead of a decision, every pending
    approval is denied and every pending external call fails, with a message to the model saying why.
    """

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
            payload = await wait_for_external_event(
                'deferred_tool_calls',
                announce=lambda token: self.announce(ctx, requests, token),
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
            return requests.build_results(
                approvals=decision.approvals, calls=decision.calls, metadata=decision.metadata
            )
        # Both a payload that is not `DeferredToolResults` (a `ValidationError`) and one naming calls
        # that are not pending, such as a decision sent for an earlier wait.
        except ValueError as error:
            raise UserError(f'The decision sent for the pending deferred tool calls is not valid: {error}') from error
