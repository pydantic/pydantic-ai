"""The `DenyUnanswered` capability: deny calls nobody approved or answered."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    HandleDeferredToolCalls,
    WrapperCapability,
)
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolApprovalResult,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDenied,
)

from ..ask_user import AskUserRequest, AskUserResponse, AskUserResult, ask_user_result

__all__ = ['DEFAULT_MESSAGE', 'DenyUnanswered', 'find_deny_unanswered']

DEFAULT_MESSAGE = 'No one was available to approve it, so it did not run.'


@dataclass(kw_only=True)
class DenyUnanswered(AbstractCapability[AgentDepsT]):
    """Deny approval-required calls and unanswered questions that nobody attended.

    For runs that may end with no one available to approve or answer: a held approval is denied
    with `message`, and a deferred `ask_user_question` call is answered as cancelled through the
    `ask_user` request/result plumbing, so the run finishes instead of pausing on
    `DeferredToolRequests`. Any other deferred call stays deferred for its own handler. Without
    this capability nothing changes.

    Emits nothing into telemetry.
    """

    message: str = DEFAULT_MESSAGE

    def get_ordering(self) -> CapabilityOrdering:
        """After every handler that may ask the user."""
        return CapabilityOrdering(position='innermost', wrapped_by=[HandleDeferredToolCalls])

    async def handle_deferred_tool_calls(
        self, ctx: RunContext[AgentDepsT], *, requests: DeferredToolRequests
    ) -> DeferredToolResults | None:
        """Deny the calls nobody attended."""
        approvals: dict[str, bool | DeferredToolApprovalResult] = {
            part.tool_call_id: ToolDenied(self.message) for part in requests.approvals
        }
        declines: dict[str, AskUserResult] = {}
        for call in requests.calls:
            try:
                request = AskUserRequest.from_tool_call(call)
            except ValueError:
                continue
            declines[call.tool_call_id] = ask_user_result(request, AskUserResponse(cancelled=True))
        if not approvals and not declines:
            return None
        return DeferredToolResults(approvals=approvals, calls=declines)


def find_deny_unanswered(ctx: RunContext[AgentDepsT]) -> DenyUnanswered[AgentDepsT] | None:
    """Return the run's `DenyUnanswered`, unwrapping `WrapperCapability` chains."""
    candidate: AbstractCapability[AgentDepsT] | None
    for candidate in ctx.capabilities.values():
        while candidate is not None:
            if isinstance(candidate, DenyUnanswered):
                return candidate
            candidate = candidate.wrapped if isinstance(candidate, WrapperCapability) else None
    return None
