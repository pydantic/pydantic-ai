"""The `DenyUnanswered` capability: deny calls nobody approved or answered."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    HandleDeferredToolCalls,
    WrapperCapability,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolApprovalResult,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDenied,
)
from pydantic_ai_harness.ask_user import TOOL_NAME

__all__ = ['DEFAULT_MESSAGE', 'DEFAULT_QUESTION_MESSAGE', 'DenyUnanswered']

DEFAULT_MESSAGE = 'No one was available to approve it, so it did not run.'
DEFAULT_QUESTION_MESSAGE = 'No one is available to answer in this run; continue without the answer.'


@dataclass(kw_only=True)
class DenyUnanswered(AbstractCapability[AgentDepsT]):
    """Deny approval-required calls and unanswered questions that nobody attended.

    For runs that may end with no one available to approve or answer: a held approval is denied
    with `message`, and a deferred `ask_user_question` call is answered with `question_message`,
    so the run finishes instead of pausing on `DeferredToolRequests`. Any other deferred call
    stays deferred for its own handler. Without this capability nothing changes.

    Emits nothing into telemetry.
    """

    # It contributes no toolsets or durable operations and its handler is deterministic, so `SubAgents` and
    # `DynamicWorkflow` can add it to a child run under durable execution.
    _safe_at_runtime: ClassVar[bool] = True

    message: str = DEFAULT_MESSAGE
    """The tool result for a denied approval."""

    question_message: str = DEFAULT_QUESTION_MESSAGE
    """The tool result for an unanswered `ask_user_question` call."""

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
        questions = [call.tool_call_id for call in requests.calls if call.tool_name == TOOL_NAME]
        if not approvals and not questions:
            return None
        return DeferredToolResults(
            approvals=approvals, calls={tool_call_id: self.question_message for tool_call_id in questions}
        )


def find_deny_unanswered(ctx: RunContext[AgentDepsT]) -> DenyUnanswered[AgentDepsT] | None:
    """Return the run's `DenyUnanswered`, unwrapping `WrapperCapability` chains."""
    try:
        capabilities = ctx.capabilities
    except UserError:
        # A durable-run context does not carry capabilities across the engine
        # boundary, so there is nothing to forward inside a durable unit.
        return None
    candidate: AbstractCapability[AgentDepsT] | None
    for candidate in capabilities.values():
        while candidate is not None:
            if isinstance(candidate, DenyUnanswered):
                return candidate
            candidate = candidate.wrapped if isinstance(candidate, WrapperCapability) else None
    return None
