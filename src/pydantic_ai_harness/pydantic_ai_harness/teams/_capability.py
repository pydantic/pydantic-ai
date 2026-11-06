"""The `Team` capability: gate listeners installed on the lead's run and on every member's run."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability, AgentNode, CapabilityOrdering, CombinedCapability, NodeResult
from pydantic_ai.output import PendingModelResponse
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, RunContext
from pydantic_ai_harness.subagents._events import bounded_text
from pydantic_ai_harness.teams._events import TeammateIdleEvent
from pydantic_ai_harness.teams._gates import joined, trace_rejection
from pydantic_graph import End


@dataclass
class _IdleGate(AbstractCapability[AgentDepsT]):
    """Asks the team's gates whether a member may go idle, as its run is about to end."""

    member: str

    def get_ordering(self) -> CapabilityOrdering:
        # Outermost runs `after_node_run` last, so the gate sees what other capabilities enqueued
        # there (a background tool's result, say) and does not fire for a run that continues anyway.
        return CapabilityOrdering(position='outermost')

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        # Queued messages turn the end into another model request, so the member is not going idle.
        if not isinstance(result, End) or ctx.pending_messages:
            return result
        output = result.data.output
        # A run paused for deferred tools or a pending provider response is waiting, not idle.
        if isinstance(output, DeferredToolRequests | PendingModelResponse):
            return result
        text, truncated = bounded_text(str(output))
        event = await ctx.emit(TeammateIdleEvent(member=self.member, final_output=text, truncated=truncated))
        if event.feedback:
            trace_rejection(ctx, 'teammate_idle', event.feedback, member=self.member, item_id=None)
            # Core's end-of-run drain turns content enqueued here into the member's next model request.
            ctx.enqueue(joined(event.feedback), priority='asap')
        return result


class Team(CombinedCapability[AgentDepsT]):
    """Gates that check a team's work: when members go idle, and what goes on the shared board.

    A gate is a capability that listens to the `teams` gate events (`TeammateIdleEvent`,
    `TeamTaskCreatingEvent`, `TeamTaskCompletingEvent`) and rejects with feedback. Pass `Team` to the
    lead's agent run, and `member_capability(name)` to each member's run, so the same gates check
    both: a task the lead adds is gated like one a member adds.

    Gates run only inside agent runs. Changes the application or the user makes to the board
    directly, outside any run, are not gated.

    Args:
        gates: Capabilities whose event listeners decide the gates, such as a `Hooks` with
            `hooks.on.event(TeamTaskCompletingEvent)` listeners. The same instances are used in the
            lead's run and in every member's run.
    """

    gates: tuple[AbstractCapability[AgentDepsT], ...]

    def __init__(self, *, gates: Sequence[AbstractCapability[AgentDepsT]] = ()) -> None:
        self.gates = tuple(gates)
        super().__init__(list(self.gates))

    def member_capability(self, name: str) -> AbstractCapability[AgentDepsT]:
        """The capability to add to the run of the member called `name`.

        It holds the team's gates and the idle gate: when the member's run is about to end with a
        final output, it emits a `TeammateIdleEvent`, and feedback from `keep_working` becomes the
        member's next user message, so the run continues.
        """
        return CombinedCapability[AgentDepsT]([*self.gates, _IdleGate[AgentDepsT](member=name)])
