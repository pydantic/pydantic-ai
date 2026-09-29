"""Enforce a caller-defined completion condition through core output retries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import AgentDepsT


class GoalVerifier(Protocol[AgentDepsT]):
    """Return `None` when complete, or a nonempty explanation of what remains.

    Receives the processed output unchanged and can inspect dependencies and
    message history through `ctx`. Exceptions propagate to the caller.
    """

    async def __call__(self, ctx: RunContext[AgentDepsT], output: object, /) -> str | None:  # pragma: no cover
        """Evaluate fresh evidence of completion, not just the model's claim."""
        ...


@dataclass(kw_only=True)
class Goal(AbstractCapability[AgentDepsT]):
    """Keep a headless run working until its verifier accepts the final output.

    Rejections consume the agent's output retry budget. Configure
    `Agent(retries={'output': ...})` to bound continuation; usage limits still apply.
    """

    goal: str
    """The objective and any constraints the model should work toward."""

    verify: GoalVerifier[AgentDepsT]
    """Completion check, called only for complete output in headless mode."""

    headless: bool = True
    """Set to `False` to allow interactive questions without enforcing completion."""

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise UserError('Goal requires a nonempty goal.')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Callable verifiers cannot be serialized as agent specifications."""
        return None

    def get_ordering(self) -> CapabilityOrdering:
        """Check the processed output inside instrumentation."""
        return CapabilityOrdering(position='outermost', wrapped_by=[Instrumentation])

    def get_instructions(self) -> str:
        instructions = f'Your goal is:\n{self.goal}'
        if self.headless:
            instructions += (
                '\nThis run is unattended: nobody is available to answer questions. '
                'Work toward the goal before producing a final answer. '
                'Make reasonable decisions within the supplied constraints. '
                'Do not invent missing credentials, bypass approvals, or claim success without evidence. '
                'If completion is impossible, explain the blocker accurately.'
            )
        return instructions

    async def after_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: object,
    ) -> object:
        """Reject incomplete final answers using the runtime's retry machinery."""
        if not self.headless or ctx.partial_output:
            return output
        with ctx.tracer.start_as_current_span('goal.verify') as span:
            gap = await self.verify(ctx, output)
            if gap is not None and not gap.strip():
                raise UserError('Goal verifier must return None or a nonempty explanation.')
            if span.is_recording():
                span.set_attribute('goal.met', gap is None)
                if ctx.trace_include_content:
                    span.set_attribute('goal.description', self.goal)
                    if gap is not None:
                        span.set_attribute('goal.gap', gap)
        if gap is not None:
            raise ModelRetry(
                f'Goal not met: {self.goal}\nWhat remains: {gap}\n'
                'Nobody is available to answer questions. Continue within the supplied constraints; '
                'do not bypass approvals or fabricate success.'
            )
        return output
