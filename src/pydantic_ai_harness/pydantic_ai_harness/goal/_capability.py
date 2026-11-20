"""Enforce a caller-defined completion condition through core output retries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, durable_operation
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness._combine import one_per_id

_HEADLESS_INSTRUCTIONS = (
    'This run is unattended: nobody is available to answer questions. '
    'Work toward the goal before producing a final answer. '
    'Make reasonable decisions within the supplied constraints. '
    'Do not invent missing credentials, bypass approvals, or claim success without evidence. '
    'If completion is impossible, explain the blocker accurately.'
)


class GoalVerifier(Protocol[AgentDepsT]):
    """Return `None` when complete, or a nonempty explanation of what remains.

    Receives the processed output and can inspect dependencies and message history
    through `ctx`. Under durable execution both cross the engine's serializer: on
    Temporal a structured output arrives as JSON data and `ctx.messages` is unavailable.
    """

    async def __call__(self, ctx: RunContext[AgentDepsT], output: object, /) -> str | None:  # pragma: no cover
        """Evaluate fresh evidence of completion, not just the model's claim."""
        ...


@dataclass(kw_only=True)
class Goal(AbstractCapability[AgentDepsT]):
    """State a run's goal and, with a verifier, keep it working until the verifier accepts the final output.

    Rejections consume the agent's output retry budget. Configure
    `Agent(retries={'output': ...})` to bound continuation; usage limits still apply.
    """

    goal: str
    """The objective and any constraints the model should work toward."""

    verify: GoalVerifier[AgentDepsT] | None = None
    """Completion check, called for complete output only. `None` adds the goal instructions without enforcing them."""

    headless: bool = True
    """Tell the model nobody is available to answer questions. Set to `False` for interactive runs."""

    id: str | None = 'goal'
    """Stable capability ID, so its instructions are addressable and its verifier is a durable operation.

    Two `Goal`s sharing this `id` are one goal stated twice when they agree, and an error when they
    differ. To enforce several goals, give each its own `id`.
    """

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise UserError('`Goal` requires a nonempty `goal`.')

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Two under one `id` are one goal stated twice; two that disagree raise rather than merge."""
        return one_per_id(capabilities, remedy='Give each its own `id`')

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
            instructions += f'\n{_HEADLESS_INSTRUCTIONS}'
        return instructions

    async def after_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: object,
    ) -> object:
        """Reject incomplete final answers using the runtime's retry machinery."""
        if self.verify is None or ctx.partial_output:
            return output
        with ctx.tracer.start_as_current_span(
            'goal.verify',
            record_exception=ctx.trace_include_content,
            set_status_on_exception=ctx.trace_include_content,
        ) as span:
            gap = await self._verify(ctx, output)
            if gap is not None and not gap.strip():
                raise UserError('The `Goal` verifier must return `None` or a nonempty explanation.')
            if span.is_recording():
                span.set_attribute('goal.met', gap is None)
                if ctx.trace_include_content:
                    span.set_attribute('goal.description', self.goal)
                    if gap is not None:
                        span.set_attribute('goal.gap', gap)
        if gap is not None:
            unattended = 'Nobody is available to answer questions. ' if self.headless else ''
            raise ModelRetry(
                f'Goal not met: {self.goal}\nWhat remains: {gap}\n{unattended}'
                'Continue within the supplied constraints; do not bypass approvals or fabricate success.'
            )
        return output

    @durable_operation('verify')
    async def _verify(self, ctx: RunContext[AgentDepsT], output: object) -> str | None:
        """Run the verifier, as a durable operation inside a workflow or flow."""
        assert self.verify is not None
        return await self.verify(ctx, output)
