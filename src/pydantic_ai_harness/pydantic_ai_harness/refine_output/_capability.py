"""Send an agent back to review and revise its output before the run ends."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from pydantic_ai import CallToolsNode, DeferredToolRequests, PendingModelResponse, RunContext
from pydantic_ai._capability_parts import check_operator_text
from pydantic_ai.capabilities import AbstractCapability, AgentNode, NodeResult
from pydantic_ai.messages import InstructionCapabilityPart, OperatorSource, OperatorText
from pydantic_ai.tools import AgentDepsT
from pydantic_graph import End

DEFAULT_REFINE_OUTPUT_INSTRUCTIONS = (
    'Review your last answer against the request. Look for mistakes, gaps, and anything unclear or unsupported. '
    'Fix what you find, using tools where needed, then give the complete final answer again, revised. '
    'Leave the review itself out of the answer.'
)
"""What `RefineOutput` tells the agent at the start of each review round, unless `instructions` is set."""


@dataclass(kw_only=True)
class ReviewRoundPart(InstructionCapabilityPart, namespace='refine_output', name='review_round'):
    """The instruction `RefineOutput` adds to the conversation at the start of a review round.

    Its `content` is the capability's `instructions`, in the application's voice. Find review rounds in history by this
    type.
    """

    round: int
    """Which of the run's review rounds it starts, from 1. Never sent to the model."""


@dataclass(kw_only=True)
class RefineOutput(AbstractCapability[AgentDepsT]):
    """Send the agent back to review and revise its output before the run ends.

    Each time the run would end with an output, a [`ReviewRoundPart`][pydantic_ai_harness.refine_output.ReviewRoundPart]
    carrying `instructions` is added to the conversation and the model answers again, up to `rounds` times per run. The last answer is the
    run's output.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import RefineOutput

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[RefineOutput()])
    ```

    Not reviewed: a run that ends for its caller to resume, an output that did not come from a model
    request, and `Agent.run_stream()`.
    """

    rounds: int = 1
    """How many times per run the agent reviews its output."""

    instructions: str = DEFAULT_REFINE_OUTPUT_INSTRUCTIONS
    """What the agent is told at the start of each review round."""

    _rounds_started: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError(f'`rounds` must be at least 1, got {self.rounds}.')
        check_operator_text(self.instructions, what='`RefineOutput.instructions`')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> RefineOutput[AgentDepsT]:
        """Return a fresh instance, so runs don't share the rounds they used."""
        return replace(self)

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        # `Agent.run_stream()` ends through a node of its own, after which the run cannot continue.
        # A response without a `model_name` came from no model, like an input guardrail's refusal;
        # a custom `Model` that leaves `model_name` unset is therefore not reviewed either.
        if (
            self._rounds_started < self.rounds
            and isinstance(node, CallToolsNode)
            and node.model_response.model_name is not None
            and isinstance(result, End)
            and not isinstance(result.data.output, DeferredToolRequests | PendingModelResponse)
        ):
            self._rounds_started += 1
            review = ReviewRoundPart(
                content=OperatorText(self.instructions), source=OperatorSource(), round=self._rounds_started
            )
            ctx.enqueue(review, priority='when_idle')
        return result
