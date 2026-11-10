"""Send an agent back to review and revise its output before the run ends."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentNode, NodeResult
from pydantic_ai.messages import ModelRequest, SystemPromptPart
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests
from pydantic_graph import End

DEFAULT_REFINE_OUTPUT_INSTRUCTIONS = (
    'Review your last answer against the request. Look for mistakes, gaps, and anything unclear or unsupported. '
    'Fix what you find, using tools where needed, then give the complete final answer again, revised. '
    'Leave the review itself out of the answer.'
)
"""What `RefineOutput` tells the agent at the start of each review round, unless `instructions` is set."""

_ORIGIN = 'pydantic-ai-harness.refine_output.v1'
"""The `origin` in the `metadata` of the `SystemPromptPart` a review round is delivered as."""


@dataclass(kw_only=True)
class RefineOutput(AbstractCapability[AgentDepsT]):
    """Send the agent back to review and revise its output before the run ends.

    Each time the run would end with an output, a system prompt carrying `instructions` is added to the
    conversation and the model answers again, up to `rounds` times per run. The last answer is the
    run's output.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import RefineOutput

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[RefineOutput()])
    ```

    Rounds are counted from the run's own messages, so a run that continues an earlier conversation
    starts with none used. A run that pauses for deferred tools is not reviewed.
    """

    rounds: int = 1
    """How many times per run the agent reviews its output."""

    instructions: str = DEFAULT_REFINE_OUTPUT_INSTRUCTIONS
    """What the agent is told at the start of each review round."""

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError(f'`rounds` must be at least 1, got {self.rounds}.')

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        if not isinstance(result, End) or isinstance(result.data.output, DeferredToolRequests):
            return result
        round_number = _rounds_started(ctx) + 1
        if round_number <= self.rounds:
            part = SystemPromptPart(self.instructions, metadata={'origin': _ORIGIN, 'round': round_number})
            ctx.enqueue(part, priority='when_idle')
        return result


def _rounds_started(ctx: RunContext[AgentDepsT]) -> int:
    """How many review rounds this run's messages already hold."""
    return max(
        (
            part.metadata['round']
            for message in ctx.messages
            if isinstance(message, ModelRequest) and message.run_id == ctx.run_id
            for part in message.parts
            if isinstance(part, SystemPromptPart) and part.metadata and part.metadata.get('origin') == _ORIGIN
        ),
        default=0,
    )
