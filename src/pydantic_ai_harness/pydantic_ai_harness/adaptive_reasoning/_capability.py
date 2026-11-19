"""Pick the thinking effort for each model request from what the request answers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import BaseToolReturnPart, ModelRequest, RetryFeedbackPart, UserPromptPart
from pydantic_ai.settings import ModelSettings, ThinkingLevel
from pydantic_ai.tools import AgentDepsT, RunContext

__all__ = ['AdaptiveReasoning']


@dataclass(kw_only=True)
class AdaptiveReasoning(AbstractCapability[AgentDepsT]):
    """Set the unified `thinking` setting per model request instead of once per run.

    By default a request that only returns successful tool results gets `routine` effort, and every other request
    (a new user prompt, a tool result that isn't successful, retry feedback) gets `deliberate` effort. Pass `select` to
    replace that policy.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import AdaptiveReasoning

    agent = Agent('openai:gpt-6-astra', capabilities=[AdaptiveReasoning(deliberate='high', routine='low')])
    ```
    """

    deliberate: ThinkingLevel = 'high'
    """The effort for a request that starts a turn or follows a failure."""

    routine: ThinkingLevel = 'medium'
    """The effort for a request that only returns successful tool results."""

    select: Callable[[RunContext[AgentDepsT]], ThinkingLevel | None] | None = None
    """Replaces the default policy, so `deliberate` and `routine` don't apply. Called once per step with the `RunContext`.

    The last `ModelRequest` in `ctx.messages` is the one the step answers. `ctx.messages` ends in a suspended
    `ModelResponse` instead when the step resumes a turn the provider paused.
    Return `None` to leave `thinking` as the model, agent and earlier capabilities set it.
    """

    id: str | None = 'adaptive_reasoning'
    """One per agent: two resolve to one, and the later one's settings win."""

    def get_model_settings(self) -> Callable[[RunContext[AgentDepsT]], ModelSettings]:
        def resolve(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            effort = self.select(ctx) if self.select is not None else self._default_effort(ctx)
            return ModelSettings() if effort is None else ModelSettings(thinking=effort)

        return resolve

    def _default_effort(self, ctx: RunContext[AgentDepsT]) -> ThinkingLevel:
        parts = next(message for message in reversed(ctx.messages) if isinstance(message, ModelRequest)).parts
        returns = [part for part in parts if isinstance(part, BaseToolReturnPart)]
        routine = (
            bool(returns)
            and all(part.outcome == 'success' for part in returns)
            and not any(isinstance(part, UserPromptPart | RetryFeedbackPart) for part in parts)
        )
        return self.routine if routine else self.deliberate
