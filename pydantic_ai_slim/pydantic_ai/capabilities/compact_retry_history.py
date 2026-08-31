from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, RetryPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass
class CompactRetryHistory(AbstractCapability[AgentDepsT]):
    """Keep only the last failed attempt and its retry prompt in history sent to the model.

    By default, every failed output or tool attempt stays in the message history, so
    tokens grow with `retries=N`. This capability is opt-in: without it, the full retry
    streak is preserved. With it, a trailing streak of
    (`ModelResponse`, retry-only `ModelRequest`) pairs is collapsed to the most recent
    pair before each model request.

    A request is retry-only when every part is a [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart].
    Mixed requests (a tool return alongside a retry prompt) are left untouched, as are
    earlier conversation turns and successful tool exchanges.

    History processors replace the run's message history, so `all_messages()` after the
    run also reflects the compacted streak.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import CompactRetryHistory

    agent = Agent('openai:gpt-5.2', retries=5, capabilities=[CompactRetryHistory()])
    ```
    """

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        compacted = _keep_last_retry_pair(request_context.messages)
        if compacted is not request_context.messages:
            request_context.messages = compacted
        return request_context


def _is_retry_only_request(message: ModelMessage) -> bool:
    return (
        isinstance(message, ModelRequest)
        and bool(message.parts)
        and all(isinstance(part, RetryPromptPart) for part in message.parts)
    )


def _keep_last_retry_pair(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop older trailing retry pairs, keeping the last (`ModelResponse`, retry request)."""
    pair_starts: list[int] = []
    index = len(messages) - 1
    while index >= 1 and _is_retry_only_request(messages[index]) and isinstance(messages[index - 1], ModelResponse):
        pair_starts.append(index - 1)
        index -= 2

    if len(pair_starts) <= 1:
        return messages

    earliest_pair = pair_starts[-1]
    last_pair = pair_starts[0]
    return [*messages[:earliest_pair], *messages[last_pair:]]
