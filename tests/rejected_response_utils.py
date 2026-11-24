"""A model that announces a lookup before it answers, and the hooks that reject the announcement.

The tests of `ModelResponseRejectedEvent` across the agent, UI adapter and durable execution suites share them, so
every suite rejects the same response for the same reason.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pydantic_ai import ModelResponse, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo

ANNOUNCEMENT = "I'll look that up."
ANSWER = 'The store opens at 9 am.'
FEEDBACK = 'Answer the question.'


async def announce_then_answer(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    """Announce a lookup on the first request, and answer once asked again."""
    yield ANSWER if len(messages) > 1 else ANNOUNCEMENT


async def always_announce(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield ANNOUNCEMENT


def reject_announcements(ctx: RunContext[Any], output: str) -> str:
    """An output validator that asks the model again when it announced a lookup instead of answering."""
    if output == ANNOUNCEMENT:
        raise ModelRetry(FEEDBACK)
    return output


@dataclass
class RejectAnnouncements(AbstractCapability[Any]):
    """Rejects the announcement from `after_model_request`."""

    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        if response.text == ANNOUNCEMENT:
            raise ModelRetry(FEEDBACK)
        return response
