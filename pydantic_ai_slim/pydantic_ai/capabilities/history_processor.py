from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import _history_processor
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass
class HistoryProcessor(AbstractCapability[AgentDepsT]):
    """A capability that processes message history before model requests."""

    processor: _history_processor.HistoryProcessor[AgentDepsT]

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        request_context.messages = await _history_processor.run_history_processor(
            self.processor, ctx, request_context.messages
        )

        return request_context

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)
