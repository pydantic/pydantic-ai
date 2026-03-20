from dataclasses import dataclass

from pydantic_ai import _history_processor
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability, BeforeModelRequestContext


@dataclass
class HistoryProcessorCapability(AbstractCapability[AgentDepsT]):
    """A capability that processes message history before model requests."""

    processor: _history_processor.HistoryProcessor[AgentDepsT]

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: BeforeModelRequestContext,
    ) -> BeforeModelRequestContext:
        request_context.messages = await _history_processor.run_history_processor(
            self.processor, ctx, request_context.messages
        )

        return request_context

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None
