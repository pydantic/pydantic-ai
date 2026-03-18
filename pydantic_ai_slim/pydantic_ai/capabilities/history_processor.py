from dataclasses import dataclass

from pydantic_ai import _history_processor
from pydantic_ai._utils import now_utc
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability


@dataclass
class HistoryProcessorCapability(AbstractCapability[AgentDepsT]):
    """A capability that processes message history before model requests."""

    processor: _history_processor.HistoryProcessor[AgentDepsT]

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        messages: list[ModelMessage],
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[ModelMessage], ModelSettings, ModelRequestParameters]:
        messages = await _history_processor.run_history_processor(self.processor, ctx, messages)

        # Ensure the last request has a timestamp (history processors may create new ModelRequest objects without one)
        if messages and messages[-1].timestamp is None:
            messages[-1].timestamp = now_utc()

        return messages, model_settings, model_request_parameters

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None
