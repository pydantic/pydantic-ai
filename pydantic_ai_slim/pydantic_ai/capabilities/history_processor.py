from dataclasses import dataclass
from typing import cast

from pydantic_ai import _function_schema, _history_processor, exceptions
from pydantic_ai._utils import now_utc, run_in_executor
from pydantic_ai.messages import ModelMessage, ModelRequest
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
        processor = self.processor
        takes_ctx = _function_schema._takes_ctx(processor)  # type: ignore[reportPrivateUsage]

        if _function_schema.is_async_callable(processor):
            if takes_ctx:
                messages = await processor(ctx, messages)
            else:
                messages = await processor(messages)
        else:
            if takes_ctx:
                sync_processor_with_ctx = cast(_history_processor._HistoryProcessorSyncWithCtx[AgentDepsT], processor)  # type: ignore[reportPrivateUsage]
                messages = await run_in_executor(sync_processor_with_ctx, ctx, messages)
            else:
                sync_processor = cast(_history_processor._HistoryProcessorSync, processor)  # type: ignore[reportPrivateUsage]
                messages = await run_in_executor(sync_processor, messages)

        if len(messages) == 0:
            raise exceptions.UserError('Processed history cannot be empty.')

        if not isinstance(messages[-1], ModelRequest):
            raise exceptions.UserError('Processed history must end with a `ModelRequest`.')

        # Ensure the last request has a timestamp (history processors may create new ModelRequest objects without one)
        if messages[-1].timestamp is None:
            messages[-1].timestamp = now_utc()

        return messages, model_settings, model_request_parameters

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None
