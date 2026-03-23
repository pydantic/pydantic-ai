from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic_ai import messages as _messages
from pydantic_ai._history_processor import (
    HistoryProcessor as HistoryProcessorFunc,
    _HistoryProcessorAsync,
    _HistoryProcessorAsyncWithCtx,
    _HistoryProcessorSync,
    _HistoryProcessorSyncWithCtx,
)
from pydantic_ai._utils import is_async_callable, run_in_executor, takes_run_context
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass
class HistoryProcessor(AbstractCapability[AgentDepsT]):
    """A capability that processes message history before model requests."""

    processor: HistoryProcessorFunc[AgentDepsT]

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        request_context.messages = await _run_history_processor(
            self.processor, ctx, request_context.messages
        )

        return request_context

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)


async def _run_history_processor(
    processor: HistoryProcessorFunc[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    messages: list[_messages.ModelMessage],
) -> list[_messages.ModelMessage]:
    """Run a history processor, handling sync/async and with/without context variants."""
    takes_ctx = takes_run_context(processor)

    if is_async_callable(processor):
        if takes_ctx:
            async_with_ctx = cast(_HistoryProcessorAsyncWithCtx[AgentDepsT], processor)
            return await async_with_ctx(ctx, messages)
        else:
            async_processor = cast(_HistoryProcessorAsync, processor)
            return await async_processor(messages)
    else:
        if takes_ctx:
            sync_with_ctx = cast(_HistoryProcessorSyncWithCtx[AgentDepsT], processor)
            return await run_in_executor(sync_with_ctx, ctx, messages)
        else:
            sync_processor = cast(_HistoryProcessorSync, processor)
            return await run_in_executor(sync_processor, messages)
