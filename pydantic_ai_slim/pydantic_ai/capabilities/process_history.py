from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import deprecated

from pydantic_ai import messages as _messages
from pydantic_ai._history_processor import HistoryProcessor as HistoryProcessorFunc
from pydantic_ai._utils import is_async_callable, run_in_executor, takes_run_context
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic_ai.models import ModelRequestContext

    _MsgList = list[_messages.ModelMessage]
    _AsyncWithCtx = Callable[[RunContext[Any], _MsgList], Awaitable[_MsgList]]
    _AsyncNoCtx = Callable[[_MsgList], Awaitable[_MsgList]]
    _SyncWithCtx = Callable[[RunContext[Any], _MsgList], _MsgList]
    _SyncNoCtx = Callable[[_MsgList], _MsgList]


@dataclass
class ProcessHistory(AbstractCapability[AgentDepsT]):
    """A capability that processes message history before model requests."""

    processor: HistoryProcessorFunc[AgentDepsT]

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        request_context.messages = await _run_history_processor(self.processor, ctx, request_context.messages)

        return request_context

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)


@deprecated('`HistoryProcessor` is deprecated, use `ProcessHistory` instead.')
@dataclass
class HistoryProcessor(ProcessHistory[AgentDepsT]):
    """Deprecated alias for [`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory]."""


async def _run_history_processor(
    processor: HistoryProcessorFunc[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    messages: list[_messages.ModelMessage],
) -> list[_messages.ModelMessage]:
    """Run a history processor, handling sync/async and with/without context variants."""
    takes_ctx = takes_run_context(processor)

    if is_async_callable(processor):
        if takes_ctx:
            return await cast('_AsyncWithCtx', processor)(ctx, messages)
        else:
            return await cast('_AsyncNoCtx', processor)(messages)
    else:
        if takes_ctx:
            return await run_in_executor(cast('_SyncWithCtx', processor), ctx, messages)
        else:
            return await run_in_executor(cast('_SyncNoCtx', processor), messages)
