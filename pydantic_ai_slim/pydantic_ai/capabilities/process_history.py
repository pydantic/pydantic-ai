from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai import messages as _messages
from pydantic_ai._history_processor import HistoryProcessor as HistoryProcessorFunc
from pydantic_ai._utils import await_maybe, is_async_callable, run_in_executor, takes_run_context
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.realtime import RealtimeConnectContext

    _MsgList = list[_messages.ModelMessage]
    _AsyncWithCtx = Callable[[RunContext[Any], _MsgList], Awaitable[_MsgList]]
    _AsyncNoCtx = Callable[[_MsgList], Awaitable[_MsgList]]
    _SyncWithCtx = Callable[[RunContext[Any], _MsgList], _MsgList]
    _SyncNoCtx = Callable[[_MsgList], _MsgList]


@dataclass
class ProcessHistory(AbstractCapability[AgentDepsT]):
    """A capability that processes message history before model requests.

    In a [realtime session](../realtime/capabilities.md#processing-seeded-history), which has no per-request
    boundary, the processor runs once over the history the session is seeded with, before it connects.
    """

    processor: HistoryProcessorFunc[AgentDepsT]

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        messages = await _run_history_processor(self.processor, ctx, list(request_context.messages))
        request_context.messages = messages.copy()
        ctx.messages[:] = messages

        return request_context

    async def before_realtime_connect(
        self,
        ctx: RunContext[AgentDepsT],
        connect_context: RealtimeConnectContext,
    ) -> RealtimeConnectContext:
        # A realtime session can't rewrite the provider's conversation before each response, so the processor runs
        # once, over the history the session is seeded with. With nothing to seed, there's nothing to process.
        if connect_context.messages:
            messages = await _run_history_processor(self.processor, ctx, list(connect_context.messages))
            connect_context.messages = messages.copy()
            ctx.messages[:] = messages
        return connect_context

    def _realtime_uncalled_hooks(self) -> tuple[str, ...]:
        # A realtime session runs the processor once over the history it's seeded with, in `before_realtime_connect`.
        return self._realtime_uncalled_hooks_overriding(ProcessHistory)

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
            return await cast('_AsyncWithCtx', processor)(ctx, messages)
        else:
            return await cast('_AsyncNoCtx', processor)(messages)
    else:
        # A plain `def` may still return an awaitable, which `run_in_executor` would leave un-awaited.
        if takes_ctx:
            return await await_maybe(await run_in_executor(cast('_SyncWithCtx', processor), ctx, messages))
        else:
            return await await_maybe(await run_in_executor(cast('_SyncNoCtx', processor), messages))
