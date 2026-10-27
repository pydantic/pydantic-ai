from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai import messages as _messages
from pydantic_ai._history_processor import HistoryProcessor as HistoryProcessorFunc
from pydantic_ai._utils import await_maybe, is_async_callable, run_in_executor, takes_run_context
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability
from .combined import CombinedCapability, _ctx_for_active_cap  # pyright: ignore[reportPrivateUsage]
from .wrapper import WrapperCapability

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
        messages = await _run_history_processor(self.processor, ctx, list(request_context.messages))
        request_context.messages = messages.copy()
        ctx.messages[:] = messages

        return request_context

    def _realtime_uncalled_hooks(self) -> tuple[str, ...]:
        # A realtime session runs the processor once over the history it's seeded with; see `process_seeded_history`.
        return self._realtime_uncalled_hooks_overriding(ProcessHistory)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)


async def process_seeded_history(
    capability: AbstractCapability[AgentDepsT],
    ctx: RunContext[AgentDepsT],
    messages: list[_messages.ModelMessage],
) -> list[_messages.ModelMessage]:
    """Run the `ProcessHistory` processors in a capability tree over the history a realtime session is seeded with.

    A session can't rewrite the provider's conversation before each response the way a standard run rewrites its
    history before each model request, so it processes the history once, before seeding it. The tree is walked as
    `before_model_request` is dispatched: in application order, through wrappers, and past deferred capabilities
    that aren't loaded.
    """
    if isinstance(capability, ProcessHistory):
        messages = await _run_history_processor(capability.processor, ctx, messages)
        ctx.messages[:] = messages
    elif isinstance(capability, CombinedCapability):
        for child in capability.capabilities:
            if (child_ctx := _ctx_for_active_cap(child, ctx)) is not None:
                messages = await process_seeded_history(child, child_ctx, messages)
    elif isinstance(capability, WrapperCapability):
        messages = await process_seeded_history(capability.wrapped, ctx, messages)
    return messages


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
