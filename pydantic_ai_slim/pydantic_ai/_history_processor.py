from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar, cast, get_origin

from pydantic_ai import messages as _messages
from pydantic_ai._utils import get_first_param_type, is_async_callable, run_in_executor
from pydantic_ai.tools import RunContext

DepsT = TypeVar('DepsT')

_HistoryProcessorSync = Callable[[list[_messages.ModelMessage]], list[_messages.ModelMessage]]
_HistoryProcessorAsync = Callable[[list[_messages.ModelMessage]], Awaitable[list[_messages.ModelMessage]]]
_HistoryProcessorSyncWithCtx = Callable[[RunContext[DepsT], list[_messages.ModelMessage]], list[_messages.ModelMessage]]
_HistoryProcessorAsyncWithCtx = Callable[
    [RunContext[DepsT], list[_messages.ModelMessage]], Awaitable[list[_messages.ModelMessage]]
]
HistoryProcessor = (
    _HistoryProcessorSync
    | _HistoryProcessorAsync
    | _HistoryProcessorSyncWithCtx[DepsT]
    | _HistoryProcessorAsyncWithCtx[DepsT]
)
"""A function that processes a list of model messages and returns a list of model messages.

Can optionally accept a `RunContext` as a parameter.
"""


def _takes_ctx(processor: HistoryProcessor[DepsT]) -> bool:
    """Check if a history processor takes a RunContext as its first argument."""
    first_param_type = get_first_param_type(processor)
    if first_param_type is None:
        return False
    return first_param_type is RunContext or get_origin(first_param_type) is RunContext


async def run_history_processor(
    processor: HistoryProcessor[DepsT],
    ctx: RunContext[DepsT],
    messages: list[_messages.ModelMessage],
) -> list[_messages.ModelMessage]:
    """Run a history processor, handling sync/async and with/without context variants."""
    takes_ctx = _takes_ctx(processor)

    if is_async_callable(processor):
        if takes_ctx:
            async_with_ctx = cast(_HistoryProcessorAsyncWithCtx[DepsT], processor)
            return await async_with_ctx(ctx, messages)
        else:
            async_processor = cast(_HistoryProcessorAsync, processor)
            return await async_processor(messages)
    else:
        if takes_ctx:
            sync_with_ctx = cast(_HistoryProcessorSyncWithCtx[DepsT], processor)
            return await run_in_executor(sync_with_ctx, ctx, messages)
        else:
            sync_processor = cast(_HistoryProcessorSync, processor)
            return await run_in_executor(sync_processor, messages)
