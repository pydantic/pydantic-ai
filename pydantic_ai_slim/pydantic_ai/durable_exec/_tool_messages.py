from __future__ import annotations

from collections.abc import Awaitable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import KW_ONLY, dataclass
from typing import Annotated, Literal

from pydantic import Discriminator

from pydantic_ai._enqueue import PendingMessage
from pydantic_ai._run_context import RunContext, set_current_run_context

from ._toolset import CallToolResult, EnqueueGuard, wrap_tool_call_result


@dataclass
class ToolCallWithMessages:
    result: CallToolResult
    messages: list[PendingMessage]
    _: KW_ONLY
    kind: Literal['tool_call_with_messages'] = 'tool_call_with_messages'


RecordedToolCallResult = Annotated[CallToolResult | ToolCallWithMessages, Discriminator('kind')]

_replayed_message_ids: ContextVar[set[str]] = ContextVar('durable_replayed_message_ids')


@contextmanager
def tool_message_replay_scope() -> Generator[None]:
    # The queue can be drained between two loads of the same result. Keep identities for the
    # whole reconstructed run, while allowing another run to consume the same cached messages.
    token = _replayed_message_ids.set(set())
    try:
        yield
    finally:
        _replayed_message_ids.reset(token)


async def record_tool_call_result(ctx: RunContext[object], coro: Awaitable[object]) -> RecordedToolCallResult:
    # Some engines also use these handlers for inline calls outside a workflow. Those calls
    # already have a live queue, whose delivery timing must remain unchanged.
    if not isinstance(ctx.pending_messages, EnqueueGuard):
        return await wrap_tool_call_result(coro)

    messages: list[PendingMessage] = []
    original_pending = ctx.pending_messages
    ctx.pending_messages = messages
    try:
        with set_current_run_context(ctx):
            result = await wrap_tool_call_result(coro)
    finally:
        ctx.pending_messages = original_pending

    # Ordinary results retain their wire shape for rolling upgrades. Tools that enqueue were
    # previously rejected, so only those newly supported calls need the envelope.
    if messages:
        return ToolCallWithMessages(result=result, messages=messages)
    return result


def replay_tool_messages(result: RecordedToolCallResult, ctx: RunContext[object]) -> CallToolResult:
    if not isinstance(result, ToolCallWithMessages):
        return result

    assert ctx.pending_messages is not None
    replayed = _replayed_message_ids.get()
    for pending in result.messages:
        if pending.enqueue_id not in replayed:
            # History processing may stamp or transform messages after draining the queue.
            # A cached record must remain reusable by subsequent runs.
            ctx.pending_messages.append(deepcopy(pending))
            replayed.add(pending.enqueue_id)
    return result.result
