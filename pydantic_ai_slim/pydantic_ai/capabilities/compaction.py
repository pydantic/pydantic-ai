"""Ask the agent's compaction capabilities to compact the history before a model request."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CapabilityEvent

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext

__all__ = ('CompactionRequestedEvent', 'request_compaction')

# Set only while the event's listeners run: the request context can't be an event field, since every field is
# serialized with the event stream.
_request_context: ContextVar[ModelRequestContext | None] = ContextVar('_compaction_request_context', default=None)


@dataclass(kw_only=True)
class CompactionRequestedEvent(CapabilityEvent, namespace='compaction', name='requested', dispatch='immediate'):
    """A capability asks for the history to be compacted before the current model request.

    Emit it with [`request_compaction`][pydantic_ai.capabilities.request_compaction]. A compaction
    capability listens with [`@on_event(CompactionRequestedEvent)`][pydantic_ai.capabilities.on_event], replaces
    [`request_context.messages`][pydantic_ai.capabilities.CompactionRequestedEvent.request_context] with
    the compacted history, and sets `compacted_by`. Listeners run in capability order and a listener should leave an
    event that already has `compacted_by` set alone, so the first capability that compacts handles the request.
    """

    reason: str
    """Why compaction was requested, such as `'cache_expired'`."""

    compacted_by: str | None = None
    """Name of the capability that compacted the history, or `None` while none has."""

    @property
    def request_context(self) -> ModelRequestContext:
        """The model request being prepared. Only available to the event's listeners."""
        request_context = _request_context.get()
        if request_context is None:
            raise UserError(
                '`CompactionRequestedEvent.request_context` is only available to its listeners, '
                'and only when the event is emitted with `request_compaction()`.'
            )
        return request_context


async def request_compaction(
    ctx: RunContext[Any], request_context: ModelRequestContext, *, reason: str
) -> CompactionRequestedEvent:
    """Ask the agent's compaction capabilities to compact the history of `request_context`.

    Call it from a capability's `before_model_request` hook, where the request's model is the one the run will call
    (under durable execution, the recorded one). When a listener compacted,
    `request_context.messages` holds the compacted history and it is also written to `ctx.messages`, so the run
    keeps it. The returned event's `compacted_by` names that listener, or is `None` when no listener compacted.
    """
    token = _request_context.set(request_context)
    try:
        event = await ctx.emit(CompactionRequestedEvent(reason=reason))
    finally:
        _request_context.reset(token)
    if event.compacted_by is not None:
        ctx.messages[:] = request_context.messages
    return event
