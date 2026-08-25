"""Internal helpers for the `RunContext.enqueue` / `AgentRun.enqueue` APIs.

These types live here (rather than in `messages.py`) because they're internal runtime
state for the pending message queue, not part of the wire-serializable message history.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator, Sequence
from concurrent.futures import CancelledError as FutureCancelledError, Future, InvalidStateError
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Event, Lock
from typing import TYPE_CHECKING, Literal, TypeAlias

from ._uuid import uuid7
from .exceptions import UserError
from .messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    ToolAvailabilityDeltaPart,
    ToolReturnPart,
    ToolSearchReturnPart,
    UserPromptPart,
)

if TYPE_CHECKING:
    from .messages import UserContent


PendingMessagePriority: TypeAlias = Literal['asap', 'when_idle']
"""When to deliver a pending message.

- `'asap'`: Delivered at the earliest opportunity — either prepended to the next
    [`ModelRequest`][pydantic_ai.messages.ModelRequest], or, if the agent would
    otherwise terminate before another request, used to redirect the run into one
    more request.
- `'when_idle'`: Delivered only when the agent would otherwise terminate, after
    any `'asap'` messages. Doesn't interrupt in-flight work.
"""


EnqueueContent: TypeAlias = 'UserContent | ModelRequestPart | ModelMessage'
"""A single item accepted by [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue]
and [`AgentRun.enqueue`][pydantic_ai.run.AgentRun.enqueue].

`enqueue` is variadic, so each item is one positional argument:

- [`UserContent`][pydantic_ai.messages.UserContent] (a `str` or a piece of multi-modal content
    like an [`ImageUrl`][pydantic_ai.messages.ImageUrl]): adjacent user content is gathered into a
    single [`UserPromptPart`][pydantic_ai.messages.UserPromptPart], so `enqueue('caption', image)`
    forms one user turn. To pass an existing list, spread it: `enqueue(*items)`.
- [`ModelRequestPart`][pydantic_ai.messages.ModelRequestPart] (e.g. a
    [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]): included verbatim.
- [`ModelMessage`][pydantic_ai.messages.ModelMessage] (a complete
    [`ModelRequest`][pydantic_ai.messages.ModelRequest] or
    [`ModelResponse`][pydantic_ai.messages.ModelResponse]): emitted as its own message.

Consecutive part-style items (user content and `ModelRequestPart`s) are coalesced into a single
`ModelRequest`; complete `ModelMessage`s stay separate. This lets one `enqueue` call inject an
interleaved exchange (e.g. a synthetic tool call + result — a `ModelResponse` followed by a
`ModelRequest`). The assembled sequence must end in a `ModelRequest` so the agent has something to
respond to.
"""

_RUN_ENDED_MESSAGE = '`enqueue` is not available because the agent run is no longer accepting messages.'


def _build_enqueue_messages(items: Sequence[EnqueueContent]) -> list[ModelMessage]:
    """Assemble enqueue items into a list of [`ModelMessage`][pydantic_ai.messages.ModelMessage]s.

    Adjacent [`UserContent`][pydantic_ai.messages.UserContent] items are gathered into one
    [`UserPromptPart`][pydantic_ai.messages.UserPromptPart], and part-style items (user content and
    [`ModelRequestPart`][pydantic_ai.messages.ModelRequestPart]s) are coalesced into a single
    [`ModelRequest`][pydantic_ai.messages.ModelRequest]; complete `ModelMessage`s are emitted as-is.
    Order is preserved, so a `ModelResponse` followed by part-style items produces the response then
    a request built from those parts.
    """
    messages: list[ModelMessage] = []
    parts: list[ModelRequestPart] = []
    content: list[UserContent] = []

    def flush_content() -> None:
        if content:
            # Collapse a lone string to `str` content, matching `Agent.run('...')`; anything else
            # (multiple items, or a single non-string like an image) becomes a content list.
            single = content[0] if len(content) == 1 and isinstance(content[0], str) else list(content)
            parts.append(UserPromptPart(content=single))
            content.clear()

    def flush_request() -> None:
        flush_content()
        if parts:
            messages.append(ModelRequest(parts=list(parts)))
            parts.clear()

    for item in items:
        if isinstance(item, (ModelRequest, ModelResponse)):
            flush_request()
            messages.append(item)
        elif isinstance(
            item,
            (
                SystemPromptPart,
                UserPromptPart,
                ToolReturnPart,
                RetryPromptPart,
                ToolSearchReturnPart,
                ToolAvailabilityDeltaPart,
                SpeechPart,
            ),
        ):
            flush_content()
            parts.append(item)
        else:
            content.append(item)
    flush_request()
    return messages


@dataclass
class PendingMessage:
    """One or more [`ModelMessage`][pydantic_ai.messages.ModelMessage]s queued for injection into the agent conversation.

    Enqueued via [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] or
    [`AgentRun.enqueue`][pydantic_ai.run.AgentRun.enqueue] and automatically drained
    at the appropriate time during the agent run by the internal `PendingMessageDrainCapability`.
    """

    messages: list[ModelMessage]
    """The message(s) to inject, in order. Always ends in a
    [`ModelRequest`][pydantic_ai.messages.ModelRequest]."""

    priority: PendingMessagePriority = 'asap'
    """When to deliver these messages:

    - `'asap'`: at the earliest opportunity (next model request, or redirect if the agent
        would otherwise terminate).
    - `'when_idle'`: only when the agent would otherwise terminate, after `'asap'` messages.
    """

    enqueue_id: str = field(default_factory=lambda: str(uuid7()))
    """Unique identifier for this enqueue call, surfaced on the
    [`EnqueuedMessagesEvent`][pydantic_ai.messages.EnqueuedMessagesEvent] emitted when the messages
    are delivered, and returned by [`enqueue`][pydantic_ai.tools.RunContext.enqueue]."""

    @classmethod
    def from_content(cls, *content: EnqueueContent, priority: PendingMessagePriority = 'asap') -> PendingMessage | None:
        """Build a `PendingMessage` from `enqueue` arguments, or `None` when there's nothing to send.

        Returns `None` for an empty call (enqueueing nothing is a no-op rather than an error).

        Raises:
            UserError: If the assembled messages don't end in a
                [`ModelRequest`][pydantic_ai.messages.ModelRequest] — e.g. a lone `ModelResponse` —
                since the agent needs a request to respond to.
        """
        messages = _build_enqueue_messages(content)
        if not messages:
            return None
        if not isinstance(messages[-1], ModelRequest):
            raise UserError(
                'Enqueued content must end with a `ModelRequest` (or user content / `ModelRequestPart` '
                'items that form one), so the agent has a request to respond to.'
            )
        return cls(messages=messages, priority=priority)


class _PendingMessageBridge:
    def __init__(self, queue: list[PendingMessage]) -> None:
        self.queue = queue
        self._loop = asyncio.get_running_loop()
        self._closed = Event()
        self._waiting: set[Future[None]] = set()
        self._lock = Lock()

    def append(self, pending: PendingMessage) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            if running_loop is not self._loop:
                raise UserError('`enqueue` cannot be called from a different event loop than the agent run.')
            self._append(pending)
            return

        result: Future[None] = Future()
        with self._lock:
            if self._closed.is_set():
                raise UserError(_RUN_ENDED_MESSAGE)
            self._waiting.add(result)

        def append() -> None:
            try:
                self._append(pending)
            except BaseException as e:
                with suppress(InvalidStateError):
                    result.set_exception(e)
            else:
                with suppress(InvalidStateError):
                    result.set_result(None)

        try:
            self._loop.call_soon_threadsafe(append)
            result.result()
        except FutureCancelledError as e:
            raise UserError(_RUN_ENDED_MESSAGE) from e
        finally:
            with self._lock:
                self._waiting.discard(result)

    def _append(self, pending: PendingMessage) -> None:
        if self._closed.is_set():
            raise UserError(_RUN_ENDED_MESSAGE)
        self.queue.append(pending)

    def close(self) -> None:
        with self._lock:
            self._closed.set()
            waiting = tuple(self._waiting)
        for result in waiting:
            result.cancel()


_pending_message_bridge: ContextVar[_PendingMessageBridge | None] = ContextVar('_pending_message_bridge', default=None)


@contextmanager
def bind_pending_message_queue(queue: list[PendingMessage]) -> Generator[None]:
    """Bind `queue` to its run loop for the lifetime of an agent run."""
    bridge = _PendingMessageBridge(queue)
    token = _pending_message_bridge.set(bridge)
    try:
        yield
    finally:
        bridge.close()
        _pending_message_bridge.reset(token)


def append_pending_message(queue: list[PendingMessage], pending: PendingMessage) -> None:
    """Append on the run loop when called from a sync callback owned by that run."""
    bridge = _pending_message_bridge.get()
    if bridge is None or bridge.queue is not queue:
        queue.append(pending)
    else:
        bridge.append(pending)


def close_pending_message_queue(queue: list[PendingMessage]) -> None:
    """Stop sync callbacks from appending after the run's final drain."""
    bridge = _pending_message_bridge.get()
    if bridge is not None and bridge.queue is queue:
        bridge.close()
