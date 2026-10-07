"""Session ownership shared by the ordinary and live execution drivers.

The binding is consumed before user code runs, like the event-stream run binding. It carries an
explicit session through existing agent wrappers without making nested runs inherit that session.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Generator
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import TYPE_CHECKING

import anyio
from anyio.abc import TaskGroup, TaskStatus

from ._cancel import RunCancellation
from ._enqueue import PendingMessage, PendingMessageQueue
from ._run_context import get_current_run_context
from .conversation import Conversation
from .exceptions import UserError
from .models import Model

if TYPE_CHECKING:
    from ._agent_graph import GraphAgentState


@dataclasses.dataclass
class ModelResources:
    """Models entered on one execution owner's stack, deduplicated by identity."""

    entered_model_ids: set[int] = dataclasses.field(default_factory=set[int])
    _stack: AsyncExitStack | None = dataclasses.field(default=None, init=False, repr=False)
    _group: TaskGroup | None = dataclasses.field(default=None, init=False, repr=False)
    _closed: anyio.Event | None = dataclasses.field(default=None, init=False, repr=False)

    def bind_stack(self, stack: AsyncExitStack) -> None:
        assert self._stack is None
        self._stack = stack

    def bind_group(self, group: TaskGroup) -> None:
        self._group = group
        self._closed = anyio.Event()

    def close(self) -> None:
        assert self._closed is not None
        self._closed.set()

    async def enter_model(self, selected_model: Model) -> None:
        if id(selected_model) in self.entered_model_ids:
            return
        if self._group is not None:
            # A custom model may own task groups or cancel scopes. Its entry and exit must stay
            # in one persistent task, even when different tasks drive successive session runs.
            await self._group.start(self._hold_model, selected_model)
        else:
            assert self._stack is not None
            await self._stack.enter_async_context(selected_model)
        self.entered_model_ids.add(id(selected_model))

    async def _hold_model(self, model: Model, *, task_status: TaskStatus[None]) -> None:
        assert self._closed is not None
        with anyio.CancelScope() as cleanup_scope:
            async with model:
                # Acquisition remains cancellable, but an entered model must outlive run cleanup.
                # Session.__aexit__ signals closure only after its active run has unwound.
                cleanup_scope.shield = True
                task_status.started()
                await self._closed.wait()


class SessionRuntime:
    """Own the conversation and grant one run an exclusive, revocable input queue lease.

    A closed run queue is never reopened: retained RunContexts must not send into the next run.
    Session submissions racing the final drain go into the idle inbox instead.
    """

    def __init__(
        self, conversation: Conversation, pending: list[PendingMessage] | None = None, *, persistent: bool = True
    ) -> None:
        self.persistent = persistent
        self.conversation = deepcopy(conversation) if persistent else conversation
        self.resources = ModelResources()
        self.inferred_models: dict[str, Model] = {}
        self._inbox = PendingMessageQueue(deepcopy(pending) if pending else ())
        self._active: GraphAgentState | None = None
        self._claimed = False
        self._cancellation: RunCancellation | None = None
        self._cancel_requested = False
        self._closed = False
        self._lock = threading.Lock()

    def claim(self) -> None:
        with self._lock:
            if self._closed:
                raise UserError('The agent session has closed.')
            if self._claimed:
                raise UserError('An agent session can execute only one run at a time.')
            self._claimed = True

    def bind_cancellation(self, cancellation: RunCancellation) -> None:
        with self._lock:
            self._cancellation = cancellation
            requested = self._cancel_requested
        if requested:
            cancellation.cancel()

    def cancel(self) -> None:
        ctx = get_current_run_context()
        with self._lock:
            if not self._claimed:
                return
            self._cancel_requested = True
            cancellation = self._cancellation
            run_id = self._active.run_id if self._active is not None else None
        # Inside a durable unit the context owns a guard, not the live controller. A closure over
        # a session must not bypass that unit's replay-safety restriction.
        if ctx is not None and ctx.run_id == run_id:
            ctx.cancel()
        elif cancellation is not None:
            cancellation.cancel()

    def attach(self, state: GraphAgentState) -> None:
        with self._lock:
            assert self._claimed and self._active is None
            state.pending_messages = self._inbox
            self._inbox = PendingMessageQueue()
            self._active = state

    def require_attached(self) -> None:
        if self._active is None:
            raise UserError('The agent wrapper did not delegate to its wrapped agent with the session conversation.')

    def enqueue(self, pending: PendingMessage) -> None:
        with self._lock:
            if self._closed:
                raise UserError('`enqueue` is not available because the agent session has closed.')
            if self._active is not None:
                queue = self._active.pending_messages
                assert isinstance(queue, PendingMessageQueue)
                ctx = get_current_run_context()
                if (
                    ctx is not None
                    and ctx.run_id == self._active.run_id
                    and ctx.pending_messages is not None
                    and ctx.pending_messages is not queue
                ):
                    ctx.pending_messages.append(pending)
                    return
                if queue.try_append(pending):
                    return
            self._inbox.append(pending)

    def snapshot(self) -> tuple[Conversation, list[PendingMessage], str | None]:
        with self._lock:
            if self._claimed and self._active is None:
                raise UserError('The session is preparing a run; its checkpoint is not available yet.')
            conversation = self.conversation
            pending: list[PendingMessage] = []
            run_id = None
            if self._active is not None:
                state = self._active
                conversation = Conversation(
                    messages=state.message_history,
                    usage=state.usage,
                    conversation_id=state.conversation_id,
                )
                queue = state.pending_messages
                assert isinstance(queue, PendingMessageQueue)
                pending.extend(queue.snapshot())
                run_id = state.run_id
            pending.extend(self._inbox.snapshot())
            return deepcopy(conversation), deepcopy(pending), run_id

    def release(self, conversation: Conversation | None = None) -> None:
        with self._lock:
            if self._active is not None:
                state = self._active
                self.conversation = conversation or Conversation(
                    messages=state.message_history,
                    usage=state.usage,
                    conversation_id=state.conversation_id,
                    deferred_tool_requests=(
                        self.conversation.deferred_tool_requests
                        if state.message_history == self.conversation.messages
                        else None
                    ),
                )
                queue = state.pending_messages
                assert isinstance(queue, PendingMessageQueue)
                if self.persistent:
                    # Close and transfer atomically; a worker thread can still hold this run's context.
                    self._inbox = PendingMessageQueue([*queue.close_and_take(), *self._inbox.snapshot()])
                else:
                    # Legacy run handles keep their undelivered messages visible after termination.
                    queue.close()
                self._active = None
            self._claimed = False
            self._cancellation = None
            self._cancel_requested = False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._inbox.close()


_SESSION_BINDING: ContextVar[tuple[object, SessionRuntime] | None] = ContextVar(
    'pydantic_ai_session_binding', default=None
)


@contextmanager
def bind_session(agent: object, session: SessionRuntime) -> Generator[None]:
    token = _SESSION_BINDING.set((agent, session))
    try:
        yield
    finally:
        _SESSION_BINDING.reset(token)


def take_session(agent: object, conversation: Conversation | None) -> SessionRuntime | None:
    binding = _SESSION_BINDING.get()
    if binding is None or binding[0] is not agent or binding[1].conversation is not conversation:
        return None
    _SESSION_BINDING.set(None)
    return binding[1]
