"""Session ownership shared by the ordinary and live execution drivers.

The binding is consumed before user code runs, like the event-stream run binding. It carries an
explicit session through existing agent wrappers without making nested runs inherit that session.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Generator
from contextlib import AsyncExitStack, contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import TYPE_CHECKING

import anyio
from anyio.abc import TaskGroup, TaskStatus

from . import _utils
from ._cancel import RunCancellation
from ._enqueue import PendingMessage, PendingMessageQueue
from ._operations import ToolOperation
from ._run_context import get_current_run_context
from .conversation import Conversation
from .exceptions import UserError
from .models import Model
from .usage import RequestUsage, RunUsage

if TYPE_CHECKING:
    from .realtime._persistent import RealtimeAttachment


@dataclasses.dataclass(frozen=True, kw_only=True)
class ActiveRun:
    """The session's driver-independent lease: identity, inbox, and current conversation.

    The driver keeps its own graph or duplex state. Its queue is never replaced, so live
    notification and validation remain attached to the same object held by RunContexts.
    """

    run_id: str
    pending_messages: PendingMessageQueue
    snapshot: Callable[[], Conversation]


@dataclasses.dataclass
class ModelResources:
    """Models entered on one execution owner's stack, deduplicated by identity."""

    entered_model_ids: set[int] = dataclasses.field(default_factory=set[int])
    # Keep definitions strongly referenced as well as handles: arbitrary custom models can be
    # unhashable, and an id must not be reused after a dynamic selector discards a definition.
    _models: dict[int, tuple[Model, Model]] = dataclasses.field(
        default_factory=dict[int, tuple[Model, Model]], init=False, repr=False
    )
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

    async def get_model(self, selected_model: Model, *, enter_model: bool = True) -> Model:
        if selected_model._model_resources_in_durable_units:  # pyright: ignore[reportPrivateUsage]
            return selected_model
        existing = self._models.get(id(selected_model))
        if existing is not None:
            original, bound = existing
            if enter_model and id(original) not in self.entered_model_ids:
                assert self._stack is not None
                await self._stack.enter_async_context(original)
                self.entered_model_ids.add(id(original))
            return bound
        if self._group is not None:
            # A custom model may own task groups or cancel scopes. Its entry and exit must stay
            # in one persistent task, even when different tasks drive successive session runs.
            bound = await self._group.start(self._hold_model, selected_model)
        else:
            assert self._stack is not None
            if enter_model and id(selected_model) not in self.entered_model_ids:
                await self._stack.enter_async_context(selected_model)
                self.entered_model_ids.add(id(selected_model))
            bound = await self._stack.enter_async_context(selected_model.open_session())
        self._models[id(selected_model)] = (selected_model, bound)
        self._models[id(bound)] = (selected_model, bound)
        return bound

    async def _hold_model(self, model: Model, *, task_status: TaskStatus[Model]) -> None:
        assert self._closed is not None
        with anyio.CancelScope() as cleanup_scope:
            async with model, model.open_session() as bound:
                # Acquisition remains cancellable, but an entered model must outlive run cleanup.
                # Session.__aexit__ signals closure only after its active run has unwound.
                cleanup_scope.shield = True
                self.entered_model_ids.add(id(model))
                task_status.started(bound)
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
        self.realtime: RealtimeAttachment | None = None
        self.inferred_models: dict[str, Model] = {}
        self.operations: dict[str, ToolOperation] = {}
        self._inbox = PendingMessageQueue(deepcopy(pending) if pending else ())
        self._active: ActiveRun | None = None
        self._result_conversation: Conversation | None = None
        self._connection_usage = RunUsage()
        self._claimed = False
        self._cancellation: RunCancellation | None = None
        self._cancel_requested = False
        self._closed = False
        self._lock = threading.Lock()

    def claim(self, *, realtime: bool = False) -> None:
        with self._lock:
            if self._closed:
                raise UserError('The agent session has closed.')
            if self._claimed:
                raise UserError('An agent session can execute only one run at a time.')
            if self.realtime is not None and not realtime:
                raise UserError('Close the realtime connection before starting an ordinary run on this session.')
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

    def attach(self, run: ActiveRun, *, transfer_pending: bool = True) -> None:
        with self._lock:
            assert self._claimed and self._active is None
            if transfer_pending:
                # Validate the whole transfer before consuming the idle inbox. An incompatible
                # input (e.g. non-text realtime enqueue) must remain available to another driver.
                for pending in self._inbox.snapshot():
                    run.pending_messages.append(pending)
                self._inbox.close_and_take()
                self._inbox = PendingMessageQueue()
            self._active = run

    def require_attached(self) -> None:
        if self._active is None:
            raise UserError('The agent wrapper did not delegate to its wrapped agent with the session conversation.')

    def enqueue(self, pending: PendingMessage) -> None:
        with self._lock:
            if self._closed:
                raise UserError('`enqueue` is not available because the agent session has closed.')
            if self._active is not None:
                queue = self._active.pending_messages
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
                conversation = state.snapshot()
                queue = state.pending_messages
                pending.extend(queue.snapshot())
                run_id = state.run_id
            pending.extend(self._inbox.snapshot())
            conversation = deepcopy(conversation)
            conversation.usage.incr(self._connection_usage)
            return conversation, deepcopy(pending), run_id

    def record_connection_usage(self, usage: RequestUsage) -> None:
        """Account for billing outside a run without changing its frozen result.

        Run hooks may still be unwinding. Hold this delta until their result is committed so
        replacing the conversation cannot erase it or charge it to the finished run. During
        preparation there is no attached run yet: update the conversation usage shared by the
        new run context so admission checks include billing received during its hooks.
        """
        with self._lock:
            if self._active is not None:
                self._connection_usage.incr(usage)
            else:
                self.conversation.usage.incr(usage)

    def record_result(self, conversation: Conversation) -> None:
        with self._lock:
            self._result_conversation = conversation

    def release(self, conversation: Conversation | None = None) -> None:
        with self._lock:
            conversation = conversation or self._result_conversation
            self._result_conversation = None
            if self._active is not None:
                state = self._active
                self.conversation = conversation or state.snapshot()
                queue = state.pending_messages
                if self.persistent:
                    # Completed run handles remain user-owned. Detach both history and effect facts
                    # together, preserving their internal associations without exposing next-run state.
                    self.conversation, self.operations = deepcopy((self.conversation, self.operations))
                    # Close and transfer atomically; a worker thread can still hold this run's context.
                    self._inbox = PendingMessageQueue([*queue.close_and_take(), *self._inbox.snapshot()])
                else:
                    # Legacy run handles keep their undelivered messages visible after termination.
                    queue.close()
                self._active = None
            self.conversation.usage.incr(self._connection_usage)
            self._connection_usage = RunUsage()
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


def take_session(
    agent: object, conversation: Conversation | None | _utils.Unset = _utils.UNSET
) -> SessionRuntime | None:
    binding = _SESSION_BINDING.get()
    if binding is None or binding[0] is not agent:
        return None
    if _utils.is_set(conversation) and binding[1].conversation is not conversation:
        return None
    _SESSION_BINDING.set(None)
    return binding[1]
