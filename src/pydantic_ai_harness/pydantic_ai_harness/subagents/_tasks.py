"""Session-owned delegation lifetime, independent of terminal rendering."""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import time
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeVar
from uuid import uuid4

import anyio
from pydantic import SerializerFunctionWrapHandler, TypeAdapter, WrapSerializer

from pydantic_ai import AgentRunResult, AgentStreamEvent, CapabilityEvent, RunContext, capture_run_messages
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler, on_event
from pydantic_ai.exceptions import ModelRetry, RunCancelled, UserError
from pydantic_ai.messages import (
    EnqueuedMessagesEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    UserPromptPart,
)
from pydantic_ai.output import OutputContext
from pydantic_ai_harness._task_report import make_task_output, make_task_question
from pydantic_ai_harness.step_persistence import StepPersistence, StepStore
from pydantic_ai_harness.subagents._events import DelegationOutcome

if TYPE_CHECKING:
    from pydantic_ai_harness.subagents._toolset import SubAgent

_CURRENT: ContextVar[DelegationTasks | None] = ContextVar('delegation_tasks', default=None)
_CHILD: ContextVar[str | None] = ContextVar('delegation_child', default=None)

NOT_BACKGROUNDABLE = 'This workspace is owned by the parent run and cannot run background children'
"""Why a delegation can't run in the background."""

RESUME_DIRECTORY_MISMATCH = 'A resumed task cannot change its working directory.'
"""Why a resume cannot select a different workspace."""


def _dump_messages(messages: list[ModelMessage], handler: SerializerFunctionWrapHandler) -> object:
    try:
        return handler(messages)
    except UnicodeDecodeError:
        # Non-UTF-8 `bytes` in an `Any` field: only an outermost adapter's base64 config reaches them.
        return ModelMessagesTypeAdapter.dump_python(messages, mode='json')


@dataclass(kw_only=True)
class DelegationTask:
    """A stable child conversation, retained across explicit resumes."""

    id: str
    agent_name: str
    prompt: str
    conversation_id: str
    parent_id: str | None = None
    model: str | None = None
    working_directory: str | None = None
    """Resolved directory selected for this task, or `None` to inherit the parent workspace."""
    background: bool = False
    backgroundable: bool = True
    resumable: bool = True
    user_stopped: bool = False
    status: Literal['running', 'finished'] = 'running'
    outcome: DelegationOutcome | None = None
    output: str = ''
    pending_question: str | None = None
    """The question a running task is waiting on an answer to, saved so a process exit doesn't lose it."""
    messages: Annotated[list[ModelMessage], WrapSerializer(_dump_messages, when_used='json')] = field(
        default_factory=list[ModelMessage]
    )
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    delivered: bool = False
    generation: int = 1
    run_id: str = field(default_factory=lambda: uuid4().hex)

    def report(self) -> UserPromptPart:
        """The settled task's report: the child's output, framed as `<task_output>` data on the user channel.

        The output was written by a model and may carry instructions from whatever the child read, so it
        never enters system voice. The frame's escaped attributes identify the task (its id, agent name,
        and an `outcome` other than `ok`). The acceptance receipt the parent got when the task started
        already says a report will follow, so nothing precedes the frame. The `TextContent` metadata
        identifies the report without matching its text.
        """
        return make_task_output(
            self.output,
            task_id=self.id,
            agent_name=self.agent_name,
            outcome=self.outcome or 'ok',
            generation=self.generation,
        )


@dataclass(kw_only=True)
class DelegationTaskEvent:
    """A task snapshot changed. Child stream events are correlated without entering the parent stream."""

    task: DelegationTask
    event: AgentStreamEvent | None = None
    model_name: str | None = None
    """The model the child run is using, for events from the child's own stream; `None` otherwise."""
    context_window: int | None = None
    """That model's context window, when its profile knows one, so a UI can show the child's context usage."""


Observer = Callable[[DelegationTaskEvent], Awaitable[None]]
Runner = Callable[[DelegationTask], Awaitable[str]]
Emitter = Callable[[CapabilityEvent], Awaitable[object]]
"""Emits a capability event into a running delegating run, like `RunContext.emit`."""
CapabilityEventT = TypeVar('CapabilityEventT', bound=CapabilityEvent)

STOPPED_OUTPUT = 'Task stopped; no successful result is available.'
"""The output of a task that was stopped before it produced one."""

UNSTARTED_OUTPUT = 'Task stopped before execution.'
"""The output of a task that was stopped before its worker was scheduled."""

CANCELLED_ANSWER = 'No answer: this task is being cancelled. Stop working on it.'
"""What `ask_parent` returns to a task that was asked to stop while it waits for an answer."""


@dataclass(kw_only=True)
class _Question:
    question_id: str
    text: str
    answer: asyncio.Future[str]
    queued: bool = False


def _question_part(record: DelegationTask, question: _Question) -> UserPromptPart:
    return make_task_question(
        question.text, task_id=record.id, agent_name=record.agent_name, question_id=question.question_id
    )


_KeyT = TypeVar('_KeyT')
_ItemT = TypeVar('_ItemT')


def _unregister(stacks: dict[_KeyT, list[_ItemT]], key: _KeyT, item: _ItemT) -> None:
    """Remove `item`, by identity, from the stack under `key`, and the stack once it is empty."""
    stack = stacks[key]
    del stack[max(index for index, registered in enumerate(stack) if registered is item)]
    if not stack:
        del stacks[key]


_ADAPTER = TypeAdapter(DelegationTask)
_SAFE_ID = re.compile(r'[0-9a-f]{32}\Z')


async def _read(path: anyio.Path) -> DelegationTask:
    """A saved task, checked to have the identity its file name gives it."""
    record = _ADAPTER.validate_json(await path.read_bytes())
    if not _SAFE_ID.fullmatch(record.id) or not _SAFE_ID.fullmatch(record.run_id) or path.name != f'{record.id}.json':
        raise ValueError(f'Invalid saved task identity in {path.name!r}')
    return record


class DelegationTasks:
    """Own detached children until `opened` exits; bind to runs with `bind`.

    Outside this opt-in scope, `SubAgents` retains its synchronous delegation contract.
    Use one owner per application session. The caller owns workspace and plugin resources
    until this owner's exit has cancelled and drained all children.
    """

    def __init__(
        self,
        *,
        observer: Observer | None = None,
        directory: Path | None = None,
        agents: Mapping[str, SubAgent[object]] | None = None,
        aliases: Mapping[str, str] | None = None,
        one_shot: frozenset[str] = frozenset(),
        max_depth: int = 4,
        instructions: str = '',
        step_store: StepStore | None = None,
    ) -> None:
        if max_depth < 1:
            raise ValueError('`max_depth` must be at least one')
        self.instructions = instructions
        self.step_store = step_store
        self.observer = observer
        self.directory = directory
        self.agents = dict(agents or {})
        self.aliases = dict(aliases or {})
        self.one_shot = one_shot
        self.max_depth = max_depth
        self.records: dict[str, DelegationTask] = {}
        self._workers: dict[str, asyncio.Task[str]] = {}
        self._released: dict[str, asyncio.Event] = {}
        self._stopping: set[str] = set()
        # Stacks, like `_emitters`: the latest registration receives, and each run removes only its own.
        self._receivers: dict[tuple[str, str | None], list[Callable[[UserPromptPart], str | None]]] = {}
        self._steering: dict[str, list[Callable[[str], str | None]]] = {}
        self._unsteered: dict[str, list[str]] = {}
        self._queued: dict[str, tuple[str, int]] = {}
        self._save_locks: dict[str, anyio.Lock] = {}
        self._questions: dict[str, _Question] = {}
        self._queued_questions: dict[str, str] = {}
        self._woken = anyio.Event()
        self._stops: set[str] = set()
        self._emitters: dict[tuple[str, str | None], list[Emitter]] = {}
        self._starting: dict[str, int] = {}
        self._open = False

    @staticmethod
    def current() -> DelegationTasks | None:
        """The owner bound to this execution context, if any."""
        return _CURRENT.get()

    @staticmethod
    def child_id() -> str | None:
        """The child whose tools are currently executing, including permission tools."""
        return _CHILD.get()

    @contextmanager
    def bind(self) -> Generator[None]:
        """Route this run's delegations to the owner without taking ownership of the run."""
        if not self._open:
            raise RuntimeError('Open the delegation task owner before binding it')
        # A child of another owner, whose tool binds this one, is none of this owner's children.
        child = _CHILD.set(None) if _CURRENT.get() is not self else None
        token = _CURRENT.set(self)
        try:
            yield
        finally:
            _CURRENT.reset(token)
            if child is not None:
                _CHILD.reset(child)

    @asynccontextmanager
    async def opened(self) -> AsyncGenerator[DelegationTasks]:
        """Restore saved children, then cancel and drain workers before releasing resources."""
        if self._open:
            raise RuntimeError('Delegation task owner is already open')
        self._open = True
        try:
            if self.directory is not None:
                directory = anyio.Path(self.directory)
                await directory.mkdir(parents=True, exist_ok=True)
                async for path in directory.glob('*.json'):
                    record = await _read(path)
                    if record.status == 'running':
                        if self.step_store is not None:
                            snapshot = await self.step_store.latest_snapshot(
                                run_id=record.run_id, include_interrupted=True
                            )
                            if snapshot is not None:
                                record.messages = snapshot.messages
                        record.status, record.outcome = 'finished', 'cancelled'
                        record.output = (
                            'Execution interrupted by process exit while waiting for an answer to:\n'
                            f'{record.pending_question}\nResume it with the answer. Inspect effects before resuming.'
                            if record.pending_question is not None
                            else 'Execution interrupted by process exit. Inspect effects before resuming.'
                        )
                        record.pending_question = None
                        record.finished_at = time.time()
                    self.records[record.id] = record
                for record in self.records.values():
                    ancestors = {record.id}
                    parent = record.parent_id
                    while parent is not None and parent in self.records:
                        if parent in ancestors:
                            raise ValueError('Cyclic saved task ancestry')
                        ancestors.add(parent)
                        parent = self.records[parent].parent_id
            yield self
        finally:
            # Closed first, so a delegation still persisting its acceptance schedules no worker after this drain.
            self._open = False
            with anyio.move_on_after(10, shield=True) as cleanup:
                for worker in self._workers.values():
                    worker.cancel()
                await asyncio.gather(*self._workers.values(), return_exceptions=True)
                for record in self.records.values():
                    if record.status == 'running':
                        record.status, record.outcome = 'finished', 'cancelled'
                        record.finished_at = time.time()
                    await self.save(record)
            if cleanup.cancel_called:
                raise RuntimeError('Delegation cleanup timed out; children did not release their resources')

    def persistence_capabilities(self) -> list[AbstractCapability[object]]:
        """Capture child steps using the application's existing persistence store."""
        return [StepPersistence(store=self.step_store, capture_frontier=True)] if self.step_store is not None else []

    async def notify(
        self,
        record: DelegationTask,
        event: AgentStreamEvent | None = None,
        *,
        model_name: str | None = None,
        context_window: int | None = None,
    ) -> None:
        """Send correlated activity, and the child's model when known, to the application's observer."""
        if self.observer is not None:
            await self.observer(
                DelegationTaskEvent(task=record, event=event, model_name=model_name, context_window=context_window)
            )

    async def save(self, record: DelegationTask) -> None:
        """Atomically replace one child's persisted history."""
        if not _SAFE_ID.fullmatch(record.id):
            raise ValueError('Invalid task identity')
        if self.directory is not None:
            async with self._save_locks.setdefault(record.id, anyio.Lock()):
                destination = anyio.Path(self.directory / f'{record.id}.json')
                temporary = anyio.Path(self.directory / f'{record.id}.tmp')
                await temporary.write_bytes(_ADAPTER.dump_json(record))
                await temporary.replace(destination)

    async def restore(self, record: DelegationTask) -> bool:
        """Add a finished task made elsewhere, such as one imported from another agent's session.

        Works whether or not the owner is open; a closed owner loads it when it opens. A task with the
        same ID that has run since, with a different `run_id`, is kept instead, as is the saved copy of
        one this owner has not loaded yet. Returns whether `record` was added.
        """
        if record.status != 'finished' or not _SAFE_ID.fullmatch(record.id) or not _SAFE_ID.fullmatch(record.run_id):
            raise ValueError('Only a finished task with a valid identity can be restored')
        current = self.records.get(record.id)
        if current is None and self.directory is not None:
            path = anyio.Path(self.directory / f'{record.id}.json')
            current = await _read(path) if await path.exists() else None
        if current is not None and current.run_id != record.run_id:
            return False
        if self.directory is not None:
            await anyio.Path(self.directory).mkdir(parents=True, exist_ok=True)
        self.records[record.id] = record
        await self.save(record)
        if self._open:
            await self.notify(record)
        return True

    async def delegate(
        self,
        *,
        agent_name: str,
        prompt: str,
        conversation_id: str,
        model: str | None,
        background: bool,
        resume: str | None,
        run: Runner,
        backgroundable: bool = True,
        task_id: str | None = None,
        resumable: bool = True,
        working_directory: str | None = None,
    ) -> str:
        """Start or resume a child; foreground waiting can be released by `background`.

        `task_id` names a new child, instead of a generated ID. A child is resumable unless `resumable`
        is off or its agent is one of `one_shot`.
        """
        if not self._open:
            raise RuntimeError('Delegation task owner is closed')
        if background and not backgroundable:
            raise ModelRetry(NOT_BACKGROUNDABLE)
        parent_id = _CHILD.get()
        if parent_id is not None:
            conversation_id = self.records[parent_id].conversation_id
        if resume is not None:
            record = self.check_resume(resume, conversation_id=conversation_id, agent_name=agent_name)
            if working_directory != record.working_directory:
                raise ModelRetry(RESUME_DIRECTORY_MISMATCH)
            record.prompt, record.model = prompt, model
            record.background, record.status = background, 'running'
            record.outcome, record.output, record.finished_at = None, '', None
            record.started_at, record.delivered = time.time(), False
            self._stopping.discard(record.id)
            self._stops.discard(record.id)
            self._unsteered.pop(record.id, None)
            record.generation += 1
            record.run_id = uuid4().hex
            record.parent_id = parent_id
            record.backgroundable = backgroundable
        else:
            record = DelegationTask(
                id=task_id or uuid4().hex,
                agent_name=agent_name,
                prompt=prompt,
                conversation_id=conversation_id,
                parent_id=_CHILD.get(),
                model=model,
                working_directory=working_directory,
                background=background,
                resumable=resumable and agent_name not in self.one_shot,
                backgroundable=backgroundable,
            )
            self.records[record.id] = record
        release = self._released[record.id] = asyncio.Event()
        # Persist acceptance before scheduling: cancellation here cannot leave an orphan worker.
        try:
            await self.save(record)
        except BaseException:
            # Never scheduled, so it settles here: nothing may wait for a worker it will not get.
            self._settle_unstarted(record)
            raise
        if record.user_stopped or record.status != 'running' or not self._open:
            # Stopped meanwhile, or its owner is closing: it settles without a worker.
            if record.status == 'running':
                self._settle_unstarted(record)
            return f'Task {record.id} was stopped before execution.'
        worker = asyncio.create_task(self._run(record, run), name=f'delegate-{record.id}')
        # Even a worker cancelled before it ran ends a wait for its task.
        worker.add_done_callback(lambda _: self._wake())
        self._workers[record.id] = worker
        if background:
            return self._receipt(record)
        waiter = asyncio.create_task(release.wait(), name=f'delegate-wait-{record.id}')
        try:
            await asyncio.wait((worker, waiter), return_when=asyncio.FIRST_COMPLETED)
            if worker.done():
                output = (
                    f'Task {record.id} (cancelled): stopped before execution.' if worker.cancelled() else await worker
                )
                record.delivered = True
                await self.save(record)
                return output
            return self._receipt(record)
        except asyncio.CancelledError:
            if not record.background:
                worker.cancel()
                with anyio.move_on_after(10, shield=True):
                    await asyncio.gather(worker, return_exceptions=True)
            raise
        finally:
            waiter.cancel()
            with anyio.move_on_after(5, shield=True):
                await asyncio.gather(waiter, return_exceptions=True)

    def check_resume(self, task_id: str, *, conversation_id: str, agent_name: str) -> DelegationTask:
        """The settled child `task_id` names, when `agent_name` can resume it in this conversation.

        Raises:
            ModelRetry: the task is unknown in this conversation, still running, not resumable, or of another agent.
        """
        record = self.records.get(task_id)
        if record is None or record.conversation_id != conversation_id:
            raise ModelRetry(f'Unknown task {task_id!r} in this conversation')
        if record.status == 'running' or (task_id in self._workers and not self._workers[task_id].done()):
            raise ModelRetry(f'Task {task_id!r} is still running')
        if not record.resumable or record.user_stopped:
            raise ModelRetry(f'Task {task_id!r} cannot resume; it is one-shot or was stopped by the user')
        if record.agent_name != agent_name:
            raise ModelRetry('Resume must use the original agent type')
        return record

    async def _run(self, record: DelegationTask, run: Runner) -> str:
        token = _CHILD.set(record.id)
        try:
            await self.notify(record)
            with capture_run_messages() as messages:
                try:
                    record.output = await run(record)
                finally:
                    if messages:
                        record.messages = messages
            record.outcome = record.outcome or 'ok'
        except (asyncio.CancelledError, RunCancelled):
            record.outcome, record.output = 'cancelled', STOPPED_OUTPUT
        except Exception as exc:
            record.outcome = record.outcome or 'failed'
            record.output = f'{type(exc).__name__}: {exc}'
        finally:
            _CHILD.reset(token)
            self._unsteered.pop(record.id, None)
            self._stops.discard(record.id)
            record.status, record.finished_at = 'finished', time.time()
            # A cancelled/failed parent cannot leave descendants without an observer.
            if record.outcome != 'ok':
                for child in tuple(self.records.values()):
                    if child.parent_id == record.id and child.status == 'running':
                        await self.cancel(child.id, user=record.user_stopped)
            with anyio.move_on_after(5, shield=True):
                await self.save(record)
                try:
                    await self.notify(record)
                except Exception:
                    logging.getLogger(__name__).exception('Task completion observer failed for %s', record.id)
                self.queue_reports(conversation_id=record.conversation_id, parent_id=record.parent_id)
        return f'Task {record.id} ({record.outcome}):\n{record.output}'

    @staticmethod
    def _receipt(record: DelegationTask) -> str:
        return (
            f'Task {record.id} is running in the background. This is an acceptance receipt, not a result. '
            'Do not claim completion or invent findings. A task report will be delivered after it settles.'
        )

    def background(self, task_id: str) -> None:
        """Release the foreground delegate tool while its child continues."""
        record = self.records[task_id]
        if not record.backgroundable:
            raise ValueError('This workspace cannot outlive its parent run')
        if record.status == 'running':
            record.background = True
            self._released[task_id].set()

    async def cancel(self, task_id: str, *, user: bool = True) -> None:
        """Stop and drain this child and descendants, leaving unrelated siblings running."""
        record = self.records[task_id]
        if user:
            record.user_stopped = True
        worker = self._workers.get(task_id)
        if worker is not None and not worker.done() and task_id not in self._stopping:
            self._stopping.add(task_id)
            worker.cancel()
        for child in tuple(self.records.values()):
            if child.parent_id == task_id:
                await self.cancel(child.id, user=user)
        with anyio.move_on_after(10, shield=True) as cleanup:
            if worker is not None:
                await asyncio.gather(worker, return_exceptions=True)
            if record.status == 'running':
                # Its run never started to read steering sent meanwhile.
                self._unsteered.pop(task_id, None)
                self._settle_unstarted(record)
                await self.notify(record)
            await self.save(record)
        if cleanup.cancel_called:
            raise RuntimeError(f'Task {task_id} did not finish cancellation cleanup')

    def reports(self, *, conversation_id: str, parent_id: str | None = None) -> list[DelegationTask]:
        """Settled background reports awaiting delivery to their own parent only."""
        return [
            r
            for r in self.records.values()
            if r.conversation_id == conversation_id
            and r.parent_id == parent_id
            and r.background
            and r.status == 'finished'
            and not r.delivered
        ]

    def questions(self, *, conversation_id: str, parent_id: str | None = None) -> list[DelegationTask]:
        """Running children whose question awaits delivery to their own parent, like `reports`."""
        return [
            record
            for task_id, question in self._questions.items()
            if not question.queued
            and (record := self.records[task_id]).conversation_id == conversation_id
            and record.parent_id == parent_id
        ]

    def steer(self, task_id: str, text: str) -> bool:
        """Send a running child a user message, which its run reads at its next model request.

        Returns `False` when the child is not running, so a caller can offer to resume it instead.
        A message sent before the child's run starts waits for it, and is dropped if the child is stopped
        first; one sent as its final response settles may arrive too late for it to read. Raises `ValueError` for blank text and `KeyError`
        for an unknown task.
        """
        if not text.strip():
            raise ValueError('Steering text must not be empty')
        if self.records[task_id].status != 'running':
            return False
        steering = self._steering.get(task_id)
        if steering is None:
            self._unsteered.setdefault(task_id, []).append(text)
        else:
            steering[-1](text)
        return True

    def history(self, task_id: str) -> list[ModelMessage]:
        """Independent replay history; callers cannot mutate the saved child through it."""
        return copy.deepcopy(self.records[task_id].messages)

    async def allow_resume(self, task_id: str) -> None:
        """Explicit user authorization to resume a stopped, resumable child."""
        record = self.records[task_id]
        if (
            not record.resumable
            or record.status == 'running'
            or (task_id in self._workers and not self._workers[task_id].done())
        ):
            raise ValueError('Only a settled, resumable task can be resumed')
        record.user_stopped = False
        await self.save(record)

    def queue_reports(self, *, conversation_id: str, parent_id: str | None) -> None:
        """Queue only settled direct-child reports on an active parent's native queue."""
        receivers = self._receivers.get((conversation_id, parent_id))
        if receivers is None:
            return
        receiver = receivers[-1]
        for record in self.reports(conversation_id=conversation_id, parent_id=parent_id):
            identity = (record.id, record.generation)
            if identity in self._queued.values():
                continue
            enqueue_id = receiver(record.report())
            if enqueue_id is not None:
                self._queued[enqueue_id] = identity
        for task_id, question in self._questions.items():
            record = self.records[task_id]
            if question.queued or record.conversation_id != conversation_id or record.parent_id != parent_id:
                continue
            enqueue_id = receiver(_question_part(record, question))
            if enqueue_id is not None:
                question.queued = True
                self._queued_questions[enqueue_id] = task_id

    @contextmanager
    def receiving(
        self,
        *,
        conversation_id: str,
        parent_id: str | None,
        enqueue: Callable[[UserPromptPart], str | None],
        steer: Callable[[str], str | None] | None = None,
    ) -> Generator[None]:
        """Keep a queue attached only for the lifetime of its owning run.

        `steer` queues a user message on a child's own run, for `DelegationTasks.steer`.
        """
        key = (conversation_id, parent_id)
        # What this run queued, which it alone can still deliver.
        queued: set[str] = set()

        def receive(part: UserPromptPart) -> str | None:
            enqueue_id = enqueue(part)
            if enqueue_id is not None:
                queued.add(enqueue_id)
            return enqueue_id

        self._receivers.setdefault(key, []).append(receive)
        if steer is not None and parent_id is not None:
            self._steering.setdefault(parent_id, []).append(steer)
            for text in self._unsteered.pop(parent_id, ()):
                steer(text)
        self.queue_reports(conversation_id=conversation_id, parent_id=parent_id)
        try:
            yield
        finally:
            _unregister(self._receivers, key, receive)
            if steer is not None and parent_id is not None:
                _unregister(self._steering, parent_id, steer)
            self._queued = {
                enqueue_id: identity for enqueue_id, identity in self._queued.items() if enqueue_id not in queued
            }
            # A question queued on a run that ended before delivering it is queued again on the next one.
            for enqueue_id in queued & self._queued_questions.keys():
                task_id = self._queued_questions.pop(enqueue_id)
                if (question := self._questions.get(task_id)) is not None:
                    question.queued = False
            # Another run of the same parent may still be receiving: it gets what this one could not deliver.
            self.queue_reports(conversation_id=conversation_id, parent_id=parent_id)

    async def acknowledge(self, enqueue_id: str) -> None:
        """Persist delivery only after core confirms it entered message history."""
        self._queued_questions.pop(enqueue_id, None)
        identity = self._queued.pop(enqueue_id, None)
        if identity is not None:
            task_id, generation = identity
            record = self.records[task_id]
            if record.generation == generation:
                record.delivered = True
                await self.save(record)

    async def wait_children(self, task_id: str | None) -> None:
        """Wait for direct children before the child's final output can settle, or until one asks a question.

        `None` waits for the children of the top-level run. A child waiting for an answer would otherwise
        wait for its parent forever, so its question ends the wait and the parent gets a turn to answer it.
        """
        while True:
            running = [r.id for r in self.records.values() if r.parent_id == task_id and r.status == 'running']
            if not running or any(child in self._questions for child in running):
                return
            # Cancelling this wait leaves the children running.
            await self._woken.wait()

    def _settle_unstarted(self, record: DelegationTask) -> None:
        """Settle a task that was stopped before it got a worker, and wake whoever waits for it."""
        record.status, record.outcome = 'finished', 'cancelled'
        record.output, record.finished_at = UNSTARTED_OUTPUT, time.time()
        self._wake()

    def _wake(self) -> None:
        """Wake `wait_children`: a child asked a question, its worker finished, or it settled without one."""
        self._woken.set()
        self._woken = anyio.Event()

    def running_count(self, conversation_id: str) -> int:
        """How many children of a conversation are running, or being started by `starting`."""
        running = sum(r.conversation_id == conversation_id and r.status == 'running' for r in self.records.values())
        return running + self._starting.get(conversation_id, 0)

    @contextmanager
    def starting(self, conversation_id: str) -> Generator[None]:
        """Count a child that is about to be delegated as running, so a concurrent delegation can't overrun a limit."""
        self._starting[conversation_id] = self._starting.get(conversation_id, 0) + 1
        try:
            yield
        finally:
            self._starting[conversation_id] -= 1

    @contextmanager
    def publishing(self, *, conversation_id: str, parent_id: str | None, emit: Emitter) -> Generator[None]:
        """Send the task events of a run's children into that run's event stream, while it lasts."""
        key = (conversation_id, parent_id)
        # Several runs, or capabilities of one run, can publish for the same parent; the latest gets the events.
        self._emitters.setdefault(key, []).append(emit)
        try:
            yield
        finally:
            _unregister(self._emitters, key, emit)

    def delegation_of(self, run_id: str | None) -> DelegationTask | None:
        """The child that the run `run_id` runs, or `None` for a run that isn't one of this owner's delegations.

        A run that a child's tool starts sees the child's ID too, but runs under an ID of its own.
        """
        child = _CHILD.get()
        if child is None:
            return None
        record = self.records[child]
        return record if record.run_id == run_id else None

    async def publish(
        self, record: DelegationTask, event: CapabilityEventT, *, observe: bool = True
    ) -> CapabilityEventT:
        """Emit a task event into its parent's run, when that run is publishing, then send it to the observer.

        Returns the event once the parent run's listeners have seen it, so a caller can read their decisions.
        """
        emitters = self._emitters.get((record.conversation_id, record.parent_id))
        if emitters is not None:
            await emitters[-1](event)
        if observe:
            await self.notify(record, event)
        return event

    def pending_question(self, task_id: str) -> tuple[str, str] | None:
        """The `(question_id, question)` a running child is waiting on an answer to, if any."""
        question = self._questions.get(task_id)
        return None if question is None else (question.question_id, question.text)

    async def ask(self, task_id: str, *, question_id: str, question: str) -> str:
        """Put a child's question to its parent, and wait for `answer`.

        The question is queued on the parent's run like a report. `request_stop` answers it with
        `CANCELLED_ANSWER`.
        """
        if task_id in self._stops:
            # Asked to stop while it was putting the question, so it gets no turn to wait for an answer.
            return CANCELLED_ANSWER
        record = self.records[task_id]
        pending = self._questions[task_id] = _Question(
            question_id=question_id, text=question, answer=asyncio.get_running_loop().create_future()
        )
        self._wake()
        self.queue_reports(conversation_id=record.conversation_id, parent_id=record.parent_id)
        # Saved, so a process exit while it waits leaves the question in the interrupted task's output.
        record.pending_question = question
        await self.save(record)
        try:
            try:
                # The observer learns the task waits, so an application can start its idle parent's next run.
                await self.notify(record)
            except Exception:
                logging.getLogger(__name__).exception('Task question observer failed for %s', record.id)
            return await pending.answer
        finally:
            record.pending_question = None
            if self._questions.get(task_id) is pending:
                del self._questions[task_id]

    def answer(self, task_id: str, *, question_id: str, answer: str) -> bool:
        """Answer the question a child is waiting on. `False` if it is not waiting on `question_id`."""
        question = self._questions.get(task_id)
        if question is None or question.question_id != question_id or question.answer.done():
            return False
        question.answer.set_result(answer)
        return True

    def request_stop(self, task_id: str) -> None:
        """Ask a running child to stop before its next model request, and to stop waiting for an answer."""
        self._stops.add(task_id)
        question = self._questions.get(task_id)
        if question is not None and not question.answer.done():
            question.answer.set_result(CANCELLED_ANSWER)

    def stop_requested(self, task_id: str) -> bool:
        """Whether `request_stop` asked this child to stop."""
        return task_id in self._stops


@contextmanager
def isolated(owner: DelegationTasks | None) -> Generator[None]:
    """Bind `owner`, or no owner, for a run that isn't a delegation of the bound owner, with no current child."""
    current, child = _CURRENT.set(owner), _CHILD.set(None)
    try:
        yield
    finally:
        _CHILD.reset(child)
        _CURRENT.reset(current)


def enqueue_while_open(
    ctx: RunContext[object], part: UserPromptPart, *, priority: Literal['asap', 'when_idle']
) -> str | None:
    """Queue `part` on the run of `ctx`, or return `None` once that run takes no more messages.

    A run closes its queue when it reads its last queued messages, but its capabilities still receive until it
    returns. What it no longer takes stays with the owner, which queues it on the conversation's next run.
    """
    try:
        return ctx.enqueue(part, priority=priority)
    except UserError:
        # The only `UserError` a running agent's `enqueue` raises for a part: the run has ended.
        return None


class DelegationReports(AbstractCapability[object]):
    """Use core's queued-message seam for automated reports and child joins."""

    def __init__(
        self,
        owner: DelegationTasks,
        *,
        conversation_id: str,
        task_id: str | None = None,
        priority: Literal['asap', 'when_idle'] = 'when_idle',
    ) -> None:
        self.priority: Literal['asap', 'when_idle'] = priority
        self.owner = owner
        self.conversation_id = conversation_id
        self.task_id = task_id
        self.id = 'delegation_reports'

    async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        with self.owner.receiving(
            conversation_id=self.conversation_id,
            parent_id=self.task_id,
            enqueue=lambda part: enqueue_while_open(ctx, part, priority=self.priority),
            # Like a report, steering a run that has read its last queued messages is dropped.
            steer=lambda text: enqueue_while_open(ctx, UserPromptPart(text), priority='asap'),
        ):
            return await handler()

    async def after_output_process(
        self,
        ctx: RunContext[object],
        *,
        output_context: OutputContext,
        output: object,
    ) -> object:
        if self.task_id is not None and not ctx.partial_output:
            await self.owner.wait_children(self.task_id)
            self.owner.queue_reports(conversation_id=self.conversation_id, parent_id=self.task_id)
        return output

    @on_event(EnqueuedMessagesEvent)
    async def delivered(self, ctx: RunContext[object], event: EnqueuedMessagesEvent) -> None:
        await self.owner.acknowledge(event.enqueue_id)
