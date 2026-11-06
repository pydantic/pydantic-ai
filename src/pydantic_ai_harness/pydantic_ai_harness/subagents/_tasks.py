"""Session-owned delegation lifetime, independent of terminal rendering."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import hashlib
import logging
import re
import time
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeVar, overload
from uuid import uuid4

import anyio
from pydantic import SerializerFunctionWrapHandler, TypeAdapter, WrapSerializer
from pydantic_core import PydanticSerializationError

from pydantic_ai import AgentRunResult, AgentStreamEvent, CapabilityEvent, RunContext, capture_run_messages
from pydantic_ai.agent import AbstractAgent, EventStreamHandler
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, WrapRunHandler, on_event
from pydantic_ai.exceptions import ModelRetry, RunCancelled, UserError
from pydantic_ai.messages import (
    EnqueuedMessagesEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    UserPromptPart,
    repair_messages,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputContext
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.workspaces import ReadOnlyWorkspace, Workspace, WorkspaceBackend
from pydantic_ai_harness._history import STORED_HISTORY_CONTEXT
from pydantic_ai_harness._task_report import make_task_output, make_task_question, make_team_message
from pydantic_ai_harness.planning import TaskStatus
from pydantic_ai_harness.step_persistence import StepPersistence, StepStore
from pydantic_ai_harness.subagents._events import DelegationOutcome
from pydantic_ai_harness.subagents._team import LEAD, AgentMessage, MemberClaims, SentMessage, TeamMessages

if TYPE_CHECKING:
    from pydantic_ai_harness.subagents._toolset import SubAgent

_CURRENT: ContextVar[DelegationTasks | None] = ContextVar('delegation_tasks', default=None)
_CHILD: ContextVar[str | None] = ContextVar('delegation_child', default=None)

NOT_BACKGROUNDABLE = 'This workspace is owned by the parent run and cannot run background children'
"""Why a delegation can't run in the background."""

RESUME_DIRECTORY_MISMATCH = 'A resumed task cannot change its working directory.'
"""Why a resume cannot select a different workspace."""

SELF_AGENT_NAME = 'self'
"""The name a delegate to the running agent itself is listed and called by (`SubAgents.include_self`)."""

_MEMBER_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,63}\Z')


def _dump_messages(messages: list[ModelMessage], handler: SerializerFunctionWrapHandler) -> object:
    try:
        return handler(messages)
    except (UnicodeDecodeError, PydanticSerializationError):
        # Non-UTF-8 `bytes` in an `Any` field: only an outermost adapter's base64 config reaches them. The
        # error arrives wrapped when a serializer function sits between, as tool results' do.
        return ModelMessagesTypeAdapter.dump_python(messages, mode='json')


@dataclass(kw_only=True)
class InboxMessage:
    """A saved message awaiting delivery, removed once core confirms it entered its recipient's history."""

    id: str = field(default_factory=lambda: uuid4().hex)
    part: UserPromptPart
    """What the recipient's run receives: the user's words, or a framed message or report from another agent."""


@dataclass(kw_only=True)
class DelegationTask:
    """A stable child conversation, retained across explicit resumes."""

    id: str
    name: str | None = None
    """The team member's name, unique in its conversation, for a member `DelegationTasks.start` started."""
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
    inbox: list[InboxMessage] = field(default_factory=list[InboxMessage])
    """A team member's messages that none of its runs has acknowledged yet, oldest first."""

    def report(self) -> UserPromptPart:
        """The settled task's report: the child's output, framed as `<task_output>` data on the user channel.

        The output was written by a model and may carry instructions from whatever the child read, so it
        never enters system voice. The frame's escaped attributes identify the task (its id, a team member's
        name, its agent name, and an `outcome` other than `ok`). The acceptance receipt the parent got when the
        task started already says a report will follow, so nothing precedes the frame. The `TextContent`
        metadata identifies the report without matching its text.
        """
        return make_task_output(
            self.output,
            task_id=self.id,
            agent_name=self.agent_name,
            outcome=self.outcome or 'ok',
            generation=self.generation,
            name=self.name,
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
ClaimsResolver = Callable[[str], MemberClaims | None]
"""Returns the board of a conversation, by its ID, whose claims a team member hands back when it stops."""
LeadWaker = Callable[[str], Awaitable[object]]
"""Starts a run of the lead of a conversation, by its ID, to read its inbox."""


@dataclass(kw_only=True)
class MemberRun:
    """What a `member_runner` needs to run a turn of a team member, as `delegate_task` sets up a child's run.

    Pass it everything, or call `run`.
    """

    agent: AbstractAgent[Any, Any]
    """The member's agent: the runner's own agent for `self`, else the owner's sub-agent of the member's type."""

    prompt: str | None
    """The member's task for its first turn; `None` when it has history, so a wake adds only its messages."""

    model: str | None
    """The member's model ID, for the runner to resolve, or `None` for the lead's current model."""

    keeps_own_model: bool
    """Whether a member with no model ID runs on its agent's own model, as a sub-agent with one does under
    `delegate_task`, rather than on the model the runner passes."""

    read_only: bool
    """Whether its sub-agent is `read_only`, so it runs in a `ReadOnlyWorkspace`."""

    capabilities: list[AbstractCapability[Any]]
    """Delivery of its inbox and its children's reports, step persistence, and `TeamMessages`."""

    event_stream_handler: EventStreamHandler[Any]
    """Keeps the member's saved history current, and sends its events to the owner's observer."""

    message_history: list[ModelMessage]
    conversation_id: str
    """The member's own ID, as a child's run has."""

    run_id: str

    async def run(
        self,
        *,
        model: Model | KnownModelName | str | None = None,
        deps: Any = None,
        workspace: WorkspaceBackend | None = None,
        usage: RunUsage | None = None,
        usage_limits: UsageLimits | None = None,
        capabilities: Sequence[AgentCapability[Any]] = (),
    ) -> AgentRunResult[Any]:
        """Run this turn of the member.

        Args:
            model: The model the runner resolved from `model`, or the lead's current model. Unused when
                `keeps_own_model` is set.
            deps: The dependencies to run with.
            workspace: A workspace the runner owns, which outlives the turn; wrapped in a `ReadOnlyWorkspace`
                when `read_only` is set.
            usage: The member's own usage, to accumulate across its turns.
            usage_limits: The member's budget. A turn that exceeds it fails, and the member is not woken again
                until a message arrives.
            capabilities: More capabilities for the turn, such as the team's gates.
        """
        if self.read_only:
            if workspace is None:
                raise UserError('A `read_only` team member needs the `workspace` to make read-only.')
            workspace = ReadOnlyWorkspace(workspace if isinstance(workspace, Workspace) else Workspace(workspace))
        return await self.agent.run(
            self.prompt,
            model=None if self.keeps_own_model else model,
            deps=deps,
            workspace=workspace,
            usage=usage,
            usage_limits=usage_limits,
            capabilities=[*self.capabilities, *capabilities],
            event_stream_handler=self.event_stream_handler,
            message_history=self.message_history,
            conversation_id=self.conversation_id,
            run_id=self.run_id,
        )


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


@dataclass(kw_only=True)
class _LeadInbox:
    conversation_id: str
    messages: list[InboxMessage]


_ADAPTER = TypeAdapter(DelegationTask)
_LEAD_INBOX_ADAPTER = TypeAdapter(_LeadInbox)
_SAFE_ID = re.compile(r'[0-9a-f]{32}\Z')


def _inbox_name(conversation_id: str) -> str:
    return hashlib.sha256(conversation_id.encode()).hexdigest()


async def _read(path: anyio.Path) -> DelegationTask:
    """A saved task, checked to have the identity its file name gives it."""
    record = _ADAPTER.validate_json(await path.read_bytes(), context=STORED_HISTORY_CONTEXT)
    if not _SAFE_ID.fullmatch(record.id) or not _SAFE_ID.fullmatch(record.run_id) or path.name != f'{record.id}.json':
        raise ValueError(f'Invalid saved task identity in {path.name!r}')
    return record


class DelegationTasks:
    """Own detached children until `opened` exits; bind to runs with `bind`.

    Outside this opt-in scope, `SubAgents` retains its synchronous delegation contract.
    Use one owner per application session. The caller owns workspace and plugin resources
    until this owner's exit has cancelled and drained all children.

    With a `member_runner`, the owner also runs teams: named, long-lived members that `start` starts, and that a
    message `send` saves for them wakes once their run has settled.
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
        member_runner: Runner | None = None,
        max_active: int | None = None,
        claims: ClaimsResolver | None = None,
        lead_waker: LeadWaker | None = None,
        lead_wake_delay: float = 1.0,
    ) -> None:
        """Create an owner.

        Args:
            observer: Receives every task's events.
            directory: Where tasks, and team leads' inboxes, are saved.
            agents: Specialist sub-agents, by name, for every delegation and team member.
            aliases: Other names for agents.
            one_shot: Agent names whose tasks never resume.
            max_depth: How many levels a delegation tree may have, counting the top-level run.
            instructions: More guidance for the delegate tool.
            step_store: Where child runs checkpoint their steps.
            member_runner: Runs a turn of a team member, with `member_run`. Required to start or wake members,
                including ones loaded from `directory`.
            max_active: How many team members run at once; others wait for a turn. `None` sets no limit.
            claims: The board of a conversation, whose claims a team member that stops, fails or was
                interrupted by a process exit hands back to `pending`.
            lead_waker: Starts a run of a conversation's lead to read the messages and reports in its inbox,
                when no run of it is active. Without one, they wait for its next run.
            lead_wake_delay: How long, in seconds, the owner gathers the lead's messages before it calls
                `lead_waker`, unless every member of the conversation has settled first.
        """
        if max_depth < 1:
            raise ValueError('`max_depth` must be at least one')
        if max_active is not None and max_active < 1:
            raise ValueError('`max_active` must be at least one')
        self.member_runner = member_runner
        self.claims = claims
        self.lead_waker = lead_waker
        self.lead_wake_delay = lead_wake_delay
        self._slots = anyio.Semaphore(max_active) if max_active is not None else None
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
        self._lead_inboxes: dict[str, list[InboxMessage]] = {}
        # Like `_steering`: a team member's run queues its messages, the latest registration receiving.
        self._messaging: dict[str, list[Callable[[UserPromptPart], str | None]]] = {}
        # Messages queued on a live run, and what each enqueue delivers: `(conversation_id, task_id, message_id)`,
        # with `None` for the lead. Kept apart from report delivery, so a message never marks a report delivered.
        self._queued_messages: set[str] = set()
        self._message_acks: dict[str, tuple[str, str | None, str]] = {}
        self._lead_wakes: dict[str, anyio.Event] = {}
        self._lead_tasks: set[asyncio.Task[None]] = set()

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
            interrupted: list[DelegationTask] = []
            if self.directory is not None:
                directory = anyio.Path(self.directory)
                await directory.mkdir(parents=True, exist_ok=True)
                async for path in directory.glob('*.json'):
                    record = await _read(path)
                    if record.status == 'running':
                        await self._settle_interrupted(record)
                        interrupted.append(record)
                    self.records[record.id] = record
                for record in self.records.values():
                    ancestors = {record.id}
                    parent = record.parent_id
                    while parent is not None and parent in self.records:
                        if parent in ancestors:
                            raise ValueError('Cyclic saved task ancestry')
                        ancestors.add(parent)
                        parent = self.records[parent].parent_id
                await self._load_lead_inboxes(directory / 'inbox')
                await self._reconcile_members(interrupted)
            yield self
        finally:
            # Closed first, so a delegation still persisting its acceptance schedules no worker after this drain.
            self._open = False
            with anyio.move_on_after(10, shield=True) as cleanup:
                for task in self._lead_tasks:
                    task.cancel()
                await asyncio.gather(*self._lead_tasks, return_exceptions=True)
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

    async def _settle_interrupted(self, record: DelegationTask) -> None:
        """Load a task a process exit interrupted as cancelled, with its latest saved steps."""
        if self.step_store is not None:
            snapshot = await self.step_store.latest_snapshot(run_id=record.run_id, include_interrupted=True)
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

    async def _load_lead_inboxes(self, directory: anyio.Path) -> None:
        if not await directory.exists():
            return
        async for path in directory.glob('*.json'):
            saved = _LEAD_INBOX_ADAPTER.validate_json(await path.read_bytes(), context=STORED_HISTORY_CONTEXT)
            if path.name != f'{_inbox_name(saved.conversation_id)}.json':
                raise ValueError(f'Invalid saved inbox identity in {path.name!r}')
            self._lead_inboxes[saved.conversation_id] = saved.messages

    async def _reconcile_members(self, interrupted: list[DelegationTask]) -> None:
        """Settle team members a process exit interrupted, and pass on reports that never reached the lead's inbox."""
        for record in interrupted:
            if record.name is not None:
                await self._release_claims(record)
        for record in list(self.records.values()):
            if record.name is not None and not record.delivered:
                await self._report_to_lead(record)
        for conversation_id in list(self._lead_inboxes):
            self._schedule_lead_wake(conversation_id)

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
            ModelRetry: the task is unknown in this conversation, a team member, still running, not resumable, or of
                another agent.
        """
        record = self.records.get(task_id)
        if record is None or record.conversation_id != conversation_id:
            raise ModelRetry(f'Unknown task {task_id!r} in this conversation')
        if record.name is not None:
            raise ModelRetry(
                f'Task {task_id!r} is the team member {record.name!r}: message it with `send_message` instead of '
                'resuming it.'
            )
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
            async with self._slot(record):
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
                if record.name is not None:
                    await self._hand_over(record)
                try:
                    await self.notify(record)
                except Exception:
                    logging.getLogger(__name__).exception('Task completion observer failed for %s', record.id)
                self.queue_reports(conversation_id=record.conversation_id, parent_id=record.parent_id)
                if record.name is not None:
                    await self._wake_if_messaged(record)
        return f'Task {record.id} ({record.outcome}):\n{record.output}'

    def _slot(self, record: DelegationTask) -> AbstractAsyncContextManager[object]:
        """A team member's turn among `max_active`; any other task runs at once."""
        return self._slots if self._slots is not None and record.name is not None else nullcontext()

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
        """Settled background reports awaiting delivery to their own parent only.

        A team member's report goes to its lead's inbox instead, which `lead_inbox` lists.
        """
        return [
            r
            for r in self.records.values()
            if r.conversation_id == conversation_id
            and r.parent_id == parent_id
            and r.name is None
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

        Returns `False` when the child is not running, so a caller can offer to resume it instead, and for a team
        member, which takes messages with `send` instead. A message sent before the child's run starts waits for
        it, and is dropped if the child is stopped first; one sent as its final response settles may arrive too
        late for it to read. Raises `ValueError` for blank text and `KeyError` for an unknown task.
        """
        if not text.strip():
            raise ValueError('Steering text must not be empty')
        record = self.records[task_id]
        if record.status != 'running' or record.name is not None:
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

    def members(self, conversation_id: str) -> list[DelegationTask]:
        """The team members of a conversation, oldest first."""
        return [r for r in self.records.values() if r.name is not None and r.conversation_id == conversation_id]

    def lead_inbox(self, conversation_id: str) -> list[InboxMessage]:
        """The messages and reports a conversation's lead has not received yet, oldest first.

        Check it, rather than `reports`, to decide whether a team's lead has something to read.
        """
        return list(self._lead_inboxes.get(conversation_id, ()))

    async def start(
        self, *, name: str, agent_name: str, prompt: str, conversation_id: str, model: str | None = None
    ) -> DelegationTask:
        """Start a team member in the background, and return its record without waiting for it.

        A member is a top-level task of the conversation, with no parent, named `name`: its reports go to the
        lead's inbox, and it settles into an idle state that a message wakes. `member_runner` runs each of its turns,
        so a member never borrows the run that started it, which may have ended.

        Args:
            name: The member's name, unique in the conversation: a letter, then up to 63 letters, digits, `_` or `-`.
            agent_name: `self`, for the runner's own agent, or one of `agents`, which `one_shot` and
                `SubAgent(resumable=False)` exclude: a member must be wakeable. A member gets no workspace of
                the starting run's, which could end before it does; the runner passes one it owns.
            prompt: The member's task.
            conversation_id: The lead's conversation.
            model: A model ID for the runner to resolve, or `None` for the lead's current model.

        Raises:
            RuntimeError: The owner is closed or has no `member_runner`.
            ValueError: The name is invalid or taken, or the agent is unknown or one-shot.
        """
        if not self._open:
            raise RuntimeError('Delegation task owner is closed')
        if self.member_runner is None:
            raise RuntimeError('Pass `DelegationTasks(member_runner=...)` to start team members')
        agent_name = self.aliases.get(agent_name, agent_name)
        if agent_name != SELF_AGENT_NAME and agent_name not in self.agents:
            raise ValueError(f'Unknown agent {agent_name!r}')
        if agent_name in self.one_shot or (agent_name in self.agents and not self.agents[agent_name].resumable):
            raise ValueError(f'Agent {agent_name!r} is one-shot, and a team member must be wakeable')
        if not _MEMBER_NAME.match(name) or name == LEAD:
            raise ValueError(f'Invalid team member name {name!r}')
        if any(member.name == name for member in self.members(conversation_id)):
            raise ValueError(f'The team already has a member named {name!r}')
        record = DelegationTask(
            id=uuid4().hex,
            name=name,
            agent_name=agent_name,
            prompt=prompt,
            conversation_id=conversation_id,
            model=model,
            background=True,
        )
        self.records[record.id] = record
        try:
            await self.save(record)
        except BaseException:
            del self.records[record.id]
            raise
        self._schedule_member(record)
        return record

    def member_run(self, record: DelegationTask, *, self_agent: AbstractAgent[Any, Any]) -> MemberRun:
        """What `member_runner` runs for this turn of the team member `record`.

        Args:
            record: The member the runner was called with.
            self_agent: The agent a member of type `self` runs: the lead's agent, as it is now.
        """
        if record.name is None:
            raise ValueError(f'Task {record.id!r} is not a team member')
        if record.agent_name == SELF_AGENT_NAME:
            agent, read_only, keeps_own_model = self_agent, False, False
        else:
            sub_agent = self.agents[record.agent_name]
            agent, read_only = sub_agent.agent, sub_agent.read_only
            keeps_own_model = record.model is None and sub_agent.agent.model is not None
        return MemberRun(
            agent=agent,
            prompt=None if record.messages else record.prompt,
            model=record.model,
            keeps_own_model=keeps_own_model,
            read_only=read_only,
            capabilities=[
                DelegationReports(self, conversation_id=record.conversation_id, task_id=record.id),
                *self.persistence_capabilities(),
                TeamMessages(self),
            ],
            event_stream_handler=partial(self._stream_member, record),
            message_history=repair_messages(record.messages),
            conversation_id=record.id,
            run_id=record.run_id,
        )

    async def _stream_member(
        self, record: DelegationTask, ctx: RunContext[Any], events: AsyncIterable[AgentStreamEvent]
    ) -> None:
        async for event in events:
            record.messages = ctx.messages
            model = ctx.model
            await self.notify(record, event, model_name=model.model_name, context_window=model.context_window)

    @overload
    async def send(self, content: str | AgentMessage, *, to: str, conversation_id: str) -> SentMessage: ...

    @overload
    async def send(self, content: AgentMessage, *, to: None, conversation_id: str) -> SentMessage: ...

    async def send(self, content: str | AgentMessage, *, to: str | None, conversation_id: str) -> SentMessage:
        """Save a message for a team member, or the conversation's lead, then deliver it as soon as it can be.

        A `str` is what the user typed to the member, and arrives as their words. It clears a user stop, as
        `allow_resume` does. An `AgentMessage` comes from another agent of the team, and arrives framed as
        `<peer_message>` data. The lead takes agent messages only: the user talks to it through its prompt.

        The message is saved first, and leaves the recipient's inbox only when core confirms it entered history.
        A running member reads it at its next model request; an idle member is woken with it, unless the user
        stopped it or it can't be resumed. A member whose turn ended in failure is woken by a new message only,
        never by one it already had. The lead's active run reads it once its current work is done; with no active
        run, it waits for the next one, which `lead_waker` can start.

        Args:
            content: The message.
            to: A team member's task ID, or `None` for the conversation's lead.
            conversation_id: The team's conversation.

        Raises:
            RuntimeError: The owner is closed, or a member must be woken and there is no `member_runner`.
            ValueError: The message is empty, or `to` isn't a member of the conversation.
        """
        if not self._open:
            raise RuntimeError('Delegation task owner is closed')
        text = content if isinstance(content, str) else content.text
        if not text.strip():
            raise ValueError('A message must not be empty')
        message_id = uuid4().hex
        if to is None:
            if isinstance(content, str):
                raise ValueError('The lead takes agent messages only: the user talks to it through its prompt')
            part = make_team_message(content.text, sender=content.sender, message_id=message_id)
            await self._append_lead(conversation_id, InboxMessage(id=message_id, part=part))
            return SentMessage(id=message_id, status='running' if self._lead_active(conversation_id) else 'saved')
        record = self.records.get(to)
        if record is None or record.name is None or record.conversation_id != conversation_id:
            raise ValueError(f'Unknown team member {to!r} in this conversation')
        if isinstance(content, str):
            part = UserPromptPart(content)
            record.user_stopped = False
        else:
            part = make_team_message(content.text, sender=content.sender, message_id=message_id)
        record.inbox.append(InboxMessage(id=message_id, part=part))
        await self.save(record)
        worker = self._workers.get(record.id)
        if record.status == 'running' or (worker is not None and not worker.done()):
            # Not attached yet, or no longer reading: its run reads it when it starts, or it settles into a wake.
            self._deliver_member(record)
            return SentMessage(id=message_id, status='running')
        if record.user_stopped or not record.resumable:
            await self.notify(record)
            return SentMessage(id=message_id, status='saved')
        await self._wake_member(record)
        return SentMessage(id=message_id, status='woken')

    def _schedule_member(self, record: DelegationTask) -> None:
        """Run a turn of a team member, bound to this owner and no other run, as a top-level task."""
        assert self.member_runner is not None, 'Checked before a member is started or woken.'
        self._released[record.id] = asyncio.Event()
        # A fresh context: whoever starts or wakes the member, its run is no child of their run.
        worker = asyncio.create_task(
            self._run_member(record, self.member_runner), name=f'member-{record.id}', context=contextvars.Context()
        )
        worker.add_done_callback(lambda _: self._wake())
        self._workers[record.id] = worker

    async def _run_member(self, record: DelegationTask, run: Runner) -> str:
        with isolated(self):
            return await self._run(record, run)

    async def _wake_member(self, record: DelegationTask) -> None:
        """Start the next turn of a settled team member, keeping its history, its name, and no parent."""
        if self.member_runner is None:
            raise RuntimeError('Pass `DelegationTasks(member_runner=...)` to wake team members')
        record.status, record.outcome, record.output, record.finished_at = 'running', None, '', None
        record.started_at, record.delivered = time.time(), False
        record.generation += 1
        record.run_id = uuid4().hex
        self._stopping.discard(record.id)
        self._stops.discard(record.id)
        try:
            await self.save(record)
        except BaseException:
            self._settle_unstarted(record)
            raise
        if not self._open or record.user_stopped:
            self._settle_unstarted(record)
            return
        self._schedule_member(record)

    async def _hand_over(self, record: DelegationTask) -> None:
        """Pass a settled member's report to the lead, and the board items it claimed back if it didn't succeed."""
        if record.outcome != 'ok':
            await self._release_claims(record)
        await self._report_to_lead(record)

    async def _wake_if_messaged(self, record: DelegationTask) -> None:
        """Wake a member that succeeded, if messages arrived too late for its turn to read.

        A member the user stopped, or whose turn failed, keeps them for its next allowed wake, so a failure
        can't loop.
        """
        if (
            self._open
            and record.outcome == 'ok'
            and record.resumable
            and not record.user_stopped
            and any(message.id not in self._queued_messages for message in record.inbox)
        ):
            await self._wake_member(record)

    async def _release_claims(self, record: DelegationTask) -> None:
        """Hand the board items a stopped, failed or interrupted member claimed back to `pending`."""
        board = self.claims(record.conversation_id) if self.claims is not None else None
        assert record.name is not None, 'Only team members claim items.'
        if board is None:
            return
        try:
            for item in await board.get_owned_items():
                if item.owner == record.name and item.status == TaskStatus.in_progress:
                    await board.release(item.id, owner=record.name, status=TaskStatus.pending)
        except Exception:
            logging.getLogger(__name__).exception('Releasing the claims of team member %s failed', record.name)

    async def _report_to_lead(self, record: DelegationTask) -> None:
        """Move a settled member's report into the lead's inbox, so a later wake can't overwrite it."""
        await self._append_lead(record.conversation_id, InboxMessage(part=record.report()))
        record.delivered = True
        await self.save(record)

    async def _append_lead(self, conversation_id: str, message: InboxMessage) -> None:
        self._lead_inboxes.setdefault(conversation_id, []).append(message)
        await self._save_lead(conversation_id)
        self._deliver_lead(conversation_id)

    async def _save_lead(self, conversation_id: str) -> None:
        """Atomically replace a lead's saved inbox, keyed by a hash so any conversation ID is a safe file name."""
        if self.directory is None:
            return
        name = _inbox_name(conversation_id)
        folder = anyio.Path(self.directory / 'inbox')
        async with self._save_locks.setdefault(f'inbox-{name}', anyio.Lock()):
            await folder.mkdir(parents=True, exist_ok=True)
            saved = _LeadInbox(conversation_id=conversation_id, messages=self.lead_inbox(conversation_id))
            temporary = folder / f'{name}.tmp'
            await temporary.write_bytes(_LEAD_INBOX_ADAPTER.dump_json(saved))
            await temporary.replace(folder / f'{name}.json')

    def _lead_active(self, conversation_id: str) -> bool:
        return (conversation_id, None) in self._receivers

    def _deliver_lead(self, conversation_id: str) -> None:
        """Queue the lead's inbox on its active run, or have `lead_waker` start one."""
        receivers = self._receivers.get((conversation_id, None))
        if receivers is not None:
            self._queue_messages(conversation_id, None, self._lead_inboxes.get(conversation_id, ()), receivers[-1])
        if any(message.id not in self._queued_messages for message in self._lead_inboxes.get(conversation_id, ())):
            self._schedule_lead_wake(conversation_id)

    def _deliver_member(self, record: DelegationTask) -> None:
        """Queue a team member's inbox on its active run, if one is reading it."""
        callbacks = self._messaging.get(record.id)
        if callbacks is not None:
            self._queue_messages(record.conversation_id, record.id, record.inbox, callbacks[-1])

    def _queue_messages(
        self,
        conversation_id: str,
        task_id: str | None,
        messages: Sequence[InboxMessage],
        enqueue: Callable[[UserPromptPart], str | None],
    ) -> None:
        for message in messages:
            if message.id in self._queued_messages:
                continue
            enqueue_id = enqueue(message.part)
            if enqueue_id is None:
                # The run takes no more messages; the rest wait with this one.
                return
            self._queued_messages.add(message.id)
            self._message_acks[enqueue_id] = (conversation_id, task_id, message.id)

    async def _remove_message(self, conversation_id: str, task_id: str | None, message_id: str) -> None:
        """Drop a message from its inbox once core confirmed it entered its recipient's history."""
        self._queued_messages.discard(message_id)
        if task_id is None:
            inbox = self._lead_inboxes.get(conversation_id, [])
            self._lead_inboxes[conversation_id] = [m for m in inbox if m.id != message_id]
            await self._save_lead(conversation_id)
        else:
            record = self.records[task_id]
            record.inbox = [m for m in record.inbox if m.id != message_id]
            await self.save(record)

    def _members_working(self, conversation_id: str) -> bool:
        return any(member.status == 'running' for member in self.members(conversation_id))

    def _schedule_lead_wake(self, conversation_id: str) -> None:
        """Gather a lead's messages for `lead_wake_delay`, or until every member settled, into one lead run."""
        if self.lead_waker is None or not self._open or self._lead_active(conversation_id):
            return
        due = self._lead_wakes.get(conversation_id)
        if due is None:
            due = self._lead_wakes[conversation_id] = anyio.Event()
            task = asyncio.create_task(self._wake_lead(conversation_id, due), name=f'lead-wake-{conversation_id}')
            self._lead_tasks.add(task)
            task.add_done_callback(self._lead_tasks.discard)
        if not self._members_working(conversation_id):
            due.set()

    async def _wake_lead(self, conversation_id: str, due: anyio.Event) -> None:
        with anyio.move_on_after(self.lead_wake_delay):
            await due.wait()
        # Messages arriving from here on gather for the next wake, or reach the run this one starts.
        del self._lead_wakes[conversation_id]
        if self._lead_active(conversation_id) or not self._lead_inboxes.get(conversation_id):
            return
        assert self.lead_waker is not None, 'Only scheduled with a `lead_waker`.'
        try:
            await self.lead_waker(conversation_id)
        except Exception:
            logging.getLogger(__name__).exception('Waking the lead of %s failed', conversation_id)

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
        message: Callable[[UserPromptPart], str | None] | None = None,
    ) -> Generator[None]:
        """Keep a queue attached only for the lifetime of its owning run.

        `steer` queues a user message on a child's own run, for `DelegationTasks.steer`. `message` queues a team
        member's inbox on its own run, for `DelegationTasks.send`, at its next model request.
        """
        key = (conversation_id, parent_id)
        # What this run queued, which it alone can still deliver.
        queued: set[str] = set()

        def receive(part: UserPromptPart) -> str | None:
            enqueue_id = enqueue(part)
            if enqueue_id is not None:
                queued.add(enqueue_id)
            return enqueue_id

        def deliver(part: UserPromptPart) -> str | None:
            assert message is not None, 'Only registered with a `message` callback.'
            enqueue_id = message(part)
            if enqueue_id is not None:
                queued.add(enqueue_id)
            return enqueue_id

        self._receivers.setdefault(key, []).append(receive)
        if steer is not None and parent_id is not None:
            self._steering.setdefault(parent_id, []).append(steer)
            for text in self._unsteered.pop(parent_id, ()):
                steer(text)
        if message is not None and parent_id in self.records:
            self._messaging.setdefault(parent_id, []).append(deliver)
        self._deliver_all(conversation_id, parent_id)
        try:
            yield
        finally:
            _unregister(self._receivers, key, receive)
            if steer is not None and parent_id is not None:
                _unregister(self._steering, parent_id, steer)
            if message is not None and parent_id in self.records:
                _unregister(self._messaging, parent_id, deliver)
            self._release_undelivered(queued)
            # Another run of the same parent may still be receiving: it gets what this one could not deliver.
            self._deliver_all(conversation_id, parent_id)

    def _release_undelivered(self, queued: set[str]) -> None:
        """Forget what an ended run queued and never delivered, so the next run queues it again."""
        self._queued = {
            enqueue_id: identity for enqueue_id, identity in self._queued.items() if enqueue_id not in queued
        }
        # A question queued on a run that ended before delivering it is queued again on the next one.
        for enqueue_id in queued & self._queued_questions.keys():
            task_id = self._queued_questions.pop(enqueue_id)
            if (question := self._questions.get(task_id)) is not None:
                question.queued = False
        # So is a message. Its acknowledgement is kept, in case core confirms it after all.
        for enqueue_id in queued & self._message_acks.keys():
            self._queued_messages.discard(self._message_acks[enqueue_id][2])

    def _deliver_all(self, conversation_id: str, parent_id: str | None) -> None:
        """Queue what the runs receiving for `parent_id` take: reports, questions, and the lead's or a member's inbox."""
        self.queue_reports(conversation_id=conversation_id, parent_id=parent_id)
        if parent_id is None:
            self._deliver_lead(conversation_id)
        elif parent_id in self._messaging:
            self._deliver_member(self.records[parent_id])

    async def acknowledge(self, enqueue_id: str) -> None:
        """Persist delivery only after core confirms it entered message history."""
        self._queued_questions.pop(enqueue_id, None)
        if (delivered := self._message_acks.pop(enqueue_id, None)) is not None:
            await self._remove_message(*delivered)
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
            # A team member's message isn't: it stays in the member's inbox for its next turn.
            message=lambda part: enqueue_while_open(ctx, part, priority='asap'),
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
