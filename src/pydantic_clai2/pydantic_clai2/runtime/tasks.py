"""CLAI's presentation and specialist roster for harness-owned child tasks."""

import asyncio
import copy
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable, Coroutine, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import anyio
from rich.console import Console
from rich.text import Text

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunContext,
    TextPart,
    TextPartDelta,
    _utils,
)
from pydantic_ai._context_size import context_tokens
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import FunctionToolCallEvent, ModelMessage, ModelResponse, ToolReturnPart
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.step_persistence import StepStore
from pydantic_ai_harness.subagents import (
    DelegationEndEvent,
    DelegationStartEvent,
    DelegationTask,
    DelegationTaskEvent,
    DelegationTasks,
    SubAgent,
    SubAgents,
)
from pydantic_clai2.plugins import TaskEvent, TaskInfo
from pydantic_clai2.runtime.sandbox_calls import DelegationToolCallEvent
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.status import Status
from pydantic_clai2.ui.rendering.tool_output import terminal_text

PromptTask = Callable[[DelegationTask, str], Coroutine[object, object, object]]
"""Runs a finished task again with the user's message, apart from the conversation, for `Tasks.prompt`."""


def specialists() -> dict[str, SubAgent[object]]:
    """Read-only agents have neither parent plugin tools nor an executable-code escape."""
    return {
        name: SubAgent(
            Agent(
                deps_type=object,
                name=name,
                description=description,
                instructions=f'{description} Return evidence with paths and line numbers. Do not modify files.',
                capabilities=[FileSystem(read_only=True)],
            ),
            read_only=True,
        )
        for name, description in (
            ('Explore', 'Search and explain the codebase with read-only file tools.'),
            ('Plan', 'Research an implementation plan, tradeoffs, and verification steps with read-only file tools.'),
        )
    }


def task_tree(records: Sequence[DelegationTask]) -> list[tuple[int, DelegationTask]]:
    """Parents before descendants, retaining stable creation order."""
    ordered = sorted(records, key=lambda record: record.started_at)
    ids = {record.id for record in records}
    result: list[tuple[int, DelegationTask]] = []

    def visit(record: DelegationTask, depth: int) -> None:
        result.append((depth, record))
        for child in ordered:
            if child.parent_id == record.id:
                visit(child, depth + 1)

    for record in ordered:
        if record.parent_id not in ids:
            visit(record, 0)
    return result


def task_row(event: AgentStreamEvent) -> Text | None:
    """Compact lifecycle rows, with child output reserved for the inspector."""
    if isinstance(event, DelegationStartEvent):
        name = _display_name(event.agent_name)
        prompt = ' '.join(terminal_text(event.task).split())
        # Styled like a tool-call header: muted marker, accent name, then dim details on one line.
        row = Text('● ', style=theme.color(theme.MUTED), no_wrap=True, overflow='ellipsis')
        row.append(name, style=theme.color(theme.ACCENT))
        row.append(f' [{(event.task_id or "")[:8]}] {prompt}', style=theme.color(theme.MUTED))
        return row
    if isinstance(event, DelegationEndEvent) and event.outcome != 'ok':
        return Text(
            f'  └ {event.outcome} · {event.duration_seconds:.1f}s · /tasks to inspect',
            style=theme.color(theme.WARNING),
        )
    return None


def resume_prompt(record: DelegationTask) -> str:
    """What asks the agent to resume a task, once the user has allowed it to."""
    return (
        f'Resume task {record.id} using delegate_task with agent_name={record.agent_name!r}, '
        f'resume={record.id!r}. Continue from its saved history and report the result.'
    )


class TaskRouter:
    """The process's one delegated-task owner, and the routing of its events to each live session's `Tasks`.

    Every task's progress is kept here, so a session that switches back to a conversation sees its tasks'
    progress. Plugin events, wake-ups and lifecycle rows go only to the session holding the task's conversation.
    """

    def __init__(self, *, directory: Path | None, step_store: StepStore | None = None) -> None:
        """Save tasks in `directory`, and capture their steps in `step_store`; neither is kept without one."""
        self.views: list[Tasks] = []
        self.started: dict[str, Tasks] = {}
        """The session each task last started or resumed in, whose plugins it runs with after that session moves on."""
        self.progress: dict[str, Status] = {}
        self.partial: dict[str, str] = {}
        self.partial_truncated: set[str] = set()
        self.owner = DelegationTasks(
            observer=self.observe,
            directory=directory,
            agents=specialists(),
            aliases={'general-purpose': 'self'},
            one_shot=frozenset({'Explore', 'Plan'}),
            step_store=step_store,
            instructions=(
                'Use Explore for code discovery and Plan for design research. Both are read-only and one-shot. '
                'General-purpose work uses self (alias general-purpose) and inherits your tools and guardrails.'
            ),
        )
        self._users = 0
        self._lock = anyio.Lock()
        self._stack = AsyncExitStack()

    def view(self, conversation_id: str) -> 'Tasks | None':
        """The live session's `Tasks` holding `conversation_id`, if one does."""
        return next((view for view in self.views if view.conversation_id() == conversation_id), None)

    @asynccontextmanager
    async def opened(self, *, view: 'Tasks') -> AsyncGenerator[DelegationTasks]:
        """Open the owner for the first session and close it after the last, cancelling the tasks still running.

        A session's `view` that leaves before the others cancels its own tasks, which run with its plugins.
        """
        async with self._lock:
            if not self._users:
                await self._stack.enter_async_context(self.owner.opened())
            self._users += 1
        try:
            yield self.owner
        finally:
            with anyio.CancelScope(shield=True):
                await view.stop_prompts()
                async with self._lock:
                    self._users -= 1
                    last = not self._users
                    if last:
                        await self._stack.aclose()
                if not last:
                    for record in [record for record in self.owner.records.values() if view.runs(record)]:
                        await self.owner.cancel(record.id, user=False)

    async def observe(self, update: DelegationTaskEvent) -> None:
        """Record the task's progress, then tell the session holding its conversation."""
        record, event = update.task, update.event
        progress = self.progress.setdefault(
            record.id,
            Status(agent=f'{_display_name(record.agent_name)} [{record.id[:8]}]', model='', activity='starting'),
        )
        if update.model_name is not None:
            progress.model, progress.context_window = update.model_name, update.context_window
        for message in reversed(record.messages):
            if isinstance(message, ModelResponse) and message.usage.input_tokens:
                progress.context_tokens = context_tokens(message)
                break
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            self.partial[record.id] = event.part.content
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            self.partial[record.id] = self.partial.get(record.id, '') + event.delta.content_delta
        elif isinstance(event, (PartEndEvent, DelegationEndEvent)) or record.status == 'finished':
            self.partial.pop(record.id, None)
            self.partial_truncated.discard(record.id)
        if len(self.partial.get(record.id, '')) > 65536:
            self.partial[record.id] = self.partial[record.id][-65536:]
            self.partial_truncated.add(record.id)
        if event is not None:
            progress.observe(event)
        view = self.view(record.conversation_id)
        # A child runs with its parent's plugins; a resumed task, with those of the session that resumed it.
        if record.parent_id is not None and (parent := self.started.get(record.parent_id)) is not None:
            self.started.setdefault(record.id, parent)
        elif view is not None and isinstance(event, DelegationStartEvent):
            self.started[record.id] = view
        elif view is not None:
            self.started.setdefault(record.id, view)
        if record.status == 'finished':
            # Resuming it starts it again, and records its session anew.
            self.started.pop(record.id, None)
        if view is not None:
            await view.deliver(update)


_CLOSED_BEFORE_START = 'This session closed before the task could start.'


class Tasks:
    """One session's view of the process's tasks, and the `TaskCatalog` plugins reach as `host.tasks`.

    Execution and histories belong to the harness owner, which `router` shares between sessions.
    """

    def __init__(
        self,
        *,
        console: Console,
        conversation_id: Callable[[], str],
        router: TaskRouter | None = None,
        directory: Path | None = None,
        step_store: StepStore | None = None,
        fire: Callable[[TaskEvent], Awaitable[None]] | None = None,
        listening: Callable[[], bool] = lambda: True,
        send_input: Callable[[str], None] | None = None,
        prompt_task: PromptTask | None = None,
    ) -> None:
        """`fire` tells plugins about the current conversation's tasks, while `listening` says one handles them.

        Without a `router`, it gets one of its own over `directory` and `step_store`. `send_input` enters
        `resume`'s prompt, and `prompt_task` runs a task again with the user's message for `prompt`.
        """
        self.presentation = TaskPresentation()
        self.console = console
        self.conversation_id = conversation_id
        self.router = router or TaskRouter(directory=directory, step_store=step_store)
        self.router.views.append(self)
        self.fire = fire
        self.listening = listening
        self.send_input = send_input
        self.prompt_task = prompt_task
        self._prompts: dict[str, asyncio.Task[Exception | None]] = {}
        """The runs `prompt` started, by task ID: nothing else waits for these tasks."""
        self._starting: dict[str, asyncio.Event] = {}
        self._closes = 0
        """How many times `stop_prompts` ran, so a prompt that was awaiting `allow_resume` meanwhile can tell."""
        self.sink: Callable[[AgentStreamEvent], Awaitable[None]] | None = None
        self.wake: Callable[[], None] | None = None

    @property
    def owner(self) -> DelegationTasks:
        return self.router.owner

    @property
    def progress(self) -> dict[str, Status]:
        return self.router.progress

    @property
    def partial(self) -> dict[str, str]:
        return self.router.partial

    @property
    def partial_truncated(self) -> set[str]:
        return self.router.partial_truncated

    def records(self) -> tuple[DelegationTask, ...]:
        """Only children of the active conversation, including its descendants."""
        return tuple(r for r in self.owner.records.values() if r.conversation_id == self.conversation_id())

    def runs(self, record: DelegationTask) -> bool:
        """Whether `record` runs for this session: one it started, or one of its conversation no session started."""
        starter = self.router.started.get(record.id)
        if starter is None:
            return record.status == 'running' and record.conversation_id == self.conversation_id()
        return record.status == 'running' and starter is self

    def running(self) -> bool:
        """Whether a task of this session still runs."""
        return any(self.runs(record) for record in self.owner.records.values())

    async def observe(self, update: DelegationTaskEvent) -> None:
        """What the owner tells the router, which reaches this view when it holds the task's conversation."""
        await self.router.observe(update)

    async def deliver(self, update: DelegationTaskEvent) -> None:
        """Tell plugins, wake the prompt, and show lifecycle rows for a task of the active conversation."""
        record, event = update.task, update.event
        if record.status == 'running' and (starting := self._starting.pop(record.id, None)) is not None:
            starting.set()
        # It runs for every event of the child's stream, so build nothing that no plugin reads.
        if self.fire is not None and self.listening():
            await self.fire(TaskEvent(task=self.info(record), event=event))
        if (
            record.status == 'finished'
            and record.background
            and record.parent_id is None
            and not record.delivered
            and self.wake is not None
        ):
            self.wake()
        if not isinstance(event, (DelegationStartEvent, DelegationEndEvent)):
            return
        if self.sink is not None:
            await self.sink(event)
        elif (row := task_row(event)) is not None:
            self.console.print(row)

    def focused(self) -> Status | None:
        """The newest subagent the main run is waiting on, whose figures the status row shows until it settles.

        A background task, or one below a background ancestor, leaves the row to the main run.
        """
        waited = [record for record in self.records() if record.status == 'running' and self._waited_on(record)]
        newest = max(waited, key=lambda record: record.started_at, default=None)
        return self.progress.get(newest.id) if newest is not None else None

    def _waited_on(self, record: DelegationTask) -> bool:
        current: DelegationTask | None = record
        while current is not None:
            if current.background or current.id in self._prompts:
                return False
            current = self.owner.records.get(current.parent_id) if current.parent_id is not None else None
        return True

    def _foreground_tasks(self) -> list[DelegationTask]:
        """The tasks the main run waits for, which leaves out those the user prompted."""
        return [
            r
            for r in self.records()
            if r.status == 'running' and not r.background and r.parent_id is None and r.id not in self._prompts
        ]

    def promote(self) -> str:
        """Ctrl+B backgrounds all foreground siblings directly delegated by the main run."""
        running = self._foreground_tasks()
        if not running:
            return 'No foreground tasks. /tasks opens the task inspector.'
        for record in running:
            try:
                self.owner.background(record.id)
            except ValueError as exc:
                return str(exc)
        return f'{len(running)} task(s) moved to background. /tasks to inspect.'

    def rows(self, glyph: str) -> tuple[str, ...]:
        now = time.time()
        rows: list[str] = []
        recent = False
        records = self.records()
        for depth, record in task_tree(records):
            if record.status == 'finished':
                if record.finished_at is None or now - record.finished_at > 30:
                    continue
                recent = True
                if record.outcome == 'ok':
                    continue
            descendants = sum(self._descends(child, record.id) for child in records)
            activity = self.progress.get(record.id, Status(activity='starting')).activity
            state = activity.removeprefix('running: ').removeprefix('tool: ')
            state = state if record.status == 'running' else record.outcome
            elapsed = (record.finished_at or now) - record.started_at
            name = _display_name(record.agent_name)
            mode = 'background' if record.background else 'foreground'
            suffix = f' · {descendants} descendants' if descendants else ''
            state_color = theme.ACCENT if record.status == 'running' else theme.ERROR
            if record.outcome in ('cancelled', 'interrupted'):
                state_color = theme.WARNING
            marker = glyph if record.status == 'running' else '!'
            muted = theme.sgr(theme.MUTED)
            rows.append(
                f'{"  " * depth}{theme.sgr(state_color)}{marker} {theme.sgr(theme.INFO)}{name} '
                f'{muted}[{record.id[:8]}] {elapsed:.0f}s · {mode} · '
                f'{theme.sgr(state_color)}{terminal_text(state or "")}{muted}{suffix}\x1b[0m'
            )
        if rows or recent:
            hint = f'{theme.sgr(theme.ACCENT)}/tasks{theme.sgr(theme.MUTED)} inspect'
            if any(record.backgroundable for record in self._foreground_tasks()):
                hint += f' · {theme.sgr(theme.ACCENT)}Ctrl+B{theme.sgr(theme.MUTED)} background'
            rows.append(hint + '\x1b[0m')
        return tuple(rows)

    def _descends(self, record: DelegationTask, ancestor: str) -> bool:
        parent = record.parent_id
        while parent is not None:
            if parent == ancestor:
                return True
            above = self.owner.records.get(parent)
            parent = above.parent_id if above is not None else None
        return False

    def snapshots(self) -> tuple[DelegationTask, ...]:
        """Display-only copies include uncommitted streamed text without altering replay history."""
        records = copy.deepcopy(self.records())
        for record in records:
            if text := self.partial.get(record.id):
                if record.id in self.partial_truncated:
                    text = '[Live preview tail; full text is available when this response settles.]\n' + text
                record.messages.append(ModelResponse(parts=[TextPart(text)]))
        return records

    def info(self, record: DelegationTask) -> TaskInfo:
        """A task as plugins see it."""
        progress = self.progress.get(record.id)
        usage = RunUsage()
        for message in record.messages:
            if isinstance(message, ModelResponse):
                usage.incr(message.usage)
                usage.requests += 1
            else:
                # As a run counts them: the calls that returned, not those retried or still waiting.
                usage.tool_calls += sum(
                    isinstance(part, ToolReturnPart) and part.outcome != 'retried' for part in message.parts
                )
        return TaskInfo(
            id=record.id,
            parent_id=record.parent_id,
            agent=_display_name(record.agent_name),
            model=progress.model if progress is not None and progress.model else record.model,
            prompt=record.prompt,
            status=record.status,
            outcome=record.outcome,
            started_at=datetime.fromtimestamp(record.started_at, UTC),
            finished_at=None if record.finished_at is None else datetime.fromtimestamp(record.finished_at, UTC),
            background=record.background,
            backgroundable=record.status == 'running'
            and not record.background
            and record.backgroundable
            and record.id not in self._prompts,
            resumable=record.resumable,
            activity=progress.activity if progress is not None else 'starting',
            usage=usage,
        )

    def listing(self) -> list[TaskInfo]:
        return [self.info(record) for _, record in task_tree(self.records())]

    def history(self, task_id: str) -> list[ModelMessage]:
        return self.owner.history(self._record(task_id).id)

    async def stop(self, task_id: str) -> None:
        record = self._record(task_id)
        await self.owner.cancel(record.id)
        await self.owner.save(record)

    def background(self, task_id: str) -> None:
        record = self._record(task_id)
        if record.id in self._prompts:
            raise ValueError('Nothing waits for a task you prompted; it already runs on its own.')
        self.owner.background(record.id)

    async def resume(self, task_id: str) -> None:
        record = self._record(task_id)
        if self.send_input is None:
            raise RuntimeError('This CLAI has no prompt to ask the agent to resume a task at.')
        stopped = record.user_stopped
        await self.owner.allow_resume(record.id)
        try:
            self.send_input(resume_prompt(record))
        except RuntimeError:
            # The shell's line-mode prompt takes no input: the user's stop still holds.
            record.user_stopped = stopped
            await self.owner.save(record)
            raise

    async def prompt(self, task_id: str, text: str) -> None:
        """Run a finished task again with `text` as the user's message, on its own; see `TaskCatalog.prompt`."""
        record = self._record(task_id)
        text = text.strip()
        if not text:
            raise ValueError('Type a message for the task.')
        if self.prompt_task is None:
            raise RuntimeError('This CLAI cannot prompt a task on its own.')
        # It reads `finished` until its run starts it, so `allow_resume` cannot tell a second prompt meanwhile.
        if record.id in self._starting:
            raise ValueError('This task is starting from a prompt already; wait for it to run.')
        stopped, closes = record.user_stopped, self._closes
        starting = self._starting[record.id] = asyncio.Event()
        try:
            # Prompting it is the user's leave to resume it, as asking the agent to is.
            await self.owner.allow_resume(record.id)
            if self._closes != closes:
                # `stop_prompts` ran meanwhile and could not see a run that had not started yet.
                raise RuntimeError(_CLOSED_BEFORE_START)
        except BaseException:
            # Another run of the session may have started the task meanwhile, which `deliver` saw.
            if self._starting.get(record.id) is starting:
                del self._starting[record.id]
            if not starting.is_set():
                await self._restore_stop(record, stopped)
            raise
        # From here the run settles the prompt, also once this caller is cancelled, as a dropped request is.
        run = self._prompts[record.id] = asyncio.create_task(
            self._run_prompt(self.prompt_task, record, text, starting, stopped=stopped),
            name=f'clai2-task-prompt-{record.id}',
        )
        started = asyncio.create_task(starting.wait())
        try:
            await asyncio.wait((run, started), return_when=asyncio.FIRST_COMPLETED)
        finally:
            started.cancel()
            await asyncio.gather(started, return_exceptions=True)
        if not starting.is_set():
            if run.cancelled():
                raise RuntimeError(_CLOSED_BEFORE_START)
            if (error := run.result()) is not None:
                raise error

    async def _run_prompt(
        self,
        prompt_task: PromptTask,
        record: DelegationTask,
        text: str,
        starting: asyncio.Event,
        *,
        stopped: bool,
    ) -> Exception | None:
        """Prompt `record`, then forget this run; the failure that kept its task from starting, for `prompt`.

        `stopped` is the user's stop from before `allow_resume` lifted it, which a task that never ran keeps.
        A failure after the task started has no caller left to hear it, so it is logged.
        """
        try:
            await prompt_task(record, text)
        except asyncio.CancelledError:
            # `stop_prompts` closing the session; a task that never ran keeps the user's stop.
            if not starting.is_set():
                await self._restore_stop(record, stopped)
            raise
        except Exception as error:
            if not starting.is_set():
                await self._restore_stop(record, stopped)
                return error
            logging.getLogger(__name__).exception('Prompting task %s failed', record.id)
        finally:
            if self._starting.get(record.id) is starting:
                del self._starting[record.id]
            # The task may have been prompted again meanwhile, by a run that is still going.
            if self._prompts.get(record.id) is asyncio.current_task():
                del self._prompts[record.id]
        return None

    async def _restore_stop(self, record: DelegationTask, stopped: bool) -> None:
        """Put back the stop `allow_resume` lifted for a task that never ran, keeping one the user sent since."""
        if stopped and not record.user_stopped:
            record.user_stopped = True
            await self.owner.save(record)

    async def stop_prompts(self) -> None:
        """Stop the runs `prompt` started, which stops their tasks, before this session closes."""
        self._closes += 1
        runs = list(self._prompts.values())
        for run in runs:
            run.cancel()
        await asyncio.gather(*runs, return_exceptions=True)

    def steer(self, task_id: str, text: str) -> bool:
        return self.owner.steer(self._record(task_id).id, text)

    def _record(self, task_id: str) -> DelegationTask:
        record = self.owner.records.get(task_id)
        if record is None or record.conversation_id != self.conversation_id():
            raise LookupError(f'No task {task_id!r} in this conversation.')
        return record

    def resolve(self, prefix: str) -> DelegationTask:
        matches = [record for record in self.records() if record.id.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f'Task ID {prefix!r} is unknown or ambiguous. Use /tasks.')
        return matches[0]


def _display_name(agent_name: str) -> str:
    return 'general-purpose' if agent_name == 'self' else agent_name


class TaskPresentation(AbstractCapability[object]):
    """Classify calls by their capability owner, not their configurable tool name."""

    async def wrap_run_event_stream(
        self,
        ctx: RunContext[object],
        *,
        stream: AsyncIterable[AgentStreamEvent],
    ) -> AsyncIterable[AgentStreamEvent]:
        try:
            async for event in stream:
                if isinstance(event, FunctionToolCallEvent) and event.args_valid is not False:
                    definition = ctx.tools.get(event.part.tool_name)
                    owner = ctx.capabilities.get(definition.capability_id or '') if definition is not None else None
                    if isinstance(owner, SubAgents):
                        event = DelegationToolCallEvent(event.part, args_valid=event.args_valid)
                yield event
        finally:
            await _utils.aclose_if_supported(stream)
