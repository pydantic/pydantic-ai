"""CLAI's presentation and specialist roster for harness-owned child tasks."""

import copy
import time
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from pathlib import Path

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
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import FunctionToolCallEvent, ModelResponse
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
from pydantic_clai2.runtime.sandbox_calls import DelegationToolCallEvent
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.status import Status
from pydantic_clai2.ui.rendering.tool_output import terminal_text


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


class Tasks:
    """One shell's task display. Execution and histories belong to the harness owner."""

    def __init__(
        self,
        *,
        console: Console,
        conversation_id: Callable[[], str],
        directory: Path | None,
        step_store: StepStore | None = None,
    ) -> None:
        self.presentation = TaskPresentation()
        self.console = console
        self.conversation_id = conversation_id
        self.sink: Callable[[AgentStreamEvent], Awaitable[None]] | None = None
        self.wake: Callable[[], None] | None = None
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

    def records(self) -> tuple[DelegationTask, ...]:
        """Only children of the active conversation, including its descendants."""
        return tuple(r for r in self.owner.records.values() if r.conversation_id == self.conversation_id())

    async def observe(self, update: DelegationTaskEvent) -> None:
        record, event = update.task, update.event
        progress = self.progress.setdefault(
            record.id,
            Status(agent=f'{_display_name(record.agent_name)} [{record.id[:8]}]', model='', activity='starting'),
        )
        if update.model_name is not None:
            progress.model, progress.context_window = update.model_name, update.context_window
        for message in reversed(record.messages):
            if isinstance(message, ModelResponse) and message.usage.input_tokens:
                progress.context_tokens = message.usage.total_tokens
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
        if (
            record.status == 'finished'
            and record.background
            and record.parent_id is None
            and not record.delivered
            and record.conversation_id == self.conversation_id()
            and self.wake is not None
        ):
            self.wake()
        if not isinstance(event, (DelegationStartEvent, DelegationEndEvent)):
            return
        if record.conversation_id != self.conversation_id():
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
            if current.background:
                return False
            current = self.owner.records.get(current.parent_id) if current.parent_id is not None else None
        return True

    def _foreground_tasks(self) -> list[DelegationTask]:
        return [r for r in self.records() if r.status == 'running' and not r.background and r.parent_id is None]

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
