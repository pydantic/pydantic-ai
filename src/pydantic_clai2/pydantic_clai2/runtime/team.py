"""The team a session's lead runs, as the shell, its plugins and the web UI see it.

A team's events come from three places: the lead's run emits those its tools cause, each member's run
emits its own (which reach the task owner's observer), and the harness `Team` tells its `observer` about
those the owner or the user cause. `Teams` merges them into one ordered feed per conversation, keeps each
conversation's latest `TeamSnapshot` for the roster panel (which repaints ten times a second) and the web
UI, and carries out the user's own actions on the team. `SessionTeam` is one session's view of it, the
`TeamCatalog` plugins reach as `host.team`.
"""

import asyncio
import logging
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import get_args

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness.planning import BoardChange, OwnedPlanItem, TaskStatus
from pydantic_ai_harness.subagents import DelegationTask, DelegationTaskEvent, DelegationTasks, DeliveryStatus
from pydantic_ai_harness.teams import (
    USER,
    BoardItemChangedEvent,
    BoardResolver,
    Team,
    TeamEvent,
    TeammateSpawnedEvent,
    TeammateStateEvent,
    TeamMember,
    TeamSnapshot,
)
from pydantic_clai2.plugins import TeamProgress
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.status import Status
from pydantic_clai2.ui.rendering.tool_output import terminal_text

TEAM_EVENTS: tuple[type[TeamEvent], ...] = get_args(TeamEvent)
"""The events of the team feed: every kind harness `TeamEvent` names, so a new kind joins the feed unchanged."""

TeamDelivery = Callable[[str, TeamProgress], Awaitable[None]]
"""Hands a conversation's team change to the session holding that conversation."""

_CONVERSATION: ContextVar[str | None] = ContextVar('clai2_team_conversation', default=None)
"""The conversation whose team a `Team` method called from here acts on, as its `observer` cannot tell."""

_log = logging.getLogger(__name__)


class Teams:
    """Every conversation's team in this process: one `Team` over the task owner, its feed, and its cache."""

    def __init__(self, owner: DelegationTasks, *, board: BoardResolver, deliver: TeamDelivery) -> None:
        """`deliver` hands each change, in order, to the session holding its conversation, if any."""
        self.owner = owner
        self.team = Team[object](owner, board=board, observer=self._observed)
        """Every conversation's team, bound to a lead's turns with `team.enabled`, and to its members' turns."""
        self.stream = TeamStream(self)
        """Added to a lead's turn, to feed the team events its tools emit."""
        self._deliver = deliver
        self._snapshots: dict[str, TeamSnapshot] = {}
        self._pending: deque[tuple[str, TeamEvent | None]] = deque()
        self._draining = False
        self._refreshing: dict[str, asyncio.Task[None]] = {}

    def claims(self, conversation_id: str) -> '_Claims':
        """The board the owner hands a stopped or failed member's claims back to, telling the feed."""
        return _Claims(self, conversation_id)

    @contextmanager
    def about(self, conversation_id: str) -> Generator[None]:
        """Tell the `Team`'s observer that the changes made within are about `conversation_id`."""
        token = _CONVERSATION.set(conversation_id)
        try:
            yield
        finally:
            _CONVERSATION.reset(token)

    async def _observed(self, event: TeamEvent) -> None:
        """The `Team`'s observer: a change the owner or the user made, about the conversation acted on."""
        conversation_id = _CONVERSATION.get() or self._conversation_of(event)
        if conversation_id is None:
            _log.warning('A team event of no known conversation was dropped: %s', type(event).__name__)
            return
        await self.publish(conversation_id, event)

    def _conversation_of(self, event: TeamEvent) -> str | None:
        if isinstance(event, TeammateSpawnedEvent | TeammateStateEvent):
            record = self.owner.records.get(event.member.task_id)
            return None if record is None else record.conversation_id
        return None

    async def observe(self, update: DelegationTaskEvent) -> None:
        """What the owner tells the shell about a member: the team events of its own run, then its state."""
        record = update.task
        if record.name is None:
            return
        if isinstance(update.event, TEAM_EVENTS):
            await self.publish(record.conversation_id, update.event)
        with self.about(record.conversation_id):
            await self.team.observe(update)

    async def publish(self, conversation_id: str, event: TeamEvent | None) -> None:
        """Add a change to the feed, which refreshes the conversation's snapshot and delivers it, in order.

        `None` delivers the team read again with no change, as when it is first read. A change published
        while the feed delivers another, such as by a plugin acting on the team from `on_team_progress`,
        waits its turn instead of overtaking it.
        """
        self._pending.append((conversation_id, event))
        if self._draining:
            return
        self._draining = True
        try:
            while self._pending:
                conversation, change = self._pending.popleft()
                try:
                    snapshot = await self.refresh(conversation)
                    await self._deliver(conversation, TeamProgress(event=change, team=snapshot))
                except Exception:  # noqa: BLE001 -- logged; see below.
                    # The feed only shows the team: a failure here must not fail the run that caused the change.
                    _log.exception('Delivering a team change failed')
        finally:
            self._draining = False

    async def refresh(self, conversation_id: str) -> TeamSnapshot:
        """Read a conversation's team again, and keep it for `snapshot`."""
        snapshot = self._snapshots[conversation_id] = await self.team.snapshot(conversation_id)
        return snapshot

    def snapshot(self, conversation_id: str) -> TeamSnapshot:
        """A conversation's team as last read, without waiting.

        The first time, before anything was read, it is empty, and the team is read in the background and
        delivered as a `TeamProgress` with no `event`.
        """
        if (snapshot := self._snapshots.get(conversation_id)) is not None:
            return snapshot
        self._refresh_soon(conversation_id)
        return TeamSnapshot(conversation_id=conversation_id, members=[], board=[], messages=[])

    def _refresh_soon(self, conversation_id: str) -> None:
        if conversation_id in self._refreshing:
            return
        try:
            task = asyncio.get_running_loop().create_task(self.publish(conversation_id, None))
        except RuntimeError:
            return  # No loop runs: the next read from one starts it.
        self._refreshing[conversation_id] = task
        task.add_done_callback(lambda _: self._refreshing.pop(conversation_id, None))

    def record(self, conversation_id: str, name: str) -> DelegationTask:
        """The team member called `name` in a conversation."""
        record = next((member for member in self.owner.members(conversation_id) if member.name == name), None)
        if record is None:
            raise LookupError(f'No teammate {name!r} in this conversation.')
        return record

    async def spawn(self, conversation_id: str, *, name: str, prompt: str, model: str | None = None) -> TeamMember:
        """Start a teammate for the user; see `Team.spawn`."""
        with self.about(conversation_id):
            return await self.team.spawn(conversation_id, name=name, prompt=prompt, model=model)

    async def send(self, conversation_id: str, *, to: str, text: str) -> DeliveryStatus:
        """Send a teammate the user's `text`, which wakes it if idle; see `Team.send`."""
        if not text.strip():
            raise ValueError('Type a message for the teammate.')
        self.record(conversation_id, to)
        with self.about(conversation_id):
            sent = await self.team.send(conversation_id, to=to, text=text.strip())
        return sent.status

    async def stop(self, conversation_id: str, name: str) -> None:
        """Stop a teammate as the user: only the user's next message wakes it."""
        self.record(conversation_id, name)
        with self.about(conversation_id):
            await self.team.stop(conversation_id, name)

    async def assign(self, conversation_id: str, item_id: str, name: str) -> None:
        """Hand a pending board item to a teammate, and tell it, as the lead's `assign_task` does."""
        self.record(conversation_id, name)
        change = await self.team.store(conversation_id).claim(item_id, owner=name)
        if change is None:
            await self._item(conversation_id, item_id)
            raise ValueError(f'#{item_id} is not available: it may be taken, done, or waiting on another item.')
        await self._changed(conversation_id, [change])
        await self.send(
            conversation_id,
            to=name,
            text=f'You are assigned board item #{item_id}: {change.item.content}. Finish it with `finish_task`.',
        )

    async def unclaim(self, conversation_id: str, item_id: str) -> None:
        """Hand a board item in progress back to `pending`, so any teammate can claim it again."""
        item = await self._item(conversation_id, item_id)
        if item.status is not TaskStatus.in_progress or item.owner is None:
            raise ValueError(f'#{item_id} is not in progress for anyone.')
        await self._release(conversation_id, item, owner=item.owner, status=TaskStatus.pending, news='took back')

    async def cancel(self, conversation_id: str, item_id: str) -> None:
        """Cancel a board item that is not done, which unblocks the items waiting only on it."""
        item = await self._item(conversation_id, item_id)
        if item.status in (TaskStatus.completed, TaskStatus.cancelled):
            raise ValueError(f'#{item_id} is {item.status.value} already.')
        owner = item.owner
        if item.status is not TaskStatus.in_progress or owner is None:
            # Taken by the user first, so no teammate claims it meanwhile.
            change = await self.team.store(conversation_id).claim(item_id, owner=USER)
            if change is None:
                raise ValueError(f'#{item_id} waits on another item; cancel or finish that one first.')
            await self._changed(conversation_id, [change])
            owner = USER
        await self._release(conversation_id, item, owner=owner, status=TaskStatus.cancelled, news='cancelled')

    async def _item(self, conversation_id: str, item_id: str) -> OwnedPlanItem:
        item = await self.team.store(conversation_id).get_owned_item(item_id)
        if item is None:
            raise LookupError(f'No board item #{item_id}.')
        return item

    async def _release(
        self, conversation_id: str, item: OwnedPlanItem, *, owner: str, status: TaskStatus, news: str
    ) -> None:
        changes = await self.team.store(conversation_id).release(item.id, owner=owner, status=status)
        await self._changed(conversation_id, changes)
        holder = next((member for member in self.owner.members(conversation_id) if member.name == owner), None)
        if holder is not None and holder.status == 'running':
            # A teammate still working on it hears so at its next model request.
            await self.send(
                conversation_id, to=owner, text=f'The user {news} board item #{item.id}; stop working on it.'
            )

    async def _changed(self, conversation_id: str, changes: Sequence[BoardChange]) -> None:
        for change in changes:
            await self.publish(conversation_id, BoardItemChangedEvent(item=change.item, previous=change.previous))


@dataclass
class _Claims:
    """A conversation's board, as the owner hands back a member's claims; the `Team` publishes each change."""

    teams: Teams
    conversation_id: str

    async def get_owned_items(self) -> list[OwnedPlanItem]:
        return await self.teams.team.store(self.conversation_id).get_owned_items()

    async def release(self, item_id: str, *, owner: str, status: TaskStatus) -> object:
        with self.teams.about(self.conversation_id):
            return await self.teams.team.claims(self.conversation_id).release(item_id, owner=owner, status=status)


@dataclass
class TeamStream(AbstractCapability[object]):
    """Feeds the team events of a lead's run, those its own tools cause, to `Teams`."""

    teams: Teams

    @on_event(*TEAM_EVENTS)
    async def _team_event(self, ctx: RunContext[object], event: TeamEvent) -> None:
        await self.teams.publish(ctx.conversation_id or '', event)


class SessionTeam:
    """One session's team: that of the conversation it holds, the `TeamCatalog` plugins reach as `host.team`."""

    def __init__(
        self,
        teams: Teams,
        *,
        conversation_id: Callable[[], str],
        history: Callable[[str], list[ModelMessage]],
        models: Callable[[], Mapping[str, str]] = dict,
    ) -> None:
        """`history` reads a task's messages so far, by its ID; `models` the configured model of each agent type."""
        self.teams = teams
        self.conversation_id = conversation_id
        self._history = history
        self._models = models

    def snapshot(self) -> TeamSnapshot:
        return self.teams.snapshot(self.conversation_id())

    def history(self, name: str) -> list[ModelMessage]:
        return self._history(self.teams.record(self.conversation_id(), name).id)

    async def spawn(self, name: str, prompt: str, *, model: str | None = None) -> TeamMember:
        # The type defaults as configured now, also before the first turn binds them.
        self.teams.team.models = self._models()
        return await self.teams.spawn(self.conversation_id(), name=name, prompt=prompt, model=model)

    async def send(self, name: str, text: str) -> DeliveryStatus:
        return await self.teams.send(self.conversation_id(), to=name, text=text)

    async def stop(self, name: str) -> None:
        await self.teams.stop(self.conversation_id(), name)

    async def assign(self, item_id: str, name: str) -> None:
        await self.teams.assign(self.conversation_id(), item_id, name)

    async def unclaim(self, item_id: str) -> None:
        await self.teams.unclaim(self.conversation_id(), item_id)

    async def cancel(self, item_id: str) -> None:
        await self.teams.cancel(self.conversation_id(), item_id)


# The roster panel.

STATE_MARKS: Mapping[str, tuple[str | None, str]] = {
    'working': (None, theme.ACCENT),
    'idle': ('✓', theme.MUTED),
    'stopped': ('■', theme.WARNING),
    'failed': ('✗', theme.ERROR),
}
"""Each member state's mark (`None` for the spinner's frame) and colour. A state not listed, such as one a
newer harness adds, shows as `•` in the info colour, so the panel never has to know every state."""

STATE_ORDER = ('working', 'failed', 'stopped', 'idle')
"""The order the summary counts states in; any other state follows, by name."""

COLLAPSE_IDLE = 4
"""With more members than this, idle ones collapse into one `N idle` row."""

SummarySegment = Callable[[TeamSnapshot], str | None]
"""One part of the roster's summary row, or `None` to leave it out."""


def state_counts(snapshot: TeamSnapshot) -> str | None:
    """How many members are in each state, such as `2 working · 1 idle`."""
    counts = Counter(member.state for member in snapshot.members)
    states = sorted(counts, key=lambda state: (STATE_ORDER.index(state) if state in STATE_ORDER else 99, state))
    return ' · '.join(f'{counts[state]} {state}' for state in states) or None


def board_progress(snapshot: TeamSnapshot) -> str | None:
    """How much of the board is done, such as `board 3/7 ✓`."""
    if not snapshot.board:
        return None
    done = sum(item.status is TaskStatus.completed for item in snapshot.board)
    return f'board {done}/{len(snapshot.board)} ✓'


SUMMARY_SEGMENTS: tuple[SummarySegment, ...] = (state_counts, board_progress)
"""The summary row's segments, in order. A team spend segment joins here once `TeamSnapshot` carries the
team's usage total."""


def roster_rows(
    snapshot: TeamSnapshot,
    *,
    glyph: str,
    progress: Mapping[str, Status],
    started: Mapping[str, float],
    now: float | None = None,
) -> tuple[str, ...]:
    """The roster panel: a summary row, a row per member, and the keys, from cached state only.

    `progress` is each member's last activity, and `started` when its current turn started, both by task ID.
    """
    if not snapshot.members:
        return ()
    now = time.time() if now is None else now
    muted, reset = theme.sgr(theme.MUTED), '\x1b[0m'
    summary = ' · '.join(part for segment in SUMMARY_SEGMENTS if (part := segment(snapshot)) is not None)
    rows = [f'{theme.sgr(theme.ACCENT)}team{muted} {summary}{reset}']
    items = {item.id: item for item in snapshot.board}
    collapse = len(snapshot.members) > COLLAPSE_IDLE
    idle = [member for member in snapshot.members if member.state == 'idle']
    for member in snapshot.members:
        if collapse and member.state == 'idle':
            continue
        mark, color = STATE_MARKS.get(member.state, ('•', theme.INFO))
        status = progress.get(member.task_id)
        if member.state == 'working':
            elapsed = now - started.get(member.task_id, now)
            activity = status.activity if status is not None else 'starting'
            detail = f'{elapsed:.0f}s · {activity.removeprefix("running: ").removeprefix("tool: ")}'
        else:
            detail = member.state
        item = items.get(member.current_task) if member.current_task is not None else None
        task = f' #{item.id} {" ".join(terminal_text(item.content).split())}' if item is not None else ''
        rows.append(
            f'{theme.sgr(color)}{mark or glyph} {theme.sgr(theme.INFO)}{terminal_text(member.name)}{muted} '
            f'{display_name(member.agent_name)} · {terminal_text(member.model or "session model")} · '
            f'{theme.sgr(color)}{terminal_text(detail)}{muted}{task}{reset}'
        )
    hint = f'{theme.sgr(theme.ACCENT)}ctrl-x a{muted} team · {theme.sgr(theme.ACCENT)}ctrl-t{muted} board'
    if collapse and idle:
        hint = f'{theme.sgr(STATE_MARKS["idle"][1])}✓ {len(idle)} idle{muted} · {hint}'
    rows.append(hint + reset)
    return tuple(rows)


def display_name(agent_name: str) -> str:
    """An agent type as the user knows it: `general-purpose` for the agent itself."""
    return 'general-purpose' if agent_name == 'self' else agent_name


def item_label(item: OwnedPlanItem, *, board: Sequence[OwnedPlanItem] = ()) -> str:
    """A board item on one line: its ID, status, owner, what it still waits for, and what it says."""
    waiting = {entry.id for entry in board if entry.status not in (TaskStatus.completed, TaskStatus.cancelled)}
    blockers = [dependency for dependency in item.depends_on if dependency in waiting]
    parts = [f'#{item.id}', item.status.value]
    if item.owner is not None:
        parts.append(item.owner)
    if blockers:
        parts.append('⛓ blocked by ' + ', '.join(f'#{dependency}' for dependency in blockers))
    return f'{" · ".join(parts)}  {" ".join(terminal_text(item.content).split())}'
