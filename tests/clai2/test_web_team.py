"""The web UI's Team pane: the team streamed to the browsers showing it, teammates' transcripts, and the user's actions."""

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from starlette.testclient import WebSocketTestSession

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, PartStartEvent, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.planning import OwnedPlanItem, TaskStatus
from pydantic_ai_harness.subagents import DelegationStartEvent, DeliveryStatus
from pydantic_ai_harness.teams import (
    BoardItemChangedEvent,
    TeammateSpawnedEvent,
    TeammateStateEvent,
    TeamMember,
    TeamMessage,
    TeamMessageEvent,
    TeamSnapshot,
)
from pydantic_clai2.builtin_plugins.web.live import Hub
from pydantic_clai2.builtin_plugins.web.protocol import (
    PROTOCOL,
    BoardItemChanged,
    BoardItemEntry,
    ItemAdded,
    Notice,
    ServerMessage,
    Snapshot,
    TaskChanged,
    TaskItems,
    TeamChanged,
    TeamDelta,
    TeamMemberChanged,
    TeamMessageAdded,
)
from pydantic_clai2.builtin_plugins.web.team import TeamFeed, team_state
from pydantic_clai2.plugins import TaskInfo, TaskProgress, TeamProgress
from tests.clai2.test_web import TOKEN, WIRE, MakeWeb, Web, make_web, receive_until

__all__ = ['make_web']

ORIGIN = {'origin': 'http://testserver'}


def member(name: str, task_id: str, *, state: str = 'working', current_task: str | None = None) -> TeamMember:
    return TeamMember(
        name=name,
        task_id=task_id,
        agent_name='self',
        model=None,
        state=state,  # pyright: ignore[reportArgumentType] -- a state a newer harness may add, too
        current_task=current_task,
    )


def item(
    item_id: str, status: TaskStatus, *, owner: str | None = None, depends_on: Sequence[str] = ()
) -> OwnedPlanItem:
    return OwnedPlanItem(id=item_id, content=f'Item {item_id}', status=status, owner=owner, depends_on=list(depends_on))


def snapshot(
    members: Sequence[TeamMember], board: Sequence[OwnedPlanItem], messages: Sequence[TeamMessage] = ()
) -> TeamSnapshot:
    return TeamSnapshot(conversation_id='c', members=list(members), board=list(board), messages=list(messages))


ALICE = member('alice', 'tA', current_task='1')
FIRST = snapshot(
    [ALICE],
    [item('1', TaskStatus.in_progress, owner='alice'), item('2', TaskStatus.pending, depends_on=['1'])],
)


@dataclass
class FakeTeam:
    """A `TeamCatalog` that records what the user did; `alice` is its only teammate, `#1` and `#2` its items."""

    team: TeamSnapshot = FIRST
    done: list[str] = field(default_factory=list[str])

    def snapshot(self) -> TeamSnapshot:
        return self.team

    def history(self, name: str) -> list[ModelMessage]:
        return [ModelRequest(parts=[UserPromptPart('alice: go')]), ModelResponse(parts=[TextPart('on it')])]

    async def spawn(self, name: str, prompt: str, *, model: str | None = None) -> TeamMember:  # pragma: no cover
        raise NotImplementedError

    def _member(self, name: str) -> None:
        if name != 'alice':
            raise LookupError(f'No teammate {name!r} in this conversation.')

    def _item(self, item_id: str) -> None:
        if item_id not in ('1', '2'):
            raise LookupError(f'No board item #{item_id}.')

    async def send(self, name: str, text: str) -> DeliveryStatus:
        self._member(name)
        if not text.strip():
            raise ValueError('Type a message for the teammate.')
        self.done.append(f'send {name} {text}')
        return 'woken'

    async def stop(self, name: str) -> None:
        self._member(name)
        self.done.append(f'stop {name}')

    async def assign(self, item_id: str, name: str) -> None:
        self._item(item_id)
        self._member(name)
        if item_id == '1':
            raise ValueError('#1 is not available: it may be taken, done, or waiting on another item.')
        self.done.append(f'assign {item_id} {name}')

    async def unclaim(self, item_id: str) -> None:
        self._item(item_id)
        self.done.append(f'unclaim {item_id}')

    async def cancel(self, item_id: str) -> None:
        self._item(item_id)
        self.done.append(f'cancel {item_id}')


def with_team(web_: Web, monkeypatch: pytest.MonkeyPatch) -> FakeTeam:
    """Have the shell lead `FakeTeam`, as it does with `team.enabled` on."""
    team = FakeTeam()
    monkeypatch.setattr(web_.shell, 'team_enabled', lambda: True)
    monkeypatch.setattr(web_.shell.team, 'catalog', team)
    return team


@contextmanager
def connected(web_: Web) -> Generator[tuple[WebSocketTestSession, Snapshot]]:
    """A browser past its first snapshot, `live`, and the orchestrator's state."""
    with web_.client.websocket_connect(f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers=ORIGIN) as socket:
        first = WIRE.validate_json(socket.receive_text())
        assert isinstance(first, Snapshot)
        socket.receive_text()
        socket.receive_text()
        yield socket, first


def team_watch(socket: WebSocketTestSession, sessions: Sequence[str], members: Sequence[str] = ()) -> None:
    socket.send_text(json.dumps({'type': 'team_watch', 'sessions': list(sessions), 'members': list(members)}))


def until_marker(web_: Web, *sockets: WebSocketTestSession) -> list[list[ServerMessage]]:
    """What each socket got before a notice published now to every browser."""
    hub = web_.plugin.service.hub
    web_.portal.call(lambda: hub.publish(Notice(text='marker')))
    return [receive_until(socket, Notice)[0][:-1] for socket in sockets]


def progress(web_: Web, event: TeamProgress) -> None:
    web_.portal.call(web_.plugin.on_team_progress, event)


def task(task_id: str, *, name: str | None) -> TaskInfo:
    return TaskInfo(
        id=task_id,
        name=name,
        parent_id=None,
        agent='general-purpose',
        model=None,
        prompt=f'{task_id} prompt',
        status='running',
        outcome=None,
        started_at=datetime.now(UTC),
        finished_at=None,
        background=True,
        backgroundable=False,
        resumable=True,
        activity='starting',
        usage=RunUsage(),
    )


def started(task_id: str) -> DelegationStartEvent:
    start = DelegationStartEvent(
        agent_name='self', task=f'{task_id} prompt', truncated=False, inherits_tools=True, model=None
    )
    start.task_id = task_id
    return start


def test_without_teams_there_is_no_team_pane(make_web: MakeWeb) -> None:
    web_ = make_web(Agent(TestModel()))
    session = web_.plugin.service.default().id
    off = 'This session leads no team. /set team.enabled true turns teams on.'
    with connected(web_) as (socket, first):
        assert first.team is None
        team_watch(socket, [session])
        _, changed = receive_until(socket, TeamChanged)
        assert changed.team is None
    assert web_.get('/api/team').json() == {'detail': off}
    assert web_.get(f'/api/sessions/{session}/team').status_code == 404
    assert web_.get('/api/team/members/alice').status_code == 404
    assert web_.post('/api/team/members/alice/stop').status_code == 404
    assert web_.post('/api/team/board/1/cancel').status_code == 404
    assert web_.get('/api/sessions/nope/team').status_code == 404


def test_team_changes_reach_only_the_browsers_showing_the_team(
    make_web: MakeWeb, monkeypatch: pytest.MonkeyPatch
) -> None:
    web_ = make_web(Agent(TestModel()))
    with_team(web_, monkeypatch)
    session = web_.plugin.service.default().id
    with connected(web_) as (shown, first), connected(web_) as (other, _):
        assert first.team is not None and first.team == team_state(FIRST)
        [entry] = first.team.members
        assert (entry.agent, entry.state, entry.current_task) == ('general-purpose', 'working', '1')
        assert [(entry.id, entry.blocked_by) for entry in first.team.board] == [('1', []), ('2', ['1'])]

        # Heard before anyone shows the team: nobody gets it.
        progress(web_, TeamProgress(event=TeammateSpawnedEvent(member=ALICE, prompt='go'), team=FIRST))
        assert until_marker(web_, shown, other) == [[], []]

        team_watch(shown, [session], ['not-running'])
        _, whole = receive_until(shown, TeamChanged)
        assert whole.team == team_state(FIRST)

        # #1 done unblocks #2, alice goes idle, and the lead hears from her.
        done = item('1', TaskStatus.completed, owner='alice')
        idle = member('alice', 'tA', state='idle')
        note = TeamMessage(id='m1', sender='alice', recipient='lead', text='#1 done')
        after = snapshot([idle], [done, item('2', TaskStatus.pending, depends_on=['1'])], [note])
        progress(web_, TeamProgress(event=BoardItemChangedEvent(item=done, previous=FIRST.board[0]), team=after))
        progress(web_, TeamProgress(event=TeamMessageEvent(message=note), team=after))
        [received, nothing] = until_marker(web_, shown, other)
        assert nothing == []
        assert all(isinstance(message, TeamDelta) for message in received)
        member_changed, first_item, second_item, message = received
        assert isinstance(member_changed, TeamMemberChanged) and member_changed.member.state == 'idle'
        assert isinstance(first_item, BoardItemChanged) and first_item.item.status == 'completed'
        assert first_item.previous is not None and first_item.previous.status == 'in_progress'
        assert isinstance(second_item, BoardItemChanged) and second_item.item.blocked_by == []
        assert isinstance(message, TeamMessageAdded) and message.message.text == '#1 done'

        # A state this CLAI does not know is passed on as it is.
        resting = member('alice', 'tA', state='resting')
        later = snapshot([resting], after.board, after.messages)
        progress(web_, TeamProgress(event=TeammateStateEvent(member=resting, previous='idle'), team=later))
        [[changed], []] = until_marker(web_, shown, other)
        assert isinstance(changed, TeamMemberChanged) and changed.member.state == 'resting'


def test_a_teammates_transcript_reaches_only_the_browsers_showing_it(
    make_web: MakeWeb, monkeypatch: pytest.MonkeyPatch
) -> None:
    web_ = make_web(Agent(TestModel()))
    with_team(web_, monkeypatch)
    session = web_.plugin.service.default().id
    on_task = web_.plugin.on_task_progress
    alice, helper = task('tA', name='alice'), task('tB', name=None)
    with connected(web_) as (shown, _), connected(web_) as (other, _):
        web_.portal.call(on_task, TaskProgress(task=alice, event=started('tA')))
        # Its row reaches every browser, its transcript none yet.
        for messages in until_marker(web_, shown, other):
            assert [type(message) for message in messages] == [TaskChanged]

        team_watch(shown, [session], ['tA'])
        receive_until(shown, TeamChanged)
        _, joined = receive_until(shown, TaskItems)
        assert joined.reset and joined.id == 'tA'
        assert [message.item.text for message in joined.messages if isinstance(message, ItemAdded)] == ['tA prompt']

        web_.portal.call(on_task, TaskProgress(task=alice, event=PartStartEvent(index=0, part=TextPart('working'))))
        web_.portal.call(on_task, TaskProgress(task=helper, event=started('tB')))
        [mine, theirs] = until_marker(web_, shown, other)
        assert [(message.id, message.reset) for message in mine if isinstance(message, TaskItems)] == [
            ('tA', False),
            ('tB', False),
        ]
        assert [message.id for message in theirs if isinstance(message, TaskItems)] == ['tB'], (
            'only a teammate is left to the browsers showing it'
        )

        # Showing the team already, it gets no second `TeamChanged`; a newly shown transcript starts over.
        team_watch(shown, [session], ['tB'])
        messages, restarted = receive_until(shown, TaskItems)
        assert (restarted.id, restarted.reset, len(messages)) == ('tB', True, 1)


def test_team_routes_reach_the_sessions_team(make_web: MakeWeb, monkeypatch: pytest.MonkeyPatch) -> None:
    web_ = make_web(Agent(TestModel()))
    team = with_team(web_, monkeypatch)
    session = web_.plugin.service.default().id
    for base in ('/api', f'/api/sessions/{session}'):
        assert web_.get(f'{base}/team').json()['board'][1]['blocked_by'] == ['1']
        transcript = web_.get(f'{base}/team/members/alice').json()
        assert transcript['member']['name'] == 'alice'
        assert [entry['text'] for entry in transcript['items']] == ['alice: go', 'on it']
        assert web_.get(f'{base}/team/members/bob').status_code == 404
        assert web_.post(f'{base}/team/members/alice/message', {'text': 'hi'}).json() == {'status': 'woken'}
        assert web_.post(f'{base}/team/members/alice/message', {'text': ' '}).status_code == 400
        assert web_.post(f'{base}/team/members/bob/message', {'text': 'hi'}).status_code == 404
        assert web_.post(f'{base}/team/members/alice/stop').status_code == 204
        assert web_.post(f'{base}/team/board/2/assign', {'owner': 'alice'}).status_code == 204
        refused = web_.post(f'{base}/team/board/1/assign', {'owner': 'alice'})
        assert (refused.status_code, refused.json()['detail']) == (
            400,
            '#1 is not available: it may be taken, done, or waiting on another item.',
        )
        assert web_.post(f'{base}/team/board/9/assign', {'owner': 'alice'}).status_code == 404
        assert web_.post(f'{base}/team/board/2/assign', {'owner': ''}).status_code == 422
        assert web_.post(f'{base}/team/board/2/unclaim').status_code == 204
        assert web_.post(f'{base}/team/board/2/cancel').status_code == 204
        assert web_.post(f'{base}/team/board/9/cancel').status_code == 404
    assert team.done == ['send alice hi', 'stop alice', 'assign 2 alice', 'unclaim 2', 'cancel 2'] * 2


def test_the_feed_sends_the_whole_team_first_and_when_a_row_leaves() -> None:
    sent: list[TeamDelta] = []
    feed = TeamFeed(publish=sent.append)
    feed.observe(TeamProgress(event=TeammateSpawnedEvent(member=ALICE, prompt='go'), team=FIRST))
    feed.observe(TeamProgress(event=TeammateSpawnedEvent(member=ALICE, prompt='go'), team=FIRST))
    assert sent == [TeamChanged(team=team_state(FIRST))], 'an unchanged team sends nothing more'
    added = item('3', TaskStatus.pending)
    grown = snapshot(FIRST.members, [*FIRST.board, added])
    feed.observe(TeamProgress(event=BoardItemChangedEvent(item=added, previous=None), team=grown))
    assert sent[-1] == BoardItemChanged(
        item=BoardItemEntry(id='3', content='Item 3', status='pending', owner=None, depends_on=[], blocked_by=[]),
        previous=None,
    )
    feed.observe(TeamProgress(event=BoardItemChangedEvent(item=added, previous=None), team=FIRST))
    assert sent[-1] == TeamChanged(team=team_state(FIRST)), 'an item left'
    feed.reset()
    feed.observe(TeamProgress(event=TeammateSpawnedEvent(member=ALICE, prompt='go'), team=FIRST))
    assert len(sent) == 4 and isinstance(sent[-1], TeamChanged), 'a new conversation starts over'


def test_a_session_switching_conversation_keeps_its_team_watchers() -> None:
    hub = Hub()
    with hub.subscribe() as client:
        client.teams = {'old'}
        hub.rename('old', 'new')
        assert client.teams == {'new'}
        hub.rename('other', 'newer')
        assert client.teams == {'new'}
