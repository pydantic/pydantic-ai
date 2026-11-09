"""The team as the shell, its plugins and its menus see it: one ordered feed, the roster, the board, and actions."""

import io
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import ClassVar, Literal

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult

from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.planning import OwnedPlanItem, TaskStatus
from pydantic_ai_harness.subagents import DelegationTask
from pydantic_ai_harness.teams import (
    BoardItemChangedEvent,
    TeammateSpawnedEvent,
    TeammateStateEvent,
    TeamMember,
    TeamMessage,
    TeamMessageEvent,
    TeamSnapshot,
)
from pydantic_clai2.app import create_stock_agent
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import Plugin, PluginHost, SessionStart, TeamProgress
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell
from pydantic_clai2.runtime.tasks import TaskRouter, Tasks
from pydantic_clai2.runtime.team import roster_rows
from pydantic_clai2.runtime.team_shell import TEAMS_OFF
from pydantic_clai2.ui.rendering.status import Status

from .menu_script import Script, pick, typed

pytestmark = pytest.mark.anyio


def tool_call(tool: str, **args: object) -> DeltaToolCalls:
    return {0: DeltaToolCall(name=tool, json_args=json.dumps(args))}


def returned(messages: list[ModelMessage]) -> list[str]:
    return [
        part.tool_name
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def prompt_of(messages: list[ModelMessage]) -> str:
    return ' '.join(
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    )


async def lead(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    done = returned(messages)
    if 'Start' not in prompt_of(messages[-1:]) and not done:
        yield 'Heard.'
    elif not done:
        yield tool_call('board_add', content='Write the haiku')
    elif done == ['board_add']:
        yield tool_call('board_add', content='Review the haiku', depends_on=['1'])
    elif done == ['board_add', 'board_add']:
        yield tool_call('spawn_teammate', name='alice', prompt='alice: claim and finish an item.')
    else:
        yield 'Team started.'


async def member(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    done = returned(messages)
    if 'claim and finish' not in prompt_of(messages):
        yield 'Noted.'
    elif not done:
        yield tool_call('claim_task')
    elif done == ['claim_task']:
        yield tool_call('finish_task', item_id='1')
    else:
        yield 'Done with #1.'


class TeamWatcher(Plugin):
    """Keeps its host and every team change, and notes the teammate going idle."""

    hosts: ClassVar[list[PluginHost[None]]] = []
    heard: ClassVar[list[TeamProgress]] = []
    idle: ClassVar[list[anyio.Event]] = []

    async def on_session_start(self, event: SessionStart) -> None:
        self.hosts.append(self.host)

    async def on_team_progress(self, event: TeamProgress) -> None:
        self.heard.append(event)
        if isinstance(event.event, TeammateStateEvent) and event.event.member.state == 'idle':
            self.idle[-1].set()


@pytest.fixture(autouse=True)
def fresh_watcher() -> None:
    TeamWatcher.hosts.clear()
    TeamWatcher.heard.clear()
    TeamWatcher.idle[:] = [anyio.Event()]


def team_shell(tmp_path: Path, *, team: bool = True) -> SessionRuntime[None, str]:
    shell = create_shell(
        create_stock_agent(FunctionModel(stream_function=lead, model_name='lead')),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO(), force_terminal=True, width=120, height=30),
        settings=Settings(model=None, team=team, team_models={'general-purpose': 'test:member'}),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=[PluginSettings(id='team_watcher', factory='tests.clai2.test_team_ui:TeamWatcher')],
        project=ProjectSettings(),
        headless=True,
    )
    shell.session.resolve_model = lambda name: FunctionModel(stream_function=member, model_name='member')
    return shell


def kinds(heard: list[TeamProgress]) -> list[str]:
    described: list[str] = []
    for progress in heard:
        event = progress.event
        if event is None:
            continue  # The team read again, with no change.
        if isinstance(event, BoardItemChangedEvent):
            described.append(f'board #{event.item.id} {event.item.status.value} {event.item.owner}')
        elif isinstance(event, TeammateStateEvent):
            described.append(f'state {event.member.name} {event.member.state}')
        elif isinstance(event, TeammateSpawnedEvent):
            described.append(f'spawned {event.member.name}')
        else:
            described.append(f'message {event.message.sender}->{event.message.recipient}')
    return described


async def test_one_feed_from_the_lead_the_members_and_the_user(tmp_path: Path) -> None:
    shell = team_shell(tmp_path)
    await shell.loader.load_all()
    [host] = TeamWatcher.hosts
    assert TeamWatcher.hosts[0].team is None, 'no task owner runs before the prompt opens'
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(20):
        async with anyio.create_task_group() as group:
            group.start_soon(shell.run)
            pipe.send_text('Start the team.\r')
            await TeamWatcher.idle[-1].wait()
            catalog = host.team
            assert catalog is not None
            assert [(item.id, item.status, item.owner) for item in catalog.snapshot().board] == [
                ('1', TaskStatus.completed, 'alice'),
                ('2', TaskStatus.pending, None),
            ]
            assert 'claim_task' in str(catalog.history('alice'))
            terminal = shell.terminal
            assert terminal is not None
            panel = [_plain(row) for row in terminal._panel('*')]  # pyright: ignore[reportPrivateUsage]
            assert panel[0] == 'team 1 idle · board 1/2 ✓'
            assert panel[1] == '✓ alice general-purpose · test:member · idle'
            # The user hands the unblocked item to the idle teammate, which wakes to it.
            TeamWatcher.idle.append(anyio.Event())
            assert await shell.team.board_command(['assign', '#2', 'alice']) == 'Assigned #2 to alice.'
            await TeamWatcher.idle[-1].wait()
            assert shell.editor is not None
            shell.editor.submit('/exit')
    await shell.loader.close('exit')
    order = kinds(TeamWatcher.heard)
    # The lead's own tool calls, the owner's state changes, and the member's own claim, in the order they happened.
    # Starting alice and the tool reporting her spawned race each other; both come before her claim.
    assert order[:2] == ['board #1 pending None', 'board #2 blocked None'], order
    assert sorted(order[2:4]) == ['spawned alice', 'state alice working'], order
    assert order[4] == 'board #1 in_progress alice', order
    assert order.index('board #2 pending None') > order.index('board #1 completed alice')
    assert order.index('state alice idle') > order.index('board #1 completed alice')
    assigned = order.index('board #2 in_progress alice')
    assert order[assigned + 1 : assigned + 3] == ['message user->alice', 'state alice working']
    assert TeamWatcher.heard[-1].team.conversation_id == shell.session.summary.id


async def test_board_actions_go_through_the_board_rules(tmp_path: Path) -> None:
    router = TaskRouter(directory=None)
    view = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'conv', router=router)
    heard: list[TeamProgress] = []

    async def on_team(progress: TeamProgress) -> None:
        heard.append(progress)

    view.on_team = on_team
    teams = router.teams
    board = router.board('conv')
    for item in (
        OwnedPlanItem(id='1', content='first'),
        OwnedPlanItem(id='2', content='second', depends_on=['1'], status=TaskStatus.blocked),
        OwnedPlanItem(id='3', content='third'),
    ):
        await board.add_item(item)
    record = DelegationTask(
        id='a' * 32, name='alice', agent_name='self', prompt='p', conversation_id='conv', status='finished'
    )
    router.owner.records[record.id] = record
    sent: list[str] = []

    async def send(text: object, *, to: str, conversation_id: str) -> object:
        sent.append(str(text))
        from pydantic_ai_harness.subagents import SentMessage

        return SentMessage(id=str(len(sent)), status='woken')

    router.owner.send = send  # type: ignore[method-assign]
    with pytest.raises(LookupError, match='No teammate'):
        await teams.assign('conv', '1', 'bob')
    with pytest.raises(LookupError, match='No board item #9'):
        await teams.assign('conv', '9', 'alice')
    with pytest.raises(ValueError, match='#2 is not available'):
        await teams.assign('conv', '2', 'alice')
    await teams.assign('conv', '1', 'alice')
    assert sent == ['You are assigned board item #1: first. Finish it with `finish_task`.']
    with pytest.raises(ValueError, match='#3 is not in progress'):
        await teams.unclaim('conv', '3')
    await teams.unclaim('conv', '1')
    assert (await board.get_owned_item('1')) == OwnedPlanItem(id='1', content='first')
    with pytest.raises(ValueError, match='waits on another item'):
        await teams.cancel('conv', '2')
    await teams.cancel('conv', '1')
    cancelled = await board.get_owned_item('1')
    assert cancelled is not None and (cancelled.status, cancelled.owner) == (TaskStatus.cancelled, 'user')
    # Cancelling what #2 waited on unblocks it.
    unblocked = await board.get_owned_item('2')
    assert unblocked is not None and unblocked.status is TaskStatus.pending
    with pytest.raises(ValueError, match='#1 is cancelled already'):
        await teams.cancel('conv', '1')
    # A teammate still working on an item hears when the user takes it back.
    record.status = 'running'
    await teams.assign('conv', '3', 'alice')
    await teams.cancel('conv', '3')
    assert sent[-1] == 'The user cancelled board item #3; stop working on it.'
    assert [type(progress.event).__name__ for progress in heard].count('BoardItemChangedEvent') >= 6
    assert heard[-1].team.board[2].status is TaskStatus.cancelled


async def test_the_feed_keeps_order_and_survives_a_failing_delivery(caplog: pytest.LogCaptureFixture) -> None:
    router = TaskRouter(directory=None)
    view = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'conv', router=router)
    teams = router.teams
    heard: list[str] = []
    item = OwnedPlanItem(id='1', content='x')

    async def on_team(progress: TeamProgress) -> None:
        event = progress.event
        assert isinstance(event, BoardItemChangedEvent)
        heard.append(event.item.content)
        if event.item.content == 'first':
            # A change published while one is delivered waits its turn.
            await teams.publish(
                'conv', BoardItemChangedEvent(item=item.model_copy(update={'content': 'nested'}), previous=None)
            )
            heard.append('after nested publish')
        if event.item.content == 'boom':
            raise RuntimeError('a plugin failed')

    view.on_team = on_team
    await teams.publish('conv', BoardItemChangedEvent(item=item.model_copy(update={'content': 'first'}), previous=None))
    assert heard == ['first', 'after nested publish', 'nested']
    with caplog.at_level(logging.ERROR):
        await teams.publish(
            'conv', BoardItemChangedEvent(item=item.model_copy(update={'content': 'boom'}), previous=None)
        )
    assert 'Delivering a team change failed' in caplog.text
    # An event no conversation can be found for is dropped, with a warning.
    member_ = TeamMember(name='ghost', task_id='0' * 32, agent_name='self', model=None, state='idle')
    with caplog.at_level(logging.WARNING):
        await teams.team.publish(TeammateStateEvent(member=member_, previous=None))
        await teams.team.publish(BoardItemChangedEvent(item=item, previous=None))
    assert caplog.text.count('no known conversation') == 2
    # A teammate's own change finds its conversation from its record.
    router.owner.records[member_.task_id] = DelegationTask(
        id=member_.task_id, name='ghost', agent_name='self', prompt='p', conversation_id='conv'
    )
    await teams.team.publish(TeammateStateEvent(member=member_, previous=None))
    assert len(heard) == 4
    # Without a session holding it, a change is only cached.
    await teams.publish(
        'elsewhere', TeamMessageEvent(message=TeamMessage(id='m', sender='lead', recipient='bob', text='hi'))
    )
    assert teams.snapshot('elsewhere').conversation_id == 'elsewhere'


def snapshot_of(*members: TeamMember, board: list[OwnedPlanItem] | None = None) -> TeamSnapshot:
    return TeamSnapshot(conversation_id='conv', members=list(members), board=board or [], messages=[])


def test_the_roster_renders_any_state_and_collapses_idle_members() -> None:
    working = TeamMember(
        name='alice', task_id='a', agent_name='self', model='test:member', state='working', current_task='1'
    )
    queued = TeamMember(name='bob', task_id='b', agent_name='Review', model=None, state='queued')  # type: ignore[arg-type]
    board = [OwnedPlanItem(id='1', content='Write the haiku', status=TaskStatus.in_progress, owner='alice')]
    rows = roster_rows(
        snapshot_of(working, queued, board=board),
        glyph='*',
        progress={'a': Status(activity='running: write_file')},
        started={'a': 100.0},
        now=112.0,
    )
    plain = [_plain(row) for row in rows]
    assert plain == [
        'team 1 working · 1 queued · board 0/1 ✓',
        '* alice general-purpose · test:member · 12s · write_file #1 Write the haiku',
        '• bob Review · session model · queued',
        'ctrl-x a team · ctrl-t board',
    ]
    idle = [
        TeamMember(name=f'm{index}', task_id=str(index), agent_name='self', model=None, state='idle')
        for index in range(5)
    ]
    plain = [_plain(row) for row in roster_rows(snapshot_of(*idle), glyph='*', progress={}, started={})]
    assert plain == ['team 5 idle', '✓ 5 idle · ctrl-x a team · ctrl-t board']
    assert roster_rows(snapshot_of(), glyph='*', progress={}, started={}) == ()


def _plain(row: str) -> str:
    import re

    return re.sub(r'\x1b\[[0-9;]*m', '', row)


async def test_teams_off_hides_the_team(tmp_path: Path) -> None:
    shell = team_shell(tmp_path, team=False)
    assert shell.loader._base.team is None  # pyright: ignore[reportPrivateUsage]
    assert shell.team.rows('*') == ()
    heard: list[object] = []
    shell.team._fire = heard.append  # type: ignore[assignment]
    progress = TeamProgress(
        event=TeamMessageEvent(message=TeamMessage(id='m', sender='lead', recipient='x', text='')), team=snapshot_of()
    )
    await shell.team.deliver(progress)
    assert heard == []
    assert shell.team.key('/board') == TEAMS_OFF
    with pytest.raises(ValueError, match='Teams are off'):
        await shell.team.board_command(['cancel', '1'])
    assert await shell.team.board_command([]) == 'The board is empty.'
    await shell.forks.close()


async def test_the_menus_message_assign_and_act_on_the_board(tmp_path: Path) -> None:
    from pydantic_clai2.ui.menus.team_menu import TeamAction, TeamView, run_board, run_roster

    alice = TeamMember(name='alice', task_id='a', agent_name='self', model=None, state='idle')
    board = [
        OwnedPlanItem(id='1', content='first'),
        OwnedPlanItem(id='2', content='second', status=TaskStatus.in_progress, owner='alice'),
    ]
    done: list[TeamAction] = []

    def act(action: TeamAction) -> str:
        done.append(action)
        if action.kind == 'cancel':
            raise ValueError('nope')
        return f'{action.kind} ok'

    record = DelegationTask(id='a' * 32, name='alice', agent_name='self', prompt='go', conversation_id='conv')
    view = TeamView(
        snapshot=lambda: snapshot_of(alice, board=board),
        member=lambda name: record if name == 'alice' else None,
        act=act,
        key_source=lambda: '',
    )
    script = Script(lists=[pick('alice'), pick('alice')], choices=[], texts=[typed('hello')])
    # Enter opens alice's transcript; `m` there asks for a message; Esc twice returns.
    from termflow.tui.menu import MenuResult

    script = Script(
        lists=[pick('alice'), MenuResult(cancelled=True)],
        choices=[_opened('message', 'alice'), MenuResult(cancelled=True)],
        texts=[typed('hello')],
    )
    run_roster(view, runners=script.runners)
    assert done == [TeamAction(kind='message', target='alice', value='hello')]
    assert 'Task: go' in view.transcript('alice')
    script = Script(
        lists=[_opened('assign', '1'), _opened('focus', 'alice'), MenuResult(cancelled=True)],
        choices=[pick('alice'), MenuResult(cancelled=True)],
        texts=[],
    )
    run_board(view, runners=script.runners)
    assert done[-1] == TeamAction(kind='assign', target='1', value='alice')
    view.act(TeamAction(kind='cancel', target='2'))
    assert view.notice == 'nope'


def _opened(kind: Literal['focus', 'message', 'assign', 'board'], target: str) -> MenuResult:
    from pydantic_clai2.ui.menus.team_menu import _Open  # pyright: ignore[reportPrivateUsage]

    return MenuResult(item=MenuItem(kind, value=_Open(kind, target)))


def _no_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import nullcontext

    monkeypatch.setattr('termflow.tui.menu.raw_mode', nullcontext)

    def alternate(output: object) -> nullcontext[None]:
        return nullcontext()

    monkeypatch.setattr('termflow.tui.menu.alt_screen', alternate)


def _view(keys: list[str], done: list[object], board: list[OwnedPlanItem]) -> object:
    from pydantic_clai2.ui.menus.team_menu import TeamAction, TeamView

    members = [
        TeamMember(name='alice', task_id='a', agent_name='self', model=None, state='working', current_task='2'),
        TeamMember(name='bob', task_id='b', agent_name='self', model='test:mid', state='idle'),
    ]
    pending = iter(keys)

    def act(action: TeamAction) -> str:
        done.append(action)
        return 'ok'

    record = DelegationTask(id='a' * 32, name='alice', agent_name='self', prompt='go', conversation_id='conv')
    return TeamView(
        snapshot=lambda: snapshot_of(*members, board=board),
        member=lambda name: record if name == 'alice' else None,
        act=act,
        key_source=lambda: next(pending),
    )


def test_the_roster_and_board_widgets_answer_their_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_clai2.ui.menus.team_menu import BoardMenu, MemberFocus, RosterMenu, TeamAction, TeamView

    _no_terminal(monkeypatch)
    board = [
        OwnedPlanItem(id='1', content='first', status=TaskStatus.pending),
        OwnedPlanItem(id='2', content='second', status=TaskStatus.in_progress, owner='alice'),
        OwnedPlanItem(id='3', content='third', status=TaskStatus.blocked, depends_on=['1']),
        OwnedPlanItem(id='4', content='fourth', status=TaskStatus.completed, owner='ghost'),
    ]
    done: list[object] = []
    view = _view(['', 'x', 'm'], done, board)
    assert isinstance(view, TeamView)
    roster = RosterMenu(view)
    result = roster.build().run()
    assert done == [TeamAction(kind='stop', target='alice')]
    assert result.item is not None and str(result.item.value) == "_Open(kind='message', target='alice')"
    assert 'Task: go' in roster.preview(roster.items()[0])
    assert roster.preview(MenuItem('none', value=42)) == 'ok'
    view.notice = ''
    assert view.transcript('bob') == "No teammate 'bob'."
    view = _view(['b'], done, board)
    assert isinstance(view, TeamView)
    result = RosterMenu(view).build().run()
    assert result.item is not None and "'board'" in str(result.item.value)

    # The board: headings by status, blockers, and `u`, `c`, `a`, Enter on the item highlighted.
    view = _view(['', 'u', 'c', 'enter', 'enter', 'a'], done, board)
    assert isinstance(view, TeamView)
    menu = BoardMenu(view)
    labels = [item.label for item in menu.items()]
    assert labels == [
        '#2 · in_progress · alice  second',
        '#1 · pending  first',
        '#3 · blocked · ⛓ blocked by #1  third',
        '#4 · completed · ghost  fourth',
    ]
    assert 'After: #1' in menu.preview(next(item for item in menu.items() if item.value == '3'))
    assert menu.preview(MenuItem('heading', value=42)) == ''
    view.selected = None
    result = menu.build().run()
    assert done[-2:] == [TeamAction(kind='unclaim', target='2'), TeamAction(kind='cancel', target='2')]
    assert result.item is not None and str(result.item.value) == "_Open(kind='focus', target='alice')"
    # Enter on an item whose owner is no teammate does nothing; `a` asks whom to assign it to.
    # A row stays where it was when its item changes status, so the next key acts on the same item.
    board[1] = board[1].model_copy(update={'status': TaskStatus.cancelled})
    view.refresh()
    assert [item.value for item in menu.items()] == ['2', '1', '3', '4']
    view.selected = '4'
    result = menu.build().run()
    assert result.item is not None and str(result.item.value) == "_Open(kind='assign', target='4')"
    view = _view(['', 'm', 'x', 'enter', 'escape'], done, board)
    assert isinstance(view, TeamView)
    focus = MemberFocus(view, 'alice', size=lambda: (60, 20))
    result = focus.build().run()
    assert result.item is not None and "'message'" in str(result.item.value)
    assert BoardMenu(_view([], done, [])).items()[0].disabled  # type: ignore[arg-type]
    # Rows that name no teammate or item do nothing.
    roster._stop(MenuItem('heading', value=42))  # pyright: ignore[reportPrivateUsage]
    assert roster._message(MenuItem('heading', value=42)) is None  # pyright: ignore[reportPrivateUsage]
    menu._act(MenuItem('heading', value=42), 'cancel')  # pyright: ignore[reportPrivateUsage]
    assert menu._assign(MenuItem('heading')) is None  # pyright: ignore[reportPrivateUsage]
    from termflow.tui.menu import MenuResult as Result

    from pydantic_clai2.ui.menus.team_menu import _opened as opened  # pyright: ignore[reportPrivateUsage]

    assert opened(Result(item=MenuItem('odd', value=42))) is None
    empty = TeamView(
        snapshot=lambda: snapshot_of(), member=lambda name: None, act=lambda action: '', key_source=lambda: ''
    )
    assert RosterMenu(empty).items()[0].disabled


async def test_open_team_runs_the_menus_on_a_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    from termflow.tui.menu import MenuResult

    from pydantic_clai2.ui.menus.field_menu import Runners
    from pydantic_clai2.ui.menus.team_menu import TeamAction, open_team

    alice = TeamMember(name='alice', task_id='a', agent_name='self', model=None, state='idle')
    done: list[TeamAction] = []

    class Controls:
        def snapshot(self) -> TeamSnapshot:
            return snapshot_of(alice, board=[OwnedPlanItem(id='1', content='first')])

        def member(self, name: str) -> DelegationTask | None:
            return None

        async def act(self, action: TeamAction) -> str:
            done.append(action)
            if action.kind == 'stop':
                raise LookupError('gone')
            return 'sent'

    lists = iter(
        [_opened('board', ''), MenuResult(cancelled=True), _opened('message', 'alice'), MenuResult(cancelled=True)]
    )
    texts = iter([typed('  ')])
    runners = Runners(run_list=lambda menu: next(lists), run_text=lambda widget: next(texts))
    assert await open_team(Controls(), runners=runners, key_source=lambda: '') == ''
    assert done == []
    lists = iter([_opened('assign', '1'), MenuResult(cancelled=True)])
    choices = iter([MenuResult(cancelled=True)])
    runners = Runners(run_list=lambda menu: next(lists), run_choice=lambda menu: next(choices))
    assert await open_team(Controls(), board=True, runners=runners, key_source=lambda: '') == ''
    lists = iter([pick(42), MenuResult(cancelled=True)])
    runners = Runners(run_list=lambda menu: next(lists))
    assert await open_team(Controls(), board=True, runners=runners, key_source=lambda: '') == ''
    # A teammate's transcript reads its record on the loop; `x` there stops it, which can fail.
    lists = iter([_opened('focus', 'alice'), MenuResult(cancelled=True)])
    choices = iter([MenuResult(cancelled=True)])
    runners = Runners(run_list=lambda menu: next(lists), run_choice=lambda menu: next(choices))
    assert await open_team(Controls(), runners=runners, key_source=lambda: '') == ''


async def settled(record: DelegationTask) -> None:
    with anyio.fail_after(10):
        while running(record):
            await anyio.sleep(0.01)


def running(record: DelegationTask) -> bool:
    return record.status == 'running'


async def test_shell_commands_drive_the_team(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = team_shell(tmp_path)
    owner = shell.tasks.owner
    async with shell.process.tasks.opened(view=shell.tasks) as opened:
        shell.session.delegations = opened
        team = shell.team
        assert (
            await team.command([''])
            == 'No teammates in this conversation. /team spawn NAME [@MODEL] PROMPT starts one.'
        )
        assert await team.command(['spawn dan @test:member dan: hi']) == 'Started dan on test:member.'
        [dan] = owner.members(shell.session.summary.id)
        await settled(dan)
        assert await team.command(['list']) == 'dan · general-purpose · test:member · idle'
        assert tuple(team.complete(['msg', ''])) == ('dan',)
        assert tuple(team.complete(['dance', 'x', 'y', 'z'])) == ()
        assert await team.command(['stop dan']) == 'Stopped dan; your next message wakes it.'
        with pytest.raises(LookupError, match='No teammate'):
            await team.command(['msg bob hi'])
        with pytest.raises(ValueError, match='Usage: /team'):
            await team.command(['stop'])
        board = shell.process.tasks.board(shell.session.summary.id)
        await board.add_item(OwnedPlanItem(id='1', content='first'))
        assert await team.board_command(['list']) == '#1 · pending  first'
        assert tuple(team.complete_board([])) == ('list', 'assign', 'unclaim', 'cancel')
        assert tuple(team.complete_board(['assign', ''])) == ('1',)
        assert tuple(team.complete_board(['assign', '1', ''])) == ('dan',)
        assert tuple(team.complete_board(['list', '1', 'x', 'y'])) == ()
        assert await team.board_command(['cancel', '#1']) == 'Cancelled #1.'
        with pytest.raises(ValueError, match='#1 is not in progress'):
            await team.board_command(['unclaim', '1'])
        with pytest.raises(ValueError, match='Usage: /board'):
            await team.board_command(['assign', '1'])
        # A message reaches dan, the user's word waking him; the menus' actions say what they did.
        from pydantic_clai2.ui.menus.team_menu import TeamAction

        assert (await team.act(TeamAction(kind='message', target='dan', value='hi'))).startswith('Message ')
        assert team.member('dan') is not None and team.member('nobody') is None
        assert team.snapshot().conversation_id == shell.session.summary.id
        # In a terminal, the bare commands open the menus instead.
        opened_menus: list[bool] = []

        async def fake_open(controls: object, *, board: bool = False) -> str:
            opened_menus.append(board)
            return 'closed'

        monkeypatch.setattr('pydantic_clai2.runtime.team_shell.open_team', fake_open)
        team.interactive = lambda: True
        assert await team.command([]) == 'closed' and await team.board_command([]) == 'closed'
        assert opened_menus == [False, True]
        entered: list[str] = []
        team._send_input = entered.append  # pyright: ignore[reportPrivateUsage]
        assert team.key('/board') == '' and entered == ['/board']
        assert team.rows('*')[0].startswith('\x1b')
        catalog = team.catalog
        await settled(dan)
        assert 'dan: hi' in str(catalog.history('dan'))
        await board.add_item(OwnedPlanItem(id='2', content='second'))
        await catalog.assign('2', 'dan')
        assert await team.board_command(['unclaim', '2']) == '#2 is pending again.'
        await catalog.cancel('2')
        with pytest.raises(ValueError, match='Type a message'):
            await catalog.send('dan', ' ')
        await catalog.stop('dan')
        member_ = await catalog.spawn('eve', 'eve: hi')
        assert member_.model == 'test:member'
    await shell.forks.close()


async def test_claims_handed_back_by_the_owner_reach_the_feed() -> None:
    router = TaskRouter(directory=None)
    view = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'conv', router=router)
    heard: list[TeamProgress] = []

    async def on_team(progress: TeamProgress) -> None:
        heard.append(progress)

    view.on_team = on_team
    board = router.board('conv')
    await board.add_item(OwnedPlanItem(id='1', content='first'))
    await board.claim('1', owner='alice')
    claims = router.owner.claims
    assert claims is not None
    handed = claims('conv')
    assert handed is not None
    assert [item.owner for item in await handed.get_owned_items()] == ['alice']
    await handed.release('1', owner='alice', status=TaskStatus.pending)
    assert [type(progress.event).__name__ for progress in heard] == ['BoardItemChangedEvent']
    # A task that is no team member never reaches the team.
    from pydantic_ai_harness.subagents import DelegationTaskEvent

    plain = DelegationTask(id='b' * 32, agent_name='self', prompt='p', conversation_id='conv')
    await router.teams.observe(DelegationTaskEvent(task=plain))
    assert len(heard) == 1
    # Read before any change, a conversation's team is empty, then read in the background and delivered.
    read = anyio.Event()

    async def on_read(progress: TeamProgress) -> None:
        heard.append(progress)
        read.set()

    view.on_team = on_read
    assert router.teams.snapshot('fresh').members == []
    view.conversation_id = lambda: 'fresh'
    assert router.teams.snapshot('fresh').board == [], 'one read at a time'
    with anyio.fail_after(5):
        await read.wait()
    assert heard[-1].event is None and heard[-1].team.conversation_id == 'fresh'


def test_the_cache_reads_nothing_without_a_loop() -> None:
    router = TaskRouter(directory=None)
    assert router.teams.snapshot('conv').board == []


def test_a_host_built_outside_the_shell_has_no_team() -> None:
    from pydantic_clai2.builtin_plugins.web.host_session import HostSession

    host = PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={})
    assert host.team is None and HostSession(host).team is None
