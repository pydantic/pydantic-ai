"""Agent teams in the shell: the lead spawns teammates on their own models, which report back and wake it."""

import io
import json
from collections.abc import AsyncIterator, Callable
from decimal import Decimal
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.planning import FileClaimablePlanStore, InMemoryClaimablePlanStore
from pydantic_ai_harness.subagents import DelegationTask, DelegationTaskEvent, PlanRequest, is_task_report
from pydantic_ai_harness.teams import TeamMember
from pydantic_clai2.app import create_stock_agent
from pydantic_clai2.config import Settings, resolve_settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime.session_runtime import create_shell
from pydantic_clai2.runtime.tasks import CONTINUE, TaskRouter, Tasks, task_label
from pydantic_clai2.runtime.team_summary import member_line, usage_text

pytestmark = pytest.mark.anyio


def first_prompt(messages: list[ModelMessage]) -> str:
    request = messages[0]
    assert isinstance(request, ModelRequest)
    return next(str(part.content) for part in request.parts if isinstance(part, UserPromptPart))


def reports(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if is_task_report(part)
    ]


def member_model(name: str) -> FunctionModel:
    """A teammate model that says which model it is, and which teammate it ran for."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        teammate = first_prompt(messages).split(':')[0]
        yield f'{teammate} ran on {name}'

    return FunctionModel(stream_function=stream, model_name=name)


def tool_call(tool: str, **args: object) -> DeltaToolCalls:
    return {0: DeltaToolCall(name=tool, json_args=json.dumps(args))}


async def test_a_lead_and_two_teammates_on_their_own_models(tmp_path: Path) -> None:
    heard: list[str] = []
    reported = anyio.Event()

    async def lead(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if seen := reports(messages[-1:]):
            heard.extend(seen)
            if len(heard) >= 2:
                reported.set()
            yield 'Heard from the team.'
        elif len(messages) == 1:
            yield tool_call('spawn_teammate', name='alice', prompt='alice: report.')
        elif len(messages) == 3:
            yield tool_call('spawn_teammate', name='bob', prompt='bob: report.', model='test:mid')
        else:
            yield 'Team started.'

    models: dict[str, Model] = {'test:cheap': member_model('cheap'), 'test:mid': member_model('mid')}
    shell = create_shell(
        create_stock_agent(FunctionModel(stream_function=lead, model_name='lead')),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO(), force_terminal=True, width=100, height=24),
        settings=Settings(model=None, team=True, team_models={'general-purpose': 'test:cheap'}),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    shell.session.resolve_model = lambda name: models.get(name, name)
    owner = shell.tasks.owner
    settled = anyio.Event()
    original_observer = owner.observer
    assert original_observer is not None

    async def observe(update: DelegationTaskEvent) -> None:
        await original_observer(update)
        if update.task.status == 'finished':
            settled.set()

    owner.observer = observe

    async def member_settles(record: DelegationTask, generation: int) -> None:
        while not (record.generation == generation and record.status == 'finished' and record.delivered):
            nonlocal settled
            settled = anyio.Event()
            await settled.wait()

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(20):
        async with anyio.create_task_group() as group:
            group.start_soon(shell.run)
            pipe.send_text('Start the team.\r')
            # The members' reports reach the idle lead without the user typing anything.
            await reported.wait()
            alice, bob = owner.members(shell.session.summary.id)
            assert (alice.model, bob.model) == ('test:cheap', 'test:mid')
            assert (alice.output, bob.output) == ('alice ran on cheap', 'bob ran on mid')
            assert shell.session.team is shell.process.tasks.team

            # The user overrides a teammate's model at spawn, and wakes one that went idle.
            assert await shell.team_command(['spawn carol @test:mid carol: report.']) == 'Started carol on test:mid.'
            (carol,) = [record for record in owner.members(shell.session.summary.id) if record.name == 'carol']
            await member_settles(carol, 1)
            assert carol.output == 'carol ran on mid'
            assert await shell.tasks_command(['resume', 'alice']) == 'Woke teammate alice.'
            await member_settles(alice, 2)
            assert CONTINUE in str(alice.messages)
            assert 'Message woken for bob.' == await shell.team_command(['msg bob Anything else?'])
            await member_settles(bob, 2)
            assert shell.editor is not None
            shell.editor.submit('/exit')

    alice_row, bob_row, carol_row, total = (await shell.team_command([])).splitlines()
    assert alice_row.startswith('alice (general-purpose) · test:cheap · idle · 2 req · ')
    assert bob_row.startswith('bob (general-purpose) · test:mid · idle · 2 req · ')
    assert carol_row.startswith('carol (general-purpose) · test:mid · idle · 1 req · ')
    # The lead's turns count too, while it leads the team.
    team_usage = shell.process.tasks.team.usage(shell.session.summary.id)
    assert team_usage.requests > 5 and total == f'team total, lead included: {usage_text(team_usage)}'
    # One automated lead turn read both first reports, batched; later reports woke it again.
    assert all('<task_output' in report for report in heard) and len(heard) >= 4
    assert sorted(info.name or '' for info in shell.tasks.listing()) == ['alice', 'bob', 'carol']


async def test_team_commands_refuse_without_a_team(tmp_path: Path) -> None:
    shell = create_shell(
        create_stock_agent(FunctionModel(stream_function=lambda messages, info: _empty())),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO(), width=200),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    shell._bind_team()  # pyright: ignore[reportPrivateUsage]
    assert shell.session.team is None
    assert await shell.team_command([]) == (
        'No teammates in this conversation. /team spawn NAME [@MODEL] PROMPT starts one.'
    )
    with pytest.raises(ValueError, match='Teams are off'):
        await shell.team_command(['spawn alice go'])
    with pytest.raises(ValueError, match='Usage'):
        await shell.team_command(['dance'])
    assert tuple(shell.complete_team([])) == ('spawn', 'msg', 'shutdown', 'approve', 'reject')
    assert shell._team_segment() == ''  # pyright: ignore[reportPrivateUsage]
    assert tuple(shell.complete_team(['msg', 'bob', ''])) == ()
    shell.context.store.add_model(name='test:mid')
    assert '@test:mid' in tuple(shell.complete_team(['spawn', 'bob', '']))
    await shell.forks.close()


async def _empty() -> AsyncIterator[str]:
    yield 'ok'


async def test_spawn_without_a_prompt_is_refused(tmp_path: Path) -> None:
    shell = create_shell(
        create_stock_agent(FunctionModel(stream_function=lambda messages, info: _empty())),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO(), width=200),
        settings=Settings(model=None, team=True),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    finished = anyio.Event()
    original_observer = shell.tasks.owner.observer
    assert original_observer is not None

    async def observe(update: DelegationTaskEvent) -> None:
        await original_observer(update)
        if update.task.status == 'finished' and update.task.delivered:
            finished.set()

    shell.tasks.owner.observer = observe
    async with shell.process.tasks.opened(view=shell.tasks) as owner:
        shell.session.delegations = owner
        with pytest.raises(ValueError, match='Usage: /team spawn'):
            await shell.team_command(['spawn alice @test:mid'])
        # With no model for its type, a teammate runs on the session's model.
        assert await shell.team_command(['spawn dan dan: hi']) == 'Started dan on the session model.'
        with anyio.fail_after(10):
            await finished.wait()
        (dan,) = owner.members(shell.session.summary.id)
        assert (dan.model, dan.output) == (None, 'ok')
        finished = anyio.Event()
        await shell.tasks.resume(dan.id)
        with anyio.fail_after(10):
            await finished.wait()
        assert dan.generation == 2
    await shell.forks.close()


async def test_the_router_keeps_a_board_per_conversation(tmp_path: Path) -> None:
    saved = TaskRouter(directory=tmp_path)
    board = saved.board('conversation')
    assert isinstance(board, FileClaimablePlanStore) and board.path.parent == tmp_path / 'boards'
    assert saved.board('conversation') is board and saved.board('other') is not board
    assert isinstance(TaskRouter(directory=None).board('conversation'), InMemoryClaimablePlanStore)


async def test_a_member_runs_and_wakes_in_the_session_holding_its_conversation() -> None:
    router = TaskRouter(directory=None)
    view = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'conv', router=router)
    record = DelegationTask(id='a' * 32, name='alice', agent_name='self', prompt='p', conversation_id='conv')
    with pytest.raises(RuntimeError, match='No session holds this conversation'):
        await router.run_member(record)
    ran: list[str] = []

    async def run_member(member: DelegationTask) -> str:
        ran.append(member.id)
        return 'ok'

    view.run_member = run_member
    assert await router.run_member(record) == 'ok' and ran == [record.id]
    woken: list[bool] = []
    await router.wake_lead('conv')
    view.wake = lambda: woken.append(True)
    await router.wake_lead('conv')
    await router.wake_lead('elsewhere')
    assert woken == [True]
    assert task_label(record) == 'alice (general-purpose)'
    router.owner.records[record.id] = record
    assert view.resolve('alice') is record
    rows = '\n'.join(view.rows('*'))
    assert 'alice (general-purpose)' in rows and 'teammate' not in rows, 'a member started by nobody runs in front'
    record.background = True
    assert 'teammate' in '\n'.join(view.rows('*'))


def test_team_settings_validate_their_models() -> None:
    settings = resolve_settings({'team.enabled': True, 'team.models': {'general-purpose': 'openai:cheap'}})
    assert (settings.team, settings.team_models) == (True, {'general-purpose': 'openai:cheap'})
    assert (settings.team_max_active, settings.team_plan_reviewer) == (0, 'lead')
    settings = resolve_settings({'team.max_active': 2, 'team.plan_reviewer': 'user'})
    assert (settings.team_max_active, settings.team_plan_reviewer) == (2, 'user')
    with pytest.raises(ValueError, match='greater than or equal to 0'):
        resolve_settings({'team.max_active': -1})
    with pytest.raises(ValueError, match="'lead' or 'user'"):
        resolve_settings({'team.plan_reviewer': 'anyone'})
    with pytest.raises(ValueError, match='PROVIDER:MODEL'):
        resolve_settings({'team.models': {'general-purpose': 'cheap'}})
    with pytest.raises(ValueError, match='nonempty'):
        resolve_settings({'team.models': {' general-purpose': 'openai:cheap'}})


async def test_waking_a_running_member_is_refused() -> None:
    view = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'conv', directory=None)
    record = DelegationTask(id='a' * 32, name='alice', agent_name='self', prompt='p', conversation_id='conv')
    view.owner.records[record.id] = record
    with pytest.raises(ValueError, match='still running'):
        await view.resume(record.id)


async def test_queue_plan_review_and_shutdown_from_the_shell(tmp_path: Path) -> None:
    async def planner(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        latest = str(messages[-1])
        if 'Shutdown requested' in latest:
            yield 'Wrapped up.'
        elif 'Your plan was approved' in latest:
            yield 'Building it.'
        elif 'Your plan was rejected' in latest:
            yield tool_call('request_plan_approval', plan='Write it, with tests.')
        else:
            yield tool_call('request_plan_approval', plan='Write it.')

    models: dict[str, Model] = {'test:planner': FunctionModel(stream_function=planner, model_name='planner')}
    shell = create_shell(
        create_stock_agent(FunctionModel(stream_function=lambda messages, info: _empty())),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO(), width=200),
        settings=Settings(model=None, team=True, team_max_active=1, team_plan_reviewer='user'),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    shell.session.resolve_model = lambda name: models.get(name, name)

    async def until(predicate: Callable[[], bool]) -> None:
        with anyio.fail_after(10):
            while not predicate():
                await anyio.sleep(0.01)

    async with shell.process.tasks.opened(view=shell.tasks) as owner:
        shell.session.delegations = owner
        conversation_id = shell.session.summary.id
        await shell.team_command(['spawn ann @test:planner ann: plan first.'])
        await shell.team_command(['spawn ben ben: hi.'])
        ann, ben = owner.members(conversation_id)
        await until(lambda: ann.plan_request is not None and ben.queued)

        # The user reviews: the lead gets no `review_plan` tool, and ben waits for ann's turn.
        assert shell.process.tasks.team.plan_reviewer == 'user'
        ann_row, plan_row, ben_row, _ = (await shell.team_command([])).splitlines()
        assert ann_row.startswith('ann (general-purpose) · test:planner · working · 1 req')
        assert plan_row == '  plan waiting for approval: Write it.'
        assert ben_row.startswith('ben (general-purpose) · session model · queued · 0 req')
        assert shell._team_segment().startswith('team 1 working, 1 queued, ')  # pyright: ignore[reportPrivateUsage]
        # A queued teammate gets no turn to wrap up in.
        await shell.team_command(['spawn cat cat: hi.'])
        assert await shell.team_command(['shutdown cat']) == 'cat is shut down.'
        with pytest.raises(ValueError, match='Usage'):
            await shell.team_command(['reject ann'])
        assert await shell.team_command(['reject ann Add tests.']) == "Rejected ann's plan."
        await until(lambda: ann.plan_request is not None and ann.plan_request.plan == 'Write it, with tests.')
        assert await shell.team_command(['approve ann Go.']) == "Approved ann's plan."
        await until(lambda: ann.status == 'finished' and ben.status == 'finished' and ben.delivered)
        assert ann.output == 'Building it.'

        assert await shell.team_command(['shutdown ben Done.']) == (
            'Asked ben to wrap up; it is shut down when its turn ends.'
        )
        await until(lambda: ben.shut_down and ben.delivered)
        ben_row = (await shell.team_command([])).splitlines()[1]
        assert ' · stopped · ' in ben_row and ben_row.endswith(' · shut down')
        assert await shell.team_command(['shutdown ann']) == 'Asked ann to wrap up; it is shut down when its turn ends.'
        await until(lambda: ann.shut_down)
        with pytest.raises(ValueError, match='No team member'):
            await shell.team_command(['shutdown nobody'])
        with pytest.raises(ValueError, match='no plan waiting'):
            await shell.team_command(['approve ann'])
    await shell.forks.close()


def test_team_rows_show_spend_shutdowns_and_long_plans() -> None:
    usage = RunUsage(requests=3, input_tokens=12_000, output_tokens=500)
    usage.cost = Decimal('0.0123')
    assert usage_text(usage) == '3 req · 12.5k tok · $0.0123'
    assert usage_text(RunUsage(), ', ') == '0 req, 0 tok'
    member = TeamMember(
        name='ann',
        task_id='a' * 32,
        agent_name='self',
        model=None,
        state='working',
        current_task='2',
        shutdown='requested',
        plan=PlanRequest(plan='step ' * 50),
        usage=usage,
    )
    row, plan = member_line(member).splitlines()
    assert row == 'ann (general-purpose) · session model · working · #2 · 3 req · 12.5k tok · $0.0123 · shutting down'
    assert plan.startswith('  plan waiting for approval: step step') and plan.endswith('…')
