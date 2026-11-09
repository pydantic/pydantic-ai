"""`Team` composed over a real `DelegationTasks` owner and real claimable boards, driven through agent runs."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import Team
from pydantic_ai_harness.planning import (
    ClaimablePlanStore,
    FileClaimablePlanStore,
    InMemoryClaimablePlanStore,
    OwnedPlanItem,
    TaskStatus,
)
from pydantic_ai_harness.subagents import DelegationTask, DelegationTaskEvent, DelegationTasks, SubAgent
from pydantic_ai_harness.teams import (
    BoardItemChangedEvent,
    TeamEvent,
    TeammateIdleEvent,
    TeammateSpawnedEvent,
    TeammateStateEvent,
    TeamMember,
    TeamMessage,
    TeamMessageEvent,
    TeamTaskCompletingEvent,
    TeamTaskCreatingEvent,
)

pytestmark = pytest.mark.anyio

WAIT = 10

Reply = str | ToolCallPart
Respond = Callable[[list[ModelMessage]], Reply]


def call(tool_name: str, **args: object) -> ToolCallPart:
    return ToolCallPart(tool_name, args, tool_call_id=f'{tool_name}-{uuid4().hex[:8]}')


def model_of(respond: Respond, *, name: str = 'function') -> FunctionModel:
    """A model whose replies `respond` picks from the conversation so far, streamed or not."""

    async def request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        reply = respond(messages)
        return ModelResponse(parts=[TextPart(reply) if isinstance(reply, str) else reply])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        reply = respond(messages)
        if isinstance(reply, str):
            yield reply
        else:
            yield {
                0: DeltaToolCall(
                    name=reply.tool_name, json_args=reply.args_as_json_str(), tool_call_id=reply.tool_call_id
                )
            }

    return FunctionModel(function=request, stream_function=stream, model_name=name)


def scripted(*replies: Reply, name: str = 'function') -> FunctionModel:
    """A model that sends `replies` in order, then `'done'`."""
    remaining = list(replies)
    return model_of(lambda messages: remaining.pop(0) if remaining else 'done', name=name)


def last_return(messages: list[ModelMessage]) -> str | None:
    """The content of the latest tool return or retry, if the latest request holds one."""
    request = messages[-1]
    assert isinstance(request, ModelRequest)
    returns = [part for part in request.parts if isinstance(part, ToolReturnPart)]
    return str(returns[-1].content) if returns else None


def tool_returns(messages: list[ModelMessage]) -> list[str]:
    return [str(part.content) for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


def worker(ran: list[tuple[str, str]], name: str) -> FunctionModel:
    """A member model that claims the next board item, completes it, and repeats until none is left.

    Each request it answers is recorded in `ran` as `(model name, member)`, which shows which model a member
    really ran on.
    """

    def respond(messages: list[ModelMessage]) -> Reply:
        ran.append((name, str(messages[0])))
        result = last_return(messages)
        if result is not None and result.startswith('Claimed #'):
            return call('finish_task', item_id=result.removeprefix('Claimed #').split(' ')[0])
        if result is None or result.startswith('#'):
            return call('claim_task')
        return f'Nothing left to claim, on {name}.'

    return model_of(respond, name=name)


class Watch:
    """The owner's observer: forwards every update to the team, and lets a test wait for a state."""

    def __init__(self) -> None:
        self.team: Team[object] | None = None
        self.updates: list[DelegationTaskEvent] = []
        self.finished: set[tuple[str, int]] = set()
        """The turns the observer, and so the team, has seen finish, by task ID and generation."""
        self._changed = anyio.Event()

    async def __call__(self, update: DelegationTaskEvent) -> None:
        self.updates.append(update)
        assert self.team is not None, 'Set as the host is wired up, before the owner opens.'
        await self.team.observe(update)
        if update.task.status == 'finished':
            self.finished.add((update.task.id, update.task.generation))
        self._changed.set()
        self._changed = anyio.Event()

    async def until(self, predicate: Callable[[], bool]) -> None:
        with anyio.fail_after(WAIT):
            while not predicate():
                await self._changed.wait()

    async def settled(self, record: DelegationTask, generation: int = 1) -> None:
        await self.until(lambda: (record.id, generation) in self.finished and record.delivered)

    def member_events(self, event_type: type[Any]) -> list[Any]:
        """The events of `event_type` that members' runs emitted, which reach the owner's observer."""
        return [update.event for update in self.updates if isinstance(update.event, event_type)]


@dataclass
class Host:
    """What an application wires up: an owner whose runner runs members through the team, on their own models."""

    owner: DelegationTasks
    team: Team[object]
    lead: Agent[object, str]
    lead_model: FunctionModel
    models: dict[str, FunctionModel]
    watch: Watch
    observed: list[TeamEvent] = field(default_factory=list[TeamEvent])

    async def run_member(self, record: DelegationTask) -> str:
        run = self.team.member_run(record, self_agent=self.lead)
        # The member's pinned model ID, resolved as the host resolves model IDs, else the lead's model.
        result = await run.run(model=self.models[run.model] if run.model is not None else self.lead_model)
        return str(result.output)

    async def lead_turn(self, prompt: str) -> list[ModelMessage]:
        with self.owner.bind():
            result = await self.lead.run(prompt, conversation_id='conv')
        return result.all_messages()


@asynccontextmanager
async def hosting(
    directory: Path,
    board: ClaimablePlanStore,
    *,
    lead_model: FunctionModel,
    models: dict[str, FunctionModel] | None = None,
    team_models: dict[str, str] | None = None,
    gates: list[Hooks[object]] | None = None,
    member_usage_limits: UsageLimits | None = None,
    agents: dict[str, SubAgent[object]] | None = None,
    max_active: int | None = 1,
    plan_reviewer: Literal['lead', 'user'] = 'lead',
    usage_limits: UsageLimits | None = None,
    shutdown_grace: float | None = None,
) -> AsyncGenerator[Host]:
    watch = Watch()
    owner = DelegationTasks(directory=directory, observer=watch, max_active=max_active, agents=agents)
    host_ref: list[Host] = []
    team = Team[object](
        owner,
        board=board,
        gates=gates or [],
        models=team_models,
        member_usage_limits=member_usage_limits,
        observer=lambda event: _record(host_ref, event),
        plan_reviewer=plan_reviewer,
        usage_limits=usage_limits,
        shutdown_grace=shutdown_grace,
    )
    owner.claims = team.claims

    async def run_member(record: DelegationTask) -> str:
        return await host_ref[0].run_member(record)

    owner.member_runner = run_member
    watch.team = team
    host = Host(
        owner=owner,
        team=team,
        lead=Agent(lead_model, deps_type=object, capabilities=[team]),
        lead_model=lead_model,
        models=models or {},
        watch=watch,
    )
    host_ref.append(host)
    with anyio.fail_after(WAIT):
        async with owner.opened():
            yield host


async def _record(host_ref: list[Host], event: TeamEvent) -> None:
    host_ref[0].observed.append(event)


class TestTeamComposition:
    async def test_lead_and_two_members_work_the_board_on_their_own_models(self, tmp_path: Path) -> None:
        """The lead plans and spawns; each member claims and completes items on the model it is pinned to."""
        ran: list[tuple[str, str]] = []
        completing: list[tuple[str, str | None, str]] = []
        gate = Hooks[object]()

        @gate.on.event(TeamTaskCompletingEvent)
        async def record_completion(ctx: RunContext[object], event: TeamTaskCompletingEvent) -> None:
            completing.append((event.item.id, event.item.owner, event.owner))

        lead_model = scripted(
            call('board_add', content='Write the migration'),
            call('board_add', content='Add the endpoint', depends_on=['1']),
            call('spawn_teammate', name='alice', prompt='alice: work the board.'),
            call('spawn_teammate', name='bob', prompt='bob: work the board.', model='mid'),
            'Team started.',
            name='lead-model',
        )
        board = FileClaimablePlanStore(tmp_path / 'board.json')
        async with hosting(
            tmp_path / 'tasks',
            board,
            lead_model=lead_model,
            models={'cheap': worker(ran, 'cheap'), 'mid': worker(ran, 'mid')},
            team_models={'self': 'cheap'},
            gates=[gate],
        ) as host:
            messages = await host.lead_turn('Ship the endpoint.')
            alice, bob = host.owner.members('conv')
            await host.watch.settled(alice)
            await host.watch.settled(bob)

            assert tool_returns(messages) == [
                'Added #1 [pending] Write the migration',
                'Added #2 [blocked] Add the endpoint (after: #1)',
                'Started alice (self on cheap). It reports back when it goes idle.',
                'Started bob (self on mid). It reports back when it goes idle.',
            ]
            # `max_active=1` runs alice's turn first: she takes both items, as the first unblocks the second.
            assert {model for model, prompt in ran if 'alice: work' in prompt} == {'cheap'}
            assert {model for model, prompt in ran if 'bob: work' in prompt} == {'mid'}
            assert (alice.model, bob.model) == ('cheap', 'mid')
            assert completing == [('1', 'alice', 'alice'), ('2', 'alice', 'alice')]

            snapshot = await host.team.snapshot('conv')
            assert snapshot.members == [
                TeamMember(name='alice', task_id=alice.id, agent_name='self', model='cheap', state='idle'),
                TeamMember(name='bob', task_id=bob.id, agent_name='self', model='mid', state='idle'),
            ]
            assert [(item.id, item.status, item.owner) for item in snapshot.board] == [
                ('1', TaskStatus.completed, 'alice'),
                ('2', TaskStatus.completed, 'alice'),
            ]
            # The members' board events reach the owner's observer, as their other stream events do.
            changed: list[BoardItemChangedEvent] = host.watch.member_events(BoardItemChangedEvent)
            assert [
                (event.item.id, event.previous.status if event.previous else None, event.item.status)
                for event in changed
            ] == [
                ('1', TaskStatus.pending, TaskStatus.in_progress),
                ('1', TaskStatus.in_progress, TaskStatus.completed),
                ('2', TaskStatus.blocked, TaskStatus.pending),
                ('2', TaskStatus.pending, TaskStatus.in_progress),
                ('2', TaskStatus.in_progress, TaskStatus.completed),
            ]
            states = [
                (event.member.name, event.previous, event.member.state)
                for event in host.observed
                if isinstance(event, TeammateStateEvent)
            ]
            assert ('alice', 'working', 'idle') in states and ('bob', 'working', 'idle') in states

    async def test_a_pinned_model_survives_a_restart(self, tmp_path: Path) -> None:
        ran: list[tuple[str, str]] = []
        board = InMemoryClaimablePlanStore()
        lead_model = scripted(call('spawn_teammate', name='alice', prompt='alice: idle.', model='mid'), 'ok')
        async with hosting(tmp_path, board, lead_model=lead_model, models={'mid': worker(ran, 'mid')}) as host:
            await host.lead_turn('Start alice.')
            (alice,) = host.owner.members('conv')
            await host.watch.settled(alice)

        ran.clear()
        async with hosting(
            tmp_path, board, lead_model=scripted(name='lead-model'), models={'mid': worker(ran, 'mid')}
        ) as host:
            (alice,) = host.owner.members('conv')
            assert (await host.team.snapshot('conv')).members[0].model == 'mid'
            sent = await host.team.send('conv', to='alice', text='Anything left?')
            assert sent.status == 'woken'
            await host.watch.settled(alice, generation=2)

            assert {model for model, _ in ran} == {'mid'}
            assert [event for event in host.observed if isinstance(event, TeamMessageEvent)] == [
                TeamMessageEvent(
                    message=TeamMessage(id=sent.id, sender='user', recipient='alice', text='Anything left?')
                )
            ]

    async def test_a_member_with_no_pinned_model_runs_on_the_leads(self, tmp_path: Path) -> None:
        def lead(messages: list[ModelMessage]) -> Reply:
            if 'alice: go.' in str(messages[0]):
                return 'alice ran on the lead model.'
            if last_return(messages) is None:
                return call('spawn_teammate', name='alice', prompt='alice: go.')
            return 'ok'

        async with hosting(tmp_path, InMemoryClaimablePlanStore(), lead_model=model_of(lead)) as host:
            messages = await host.lead_turn('Start alice.')
            (alice,) = host.owner.members('conv')
            await host.watch.settled(alice)

        assert tool_returns(messages) == ['Started alice (self). It reports back when it goes idle.']
        assert (alice.model, alice.output) == (None, 'alice ran on the lead model.')

    async def test_a_failed_member_hands_its_claim_back_to_a_real_board(self, tmp_path: Path) -> None:
        def crash(messages: list[ModelMessage]) -> Reply:
            if last_return(messages) is None:
                return call('claim_task')
            raise RuntimeError('model fell over')

        board = FileClaimablePlanStore(tmp_path / 'board.json')
        await board.add_item(OwnedPlanItem(id='1', content='Fix it'))
        async with hosting(
            tmp_path / 'tasks', board, lead_model=scripted(), models={'crashy': model_of(crash)}
        ) as host:
            member = await host.team.spawn('conv', name='carol', prompt='carol: fix it.', model='crashy')
            (carol,) = host.owner.members('conv')
            await host.watch.settled(carol)

            assert (carol.outcome, member.model) == ('failed', 'crashy')
            assert await board.get_owned_items() == [OwnedPlanItem(id='1', content='Fix it')]
            # The spawn and the member's first state race each other; the release comes before the failure.
            assert {type(event).__name__ for event in host.observed[:2]} == {
                'TeammateSpawnedEvent',
                'TeammateStateEvent',
            }
            released, failed = host.observed[2:]
            assert isinstance(released, BoardItemChangedEvent)
            assert released.previous is not None and released.previous.owner == 'carol'
            assert released.item.owner is None and released.item.status is TaskStatus.pending
            assert isinstance(failed, TeammateStateEvent)
            assert (failed.previous, failed.member.state) == ('working', 'failed')

    async def test_a_gate_that_always_rejects_is_stopped_by_the_member_budget(self, tmp_path: Path) -> None:
        gate = Hooks[object]()
        seen: list[TeammateIdleEvent] = []

        @gate.on.event(TeammateIdleEvent)
        async def never_idle(ctx: RunContext[object], event: TeammateIdleEvent) -> None:
            seen.append(event)
            event.keep_working('Keep going.')

        async with hosting(
            tmp_path,
            InMemoryClaimablePlanStore(),
            lead_model=scripted(),
            models={'m': scripted(*(['Still going.'] * 10))},
            gates=[gate],
            member_usage_limits=UsageLimits(request_limit=3),
        ) as host:
            await host.team.spawn('conv', name='dave', prompt='dave: go.', model='m')
            (dave,) = host.owner.members('conv')
            await host.watch.settled(dave)
            await anyio.sleep(0.05)

            assert dave.outcome == 'failed' and 'request_limit of 3' in dave.output
            assert (len(seen), dave.generation) == (3, 1)
            assert (await host.team.snapshot('conv')).members[0].state == 'failed'

    async def test_gates_check_the_leads_board_tools(self, tmp_path: Path) -> None:
        gate = Hooks[object]()
        creating: list[OwnedPlanItem] = []

        @gate.on.event(TeamTaskCreatingEvent)
        async def require_done_criteria(ctx: RunContext[object], event: TeamTaskCreatingEvent) -> None:
            creating.append(event.item)
            if 'done when' not in event.item.content:
                event.reject('Say when it is done.')

        lead_model = scripted(
            call('board_add', content='Write docs'),
            call('board_add', content='Write docs, done when examples run'),
            call('board_add', content='Publish', depends_on=['7']),
            # Models write IDs as the tools show them, with a `#`.
            call('board_add', content='Publish, done when live', depends_on=['#1']),
            call('finish_task', item_id='#1'),
            call('assign_task', item_id='1', teammate='nobody'),
            call('board_list'),
        )
        board = InMemoryClaimablePlanStore()
        async with hosting(tmp_path, board, lead_model=lead_model, gates=[gate]) as host:
            messages = await host.lead_turn('Plan the docs.')

        assert tool_returns(messages) == [
            'The task was not added to the board.\n\nSay when it is done.',
            'Added #1 [pending] Write docs, done when examples run',
            'No board items #7. Use `board_list`.',
            'Added #2 [blocked] Publish, done when live (after: #1)',
            '#1 is not in progress for anyone.',
            "No teammate 'nobody'.",
            '#1 [pending] Write docs, done when examples run\n#2 [blocked] Publish, done when live (after: #1)',
        ]
        assert [item.id for item in creating] == ['1', '1', '2']

    async def test_members_message_each_other_and_the_lead(self, tmp_path: Path) -> None:
        def chatty(messages: list[ModelMessage]) -> Reply:
            first = str(messages[0])
            result = last_return(messages)
            if 'alice:' in first and result is None:
                return call('send_message', to='bob', text='Take #2.')
            if 'alice:' in first and result is not None and 'bob' in result:
                return call('send_message', to='lead', text='Asked bob.')
            return 'ok'

        lead_model = scripted(
            call('spawn_teammate', name='bob', prompt='bob: wait.'),
            call('spawn_teammate', name='alice', prompt='alice: coordinate.'),
            call('send_message', to='lead', text='hi'),
            call('spawn_teammate', name='alice', prompt='again'),
            call('board_list'),
        )
        async with hosting(
            tmp_path, InMemoryClaimablePlanStore(), lead_model=lead_model, models={}, max_active=None
        ) as host:
            host.lead_model = model_of(chatty)
            messages = await host.lead_turn('Start the team.')
            _, alice = host.owner.members('conv')
            await host.watch.settled(alice)

            assert tool_returns(messages)[2:] == [
                'You cannot message yourself.',
                "The team already has a member named 'alice'",
                'The board is empty.',
            ]
            snapshot = await host.team.snapshot('conv')
            assert [(m.sender, m.recipient, m.text) for m in snapshot.messages] == [
                ('alice', 'bob', 'Take #2.'),
                ('alice', 'lead', 'Asked bob.'),
            ]
            sent: list[TeamMessageEvent] = host.watch.member_events(TeamMessageEvent)
            assert [event.message for event in sent] == snapshot.messages

    async def test_the_user_stops_a_member(self, tmp_path: Path) -> None:
        async with hosting(
            tmp_path, InMemoryClaimablePlanStore(), lead_model=scripted(), models={'m': scripted()}
        ) as host:
            await host.team.spawn('conv', name='erin', prompt='erin', model='m')
            (erin,) = host.owner.members('conv')
            await host.watch.settled(erin)

            await host.team.stop('conv', 'erin')

            assert erin.user_stopped
            assert (await host.team.snapshot('conv')).members[0].state == 'stopped'
            with pytest.raises(ValueError, match='No team member'):
                await host.team.stop('conv', 'nobody')
            with pytest.raises(ValueError, match='No team member'):
                await host.team.send('conv', to='nobody', text='hi')
            with pytest.raises(ValueError, match='Invalid team member name'):
                await host.team.spawn('conv', name='lead', prompt='x')

    async def test_specialist_types_get_a_default_model_by_type_or_alias(self, tmp_path: Path) -> None:
        reviewer = SubAgent[object](Agent(deps_type=object, name='Reviewer'))
        board = InMemoryClaimablePlanStore()
        async with hosting(
            tmp_path,
            board,
            lead_model=scripted(),
            models={'small': scripted('reviewed')},
            team_models={'other': 'big', 'Reviewer': 'small'},
            agents={'Reviewer': reviewer},
        ) as host:
            host.owner.aliases['review'] = 'Reviewer'
            member = await host.team.spawn('conv', name='rita', prompt='rita', agent_name='review')
            (rita,) = host.owner.members('conv')
            await host.watch.settled(rita)

            assert (member.agent_name, member.model, rita.output) == ('Reviewer', 'small', 'reviewed')
            assert host.team.agent_types() == ['self', 'Reviewer']

    async def test_the_lead_assigns_an_item_and_the_member_hands_it_back(self, tmp_path: Path) -> None:
        lead_done = anyio.Event()

        async def member(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            if 'You are assigned board item #1' not in str(messages[-1]):
                yield 'ready'
                return
            # Handed back once the lead's turn is over, so its second `assign_task` finds the item taken.
            await lead_done.wait()
            yield {0: DeltaToolCall(name='finish_task', json_args='{"item_id": "1", "status": "pending"}')}

        lead_model = scripted(
            call('board_add', content='Fix it'),
            call('spawn_teammate', name='gus', prompt='gus', model='m'),
            call('assign_task', item_id='1', teammate='gus'),
            call('assign_task', item_id='1', teammate='gus'),
            call('finish_task', item_id='9'),
        )
        board = InMemoryClaimablePlanStore()
        async with hosting(
            tmp_path, board, lead_model=lead_model, models={'m': FunctionModel(stream_function=member)}
        ) as host:
            messages = await host.lead_turn('Plan.')
            lead_done.set()
            (gus,) = host.owner.members('conv')

            def handed_back() -> bool:
                changes: list[BoardItemChangedEvent] = host.watch.member_events(BoardItemChangedEvent)
                return any(event.item.status is TaskStatus.pending and event.item.owner is None for event in changes)

            await host.watch.until(handed_back)
            await host.watch.until(lambda: gus.status == 'finished')

            assert tool_returns(messages)[2:] == [
                'Assigned #1 [in_progress] Fix it (owner: gus)',
                '#1 is not available: it may be taken, done, or waiting. Use `board_list`.',
                'No board item #9. Use `board_list`.',
            ]
            assert await board.get_owned_items() == [OwnedPlanItem(id='1', content='Fix it')]
            snapshot = await host.team.snapshot('conv')
            assert [(m.sender, m.recipient) for m in snapshot.messages] == [('lead', 'gus')]

    async def test_board_add_takes_the_next_id_when_another_writer_took_its_own(self, tmp_path: Path) -> None:
        board = _StaleOnce()
        await board.add_item(OwnedPlanItem(id='1', content='Theirs'))
        board.stale = True
        async with hosting(tmp_path, board, lead_model=scripted(call('board_add', content='Mine'))) as host:
            messages = await host.lead_turn('Add mine.')

        assert tool_returns(messages) == ['Added #2 [pending] Mine']


class _StaleOnce(InMemoryClaimablePlanStore):
    """A board whose next read, once `stale`, misses what another writer just added, as a racing `board_add` sees."""

    stale = False

    async def get_owned_items(self) -> list[OwnedPlanItem]:
        if self.stale:
            self.stale = False
            return []
        return await super().get_owned_items()


class TestTeamBinding:
    async def test_tools_and_instructions_follow_the_role(self, tmp_path: Path) -> None:
        seen: dict[str, tuple[list[str], str]] = {}

        def capture(role: str) -> FunctionModel:
            async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                seen[role] = (sorted(tool.name for tool in info.function_tools), info.instructions or '')
                return ModelResponse(parts=[TextPart('ok')])

            async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
                await respond(messages, info)
                yield 'ok'

            return FunctionModel(respond, stream_function=stream)

        board = InMemoryClaimablePlanStore()
        async with hosting(tmp_path, board, lead_model=capture('lead')) as host:
            host.models['m'] = capture('member')
            await host.lead_turn('Hi.')
            await host.team.spawn('conv', name='fay', prompt='fay', model='m')
            (fay,) = host.owner.members('conv')
            await host.watch.settled(fay)
            # Outside the owner's binding, the lead's agent has no team.
            await host.lead.run('Hi.', model=capture('unbound'))

        assert seen['lead'][0] == [
            'assign_task',
            'board_add',
            'board_list',
            'finish_task',
            'review_plan',
            'send_message',
            'shutdown_teammate',
            'spawn_teammate',
        ]
        assert seen['member'][0] == [
            'board_add',
            'board_list',
            'claim_task',
            'finish_task',
            'request_plan_approval',
            'send_message',
        ]
        assert seen['unbound'][0] == []
        assert 'You lead the team.' in seen['lead'][1] and '`self` (default model)' in seen['lead'][1]
        assert 'You are `fay`.' in seen['member'][1]
        assert 'team' not in seen['unbound'][1]

    async def test_one_team_per_agent(self, tmp_path: Path) -> None:
        owner = DelegationTasks(directory=tmp_path)
        team = Team[object](owner, board=InMemoryClaimablePlanStore())
        other = Team[object](DelegationTasks(directory=tmp_path), board=InMemoryClaimablePlanStore())
        # The same team on the agent and again for the run is one team.
        agent = Agent(scripted('ok'), capabilities=[team])
        result = await agent.run('Hi.', capabilities=[team.member_capability('x')])
        assert result.output == 'ok'

        with pytest.raises(UserError, match="Capability id 'team' is used by multiple Team capabilities"):
            await Agent(scripted('ok'), capabilities=[team, other]).run('Hi.')
        assert Team.get_serialization_name() is None
        # No observer: nothing to tell, and a task that is not a member is not on the roster.
        await team.publish(TeammateStateEvent(member=_member(), previous=None))
        await team.observe(
            DelegationTaskEvent(task=DelegationTask(id='t', agent_name='x', prompt='p', conversation_id='c'))
        )
        assert team.default_model('x') is None

    async def test_spawned_and_state_events_are_typed(self) -> None:
        member = _member()
        assert TeammateSpawnedEvent(member=member, prompt='p').truncated is False
        assert TeammateStateEvent(member=member, previous=None).member.current_task is None


def _member() -> TeamMember:
    return TeamMember(name='a', task_id='t', agent_name='self', model=None, state='working')
