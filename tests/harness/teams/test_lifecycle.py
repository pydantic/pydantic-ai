"""A team member's lifecycle: queued turns, graceful shutdown, plan approval, and the team's usage."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from pydantic_ai import FunctionToolResultEvent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.planning import FileClaimablePlanStore, InMemoryClaimablePlanStore, OwnedPlanItem
from pydantic_ai_harness.subagents import DelegationTask, DelegationTaskEvent, DelegationTasks
from pydantic_ai_harness.teams import (
    BoardItemChangedEvent,
    PlanApprovalRequestedEvent,
    PlanReviewedEvent,
    TeammateStateEvent,
)

from .test_team import Host, Reply, call, hosting, last_return, model_of, scripted, tool_returns

pytestmark = pytest.mark.anyio


def states(host: Host, name: str) -> list[tuple[str | None, str, str | None]]:
    """The `(previous, state, shutdown)` of each state event the observer got for the member `name`."""
    return [
        (event.previous, event.member.state, event.member.shutdown)
        for event in host.observed
        if isinstance(event, TeammateStateEvent) and event.member.name == name
    ]


Respond = Callable[[list[ModelMessage]], Awaitable[ModelResponse]]


def async_model(respond: Respond) -> FunctionModel:
    """A model that answers with what `respond` awaits, streamed or not, so it can wait or send several tool calls."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        response = await respond(messages)
        calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
        if calls:
            yield {
                index: DeltaToolCall(
                    name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id
                )
                for index, part in enumerate(calls)
            }
        else:
            yield response.text or ''

    return FunctionModel(stream_function=stream)


def blocking(release: anyio.Event, ran: list[str] | None = None) -> FunctionModel:
    """A member model whose every reply waits for `release`; `ran` records whose requests it got."""

    async def respond(messages: list[ModelMessage]) -> ModelResponse:
        if ran is not None:
            ran.append(_name(messages))
        await release.wait()
        return ModelResponse(parts=[TextPart('done')])

    return async_model(respond)


def _name(messages: list[ModelMessage]) -> str:
    """The member a request belongs to, from the first word of its task: `ben: go.` is ben's."""
    first = messages[0].parts[0]
    assert isinstance(first, UserPromptPart) and isinstance(first.content, str)
    return first.content.split(':')[0]


def _result_of(update: DelegationTaskEvent, tool_call_id: str) -> bool:
    """Whether `update` is a member's tool result for `tool_call_id`."""
    return isinstance(update.event, FunctionToolResultEvent) and update.event.tool_call_id == tool_call_id


async def poll(predicate: Callable[[], bool]) -> None:
    """Wait for something the observer is not told about, such as a model request."""
    with anyio.fail_after(10):
        while not predicate():
            await anyio.sleep(0.01)


def recorder(events: list[Any], *event_types: type[Any]) -> Hooks[object]:
    """A gate that only records the events of `event_types`, from the lead's run and every member's."""
    hooks = Hooks[object]()

    @hooks.on.event(*event_types)
    async def record(ctx: RunContext[object], event: Any) -> None:
        events.append(event)

    return hooks


class TestQueue:
    async def test_members_beyond_max_active_queue_and_start_in_order(self, tmp_path: Path) -> None:
        release = anyio.Event()
        ran: list[str] = []

        def later(messages: list[ModelMessage]) -> Reply:
            ran.append(_name(messages))
            return 'done'

        async with hosting(
            tmp_path,
            InMemoryClaimablePlanStore(),
            lead_model=scripted(),
            models={'slow': blocking(release, ran), 'm': model_of(later)},
        ) as host:
            await host.team.spawn('conv', name='ann', prompt='ann: go.', model='slow')
            await host.team.spawn('conv', name='ben', prompt='ben: go.', model='m')
            await host.team.spawn('conv', name='cat', prompt='cat: go.', model='m')
            ann, ben, cat = host.owner.members('conv')
            await poll(lambda: ran == ['ann'] and ben.queued and cat.queued)

            snapshot = await host.team.snapshot('conv')
            assert [(member.name, member.state) for member in snapshot.members] == [
                ('ann', 'working'),
                ('ben', 'queued'),
                ('cat', 'queued'),
            ]
            release.set()
            for record in (ann, ben, cat):
                await host.watch.settled(record)

            assert ran == ['ann', 'ben', 'cat']
            assert states(host, 'ben') == [
                (None, 'queued', None),
                ('queued', 'working', None),
                ('working', 'idle', None),
            ]
            assert not any(record.queued for record in (ann, ben, cat))

    async def test_raising_max_active_starts_queued_members(self, tmp_path: Path) -> None:
        release = anyio.Event()
        async with hosting(
            tmp_path, InMemoryClaimablePlanStore(), lead_model=scripted(), models={'m': blocking(release)}
        ) as host:
            for name in ('ann', 'ben', 'cat'):
                await host.team.spawn('conv', name=name, prompt=name, model='m')
            ann, ben, cat = host.owner.members('conv')
            await host.watch.until(lambda: ben.queued and cat.queued)

            host.owner.max_active = 2
            await host.watch.until(lambda: not ben.queued)
            assert (host.owner.max_active, cat.queued) == (2, True)
            with pytest.raises(ValueError, match='at least one'):
                host.owner.max_active = 0

            # A queued member the user stops leaves the queue without a turn.
            await host.team.stop('conv', 'cat')
            assert (cat.outcome, cat.queued) == ('cancelled', False)
            host.owner.max_active = None
            release.set()
            await host.watch.settled(ann)
            await host.watch.settled(ben)


class TestShutdown:
    async def test_the_lead_and_the_user_shut_members_down_and_their_claims_go_back(self, tmp_path: Path) -> None:
        def member(messages: list[ModelMessage]) -> Reply:
            name, latest = _name(messages), str(messages[-1])
            if 'Shutdown requested' in latest and name == 'ann':
                # ann wraps up: she hands her item back herself.
                return call('finish_task', item_id='1', status='pending')
            if last_return(messages) is None and 'Shutdown requested' not in latest:
                return call('claim_task')
            return f'{name} is done.'

        board = FileClaimablePlanStore(tmp_path / 'board.json')
        await board.add_item(OwnedPlanItem(id='1', content='One'))
        await board.add_item(OwnedPlanItem(id='2', content='Two'))
        lead_model = scripted(
            call('shutdown_teammate', teammate='ann', reason='Done for today.'),
            call('shutdown_teammate', teammate='nobody'),
            'Shut down.',
        )
        async with hosting(
            tmp_path / 'tasks', board, lead_model=lead_model, models={'m': model_of(member)}, max_active=None
        ) as host:
            await host.team.spawn('conv', name='ann', prompt='ann: claim one.', model='m')
            (ann,) = host.owner.members('conv')
            await host.watch.settled(ann)
            await host.team.spawn('conv', name='ben', prompt='ben: claim one.', model='m')
            _, ben = host.owner.members('conv')
            await host.watch.settled(ben)
            assert [(item.id, item.owner) for item in await board.get_owned_items()] == [('1', 'ann'), ('2', 'ben')]

            messages = await host.lead_turn('Shut ann down.')
            await host.watch.settled(ann, generation=2)
            # ben ignores the request and ends his turn holding #2: the owner hands it back.
            assert await host.team.shutdown('conv', 'ben', reason='Stop.') is True
            await host.watch.settled(ben, generation=2)
            assert ann.shut_down and ben.shut_down

            assert tool_returns(messages) == [
                'Asked ann to wrap up. It is shut down when its turn ends, and its report reaches you.',
                "No team member 'nobody' in this conversation",
            ]
            with pytest.raises(ValueError, match='shut down already'):
                await host.team.shutdown('conv', 'ann')
            assert await board.get_owned_items() == [
                OwnedPlanItem(id='1', content='One'),
                OwnedPlanItem(id='2', content='Two'),
            ]
            # ann handed hers back in her own run; ben's went back through the owner, which the observer hears.
            member_changes: list[BoardItemChangedEvent] = host.watch.member_events(BoardItemChangedEvent)
            assert any(event.item.id == '1' and event.item.owner is None for event in member_changes)
            released = [event for event in host.observed if isinstance(event, BoardItemChangedEvent)]
            assert [(event.item.id, event.previous.owner if event.previous else None) for event in released] == [
                ('2', 'ben')
            ]
            # Woken idle to wrap up, then shut down; a member the user stops shows no `shutdown`.
            assert states(host, 'ann')[-2:] == [('idle', 'working', 'requested'), ('working', 'stopped', 'done')]
            snapshot = await host.team.snapshot('conv')
            assert [(m.name, m.state, m.shutdown) for m in snapshot.members] == [
                ('ann', 'stopped', 'done'),
                ('ben', 'stopped', 'done'),
            ]
            assert [(m.sender, m.recipient) for m in snapshot.messages] == [('lead', 'ann'), ('user', 'ben')]
            assert 'Shutdown requested by lead: Done for today.' in snapshot.messages[0].text
            # Both reports reached the lead.
            assert len(host.owner.lead_inbox('conv')) == 4

            # Only the user wakes a member that was shut down.
            sent = await host.team.send('conv', to='ann', text='Back to work.')
            assert sent.status == 'woken'
            await host.watch.settled(ann, generation=3)
            assert (ann.shut_down, ann.user_stopped) == (False, False)

    async def test_a_member_with_no_turn_to_wrap_up_in_is_shut_down_at_once(self, tmp_path: Path) -> None:
        release = anyio.Event()
        async with hosting(
            tmp_path,
            InMemoryClaimablePlanStore(),
            lead_model=scripted(call('shutdown_teammate', teammate='cat'), 'ok'),
            models={'m': blocking(release), 'quick': scripted('ok')},
        ) as host:
            await host.team.spawn('conv', name='ann', prompt='ann', model='m')
            await host.team.spawn('conv', name='ben', prompt='ben', model='m')
            await host.team.spawn('conv', name='cat', prompt='cat', model='quick')
            ann, ben, cat = host.owner.members('conv')
            await host.watch.until(lambda: ben.queued)

            # Queued: it never gets a turn.
            assert await host.team.shutdown('conv', 'ben') is False
            assert (ben.shut_down, ben.outcome, ben.status) == (True, 'cancelled', 'finished')
            # Stopped by the user: it is shut down as it is.
            release.set()
            await host.watch.settled(ann)
            await host.watch.settled(cat)
            await host.team.stop('conv', 'cat')
            messages = await host.lead_turn('Shut cat down.')
            assert tool_returns(messages) == ['cat was not running, and is shut down.']
            assert cat.shut_down
            with pytest.raises(ValueError, match='shut down already'):
                await host.team.shutdown('conv', 'cat')
            with pytest.raises(ValueError, match='No team member'):
                await host.team.shutdown('conv', 'nobody')

    async def test_a_wrap_up_past_the_grace_period_is_stopped(self, tmp_path: Path) -> None:
        async def respond(messages: list[ModelMessage]) -> ModelResponse:
            if _name(messages) == 'ann' and 'Shutdown requested' in str(messages[-1]):
                await anyio.sleep_forever()
            return ModelResponse(parts=[TextPart('idle')])

        async with hosting(
            tmp_path,
            InMemoryClaimablePlanStore(),
            lead_model=scripted(),
            models={'m': async_model(respond)},
            shutdown_grace=30,
            max_active=None,
        ) as host:
            await host.team.spawn('conv', name='ann', prompt='ann: idle.', model='m')
            await host.team.spawn('conv', name='ben', prompt='ben: idle.', model='m')
            ann, ben = host.owner.members('conv')
            await host.watch.settled(ann)
            await host.watch.settled(ben)
            # ben wraps up well within his grace, which ends with his shutdown.
            assert await host.team.shutdown('conv', 'ben') is True
            await host.watch.settled(ben, generation=2)
            assert (ben.shut_down, ben.outcome) == (True, 'ok')

            host.team.shutdown_grace = 0.05
            assert await host.team.shutdown('conv', 'ann') is True
            with pytest.raises(ValueError, match='already shutting down'):
                await host.team.shutdown('conv', 'ann')
            await host.watch.until(lambda: ann.shut_down)
            assert (ann.outcome, ann.generation) == ('cancelled', 2)

        with pytest.raises(RuntimeError, match='closed'):
            await host.owner.shutdown('Stop.', to=ann.id, conversation_id='conv')

    async def test_a_restart_finishes_a_shutdown_it_interrupted(self, tmp_path: Path) -> None:
        board = InMemoryClaimablePlanStore()
        await board.add_item(OwnedPlanItem(id='1', content='One'))
        await board.claim('1', owner='ann')
        saved = DelegationTasks(directory=tmp_path)
        record = DelegationTask(
            id='a' * 32, name='ann', agent_name='self', prompt='ann', conversation_id='conv', background=True
        )
        record.shutdown_request = 'request'
        await saved.save(record)

        async with hosting(tmp_path, board, lead_model=scripted()) as host:
            (ann,) = host.owner.members('conv')
            assert (ann.shut_down, ann.shutdown_request, ann.status) == (True, None, 'finished')
            assert await board.get_owned_items() == [OwnedPlanItem(id='1', content='One')]
            assert (await host.team.snapshot('conv')).members[0].shutdown == 'done'


def planner(messages: list[ModelMessage]) -> Reply:
    """A member that submits a plan, revises it once rejected, and reports the review it got.

    The review is its tool's result, or, for a run that ended while it waited, a message.
    """
    latest = [part.content for part in messages[-1].parts if isinstance(part, ToolReturnPart | UserPromptPart)]
    review = next((text for text in latest if isinstance(text, str) and text.startswith('Your plan was')), None)
    if review is None:
        return call('request_plan_approval', plan='1. Write it. 2. Test it.')
    if 'rejected' in review:
        return call('request_plan_approval', plan='1. Write it. 2. Test it. 3. Document it.')
    return f'Review: {review}'


class TestPlanApproval:
    async def test_the_lead_approves_a_plan_the_member_waits_on(self, tmp_path: Path) -> None:
        events: list[Any] = []
        lead_model = scripted(
            call('review_plan', teammate='ann', approved=False),
            call('review_plan', teammate='ann', approved=True, feedback='Keep it short.'),
            call('review_plan', teammate='ann', approved=True),
            'Reviewed.',
        )
        async with hosting(
            tmp_path,
            InMemoryClaimablePlanStore(),
            lead_model=lead_model,
            models={'m': model_of(planner)},
            gates=[recorder(events, PlanApprovalRequestedEvent, PlanReviewedEvent)],
        ) as host:
            await host.team.spawn('conv', name='ann', prompt='ann: plan first.', model='m')
            (ann,) = host.owner.members('conv')
            await host.watch.until(lambda: ann.plan_request is not None)
            snapshot = await host.team.snapshot('conv')
            assert snapshot.members[0].plan is not None
            assert snapshot.members[0].plan.plan == '1. Write it. 2. Test it.'
            assert 'Plan approval requested' in str(host.owner.lead_inbox('conv')[0].part.content)

            messages = await host.lead_turn('Review the plan.')
            await host.watch.settled(ann)

            assert tool_returns(messages) == [
                'Say what to change when rejecting a plan',
                'Approved the plan of ann.',
                'ann has no plan waiting for approval',
            ]
            assert ann.output == 'Review: Your plan was approved by the lead. Go ahead. Notes: Keep it short.'
            assert ann.plan_request is None
            requested, reviewed = events
            assert isinstance(requested, PlanApprovalRequestedEvent) and requested.member == 'ann'
            assert isinstance(reviewed, PlanReviewedEvent)
            assert (reviewed.request_id, reviewed.approved, reviewed.reviewer) == (requested.request_id, True, 'lead')

    async def test_the_user_rejects_then_approves_when_the_team_says_so(self, tmp_path: Path) -> None:
        async with hosting(
            tmp_path,
            InMemoryClaimablePlanStore(),
            lead_model=scripted(),
            models={'m': model_of(planner)},
            plan_reviewer='user',
        ) as host:
            await host.team.spawn('conv', name='ann', prompt='ann: plan first.', model='m')
            (ann,) = host.owner.members('conv')
            await host.watch.until(lambda: ann.plan_request is not None)
            first = ann.plan_request
            assert host.owner.lead_inbox('conv') == []
            with pytest.raises(ValueError, match='Say what to change'):
                await host.team.reject('conv', 'ann', ' ')
            await host.team.reject('conv', 'ann', 'Document it too.')
            await host.watch.until(lambda: ann.plan_request is not None and ann.plan_request is not first)
            await host.team.approve('conv', 'ann')
            await host.watch.settled(ann)
            with pytest.raises(ValueError, match='no plan waiting'):
                await host.team.approve('conv', 'ann')

            assert ann.output == 'Review: Your plan was approved by the user. Go ahead.'
            reviews = [event for event in host.observed if isinstance(event, PlanReviewedEvent)]
            assert [(event.approved, event.feedback, event.reviewer) for event in reviews] == [
                (False, 'Document it too.', 'user'),
                (True, '', 'user'),
            ]

    async def test_a_plan_waiting_at_a_restart_is_reviewed_after_it(self, tmp_path: Path) -> None:
        board = InMemoryClaimablePlanStore()
        async with hosting(tmp_path, board, lead_model=scripted(), models={'m': model_of(planner)}) as host:
            await host.team.spawn('conv', name='ann', prompt='ann: plan first.', model='m')
            (ann,) = host.owner.members('conv')
            await host.watch.until(lambda: ann.plan_request is not None)

        async with hosting(tmp_path, board, lead_model=scripted(), models={'m': model_of(planner)}) as host:
            (ann,) = host.owner.members('conv')
            member = (await host.team.snapshot('conv')).members[0]
            assert member.plan is not None and member.plan.plan == '1. Write it. 2. Test it.'
            await host.team.approve('conv', 'ann', 'Go.')
            # Its run ended, so the user's approval wakes it as their words.
            await host.watch.settled(ann, generation=2)
            assert ann.output == 'Review: Your plan was approved by the user. Go ahead. Notes: Go.'

    async def test_a_member_cannot_wait_on_two_plans_or_an_empty_one(self, tmp_path: Path) -> None:
        async def respond(messages: list[ModelMessage]) -> ModelResponse:
            if _name(messages) == 'ben':
                done = len(messages) > 1
                return ModelResponse(
                    parts=[TextPart('done') if done else ToolCallPart('request_plan_approval', {'plan': ' '})]
                )
            if len(messages) == 1:
                # Two at once: the second is refused while the first waits.
                return ModelResponse(
                    parts=[
                        ToolCallPart('request_plan_approval', {'plan': 'A'}, tool_call_id='a'),
                        ToolCallPart('request_plan_approval', {'plan': 'B'}, tool_call_id='b'),
                    ]
                )
            return ModelResponse(parts=[TextPart('done')])

        async with hosting(
            tmp_path,
            InMemoryClaimablePlanStore(),
            lead_model=scripted(),
            models={'m': async_model(respond)},
            max_active=None,
        ) as host:
            await host.team.spawn('conv', name='ann', prompt='ann', model='m')
            await host.team.spawn('conv', name='ben', prompt='ben', model='m')
            ann, ben = host.owner.members('conv')
            # Approved once the second request was refused, while the first waits.
            await host.watch.until(lambda: any(_result_of(update, 'b') for update in host.watch.updates))
            assert ann.plan_request is not None and ann.plan_request.plan == 'A'
            await host.team.approve('conv', 'ann')
            await host.watch.settled(ann)
            await host.watch.settled(ben)

            assert sorted(tool_returns(ann.messages)) == [
                'Your plan already waits for approval.',
                'Your plan was approved by the user. Go ahead.',
            ]
            assert tool_returns(ben.messages) == ['The plan must not be empty.']


class TestUsage:
    async def test_usage_adds_up_per_member_and_for_the_team_across_a_restart(self, tmp_path: Path) -> None:
        lead_model = scripted(
            call('spawn_teammate', name='ann', prompt='ann', model='cheap'),
            call('spawn_teammate', name='ben', prompt='ben', model='mid'),
            'Started.',
        )
        board = InMemoryClaimablePlanStore()
        models = {'cheap': scripted('ann done', name='cheap'), 'mid': scripted(call('board_list'), 'ben done')}
        async with hosting(tmp_path, board, lead_model=lead_model, models=models, max_active=None) as host:
            await host.lead_turn('Start two.')
            ann, ben = host.owner.members('conv')
            await host.watch.settled(ann)
            await host.watch.settled(ben)
            snapshot = await host.team.snapshot('conv')

        lead, (ann_usage, ben_usage) = snapshot.lead_usage, [member.usage for member in snapshot.members]
        assert (lead.requests, ann_usage.requests, ben_usage.requests) == (3, 1, 2)
        assert ann_usage.input_tokens > 0 and ben_usage.output_tokens > 0
        assert snapshot.usage.requests == 6
        assert snapshot.usage.input_tokens == lead.input_tokens + ann_usage.input_tokens + ben_usage.input_tokens

        async with hosting(tmp_path, board, lead_model=scripted()) as host:
            reloaded = await host.team.snapshot('conv')
            assert {member.name: member.usage.requests for member in reloaded.members} == {'ann': 1, 'ben': 2}
            assert (reloaded.lead_usage.requests, host.team.usage('conv').requests) == (3, 6)
            assert host.team.usage('other').requests == 0

    async def test_the_team_budget_stops_members_and_the_lead(self, tmp_path: Path) -> None:
        async with hosting(
            tmp_path,
            InMemoryClaimablePlanStore(),
            lead_model=scripted('never'),
            models={'m': scripted(call('board_list'), call('board_list'), 'done')},
            usage_limits=UsageLimits(request_limit=2),
        ) as host:
            await host.team.spawn('conv', name='ann', prompt='ann', model='m')
            (ann,) = host.owner.members('conv')
            await host.watch.settled(ann)

            assert ann.outcome == 'failed'
            assert 'Team usage limit: The next request would exceed the request_limit of 2' in ann.output
            with pytest.raises(UsageLimitExceeded, match='Team usage limit'):
                await host.lead_turn('Hi.')
            # Without a budget, nothing is checked.
            host.team.usage_limits = None
            host.team.check_usage('conv')
