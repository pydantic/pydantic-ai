"""Native input ledger and recovery contracts, independent of provider transport timing."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import replace
from typing import Literal

import anyio
import pytest

from pydantic_ai import Agent, SessionStateTypeAdapter, UserError
from pydantic_ai._steering import (
    AcceptSteering,
    CommitSteering,
    LoseSteering,
    ReconcileSteering,
    RejectSteering,
    SendSteering,
    SteeringController,
    SteeringEvent,
    SteeringSent,
    transition,
)
from pydantic_ai.messages import BinaryImage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.session import SessionState, SteeringDelivery


def delivery() -> SteeringDelivery:
    return SteeringDelivery(
        delivery_id='input-1',
        run_id='run-1',
        parent_response_id='response-A',
        messages=[
            ModelRequest(parts=[UserPromptPart(['change direction', BinaryImage(b'\xff\x00', media_type='image/png')])])
        ],
    )


def test_native_delivery_acknowledgement_does_not_commit_history():
    """Pin the pure decision function; an ACK cannot authorize the history-write effect."""
    original = delivery()
    before = deepcopy(original)
    sending, actions = transition(original, SendSteering())
    assert (original.status, sending.status, actions) == ('pending', 'sending', ('send',))
    sent, actions = transition(sending, SteeringSent())
    assert (sent.status, actions) == ('sent', ())
    accepted, actions = transition(sent, AcceptSteering('provider-steer'))
    assert (accepted.status, accepted.provider_id, accepted.successor_response_id, actions) == (
        'accepted',
        'provider-steer',
        None,
        (),
    )
    committed, actions = transition(accepted, CommitSteering('response-B'))
    assert (committed.status, committed.successor_response_id, actions) == (
        'committed',
        'response-B',
        ('record_input',),
    )
    assert transition(committed, CommitSteering('response-B')) == (committed, ())
    assert transition(committed, AcceptSteering('provider-steer')) == (committed, ())
    assert transition(committed, SteeringSent()) == (committed, ())
    assert transition(committed, LoseSteering()) == (committed, ())
    assert original == before


@pytest.mark.parametrize('status', ['sending', 'sent', 'accepted'])
def test_lost_native_delivery_never_authorizes_automatic_replay(status: Literal['sending', 'sent', 'accepted']):
    original = replace(delivery(), status=status, provider_id='provider-steer' if status == 'accepted' else None)
    uncertain, actions = transition(original, LoseSteering())
    assert (uncertain.status, actions) == ('uncertain', ())
    assert transition(uncertain, SteeringSent()) == (uncertain, ())
    late_ack, actions = transition(uncertain, AcceptSteering('provider-steer'))
    assert (late_ack.status, late_ack.provider_id, actions) == ('uncertain', 'provider-steer', ())
    confirmed, actions = transition(late_ack, CommitSteering('response-B'))
    assert (confirmed.status, actions) == ('committed', ('record_input',))
    with pytest.raises(UserError, match='already been attempted'):
        transition(uncertain, SendSteering())


@pytest.mark.parametrize(
    'event', [SendSteering(), SteeringSent(), AcceptSteering('s'), CommitSteering('B'), RejectSteering('error')]
)
def test_reconciled_native_delivery_is_not_reactivated(event: SteeringEvent):
    settled, _ = transition(delivery(), ReconcileSteering('discard'))
    with pytest.raises(UserError):
        transition(settled, event)
    assert transition(settled, LoseSteering()) == (settled, ())


def test_native_delivery_correlates_successor_and_provider_identity():
    accepted = replace(delivery(), status='accepted', provider_id='s')
    with pytest.raises(UserError, match='different provider identity'):
        transition(accepted, AcceptSteering('different'))
    with pytest.raises(UserError, match='not its parent'):
        transition(accepted, CommitSteering('response-A'))
    with pytest.raises(UserError, match='no accepted input'):
        transition(replace(delivery(), status='sent'), CommitSteering('B'))
    committed, _ = transition(accepted, CommitSteering('B'))
    with pytest.raises(UserError, match='no accepted input'):
        transition(committed, CommitSteering('C'))
    with pytest.raises(UserError, match='already been settled'):
        transition(committed, ReconcileSteering('replay'))


@pytest.mark.parametrize('active', [False, True])
@pytest.mark.parametrize('decision', ['replay', 'discard'])
@pytest.mark.parametrize('status', ['pending', 'sending', 'sent', 'accepted', 'uncertain', 'failed'])
async def test_native_delivery_checkpoint_requires_reconciliation(
    active: bool,
    decision: Literal['replay', 'discard'],
    status: Literal['pending', 'sending', 'sent', 'accepted', 'uncertain', 'failed'],
):
    state = SessionState(
        steering=[replace(delivery(), status=status, provider_id='provider-steer' if status == 'accepted' else None)],
        active_run_id='run-1' if active else None,
    )
    state = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(state))
    original = deepcopy(state)
    agent = Agent(TestModel())
    with pytest.raises(UserError, match=r'unfinished run|unresolved'):
        agent.session(state=state)
    if active:
        with pytest.raises(UserError, match='unfinished run'):
            state.recover(steering={'input-1': decision})
    with pytest.raises(UserError, match='unresolved'):
        state.recover(abandon_run=active)
    recovered = state.recover(steering={'input-1': decision}, abandon_run=active)
    assert state == original
    assert recovered.active_run_id is None
    assert recovered.steering[0].status == ('replayed' if decision == 'replay' else 'discarded')
    assert recovered.conversation.messages == []
    recovered = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(recovered))
    if decision == 'replay':
        assert len(recovered.pending) == 1
        assert recovered.pending[0].enqueue_id == 'input-1'
        assert recovered.pending[0].messages == original.steering[0].messages
    else:
        assert recovered.pending == []
    async with agent.session(state=recovered) as session:
        await session.run('resume')
        assert session.state.pending == []
        assert session.state.steering == recovered.steering
        prompts = [p for m in session.conversation.messages for p in m.parts if isinstance(p, UserPromptPart)]
        assert len(prompts) == (2 if decision == 'replay' else 1)
    assert state == original


def test_native_delivery_recovery_is_atomic_and_cannot_replay_twice():
    state = SessionState(steering=[delivery(), replace(delivery(), delivery_id='input-2')])
    original = deepcopy(state)
    with pytest.raises(UserError, match='unresolved'):
        state.recover(steering={'input-1': 'replay'})
    assert state == original
    with pytest.raises(UserError, match='Unknown steering delivery'):
        state.recover(steering={'unknown': 'discard'})
    assert state == original
    recovered = state.recover(steering={'input-1': 'replay', 'input-2': 'discard'})
    with pytest.raises(UserError, match='already been settled'):
        recovered.recover(steering={'input-1': 'replay'})
    assert len(recovered.pending) == 1
    with pytest.raises(UserError, match='duplicate steering delivery IDs'):
        SessionState(steering=[delivery(), delivery()]).recover()
    with pytest.raises(UserError, match='already in the boundary inbox'):
        SessionState(steering=[delivery()], pending=recovered.pending).recover(steering={'input-1': 'replay'})


def test_rejected_native_delivery_keeps_user_input():
    pending = replace(delivery(), status='accepted', provider_id='provider-steer')
    failed, actions = transition(pending, RejectSteering('unsupported model'))
    assert (failed.status, failed.error, failed.messages, actions) == (
        'failed',
        'unsupported model',
        pending.messages,
        (),
    )
    assert transition(failed, SteeringSent()) == (failed, ())
    with pytest.raises(UserError, match='unresolved'):
        SessionState(steering=[failed]).recover()
    assert SessionState(steering=[failed]).recover(steering={'input-1': 'discard'}).steering[0].status == 'discarded'


@pytest.mark.parametrize(
    'messages',
    [[], [ModelResponse([TextPart('assistant')])], [ModelRequest([])], [ModelRequest([SystemPromptPart('system')])]],
)
def test_native_delivery_rejects_non_user_input(messages: list[ModelRequest | ModelResponse]):
    with pytest.raises(UserError, match='nonempty user messages'):
        replace(delivery(), messages=messages)


def test_native_commit_does_not_settle_without_history_effect():
    """A broken parent reference cannot publish a committed checkpoint."""
    accepted = replace(delivery(), status='accepted', provider_id='provider-steer')
    controller = SteeringController({accepted.delivery_id: accepted}, 'run-1', lambda: [])
    controller.delivery_id = accepted.delivery_id
    with pytest.raises(UserError, match='parent response'):
        controller.observe(CommitSteering('response-B'))
    assert controller.deliveries[accepted.delivery_id] == accepted


async def test_native_close_drains_external_sender():
    """Teardown must finish the submission before its socket can serve another run."""
    entered = anyio.Event()
    finished = anyio.Event()
    wrote: list[str] = []
    controller = SteeringController({}, 'run-1', lambda: [])
    controller.parent_response_id = 'response-A'

    async def send(parent: str) -> None:
        entered.set()
        try:
            await anyio.sleep_forever()
            wrote.append(parent)
        finally:
            finished.set()

    async def prepare_send(part: UserPromptPart) -> Callable[[str], Awaitable[None]]:
        return send

    controller.prepare_send = prepare_send

    async def submit() -> None:
        try:
            await controller.steer(['new input'])
        except UserError:
            pass

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(submit)
            await entered.wait()
            await controller.close()
            assert finished.is_set()
    assert not wrote
    assert next(iter(controller.deliveries.values())).status == 'uncertain'
