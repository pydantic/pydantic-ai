"""Data-only input delivery facts for provider-controlled response successors.

This ledger is independent of tool execution and the boundary inbox. A transport acknowledgement
cannot put user input in history: only the creation of the response consuming it can. Transitions
never send, generate IDs, or replay input; the owning driver executes the returned actions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, assert_never

from ._messages_serialization import MessageHistory
from .exceptions import UserError
from .messages import ModelRequest, UserPromptPart


@dataclass(frozen=True, kw_only=True)
class SteeringDelivery:
    """Portable delivery facts for one native steering submission.

    `sent` means the write returned; `accepted` means the provider queued the input, not that it
    consumed it. `committed` identifies the successor which consumed it. A lost connection leaves
    unresolved input `uncertain`, never ready for automatic replay. `replayed` records an explicit
    decision to enqueue the input at an ordinary request boundary, not proof of its consumption.
    """

    delivery_id: str
    run_id: str
    parent_response_id: str
    messages: MessageHistory
    status: Literal[
        'pending', 'sending', 'sent', 'accepted', 'committed', 'failed', 'uncertain', 'replayed', 'discarded'
    ] = 'pending'
    provider_id: str | None = None
    successor_response_id: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.messages or any(
            not isinstance(message, ModelRequest)
            or not message.parts
            or any(not isinstance(part, UserPromptPart) for part in message.parts)
            for message in self.messages
        ):
            raise UserError('Native steering requires nonempty user messages, not system, assistant, or tool parts.')


@dataclass(frozen=True)
class SendSteering:
    pass


@dataclass(frozen=True)
class SteeringSent:
    pass


@dataclass(frozen=True)
class AcceptSteering:
    provider_id: str


@dataclass(frozen=True)
class CommitSteering:
    successor_response_id: str


@dataclass(frozen=True)
class RejectSteering:
    error: str


@dataclass(frozen=True)
class LoseSteering:
    pass


@dataclass(frozen=True)
class ReconcileSteering:
    decision: Literal['replay', 'discard']


SteeringEvent = (
    SendSteering | SteeringSent | AcceptSteering | CommitSteering | RejectSteering | LoseSteering | ReconcileSteering
)
SteeringAction = Literal['send', 'record_input', 'enqueue']


def transition(  # noqa: C901
    delivery: SteeringDelivery, event: SteeringEvent
) -> tuple[SteeringDelivery, tuple[SteeringAction, ...]]:
    """Reduce an observation without mutating the input or executing its effects."""
    if isinstance(event, SendSteering):
        if delivery.status != 'pending':
            raise UserError(f'Steering delivery {delivery.delivery_id!r} has already been attempted.')
        return replace(delivery, status='sending'), ('send',)
    if isinstance(event, SteeringSent):
        # The reader can observe acceptance/commitment while the writer is still returning.
        if delivery.status in ('accepted', 'committed', 'failed', 'uncertain'):
            return delivery, ()
        if delivery.status not in ('sending', 'sent'):
            raise UserError(f'Steering delivery {delivery.delivery_id!r} is not being sent.')
        return replace(delivery, status='sent'), ()
    if isinstance(event, AcceptSteering):
        if delivery.provider_id is not None and delivery.provider_id != event.provider_id:
            raise UserError(f'Steering delivery {delivery.delivery_id!r} has a different provider identity.')
        if delivery.status in ('accepted', 'committed', 'uncertain'):
            return replace(delivery, provider_id=event.provider_id), ()
        if delivery.status not in ('sending', 'sent'):
            raise UserError(f'Steering delivery {delivery.delivery_id!r} has no send to acknowledge.')
        return replace(delivery, status='accepted', provider_id=event.provider_id), ()
    if isinstance(event, CommitSteering):
        if event.successor_response_id == delivery.parent_response_id:
            raise UserError('Steering must commit to a successor, not its parent response.')
        if delivery.status == 'committed' and delivery.successor_response_id == event.successor_response_id:
            return delivery, ()
        if delivery.status not in ('accepted', 'uncertain') or delivery.provider_id is None:
            raise UserError(f'Steering delivery {delivery.delivery_id!r} has no accepted input to commit.')
        return replace(delivery, status='committed', successor_response_id=event.successor_response_id), (
            'record_input',
        )
    if isinstance(event, RejectSteering):
        if delivery.status not in ('sending', 'sent', 'accepted', 'uncertain'):
            raise UserError(f'Steering delivery {delivery.delivery_id!r} cannot be rejected in its current state.')
        return replace(delivery, status='failed', error=event.error), ()
    if isinstance(event, LoseSteering):
        if delivery.status in ('sending', 'sent', 'accepted'):
            return replace(delivery, status='uncertain'), ()
        return delivery, ()
    if isinstance(event, ReconcileSteering):
        # Reconciliation is an application decision made after fencing the old owner. It also
        # accepts captured in-flight states: a stale checkpoint need not have observed the loss.
        if delivery.status not in ('pending', 'sending', 'sent', 'accepted', 'uncertain', 'failed'):
            raise UserError(f'Steering delivery {delivery.delivery_id!r} has already been settled.')
        if event.decision == 'replay':
            return replace(delivery, status='replayed'), ('enqueue',)
        return replace(delivery, status='discarded'), ()
    assert_never(event)


def require_settled(deliveries: list[SteeringDelivery]) -> None:
    """A detached checkpoint cannot treat socket-local queued input as delivered or retryable."""
    for delivery in deliveries:
        if delivery.status in ('pending', 'sending', 'sent', 'accepted', 'uncertain', 'failed'):
            raise UserError(f'Steering delivery {delivery.delivery_id!r} is unresolved; use `state.recover()` first.')
