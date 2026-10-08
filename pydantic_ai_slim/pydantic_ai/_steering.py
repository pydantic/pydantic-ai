"""Data-only input delivery facts for provider-controlled response successors.

This ledger is independent of tool execution and the boundary inbox. A transport acknowledgement
cannot put user input in history: only the creation of the response consuming it can. Transitions
never send, generate IDs, or replay input; the owning driver executes the returned actions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, assert_never
from uuid import uuid4

import anyio

from ._messages_serialization import MessageHistory
from .exceptions import UserError
from .messages import ModelMessage, ModelRequest, ModelResponse, UserContent, UserPromptPart
from .models.wrapper import WrapperModel
from .usage import RequestUsage, RunUsage, UsageLimits

if TYPE_CHECKING:
    from .models import Model, ModelRequestContext, StreamedResponse


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


class SteeringController:
    """A run-owned native input port; the session owns its portable delivery records.

    No reader task is created here. Sending returns without waiting for acknowledgement, so an
    event consumer can submit input without deadlocking itself. The existing model stream reads
    acknowledgements and each successor. Run teardown fences the port and abandons queued input.
    """

    def __init__(
        self,
        deliveries: dict[str, SteeringDelivery],
        run_id: str,
        history: Callable[[], list[ModelMessage]],
    ) -> None:
        self.deliveries = deliveries
        self.run_id = run_id
        self.history = history
        self.delivery_id: str | None = None
        self.parent_response_id: str | None = None
        self.prepare_send: Callable[[UserPromptPart], Awaitable[Callable[[str], Awaitable[None]]]] | None = None
        self._writers: dict[anyio.CancelScope, anyio.Event] = {}
        self.disconnect: Callable[[], Awaitable[None]] | None = None
        self.receive: Callable[[], AbstractAsyncContextManager[StreamedResponse]] | None = None
        self.inherited_request: ModelRequestContext | None = None
        self.bound_model: Model | None = None
        self.current_usage: Callable[[], RequestUsage] = RequestUsage
        self.usage = RunUsage()
        self.limits = UsageLimits()
        self.closed = False
        self.blocked = False

    @property
    def pending(self) -> bool:
        return self.delivery_id is not None and self.deliveries[self.delivery_id].status in (
            'sending',
            'sent',
            'accepted',
            'uncertain',
        )

    def check_model(self, model: Model) -> None:
        if not self.pending:
            return
        while isinstance(model, WrapperModel):
            model = model.wrapped
        if model is not self.bound_model:
            raise UserError('A native steering continuation must use its original bound model.')

    def observe(self, event: SteeringEvent, *, delivery_id: str | None = None) -> None:
        delivery_id = delivery_id or self.delivery_id
        if delivery_id is None:
            raise UserError('Received a steering observation without an active submission.')
        delivery, actions = transition(self.deliveries[delivery_id], event)
        if 'record_input' in actions:
            history = self.history()
            # At an explicit tool continuation the output request is already in history. Native
            # input is prepended by the provider, so insert it immediately after the parent.
            index = next(
                (
                    i + 1
                    for i in range(len(history) - 1, -1, -1)
                    if isinstance(message := history[i], ModelResponse)
                    and message.provider_response_id == delivery.parent_response_id
                ),
                None,
            )
            if index is None:
                raise UserError('Cannot commit native steering without its parent response in history.')
            history[index:index] = deepcopy(delivery.messages)
        # Publish settlement only after its history effect succeeds.
        self.deliveries[delivery_id] = delivery

    async def steer(self, content: Sequence[UserContent]) -> str:
        finished = anyio.Event()
        with anyio.CancelScope() as scope:
            self._writers[scope] = finished
            try:
                return await self._steer(content)
            finally:
                self._writers.pop(scope)
                finished.set()
        raise UserError('The native steering owner closed before submission completed.')

    async def _steer(self, content: Sequence[UserContent]) -> str:
        if self.blocked:
            raise UserError('Native steering is not supported inside durable execution.')
        if self.closed or self.prepare_send is None or self.parent_response_id is None:
            raise UserError('Native steering requires an active, steering-enabled model response.')
        if self.pending:
            raise UserError('Wait for the current steering submission to commit before submitting another.')
        if not content:
            raise UserError('Native steering requires nonempty user content.')
        if self.limits.count_tokens_before_request:
            raise UserError('Native steering cannot preflight successor input with `count_tokens_before_request=True`.')
        parent = self.parent_response_id
        part = UserPromptPart(deepcopy(list(content)))
        # Mapping may download media. Failures here have not attempted a provider write and
        # must not create an uncertain delivery. Recheck admission after the suspension.
        send = await self.prepare_send(part)
        if self.closed or self.parent_response_id != parent or self.pending:
            raise UserError('The native steering response changed while preparing input; submit it again explicitly.')
        usage = deepcopy(self.usage)
        usage.incr(self.current_usage())  # usage-attribution: provisional admission-check copy
        usage.requests += 1  # usage-attribution: reserve the uncommitted parent on the provisional copy
        self.limits.check_before_request(usage)
        self.limits.check_tokens(usage)
        self.limits.check_cost(usage, warn_if_cost_unavailable=False)
        delivery = SteeringDelivery(
            delivery_id=str(uuid4()),
            run_id=self.run_id,
            parent_response_id=self.parent_response_id,
            messages=[ModelRequest(parts=[part], run_id=self.run_id)],
        )
        self.delivery_id = delivery.delivery_id
        self.deliveries[delivery.delivery_id] = delivery
        self.observe(SendSteering())
        try:
            await send(delivery.parent_response_id)
        except BaseException:
            self.observe(LoseSteering(), delivery_id=delivery.delivery_id)
            if self.disconnect is not None:
                with anyio.CancelScope(shield=True):
                    await self.disconnect()
            raise
        self.observe(SteeringSent(), delivery_id=delivery.delivery_id)
        return delivery.delivery_id

    async def close(self) -> None:
        self.closed = True
        self.prepare_send = None
        # External callers may still be mapping input or writing. Cancel and drain them before
        # releasing the run's ownership; no old writer may reach a later run's connection.
        with anyio.CancelScope(shield=True):
            writers = list(self._writers.items())
            for scope, _ in writers:
                scope.cancel()
            for _, finished in writers:
                await finished.wait()
            try:
                if self.pending:
                    self.observe(LoseSteering())
                    if self.disconnect is not None:
                        await self.disconnect()
            finally:
                # Retained run contexts must not keep a request-scoped model alive after teardown.
                self.inherited_request = None


def require_settled(deliveries: list[SteeringDelivery]) -> None:
    """A detached checkpoint cannot treat socket-local queued input as delivered or retryable."""
    for delivery in deliveries:
        if delivery.status in ('pending', 'sending', 'sent', 'accepted', 'uncertain', 'failed'):
            raise UserError(f'Steering delivery {delivery.delivery_id!r} is unresolved; use `state.recover()` first.')
