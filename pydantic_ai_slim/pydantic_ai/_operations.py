"""Tool effect state shared by request-driven and duplex session drivers.

This is a projection, not an execution journal. Durable engines still own scheduling and result
recording. The transition has no clocks, generated IDs, callbacks, or I/O; drivers execute its
actions and feed observations back. In particular, losing a send never makes a tool executable.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal, assert_never

from ._messages_serialization import MessageHistory
from .exceptions import UserError
from .messages import ModelMessage, ModelRequest, RetryPromptPart, ToolCallPart, ToolReturnPart


@dataclass(frozen=True, kw_only=True)
class ToolOperation:
    """Portable facts about one admitted tool call and the delivery of its result.

    `completed` means a normalized result exists, not that the provider received it. `sent` means
    only that the transport returned. `committed` means a completed Model interaction consumed
    the request (possibly a cached durable result), or the provider explicitly confirmed it.
    Neither interruption nor a checkpoint can establish whether an external tool effect happened.
    """

    operation_id: str
    run_id: str | None
    run_step: int
    call: ToolCallPart
    call_index: int = 0
    response_timestamp: datetime | None = None
    execution: Literal['pending', 'running', 'deferred', 'completed', 'interrupted'] = 'pending'
    delivery: Literal['pending', 'ready', 'sending', 'sent', 'accepted', 'committed', 'uncertain', 'abandoned'] = (
        'pending'
    )
    result: MessageHistory = field(default_factory=list[ModelMessage])
    """The normalized return and optional user content, independently of transport delivery."""


@dataclass(frozen=True)
class StartTool:
    pass


@dataclass(frozen=True)
class CompleteTool:
    result: MessageHistory


@dataclass(frozen=True)
class ReconcileTool:
    """Externally verified outcome of an interrupted effect; never execute it again."""

    result: MessageHistory


@dataclass(frozen=True)
class DeferTool:
    pass


@dataclass(frozen=True)
class InterruptTool:
    pass


@dataclass(frozen=True)
class StartDelivery:
    pass


@dataclass(frozen=True)
class ObserveDelivery:
    status: Literal['sent', 'accepted', 'committed']


@dataclass(frozen=True)
class LoseDelivery:
    pass


@dataclass(frozen=True)
class ReconcileDelivery:
    """An explicit external decision, never inferred from a disconnect or timeout."""

    status: Literal['ready', 'committed', 'abandoned']


OperationEvent = (
    StartTool
    | CompleteTool
    | ReconcileTool
    | DeferTool
    | InterruptTool
    | StartDelivery
    | ObserveDelivery
    | LoseDelivery
    | ReconcileDelivery
)
OperationAction = Literal['execute_tool', 'record_result', 'deliver_result']


def transition(  # noqa: C901
    operation: ToolOperation, event: OperationEvent
) -> tuple[ToolOperation, tuple[OperationAction, ...]]:
    """Return the next state and permitted effects without modifying either input."""
    if isinstance(event, StartTool):
        if operation.execution not in ('pending', 'deferred'):
            raise UserError(
                f'Tool operation {operation.operation_id!r} has already started; reconcile it before retrying.'
            )
        return replace(operation, execution='running'), ('execute_tool',)
    if isinstance(event, CompleteTool):
        if operation.execution != 'running':
            raise UserError(f'Tool operation {operation.operation_id!r} is not running.')
        return replace(operation, execution='completed', delivery='ready', result=event.result), ('record_result',)
    if isinstance(event, ReconcileTool):
        if operation.execution not in ('running', 'interrupted'):
            raise UserError(f'Tool operation {operation.operation_id!r} has no unresolved outcome.')
        return replace(operation, execution='completed', delivery='ready', result=event.result), ('record_result',)
    if isinstance(event, DeferTool):
        if operation.execution not in ('pending', 'running', 'deferred'):
            raise UserError(f'Tool operation {operation.operation_id!r} cannot be deferred.')
        return replace(operation, execution='deferred'), ()
    if isinstance(event, InterruptTool):
        if operation.execution == 'running':
            return replace(operation, execution='interrupted'), ()
        return operation, ()
    if isinstance(event, StartDelivery):
        if operation.execution != 'completed' or operation.delivery != 'ready':
            return operation, ()
        return replace(operation, delivery='sending'), ('deliver_result',)
    if isinstance(event, ObserveDelivery):
        # A late weaker acknowledgement cannot undo stronger evidence, including after a loss.
        if (
            operation.delivery == 'committed'
            or operation.delivery == 'accepted'
            and event.status == 'sent'
            or operation.delivery == 'uncertain'
            and event.status != 'committed'
        ):
            return operation, ()
        if operation.delivery not in ('sending', 'sent', 'accepted', 'uncertain'):
            raise UserError(f'Tool operation {operation.operation_id!r} has no delivery to acknowledge.')
        return replace(operation, delivery=event.status), ()
    if isinstance(event, LoseDelivery):
        if operation.delivery in ('sending', 'sent', 'accepted'):
            return replace(operation, delivery='uncertain'), ()
        return operation, ()
    if isinstance(event, ReconcileDelivery):
        if operation.execution != 'completed':
            raise UserError(f'Tool operation {operation.operation_id!r} has no completed result to reconcile.')
        if operation.delivery == 'committed' and event.status != 'committed':
            raise UserError('A committed result cannot be made pending again.')
        return replace(operation, delivery=event.status), ()
    assert_never(event)


def apply(
    operations: dict[str, ToolOperation], operation_id: str, event: OperationEvent
) -> tuple[OperationAction, ...]:
    """Runtime-owned application of the pure transition to its current projection."""
    operations[operation_id], actions = transition(operations[operation_id], event)
    return actions


def admit(
    operations: dict[str, ToolOperation],
    *,
    run_id: str | None,
    run_step: int,
    call: ToolCallPart,
    call_index: int = 0,
    response_timestamp: datetime | None = None,
) -> str:
    # Length-prefix the run ID so even caller-supplied IDs containing separators are unambiguous.
    # The scope and model step distinguish provider call IDs reused in later responses or runs.
    scope = run_id or ''
    operation_id = f'{len(scope)}:{scope}:{run_step}:{call_index}:{call.tool_call_id}'
    if operation_id not in operations:
        operations[operation_id] = ToolOperation(
            operation_id=operation_id,
            run_id=run_id,
            run_step=run_step,
            call=call,
            call_index=call_index,
            response_timestamp=response_timestamp,
        )
    return operation_id


def begin_request_delivery(operations: dict[str, ToolOperation], messages: Sequence[ModelMessage]) -> list[str]:
    """Identify completed results included in a request, without treating old history as new sends."""
    # Compare framework metadata, never arbitrary result payloads: a tool can return an object
    # whose equality is not boolean (e.g. an array). Return timestamps also distinguish repeated
    # provider call IDs on later steps, and survive checkpoint round trips. Request envelopes
    # can lose their run ID when history repair merges adjacent requests (e.g. model retries).
    results = {
        (message.run_id, part.tool_call_id, part.tool_name, part.timestamp)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, (ToolReturnPart, RetryPromptPart))
    }
    selected: list[str] = []
    for operation_id, operation in operations.items():
        if operation.delivery not in ('ready', 'uncertain'):
            continue
        if any(
            (message.run_id, part.tool_call_id, part.tool_name, part.timestamp) in results
            or (None, part.tool_call_id, part.tool_name, part.timestamp) in results
            for message in operation.result
            for part in message.parts
            if isinstance(part, (ToolReturnPart, RetryPromptPart))
        ):
            # Existing request retry policy can resend history. Observe its stronger completion
            # evidence without resetting an uncertain delivery or granting a new tool execution.
            if operation.delivery == 'uncertain' or 'deliver_result' in apply(
                operations, operation_id, StartDelivery()
            ):
                selected.append(operation_id)
    return selected


@contextmanager
def request_delivery(operations: dict[str, ToolOperation], messages: Sequence[ModelMessage]) -> Generator[list[str]]:
    """Leave unconfirmed deliveries uncertain, including streams abandoned without an exception."""
    deliveries = begin_request_delivery(operations, messages)
    try:
        yield deliveries
    finally:
        for operation_id in deliveries:
            apply(operations, operation_id, LoseDelivery())


def commit_request_delivery(operations: dict[str, ToolOperation], deliveries: Sequence[str]) -> None:
    """A completed Model interaction consumed the request; this can also be a cached durable result."""
    for operation_id in deliveries:
        apply(operations, operation_id, ObserveDelivery('committed'))
