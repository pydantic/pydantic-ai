"""Wait for external events inside a DBOS workflow: a step calls `ask`, and a DBOS message resolves the token.

The token names the workflow and a key for the wait, so whoever resolves it sends the workflow a
message on a topic of the wait's own, which the workflow receives with `DBOS.recv_async`.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import NotRequired
from uuid import uuid4

from dbos import DBOS, DBOSClient, WorkflowSerializationFormat
from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from pydantic_ai.exceptions import UserError
from pydantic_ai.external_events import (
    ExternalEventError,
    ExternalEventWaiter,
    _to_json,  # pyright: ignore[reportPrivateUsage]
)

_TOPIC_PREFIX = 'pydantic_ai__external_event:'
"""Prefixes the event key in the topic of the message that resolves a wait."""

_EVENT_KEY = re.compile(r'[0-9a-f]{32}')
"""An event key: the hex form of a UUID a step draws for each wait, so no two waits share one.

It is unique, not secret: the workflow's steps record it.
"""

_NO_TIMEOUT_SECONDS = timedelta(days=365 * 100).total_seconds()
"""How long a wait without a `timeout` lasts: `DBOS.recv_async` takes a finite timeout, which it records as a wake-up time."""


class _ExternalEventMessage(TypedDict):
    """What `resolve_external_event` and `fail_external_event` send to the workflow, in DBOS's portable JSON."""

    payload: NotRequired[JsonValue]
    error: NotRequired[str]


_message_adapter = TypeAdapter(_ExternalEventMessage)


@dataclass
class DBOSExternalEventWaiter(ExternalEventWaiter):
    """Waits for an external event in a DBOS workflow.

    A step draws the wait's event key, and another calls `ask` with the token, so a recovered workflow
    gets both from its recorded steps. Unless `ask` answers, the workflow receives the message
    `resolve_external_event` or `fail_external_event` sends with `DBOS.recv_async`, which records the
    message, or its absence when the timeout elapses.
    """

    agent_name: str

    async def wait(
        self,
        name: str,
        *,
        ask: Callable[[str], Awaitable[JsonValue]],
        timeout: timedelta | None,
    ) -> JsonValue:
        workflow_id = DBOS.workflow_id
        assert workflow_id is not None, 'the waiter is installed only in workflow code'
        step_name = f'{self.agent_name}__external_event.{name}'

        # A coroutine function, so DBOS records the step without blocking the event loop.
        async def draw_event_key() -> str:
            return uuid4().hex

        event_key = await DBOS.run_step_async({'name': step_name}, draw_event_key)
        token = f'{workflow_id}:{event_key}'

        async def call_ask() -> JsonValue:
            return await ask(token)

        answer = await DBOS.run_step_async({'name': f'{step_name}.ask'}, call_ask)
        if answer is not None:
            return answer
        seconds = _NO_TIMEOUT_SECONDS if timeout is None else timeout.total_seconds()
        message: object = await DBOS.recv_async(f'{_TOPIC_PREFIX}{event_key}', timeout_seconds=seconds)
        if message is None:
            raise ExternalEventError(f'No resolution arrived within {timeout}.')
        try:
            outcome = _message_adapter.validate_python(message)
        except ValidationError:
            raise ExternalEventError(
                'The message sent to resolve the wait was not sent by `resolve_external_event` or `fail_external_event`.'
            ) from None
        if 'error' in outcome:
            raise ExternalEventError(outcome['error'])
        return outcome.get('payload')


async def resolve_external_event(token: str, payload: object, *, client: DBOSClient | None = None) -> None:
    """End a wait for an external event in a DBOS workflow with `payload`.

    The token is the one a [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event]
    in a workflow running an agent with [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability]
    handed to its `ask`, such as the handler of
    [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls]. It names the
    workflow, which this sends a DBOS message with the payload. Call it from the endpoint your approval
    UI posts to: in a process that launched DBOS, or anywhere else with a `DBOSClient` for the
    application's system database.

    `payload` is converted to its JSON form, as it would be in process, and sent in DBOS's portable
    JSON format, so the workflow reads it whatever serializer the sender uses. The first resolution
    ends the wait. Once the wait ended, because it was already resolved or timed out, resolving it
    again has no effect.

    Args:
        token: The resume token `ask` received.
        payload: The answer, such as the JSON form of `DeferredToolResults` for `HandleDeferredToolCalls`.
        client: A DBOS client for the application's system database, to send the message from a process
            that didn't launch DBOS. Without one, the message is sent with `DBOS.send_async`.

    Raises:
        UserError: The token was not handed out by a DBOS workflow, or `payload` can't be converted to JSON.
        DBOSNonExistentWorkflowError: No workflow has the ID the token names.
    """
    workflow_id, event_key = _parse_token(token)
    await _send(workflow_id, event_key, {'payload': _to_json(payload)}, client)


async def fail_external_event(token: str, reason: str, *, client: DBOSClient | None = None) -> None:
    """End a wait for an external event in a DBOS workflow without a payload.

    The wait raises [`ExternalEventError`][pydantic_ai.external_events.ExternalEventError] with `reason`,
    as it does when it times out. [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls]
    then denies the pending calls with `reason` in the message to the model.

    Args:
        token: The resume token `ask` received.
        reason: Why no answer is coming, which the wait's `ExternalEventError` carries.
        client: A DBOS client for the application's system database, to send the message from a process
            that didn't launch DBOS. Without one, the message is sent with `DBOS.send_async`.

    Raises:
        UserError: The token was not handed out by a DBOS workflow.
        DBOSNonExistentWorkflowError: No workflow has the ID the token names.
    """
    workflow_id, event_key = _parse_token(token)
    await _send(workflow_id, event_key, {'error': reason}, client)


async def _send(workflow_id: str, event_key: str, message: _ExternalEventMessage, client: DBOSClient | None) -> None:
    topic = f'{_TOPIC_PREFIX}{event_key}'
    portable = WorkflowSerializationFormat.PORTABLE
    if client is None:
        await DBOS.send_async(workflow_id, message, topic, serialization_type=portable)
    else:
        await client.send_async(workflow_id, message, topic, serialization_type=portable)


def _parse_token(token: str) -> tuple[str, str]:
    """The workflow ID and event key a token names."""
    workflow_id, _, event_key = token.rpartition(':')
    if not workflow_id or not _EVENT_KEY.fullmatch(event_key):
        raise UserError('The token was not handed out by a wait for an external event in a DBOS workflow.')
    return workflow_id, event_key
