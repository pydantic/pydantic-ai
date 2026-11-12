"""Wait for external events inside a Temporal workflow: a local activity calls `ask`, and a signal resolves the token.

The token names the workflow and a key for the wait, so whoever resolves it sends the workflow a signal
without the workflow declaring one: the waiter registers the signal's handler on the workflow the first
time it waits.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta

from pydantic import JsonValue
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.exceptions import ApplicationError
from temporalio.workflow import LocalActivityConfig

from pydantic_ai.exceptions import UserError
from pydantic_ai.external_events import (
    ExternalEventError,
    ExternalEventWaiter,
    _to_json,  # pyright: ignore[reportPrivateUsage]
)

from ._activity_execution import execute_local_activity
from ._toolset import non_retryable_subclasses_as_application_errors

_SIGNAL_NAME = 'pydantic_ai__external_event'
"""The signal `resolve_external_event` and `fail_external_event` send, which the waiter handles on every workflow it waits in."""

_EVENT_KEY = re.compile(r'[0-9a-f]{32}')
"""An event key: the hex form of a UUID the workflow draws for each wait, so no two waits share one, even across executions.

It is unique, not secret: the workflow draws it from its replay-safe random source, and the history records it.
"""

AskActivity = Callable[[str, str], Awaitable[JsonValue]]

_pending_asks: dict[str, Callable[[str], Awaitable[JsonValue]]] = {}
"""The `ask` callbacks of the waits in progress on this worker, by workflow run and event key.

`ask` is a closure over the live run, which can't cross a process boundary, so it runs in a local
activity: Temporal runs a local activity in the worker process that runs the workflow task scheduling
it, where the waiter left the callback. The local activity's result is recorded in the workflow's
history, so a replay returns it without calling `ask` again.
"""


@dataclass
class ExternalEventSignal:
    """What `resolve_external_event` and `fail_external_event` send to the workflow."""

    event_key: str
    payload: JsonValue = None
    error: str | None = None


class _Inbox:
    """The outcomes signalled to one workflow execution, by event key; registered as the workflow's handler for the signal."""

    def __init__(self) -> None:
        self.waiting: set[str] = set()
        """The event keys of the waits in progress, the only ones whose outcomes are kept."""
        self.outcomes: dict[str, ExternalEventSignal] = {}

    def __call__(self, signal: ExternalEventSignal) -> None:
        # The first outcome wins, as in process: the wait ends with it, and a later one, or one for a wait
        # that already ended, changes nothing.
        if signal.event_key in self.waiting:
            self.outcomes.setdefault(signal.event_key, signal)

    @classmethod
    def for_current_workflow(cls) -> _Inbox:
        """The inbox of the current workflow, registering its signal handler the first time.

        Temporal delivers a signal that arrived before the handler was registered once it is, so
        registering on the first wait, before `ask` hands the token out, misses none.
        """
        # Temporal types a handler as a bare `Callable`; only whether it is this workflow's inbox matters here.
        handler: object = workflow.get_signal_handler(_SIGNAL_NAME)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(handler, cls):
            return handler
        inbox = cls()
        workflow.set_signal_handler(_SIGNAL_NAME, inbox)  # pyright: ignore[reportUnknownMemberType]
        return inbox


def ask_activity(agent_name: str) -> AskActivity:
    """The local activity that calls a wait's `ask`, registered with the agent's other activities."""

    async def external_event_ask(pending_key: str, token: str) -> JsonValue:
        ask = _pending_asks.get(pending_key)
        # The wait that scheduled this local activity, in this process, holds the callback until it ends, so it
        # is missing only when the workflow sandbox re-imported `pydantic_ai` and holds its own copy of the
        # registry, which no retry can fix.
        if ask is None:  # pragma: no cover
            raise ApplicationError(
                "The wait for an external event that scheduled this activity isn't in this worker's registry: the "
                "worker's workflow sandbox must pass `pydantic_ai` through. Configure the worker with "
                '`PydanticAIPlugin` or `AgentPlugin`, which do.',
                non_retryable=True,
            )
        with non_retryable_subclasses_as_application_errors():
            return await ask(token)

    return activity.defn(name=f'agent__{agent_name}__external_event_ask')(external_event_ask)


@dataclass
class TemporalExternalEventWaiter(ExternalEventWaiter):
    """Waits for an external event in a Temporal workflow.

    `ask` runs in a local activity whose result the history records. Unless it answers, the workflow
    waits with `workflow.wait_condition` until the token is resolved through `resolve_external_event`
    or `fail_external_event`, which signal the workflow.
    """

    ask_activity: AskActivity
    ask_config: LocalActivityConfig

    async def wait(
        self,
        name: str,
        *,
        ask: Callable[[str], Awaitable[JsonValue]],
        timeout: timedelta | None,
    ) -> JsonValue:
        info = workflow.info()
        event_key = workflow.uuid4().hex
        token = f'{info.workflow_id}:{event_key}'
        pending_key = f'{info.run_id}:{event_key}'
        inbox = _Inbox.for_current_workflow()
        inbox.waiting.add(event_key)
        _pending_asks[pending_key] = ask
        try:
            config: LocalActivityConfig = {**self.ask_config, 'summary': f'ask: {name}'}
            answer: JsonValue = await execute_local_activity(self.ask_activity, args=[pending_key, token], **config)
            if answer is not None:
                return answer
            try:
                await workflow.wait_condition(
                    lambda: event_key in inbox.outcomes, timeout=timeout, timeout_summary=name
                )
            except TimeoutError:
                raise ExternalEventError(f'No resolution arrived within {timeout}.') from None
            outcome = inbox.outcomes[event_key]
        finally:
            del _pending_asks[pending_key]
            inbox.waiting.discard(event_key)
            inbox.outcomes.pop(event_key, None)
        if outcome.error is not None:
            raise ExternalEventError(outcome.error)
        return outcome.payload


async def resolve_external_event(client: Client, token: str, payload: object) -> None:
    """End a wait for an external event in a Temporal workflow with `payload`.

    The token is the one a [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event]
    in a workflow running an agent with
    [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability] handed to its `ask`, such
    as [`RequestInput`][pydantic_ai.capabilities.RequestInput]'s. It names the workflow, which this signals
    with the payload. Call it from anywhere with a client for the workflow's namespace, such as the
    endpoint your approval UI posts to.

    `payload` is converted to its JSON form, as it would be in process. Once the wait ended, because it
    was already resolved or timed out, resolving it again has no effect.

    Args:
        client: A Temporal client for the namespace the workflow runs in.
        token: The resume token `ask` received.
        payload: The answer, such as the JSON form of `DeferredToolResults` for `RequestInput`.

    Raises:
        UserError: The token was not handed out by a Temporal workflow, or `payload` can't be converted to JSON.
        temporalio.service.RPCError: The workflow is not running, for example because it already completed.
    """
    workflow_id, event_key = _parse_token(token)
    signal = ExternalEventSignal(event_key=event_key, payload=_to_json(payload))
    await client.get_workflow_handle(workflow_id).signal(_SIGNAL_NAME, signal)


async def fail_external_event(client: Client, token: str, reason: str) -> None:
    """End a wait for an external event in a Temporal workflow without a payload.

    The wait raises [`ExternalEventError`][pydantic_ai.external_events.ExternalEventError] with `reason`,
    as it does when it times out. [`RequestInput`][pydantic_ai.capabilities.RequestInput] then denies the
    pending calls with `reason` in the message to the model.

    Args:
        client: A Temporal client for the namespace the workflow runs in.
        token: The resume token `ask` received.
        reason: Why no answer is coming, which the wait's `ExternalEventError` carries.

    Raises:
        UserError: The token was not handed out by a Temporal workflow.
        temporalio.service.RPCError: The workflow is not running, for example because it already completed.
    """
    workflow_id, event_key = _parse_token(token)
    await client.get_workflow_handle(workflow_id).signal(
        _SIGNAL_NAME, ExternalEventSignal(event_key=event_key, error=reason)
    )


def _parse_token(token: str) -> tuple[str, str]:
    """The workflow ID and event key a token names."""
    workflow_id, _, event_key = token.rpartition(':')
    if not workflow_id or not _EVENT_KEY.fullmatch(event_key):
        raise UserError('The token was not handed out by a wait for an external event in a Temporal workflow.')
    return workflow_id, event_key
