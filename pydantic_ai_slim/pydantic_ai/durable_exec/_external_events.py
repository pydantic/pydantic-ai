"""Suspend a durable agent run until something outside it resolves a token the run handed out.

Some durable engines let a run create a resume token, hand it to an outside system, and suspend
until that system resolves the token with a payload: AWS Lambda durable functions call it a
callback, Restate an awakeable, Trigger.dev a waitpoint token, Vercel Workflow a hook. A run waiting
on a person (an approval, a review) uses that to stay inside one durable execution instead of ending
and being started again with the person's answer.

An engine that supports it installs an [`ExternalEventWaiter`][pydantic_ai.durable_exec.ExternalEventWaiter]
for each durable agent run with [`external_event_scope`][pydantic_ai.durable_exec.external_event_scope],
and [`wait_for_external_event`][pydantic_ai.durable_exec.wait_for_external_event] calls it. Engines whose
external events are addressed by name rather than by token (Temporal signals, Azure Durable Task
external events, DBOS messages) can implement the same contract by encoding the execution's ID and an
event key in the token they hand out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timedelta

from pydantic import JsonValue

from pydantic_ai._utils import await_maybe, is_async_callable, run_in_executor
from pydantic_ai.exceptions import UserError

from ._operation_backend import durable_unit_scope, in_durable_unit

__all__ = (
    'ExternalEventError',
    'ExternalEventWaiter',
    'external_event_scope',
    'wait_for_external_event',
)


class ExternalEventError(Exception):
    """The external event a run waited for will not arrive.

    Raised by [`wait_for_external_event`][pydantic_ai.durable_exec.wait_for_external_event] when the
    wait's `timeout` elapsed, or the outside system reported a failure instead of resolving the token.
    The message is the engine's reason.
    """


class ExternalEventWaiter(ABC):
    """What a durable engine implements so a run can wait for an external event.

    An engine installs one for each durable agent run with
    [`external_event_scope`][pydantic_ai.durable_exec.external_event_scope]. See
    [Waiting for external events](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/#waiting-for-external-events).
    """

    @abstractmethod
    async def wait(
        self,
        name: str,
        *,
        announce: Callable[[str], Awaitable[None]],
        timeout: timedelta | None,
    ) -> JsonValue:
        """Create a resume token, announce it, and suspend the run until the token is resolved.

        The engine must journal all three parts, so that a replay of the run returns the recorded
        payload without creating a second token or announcing again.

        Args:
            name: A name for the wait in the engine's history, stable across replays.
            announce: Called once with the token, inside a durable unit, so the run can hand the
                token to whoever will resolve it.
            timeout: How long to wait before giving up, or `None` to wait as long as the engine allows.

        Returns:
            The JSON payload the token was resolved with.

        Raises:
            ExternalEventError: The timeout elapsed, or the token was resolved with a failure.
        """


_current_waiter: ContextVar[ExternalEventWaiter | None] = ContextVar('_current_external_event_waiter', default=None)


@contextmanager
def external_event_scope(waiter: ExternalEventWaiter) -> Generator[ExternalEventWaiter]:
    """Install `waiter` for the code inside, typically one durable agent run.

    [`wait_for_external_event`][pydantic_ai.durable_exec.wait_for_external_event] called inside
    the scope suspends through `waiter`.
    """
    token = _current_waiter.set(waiter)
    try:
        yield waiter
    finally:
        _current_waiter.reset(token)


async def wait_for_external_event(
    name: str,
    *,
    announce: Callable[[str], Awaitable[None] | None],
    timeout: timedelta | None = None,
) -> JsonValue:
    """Suspend the durable agent run until something outside it resolves a token, and return the payload.

    The durable engine creates a resume token and calls `announce` with it once, inside a durable
    unit: that is where you hand the token to whoever will resolve it, by posting to Slack, sending
    an email or writing a database row. The run then suspends, without holding compute on engines
    that support it, until the token is resolved with a JSON payload through the engine's API. A
    replay of the run returns the recorded payload without announcing again.

    Only works inside a durable agent run on an engine that implements it, currently AWS Lambda
    durable functions through Pydantic AI Harness's `AWSLambdaDurability`.

    Args:
        name: A name for the wait in the engine's history. Keep it stable across deployments.
        announce: Sync or async function called with the token. A sync function runs in a worker
            thread, like a sync tool. An exception it raises fails the durable unit it runs in,
            which the engine retries according to its policy.
        timeout: How long to wait before giving up, or `None` to wait as long as the engine allows.

    Returns:
        The JSON payload the token was resolved with.

    Raises:
        ExternalEventError: The timeout elapsed, or the token was resolved with a failure.
        UserError: The run is not a durable run on an engine that supports external events, or the
            wait was requested from inside a durable unit, such as a tool call or an `announce` callback.
    """
    check_timeout(timeout)
    if in_durable_unit():
        raise UserError(
            '`wait_for_external_event()` was called from inside a durable unit, such as a tool call or an '
            '`announce` callback. The wait '
            'has to be recorded by the durable run itself, so call it from code that runs in the run, such as '
            'a `handle_deferred_tool_calls` capability hook, or resolve the call with `DurableApprovals`.'
        )
    waiter = _current_waiter.get()
    if waiter is None:
        raise UserError(
            '`wait_for_external_event()` can only be called inside a durable agent run on an engine that '
            'supports waiting for external events, such as AWS Lambda durable functions with '
            '`AWSLambdaDurability` from `pydantic-ai-harness`. Outside a durable run, or on an engine '
            'that does not support it, end the run with `DeferredToolRequests` output instead and start '
            'a new run with the results.'
        )

    async def announce_token(token: str) -> None:
        # `announce` runs inside the engine's durable unit, so mark it as one: a durable operation it
        # calls runs directly, and a wait it starts is rejected above.
        with durable_unit_scope():
            if is_async_callable(announce):
                await await_maybe(announce(token))
            else:
                await await_maybe(await run_in_executor(announce, token))

    return await waiter.wait(name, announce=announce_token, timeout=timeout)


def check_timeout(timeout: timedelta | None) -> None:
    """Reject a `timeout` that is not positive."""
    if timeout is not None and timeout <= timedelta(0):
        raise UserError(f'`timeout` must be positive, got {timeout!r}.')
