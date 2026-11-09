"""Wait inside an agent run until something outside it resolves a token the run handed out.

A run waiting on a person (an approval, a review) hands a resume token to whoever will answer, by
posting a Slack message, sending an email or writing a database row, and waits until code elsewhere
resolves the token with the answer. [`AwaitApproval`][pydantic_ai.capabilities.AwaitApproval] is built
on this.

By default the wait happens in the current process: the token is a random ID, and
[`resolve_external_event`][pydantic_ai.external_events.resolve_external_event] or
[`fail_external_event`][pydantic_ai.external_events.fail_external_event], called from anywhere in the
same process (such as your approval endpoint), ends it. The wait only lives as long as the process.

Inside a durable run, the durable engine supplies the wait instead, so it survives restarts and
holds no compute while it lasts. Engines whose API hands out tokens map onto it directly: AWS Lambda
durable functions call it a callback, Restate an awakeable, Trigger.dev a waitpoint token, Vercel
Workflow a hook. An engine installs an [`ExternalEventWaiter`][pydantic_ai.external_events.ExternalEventWaiter]
for each durable agent run with [`external_event_scope`][pydantic_ai.external_events.external_event_scope].
Engines whose external events are addressed by name rather than by token (Temporal signals, Azure
Durable Task external events, DBOS messages) can implement the same contract by encoding the
execution's ID and an event key in the token they hand out.
"""

from __future__ import annotations

import asyncio
import secrets
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import timedelta

import anyio
from pydantic import JsonValue
from pydantic_core import PydanticSerializationError, to_jsonable_python

from pydantic_ai._run_context import get_current_run_context
from pydantic_ai._utils import await_maybe, is_async_callable, run_in_executor
from pydantic_ai.durable_exec._operation_backend import durable_unit_scope, in_durable_unit
from pydantic_ai.exceptions import UserError

__all__ = (
    'ExternalEventError',
    'ExternalEventWaiter',
    'external_event_scope',
    'wait_for_external_event',
    'resolve_external_event',
    'fail_external_event',
)


class ExternalEventError(Exception):
    """The external event a run waited for will not arrive.

    Raised by [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event] when the
    wait's `timeout` elapsed, or the token was resolved with a failure instead of a payload. The message
    is the reason.
    """


class ExternalEventWaiter(ABC):
    """What a durable engine implements so a run can wait for an external event durably.

    An engine installs one for each durable agent run with
    [`external_event_scope`][pydantic_ai.external_events.external_event_scope]. See
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

    [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event] called inside
    the scope waits through `waiter` instead of in the current process.
    """
    token = _current_waiter.set(waiter)
    try:
        yield waiter
    finally:
        _current_waiter.reset(token)


@dataclass(eq=False)
class _PendingEvent:
    """One in-process wait: the outcome a resolver sets, and how to wake the waiting task from any thread."""

    done: anyio.Event
    wake: Callable[[], object]
    """Sets `done` on the waiting task's event loop; safe to call from any thread."""
    payload: JsonValue = None
    error: ExternalEventError | None = None
    resolved: bool = False

    @classmethod
    def for_current_task(cls) -> _PendingEvent:
        done = anyio.Event()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Not on asyncio, so on Trio, the only other backend AnyIO runs on.
            import trio

            trio_token = trio.lowlevel.current_trio_token()
            return cls(done=done, wake=lambda: trio_token.run_sync_soon(done.set))
        return cls(done=done, wake=lambda: loop.call_soon_threadsafe(done.set))

    def finish(self) -> None:
        self.resolved = True
        self.wake()


_pending_events: dict[str, _PendingEvent] = {}
"""The in-process waits that are still pending, by token. Each wait removes its own entry when it ends."""


class _InProcessWaiter(ExternalEventWaiter):
    """The waiter used outside a durable run: the token is a random ID, resolved through `resolve_external_event`."""

    async def wait(
        self,
        name: str,
        *,
        announce: Callable[[str], Awaitable[None]],
        timeout: timedelta | None,
    ) -> JsonValue:
        token = secrets.token_urlsafe(32)
        pending = _PendingEvent.for_current_task()
        _pending_events[token] = pending
        try:
            await announce(token)
            with anyio.move_on_after(None if timeout is None else timeout.total_seconds()):
                await pending.done.wait()
        finally:
            _pending_events.pop(token, None)
        # A resolution that landed while the timeout fired still counts: its resolver was told it succeeded.
        if not pending.resolved:
            raise ExternalEventError(f'No resolution arrived within {timeout}.')
        if pending.error is not None:
            raise pending.error
        return pending.payload


_IN_PROCESS_WAITER = _InProcessWaiter()


async def wait_for_external_event(
    name: str,
    *,
    announce: Callable[[str], Awaitable[None] | None],
    timeout: timedelta | None = None,
) -> JsonValue:
    """Wait until something outside the run resolves a token, and return the payload it was resolved with.

    A resume token is created and `announce` is called with it once: that is where you hand the token
    to whoever will resolve it, by posting to Slack, sending an email or writing a database row.
    `announce` does not deliver the answer. Code you write elsewhere does, by resolving the token.

    Outside a durable run, the wait happens in this process, and
    [`resolve_external_event`][pydantic_ai.external_events.resolve_external_event] or
    [`fail_external_event`][pydantic_ai.external_events.fail_external_event] ends it. It only lives
    as long as the process does.

    Inside a durable run on an engine that supports it, currently AWS Lambda durable functions through
    Pydantic AI Harness's `AWSLambdaDurability`, the engine records the token, the announcement and the
    payload, and suspends the run while it waits, so the wait survives restarts and a replay doesn't
    announce again. The token is then resolved through the engine's API.

    Args:
        name: A name for the wait in a durable engine's history. Keep it stable across deployments.
        announce: Sync or async function called with the token. A sync function runs in a worker
            thread, like a sync tool. In a durable run, an exception it raises fails the durable unit
            it runs in, which the engine retries according to its policy.
        timeout: How long to wait before giving up, or `None` to wait without a limit (or, in a
            durable run, as long as the engine allows).

    Returns:
        The JSON payload the token was resolved with.

    Raises:
        ExternalEventError: The timeout elapsed, or the token was resolved with a failure.
        UserError: The wait was requested from inside a durable unit, such as a tool call, or inside a
            durable run on an engine that does not support waiting for external events.
    """
    check_timeout(timeout)
    if in_durable_unit():
        raise UserError(
            '`wait_for_external_event()` was called from inside a durable unit, such as a tool call or an '
            '`announce` callback. The wait has to be recorded by the durable run itself, so call it from code '
            'that runs in the run, such as a `handle_deferred_tool_calls` capability hook, or resolve the call '
            'with `AwaitApproval`.'
        )
    waiter = _current_waiter.get()
    if waiter is None:
        run_context = get_current_run_context()
        if run_context is not None and run_context.in_durable_context:
            raise UserError(
                "This agent's durable engine does not support waiting for external events, and waiting in "
                'process would not be recorded, so a recovered run would announce again and a restart would lose '
                'the wait. Use an engine that supports it, such as AWS Lambda durable functions with '
                '`AWSLambdaDurability` from `pydantic-ai-harness`, or end the run with `DeferredToolRequests` '
                'output and start a new run with the results.'
            )
        waiter = _IN_PROCESS_WAITER

    async def announce_token(token: str) -> None:
        # `announce` runs inside the engine's durable unit, so mark it as one: a durable operation it
        # calls runs directly, and a wait it starts is rejected above.
        with durable_unit_scope():
            if is_async_callable(announce):
                await await_maybe(announce(token))
            else:
                await await_maybe(await run_in_executor(announce, token))

    return await waiter.wait(name, announce=announce_token, timeout=timeout)


def resolve_external_event(token: str, payload: object) -> None:
    """End an in-process [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event] with `payload`.

    Call it from anywhere in the process the run is waiting in, such as the endpoint your approval UI
    posts to, from any thread. A token from a durable engine is resolved through that engine's API
    instead, such as AWS Lambda's `SendDurableExecutionCallbackSuccess`.

    `payload` is converted to its JSON form, as it would be on its way through a durable engine, so the
    wait gets the same value in process as on an engine.

    Raises:
        UserError: `payload` can't be converted to JSON, or no in-process wait has this token: it is
            wrong, from another process, or its wait already ended because it was resolved, timed out
            or was cancelled.
    """
    try:
        json_payload: JsonValue = to_jsonable_python(payload)
    except PydanticSerializationError as error:
        raise UserError(f'The payload for an external event must be JSON-serializable: {error}') from error
    pending = _take_pending(token)
    pending.payload = json_payload
    pending.finish()


def fail_external_event(token: str, reason: str) -> None:
    """End an in-process [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event] without a payload.

    The wait raises [`ExternalEventError`][pydantic_ai.external_events.ExternalEventError] with `reason`,
    as it does when it times out. [`AwaitApproval`][pydantic_ai.capabilities.AwaitApproval] then denies
    the pending calls with `reason` in the message to the model.

    Raises:
        UserError: No in-process wait has this token: it is wrong, from another process, or its wait
            already ended because it was resolved, timed out or was cancelled.
    """
    pending = _take_pending(token)
    pending.error = ExternalEventError(reason)
    pending.finish()


def _take_pending(token: str) -> _PendingEvent:
    """Remove the wait for `token` from the registry, so that only one resolver ever gets it."""
    pending = _pending_events.pop(token, None)
    if pending is None:
        raise UserError(
            'No wait in this process has this token: it is wrong, from another process, or its wait already '
            'ended because it was resolved, timed out or was cancelled.'
        )
    return pending


def check_timeout(timeout: timedelta | None) -> None:
    """Reject a `timeout` that is not positive."""
    if timeout is not None and timeout <= timedelta(0):
        raise UserError(f'`timeout` must be positive, got {timeout!r}.')
