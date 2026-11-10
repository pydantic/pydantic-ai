"""Wait inside an agent run until something outside it resolves a token the run handed out.

A run waiting on input from outside (an approval, an answer, a review) hands a resume token to an
`ask` function. `ask` either answers right away, as a terminal prompt or an automated policy does, or
delivers the token to whoever will answer, by posting a Slack message, sending an email or writing a
database row, and the run waits until code elsewhere resolves the token with the answer.
[`RequestInput`][pydantic_ai.capabilities.RequestInput] is built on this.

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
import threading
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager, suppress
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
        ask: Callable[[str], Awaitable[JsonValue]],
        timeout: timedelta | None,
    ) -> JsonValue:
        """Create a resume token, pass it to `ask`, and unless `ask` answered, suspend the run until the token is resolved.

        The engine must journal every part, so that a replay of the run returns the recorded payload
        without creating a second token or calling `ask` again.

        Args:
            name: A name for the wait in the engine's history, stable across replays.
            ask: Called once with the token, inside a durable unit whose result the engine records.
                It returns the payload when it answered right away, and the wait returns that
                payload without suspending. It returns `None` when it handed the token to whoever
                will resolve it, and the wait suspends until they do.
            timeout: How long to wait before giving up, or `None` to wait as long as the engine allows.

        Returns:
            The payload `ask` returned, or else the JSON payload the token was resolved with.

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
    """Set, with `payload` or `error`, by the one resolver that takes the wait out of the registry."""

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


_pending_events: dict[str, _PendingEvent] = {}
"""The in-process waits that are still pending, by token. Each wait removes its own entry when it ends."""

_pending_events_lock = threading.Lock()
"""Taking a wait out of the registry and settling its outcome is one step, for resolvers on any thread."""


class _InProcessWaiter(ExternalEventWaiter):
    """The waiter used outside a durable run: the token is a random ID, resolved through `resolve_external_event`."""

    async def wait(
        self,
        name: str,
        *,
        ask: Callable[[str], Awaitable[JsonValue]],
        timeout: timedelta | None,
    ) -> JsonValue:
        token = secrets.token_urlsafe(32)
        pending = _PendingEvent.for_current_task()
        _pending_events[token] = pending
        try:
            # The timeout also bounds `ask`, so an answerer that asks a person directly is cancelled with it.
            with anyio.move_on_after(None if timeout is None else timeout.total_seconds()):
                answer = await ask(token)
                if answer is not None:
                    return answer
                await pending.done.wait()
        finally:
            with _pending_events_lock:
                _pending_events.pop(token, None)
        # Read after the entry is gone: a resolver either settled the outcome before that, and was told it
        # succeeded, or found no wait and raised. A resolution that landed as the timeout fired still counts.
        if not pending.resolved:
            raise ExternalEventError(f'No resolution arrived within {timeout}.')
        if pending.error is not None:
            raise pending.error
        return pending.payload


_IN_PROCESS_WAITER = _InProcessWaiter()


async def wait_for_external_event(
    name: str,
    *,
    ask: Callable[[str], Awaitable[object | None] | object | None],
    timeout: timedelta | None = None,
    allow_non_durable_wait: bool = False,
) -> JsonValue:
    """Hand a resume token to `ask`, and return the answer `ask` gives or else the payload the token is resolved with.

    A resume token is created and `ask` is called with it once. `ask` either answers right away, as a
    terminal prompt or an automated policy does, by returning the payload, or hands the token to whoever
    will answer, by posting to Slack, sending an email or writing a database row, and returns `None`. Then
    code you write elsewhere delivers the answer by resolving the token. A payload of `None` can't be
    returned from `ask`; resolve the token with it instead.

    Outside a durable run, the wait happens in this process, and
    [`resolve_external_event`][pydantic_ai.external_events.resolve_external_event] or
    [`fail_external_event`][pydantic_ai.external_events.fail_external_event] ends it. It only lives
    as long as the process does.

    Inside a durable run on an engine that supports it, currently AWS Lambda durable functions through
    Pydantic AI Harness's `AWSLambdaDurability`, the engine records the token, the answer `ask` returns
    and the payload, and suspends the run while it waits, so the wait survives restarts and a replay
    doesn't ask again. The token is then resolved through the engine's API. On a durable engine that
    doesn't support it, a wait in process would not be recorded, so it is refused unless
    `allow_non_durable_wait` is set; that is only detected while the run's context is current, as it is
    in [`RequestInput`][pydantic_ai.capabilities.RequestInput], so call this directly only outside
    durable runs or on an engine that supports it.

    Args:
        name: A name for the wait in a durable engine's history. Keep it stable across deployments.
        ask: Sync or async function called with the token, returning the answer or `None`. A sync
            function runs in a worker thread, like a sync tool. An answer is converted to its JSON form,
            as it is on its way through a durable engine. In a durable run, an exception it raises fails
            the durable unit it runs in, which the engine retries according to its policy, so it must
            return quickly and be safe to call again.
        timeout: How long to wait before giving up, or `None` to wait without a limit (or, in a
            durable run, as long as the engine allows). In process, it also bounds `ask`: an async
            `ask` still running when it elapses is cancelled.
        allow_non_durable_wait: Inside a durable run whose engine doesn't support waiting for external
            events, wait in process instead of raising. The wait is then not recorded: a replay of the
            run calls `ask` again, and a restart loses the wait. It doesn't work on Temporal, whose
            workflow code can't be woken from outside the workflow.

    Returns:
        The JSON payload `ask` returned, or the token was resolved with.

    Raises:
        ExternalEventError: The timeout elapsed, or the token was resolved with a failure.
        UserError: The wait was requested from inside a durable unit, such as a tool call, or, with the
            run's context current and without `allow_non_durable_wait`, inside a durable run on an engine
            that does not support waiting for external events; or `ask` returned a value that can't be
            converted to JSON.
    """
    _check_timeout(timeout)
    if in_durable_unit():
        raise UserError(
            '`wait_for_external_event()` was called from inside a durable unit, such as a tool call or an '
            '`ask` callback. The wait has to be recorded by the durable run itself, so resolve the call '
            'with `RequestInput`, or call it from a `handle_deferred_tool_calls` capability hook only outside '
            'durable runs or on an engine that supports waiting for external events, such as AWS Lambda durable '
            'functions with `AWSLambdaDurability`.'
        )
    waiter = _current_waiter.get()
    if waiter is None:
        run_context = get_current_run_context()
        if run_context is not None and run_context.in_durable_context and not allow_non_durable_wait:
            raise UserError(
                "This agent's durable engine does not support waiting for external events, and waiting in "
                'process would not be recorded, so a recovered run would ask again and a restart would lose '
                'the wait. Use an engine that supports it, such as AWS Lambda durable functions with '
                '`AWSLambdaDurability` from `pydantic-ai-harness`, end the run with `DeferredToolRequests` '
                'output and start a new run with the results, or pass `allow_non_durable_wait=True` to accept '
                'a wait that is not recorded.'
            )
        waiter = _IN_PROCESS_WAITER

    async def ask_with_token(token: str) -> JsonValue:
        # `ask` runs inside the engine's durable unit, so mark it as one: a durable operation it
        # calls runs directly, and a wait it starts is rejected above.
        with durable_unit_scope():
            # A plain bool, so the checker keeps `ask`'s declared return type in both branches.
            is_async: bool = is_async_callable(ask)
            if is_async:
                answer = await await_maybe(ask(token))
            else:
                answer = await await_maybe(await run_in_executor(ask, token))
        return None if answer is None else _to_json(answer)

    return await waiter.wait(name, ask=ask_with_token, timeout=timeout)


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
    _settle(token, payload=_to_json(payload))


def fail_external_event(token: str, reason: str) -> None:
    """End an in-process [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event] without a payload.

    The wait raises [`ExternalEventError`][pydantic_ai.external_events.ExternalEventError] with `reason`,
    as it does when it times out. [`RequestInput`][pydantic_ai.capabilities.RequestInput] then denies
    the pending calls with `reason` in the message to the model.

    Raises:
        UserError: No in-process wait has this token: it is wrong, from another process, or its wait
            already ended because it was resolved, timed out or was cancelled.
    """
    _settle(token, error=ExternalEventError(reason))


def _settle(token: str, *, payload: JsonValue = None, error: ExternalEventError | None = None) -> None:
    """Take the wait for `token` out of the registry and settle its outcome, so that only one resolver ever does."""
    with _pending_events_lock:
        pending = _pending_events.pop(token, None)
        if pending is None:
            raise UserError(
                'No wait in this process has this token: it is wrong, from another process, or its wait already '
                'ended because it was resolved, timed out or was cancelled.'
            )
        pending.payload = payload
        pending.error = error
        pending.resolved = True
    # The outcome is settled, so a wait that ended by timeout or cancellation in the meantime, and whose
    # event loop has since closed, needs no waking. Trio's `RunFinishedError` is a `RuntimeError` too.
    with suppress(RuntimeError):
        pending.wake()


def _to_json(payload: object) -> JsonValue:
    """The JSON form of a payload, as it would arrive through a durable engine."""
    try:
        return to_jsonable_python(payload)
    except PydanticSerializationError as error:
        raise UserError(f'The payload for an external event must be JSON-serializable: {error}') from error


def _check_timeout(timeout: timedelta | None) -> None:
    """Reject a `timeout` that is not positive."""
    if timeout is not None and timeout <= timedelta(0):
        raise UserError(f'`timeout` must be positive, got {timeout!r}.')
