"""Lifecycle helpers shared by the cloud [sandbox backends][pydantic_ai.sandboxes.SandboxBackend].

Provisioning a cloud environment — or starting a command inside one — is a round trip that can
be cancelled while it is in flight, at which point the caller is gone but the thing it asked for
may already exist and be billing. These helpers give the provider backends the guarantees
[`LocalSandbox`][pydantic_ai.sandboxes.LocalSandbox] has for host subprocesses, without either of
them growing its own copy: nothing the SDK created is ever abandoned, and teardown never masks
the exception it runs alongside.

Deliberately SDK-free, so importing it costs nothing and neither backend's optional dependency
leaks into the other's.
"""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from typing import Any, TypeVar

import anyio

__all__ = ('destroy_quietly', 'guarded_create')

_T = TypeVar('_T')

# Bounds every best-effort teardown: an environment that has stopped answering must not hang the
# block that is trying to let go of it.
_TEARDOWN_TIMEOUT = 30.0

# A cleanup whose caller has already gone away is referenced by nothing else, and the event loop
# only holds weak references to running tasks.
_orphan_cleanups: set[asyncio.Task[None]] = set()


async def guarded_create(create: Coroutine[Any, Any, _T], destroy: Callable[[_T], Awaitable[object]]) -> _T:
    """Await `create`, destroying whatever it produced if the caller is cancelled meanwhile.

    Cancelling the caller of an in-flight `create` would otherwise leave a provisioned — and
    billed — environment behind with nobody holding a handle to it, because the SDK call
    completes on the control plane whether or not anyone is left to receive its result.
    """

    async def attempt() -> _T | Exception:
        try:
            return await create
        except Exception as error:
            # Returned rather than raised: on Python 3.14, `shield` reports an abandoned
            # future's exception to the event loop's exception handler.
            return error

    spawn = asyncio.ensure_future(attempt())
    try:
        outcome = await asyncio.shield(spawn)
    except asyncio.CancelledError:
        spawn.add_done_callback(lambda done: _destroy_orphan(done, destroy))
        raise
    if isinstance(outcome, Exception):
        raise outcome
    return outcome


def _destroy_orphan(spawn: asyncio.Future[Any], destroy: Callable[[Any], Awaitable[object]]) -> None:
    if spawn.cancelled():  # pragma: no cover
        return
    outcome = spawn.result()
    if isinstance(outcome, Exception):
        # Nobody is left to receive the failure, and a create that failed left nothing behind.
        return
    cleanup = asyncio.ensure_future(destroy_quietly(destroy(outcome)))
    _orphan_cleanups.add(cleanup)
    cleanup.add_done_callback(_orphan_cleanups.discard)


async def destroy_quietly(destroy: Awaitable[object]) -> None:
    """Run a teardown so that neither cancellation nor its own failure can escape it.

    Shielded because teardown routinely runs inside an already-cancelled scope, where every
    unshielded await re-raises; bounded so an unreachable environment can't hang the block that
    is letting go of it; and silent because a teardown failure must never mask the exception it
    runs alongside — nor invent one for a block that succeeded.
    """
    with anyio.CancelScope(shield=True), anyio.move_on_after(_TEARDOWN_TIMEOUT), suppress(Exception):
        await destroy
