"""Internal sandbox lifecycle routing: exactly one capability may supply a run's sandbox."""

import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import cast

import anyio

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef
from pydantic_ai.tools import AgentDepsT

from ._durable_operation import invoke_durable_operation
from .abstract import AbstractCapability, leaf_capabilities
from .wrapper import WrapperCapability


def _hook_is_async(capability: AbstractCapability[AgentDepsT], name: str) -> bool:
    while isinstance(capability, WrapperCapability):
        capability = capability.wrapped
    return inspect.iscoroutinefunction(getattr(capability, name))


def active_leaves(capability: AbstractCapability[AgentDepsT]) -> list[AbstractCapability[AgentDepsT]]:
    """The leaf capabilities that may take part in the run's sandbox lifecycle (deferred ones are inert)."""
    return [leaf for leaf in leaf_capabilities(capability) if leaf.defer_loading is not True]


async def acquire_run_sandbox(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> tuple[AbstractCapability[AgentDepsT], SandboxRef] | None:
    """Ask every active capability for the run's sandbox; exactly one may answer."""
    acquired: list[tuple[AbstractCapability[AgentDepsT], SandboxRef]] = []
    for leaf in active_leaves(capability):
        if _hook_is_async(leaf, 'acquire_sandbox'):
            handler = cast(Callable[..., Awaitable[SandboxRef | None]], leaf.acquire_sandbox)
            ref = await invoke_durable_operation(leaf, 'acquire_sandbox', ctx, handler, (ctx,), {})
        else:
            ref = leaf.acquire_sandbox(ctx)
            if inspect.isawaitable(ref):
                ref = await ref
        if ref is not None:
            acquired.append((leaf, ref))
    if len(acquired) <= 1:
        return acquired[0] if acquired else None
    release_error: Exception | None = None
    for leaf, ref in acquired:
        try:
            await release_run_sandbox(leaf, ctx, ref)
        except Exception as error:
            release_error = release_error or error
    names = ', '.join(type(leaf).__name__ for leaf, _ in acquired)
    raise UserError(f'Exactly one capability may supply the run sandbox; {names} all did.') from release_error


async def release_run_sandbox(
    supplier: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT], ref: SandboxRef
) -> None:
    """Release `ref` through the capability that acquired it, even while the run is being cancelled."""
    with anyio.CancelScope(shield=True):
        if _hook_is_async(supplier, 'release_sandbox'):
            handler = cast(Callable[..., Awaitable[None]], supplier.release_sandbox)
            await invoke_durable_operation(supplier, 'release_sandbox', ctx, handler, (ctx, ref), {})
        elif inspect.isawaitable(result := supplier.release_sandbox(ctx, ref)):
            await result


def connect_sandbox_ref(
    capabilities: Sequence[AbstractCapability[AgentDepsT]], ctx: RunContext[AgentDepsT], ref: SandboxRef
) -> SandboxBackend:
    """Connect to `ref` through `capabilities`; exactly one may answer."""
    connected: list[SandboxBackend] = []
    for capability in capabilities:
        try:
            backend = capability.get_sandbox(ctx, ref)
        except UserError:
            raise
        except Exception as error:
            raise UserError(f'Failed to connect to sandbox {ref.sandbox_id!r}.') from error
        if backend is not None:
            connected.append(backend)
    if len(connected) == 1:
        return connected[0]
    if connected:
        raise UserError(f'Exactly one capability may connect to sandbox {ref.sandbox_id!r}; {len(connected)} did.')
    raise UserError(
        f'No capability can connect to sandbox {ref.sandbox_id!r}: every `get_sandbox` returned `None`. '
        'Attach a capability whose `get_sandbox` recognizes it.'
    )
