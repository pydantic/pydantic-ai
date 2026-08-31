"""Internal sandbox provider selection and lifecycle routing."""

from dataclasses import replace

from pydantic.alias_generators import to_snake

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.sandboxes import SandboxBackend, SandboxRef
from pydantic_ai.tools import AgentDepsT

from ._durable_operation import invoke_durable_operation
from .abstract import AbstractCapability, leaf_capabilities


def capabilities_by_id(
    capability: AbstractCapability[AgentDepsT],
) -> dict[str, AbstractCapability[AgentDepsT]]:
    """Return the capability registry, deriving the same stable run-local IDs as Agent."""
    capabilities = leaf_capabilities(capability)
    explicit_ids: set[str] = set()
    for cap in capabilities:
        if cap.id is None:
            continue
        if cap.id in explicit_ids:
            raise UserError(
                f'Capability id {cap.id!r} is used by multiple capabilities. '
                'Capability ids must be unique within a run.'
            )
        explicit_ids.add(cap.id)
    by_id: dict[str, AbstractCapability[AgentDepsT]] = {}
    for cap in capabilities:
        capability_id = cap.id
        if capability_id is None:
            base_id = to_snake(type(cap).__name__)
            capability_id = base_id
            suffix = 2
            while capability_id in by_id or capability_id in explicit_ids:
                capability_id = f'{base_id}_{suffix}'
                suffix += 1
        by_id[capability_id] = cap
    return by_id


def find_sandbox_provider(
    capability: AbstractCapability[AgentDepsT],
) -> tuple[AbstractCapability[AgentDepsT], str] | None:
    """Return the sole active sandbox provider, rejecting ambiguous trees before side effects."""
    if not capability._has_sandbox_hooks:  # pyright: ignore[reportPrivateUsage]
        return None
    providers = [
        (capability_id, leaf)
        for capability_id, leaf in capabilities_by_id(capability).items()
        if leaf.defer_loading is not True and leaf._has_sandbox_hooks  # pyright: ignore[reportPrivateUsage]
    ]
    if len(providers) > 1:
        raise UserError(
            f'Exactly one capability may provide sandbox hooks; found {len(providers)}: '
            f'{", ".join(repr(provider_id) for provider_id, _ in providers)}.'
        )
    if not providers:
        return None
    provider_id, provider = providers[0]
    return provider, provider_id


def find_sandbox_ref_connector(
    capability: AbstractCapability[AgentDepsT], ref: SandboxRef
) -> AbstractCapability[AgentDepsT]:
    """Resolve and validate the capability that reconnects an explicit sandbox ref."""
    if ref.capability_id is None:
        resolved = find_sandbox_provider(capability)
        if resolved is not None and resolved[0]._has_get_sandbox:  # pyright: ignore[reportPrivateUsage]
            return resolved[0]
        raise UserError(
            f'No capability recognizes the sandbox reference for provider {ref.provider!r} '
            f'(sandbox {ref.sandbox_id!r}). Attach a capability whose `get_sandbox` can connect to it.'
        )

    match = capabilities_by_id(capability).get(ref.capability_id)
    if match is None:
        raise UserError(
            f'Cannot reconnect sandbox {ref.sandbox_id!r}: expected one capability with id '
            f'{ref.capability_id!r}, found 0.'
        )
    if match.defer_loading is True:
        raise UserError(
            f'Cannot reconnect sandbox {ref.sandbox_id!r} through deferred capability '
            f'{ref.capability_id!r}; deferred capabilities cannot provide the run sandbox.'
        )
    if not match._has_get_sandbox:  # pyright: ignore[reportPrivateUsage]
        raise UserError(
            f'Cannot reconnect sandbox {ref.sandbox_id!r} through capability {ref.capability_id!r}: '
            'the capability does not implement `get_sandbox`.'
        )
    return match


async def resolve_run_sandbox(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> tuple[AbstractCapability[AgentDepsT], str, SandboxRef | None] | None:
    """Resolve the tree's sole sandbox provider before invoking any lifecycle hook."""
    resolved = find_sandbox_provider(capability)
    if resolved is None:
        return None
    provider, provider_id = resolved
    ref = await invoke_durable_operation(provider, 'acquire_sandbox', ctx, provider.acquire_sandbox, (ctx,), {})
    if ref is not None:
        ref = replace(ref, capability_id=provider_id)
    return provider, provider_id, ref


async def resolve_sandbox_ref(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT], ref: SandboxRef
) -> SandboxBackend | None:
    """Reconnect a ref through its named capability, or use chain precedence for a legacy ref."""
    connector = find_sandbox_ref_connector(capability, ref)
    return await connector.get_sandbox(ctx, ref)


async def connect_sandbox_provider(
    capability: AbstractCapability[AgentDepsT], ctx: RunContext[AgentDepsT], capability_id: str
) -> SandboxBackend | None:
    """Connect a provider-only sandbox through the exact capability selected at run setup."""
    match = capabilities_by_id(capability).get(capability_id)
    if match is None:
        raise UserError(
            f'Cannot connect the capability-provided sandbox: expected one capability with id '
            f'{capability_id!r}, found 0.'
        )
    if match.defer_loading is True:
        raise UserError(
            f'Cannot connect the capability-provided sandbox through deferred capability '
            f'{capability_id!r}; deferred capabilities cannot provide the run sandbox.'
        )
    return await match.get_sandbox(ctx, None)
