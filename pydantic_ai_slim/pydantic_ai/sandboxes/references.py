"""Serializable sandbox identity and the provider glue that creates and re-opens sandboxes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai.exceptions import UserError

from .protocol import SandboxBackend

__all__ = ('SandboxProvider', 'SandboxRef')


@dataclass(frozen=True, kw_only=True)
class SandboxRef:
    """Serializable identity for an existing sandbox environment."""

    provider: str
    """The provider that can re-open the environment."""

    sandbox_id: str
    """The provider-specific identifier of the existing environment."""


class SandboxProvider(ABC):
    """Creates, reconnects to, and optionally destroys sandboxes for one provider.

    [`connect`][pydantic_ai.sandboxes.SandboxProvider.connect] is the one required operation;
    [`create`][pydantic_ai.sandboxes.SandboxProvider.create] and
    [`teardown`][pydantic_ai.sandboxes.SandboxProvider.teardown] have default implementations,
    so a provider implements exactly the lifecycle its platform supports.

    Pass a provider to [`ManagedSandbox`][pydantic_ai.sandboxes.ManagedSandbox] to let a run own a
    sandbox's whole lifecycle, and/or register it on a durability capability so a
    [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef] can be re-opened worker-side.
    """

    @property
    @abstractmethod
    def provider(self) -> str:
        """The provider identifier this provider handles.

        Must equal the `provider` its backends report, since that pairing is what
        [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef] resolution matches on.
        """

    @abstractmethod
    async def connect(self, sandbox_id: str) -> SandboxBackend:
        """Re-open the existing sandbox identified by `sandbox_id`.

        Must fail when `sandbox_id` no longer exists. It must never silently create a
        replacement environment, since that would violate durable-execution identity.
        """

    async def create(self) -> SandboxBackend:
        """Provision a fresh sandbox and return a live backend for it.

        Providers whose environments are created out of band — by an operator or another
        service — leave this unimplemented; the sandbox is then supplied to the run directly,
        as a backend or as a [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef].
        """
        raise NotImplementedError(
            f'Sandbox provider {self.provider!r} does not implement `create()`; pass an existing sandbox '
            'backend or a `SandboxRef` to the run instead.'
        )

    async def teardown(self, sandbox_id: str) -> None:
        """Destroy the sandbox identified by `sandbox_id`.

        The default is a no-op: most platforms reap idle sandboxes on their own timeout, which
        is the backstop for every path that cannot run teardown (a killed worker, a cancelled
        workflow). Implementations must tolerate an already-gone sandbox, because teardown also
        runs after a failure that may have destroyed it.
        """


async def connect_sandbox_ref(ref: SandboxRef, providers: Sequence[SandboxProvider]) -> SandboxBackend:
    """Connect a sandbox reference using latest-provider-wins provider resolution."""
    providers_by_name = {provider.provider: provider for provider in providers}
    provider = providers_by_name.get(ref.provider)
    if provider is None:
        registered = ', '.join(repr(name) for name in providers_by_name) or '(none)'
        raise UserError(
            f'No sandbox provider is registered for provider {ref.provider!r}. Registered providers: {registered}.'
        )
    try:
        return await provider.connect(ref.sandbox_id)
    except Exception as error:
        raise UserError(
            f'Failed to connect to sandbox provider {ref.provider!r} for sandbox {ref.sandbox_id!r}.'
        ) from error
