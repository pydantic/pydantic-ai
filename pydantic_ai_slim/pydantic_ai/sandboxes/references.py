"""Serializable sandbox identity and worker-side reconnection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic_ai.exceptions import UserError

from .protocol import SandboxBackend

__all__ = ('SandboxConnector', 'SandboxRef')


@dataclass(frozen=True, kw_only=True)
class SandboxRef:
    """Serializable identity for an existing sandbox environment."""

    provider: str
    """The provider whose connector can re-open the environment."""

    sandbox_id: str
    """The provider-specific identifier of the existing environment."""


class SandboxConnector(Protocol):
    """Worker-side configuration for re-opening existing sandboxes.

    A connector must fail when `sandbox_id` no longer exists. It must never silently create a
    replacement environment, since that would violate durable-execution identity.
    """

    @property
    def provider(self) -> str:
        """The provider identifier this connector handles."""
        ...

    async def connect(self, sandbox_id: str) -> SandboxBackend:
        """Re-open the existing sandbox identified by `sandbox_id`."""
        ...


async def connect_sandbox_ref(ref: SandboxRef, connectors: Sequence[SandboxConnector]) -> SandboxBackend:
    """Connect a sandbox reference using latest-connector-wins provider resolution."""
    connectors_by_provider = {connector.provider: connector for connector in connectors}
    connector = connectors_by_provider.get(ref.provider)
    if connector is None:
        registered = ', '.join(repr(provider) for provider in connectors_by_provider) or '(none)'
        raise UserError(
            f'No sandbox connector is registered for provider {ref.provider!r}. Registered providers: {registered}.'
        )
    try:
        return await connector.connect(ref.sandbox_id)
    except Exception as error:
        raise UserError(
            f'Failed to connect to sandbox provider {ref.provider!r} for sandbox {ref.sandbox_id!r}.'
        ) from error
