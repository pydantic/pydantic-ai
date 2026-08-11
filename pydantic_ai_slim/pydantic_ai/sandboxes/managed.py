"""A capability that gives a run ownership of a sandbox's whole lifecycle."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from pydantic_ai._run_context import RunPreparationContext
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import UserError

from .protocol import SandboxBackend
from .references import SandboxProvider

__all__ = ('ManagedSandbox',)


@dataclass
class ManagedSandbox(AbstractCapability[Any]):
    """Provision a sandbox for each run and destroy it when the run ends.

    The run owns the lifecycle: the sandbox is created before any hook sees
    [`ctx.sandbox`][pydantic_ai.tools.RunContext.sandbox] and destroyed after the last one,
    even when the run fails. When a sandbox has to outlive a single run, pass a
    [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef] to the run instead.

    The provider is also published through
    [`get_sandbox_providers`][pydantic_ai.capabilities.AbstractCapability.get_sandbox_providers],
    so durable engines can re-open the sandbox on a worker without a second registration.

    See [a sandbox per run](../sandbox.md#a-sandbox-per-run) for a full example, including
    a custom [`SandboxProvider`][pydantic_ai.sandboxes.SandboxProvider] implementation.
    """

    sandbox_provider: SandboxProvider
    """The provider that creates and destroys the run's sandbox."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        # Providers hold live clients and credentials, so this capability can't be loaded from a spec.
        return None

    async def _create_backend(self) -> SandboxBackend:
        """Provision the run's sandbox. Also called by the durable integrations, so both paths fail with the same error."""
        try:
            return await self.sandbox_provider.create()
        except NotImplementedError as error:
            raise UserError(
                f'The sandbox provider {self.sandbox_provider.provider!r} passed to `ManagedSandbox` does not '
                'implement `create()`, so it cannot provision a sandbox for this run. Implement `create()` on '
                'the provider, or pass an existing sandbox backend or a `SandboxRef` to the run instead.'
            ) from error

    def get_sandbox(self, ctx: RunPreparationContext[Any]) -> AbstractAsyncContextManager[SandboxBackend]:
        @asynccontextmanager
        async def managed_sandbox() -> AsyncGenerator[SandboxBackend]:
            backend = await self._create_backend()
            # Tear down even when the run fails. The default `teardown()` is a no-op, so
            # providers that rely on their platform's idle timeout pay nothing for this.
            try:
                yield backend
            finally:
                await self.sandbox_provider.teardown(backend.sandbox_id)

        return managed_sandbox()

    def get_sandbox_providers(self) -> Sequence[SandboxProvider]:
        return (self.sandbox_provider,)
