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
    including when the run fails. Users never handle a
    [`SandboxRef`][pydantic_ai.sandboxes.SandboxRef]; pass one to the run instead when a sandbox
    has to outlive a single run.

    The provider is also published through
    [`get_sandbox_providers`][pydantic_ai.capabilities.AbstractCapability.get_sandbox_providers],
    so durable engines that carry the sandbox's identity across their I/O boundary can re-open it
    without a second registration.

    ```python {title="managed_sandbox.py" test="skip"}
    from my_sandboxes import SandboxClient

    from pydantic_ai import Agent
    from pydantic_ai.sandboxes import ManagedSandbox, SandboxBackend, SandboxProvider


    class MySandboxProvider(SandboxProvider):
        def __init__(self, client: SandboxClient):
            self.client = client

        @property
        def provider(self) -> str:
            return 'my-sandbox'

        async def connect(self, sandbox_id: str) -> SandboxBackend:
            return await self.client.connect(sandbox_id)

        async def create(self) -> SandboxBackend:
            return await self.client.create()

        async def teardown(self, sandbox_id: str) -> None:
            await self.client.destroy(sandbox_id)


    agent = Agent(
        'anthropic:claude-sonnet-5',
        capabilities=[ManagedSandbox(MySandboxProvider(SandboxClient.from_environment()))],
    )
    ```
    """

    sandbox_provider: SandboxProvider
    """The provider that creates and destroys the run's sandbox."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        # Not spec-loadable: a provider holds live clients and worker-side credentials.
        return None

    async def _create_backend(self) -> SandboxBackend:
        """Provision the run's sandbox, explaining a provider that cannot.

        Shared with the durable integrations, which run creation inside a durable unit rather
        than through `get_sandbox`, so both paths fail the same way.
        """
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
            # Always torn down, including when the run fails: the default `teardown()` is a no-op,
            # so a provider that relies on its platform's idle timeout pays nothing for this.
            try:
                yield backend
            finally:
                await self.sandbox_provider.teardown(backend.sandbox_id)

        return managed_sandbox()

    def get_sandbox_providers(self) -> Sequence[SandboxProvider]:
        return (self.sandbox_provider,)
