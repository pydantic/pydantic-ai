"""`/sandbox render`: the GitHub token is written into the sandbox, where anything run there can read it.

Render's network has no proxy that could sign requests, so `give_github_login` writes the token into the
sandbox's home, and it goes when the sandbox does. A Render sandbox lives at most 24 hours.

Render cannot change a sandbox's environment, so the harness backend passes `env` with each command, inside
the launch argument it sends to Render's API; a key `request_env` sets travels that way too.
"""

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import anyio
from anyio import to_thread

from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import (
    SANDBOX_ENV,
    GitHubCredential,
    saved_key,
)
from pydantic_ai_pax.builtin_plugins.sandbox.seed import SandboxBackend

if TYPE_CHECKING:
    from render import RenderAsync

    from pydantic_ai_harness.render_sandbox import RenderSandboxBackend

KEY_NAME = 'RENDER_API_KEY'
WORKSPACE_KEY_NAME = 'RENDER_WORKSPACE_ID'


@dataclass(frozen=True)
class RenderProvider:
    """[Render](https://render.com/docs/sandboxes) sandboxes, with keys from `/keys` or the environment."""

    name: str = 'render'
    title: str = 'Render'
    # Commands run as `root`; the harness backend makes this folder on a new sandbox.
    working_dir: str = '/root/workspace'
    token_in_sandbox: bool = True
    _clients: 'WeakKeyDictionary[SandboxBackend, RenderAsync]' = field(
        default_factory=WeakKeyDictionary['SandboxBackend', 'RenderAsync'], init=False, compare=False, repr=False
    )
    """The client made for each backend: `RenderSandboxBackend` leaves a client it was given open, so `close` closes it."""

    def missing(self) -> str | None:
        if find_spec('render') is None:
            return "Render support is not installed. Install Pax with it: uv tool install 'pydantic-ai-pax[render]'."
        if saved_key(KEY_NAME) is None or saved_key(WORKSPACE_KEY_NAME) is None:
            return (
                f'Add your Render API key in /keys as {KEY_NAME}, and the tea-... ID of a workspace with sandboxes '
                f'as {WORKSPACE_KEY_NAME}. Make a key as https://render.com/docs/api#1-create-an-api-key says.'
            )
        return None

    def _client(self) -> 'RenderAsync':
        from render import RenderAsync

        # The constructor reads its package's metadata, so async callers run it in a thread.
        return RenderAsync(token=saved_key(KEY_NAME), owner_id=saved_key(WORKSPACE_KEY_NAME))

    def _backend(self, client: 'RenderAsync', ref: WorkspaceRef | None = None) -> 'RenderSandboxBackend':
        from pydantic_ai_harness.render_sandbox import RenderSandboxBackend

        # Render's longest lifetime, 24 hours.
        backend = RenderSandboxBackend(
            client=client, ref=ref, working_dir=self.working_dir, env=SANDBOX_ENV, sandbox_timeout=86_400
        )
        self._clients[backend] = client
        return backend

    async def create(self, *, github: GitHubCredential | None) -> tuple[WorkspaceRef, SandboxBackend]:
        client = await to_thread.run_sync(self._client)
        backend = self._backend(client)
        try:
            await backend.working_dir()  # Creates the sandbox, waits until it runs, and makes `working_dir`.
        except BaseException:
            # The sandbox may exist already: the backend waits for it and makes the working folder after creating it.
            with anyio.CancelScope(shield=True):
                try:
                    if backend.ref is not None:
                        await self._destroy(client, backend.ref)
                finally:
                    await self.close(backend)
            raise
        assert backend.ref is not None
        return backend.ref, backend

    def reopen(self, ref: WorkspaceRef) -> SandboxBackend:
        # Built here, though it reads its package's metadata: the keys may be only in `/keys`, which a backend
        # left to open its own client would not read.
        return self._backend(self._client(), ref)

    async def close(self, backend: SandboxBackend) -> None:
        if (client := self._clients.pop(backend, None)) is not None:
            # Render's client has no close of its own; its connections are in its sandbox API's HTTP client.
            await client.sandboxes.client.get_async_httpx_client().aclose()

    async def delete(self, ref: WorkspaceRef) -> None:
        client = await to_thread.run_sync(self._client)
        try:
            await self._destroy(client, ref)
        finally:
            await client.sandboxes.client.get_async_httpx_client().aclose()

    async def _destroy(self, client: 'RenderAsync', ref: WorkspaceRef) -> None:
        from pydantic_ai_harness.render_sandbox import RenderSandbox

        await RenderSandbox[None](client=client).destroy(ref)
