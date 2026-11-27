"""`/sandbox sprites`: the GitHub token is written into the Sprite, where anything run there can read it.

Sprites have no proxy that could sign requests, so `give_github_login` writes the token into the Sprite's home,
through Sprites' filesystem API, and it goes when the Sprite does. A Sprite keeps its files with no time limit,
and sleeps while idle.

The harness backend sends each command's environment in the exec request's URL, so a key `request_env` sets
reaches Sprites' API in a URL. The GitHub token does not: it goes by the filesystem API.
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
    from sprites import AsyncSpritesClient

    from pydantic_ai_harness.sprites_sandbox import SpritesSandboxBackend

KEY_NAME = 'SPRITE_TOKEN'


@dataclass(frozen=True)
class SpritesProvider:
    """[Sprites](https://sprites.dev) by Fly.io, with the token from `/keys` or the `SPRITE_TOKEN` environment variable."""

    name: str = 'sprites'
    title: str = 'Sprites'
    working_dir: str = '/home/sprite/workspace'
    token_in_sandbox: bool = True
    desktop_at_create: bool = False
    _clients: 'WeakKeyDictionary[SandboxBackend, AsyncSpritesClient]' = field(
        default_factory=WeakKeyDictionary['SandboxBackend', 'AsyncSpritesClient'], init=False, compare=False, repr=False
    )
    """The client made for each backend: `SpritesSandboxBackend` leaves a client it was given open, so `close` closes it."""

    def missing(self, *, desktop: bool = False) -> str | None:
        del desktop
        if find_spec('sprites') is None:
            return "Sprites support is not installed. Install Pax with it: uv tool install 'pydantic-ai-pax[sprites]'."
        if saved_key(KEY_NAME) is None:
            return f'Add your Sprites token in /keys as {KEY_NAME}. Make one at https://sprites.dev/account.'
        return None

    def _client(self) -> 'AsyncSpritesClient':
        from sprites import AsyncSpritesClient

        # The constructor reads `/proc` on Linux the first time, so async callers run it in a thread.
        return AsyncSpritesClient(token=saved_key(KEY_NAME) or '')

    def _backend(
        self, client: 'AsyncSpritesClient', ref: WorkspaceRef | None = None, *, desktop: bool = False
    ) -> 'SpritesSandboxBackend':
        from pydantic_ai_harness.sprites_sandbox import SpritesSandboxBackend

        backend = SpritesSandboxBackend(
            client=client,
            ref=ref,
            working_dir=self.working_dir,
            env=SANDBOX_ENV,
            desktop=desktop,
        )
        self._clients[backend] = client
        return backend

    async def create(
        self, *, github: GitHubCredential | None, desktop: bool = False
    ) -> tuple[WorkspaceRef, SandboxBackend]:
        client = await to_thread.run_sync(self._client)
        backend = self._backend(client, desktop=desktop)
        try:
            await backend.get_sandbox()
        except BaseException:
            # The Sprite may exist already: the backend makes the working folder in it after creating it.
            with anyio.CancelScope(shield=True):
                try:
                    if backend.ref is not None:
                        await self._destroy(client, backend.ref)
                finally:
                    await self.close(backend)
            raise
        assert backend.ref is not None
        return backend.ref, backend

    def reopen(self, ref: WorkspaceRef, *, desktop: bool = False) -> SandboxBackend:
        # Built here, though it reads the key store: the token may be only in `/keys`, which a backend left to
        # open its own client would not read.
        return self._backend(self._client(), ref, desktop=desktop)

    async def close(self, backend: SandboxBackend) -> None:
        if (client := self._clients.pop(backend, None)) is not None:
            await client.aclose()

    async def delete(self, ref: WorkspaceRef) -> None:
        client = await to_thread.run_sync(self._client)
        try:
            await self._destroy(client, ref)
        finally:
            await client.aclose()

    async def _destroy(self, client: 'AsyncSpritesClient', ref: WorkspaceRef) -> None:
        from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

        await SpritesSandbox[None](client=client).destroy(ref)
