"""`/sandbox boxd`: boxd's proxy swaps a stand-in in the machine's `GH_TOKEN` for the user's GitHub token.

boxd's proxy reads only secrets, so each machine gets one of its own, made just before it and deleted after
it: the user's (`private-unattached`, which no other org member can see or use), bound to `GITHUB_HOSTS` and
swapped into request headers only, so a command can't post the real token in a request body.
The machine has a variable of the secret's name holding the stand-in, and `create` copies it into `GH_TOKEN`
in the machine's spec, so `gh` and git's credential helper send it after a reopen too.
"""

import uuid
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import anyio

from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import (
    GITHUB_HOSTS,
    SANDBOX_ENV,
    GitHubCredential,
    saved_key,
)
from pydantic_ai_pax.builtin_plugins.sandbox.seed import SandboxBackend

if TYPE_CHECKING:
    import boxd

    from pydantic_ai_harness.boxd_sandbox import BoxdSandboxBackend

KEY_NAME = 'BOXD_API_KEY'

SECRET_PREFIX = 'CLAI_GITHUB_'
"""How `delete` tells Pax's GitHub secrets from the user's others."""

_SCOPE = 'private-unattached'


@dataclass(frozen=True)
class BoxdProvider:
    """[boxd](https://boxd.sh) machines, with the key from `/keys` or the `BOXD_API_KEY` environment variable."""

    name: str = 'boxd'
    title: str = 'boxd'
    working_dir: str = '/home/boxd/workspace'
    token_in_sandbox: bool = False
    desktop_at_create: bool = False
    _clients: 'WeakKeyDictionary[SandboxBackend, boxd.AsyncBoxd]' = field(
        default_factory=WeakKeyDictionary['SandboxBackend', 'boxd.AsyncBoxd'], init=False, compare=False, repr=False
    )
    """The client made for each backend: `BoxdSandboxBackend` leaves a client it was given open, so `close` closes it."""

    def missing(self, *, desktop: bool = False) -> str | None:
        del desktop
        if find_spec('boxd') is None:
            return "boxd support is not installed. Install Pax with it: uv tool install 'pydantic-ai-pax[boxd]'."
        if saved_key(KEY_NAME) is None:
            return f'Add your boxd API key in /keys as {KEY_NAME}. Make one with boxd api-keys create.'
        return None

    def _client(self) -> 'boxd.AsyncBoxd':
        import boxd

        return boxd.AsyncBoxd(api_key=saved_key(KEY_NAME))

    async def create(
        self, *, github: GitHubCredential | None, desktop: bool = False
    ) -> tuple[WorkspaceRef, SandboxBackend]:
        client = self._client()
        secret = None
        backend = None
        try:
            if github is not None:
                # boxd refuses a spec naming a secret that does not exist yet, so the secret comes first.
                secret = f'{SECRET_PREFIX}{uuid.uuid4().hex.upper()}'
                await client.secrets.set(
                    secret, github.token, scope=_SCOPE, domains=list(GITHUB_HOSTS), replace_in=['headers']
                )
            backend = self._backend(client, secrets=None if secret is None else [secret], desktop=desktop)
            machine = await backend.get_sandbox()
            if github is not None and secret is not None:
                stand_in = (await backend.run(['printenv', secret])).stdout.strip()
                if not stand_in or stand_in == github.token:
                    # The token would sit in every command's environment.
                    raise RuntimeError('boxd did not give the machine a stand-in for the GitHub token')
                await client.machines.update_spec(machine.id, {'env': {'GH_TOKEN': stand_in}})
        except BaseException:
            with anyio.CancelScope(shield=True):
                try:
                    if backend is not None and backend.ref is not None:
                        await self.delete(backend.ref)
                    elif secret is not None:
                        await client.secrets.delete(secret, scope=_SCOPE)
                finally:
                    await client.close()
            raise
        return WorkspaceRef(provider=self.name, id=machine.id), backend

    def _backend(
        self, client: 'boxd.AsyncBoxd', *, secrets: list[str] | None = None, desktop: bool = False
    ) -> 'BoxdSandboxBackend':
        from pydantic_ai_harness.boxd_sandbox import BoxdSandboxBackend

        # The backend sets `env` on each command it runs, not on the machine, so each reopen passes it again.
        backend = BoxdSandboxBackend(
            client=client, working_dir=self.working_dir, env=SANDBOX_ENV, secrets=secrets, desktop=desktop
        )
        self._clients[backend] = client
        return backend

    def reopen(self, ref: WorkspaceRef, *, desktop: bool = False) -> SandboxBackend:
        from pydantic_ai_harness.boxd_sandbox import BoxdSandboxBackend

        client = self._client()
        backend = BoxdSandboxBackend(
            client=client, ref=ref, working_dir=self.working_dir, env=SANDBOX_ENV, desktop=desktop
        )
        self._clients[backend] = client
        return backend

    async def desktop_url(self, backend: SandboxBackend) -> str | None:
        """The machine's Desktop view in boxd's console, which opens only in a browser signed in to its owner's boxd."""
        from pydantic_ai_harness.boxd_sandbox import BoxdSandboxBackend

        assert isinstance(backend, BoxdSandboxBackend)
        return (await backend.get_sandbox()).access.desktop_url or None

    async def close(self, backend: SandboxBackend) -> None:
        if (client := self._clients.pop(backend, None)) is not None:
            await client.close()

    async def delete(self, ref: WorkspaceRef) -> None:
        from pydantic_ai_harness.boxd_sandbox import BoxdSandbox

        client = self._client()
        try:
            await BoxdSandbox[None](client=client).destroy(ref)
            # A secret outlives its machine and only the machine named it, so every GitHub secret of Pax's that
            # no machine names goes: this machine's, and any whose machine boxd deleted itself, such as on an idle
            # timer. Another `/sandbox boxd` caught between making its secret and its machine fails, as boxd
            # refuses a machine naming a missing secret.
            for secret in await client.secrets.list():
                if secret.name.startswith(SECRET_PREFIX) and secret.scope == _SCOPE and secret.bound_machines == 0:
                    await client.secrets.delete(secret.name, scope=_SCOPE)
        finally:
            await client.close()
