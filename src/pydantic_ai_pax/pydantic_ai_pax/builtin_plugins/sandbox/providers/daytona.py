"""`/sandbox daytona`: Daytona's proxy swaps a stand-in in the sandbox's `GH_TOKEN` for the user's GitHub token.

The token is a secret in the user's Daytona organization, made just before the sandbox, bound to `GITHUB_HOSTS`,
and deleted with the sandbox. The sandbox's `GH_TOKEN` holds Daytona's `dtn_secret_...` stand-in, which the
proxy replaces in requests for those hosts, in any header, git's Basic one included, and turns back into the
stand-in in responses. Proven live with a dummy token: Basic with the stand-in got 200 from httpbin's
basic-auth check, a wrong password 401.
"""

import uuid
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import anyio
from anyio import to_thread

from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import (
    GITHUB_HOSTS,
    SANDBOX_ENV,
    GitHubCredential,
    saved_key,
)
from pydantic_ai_pax.builtin_plugins.sandbox.seed import SandboxBackend

if TYPE_CHECKING:
    import daytona

KEY_NAME = 'DAYTONA_API_KEY'

SECRET_LABEL = 'clai-github-secret'
"""The sandbox label naming its GitHub secret, so `delete` deletes the secret too."""


@dataclass(frozen=True)
class DaytonaProvider:
    """[Daytona](https://www.daytona.io) sandboxes, with the key from `/keys` or the `DAYTONA_API_KEY` environment variable."""

    name: str = 'daytona'
    title: str = 'Daytona'
    working_dir: str = '/home/daytona/workspace'
    token_in_sandbox: bool = False
    desktop_at_create: bool = False
    _clients: 'WeakKeyDictionary[SandboxBackend, daytona.AsyncDaytona]' = field(
        default_factory=WeakKeyDictionary['SandboxBackend', 'daytona.AsyncDaytona'],
        init=False,
        compare=False,
        repr=False,
    )
    """The client made for each backend: `DaytonaSandboxBackend` leaves a client it was given open, so `close` closes it."""

    def missing(self, *, desktop: bool = False) -> str | None:
        del desktop
        if find_spec('daytona') is None:
            return "Daytona support is not installed. Install Pax with it: uv tool install 'pydantic-ai-pax[daytona]'."
        if saved_key(KEY_NAME) is None:
            return (
                f'Add your Daytona API key in /keys as {KEY_NAME}. Make one at https://app.daytona.io/dashboard/keys.'
            )
        return None

    def _client(self) -> 'daytona.AsyncDaytona':
        import daytona

        # The constructor reads `.env` and `.env.local` in the working folder, so async callers run it in a thread.
        return daytona.AsyncDaytona(daytona.DaytonaConfig(api_key=saved_key(KEY_NAME)))

    async def create(
        self, *, github: GitHubCredential | None, desktop: bool = False
    ) -> tuple[WorkspaceRef, SandboxBackend]:
        import daytona

        from pydantic_ai_harness.daytona_sandbox import DaytonaSandboxBackend

        client = await to_thread.run_sync(self._client)
        secret = None
        backend = None
        try:
            if github is not None:
                # Daytona refuses a sandbox naming a secret that does not exist yet, so the secret comes first.
                secret = await client.secret.create(
                    daytona.CreateSecretParams(
                        name=f'CLAI_GITHUB_{uuid.uuid4().hex.upper()}', value=github.token, hosts=list(GITHUB_HOSTS)
                    )
                )
            backend = DaytonaSandboxBackend(
                client=client,
                working_dir=self.working_dir,
                env=SANDBOX_ENV,
                secrets=None if secret is None else {'GH_TOKEN': secret.name},
                desktop=desktop,
            )
            sandbox = await backend.get_sandbox()
            if github is not None and secret is not None:
                stand_in = (await backend.run(['printenv', 'GH_TOKEN'])).stdout.strip()
                if not stand_in or stand_in == github.token:
                    # The token would sit in every command's environment.
                    raise RuntimeError('Daytona did not give the sandbox a stand-in for the GitHub token')
                # `set_labels` replaces them all, so the harness's own label is kept.
                await sandbox.set_labels({**sandbox.labels, SECRET_LABEL: secret.id})
        except BaseException:
            with anyio.CancelScope(shield=True):
                # The secret, which holds the token, first, so it goes even if deleting the sandbox fails.
                try:
                    if secret is not None:
                        await _delete_secret(client, secret.id)
                finally:
                    try:
                        if backend is not None and backend.ref is not None:
                            await self._destroy(client, backend.ref)
                    finally:
                        await client.close()
            raise
        self._clients[backend] = client
        return WorkspaceRef(provider=self.name, id=sandbox.id), backend

    def reopen(self, ref: WorkspaceRef, *, desktop: bool = False) -> SandboxBackend:
        from pydantic_ai_harness.daytona_sandbox import DaytonaSandboxBackend

        # Built here, though it reads the key store and two small dotenv files: the key may be only in `/keys`,
        # which a backend left to open its own client would not read.
        client = self._client()
        backend = DaytonaSandboxBackend(
            client=client, ref=ref, working_dir=self.working_dir, env=SANDBOX_ENV, desktop=desktop
        )
        self._clients[backend] = client
        return backend

    async def close(self, backend: SandboxBackend) -> None:
        if (client := self._clients.pop(backend, None)) is not None:
            await client.close()

    async def delete(self, ref: WorkspaceRef) -> None:
        import daytona

        client = await to_thread.run_sync(self._client)
        try:
            try:
                secret = (await client.get(ref.id)).labels.get(SECRET_LABEL)
            except daytona.DaytonaNotFoundError:
                secret = None
            # The secret goes first: its id is only in the sandbox's label, so a failure here leaves the sandbox
            # for a retry to find it by, rather than the token in the organization with nothing pointing to it.
            if secret is not None:
                await _delete_secret(client, secret)
            await self._destroy(client, ref)
        finally:
            await client.close()

    async def _destroy(self, client: 'daytona.AsyncDaytona', ref: WorkspaceRef) -> None:
        from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox

        await DaytonaSandbox[None](client=client).destroy(ref)


async def _delete_secret(client: 'daytona.AsyncDaytona', secret_id: str) -> None:
    # The secrets API raises its generated client's error, not `DaytonaNotFoundError` (proven live).
    from daytona_api_client_async.exceptions import NotFoundException

    try:
        await client.secret.delete(secret_id)
    except NotFoundException:
        pass
