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

_REQUEST_TIMEOUT = 30
"""Seconds `keep_alive` waits for each Daytona request, as the harness backend does for its own."""

SECRET_LABEL = 'clai-github-secret'
"""The sandbox label naming its GitHub secret, so `delete` deletes the secret too."""


@dataclass(frozen=True)
class DaytonaProvider:
    """[Daytona](https://www.daytona.io) sandboxes, with the key from `/keys` or the `DAYTONA_API_KEY` environment variable."""

    name: str = 'daytona'
    title: str = 'Daytona'
    working_dir: str = '/home/daytona/workspace'
    token_in_sandbox: bool = False
    _clients: 'dict[WorkspaceRef, daytona.AsyncDaytona]' = field(
        default_factory=dict['WorkspaceRef', 'daytona.AsyncDaytona'], init=False, compare=False, repr=False
    )
    """The client of each sandbox, shared by its backends: `DaytonaSandboxBackend` leaves a client it was given open."""

    def missing(self) -> str | None:
        if find_spec('daytona') is None:
            return "Daytona support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[daytona]'."
        if saved_key(KEY_NAME) is None:
            return (
                f'Add your Daytona API key in /keys as {KEY_NAME}. Make one at https://app.daytona.io/dashboard/keys.'
            )
        return None

    def _client(self) -> 'daytona.AsyncDaytona':
        import daytona

        # The constructor reads `.env` and `.env.local` in the working folder, so async callers run it in a thread.
        return daytona.AsyncDaytona(daytona.DaytonaConfig(api_key=saved_key(KEY_NAME)))

    async def create(self, *, github: GitHubCredential | None) -> tuple[WorkspaceRef, SandboxBackend]:
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
        ref = WorkspaceRef(provider=self.name, id=sandbox.id)
        self._clients[ref] = client
        return ref, backend

    def reopen(self, ref: WorkspaceRef) -> SandboxBackend:
        from pydantic_ai_harness.daytona_sandbox import DaytonaSandboxBackend

        # Built here, though it reads the key store and two small dotenv files: the key may be only in `/keys`,
        # which a backend left to open its own client would not read.
        if (client := self._clients.get(ref)) is None:
            client = self._clients[ref] = self._client()
        return DaytonaSandboxBackend(client=client, ref=ref, working_dir=self.working_dir, env=SANDBOX_ENV)

    async def keep_alive(self, backend: SandboxBackend) -> None:
        """Reset the idle timer, `auto_stop_interval` (15 minutes by default).

        A running command does not reset it, nor does the backend polling the command's status: measured live, a
        `sleep 240` polled every 0.2s was stopped after 76s with `auto_stop_interval=1`.

        The sandbox is looked up by id with its client, as the backend's `get_sandbox` would start a stopped one.
        """
        import daytona

        assert backend.ref is not None  # Both `create` and `reopen` give backends that name their sandbox.
        sandbox = await self._clients[backend.ref].get(backend.ref.id, request_timeout=_REQUEST_TIMEOUT)
        if sandbox.state == daytona.SandboxState.STARTED:
            await sandbox.refresh_activity(request_timeout=_REQUEST_TIMEOUT)

    async def close(self, ref: WorkspaceRef) -> None:
        if (client := self._clients.pop(ref, None)) is not None:
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
