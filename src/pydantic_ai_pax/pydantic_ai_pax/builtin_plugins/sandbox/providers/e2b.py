"""`/sandbox e2b`: the sandbox's network rules have E2B's proxy set GitHub's `Authorization` header.

The token is in the rules as written, so E2B holds it until the sandbox is deleted: a sandbox that times out
is paused, rules and all. Until then, anyone with an API key for the E2B team can read the rules; an E2B secret
would let them use the token just the same, from a sandbox of their own, and would outlive this one.

E2B makes the sandbox's URLs public unless told otherwise (`allow_public_traffic`), so every sandbox, with or
without a GitHub login, is told otherwise: a server started in it answers only requests that carry its access token.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING

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
    import e2b
    from e2b import SandboxNetworkOpts

KEY_NAME = 'E2B_API_KEY'

LIFETIME = 3_600
"""Seconds until E2B pauses a sandbox, counted from when it was made, reopened, or last kept alive.

The most E2B's Hobby plan allows, and the harness backend's default.
"""


@dataclass(frozen=True)
class E2BProvider:
    """[E2B](https://e2b.dev) sandboxes, with the key from `/keys` or the `E2B_API_KEY` environment variable."""

    name: str = 'e2b'
    title: str = 'E2B'
    working_dir: str = '/home/user/workspace'
    token_in_sandbox: bool = False

    def missing(self) -> str | None:
        if find_spec('e2b') is None:
            return "E2B support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[e2b]'."
        if saved_key(KEY_NAME) is None:
            return f'Add your E2B API key in /keys as {KEY_NAME}. Get one at https://e2b.dev/dashboard?tab=keys.'
        return None

    def _client(self) -> 'e2b.E2B':
        import e2b

        return e2b.E2B(api_key=saved_key(KEY_NAME))

    async def create(self, *, github: GitHubCredential | None) -> tuple[WorkspaceRef, SandboxBackend]:
        from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

        env: Mapping[str, str] = SANDBOX_ENV
        network: SandboxNetworkOpts = {'allow_public_traffic': False}
        if github is not None:
            network['rules'] = {
                host: [{'transform': {'headers': {'Authorization': github.authorization(host)}}}]
                for host in GITHUB_HOSTS
            }
            env = {**env, 'GH_TOKEN': 'placeholder'}  # `gh` sends a header only with a token set; the proxy replaces it
        # E2B keeps `env` on the sandbox, so commands after a reopen get it too.
        backend = E2BSandboxBackend(
            client=self._client(), working_dir=self.working_dir, env=env, network=network, sandbox_timeout=LIFETIME
        )
        try:
            sandbox = await backend.get_sandbox()
        except BaseException:
            # The backend records a sandbox before setting it up, so one whose setup failed is deleted here,
            # taking the token in its rules with it.
            if backend.ref is not None:
                with anyio.CancelScope(shield=True):
                    await self.delete(backend.ref)
            raise
        return WorkspaceRef(provider=self.name, id=sandbox.sandbox_id), backend

    def reopen(self, ref: WorkspaceRef) -> SandboxBackend:
        from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

        return E2BSandboxBackend(client=self._client(), ref=ref, working_dir=self.working_dir, sandbox_timeout=LIFETIME)

    async def keep_alive(self, backend: SandboxBackend) -> None:
        """Push the pause back to `LIFETIME` from now, as E2B pauses a sandbox then even while a command runs.

        E2B caps it at the plan's longest run from when the sandbox started or resumed: 24 hours on Pro, and on
        Hobby the hour it already had, so there this changes nothing. Only for a running sandbox: E2B refuses
        the timeout of a paused one and leaves it paused, but asking first keeps that from depending on E2B.
        """
        import e2b

        assert backend.ref is not None  # Both `create` and `reopen` give backends that name their sandbox.
        sandboxes = self._client().AsyncSandbox
        if (await sandboxes.get_info(backend.ref.id)).state == e2b.SandboxState.RUNNING:
            await sandboxes.set_timeout(backend.ref.id, LIFETIME)

    async def close(self, backend: SandboxBackend) -> None:
        del backend  # `e2b.E2B` holds no open connection between requests.

    async def delete(self, ref: WorkspaceRef) -> None:
        from pydantic_ai_harness.e2b_sandbox import E2BSandbox

        await E2BSandbox[None](client=self._client()).destroy(ref)
