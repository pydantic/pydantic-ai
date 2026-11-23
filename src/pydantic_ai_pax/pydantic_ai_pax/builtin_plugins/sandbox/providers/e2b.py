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


@dataclass(frozen=True)
class E2BProvider:
    """[E2B](https://e2b.dev) sandboxes, with the key from `/keys` or the `E2B_API_KEY` environment variable."""

    name: str = 'e2b'
    title: str = 'E2B'
    working_dir: str = '/home/user/workspace'
    token_in_sandbox: bool = False

    def missing(self) -> str | None:
        if find_spec('e2b') is None:
            return "E2B support is not installed. Install Pax with it: uv tool install 'pydantic-ai-pax[e2b]'."
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
        backend = E2BSandboxBackend(client=self._client(), working_dir=self.working_dir, env=env, network=network)
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

        return E2BSandboxBackend(client=self._client(), ref=ref, working_dir=self.working_dir)

    async def close(self, backend: SandboxBackend) -> None:
        del backend  # `e2b.E2B` holds no open connection between requests.

    async def delete(self, ref: WorkspaceRef) -> None:
        from pydantic_ai_harness.e2b_sandbox import E2BSandbox

        await E2BSandbox[None](client=self._client()).destroy(ref)
