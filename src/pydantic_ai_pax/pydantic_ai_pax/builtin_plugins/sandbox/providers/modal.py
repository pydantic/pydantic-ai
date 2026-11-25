"""`/sandbox modal`: the GitHub token is written into the sandbox, where anything run there can read it.

Modal's network has no proxy that could sign requests, so `give_github_login` writes the token into the
sandbox's home, and it goes when the sandbox does. A Modal sandbox lives at most 24 hours, and cannot be
paused: then Modal ends it, and its files with it.

Pax uses Modal's own login (`modal token new`, `~/.modal.toml`, or `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`),
not `/keys`: a client from other credentials opens a connection when it is built, and `reopen` may not wait for
one, while the harness backend resolves Modal's own client on first use.
"""

import os
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import SANDBOX_ENV, GitHubCredential
from pydantic_ai_pax.builtin_plugins.sandbox.seed import SandboxBackend


@dataclass(frozen=True)
class ModalProvider:
    """[Modal](https://modal.com) sandboxes, signed in with Modal's own login."""

    name: str = 'modal'
    title: str = 'Modal'
    working_dir: str = '/root/workspace'
    token_in_sandbox: bool = True

    def missing(self) -> str | None:
        if find_spec('modal') is None:
            return "Modal support is not installed. Install Pax with it: uv tool install 'pydantic-ai-pax[modal]'."
        # Checked without importing Modal, which reads its config file as it loads; a file without a token
        # fails at create with Modal's own message.
        config = os.environ.get('MODAL_CONFIG_PATH') or '~/.modal.toml'
        if not (os.environ.get('MODAL_TOKEN_ID') and os.environ.get('MODAL_TOKEN_SECRET')) and not (
            Path(config).expanduser().exists()
        ):
            return 'Sign in to Modal first: run uvx modal token new, then try again.'
        return None

    async def create(self, *, github: GitHubCredential | None) -> tuple[WorkspaceRef, SandboxBackend]:
        from pydantic_ai_harness.modal_sandbox import ModalSandboxBackend

        # Modal's longest lifetime, 24 hours, and no idle timeout: Modal would end an idle sandbox, not pause it.
        backend = ModalSandboxBackend(working_dir=self.working_dir, env=SANDBOX_ENV, sandbox_timeout=86_400)
        # Nothing after Modal creates the sandbox can fail, so there is nothing here to clean up.
        await backend.get_sandbox()
        assert backend.ref is not None
        return backend.ref, backend

    def reopen(self, ref: WorkspaceRef) -> SandboxBackend:
        from pydantic_ai_harness.modal_sandbox import ModalSandboxBackend

        return ModalSandboxBackend(ref=ref, working_dir=self.working_dir, env=SANDBOX_ENV)

    async def close(self, backend: SandboxBackend) -> None:
        del backend  # Modal's own client is shared by the process, and closes with it.

    async def delete(self, ref: WorkspaceRef) -> None:
        from pydantic_ai_harness.modal_sandbox import ModalSandbox

        await ModalSandbox[None]().destroy(ref)
