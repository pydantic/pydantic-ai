"""What every `/sandbox` provider implements, and the GitHub credential they all give their proxy."""

import base64
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from pydantic_ai.workspaces import WorkspaceRef
from pydantic_clai2.builtin_plugins.sandbox.seed import SandboxBackend
from pydantic_clai2.config.api_keys import load_keys

GITHUB_HOSTS = ('github.com', 'api.github.com', 'uploads.github.com')
"""The hosts the proxy signs requests to. None of them echoes request headers back, which would show the token."""

SANDBOX_ENV: Mapping[str, str] = {'GIT_TERMINAL_PROMPT': '0'}
"""What every command in a sandbox gets: git fails instead of waiting for a password no one can type."""


@dataclass(frozen=True, kw_only=True)
class GitHubCredential:
    """The user's `gh` login, which the provider's proxy adds to requests for `GITHUB_HOSTS` while the sandbox lives."""

    login: str
    token: str = field(repr=False)

    @property
    def basic(self) -> str:
        """The `Authorization` value `git` sends over HTTPS."""
        return 'Basic ' + base64.b64encode(f'x-access-token:{self.token}'.encode()).decode()

    @property
    def bearer(self) -> str:
        """The `Authorization` value `gh` sends to the API."""
        return f'token {self.token}'

    def authorization(self, host: str) -> str:
        """The `Authorization` value for `host`, one of `GITHUB_HOSTS`: git's on `github.com`, gh's on the API hosts."""
        return self.basic if host == 'github.com' else self.bearer


class SandboxProvider(Protocol):
    """A cloud sandbox service whose proxy, outside the sandbox, holds the GitHub token for the sandbox's requests."""

    @property
    def name(self) -> str:
        """The name `/sandbox` takes and `WorkspaceRef.provider` holds, such as `e2b`."""
        ...

    @property
    def title(self) -> str:
        """The name to show, such as `E2B`."""
        ...

    @property
    def working_dir(self) -> str:
        """Where the checkout goes in the sandbox."""
        ...

    def missing(self) -> str | None:
        """What to do before a sandbox can be made, such as a key to add; `None` when nothing is missing."""
        ...

    async def create(self, *, github: GitHubCredential | None) -> tuple[WorkspaceRef, SandboxBackend]:
        """A new, running sandbox whose commands get `SANDBOX_ENV`, and its reference.

        With `github`, the provider's proxy gets the token for this sandbox only, and the sandbox's `GH_TOKEN`
        holds a stand-in the proxy replaces, so `git` and `gh` act as `github.login`. A `create` that raises,
        or is cancelled, first deletes whatever it made, as no caller has its reference to delete it later.
        """
        ...

    def reopen(self, ref: WorkspaceRef) -> SandboxBackend:
        """The sandbox `ref` names, attached on first use; its commands get `SANDBOX_ENV`."""
        ...

    async def set_env(self, backend: SandboxBackend, name: str, value: str) -> None:
        """Give every command started in the sandbox from now on the variable `name`, kept with the sandbox.

        A command that sets `name` itself keeps its own value; a process already running keeps the
        environment it started with. `RuntimeError` says why this sandbox cannot take it.
        """
        ...

    async def delete(self, ref: WorkspaceRef) -> None:
        """Delete the sandbox `ref` names, and the provider's copy of the token with it; one already gone is fine."""
        ...

    async def close(self, backend: SandboxBackend) -> None:
        """Close what `create` or `reopen` opened for `backend`, such as an API client; the sandbox keeps running."""
        ...


def saved_key(name: str) -> str | None:
    """The key saved in `/keys` as `name`, else the environment variable `name`."""
    saved = load_keys().get(name)
    return saved.get_secret_value() if saved is not None else os.environ.get(name) or None
