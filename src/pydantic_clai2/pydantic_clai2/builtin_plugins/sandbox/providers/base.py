"""What every `/sandbox` provider implements, and the GitHub credential it gives the sandbox's `git` and `gh`."""

import base64
import os
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from pydantic_ai.workspaces import Workspace, WorkspaceRef
from pydantic_clai2.builtin_plugins.sandbox.seed import GITHUB_TOKEN_FILE, SandboxBackend
from pydantic_clai2.config.api_keys import load_keys

GITHUB_HOSTS = ('github.com', 'api.github.com', 'uploads.github.com')
"""The hosts the proxy signs requests to. None of them echoes request headers back, which would show the token."""

SANDBOX_ENV: Mapping[str, str] = {'GIT_TERMINAL_PROMPT': '0'}
"""What every command in a sandbox gets: git fails instead of waiting for a password no one can type."""


@dataclass(frozen=True, kw_only=True)
class GitHubCredential:
    """The user's `gh` login, which the sandbox's `git` and `gh` act as while the sandbox lives."""

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
    """A cloud sandbox service that `/sandbox` can make, reopen, and delete sandboxes on."""

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

    @property
    def token_in_sandbox(self) -> bool:
        """Whether the GitHub token is written into the sandbox, as it is where no proxy can sign requests."""
        ...

    def missing(self) -> str | None:
        """What to do before a sandbox can be made, such as a key to add; `None` when nothing is missing."""
        ...

    async def create(self, *, github: GitHubCredential | None) -> tuple[WorkspaceRef, SandboxBackend]:
        """A new, running sandbox whose commands get `SANDBOX_ENV`, and its reference.

        With `github`, `git` and `gh` there act as `github.login`. A provider whose proxy can sign requests gets the
        token for this sandbox only, and the sandbox's `GH_TOKEN` holds a stand-in the proxy replaces; any other
        writes the token into the sandbox (`give_github_login`). A `create` that raises, or is cancelled, first
        deletes whatever it made, as no caller has its reference to delete it later.
        """
        ...

    def reopen(self, ref: WorkspaceRef) -> SandboxBackend:
        """The sandbox `ref` names, attached on first use; its commands get `SANDBOX_ENV`."""
        ...

    async def set_env(self, backend: SandboxBackend, name: str, value: str) -> None:
        """Give every command started in the sandbox from now on the variable `name`.

        It is kept with the sandbox where the provider can keep it there, as E2B and boxd can; otherwise it lasts
        for this CLAI session, and a sandbox reopened later has it no more.

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


async def give_github_login(backend: SandboxBackend, github: GitHubCredential) -> None:
    """Sign the sandbox's `git` and `gh` in as `github.login` by writing the token into the sandbox's home.

    Without a proxy to hold it, the token must be in the sandbox, so anything run there can read it. It goes
    through the backend's file API, kept off command lines a provider may log or send in a URL. A backend
    with only commands gets it base64-encoded in a command, as `Workspace` writes files there.
    """
    home = (await backend.run(['sh', '-c', 'printf %s "$HOME"'])).stdout
    if not posixpath.isabs(home):
        raise RuntimeError('the sandbox has no home folder for the GitHub login')
    token = posixpath.join(home, GITHUB_TOKEN_FILE)
    # `gh` reads its login from here when `GH_TOKEN` is unset.
    hosts = posixpath.join(home, '.config/gh/hosts.yml')
    # Folders only their user can enter, made before the files, so no other user can read a file as it is written.
    await _check(backend, ['mkdir', '-p', '-m', '700', posixpath.dirname(token), posixpath.dirname(hosts)])
    files = Workspace(backend)
    await files.write_bytes(token, github.token.encode())
    await files.write_bytes(
        hosts,
        f'github.com:\n    oauth_token: {github.token}\n    user: {github.login}\n    git_protocol: https\n'.encode(),
    )
    await _check(backend, ['chmod', '600', token, hosts])


async def _check(backend: SandboxBackend, argv: list[str]) -> None:
    result = await backend.run(argv)
    if result.exit_code != 0:
        raise RuntimeError(f'Could not sign the sandbox in to GitHub: {(result.stderr or result.stdout).strip()}')
