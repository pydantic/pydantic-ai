"""`/sandbox opensandbox`: OpenSandbox's credential proxy sets GitHub's `Authorization` header outside the sandbox.

The token goes into the sandbox's Credential Vault, which lives only in the memory of the sandbox's egress
sidecar, a container of its own next to the sandbox. The proxy there sets `Authorization` on HTTPS requests to
`GITHUB_HOSTS`, replacing the stand-in that `git` and `gh` send, which is written into the sandbox's home where the
token would be. The vault's API never returns the token, but anyone with the server's API key can reach it and
bind the token to another host, so they can use it; so can anyone who can read the sidecar's memory on the
server's machine. It goes when the sandbox is deleted, or when the sidecar restarts: then `git` and `gh` there
are signed out, and a new `/sandbox opensandbox` signs in again. Proven live with a dummy token: git's Basic with
the stand-in got 200 from httpbin's basic-auth check bound the same way, a wrong password 401.

OpenSandbox deprecates its credential proxy in a sandbox that can reach any host, so a sandbox with a GitHub
login reaches only `GITHUB_HOSTS`, and cannot install packages; one without keeps the server's network policy.
"""

from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import anyio

from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import (
    GITHUB_HOSTS,
    SANDBOX_ENV,
    GitHubCredential,
    give_github_login,
    saved_key,
)
from pydantic_ai_pax.builtin_plugins.sandbox.seed import SandboxBackend
from pydantic_ai_pax.cli.install import install_hint

if TYPE_CHECKING:
    from opensandbox.config import ConnectionConfig

DOMAIN_NAME = 'OPEN_SANDBOX_DOMAIN'
KEY_NAME = 'OPEN_SANDBOX_API_KEY'

IMAGE = 'python:3.13'
"""Debian with `git`, which the checkout is cloned with."""

LOOPBACK_HOSTS = ('localhost', '127.0.0.1', '::1')

STAND_IN = 'placeholder'
"""What the sandbox's `git` and `gh` send as the token; the proxy replaces the whole header."""


@dataclass(frozen=True)
class OpenSandboxProvider:
    """Sandboxes on your own [OpenSandbox](https://open-sandbox.ai) server, at `OPEN_SANDBOX_DOMAIN` with `OPEN_SANDBOX_API_KEY`."""

    name: str = 'opensandbox'
    title: str = 'OpenSandbox'
    working_dir: str = '/workspace'
    token_in_sandbox: bool = False
    desktop_at_create: bool = False

    def missing(self, *, desktop: bool = False) -> str | None:
        if desktop:
            # Its desktop installs its programs with apt, which the GitHub-only network policy blocks.
            return 'OpenSandbox has no desktop for /computer-use. Use another provider, such as e2b or daytona.'
        if find_spec('opensandbox') is None:
            return f'OpenSandbox support is not installed. Install Pax with it: {install_hint("opensandbox")}.'
        if saved_key(DOMAIN_NAME) is None:
            return (
                f"Add your OpenSandbox server's address in /keys as {DOMAIN_NAME}, such as localhost:8080 or "
                'https://sandbox.example.com (an address without a scheme uses HTTPS unless it is this machine), '
                f'and its API key as {KEY_NAME} if it needs one.'
            )
        return None

    def _connection(self) -> 'ConnectionConfig':
        from opensandbox.config import ConnectionConfig

        domain = saved_key(DOMAIN_NAME) or ''
        # The SDK takes the scheme of `domain` for the server's API only; command requests, and a domain without a
        # scheme, use `protocol`. Plain HTTP would carry the API key and the GitHub token readable, so a domain
        # without a scheme gets it only on this machine.
        if domain.startswith(('http://', 'https://')):
            protocol = domain.split(':', 1)[0]
        else:
            protocol = 'http' if urlsplit(f'//{domain}').hostname in LOOPBACK_HOSTS else 'https'
        return ConnectionConfig(domain=domain or None, api_key=saved_key(KEY_NAME), protocol=protocol)

    async def create(
        self, *, github: GitHubCredential | None, desktop: bool = False
    ) -> tuple[WorkspaceRef, SandboxBackend]:
        del desktop  # `missing` refuses a desktop.
        from opensandbox.models.sandboxes import (
            Credential,
            CredentialAuth,
            CredentialBinding,
            CredentialMatch,
            CredentialProxyConfig,
            CustomHeaderEntry,
            InlineCredentialSource,
            NetworkPolicy,
            NetworkRule,
        )

        from pydantic_ai_harness.opensandbox_sandbox import OpenSandboxBackend

        backend = OpenSandboxBackend(
            image=IMAGE,
            connection=self._connection(),
            working_dir=self.working_dir,
            env=SANDBOX_ENV,
            # OpenSandbox deprecates the credential proxy in a sandbox that reaches any host, which may let a
            # request for a GitHub host, token and all, go somewhere else; so it reaches only those hosts.
            network_policy=None
            if github is None
            else NetworkPolicy(
                defaultAction='deny', egress=[NetworkRule(action='allow', target=host) for host in GITHUB_HOSTS]
            ),
            credential_proxy=None if github is None else CredentialProxyConfig(enabled=True),
        )
        try:
            sandbox = await backend.get_sandbox()
            if github is not None:
                values = {host: github.authorization(host) for host in GITHUB_HOSTS}
                names = {value: f'github-{index}' for index, value in enumerate(dict.fromkeys(values.values()))}
                await sandbox.credential_vault.create(
                    credentials=[
                        Credential(name=name, source=InlineCredentialSource(value=value))
                        for value, name in names.items()
                    ],
                    bindings=[
                        CredentialBinding(
                            name=host.replace('.', '-'),
                            # HTTPS only: a plain request would carry the token readable on the way.
                            match=CredentialMatch(schemes=['https'], hosts=[host]),
                            auth=CredentialAuth(
                                type='customHeaders',
                                headers=[CustomHeaderEntry(name='Authorization', credential=names[value])],
                            ),
                        )
                        for host, value in values.items()
                    ],
                )
                # In the sandbox's files rather than the backend's `env`, so it is still there after a reopen.
                await give_github_login(backend, GitHubCredential(login=github.login, token=STAND_IN))
        except BaseException:
            with anyio.CancelScope(shield=True):
                try:
                    if backend.ref is not None:
                        await self.delete(backend.ref)
                finally:
                    await backend.aclose()
            raise
        return WorkspaceRef(provider=self.name, id=sandbox.id), backend

    def reopen(self, ref: WorkspaceRef, *, desktop: bool = False) -> SandboxBackend:
        from pydantic_ai_harness.opensandbox_sandbox import OpenSandboxBackend

        del desktop
        return OpenSandboxBackend(ref=ref, connection=self._connection(), working_dir=self.working_dir, env=SANDBOX_ENV)

    async def close(self, backend: SandboxBackend) -> None:
        from pydantic_ai_harness.opensandbox_sandbox import OpenSandboxBackend

        assert isinstance(backend, OpenSandboxBackend)
        await backend.aclose()

    async def delete(self, ref: WorkspaceRef) -> None:
        from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox

        # The sandbox's egress sidecar, and the token in its memory, go with it.
        await OpenSandbox[None](connection=self._connection()).destroy(ref)
