"""`/sandbox local`: a SmolVM microVM on this machine; the GitHub token is written into the VM, where anything run there can read it.

No proxy stands between the VM and GitHub, so `give_github_login` writes the token into the VM's home, and it goes
when the VM does. Nothing of this machine is mounted into the VM: like a cloud sandbox, it holds a copy of the
checkout, and it has network access, from this machine's network, to clone and push.

SmolVM runs on macOS (Apple silicon), Linux, and Windows. Where smolmachines is installed, which has wheels for
macOS and Linux only, the VM runs through it in CLAI's process (harness `SmolSandboxBackend`), and it ends with
that process, so `close` runs `sync`: smolmachines writes a VM's disk back only then or on a stop. Elsewhere, Windows
included, CLAI drives the `smolvm` CLI on `PATH`, whose VM would outlive CLAI, so `close` stops it. Either way the
next command starts the VM again, and a conversation resumed later finds its files.
"""

import shutil
import sys
from contextlib import suppress
from dataclasses import dataclass
from importlib.util import find_spec

import anyio

from pydantic_ai.workspaces import WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import SANDBOX_ENV, GitHubCredential
from pydantic_ai_pax.builtin_plugins.sandbox.seed import SandboxBackend

REF_PROVIDER = 'local-vm'
"""What `WorkspaceRef.provider` holds for this provider's VMs.

Not `local`, which names this machine's own folder, nor the harness's `smol`, which a `SmolSandbox` of the user's
own makes too: `/sandbox` must not take that VM for one of its sandboxes.
"""

IMAGE = 'python:3.13'
"""Debian with bash and git, which the copy of the checkout needs; smolvm pulls it the first time."""

_SYNC_TIMEOUT = 60.0


@dataclass(frozen=True)
class LocalProvider:
    """[SmolVM](https://smolmachines.com) microVMs on this machine, with no account or key."""

    name: str = 'local'
    title: str = 'SmolVM'
    working_dir: str = '/workspace'
    token_in_sandbox: bool = True

    def missing(self) -> str | None:
        if find_spec('smol') is not None or shutil.which('smolvm') is not None:
            return None
        if sys.platform == 'win32':
            return (
                'Install SmolVM: unzip the windows-x86_64 release from https://github.com/smol-machines/smolvm/releases '
                'into a folder on PATH. It needs the Windows Hypervisor Platform.'
            )
        return "SmolVM support is not installed. Install CLAI with it: uv tool install 'pydantic-ai-pax[local]'."

    def _backend(self, ref: WorkspaceRef | None) -> SandboxBackend:
        ref = None if ref is None else _vm(ref)
        if find_spec('smol') is not None:
            from pydantic_ai_harness.smol_sandbox import SmolSandbox, SmolSandboxBackend

            if ref is not None:
                # Checks the name, so a history cannot open a VM of another tool's, such as one with folders mounted.
                return SmolSandbox[None](working_dir=self.working_dir, env=SANDBOX_ENV).backend(ref)
            return SmolSandboxBackend(
                image=IMAGE, allow_internet_access=True, working_dir=self.working_dir, env=SANDBOX_ENV
            )
        from pydantic_ai_harness._smolvm_cli import SmolvmCliBackend

        return SmolvmCliBackend(
            ref=ref, image=IMAGE, allow_internet_access=True, working_dir=self.working_dir, env=SANDBOX_ENV
        )

    async def create(self, *, github: GitHubCredential | None) -> tuple[WorkspaceRef, SandboxBackend]:
        backend = self._backend(None)
        try:
            # Boots the VM and makes the working folder.
            await backend.working_dir()
        except BaseException:
            with anyio.CancelScope(shield=True):
                if backend.ref is not None:
                    await self._delete(backend.ref)
            raise
        assert backend.ref is not None
        return WorkspaceRef(provider=REF_PROVIDER, id=backend.ref.id), backend

    def reopen(self, ref: WorkspaceRef) -> SandboxBackend:
        return self._backend(ref)

    async def close(self, backend: SandboxBackend) -> None:
        from pydantic_ai_harness._smolvm_cli import SmolvmCliBackend

        # One deleted already, as by `/sandbox delete`, has nothing to write back.
        with suppress(WorkspaceUnavailableError):
            if isinstance(backend, SmolvmCliBackend):
                await backend.stop()
            else:
                await backend.run(['sync'], timeout=_SYNC_TIMEOUT)

    async def delete(self, ref: WorkspaceRef) -> None:
        await self._delete(_vm(ref))

    async def _delete(self, vm: WorkspaceRef) -> None:
        if find_spec('smol') is not None:
            from pydantic_ai_harness.smol_sandbox import SmolSandbox

            await SmolSandbox[None]().destroy(vm)
        else:
            from pydantic_ai_harness._smolvm_cli import delete_machine

            await delete_machine(vm.id)


def _vm(ref: WorkspaceRef) -> WorkspaceRef:
    """The harness's ref for the VM `ref` names."""
    return WorkspaceRef(provider='smol', id=ref.id)
