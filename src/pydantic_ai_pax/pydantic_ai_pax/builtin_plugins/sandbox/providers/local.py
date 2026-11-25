"""`/sandbox local smol` and `/sandbox local boxlite`: a microVM on this machine; the GitHub token is written into the VM.

`/sandbox local` alone asks which: SmolVM (`smol`) runs on macOS (Apple silicon), Linux, and Windows; BoxLite
(`boxlite`) on macOS (Apple silicon) and Linux. No proxy stands between the VM and GitHub, so `give_github_login`
writes the token into the VM's home, where anything run there can read it, and it goes when the VM does. Nothing of
this machine is mounted into the VM: like a cloud sandbox, it holds a copy of the checkout, and it has network
access, from this machine's network, to clone and push.

SmolVM: where smolmachines is installed, which has wheels for macOS and Linux only, the VM runs through it in
Pax's process (harness `SmolSandboxBackend`), and it ends with that process, so `close` runs `sync`: smolmachines
writes a VM's disk back only then or on a stop. Elsewhere, Windows included, Pax drives the `smolvm` CLI on
`PATH`, whose VM would outlive Pax, so `close` stops it. That path has passed the workspace conformance suite against
smolvm 1.36.1 on Linux, but not yet run on a Windows host. Either way the next command starts the VM again, and a conversation
resumed later finds its files.

BoxLite: the box runs in Pax's process (harness `BoxLiteSandboxBackend`) and stops when Pax exits, keeping its
disk, writes included, so `close` has nothing to do. One process at a time can use BoxLite's home directory.
"""

import logging
import platform
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TypeVar

import anyio

from pydantic_ai.computer_use import Point, Region
from pydantic_ai.workspaces import (
    CommandResult,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.computer_use import (
    Display,
    DisplayAction,
    SupportsCursorPosition,
    SupportsDisplay,
    SupportsZoom,
    UnsupportedActionError,
)
from pydantic_ai_pax.builtin_plugins.sandbox.providers.base import (
    SANDBOX_ENV,
    GitHubCredential,
    ref_provider,
)
from pydantic_ai_pax.builtin_plugins.sandbox.seed import SandboxBackend

_T = TypeVar('_T')

IMAGE = 'python:3.13'
"""Debian with bash and git, which the copy of the checkout needs; the engine pulls it the first time."""

_SYNC_TIMEOUT = 60.0

logger = logging.getLogger(__name__)

_INSTALL = "Install Pax with it: uv tool install 'pydantic-ai-pax[local]'."


def _not_on_intel_mac(title: str) -> str | None:
    if sys.platform == 'darwin' and platform.machine() != 'arm64':
        return f'{title} runs only on Macs with Apple silicon.'
    return None


async def _boot(
    backend: SandboxBackend, name: str, delete: Callable[[WorkspaceRef], Awaitable[None]]
) -> tuple[WorkspaceRef, SandboxBackend]:
    """Boot a new VM and make its working folder, deleting it if that fails; the ref `/sandbox` records for it."""
    try:
        await backend.working_dir()
    except BaseException:
        with anyio.CancelScope(shield=True):
            if backend.ref is not None:
                await delete(backend.ref)
        raise
    assert backend.ref is not None
    return WorkspaceRef(provider=ref_provider(name), id=backend.ref.id), backend


@dataclass(frozen=True)
class SmolProvider:
    """`/sandbox local smol`: [SmolVM](https://smolmachines.com) microVMs on this machine, with no account or key."""

    name: str = 'local smol'
    title: str = 'SmolVM'
    working_dir: str = '/workspace'
    token_in_sandbox: bool = True
    desktop_at_create: bool = False

    def missing(self, *, desktop: bool = False) -> str | None:
        from pydantic_ai_harness._smolvm_cli import find_cli

        if find_spec('smol') is not None:
            return None
        if desktop:
            if sys.platform == 'win32':
                return 'SmolVM desktop mode needs smolmachines, which has no Windows build.'
            return _not_on_intel_mac(self.title) or f'SmolVM desktop support needs smolmachines. {_INSTALL}'
        if find_cli() is not None:
            return None
        if sys.platform == 'win32':
            return (
                'Install SmolVM: unzip the windows-x86_64 release from https://github.com/smol-machines/smolvm/releases '
                'into a folder on PATH. It needs the Windows Hypervisor Platform. '
                '/sandbox local smol on Windows has not been tested on a Windows host yet.'
            )
        return _not_on_intel_mac(self.title) or f'SmolVM support is not installed. {_INSTALL}'

    def _backend(self, ref: WorkspaceRef | None, *, desktop: bool = False) -> SandboxBackend:
        ref = None if ref is None else WorkspaceRef(provider='smol', id=ref.id)
        if find_spec('smol') is not None:
            from pydantic_ai_harness.smol_sandbox import SmolSandbox, SmolSandboxBackend

            if ref is not None:
                # Checks the name, so a history cannot open a VM of another tool's, such as one with folders mounted.
                return _Reached(
                    SmolSandbox[None](working_dir=self.working_dir, env=SANDBOX_ENV, desktop=desktop).backend(ref)
                )
            return _Reached(
                SmolSandboxBackend(
                    image=IMAGE,
                    allow_internet_access=True,
                    working_dir=self.working_dir,
                    env=SANDBOX_ENV,
                    desktop=desktop,
                )
            )
        from pydantic_ai_harness._smolvm_cli import SmolvmCliBackend

        return SmolvmCliBackend(
            ref=ref, image=IMAGE, allow_internet_access=True, working_dir=self.working_dir, env=SANDBOX_ENV
        )

    async def create(
        self, *, github: GitHubCredential | None, desktop: bool = False
    ) -> tuple[WorkspaceRef, SandboxBackend]:
        return await _boot(self._backend(None, desktop=desktop), self.name, self.delete)

    def reopen(self, ref: WorkspaceRef, *, desktop: bool = False) -> SandboxBackend:
        return self._backend(ref, desktop=desktop)

    async def close(self, backend: SandboxBackend) -> None:
        from pydantic_ai_harness._smolvm_cli import SmolvmCliBackend

        try:
            if isinstance(backend, SmolvmCliBackend):
                await backend.stop()
            else:
                assert isinstance(backend, _Reached)
                # Not one that never reached its VM, which `sync` would boot for nothing.
                if backend.reached:
                    await backend.run(['sync'], timeout=_SYNC_TIMEOUT)
        except WorkspaceUnavailableError:
            # One deleted already, as by `/sandbox delete`, has nothing to write back.
            pass
        except WorkspaceError:
            # Best effort, so the session's other sandboxes still close, and `/sandbox delete` still reports success.
            logger.warning('Could not stop or sync SmolVM sandbox %s', backend.ref, exc_info=True)

    async def delete(self, ref: WorkspaceRef) -> None:
        """Delete the VM `ref` names; the harness's ref, or `/sandbox`'s, as only the id is read."""
        if find_spec('smol') is not None:
            from pydantic_ai_harness.smol_sandbox import SmolSandbox

            await SmolSandbox[None]().destroy(WorkspaceRef(provider='smol', id=ref.id))
        else:
            from pydantic_ai_harness._smolvm_cli import delete_machine

            await delete_machine(ref.id)


@dataclass(frozen=True)
class BoxLiteProvider:
    """`/sandbox local boxlite`: [BoxLite](https://github.com/boxlite-ai/boxlite) micro-VMs on this machine, with no key."""

    name: str = 'local boxlite'
    title: str = 'BoxLite'
    working_dir: str = '/workspace'
    token_in_sandbox: bool = True
    desktop_at_create: bool = True

    def missing(self, *, desktop: bool = False) -> str | None:
        del desktop
        if find_spec('boxlite') is not None:
            return None
        if sys.platform == 'win32':
            return 'BoxLite does not run on Windows: use /sandbox local smol.'
        return (
            _not_on_intel_mac(self.title)
            or f'BoxLite support is not installed. {_INSTALL} It needs Python 3.14 or older.'
        )

    def _backend(self, ref: WorkspaceRef | None, *, desktop: bool = False) -> SandboxBackend:
        from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox, BoxLiteSandboxBackend

        if ref is not None:
            # Checks the name, so a history cannot open a box of another tool's, such as one with folders mounted.
            return BoxLiteSandbox[None](
                working_dir=self.working_dir, env=SANDBOX_ENV, allow_internet_access=True, desktop=desktop
            ).backend(WorkspaceRef(provider='boxlite', id=ref.id))
        return BoxLiteSandboxBackend(
            image=IMAGE,
            allow_internet_access=True,
            working_dir=self.working_dir,
            env=SANDBOX_ENV,
            desktop=desktop,
            box_options={'disk_size_gb': 4} if desktop else None,
        )

    async def create(
        self, *, github: GitHubCredential | None, desktop: bool = False
    ) -> tuple[WorkspaceRef, SandboxBackend]:
        return await _boot(self._backend(None, desktop=desktop), self.name, self.delete)

    def reopen(self, ref: WorkspaceRef, *, desktop: bool = False) -> SandboxBackend:
        return self._backend(ref, desktop=desktop)

    async def close(self, backend: SandboxBackend) -> None:
        # The box stops when Pax exits, keeping its disk, so there is nothing to write back or stop.
        pass

    async def delete(self, ref: WorkspaceRef) -> None:
        """Delete the box `ref` names; the harness's ref, or `/sandbox`'s, as only the id is read."""
        if (missing := self.missing()) is not None:
            raise WorkspaceUnavailableError(missing)
        from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox

        await BoxLiteSandbox[None]().destroy(WorkspaceRef(provider='boxlite', id=ref.id))


class _ReachedDisplay:
    """Forwards `Display` calls for `_Reached` and records that the VM was booted so `close` syncs its disk."""

    def __init__(self, owner: '_Reached', display: Display) -> None:
        self._owner = owner
        self._display = display

    async def _call(self, op: Callable[[], Awaitable[_T]]) -> _T:
        reached = True
        try:
            return await op()
        except WorkspaceUnavailableError:
            reached = False
            raise
        finally:
            self._owner.reached = self._owner.reached or reached

    async def display_size(self) -> tuple[int, int]:
        return await self._call(self._display.display_size)

    async def screenshot(self) -> bytes:
        return await self._call(self._display.screenshot)

    async def act(self, action: DisplayAction) -> None:
        await self._call(lambda: self._display.act(action))

    async def zoom(self, region: Region) -> bytes:
        if not isinstance(self._display, SupportsZoom):
            raise UnsupportedActionError('This display cannot zoom.')
        display = self._display
        return await self._call(lambda: display.zoom(region))

    async def cursor_position(self) -> Point:
        if not isinstance(self._display, SupportsCursorPosition):
            raise UnsupportedActionError('This display cannot report the pointer position.')
        display = self._display
        return await self._call(display.cursor_position)


class _Reached(WorkspaceBackend, SupportsCommands):
    """A smolmachines backend that records whether it reached its VM, which its first operation boots."""

    def __init__(self, backend: SandboxBackend) -> None:
        self._backend = backend
        self.reached = False
        inner = backend.display if isinstance(backend, SupportsDisplay) else None
        self._display: _ReachedDisplay | None = None if inner is None else _ReachedDisplay(self, inner)

    @property
    def display(self) -> Display | None:
        return self._display

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._backend.ref

    async def working_dir(self) -> str:
        reached = True
        try:
            return await self._backend.working_dir()
        except WorkspaceUnavailableError:
            # It could not boot or attach to the VM, such as one deleted since.
            reached = False
            raise
        finally:
            self.reached = self.reached or reached

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        reached = True
        try:
            return await self._backend.run(command, shell=shell, env=env, timeout=timeout)
        except WorkspaceUnavailableError:
            reached = False
            raise
        finally:
            self.reached = self.reached or reached
