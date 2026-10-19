"""Native content transfers with Pydantic's public shell metadata implementation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol
from uuid import uuid4

import anyio
from render import RenderAsync
from render.client.errors import RateLimitError, RenderError
from render.experimental.sandbox import SandboxFileNotFoundError, SandboxNotFoundError

from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    Workspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

_HELPER = Path(__file__).with_name('_file_transfer.py').read_text()


class _Commands(WorkspaceBackend, SupportsCommands, Protocol):
    pass


class _CommandView:
    """Hide SupportsFilesystem so public Workspace metadata cannot recurse into us."""

    def __init__(self, backend: _Commands) -> None:
        self.backend = backend

    @property
    def ref(self) -> WorkspaceRef | None:
        return self.backend.ref

    async def working_dir(self) -> str:
        return await self.backend.working_dir()

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        return await self.backend.run(command, shell=shell, env=env, timeout=timeout)


class NativeFilesystem:
    def __init__(self, backend: _Commands, timeout: float) -> None:
        self.commands = _CommandView(backend)
        self.metadata = Workspace(self.commands)
        self.timeout = timeout

    async def _helper(self, action: str, path: str, directory: str, size: int = 0) -> int:
        result = await self.commands.run(
            ['python3', '-I', '-c', _HELPER, action, path, directory, str(size)], timeout=self.timeout
        )
        try:
            value = json.loads(result.stdout)
            if result.exit_code == 1 and type(value.get('errno')) is int:
                raise OSError(value['errno'], value.get('message', 'File operation failed'), path)
            if result.exit_code == 0 and type(value.get('size')) is int and value['size'] >= 0:
                return value['size']
        except (ValueError, AttributeError):
            pass
        raise WorkspaceError(f'Render file {action} returned an invalid result for {path!r}')

    async def _transfer(self, client: RenderAsync, path: str, data: bytes | None) -> bytes:
        if not path.startswith('/') or '\0' in path:
            raise ValueError('file path must be absolute and contain no NUL')
        await self.commands.working_dir()
        ref = self.commands.ref
        assert ref is not None
        # Known before dispatch, so cancellation before acknowledgement still permits
        # cleanup. Never put agent file bytes in command arguments or command output.
        remote = f'/tmp/.render-pydantic-{uuid4().hex}'
        try:
            size = await self._helper('read' if data is None else 'prepare', path, remote)
            with TemporaryDirectory(prefix='render-pydantic-') as directory:
                local = Path(directory) / 'data'
                if data is not None:
                    await anyio.Path(local).write_bytes(data)
                # The public SDK has per-connection timeouts but no total file deadline.
                # Never retry an uncertain upload or hide failure with a shell fallback.
                async with asyncio.timeout(self.timeout):
                    if data is None:
                        await client.experimental.sandboxes.copy_from(ref.id, f'{remote}/data', local)
                    else:
                        await client.experimental.sandboxes.copy_to(ref.id, local, f'{remote}/data')
                if data is None:
                    # Reject a changed snapshot before allocating the returned bytes.
                    if local.is_symlink() or local.stat().st_size != size:
                        raise WorkspaceError('Render file download size did not match its snapshot')
                    return await anyio.Path(local).read_bytes()
                await self._helper('commit', path, remote, len(data))
                return b''
        except SandboxFileNotFoundError as error:
            raise FileNotFoundError(path) from error
        except SandboxNotFoundError as error:
            raise WorkspaceUnavailableError('Render sandbox is unavailable') from error
        except RateLimitError:
            raise
        except TimeoutError as error:
            raise WorkspaceTimeoutError(f'Render file transfer exceeded {self.timeout}s') from error
        except RenderError as error:
            # SDK download errors and transient failures are not evidence of a missing file.
            raise WorkspaceError(f'Render file transfer failed for {path!r}') from error
        finally:
            # Shield bounded best-effort staging cleanup, preserving the original error.
            # Process death can leave staging until the owner terminates the sandbox.
            with anyio.move_on_after(self.timeout, shield=True):
                try:
                    await self.commands.run(
                        ['python3', '-I', '-c', _HELPER, 'cleanup', path, remote], timeout=self.timeout
                    )
                except Exception:
                    pass

    async def read_bytes(self, client: RenderAsync, path: str) -> bytes:
        return await self._transfer(client, path, None)

    async def write_bytes(self, client: RenderAsync, path: str, data: bytes) -> None:
        await self._transfer(client, path, data)

    async def stat(self, path: str) -> FileEntry:
        return await self.metadata.stat(path)

    async def list_dir(self, path: str) -> tuple[FileEntry, ...]:
        return tuple(await self.metadata.list_dir(path))

    async def make_dir(self, path: str) -> None:
        await self.metadata.make_dir(path)

    async def remove(self, path: str) -> None:
        await self.metadata.remove(path)

    async def exists(self, path: str) -> bool:
        return await self.metadata.exists(path)
