"""Filesystem and shell tools that route through the ACP client instead of local disk/processes.

The local [`FileSystem`][pydantic_ai_harness.FileSystem] and [`Shell`][pydantic_ai_harness.Shell]
capabilities operate on the agent process's own disk and subprocesses. In an editor, that misses
the source of truth: unsaved buffers, the file layout the editor (not the launching shell)
considers the workspace, and -- for a remote or containerized editor -- the machine the code
actually lives on. ACP lets the agent ask the *client* to do the I/O: `fs/read_text_file` /
`fs/write_text_file` for files, and the terminal lifecycle (`terminal/create`, `terminal/output`,
`terminal/wait_for_exit`, `terminal/release`) for commands.

[`AcpFileSystemToolset`][pydantic_ai_harness.experimental.acp.AcpFileSystemToolset] and
[`AcpTerminalToolset`][pydantic_ai_harness.experimental.acp.AcpTerminalToolset] are those editor-native
counterparts. Build them per session with
[`acp_filesystem`][pydantic_ai_harness.experimental.acp.acp_filesystem] /
[`acp_terminal`][pydantic_ai_harness.experimental.acp.acp_terminal], which return the toolset only when the
client advertised the matching capability and otherwise return `None` so the caller can fall back
to the local capability.
"""

# A read-only ACP client gets editor-native reads with writes delegated to the local `FileSystem`
# capability; otherwise these toolsets are self-contained. A future shared I/O-backend seam is
# expected to subsume them.

from __future__ import annotations

import asyncio
import contextlib
import os.path
from collections.abc import Awaitable
from typing import Protocol

import anyio
from acp import Client, schema
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.experimental.acp._session import AcpSession
from pydantic_ai_harness.filesystem import FileSystem


class _LocalFileWriter(Protocol):
    """Something that can write a file on the local disk -- structurally satisfied by `FileSystemToolset`."""

    def write_file(self, path: str, content: str) -> Awaitable[str]: ...  # pragma: no cover - structural protocol


class AcpFileSystemToolset(FunctionToolset[AgentDepsT]):
    """`read_file`/`write_file` tools backed by the ACP client connection.

    Each call invokes the client's `fs/read_text_file` / `fs/write_text_file`, so the agent sees
    the editor's live view of the workspace (including unsaved buffers). The tool names match the
    local `FileSystem` capability, so the default presenter renders them the same way.

    ACP requires absolute paths, but models routinely produce workspace-relative ones (the local
    `FileSystem` tools take them, and the same agent may run against either backend): when `cwd`
    is set, relative paths are resolved against it before reaching the wire. The client still
    resolves and authorizes every path itself (this toolset adds no sandboxing of its own).

    If `local_writer` is set (a read-only client -- see [`acp_filesystem`][pydantic_ai_harness.experimental.acp.acp_filesystem]),
    `write_file` goes there instead of to the client, while reads still route through the editor.
    """

    def __init__(
        self, *, client: Client, session_id: str, cwd: str | None = None, local_writer: _LocalFileWriter | None = None
    ) -> None:
        super().__init__()
        self._client = client
        self._session_id = session_id
        self._cwd = cwd
        self._local_writer = local_writer
        self.add_function(self.read_file, name='read_file')
        self.add_function(self.write_file, name='write_file')

    def _absolute(self, path: str) -> str:
        if self._cwd is None or os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self._cwd, path))

    async def read_file(self, path: str) -> str:
        """Read a text file's contents through the editor.

        Args:
            path: Path to the file; resolved against the session workspace when relative.
        """
        response = await self._client.read_text_file(path=self._absolute(path), session_id=self._session_id)
        return response.content

    async def write_file(self, path: str, content: str) -> str:
        """Write a text file's full contents through the editor.

        Args:
            path: Path to the file; resolved against the session workspace when relative.
            content: The complete new contents of the file.
        """
        path = self._absolute(path)
        if self._local_writer is not None:
            return await self._local_writer.write_file(path, content)
        await self._client.write_text_file(content=content, path=path, session_id=self._session_id)
        return f'Wrote {path} ({len(content)} characters).'


def acp_filesystem(session: AcpSession) -> AcpFileSystemToolset[None] | None:
    """Build an ACP-client-backed filesystem toolset for `session`, or `None` if unsupported.

    Returns an [`AcpFileSystemToolset`][pydantic_ai_harness.experimental.acp.AcpFileSystemToolset] whenever the
    client advertised `fs/read_text_file` during `initialize`:

    - read + write advertised: reads and writes both route through the editor.
    - read only (no `fs/write_text_file`): reads route through the editor, while writes go to the
      local [`FileSystem`][pydantic_ai_harness.FileSystem] rooted at `session.cwd`. This is coherent
      only when the agent shares the workspace disk with the editor (same machine, or an agent
      running inside the editor's container) -- for a *remote* editor the writes land on the agent's
      disk, not the editor's.

    Returns `None` only when the client advertised no readable filesystem, so the caller can fall
    back to a fully local toolset:

    ```python
    def session_config(session: AcpSession) -> AcpSessionConfig[None]:
        fs = acp_filesystem(session) or FileSystem(root_dir=session.cwd).get_toolset()
        return AcpSessionConfig(deps=None, toolsets=[fs])
    ```

    For an agent with non-`None` deps, construct `AcpFileSystemToolset[YourDeps](...)` directly
    (the toolset ignores deps); this helper covers the common no-deps case.
    """
    capabilities = session.client_capabilities
    fs = capabilities.fs if capabilities is not None else None
    if fs is None or not fs.read_text_file:
        return None
    local_writer = None if fs.write_text_file else FileSystem(root_dir=session.cwd).get_toolset()
    return AcpFileSystemToolset[None](
        client=session.client, session_id=session.session_id, cwd=session.cwd, local_writer=local_writer
    )


def _format_terminal_output(result: schema.TerminalOutputResponse) -> str:
    """Render a terminal's captured output plus a trailing note for truncation or non-success exit."""
    parts = [result.output]
    if result.truncated:
        parts.append('[output truncated]')
    status = result.exit_status
    if status is not None and status.exit_code not in (None, 0):
        parts.append(f'[exited with code {status.exit_code}]')
    elif status is not None and status.signal is not None:
        parts.append(f'[terminated by signal {status.signal}]')
    return '\n'.join(parts)


class AcpTerminalToolset(FunctionToolset[AgentDepsT]):
    """A `run_command` tool backed by the ACP client's terminal lifecycle.

    Each call asks the client to create a terminal, waits for it to exit, reads its output, and
    releases it, so the command runs in the editor's environment rather than as a local
    subprocess of the agent. The tool name matches the local `Shell` capability so the default
    presenter renders it as an `execute` call. If the call is cancelled while the command is
    running, the terminal is killed and released.
    """

    def __init__(self, *, client: Client, session_id: str, cwd: str | None = None) -> None:
        super().__init__()
        self._client = client
        self._session_id = session_id
        self._cwd = cwd
        self.add_function(self.run_command, name='run_command')

    # Embedding a live terminal pane in the tool call would require the terminal id at
    # call-start, before the command runs; the captured output is returned instead.
    async def run_command(self, command: str) -> str:
        """Run a shell command in the editor's terminal and return its captured output.

        Args:
            command: The shell command line to execute.
        """
        # The create runs as its own task so a cancellation landing mid-flight cannot abandon the
        # request: it may already be on the wire (the client then starts the command regardless),
        # so its response must still be read to learn the id and clean up. A raw `task.cancel()`
        # (how the adapter and pydantic-ai deliver cancellation) pierces anyio shields, so the
        # create must live outside this task; `asyncio.wait` rather than `asyncio.shield` because
        # shield on 3.12+ reports a late create failure to the loop exception handler even when
        # the cleanup below retrieves it.
        create = asyncio.ensure_future(
            self._client.create_terminal(command=command, session_id=self._session_id, cwd=self._cwd)
        )
        terminal_id: str | None = None
        try:
            await asyncio.wait([create])
            terminal_id = create.result().terminal_id
            await self._client.wait_for_terminal_exit(session_id=self._session_id, terminal_id=terminal_id)
            result = await self._client.terminal_output(session_id=self._session_id, terminal_id=terminal_id)
            return _format_terminal_output(result)
        except asyncio.CancelledError:
            # Kill the still-running terminal before unwinding, shielded so the cleanup completes
            # even though this task is being cancelled (when the cancellation landed during the
            # create, the id is learned from the still-running create first). Suppress failures: a
            # client error here must not replace the `CancelledError` the caller needs to see
            # (the spec requires the turn to end with a `cancelled` stop reason).
            with anyio.CancelScope(shield=True):
                if terminal_id is None:
                    with contextlib.suppress(Exception):
                        terminal_id = (await create).terminal_id
                if terminal_id is not None:
                    with contextlib.suppress(Exception):
                        await self._client.kill_terminal(session_id=self._session_id, terminal_id=terminal_id)
            raise
        finally:
            # Always release the terminal (if one came into existence); suppress failures so a
            # release error never masks the exception (or successful return) already in flight.
            if terminal_id is not None:
                with anyio.CancelScope(shield=True), contextlib.suppress(Exception):
                    await self._client.release_terminal(session_id=self._session_id, terminal_id=terminal_id)


def acp_terminal(session: AcpSession) -> AcpTerminalToolset[None] | None:
    """Build an ACP-client-backed terminal toolset for `session`, or `None` if unsupported.

    Returns an [`AcpTerminalToolset`][pydantic_ai_harness.experimental.acp.AcpTerminalToolset] only when the
    client advertised terminal support during `initialize`; otherwise returns `None` so the caller
    can fall back to a local toolset:

    ```python
    def session_config(session: AcpSession) -> AcpSessionConfig[None]:
        shell = acp_terminal(session) or Shell(cwd=session.cwd).get_toolset()
        return AcpSessionConfig(deps=None, toolsets=[shell])
    ```

    For an agent with non-`None` deps, construct `AcpTerminalToolset[YourDeps](...)` directly (the
    toolset ignores deps); this helper covers the common no-deps case.
    """
    capabilities = session.client_capabilities
    if capabilities is None or not capabilities.terminal:
        return None
    return AcpTerminalToolset[None](client=session.client, session_id=session.session_id, cwd=session.cwd)
