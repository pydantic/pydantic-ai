"""Shared in-memory ACP `Client` scaffolding for the adapter and toolset tests.

`Client` has no abstract methods at runtime, but pyright treats them as abstract (the SDK marks
them so), so a test client must define the whole surface. `RecordingClientBase` defines that surface
once -- it records the `session/update`s it receives and stubs every request method -- so concrete
clients subclass it and override only the capabilities they exercise. `RecordingClient` drives
filesystem and terminal state; `FakeClient` (in `test_acp.py`) answers permission requests; and
`WireClient` (in `_wire.py`) records updates only.
"""

from __future__ import annotations

import asyncio

from acp import Client, RequestError, schema


class RecordingClientBase(Client):
    """Records `session/update`s; every request method is an unused stub that raises if called.

    Subclasses override only the capabilities they exercise. A stub that fires means the adapter
    called a client capability the test did not expect, which should surface loudly.
    """

    def __init__(self) -> None:
        self.updates: list[object] = []

    async def session_update(self, session_id: str, update: object, **kwargs: object) -> None:
        self.updates.append(update)

    def on_connect(self, conn: object) -> None:
        return None  # pragma: no cover - unused

    async def request_permission(
        self,
        session_id: str,
        tool_call: schema.ToolCallUpdate,
        options: list[schema.PermissionOption],
        **kwargs: object,
    ) -> schema.RequestPermissionResponse:
        raise NotImplementedError  # pragma: no cover - unused

    async def read_text_file(
        self, session_id: str, path: str, line: int | None = None, limit: int | None = None, **kwargs: object
    ) -> schema.ReadTextFileResponse:
        raise NotImplementedError  # pragma: no cover - unused

    async def write_text_file(
        self, session_id: str, path: str, content: str, **kwargs: object
    ) -> schema.WriteTextFileResponse | None:
        raise NotImplementedError  # pragma: no cover - unused

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: list[str] | None = None,
        env: list[schema.EnvVariable] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **kwargs: object,
    ) -> schema.CreateTerminalResponse:
        raise NotImplementedError  # pragma: no cover - unused

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> schema.WaitForTerminalExitResponse:
        raise NotImplementedError  # pragma: no cover - unused

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> schema.TerminalOutputResponse:
        raise NotImplementedError  # pragma: no cover - unused

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> schema.KillTerminalResponse | None:
        raise NotImplementedError  # pragma: no cover - unused

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> schema.ReleaseTerminalResponse | None:
        raise NotImplementedError  # pragma: no cover - unused

    async def create_elicitation(
        self, message: str, mode: schema.ElicitationMode, **kwargs: object
    ) -> schema.CreateElicitationResponse:
        raise NotImplementedError  # pragma: no cover - unused

    async def complete_elicitation(self, elicitation_id: str, **kwargs: object) -> None:
        raise NotImplementedError  # pragma: no cover - unused

    async def ext_method(self, method: str, params: dict[str, object]) -> dict[str, object]:
        raise RequestError.method_not_found(method)  # pragma: no cover - unused

    async def ext_notification(self, method: str, params: dict[str, object]) -> None:
        return None  # pragma: no cover - unused


class RecordingClient(RecordingClientBase):
    """An ACP client driving in-memory filesystem/terminal state and recording session updates."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        *,
        output: str = '',
        truncated: bool = False,
        exit_code: int | None = 0,
        signal: str | None = None,
        no_exit_status: bool = False,
        block_exit: bool = False,
        block_create: bool = False,
    ) -> None:
        super().__init__()
        self.files: dict[str, str] = dict(files or {})
        self.reads: list[tuple[str, str]] = []
        self.writes: list[tuple[str, str, str]] = []
        self._output = output
        self._truncated = truncated
        self._exit_code = exit_code
        self._signal = signal
        self._no_exit_status = no_exit_status
        self._block_exit = block_exit
        self._block_create = block_create
        self.release_create = asyncio.Event()
        self.exit_event = asyncio.Event()
        self.created: list[tuple[str, str | None]] = []
        self.killed: list[str] = []
        self.released: list[str] = []
        self._terminals = 0
        self.create_event = asyncio.Event()

    # --- filesystem -----------------------------------------------------------------------

    async def read_text_file(
        self, session_id: str, path: str, line: int | None = None, limit: int | None = None, **kwargs: object
    ) -> schema.ReadTextFileResponse:
        self.reads.append((path, session_id))
        return schema.ReadTextFileResponse(content=self.files[path])

    async def write_text_file(
        self, session_id: str, path: str, content: str, **kwargs: object
    ) -> schema.WriteTextFileResponse | None:
        self.files[path] = content
        self.writes.append((path, content, session_id))
        return None

    # --- terminal -------------------------------------------------------------------------

    async def create_terminal(
        self,
        session_id: str,
        command: str,
        args: list[str] | None = None,
        env: list[schema.EnvVariable] | None = None,
        cwd: str | None = None,
        output_byte_limit: int | None = None,
        **kwargs: object,
    ) -> schema.CreateTerminalResponse:
        self._terminals += 1
        self.created.append((command, cwd))
        self.create_event.set()
        if self._block_create:
            await self.release_create.wait()
        return schema.CreateTerminalResponse(terminal_id=f'term-{self._terminals}')

    async def wait_for_terminal_exit(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> schema.WaitForTerminalExitResponse:
        self.exit_event.set()
        if self._block_exit:
            await asyncio.Event().wait()  # block until cancelled
        return schema.WaitForTerminalExitResponse(exit_code=self._exit_code, signal=self._signal)

    async def terminal_output(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> schema.TerminalOutputResponse:
        status = (
            None if self._no_exit_status else schema.TerminalExitStatus(exit_code=self._exit_code, signal=self._signal)
        )
        return schema.TerminalOutputResponse(output=self._output, truncated=self._truncated, exit_status=status)

    async def kill_terminal(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> schema.KillTerminalResponse | None:
        self.killed.append(terminal_id)
        return None

    async def release_terminal(
        self, session_id: str, terminal_id: str, **kwargs: object
    ) -> schema.ReleaseTerminalResponse | None:
        self.released.append(terminal_id)
        return None
