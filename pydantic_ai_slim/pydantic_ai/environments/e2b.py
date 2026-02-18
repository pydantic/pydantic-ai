"""Hosted sandbox via E2B (https://e2b.dev).

Runs code in a cloud-hosted micro-VM with a full Linux environment.
Requires the `e2b-code-interpreter` package: `pip install pydantic-ai-slim[e2b-sandbox]`

Note: pyright errors are suppressed at the top of this file because the
`e2b-code-interpreter` package ships incomplete type stubs (missing attributes,
unknown member types, etc.) that cannot be worked around without pervasive casts.
"""

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownLambdaType=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportUnnecessaryComparison=false

from __future__ import annotations

import pathlib
import posixpath
from typing import Any, Literal

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from typing_extensions import Self

from ._base import (
    IMAGE_EXTENSIONS,
    MAX_OUTPUT_CHARS,
    Capability,
    ExecuteResult,
    ExecutionProcess,
    FileInfo,
    ToolName,
    apply_edit,
    build_glob_cmd,
    build_grep_cmd,
    build_read_file_cmd,
    filter_grep_count_output,
    parse_glob_output,
    shell_escape,
)
from ._driver import DriverBasedEnvironment

try:
    from e2b_code_interpreter import AsyncSandbox as E2BAsyncSandbox
except ImportError as _import_error:
    raise ImportError(
        'The `e2b-code-interpreter` package is required for E2BEnvironment. '
        'Install it with: pip install pydantic-ai-slim[e2b-sandbox]'
    ) from _import_error


class E2BEnvironmentProcess(ExecutionProcess):
    """Interactive process in an E2B sandbox, bridging callback-based I/O to pull-based recv()."""

    def __init__(self, sandbox: E2BAsyncSandbox, command: str, env: dict[str, str] | None = None) -> None:
        self._sandbox = sandbox
        self._command = command
        self._env = env
        self._proc: Any = None
        self._stdout_send: MemoryObjectSendStream[bytes]
        self._stdout_recv: MemoryObjectReceiveStream[bytes]
        self._stdout_send, self._stdout_recv = anyio.create_memory_object_stream[bytes](max_buffer_size=float('inf'))
        self._stderr_send: MemoryObjectSendStream[bytes]
        self._stderr_recv: MemoryObjectReceiveStream[bytes]
        self._stderr_send, self._stderr_recv = anyio.create_memory_object_stream[bytes](max_buffer_size=float('inf'))
        self._returncode: int | None = None

    async def _start(self) -> None:
        """Start the command in the sandbox (called from __aenter__)."""
        kwargs: dict[str, Any] = {
            'on_stdout': lambda data: self._stdout_send.send_nowait(data.line.encode() + b'\n'),
            'on_stderr': lambda data: self._stderr_send.send_nowait(data.line.encode() + b'\n'),
            'on_exit': lambda exit_code: self._on_exit(exit_code),
        }
        if self._env:
            kwargs['envs'] = self._env
        self._proc = await self._sandbox.commands.start(
            self._command,
            **kwargs,
        )

    async def __aenter__(self) -> Self:
        if self._proc is None:
            await self._start()
        return self

    def _on_exit(self, exit_code: int) -> None:
        self._returncode = exit_code

    async def send(self, data: bytes) -> None:
        await self._proc.send_stdin(data.decode('utf-8', errors='replace'))

    async def recv(self, timeout: float | None = None) -> bytes:
        if timeout is not None:
            with anyio.fail_after(timeout):
                return await self._stdout_recv.receive()
        return await self._stdout_recv.receive()

    async def recv_stderr(self, timeout: float | None = None) -> bytes:
        if timeout is not None:
            with anyio.fail_after(timeout):
                return await self._stderr_recv.receive()
        return await self._stderr_recv.receive()

    @property
    def returncode(self) -> int | None:
        return self._returncode

    async def wait(self, timeout: float | None = None) -> int:
        async def _poll() -> int:
            while self._returncode is None:
                await anyio.sleep(0.1)
            return self._returncode

        if timeout is not None:
            with anyio.fail_after(timeout):
                return await _poll()
        return await _poll()

    async def kill(self) -> None:
        if self._proc is not None:
            try:
                await self._proc.kill()
            except Exception:
                pass


class E2BEnvironment(DriverBasedEnvironment):
    """Hosted sandbox via E2B (https://e2b.dev).

    Runs code in a cloud-hosted micro-VM with a full Linux environment.
    Requires an E2B API key (set via `E2B_API_KEY` env var or `api_key` parameter).

    Usage:
        ```python {test="skip" lint="skip"}
        async with E2BEnvironment(template='base') as env:
            result = await env.shell('echo hello')
            print(result.output)
        ```
    """

    def __init__(
        self,
        template: str = 'base',
        *,
        api_key: str | None = None,
        timeout: float = 300,
        metadata: dict[str, str] | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> None:
        """Create an E2B environment.

        Args:
            template: The E2B sandbox template to use.
            api_key: E2B API key. Defaults to the `E2B_API_KEY` environment variable.
            timeout: Sandbox timeout in seconds (how long the VM stays alive).
            metadata: Optional metadata to attach to the sandbox.
            env_vars: Baseline environment variables for all commands.
        """
        self._template = template
        self._api_key = api_key
        self._timeout = int(timeout)
        self._metadata = metadata or {}
        self._env_vars = env_vars or {}
        self._sandbox: E2BAsyncSandbox | None = None

    async def __aenter__(self) -> Self:
        kwargs: dict[str, Any] = {
            'template': self._template,
            'timeout': self._timeout,
        }
        if self._api_key:
            kwargs['api_key'] = self._api_key
        if self._metadata:
            kwargs['metadata'] = self._metadata
        if self._env_vars:
            kwargs['envs'] = self._env_vars
        self._sandbox = await E2BAsyncSandbox.create(**kwargs)
        return self

    async def __aexit__(self, *_args: Any) -> None:
        if self._sandbox is not None:
            try:
                await self._sandbox.close()
            except Exception:
                # Best-effort cleanup: sandbox may already be closed or unreachable
                pass
            self._sandbox = None

    @property
    def sandbox(self) -> E2BAsyncSandbox:
        if self._sandbox is None:
            raise RuntimeError('E2BEnvironment not started. Use `async with E2BEnvironment(...) as env:`')
        return self._sandbox

    @property
    def capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {
                'ls',
                'shell',
                'read_file',
                'write_file',
                'edit_file',
                'glob',
                'grep',
                'run_code',
                'run_code_with_functions',
            }
        )

    def tool_description(self, tool: ToolName) -> str | None:
        if tool == 'grep':
            return 'Uses POSIX basic regex, not Python `re` syntax.'
        if tool == 'glob':
            return 'Uses `find` for pattern matching; `**` is not supported.'
        return None

    def _merge_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        """Merge per-call env vars with baseline."""
        if not self._env_vars and not env:
            return None
        merged = {**self._env_vars}
        if env:
            merged.update(env)
        return merged

    async def create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
    ) -> ExecutionProcess:
        return E2BEnvironmentProcess(self.sandbox, command, env=self._merge_env(env))

    async def shell(
        self,
        command: str,
        *,
        timeout: float | None = 120,
        env: dict[str, str] | None = None,
    ) -> ExecuteResult:
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs['timeout'] = int(timeout)
        merged_env = self._merge_env(env)
        if merged_env:
            kwargs['envs'] = merged_env
        result = await self.sandbox.commands.run(command, **kwargs)
        output = result.stdout + result.stderr
        truncated = len(output) > MAX_OUTPUT_CHARS
        if truncated:
            output = output[:MAX_OUTPUT_CHARS]
        return ExecuteResult(
            output=output,
            exit_code=result.exit_code,
            truncated=truncated,
        )

    async def read_file(self, path: str, *, offset: int = 0, limit: int = 2000) -> str | bytes:
        ext = posixpath.splitext(path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            content = await self.sandbox.files.read(path, format='bytes')
            return content  # type: ignore[return-value]

        cmd = build_read_file_cmd(path, offset=offset, limit=limit)
        result = await self.sandbox.commands.run(cmd)
        if result.exit_code != 0:
            raise FileNotFoundError(f'File not found or not readable: {path}')
        return result.stdout

    async def write_file(self, path: str, content: str | bytes) -> None:
        # Ensure parent directory exists
        parent = path.rsplit('/', 1)[0] if '/' in path else '.'
        if parent != '.':
            await self.sandbox.commands.run(f'mkdir -p {shell_escape(parent)}')

        if isinstance(content, bytes):
            await self.sandbox.files.write(path, content, format='bytes')  # type: ignore[call-arg]
        else:
            await self.sandbox.files.write(path, content)

    async def replace_str(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> int:
        raw_content = await self.sandbox.files.read(path)
        text: str = raw_content if isinstance(raw_content, str) else raw_content.decode('utf-8')

        new_text, count = apply_edit(text, old, new, path, replace_all=replace_all)

        await self.sandbox.files.write(path, new_text)
        return count

    async def ls(self, path: str = '.') -> list[FileInfo]:
        entries_raw = await self.sandbox.files.list(path)
        entries: list[FileInfo] = []
        for entry in entries_raw:
            entries.append(
                FileInfo(
                    name=entry.name,
                    path=f'{path}/{entry.name}' if path != '.' else entry.name,
                    is_dir=entry.type == 'dir',
                    size=None,  # E2B doesn't always provide size in list
                )
            )
        return entries

    async def glob(self, pattern: str, *, path: str = '.') -> list[str]:
        cmd = build_glob_cmd(pattern, path=path)
        result = await self.sandbox.commands.run(cmd)
        return parse_glob_output(result.stdout)

    async def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        glob_pattern: str | None = None,
        output_mode: Literal['content', 'files_with_matches', 'count'] = 'content',
    ) -> str:
        cmd = build_grep_cmd(pattern, path=path, glob_pattern=glob_pattern, output_mode=output_mode)
        result = await self.sandbox.commands.run(cmd)
        text = result.stdout.strip()
        if output_mode == 'count':
            text = filter_grep_count_output(text)
        return text

    async def _copy_driver(self) -> None:
        driver_source = pathlib.Path(__file__).parents[1] / 'toolsets' / 'code_execution' / '_driver.py'
        content = driver_source.read_text()
        await self.sandbox.files.write(self.driver_script_path, content)
