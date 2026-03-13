"""Local subprocess-based execution environment for development and testing.

Runs commands directly on the host machine within a specified root directory.
**No isolation** — use `DockerEnvironment` for untrusted code.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Literal

import anyio
import anyio.abc
from typing_extensions import Self

from ._base import (
    IMAGE_EXTENSIONS,
    MAX_OUTPUT_CHARS,
    EnvToolName,
    ExecutionEnvironment,
    ExecutionProcess,
    ExecutionResult,
    FileInfo,
    apply_edit,
    collect_grep_matches,
    format_lines,
)


class _LocalEnvironmentProcess(ExecutionProcess):
    """Interactive process backed by `anyio.abc.Process`."""

    def __init__(self, proc: anyio.abc.Process) -> None:
        self._proc = proc

    async def send(self, data: bytes) -> None:
        stdin = self._proc.stdin
        if stdin is None:
            raise RuntimeError('Process stdin is not available.')
        await stdin.send(data)

    async def recv(self, timeout: float | None = None) -> bytes:
        stdout = self._proc.stdout
        if stdout is None:
            raise RuntimeError('Process stdout is not available.')
        try:
            if timeout is not None:
                with anyio.fail_after(timeout):
                    return await stdout.receive(8192)
            return await stdout.receive(8192)
        except anyio.EndOfStream:
            return b''

    async def recv_stderr(self, timeout: float | None = None) -> bytes:
        stderr = self._proc.stderr
        if stderr is None:
            raise RuntimeError('Process stderr is not available.')
        try:
            if timeout is not None:
                with anyio.fail_after(timeout):
                    return await stderr.receive(8192)
            return await stderr.receive(8192)
        except anyio.EndOfStream:
            return b''

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    async def wait(self, timeout: float | None = None) -> int:
        if timeout is not None:
            with anyio.fail_after(timeout):
                return await self._proc.wait()
        return await self._proc.wait()

    async def kill(self) -> None:
        try:
            self._proc.kill()
        except ProcessLookupError:
            pass
        await self._proc.aclose()
        _close_subprocess_transport(self._proc)


def _close_subprocess_transport(proc: anyio.abc.Process) -> None:
    """Close the underlying asyncio subprocess transport to prevent ResourceWarning on Python 3.10.

    On Python 3.10, asyncio subprocess transports are not closed by
    `Process.wait()` or `Process.aclose()` and their `__del__`
    emits `ResourceWarning: unclosed transport`.  Python 3.11+ fixed
    this, but we still support 3.10.
    """
    inner = getattr(proc, '_process', None)  # anyio wraps asyncio.subprocess.Process
    transport = getattr(inner, '_transport', None)
    if transport is not None:  # pragma: no branch
        transport.close()


class LocalEnvironment(ExecutionEnvironment):
    """Local subprocess-based execution environment for development and testing.

    Runs commands directly on the host machine within a specified root
    directory. Provides no isolation — use `DockerEnvironment` for untrusted code.

    Usage:
        ```python {test="skip" lint="skip"}
        async with LocalEnvironment(root_dir='/tmp/workspace') as env:
            result = await env.shell('python script.py')
            print(result.output)
        ```
    """

    def __init__(
        self,
        root_dir: str | Path = '.',
        *,
        env_vars: dict[str, str] | None = None,
        inherit_env: bool = True,
    ) -> None:
        """Create a local execution environment.

        Args:
            root_dir: The working directory for all operations.
                Defaults to the current directory.
            env_vars: Baseline environment variables for all commands.
            inherit_env: Whether to inherit the host's environment variables.
                When True (default), `env_vars` and per-call `env` are merged
                on top of `os.environ`. When False, only `env_vars` and per-call
                `env` are used (useful for reproducibility and testing).
        """
        self._root_dir = Path(root_dir).resolve()
        self._env_vars = env_vars or {}
        self._inherit_env = inherit_env

    @property
    def capabilities(self) -> frozenset[EnvToolName]:
        return frozenset({'ls', 'shell', 'read_file', 'write_file', 'edit_file', 'glob', 'grep'})

    async def __aenter__(self) -> Self:
        self._root_dir.mkdir(parents=True, exist_ok=True)
        return self

    async def __aexit__(self, *_args: Any) -> None:
        pass

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to root_dir, preventing traversal."""
        resolved = (self._root_dir / path).resolve()
        if not resolved.is_relative_to(self._root_dir):
            raise PermissionError(f'Path {path!r} resolves outside the environment root.')
        return resolved

    def _build_env(self, env: dict[str, str] | None) -> dict[str, str] | None:
        """Merge baseline env vars with per-call overrides."""
        if not self._env_vars and not env and self._inherit_env:
            return None  # subprocess inherits naturally
        import os

        merged = {**os.environ} if self._inherit_env else {}
        merged.update(self._env_vars)
        if env:
            merged.update(env)
        return merged

    async def create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
    ) -> ExecutionProcess:
        proc = await anyio.open_process(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._root_dir,
            env=self._build_env(env),
        )
        return _LocalEnvironmentProcess(proc)

    async def shell(
        self,
        command: str,
        *,
        timeout: float | None = 120,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a command using subprocess for simplicity and reliability."""
        proc = await anyio.open_process(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=self._root_dir,
            env=self._build_env(env),
        )
        try:
            assert proc.stdout is not None
            chunks: list[bytes] = []
            if timeout is not None:
                with anyio.fail_after(timeout):
                    async for chunk in proc.stdout:
                        chunks.append(chunk)
                    await proc.wait()
            else:
                async for chunk in proc.stdout:
                    chunks.append(chunk)
                await proc.wait()
        except TimeoutError:
            proc.kill()
            with anyio.CancelScope(shield=True):
                await proc.wait()
            _close_subprocess_transport(proc)
            return ExecutionResult(output='[Command timed out]', exit_code=-1)

        _close_subprocess_transport(proc)
        stdout = b''.join(chunks)
        output = stdout.decode('utf-8', errors='replace')
        truncated = len(output) > MAX_OUTPUT_CHARS
        if truncated:
            output = output[:MAX_OUTPUT_CHARS]
        return ExecutionResult(
            output=output,
            exit_code=proc.returncode if proc.returncode is not None else 0,
            truncated=truncated,
        )

    async def read_file(self, path: str, *, offset: int = 0, limit: int = 2000) -> str | bytes:
        resolved = self._resolve_path(path)
        if not resolved.is_file():
            if resolved.is_dir():
                raise FileNotFoundError(f"'{path}' is a directory, not a file.")
            raise FileNotFoundError(f'File not found: {path}')

        if resolved.suffix.lower() in IMAGE_EXTENSIONS:
            return resolved.read_bytes()

        raw = resolved.read_bytes()
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw
        return format_lines(text, offset, limit)

    async def write_file(self, path: str, content: str | bytes) -> None:
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            resolved.write_bytes(content)
        else:
            resolved.write_text(content, encoding='utf-8')

    async def replace_str(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> int:
        resolved = self._resolve_path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f'File not found: {path}')

        text = resolved.read_text(encoding='utf-8')
        new_text, count = apply_edit(text, old, new, path, replace_all=replace_all)
        resolved.write_text(new_text, encoding='utf-8')
        return count

    async def ls(self, path: str = '.') -> list[FileInfo]:
        resolved = self._resolve_path(path)
        if not resolved.is_dir():
            raise NotADirectoryError(f'Not a directory: {path}')

        entries: list[FileInfo] = []
        for entry in sorted(resolved.iterdir()):
            try:
                stat = entry.stat()
                entries.append(
                    FileInfo(
                        name=entry.name,
                        path=str(entry.relative_to(self._root_dir)),
                        is_dir=entry.is_dir(),
                        size=stat.st_size if not entry.is_dir() else None,
                    )
                )
            except OSError:  # pragma: no cover
                continue
        return entries

    async def glob(self, pattern: str, *, path: str = '.') -> list[str]:
        resolved = self._resolve_path(path)
        matches: list[str] = []
        for match in sorted(resolved.glob(pattern)):
            try:
                rel = str(match.relative_to(self._root_dir))
                matches.append(rel)
            except ValueError:  # pragma: no cover
                continue
        return matches

    async def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        glob_pattern: str | None = None,
        output_mode: Literal['content', 'files_with_matches', 'count'] = 'content',
    ) -> str:
        """Search file contents using a regex pattern (Python `re` module syntax)."""
        search_dir = self._resolve_path(path or '.')
        compiled = re.compile(pattern)

        is_explicit_file = search_dir.is_file()
        if is_explicit_file:
            files = [search_dir]
        elif glob_pattern:
            files = sorted(search_dir.rglob(glob_pattern))
        else:
            files = sorted(search_dir.rglob('*'))

        results: list[str] = []
        for file_path in files:
            if not file_path.is_file():
                continue
            # Skip hidden files/directories (e.g. .git/, .venv/) unless explicitly specified
            if not is_explicit_file and any(
                part.startswith('.') for part in file_path.relative_to(self._root_dir).parts
            ):
                continue
            try:
                raw = file_path.read_bytes()
            except OSError:  # pragma: no cover
                continue

            # Skip binary files (null byte in first 8KB)
            if b'\x00' in raw[:8192]:
                continue

            text = raw.decode('utf-8', errors='replace')
            rel_path = str(file_path.relative_to(self._root_dir))
            collect_grep_matches(rel_path, text, compiled, output_mode, results)

            if len(results) > 1000:
                results.append('[... truncated at 1000 matches]')
                break

        return '\n'.join(results)
