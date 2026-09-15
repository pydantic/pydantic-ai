"""The six tools exposed by Coder."""

from __future__ import annotations

import os
import re
import stat
import subprocess
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Literal

import anyio
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.coder._shell import shell
from pydantic_ai_harness.filesystem import FileSystem, FileSystemToolset
from pydantic_ai_harness.filesystem._changes import Change
from pydantic_ai_harness.filesystem._toolset import (
    _content_hash,  # pyright: ignore[reportPrivateUsage]
    _write_content,  # pyright: ignore[reportPrivateUsage]
)


@dataclass
class Replacement:
    """One exact, unique replacement in an edit batch."""

    _: KW_ONLY
    old_text: str
    new_text: str


class CoderToolset(FunctionToolset[AgentDepsT]):
    """Focused local coding tools. Shell access requires a trusted workspace."""

    def __init__(self, workspace: Path) -> None:
        super().__init__(id='coder')
        self.workspace = workspace.resolve()
        filesystem = FileSystem[AgentDepsT](self.workspace).get_toolset()
        assert isinstance(filesystem, FileSystemToolset)
        self.filesystem = filesystem
        self.add_function(self.read_file)
        self.add_function(self.write_file)
        self.add_function(self.edit_file)
        self.add_function(self.list_files)
        self.add_function(self.grep)
        self.add_function(self.shell)

    async def read_file(
        self, ctx: RunContext[AgentDepsT], path: str, *, offset: int = 0, limit: int | None = None
    ) -> str:
        """Read a file with zero-based offset and one-based line numbers, without hashes."""
        if offset < 0 or (limit is not None and limit <= 0):
            raise ModelRetry('offset must be non-negative and limit positive.')
        limit = min(limit or 2000, 2000)
        try:
            resolved = self.filesystem._safe_resolve(path)  # pyright: ignore[reportPrivateUsage]
            flags = os.O_RDONLY | (os.O_BINARY if os.name == 'nt' else os.O_NONBLOCK | os.O_NOFOLLOW)
            descriptor = os.open(resolved, flags)
            with os.fdopen(descriptor, 'rb') as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ModelRetry('Reads require a regular file.')
                output: list[str] = []
                number = 0
                budget = 60000
                while budget > 0:
                    line = source.readline(65536)
                    if not line:
                        break
                    if b'\0' in line:
                        return '[Binary file; use a binary-aware tool.]'
                    number += 1
                    if len(line) == 65536 and not line.endswith(b'\n'):
                        return '[Line exceeds 65,536 bytes; use shell for a bounded byte-range inspection.]'
                    if number <= offset:
                        continue
                    rendered = f'{number}: {line.decode("utf-8", errors="replace")}'
                    output.append(rendered[:budget])
                    budget -= len(rendered)
                    if len(output) >= limit:
                        break
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelRetry(f'Cannot read {path!r}: {exc}') from exc
        return f'[{path}]\n' + ''.join(output) + '\n[Read window; use offset/limit to continue.]'

    async def write_file(self, ctx: RunContext[AgentDepsT], path: str, content: str) -> str:
        """Write a complete file. Create missing parent directories with shell mkdir first."""
        result = await self.filesystem._write_file(ctx, path, content)  # pyright: ignore[reportPrivateUsage]
        return re.sub(r' \[hash:[0-9a-f]+\]', '', result)

    async def edit_file(
        self,
        ctx: RunContext[AgentDepsT],
        path: str,
        *,
        old_text: str | None = None,
        new_text: str | None = None,
        replacements: list[Replacement] | None = None,
    ) -> str:
        """Apply one exact replacement or a sequential batch; each match must occur once.

        All replacements are checked in memory before the file is changed.
        """
        if replacements is None:
            if old_text is None or new_text is None:
                raise ModelRetry('Provide old_text and new_text, or a non-empty replacements list.')
            replacements = [Replacement(old_text=old_text, new_text=new_text)]
        elif old_text is not None or new_text is not None or not replacements:
            raise ModelRetry('Use either old_text/new_text or a non-empty replacements list, not both.')
        try:
            resolved = self.filesystem._safe_resolve(path, write=True)  # pyright: ignore[reportPrivateUsage]
            flags = os.O_RDONLY | (os.O_BINARY if os.name == 'nt' else os.O_NONBLOCK | os.O_NOFOLLOW)
            descriptor = os.open(resolved, flags)
            with os.fdopen(descriptor, 'rb') as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ModelRetry('Edits require a regular file.')
                original = source.read().decode('utf-8')
                if '\0' in original:
                    raise ModelRetry('Edits require a text file without NUL bytes.')
        except (OSError, UnicodeError) as exc:
            raise ModelRetry(f'Cannot read {path!r}: {exc}') from exc
        content = original
        for replacement in replacements:
            if not replacement.old_text or content.count(replacement.old_text) != 1:
                raise ModelRetry('Each non-empty old_text must occur exactly once; no changes were written.')
            content = content.replace(replacement.old_text, replacement.new_text, 1)
        change = Change.propose(
            **self.filesystem._event_location(resolved),  # pyright: ignore[reportPrivateUsage]
            operation='edit',
            old=original,
            new=content,
        )
        refusal = await self.filesystem._request(  # pyright: ignore[reportPrivateUsage]
            ctx, change, path=path, resolved=resolved
        )
        if refusal is not None:
            return refusal
        try:
            _write_content(resolved, path, content, expected_hash=_content_hash(original), create=False)
        except (OSError, ValueError) as exc:
            raise ModelRetry(f'Cannot edit {path!r}: {exc}') from exc
        await ctx.emit(change.edited(content_hash=_content_hash(content)))
        return f'Edited {path}.'

    def _directory(self, path: str) -> Path:
        try:
            directory = (self.workspace / path).resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ModelRetry(f'Cannot resolve directory: {exc}') from exc
        if not directory.is_relative_to(self.workspace) or not directory.is_dir():
            raise ModelRetry('path must be an existing directory inside the workspace.')
        return directory

    async def _rg(self, arguments: list[str], *, path: str, limit: int) -> str:
        if not 1 <= limit <= 1000:
            raise ModelRetry('limit must be between 1 and 1000.')
        try:
            async with await anyio.open_process(
                ['rg', *arguments], cwd=self._directory(path), stderr=subprocess.DEVNULL
            ) as process:
                assert process.stdout is not None
                output = bytearray()
                truncated = False
                async for chunk in process.stdout:
                    output.extend(chunk)
                    if output.count(b'\n') > limit or len(output) > 64000:
                        truncated = True
                        process.terminate()
                        break
                await process.wait()
                if process.returncode not in (0, 1) and not truncated:
                    raise ModelRetry('ripgrep failed; check the pattern and filters.')
        except FileNotFoundError as exc:
            raise ModelRetry('Install ripgrep (rg) to use list_files and grep.') from exc
        lines = output.decode('utf-8', errors='replace').splitlines()
        marker = '\n[truncated; narrow the search]' if truncated else ''
        result = '\n'.join(lines[:limit])[: 64000 - len(marker)]
        return result + marker

    async def list_files(self, path: str = '.', *, glob: str | None = None, limit: int = 200) -> str:
        """List files with rg --files, respecting ignore rules and optional glob filtering."""
        arguments = ['--files', '--color=never']
        if glob is not None:
            arguments.extend(['--glob', glob])
        return await self._rg(arguments, path=path, limit=limit)

    async def grep(
        self,
        pattern: str,
        *,
        path: str = '.',
        glob: str | None = None,
        file_type: str | None = None,
        ignore_case: bool = False,
        literal: bool = False,
        context: int = 0,
        limit: int = 200,
    ) -> str:
        """Search with ripgrep, bounding returned lines including optional context."""
        if not 0 <= context <= 20:
            raise ModelRetry('context must be between 0 and 20.')
        arguments = ['--line-number', '--with-filename', '--color=never', '--context', str(context)]
        if glob is not None:
            arguments.extend(['--glob', glob])
        if file_type is not None:
            arguments.extend(['--type', file_type])
        if ignore_case:
            arguments.append('--ignore-case')
        if literal:
            arguments.append('--fixed-strings')
        arguments.extend(['--regexp', pattern, '--', '.'])
        return await self._rg(arguments, path=path, limit=limit)

    async def shell(
        self,
        command: str,
        *,
        mode: Literal['foreground', 'background'] = 'foreground',
        timeout: float = 270,
    ) -> str:
        """Run unrestricted commands. Foreground promotes after timeout (maximum 270s).

        Returns PID, output log and status file paths. Background processes survive
        agent runs; use shell to inspect logs/status and kill processes when done.
        """
        return await shell(self.workspace, command, mode=mode, timeout=timeout)
