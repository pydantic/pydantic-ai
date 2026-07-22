"""Modal sandbox toolset: gives agents a cloud sandbox to work in."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated

from pydantic import Field
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from typing_extensions import Self

from pydantic_ai_harness.modal_sandbox._session import (
    ModalSandboxError,
    ModalSandboxSession,
    ModalSandboxTerminalError,
)
from pydantic_ai_harness.modal_sandbox._tool_output import guard_read_size, render_file_window, truncate_output


class ModalSandboxToolset(FunctionToolset[AgentDepsT]):
    """Gives an agent a Modal sandbox to run commands and manage files in.

    Holds the sandbox configuration and, for each run, opens a `ModalSandboxSession`
    (creating a fresh sandbox, or attaching to `sandbox_id`) that the tools execute
    against. When the run ends, an owned session requests termination and waits for
    a bounded period; `sandbox_timeout` is the server-side cleanup backstop.
    """

    def __init__(
        self,
        *,
        image: str,
        sandbox_id: str | None,
        app_name: str,
        create_app_if_missing: bool,
        sandbox_timeout: int,
        workdir: str | None,
        default_command_timeout: float,
        max_command_timeout: int | None,
        max_output_bytes: int,
        max_output_lines: int,
        max_read_bytes: int,
        env: Mapping[str, str] | None = None,
        session: ModalSandboxSession | None = None,
        _run_scoped: bool = False,
    ) -> None:
        super().__init__()
        self._image = image
        self._sandbox_id = sandbox_id
        self._app_name = app_name
        self._create_app_if_missing = create_app_if_missing
        self._sandbox_timeout = sandbox_timeout
        self._workdir = workdir
        self._default_command_timeout = default_command_timeout
        self._max_command_timeout = max_command_timeout
        self._max_output_bytes = max_output_bytes
        self._max_output_lines = max_output_lines
        self._max_read_bytes = max_read_bytes
        self._env = dict(env) if env is not None else None
        # A caller-owned session to reuse instead of opening one per run; when set, this
        # toolset uses it but never opens or closes it.
        self._external_session = session
        self._session: ModalSandboxSession | None = None
        self._run_scoped = _run_scoped

        self.add_function(self.run_command, name='run_command')
        self.add_function(self.read_file, name='read_file')
        self.add_function(self.write_file, name='write_file')
        self.add_function(self.list_directory, name='list_directory')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        """Return a fresh instance per run so each run gets its own sandbox session.

        `get_toolset` builds one shared instance at agent construction; this toolset
        opens a per-run sandbox in `__aenter__`, so each run needs its own instance
        whose `__aexit__` requests that sandbox's termination.
        """
        return ModalSandboxToolset[AgentDepsT](
            image=self._image,
            sandbox_id=self._sandbox_id,
            app_name=self._app_name,
            create_app_if_missing=self._create_app_if_missing,
            sandbox_timeout=self._sandbox_timeout,
            workdir=self._workdir,
            default_command_timeout=self._default_command_timeout,
            max_command_timeout=self._max_command_timeout,
            max_output_bytes=self._max_output_bytes,
            max_output_lines=self._max_output_lines,
            max_read_bytes=self._max_read_bytes,
            env=self._env,
            session=self._external_session,
            _run_scoped=True,
        )

    async def __aenter__(self) -> Self:
        """Make a sandbox session available before tools run.

        With a caller-owned `session`, use it as-is (the caller opened it). Otherwise open a
        per-run session here so each run gets its own sandbox.
        """
        if not self._run_scoped:
            return self
        if self._external_session is not None:
            # The caller owns this session and must open it before the run; check here so an
            # unopened session fails at run start with a clear message, not mid-tool-call.
            if self._external_session.sandbox_id is None:
                raise ModalSandboxError(
                    'The injected session is not open. Enter it with `async with session:` before running the agent.'
                )
            self._session = self._external_session
            return self
        session = ModalSandboxSession(
            image=self._image,
            sandbox_id=self._sandbox_id,
            app_name=self._app_name,
            create_app_if_missing=self._create_app_if_missing,
            sandbox_timeout=self._sandbox_timeout,
            workdir=self._workdir,
            env=self._env,
        )
        await session.__aenter__()
        self._session = session
        return self

    async def __aexit__(self, *args: object) -> None:
        """Close the per-run session; leave a caller-owned session for its owner to close."""
        session = self._session
        self._session = None
        if session is not None and self._external_session is None:
            await session.__aexit__(*args)

    def _require_session(self) -> ModalSandboxSession:
        if self._session is None:
            # Reachable by calling a tool on an instance that was never entered (e.g. the
            # base toolset outside an agent run). That is a caller error, not a model
            # mistake, so the typed error propagates rather than becoming a ModelRetry.
            raise ModalSandboxError('The Modal sandbox session is not open.')
        return self._session

    def _truncate_stream(self, text: str, already_truncated: bool) -> str:
        return truncate_output(
            text,
            max_lines=self._max_output_lines,
            max_bytes=self._max_output_bytes,
            direction='tail',
            already_truncated=already_truncated,
        )

    def _command_timeout(self, timeout_seconds: float | None) -> int:
        if timeout_seconds is not None and (not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            # Reject rather than let the session floor it to 1s: a 0 or negative request is a
            # model mistake, and a surprise "[timed out after 1s]" hides that from the model.
            raise ModelRetry(f'timeout_seconds must be greater than 0, got {timeout_seconds}.')
        requested = timeout_seconds if timeout_seconds is not None else self._default_command_timeout
        # Clamp to a hard ceiling. Modal cannot kill a running command, so a cancelled one
        # runs until its deadline; the ceiling bounds that worst case. It defaults to the
        # sandbox lifetime, beyond which an owned command cannot run anyway. Round up before
        # clamping so the whole-second Modal deadline cannot exceed the configured ceiling.
        ceiling = self._max_command_timeout if self._max_command_timeout is not None else self._sandbox_timeout
        return min(max(1, math.ceil(requested)), ceiling)

    async def run_command(self, command: str, *, timeout_seconds: float | None = None) -> str:
        """Run a shell command in the sandbox and return its output.

        The command runs through `sh -c`, so pipes, redirection, `&&`, and globs
        work. A non-zero exit is reported, not raised, so you can react to it.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: the configured timeout).

        Returns:
            Labelled stdout/stderr output, with an exit code on non-zero exit.
        """
        session = self._require_session()
        # Surface a recoverable sandbox-side failure as a retryable tool error, matching the
        # file tools. A terminal failure (the sandbox is gone, or credentials were rejected)
        # propagates instead: retrying the command cannot fix it, so end the run cleanly.
        try:
            result = await session.exec(
                ['sh', '-c', command],
                timeout=self._command_timeout(timeout_seconds),
                max_output_bytes=self._max_output_bytes,
            )
        except ModalSandboxTerminalError:
            raise
        except ModalSandboxError as e:
            raise ModelRetry(str(e))
        # Truncate each stream separately and attach its label afterwards, so the
        # `[stdout]` / `[stderr]` markers always survive truncation and a large stderr
        # cannot crowd stdout out of a shared budget. Tail direction: errors and the
        # exit status live at the end.
        parts: list[str] = []
        if result.stdout:
            parts.append(f'[stdout]\n{self._truncate_stream(result.stdout, result.stdout_truncated)}')
        if result.stderr:
            parts.append(f'[stderr]\n{self._truncate_stream(result.stderr, result.stderr_truncated)}')
        output = '\n'.join(parts) if parts else '(no output)'
        if result.timed_out:
            return f'{output}\n[timed out after {result.applied_timeout}s]'
        if result.returncode:
            return f'{output}\n[exit code: {result.returncode}]'
        return output

    async def read_file(
        self,
        path: str,
        *,
        offset: Annotated[int | None, Field(description='Line number to start reading from (1-indexed)')] = None,
        limit: Annotated[int | None, Field(description='Maximum number of lines to read')] = None,
    ) -> str:
        """Read a text file from the sandbox and return its contents.

        Large files are truncated to a safety cap; the result ends with the next
        `offset` to use to page through the rest.

        Args:
            path: Path to the file inside the sandbox. Relative paths are resolved
                against the working directory used by `run_command`.
            offset: Line number to start reading from (1-indexed).
            limit: Maximum number of lines to read.
        """
        session = self._require_session()
        try:
            # Check size first: read_bytes pulls the whole file into memory before windowing,
            # so refuse an oversized file rather than transfer and decode all of it for a slice.
            guard_read_size(await session.file_size(path), max_bytes=self._max_read_bytes)
            data = await session.read_bytes(path)
        except ModalSandboxTerminalError:
            raise
        except ModalSandboxError as e:
            raise ModelRetry(f'Could not read {path!r}: {e}')
        # Re-check against the bytes actually returned. The stat and the read are separate
        # round-trips, so the file could have grown past the limit in between. Modal's API
        # has no bounded read, so the transfer already happened, but refusing here still
        # avoids the large UTF-8 decode and windowing that would otherwise follow.
        guard_read_size(len(data), max_bytes=self._max_read_bytes)
        return render_file_window(
            data, offset=offset, limit=limit, max_lines=self._max_output_lines, max_bytes=self._max_output_bytes
        )

    async def write_file(self, path: str, content: str) -> str:
        """Write text to a file in the sandbox, creating parent directories.

        Args:
            path: Path to the file inside the sandbox. Relative paths are resolved
                against the working directory used by `run_command`.
            content: The text to write.
        """
        session = self._require_session()
        try:
            data = content.encode('utf-8')
        except UnicodeEncodeError:
            # Reachable when a provider's pre-parsed tool arguments carry an unpaired
            # surrogate; a model mistake, so retry rather than abort the run.
            raise ModelRetry('content contains characters that cannot be encoded as UTF-8 (unpaired surrogates).')
        try:
            await session.write_bytes(path, data)
        except ModalSandboxTerminalError:
            raise
        except ModalSandboxError as e:
            raise ModelRetry(f'Could not write {path!r}: {e}')
        return f'Wrote {len(data)} bytes to {path!r}.'

    async def list_directory(self, path: str = '.') -> str:
        """List the entries in a sandbox directory (directories shown with a trailing `/`).

        Args:
            path: Directory to list. Relative paths (including the default `.`) are
                resolved against the working directory used by `run_command`.
        """
        session = self._require_session()
        try:
            entries = await session.list_files(path)
        except ModalSandboxTerminalError:
            raise
        except ModalSandboxError as e:
            raise ModelRetry(f'Could not list {path!r}: {e}')
        if not entries:
            return '(empty)'
        # Sort by name before adding the `/` suffix so directories keep plain name order
        # ('/' sorts after '-' and '.', which would misplace suffixed names).
        names = [f'{name}/' if is_dir else name for name, is_dir in sorted(entries)]
        # Directory listing is sorted, so keep the head if it overflows the cap.
        return truncate_output(
            '\n'.join(names), max_lines=self._max_output_lines, max_bytes=self._max_output_bytes, direction='head'
        )
