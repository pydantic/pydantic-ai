"""Docker container-based environment for isolated code execution.

Requires the `docker` package: `pip install pydantic-ai-slim[docker-environment]`
"""

from __future__ import annotations

import io
import math
import posixpath
import struct
import tarfile
from pathlib import PurePosixPath
from typing import Any, Literal, cast

import anyio
import anyio.to_thread
from typing_extensions import Self

from ._base import (
    IMAGE_EXTENSIONS,
    MAX_OUTPUT_CHARS,
    EnvCapability,
    ExecutionEnvironment,
    ExecutionProcess,
    ExecutionResult,
    FileInfo,
    apply_edit,
    build_glob_cmd,
    build_grep_cmd,
    build_read_file_cmd,
    filter_grep_count_output,
    parse_glob_output,
    shell_escape,
)

try:
    import docker
    from docker.errors import DockerException
    from docker.models.containers import Container
except ImportError as _import_error:
    raise ImportError(
        'The `docker` package is required for DockerEnvironment. '
        'Install it with: pip install pydantic-ai-slim[docker-environment]'
    ) from _import_error


def _put_file(container: Container, path: str, data: bytes) -> None:
    """Write file data into a container via put_archive."""
    parent = str(PurePosixPath(path).parent)
    filename = PurePosixPath(path).name
    f = io.BytesIO()
    with tarfile.open(fileobj=f, mode='w') as tar:
        info = tarfile.TarInfo(name=filename)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    f.seek(0)
    container.put_archive(parent, f)  # pyright: ignore[reportUnknownMemberType]


class DockerEnvironmentProcess(ExecutionProcess):
    """Interactive process inside a Docker container using exec with socket I/O.

    Docker's exec socket uses a multiplexed stream protocol where stdout and
    stderr frames are interleaved with 8-byte headers indicating the stream
    type.  This class properly separates the two streams so that `recv()`
    returns only stdout data and `recv_stderr()` returns only stderr data.
    When one stream is requested but the other arrives first, the unexpected
    frame is buffered for the next call to the appropriate method.
    """

    _STDOUT = 1
    _STDERR = 2

    def __init__(self, container: Container, command: str, work_dir: str, env: dict[str, str] | None = None) -> None:
        self._container = container
        self._command = command
        self._work_dir = work_dir
        self._env = env
        self._exec_id: str | None = None
        self._socket: Any = None
        self._returncode: int | None = None
        self._stdout_buffer: list[bytes] = []
        self._stderr_buffer: list[bytes] = []
        self._eof = False

    async def _start(self) -> None:
        """Start the exec and open the socket (called from __aenter__)."""

        def _do_start() -> tuple[str, Any]:
            client: Any = self._container.client
            kwargs: dict[str, Any] = {
                'stdin': True,
                'stdout': True,
                'stderr': True,
                'workdir': self._work_dir,
            }
            if self._env:
                kwargs['environment'] = self._env
            exec_id: str = client.api.exec_create(
                self._container.id,
                ['sh', '-c', self._command],
                **kwargs,
            )['Id']
            sock = client.api.exec_start(exec_id, socket=True)
            # docker-py returns a SocketIO wrapper; get the raw socket
            raw = getattr(sock, '_sock', sock)
            return exec_id, raw

        self._exec_id, self._socket = await anyio.to_thread.run_sync(_do_start)

    async def __aenter__(self) -> Self:
        if self._exec_id is None:  # pragma: no branch
            await self._start()
        return self

    async def send(self, data: bytes) -> None:
        await anyio.to_thread.run_sync(self._socket.sendall, data)

    async def recv(self, timeout: float | None = None) -> bytes:
        if self._stdout_buffer:
            return self._stdout_buffer.pop(0)
        if timeout is not None:
            with anyio.fail_after(timeout):
                return await self._recv_stream(self._STDOUT)
        return await self._recv_stream(self._STDOUT)

    async def recv_stderr(self, timeout: float | None = None) -> bytes:
        if self._stderr_buffer:
            return self._stderr_buffer.pop(0)
        if timeout is not None:
            with anyio.fail_after(timeout):
                return await self._recv_stream(self._STDERR)
        return await self._recv_stream(self._STDERR)

    async def _recv_stream(self, wanted: int) -> bytes:
        """Read frames until one for the wanted stream type arrives."""
        while True:
            stream_type, data = await anyio.to_thread.run_sync(self._read_frame)
            if not data and self._eof:
                return b''
            if stream_type == wanted:
                return data
            # Buffer the frame for the other stream
            if stream_type == self._STDOUT:
                self._stdout_buffer.append(data)
            else:
                self._stderr_buffer.append(data)

    def _read_frame(self) -> tuple[int, bytes]:
        """Read one frame from the Docker multiplexed stream.

        Docker exec socket uses a multiplexed protocol:
        - 8 byte header: [stream_type(1), 0, 0, 0, size(4)]
        - followed by `size` bytes of data

        Returns:
            A `(stream_type, data)` tuple.  `stream_type` is 1 for stdout
            and 2 for stderr.  Returns `(0, b'')` on EOF.
        """
        if self._eof:
            return 0, b''

        header = b''
        while len(header) < 8:
            chunk = self._socket.recv(8 - len(header))
            if not chunk:
                self._eof = True
                return 0, b''
            header += chunk

        stream_type = header[0]
        size = struct.unpack('>I', header[4:8])[0]
        if size == 0:
            return stream_type, b''

        data = b''
        while len(data) < size:
            chunk = self._socket.recv(size - len(data))
            if not chunk:
                self._eof = True
                break
            data += chunk
        return stream_type, data

    @property
    def returncode(self) -> int | None:
        if self._returncode is not None:
            return self._returncode
        if self._exec_id is None:
            return None
        try:
            client: Any = self._container.client
            info = client.api.exec_inspect(self._exec_id)
            rc = info.get('ExitCode')
            if not info.get('Running', False) and rc is not None:
                self._returncode = rc
                return rc
        except (DockerException, OSError):
            # Docker API may raise various errors (connection, not found, etc.)
            # when inspecting exec state — treat as "still running"
            pass
        return None

    async def wait(self, timeout: float | None = None) -> int:
        async def _poll() -> int:
            while True:
                rc = self.returncode
                if rc is not None:
                    return rc
                await anyio.sleep(0.1)

        if timeout is not None:
            with anyio.fail_after(timeout):
                return await _poll()
        return await _poll()

    async def kill(self) -> None:
        # Docker exec doesn't provide a direct kill; close the socket
        try:
            self._socket.close()
        except OSError:
            pass


class DockerEnvironment(ExecutionEnvironment):
    """Docker container-based environment for isolated code execution.

    Provides isolated code execution with configurable resource limits,
    network access, and persistent or ephemeral workspaces.

    Usage:
        ```python {test="skip" lint="skip"}
        async with DockerEnvironment(image='python:3.12-slim') as env:
            result = await env.shell('python -c "print(42)"')
            print(result.output)
        ```
    """

    def __init__(
        self,
        *,
        image: str = 'python:3.12-slim',
        env_vars: dict[str, str] | None = None,
        work_dir: str = '/workspace',
        volumes: dict[str, dict[str, str]] | None = None,
        memory_limit: str | None = None,
        cpu_limit: float | None = None,
        pids_limit: int | None = None,
        network_disabled: bool = False,
        read_only: bool = False,
        cap_drop: list[str] | None = None,
        security_opt: list[str] | None = None,
        user: str | None = None,
        tmpfs: dict[str, str] | None = None,
        init: bool = False,
    ) -> None:
        """Create a Docker environment.

        Args:
            image: Docker image to use. Pre-build custom images with any
                required packages before passing them here.
            env_vars: Baseline environment variables to set in the container.
            work_dir: Working directory inside the container.
            volumes: Volume mounts (Docker format).
            memory_limit: Memory limit (e.g. '512m', '1g').
            cpu_limit: CPU limit (e.g. 1.0 for one CPU).
            pids_limit: Maximum number of PIDs in the container (e.g. 256).
                Prevents fork bombs.
            network_disabled: Whether to disable network access.
            read_only: Whether to mount the root filesystem as read-only.
                Use with `tmpfs` to provide writable scratch space.
            cap_drop: Linux capabilities to drop (e.g. `['ALL']`).
            security_opt: Security options (e.g. `['no-new-privileges']`).
            user: User to run as inside the container (e.g. `'nobody'`).
            tmpfs: tmpfs mounts as `{path: options}`
                (e.g. `{'/tmp': 'noexec,nosuid,size=64m'}`).
            init: Whether to use `--init` to run an init process as PID 1.
                Ensures proper signal handling and zombie reaping.
        """
        self._image = image
        self._env_vars = env_vars or {}
        self._work_dir = work_dir
        self._volumes = volumes
        self._memory_limit = memory_limit
        self._cpu_limit = cpu_limit
        self._pids_limit = pids_limit
        self._network_disabled = network_disabled
        self._read_only = read_only
        self._cap_drop = cap_drop
        self._security_opt = security_opt
        self._user = user
        self._tmpfs = tmpfs
        self._init = init

        self._client: docker.DockerClient | None = None
        self._container: Container | None = None

    @classmethod
    def hardened(
        cls,
        *,
        image: str = 'python:3.12-slim',
        env_vars: dict[str, str] | None = None,
        work_dir: str = '/workspace',
        memory_limit: str = '512m',
        cpu_limit: float = 1.0,
        pids_limit: int = 256,
    ) -> DockerEnvironment:
        """Create a hardened Docker environment with security best practices.

        This is a convenience constructor that sets sensible security defaults:
        network disabled, read-only root filesystem, all capabilities dropped,
        no privilege escalation, runs as `nobody`, and uses an init process.

        The root filesystem is read-only; writable tmpfs mounts are provided at
        `/tmp` and the working directory.

        Args:
            image: Docker image to use.
            env_vars: Baseline environment variables to set in the container.
            work_dir: Working directory inside the container.
            memory_limit: Memory limit (e.g. '512m', '1g').
            cpu_limit: CPU limit (e.g. 1.0 for one CPU).
            pids_limit: Maximum number of PIDs in the container.
        """
        return cls(
            image=image,
            env_vars=env_vars,
            work_dir=work_dir,
            network_disabled=True,
            read_only=True,
            cap_drop=['ALL'],
            security_opt=['no-new-privileges'],
            user='nobody',
            pids_limit=pids_limit,
            tmpfs={'/tmp': 'noexec,nosuid,size=64m', work_dir: 'size=128m'},
            init=True,
            memory_limit=memory_limit,
            cpu_limit=cpu_limit,
        )

    @property
    def capabilities(self) -> frozenset[EnvCapability]:  # pragma: lax no cover
        return frozenset(
            {
                'ls',
                'shell',
                'read_file',
                'write_file',
                'edit_file:replace_str',
                'glob',
                'grep',
            }
        )

    def instructions(self, capability: EnvCapability) -> str | None:
        if capability == 'grep':  # pragma: lax no cover
            return 'Uses POSIX basic regex, not Python `re` syntax.'
        elif capability == 'glob':  # pragma: lax no cover
            return 'Uses `find` for pattern matching; `**` is not supported.'
        elif capability == 'shell':  # pragma: lax no cover
            return 'Runs inside a Docker container.'
        return None  # pragma: lax no cover

    async def __aenter__(self) -> Self:
        await anyio.to_thread.run_sync(self._setup)
        return self

    def _setup(self) -> None:
        """Start container (sync, runs in executor)."""
        if self._container is not None:
            return
        self._client = docker.from_env()

        # Create and start container
        kwargs: dict[str, Any] = {
            'image': self._image,
            'command': 'sleep infinity',
            'detach': True,
            'working_dir': self._work_dir,
            'environment': self._env_vars,
            'auto_remove': False,
        }
        if self._volumes:
            kwargs['volumes'] = self._volumes
        if self._memory_limit:
            kwargs['mem_limit'] = self._memory_limit
        if self._cpu_limit:
            kwargs['nano_cpus'] = int(self._cpu_limit * 1e9)
        if self._pids_limit is not None:
            kwargs['pids_limit'] = self._pids_limit
        if self._network_disabled:
            kwargs['network_disabled'] = True
        if self._read_only:
            kwargs['read_only'] = True
        if self._cap_drop:
            kwargs['cap_drop'] = self._cap_drop
        if self._security_opt:
            kwargs['security_opt'] = self._security_opt
        if self._user:
            kwargs['user'] = self._user
        if self._tmpfs:
            kwargs['tmpfs'] = self._tmpfs
        if self._init:
            kwargs['init'] = True

        self._container = cast(Container, self._client.containers.run(**kwargs))

        # Ensure work_dir exists
        self._container.exec_run(['mkdir', '-p', self._work_dir])

    async def __aexit__(self, *_args: Any) -> None:
        if self._container is not None:  # pragma: no branch
            await anyio.to_thread.run_sync(self._teardown)

    def _teardown(self) -> None:
        """Stop and remove container (sync, runs in executor)."""
        if self._container is not None:  # pragma: no branch
            try:
                self._container.stop(timeout=5)
            except Exception:
                # Best-effort cleanup: container may already be stopped or removed
                pass
            try:
                self._container.remove(force=True)
            except Exception:
                # Best-effort cleanup: container may already be removed
                pass
            self._container = None

    @property
    def container(self) -> Container:
        if self._container is None:
            raise RuntimeError('DockerEnvironment not started. Use `async with DockerEnvironment(...) as env:`')
        return self._container

    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to work_dir for Docker API calls.

        Docker API methods like `put_archive` and `get_archive` resolve
        paths against the container root `/`, not the working directory.
        This helper ensures relative paths are resolved against `work_dir`.
        """
        if not path.startswith('/'):
            return f'{self._work_dir}/{path}'
        return path

    async def create_process(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
    ) -> ExecutionProcess:
        return DockerEnvironmentProcess(self.container, command, self._work_dir, env=env)

    async def shell(
        self,
        command: str,
        *,
        timeout: float | None = 120,
        env: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a command in the container."""

        def _exec() -> tuple[int, bytes]:
            if timeout is not None:
                wrapped = f'timeout {math.ceil(timeout)} sh -c {shell_escape(command)}'
            else:
                wrapped = command
            exec_kwargs: dict[str, Any] = {'workdir': self._work_dir}
            if env:
                exec_kwargs['environment'] = env
            exit_code, output = self.container.exec_run(
                ['sh', '-c', wrapped],
                **exec_kwargs,
            )
            return exit_code, output

        exit_code, output_bytes = await anyio.to_thread.run_sync(_exec)
        output = output_bytes.decode('utf-8', errors='replace')
        truncated = len(output) > MAX_OUTPUT_CHARS
        if truncated:
            output = output[:MAX_OUTPUT_CHARS]
        # timeout command returns 124 on timeout
        if exit_code == 124 and timeout is not None:
            output += '\n[Command timed out]'
        return ExecutionResult(output=output, exit_code=exit_code, truncated=truncated)

    async def read_file(self, path: str, *, offset: int = 0, limit: int = 2000) -> str | bytes:
        ext = posixpath.splitext(path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return await anyio.to_thread.run_sync(self._read_file_bytes_sync, path)

        def _read() -> str | bytes:
            cmd = build_read_file_cmd(path, offset=offset, limit=limit)
            exit_code, output = self.container.exec_run(['sh', '-c', cmd], workdir=self._work_dir)
            if exit_code != 0:
                raise FileNotFoundError(f'File not found or not readable: {path}')
            try:
                return output.decode('utf-8')
            except UnicodeDecodeError:
                return self._read_file_bytes_sync(path)

        return await anyio.to_thread.run_sync(_read)

    def _read_file_bytes_sync(self, path: str) -> bytes:
        """Read raw file bytes using Docker's get_archive API."""
        bits, _ = self.container.get_archive(self._resolve_path(path))
        # get_archive returns a tar stream
        tar_bytes = b''.join(bits)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
            members = tar.getmembers()
            if not members:
                raise FileNotFoundError(f'File not found: {path}')
            extracted = tar.extractfile(members[0])
            if extracted is None:
                raise FileNotFoundError(f'Cannot read file: {path}')
            return extracted.read()

    async def write_file(self, path: str, content: str | bytes) -> None:
        def _write() -> None:
            full_path = self._resolve_path(path)
            # Ensure parent directory exists
            parent = str(PurePosixPath(full_path).parent)
            self.container.exec_run(['mkdir', '-p', parent])

            data = content.encode('utf-8') if isinstance(content, str) else content
            _put_file(self.container, full_path, data)

        await anyio.to_thread.run_sync(_write)

    async def replace_str(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> int:
        def _edit() -> int:
            raw = self._read_file_bytes_sync(path)
            text = raw.decode('utf-8')
            new_text, count = apply_edit(text, old, new, path, replace_all=replace_all)
            _put_file(self.container, self._resolve_path(path), new_text.encode('utf-8'))
            return count

        return await anyio.to_thread.run_sync(_edit)

    async def ls(self, path: str = '.') -> list[FileInfo]:
        def _ls() -> list[FileInfo]:
            cmd = f'ls -la {shell_escape(path)}'
            exit_code, output = self.container.exec_run(['sh', '-c', cmd], workdir=self._work_dir)
            if exit_code != 0:
                raise NotADirectoryError(f'Not a directory or not found: {path}')

            entries: list[FileInfo] = []
            for line in output.decode('utf-8', errors='replace').splitlines():
                # Skip total line and empty lines
                if not line or line.startswith('total'):
                    continue
                parts = line.split(None, 8)
                if len(parts) < 9:
                    continue
                perms, _, _, _, size_str, _, _, _, name = parts
                is_dir = perms.startswith('d')
                try:
                    size = int(size_str) if not is_dir else None
                except ValueError:
                    size = None
                entry_path = f'{path}/{name}' if path != '.' else name
                entries.append(FileInfo(name=name, path=entry_path, is_dir=is_dir, size=size))
            return entries

        return await anyio.to_thread.run_sync(_ls)

    async def glob(self, pattern: str, *, path: str = '.') -> list[str]:
        def _glob() -> list[str]:
            cmd = build_glob_cmd(pattern, path=path)
            _, output = self.container.exec_run(['sh', '-c', cmd], workdir=self._work_dir)
            return parse_glob_output(output.decode('utf-8', errors='replace'))

        return await anyio.to_thread.run_sync(_glob)

    async def grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        glob_pattern: str | None = None,
        output_mode: Literal['content', 'files_with_matches', 'count'] = 'content',
    ) -> str:
        def _grep() -> str:
            cmd = build_grep_cmd(pattern, path=path, glob_pattern=glob_pattern, output_mode=output_mode)
            _, output = self.container.exec_run(['sh', '-c', cmd], workdir=self._work_dir)
            text = output.decode('utf-8', errors='replace').strip()
            if output_mode == 'count':
                text = filter_grep_count_output(text)
            return text

        return await anyio.to_thread.run_sync(_grep)

    async def is_alive(self) -> bool:
        """Check if the container is running.

        Returns:
            True if the container is running, False otherwise.
        """
        if self._container is None:
            return False

        def _check() -> bool:
            assert self._container is not None
            try:
                self._container.reload()
                return self._container.status == 'running'
            except Exception:
                return False

        return await anyio.to_thread.run_sync(_check)
