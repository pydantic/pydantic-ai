"""An Amazon Bedrock AgentCore Code Interpreter session behind Pydantic AI's `WorkspaceBackend` protocol.

Service behavior verified 2026-11-02 against a live `aws.codeinterpreter.v1` session in us-east-1
with boto3 1.42.63:

- `startCommandExecution` runs a command on a pseudo-terminal: stdin is a TTY, and stdout and
  stderr arrive merged with CRLF line endings. So each command runs as a background task with
  stdin at EOF and both streams redirected to files, which are read once the task ends. `stopTask`
  kills the task's process.
- `executeCode` runs Python 3.12 in a persistent Jupyter kernel, as the same user and on the same
  filesystem as the commands, with no pseudo-terminal. The native file tools (`readFiles`,
  `writeFiles`, ...) refuse absolute paths and read a symlink as empty, so every file operation is
  a short Python program run by `executeCode` instead. An exception it raises comes back as an
  `isError` result with a traceback, so the program catches `OSError` itself and reports the
  exception's class name, which is mapped back to the builtin error here: errno numbers differ
  between Linux and macOS.
- `sessionTimeoutSeconds` is an absolute lifetime, not the idle timeout the botocore model
  describes: a 60-second session called every 15 seconds was refused as not active at 61 seconds.
  The service also accepted 30 seconds, below the model's stated minimum of 60.
- A request payload over 24 MiB is refused, and `executeCode` returns large output slowly (4 MiB of
  stdout took 57 seconds, close to botocore's 60-second read timeout), so file contents travel in
  chunks.
- botocore retries a request that timed out or lost its connection, and its `max_attempts` counts
  retries after the first request. Sending `startCommandExecution`, a chunk `write`, `remove` or
  `finished` (which deletes the command's output) twice changes the outcome, so those go through a
  client limited to one attempt (`total_max_attempts=1`) and surface throttling instead of
  retrying it. Every other call retries with botocore's standard mode. `StartCodeInterpreterSession`
  carries a `clientToken` botocore fills in, so retrying it cannot start a second session.
  `stopTask` on a task that has already ended, or was already stopped, answers `isError` with
  `Task cannot be canceled (current status: ...)` and changes nothing, so it retries too.
"""

from __future__ import annotations

import base64
import json
import logging
import shlex
import uuid
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, NotRequired, TypeVar

import anyio
import anyio.to_thread
from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    SupportsCommands,
    SupportsFilesystem,
    SupportsRealpath,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_env,
    check_provider,
    check_timeout,
    command_argv,
    command_deadline,
    run_to_completion,
    running_on_asyncio,
    stop_shielded,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_agentcore import BedrockAgentCoreClient
    from mypy_boto3_bedrock_agentcore.literals import ToolNameType
    from mypy_boto3_bedrock_agentcore.type_defs import CodeInterpreterResultTypeDef, ToolArgumentsTypeDef

    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('AgentCoreSandboxBackend',)

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
except ImportError as _import_error:
    raise ImportError('Install `pydantic-ai-harness[agentcore]` to use AgentCoreSandbox.') from _import_error

logger = logging.getLogger(__name__)

_T = TypeVar('_T')

PROVIDER = 'agentcore'
DEFAULT_CODE_INTERPRETER = 'aws.codeinterpreter.v1'
# AgentCore's own default, and the most it allows (8 hours).
DEFAULT_SESSION_TIMEOUT = 900
MAX_SESSION_TIMEOUT = 28_800

_POLL_INTERVAL = 0.25
_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_OUTPUT_PREVIEW_BYTES = 64 * 1024
_PARTIAL_OUTPUT_BYTES = 4096
# Per `executeCode` call; see the module docstring.
_READ_CHUNK_BYTES = 1024 * 1024
_WRITE_CHUNK_BYTES = 4 * 1024 * 1024
_CLEAN_UP_GRACE = 10
_RETRYING_CONFIG = Config(retries={'mode': 'standard', 'total_max_attempts': 3}, read_timeout=60, connect_timeout=10)
_SINGLE_ATTEMPT_CONFIG = Config(retries={'total_max_attempts': 1}, read_timeout=60, connect_timeout=10)
# `_PROGRAM` operations that change state in a way a second attempt would see; see the module docstring.
_SINGLE_ATTEMPT_OPERATIONS = frozenset({'write', 'remove', 'finished'})
_FINISHED = frozenset({'completed', 'failed', 'canceled'})
_CREDENTIAL_ERRORS = frozenset(
    {'AccessDeniedException', 'UnrecognizedClientException', 'ExpiredTokenException', 'InvalidSignatureException'}
)
_PATH_ERRORS: dict[str, type[OSError]] = {
    error.__name__: error
    for error in (FileNotFoundError, FileExistsError, IsADirectoryError, NotADirectoryError, PermissionError)
}
# Written to a command's exit code file in place of a code when it cannot enter the working directory.
_NO_WORKING_DIR = 'no-working-dir'

# Runs in the session's Python kernel, which the agent's own `executeCode` calls could share, so it
# defines one prefixed name. Each call sends it whole, since the kernel may have restarted since the
# last one. It prints one JSON object, `{"ok": result}` or `{"error": [class name, message]}`.
_PROGRAM = """
def _pydantic_ai_workspace(encoded):
    import base64, json, os, shutil, stat

    def entry(path, strict):
        name = os.path.basename(path.rstrip('/')) or path
        try:
            info = os.stat(path)
        except OSError:
            if strict:
                raise
            return {'name': name, 'path': path, 'is_dir': False, 'size': None}
        size = info.st_size if stat.S_ISREG(info.st_mode) else None
        return {'name': name, 'path': path, 'is_dir': stat.S_ISDIR(info.st_mode), 'size': size}

    def check_regular(mode, path):
        if stat.S_ISDIR(mode):
            raise IsADirectoryError(21, 'Is a directory', path)
        if not stat.S_ISREG(mode):
            raise OSError(f'{path!r} is not a regular file')

    def open_regular(path, flags=0):
        # Checked before opening, so a FIFO's writer or a device is never touched, and again on the
        # descriptor, which no swap in between can fool. Opened without waiting for a writer: a FIFO
        # left where a file is expected would otherwise block this kernel for the rest of the session.
        check_regular((os.lstat if flags & os.O_NOFOLLOW else os.stat)(path).st_mode, path)
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | flags)
        try:
            check_regular(os.fstat(fd).st_mode, path)
        except OSError:
            os.close(fd)
            raise
        return os.fdopen(fd, 'rb')

    def captured(directory, name):
        # A command's output artifact, or None before the command wrote it. The command could have
        # replaced it with a link, so links are refused too.
        try:
            return open_regular(os.path.join(directory, name), os.O_NOFOLLOW)
        except FileNotFoundError:
            return None

    def read(path, offset, size):
        with open_regular(path) as file:
            file.seek(offset)
            data = file.read(size)
        return {'data': base64.b64encode(data).decode(), 'eof': len(data) < size}

    def write(path, data, append):
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            try:
                os.makedirs(parent, exist_ok=True)
            except FileExistsError:
                raise NotADirectoryError(20, 'Not a directory', parent) from None
        if os.path.exists(path):
            check_regular(os.stat(path).st_mode, path)
        with open(path, 'ab' if append else 'wb') as file:
            file.write(base64.b64decode(data))

    def remove(path, root):
        if not os.path.lexists(path):
            raise FileNotFoundError(2, 'No such file or directory', path)
        parent = os.path.realpath(os.path.dirname(path))
        target = os.path.normpath(os.path.join(parent, os.path.basename(path)))
        if target == root or root.startswith(target.rstrip('/') + '/'):
            return False
        if os.path.islink(path) or not os.path.isdir(path):
            os.unlink(path)
        else:
            shutil.rmtree(path)
        return True

    def working_dir(path, create):
        path = path or os.getcwd()
        if create:
            os.makedirs(path, exist_ok=True)
        return os.path.realpath(path) if os.path.isdir(path) else None

    def finished(directory, inline):
        def head(name):
            file = captured(directory, name)
            if file is None:
                return b'', 0
            with file:
                return file.read(inline + 1), os.fstat(file.fileno()).st_size

        reply = {'exit_code': None, 'size': 0, 'out': None, 'err': None, 'replaced': False}
        try:
            rc = captured(directory, 'rc')
            if rc is None:
                return reply
            with rc:
                reply['exit_code'] = rc.read().decode().strip()
            (out, out_size), (err, err_size) = head('out'), head('err')
        except OSError:
            # The command replaced an output file with something that is not a regular file, or made it unreadable.
            reply['replaced'] = True
            return reply
        reply['size'] = out_size + err_size
        # Most output is small: return it now and clean up, saving a round trip per stream. Decided on
        # what was read, as a process the command left in the background can still be writing.
        if len(out) + len(err) <= inline:
            reply.update(out=base64.b64encode(out).decode(), err=base64.b64encode(err).decode())
            shutil.rmtree(directory, ignore_errors=True)
        return reply

    def tail(directory, name, limit):
        try:
            file = captured(directory, name)
        except OSError:
            return ''
        if file is None:
            return ''
        with file:
            file.seek(max(os.fstat(file.fileno()).st_size - limit, 0))
            return base64.b64encode(file.read(limit)).decode()

    operations = {
        'read': read,
        'write': write,
        'stat': lambda path: entry(path, True),
        'list': lambda path: [entry(os.path.join(path, name), False) for name in sorted(os.listdir(path))],
        'make_dir': lambda path: os.makedirs(path, exist_ok=True),
        'remove': remove,
        'exists': lambda path: os.path.exists(path),
        'realpath': lambda path: os.path.realpath(path),
        'working_dir': working_dir,
        'finished': finished,
        'tail': lambda directory, limit: [tail(directory, name, limit) for name in ('out', 'err')],
        'clean_up': lambda directory: shutil.rmtree(directory, ignore_errors=True),
    }
    request = json.loads(encoded)
    operation = operations[request.pop('operation')]
    try:
        reply = {'ok': operation(**request)}
    except OSError as error:
        reply = {'error': [type(error).__name__, str(error)]}
    print(json.dumps(reply))
"""


class _Reply(TypedDict):
    ok: NotRequired[JsonValue]
    error: NotRequired[tuple[str, str]]


class _Chunk(TypedDict):
    data: str
    eof: bool


class _Finished(TypedDict):
    exit_code: str | None
    size: int
    out: str | None
    err: str | None
    replaced: bool


_REPLY = TypeAdapter(_Reply)
_CHUNK = TypeAdapter(_Chunk)
_FINISHED_REPLY = TypeAdapter(_Finished)
_ENTRY = TypeAdapter(FileEntry)
_ENTRIES = TypeAdapter(list[FileEntry])
_BOOL = TypeAdapter(bool)
_STR = TypeAdapter(str)
_OPTIONAL_STR = TypeAdapter(str | None)
_NONE = TypeAdapter(None)
_TAIL = TypeAdapter(tuple[str, str])


def new_client(region_name: str | None, *, retry: bool) -> BedrockAgentCoreClient:
    """A `bedrock-agentcore` client, retrying idempotent calls or making one attempt only.

    Building one reads files: do it off the event loop.
    """
    config = _RETRYING_CONFIG if retry else _SINGLE_ATTEMPT_CONFIG
    # boto3-stubs types `client` as one overload per AWS service, and those whose stubs are not
    # installed return an unknown type, which leaves the overloaded member partly unknown.
    return boto3.client('bedrock-agentcore', region_name=region_name, config=config)  # pyright: ignore[reportUnknownMemberType]


async def _complete(call: Callable[[], Awaitable[None]]) -> None:
    """Await `call` to its end even if the caller is cancelled meanwhile, so what it records is not lost."""
    if running_on_asyncio():
        await run_to_completion(call)
    else:
        with anyio.CancelScope(shield=True):
            await call()


def split_ref(ref: WorkspaceRef) -> tuple[str, str]:
    """The code interpreter and session id an AgentCore `WorkspaceRef` names."""
    check_provider(ref, PROVIDER)
    code_interpreter, _, session_id = ref.id.rpartition('/')
    if not code_interpreter or not session_id:
        raise ValueError(f"AgentCore workspace ref id must be '<code interpreter>/<session id>', got {ref.id!r}")
    return code_interpreter, session_id


def is_inactive(error: ClientError) -> bool:
    """Whether AgentCore refused a call because the session has ended or never existed."""
    code = error.response.get('Error', {}).get('Code', '')
    message = str(error).lower()
    # Stopping a stopped session is a conflict: `Session already terminated: <id>`.
    return code == 'ResourceNotFoundException' or (
        code in ('ValidationException', 'ConflictException') and ('not active' in message or 'terminated' in message)
    )


def translate(error: Exception, session_id: str | None) -> Exception:
    """Map one boto3 failure onto the protocol's typed failures, or return it unchanged.

    An ended session, missing or refused credentials, and a request AgentCore validates and rejects
    (an unknown code interpreter, a session timeout out of range) end the run: retrying cannot
    succeed. Throttling, server errors and transport failures come back unchanged, for durable
    engines to retry.
    """
    if isinstance(error, NoCredentialsError):
        return WorkspaceUnavailableError('No AWS credentials found for AgentCore; configure them as boto3 expects.')
    if not isinstance(error, ClientError):
        return error
    if session_id is not None and is_inactive(error):
        return WorkspaceUnavailableError(
            f'The AgentCore Code Interpreter session {session_id!r} is not active in this region: it was stopped, '
            'reached its session timeout, or was started in another region. A session cannot be restarted; '
            "pass `workspace='new'` to start a fresh one."
        )
    code = error.response.get('Error', {}).get('Code', '')
    if code in _CREDENTIAL_ERRORS:
        return WorkspaceUnavailableError(f'AgentCore refused the AWS credentials ({code}).')
    if code in ('ValidationException', 'ResourceNotFoundException'):
        # Refusing to start or stop a session is final; refusing one call in a live session ends only that call.
        if session_id is None:
            return WorkspaceUnavailableError(f'AgentCore refused the request: {error}')
        return WorkspaceError(f'AgentCore refused the request: {error}')
    return error


@contextmanager
def translated(session_id: str | None) -> Generator[None]:
    """Raise boto3 failures inside the block as the protocol's typed failures; see `translate`."""
    try:
        yield
    except Exception as error:
        mapped = translate(error, session_id)
        if mapped is error:
            raise
        raise mapped from error


async def stop_session(ref: WorkspaceRef, *, region_name: str | None) -> None:
    """Stop the session `ref` names; one that has already ended returns quietly."""
    code_interpreter, session_id = split_ref(ref)
    # Stopping twice is harmless: the second attempt finds the session terminated.
    client = await anyio.to_thread.run_sync(lambda: new_client(region_name, retry=True))

    def stop() -> None:
        try:
            client.stop_code_interpreter_session(codeInterpreterIdentifier=code_interpreter, sessionId=session_id)
        except ClientError as error:
            if not is_inactive(error):
                raise

    with translated(None):
        await anyio.to_thread.run_sync(stop)


def _decode(encoded: str) -> str:
    return base64.b64decode(encoded).decode(errors='replace')


class AgentCoreSandboxBackend(WorkspaceBackend, SupportsCommands, SupportsFilesystem, SupportsRealpath):
    """An [Amazon Bedrock AgentCore Code Interpreter](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html) session as a Pydantic AI [`WorkspaceBackend`][pydantic_ai.workspaces.WorkspaceBackend].

    Commands and file operations run inside the session's microVM, not on the host.

    Building one does no I/O. The first operation starts a session, or attaches to the one `ref`
    names. The backend does not stop the session: `AgentCoreSandbox.destroy(ref)` does, or it ends
    on its own when `session_timeout` runs out. A session that has ended cannot be resumed.

    Each command runs as a background task under `sh -c`, with its output redirected to files that
    are read once it exits, because AgentCore runs commands on a pseudo-terminal that would merge
    stdout and stderr. File operations run as short Python programs in the session's code kernel.

    Args:
        ref: Identity of a running session to attach to.
        region_name: AWS region of the code interpreter; boto3's configured region when `None`. A ref
            does not record its region, so attach with the region the session was started in.
        code_interpreter: The AWS-managed `aws.codeinterpreter.v1`, or the id of a code interpreter
            you created, for example one with network access. An unknown one raises
            `WorkspaceUnavailableError` on first use.
        session_timeout: Total lifetime of a newly started session in seconds, from 1 to 28800. It is not
            an idle timeout: the session ends when it runs out, whatever it is doing.
        working_dir: Absolute directory commands start in and relative paths resolve against.
            Created in a session this backend starts; an attached session must already have it.
            `None` uses the session's own (`/opt/amazon/genesis1p-tools/var` on `aws.codeinterpreter.v1`).
        env: Environment variables every command gets, also on an attached session; a command's
            `env` is layered on top. Nothing is read from the host.
    """

    def __init__(
        self,
        *,
        ref: WorkspaceRef | None = None,
        region_name: str | None = None,
        code_interpreter: str = DEFAULT_CODE_INTERPRETER,
        session_timeout: int = DEFAULT_SESSION_TIMEOUT,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        check_env(env)
        if type(session_timeout) is not int or not 1 <= session_timeout <= MAX_SESSION_TIMEOUT:
            raise ValueError(
                f'session_timeout must be an integer from 1 to {MAX_SESSION_TIMEOUT}, got {session_timeout!r}.'
            )
        self._code_interpreter = code_interpreter
        self._session_id: str | None = None
        if ref is not None:
            self._code_interpreter, self._session_id = split_ref(ref)
        self._ref = ref
        self._region_name = region_name
        self._session_timeout = session_timeout
        self._working_dir = absolute_path('working_dir', working_dir)
        # Made only in a session this backend starts, as the other sandboxes do.
        self._create_working_dir = ref is None
        self._resolved_working_dir: str | None = None
        self._env = dict(env or {})
        # Built on first use: one that retries idempotent calls, and one that makes a single attempt.
        self._clients: tuple[BedrockAgentCoreClient, BedrockAgentCoreClient] | None = None
        self._lock = anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        """Identity of the session, or `None` before one has been started."""
        return self._ref

    async def _session(self) -> tuple[tuple[BedrockAgentCoreClient, BedrockAgentCoreClient], str]:
        async with self._lock:
            if self._clients is None:
                region_name = self._region_name
                self._clients = await anyio.to_thread.run_sync(
                    lambda: (new_client(region_name, retry=True), new_client(region_name, retry=False))
                )
            clients = self._clients
            client = clients[0]
            if self._session_id is None:

                async def start() -> None:
                    with translated(None):
                        response = await anyio.to_thread.run_sync(
                            lambda: client.start_code_interpreter_session(
                                codeInterpreterIdentifier=self._code_interpreter,
                                sessionTimeoutSeconds=self._session_timeout,
                            )
                        )
                    # Recorded inside the call that outlives a cancelled caller, so the billed
                    # session can always be found and stopped.
                    session_id = self._session_id = response['sessionId']
                    self._ref = WorkspaceRef(provider=PROVIDER, id=f'{self._code_interpreter}/{session_id}')
                    logger.info('Started AgentCore Code Interpreter session %s', session_id)

                await _complete(start)
            assert self._session_id is not None
            return clients, self._session_id

    async def _invoke(
        self, name: ToolNameType, arguments: ToolArgumentsTypeDef, *, retry: bool
    ) -> CodeInterpreterResultTypeDef:
        """Call one Code Interpreter tool and return its result; `retry` only when a second attempt is harmless."""
        (retrying, single_attempt), session_id = await self._session()
        client = retrying if retry else single_attempt

        def call() -> CodeInterpreterResultTypeDef | None:
            response = client.invoke_code_interpreter(
                codeInterpreterIdentifier=self._code_interpreter, sessionId=session_id, name=name, arguments=arguments
            )
            result: CodeInterpreterResultTypeDef | None = None
            # botocore raises the stream's exception events itself; drain it to release the connection.
            for event in response['stream']:
                if 'result' in event:
                    result = event['result']
            return result

        with translated(session_id):
            result = await anyio.to_thread.run_sync(call)
        if result is None:
            raise WorkspaceError(f'AgentCore {name} returned no result')
        if result.get('isError'):
            text = ''.join(block.get('text', '') for block in result['content'])
            raise WorkspaceError(f'AgentCore {name} failed: {text[-2000:]}')
        return result

    async def _code(self, adapter: TypeAdapter[_T], operation: str, **arguments: JsonValue) -> _T:
        """Run one operation of `_PROGRAM` in the session's code kernel and return its validated result."""
        request = json.dumps({'operation': operation, **arguments})
        code = f'{_PROGRAM}\n_pydantic_ai_workspace({request!r})\n'
        result = await self._invoke(
            'executeCode', {'code': code, 'language': 'python'}, retry=operation not in _SINGLE_ATTEMPT_OPERATIONS
        )
        stdout = result.get('structuredContent', {}).get('stdout', '')
        try:
            reply = _REPLY.validate_json(stdout)
            if 'error' in reply:
                name, message = reply['error']
                path = arguments.get('path')
                raise _PATH_ERRORS.get(name, OSError)(f'{message} (in the AgentCore session, for {path!r})')
            return adapter.validate_python(reply.get('ok'))
        except ValidationError as error:
            raise WorkspaceError(
                f'AgentCore returned an unexpected reply to {operation!r}: {stdout[:200]!r}'
            ) from error

    async def _read(self, path: str, limit: int | None = None) -> bytes:
        chunks: list[bytes] = []
        offset = 0
        while True:
            size = _READ_CHUNK_BYTES if limit is None else min(_READ_CHUNK_BYTES, limit - offset)
            chunk = await self._code(_CHUNK, 'read', path=path, offset=offset, size=size)
            data = base64.b64decode(chunk['data'])
            chunks.append(data)
            offset += len(data)
            if chunk['eof'] or offset == limit:
                return b''.join(chunks)

    async def working_dir(self) -> str:
        """The canonical absolute directory commands start in."""
        if self._resolved_working_dir is None:
            resolved = await self._code(
                _OPTIONAL_STR, 'working_dir', path=self._working_dir, create=self._create_working_dir
            )
            if resolved is None:
                raise WorkspaceError(
                    f'working_dir {self._working_dir!r} does not exist in AgentCore session {self._session_id!r}. '
                    'Create it there, or pass a working_dir that exists.'
                )
            self._resolved_working_dir = resolved
        return self._resolved_working_dir

    async def read_bytes(self, path: str) -> bytes:
        return await self._read(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        # An empty file is still one write, which creates it.
        for start in range(0, max(len(data), 1), _WRITE_CHUNK_BYTES):
            chunk = base64.b64encode(data[start : start + _WRITE_CHUNK_BYTES]).decode()
            await self._code(_NONE, 'write', path=path, data=chunk, append=start > 0)

    async def stat(self, path: str) -> FileEntry:
        return await self._code(_ENTRY, 'stat', path=path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        return await self._code(_ENTRIES, 'list', path=path)

    async def make_dir(self, path: str) -> None:
        await self._code(_NONE, 'make_dir', path=path)

    async def remove(self, path: str) -> None:
        # The removal is recursive, so refuse the working directory and its ancestors, like the
        # built-in backends. The leaf stays unresolved: removing a link removes only the link.
        if not await self._code(_BOOL, 'remove', path=path, root=await self.working_dir()):
            raise ValueError('cannot remove the workspace root or its ancestor')

    async def exists(self, path: str) -> bool:
        return await self._code(_BOOL, 'exists', path=path)

    async def realpath(self, path: str) -> str:
        return await self._code(_STR, 'realpath', path=path)

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run a command as a background task and return its output once it exits, stopping it on timeout or cancellation."""
        check_timeout(timeout)
        argv = command_argv(command, shell)
        check_env(env)
        cwd = await self.working_dir()
        merged_env = {**self._env, **(env or {})}
        if merged_env:
            argv = ['env', '--', *(f'{name}={value}' for name, value in merged_env.items()), *argv]
        directory = f'/tmp/pydantic-ai-{uuid.uuid4().hex}'
        # The exit code file is written last, under a temporary name, so it is complete once it exists.
        script = (
            f'mkdir -p {directory} && cd -- {shlex.quote(cwd)} || {{ echo {_NO_WORKING_DIR} >{directory}/rc; exit; }}; '
            f'"$@" </dev/null >{directory}/out 2>{directory}/err; echo $? >{directory}/rc.tmp; '
            f'mv {directory}/rc.tmp {directory}/rc'
        )
        task_id: str | None = None
        partial = ('', '')

        async def stop() -> None:
            nonlocal partial
            if task_id is not None:
                try:
                    await self._invoke('stopTask', {'taskId': task_id}, retry=True)
                except (WorkspaceError, ClientError, BotoCoreError):
                    # The task may have ended on its own meanwhile, which AgentCore reports as an
                    # error; read what it wrote and clean up whatever the outcome.
                    logger.debug('Could not stop AgentCore task %s', task_id, exc_info=True)
            out, err = await self._code(_TAIL, 'tail', directory=directory, limit=_PARTIAL_OUTPUT_BYTES)
            partial = (_decode(out), _decode(err))
            await self._code(_NONE, 'clean_up', directory=directory)

        # The deadline covers the command only; `stop` makes three calls, so it gets more than the default grace.
        async with command_deadline(timeout, stop=stop, output=lambda: partial, grace=_CLEAN_UP_GRACE):

            async def start() -> None:
                nonlocal task_id
                started = await self._invoke(
                    'startCommandExecution', {'command': shlex.join(['sh', '-c', script, 'sh', *argv])}, retry=False
                )
                task_id = started.get('structuredContent', {}).get('taskId')
                if task_id is None:
                    raise WorkspaceError('AgentCore startCommandExecution returned no task id')

            # Run to its end even if the caller is cancelled meanwhile: a task started remotely
            # whose id is lost could not be stopped.
            await _complete(start)
            assert task_id is not None
            started_id = task_id
            while True:
                task = await self._invoke('getTask', {'taskId': started_id}, retry=True)
                if task.get('structuredContent', {}).get('taskStatus') in _FINISHED:
                    break
                await anyio.sleep(_POLL_INTERVAL)
        cleaned_up = False
        try:
            finished = await self._code(_FINISHED_REPLY, 'finished', directory=directory, inline=_READ_CHUNK_BYTES)
            cleaned_up = finished['out'] is not None
            return await self._result(directory, finished)
        finally:
            if not cleaned_up:
                await stop_shielded(lambda: self._code(_NONE, 'clean_up', directory=directory), grace=_CLEAN_UP_GRACE)

    async def _result(self, directory: str, finished: _Finished) -> CommandResult:
        if finished['replaced']:
            raise WorkspaceError(
                'The AgentCore command output could not be read: the command replaced an output file with '
                'something other than a regular file, or made it unreadable'
            )
        exit_code = finished['exit_code']
        if exit_code is None:
            raise WorkspaceError('The AgentCore command ended without recording an exit code')
        if exit_code == _NO_WORKING_DIR:
            raise WorkspaceError(
                f'working_dir {self._resolved_working_dir!r} no longer exists in AgentCore session {self._session_id!r}'
            )
        if finished['out'] is not None and finished['err'] is not None:
            return CommandResult(
                exit_code=int(exit_code), stdout=_decode(finished['out']), stderr=_decode(finished['err'])
            )
        out, err = f'{directory}/out', f'{directory}/err'
        if finished['size'] <= _MAX_OUTPUT_BYTES:
            # A process the command left in the background can still be writing, so the reads stop at the limit too.
            stdout = await self._read(out, _MAX_OUTPUT_BYTES + 1)
            stderr = await self._read(err, _MAX_OUTPUT_BYTES + 1 - len(stdout))
            if len(stdout) + len(stderr) <= _MAX_OUTPUT_BYTES:
                return CommandResult(
                    exit_code=int(exit_code),
                    stdout=stdout.decode(errors='replace'),
                    stderr=stderr.decode(errors='replace'),
                )
        raise WorkspaceOutputLimitError(
            'AgentCore command output exceeded the 4 MiB limit; redirect it to a file and read part of it instead',
            limit=_MAX_OUTPUT_BYTES,
            stdout=(await self._read(out, _OUTPUT_PREVIEW_BYTES)).decode(errors='replace'),
            stderr=(await self._read(err, _OUTPUT_PREVIEW_BYTES)).decode(errors='replace'),
        )
