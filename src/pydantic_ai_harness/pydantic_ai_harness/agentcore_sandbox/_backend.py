"""An Amazon Bedrock AgentCore Code Interpreter session behind Pydantic AI's `WorkspaceBackend` protocol.

Code Interpreter runs commands on a pseudo-terminal: stdin is a TTY, stdout and stderr arrive merged
with CRLF line endings, and undecodable bytes are not replaced. So each command runs as a background
task with stdin at EOF and both streams redirected to files, and the files are fetched base64-encoded
once the task ends. The native file tools only accept paths relative to the session's home directory,
so file operations go through `Workspace`'s shell fallback.
"""

from __future__ import annotations

import base64
import logging
import posixpath
import shlex
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import anyio
import anyio.to_thread

from pydantic_ai.workspaces import (
    CommandResult,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_env_types,
    check_timeout,
    command_argv,
    command_deadline,
    run_to_completion,
    stop_shielded,
)

if TYPE_CHECKING:
    from pydantic_ai.workspaces import WorkspaceCommand

__all__ = ('AgentCoreSandboxBackend', 'destroy_session')

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError as error:
    raise ImportError('Install `boto3` to use AgentCoreSandboxBackend.') from error

PROVIDER = 'agentcore'
DEFAULT_IDENTIFIER = 'aws.codeinterpreter.v1'
_POLL_INTERVAL = 0.25
_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
_PARTIAL_OUTPUT_LIMIT = 4096
# No retries: a retried `startCommandExecution` would run the command twice.
_CLIENT_CONFIG = Config(retries={'max_attempts': 1}, read_timeout=60, connect_timeout=10)
_FINISHED = frozenset({'completed', 'failed', 'canceled'})

logger = logging.getLogger(__name__)


def _new_client(region_name: str | None) -> Any:
    return boto3.client('bedrock-agentcore', region_name=region_name, config=_CLIENT_CONFIG)


def _split_ref(ref: WorkspaceRef) -> tuple[str, str]:
    if ref.provider != PROVIDER:
        raise ValueError(f'unsupported workspace provider {ref.provider!r}; expected {PROVIDER!r}')
    identifier, _, session_id = ref.id.rpartition('/')
    if not identifier or not session_id:
        raise ValueError(f"AgentCore workspace ref id must be '<code interpreter id>/<session id>', got {ref.id!r}")
    return identifier, session_id


def _unavailable(session_id: str, error: Exception) -> WorkspaceUnavailableError:
    return WorkspaceUnavailableError(
        f'The AgentCore Code Interpreter session {session_id!r} is no longer active ({error}). '
        "Sessions end at their timeout and cannot be restarted; pass `workspace='new'` to start a fresh one."
    )


def _is_inactive(error: ClientError) -> bool:
    code = error.response.get('Error', {}).get('Code', '')
    message = str(error).lower()
    return code == 'ResourceNotFoundException' or (code == 'ValidationException' and 'not active' in message)


async def destroy_session(ref: WorkspaceRef, *, region_name: str | None = None) -> None:
    """Stop the Code Interpreter session `ref` names; a session that already ended returns quietly."""
    identifier, session_id = _split_ref(ref)
    client = _new_client(region_name)

    def stop() -> None:
        try:
            client.stop_code_interpreter_session(codeInterpreterIdentifier=identifier, sessionId=session_id)
        except ClientError as error:
            if not _is_inactive(error):
                raise

    await anyio.to_thread.run_sync(stop)


class AgentCoreSandboxBackend(WorkspaceBackend, SupportsCommands):
    """An Amazon Bedrock AgentCore Code Interpreter session as a Pydantic AI workspace backend.

    Building one does no I/O. The first operation starts a session, or attaches to the one `ref`
    names. The backend does not stop the session; call `destroy_session(ref)`, or let it reach its
    timeout.

    Args:
        region_name: AWS region of the code interpreter; boto3's default when `None`.
        code_interpreter_identifier: The AWS-managed `aws.codeinterpreter.v1` or a custom interpreter.
        session_timeout_seconds: Absolute lifetime of a new session, not an idle timeout.
        ref: Identity of a running session to attach to.
        working_dir: Absolute directory commands start in; the session's home directory when `None`.
        env: Environment variables every command gets; a command's own `env` is layered on top.
    """

    def __init__(
        self,
        *,
        region_name: str | None = None,
        code_interpreter_identifier: str = DEFAULT_IDENTIFIER,
        session_timeout_seconds: int = 900,
        ref: WorkspaceRef | None = None,
        working_dir: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        check_env_types(env)
        self._identifier = code_interpreter_identifier
        self._session_id: str | None = None
        if ref is not None:
            self._identifier, self._session_id = _split_ref(ref)
        self._ref = ref
        self._region_name = region_name
        self._session_timeout_seconds = session_timeout_seconds
        self._working_dir = absolute_path('working_dir', working_dir)
        self._resolved_working_dir: str | None = None
        self._env = dict(env or {})
        self._client: Any = None
        self._lock = anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    async def _session(self) -> str:
        async with self._lock:
            if self._client is None:
                self._client = _new_client(self._region_name)
            if self._session_id is None:

                async def start() -> str:
                    response = await anyio.to_thread.run_sync(
                        lambda: self._client.start_code_interpreter_session(
                            codeInterpreterIdentifier=self._identifier,
                            sessionTimeoutSeconds=self._session_timeout_seconds,
                        )
                    )
                    session_id: str = response['sessionId']
                    self._session_id = session_id
                    self._ref = WorkspaceRef(provider=PROVIDER, id=f'{self._identifier}/{session_id}')
                    logger.info('Started AgentCore Code Interpreter session %s', session_id)
                    return session_id

                return await run_to_completion(start)
            return self._session_id

    async def _invoke(self, name: str, **arguments: str) -> dict[str, Any]:
        """Call one Code Interpreter tool and return its `result`, mapping an ended session to unavailable."""
        session_id = await self._session()

        def call() -> dict[str, Any]:
            response = self._client.invoke_code_interpreter(
                codeInterpreterIdentifier=self._identifier, sessionId=session_id, name=name, arguments=arguments
            )
            result: dict[str, Any] | None = None
            for event in response['stream']:
                if 'result' in event:
                    result = event['result']
                else:
                    raise WorkspaceError(f'AgentCore {name} failed: {event}')
            if result is None:
                raise WorkspaceError(f'AgentCore {name} returned no result')
            return result

        try:
            return await anyio.to_thread.run_sync(call)
        except ClientError as error:
            if _is_inactive(error):
                raise _unavailable(session_id, error) from error
            raise

    async def _shell(self, script: str) -> str:
        """Run a short helper script that exits 0 and return its output with CRLF undone."""
        result = await self._invoke('executeCommand', command=shlex.join(['sh', '-c', f'{script} </dev/null']))
        content = result.get('structuredContent', {})
        output = (content.get('stdout', '') + content.get('stderr', '')).replace('\r\n', '\n')
        if content.get('exitCode') != 0:
            raise WorkspaceError(f'AgentCore helper command failed: {output.strip()}')
        return output

    async def working_dir(self) -> str:
        if self._resolved_working_dir is None:
            prefix = f'cd -- {shlex.quote(self._working_dir)} && ' if self._working_dir else ''
            printed = (await self._shell(f'{prefix}pwd -P')).strip()
            if not posixpath.isabs(printed):
                raise WorkspaceError(f'Could not determine the working directory: {printed!r}')
            self._resolved_working_dir = printed
        return self._resolved_working_dir

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        check_timeout(timeout)
        argv = command_argv(command, shell)
        check_env_types(env)
        cwd = await self.working_dir()
        merged_env = {**self._env, **(env or {})}
        if merged_env:
            argv = ['env', '--', *(f'{name}={value}' for name, value in merged_env.items()), *argv]
        directory = f'/tmp/pydantic-ai-{uuid.uuid4().hex}'
        out, err, rc = f'{directory}/out', f'{directory}/err', f'{directory}/rc'
        script = (
            f'mkdir -p {directory} && cd -- {shlex.quote(cwd)} || exit 125; '
            f'"$@" </dev/null >{out} 2>{err}; echo $? >{rc}.tmp; mv {rc}.tmp {rc}'
        )
        line = shlex.join(['sh', '-c', script, 'sh', *argv])
        task_id: str | None = None
        partial = ('', '')

        async def kill() -> None:
            nonlocal partial
            if task_id is not None:
                await self._invoke('stopTask', taskId=task_id)
            with anyio.move_on_after(10):
                excerpt = await self._shell(
                    f'for f in {out} {err}; do tail -c {_PARTIAL_OUTPUT_LIMIT} "$f" 2>/dev/null | base64 -w0; echo; done; '
                    f'rm -rf {directory}'
                )
                lines = excerpt.split('\n')
                partial = (_decode(lines[0]), _decode(lines[1]))

        async with command_deadline(timeout, stop=kill, output=lambda: partial):
            started = await self._invoke('startCommandExecution', command=line)
            task_id = started_id = str(started['structuredContent']['taskId'])
            while True:
                task = await self._invoke('getTask', taskId=started_id)
                if task.get('structuredContent', {}).get('taskStatus') in _FINISHED:
                    break
                await anyio.sleep(_POLL_INTERVAL)
            result = await self._collect(directory, out, err, rc)
        return result

    async def _collect(self, directory: str, out: str, err: str, rc: str) -> CommandResult:
        try:
            report = await self._shell(
                f'cat {rc} 2>/dev/null || echo missing; '
                f'o=$(wc -c <{out} 2>/dev/null || echo 0); e=$(wc -c <{err} 2>/dev/null || echo 0); echo $((o + e)); '
                f'if [ $((o + e)) -le {_MAX_OUTPUT_BYTES} ]; then base64 -w0 {out}; echo; base64 -w0 {err}; echo; fi'
            )
        finally:
            await stop_shielded(lambda: self._shell(f'rm -rf {directory}'), grace=10)
        lines = report.split('\n')
        if lines[0].strip() == 'missing':
            raise WorkspaceError('The AgentCore command ended without recording an exit code')
        exit_code = int(lines[0])
        if exit_code == 125 and int(lines[1]) == 0:
            raise WorkspaceError(f'working_dir {self._resolved_working_dir!r} does not exist in the session')
        if int(lines[1]) > _MAX_OUTPUT_BYTES:
            raise WorkspaceOutputLimitError(
                'AgentCore command output exceeded the 4 MiB limit; redirect it to a file and read part of it instead',
                limit=_MAX_OUTPUT_BYTES,
                stdout='',
                stderr='',
            )
        return CommandResult(exit_code=exit_code, stdout=_decode(lines[2]), stderr=_decode(lines[3]))


def _decode(encoded: str) -> str:
    return base64.b64decode(encoded.strip()).decode(errors='replace')
