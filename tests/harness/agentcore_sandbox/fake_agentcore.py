"""A host-backed fake of the `bedrock-agentcore` client calls `AgentCoreSandboxBackend` makes.

Each session works in `home`: `startCommandExecution` runs `/bin/sh -c` there in a process group
of its own, and `executeCode` runs the Python program in this process with `exec`, with `print`
captured and `os.getcwd()` reporting `home`, so no interpreter is started. Like the real service,
a stopped session refuses every call with a `ValidationException`, and stopping it again is a
`ConflictException`.

`queue(name, outcome)` replaces the next call of one tool (or `'start'` / `'stop'` for the session
calls) with an exception to raise or a list of stream events to return, for the failure paths
the real service only produces on its own schedule.
"""

from __future__ import annotations

import builtins
import itertools
import os
import re
import signal
import subprocess
import threading
import traceback
import types
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

Outcome = BaseException | list[dict[str, Any]]


def client_error(code: str, message: str, operation: str = 'InvokeCodeInterpreter') -> ClientError:
    return ClientError({'Error': {'Code': code, 'Message': message}}, operation)


class _SessionOs(types.ModuleType):
    """The `os` module, except that `getcwd()` reports the session's home rather than this process's."""

    def __init__(self, cwd: str) -> None:
        super().__init__('os')
        self._cwd = cwd

    def getcwd(self) -> str:
        return self._cwd

    def __getattr__(self, name: str) -> Any:
        return getattr(os, name)


@dataclass
class Hold:
    """A held tool call: `reached` is set once it has done its work, and it answers once `release` is set."""

    reached: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)


@dataclass
class FakeSession:
    session_id: str
    code_interpreter: str
    timeout: int
    active: bool = True
    tasks: dict[str, subprocess.Popen[bytes]] = field(default_factory=dict[str, 'subprocess.Popen[bytes]'])
    canceled: set[str] = field(default_factory=set[str])

    def kill(self) -> None:
        for process in self.tasks.values():
            _kill(process)


def _kill(process: subprocess.Popen[bytes]) -> None:
    """Kill a task's process group, as `stopTask` does, if it is still running."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:  # pragma: no cover - the process exited between poll and kill
            pass
    process.wait()


def _result(stdout: str = '', *, error: bool = False, **structured: Any) -> list[dict[str, Any]]:
    content = [{'type': 'text', 'text': stdout}] if stdout else []
    return [
        {
            'result': {
                'content': content,
                'structuredContent': {'stdout': stdout, 'stderr': '', 'exitCode': int(error), **structured},
                'isError': error,
            }
        }
    ]


class FakeClient:
    def __init__(self, owner: FakeAgentCore, *, single_attempt: bool) -> None:
        self._owner = owner
        self._single_attempt = single_attempt

    def start_code_interpreter_session(
        self, *, codeInterpreterIdentifier: str, sessionTimeoutSeconds: int
    ) -> dict[str, Any]:
        self._owner.take('start')
        if codeInterpreterIdentifier not in self._owner.code_interpreters:
            raise client_error(
                'ResourceNotFoundException',
                f'Code Interpreter not found: {codeInterpreterIdentifier}',
                'StartCodeInterpreterSession',
            )
        session_id = f'session{next(self._owner.ids)}'
        self._owner.sessions[session_id] = FakeSession(session_id, codeInterpreterIdentifier, sessionTimeoutSeconds)
        return {'sessionId': session_id}

    def stop_code_interpreter_session(self, *, codeInterpreterIdentifier: str, sessionId: str) -> dict[str, Any]:
        self._owner.take('stop')
        session = self._owner.sessions.get(sessionId)
        if session is None:
            raise client_error(
                'ResourceNotFoundException', f'Session not found: {sessionId}', 'StopCodeInterpreterSession'
            )
        if not session.active:
            raise client_error(
                'ConflictException', f'Session already terminated: {sessionId}', 'StopCodeInterpreterSession'
            )
        session.active = False
        session.kill()
        return {}

    def invoke_code_interpreter(
        self, *, codeInterpreterIdentifier: str, sessionId: str, name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        owner = self._owner
        owner.calls.append(name)
        operation = re.search(r'"operation": "(\w+)"', arguments.get('code', ''))
        label = f'{name}:{operation[1]}' if operation else name
        (owner.single_attempt_calls if self._single_attempt else owner.retrying_calls).add(label)
        session = owner.sessions.get(sessionId)
        if session is None or not session.active:
            raise client_error('ValidationException', f'Code interpreter session {sessionId} is not active')
        queued = owner.take(name)
        events = queued if queued is not None else self._run(session, name, arguments)
        # The call has done its work; a held response is one that is slow or lost on its way back.
        if (hold := owner.holds.pop(name, None)) is not None:
            hold.reached.set()
            hold.release.wait()
        return {'sessionId': sessionId, 'stream': iter(events)}

    def _run(self, session: FakeSession, name: str, arguments: Mapping[str, Any]) -> list[dict[str, Any]]:
        if name == 'executeCode':
            return self._execute_code(arguments['code'])
        if name == 'startCommandExecution':
            task_id = f'task{next(self._owner.ids)}'
            session.tasks[task_id] = subprocess.Popen(
                ['/bin/sh', '-c', arguments['command']],
                cwd=self._owner.home,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return _result(taskId=task_id, taskStatus='submitted')
        task_id = arguments['taskId']
        process = session.tasks[task_id]
        status = 'canceled' if task_id in session.canceled else 'working' if process.poll() is None else 'completed'
        if name == 'stopTask':
            if status != 'working':
                # AgentCore's own wording for a task that already ended.
                return _result(
                    f'Error executing tool stop_task: Task cannot be canceled (current status: TaskStatus.{status.upper()})',
                    error=True,
                )
            _kill(process)
            session.canceled.add(task_id)
            return _result('Successfully canceled a command execution task')
        assert name == 'getTask', name
        return _result(taskStatus=status)

    def _execute_code(self, code: str) -> list[dict[str, Any]]:
        printed: list[str] = []
        home = str(self._owner.home)

        def capture(*values: object, end: str = '\n', **_: object) -> None:
            printed.append(' '.join(str(value) for value in values) + end)

        def session_import(module: str, *args: Any, **kwargs: Any) -> Any:
            return _SessionOs(home) if module == 'os' else builtins.__import__(module, *args, **kwargs)

        namespace: dict[str, Any] = {'__builtins__': {**vars(builtins), 'print': capture, '__import__': session_import}}
        try:
            exec(code, namespace)
        except Exception:
            return _result(traceback.format_exc(), error=True)
        return _result(''.join(printed).removesuffix('\n'))


class FakeAgentCore:
    """Control surface for the fake: its `module` stands in for `boto3`."""

    def __init__(self, home: Path) -> None:
        self.home = home
        self.holds: dict[str, Hold] = {}
        self._issued_holds: list[Hold] = []
        self.code_interpreters = {'aws.codeinterpreter.v1'}
        self.sessions: dict[str, FakeSession] = {}
        self.ids = itertools.count(1)
        self.calls: list[str] = []
        self.single_attempt_calls: set[str] = set()
        """Each call made through the client limited to one attempt, as `executeCode:<operation>` for a program."""
        self.retrying_calls: set[str] = set()
        self.clients: list[dict[str, Any]] = []
        self._queued: dict[str, list[Outcome]] = {}
        self._lock = threading.Lock()
        self.module = types.SimpleNamespace(client=self.client)

    def client(self, service_name: str, **kwargs: Any) -> FakeClient:
        assert service_name == 'bedrock-agentcore'
        self.clients.append(kwargs)
        return FakeClient(self, single_attempt=kwargs['config'].retries == {'total_max_attempts': 1})

    def queue(self, name: str, outcome: Outcome) -> None:
        """Replace the next `name` call with `outcome`: an exception to raise, or stream events to return."""
        with self._lock:
            self._queued.setdefault(name, []).append(outcome)

    def take(self, name: str) -> list[dict[str, Any]] | None:
        with self._lock:
            pending = self._queued.get(name)
            outcome = pending.pop(0) if pending else None
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def queue_code_reply(self, stdout: str, *, error: bool = False) -> None:
        """Answer the next `executeCode` call with `stdout`, as a failed execution when `error`."""
        self.queue('executeCode', _result(stdout, error=error))

    def queue_task(self, name: str, **structured: Any) -> None:
        """Answer the next `name` task call with this structured content."""
        self.queue(name, _result(**structured))

    def active(self) -> Iterator[FakeSession]:
        return (session for session in self.sessions.values() if session.active)

    def hold(self, name: str) -> Hold:
        """Hold the response to the next `name` call, after it has done its work, until `release` is set."""
        hold = self.holds[name] = Hold()
        self._issued_holds.append(hold)
        return hold

    def close(self) -> None:
        # Also the ones a test reached but never released, so no worker thread stays blocked.
        for hold in self._issued_holds:
            # Waking `reached` too frees a test thread still waiting for a call that never came.
            hold.reached.set()
            hold.release.set()
        for session in self.sessions.values():
            session.kill()


__all__ = ['FakeAgentCore', 'FakeClient', 'FakeSession', 'Hold', 'Outcome', 'client_error']
