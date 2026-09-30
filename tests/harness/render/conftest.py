from __future__ import annotations

import importlib.util
import inspect
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

try:
    _render_spec = importlib.util.find_spec('render')
except ModuleNotFoundError as exc:
    if exc.name != 'render':
        raise
    _render_spec = None

if TYPE_CHECKING:
    from render.workflows import TaskContext, TaskDefinition
elif _render_spec is None:
    collect_ignore_glob = ['*.py']

    class TaskContext:
        """Placeholder used only while pytest ignores Render-extra tests."""
else:
    from render.workflows import TaskContext, TaskDefinition


def pytest_ignore_collect(collection_path: Path) -> bool:
    """Ignore Render-extra tests only when the top-level optional package is absent."""
    return _render_spec is None and collection_path.suffix == '.py'


P = ParamSpec('P')
R = TypeVar('R')

_RENDER_CREDENTIAL_KEYS = frozenset(
    {
        'RENDER_API_KEY',
        'RENDER_CLI_TOKEN',
        'RENDER_TOKEN',
        'RENDER_WORKSPACE_ID',
    }
)


def renderless_environment() -> dict[str, str]:
    """Return the current environment without Render credentials."""
    return {
        key: value
        for key, value in os.environ.items()
        if key not in _RENDER_CREDENTIAL_KEYS and not key.startswith('COVERAGE_')
    }


class RecordingTaskContext(TaskContext):
    """Execute child tasks locally while recording their public names."""

    def __init__(self) -> None:
        self.task_names: list[str] = []

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        self.task_names.append(task.name)
        result = task.func(self, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result


class LocalTask(BaseModel):
    """A task registered on the local Render development server."""

    model_config = ConfigDict(extra='ignore')

    id: str
    name: str


class LocalTaskRun(BaseModel):
    """A local task-run record returned by the Render CLI."""

    model_config = ConfigDict(extra='ignore', populate_by_name=True)

    id: str
    status: str
    parent_task_run_id: str = Field(alias='parentTaskRunId')
    root_task_run_id: str = Field(alias='rootTaskRunId')
    results: list[object] | None = None


_JSON_VALUE = TypeAdapter(object)
_LOCAL_TASKS = TypeAdapter(list[LocalTask])
_LOCAL_RUN = TypeAdapter(LocalTaskRun)
_LOCAL_RUNS = TypeAdapter(list[LocalTaskRun])


@dataclass(frozen=True)
class LocalRenderRuntime:
    """Handle for a keyless local Render Workflows development server."""

    port: int
    process: subprocess.Popen[str]
    log_path: Path
    repository: Path

    def cli(self, arguments: Sequence[str], *, check: bool = True) -> object | None:
        """Run a non-interactive CLI command and decode its JSON response."""
        command = [
            'render',
            'workflows',
            *arguments,
            '--local',
            '--port',
            str(self.port),
            '--confirm',
            '--output',
            'json',
        ]
        completed = subprocess.run(
            command,
            cwd=self.repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            env=renderless_environment(),
        )
        if check and completed.returncode != 0:
            pytest.fail(
                f'Command failed ({completed.returncode}): {" ".join(command)}\n'
                f'stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n'
                f'local runtime log: {self.log_path}\n{self.logs()}'
            )
        if completed.returncode != 0:
            return None
        if not completed.stdout.strip():
            if not check:
                return None
            pytest.fail(
                f'Command returned empty output: {" ".join(command)}\n'
                f'stderr:\n{completed.stderr}\n'
                f'local runtime log: {self.log_path}\n{self.logs()}'
            )
        try:
            return _JSON_VALUE.validate_json(completed.stdout)
        except ValidationError:
            if not check:
                return None
            pytest.fail(
                f'Command returned non-JSON output: {" ".join(command)}\n'
                f'stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\n'
                f'local runtime log: {self.log_path}\n{self.logs()}'
            )

    def list_tasks(self, *, check: bool = True) -> list[LocalTask]:
        """Return the tasks currently registered on the local server."""
        payload = self.cli(['tasks', 'list'], check=check)
        if payload is None:
            return []
        try:
            return _LOCAL_TASKS.validate_python(payload)
        except ValidationError:
            if not check:
                return []
            raise

    def start_task(self, task_slug: str, input_json: str) -> LocalTaskRun:
        """Start one local task run with JSON-array input."""
        return _LOCAL_RUN.validate_python(self.cli(['tasks', 'start', task_slug, '--input', input_json]))

    def list_runs(self, task_name: str) -> list[LocalTaskRun]:
        """Return local runs for one registered task name."""
        return _LOCAL_RUNS.validate_python(self.cli(['runs', 'list', task_name]))

    def wait_for_run(self, run_id: str, *, timeout: float = 60) -> LocalTaskRun:
        """Poll a run until the local task server reports a terminal state."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            run = _LOCAL_RUN.validate_python(self.cli(['runs', 'show', run_id]))
            if run.status in {'completed', 'failed', 'canceled'}:
                return run
            if self.process.poll() is not None:
                break
            time.sleep(0.2)
        pytest.fail(f'Run {run_id} did not finish\nlocal runtime log: {self.log_path}\n{self.logs()}')

    def logs(self) -> str:
        """Read diagnostics emitted by the local task server and workers."""
        return self.log_path.read_text(errors='replace') if self.log_path.exists() else '<missing>'


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(('127.0.0.1', 0))
        return int(server.getsockname()[1])


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    """Stop the dev server and the task workers it spawned, within bounded time.

    The server is started in its own process session, so signalling the group reaches the worker
    processes too. Each wait is bounded, and a wait that expires escalates rather than blocks.
    """
    for send, timeout in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        if process.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(process.pid), send)
        except (AttributeError, OSError):
            # No process group to signal: fall back to the direct child.
            if send == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        return


def _require_render_cli() -> None:
    executable = shutil.which('render')
    if executable is None:
        pytest.skip('local Render Workflows runtime requires the `render` CLI')
    completed = subprocess.run(
        [executable, '--version'],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
        env=renderless_environment(),
    )
    version_match = re.search(r'render v(\d+)\.(\d+)\.(\d+)', completed.stdout + completed.stderr)
    if completed.returncode != 0 or version_match is None:
        pytest.skip(f'could not determine Render CLI version: {completed.stdout}{completed.stderr}'.strip())
    version = tuple(int(part) for part in version_match.groups())
    if version < (2, 28, 0):
        pytest.skip(f'local runtime test requires Render CLI >=2.28.0, found {version_match.group(0)}')


@pytest.fixture
def local_render_runtime(tmp_path: Path) -> Iterator[LocalRenderRuntime]:
    """Start and cleanly stop the opt-in process-isolated local runtime."""
    if os.getenv('PYDANTIC_AI_HARNESS_RENDER_LOCAL_RUNTIME') != '1':
        pytest.skip('set PYDANTIC_AI_HARNESS_RENDER_LOCAL_RUNTIME=1 to run the local Render runtime test')
    _require_render_cli()

    repository = Path(__file__).resolve().parents[3]
    log_path = tmp_path / 'render-workflows-dev.log'
    log_file = log_path.open('w')
    port = _unused_local_port()
    command = [
        'render',
        'workflows',
        'dev',
        '--port',
        str(port),
        '--confirm',
        '--output',
        'text',
        '--',
        sys.executable,
        'tests/harness/render/runtime_app.py',
    ]
    if os.getenv('PYDANTIC_AI_HARNESS_RENDER_LOCAL_RUNTIME_DEBUG') == '1':
        command.insert(3, '--debug')
    process = subprocess.Popen(
        command,
        cwd=repository,
        env=renderless_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    runtime = LocalRenderRuntime(port=port, process=process, log_path=log_path, repository=repository)

    def capture_logs() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
        except (OSError, ValueError):
            # Teardown closed the stream or the log file while this thread was still reading.
            pass

    log_thread = threading.Thread(target=capture_logs, name='render-workflows-log-capture', daemon=True)
    log_thread.start()

    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if runtime.list_tasks(check=False):
                break
            if process.poll() is not None:
                pytest.fail(f'local runtime exited before readiness\nlog: {log_path}\n{runtime.logs()}')
            time.sleep(0.2)
        else:
            pytest.fail(f'local runtime was not ready within 30 seconds\nlog: {log_path}\n{runtime.logs()}')
        yield runtime
    finally:
        try:
            _stop_process_group(process)
        finally:
            # Release the capture thread and both file handles even when stopping the group failed.
            with log_file:
                log_thread.join(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()
