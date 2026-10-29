"""OpenTelemetry spans for workspace operations, and for the shell commands that carry out file operations.

A workspace has no `RunContext`: it is selected before the run's tracer is known and can outlive the
run. The run publishes its tracer and content setting in a context variable instead, so they reach
every tool, hook and task the run starts, the way `_instrumentation.ContentPolicy` carries a model
request's tracer to the spans opened beneath it. A run without instrumentation publishes none, and a
call outside any run finds none, so neither emits anything.
"""

from __future__ import annotations as _annotations

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, TypeAlias

from opentelemetry.trace import INVALID_SPAN, NoOpTracer, Span, Tracer
from opentelemetry.util.types import AttributeValue

from .._instrumentation import record_exception, set_error_status
from .protocol import WorkspaceBackend, WorkspaceCommand, WorkspaceRef

__all__ = (
    'Lifecycle',
    'Operation',
    'Step',
    'command_span',
    'lifecycle_span',
    'operation_span',
    'publish_workspace_telemetry',
    'set_command',
    'set_content',
    'stop_span',
    'workspace_telemetry_scope',
)


Operation: TypeAlias = Literal[
    'working_dir',
    'run',
    'read_bytes',
    'write_bytes',
    'stat',
    'list_dir',
    'make_dir',
    'remove',
    'exists',
    'realpath',
]
Lifecycle: TypeAlias = Literal['create', 'attach', 'stop', 'destroy']
"""What a backend does to its environment or a running command, apart from the facade's operations."""
Step: TypeAlias = Literal[
    'probe',
    'read_chunk',
    'write_chunk',
    'commit',
    'cleanup',
    'stat',
    'list',
    'make_dir',
    'remove',
    'exists',
    'realpath',
]
"""What a shell-filesystem command does for the file operation it is part of."""


@dataclass(frozen=True)
class _Telemetry:
    tracer: Tracer
    include_content: bool


_telemetry: ContextVar[_Telemetry | None] = ContextVar('workspace_telemetry', default=None)

_in_operation: ContextVar[bool] = ContextVar('workspace_operation', default=False)
"""Set while a workspace operation's span is open, so facade calls made on the way, such as a backend
resolving a path through `Workspace(self)`, stay inside that span rather than adding their own."""


def publish_workspace_telemetry(tracer: Tracer, *, include_content: bool) -> None:
    """Trace workspace calls in the current context with `tracer`; a `NoOpTracer` turns tracing off.

    A run started inside a workspace operation, such as from a backend, traces its own operations again.
    """
    _telemetry.set(None if isinstance(tracer, NoOpTracer) else _Telemetry(tracer, include_content))
    _in_operation.set(False)


@contextmanager
def workspace_telemetry_scope(tracer: Tracer, *, include_content: bool) -> Generator[None]:
    """Publish workspace telemetry for the block only, for a durable unit that may run in its caller's context."""
    tokens = _telemetry.set(None), _in_operation.set(False)
    try:
        publish_workspace_telemetry(tracer, include_content=include_content)
        yield
    finally:
        _telemetry.reset(tokens[0])
        _in_operation.reset(tokens[1])


def set_content(span: Span, key: str, value: AttributeValue) -> None:
    """Set an attribute that can quote what the model or the user wrote, only when the run captures content."""
    if (telemetry := _telemetry.get()) is not None and telemetry.include_content:
        span.set_attribute(key, value)


def set_command(span: Span, command: WorkspaceCommand) -> None:
    if isinstance(command, str):
        set_content(span, 'process.command_line', command)
    else:
        set_content(span, 'process.command_args', list(command))


@contextmanager
def operation_span(operation: Operation, backend: WorkspaceBackend) -> Generator[Span]:
    """Open a `workspace {operation}` span, unless tracing is off or a workspace operation is already open."""
    if _in_operation.get():
        yield INVALID_SPAN
        return
    with _span(operation, backend) as span:
        yield span


@contextmanager
def command_span(step: Step) -> Generator[Span]:
    """Open a `workspace run` span for a command the shell filesystem runs as part of a file operation.

    It opens inside the file operation's span, which names the workspace. `step` says what the command
    does; the command's text is never recorded, as it can carry the file's contents.
    """
    with _span('run') as span:
        span.set_attribute('pydantic_ai.workspace.command.step', step)
        yield span


def stop_span() -> AbstractContextManager[Span]:
    """Open a `workspace stop` span while a backend stops a command that timed out, was cancelled or failed.

    It opens inside the command's `workspace run` span, which names the workspace and records why.
    """
    return _span('stop')


def lifecycle_span(operation: Lifecycle, workspace: WorkspaceBackend | WorkspaceRef) -> AbstractContextManager[Span]:
    """Open a `workspace create`, `attach` or `destroy` span around a sandbox lifecycle call.

    It opens even inside a workspace operation's span, as the first operation on a lazy backend is
    often what creates or attaches to its sandbox.
    """
    return _span(operation, workspace)


@contextmanager
def _span(
    operation: Operation | Lifecycle, workspace: WorkspaceBackend | WorkspaceRef | None = None
) -> Generator[Span]:
    telemetry = _telemetry.get()
    if telemetry is None:
        yield INVALID_SPAN
        return
    with telemetry.tracer.start_as_current_span(
        f'workspace {operation}', record_exception=False, set_status_on_exception=False
    ) as span:
        # A failed create may drop the ref it had allocated; the span still names that sandbox.
        entry_ref = _ref(workspace)
        token = _in_operation.set(True)
        try:
            yield span
        except Exception as error:
            record_exception(span, error, include_content=telemetry.include_content, escaped=False)
            # A missing file is often the answer the caller asked for (a status file not written yet), not a failure.
            if not isinstance(error, FileNotFoundError):
                set_error_status(span, error, include_content=telemetry.include_content)
            raise
        finally:
            _in_operation.reset(token)
            # A sandbox this operation created has its ref only now.
            if (ref := _ref(workspace) or entry_ref) is not None:
                span.set_attribute('pydantic_ai.workspace.provider', ref.provider)
                span.set_attribute('pydantic_ai.workspace.id', ref.id)


def _ref(workspace: WorkspaceBackend | WorkspaceRef | None) -> WorkspaceRef | None:
    return workspace if isinstance(workspace, WorkspaceRef) or workspace is None else workspace.ref
