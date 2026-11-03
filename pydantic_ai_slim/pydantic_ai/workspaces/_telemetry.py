"""OpenTelemetry spans for workspace calls.

A workspace has no `RunContext` and can outlive a run, so the run publishes its tracer in a context
variable instead. Outside an instrumented run there is none, and nothing is traced.
"""

from __future__ import annotations as _annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, TypeAlias

from opentelemetry.trace import INVALID_SPAN, NoOpTracer, Span, Tracer
from opentelemetry.util.types import AttributeValue

from .._instrumentation import record_uncaught_errors
from ..messages import WorkspaceRef
from .protocol import WorkspaceBackend, WorkspaceCommand

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
Lifecycle: TypeAlias = Literal['create', 'attach', 'destroy']
"""What a backend does to its environment, apart from the facade's operations."""
Step: TypeAlias = Literal[
    'probe',
    'read_chunk',
    'write_chunk',
    'commit',
    'cleanup',
    'stat',
    'list_dir',
    'make_dir',
    'remove',
    'exists',
    'realpath',
]
"""What a shell-filesystem command does within its file operation."""


@dataclass(frozen=True)
class _Telemetry:
    tracer: Tracer
    include_content: bool


_telemetry: ContextVar[_Telemetry | None] = ContextVar('workspace_telemetry', default=None)

_in_operation: ContextVar[bool] = ContextVar('workspace_operation', default=False)
"""True inside an operation's span, so nested facade calls (such as `Workspace(self)` in a backend) add no span."""


def publish_workspace_telemetry(tracer: Tracer, *, include_content: bool) -> None:
    """Trace workspace calls in the current context with `tracer`; a `NoOpTracer` turns tracing off."""
    _telemetry.set(None if isinstance(tracer, NoOpTracer) else _Telemetry(tracer, include_content))
    # A run started from inside an operation traces its own calls.
    _in_operation.set(False)


@contextmanager
def workspace_telemetry_scope(tracer: Tracer, *, include_content: bool) -> Generator[None]:
    """`publish_workspace_telemetry` for the duration of the block."""
    telemetry, in_operation = _telemetry.get(), _in_operation.get()
    publish_workspace_telemetry(tracer, include_content=include_content)
    try:
        yield
    finally:
        _telemetry.set(telemetry)
        _in_operation.set(in_operation)


def set_content(span: Span, key: str, value: AttributeValue) -> None:
    """Set an attribute only when the run captures content."""
    if (telemetry := _telemetry.get()) is not None and telemetry.include_content:
        span.set_attribute(key, value)


def set_command(span: Span, command: WorkspaceCommand) -> None:
    """Record `command` on a `workspace run` span, only when the run captures content."""
    if isinstance(command, str):
        set_content(span, 'process.command_line', command)
    else:
        set_content(span, 'process.command_args', list(command))


@contextmanager
def operation_span(operation: Operation, workspace: WorkspaceBackend) -> Generator[Span]:
    """Open a `workspace {operation}` span, unless tracing is off or an operation's span is already open."""
    if _in_operation.get():
        yield INVALID_SPAN
        return
    with _span(operation) as span:
        try:
            yield span
        finally:
            # Read at the end: a lazy backend creates its sandbox during its first operation.
            _record_ref(span, workspace.ref)


@contextmanager
def lifecycle_span(action: Lifecycle, workspace: WorkspaceBackend | WorkspaceRef) -> Generator[Span]:
    """Open a `workspace create`, `attach` or `destroy` span around a backend's call to its provider.

    It opens even inside an operation's span: a lazy backend creates or attaches during its first
    operation, and this span is what separates that time from the operation's own.
    """
    with _span(action) as span:
        # A failed create may drop the ref it had allocated; the span still names that sandbox.
        entry_ref = _ref_of(workspace)
        try:
            yield span
        finally:
            _record_ref(span, _ref_of(workspace) or entry_ref)


@contextmanager
def command_span(step: Step) -> Generator[Span]:
    """Open a `workspace run` span for a shell-filesystem command, recording its `step` but never its text, which can carry file contents."""
    # The message names the step, so it can't pass for a command the agent ran. It never holds the
    # command: Logfire does not scrub `logfire.msg`.
    with _span('run', f'workspace run ({step})') as span:
        span.set_attribute('pydantic_ai.workspace.command.step', step)
        yield span


def _ref_of(workspace: WorkspaceBackend | WorkspaceRef) -> WorkspaceRef | None:
    return workspace if isinstance(workspace, WorkspaceRef) else workspace.ref


def _record_ref(span: Span, ref: WorkspaceRef | None) -> None:
    if span.is_recording() and ref is not None:
        span.set_attribute('pydantic_ai.workspace.provider', ref.provider)
        span.set_attribute('pydantic_ai.workspace.id', ref.id)


@contextmanager
def _span(operation: Operation | Lifecycle, message: str | None = None) -> Generator[Span]:
    if (telemetry := _telemetry.get()) is None:
        yield INVALID_SPAN
        return
    # Set at the start, so Logfire's console and pending spans show it too.
    with telemetry.tracer.start_as_current_span(
        f'workspace {operation}',
        attributes=None if message is None else {'logfire.msg': message},
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        token = _in_operation.set(True)
        try:
            # A missing file is often an expected answer, not a failure.
            with record_uncaught_errors(
                span, include_content=telemetry.include_content, not_errors=(FileNotFoundError,)
            ):
                yield span
        finally:
            _in_operation.reset(token)
