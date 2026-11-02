"""OpenTelemetry spans for workspace operations, and for the shell commands that carry out file operations.

A workspace has no `RunContext`: it is selected before the run's tracer is known and can outlive the
run. The run publishes its tracer and content setting in a context variable instead, so they reach
every tool, hook and task the run starts, the way `_instrumentation.ContentPolicy` carries a model
request's tracer to the spans opened beneath it. A run without instrumentation publishes none, and a
call outside any run finds none, so neither emits anything.
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
from .protocol import WorkspaceBackend, WorkspaceCommand

__all__ = (
    'Operation',
    'Step',
    'command_span',
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
    """Trace workspace calls in the current context with `tracer`; a `NoOpTracer` turns tracing off."""
    _telemetry.set(None if isinstance(tracer, NoOpTracer) else _Telemetry(tracer, include_content))
    # A run started inside a workspace operation, such as from a backend, traces its own operations again.
    _in_operation.set(False)


@contextmanager
def workspace_telemetry_scope(tracer: Tracer, *, include_content: bool) -> Generator[None]:
    """Publish workspace telemetry for the block only, for a durable unit that may run in its caller's context."""
    telemetry, in_operation = _telemetry.get(), _in_operation.get()
    publish_workspace_telemetry(tracer, include_content=include_content)
    try:
        yield
    finally:
        _telemetry.set(telemetry)
        _in_operation.set(in_operation)


def set_content(span: Span, key: str, value: AttributeValue) -> None:
    """Set an attribute that can quote what the model or the user wrote, only when the run captures content."""
    if (telemetry := _telemetry.get()) is not None and telemetry.include_content:
        span.set_attribute(key, value)


def set_command(span: Span, command: WorkspaceCommand) -> None:
    """Record the command a `workspace run` span runs, only when the run captures content."""
    if isinstance(command, str):
        set_content(span, 'process.command_line', command)
    else:
        set_content(span, 'process.command_args', list(command))


@contextmanager
def operation_span(operation: Operation, workspace: WorkspaceBackend) -> Generator[Span]:
    """Open a `workspace {operation}` span, unless tracing is off or a workspace operation is already open.

    The first workspace layer an operation enters opens its span; the layers and backend calls beneath it
    run inside that span instead of adding their own.
    """
    if _in_operation.get():
        yield INVALID_SPAN
        return
    with _span(operation) as span:
        try:
            yield span
        finally:
            # Read at the end: the first operation on a lazy backend is what creates its sandbox.
            if span.is_recording() and (ref := workspace.ref) is not None:
                span.set_attribute('pydantic_ai.workspace.provider', ref.provider)
                span.set_attribute('pydantic_ai.workspace.id', ref.id)


@contextmanager
def command_span(step: Step) -> Generator[Span]:
    """Open a `workspace run` span for a command the shell filesystem runs as part of a file operation.

    It opens inside the file operation's span, which names the workspace. `step` says what the command
    does; the command's text is never recorded, as it can carry the file's contents.
    """
    with _span('run') as span:
        span.set_attribute('pydantic_ai.workspace.command.step', step)
        yield span


@contextmanager
def _span(operation: Operation) -> Generator[Span]:
    if (telemetry := _telemetry.get()) is None:
        yield INVALID_SPAN
        return
    with telemetry.tracer.start_as_current_span(
        f'workspace {operation}', record_exception=False, set_status_on_exception=False
    ) as span:
        token = _in_operation.set(True)
        try:
            # A missing file is often the answer the caller asked for (a status file not written yet), not a failure.
            with record_uncaught_errors(
                span, include_content=telemetry.include_content, not_errors=(FileNotFoundError,)
            ):
                yield span
        finally:
            _in_operation.reset(token)
