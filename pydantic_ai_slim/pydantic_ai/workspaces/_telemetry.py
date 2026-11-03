"""OpenTelemetry spans for workspace calls.

A workspace has no `RunContext` and can outlive a run, so the run publishes its tracer in a context
variable instead. Outside an instrumented run there is none, and nothing is traced.
"""

from __future__ import annotations as _annotations

import shlex
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
def operation_span(
    operation: Operation, workspace: WorkspaceBackend, subject: WorkspaceCommand | None = None
) -> Generator[Span]:
    """Open a `workspace {operation}` span, unless tracing is off or an operation's span is already open.

    `subject`, the path or command, is shown in the span's message only when the run captures content.
    """
    if _in_operation.get():
        yield INVALID_SPAN
        return
    unattached = workspace.ref is None
    with _span(operation) as span:
        message = f'workspace {operation}'
        if subject is not None and (telemetry := _telemetry.get()) is not None and telemetry.include_content:
            message += f': {_one_line(subject)}'
        span.set_attribute('logfire.msg', message)
        try:
            yield span
        finally:
            # Read at the end: a lazy backend creates its sandbox during its first operation.
            if span.is_recording() and (ref := workspace.ref) is not None:
                span.set_attribute('pydantic_ai.workspace.provider', ref.provider)
                span.set_attribute('pydantic_ai.workspace.id', ref.id)
                if unattached:
                    span.set_attribute('pydantic_ai.workspace.created', True)
                    span.set_attribute('logfire.msg', f'{message} (sandbox created)')


@contextmanager
def command_span(step: Step) -> Generator[Span]:
    """Open a `workspace run` span for a shell-filesystem command, recording its `step` but never its text, which can carry file contents."""
    with _span('run') as span:
        span.set_attribute('pydantic_ai.workspace.command.step', step)
        span.set_attribute('logfire.msg', f'workspace run ({step})')
        yield span


def _one_line(subject: WorkspaceCommand) -> str:
    text = subject if isinstance(subject, str) else shlex.join(subject)
    first, newline, _ = text.strip().partition('\n')
    return f'{first} …' if newline else first


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
            # A missing file is often an expected answer, not a failure.
            with record_uncaught_errors(
                span, include_content=telemetry.include_content, not_errors=(FileNotFoundError,)
            ):
                yield span
        finally:
            _in_operation.reset(token)
