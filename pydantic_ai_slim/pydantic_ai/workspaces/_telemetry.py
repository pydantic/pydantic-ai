"""OpenTelemetry spans for workspace calls.

A workspace has no `RunContext` and can outlive a run, so the run publishes its tracer in a context
variable instead. Outside an instrumented run there is none, and nothing is traced.

Every span opens through `_span`. Each public helper only picks its name and what it records:

- `operation_span`: `workspace {operation}`, one per `Workspace` call, skipped inside another workspace span.
- `acquire_span`: `workspace create` or `workspace attach`, around a backend getting its environment.
- `destroy_span`: `workspace destroy`, around a capability deleting an environment.
- `command_span`: `workspace run`, one per shell command that emulates a file operation.
"""

from __future__ import annotations as _annotations

from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

from opentelemetry.trace import INVALID_SPAN, NoOpTracer, Span, Tracer
from opentelemetry.util.types import Attributes, AttributeValue

from .._instrumentation import record_uncaught_errors
from ..messages import WorkspaceRef
from .protocol import WorkspaceBackend, WorkspaceCommand

__all__ = (
    'Operation',
    'Step',
    'acquire_span',
    'command_span',
    'destroy_span',
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
"""A call on the `Workspace` facade."""
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
_SpanName: TypeAlias = Operation | Literal['create', 'attach', 'destroy']


@dataclass(frozen=True)
class _Telemetry:
    tracer: Tracer
    include_content: bool
    in_workspace_span: bool = False
    """True inside a workspace span, so a nested `Workspace` call (such as `Workspace(self)` in a backend) adds none."""


_run_telemetry: ContextVar[_Telemetry | None] = ContextVar('workspace_telemetry', default=None)


def publish_workspace_telemetry(tracer: Tracer, *, include_content: bool) -> None:
    """Trace workspace calls in the current context with `tracer`; a `NoOpTracer` turns tracing off."""
    _run_telemetry.set(_new_telemetry(tracer, include_content))


@contextmanager
def workspace_telemetry_scope(tracer: Tracer, *, include_content: bool) -> Generator[None]:
    """`publish_workspace_telemetry` for the duration of the block."""
    token = _run_telemetry.set(_new_telemetry(tracer, include_content))
    try:
        yield
    finally:
        _run_telemetry.reset(token)


def _new_telemetry(tracer: Tracer, include_content: bool) -> _Telemetry | None:
    # Starts outside any span: a run started from inside a workspace span traces its own calls.
    return None if isinstance(tracer, NoOpTracer) else _Telemetry(tracer, include_content)


def set_content(span: Span, key: str, value: AttributeValue) -> None:
    """Set an attribute only when the published run captures content."""
    if (telemetry := _run_telemetry.get()) is not None and telemetry.include_content:
        span.set_attribute(key, value)


def set_command(span: Span, command: WorkspaceCommand) -> None:
    """Record `command` on a `workspace run` span, only when the run captures content."""
    if isinstance(command, str):
        set_content(span, 'process.command_line', command)
    else:
        set_content(span, 'process.command_args', list(command))


def operation_span(operation: Operation, workspace: WorkspaceBackend) -> AbstractContextManager[Span]:
    """Open a `workspace {operation}` span, unless a workspace span is already open."""
    if (telemetry := _run_telemetry.get()) is not None and telemetry.in_workspace_span:
        return nullcontext(INVALID_SPAN)
    return _span(operation, backend=workspace)


def acquire_span(action: Literal['create', 'attach'], backend: WorkspaceBackend) -> AbstractContextManager[Span]:
    """Open a `workspace create` or `workspace attach` span around `backend`'s call to its provider.

    It opens even inside an operation's span: a lazy backend creates or attaches during its first
    operation, and this span is what separates that time from the operation's own.
    """
    return _span(action, backend=backend)


def destroy_span(ref: WorkspaceRef) -> AbstractContextManager[Span]:
    """Open a `workspace destroy` span around the provider call that deletes `ref`'s environment."""
    return _span('destroy', attributes=_ref_attributes(ref))


def command_span(step: Step) -> AbstractContextManager[Span]:
    """Open a `workspace run` span for a shell-filesystem command, recording its `step` but never its text, which can carry file contents."""
    # The message names the step, so it can't pass for a command the agent ran. It never holds the
    # command: Logfire does not scrub `logfire.msg`.
    return _span(
        'run', attributes={'logfire.msg': f'workspace run ({step})', 'pydantic_ai.workspace.command.step': step}
    )


@contextmanager
def _span(
    operation: _SpanName,
    *,
    attributes: Attributes = None,
    backend: WorkspaceBackend | None = None,
) -> Generator[Span]:
    """Open `workspace {operation}`, recording errors and, given `backend`, the environment it ran in.

    `attributes` are set at the start, so Logfire's console and pending spans show them too.
    """
    if (telemetry := _run_telemetry.get()) is None:
        yield INVALID_SPAN
        return
    with telemetry.tracer.start_as_current_span(
        f'workspace {operation}', attributes=attributes, record_exception=False, set_status_on_exception=False
    ) as span:
        if backend is not None:
            # Read at the start: a failed attach may drop the ref it was given.
            span.set_attributes(_ref_attributes(backend.ref))
        token = _run_telemetry.set(replace(telemetry, in_workspace_span=True))
        try:
            # A missing file is often an expected answer, not a failure.
            with record_uncaught_errors(
                span, include_content=telemetry.include_content, not_errors=(FileNotFoundError,)
            ):
                yield span
        finally:
            _run_telemetry.reset(token)
            if backend is not None:
                # Read at the end too: a lazy backend creates its environment during its first operation.
                span.set_attributes(_ref_attributes(backend.ref))


def _ref_attributes(ref: WorkspaceRef | None) -> dict[str, AttributeValue]:
    if ref is None:
        return {}
    return {'pydantic_ai.workspace.provider': ref.provider, 'pydantic_ai.workspace.id': ref.id}
