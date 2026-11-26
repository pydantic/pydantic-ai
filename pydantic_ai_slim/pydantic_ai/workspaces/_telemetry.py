"""OpenTelemetry spans for workspace calls, on the calling run's telemetry (`pydantic_ai.telemetry`).

- `operation_span`: `workspace {operation}`, one per `Workspace` call, skipped inside another workspace span.
- `acquire_span`: `workspace create` or `workspace attach`, around a backend getting its environment.
- `destroy_span`: `workspace destroy`, around a capability deleting an environment.
- `command_span`: `workspace run`, one per shell command that emulates a file operation.
"""

from __future__ import annotations as _annotations

from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from typing import Literal, TypeAlias

from opentelemetry.trace import INVALID_SPAN, Span
from opentelemetry.util.types import AttributeValue

from .._instrumentation import record_exception
from ..messages import WorkspaceRef
from ..telemetry import RunTelemetry, current_run_telemetry, open_span, records_content, set_content_attribute
from .protocol import WorkspaceBackend, WorkspaceCommand

__all__ = (
    'Operation',
    'Step',
    'acquire_span',
    'command_span',
    'destroy_span',
    'operation_span',
    'set_command',
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
_LOOKUPS: frozenset[Operation] = frozenset({'stat', 'read_bytes', 'list_dir', 'remove', 'realpath'})
"""Operations where a missing file is often the expected answer, so `FileNotFoundError` leaves the status unset.
Every other span, such as `workspace run` with a missing `cwd`, records it as an error."""
_SpanName: TypeAlias = Operation | Literal['create', 'attach', 'destroy']

_opened_by: ContextVar[RunTelemetry | None] = ContextVar('pydantic_ai.workspace_span_opened_by', default=None)
"""The run whose workspace span is open, so that run's nested `Workspace` calls (such as `Workspace(self)`
in a backend) add none, while a run started inside it, with its own telemetry, traces its own."""


def operation_span(operation: Operation, workspace: WorkspaceBackend) -> AbstractContextManager[Span]:
    """Open a `workspace {operation}` span, unless the calling run's workspace span is already open."""
    # By identity: each run publishes a new `RunTelemetry`, so a run started inside this one is never taken for it.
    if (telemetry := current_run_telemetry()) is not None and _opened_by.get() is telemetry:
        return nullcontext(INVALID_SPAN)
    return _span(operation, backend=workspace, missing_is_answer=operation in _LOOKUPS)


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
        'run',
        attributes={'logfire.msg': f'workspace run ({step})', 'pydantic_ai.workspace.command.step': step},
    )


def set_command(span: Span, command: WorkspaceCommand) -> None:
    """Record `command` on a `workspace run` span, only when the run captures content."""
    if isinstance(command, str):
        set_content_attribute(current_run_telemetry(), span, 'process.command_line', command)
    else:
        set_content_attribute(current_run_telemetry(), span, 'process.command_args', command)


@contextmanager
def _span(
    name: _SpanName,
    *,
    attributes: Mapping[str, AttributeValue] | None = None,
    backend: WorkspaceBackend | None = None,
    missing_is_answer: bool = False,
) -> Generator[Span]:
    """Open `workspace {name}`, naming the environment it ran in when given `backend`."""
    telemetry = current_run_telemetry()
    token = _opened_by.set(telemetry)
    missing: FileNotFoundError | None = None
    try:
        with open_span(telemetry, f'workspace {name}', attributes=attributes) as span:
            # The ref is read only when recording, so an untraced call never touches it.
            traced = backend if span.is_recording() else None
            if traced is not None:
                # Read at the start: a failed attach may drop the ref it was given.
                span.set_attributes(_ref_attributes(traced.ref))
            try:
                yield span
            except FileNotFoundError as error:
                if not missing_is_answer:
                    raise
                # Recorded, then raised once the span has ended, so `open_span` doesn't mark it ERROR.
                record_exception(span, error, include_content=records_content(telemetry), escaped=False)
                missing = error
            finally:
                if traced is not None:
                    # Read at the end too: a lazy backend creates its environment during its first operation.
                    span.set_attributes(_ref_attributes(traced.ref))
    finally:
        _opened_by.reset(token)
    if missing is not None:
        raise missing


def _ref_attributes(ref: WorkspaceRef | None) -> dict[str, AttributeValue]:
    if ref is None:
        return {}
    return {'pydantic_ai.workspace.provider': ref.provider, 'pydantic_ai.workspace.id': ref.id}
