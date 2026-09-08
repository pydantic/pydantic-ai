from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import anyio

from pydantic_ai._run_context import get_current_run_context
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    FileWindow,
    UnavailableWorkspace,
    Workspace,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceFileEntry,
    WorkspaceRef,
    WorkspaceResult,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

from ._operation import CacheIdentity, WorkspaceMethod
from ._operation_backend import BoundDurableOperation

WorkspaceOperationValue: TypeAlias = CommandResult | str | bytes | FileEntry | list[FileEntry] | bool | FileWindow | None
WorkspaceErrorKind: TypeAlias = Literal[
    'timeout',
    'unavailable',
    'workspace',
    'not_found',
    'not_a_directory',
    'is_a_directory',
    'not_implemented',
]


@dataclass(frozen=True, kw_only=True)
class WorkspaceOperationError:
    kind: WorkspaceErrorKind
    message: str
    stdout: str = ''
    stderr: str = ''
    timeout: float | None = None


@dataclass(frozen=True, kw_only=True)
class WorkspaceOperationParams:
    run_context: RunContext[Any]
    supplier_id: str
    ref: WorkspaceRef | None
    arguments: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class WorkspaceOperationResult:
    value: WorkspaceOperationValue = None
    ref: WorkspaceRef | None = None
    error: WorkspaceOperationError | None = None


class WorkspaceOperationCacheIdentity(CacheIdentity[WorkspaceOperationParams]):
    def project(self, params: WorkspaceOperationParams) -> tuple[object, ...]:
        return (params.supplier_id, params.ref, params.arguments, params.run_context)


class DurableWorkspaceDispatcher:
    """Route one run's user-facing workspace calls through durable operations."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        supplier: AbstractCapability[Any],
        operations: Mapping[WorkspaceMethod, BoundDurableOperation[WorkspaceOperationParams, Any, WorkspaceOperationResult]],
        in_durable_context: Callable[[], bool],
    ) -> None:
        supplier_id = supplier.id
        if supplier_id is None:
            raise UserError(
                f'Capability {type(supplier).__name__!r} supplies a workspace and needs an explicit `id` '
                'because durable workspace operation identity must remain stable.'
            )
        self._workspace = workspace
        self._supplier = supplier
        self._supplier_id = supplier_id
        self._operations = operations
        self._in_durable_context = in_durable_context
        self._ref = workspace._raw_backend().ref  # pyright: ignore[reportPrivateUsage]
        self._first_operation_lock = anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref

    @property
    def backend(self) -> WorkspaceBackend:
        if self._in_durable_context():
            raise UserError(
                '`workspace.backend` is not available in durable workflow code because calling the provider '
                'backend directly would bypass durable execution. Use the `Workspace` methods instead.'
            )
        return self._workspace._raw_backend()  # pyright: ignore[reportPrivateUsage]

    async def __call__(self, method: str, arguments: Mapping[str, Any]) -> Any:
        workspace_method = cast(WorkspaceMethod, method)
        if not self._in_durable_context():
            return await self._call_direct(workspace_method, arguments)

        ctx = get_current_run_context()
        if ctx is None:
            raise RuntimeError('A durable workspace operation requires the current agent run context.')

        if self._ref is None:
            async with self._first_operation_lock:
                if self._ref is None:
                    return await self._dispatch(ctx, workspace_method, arguments)
        return await self._dispatch(ctx, workspace_method, arguments)

    async def _dispatch(self, ctx: RunContext[Any], method: WorkspaceMethod, arguments: Mapping[str, Any]) -> Any:
        operation = self._operations.get(method)
        if operation is None:
            raise UserError(
                f'Workspace method {method!r} was not registered for capability {self._supplier_id!r}. '
                'Attach the workspace capability when constructing the agent.'
            )
        outcome = await operation(
            WorkspaceOperationParams(
                run_context=ctx,
                supplier_id=self._supplier_id,
                ref=self._ref,
                arguments=dict(arguments),
            )
        )
        if outcome.ref is not None and self._ref != outcome.ref:
            self._ref = outcome.ref
            backend = self._supplier.get_workspace(ctx, ref=outcome.ref)
            if backend is None:
                raise RuntimeError(
                    f'Workspace capability {self._supplier_id!r} declined the environment it just created.'
                )
            self._workspace._replace_raw_backend(backend)  # pyright: ignore[reportPrivateUsage]
        if outcome.error is not None:
            _raise_operation_error(outcome.error)
        if outcome.ref is None:
            raise RuntimeError(
                f'Workspace capability {self._supplier_id!r} completed {method!r} without assigning a `WorkspaceRef`.'
            )
        return outcome.value

    async def _call_direct(self, method: WorkspaceMethod, arguments: Mapping[str, Any]) -> Any:
        direct = Workspace(self._workspace._raw_backend())  # pyright: ignore[reportPrivateUsage]
        return await cast(Callable[..., Any], getattr(direct, method))(**arguments)


def normalize_workspace_value(method: WorkspaceMethod, value: Any) -> WorkspaceOperationValue:
    if method == 'run':
        result = cast(WorkspaceResult, value)
        return CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)
    if method == 'stat':
        return _file_entry(cast(WorkspaceFileEntry, value))
    if method == 'list_dir':
        return [_file_entry(entry) for entry in cast(Sequence[WorkspaceFileEntry], value)]
    return cast(WorkspaceOperationValue, value)


def workspace_operation_error(error: BaseException) -> WorkspaceOperationError | None:
    if isinstance(error, WorkspaceTimeoutError):
        return WorkspaceOperationError(
            kind='timeout', message=str(error), stdout=error.stdout, stderr=error.stderr, timeout=error.timeout
        )
    if isinstance(error, WorkspaceUnavailableError):
        return WorkspaceOperationError(kind='unavailable', message=str(error))
    if isinstance(error, WorkspaceError):
        return WorkspaceOperationError(kind='workspace', message=str(error))
    for error_type, kind in (
        (FileNotFoundError, 'not_found'),
        (NotADirectoryError, 'not_a_directory'),
        (IsADirectoryError, 'is_a_directory'),
        (NotImplementedError, 'not_implemented'),
    ):
        if isinstance(error, error_type):
            return WorkspaceOperationError(kind=cast(WorkspaceErrorKind, kind), message=str(error))
    return None


def _raise_operation_error(error: WorkspaceOperationError) -> None:
    if error.kind == 'timeout':
        raise WorkspaceTimeoutError(error.message, stdout=error.stdout, stderr=error.stderr, timeout=error.timeout)
    if error.kind == 'unavailable':
        raise WorkspaceUnavailableError(error.message)
    if error.kind == 'workspace':
        raise WorkspaceError(error.message)
    error_types: dict[WorkspaceErrorKind, type[Exception]] = {
        'not_found': FileNotFoundError,
        'not_a_directory': NotADirectoryError,
        'is_a_directory': IsADirectoryError,
        'not_implemented': NotImplementedError,
    }
    raise error_types[error.kind](error.message)


def _file_entry(entry: WorkspaceFileEntry) -> FileEntry:
    return FileEntry(name=entry.name, path=entry.path, is_dir=entry.is_dir, size=entry.size)


def live_workspace_error(*, run_location: str, workspace_constraint: str) -> str:
    return (
        f'A live workspace handle cannot be passed {run_location}: {workspace_constraint}. '
        'Pass a `WorkspaceRef` instead and attach a capability whose `get_workspace` can supply it.'
    )


def guard_workflow_workspace(
    workspace: WorkspaceBackend | WorkspaceRef | None,
    *,
    live_error: str,
    ref_error: str | None = None,
) -> WorkspaceRef | UnavailableWorkspace | None:
    """Reject a workspace argument an older durable wrapper cannot support safely.

    Live handles never survive serialization. Wrappers that cannot route a reconstructed
    environment through their durable units also provide `ref_error` to reject references.
    """
    if workspace is not None and not isinstance(workspace, (WorkspaceRef, UnavailableWorkspace)):
        raise UserError(live_error)
    if isinstance(workspace, WorkspaceRef) and ref_error is not None:
        raise UserError(ref_error)
    return workspace
