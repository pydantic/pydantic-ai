from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import anyio

from pydantic_ai._run_context import get_current_run_context
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.sandboxes import (
    CommandResult,
    FileEntry,
    FileWindow,
    Sandbox,
    SandboxBackend,
    SandboxError,
    SandboxFileEntry,
    SandboxRef,
    SandboxResult,
    SandboxTimeoutError,
    SandboxUnavailableError,
    UnavailableSandbox,
)
from pydantic_ai.tools import RunContext

from ._operation import CacheIdentity, SandboxMethod
from ._operation_backend import BoundDurableOperation

SandboxOperationValue: TypeAlias = CommandResult | str | bytes | FileEntry | list[FileEntry] | bool | FileWindow | None
SandboxErrorKind: TypeAlias = Literal[
    'timeout',
    'unavailable',
    'sandbox',
    'not_found',
    'not_a_directory',
    'is_a_directory',
    'not_implemented',
]


@dataclass(frozen=True, kw_only=True)
class SandboxOperationError:
    kind: SandboxErrorKind
    message: str
    stdout: str = ''
    stderr: str = ''
    timeout: float | None = None


@dataclass(frozen=True, kw_only=True)
class SandboxOperationParams:
    run_context: RunContext[Any]
    supplier_id: str
    ref: SandboxRef | None
    arguments: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class SandboxOperationResult:
    value: SandboxOperationValue = None
    ref: SandboxRef | None = None
    error: SandboxOperationError | None = None


class SandboxOperationCacheIdentity(CacheIdentity[SandboxOperationParams]):
    def project(self, params: SandboxOperationParams) -> tuple[object, ...]:
        return (params.supplier_id, params.ref, params.arguments, params.run_context)


class DurableSandboxDispatcher:
    """Route one run's user-facing sandbox calls through durable operations."""

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        supplier: AbstractCapability[Any],
        operations: Mapping[SandboxMethod, BoundDurableOperation[SandboxOperationParams, Any, SandboxOperationResult]],
        in_durable_context: Callable[[], bool],
    ) -> None:
        supplier_id = supplier.id
        if supplier_id is None:
            raise UserError(
                f'Capability {type(supplier).__name__!r} supplies a sandbox and needs an explicit `id` '
                'because durable sandbox operation identity must remain stable.'
            )
        self._sandbox = sandbox
        self._supplier = supplier
        self._supplier_id = supplier_id
        self._operations = operations
        self._in_durable_context = in_durable_context
        self._ref = sandbox._raw_backend().ref  # pyright: ignore[reportPrivateUsage]
        self._first_operation_lock = anyio.Lock()

    @property
    def ref(self) -> SandboxRef | None:
        return self._ref

    @property
    def backend(self) -> SandboxBackend:
        if self._in_durable_context():
            raise UserError(
                '`sandbox.backend` is not available in durable workflow code because calling the provider '
                'backend directly would bypass durable execution. Use the `Sandbox` methods instead.'
            )
        return self._sandbox._raw_backend()  # pyright: ignore[reportPrivateUsage]

    async def __call__(self, method: str, arguments: Mapping[str, Any]) -> Any:
        sandbox_method = cast(SandboxMethod, method)
        if not self._in_durable_context():
            return await self._call_direct(sandbox_method, arguments)

        ctx = get_current_run_context()
        if ctx is None:
            raise RuntimeError('A durable sandbox operation requires the current agent run context.')

        if self._ref is None:
            async with self._first_operation_lock:
                if self._ref is None:
                    return await self._dispatch(ctx, sandbox_method, arguments)
        return await self._dispatch(ctx, sandbox_method, arguments)

    async def _dispatch(self, ctx: RunContext[Any], method: SandboxMethod, arguments: Mapping[str, Any]) -> Any:
        operation = self._operations.get(method)
        if operation is None:
            raise UserError(
                f'Sandbox method {method!r} was not registered for capability {self._supplier_id!r}. '
                'Attach the sandbox capability when constructing the agent.'
            )
        outcome = await operation(
            SandboxOperationParams(
                run_context=ctx,
                supplier_id=self._supplier_id,
                ref=self._ref,
                arguments=dict(arguments),
            )
        )
        if outcome.ref is not None and self._ref != outcome.ref:
            self._ref = outcome.ref
            backend = self._supplier.get_sandbox(ctx, ref=outcome.ref)
            if backend is None:
                raise RuntimeError(
                    f'Sandbox capability {self._supplier_id!r} declined the environment it just created.'
                )
            self._sandbox._replace_raw_backend(backend)  # pyright: ignore[reportPrivateUsage]
        if outcome.error is not None:
            _raise_operation_error(outcome.error)
        if outcome.ref is None:
            raise RuntimeError(
                f'Sandbox capability {self._supplier_id!r} completed {method!r} without assigning a `SandboxRef`.'
            )
        return outcome.value

    async def _call_direct(self, method: SandboxMethod, arguments: Mapping[str, Any]) -> Any:
        direct = Sandbox(self._sandbox._raw_backend())  # pyright: ignore[reportPrivateUsage]
        return await cast(Callable[..., Any], getattr(direct, method))(**arguments)


def normalize_sandbox_value(method: SandboxMethod, value: Any) -> SandboxOperationValue:
    if method == 'run':
        result = cast(SandboxResult, value)
        return CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)
    if method == 'stat':
        return _file_entry(cast(SandboxFileEntry, value))
    if method == 'list_dir':
        return [_file_entry(entry) for entry in cast(Sequence[SandboxFileEntry], value)]
    return cast(SandboxOperationValue, value)


def sandbox_operation_error(error: BaseException) -> SandboxOperationError | None:
    if isinstance(error, SandboxTimeoutError):
        return SandboxOperationError(
            kind='timeout', message=str(error), stdout=error.stdout, stderr=error.stderr, timeout=error.timeout
        )
    if isinstance(error, SandboxUnavailableError):
        return SandboxOperationError(kind='unavailable', message=str(error))
    if isinstance(error, SandboxError):
        return SandboxOperationError(kind='sandbox', message=str(error))
    for error_type, kind in (
        (FileNotFoundError, 'not_found'),
        (NotADirectoryError, 'not_a_directory'),
        (IsADirectoryError, 'is_a_directory'),
        (NotImplementedError, 'not_implemented'),
    ):
        if isinstance(error, error_type):
            return SandboxOperationError(kind=cast(SandboxErrorKind, kind), message=str(error))
    return None


def _raise_operation_error(error: SandboxOperationError) -> None:
    if error.kind == 'timeout':
        raise SandboxTimeoutError(error.message, stdout=error.stdout, stderr=error.stderr, timeout=error.timeout)
    if error.kind == 'unavailable':
        raise SandboxUnavailableError(error.message)
    if error.kind == 'sandbox':
        raise SandboxError(error.message)
    error_types: dict[SandboxErrorKind, type[Exception]] = {
        'not_found': FileNotFoundError,
        'not_a_directory': NotADirectoryError,
        'is_a_directory': IsADirectoryError,
        'not_implemented': NotImplementedError,
    }
    raise error_types[error.kind](error.message)


def _file_entry(entry: SandboxFileEntry) -> FileEntry:
    return FileEntry(name=entry.name, path=entry.path, is_dir=entry.is_dir, size=entry.size)


def live_sandbox_error(*, run_location: str, sandbox_constraint: str) -> str:
    return (
        f'A live sandbox handle cannot be passed {run_location}: {sandbox_constraint}. '
        'Pass a `SandboxRef` instead and attach a capability whose `get_sandbox` can supply it.'
    )


def guard_workflow_sandbox(
    sandbox: SandboxBackend | SandboxRef | None,
    *,
    live_error: str,
    ref_error: str | None = None,
) -> SandboxRef | UnavailableSandbox | None:
    """Reject a sandbox argument an older durable wrapper cannot support safely.

    Live handles never survive serialization. Wrappers that cannot route a reconstructed
    environment through their durable units also provide `ref_error` to reject references.
    """
    if sandbox is not None and not isinstance(sandbox, (SandboxRef, UnavailableSandbox)):
        raise UserError(live_error)
    if isinstance(sandbox, SandboxRef) and ref_error is not None:
        raise UserError(ref_error)
    return sandbox
