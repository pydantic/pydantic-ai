"""Workspace operations as durable units.

Inside a durable container, [`RunContext.workspace`][pydantic_ai.tools.RunContext.workspace] is a
[`DurableWorkspace`][pydantic_ai.durable_exec._workspace.DurableWorkspace]: a wrapper installed
innermost around the run's selected workspace whose every operation runs in its own durable unit,
so workflow-side code (capability hooks, output functions, `result.workspace`) never performs
workspace I/O in the container. Inside a unit (a tool activity, a `@durable_operation` hook)
`ctx.workspace` is the plain facade and calls reach the backend directly.

One `ensure` unit runs at the start of every run in a container. It forces the environment to
exist, and journals its [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] and canonical
working directory. From then on every unit carries the same ref, so parallel tools, retries,
replay and recovery all reattach to one environment, and `working_dir()` and `resolve()` are
answered locally from the journaled value.

Parameters and results are pydantic dataclasses with `bytes` encoded as base64, which is the one
shape that survives Temporal's payload converter, `JSON_CODEC` and pickle unchanged. Errors a
workspace is expected to raise cross as data and are re-raised as the same types on the other
side, so a durable unit only fails for infrastructure errors the engine should retry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeAlias, TypeVar, cast

import anyio
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import Never

from pydantic_ai._run_context import get_current_run_context
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    FileWindow,
    UnavailableWorkspace,
    Workspace,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceFileEntry,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
    WrapperWorkspace,
)
from pydantic_ai.workspaces.workspace import (
    _DEFAULT_READ_BYTES,  # pyright: ignore[reportPrivateUsage]
    _DEFAULT_READ_LINES,  # pyright: ignore[reportPrivateUsage]
)

from ._operation import CacheIdentity, WorkspaceMethod, WorkspaceOperationId
from ._operation_backend import BoundDurableOperation, in_durable_unit

if TYPE_CHECKING:
    from ._base import BaseDurabilityCapability

__all__ = (
    'DurableWorkspace',
    'WorkspaceMethod',
    'WorkspaceOperationId',
    'WorkspaceOperationParams',
    'WorkspaceOperationResult',
)

_BYTES_CONFIG = ConfigDict(ser_json_bytes='base64', val_json_bytes='base64')
"""Serialize `bytes` fields as base64 text.

Temporal's payload converter serializes by runtime type and ignores field annotations, so a
`Base64Bytes` annotation never sees non-UTF-8 data; a dataclass-level config does.
"""

ValueT = TypeVar('ValueT')
ValueT_co = TypeVar('ValueT_co', covariant=True)


@dataclass(frozen=True, kw_only=True)
class WorkspaceOperationParams(Generic[ValueT]):
    """Semantic parameters of a workspace unit: the run context, the journaled ref, and the call."""

    run_context: RunContext[Any]
    ref: WorkspaceRef | None
    arguments: WorkspaceArguments[ValueT]


class WorkspaceArguments(Protocol[ValueT_co]):
    """The typed arguments of one workspace call, which know how to make that call."""

    @property
    def method(self) -> WorkspaceMethod | Literal['ensure']: ...

    async def call(self, workspace: Workspace) -> ValueT_co: ...


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class EnsuredWorkspace:
    """What the `ensure` unit journals: the environment's identity and its canonical working directory."""

    ref: WorkspaceRef
    working_dir: str


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class EnsureArguments:
    @property
    def method(self) -> Literal['ensure']:
        return 'ensure'

    async def call(self, workspace: Workspace) -> EnsuredWorkspace:
        # `working_dir()` is the one operation every backend has, and it creates or attaches. The
        # ref is read afterwards because a fresh environment only gets one from its provider.
        working_dir = await workspace.working_dir()
        ref = workspace.ref
        if ref is None:
            raise UserError(
                'The workspace backend completed an operation without reporting a `WorkspaceRef`. A backend '
                'used under durable execution must report a `ref` once any operation has completed, so that '
                'every durable unit can reattach to the same environment.'
            )
        return EnsuredWorkspace(ref=ref, working_dir=working_dir)


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class RunArguments:
    @property
    def method(self) -> Literal['run']:
        return 'run'

    command: WorkspaceCommand
    shell: bool = False
    cwd: str | None = None
    env: Mapping[str, str] | None = None
    timeout: float | None = None

    async def call(self, workspace: Workspace) -> CommandResult:
        result = await workspace.run(self.command, shell=self.shell, cwd=self.cwd, env=self.env, timeout=self.timeout)
        return CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class ReadBytesArguments:
    @property
    def method(self) -> Literal['read_bytes']:
        return 'read_bytes'

    path: str

    async def call(self, workspace: Workspace) -> bytes:
        return await workspace.read_bytes(self.path)


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class WriteBytesArguments:
    @property
    def method(self) -> Literal['write_bytes']:
        return 'write_bytes'

    path: str
    data: bytes

    async def call(self, workspace: Workspace) -> None:
        await workspace.write_bytes(self.path, self.data)


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class StatArguments:
    @property
    def method(self) -> Literal['stat']:
        return 'stat'

    path: str

    async def call(self, workspace: Workspace) -> FileEntry:
        return _file_entry(await workspace.stat(self.path))


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class ListDirArguments:
    @property
    def method(self) -> Literal['list_dir']:
        return 'list_dir'

    path: str

    async def call(self, workspace: Workspace) -> list[FileEntry]:
        return [_file_entry(entry) for entry in await workspace.list_dir(self.path)]


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class MakeDirArguments:
    @property
    def method(self) -> Literal['make_dir']:
        return 'make_dir'

    path: str

    async def call(self, workspace: Workspace) -> None:
        await workspace.make_dir(self.path)


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class RemoveArguments:
    @property
    def method(self) -> Literal['remove']:
        return 'remove'

    path: str

    async def call(self, workspace: Workspace) -> None:
        await workspace.remove(self.path)


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class ExistsArguments:
    @property
    def method(self) -> Literal['exists']:
        return 'exists'

    path: str

    async def call(self, workspace: Workspace) -> bool:
        return await workspace.exists(self.path)


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class ReadTextArguments:
    @property
    def method(self) -> Literal['read_text']:
        return 'read_text'

    path: str
    encoding: str = 'utf-8'

    async def call(self, workspace: Workspace) -> str:
        return await workspace.read_text(self.path, encoding=self.encoding)


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class WriteTextArguments:
    @property
    def method(self) -> Literal['write_text']:
        return 'write_text'

    path: str
    content: str
    encoding: str = 'utf-8'

    async def call(self, workspace: Workspace) -> None:
        await workspace.write_text(self.path, self.content, encoding=self.encoding)


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class ReadFileArguments:
    @property
    def method(self) -> Literal['read_file']:
        return 'read_file'

    path: str
    offset: int
    limit: int | None
    max_bytes: int | None

    async def call(self, workspace: Workspace) -> FileWindow:
        return await workspace.read_file(self.path, offset=self.offset, limit=self.limit, max_bytes=self.max_bytes)


def _file_entry(entry: WorkspaceFileEntry) -> FileEntry:
    return FileEntry(name=entry.name, path=entry.path, is_dir=entry.is_dir, size=entry.size)


@dataclass(frozen=True)
class WorkspaceOperationSpec:
    """The static shape of one workspace unit: its id and the argument and result types it carries."""

    operation_id: WorkspaceOperationId
    arguments_type: type[Any]
    result_type: object
    """The `WorkspaceOperationResult[...]` type form of the unit's result."""


WorkspaceErrorKind: TypeAlias = Literal[
    'timeout',
    'unavailable',
    'workspace',
    'not_found',
    'not_a_directory',
    'is_a_directory',
    'permission',
    'file_exists',
    'unicode_decode',
    'not_implemented',
    'user',
    'type',
    'value',
]


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class UnicodeDecodeDetails:
    """The constructor arguments of a `UnicodeDecodeError`, which carries the undecodable bytes."""

    encoding: str
    object: bytes
    start: int
    end: int
    reason: str


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class WorkspaceOperationError:
    """An expected workspace failure, carried as data so the unit succeeds and the caller re-raises it."""

    kind: WorkspaceErrorKind
    message: str
    stdout: str = ''
    stderr: str = ''
    timeout: float | None = None
    decode: UnicodeDecodeDetails | None = None


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class WorkspaceOperationResult(Generic[ValueT]):
    """The journaled outcome of a workspace unit: the call's value, or the error it raised."""

    value: ValueT | None = None
    error: WorkspaceOperationError | None = None


_ERROR_TYPES: dict[WorkspaceErrorKind, type[Exception]] = {
    'not_found': FileNotFoundError,
    'not_a_directory': NotADirectoryError,
    'is_a_directory': IsADirectoryError,
    'permission': PermissionError,
    'file_exists': FileExistsError,
    'not_implemented': NotImplementedError,
    'user': UserError,
    'type': TypeError,
    'value': ValueError,
}
"""Error kinds that rebuild from their message alone. The others need extra fields; see `raise_operation_error`."""


def workspace_operation_error(error: Exception) -> WorkspaceOperationError | None:
    """Map an exception a workspace is expected to raise to its data form, or `None` for anything else.

    Subclasses are checked before their bases: a `WorkspaceTimeoutError` is also a `WorkspaceError`
    and a `UnicodeDecodeError` is also a `ValueError`. `WorkspaceUnavailableError` crosses as data
    too: retrying the same unit cannot succeed, so the caller must get to decide, not the engine.
    """
    if isinstance(error, WorkspaceTimeoutError):
        return WorkspaceOperationError(
            kind='timeout', message=str(error), stdout=error.stdout, stderr=error.stderr, timeout=error.timeout
        )
    if isinstance(error, WorkspaceUnavailableError):
        return WorkspaceOperationError(kind='unavailable', message=str(error))
    if isinstance(error, WorkspaceError):
        return WorkspaceOperationError(kind='workspace', message=str(error))
    if isinstance(error, UnicodeDecodeError):
        return WorkspaceOperationError(
            kind='unicode_decode',
            message=str(error),
            decode=UnicodeDecodeDetails(
                encoding=error.encoding, object=error.object, start=error.start, end=error.end, reason=error.reason
            ),
        )
    for kind, error_type in _ERROR_TYPES.items():
        if isinstance(error, error_type):
            return WorkspaceOperationError(kind=kind, message=str(error))
    return None


def raise_operation_error(error: WorkspaceOperationError) -> Never:
    """Re-raise a workspace failure that crossed a durable boundary as data, as its original type."""
    if error.kind == 'timeout':
        raise WorkspaceTimeoutError(error.message, stdout=error.stdout, stderr=error.stderr, timeout=error.timeout)
    if error.kind == 'unavailable':
        raise WorkspaceUnavailableError(error.message)
    if error.kind == 'workspace':
        raise WorkspaceError(error.message)
    if error.kind == 'unicode_decode':
        details = error.decode
        assert details is not None
        raise UnicodeDecodeError(details.encoding, details.object, details.start, details.end, details.reason)
    raise _ERROR_TYPES[error.kind](error.message)


async def execute_workspace_operation(
    workspace: Workspace, arguments: WorkspaceArguments[ValueT]
) -> WorkspaceOperationResult[ValueT]:
    """Run one workspace call inside a durable unit, capturing expected failures as data.

    Anything not in the error table propagates and fails the unit, so a provider's own transient
    errors get the engine's retry policy while a missing file does not.
    """
    try:
        value = await arguments.call(workspace)
    except Exception as error:
        operation_error = workspace_operation_error(error)
        if operation_error is None:
            raise
        return WorkspaceOperationResult(error=operation_error)
    return WorkspaceOperationResult(value=value)


class WorkspaceCacheIdentity(CacheIdentity[WorkspaceOperationParams[Any]]):
    """Project the call and the environment it targets; hash-keyed engines add their own sequence."""

    def project(self, params: WorkspaceOperationParams[Any]) -> tuple[object, ...]:
        return (params.arguments, params.ref)


WORKSPACE_OPERATIONS: tuple[WorkspaceOperationSpec, ...] = (
    WorkspaceOperationSpec(WorkspaceOperationId('ensure'), EnsureArguments, WorkspaceOperationResult[EnsuredWorkspace]),
    WorkspaceOperationSpec(WorkspaceOperationId('run'), RunArguments, WorkspaceOperationResult[CommandResult]),
    WorkspaceOperationSpec(WorkspaceOperationId('read_bytes'), ReadBytesArguments, WorkspaceOperationResult[bytes]),
    WorkspaceOperationSpec(WorkspaceOperationId('write_bytes'), WriteBytesArguments, WorkspaceOperationResult[None]),
    WorkspaceOperationSpec(WorkspaceOperationId('stat'), StatArguments, WorkspaceOperationResult[FileEntry]),
    WorkspaceOperationSpec(
        WorkspaceOperationId('list_dir'), ListDirArguments, WorkspaceOperationResult[list[FileEntry]]
    ),
    WorkspaceOperationSpec(WorkspaceOperationId('make_dir'), MakeDirArguments, WorkspaceOperationResult[None]),
    WorkspaceOperationSpec(WorkspaceOperationId('remove'), RemoveArguments, WorkspaceOperationResult[None]),
    WorkspaceOperationSpec(WorkspaceOperationId('exists'), ExistsArguments, WorkspaceOperationResult[bool]),
    WorkspaceOperationSpec(WorkspaceOperationId('read_text'), ReadTextArguments, WorkspaceOperationResult[str]),
    WorkspaceOperationSpec(WorkspaceOperationId('write_text'), WriteTextArguments, WorkspaceOperationResult[None]),
    WorkspaceOperationSpec(WorkspaceOperationId('read_file'), ReadFileArguments, WorkspaceOperationResult[FileWindow]),
)
"""Every workspace unit a durability capability binds, `ensure` first."""

MUTATING_WORKSPACE_METHODS: frozenset[WorkspaceMethod] = frozenset(
    {'run', 'write_bytes', 'write_text', 'make_dir', 'remove'}
)
"""Methods whose unit an engine should attempt once by default: a retry would repeat the side effect."""

WorkspaceBoundOperation: TypeAlias = BoundDurableOperation[
    WorkspaceOperationParams[Any], Any, WorkspaceOperationResult[Any]
]


def attached_workspace(workspace: Workspace) -> bool:
    """Whether `workspace` reaches a real backend rather than the unattached placeholder.

    Walks wrappers through `wrapped` rather than `backend`, which a `DurableWorkspace` refuses to
    answer inside a durable container.
    """
    while isinstance(workspace, WrapperWorkspace):
        workspace = workspace.wrapped
    return not isinstance(workspace.backend, UnavailableWorkspace)


def resolve_run_workspace(
    capability: AbstractCapability[Any], ctx: RunContext[Any], ref: WorkspaceRef | None
) -> Workspace | None:
    """Rebuild the run's workspace from the capability tree, the way the agent selects it.

    A bare backend is wrapped in a `Workspace`; a facade or wrapper is returned as is, which is how
    policy such as `ReadOnlyWorkspace` comes back on every side of a durable boundary.
    """
    selection = capability.get_workspace(ctx, ref=ref)
    if selection is None:
        return None
    return selection if isinstance(selection, Workspace) else Workspace(selection)


class DurableWorkspace(WrapperWorkspace):
    """The run's workspace inside a durable container: each operation is its own durable unit.

    Installed by a durability capability's `_wrap_workspace` hook, innermost around the workspace
    the run selected. Outside the container, and inside a durable unit, calls go straight to the
    wrapped workspace. The first dispatched call runs `ensure` if the run's companion capability
    has not already, so a capability whose `wrap_run` touches the workspace before the run body
    still gets one environment.

    [`ref`][pydantic_ai.workspaces.Workspace.ref] and
    [`working_dir`][pydantic_ai.workspaces.Workspace.working_dir] report the values `ensure`
    journaled, which are deterministic under replay; `working_dir` is fixed for the life of a
    workspace, so answering it locally loses nothing and saves hooks that resolve paths a unit each.
    """

    def __init__(self, wrapped: Workspace, *, durability: BaseDurabilityCapability[Any], ctx: RunContext[Any]) -> None:
        super().__init__(wrapped)
        self._durability = durability
        self._ctx = ctx
        self._ref: WorkspaceRef | None = None
        self._working_dir: str | None = None
        self._ensured = False
        self._ensure_lock = anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref if self._ref is not None else self.wrapped.ref

    @property
    def backend(self) -> Workspace:
        if self._in_container():
            raise UserError(
                '`workspace.backend` is not available in durable workflow code: calling the provider backend '
                f'directly would bypass the {self._durability.engine_name} {self._durability.durable_unit_plural} '
                'that make workspace operations durable. Use the `Workspace` methods instead, or reach the '
                f'backend from a tool, which runs inside {self._durability.durable_unit_noun}.'
            )
        return self.wrapped

    def _in_container(self) -> bool:
        return self._durability.in_durable_context and not in_durable_unit()

    async def working_dir(self) -> str:
        if self._working_dir is not None:
            return self._working_dir
        if not self._in_container():
            return await self.wrapped.working_dir()
        await self._ensure()
        assert self._working_dir is not None
        return self._working_dir

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if not self._in_container():
            result = await self.wrapped.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
            return CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)
        return await self._dispatch(RunArguments(command=command, shell=shell, cwd=cwd, env=env, timeout=timeout))

    async def read_bytes(self, path: str) -> bytes:
        if not self._in_container():
            return await self.wrapped.read_bytes(path)
        return await self._dispatch(ReadBytesArguments(path=path))

    async def write_bytes(self, path: str, data: bytes) -> None:
        if not self._in_container():
            return await self.wrapped.write_bytes(path, data)
        await self._dispatch(WriteBytesArguments(path=path, data=data))

    async def stat(self, path: str) -> WorkspaceFileEntry:
        if not self._in_container():
            return await self.wrapped.stat(path)
        return await self._dispatch(StatArguments(path=path))

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        if not self._in_container():
            return await self.wrapped.list_dir(path)
        return await self._dispatch(ListDirArguments(path=path))

    async def make_dir(self, path: str) -> None:
        if not self._in_container():
            return await self.wrapped.make_dir(path)
        await self._dispatch(MakeDirArguments(path=path))

    async def remove(self, path: str) -> None:
        if not self._in_container():
            return await self.wrapped.remove(path)
        await self._dispatch(RemoveArguments(path=path))

    async def exists(self, path: str) -> bool:
        if not self._in_container():
            return await self.wrapped.exists(path)
        return await self._dispatch(ExistsArguments(path=path))

    async def read_text(self, path: str, *, encoding: str = 'utf-8') -> str:
        if not self._in_container():
            return await self.wrapped.read_text(path, encoding=encoding)
        return await self._dispatch(ReadTextArguments(path=path, encoding=encoding))

    async def write_text(self, path: str, content: str, *, encoding: str = 'utf-8') -> None:
        if not self._in_container():
            return await self.wrapped.write_text(path, content, encoding=encoding)
        await self._dispatch(WriteTextArguments(path=path, content=content, encoding=encoding))

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 1,
        limit: int | None = _DEFAULT_READ_LINES,
        max_bytes: int | None = _DEFAULT_READ_BYTES,
    ) -> FileWindow:
        # Overridden rather than inherited: the base implementation windows a file it has read
        # through `read_bytes`, which would ship the whole file through a unit. The unit windows
        # it inside the workspace, so only the window crosses.
        if not self._in_container():
            return await self.wrapped.read_file(path, offset=offset, limit=limit, max_bytes=max_bytes)
        return await self._dispatch(ReadFileArguments(path=path, offset=offset, limit=limit, max_bytes=max_bytes))

    async def _dispatch(self, arguments: WorkspaceArguments[ValueT]) -> ValueT:
        await self._ensure()
        # The ambient context is only set around `before_run` and model requests; elsewhere (tool
        # hooks, `after_run`, `result.workspace` after the run) the run's own context serves.
        ctx = get_current_run_context() or self._ctx
        operation = self._durability._workspace_operation(arguments.method)  # pyright: ignore[reportPrivateUsage]
        config = self._durability._workspace_operation_config(operation, arguments)  # pyright: ignore[reportPrivateUsage]
        result = await operation(
            WorkspaceOperationParams(run_context=ctx, ref=self._ref, arguments=arguments), config=config
        )
        if result.error is not None:
            raise_operation_error(result.error)
        # The bound operation is stored untyped alongside the other methods'; the arguments'
        # `call` signature is what fixes the value type.
        return cast(ValueT, result.value)

    async def _ensure(self, ctx: RunContext[Any] | None = None) -> None:
        """Run the `ensure` unit once for the run, and rebuild the wrapped workspace on its ref.

        Every dispatched operation calls this first, under a lock, so parallel first uses share one
        environment even when the companion capability's eager call did not come first.
        """
        if self._ensured:
            return
        async with self._ensure_lock:
            if self._ensured:
                return
            if ctx is not None:
                self._ctx = ctx
            ctx = get_current_run_context() or self._ctx
            arguments = EnsureArguments()
            operation = self._durability._workspace_operation(arguments.method)  # pyright: ignore[reportPrivateUsage]
            result = await operation(
                WorkspaceOperationParams(run_context=ctx, ref=self.wrapped.ref, arguments=arguments)
            )
            if result.error is not None:
                raise_operation_error(result.error)
            ensured = cast(EnsuredWorkspace, result.value)
            if self.wrapped.ref != ensured.ref:
                # A fresh environment: rebuild the selection on its identity, so this side attaches
                # to what the unit created instead of creating another on first use.
                root_capability = ctx.root_capability
                assert root_capability is not None
                rebuilt = resolve_run_workspace(root_capability, ctx, ensured.ref)
                if rebuilt is None:
                    raise UserError(
                        f'No capability can supply workspace {ensured.ref.id!r} from provider '
                        f'{ensured.ref.provider!r}, which the run just created. A `get_workspace` hook that '
                        'creates an environment must also recognize its ref.'
                    )
                self._backend = rebuilt
            self._ref = ensured.ref
            self._working_dir = ensured.working_dir
            self._ensured = True


class WorkspaceEnsurer(AbstractCapability[AgentDepsT]):
    """Runs the `ensure` unit before the run body, from the `outermost` tier.

    A durability capability is `innermost`, so its own `wrap_run` runs after every other
    capability's pre-handler code; those hooks would otherwise dispatch the first workspace unit
    themselves through the lazy fallback. The durability capability's `for_agent` composes this
    companion around itself, the way `TemporalDurability` pairs its terminal-event publisher.
    """

    def __init__(self, durability: BaseDurabilityCapability[AgentDepsT]) -> None:
        self._durability = durability

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        workspace = ctx.workspace
        if isinstance(workspace, DurableWorkspace):
            await workspace._ensure(ctx)  # pyright: ignore[reportPrivateUsage]
        return await handler()


class RejectWorkspaceInContainer(AbstractCapability[AgentDepsT]):
    """Refuse a real workspace for an agent used through a deprecated wrapper agent inside its container.

    The wrapper agents have no durability capability, so nothing would route workspace operations
    through durable units: a hook would do provider I/O in workflow code, and a tool activity would
    create a new environment per call. Added per-run by the wrapper agents' `iter`.
    """

    _safe_at_runtime = True

    def __init__(self, *, engine: str, container_noun: str, capability: str) -> None:
        self._engine = engine
        self._container_noun = container_noun
        self._capability = capability

    def _wrap_workspace(self, ctx: RunContext[AgentDepsT], workspace: Workspace, *, explicit: bool) -> Workspace:
        if attached_workspace(workspace):
            raise UserError(
                f'Workspaces are not supported inside a {self._engine} {self._container_noun} through the deprecated '
                f'wrapper agent. Use `Agent(..., capabilities=[{self._capability}()])`, which runs every workspace '
                f'operation as a durable unit, and attach the workspace through a capability such as `LocalWorkspace`.'
            )
        return workspace
