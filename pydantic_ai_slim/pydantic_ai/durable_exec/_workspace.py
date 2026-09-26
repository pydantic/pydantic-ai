"""Workspace calls as one durable operation.

Inside a durable container, [`RunContext.workspace`][pydantic_ai.tools.RunContext.workspace] is a
`DurableWorkspace`: every call made in workflow code (capability hooks, output functions,
`result.workspace`) runs as the durability capability's workspace operation, so it is journaled and
never repeated on replay. Inside a durable unit (a tool, a `@durable_operation`) calls go straight to
the workspace, rebuilt from the run's `WorkspaceRef`.

An `ensure` call runs first, once per run: it creates or attaches to the environment and journals its
ref and working directory, so every later unit, parallel tools included, reattaches to that one
environment. Errors a workspace is expected to raise cross as data and are re-raised with their
original type, so a hook can catch them and the engine only retries infrastructure failures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import anyio
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import Never, assert_never

from pydantic_ai._run_context import get_current_run_context
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    Workspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceFileEntry,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceResult,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
    WrapperWorkspace,
)

from ._operation import CacheIdentity, CapabilityOperationId
from ._operation_backend import in_durable_unit

if TYPE_CHECKING:
    from ._base import BaseDurabilityCapability

__all__ = ('DurableWorkspace',)

WORKSPACE_OPERATION_ID = CapabilityOperationId('workspace', operation='call')
"""The persisted identity of the workspace operation, named and configured like a capability operation."""

# Base64 for `bytes`: Temporal's payload converter serializes by runtime type, ignoring annotations.
_BYTES_CONFIG = ConfigDict(ser_json_bytes='base64', val_json_bytes='base64')

WorkspaceMethod: TypeAlias = Literal[
    'ensure', 'run', 'read_bytes', 'write_bytes', 'stat', 'list_dir', 'make_dir', 'remove', 'exists', 'realpath'
]


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class WorkspaceCallError:
    """An expected workspace error, carried as data so the unit succeeds and the caller re-raises it."""

    type: str
    message: str
    stdout: str = ''
    stderr: str = ''


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class WorkspaceCallResult:
    """The journaled outcome of a call: the field for the method's return shape, or the error."""

    data: bytes = b''
    text: str = ''
    flag: bool = False
    entries: list[FileEntry] = field(default_factory=list[FileEntry])
    command: CommandResult | None = None
    ref: WorkspaceRef | None = None
    error: WorkspaceCallError | None = None


def _entry(entry: WorkspaceFileEntry) -> FileEntry:
    return FileEntry(name=entry.name, path=entry.path, is_dir=entry.is_dir, size=entry.size)


@pydantic_dataclass(frozen=True, kw_only=True, config=_BYTES_CONFIG)
class WorkspaceCall:
    """One `Workspace` method call: the method and the arguments it takes."""

    method: WorkspaceMethod
    path: str = ''
    data: bytes = b''
    command: WorkspaceCommand = ''
    shell: bool = False
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout: float | None = None

    async def execute(self, workspace: Workspace) -> WorkspaceCallResult:
        match self.method:
            case 'ensure':
                # `working_dir()` is the one operation every backend has, and it creates or attaches;
                # the ref is read afterwards because a fresh environment only then has one.
                working_dir = await workspace.working_dir()
                if workspace.ref is None:
                    raise UserError(
                        'The workspace backend completed an operation without reporting a `WorkspaceRef`. Under '
                        'durable execution a backend must report its `ref` once an operation has completed, so '
                        'every durable unit can reattach to the same environment.'
                    )
                return WorkspaceCallResult(text=working_dir, ref=workspace.ref)
            case 'run':
                result = await workspace.run(
                    self.command, shell=self.shell, cwd=self.cwd, env=self.env, timeout=self.timeout
                )
                return WorkspaceCallResult(
                    command=CommandResult(exit_code=result.exit_code, stdout=result.stdout, stderr=result.stderr)
                )
            case 'read_bytes':
                return WorkspaceCallResult(data=await workspace.read_bytes(self.path))
            case 'write_bytes':
                await workspace.write_bytes(self.path, self.data)
                return WorkspaceCallResult()
            case 'stat':
                return WorkspaceCallResult(entries=[_entry(await workspace.stat(self.path))])
            case 'list_dir':
                return WorkspaceCallResult(entries=[_entry(entry) for entry in await workspace.list_dir(self.path)])
            case 'make_dir':
                await workspace.make_dir(self.path)
                return WorkspaceCallResult()
            case 'remove':
                await workspace.remove(self.path)
                return WorkspaceCallResult()
            case 'exists':
                return WorkspaceCallResult(flag=await workspace.exists(self.path))
            case 'realpath':
                return WorkspaceCallResult(text=await workspace.realpath(self.path))
        assert_never(self.method)


@dataclass(frozen=True)
class WorkspaceCallParams:
    """The workspace operation's parameters: the run context, the environment's ref, and the call."""

    run_context: RunContext[Any]
    ref: WorkspaceRef | None
    call: WorkspaceCall


class WorkspaceCallCacheIdentity(CacheIdentity[WorkspaceCallParams]):
    def project(self, params: WorkspaceCallParams) -> tuple[object, ...]:
        return (params.call, params.ref)


_EXPECTED_ERRORS: tuple[type[Exception], ...] = (
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
    WorkspaceReadOnlyError,
    WorkspaceError,
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    PermissionError,
    FileExistsError,
    NotImplementedError,
    UserError,
    TypeError,
    ValueError,
)
"""Subclasses before their bases: an error crosses as the first of these it is an instance of."""
_EXPECTED_ERRORS_BY_NAME = {error_type.__name__: error_type for error_type in _EXPECTED_ERRORS}


def error_as_data(error: Exception) -> WorkspaceCallError | None:
    """The data form of an error a workspace is expected to raise, or `None` for anything else."""
    error_type = next((error_type for error_type in _EXPECTED_ERRORS if isinstance(error, error_type)), None)
    if error_type is None:
        return None
    if isinstance(error, WorkspaceTimeoutError):
        return WorkspaceCallError(
            type=error_type.__name__, message=str(error), stdout=error.stdout, stderr=error.stderr
        )
    return WorkspaceCallError(type=error_type.__name__, message=str(error))


def raise_error(error: WorkspaceCallError) -> Never:
    """Re-raise an error that crossed a durable boundary as data, with its original type."""
    error_type = _EXPECTED_ERRORS_BY_NAME[error.type]
    if error_type is WorkspaceTimeoutError:
        raise WorkspaceTimeoutError(error.message, stdout=error.stdout, stderr=error.stderr)
    raise error_type(error.message)


async def execute_call(workspace: Workspace, call: WorkspaceCall) -> WorkspaceCallResult:
    """Run a call inside a durable unit; anything outside the error table fails the unit, so the engine retries it."""
    try:
        return await call.execute(workspace)
    except Exception as error:
        if (data := error_as_data(error)) is None:
            raise
        return WorkspaceCallResult(error=data)


def select_workspace(
    capability: AbstractCapability[Any], ctx: RunContext[Any], ref: WorkspaceRef | None
) -> Workspace | None:
    """The workspace the capabilities supply for `ref`, the way the agent selects it, policy wrappers included."""
    selected = capability.get_workspace(ctx, ref=ref)
    return selected if selected is None or isinstance(selected, Workspace) else Workspace(selected)


class DurableWorkspace(WrapperWorkspace):
    """The run's workspace inside a durable container: each call from workflow code is a durable operation.

    Outside the container, and inside a durable unit, calls go straight to the wrapped workspace.
    `ref` and `working_dir` report what `ensure` journaled.
    """

    def __init__(self, wrapped: Workspace, *, durability: BaseDurabilityCapability[Any], ctx: RunContext[Any]) -> None:
        super().__init__(wrapped)
        self._durability = durability
        self._ctx = ctx
        self._ref: WorkspaceRef | None = None
        self._working_dir: str | None = None
        self._ensure_lock = anyio.Lock()

    @property
    def ref(self) -> WorkspaceRef | None:
        return self._ref if self._ref is not None else self.wrapped.ref

    @property
    def backend(self) -> WorkspaceBackend:
        if self._in_container():
            raise UserError(
                '`workspace.backend` is not available in durable workflow code: calling the provider backend '
                f'directly would bypass the {self._durability.engine_name} {self._durability.durable_unit_plural} '
                'that make workspace operations durable. Use the `Workspace` methods instead, or reach the '
                f'backend from a tool, which runs inside {self._durability.durable_unit_noun}.'
            )
        return super().backend

    def _in_container(self) -> bool:
        return self._durability.in_durable_context and not in_durable_unit()

    async def working_dir(self) -> str:
        if self._working_dir is None and not self._in_container():
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
    ) -> WorkspaceResult:
        if not self._in_container():
            return await self.wrapped.run(command, shell=shell, cwd=cwd, env=env, timeout=timeout)
        call = WorkspaceCall(
            method='run', command=command, shell=shell, cwd=cwd, env=dict(env) if env else None, timeout=timeout
        )
        result = (await self._call(call)).command
        assert result is not None
        return result

    async def read_bytes(self, path: str) -> bytes:
        if not self._in_container():
            return await self.wrapped.read_bytes(path)
        return (await self._call(WorkspaceCall(method='read_bytes', path=path))).data

    async def write_bytes(self, path: str, data: bytes) -> None:
        if not self._in_container():
            return await self.wrapped.write_bytes(path, data)
        await self._call(WorkspaceCall(method='write_bytes', path=path, data=data))

    async def stat(self, path: str) -> WorkspaceFileEntry:
        if not self._in_container():
            return await self.wrapped.stat(path)
        return (await self._call(WorkspaceCall(method='stat', path=path))).entries[0]

    async def list_dir(self, path: str) -> Sequence[WorkspaceFileEntry]:
        if not self._in_container():
            return await self.wrapped.list_dir(path)
        return (await self._call(WorkspaceCall(method='list_dir', path=path))).entries

    async def make_dir(self, path: str) -> None:
        if not self._in_container():
            return await self.wrapped.make_dir(path)
        await self._call(WorkspaceCall(method='make_dir', path=path))

    async def remove(self, path: str) -> None:
        if not self._in_container():
            return await self.wrapped.remove(path)
        await self._call(WorkspaceCall(method='remove', path=path))

    async def exists(self, path: str) -> bool:
        if not self._in_container():
            return await self.wrapped.exists(path)
        return (await self._call(WorkspaceCall(method='exists', path=path))).flag

    async def realpath(self, path: str) -> str:
        if not self._in_container():
            return await self.wrapped.realpath(path)
        return (await self._call(WorkspaceCall(method='realpath', path=path))).text

    async def _call(self, call: WorkspaceCall) -> WorkspaceCallResult:
        await self._ensure()
        return await self._dispatch(call, ref=self._ref)

    async def _dispatch(self, call: WorkspaceCall, *, ref: WorkspaceRef | None) -> WorkspaceCallResult:
        # The ambient context is set around `before_run` and model requests; elsewhere (tool hooks,
        # `after_run`, `result.workspace` after the run) the run's own context serves.
        ctx = get_current_run_context() or self._ctx
        result = await self._durability._call_workspace(  # pyright: ignore[reportPrivateUsage]
            WorkspaceCallParams(run_context=ctx, ref=ref, call=call)
        )
        if result.error is not None:
            raise_error(result.error)
        return result

    async def _ensure(self, ctx: RunContext[Any] | None = None) -> None:
        """Run `ensure` once for the run, under a lock so parallel first calls share one environment."""
        if self._working_dir is not None:
            return
        async with self._ensure_lock:
            if self._working_dir is not None:
                return
            if ctx is not None:
                self._ctx = ctx
            result = await self._dispatch(WorkspaceCall(method='ensure'), ref=self.wrapped.ref)
            assert result.ref is not None
            if self.wrapped.ref != result.ref:
                # A fresh environment: rebuild the selection on its ref, so this side attaches to what
                # the unit created instead of creating another on first use.
                ctx = get_current_run_context() or self._ctx
                assert ctx.root_capability is not None
                rebuilt = select_workspace(ctx.root_capability, ctx, result.ref)
                if rebuilt is None:
                    raise UserError(
                        f'No capability can supply workspace {result.ref.id!r} from provider '
                        f'{result.ref.provider!r}, which the run just created. A `get_workspace` hook that '
                        'creates an environment must also recognize its ref.'
                    )
                self._backend = rebuilt
            self._ref = result.ref
            self._working_dir = result.text


class WorkspaceEnsurer(AbstractCapability[AgentDepsT]):
    """Runs `ensure` before the run body, from the `outermost` tier.

    The durability capability is `innermost`, so without this the first workspace call from another
    capability's `wrap_run` would run `ensure` itself through the lazy fallback.
    """

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        workspace = ctx.workspace
        if isinstance(workspace, DurableWorkspace):
            await workspace._ensure(ctx)  # pyright: ignore[reportPrivateUsage]
        return await handler()


class RejectWorkspaceInContainer(AbstractCapability[AgentDepsT]):
    """Refuse a workspace inside the container for the deprecated wrapper agents, which cannot make it durable."""

    _safe_at_runtime = True

    def __init__(self, *, engine: str, container_noun: str, capability: str) -> None:
        self._engine = engine
        self._container_noun = container_noun
        self._capability = capability

    def _prepare_workspace(self, ctx: RunContext[AgentDepsT], workspace: Workspace, *, explicit: bool) -> Workspace:
        raise UserError(
            f'Workspaces are not supported inside a {self._engine} {self._container_noun} through the deprecated '
            f'wrapper agent. Use `Agent(..., capabilities=[{self._capability}()])`, which runs every workspace '
            f'operation as a durable unit, and attach the workspace through a capability such as `LocalWorkspace`.'
        )
