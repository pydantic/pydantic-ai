"""Registered Pydantic AI operations backed by Render Workflows tasks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeGuard, TypeVar

from render import Options, Retry, TaskContext, Workflows
from render.workflows import TaskDefinition

from pydantic_ai.durable_exec import (
    JournalOperationNamer,
    RegisteredOperationBackend,
    RoleBasedOperationConfig,
    ToolsetCallToolId,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from ._compat import (
    BoundDurableOperation,
    DurableOperation,
    dump_operation_params,
    load_operation_params,
    operation_run_context,
)
from ._protocol import (
    OperationRequest,
    OperationResult,
    RenderProtocolError,
    apply_effects,
    control_flow_error,
    make_request,
    permanent_error,
    read_outcome,
    read_request,
    recording_effects,
    success,
)

if TYPE_CHECKING:
    from ._capability import RenderWorkflows

ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')
RuntimeDepsT = TypeVar('RuntimeDepsT')


def _snapshot_options(options: Options) -> Options:
    retry = options.retry
    return Options(
        retry=(
            Retry(
                max_retries=retry.max_retries,
                wait_duration_ms=retry.wait_duration_ms,
                backoff_scaling=retry.backoff_scaling,
            )
            if retry is not None
            else None
        ),
        timeout_seconds=options.timeout_seconds,
        plan=options.plan,
    )


@dataclass
class _RegisteredTask:
    """A task handle filled after core binding and its snapshotted options."""

    name: str
    options: Options | None
    definition: TaskDefinition[[OperationRequest], OperationResult] | None = None

    @property
    def task(self) -> TaskDefinition[[OperationRequest], OperationResult]:
        assert self.definition is not None, 'Render tasks must be registered before an operation runs.'
        return self.definition


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def _completed_tool_result(operation_id: object, payload: object) -> bool:
    """Whether a wrapped tool result represents a completed call rather than control flow."""
    if not isinstance(operation_id, ToolsetCallToolId | ToolsetValidateToolArgumentsId):
        return True
    return _is_object_dict(payload) and payload.get('kind') in {'tool_return', 'tool_content_result'}


class RenderBoundOperation(
    BoundDurableOperation[ParamsT, WireT, ResultT],
    Generic[ParamsT, WireT, ResultT, RuntimeDepsT],
):
    """Dispatch one operation through its statically registered Render task."""

    def __init__(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        registered: _RegisteredTask,
        runtime: RenderWorkflows[RuntimeDepsT],
    ) -> None:
        self._operation = operation
        self._registered = registered
        self._runtime = runtime

    @property
    def operation(
        self,
    ) -> DurableOperation[ParamsT, WireT, ResultT]:  # pragma: no cover - backend protocol introspection
        return self._operation

    async def __call__(self, params: ParamsT, *, config: object | None = None) -> ResultT:
        context = self._runtime.current_task_context
        if context is None:  # pragma: no cover - core bypasses bound operations outside durable context
            return await self._operation.handler(params)

        registered = self._registered
        self._check_static_config(config, registered.options)
        wire_params = dump_operation_params(self._operation, params)
        request = make_request(registered.name, wire_params)
        result = await context.run(registered.task, request)
        outcome = read_outcome(result)
        caller_ctx = operation_run_context(params)
        if outcome.effects is not None:
            if caller_ctx is None:  # pragma: no cover - operations that produce effects carry a context
                raise UserError(f'Render operation {registered.name!r} returned effects without a run context.')
            await apply_effects(outcome.effects, ctx=caller_ctx)
        return self._operation.result_codec.load(outcome.payload)

    def _check_static_config(self, config: object | None, options: Options | None) -> None:
        if config is not None and config != options:
            raise UserError(
                'Render Workflows task options are fixed when an agent is bound and cannot vary per tool or invocation. '
                'Put tools needing different settings in separate named toolsets.'
            )


class RenderOperationBackend(RegisteredOperationBackend[Options | None], Generic[RuntimeDepsT]):
    """Register each supported Pydantic AI operation on a Workflows app."""

    def __init__(
        self,
        app: Workflows,
        *,
        runtime: RenderWorkflows[RuntimeDepsT],
        agent_name: str,
        config: RoleBasedOperationConfig[Options | None],
    ) -> None:
        super().__init__(namer=JournalOperationNamer(agent_name), config=config)
        self._app = app
        self._runtime = runtime
        self._pending_registrations: list[Callable[[], None]] = []

    def register(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        name: str,
        config: Options | None,
    ) -> tuple[BoundDurableOperation[ParamsT, WireT, ResultT], Sequence[Callable[..., object]]]:
        self._validate_tool_options(operation, config=config)
        bound = RenderBoundOperation(
            operation,
            registered=self._prepare_task(operation, name=name, config=config),
            runtime=self._runtime,
        )
        # RenderWorkflows.for_agent registers the queued SDK definitions after core binding.
        # The Render SDK needs no additional worker-registration callables.
        return bound, ()

    def _prepare_task(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        name: str,
        config: Options | None,
    ) -> _RegisteredTask:
        async def operation_task(context: TaskContext, request: OperationRequest) -> OperationResult:
            with recording_effects() as recorder:
                try:
                    wire_params = read_request(request, expected_operation=name)
                    params = load_operation_params(operation, wire_params, runtime=self._runtime)
                except Exception as exc:
                    # Retrying cannot repair persisted request bytes or worker-side decoding.
                    return permanent_error('invalid-request', exc)

                try:
                    with self._runtime.activate(context):
                        value = await operation.handler(params)
                except RenderProtocolError as exc:
                    return permanent_error('invalid-result', exc)
                except Exception as exc:
                    try:
                        expected_error = control_flow_error(exc)
                    except Exception as encoding_error:
                        return permanent_error('invalid-result', encoding_error)
                    if expected_error is not None:
                        return expected_error
                    raise

                try:
                    payload = operation.result_codec.dump(value)
                    effects = recorder.effects() if _completed_tool_result(operation.operation_id, payload) else None
                    return success(payload, effects=effects)
                except Exception as exc:
                    # The handler may already have committed an external side effect.
                    return permanent_error('invalid-result', exc)

        registered_config = _snapshot_options(config) if config is not None else None
        options = registered_config or Options()
        # A `None` field intentionally lets the public Workflows API resolve its app default
        # when this task is registered; only explicit per-operation values are snapshotted here.
        registered = _RegisteredTask(name=name, options=registered_config)

        def commit() -> None:
            registered.definition = self._app.task(
                name=name,
                retry=options.retry,
                timeout_seconds=options.timeout_seconds,
                plan=options.plan,
            )(operation_task)

        self._pending_registrations.append(commit)
        return registered

    def register_tasks(self) -> None:
        """Commit the pending definitions after core binding succeeds.

        The SDK cannot roll back registrations. Discard the app if an SDK call fails.
        """
        for commit in self._pending_registrations:
            commit()
        self._pending_registrations.clear()

    def _validate_tool_options(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        config: Options | None,
    ) -> None:
        """Reject per-tool policies before committing any SDK task definitions."""
        operation_id = operation.operation_id
        if not isinstance(operation_id, ToolsetCallToolId | ToolsetValidateToolArgumentsId):
            return
        if operation_id.toolset_kind != 'function':
            return
        toolset = self._static_function_toolset(operation_id.toolset_id)
        if toolset is None:
            return
        for tool_name, tool in toolset.tools.items():
            resolved = self.config_for_tool(operation, tool=tool, tool_name=tool_name)
            if resolved is not False and resolved != config:
                raise UserError(
                    'Render Workflows task options are fixed per toolset. '
                    'Put tools needing different settings in separate named toolsets, '
                    'or return False to run a function tool inline.'
                )

    def _static_function_toolset(self, toolset_id: str) -> FunctionToolset[RuntimeDepsT] | None:
        """The bound agent's construction-time function toolset with this ID, if there is one."""
        agent = self._runtime.agent
        if agent is None:  # pragma: no cover - the backend is only built while binding an agent
            return None
        found: list[FunctionToolset[RuntimeDepsT]] = []

        def visit(leaf: AbstractToolset[RuntimeDepsT]) -> None:
            if isinstance(leaf, FunctionToolset) and leaf.id == toolset_id:
                found.append(leaf)

        for toolset in agent.toolsets:
            toolset.apply(visit)
        return found[0] if len(found) == 1 else None
