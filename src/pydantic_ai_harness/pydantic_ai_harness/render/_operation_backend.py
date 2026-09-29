"""Registered Pydantic AI operations backed by Render Workflows tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeGuard, TypeVar

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
    ToolsetCallToolParams,
    dump_operation_params,
    function_tool_original_name,
    load_operation_params,
    operation_run_context,
)
from ._protocol import (
    PROTOCOL_VERSION,
    OperationRequest,
    OperationResult,
    RenderProtocolError,
    apply_effects,
    control_flow_error,
    make_request,
    permanent_error,
    read_outcome,
    read_protocol_version,
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


@dataclass(frozen=True)
class _RegisteredTask:
    """One registered Render task and the options fixed on it at registration."""

    name: str
    task: TaskDefinition[[OperationRequest], OperationResult]
    options: Options | None


def _routed_tool_name(params: object) -> str | None:
    """The toolset's own name for the tool a call is for, or `None` for other operations."""
    if not isinstance(params, ToolsetCallToolParams):
        return None
    tool = params.tool
    if tool is None:
        return params.name
    # A `prepare` function can rename a tool for the model; the toolset still holds it, and
    # registered it, under its original name.
    return function_tool_original_name(tool) or params.name


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
        operation_name: str,
        shared: _RegisteredTask | None = None,
        per_tool: Mapping[str, _RegisteredTask] | None = None,
        runtime: RenderWorkflows[RuntimeDepsT],
    ) -> None:
        self._operation = operation
        self._operation_name = operation_name
        self._shared = shared
        self._per_tool: Mapping[str, _RegisteredTask] = per_tool or {}
        self._runtime = runtime

    @property
    def operation(self) -> DurableOperation[ParamsT, WireT, ResultT]:
        return self._operation

    async def __call__(self, params: ParamsT, *, config: object | None = None) -> ResultT:
        context = self._runtime.current_task_context
        if context is None:
            return await self._operation.handler(params)

        registered = self._route(params)
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

    def _route(self, params: ParamsT) -> _RegisteredTask:
        shared = self._shared
        if shared is not None:
            return shared
        tool_name = _routed_tool_name(params)
        registered = self._per_tool.get(tool_name) if tool_name is not None else None
        if registered is None:
            raise UserError(
                f'Render operation {self._operation_name!r} has no registered task for tool {tool_name!r}. '
                'Each function tool that resolves its own task options is registered as its own Render '
                'task while the agent is bound, and Render cannot register tasks once the worker has '
                'started. Add the tool to its toolset before the agent is constructed.'
            )
        return registered

    def _check_static_config(self, config: object | None, options: Options | None) -> None:
        if config is not None and config != options:
            raise UserError(
                'Render Workflows task options are fixed when an agent is bound and cannot vary per invocation.'
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
        self._agent_name = agent_name
        self._per_tool_task_names: set[str] = set()

    def register(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        name: str,
        config: Options | None,
    ) -> tuple[BoundDurableOperation[ParamsT, WireT, ResultT], Sequence[Callable[..., object]]]:
        per_tool = self._per_tool_options(operation, config=config)
        if per_tool is None:
            bound = RenderBoundOperation(
                operation,
                operation_name=name,
                shared=self._register_task(operation, name=name, config=config),
                runtime=self._runtime,
            )
        else:
            # The operation's own name is the stable prefix, so each tool's task name stays
            # derived from the agent, the toolset, the tool, and the operation alone.
            prefix, _, suffix = name.rpartition('.')
            bound = RenderBoundOperation(
                operation,
                operation_name=name,
                per_tool={
                    tool_name: self._register_task(
                        operation,
                        name=self._per_tool_task_name(prefix, tool_name, suffix),
                        config=tool_config,
                    )
                    for tool_name, tool_config in per_tool.items()
                },
                runtime=self._runtime,
            )
        # `app.task` has already registered the task definitions. The generic backend's second
        # return value is for worker-registration callables, not task runs, so it is empty here.
        return bound, ()

    def _register_task(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        name: str,
        config: Options | None,
    ) -> _RegisteredTask:
        async def operation_task(context: TaskContext, request: OperationRequest) -> OperationResult:
            with recording_effects() as recorder:
                response_version = PROTOCOL_VERSION
                try:
                    response_version = read_protocol_version(request)
                    wire_params = read_request(request, expected_operation=name)
                    params = load_operation_params(operation, wire_params, runtime=self._runtime)
                except Exception as exc:
                    # Retrying cannot repair persisted request bytes or worker-side decoding.
                    return permanent_error('invalid-request', exc, version=response_version)

                try:
                    with self._runtime.activate(context):
                        value = await operation.handler(params)
                except RenderProtocolError as exc:
                    return permanent_error('invalid-result', exc, version=response_version)
                except Exception as exc:
                    try:
                        expected_error = control_flow_error(exc, version=response_version)
                    except Exception as encoding_error:
                        return permanent_error('invalid-result', encoding_error, version=response_version)
                    if expected_error is not None:
                        return expected_error
                    raise

                try:
                    payload = operation.result_codec.dump(value)
                    effects = recorder.effects() if _completed_tool_result(operation.operation_id, payload) else None
                    return success(payload, effects=effects, version=response_version)
                except Exception as exc:
                    # The handler may already have committed an external side effect.
                    return permanent_error('invalid-result', exc, version=response_version)

        registered_config = _snapshot_options(config) if config is not None else None
        options = registered_config or Options()
        # A `None` field intentionally lets the public Workflows API resolve its app default
        # when this task is registered; only explicit per-operation values are snapshotted here.
        task = self._app.task(
            name=name,
            retry=options.retry,
            timeout_seconds=options.timeout_seconds,
            plan=options.plan,
        )(operation_task)
        return _RegisteredTask(name=name, task=task, options=registered_config)

    def _per_tool_options(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        config: Options | None,
    ) -> dict[str, Options | None] | None:
        """Resolve one set of task options per statically known function tool.

        `None` means the toolset keeps a single shared task: nothing asked for per-tool options,
        the toolset holds no tools yet, or the leaf is not a plain `FunctionToolset`. Task names
        are persisted workflow identity, so a toolset that needs nothing more than the shared
        task keeps the name it already had.
        """
        operation_id = operation.operation_id
        if not isinstance(operation_id, ToolsetCallToolId | ToolsetValidateToolArgumentsId):
            return None
        if operation_id.toolset_kind != 'function':
            return None
        toolset = self._static_function_toolset(operation_id.toolset_id)
        if toolset is None:
            return None
        resolved: dict[str, Options | None | Literal[False]] = {
            tool_name: self.config_for_tool(operation, tool=tool, tool_name=tool_name)
            for tool_name, tool in toolset.tools.items()
        }
        if not resolved or all(tool_config == config for tool_config in resolved.values()):
            return None
        # `False` opts the tool out of a Render task entirely: the toolset wrapper resolves the
        # same `False` for the call and runs the tool inline, so it gets no task definition.
        return {tool_name: tool_config for tool_name, tool_config in resolved.items() if tool_config is not False}

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

    def _per_tool_task_name(self, prefix: str, tool_name: str, suffix: str) -> str:
        """Build a stable per-tool name that cannot shadow a shared function task."""
        reserved = self._shared_function_task_names()
        preferred = f'{prefix}.{tool_name}.{suffix}'
        if preferred not in reserved and preferred not in self._per_tool_task_names:
            self._per_tool_task_names.add(preferred)
            return preferred
        base = f'{prefix}.__tool__.{tool_name}.{suffix}'
        candidate = base
        index = 2
        while candidate in reserved or candidate in self._per_tool_task_names:
            candidate = f'{base}.{index}'
            index += 1
        self._per_tool_task_names.add(candidate)
        return candidate

    def _shared_function_task_names(self) -> set[str]:
        """Return every shared function task name this agent can register later."""
        agent = self._runtime.agent
        if agent is None:  # pragma: no cover - task names are built while binding an agent
            return set()
        names: set[str] = set()

        def visit(leaf: AbstractToolset[RuntimeDepsT]) -> None:
            if isinstance(leaf, FunctionToolset) and leaf.id is not None:
                prefix = f'{self._agent_name}__function_toolset__{leaf.id}'
                names.update((f'{prefix}.call_tool', f'{prefix}.validate_args'))

        for toolset in agent.toolsets:
            toolset.apply(visit)
        return names
