from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Generic, Literal, Protocol, TypeVar, cast

from temporalio import activity, workflow
from temporalio.exceptions import ApplicationError, TemporalError
from temporalio.workflow import ActivityConfig

from pydantic_ai.durable_exec._operation import (
    CapabilityOperationId,
    DurableOperation,
    DurableOperationConfig,
    DurableOperationId,
    EventStreamHandlerId,
    ModelCancelSuspendedResponseId,
    ModelCompactMessagesId,
    ModelRequestId,
    OperationConfigRole,
    ParameterTransport,
    ParameterTransports,
    ToolsetCallToolId,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._operation_backend import BoundDurableOperation, RegisteredOperationBackend
from pydantic_ai.exceptions import ModelAPIError, UserError

from ._activity_execution import execute_activity, execute_child_workflow
from ._model_errors import model_errors_as_application_errors, rebuilt_model_errors
from ._operation_names import TemporalOperationNamer
from ._toolset import (
    ChildWorkflowOperationConfig,
    heartbeating,
    model_response_payload_errors,
)

ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')


registered_child_workflow_names: set[str] = set()
"""Child workflow types that a worker in this process was configured with, by name.

`worker_registrations` adds to it when a plugin configures a worker. It lives at module level, in a module the
workflow sandbox passes through, because the sandbox re-imports the user's workflow module and so runs calls on
re-created agents whose backends never saw the worker being configured.
"""

TemporalOperationConfigValue = ActivityConfig | ChildWorkflowOperationConfig


class TemporalParameterTransport(ParameterTransport[ParamsT, WireT], Protocol[ParamsT, WireT]):
    wire_type: object
    result_type: object

    @abstractmethod
    def dump(self, params: ParamsT) -> WireT: ...

    @abstractmethod
    def load(self, payload: WireT, *, runtime: object) -> ParamsT: ...


class _ModelParams(Protocol):
    model_id: str | None


class _EventParams(Protocol):
    event: Any


class TemporalOperationConfig(DurableOperationConfig[TemporalOperationConfigValue]):
    def __init__(
        self,
        *,
        model: ActivityConfig,
        event: ActivityConfig,
        tool: ActivityConfig,
        resolve_tool: Callable[[DurableOperationId, object | None, str], TemporalOperationConfigValue | Literal[False]],
    ) -> None:
        self._model = model
        self._event = event
        self._tool = tool
        self._resolve_tool = resolve_tool

    def base(self, role: OperationConfigRole, *, operation_id: DurableOperationId) -> ActivityConfig:
        if role == 'model':
            return self._model
        if role == 'event':
            return self._event
        return self._tool

    def for_tool(
        self,
        role: OperationConfigRole,
        *,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> TemporalOperationConfigValue | Literal[False]:
        return self._resolve_tool(operation_id, tool, tool_name)


class TemporalBoundOperation(BoundDurableOperation[ParamsT, WireT, ResultT], Generic[ParamsT, WireT, ResultT]):
    def __init__(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        registration: Callable[..., Awaitable[ResultT]],
        child_workflow: tuple[type, str] | None = None,
        config: TemporalOperationConfigValue,
    ) -> None:
        self._operation = operation
        self.registration = registration
        self.child_workflow = child_workflow
        self._config = config

    @property
    def operation(self) -> DurableOperation[ParamsT, WireT, ResultT]:
        return self._operation

    async def __call__(self, params: ParamsT, *, config: object | None = None) -> ResultT:
        payload = self._operation.parameter_transport.dump(params)
        resolved_config = cast(TemporalOperationConfigValue, config or self._config)
        if 'child_workflow' in resolved_config:
            toolset_id = cast(Any, self._operation.operation_id).toolset_id
            # A replay re-issues what the original execution did, which already passed this check, and may run
            # where no worker was configured (e.g. a `Replayer` in a fresh process), so only new execution checks.
            if self.child_workflow is None or (
                not workflow.unsafe.is_replaying() and self.child_workflow[1] not in registered_child_workflow_names
            ):
                raise UserError(
                    f'Toolset {toolset_id!r} has a tool configured to run as a Temporal child workflow, but its '
                    'child workflow was not registered with the worker, because no tool in the toolset had '
                    '`child_workflow` metadata when the worker was configured. Set the metadata on the tool, or with '
                    'a `SetToolMetadata` capability, before creating the worker, and register the worker through '
                    '`PydanticAIPlugin` or `AgentPlugin`.'
                )
            child_config = resolved_config['child_workflow'].copy()
            child_config.setdefault('id', f'{workflow.info().workflow_id}--{workflow.uuid4()}')
            transport = cast(TemporalParameterTransport[ParamsT, WireT], self._operation.parameter_transport)
            return await execute_child_workflow(
                cast(Any, self.child_workflow[0]).run,
                args=cast(Sequence[Any], payload),
                result_type=transport.result_type,
                **child_config,
            )
        activity_config = resolved_config.copy()
        operation_id = self._operation.operation_id
        model_name = ''
        if isinstance(operation_id, ModelRequestId):
            model_name = cast(_ModelParams, params).model_id or operation_id.model_name
            suffix = ' (stream)' if operation_id.streaming else ''
            activity_config['summary'] = f'request model: {model_name}{suffix}'
        elif isinstance(operation_id, ModelCancelSuspendedResponseId):
            model_name = cast(_ModelParams, params).model_id or operation_id.model_name
            activity_config['summary'] = f'cancel suspended response: {model_name}'
        elif isinstance(operation_id, ModelCompactMessagesId):
            model_name = cast(_ModelParams, params).model_id or operation_id.model_name
            activity_config['summary'] = f'compact messages: {model_name}'
        elif isinstance(operation_id, ToolsetCallToolId):
            tool_name = cast(Any, params).name
            activity_config['summary'] = f'call tool: {operation_id.toolset_id}:{tool_name}'
        elif isinstance(operation_id, ToolsetValidateToolArgumentsId):
            tool_name = cast(Any, params).name
            activity_config['summary'] = f'validate tool args: {operation_id.toolset_id}:{tool_name}'
        elif isinstance(operation_id, ToolsetGetToolsId):
            activity_config['summary'] = f'get tools: {operation_id.toolset_id}'
        elif isinstance(operation_id, ToolsetGetInstructionsId):
            activity_config['summary'] = f'get instructions: {operation_id.toolset_id}'
        elif isinstance(operation_id, CapabilityOperationId):
            activity_config['summary'] = f'capability: {operation_id.capability_id}.{operation_id.operation}'
        elif isinstance(operation_id, EventStreamHandlerId):
            event = cast(_EventParams, params).event
            activity_config['summary'] = f'handle event: {event.event_kind}'
        else:
            # New operation ids use their stable activity name as the default summary. Their
            # parameter transport must implement `TemporalParameterTransport`, including
            # `wire_type` and `result_type`, so the payload converter can inspect the activity.
            activity_config['summary'] = self.registration.__name__

        if isinstance(operation_id, ModelRequestId | ModelCompactMessagesId):
            with rebuilt_model_errors(), model_response_payload_errors(model_name):
                return await execute_activity(
                    activity=self.registration, args=cast(Sequence[Any], payload), **activity_config
                )
        if isinstance(operation_id, ModelCancelSuspendedResponseId):
            with rebuilt_model_errors():
                return await execute_activity(
                    activity=self.registration, args=cast(Sequence[Any], payload), **activity_config
                )
        return await execute_activity(activity=self.registration, args=cast(Sequence[Any], payload), **activity_config)


class TemporalOperationBackend(RegisteredOperationBackend[TemporalOperationConfigValue]):
    """Own Temporal activity definitions while preserving their existing callables."""

    def __init__(
        self,
        *,
        agent_name: str,
        deps_type: type[Any],
        model_config: ActivityConfig,
        event_config: ActivityConfig,
        tool_config: ActivityConfig,
        resolve_tool_config: Callable[
            [DurableOperationId, object | None, str], TemporalOperationConfigValue | Literal[False]
        ],
        non_retryable_model_errors: tuple[type[ModelAPIError], ...] = (),
        runtime: object | None = None,
        parameter_transports: ParameterTransports | None = None,
    ) -> None:
        super().__init__(
            namer=TemporalOperationNamer(agent_name),
            config=TemporalOperationConfig(
                model=model_config, event=event_config, tool=tool_config, resolve_tool=resolve_tool_config
            ),
            parameter_transports=parameter_transports,
        )
        self._deps_type = deps_type
        self._non_retryable_model_errors = non_retryable_model_errors
        self._runtime = runtime
        # Every function toolset gets a child workflow definition, but a worker only registers the ones whose toolset
        # selects a child workflow when it's configured (see `worker_registrations`).
        self.child_workflow_toolsets: dict[type, tuple[str, str]] = {}
        """Each child workflow definition's toolset id and workflow name."""

    def move_registration_to_end(self, registration: Callable[..., object]) -> None:
        self._registrations.remove(registration)
        self._registrations.append(registration)

    def register(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        name: str,
        config: TemporalOperationConfigValue,
    ) -> tuple[BoundDurableOperation[ParamsT, WireT, ResultT], Sequence[Callable[..., object]]]:
        transport = cast(TemporalParameterTransport[ParamsT, WireT], operation.parameter_transport)

        model_operation = isinstance(
            operation.operation_id, ModelRequestId | ModelCompactMessagesId | ModelCancelSuspendedResponseId
        )

        async def activity_handler(params: Any, deps: Any = None) -> ResultT:
            semantic_params = transport.load(cast(WireT, (params, deps)), runtime=self._runtime)
            async with heartbeating():
                if model_operation:
                    with model_errors_as_application_errors(self._non_retryable_model_errors):
                        return await operation.handler(semantic_params)
                return await operation.handler(semantic_params)

        # Existing operation transports retain their shipped wire dataclasses and activity
        # signatures. New operation ids ride this generic registration path and only need a
        # `TemporalParameterTransport` with `wire_type` and `result_type`.
        # Temporal's Pydantic payload converter deserializes `deps` by inspecting the
        # registered callable, so patch the exact function that the SDK will inspect.
        activity_handler.__annotations__ = {
            'params': transport.wire_type,
            'deps': self._deps_type | None,
            'return': transport.result_type,
        }
        registration = activity.defn(name=name)(activity_handler)
        child_workflow: tuple[type, str] | None = None
        registrations: list[Callable[..., object]] = [registration]
        operation_id = operation.operation_id
        if isinstance(operation_id, ToolsetCallToolId) and operation_id.toolset_kind == 'function':
            runtime = self._runtime

            async def run(self: object, params: Any, deps: Any = None) -> ResultT:
                if workflow.info().parent is None:
                    raise ApplicationError(
                        'Pydantic AI tool-call workflows must be started as child workflows.', non_retryable=True
                    )
                semantic_params = transport.load(cast(WireT, (params, deps)), runtime=runtime)
                try:
                    return await operation.handler(semantic_params)
                except TemporalError:
                    raise
                except Exception as e:
                    # Like an exception in an activity, fail this child workflow (subject to its retry policy) so
                    # the error reaches the parent's tool call, instead of failing the workflow task, which
                    # Temporal would retry forever.
                    raise ApplicationError(str(e), type=type(e).__name__) from e

            # UTF-8 hex preserves the full operation name while using only Python identifier characters.
            encoded_name = name.encode().hex()
            class_name = f'_{encoded_name}__child_workflow'
            run.__qualname__ = f'{class_name}.run'
            run.__annotations__ = activity_handler.__annotations__
            child_workflow_name = f'{name}__child_workflow'
            child_workflow_registration = workflow.defn(name=child_workflow_name)(
                type(class_name, (), {'run': workflow.run(run)})
            )
            globals()[class_name] = child_workflow_registration
            child_workflow = (child_workflow_registration, child_workflow_name)
            self.child_workflow_toolsets[child_workflow_registration] = (operation_id.toolset_id, child_workflow_name)
            registrations.append(cast(Callable[..., object], child_workflow_registration))
        bound = TemporalBoundOperation(
            operation,
            registration=cast(Callable[..., Awaitable[ResultT]], registration),
            child_workflow=child_workflow,
            config=config,
        )
        return bound, tuple(registrations)
