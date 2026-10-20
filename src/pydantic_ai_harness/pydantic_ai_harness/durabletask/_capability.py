from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, ClassVar

from durabletask.task import RetryPolicy
from pydantic import JsonValue, TypeAdapter, ValidationError
from pydantic_core import to_json

from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.durable_exec import JSON_CODEC, BaseDurabilityCapability, DurabilityEngineSpec
from pydantic_ai.durable_exec._base import conversation_id_from_run_id
from pydantic_ai.durable_exec._capability_operation import CapabilityMethodDeclaration, CapabilityOperationParams
from pydantic_ai.durable_exec._operation import (
    DurableOperationConfig,
    DynamicToolsetCallToolParams,
    EventStreamHandlerParams,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesParams,
    ModelRequestParams,
    ParameterTransport,
    ToolsetCallToolParams,
    ToolsetGetToolsParams,
)
from pydantic_ai.durable_exec._toolset import validation_context_from_agent
from pydantic_ai.durable_exec._workspace import WorkspaceCallParams
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset

from . import _transports
from ._bridge import current_loop
from ._operation_backend import (
    DurableTaskOperationBackend,
    _SerializationFailure,  # pyright: ignore[reportPrivateUsage]
)
from ._run_context import _ContextSnapshot, restore_context  # pyright: ignore[reportPrivateUsage]


@dataclass(init=False)
class DurableTaskDurability(BaseDurabilityCapability[AgentDepsT]):
    """Checkpoint model, tool and capability operations as Durable Task activities.

    Attach to an agent, then construct `DurableTaskAgent` to register its activities
    and orchestrator with a worker. Outside a Durable Task orchestration, the agent
    behaves normally.

    Activities can execute more than once. Keep side effects idempotent, using
    `RunContext.run_id` and `RunContext.tool_call_id` as application-level keys.
    Orchestrator-side code must be deterministic and perform no external I/O.
    """

    engine_spec: ClassVar = DurabilityEngineSpec(
        engine_name='Durable Task',
        durable_unit_noun='activity',
        durable_unit_plural='activities',
        durable_container_noun='orchestration',
        codec=JSON_CODEC,
        serialization_failure=_SerializationFailure,
        toolset_lifecycles={
            'function': 'enter-outside-durable',
            'mcp': 'enter-outside-durable',
            'dynamic': 'enter-never',
        },
        unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
    )

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        retry_policy: RetryPolicy | None = None,
        operation_config: DurableOperationConfig[RetryPolicy | None] | None = None,
        deps_factory: Callable[[JsonValue], AbstractAsyncContextManager[AgentDepsT]] | None = None,
    ) -> None:
        """Configure activity retries and optional worker-side dependency reconstruction.

        `deps_factory` receives the JSON dependency input and owns its resources through
        an async context manager, entered and exited for each activity attempt. Without
        a factory, dependencies must be JSON serializable and are validated against the
        agent's `deps_type` on the worker.

        Use `operation_config=RoleBasedOperationConfig(...)` for per-role policies
        and per-tool configuration. Its `resolve_tool` callback may return `False`
        for a deterministic, I/O-free inline tool. Keep native policy objects out
        of tool metadata, which crosses JSON activity boundaries.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        if retry_policy is not None and operation_config is not None:
            raise UserError('Pass `retry_policy` or `operation_config`, not both.')
        self._retry_policy = retry_policy
        self._operation_config = operation_config
        self._deps_factory = deps_factory
        self._backend: DurableTaskOperationBackend | None = None
        self._deps_adapter: TypeAdapter[AgentDepsT] | None = None
        self._activity_deps: ContextVar[AgentDepsT] = ContextVar('durabletask_activity_deps')
        self._activity_toolsets: ContextVar[AsyncExitStack] = ContextVar('durabletask_activity_toolsets')

    @property
    def in_durable_context(self) -> bool:
        return current_loop() is not None

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        self._deps_adapter = TypeAdapter[AgentDepsT](agent.deps_type) if self._deps_factory is None else None
        self._backend = DurableTaskOperationBackend(
            agent_name=self.name,
            default_model_id=self.default_model_id,
            retry_policy=self._retry_policy,
            operation_config=self._operation_config,
            activity_scope=self._activity_scope,
        )
        super()._bind_to_agent(agent)
        if self._event_stream_handler is not None:
            self._bound_event_operation = self._bind_event_operation(self._backend)

    def get_durable_operation_backend(self) -> DurableTaskOperationBackend:
        if self._backend is None:
            raise UserError('Bind `DurableTaskDurability` to an agent before registering its activities.')
        return self._backend

    @asynccontextmanager
    async def _activity_scope(self, deps_input: JsonValue) -> AsyncGenerator[None]:
        async with AsyncExitStack() as stack:
            if self._deps_factory is not None:
                deps = await stack.enter_async_context(self._deps_factory(deps_input))
            else:
                assert self._deps_adapter is not None
                try:
                    deps = self._deps_adapter.validate_python(deps_input)
                except ValidationError as exc:
                    raise _SerializationFailure(exc) from exc
            deps_token = self._activity_deps.set(deps)
            toolsets_token = self._activity_toolsets.set(stack)
            try:
                yield
            finally:
                self._activity_toolsets.reset(toolsets_token)
                self._activity_deps.reset(deps_token)

    def _restore_context(self, snapshot: _ContextSnapshot) -> RunContext[AgentDepsT]:
        assert self._agent is not None
        return restore_context(snapshot, deps=self._activity_deps.get(), agent=self._agent)

    def _default_run_id(self) -> str | None:
        loop = current_loop()
        if loop is None:
            return None
        loop.run_counter += 1
        return f'{loop.context.instance_id}:{self.name}:{loop.run_counter}'

    def _default_conversation_id(self, run_id: str) -> str | None:
        return conversation_id_from_run_id(run_id) if self.in_durable_context else None

    async def _run_start_time(self, run_id: str) -> datetime | None:
        loop = current_loop()
        return loop.context.current_utc_datetime.replace(tzinfo=UTC) if loop is not None else None

    def _run_clock(self) -> Callable[[], datetime] | None:
        loop = current_loop()
        return (lambda: loop.context.current_utc_datetime.replace(tzinfo=UTC)) if loop is not None else None

    def _validation_context(self, ctx: RunContext[Any]) -> Any:
        assert self._agent is not None
        return validation_context_from_agent(self._agent)(ctx)

    async def _prepare_function_call_params(
        self, toolset: FunctionToolset[AgentDepsT], params: ToolsetCallToolParams
    ) -> ToolsetCallToolParams:
        assert params.tool is not None
        # Serialized function handlers cannot reuse a run-held toolset. The activity
        # owns its entry, closing it before the dependency factory on failure or cancel.
        await self._activity_toolsets.get().enter_async_context(toolset)
        # The activity wire contains JSON, while function tools expect schema-validated Python values.
        args = params.tool.args_validator.validate_json(
            to_json(params.tool_args), context=self._validation_context(params.ctx)
        )
        return replace(params, tool_args=args)

    def _function_call_parameter_transport(
        self, toolset: FunctionToolset[AgentDepsT]
    ) -> ParameterTransport[ToolsetCallToolParams, JsonValue]:
        return _transports.function_transport(toolset, self._restore_context)

    def _get_tools_parameter_transport(
        self, toolset: AbstractToolset[AgentDepsT]
    ) -> ParameterTransport[ToolsetGetToolsParams, JsonValue]:
        return _transports.discovery_transport(self._restore_context)

    def _get_instructions_parameter_transport(
        self, toolset: AbstractToolset[AgentDepsT]
    ) -> ParameterTransport[ToolsetGetToolsParams, JsonValue]:
        return _transports.discovery_transport(self._restore_context)

    def _dynamic_get_tools_parameter_transport(
        self, toolset: DynamicToolset[AgentDepsT]
    ) -> ParameterTransport[ToolsetGetToolsParams, JsonValue]:
        return _transports.discovery_transport(self._restore_context)

    def _dynamic_call_parameter_transport(
        self, toolset: DynamicToolset[AgentDepsT]
    ) -> ParameterTransport[DynamicToolsetCallToolParams, JsonValue]:
        return _transports.dynamic_transport(self._restore_context)

    def _mcp_call_parameter_transport(
        self, toolset: AbstractToolset[AgentDepsT]
    ) -> ParameterTransport[ToolsetCallToolParams, JsonValue]:
        from pydantic_ai.mcp import MCPToolset

        assert isinstance(toolset, MCPToolset)
        return _transports.mcp_transport(
            lambda definition, ctx: toolset.tool_for_tool_def(definition, ctx=ctx), self._restore_context
        )

    def _model_request_parameter_transport(
        self, result_type: object
    ) -> ParameterTransport[ModelRequestParams, JsonValue]:
        return _transports.request_transport(self._restore_context)

    def _compact_messages_parameter_transport(self) -> ParameterTransport[ModelCompactMessagesParams, JsonValue]:
        return _transports.compact_transport(self._restore_context)

    def _cancel_suspended_response_parameter_transport(
        self,
    ) -> ParameterTransport[ModelCancelSuspendedResponseParams, JsonValue]:
        return _transports.cancel_transport(self._restore_context)

    def _event_stream_handler_parameter_transport(self) -> ParameterTransport[EventStreamHandlerParams, JsonValue]:
        return _transports.event_transport(self._restore_context)

    def _capability_operation_parameter_transport(
        self, declaration: CapabilityMethodDeclaration
    ) -> ParameterTransport[CapabilityOperationParams, JsonValue]:
        return _transports.capability_transport(self._restore_context)

    def _workspace_call_transport(self) -> ParameterTransport[WorkspaceCallParams, JsonValue]:
        return _transports.workspace_transport(self._restore_context)
