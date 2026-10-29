from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import AbstractContextManager, AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from durabletask.task import RetryPolicy
from durabletask.worker import TaskHubGrpcWorker
from pydantic import JsonValue, TypeAdapter, ValidationError
from pydantic_core import to_json

from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    DurabilityEngineSpec,
    DurableOperationId,
    RoleBasedOperationConfig,
    ToolsetCallToolId,
    ToolsetValidateToolArgumentsId,
    conversation_id_from_run_id,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolsetTool

from . import _transports
from ._bridge import current_loop
from ._operation_backend import (
    AzureDurableTaskOperationBackend,
    SerializationFailure,
)
from ._run_context import ActivityRunContext, ContextSnapshot

ToolOptionsResolver = Callable[
    [DurableOperationId, object | None, str],
    RetryPolicy | Literal[False] | None,
]


@dataclass(init=False)
class AzureDurableTaskDurability(BaseDurabilityCapability[AgentDepsT]):
    """Run agent operations as activities in user-written Azure Durable Task orchestrations.

    Call `run_durable` from the orchestrator. Retrieve the bound capability with
    `from_agent` and enter `register(worker)` before starting its SDK worker.
    Outside an orchestration, the agent behaves normally.

    Activities can execute more than once. Keep side effects idempotent, using
    `RunContext.run_id` and `RunContext.tool_call_id` as application-level keys.
    Dependencies must be JSON serializable and are rebuilt using the agent's `deps_type`.
    """

    engine_spec: ClassVar = DurabilityEngineSpec(
        engine_name='Azure Durable Task',
        durable_unit_noun='activity',
        durable_unit_plural='activities',
        durable_container_noun='orchestration',
        codec=JSON_CODEC,
        serialization_failure=SerializationFailure,
        toolset_lifecycles={
            'function': 'enter-outside-durable',
            'mcp': 'enter-outside-durable',
            'dynamic': 'enter-never',
        },
        unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        tool_config_key='azure_durable_task',
    )

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        model_options: RetryPolicy | None = None,
        tool_options: RetryPolicy | None = None,
        event_options: RetryPolicy | None = None,
        capability_options: RetryPolicy | None = None,
        resolve_tool_options: ToolOptionsResolver | None = None,
    ) -> None:
        """Configure worker-side models, event handling and native activity retry policies.

        Args:
            models: Additional models keyed by their registered model ID.
            event_stream_handler: Optional handler for agent stream events.
            name: Stable prefix for generated activity names. Defaults to the agent name.
            model_options: Native retry policy for model activities.
            tool_options: Native retry policy for tool activities, including discovery.
            event_options: Native retry policy for event handler activities.
            capability_options: Native retry policy for other capabilities' `@durable_operation` methods.
            resolve_tool_options: Optional resolver for toolset defaults and per-tool policies.
                The framework asks for a toolset default with `tool=None`, then resolves each
                function tool with its concrete tool and original name. `None` keeps `tool_options`;
                `False` runs a function tool inline. Inline tools must be deterministic and I/O-free.
                Function tools can also opt out with `metadata={'azure_durable_task': False}`.
                Keep native policies out of JSON-serialized tool metadata.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)

        def resolve_options(
            operation_id: DurableOperationId, tool: object | None, tool_name: str
        ) -> RetryPolicy | Literal[False] | None:
            # Keep the declared type: the `isinstance` checks below would narrow `tool` to a partially unknown generic.
            resolver_tool: object | None = tool
            from_metadata = (
                isinstance(tool, ToolsetTool)
                and tool.tool_def.metadata is not None
                and tool.tool_def.metadata.get('azure_durable_task') is False
            )
            if from_metadata:
                resolved = False
            elif resolve_tool_options is not None:
                static_name = tool.original_name if isinstance(tool, FunctionToolsetTool) else None
                resolved = resolve_tool_options(operation_id, resolver_tool, static_name or tool_name)
            else:
                resolved = None
            # The shared activity is registered before a concrete tool is known.
            if tool is None and resolved is False:
                return tool_options
            if resolved is False and (
                not isinstance(operation_id, ToolsetCallToolId | ToolsetValidateToolArgumentsId)
                or operation_id.toolset_kind != 'function'
            ):
                source = "`metadata={'azure_durable_task': False}`" if from_metadata else '`resolve_tool_options`'
                raise UserError(
                    f'{source} can opt only function tools out of durable execution; '
                    f'{tool_name!r} is an MCP or dynamic tool, so it must run as an Azure Durable Task activity.'
                )
            return tool_options if resolved is None else resolved

        self._operation_config = RoleBasedOperationConfig[RetryPolicy | None](
            model=model_options,
            tool=tool_options,
            event=event_options,
            capability=capability_options,
            resolve_tool=resolve_options,
        )
        self._backend: AzureDurableTaskOperationBackend | None = None
        # AbstractAgent.deps_type erases its generic parameter; the runtime schema validates it in activities.
        self._deps_adapter: TypeAdapter[Any] | None = None
        self._activity_deps: ContextVar[AgentDepsT] = ContextVar('azure_durable_task_activity_deps')
        self._activity_toolsets: ContextVar[AsyncExitStack] = ContextVar('azure_durable_task_activity_toolsets')

    @property
    def in_durable_context(self) -> bool:
        return current_loop() is not None

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        self._deps_adapter = TypeAdapter(agent.deps_type)
        self._backend = AzureDurableTaskOperationBackend(
            agent_name=self.name,
            default_model_id=self.default_model_id,
            config=self._operation_config,
            activity_scope=self._activity_scope,
            parameter_transports=_transports.AzureDurableTaskParameterTransports(
                self._restore_context, self._activity_toolsets, engine_spec=self.engine_spec
            ),
        )
        if self._event_stream_handler is not None:
            self._bound_event_operation = self._bind_event_operation(self._backend)
        super()._bind_to_agent(agent)

    def get_durable_operation_backend(self) -> AzureDurableTaskOperationBackend:
        if self._backend is None:
            raise UserError('Bind `AzureDurableTaskDurability` to an agent before registering its activities.')
        return self._backend

    def register(self, worker: TaskHubGrpcWorker) -> AbstractContextManager[None]:
        """Register activities and own their runtime until the worker has stopped and drained.

        Use the bound capability returned by `AzureDurableTaskDurability.from_agent(agent)`.
        Register your orchestrators directly with the SDK worker.
        """
        return self.get_durable_operation_backend().worker_scope(worker)

    @asynccontextmanager
    async def _activity_scope(self, deps_input: JsonValue) -> AsyncGenerator[None]:
        assert self._deps_adapter is not None
        try:
            deps = self._deps_adapter.validate_json(to_json(deps_input))
        except ValidationError as exc:
            raise SerializationFailure(exc) from exc
        async with AsyncExitStack() as stack:
            deps_token = self._activity_deps.set(deps)
            toolsets_token = self._activity_toolsets.set(stack)
            try:
                yield
            finally:
                self._activity_toolsets.reset(toolsets_token)
                self._activity_deps.reset(deps_token)

    def _restore_context(self, snapshot: ContextSnapshot) -> RunContext[AgentDepsT]:
        return ActivityRunContext.restore(
            snapshot.context, deps=self._activity_deps.get(), agent=self._agent, engine_spec=self.engine_spec
        )

    def get_default_run_id(self) -> str | None:
        loop = current_loop()
        if loop is None:
            return None
        return f'{loop.context.instance_id}:{self.name}:{loop.context.new_uuid()}'

    def get_default_conversation_id(self, run_id: str) -> str | None:
        return conversation_id_from_run_id(run_id) if self.in_durable_context else None

    async def get_run_start_time(self, run_id: str) -> datetime | None:
        loop = current_loop()
        return loop.context.current_utc_datetime.replace(tzinfo=UTC) if loop is not None else None

    def get_run_clock(self) -> Callable[[], datetime] | None:
        loop = current_loop()
        return (lambda: loop.context.current_utc_datetime.replace(tzinfo=UTC)) if loop is not None else None
