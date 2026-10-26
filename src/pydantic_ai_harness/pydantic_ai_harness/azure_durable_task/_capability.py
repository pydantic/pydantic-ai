from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import AbstractContextManager, AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar

from durabletask.task import RetryPolicy
from durabletask.worker import TaskHubGrpcWorker
from pydantic import JsonValue, TypeAdapter, ValidationError
from pydantic_core import to_json

from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.durable_exec import JSON_CODEC, BaseDurabilityCapability, DurabilityEngineSpec, DurableOperationConfig
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT, RunContext

from . import _transports
from ._bridge import current_loop
from ._operation_backend import (
    AzureDurableTaskOperationBackend,
    _SerializationFailure,  # pyright: ignore[reportPrivateUsage]
)
from ._run_context import _ContextSnapshot, restore_context  # pyright: ignore[reportPrivateUsage]


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
    ) -> None:
        """Configure worker-side models, event handling and native activity retry policies.

        Pass either `retry_policy` or `operation_config`, not both. A
        `RoleBasedOperationConfig` can select per-role and per-tool policies.
        Returning `False` for a function tool runs it inline; it must be deterministic
        and I/O-free. Keep native policies out of JSON-serialized tool metadata.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        if retry_policy is not None and operation_config is not None:
            raise UserError('Pass `retry_policy` or `operation_config`, not both.')
        self._retry_policy = retry_policy
        self._operation_config = operation_config
        self._backend: AzureDurableTaskOperationBackend | None = None
        self._deps_adapter: TypeAdapter[AgentDepsT] | None = None
        self._activity_deps: ContextVar[AgentDepsT] = ContextVar('azure_durable_task_activity_deps')
        self._activity_toolsets: ContextVar[AsyncExitStack] = ContextVar('azure_durable_task_activity_toolsets')

    @property
    def in_durable_context(self) -> bool:
        return current_loop() is not None

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        self._deps_adapter = TypeAdapter[AgentDepsT](agent.deps_type)
        self._backend = AzureDurableTaskOperationBackend(
            agent_name=self.name,
            default_model_id=self.default_model_id,
            retry_policy=self._retry_policy,
            operation_config=self._operation_config,
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
            raise _SerializationFailure(exc) from exc
        async with AsyncExitStack() as stack:
            deps_token = self._activity_deps.set(deps)
            toolsets_token = self._activity_toolsets.set(stack)
            try:
                yield
            finally:
                self._activity_toolsets.reset(toolsets_token)
                self._activity_deps.reset(deps_token)

    def _restore_context(self, snapshot: _ContextSnapshot) -> RunContext[AgentDepsT]:
        return restore_context(
            snapshot, deps=self._activity_deps.get(), agent=self._agent, engine_spec=self.engine_spec
        )

    def get_default_run_id(self) -> str | None:
        loop = current_loop()
        if loop is None:
            return None
        return f'{loop.context.instance_id}:{self.name}:{loop.context.new_uuid()}'

    def get_default_conversation_id(self, run_id: str) -> str | None:
        return run_id if self.in_durable_context else None

    async def get_run_start_time(self, run_id: str) -> datetime | None:
        loop = current_loop()
        return loop.context.current_utc_datetime.replace(tzinfo=UTC) if loop is not None else None

    def get_run_clock(self) -> Callable[[], datetime] | None:
        loop = current_loop()
        return (lambda: loop.context.current_utc_datetime.replace(tzinfo=UTC)) if loop is not None else None
