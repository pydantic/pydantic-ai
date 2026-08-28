from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai.durable_exec._operation import (
    CapabilityOperationId,
    DurableOperationConfig,
    DurableOperationId,
    EventStreamHandlerId,
    OperationConfigRole,
    ToolsetCallToolId,
    ToolsetKind,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._operation_backend import CallableOperationBackend

from ._operation_names import PrefectOperationNamer
from ._types import TaskConfig


class PrefectOperationConfig(DurableOperationConfig[TaskConfig]):
    def __init__(
        self,
        *,
        model: TaskConfig,
        event: TaskConfig,
        capability: TaskConfig,
        tool: Callable[[ToolsetKind, object | None, str], TaskConfig | Literal[False]],
    ) -> None:
        self._model = model
        self._event = event
        self._capability = capability
        self._tool = tool

    def base(self, role: OperationConfigRole, *, operation_id: DurableOperationId) -> TaskConfig:
        if role == 'model':
            return self._model
        if role == 'event':
            return self._event
        if role == 'capability':
            assert isinstance(operation_id, CapabilityOperationId)
            return self._capability
        assert isinstance(operation_id, ToolsetCallToolId | ToolsetValidateToolArgumentsId)
        config = self._tool(operation_id.toolset_kind, None, '')
        assert config is not False
        return config

    def for_tool(
        self,
        role: OperationConfigRole,
        *,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> TaskConfig | Literal[False]:
        assert role == 'tool'
        assert isinstance(operation_id, ToolsetCallToolId | ToolsetValidateToolArgumentsId)
        return self._tool(operation_id.toolset_kind, tool, tool_name)


class PrefectOperationBackend(CallableOperationBackend[TaskConfig]):
    def __init__(self, *, config: PrefectOperationConfig, event_sequence_key: str) -> None:
        super().__init__(namer=PrefectOperationNamer(), config=config)
        self._event_sequence_key = event_sequence_key

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: TaskConfig,
    ) -> object:
        sequence_key: str | None = None
        if isinstance(operation_id, EventStreamHandlerId):
            sequence_key = self._event_sequence_key
        elif isinstance(operation_id, CapabilityOperationId):
            capability_id = operation_id.capability_id
            sequence_key = (
                f'{self._event_sequence_key}:capability:{len(capability_id)}:{capability_id}{operation_id.operation}'
            )

        if sequence_key is not None:
            flow_context = FlowRunContext.get()
            assert flow_context is not None
            # Prefect rebuilds dynamic task keys in the same order on flow retry. A counter per
            # semantic operation therefore distinguishes repeated live invocations while producing
            # the same cache keys during replay. Capability operations use separate counters so an
            # unrelated operation cannot shift their replay identities.
            sequence = flow_context.task_run_dynamic_keys.get(sequence_key, 0)
            assert isinstance(sequence, int)
            flow_context.task_run_dynamic_keys[sequence_key] = sequence + 1
            cache_key = (*cache_key, sequence)

        @task
        async def operation(operation_name: str, *logical_inputs: object) -> object:
            return await body()

        options = config or {}
        return await operation.with_options(name=name, **options)(name, *cache_key)
