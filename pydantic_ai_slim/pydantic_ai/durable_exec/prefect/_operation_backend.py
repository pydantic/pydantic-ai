from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal, cast

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai.durable_exec._operation import (
    CallToolId,
    CapabilityOperationId,
    DurableOperationId,
    EventStreamHandlerId,
    OperationConfigRole,
    ToolsetKind,
    ValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._operation_backend import CallableOperationBackend
from pydantic_ai.durable_exec._operation_names import PrefectOperationNamer

from ._types import TaskConfig


class PrefectOperationConfig:
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

    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> TaskConfig:
        if role is OperationConfigRole.MODEL:
            return self._model
        if role is OperationConfigRole.EVENT:
            return self._event
        if role is OperationConfigRole.CAPABILITY:
            assert isinstance(operation_id, CapabilityOperationId)
            return self._capability
        assert isinstance(operation_id, CallToolId | ValidateToolArgumentsId)
        config = self._tool(operation_id.toolset_kind, None, '')
        assert config is not False
        return config

    def for_tool(
        self,
        role: OperationConfigRole,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> TaskConfig | Literal[False]:
        assert role in (OperationConfigRole.TOOL_CALL, OperationConfigRole.TOOL_VALIDATION)
        assert isinstance(operation_id, CallToolId | ValidateToolArgumentsId)
        return self._tool(operation_id.toolset_kind, tool, tool_name)


class PrefectOperationBackend(CallableOperationBackend[TaskConfig]):
    def __init__(self, *, config: PrefectOperationConfig, event_sequence_key: str) -> None:
        super().__init__(namer=PrefectOperationNamer(), config=config)
        self._event_sequence_key = event_sequence_key

    async def _execute(
        self,
        *,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: object,
    ) -> object:
        if name == PrefectOperationNamer().operation_name(EventStreamHandlerId()):
            flow_context = FlowRunContext.get()
            assert flow_context is not None
            sequence = flow_context.task_run_dynamic_keys.get(self._event_sequence_key, 0)
            assert isinstance(sequence, int)
            flow_context.task_run_dynamic_keys[self._event_sequence_key] = sequence + 1
            cache_key = (*cache_key, sequence)

        @task
        async def operation(operation_name: str, *logical_inputs: object) -> object:
            return await body()

        options = cast(TaskConfig, config or {})
        return await operation.with_options(name=name, **options)(name, *cache_key)
