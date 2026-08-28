from __future__ import annotations

from collections.abc import Awaitable, Callable

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai.durable_exec._operation import CapabilityOperationId, DurableOperationId, EventStreamHandlerId
from pydantic_ai.durable_exec._operation_backend import CallableOperationBackend, RoleBasedOperationConfig

from ._operation_names import PrefectOperationNamer
from ._types import TaskConfig

PrefectOperationConfig = RoleBasedOperationConfig[TaskConfig]


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
