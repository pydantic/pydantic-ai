from __future__ import annotations

from collections.abc import Awaitable, Callable

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai.durable_exec._operation import CapabilityOperationId, DurableOperationId, EventStreamHandlerId
from pydantic_ai.durable_exec._operation_backend import CallableOperationBackend, RoleBasedOperationConfig
from pydantic_ai.durable_exec._workspace import WORKSPACE_OPERATION_ID
from pydantic_ai.tools import RunContext

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
            # Match the singleton bound by _bind_workspace_operation: equal custom capability IDs
            # still use the custom operation's projection and counter namespace.
            if operation_id is WORKSPACE_OPERATION_ID:
                # The projection carries the workspace's own run ID, including after the run ends.
                # Concurrent runs can arrive in a different order when a flow retries.
                run_id = cache_key[-1]
                assert isinstance(run_id, str)
                sequence_key = f'{self._event_sequence_key}:workspace:{len(run_id)}:{run_id}'
            else:
                # CapabilityCacheIdentity.project supplies the trailing RunContext. Scope the
                # counter to its run so reordered runs replay their own identical operations.
                run_context = cache_key[-1]
                assert isinstance(run_context, RunContext)
                run_id = run_context.run_id
                assert run_id is not None
                capability_id = operation_id.capability_id
                sequence_key = (
                    f'pydantic_ai_capability_sequence:{len(self._event_sequence_key)}:{self._event_sequence_key}:'
                    f'{len(run_id)}:{run_id}:{len(capability_id)}:{capability_id}{operation_id.operation}'
                )

        if sequence_key is not None:
            flow_context = FlowRunContext.get()
            assert flow_context is not None
            # Each counter is rebuilt on flow retry. Its ordinal distinguishes identical calls;
            # separate operation counters keep unrelated calls from shifting replay identities.
            # Include the counter key so distinct agents and capability/workspace runs cannot share results.
            sequence = flow_context.task_run_dynamic_keys.get(sequence_key, 0)
            assert isinstance(sequence, int)
            flow_context.task_run_dynamic_keys[sequence_key] = sequence + 1
            cache_key = (*cache_key, sequence_key, sequence)

        @task
        async def operation(operation_name: str, *logical_inputs: object) -> object:
            return await body()

        options = config or {}
        return await operation.with_options(name=name, **options)(name, *cache_key)
