"""Durable execution for Pydantic AI agents on the Absurd engine."""

from __future__ import annotations

try:
    import absurd_sdk  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:
    raise ImportError(
        'Please install the `absurd-sdk` package to use the Absurd durability capability, '
        'you can use the `absurd` optional group -- `pip install "pydantic-ai-harness[absurd]"`'
    ) from _import_error

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic_ai.agent import EventStreamHandler, ParallelExecutionMode
from pydantic_ai.capabilities import WrapRunHandler
from pydantic_ai.durable_exec import JSON_CODEC, BaseDurabilityCapability, DurabilityEngineSpec
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext

from ._context import ENGINE_NAME, current_async_task_context
from ._operation_backend import AbsurdOperationBackend

AbsurdParallelExecutionMode = Literal['sequential', 'parallel_ordered_events']
"""Tool-call execution modes usable with Absurd: a subset of `ParallelExecutionMode`. `'parallel'` is
excluded because it emits tool events in completion order, which can differ on replay."""


@dataclass(init=False)
class AbsurdDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by checkpointing its I/O into Absurd steps.

    Run the agent inside an async Absurd task handler: every model request, MCP call, and function
    tool call becomes a step, so a retried task resumes from the last completed step. Outside a task
    the capability is transparent.

    Example:
        ```python {test="skip"}
        from absurd_sdk import AsyncAbsurd, AsyncTaskContext, JsonValue
        from pydantic_ai import Agent
        from pydantic_ai_harness.absurd import AbsurdDurability

        absurd = AsyncAbsurd('postgresql://localhost/absurd', queue_name='agents')
        agent = Agent('openai:gpt-5', name='analyst', capabilities=[AbsurdDurability()])


        @absurd.register_task(name='analyse')
        async def analyse(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
            assert isinstance(params, dict)
            result = await agent.run(params['prompt'])
            return {'output': result.output}
        ```
    """

    engine_spec: ClassVar = DurabilityEngineSpec(
        engine_name=ENGINE_NAME,
        durable_unit_noun='step',
        durable_container_noun='task',
        codec=JSON_CODEC,
        wrapped_toolset_kinds=frozenset({'function', 'mcp'}),
        toolset_lifecycles={'function': 'enter-never', 'mcp': 'enter-never'},
        unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
    )
    # Absurd has no non-retryable error type, so a serialization failure retries under the task's `RetryStrategy`.

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        parallel_execution_mode: AbsurdParallelExecutionMode = 'sequential',
    ) -> None:
        """Create an `AbsurdDurability` capability.

        Args:
            models: Optional additional models keyed by ID for runtime model switching via
                `agent.run(model='<id>')`. The agent's primary model is always registered as
                `'default'`; the ID is folded into the checkpoint step name so a replay resolves
                to the same model.
            event_stream_handler: Optional event stream handler. Model events are handled live
                inside the model-request step; each tool event is handled in its own checkpointed
                step.
            name: Unique agent name used as the prefix for every checkpoint step. Defaults to the
                agent's `name` when the capability is bound.
            parallel_execution_mode: Tool-call execution mode applied to every run. `'parallel'` is
                excluded; see `AbsurdParallelExecutionMode`.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self._parallel_execution_mode: ParallelExecutionMode = parallel_execution_mode

    @property
    def in_durable_context(self) -> bool:
        return current_async_task_context() is not None

    def get_durable_operation_backend(self) -> AbsurdOperationBackend:
        return AbsurdOperationBackend(agent_name=self.name, default_model_id=self.default_model_id)

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Apply the configured parallel-execution mode to every run, inside a task or not."""
        agent = self.agent
        assert agent is not None
        with agent.parallel_tool_call_execution_mode(self._parallel_execution_mode):
            return await handler()
