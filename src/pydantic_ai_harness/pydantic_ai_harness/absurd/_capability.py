"""Durable execution for Pydantic AI agents on the Absurd engine.

Absurd (`absurd-sdk`) is a Postgres-based durable-execution engine. This module
checkpoints an agent's I/O -- model requests, MCP calls, and function tool calls
-- into Absurd steps (`ctx.step(...)`), so a worker crash mid-run resumes from
the last completed step instead of restarting the run.

The step names and checkpoint payloads match the `pydantic-ai-absurd` package by
Marcelo Trylesinski, so a run started under one package can resume under the other.
"""

from __future__ import annotations

try:
    import absurd_sdk  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `absurd-sdk` package to use the Absurd durability capability, '
        'you can use the `absurd` optional group -- `pip install "pydantic-ai-harness[absurd]"`'
    ) from _import_error

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic_ai.agent import EventStreamHandler, ParallelExecutionMode
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities import WrapRunHandler
from pydantic_ai.durable_exec import JSON_CODEC, BaseDurabilityCapability, DurabilityEngineSpec
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset

from ._context import ENGINE_NAME, current_async_task_context
from ._operation_backend import AbsurdOperationBackend

AbsurdParallelExecutionMode = Literal['sequential', 'parallel_ordered_events']
"""Tool-call execution modes usable with Absurd. A subset of `ParallelExecutionMode`.

Absurd disambiguates repeated step names by encounter order (the second `ctx.step(name, ...)` for a
given `name` records under `name#2`, the third under `name#3`, ...). A replay lines up with its
checkpoints only if each repeated step name claims the same slot it did on the first run.

The slot is claimed synchronously when `ctx.step(...)` is entered, before the step body runs. Tool
calls are scheduled in the model's tool-call order under both parallel modes, so their step names are
assigned in that order regardless of which call finishes first -- completion order does not move a
tool call's slot. `'parallel'` is nonetheless excluded because it emits tool-result events (and so
the per-event `event_stream_handler` steps) in completion order, which races and could assign one of
those repeated step names a different slot on replay. `'parallel_ordered_events'` emits those events
in the model's tool-call order once the whole batch completes, so every repeated step name -- tool
calls and event-handler steps alike -- lines up on replay."""


@dataclass(init=False)
class AbsurdDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by checkpointing its I/O into Absurd steps.

    Attach it via `capabilities=[AbsurdDurability()]` and call `agent.run()` inside an Absurd
    task handler: every model request, MCP call, and function tool call is wrapped in
    `ctx.step(...)`, so a worker crash mid-run resumes from the last completed step instead of
    restarting. A completed step is served from its checkpoint on
    replay instead of being recomputed, so tokens are not re-spent on work that already finished.
    A step is checkpointed after it runs, so a crash between a tool's side effect and its checkpoint
    re-runs the tool on recovery: keep tool side effects idempotent. Outside a task the capability
    is transparent and the run is a normal, non-durable agent run.

    The capability discovers the agent's model, name, and toolsets automatically when it is bound
    to the agent. Step results are stored in Postgres as JSON, so a checkpointed tool's return
    value must be JSON-serializable.

    Step names and checkpoint payloads match the `pydantic-ai-absurd` package, so a task started
    under it resumes under this capability.

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
        # `pydantic-ai-absurd` checkpoints function and MCP toolsets only; dynamic toolsets run as-is.
        wrapped_toolset_kinds=frozenset({'function', 'mcp'}),
        toolset_lifecycles={'function': 'enter-never', 'mcp': 'enter-never'},
        unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
    )
    # Absurd has no raise-time non-retryable exception equivalent to Lambda's `ExecutionError` or
    # Restate's `TerminalError`. Serialization failures therefore use the base behavior: the task's
    # `RetryStrategy`/`max_attempts` governs retries, and the task fails after those attempts are exhausted.

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        parallel_execution_mode: AbsurdParallelExecutionMode = 'sequential',
    ) -> None:
        """Create an `AbsurdDurability` capability.

        The agent's model, name, and toolsets are discovered automatically.

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
            parallel_execution_mode: Tool-call execution mode applied to every run. Defaults to
                `'sequential'`. `'parallel'` is excluded by type because it emits tool-result and
                event-handler steps in completion order, which races with Absurd's encounter-order
                step naming; see `AbsurdParallelExecutionMode` for the full invariant.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self._parallel_execution_mode: ParallelExecutionMode = parallel_execution_mode
        self._id_less_wrappers: dict[int, AbstractToolset[AgentDepsT]] = {}

    @property
    def in_durable_context(self) -> bool:
        return current_async_task_context() is not None

    def get_durable_operation_backend(self) -> AbsurdOperationBackend:
        return AbsurdOperationBackend(
            agent_name=self.name,
            default_model_id=self.default_model_id,
        )

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        self._id_less_wrappers = {}
        super()._bind_to_agent(agent)

    def _wrap_and_register_leaf(self, ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # `pydantic-ai-absurd` accepts toolsets without an `id` and keys their wrappers by instance.
        if ts.id is not None:
            return super()._wrap_and_register_leaf(ts)
        if (existing := self._id_less_wrappers.get(id(ts))) is not None:
            return existing
        wrapped = self._wrap_leaf_toolset(ts)
        if wrapped is None:
            return ts
        self._id_less_wrappers[id(ts)] = wrapped
        return wrapped

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        wrapped = super().get_wrapper_toolset(toolset)
        if not self._id_less_wrappers:
            return wrapped
        return (wrapped or toolset).visit_and_replace(lambda ts: self._id_less_wrappers.get(id(ts), ts))

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Apply the configured parallel-execution mode for every run, as `pydantic-ai-absurd` does."""
        agent = self.agent
        assert agent is not None
        with agent.parallel_tool_call_execution_mode(self._parallel_execution_mode):
            return await handler()
