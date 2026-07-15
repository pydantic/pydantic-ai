from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from dbos import DBOS

from pydantic_ai import messages as _messages
from pydantic_ai._run_context import set_current_run_context
from pydantic_ai.agent import ParallelExecutionMode
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import (
    CapabilityOrdering,
    WrapModelRequestHandler,
    WrapRunHandler,
)
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._runtime_toolsets import reject_unsupported_runtime_toolsets
from pydantic_ai.durable_exec._utils import (
    DurableModel,
    StreamedActivityResult,
    process_event_stream,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

from ._agent import DBOSParallelExecutionMode
from ._utils import StepConfig


@dataclass(init=False)
class DBOSDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by routing I/O through DBOS steps.

    When added to an agent, this capability intercepts model requests and
    optionally wraps MCP toolsets to route their I/O through DBOS steps.
    Outside of DBOS workflows, the capability is transparent.

    The capability discovers the agent's model, name, and toolsets
    automatically via `for_agent()`.

    Example:
        ```python {test="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.dbos import DBOSDurability

        durability = DBOSDurability()
        agent = Agent('openai:gpt-5.2', name='my_agent', capabilities=[durability])
        ```
    """

    engine_name = 'DBOS'

    name: str
    """Unique agent name used as a prefix for DBOS step names."""

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        model_step_config: StepConfig | None = None,
        mcp_step_config: StepConfig | None = None,
        parallel_execution_mode: DBOSParallelExecutionMode = 'parallel_ordered_events',
    ):
        """Create a DBOSDurability capability.

        The agent's model, name, and toolsets are discovered automatically.

        Args:
            models: Optional additional models keyed by ID for runtime model
                switching. The agent's primary model is always registered as
                `'default'`. A `Model` instance can't be serialized across the
                step boundary, so a run-time model (via `agent.run(model=...)`
                / `agent.override(model=...)`, or swapped in by an outer capability)
                is sent as its `model_id` string and rebuilt inside the step by
                registry lookup, then the agent's `resolve_model_id` capability
                chain / `infer_model`. Register an instance here (and reference it
                by key or pass the registered instance) whenever its `model_id`
                alone wouldn't rebuild it faithfully — e.g. a custom provider,
                client, or settings. Model-name strings never need registering;
                to customize how they're built (e.g. a custom provider), use the
                [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] capability.
            model_step_config: DBOS step config for model request steps.
            mcp_step_config: DBOS step config for MCP server steps.
            parallel_execution_mode: Tool-call execution mode applied for the duration
                of every run. Defaults to `'parallel_ordered_events'` so events
                replay deterministically. Set to `'sequential'` for strict ordering.
        """
        super().__init__(models=models)
        self.name = ''
        self._agent: AbstractAgent[Any, Any] | None = None
        self._model_step_config = model_step_config or {}
        self._mcp_step_config = mcp_step_config or {}
        self._parallel_execution_mode: ParallelExecutionMode = cast(ParallelExecutionMode, parallel_execution_mode)
        self._dbos_toolsets_by_id: dict[str, WrapperToolset[Any]] = {}
        # Populated by for_agent when the capability is attached to an agent.
        self._request_step: Any = None
        self._request_stream_step: Any = None
        self._cancel_suspended_response_step: Any = None
        self._auto_run_workflow: Callable[..., Awaitable[Any]] | None = None
        self._auto_run_sync_workflow: Callable[..., Any] | None = None

    @classmethod
    def from_agent(cls, agent: AbstractAgent[Any, Any]) -> DBOSDurability[Any] | None:
        """Return the bound `DBOSDurability` on an agent, walking its capability chain."""
        found: list[DBOSDurability[Any]] = []

        def visitor(cap: Any) -> None:
            if isinstance(cap, cls):
                found.append(cap)

        agent.root_capability.apply(visitor)
        return found[0] if found else None

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> DBOSDurability[AgentDepsT]:
        """Bind to the agent: discover model, name, toolsets and register DBOS steps.

        Returns a new bound instance; the original capability is left pristine so the
        same instance can be passed to multiple agents.
        """
        if not agent.name:
            raise UserError('An agent needs to have a unique `name` in order to be used with DBOS.')

        bound = copy.copy(self)
        bound.name = agent.name
        bound._agent = agent

        bound._dbos_toolsets_by_id = {}

        # Build model registry (shared with the other durability capabilities)
        bound._bind_models(agent)

        # --- Model request steps ---

        @DBOS.step(name=f'{bound.name}__model.request', **bound._model_step_config)
        async def request_step(
            model_id: str | None,
            messages: list[_messages.ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any],
        ) -> ModelResponse:
            model = await bound._resolve_model_for_request(model_id, run_context)
            with set_current_run_context(run_context):
                return await model.request(messages, model_settings, model_request_parameters)

        bound._request_step = request_step

        @DBOS.step(name=f'{bound.name}__model.request_stream', **bound._model_step_config)
        async def request_stream_step(
            model_id: str | None,
            messages: list[_messages.ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any],
        ) -> StreamedActivityResult:
            model = await bound._resolve_model_for_request(model_id, run_context)
            request_context = ModelRequestContext(
                model=model,
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
            )
            with set_current_run_context(run_context):
                async with model.request_stream(
                    messages, model_settings, model_request_parameters, run_context
                ) as streamed_response:
                    events = await process_event_stream(
                        run_context=run_context,
                        request_context=request_context,
                        stream=streamed_response,
                    )
            return StreamedActivityResult(response=streamed_response.get(), events=events)

        bound._request_stream_step = request_stream_step

        @DBOS.step(name=f'{bound.name}__model.cancel_suspended_response', **bound._model_step_config)
        async def cancel_suspended_response_step(
            model_id: str | None, response: ModelResponse, run_context: RunContext[Any]
        ) -> None:
            model = await bound._resolve_model_for_request(model_id, run_context)
            with set_current_run_context(run_context):
                await model.cancel_suspended_response(response)

        bound._cancel_suspended_response_step = cancel_suspended_response_step

        # --- MCP toolset wrapping ---
        for toolset in agent.toolsets:
            bound._dbosify_leaf_toolsets(toolset)

        # --- Auto-wrap agent.run / run_sync as DBOS workflows ---
        bound._install_workflow_wrappers(agent)

        return bound

    def _install_workflow_wrappers(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        """Replace `agent.run` and `agent.run_sync` with DBOS-workflow-decorated wrappers.

        When called outside an active DBOS workflow, the wrapper enters a workflow so
        all model and toolset steps recorded inside become durable. When already inside a
        workflow, it skips the redundant wrap and just applies the configured parallel
        execution mode.

        `agent.iter` is wrapped in-place by `wrap_run` instead — its async-generator
        signature can't be cleanly forwarded through `@DBOS.workflow`.

        Idempotent: if `for_agent` is called twice (e.g. an agent is bound to two
        `DBOSDurability` instances by mistake), the second call is a no-op rather
        than stacking wrappers.
        """
        if getattr(agent.run, '_pydantic_ai_dbos_wrapped', False):
            return
        original_run = agent.run
        original_run_sync = agent.run_sync
        parallel_mode = self._parallel_execution_mode

        @DBOS.workflow(name=f'{self.name}.run')
        async def _auto_run_workflow(*args: Any, **kwargs: Any) -> Any:
            with agent.parallel_tool_call_execution_mode(parallel_mode):
                return await original_run(*args, **kwargs)

        @DBOS.workflow(name=f'{self.name}.run_sync')
        def _auto_run_sync_workflow(*args: Any, **kwargs: Any) -> Any:
            with agent.parallel_tool_call_execution_mode(parallel_mode):
                return original_run_sync(*args, **kwargs)

        self._auto_run_workflow = _auto_run_workflow
        self._auto_run_sync_workflow = _auto_run_sync_workflow

        def _reject_runtime_toolsets_kwarg(kwargs: dict[str, Any]) -> None:
            # Fast path for the explicit `run(toolsets=...)` kwarg: it must be rejected
            # *before* `_auto_run_workflow` hands the kwargs to DBOS, which pickles workflow
            # inputs — an executing toolset (e.g. an `MCPToolset` holding a live client)
            # isn't picklable, so letting it through would surface an opaque serialization
            # error instead of the `UserError`. Toolsets that arrive through other routes
            # (per-run capabilities/specs, `iter`) are caught by `_reject_runtime_toolsets`
            # in run setup.
            reject_unsupported_runtime_toolsets(
                kwargs.get('toolsets'), unsupported_kinds=frozenset({'mcp', 'dynamic'}), engine='DBOS'
            )

        async def patched_run(*args: Any, **kwargs: Any) -> Any:
            _reject_runtime_toolsets_kwarg(kwargs)
            if DBOS.workflow_id is not None and DBOS.step_id is None:
                # Already inside a DBOS workflow — skip the auto-wrap and just apply the mode.
                with agent.parallel_tool_call_execution_mode(parallel_mode):
                    return await original_run(*args, **kwargs)
            return await _auto_run_workflow(*args, **kwargs)

        def patched_run_sync(*args: Any, **kwargs: Any) -> Any:
            _reject_runtime_toolsets_kwarg(kwargs)
            if DBOS.workflow_id is not None and DBOS.step_id is None:  # pragma: lax no cover
                with agent.parallel_tool_call_execution_mode(parallel_mode):
                    return original_run_sync(*args, **kwargs)
            return _auto_run_sync_workflow(*args, **kwargs)

        patched_run._pydantic_ai_dbos_wrapped = True  # pyright: ignore[reportFunctionMemberAccess]
        agent.run = patched_run
        agent.run_sync = patched_run_sync

    def _reject_runtime_toolsets(self, toolset: AbstractToolset[AgentDepsT]) -> None:
        """Reject executing toolsets added per-run inside a workflow.

        The run toolset assembled by the agent contains construction-time toolsets (whose
        MCP I/O `for_agent` wrapped as steps) plus any per-run extras — `run(toolsets=...)`,
        or toolsets contributed by per-run capabilities/specs. An executing extra would run
        un-checkpointed inside the workflow and re-execute on recovery, so it's rejected
        explicitly, like the deprecated `DBOSAgent` does. DBOS runs function tools inline,
        so `FunctionToolset` is allowed at runtime, as are non-executing toolsets like
        `ExternalToolset`. Checked here in run setup rather than in the patched
        `run`/`run_sync` so that `iter`/`run_stream` inside a user workflow are covered
        too; only applies inside a workflow — outside one the capability is transparent
        and any toolset is fine.
        """
        if DBOS.workflow_id is None or DBOS.step_id is not None:
            return

        construction_leaves: set[int] = set()
        if self._agent is not None:  # pragma: no branch — `for_agent` always binds before a run
            for agent_toolset in self._agent.toolsets:
                agent_toolset.apply(lambda leaf: construction_leaves.add(id(leaf)))

        runtime_leaves: list[AbstractToolset[AgentDepsT]] = []

        def collect(leaf: AbstractToolset[AgentDepsT]) -> None:
            if id(leaf) not in construction_leaves:
                runtime_leaves.append(leaf)

        toolset.apply(collect)
        reject_unsupported_runtime_toolsets(
            runtime_leaves, unsupported_kinds=frozenset({'mcp', 'dynamic'}), engine='DBOS'
        )

    def _dbosify_leaf_toolsets(self, toolset: AbstractToolset[AgentDepsT]) -> None:
        """Wrap MCP leaf toolsets as DBOS steps."""

        def dbosify(ts: AbstractToolset[Any]) -> AbstractToolset[Any]:
            try:
                from pydantic_ai.mcp import MCPToolset

                from ._mcp_toolset import DBOSMCPToolset
            except ImportError:
                pass
            else:
                if isinstance(ts, MCPToolset):
                    # Without an ID the wrapper can't be swapped in at run time (see
                    # `get_wrapper_toolset`), so the toolset's I/O would silently run
                    # un-checkpointed inside the DBOS workflow and re-execute on recovery.
                    if ts.id is None:
                        raise UserError(
                            'MCP toolsets need to have a unique `id` in order to be used with DBOS. '
                            "The ID will be used to identify the toolset's steps within the workflow."
                        )
                    existing = self._dbos_toolsets_by_id.get(ts.id)
                    if existing is not None:
                        if existing.wrapped is ts:
                            # The same toolset instance can appear in more than one place
                            # in the tree; reuse its wrapper.
                            return existing
                        # A distinct toolset under an already-registered `id` would silently
                        # replace it in the registry and route both toolsets' calls to one wrapper.
                        raise UserError(
                            f'Two toolsets have the same `id` {ts.id!r}. Toolset `id`s must be unique among all '
                            "toolsets registered with the same agent, as they identify the toolset's steps "
                            'within the workflow.'
                        )
                    wrapped = DBOSMCPToolset(
                        wrapped=ts,
                        step_name_prefix=self.name,
                        step_config=self._mcp_step_config,
                    )
                    self._dbos_toolsets_by_id[ts.id] = wrapped
                    return wrapped

            return ts

        toolset.visit_and_replace(dbosify)

    # --- Capability hooks ---

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Apply the configured parallel-execution mode for the duration of the run.

        Auto-wrapping into a DBOS workflow is handled by `_install_workflow_wrappers`
        for `run`/`run_sync`. This hook is the single chokepoint that applies the
        execution mode for every entry point — including `iter`, which the run/sync
        wrappers don't cover.
        """
        agent = self._agent
        if agent is None:  # pragma: no cover
            return await handler()
        with agent.parallel_tool_call_execution_mode(self._parallel_execution_mode):
            return await handler()

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Route model requests through DBOS steps when inside a workflow."""
        if DBOS.workflow_id is None or DBOS.step_id is not None:
            return await handler(request_context)

        # A `Model` instance can't be serialized across the step boundary, so the
        # request carries a `model_id` (None for the default, the run's original
        # model-id string, a `models=` registry key, or a model-name string) and the
        # step rebuilds the model deps-aware via `_resolve_model_for_request`.
        # A model swapped in by an outer capability's `before_model_request`
        # round-trips via `_find_model_id` on `request_context.model`.
        model_id = self._model_id_for_request(ctx, request_context)

        async def request_segment(
            messages: list[_messages.ModelMessage],
            settings: ModelSettings | None,
            parameters: ModelRequestParameters,
        ) -> ModelResponse:
            return await self._request_step(model_id, messages, settings, parameters, ctx)

        async def request_stream_segment(
            messages: list[_messages.ModelMessage],
            settings: ModelSettings | None,
            parameters: ModelRequestParameters,
        ) -> StreamedActivityResult:
            return await self._request_stream_step(model_id, messages, settings, parameters, ctx)

        async def cancel_suspended_response_segment(response: ModelResponse) -> None:
            await self._cancel_suspended_response_step(model_id, response, ctx)

        request_context.model = DurableModel(
            request_context.model,
            request_segment=request_segment,
            request_stream_segment=request_stream_segment,
            cancel_suspended_response_segment=cancel_suspended_response_segment,
        )
        request_context._hooks_already_applied = True  # pyright: ignore[reportPrivateUsage]
        return await handler(request_context)

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Replace MCP leaf toolsets with their DBOS-wrapped versions."""
        self._reject_runtime_toolsets(toolset)

        if not self._dbos_toolsets_by_id:
            return None

        def swap(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            ts_id = ts.id
            if ts_id is not None and ts_id in self._dbos_toolsets_by_id:
                return self._dbos_toolsets_by_id[ts_id]
            return ts  # pragma: lax no cover

        return toolset.visit_and_replace(swap)

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None
