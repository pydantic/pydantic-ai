from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, cast
from weakref import WeakKeyDictionary

from dbos import DBOS, SetWorkflowTimeout
from dbos._error import DBOSWorkflowCancelledError, DBOSWorkflowConflictIDError
from pydantic import TypeAdapter, ValidationError

from pydantic_ai._cancel import provide_run_deadline
from pydantic_ai._utils import now_utc
from pydantic_ai.agent import EventStreamHandler, ParallelExecutionMode
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import AbstractCapability, WrapRunHandler
from pydantic_ai.durable_exec._base import (
    BaseDurabilityCapability,
    capability_selects_child_workflow,
    conversation_id_from_run_id,
    selects_child_workflow,
    toolset_selects_child_workflow,
)
from pydantic_ai.durable_exec._codec import IDENTITY_CODEC
from pydantic_ai.durable_exec._operation import EventStreamHandlerParams, ToolsetCallToolId, ToolsetKind
from pydantic_ai.durable_exec._run_context import DurableRunContext
from pydantic_ai.durable_exec._spec import DurabilityEngineSpec
from pydantic_ai.durable_exec._toolset import CallToolResult, DurableFunctionToolset, wrap_tool_call_result
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolsetTool

from ._agent import DBOSParallelExecutionMode
from ._operation_backend import DBOSBoundOperation, DBOSOperationBackend, DBOSOperationConfig
from ._operation_names import DBOSOperationNamer
from ._utils import ChildWorkflowConfig, StepConfig, guard_enqueue_in_workflow, in_workflow_code, using_durable_sleep

if TYPE_CHECKING:
    from dbos._context import DBOSContext

_default_run_ids_by_step: WeakKeyDictionary[DBOSContext, dict[int, int]] = WeakKeyDictionary()
"""Agent runs that drew a default `run_id` at each step position, per workflow execution.

Runs started concurrently (e.g. with `asyncio.gather`) resolve their IDs before either starts a step,
so they share a position and need the count to tell them apart.
"""


@dataclass(init=False, kw_only=True)
class DBOSDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by routing I/O through DBOS steps.

    The capability routes model requests, MCP I/O, and optionally event-stream
    handling through DBOS steps when the agent runs inside a DBOS workflow. Call
    `agent.run()` inside your own `@DBOS.workflow` to make that run durable;
    outside a workflow the capability is transparent and the run is a normal,
    non-durable agent run.

    The capability discovers the agent's model, name, and toolsets
    automatically via `for_agent()`.

    Example:
        ```python {test="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.dbos import DBOSDurability

        durability = DBOSDurability()
        agent = Agent('openai:gpt-5.6-sol', name='my_agent', capabilities=[durability])
        ```
    """

    engine_spec = DurabilityEngineSpec(
        engine_name='DBOS',
        durable_unit_noun='step',
        durable_unit_plural='steps',
        durable_container_noun='workflow',
        codec=IDENTITY_CODEC,
        unsupported_runtime_toolset_kinds=frozenset({'mcp', 'dynamic'}),
        wrapped_toolset_kinds=frozenset({'mcp', 'dynamic'}),
        toolset_lifecycles={'mcp': 'enter-in-durable-unit', 'dynamic': 'enter-never'},
        tool_call_result_upgrade_lenient=True,
        journal_discovery=True,
        sequential_tools_in_durable_context=False,
        tool_config_key=None,
        # `DBOS.cancel_workflow()` aborts the run with one of these `BaseException`s rather than a
        # `CancelledError`: `DBOSWorkflowCancelledError` when a step starts (or a preemptible step is
        # aborted), and, since dbos 3.2, `DBOSWorkflowConflictIDError` when a step that was running as
        # the workflow was cancelled tries to record its result after the cancel handed ownership off.
        cancellation_error_types=(DBOSWorkflowCancelledError, DBOSWorkflowConflictIDError),
    )
    # No `tool_config_key`: DBOS takes no per-tool step config, and tool metadata other than
    # `{'dbos': {'child_workflow': ...}}` is ignored (as it was before this capability existed). It
    # can't be supported without changing durable history: a step is registered once per name, and
    # DBOS tool-call step names deliberately carry no tool name (every tool in a toolset shares one
    # step), so per-tool config would be first-tool-wins.

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        model_step_config: StepConfig | None = None,
        event_stream_handler_step_config: StepConfig | None = None,
        mcp_step_config: StepConfig | None = None,
        parallel_execution_mode: DBOSParallelExecutionMode = 'parallel_ordered_events',
        register_legacy_workflows: bool = False,
    ):
        """Create a DBOSDurability capability.

        The agent's model, name, and toolsets are discovered automatically.

        Args:
            models: Optional additional models keyed by ID for runtime model
                switching. The agent's primary model is always registered as
                `'default'`. A `Model` instance can't be serialized across the
                step boundary, so a run-time model (via `agent.run(model=...)`
                / `agent.override(model=...)`, or swapped in by an outer capability)
                has to be registered here and referenced by key (or passed as the
                registered instance); an unregistered instance is rejected, because
                rebuilding it from its `model_id` would build a different model.
                Model-name strings never need registering: they cross as the string
                the caller wrote and are built inside the step by the agent's
                `resolve_model_id` capability chain, then `infer_model`. To build a
                specific instance inside the step from such a string — a custom
                provider, or per-user credentials carried on `deps` — use the
                [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] capability.
            event_stream_handler: Optional event stream handler. Model events are handled
                live inside model-request steps, and each tool event is handled in its own
                event-handler step. Tool emissions reserve that step's position during `ctx.emit()`;
                the handler still runs at stream position, without blocking the emitter.
            name: Unique agent name used in the DBOS step names. Defaults to the agent's
                `name` when the capability is bound.
            model_step_config: DBOS step config for model request steps.
            event_stream_handler_step_config: DBOS step config for event stream handler steps.
            mcp_step_config: DBOS step config for MCP server steps.
            parallel_execution_mode: Tool-call execution mode applied for the duration
                of every run. Defaults to `'parallel_ordered_events'` so events
                replay deterministically. Set to `'sequential'` for strict ordering.
                A run with a workspace always runs its tool calls sequentially.
            register_legacy_workflows: Register the workflow names used by the deprecated
                `DBOSAgent` so in-flight wrapper-era workflows can recover during migration.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self._model_step_config = model_step_config or {}
        self._event_stream_handler_step_config = event_stream_handler_step_config or {}
        self._mcp_step_config = mcp_step_config or {}
        self._parallel_execution_mode: ParallelExecutionMode = cast(ParallelExecutionMode, parallel_execution_mode)
        self._register_legacy_workflows = register_legacy_workflows
        # Populated by for_agent when the capability is attached to an agent.
        self._legacy_run_workflow: Any = None
        self._legacy_run_sync_workflow: Any = None
        self._operation_backend: DBOSOperationBackend | None = None
        self._event_step_contexts: dict[tuple[str | None, str | None], dict[int, deque[DBOSContext | None]]] = {}
        self._child_workflow_toolsets: dict[str, DurableFunctionToolset[AgentDepsT]] = {}
        self._child_workflow_metadata_capability = False
        self._init_legacy_context_vars()

    def _init_legacy_context_vars(self) -> None:
        # A wrapper-era workflow recorded `event_stream_handler=` as a workflow input; the legacy
        # workflows stash it here so the model-request steps deliver model events to it live,
        # exactly like the wrapper's `ContextVar`-stashed per-run handler.
        self._legacy_run_event_stream_handler: ContextVar[EventStreamHandler[AgentDepsT] | None] = ContextVar(
            '_legacy_run_event_stream_handler', default=None
        )
        # Whether the current run entered through a legacy `{name}.run`/`{name}.run_sync` workflow,
        # whose recorded step sequence must be preserved on recovery.
        self._in_legacy_workflow: ContextVar[bool] = ContextVar('_in_legacy_workflow', default=False)

    def _effective_event_stream_handler(self) -> EventStreamHandler[AgentDepsT] | None:
        return self._legacy_run_event_stream_handler.get() or self._event_stream_handler

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        # `for_agent` shallow-copies the user's instance, so without fresh `ContextVar`s here,
        # one capability instance attached to several agents would leak one agent's per-run
        # legacy state into another's runs.
        self._init_legacy_context_vars()
        self._operation_backend = DBOSOperationBackend(
            agent_name=self.name,
            config=DBOSOperationConfig(
                model=self._model_step_config,
                event=self._event_stream_handler_step_config,
                tool=self._mcp_step_config,
            ),
        )
        self._event_step_contexts = {}
        self._bound_model_operations = self._bind_model_operations(
            self._operation_backend, model_id=None, model_name='default'
        )
        if self._event_stream_handler is not None:
            event = self._bind_event_operation(self._operation_backend)
            assert isinstance(event, DBOSBoundOperation)
            self._bound_event_operation = event

        # --- MCP toolset wrapping ---
        self._register_toolsets(agent)
        self._child_workflow_metadata_capability = False
        agent.root_capability.apply(self._detect_child_workflow_metadata)
        self._child_workflow_toolsets = {}
        for toolset in agent.toolsets:
            toolset.apply(self._register_child_workflow)

        if self._register_legacy_workflows:
            # A wrapper-era workflow recorded only model and MCP steps: `DBOSAgent` delivered model
            # events to the handler live inside the `__model.request_stream` step, and graph-level
            # events with a *direct* workflow-level handler call that consumed no step at all.
            # Legacy runs flag themselves via `_in_legacy_workflow` so `_dispatch_event_stream_event`
            # mirrors that delivery — routing graph events through the `__event_stream_handler` step
            # would insert step ids the recording doesn't have and fail recovery with
            # `DBOSUnexpectedStepError`.

            @DBOS.workflow(name=f'{self.name}.run')
            async def legacy_run_workflow(*args: Any, **kwargs: Any) -> AgentRunResult[Any]:
                handler = kwargs.pop('event_stream_handler', None)
                legacy_token = self._in_legacy_workflow.set(True)
                token = self._legacy_run_event_stream_handler.set(handler) if handler is not None else None
                try:
                    return await agent.run(*args, **kwargs)
                finally:
                    self._in_legacy_workflow.reset(legacy_token)
                    if token is not None:
                        self._legacy_run_event_stream_handler.reset(token)

            self._legacy_run_workflow = legacy_run_workflow

            @DBOS.workflow(name=f'{self.name}.run_sync')
            def legacy_run_sync_workflow(*args: Any, **kwargs: Any) -> AgentRunResult[Any]:
                handler = kwargs.pop('event_stream_handler', None)
                legacy_token = self._in_legacy_workflow.set(True)
                token = self._legacy_run_event_stream_handler.set(handler) if handler is not None else None
                try:
                    return agent.run_sync(*args, **kwargs)
                finally:
                    self._in_legacy_workflow.reset(legacy_token)
                    if token is not None:
                        self._legacy_run_event_stream_handler.reset(token)

            self._legacy_run_sync_workflow = legacy_run_sync_workflow

    @property
    def in_durable_context(self) -> bool:
        return in_workflow_code()

    def get_default_run_id(self) -> str | None:
        if not self.in_durable_context:
            return None
        from dbos._context import get_local_dbos_context

        context = get_local_dbos_context()
        assert context is not None and context.workflow_id is not None
        step = context.function_id + 1
        runs = _default_run_ids_by_step.setdefault(context, {})
        index = runs.get(step, 0)
        runs[step] = index + 1
        # The first run at a position keeps the ID it had before concurrent runs were told apart.
        return f'{context.workflow_id}:{step}' if index == 0 else f'{context.workflow_id}:{step}:{index}'

    def get_default_conversation_id(self, run_id: str) -> str | None:
        return conversation_id_from_run_id(run_id) if self.in_durable_context else None

    async def get_run_start_time(self, run_id: str) -> datetime | None:
        if not self.in_durable_context:
            return None

        # Recorded as a step, so a recovered workflow counts the run's `timeout=` from when it first started.
        # A coroutine function, so DBOS records the step without blocking the event loop.
        async def run_start_time() -> datetime:
            return now_utc()

        return await DBOS.run_step_async({'name': f'{self.name}.run_start_time'}, run_start_time)

    def _durable_run_context(self, ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
        # A DBOS step degrades to a plain inline call outside a workflow, where enqueueing is
        # safe, so only guard once actually inside a workflow.
        return guard_enqueue_in_workflow(ctx)

    def get_durable_operation_backend(self) -> DBOSOperationBackend:
        assert self._operation_backend is not None
        return self._operation_backend

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        wrapped = super().get_wrapper_toolset(toolset)
        swapped = False

        def swap(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            nonlocal swapped
            child = self._child_workflow_toolset(ts)
            swapped = swapped or child is not None
            return child or ts

        replaced = (wrapped or toolset).visit_and_replace(swap)
        return replaced if swapped else wrapped

    def _child_workflow_toolset(self, ts: AbstractToolset[AgentDepsT]) -> DurableFunctionToolset[AgentDepsT] | None:
        """The wrapper that dispatches `ts`'s `child_workflow` tools; toolsets without them stay unwrapped, as before."""
        if not isinstance(ts, FunctionToolset):
            return None
        if not self._child_workflow_metadata_capability and not toolset_selects_child_workflow(ts, 'dbos'):
            return None
        child = self._child_workflow_toolsets.get(ts._explicit_id or '')  # pyright: ignore[reportPrivateUsage]
        if child is not None and child.wrapped is ts:
            return child
        return DurableFunctionToolset(
            ts,
            in_durable_context=in_workflow_code,
            call_tool_operation=_reject_unregistered_child_workflow,
            resolve_tool_config=_child_workflow_config,
            lifecycle='enter-always',
        )

    def _detect_child_workflow_metadata(self, capability: AbstractCapability[AgentDepsT]) -> None:
        # `SetToolMetadata` can select tools in any toolset, so it enables child workflows for all of them.
        if capability_selects_child_workflow(capability, 'dbos'):
            self._child_workflow_metadata_capability = True

    def _register_child_workflow(self, toolset: AbstractToolset[AgentDepsT]) -> None:
        """Register the DBOS workflow that runs `toolset`'s tools with `child_workflow` metadata."""
        toolset_id = toolset._explicit_id  # pyright: ignore[reportPrivateUsage]
        if (
            not isinstance(toolset, FunctionToolset)
            or toolset_id is None
            or toolset_id in self._child_workflow_toolsets
            or not (self._child_workflow_metadata_capability or toolset_selects_child_workflow(toolset, 'dbos'))
        ):
            # DBOS hashes registered workflows into the application version, so only opted-in toolsets register one.
            return
        operation_name = DBOSOperationNamer(self.name).operation_name(
            ToolsetCallToolId('function', toolset_id=toolset_id)
        )

        @DBOS.workflow(name=f'{operation_name}__child_workflow')
        async def call_tool_workflow(*args: Any) -> CallToolResult:
            # DBOS hashes registered workflow source into the application version, so the body lives elsewhere.
            return await self._call_tool_in_child_workflow(toolset, *args)

        async def call_tool_operation(
            name: str,
            tool_args: dict[str, Any],
            *,
            ctx: RunContext[AgentDepsT],
            tool: ToolsetTool[AgentDepsT],
            config: Mapping[str, Any],
        ) -> Any:
            assert isinstance(tool, FunctionToolsetTool)  # `_child_workflow_config` admits only these
            timeout = config.get('workflow_timeout_sec')
            with SetWorkflowTimeout(timeout) if timeout is not None else nullcontext():
                payload = await call_tool_workflow(
                    name,
                    tool_args,
                    tool.tool_def,
                    tool.original_name,
                    DurableRunContext.serialize_run_context(ctx),
                    ctx.deps,
                )
            return self._unwrap_tool_result(payload)

        self._child_workflow_toolsets[toolset_id] = DurableFunctionToolset(
            toolset,
            in_durable_context=in_workflow_code,
            call_tool_operation=call_tool_operation,
            resolve_tool_config=_child_workflow_config,
            lifecycle='enter-always',
            durable_registrations=[call_tool_workflow],
        )

    async def _call_tool_in_child_workflow(
        self,
        toolset: FunctionToolset[AgentDepsT],
        tool_name: str,
        tool_args: dict[str, Any],
        tool_def: ToolDefinition,
        original_name: str | None,
        serialized_ctx: dict[str, Any],
        deps: Any,
    ) -> CallToolResult:
        ctx = DurableRunContext.restore(serialized_ctx, deps=deps, agent=self._agent, engine_spec=self.engine_spec)
        # The tool was prepared in the parent, against the full run context.
        tool = toolset.tool_for_tool_def(tool_def, ctx=ctx, original_name=original_name)
        # A recovered child doesn't run inside the parent's run, so nested runs inherit its deadline from `ctx`.
        with (
            provide_run_deadline(ctx.deadline) if ctx.deadline is not None else nullcontext(),
            self._tool_run_context_scope(ctx) as durable_ctx,
        ):
            return await wrap_tool_call_result(toolset.call_tool(tool_name, tool_args, durable_ctx, tool))

    def _toolset_base_config(self, kind: ToolsetKind) -> StepConfig:
        return self._mcp_step_config

    def _toolset_in_durable_context(self) -> bool:
        # DBOS steps degrade to inline calls outside a workflow, preserving the wrapper-era lifecycle.
        return True

    def _prepare_event_dispatch(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        reservations = self._event_step_contexts.get((DBOS.workflow_id, ctx.run_id))
        if reservations is not None:
            # Reserve in buffer order, before immediate listeners can yield or emit another event.
            step_ctx = None
            if ctx.tool_call_id is not None and self.in_durable_context:
                # TODO: Replace the private reservation API: https://github.com/dbos-inc/dbos-transact-py/issues/915
                from dbos._context import snapshot_step_context

                step_ctx = snapshot_step_context()
            reservations.setdefault(id(event), deque()).append(step_ctx)

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        if self._in_legacy_workflow.get():
            # Wrapper-era recordings contain no `__event_stream_handler` steps (the wrapper called
            # the handler directly in workflow code), so a legacy run must do the same to keep the
            # recorded step sequence replayable. The handler runs at workflow level here, not inside
            # a step, so the enqueue guard doesn't apply — matching how the wrapper delivered it.
            handler = self._effective_event_stream_handler()
            assert handler is not None
            await handler(ctx, self._single_event_stream(event))
            return
        reservations = self._event_step_contexts.get((DBOS.workflow_id, ctx.run_id))
        if reservations is not None and (pending := reservations.get(id(event))):
            step_ctx = pending.popleft()
            if not pending:
                del reservations[id(event)]
            if step_ctx is not None:
                event_operation = self._bound_event_operation
                assert isinstance(event_operation, DBOSBoundOperation)
                await event_operation.call_reserved(step_ctx, EventStreamHandlerParams(event, run_context=ctx))
                return
        await super()._dispatch_event_stream_event(ctx, event)

    # --- Capability hooks ---

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Apply the configured parallel-execution mode, and make agent-graph waits durable in workflows."""
        agent = self._agent
        if agent is None:  # pragma: no cover
            return await handler()
        # DBOS numbers steps as they start; parallel tools making several workspace steps each could
        # replay recorded results into the wrong calls on recovery.
        mode = self._parallel_execution_mode
        if self.in_durable_context and ctx.workspace.attached:
            mode = 'sequential'
        key = (DBOS.workflow_id, ctx.run_id)
        reserve_events = (
            self.in_durable_context and self._event_stream_handler is not None and not self._in_legacy_workflow.get()
        )
        if reserve_events:
            self._event_step_contexts[key] = {}
        try:
            with agent.parallel_tool_call_execution_mode(mode), using_durable_sleep():
                return await handler()
        finally:
            if reserve_events:
                self._event_step_contexts.pop(key)


_child_workflow_config_adapter = TypeAdapter(ChildWorkflowConfig)


def _child_workflow_config(tool: ToolsetTool[Any] | None, name: str) -> ChildWorkflowConfig | Literal[False]:
    if tool is None or not selects_child_workflow(tool.tool_def.metadata, 'dbos'):
        return False
    config = (tool.tool_def.metadata or {})['dbos']
    if len(config) != 1:
        raise UserError(f'Tool {name!r} has invalid DBOS metadata: `child_workflow` must be the only key.')
    try:
        child_config = _child_workflow_config_adapter.validate_python(config['child_workflow'])
    except ValidationError as e:
        raise UserError(f'Invalid DBOS `ChildWorkflowConfig` in tool {name!r} metadata: {e}') from e
    if not (isinstance(tool, FunctionToolsetTool) and tool.is_async):
        raise UserError(
            f'DBOS metadata for tool {name!r} selects a child workflow, but only async function tools can run as one.'
        )
    return child_config


async def _reject_unregistered_child_workflow(
    name: str, tool_args: dict[str, Any], *, ctx: RunContext[Any], tool: ToolsetTool[Any], config: Mapping[str, Any]
) -> Any:
    raise UserError(
        f'Tool {name!r} selects a DBOS child workflow, but its toolset has no registered child workflow. '
        'Give the `FunctionToolset` a unique `id`, and select the child workflow before the agent is constructed, '
        "in the tool's metadata or with a `SetToolMetadata` capability."
    )
