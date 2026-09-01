from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from dbos import DBOS

from pydantic_ai.agent import EventStreamHandler, ParallelExecutionMode
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import WrapRunHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._codec import IDENTITY_CODEC
from pydantic_ai.durable_exec._operation import ToolsetKind
from pydantic_ai.durable_exec._spec import DurabilityEngineSpec
from pydantic_ai.durable_exec._utils import StreamedActivityResult
from pydantic_ai.messages import AgentStreamEvent, ModelResponse
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestParameters
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext

from ._agent import DBOSParallelExecutionMode
from ._operation_backend import DBOSBoundOperation, DBOSOperationBackend, DBOSOperationConfig
from ._utils import StepConfig, guard_enqueue_in_workflow

if TYPE_CHECKING:
    pass


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
        toolset_lifecycles={'mcp': 'enter-never', 'dynamic': 'enter-never'},
        tool_call_result_upgrade_lenient=True,
        journal_discovery=True,
        sequential_tools_in_durable_context=False,
        tool_config_key=None,
    )
    # No `tool_config_key`: DBOS takes no per-tool config, and tool metadata is ignored (as it was
    # before this capability existed). It can't be supported without changing durable history: a step
    # is registered once per name, and DBOS tool-call step names deliberately carry no tool name
    # (every tool in a toolset shares one step), so per-tool config would be first-tool-wins.

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
                event-handler step.
            name: Unique agent name used in the DBOS step names. Defaults to the agent's
                `name` when the capability is bound.
            model_step_config: DBOS step config for model request steps.
            event_stream_handler_step_config: DBOS step config for event stream handler steps.
            mcp_step_config: DBOS step config for MCP server steps.
            parallel_execution_mode: Tool-call execution mode applied for the duration
                of every run. Defaults to `'parallel_ordered_events'` so events
                replay deterministically. Set to `'sequential'` for strict ordering.
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
        self._bound_model_operations = self._bind_model_operations(
            self._operation_backend, model_id=None, model_name='default'
        )
        if self._event_stream_handler is not None:
            event = self._bind_event_operation(self._operation_backend)
            assert isinstance(event, DBOSBoundOperation)
            self._bound_event_operation = event

        # --- MCP toolset wrapping ---
        self._register_toolsets(agent)

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
        return DBOS.workflow_id is not None and DBOS.step_id is None

    def _durable_run_context(self, ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
        # A DBOS step degrades to a plain inline call outside a workflow, where enqueueing is
        # safe, so only guard once actually inside a workflow.
        return guard_enqueue_in_workflow(ctx)

    def get_durable_operation_backend(self) -> DBOSOperationBackend:
        assert self._operation_backend is not None
        return self._operation_backend

    def _toolset_base_config(self, kind: ToolsetKind) -> StepConfig:
        return self._mcp_step_config

    def _toolset_in_durable_context(self) -> bool:
        # DBOS steps degrade to inline calls outside a workflow, preserving the wrapper-era lifecycle.
        return True

    async def _load_streamed_activity_result(
        self, result: object, model_request_parameters: ModelRequestParameters
    ) -> StreamedActivityResult:
        if isinstance(result, ModelResponse):
            # Legacy-history-only: `DBOSAgent` recorded a bare response for stream steps.
            stream = CompletedStreamedResponse(
                result,
                model_request_parameters=model_request_parameters,
                replay_events=True,
            )
            return StreamedActivityResult(response=result, events=[event async for event in stream])
        return cast(StreamedActivityResult, result)

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
        await super()._dispatch_event_stream_event(ctx, event)

    # --- Capability hooks ---

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Apply the configured parallel-execution mode for every entry point."""
        agent = self._agent
        if agent is None:  # pragma: no cover
            return await handler()
        with agent.parallel_tool_call_execution_mode(self._parallel_execution_mode):
            return await handler()
