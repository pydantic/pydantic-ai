from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from dbos import DBOS

from pydantic_ai import messages as _messages
from pydantic_ai._run_context import set_current_run_context
from pydantic_ai.agent import EventStreamHandler, ParallelExecutionMode
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import WrapModelRequestHandler, WrapRunHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._runtime_toolsets import RuntimeToolsetKind
from pydantic_ai.durable_exec._utils import (
    DurableModel,
    StreamedActivityResult,
    capture_event_stream,
)
from pydantic_ai.messages import AgentStreamEvent, ModelResponse
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._agent import DBOSParallelExecutionMode
from ._utils import StepConfig


@dataclass(init=False)
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

    engine_name = 'DBOS'
    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]] = frozenset({'mcp', 'dynamic'})

    _durable_unit_noun = 'step'
    _durable_container_noun = 'workflow'

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
                is sent as its `model_id` string and rebuilt inside the step by
                registry lookup, then the agent's `resolve_model_id` capability
                chain / `infer_model`. Register an instance here (and reference it
                by key or pass the registered instance) whenever its `model_id`
                alone wouldn't rebuild it faithfully — e.g. a custom provider,
                client, or settings. Model-name strings never need registering;
                to customize how they're built (e.g. a custom provider), use the
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
        self._request_step: Any = None
        self._request_stream_step: Any = None
        self._cancel_suspended_response_step: Any = None
        self._event_stream_handler_step: Any = None
        self._legacy_run_workflow: Any = None
        self._legacy_run_sync_workflow: Any = None
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

        # --- Model request steps ---

        @DBOS.step(name=f'{self.name}__model.request', **self._model_step_config)
        async def request_step(
            model_id: str | None,
            messages: list[_messages.ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any],
        ) -> ModelResponse:
            model = await self._resolve_model_for_request(model_id, run_context)
            with set_current_run_context(run_context):
                return await model.request(messages, model_settings, model_request_parameters)

        self._request_step = request_step

        @DBOS.step(name=f'{self.name}__model.request_stream', **self._model_step_config)
        async def request_stream_step(
            model_id: str | None,
            messages: list[_messages.ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any],
        ) -> StreamedActivityResult:
            model = await self._resolve_model_for_request(model_id, run_context)
            with set_current_run_context(run_context):
                async with model.request_stream(
                    messages, model_settings, model_request_parameters, run_context
                ) as streamed_response:
                    events = await capture_event_stream(
                        run_context=run_context,
                        stream=streamed_response,
                        handler=self._effective_event_stream_handler(),
                    )
            return StreamedActivityResult(response=streamed_response.get(), events=events)

        self._request_stream_step = request_stream_step

        @DBOS.step(name=f'{self.name}__model.cancel_suspended_response', **self._model_step_config)
        async def cancel_suspended_response_step(
            model_id: str | None, response: ModelResponse, run_context: RunContext[Any]
        ) -> None:
            model = await self._resolve_model_for_request(model_id, run_context)
            with set_current_run_context(run_context):
                await model.cancel_suspended_response(response)

        self._cancel_suspended_response_step = cancel_suspended_response_step

        if self._event_stream_handler is not None:

            @DBOS.step(name=f'{self.name}__event_stream_handler', **self._event_stream_handler_step_config)
            async def event_stream_handler_step(
                event: _messages.AgentStreamEvent, run_context: RunContext[Any]
            ) -> None:
                handler = self._effective_event_stream_handler()
                assert handler is not None
                await handler(run_context, self._single_event_stream(event))

            self._event_stream_handler_step = event_stream_handler_step

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

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        if self._in_legacy_workflow.get():
            # Wrapper-era recordings contain no `__event_stream_handler` steps (the wrapper called
            # the handler directly in workflow code), so a legacy run must do the same to keep the
            # recorded step sequence replayable.
            handler = self._effective_event_stream_handler()
            assert handler is not None
            await handler(ctx, self._single_event_stream(event))
            return
        # Route the handler through a DBOS step so its side effects are checkpointed and
        # don't re-run when the workflow recovers.
        assert self._event_stream_handler_step is not None
        await self._event_stream_handler_step(event, ctx)

    def _wrap_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        if isinstance(ts, DynamicToolset):
            from ._dynamic_toolset import dbosify_dynamic_toolset

            return dbosify_dynamic_toolset(wrapped=ts, step_name_prefix=self.name, step_config=self._mcp_step_config)
        try:
            from pydantic_ai.mcp import MCPToolset

            from ._mcp_toolset import dbosify_mcp_toolset
        except ImportError:  # pragma: no cover
            return None
        if isinstance(ts, MCPToolset):
            return dbosify_mcp_toolset(wrapped=ts, step_name_prefix=self.name, step_config=self._mcp_step_config)
        return None

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

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Route model requests through DBOS steps when inside a workflow."""
        if not self.in_durable_context:
            return await handler(request_context)

        # A `Model` instance can't be serialized across the step boundary, so the
        # request carries a `model_id` (None for the default, the run's original
        # model-id string, a `models=` registry key, or a model-name string) and the
        # step rebuilds the model deps-aware via `_resolve_model_for_request`.
        # A model swapped in by an outer capability's `before_model_request`
        # round-trips via `_find_model_id` on `request_context.model`.
        model_id = self._model_id_for_request(ctx, request_context)

        async def request_segment(request: ModelRequestContext) -> ModelResponse:
            return await self._request_step(
                model_id, request.messages, request.model_settings, request.model_request_parameters, ctx
            )

        async def request_stream_segment(request: ModelRequestContext) -> StreamedActivityResult:
            result = await self._request_stream_step(
                model_id, request.messages, request.model_settings, request.model_request_parameters, ctx
            )
            if isinstance(result, ModelResponse):
                # Legacy-history-only: `DBOSAgent` recorded a bare response for stream steps.
                stream = CompletedStreamedResponse(
                    result,
                    model_request_parameters=request.model_request_parameters,
                    replay_events=True,
                )
                return StreamedActivityResult(response=result, events=[event async for event in stream])
            return result

        async def cancel_suspended_response_segment(response: ModelResponse) -> None:
            await self._cancel_suspended_response_step(model_id, response, ctx)

        request_context.model = DurableModel(
            request_context.model,
            request_segment=request_segment,
            request_stream_segment=request_stream_segment,
            cancel_suspended_response_segment=cancel_suspended_response_segment,
        )
        return await handler(request_context)
