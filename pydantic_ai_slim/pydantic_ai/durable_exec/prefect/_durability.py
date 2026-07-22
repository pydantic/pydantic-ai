from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai import messages as _messages
from pydantic_ai._run_context import set_current_run_context
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import WrapModelRequestHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._runtime_toolsets import RuntimeToolsetKind
from pydantic_ai.durable_exec._toolset import DurableDynamicToolset, DurableFunctionToolset, DurableMCPToolset
from pydantic_ai.durable_exec._utils import (
    DurableModel,
    StreamedActivityResult,
    capture_event_stream,
)
from pydantic_ai.messages import AgentStreamEvent, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

from ._model import _stamp_response_provenance  # pyright: ignore[reportPrivateUsage]
from ._toolset import prefectify_toolset as _default_prefectify_toolset, with_non_retryable_errors
from ._types import TaskConfig, default_task_config


@dataclass(init=False)
class PrefectDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by routing I/O through Prefect tasks.

    The capability routes model requests, tool calls, MCP I/O, and optionally
    event-stream handling through Prefect tasks when the agent runs inside a
    Prefect flow. Call `agent.run()` inside your own `@flow` to make that run
    durable; outside a flow the capability is transparent and the run is a
    normal, non-durable agent run.

    The capability discovers the agent's model, name, and toolsets
    automatically via `for_agent()`.

    Example:
        ```python {test="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.prefect import PrefectDurability

        durability = PrefectDurability()
        agent = Agent('openai:gpt-5.6-sol', name='my_agent', capabilities=[durability])
        ```
    """

    engine_name = 'Prefect'
    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]] = frozenset(
        {'function', 'mcp', 'dynamic'}
    )

    _durable_unit_noun = 'task'
    _durable_container_noun = 'flow'
    _tool_config_key = 'prefect'

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        event_stream_handler_task_config: TaskConfig | None = None,
        model_task_config: TaskConfig | None = None,
        mcp_task_config: TaskConfig | None = None,
        tool_task_config: TaskConfig | None = None,
    ):
        """Create a PrefectDurability capability.

        The agent's model, name, and toolsets are discovered automatically.

        Args:
            models: Optional additional models keyed by ID for runtime model
                switching. The agent's primary model is always registered as
                `'default'`. A `Model` instance can't be serialized across the
                task boundary, so a run-time model (via `agent.run(model=...)`
                / `agent.override(model=...)`, or swapped in by an outer capability)
                is sent as its `model_id` string and rebuilt inside the task by
                registry lookup, then the agent's `resolve_model_id` capability
                chain / `infer_model`. Register an instance here (and reference it
                by key or pass the registered instance) whenever its `model_id`
                alone wouldn't rebuild it faithfully — e.g. a custom provider,
                client, or settings. Model-name strings never need registering;
                to customize how they're built (e.g. a custom provider), use the
                [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] capability.
            event_stream_handler: Optional event stream handler. Model events are handled
                live inside model-request tasks, and tool events are handled in per-event tasks.
            name: Unique agent name used in the Prefect task names. Defaults to the agent's
                `name` when the capability is bound.
            event_stream_handler_task_config: Prefect task config for event stream handler tasks.
            model_task_config: Prefect task config for model request tasks.
            mcp_task_config: Prefect task config for MCP server tasks.
            tool_task_config: Default Prefect task config for tool call tasks. Per-tool
                overrides are configured via tool metadata, e.g.
                `@my_toolset.tool(metadata={'prefect': TaskConfig(...)})` (or `False` to skip
                task wrapping), or via the
                [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] capability.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)

        # Model and event-handler tasks compose the same non-retryable condition as tool tasks: a
        # `UserError`/`UnexpectedModelBehavior` raised inside them (e.g. a model that can't be
        # rebuilt on the worker) is a framework misconfiguration that retrying can't fix.
        self._model_task_config = with_non_retryable_errors(default_task_config | (model_task_config or {}))
        self._mcp_task_config = default_task_config | (mcp_task_config or {})
        self._tool_task_config = default_task_config | (tool_task_config or {})
        self._event_stream_handler_task_config = with_non_retryable_errors(
            default_task_config | (event_stream_handler_task_config or {})
        )

        # Populated by for_agent when the capability is attached to an agent.
        self._request_task: Any = None
        self._request_stream_task: Any = None
        self._cancel_suspended_response_task: Any = None

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        # --- Model request tasks ---

        @task
        async def request_task(
            model_id: str | None,
            messages: list[_messages.ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any],
        ) -> ModelResponse:
            model = await self._resolve_model_for_request(model_id, run_context)
            with set_current_run_context(run_context):
                response = await model.request(messages, model_settings, model_request_parameters)
            _stamp_response_provenance(response, messages)
            return response

        self._request_task = request_task

        @task
        async def request_stream_task(
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
                        handler=self._event_stream_handler,
                    )
            response = streamed_response.get()
            _stamp_response_provenance(response, messages)
            return StreamedActivityResult(response=response, events=events)

        self._request_stream_task = request_stream_task

        @task
        async def cancel_suspended_response_task(
            model_id: str | None, response: ModelResponse, run_context: RunContext[Any]
        ) -> None:
            model = await self._resolve_model_for_request(model_id, run_context)
            with set_current_run_context(run_context):
                await model.cancel_suspended_response(response)

        self._cancel_suspended_response_task = cancel_suspended_response_task

        # --- Toolset wrapping ---
        self._register_toolsets(agent)

    @property
    def in_durable_context(self) -> bool:
        return FlowRunContext.get() is not None

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        assert self._event_stream_handler is not None
        handler = self._event_stream_handler

        @task(name='Handle Stream Event', **self._event_stream_handler_task_config)
        async def event_stream_handler_task(stream_event: AgentStreamEvent, sequence: int) -> None:
            await handler(ctx, self._single_event_stream(stream_event))

        # The sequence number makes content-identical events within one flow run each fire
        # (distinct task-cache keys) while a flow retry that re-executes the same run
        # reproduces the same numbers and replays from cache. `task_run_dynamic_keys` is
        # Prefect's own per-flow-run counter store for task-call disambiguation, so a
        # namespaced key gets exactly the retry-lineage lifetime Prefect's task naming
        # relies on.
        flow_context = FlowRunContext.get()
        assert flow_context is not None
        sequence_key = f'pydantic_ai_event_sequence:{self.name}'
        sequence = flow_context.task_run_dynamic_keys.get(sequence_key, 0)
        assert isinstance(sequence, int)
        flow_context.task_run_dynamic_keys[sequence_key] = sequence + 1
        await event_stream_handler_task(event, sequence)

    def _wrap_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        wrapped = _default_prefectify_toolset(ts, self._mcp_task_config, self._tool_task_config, {})
        return (
            wrapped if isinstance(wrapped, (DurableDynamicToolset, DurableFunctionToolset, DurableMCPToolset)) else None
        )

    # --- Capability hooks ---

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Route model requests through Prefect tasks when inside a flow."""
        if not self.in_durable_context:
            return await handler(request_context)

        # A `Model` instance can't be serialized across the task boundary, so the
        # request carries a `model_id` (None for the default, the run's original
        # model-id string, a `models=` registry key, or a model-name string) and the
        # task rebuilds the model deps-aware via `_resolve_model_for_request`.
        # A model swapped in by an outer capability's `before_model_request`
        # round-trips via `_find_model_id` on `request_context.model`.
        model_id = self._model_id_for_request(ctx, request_context)
        model_name = request_context.model.model_name

        async def request_segment(request: ModelRequestContext) -> ModelResponse:
            return await self._request_task.with_options(
                name=f'Model Request: {model_name}', **self._model_task_config
            )(model_id, request.messages, request.model_settings, request.model_request_parameters, ctx)

        async def request_stream_segment(request: ModelRequestContext) -> StreamedActivityResult:
            return await self._request_stream_task.with_options(
                name=f'Model Request (Streaming): {model_name}', **self._model_task_config
            )(model_id, request.messages, request.model_settings, request.model_request_parameters, ctx)

        async def cancel_suspended_response_segment(response: ModelResponse) -> None:
            await self._cancel_suspended_response_task.with_options(
                name=f'Cancel Suspended Response: {model_name}', **self._model_task_config
            )(model_id, response, ctx)

        request_context.model = DurableModel(
            request_context.model,
            request_segment=request_segment,
            request_stream_segment=request_stream_segment,
            cancel_suspended_response_segment=cancel_suspended_response_segment,
        )
        return await handler(request_context)
