from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai import messages as _messages
from pydantic_ai._run_context import set_current_run_context
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import (
    CapabilityOrdering,
    WrapModelRequestHandler,
)
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._runtime_toolsets import reject_unsupported_runtime_toolsets
from pydantic_ai.durable_exec._utils import (
    DurableModel,
    StreamedActivityResult,
    capture_event_stream,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset

from ._toolset import PrefectWrapperToolset, prefectify_toolset as _default_prefectify_toolset
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
        agent = Agent('openai:gpt-5.2', name='my_agent', capabilities=[durability])
        ```
    """

    engine_name = 'Prefect'

    name: str
    """Unique agent name used as a prefix for Prefect task names."""

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
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
            event_stream_handler_task_config: Prefect task config for event stream handler tasks.
            model_task_config: Prefect task config for model request tasks.
            mcp_task_config: Prefect task config for MCP server tasks.
            tool_task_config: Default Prefect task config for tool call tasks. Per-tool
                overrides are configured via tool metadata, e.g.
                `@my_toolset.tool(metadata={'prefect': TaskConfig(...)})` (or `False` to skip
                task wrapping), or via the
                [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] capability.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler)
        self.name = ''
        self._agent: AbstractAgent[Any, Any] | None = None

        self._model_task_config = default_task_config | (model_task_config or {})
        self._mcp_task_config = default_task_config | (mcp_task_config or {})
        self._tool_task_config = default_task_config | (tool_task_config or {})
        self._event_stream_handler_task_config = default_task_config | (event_stream_handler_task_config or {})

        self._prefect_toolsets_by_id: dict[str, WrapperToolset[AgentDepsT]] = {}
        # Populated by for_agent when the capability is attached to an agent.
        self._request_task: Any = None
        self._request_stream_task: Any = None
        self._cancel_suspended_response_task: Any = None

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> PrefectDurability[AgentDepsT]:
        """Bind to the agent: discover model, name, toolsets and register Prefect tasks.

        Returns a new bound instance; the original capability is left pristine so the
        same instance can be passed to multiple agents.
        """
        if not agent.name:
            raise UserError('An agent needs to have a unique `name` in order to be used with Prefect.')

        bound = copy.copy(self)
        bound.name = agent.name
        bound._agent = agent

        bound._prefect_toolsets_by_id = {}

        # Build model registry (shared with the other durability capabilities)
        bound._bind_models(agent)

        # --- Model request tasks ---

        @task
        async def request_task(
            model_id: str | None,
            messages: list[_messages.ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any],
        ) -> ModelResponse:
            model = await bound._resolve_model_for_request(model_id, run_context)
            with set_current_run_context(run_context):
                return await model.request(messages, model_settings, model_request_parameters)

        bound._request_task = request_task

        @task
        async def request_stream_task(
            model_id: str | None,
            messages: list[_messages.ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any],
        ) -> StreamedActivityResult:
            model = await bound._resolve_model_for_request(model_id, run_context)
            with set_current_run_context(run_context):
                async with model.request_stream(
                    messages, model_settings, model_request_parameters, run_context
                ) as streamed_response:
                    events = await capture_event_stream(
                        run_context=run_context,
                        stream=streamed_response,
                        handler=bound._event_stream_handler,
                    )
            return StreamedActivityResult(response=streamed_response.get(), events=events)

        bound._request_stream_task = request_stream_task

        @task
        async def cancel_suspended_response_task(
            model_id: str | None, response: ModelResponse, run_context: RunContext[Any]
        ) -> None:
            model = await bound._resolve_model_for_request(model_id, run_context)
            with set_current_run_context(run_context):
                await model.cancel_suspended_response(response)

        bound._cancel_suspended_response_task = cancel_suspended_response_task

        # --- Toolset wrapping ---
        for toolset in agent.toolsets:
            bound._prefectify_leaf_toolsets(toolset)

        return bound

    def _in_durable_context(self) -> bool:
        return FlowRunContext.get() is not None

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        assert self._event_stream_handler is not None
        handler = self._event_stream_handler

        @task(name='Handle Stream Event', **self._event_stream_handler_task_config)
        async def event_stream_handler_task(stream_event: AgentStreamEvent) -> None:
            await handler(ctx, self._single_event_stream(stream_event))

        await event_stream_handler_task(event)

    def _reject_runtime_toolsets(self, toolset: AbstractToolset[AgentDepsT]) -> None:
        """Reject executing toolsets added per-run inside a flow.

        The run toolset assembled by the agent contains construction-time toolsets (whose
        function tools and MCP I/O `for_agent` wrapped as tasks) plus any per-run extras —
        `run(toolsets=...)`, or toolsets contributed by per-run capabilities/specs. An
        executing extra would run un-tasked inside the flow and re-execute on retries, so
        it's rejected explicitly, like the deprecated `PrefectAgent` does. Non-executing
        toolsets like `ExternalToolset` are allowed. Checked here in run setup so every
        entry point inside a user flow is covered; only applies inside a flow — outside
        one the capability is transparent and any toolset is fine.
        """
        if FlowRunContext.get() is None:
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
            runtime_leaves, unsupported_kinds=frozenset({'function', 'mcp', 'dynamic'}), engine='Prefect'
        )

    def _prefectify_leaf_toolsets(self, toolset: AbstractToolset[AgentDepsT]) -> None:
        """Wrap leaf toolsets as Prefect tasks."""

        def prefectify(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            if ts.id is not None and (existing := self._prefect_toolsets_by_id.get(ts.id)) is not None:
                if existing.wrapped is ts:
                    # The same toolset instance can appear in more than one place in the
                    # tree; reuse its wrapper.
                    return existing
                # A distinct toolset under an already-registered `id` would silently
                # replace it in the registry and route both toolsets' calls to one wrapper.
                raise UserError(
                    f'Two toolsets have the same `id` {ts.id!r}. Toolset `id`s must be unique among all '
                    "toolsets registered with the same agent, as they identify the toolset's tasks "
                    'within the flow.'
                )
            wrapped = _default_prefectify_toolset(
                ts,
                self._mcp_task_config,
                self._tool_task_config,
                {},  # per-tool config comes from tool metadata on the capability path
            )
            if isinstance(wrapped, PrefectWrapperToolset):
                # Without an ID the wrapper can't be swapped in at run time (see
                # `get_wrapper_toolset`), so the toolset's calls would silently run
                # untracked inside the Prefect flow and re-execute on retries.
                if ts.id is None:
                    raise UserError(
                        "Toolsets that are 'leaves' (i.e. those that implement their own tool listing and calling) "
                        'need to have a unique `id` in order to be used with Prefect. '
                        "The ID will be used to identify the toolset's tasks within the flow."
                    )
                self._prefect_toolsets_by_id[ts.id] = wrapped
            return wrapped

        toolset.visit_and_replace(prefectify)

    # --- Capability hooks ---

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Route model requests through Prefect tasks when inside a flow."""
        if FlowRunContext.get() is None:
            return await handler(request_context)

        # A `Model` instance can't be serialized across the task boundary, so the
        # request carries a `model_id` (None for the default, the run's original
        # model-id string, a `models=` registry key, or a model-name string) and the
        # task rebuilds the model deps-aware via `_resolve_model_for_request`.
        # A model swapped in by an outer capability's `before_model_request`
        # round-trips via `_find_model_id` on `request_context.model`.
        model_id = self._model_id_for_request(ctx, request_context)
        model_name = request_context.model.model_name

        async def request_segment(
            messages: list[_messages.ModelMessage],
            settings: ModelSettings | None,
            parameters: ModelRequestParameters,
        ) -> ModelResponse:
            return await self._request_task.with_options(
                name=f'Model Request: {model_name}', **self._model_task_config
            )(model_id, messages, settings, parameters, ctx)

        async def request_stream_segment(
            messages: list[_messages.ModelMessage],
            settings: ModelSettings | None,
            parameters: ModelRequestParameters,
        ) -> StreamedActivityResult:
            return await self._request_stream_task.with_options(
                name=f'Model Request (Streaming): {model_name}', **self._model_task_config
            )(model_id, messages, settings, parameters, ctx)

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

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Replace leaf toolsets with their Prefect-wrapped versions."""
        self._reject_runtime_toolsets(toolset)

        if not self._prefect_toolsets_by_id:  # pragma: no cover
            # An agent always has its built-in `<agent>` `FunctionToolset`, which is registered
            # here, so this is never empty at run time; the guard mirrors DBOS/Temporal for parity.
            return None

        def swap(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            ts_id = ts.id
            if ts_id is not None and ts_id in self._prefect_toolsets_by_id:
                return self._prefect_toolsets_by_id[ts_id]
            return ts

        return toolset.visit_and_replace(swap)

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None
