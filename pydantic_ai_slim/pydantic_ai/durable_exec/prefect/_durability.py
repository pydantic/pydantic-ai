from __future__ import annotations

import copy
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from prefect import flow, task
from prefect.context import FlowRunContext
from prefect.utilities.asyncutils import run_coro_as_sync

from pydantic_ai import messages as _messages
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import (
    CapabilityOrdering,
    WrapModelRequestHandler,
)
from pydantic_ai.durable_exec._base import BaseDurability
from pydantic_ai.durable_exec._runtime_toolsets import reject_unsupported_runtime_toolsets
from pydantic_ai.durable_exec._utils import (
    StreamedActivityResult,
    model_request,
    model_request_stream,
    process_event_stream,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset

from ._toolset import PrefectWrapperToolset, prefectify_toolset as _default_prefectify_toolset
from ._types import TaskConfig, default_task_config


def _wrap_event_stream_handler_in_tasks(
    handler: EventStreamHandler[Any] | None,
    task_config: TaskConfig,
) -> EventStreamHandler[Any] | None:
    """Wrap an event stream handler so each event is handled in its own Prefect task.

    Mirrors the deprecated `PrefectAgent`, which runs the handler once per event inside a
    `@task` (named `'Handle Stream Event'`, configurable via `event_stream_handler_task_config`)
    so each event's side effects are individually checkpointed. Returns `None` unchanged so
    the caller can pass the result straight through to `process_event_stream`.
    """
    if handler is None:
        return None

    async def handle_events_in_flow(ctx: RunContext[Any], stream: AsyncIterable[_messages.AgentStreamEvent]) -> None:
        @task(name='Handle Stream Event', **task_config)
        async def handle_event(event: _messages.AgentStreamEvent) -> None:
            async def single_event() -> AsyncIterator[_messages.AgentStreamEvent]:
                yield event

            await handler(ctx, single_event())

        async for event in stream:
            await handle_event(event)

    return handle_events_in_flow


@dataclass(init=False)
class PrefectDurability(BaseDurability[AgentDepsT]):
    """Capability that makes an agent durable by routing I/O through Prefect tasks.

    When added to an agent, this capability intercepts model requests and
    wraps toolsets to route their I/O through Prefect tasks.
    Outside of Prefect flows, the capability is transparent.

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
        model_task_config: TaskConfig | None = None,
        mcp_task_config: TaskConfig | None = None,
        tool_task_config: TaskConfig | None = None,
        event_stream_handler_task_config: TaskConfig | None = None,
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
            event_stream_handler: Optional handler for streaming events.
            model_task_config: Prefect task config for model request tasks.
            mcp_task_config: Prefect task config for MCP server tasks.
            tool_task_config: Default Prefect task config for tool call tasks. Per-tool
                overrides are configured via tool metadata, e.g.
                `@my_toolset.tool(metadata={'prefect': TaskConfig(...)})` (or `False` to skip
                task wrapping), or via the
                [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] capability.
            event_stream_handler_task_config: Prefect task config for event handler tasks.
        """
        super().__init__(models=models)
        self.name = ''
        self._agent: AbstractAgent[Any, Any] | None = None
        self._event_stream_handler = event_stream_handler

        self._model_task_config = default_task_config | (model_task_config or {})
        self._mcp_task_config = default_task_config | (mcp_task_config or {})
        self._tool_task_config = default_task_config | (tool_task_config or {})
        self._event_stream_handler_task_config = default_task_config | (event_stream_handler_task_config or {})

        self._prefect_toolsets_by_id: dict[str, AbstractToolset[AgentDepsT]] = {}
        # Populated by for_agent when the capability is attached to an agent.
        self._request_task: Any = None
        self._request_stream_task: Any = None

    @classmethod
    def from_agent(cls, agent: AbstractAgent[Any, Any]) -> PrefectDurability[Any] | None:
        """Return the bound `PrefectDurability` on an agent, walking its capability chain."""
        found: list[PrefectDurability[Any]] = []

        def visitor(cap: Any) -> None:
            if isinstance(cap, cls):
                found.append(cap)

        agent.root_capability.apply(visitor)
        return found[0] if found else None

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

        # If no handler was passed to the capability, fall back to the agent's
        # instance-level one so it fires inside the task alongside the capability chain.
        if bound._event_stream_handler is None:
            bound._event_stream_handler = agent.event_stream_handler
        # Process each streamed event in its own Prefect task (tunable via
        # `event_stream_handler_task_config`), mirroring the deprecated `PrefectAgent` so
        # completed event side effects are checkpointed within the streaming task.
        event_stream_handler = _wrap_event_stream_handler_in_tasks(
            bound._event_stream_handler, bound._event_stream_handler_task_config
        )
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
            request_context = ModelRequestContext(
                model=model,
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
            )
            return await model_request(model, request_context=request_context, run_context=run_context)

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
            request_context = ModelRequestContext(
                model=model,
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
            )
            async with model_request_stream(
                model, request_context=request_context, run_context=run_context
            ) as streamed_response:
                # Fire the full capability chain's wrap_run_event_stream hooks against
                # the live stream inside the Prefect task.
                events = await process_event_stream(
                    run_context=run_context,
                    request_context=request_context,
                    stream=streamed_response,
                    handler=event_stream_handler,
                )
            return StreamedActivityResult(response=streamed_response.get(), events=events)

        bound._request_stream_task = request_stream_task

        # --- Toolset wrapping ---
        for toolset in agent.toolsets:
            bound._prefectify_leaf_toolsets(toolset)

        # --- Auto-wrap agent.run / run_sync as Prefect flows ---
        bound._install_flow_wrappers(agent)

        return bound

    def _install_flow_wrappers(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        """Replace `agent.run` and `agent.run_sync` with Prefect-flow-decorated wrappers.

        When called outside an active flow run, the wrapper enters a new flow so
        every model and toolset task recorded inside it is observable in the Prefect
        UI. When already inside a flow, the wrapper passes through to avoid a
        redundant nested flow.

        Idempotent: if `for_agent` is called twice (e.g. an agent is bound to two
        `PrefectDurability` instances by mistake), the second call is a no-op
        rather than stacking wrappers.
        """
        if getattr(agent.run, '_pydantic_ai_prefect_wrapped', False):
            return
        original_run = agent.run
        original_run_sync = agent.run_sync

        @flow(name=f'{self.name} Run')
        async def _auto_run_flow(*args: Any, **kwargs: Any) -> Any:
            return await original_run(*args, **kwargs)

        @flow(name=f'{self.name} Sync Run')
        def _auto_run_sync_flow(*args: Any, **kwargs: Any) -> Any:
            # Prefect's sync flow body must be sync; bridge to the async run() the
            # same way `PrefectAgent.run_sync` does.
            return run_coro_as_sync(original_run(*args, **kwargs))

        async def patched_run(*args: Any, **kwargs: Any) -> Any:
            self._reject_runtime_toolsets(kwargs.get('toolsets'))
            if FlowRunContext.get() is not None:
                return await original_run(*args, **kwargs)
            return cast(Any, await _auto_run_flow(*args, **kwargs))

        def patched_run_sync(*args: Any, **kwargs: Any) -> Any:
            self._reject_runtime_toolsets(kwargs.get('toolsets'))
            if FlowRunContext.get() is not None:  # pragma: lax no cover
                return original_run_sync(*args, **kwargs)
            return cast(Any, _auto_run_sync_flow(*args, **kwargs))

        patched_run._pydantic_ai_prefect_wrapped = True  # pyright: ignore[reportFunctionMemberAccess]
        agent.run = patched_run
        agent.run_sync = patched_run_sync

    def _reject_runtime_toolsets(self, toolsets: Sequence[AbstractToolset[AgentDepsT]] | None) -> None:
        """Reject executing toolsets added per-run.

        Every run on an agent with `PrefectDurability` is a Prefect flow. Prefect wraps
        both function tools and MCP servers in tasks registered up front, and dynamic
        toolsets can't be introspected ahead of time, so a per-run executing toolset
        would run un-tasked inside the flow — rejected explicitly, like the deprecated
        `PrefectAgent` does. Non-executing toolsets like `ExternalToolset` are allowed.
        """
        reject_unsupported_runtime_toolsets(
            toolsets, unsupported_kinds=frozenset({'function', 'mcp', 'dynamic'}), engine='Prefect'
        )

    def _prefectify_leaf_toolsets(self, toolset: AbstractToolset[AgentDepsT]) -> None:
        """Wrap leaf toolsets as Prefect tasks."""

        def prefectify(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
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
        # request carries a `model_id` (None for the default, a `models=` registry
        # key, or a model-name string) and the task rebuilds the model deps-aware
        # via `_resolve_model_for_request`. `request_context.model` (not `ctx.model`)
        # is used so a model swapped in by an outer capability's
        # `before_model_request` round-trips too.
        model_id = self._find_model_id(request_context.model)

        model_name = request_context.model.model_name

        # Use the streaming task when either the agent loop expects an event
        # stream (per-run/instance handler, or a chain capability that overrides
        # `wrap_run_event_stream`) OR this capability has its own construction-
        # time handler that needs to fire inside the task. The streaming task
        # fires the chain against live events inside the boundary and buffers
        # events for replay through any per-run handler on the workflow side.
        if request_context.streaming or self._event_stream_handler is not None:
            result: StreamedActivityResult = await self._request_stream_task.with_options(
                name=f'Model Request (Streaming): {model_name}', **self._model_task_config
            )(
                model_id,
                request_context.messages,
                request_context.model_settings,
                request_context.model_request_parameters,
                ctx,
            )
            return result.apply_to(request_context)

        return await self._request_task.with_options(name=f'Model Request: {model_name}', **self._model_task_config)(
            model_id,
            request_context.messages,
            request_context.model_settings,
            request_context.model_request_parameters,
            ctx,
        )

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Replace leaf toolsets with their Prefect-wrapped versions."""
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
