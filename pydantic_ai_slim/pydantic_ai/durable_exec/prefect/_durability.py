from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from prefect import flow, task
from prefect.context import FlowRunContext
from prefect.utilities.asyncutils import run_coro_as_sync

from pydantic_ai import messages as _messages
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    CapabilityOrdering,
    WrapModelRequestHandler,
)
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset

from ._toolset import PrefectWrapperToolset, prefectify_toolset as _default_prefectify_toolset
from ._types import TaskConfig, default_task_config


@dataclass(init=False)
class PrefectDurability(AbstractCapability[AgentDepsT]):
    """Capability that makes an agent durable by routing I/O through Prefect tasks.

    When added to an agent, this capability intercepts model requests and
    wraps toolsets to route their I/O through Prefect tasks.
    Outside of Prefect flows, the capability is transparent.

    The capability discovers the agent's model, name, and toolsets
    automatically via ``for_agent()``.

    Example:
        ```python {test="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.prefect import PrefectDurability

        durability = PrefectDurability()
        agent = Agent('openai:gpt-5.2', name='my_agent', capabilities=[durability])
        ```
    """

    name: str
    """Unique agent name used as a prefix for Prefect task names."""

    def __init__(
        self,
        *,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        model_task_config: TaskConfig | None = None,
        mcp_task_config: TaskConfig | None = None,
        tool_task_config: TaskConfig | None = None,
        tool_task_config_by_name: dict[str, TaskConfig | None] | None = None,
        event_stream_handler_task_config: TaskConfig | None = None,
        prefectify_toolset_func: Callable[
            [AbstractToolset[AgentDepsT], TaskConfig, TaskConfig, dict[str, TaskConfig | None]],
            AbstractToolset[AgentDepsT],
        ] = _default_prefectify_toolset,
    ):
        """Create a PrefectDurability capability.

        The agent's model, name, and toolsets are discovered automatically.

        Args:
            event_stream_handler: Optional handler for streaming events.
            model_task_config: Prefect task config for model request tasks.
            mcp_task_config: Prefect task config for MCP server tasks.
            tool_task_config: Default Prefect task config for tool call tasks.
            tool_task_config_by_name: Per-tool task configs keyed by tool name.
            event_stream_handler_task_config: Prefect task config for event handler tasks.
            prefectify_toolset_func: Custom function for wrapping leaf toolsets.
        """
        self.name = ''
        self._agent: AbstractAgent[Any, Any] | None = None
        self._event_stream_handler = event_stream_handler
        self._prefectify_toolset_func = prefectify_toolset_func

        self._model_task_config = default_task_config | (model_task_config or {})
        self._mcp_task_config = default_task_config | (mcp_task_config or {})
        self._tool_task_config = default_task_config | (tool_task_config or {})
        self._tool_task_config_by_name = tool_task_config_by_name or {}
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
        from pydantic_ai.exceptions import UserError

        if not agent.name:
            raise UserError('An agent needs to have a unique `name` in order to be used with Prefect.')
        if not isinstance(agent.model, Model):
            raise UserError('An agent needs to have a concrete `model` in order to be used with Prefect.')

        bound = copy.copy(self)
        bound.name = agent.name
        bound._agent = agent
        model = agent.model

        # If no handler was passed to the capability, fall back to the agent's
        # instance-level one so it fires inside the task alongside the capability chain.
        if bound._event_stream_handler is None:
            bound._event_stream_handler = agent.event_stream_handler
        event_stream_handler = bound._event_stream_handler
        bound._prefect_toolsets_by_id = {}

        # --- Model request tasks ---

        @task
        async def request_task(
            messages: list[_messages.ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any],
        ) -> ModelResponse:
            from pydantic_ai.durable_exec import call_model

            request_context = ModelRequestContext(
                model=model,
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
            )
            return await call_model(model, request_context, run_context)

        bound._request_task = request_task

        @task
        async def request_stream_task(
            messages: list[_messages.ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any],
        ) -> ModelResponse:
            from pydantic_ai.durable_exec import open_model_stream

            request_context = ModelRequestContext(
                model=model,
                messages=messages,
                model_settings=model_settings,
                model_request_parameters=model_request_parameters,
            )
            async with open_model_stream(model, request_context, run_context) as streamed_response:
                # Fire the full capability chain's wrap_run_event_stream hooks against
                # the live stream inside the Prefect task.
                wrapped_stream = agent.root_capability.wrap_run_event_stream(run_context, stream=streamed_response)
                if event_stream_handler is not None:
                    await event_stream_handler(run_context, wrapped_stream)
                else:
                    async for _ in wrapped_stream:
                        pass
            return streamed_response.get()

        bound._request_stream_task = request_stream_task

        # --- Toolset wrapping ---
        for toolset in agent.toolsets:
            bound._prefectify_leaf_toolsets(toolset)

        # --- Auto-wrap agent.run / run_sync as Prefect flows ---
        bound._install_flow_wrappers(agent)

        return bound

    def _install_flow_wrappers(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        """Replace ``agent.run`` and ``agent.run_sync`` with Prefect-flow-decorated wrappers.

        When called outside an active flow run, the wrapper enters a new flow so
        every model and toolset task recorded inside it is observable in the Prefect
        UI. When already inside a flow, the wrapper passes through to avoid a
        redundant nested flow.
        """
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
            if FlowRunContext.get() is not None:
                return await original_run(*args, **kwargs)
            return cast(Any, await _auto_run_flow(*args, **kwargs))

        def patched_run_sync(*args: Any, **kwargs: Any) -> Any:
            if FlowRunContext.get() is not None:  # pragma: lax no cover
                return original_run_sync(*args, **kwargs)
            return cast(Any, _auto_run_sync_flow(*args, **kwargs))

        agent.run = patched_run
        agent.run_sync = patched_run_sync

    def _prefectify_leaf_toolsets(self, toolset: AbstractToolset[AgentDepsT]) -> None:
        """Wrap leaf toolsets as Prefect tasks."""

        def prefectify(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            wrapped = self._prefectify_toolset_func(
                ts,
                self._mcp_task_config,
                self._tool_task_config,
                self._tool_task_config_by_name,
            )
            if isinstance(wrapped, PrefectWrapperToolset):
                ts_id = ts.id
                if ts_id is not None:
                    self._prefect_toolsets_by_id[ts_id] = wrapped
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

        model_name = ctx.model.model_name

        # Use the streaming task when we need to fire the capability chain's
        # wrap_run_event_stream hooks against live events (outer capabilities that
        # override the hook) or to run the event_stream_handler inside the task.
        agent = self._agent
        needs_chain = agent is not None and agent.root_capability.has_wrap_run_event_stream
        if self._event_stream_handler is not None or needs_chain:
            response = await self._request_stream_task.with_options(
                name=f'Model Request (Streaming): {model_name}', **self._model_task_config
            )(
                request_context.messages,
                request_context.model_settings,
                request_context.model_request_parameters,
                ctx,
            )
            request_context.capabilities_already_applied = True
            return response

        return await self._request_task.with_options(name=f'Model Request: {model_name}', **self._model_task_config)(
            request_context.messages,
            request_context.model_settings,
            request_context.model_request_parameters,
            ctx,
        )

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Replace leaf toolsets with their Prefect-wrapped versions."""
        if not self._prefect_toolsets_by_id:
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
