from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai import messages as _messages
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
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
    optionally wraps toolsets to route their I/O through Prefect tasks.
    Outside of Prefect flows, the capability is transparent.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.prefect import PrefectDurability
        from pydantic_ai.models.openai import OpenAIChatModel

        model = OpenAIChatModel('gpt-5.2')
        durability = PrefectDurability(name='my_agent', model=model)
        agent = Agent(model=model, capabilities=[durability])
        ```
    """

    name: str
    """Unique agent name used as a prefix for Prefect task names."""

    def __init__(
        self,
        *,
        name: str,
        model: Model,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        model_task_config: TaskConfig | None = None,
        mcp_task_config: TaskConfig | None = None,
        tool_task_config: TaskConfig | None = None,
        tool_task_config_by_name: dict[str, TaskConfig | None] | None = None,
        event_stream_handler_task_config: TaskConfig | None = None,
        toolsets: list[AbstractToolset[AgentDepsT]] | None = None,
        prefectify_toolset_func: Callable[
            [AbstractToolset[AgentDepsT], TaskConfig, TaskConfig, dict[str, TaskConfig | None]],
            AbstractToolset[AgentDepsT],
        ] = _default_prefectify_toolset,
    ):
        """Create a PrefectDurability capability.

        Args:
            name: Unique agent name used as a prefix for Prefect task names.
            model: The model instance to use for requests. Prefect requires a
                single concrete model (no runtime model switching).
            event_stream_handler: Optional handler for streaming events. When
                set, model requests use a streaming task that invokes this
                handler inside the task.
            model_task_config: Prefect task config for model request tasks.
            mcp_task_config: Prefect task config for MCP server tasks.
            tool_task_config: Default Prefect task config for tool call tasks.
            tool_task_config_by_name: Per-tool task configs keyed by tool name.
                Set a tool's config to ``None`` to skip task wrapping.
            event_stream_handler_task_config: Prefect task config for event
                stream handler tasks.
            toolsets: Agent toolsets to wrap for Prefect task execution.
            prefectify_toolset_func: Custom function for wrapping leaf toolsets.
        """
        self.name = name
        self._model = model
        self._event_stream_handler = event_stream_handler

        # Merge configs with defaults
        self._model_task_config = default_task_config | (model_task_config or {})
        self._mcp_task_config = default_task_config | (mcp_task_config or {})
        self._tool_task_config = default_task_config | (tool_task_config or {})
        self._tool_task_config_by_name = tool_task_config_by_name or {}
        self._event_stream_handler_task_config = default_task_config | (event_stream_handler_task_config or {})

        # --- Model request tasks ---

        @task
        async def request_task(
            messages: list[_messages.ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            return await model.request(messages, model_settings, model_request_parameters)

        self._request_task = request_task

        @task
        async def request_stream_task(
            messages: list[_messages.ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext[Any] | None = None,
        ) -> ModelResponse:
            async with model.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                if event_stream_handler is not None:
                    assert run_context is not None
                    await event_stream_handler(run_context, streamed_response)
                async for _ in streamed_response:
                    pass
            return streamed_response.get()

        self._request_stream_task = request_stream_task

        # --- Toolset wrapping ---
        self._prefect_toolsets_by_id: dict[str, AbstractToolset[AgentDepsT]] = {}

        if toolsets:
            for ts in toolsets:
                self._prefectify_leaf_toolsets(ts, prefectify_toolset_func)

    def _prefectify_leaf_toolsets(
        self,
        toolset: AbstractToolset[AgentDepsT],
        prefectify_func: Callable[..., AbstractToolset[AgentDepsT]],
    ) -> None:
        """Wrap leaf toolsets as Prefect tasks."""

        def prefectify(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            wrapped = prefectify_func(
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

        model_name = self._model.model_name

        # Use streaming task when event_stream_handler is set
        if self._event_stream_handler is not None:
            return await self._request_stream_task.with_options(
                name=f'Model Request (Streaming): {model_name}', **self._model_task_config
            )(
                request_context.messages,
                request_context.model_settings,
                request_context.model_request_parameters,
                ctx,
            )

        return await self._request_task.with_options(name=f'Model Request: {model_name}', **self._model_task_config)(
            request_context.messages,
            request_context.model_settings,
            request_context.model_request_parameters,
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

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None
