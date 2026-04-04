from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from dbos import DBOS

from pydantic_ai import messages as _messages
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    CapabilityOrdering,
    WrapModelRequestHandler,
    WrapRunHandler,
)
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset

from ._utils import StepConfig

DBOSParallelExecutionMode = Literal['sequential', 'parallel_ordered_events']
"""Parallel execution modes safe for DBOS deterministic replay."""


@dataclass(init=False)
class DBOSDurability(AbstractCapability[AgentDepsT]):
    """Capability that makes an agent durable by routing I/O through DBOS steps.

    When added to an agent, this capability intercepts model requests and
    optionally wraps MCP toolsets to route their I/O through DBOS steps.
    Outside of DBOS workflows, the capability is transparent.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.dbos import DBOSDurability
        from pydantic_ai.models.openai import OpenAIChatModel

        model = OpenAIChatModel('gpt-5.2')
        durability = DBOSDurability(name='my_agent', model=model)
        agent = Agent(model=model, capabilities=[durability])
        ```
    """

    name: str
    """Unique agent name used as a prefix for DBOS step names."""

    def __init__(
        self,
        *,
        name: str,
        model: Model,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        model_step_config: StepConfig | None = None,
        mcp_step_config: StepConfig | None = None,
        toolsets: list[AbstractToolset[AgentDepsT]] | None = None,
    ):
        """Create a DBOSDurability capability.

        Args:
            name: Unique agent name used as a prefix for DBOS step names.
            model: The model instance to use for requests. DBOS requires a
                single concrete model (no runtime model switching).
            event_stream_handler: Optional handler for streaming events. When
                set, model requests use a streaming step that invokes this
                handler inside the step.
            model_step_config: DBOS step config for model request steps.
            mcp_step_config: DBOS step config for MCP server steps.
            toolsets: Agent toolsets to wrap for DBOS step execution. Only MCP
                toolsets are wrapped; function toolsets pass through unchanged.
        """
        self.name = name
        self._model = model
        self._event_stream_handler = event_stream_handler
        self._model_step_config = model_step_config or {}
        self._mcp_step_config = mcp_step_config or {}

        # --- Model request steps ---

        @DBOS.step(name=f'{name}__model.request', **self._model_step_config)
        async def request_step(
            messages: list[_messages.ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            return await model.request(messages, model_settings, model_request_parameters)

        self._request_step = request_step

        @DBOS.step(name=f'{name}__model.request_stream', **self._model_step_config)
        async def request_stream_step(
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

        self._request_stream_step = request_stream_step

        # --- MCP toolset wrapping ---
        self._dbos_toolsets_by_id: dict[str, AbstractToolset[Any]] = {}

        if toolsets:
            for toolset in toolsets:
                self._dbosify_leaf_toolsets(toolset)

    def _dbosify_leaf_toolsets(self, toolset: AbstractToolset[AgentDepsT]) -> None:
        """Wrap MCP leaf toolsets as DBOS steps."""

        def dbosify(ts: AbstractToolset[Any]) -> AbstractToolset[Any]:
            try:
                from pydantic_ai.mcp import MCPServer

                from ._mcp_server import DBOSMCPServer
            except ImportError:
                pass
            else:
                if isinstance(ts, MCPServer):
                    wrapped = DBOSMCPServer(
                        wrapped=ts,
                        step_name_prefix=self.name,
                        step_config=self._mcp_step_config,
                    )
                    if ts.id is not None:
                        self._dbos_toolsets_by_id[ts.id] = wrapped
                    return wrapped

            try:
                from pydantic_ai.toolsets.fastmcp import FastMCPToolset

                from ._fastmcp_toolset import DBOSFastMCPToolset
            except ImportError:
                pass
            else:
                if isinstance(ts, FastMCPToolset):
                    wrapped = DBOSFastMCPToolset(
                        wrapped=ts,
                        step_name_prefix=self.name,
                        step_config=self._mcp_step_config,
                    )
                    if ts.id is not None:
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
        """No-op outside DBOS workflows; inside, just delegates."""
        return await handler()

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Route model requests through DBOS steps when inside a workflow."""
        # If not in a workflow, or already in a step, delegate to the real model
        if DBOS.workflow_id is None or DBOS.step_id is not None:
            return await handler(request_context)

        # Use streaming step when event_stream_handler is set
        if self._event_stream_handler is not None:
            return await self._request_stream_step(
                request_context.messages,
                request_context.model_settings,
                request_context.model_request_parameters,
                ctx,
            )

        return await self._request_step(
            request_context.messages,
            request_context.model_settings,
            request_context.model_request_parameters,
        )

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Replace MCP leaf toolsets with their DBOS-wrapped versions."""
        if not self._dbos_toolsets_by_id:
            return None

        def swap(ts: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            ts_id = ts.id
            if ts_id is not None and ts_id in self._dbos_toolsets_by_id:
                return self._dbos_toolsets_by_id[ts_id]
            return ts

        return toolset.visit_and_replace(swap)

    @classmethod
    def get_ordering(cls) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None
