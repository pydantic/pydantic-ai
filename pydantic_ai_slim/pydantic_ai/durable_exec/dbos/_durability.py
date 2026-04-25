from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from dbos import DBOS

from pydantic_ai import messages as _messages
from pydantic_ai.agent import EventStreamHandler, ParallelExecutionMode
from pydantic_ai.agent.abstract import AbstractAgent
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
"""Parallel execution modes safe for DBOS deterministic replay.

`'parallel'` is excluded because it cannot guarantee deterministic event ordering,
which DBOS replay requires.
"""


@dataclass(init=False)
class DBOSDurability(AbstractCapability[AgentDepsT]):
    """Capability that makes an agent durable by routing I/O through DBOS steps.

    When added to an agent, this capability intercepts model requests and
    optionally wraps MCP toolsets to route their I/O through DBOS steps.
    Outside of DBOS workflows, the capability is transparent.

    The capability discovers the agent's model, name, and toolsets
    automatically via ``for_agent()``.

    Example:
        ```python {test="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.dbos import DBOSDurability

        durability = DBOSDurability()
        agent = Agent('openai:gpt-5.2', name='my_agent', capabilities=[durability])
        ```
    """

    name: str
    """Unique agent name used as a prefix for DBOS step names."""

    def __init__(
        self,
        *,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        model_step_config: StepConfig | None = None,
        mcp_step_config: StepConfig | None = None,
        parallel_execution_mode: DBOSParallelExecutionMode = 'parallel_ordered_events',
    ):
        """Create a DBOSDurability capability.

        The agent's model, name, and toolsets are discovered automatically.

        Args:
            event_stream_handler: Optional handler for streaming events.
            model_step_config: DBOS step config for model request steps.
            mcp_step_config: DBOS step config for MCP server steps.
            parallel_execution_mode: Tool-call execution mode applied for the duration
                of every run. Defaults to ``'parallel_ordered_events'`` so events
                replay deterministically. Set to ``'sequential'`` for strict ordering.
        """
        self.name = ''
        self._agent: AbstractAgent[Any, Any] | None = None
        self._event_stream_handler = event_stream_handler
        self._model_step_config = model_step_config or {}
        self._mcp_step_config = mcp_step_config or {}
        self._parallel_execution_mode: ParallelExecutionMode = cast(ParallelExecutionMode, parallel_execution_mode)
        self._dbos_toolsets_by_id: dict[str, AbstractToolset[Any]] = {}
        # Populated by for_agent when the capability is attached to an agent.
        self._request_step: Any = None
        self._request_stream_step: Any = None
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
        from pydantic_ai.exceptions import UserError

        if not agent.name:
            raise UserError('An agent needs to have a unique `name` in order to be used with DBOS.')
        if not isinstance(agent.model, Model):
            raise UserError('An agent needs to have a concrete `model` in order to be used with DBOS.')

        bound = copy.copy(self)
        bound.name = agent.name
        bound._agent = agent
        model = agent.model

        # If no handler was passed to the capability, fall back to the agent's
        # instance-level one so it fires inside the step alongside the capability chain.
        if bound._event_stream_handler is None:
            bound._event_stream_handler = agent.event_stream_handler
        event_stream_handler = bound._event_stream_handler
        bound._dbos_toolsets_by_id = {}

        # --- Model request steps ---

        @DBOS.step(name=f'{bound.name}__model.request', **bound._model_step_config)
        async def request_step(
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

        bound._request_step = request_step

        @DBOS.step(name=f'{bound.name}__model.request_stream', **bound._model_step_config)
        async def request_stream_step(
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
                # the live stream inside the DBOS step.
                wrapped_stream = agent.root_capability.wrap_run_event_stream(run_context, stream=streamed_response)
                if event_stream_handler is not None:
                    await event_stream_handler(run_context, wrapped_stream)
                else:
                    async for _ in wrapped_stream:
                        pass
            return streamed_response.get()

        bound._request_stream_step = request_stream_step

        # --- MCP toolset wrapping ---
        for toolset in agent.toolsets:
            bound._dbosify_leaf_toolsets(toolset)

        # --- Auto-wrap agent.run / run_sync as DBOS workflows ---
        bound._install_workflow_wrappers(agent)

        return bound

    def _install_workflow_wrappers(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        """Replace ``agent.run`` and ``agent.run_sync`` with DBOS-workflow-decorated wrappers.

        When called outside an active DBOS workflow, the wrapper enters a workflow so
        all model and toolset steps recorded inside become durable. When already inside a
        workflow, it skips the redundant wrap and just applies the configured parallel
        execution mode.

        ``agent.iter`` is wrapped in-place by `wrap_run` instead — its async-generator
        signature can't be cleanly forwarded through ``@DBOS.workflow``.
        """
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

        async def patched_run(*args: Any, **kwargs: Any) -> Any:
            if DBOS.workflow_id is not None and DBOS.step_id is None:
                # Already inside a DBOS workflow — skip the auto-wrap and just apply the mode.
                with agent.parallel_tool_call_execution_mode(parallel_mode):
                    return await original_run(*args, **kwargs)
            return await _auto_run_workflow(*args, **kwargs)

        def patched_run_sync(*args: Any, **kwargs: Any) -> Any:
            if DBOS.workflow_id is not None and DBOS.step_id is None:  # pragma: lax no cover
                with agent.parallel_tool_call_execution_mode(parallel_mode):
                    return original_run_sync(*args, **kwargs)
            return _auto_run_sync_workflow(*args, **kwargs)

        agent.run = patched_run
        agent.run_sync = patched_run_sync

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
                    if ts.id is not None:  # pragma: no branch
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
                    if ts.id is not None:  # pragma: no branch
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
        for ``run``/``run_sync``. This hook is the single chokepoint that applies the
        execution mode for every entry point — including ``iter``, which the run/sync
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

        # Use the streaming step when we need to fire the capability chain's
        # wrap_run_event_stream hooks against live events (outer capabilities that
        # override the hook) or to run the event_stream_handler inside the step.
        agent = self._agent
        needs_chain = agent is not None and agent.root_capability.has_wrap_run_event_stream
        if self._event_stream_handler is not None or needs_chain:
            response = await self._request_stream_step(
                request_context.messages,
                request_context.model_settings,
                request_context.model_request_parameters,
                ctx,
            )
            request_context.capabilities_already_applied = True
            return response

        return await self._request_step(
            request_context.messages,
            request_context.model_settings,
            request_context.model_request_parameters,
            ctx,
        )

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """Replace MCP leaf toolsets with their DBOS-wrapped versions."""
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
