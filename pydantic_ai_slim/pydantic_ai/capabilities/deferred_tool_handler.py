from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic_ai import _agent_graph, exceptions
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, DeferredToolResults, RunContext

from .abstract import AbstractCapability, AgentNode, NodeResult


@dataclass
class DeferredToolHandler(AbstractCapability[AgentDepsT]):
    """Handles deferred tool requests inline during an agent run.

    When tools require approval or external execution, the agent normally pauses the run
    and raises [`DeferredToolRequestsPending`][pydantic_ai.exceptions.DeferredToolRequestsPending].
    This capability intercepts that exception, calls the provided handler to resolve all
    requests, and continues the agent run automatically.

    The handler receives the [`RunContext`][pydantic_ai.tools.RunContext] and the
    [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests], and must return
    [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] with results for
    all pending tool calls.

    When `DeferredToolRequests` is included in the agent's output type, no exception
    is raised and the `DeferredToolRequests` is returned as the agent output — this
    capability only acts on the exception path.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.capabilities import DeferredToolHandler
        from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext


        async def handle_deferred(
            ctx: RunContext[None], requests: DeferredToolRequests
        ) -> DeferredToolResults:
            # Auto-approve all tools that need approval
            approvals = {call.tool_call_id: True for call in requests.approvals}
            return DeferredToolResults(approvals=approvals)


        agent = Agent(
            'openai:gpt-5',
            capabilities=[DeferredToolHandler(handler=handle_deferred)],
        )
        ```
    """

    handler: Callable[
        [RunContext[AgentDepsT], DeferredToolRequests],
        DeferredToolResults | Awaitable[DeferredToolResults],
    ]
    """The handler function that resolves deferred tool requests.

    Receives the run context and the deferred tool requests, and must return
    `DeferredToolResults` with results for all pending tool calls.
    Can be sync or async.
    """

    async def on_node_run_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        error: Exception,
    ) -> NodeResult[AgentDepsT]:
        if not isinstance(error, exceptions.DeferredToolRequestsPending):
            raise error

        handler_result = self.handler(ctx, error.deferred_tool_requests)
        if inspect.isawaitable(handler_result):
            deferred_tool_results = await handler_result
        else:
            deferred_tool_results = handler_result

        model_response, tool_call_results, metadata = _agent_graph.build_tool_call_results(
            deferred_tool_results, ctx.messages
        )

        return _agent_graph.CallToolsNode(
            model_response,
            tool_call_results=tool_call_results,
            tool_call_metadata=metadata,
        )
