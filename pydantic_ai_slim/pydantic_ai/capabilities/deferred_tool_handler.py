"""Capability that resolves deferred tool calls using a user-supplied handler function."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, DeferredToolResults, RunContext

from .abstract import AbstractCapability


@dataclass
class HandleDeferredToolCalls(AbstractCapability[AgentDepsT]):
    """Resolves deferred tool calls inline during an agent run using a handler function.

    When tools require approval or external execution, the agent normally pauses the run
    and returns [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] as output.
    This capability intercepts deferred tool calls, calls the provided handler to resolve
    them, and continues the agent run automatically.

    The handler receives the [`RunContext`][pydantic_ai.tools.RunContext] and the
    [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests], and must return
    [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] with results for
    some or all pending tool calls. Unresolved calls are passed to the next capability
    in the chain, or bubble up as `DeferredToolRequests` output.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.capabilities import HandleDeferredToolCalls
        from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext


        async def handle_deferred(
            ctx: RunContext[None], requests: DeferredToolRequests
        ) -> DeferredToolResults:
            # Auto-approve all tools that need approval
            approvals = {call.tool_call_id: True for call in requests.approvals}
            return DeferredToolResults(approvals=approvals)


        agent = Agent(
            'openai:gpt-5',
            capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
        )
        ```
    """

    handler: Callable[
        [RunContext[AgentDepsT], DeferredToolRequests],
        DeferredToolResults | Awaitable[DeferredToolResults],
    ]
    """The handler function that resolves deferred tool requests.

    Receives the run context and the deferred tool requests, and must return
    `DeferredToolResults` with results for some or all pending tool calls.
    Can be sync or async.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        result = self.handler(ctx, requests)
        if inspect.isawaitable(result):
            return await result
        return result
