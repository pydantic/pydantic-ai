"""Capability that requires approval before executing selected tools."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
    ToolSelector,
    matches_tool_selector,
)
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset

from .abstract import AbstractCapability

ApprovalHandler = Callable[
    [RunContext[AgentDepsT], ToolCallPart],
    bool | ToolApproved | ToolDenied | Awaitable[bool | ToolApproved | ToolDenied],
]
"""Per-call handler returning whether a tool call is approved.

Receives the run context and the [`ToolCallPart`][pydantic_ai.messages.ToolCallPart]
for the call awaiting approval. Returns either:

- `True` / `False`: shorthand for [`ToolApproved`][pydantic_ai.tools.ToolApproved] /
  [`ToolDenied`][pydantic_ai.tools.ToolDenied].
- A [`ToolApproved`][pydantic_ai.tools.ToolApproved] (optionally with `override_args`).
- A [`ToolDenied`][pydantic_ai.tools.ToolDenied] (optionally with a custom denial message).

Sync and async are both accepted.
"""


@dataclass
class RequireToolApproval(AbstractCapability[AgentDepsT]):
    """Capability that requires approval before executing selected tools.

    Tools matching the [`tools`][pydantic_ai.tools.ToolSelector] selector cannot run
    until they're approved: when the model calls one, the framework raises
    [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] under the hood and
    the call surfaces in the
    [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output.

    Without a `handler`, that's all this capability does — the user (or another
    capability) decides what to do with the approval request. Pass a `handler` to
    resolve approvals inline during the run: the capability iterates only the
    approvals for its own matched tools, calls the handler per request, and lets
    any non-matching approvals pass through to other capabilities (or bubble up
    as `DeferredToolRequests`).

    Multiple `RequireToolApproval` instances compose cleanly with different
    selectors and handlers — each only handles its own matched tools.

    ```python
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.capabilities import RequireToolApproval
    from pydantic_ai.messages import ToolCallPart

    # Mark tools as needing approval; let the agent return DeferredToolRequests
    # for the caller to resolve.
    agent = Agent('openai:gpt-5', capabilities=[RequireToolApproval(tools=['delete'])])


    # Resolve approvals inline with a handler.
    async def approve(ctx: RunContext[None], call: ToolCallPart) -> bool:
        return call.tool_name != 'delete' or call.args_as_dict().get('safe') is True


    agent = Agent(
        'openai:gpt-5',
        capabilities=[RequireToolApproval(tools=['delete', 'modify'], handler=approve)],
    )
    ```
    """

    tools: ToolSelector[AgentDepsT] = 'all'
    """A [`ToolSelector`][pydantic_ai.tools.ToolSelector] specifying which tools require approval."""

    handler: ApprovalHandler[AgentDepsT] | None = None
    """Optional per-call handler for resolving approvals inline.

    When `None`, this capability only marks tools — approvals bubble up as
    [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output (or are
    handled by another capability). When provided, the handler is invoked for each
    approval request matching this capability's `tools` selector; non-matching
    approvals pass through unchanged.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'RequireToolApproval'

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return ApprovalRequiredToolset(toolset, tools=self.tools)

    async def handle_deferred_tool_calls(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        if self.handler is None or not requests.approvals:
            return None
        if ctx.tool_manager is None:
            return None  # pragma: no cover

        results = DeferredToolResults()
        for call in requests.approvals:
            tool_def = ctx.tool_manager.get_tool_def(call.tool_name)
            if tool_def is None or not await matches_tool_selector(self.tools, ctx, tool_def):
                # Not our tool — leave for other handlers / DeferredToolRequests output.
                continue
            outcome = self.handler(ctx, call)
            if inspect.isawaitable(outcome):
                outcome = await outcome
            results.approvals[call.tool_call_id] = outcome

        return results if results.approvals else None
