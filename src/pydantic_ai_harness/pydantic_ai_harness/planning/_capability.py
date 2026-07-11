"""Planning capability: model-owned task plans surfaced without busting the cache."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import CachePoint, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.planning._toolset import PlanningToolset, PlanState, render_plan

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.capabilities.abstract import WrapModelRequestHandler
    from pydantic_ai.models import ModelRequestContext

_DEFAULT_GUIDANCE = (
    'You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the '
    'steps, then call it again to update statuses as you start and finish each step. Pass the '
    'full plan every time and keep exactly one step `in_progress`.'
)


@dataclass
class Planning(AbstractCapability[AgentDepsT]):
    """Structured task planning that never invalidates the prompt cache.

    The plan is owned by the model through a single `write_plan` tool. The
    current plan is surfaced back as an *ephemeral* reminder appended to the tail
    of each request (after the latest message), with a cache breakpoint placed
    in front of it. Because the reminder always sits after the breakpoint and is
    never written to the durable message history, the cached prefix stays
    byte-identical across turns -- only the small reminder is re-read each turn.

    Static usage guidance goes into the system prompt via `get_instructions`,
    which is cache-stable; the mutable plan is *never* injected there.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.planning import Planning

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Planning()])
    ```
    """

    guidance: str | None = None
    """Static planning guidance for the system prompt. Cache-stable (identical every
    request). Leave as `None` for the default, or set `''` to omit guidance entirely."""

    cache_ttl: Literal['5m', '1h'] = '5m'
    """TTL for the cache breakpoint placed before the plan reminder."""

    _state: PlanState = field(default_factory=PlanState, init=False, repr=False, compare=False)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Planning[AgentDepsT]:
        """Return a fresh per-run instance with isolated plan state (config preserved)."""
        return replace(self)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable guidance on using the planning tool."""
        guidance = _DEFAULT_GUIDANCE if self.guidance is None else self.guidance
        return guidance or None

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Toolset providing `write_plan` over this run's plan state."""
        return PlanningToolset[AgentDepsT](self._state)

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Append the current plan as an ephemeral tail reminder behind a cache breakpoint.

        This runs *after* core has persisted the durable history, and the
        per-request message list it mutates is never written back. So the
        reminder and its `CachePoint` reach the model but never enter
        `ctx.state.message_history` -- the cached prefix stays byte-identical
        across turns and no stale reminders accumulate. The `CachePoint` sits
        before the reminder text, so the reminder falls outside the cached
        region and cannot invalidate it.
        """
        items = self._state.items
        if not items:
            return await handler(request_context)
        messages = request_context.messages
        last = messages[-1]
        if isinstance(last, ModelRequest):
            reminder = UserPromptPart(content=[CachePoint(ttl=self.cache_ttl), _reminder_text(render_plan(items))])
            messages[-1] = replace(last, parts=[*last.parts, reminder])
        return await handler(request_context)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Serialization name for agent-spec support."""
        return 'Planning'


def _reminder_text(plan: str) -> str:
    return f'<plan-reminder>\nYour current plan (keep it updated with `write_plan`):\n\n{plan}\n</plan-reminder>'
