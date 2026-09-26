"""The optional tool that lets an agent read its own spend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset

if TYPE_CHECKING:
    from pydantic_ai_harness.spend._capability import SpendLimits


def build_toolset(limits: SpendLimits[AgentDepsT]) -> FunctionToolset[AgentDepsT]:
    """A one-tool toolset reporting what the agent has spent so far."""

    async def get_spend(ctx: RunContext[AgentDepsT]) -> str:
        """Report spend so far against each budget, and how much of each is left."""
        statuses = await limits.status(ctx)
        if not statuses:
            return 'No budgets are configured.'
        return '\n'.join(
            f'{s.budget.name} ({s.budget.window}): ${s.spent.usd} spent, '
            f'{"no limit" if s.remaining_usd is None else f"${s.remaining_usd} left"}; '
            f'{s.spent.tokens} tokens used, '
            f'{"no limit" if s.remaining_tokens is None else f"{s.remaining_tokens} left"}'
            for s in statuses
        )

    return FunctionToolset[AgentDepsT]([get_spend], id='spend')
