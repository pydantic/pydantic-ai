"""The voice front: a thin agent that delegates real work to the finance supervisor.

It owns the conversation and exposes tools that demonstrate both subagent dispatch modes:

- `ask_analyst` runs **synchronously** — the model waits for the answer (quick questions).
- `run_deep_analysis` and `plan_savings_goal` run in the **background** — the model keeps talking
  while a longer multi-step run happens, and the result is delivered when ready.

Each call collects the supervisor's typed widgets into `VoiceDeps.widgets`, keyed by the realtime
tool call id, so the app can render them as cards under the right tool — even when an async and a
sync call overlap.
"""

# pyright: reportUnusedFunction=false
# The delegation tools are registered via the `@agent.tool` decorator, not by being referenced.
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext

from .data import DEFAULT_USER_ID
from .supervisor import FinanceDeps
from .widgets import Widget

RUN_DEEP_ANALYSIS = 'run_deep_analysis'
PLAN_SAVINGS_GOAL = 'plan_savings_goal'
BACKGROUND_TOOLS = {RUN_DEEP_ANALYSIS, PLAN_SAVINGS_GOAL}

VOICE_INSTRUCTIONS = """\
You are a polished, concise finance voice assistant for one user. You can answer about balances,
spending, transactions, portfolio, net worth, subscriptions, budgets, insights, and projections.

For a quick, single-topic question:
1. FIRST say a short acknowledgement OUT LOUD, e.g. "Sure, one moment." Keep it to a few words.
2. THEN call `ask_analyst` with the user's question.
3. When the result arrives, answer in one short, conversational sentence.

For a broad request (a "full review", "how am I doing overall", several areas at once):
1. Say you'll run a deeper analysis and it'll take a few seconds.
2. Call `run_deep_analysis`. Keep chatting meanwhile; the result arrives shortly.

For a savings goal or planning request ("can I save X by then", "help me plan", "what if I save more"):
1. Say you'll work out a plan and it'll take a moment.
2. Call `plan_savings_goal` with the target amount and number of years.
3. Then summarise whether it's on track and the key lever.

Always reply in the SAME language the user spoke in. Never read out tables or number lists — the UI
shows them as cards. Never invent figures; use the exact numbers from the tool result.\
"""


@dataclass
class VoiceDeps:
    """Dependencies for the voice front: the supervisor it delegates to and widgets keyed by call id."""

    supervisor: Agent[FinanceDeps, str]
    user_id: str = DEFAULT_USER_ID
    widgets: dict[str, list[Widget]] = field(default_factory=dict[str, list[Widget]])


def create_voice_front() -> Agent[VoiceDeps, str]:
    """Build the thin voice-front agent that delegates to the finance supervisor."""
    agent = Agent(deps_type=VoiceDeps, instructions=VOICE_INSTRUCTIONS)

    async def _delegate(ctx: RunContext[VoiceDeps], question: str) -> str:
        # Collect this delegation's widgets keyed by the realtime tool call id, so concurrent
        # (async) and synchronous tool calls never mix their widgets in the UI.
        fin_deps = FinanceDeps(user_id=ctx.deps.user_id)
        result = await ctx.deps.supervisor.run(question, deps=fin_deps)
        if ctx.tool_call_id:
            ctx.deps.widgets[ctx.tool_call_id] = fin_deps.widgets
        return result.output

    @agent.tool
    async def ask_analyst(ctx: RunContext[VoiceDeps], question: str) -> str:
        """Answer a single finance question about the user's money."""
        return await _delegate(ctx, question)

    @agent.tool
    async def run_deep_analysis(
        ctx: RunContext[VoiceDeps], focus: str = 'overall finances'
    ) -> str:
        """Run a broader multi-area financial review (slower; runs in the background)."""
        prompt = (
            f'Do a full financial review of the user covering balances, spending, subscriptions, '
            f'portfolio, net worth and key insights. Focus on: {focus}. Call the relevant tools for '
            f'each area.'
        )
        return await _delegate(ctx, prompt)

    @agent.tool
    async def plan_savings_goal(
        ctx: RunContext[VoiceDeps], target_amount: float, years: int = 3
    ) -> str:
        """Build a plan to reach a savings target (slower; runs in the background)."""
        prompt = (
            f'The user wants to reach {target_amount:.0f} in {years} years. '
            f'Call savings_projection (years={years}), budget_plan and spending_insights, then say '
            f'whether the goal is on track and the single biggest lever to get there.'
        )
        return await _delegate(ctx, prompt)

    return agent
