"""The supervisor: a text agent that does the finance reasoning the voice model delegates to.

Realtime voice models can't produce structured output, so the voice front (see `voice.py`) calls
this agent through tools. Each tool returns a typed [`Widget`][] (Pydantic) which is both fed to the
LLM as JSON (so it narrates from exact data) and captured in `FinanceDeps.widgets` so the UI can
render it as a card. The agent's own output is just the short spoken narration.
"""

# pyright: reportUnusedFunction=false
# Tools are registered via the `@agent.tool` decorator, not by being referenced.
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import KnownModelName, Model

from .data import DEFAULT_USER_ID, get_user_finances
from .widgets import (
    AccountsWidget,
    BudgetWidget,
    InsightsWidget,
    NetWorthWidget,
    PortfolioWidget,
    ProjectionWidget,
    SpendingWidget,
    SubscriptionsWidget,
    TransactionsWidget,
    Widget,
    accounts_widget,
    budget_widget,
    insights_widget,
    net_worth_widget,
    portfolio_widget,
    savings_projection_widget,
    spending_widget,
    subscriptions_widget,
    transactions_widget,
)


@dataclass
class FinanceDeps:
    """Run dependencies: which user, and where tool widgets are collected for the UI."""

    user_id: str = DEFAULT_USER_ID
    widgets: list[Widget] = field(default_factory=list[Widget])


INSTRUCTIONS = """\
You are a personal finance analyst. Always call the relevant tool(s) to read the user's real data
before answering; for a full review, call several. Then reply with ONE short, natural sentence
summarising the result, quoting the EXACT figures from the tools (never round or invent numbers).
Do not describe tables — the UI shows the data as cards.\
"""


def create_finance_supervisor(
    model: Model | KnownModelName | str,
) -> Agent[FinanceDeps, str]:
    """Build the finance supervisor agent backed by `model` (e.g. `openai:gpt-5`)."""
    agent = Agent(
        model, deps_type=FinanceDeps, output_type=str, instructions=INSTRUCTIONS
    )

    @agent.tool
    async def get_balances(ctx: RunContext[FinanceDeps]) -> AccountsWidget:
        """Account balances and total cash plus investments."""
        widget = accounts_widget(get_user_finances(ctx.deps.user_id))
        ctx.deps.widgets.append(widget)
        return widget

    @agent.tool
    async def spending_by_category(ctx: RunContext[FinanceDeps]) -> SpendingWidget:
        """Total spending per category for the period (income excluded)."""
        widget = spending_widget(get_user_finances(ctx.deps.user_id))
        ctx.deps.widgets.append(widget)
        return widget

    @agent.tool
    async def portfolio(ctx: RunContext[FinanceDeps]) -> PortfolioWidget:
        """Investment holdings by market value."""
        widget = portfolio_widget(get_user_finances(ctx.deps.user_id))
        ctx.deps.widgets.append(widget)
        return widget

    @agent.tool
    async def net_worth_trend(ctx: RunContext[FinanceDeps]) -> NetWorthWidget:
        """Recent net worth history and the change over the period."""
        widget = net_worth_widget(get_user_finances(ctx.deps.user_id))
        ctx.deps.widgets.append(widget)
        return widget

    @agent.tool
    async def recent_transactions(
        ctx: RunContext[FinanceDeps], limit: int = 6
    ) -> TransactionsWidget:
        """The most recent transactions, newest first."""
        widget = transactions_widget(get_user_finances(ctx.deps.user_id), limit=limit)
        ctx.deps.widgets.append(widget)
        return widget

    @agent.tool
    async def subscriptions(ctx: RunContext[FinanceDeps]) -> SubscriptionsWidget:
        """Recurring subscriptions with their monthly and annual cost."""
        widget = subscriptions_widget(get_user_finances(ctx.deps.user_id))
        ctx.deps.widgets.append(widget)
        return widget

    @agent.tool
    async def budget_plan(ctx: RunContext[FinanceDeps]) -> BudgetWidget:
        """A suggested budget that trims discretionary categories, with the monthly saving."""
        widget = budget_widget(get_user_finances(ctx.deps.user_id))
        ctx.deps.widgets.append(widget)
        return widget

    @agent.tool
    async def spending_insights(ctx: RunContext[FinanceDeps]) -> InsightsWidget:
        """Notable insights about the user's finances (top category, savings rate, ...)."""
        widget = insights_widget(get_user_finances(ctx.deps.user_id))
        ctx.deps.widgets.append(widget)
        return widget

    @agent.tool
    async def savings_projection(
        ctx: RunContext[FinanceDeps], years: int = 3, extra_monthly: float = 0.0
    ) -> ProjectionWidget:
        """Project net worth over `years`, optionally saving `extra_monthly` more each month."""
        widget = savings_projection_widget(
            get_user_finances(ctx.deps.user_id),
            years=years,
            extra_monthly=extra_monthly,
        )
        ctx.deps.widgets.append(widget)
        return widget

    return agent
