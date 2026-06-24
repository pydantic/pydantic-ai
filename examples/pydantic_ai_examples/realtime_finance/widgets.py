"""Structured tool outputs (Pydantic) that the frontend renders as rich cards in the chat.

The supervisor's tools return these models. pydantic-ai serializes them to JSON for the LLM (so it
narrates from exact data), and the app forwards the same JSON to the browser which renders a typed
card per `kind` — no ASCII, no separate panel.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel

from .data import UserFinances


class AccountRow(BaseModel):
    name: str
    kind: str
    balance: float


class AccountsWidget(BaseModel):
    kind: Literal['accounts'] = 'accounts'
    title: str = 'Accounts'
    currency: str
    total: float
    rows: list[AccountRow]


class CategoryAmount(BaseModel):
    label: str
    amount: float


class SpendingWidget(BaseModel):
    kind: Literal['spending'] = 'spending'
    title: str = 'Spending by category'
    currency: str
    total: float
    categories: list[CategoryAmount]


class HoldingRow(BaseModel):
    symbol: str
    shares: float
    price: float
    value: float


class PortfolioWidget(BaseModel):
    kind: Literal['portfolio'] = 'portfolio'
    title: str = 'Portfolio'
    currency: str
    total: float
    holdings: list[HoldingRow]


class NetWorthWidget(BaseModel):
    kind: Literal['net_worth'] = 'net_worth'
    title: str = 'Net worth'
    currency: str
    history: list[float]
    start: float
    end: float
    change: float


class TransactionRow(BaseModel):
    date: str
    description: str
    category: str
    amount: float


class TransactionsWidget(BaseModel):
    kind: Literal['transactions'] = 'transactions'
    title: str = 'Recent transactions'
    currency: str
    rows: list[TransactionRow]


class SubscriptionRow(BaseModel):
    name: str
    category: str
    amount: float
    cadence: str


class SubscriptionsWidget(BaseModel):
    kind: Literal['subscriptions'] = 'subscriptions'
    title: str = 'Recurring subscriptions'
    currency: str
    monthly_total: float
    annual_total: float
    rows: list[SubscriptionRow]


class BudgetRow(BaseModel):
    category: str
    current: float
    suggested: float


class BudgetWidget(BaseModel):
    kind: Literal['budget'] = 'budget'
    title: str = 'Suggested budget'
    currency: str
    rows: list[BudgetRow]
    monthly_savings: float


class ProjectionPoint(BaseModel):
    label: str
    value: float


class ProjectionWidget(BaseModel):
    kind: Literal['projection'] = 'projection'
    title: str = 'Net worth projection'
    currency: str
    points: list[ProjectionPoint]
    final: float
    monthly_contribution: float
    years: int
    note: str = ''


class Insight(BaseModel):
    title: str
    detail: str
    severity: Literal['info', 'good', 'warn'] = 'info'
    value: float | None = None
    """Exact headline figure (unrounded), so the supervisor quotes precise numbers; `None` if none."""
    unit: Literal['currency', 'percent', ''] = ''
    """How the browser should format `value` for display."""


class InsightsWidget(BaseModel):
    kind: Literal['insights'] = 'insights'
    title: str = 'Insights'
    items: list[Insight]


Widget: TypeAlias = (
    AccountsWidget
    | SpendingWidget
    | PortfolioWidget
    | NetWorthWidget
    | TransactionsWidget
    | SubscriptionsWidget
    | BudgetWidget
    | ProjectionWidget
    | InsightsWidget
)


def accounts_widget(fin: UserFinances) -> AccountsWidget:
    return AccountsWidget(
        currency=fin.currency,
        total=sum(a.balance for a in fin.accounts),
        rows=[
            AccountRow(name=a.name, kind=a.kind, balance=a.balance)
            for a in fin.accounts
        ],
    )


def spending_widget(fin: UserFinances) -> SpendingWidget:
    totals: dict[str, float] = {}
    for t in fin.transactions:
        if t.amount < 0:
            totals[t.category] = totals.get(t.category, 0.0) + -t.amount
    categories = [
        CategoryAmount(label=k, amount=v)
        for k, v in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return SpendingWidget(
        currency=fin.currency,
        total=sum(c.amount for c in categories),
        categories=categories,
    )


def portfolio_widget(fin: UserFinances) -> PortfolioWidget:
    holdings = [
        HoldingRow(symbol=h.symbol, shares=h.shares, price=h.price, value=h.value)
        for h in fin.holdings
    ]
    return PortfolioWidget(
        currency=fin.currency, total=sum(h.value for h in holdings), holdings=holdings
    )


def net_worth_widget(fin: UserFinances) -> NetWorthWidget:
    history = fin.net_worth_history
    start = history[0] if history else 0.0
    end = history[-1] if history else 0.0
    return NetWorthWidget(
        currency=fin.currency, history=history, start=start, end=end, change=end - start
    )


def transactions_widget(fin: UserFinances, limit: int = 6) -> TransactionsWidget:
    recent = sorted(fin.transactions, key=lambda t: t.date, reverse=True)[:limit]
    rows = [
        TransactionRow(
            date=t.date, description=t.description, category=t.category, amount=t.amount
        )
        for t in recent
    ]
    return TransactionsWidget(currency=fin.currency, rows=rows)


_DISCRETIONARY = {'food', 'entertainment', 'shopping', 'travel', 'news'}


def _monthly_expenses(fin: UserFinances) -> float:
    return sum(-t.amount for t in fin.transactions if t.amount < 0)


def subscriptions_widget(fin: UserFinances) -> SubscriptionsWidget:
    rows = [
        SubscriptionRow(
            name=s.name, category=s.category, amount=s.amount, cadence=s.cadence
        )
        for s in fin.subscriptions
    ]
    monthly = sum(s.amount for s in fin.subscriptions)
    return SubscriptionsWidget(
        currency=fin.currency,
        monthly_total=monthly,
        annual_total=monthly * 12,
        rows=rows,
    )


def budget_widget(fin: UserFinances) -> BudgetWidget:
    spend = spending_widget(fin)
    rows: list[BudgetRow] = []
    saved = 0.0
    for c in spend.categories:
        cut = 0.2 if c.label in _DISCRETIONARY else 0.0
        suggested = round(c.amount * (1 - cut), 2)
        saved += c.amount - suggested
        rows.append(BudgetRow(category=c.label, current=c.amount, suggested=suggested))
    return BudgetWidget(
        currency=fin.currency, rows=rows, monthly_savings=round(saved, 2)
    )


def savings_projection_widget(
    fin: UserFinances, years: int = 3, extra_monthly: float = 0.0
) -> ProjectionWidget:
    start = fin.net_worth_history[-1] if fin.net_worth_history else 0.0
    monthly_net = fin.monthly_income - _monthly_expenses(fin) + extra_monthly
    growth = 0.05  # assumed annual return on the running balance
    value = start
    points = [ProjectionPoint(label='now', value=round(value, 2))]
    for y in range(1, years + 1):
        value = value * (1 + growth) + monthly_net * 12
        points.append(ProjectionPoint(label=f'yr {y}', value=round(value, 2)))
    return ProjectionWidget(
        currency=fin.currency,
        points=points,
        final=round(value, 2),
        monthly_contribution=round(monthly_net, 2),
        years=years,
        note=f'Assumes ~{int(growth * 100)}% annual return and {fin.currency} {round(monthly_net):,}/mo saved.',
    )


def insights_widget(fin: UserFinances) -> InsightsWidget:
    spend = spending_widget(fin)
    subs = subscriptions_widget(fin)
    nw = net_worth_widget(fin)
    monthly_net = fin.monthly_income - _monthly_expenses(fin)
    rate = monthly_net / fin.monthly_income if fin.monthly_income else 0.0
    items: list[Insight] = []
    if spend.categories:
        top = spend.categories[0]
        items.append(
            Insight(
                title='Top spending',
                detail=f'{top.label.title()} is your largest category.',
                value=top.amount,
                unit='currency',
            )
        )
    items.append(
        Insight(
            title='Subscriptions',
            detail=f'{len(subs.rows)} subscriptions billed each month.',
            value=subs.monthly_total,
            unit='currency',
            severity='warn',
        )
    )
    items.append(
        Insight(
            title='Savings rate',
            detail='Share of your income you keep each month.',
            value=rate * 100,
            unit='percent',
            severity='good' if rate >= 0.2 else 'warn',
        )
    )
    items.append(
        Insight(
            title='Net worth',
            detail='Change over the tracked period.',
            value=nw.change,
            unit='currency',
            severity='good' if nw.change >= 0 else 'warn',
        )
    )
    return InsightsWidget(items=items)
