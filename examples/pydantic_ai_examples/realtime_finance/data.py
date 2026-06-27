"""Mocked user finance data for the realtime finance demo.

No database or network — a couple of hand-built profiles so the demo runs offline. Lookups raise
`KeyError` for unknown users; the agent tools translate that into a friendly message.
"""

from __future__ import annotations

from pydantic import BaseModel


class Account(BaseModel):
    """A single bank or brokerage account."""

    name: str
    kind: str
    balance: float


class Transaction(BaseModel):
    """A dated spending or income entry."""

    date: str
    description: str
    category: str
    amount: float


class Holding(BaseModel):
    """An investment position."""

    symbol: str
    shares: float
    price: float

    @property
    def value(self) -> float:
        return self.shares * self.price


class Subscription(BaseModel):
    """A recurring charge."""

    name: str
    category: str
    amount: float
    cadence: str = 'monthly'


class UserFinances(BaseModel):
    """Everything the demo knows about one user."""

    user_id: str
    name: str
    currency: str
    monthly_income: float
    accounts: list[Account]
    transactions: list[Transaction]
    holdings: list[Holding]
    subscriptions: list[Subscription]
    net_worth_history: list[float]


_USERS: dict[str, UserFinances] = {
    'u_001': UserFinances(
        user_id='u_001',
        name='Ada',
        currency='USD',
        monthly_income=6200.00,
        accounts=[
            Account(name='Checking', kind='cash', balance=4231.55),
            Account(name='Savings', kind='cash', balance=18750.00),
            Account(name='Brokerage', kind='investment', balance=52310.42),
        ],
        transactions=[
            Transaction(
                date='2026-06-01',
                description='Salary',
                category='income',
                amount=6200.00,
            ),
            Transaction(
                date='2026-06-03',
                description='Rent',
                category='housing',
                amount=-2100.00,
            ),
            Transaction(
                date='2026-06-05',
                description='Groceries',
                category='food',
                amount=-184.32,
            ),
            Transaction(
                date='2026-06-09',
                description='Restaurant',
                category='food',
                amount=-72.10,
            ),
            Transaction(
                date='2026-06-12',
                description='Electricity',
                category='utilities',
                amount=-96.40,
            ),
            Transaction(
                date='2026-06-15',
                description='Flight',
                category='travel',
                amount=-512.00,
            ),
            Transaction(
                date='2026-06-18',
                description='Bookshop',
                category='shopping',
                amount=-43.99,
            ),
            Transaction(
                date='2026-06-21',
                description='Groceries',
                category='food',
                amount=-201.77,
            ),
        ],
        holdings=[
            Holding(symbol='AAPL', shares=40, price=212.10),
            Holding(symbol='VOO', shares=55, price=512.30),
            Holding(symbol='BTC', shares=0.25, price=61200.00),
        ],
        subscriptions=[
            Subscription(name='Netflix', category='entertainment', amount=15.99),
            Subscription(name='Spotify', category='entertainment', amount=9.99),
            Subscription(name='iCloud+', category='utilities', amount=2.99),
            Subscription(name='Gym membership', category='health', amount=49.00),
            Subscription(name='NYTimes', category='news', amount=17.00),
            Subscription(name='Adobe CC', category='software', amount=59.99),
        ],
        net_worth_history=[61200, 62050, 61480, 63310, 64200, 66120, 65880, 67340],
    ),
}


def get_user_finances(user_id: str) -> UserFinances:
    """Return the finances for `user_id`, raising `KeyError` if there is no such user."""
    return _USERS[user_id]


DEFAULT_USER_ID = 'u_001'
