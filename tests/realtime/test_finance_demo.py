"""Tests for the realtime finance demo core (widgets, data, supervisor, voice delegation)."""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import pytest
from pydantic_ai_examples.realtime_finance.data import DEFAULT_USER_ID, get_user_finances
from pydantic_ai_examples.realtime_finance.supervisor import FinanceDeps, create_finance_supervisor
from pydantic_ai_examples.realtime_finance.voice import BACKGROUND_TOOLS, VoiceDeps, create_voice_front
from pydantic_ai_examples.realtime_finance.widgets import (
    SpendingWidget,
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

from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.realtime import (
    RealtimeConnection,
    RealtimeEvent,
    RealtimeInput,
    RealtimeModel,
    TextInput,
    ToolCall,
    ToolResult,
    Transcript,
    TurnComplete,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

pytestmark = pytest.mark.anyio


class _RoutingConnection(RealtimeConnection):
    """A fake connection: routes a text turn to `ask_analyst` and speaks the tool result back."""

    def __init__(self) -> None:
        self._events: asyncio.Queue[RealtimeEvent] = asyncio.Queue()
        self._ids = itertools.count(1)

    async def send(self, content: RealtimeInput) -> None:
        if isinstance(content, TextInput):
            await self._events.put(
                ToolCall(
                    tool_call_id=f'call_{next(self._ids)}',
                    tool_name='ask_analyst',
                    args=json.dumps({'question': content.text}),
                )
            )
        else:
            assert isinstance(content, ToolResult)
            await self._events.put(Transcript(text=content.output, is_final=True))
            await self._events.put(TurnComplete())

    async def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            yield await self._events.get()


class _RoutingModel(RealtimeModel):
    @property
    def model_name(self) -> str:
        return 'fake-realtime'

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str,
        tools: list[ToolDefinition] | None = None,
        native_tools: list[AbstractNativeTool] | None = None,
        model_settings: ModelSettings | None = None,
    ) -> AsyncGenerator[_RoutingConnection]:
        yield _RoutingConnection()


def test_widget_builders() -> None:
    fin = get_user_finances(DEFAULT_USER_ID)
    accounts = accounts_widget(fin)
    assert accounts.total == sum(a.balance for a in fin.accounts)

    spending = spending_widget(fin)
    assert spending.total == sum(c.amount for c in spending.categories)
    assert all(c.amount > 0 for c in spending.categories)  # income excluded, positives only
    assert spending.categories == sorted(spending.categories, key=lambda c: c.amount, reverse=True)

    port = portfolio_widget(fin)
    assert port.total == sum(h.value for h in port.holdings)
    assert port.holdings[0].value == port.holdings[0].shares * port.holdings[0].price

    nw = net_worth_widget(fin)
    assert nw.change == nw.end - nw.start

    tx = transactions_widget(fin, limit=3)
    assert len(tx.rows) == 3
    assert tx.rows == sorted(tx.rows, key=lambda r: r.date, reverse=True)


def test_new_widget_builders() -> None:
    fin = get_user_finances(DEFAULT_USER_ID)

    subs = subscriptions_widget(fin)
    assert subs.monthly_total == sum(s.amount for s in fin.subscriptions)
    assert subs.annual_total == subs.monthly_total * 12

    budget = budget_widget(fin)
    assert budget.monthly_savings >= 0
    assert all(r.suggested <= r.current for r in budget.rows)

    proj = savings_projection_widget(fin, years=2, extra_monthly=500)
    assert len(proj.points) == 3  # now + 2 years
    assert proj.points[0].label == 'now'
    assert proj.final == proj.points[-1].value

    insights = insights_widget(fin)
    assert insights.items
    assert all(i.severity in ('info', 'good', 'warn') for i in insights.items)


def test_widget_serializes_with_kind() -> None:
    dumped = spending_widget(get_user_finances(DEFAULT_USER_ID)).model_dump()
    assert dumped['kind'] == 'spending'
    assert isinstance(dumped['categories'], list)


def test_get_user_finances_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_user_finances('nope')


async def test_supervisor_collects_widgets_with_test_model() -> None:
    # TestModel calls every tool, so each appends its widget to the run deps.
    agent = create_finance_supervisor(TestModel())
    deps = FinanceDeps()
    result = await agent.run('full review', deps=deps)
    assert isinstance(result.output, str)
    kinds = {w.kind for w in deps.widgets}
    assert {
        'accounts',
        'spending',
        'portfolio',
        'net_worth',
        'transactions',
        'subscriptions',
        'budget',
        'projection',
        'insights',
    } <= kinds


async def test_voice_front_bubbles_widgets_through_mock() -> None:
    voice = create_voice_front()
    deps = VoiceDeps(supervisor=create_finance_supervisor(TestModel()))
    async with voice.realtime_session(model=_RoutingModel(), deps=deps, background_tools=BACKGROUND_TOOLS) as session:
        await session.send_text('how much did I spend last month?')
        async for event in session:
            if isinstance(event, TurnComplete):
                break

    # The analyst delegation ran and surfaced typed widgets keyed by the tool call id.
    assert deps.widgets
    all_widgets = [w for batch in deps.widgets.values() for w in batch]
    assert all(hasattr(w, 'kind') for w in all_widgets)
    assert any(isinstance(w, SpendingWidget) for w in all_widgets)


def test_background_tools_marks_long_scenarios_async() -> None:
    assert {'run_deep_analysis', 'plan_savings_goal'} <= BACKGROUND_TOOLS
    assert 'ask_analyst' not in BACKGROUND_TOOLS
