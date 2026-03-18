"""VCR integration tests for tool search functionality."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart

from .conftest import try_import

with try_import() as anthropic_available:
    import anthropic  # pyright: ignore[reportUnusedImport]  # noqa: F401

with try_import() as google_available:
    import google.genai  # pyright: ignore[reportUnusedImport]  # noqa: F401

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
]

MOCK_API_KEYS: dict[str, str] = {
    'OPENAI_API_KEY': 'mock-api-key',
    'ANTHROPIC_API_KEY': 'mock-api-key',
    'GOOGLE_API_KEY': 'mock-api-key',
}


@pytest.fixture(autouse=True)
def _mock_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, default in MOCK_API_KEYS.items():
        if not os.getenv(key):  # pragma: no branch
            monkeypatch.setenv(key, default)


def _build_agent() -> Agent[None, str]:
    """Build an agent with a visible tool and several deferred tools for testing."""
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'The weather in {city} is sunny and 72°F.'

    @agent.tool_plain(defer_loading=True)
    def get_exchange_rate(from_currency: str, to_currency: str) -> str:
        """Look up the current exchange rate between two currencies."""
        rates: dict[str, float] = {
            'USD_EUR': 0.92,
            'EUR_USD': 1.09,
            'USD_GBP': 0.79,
            'GBP_USD': 1.27,
        }
        key = f'{from_currency}_{to_currency}'
        rate = rates.get(key, 1.0)
        return f'1 {from_currency} = {rate} {to_currency}'

    @agent.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:  # pragma: no cover
        """Look up stock price by ticker symbol."""
        return f'Stock {symbol}: $150.00'

    @agent.tool_plain(defer_loading=True)
    def mortgage_calculator(principal: float, rate: float, years: int) -> str:  # pragma: no cover
        """Calculate monthly mortgage payment for a home loan."""
        monthly_rate = rate / 12 / 100
        num_payments = years * 12
        if monthly_rate == 0:
            payment = principal / num_payments
        else:
            payment = (
                principal
                * (monthly_rate * (1 + monthly_rate) ** num_payments)
                / ((1 + monthly_rate) ** num_payments - 1)
            )
        return f'Monthly payment: ${payment:.2f}'

    return agent


@dataclass
class Case:
    model_name: str
    expected_output: str


@pytest.mark.parametrize(
    'case',
    [
        pytest.param(
            Case(
                model_name='openai:gpt-5-mini',
                expected_output=snapshot('The current exchange rate is 1 USD = 0.92 EUR.'),
            ),
            id='openai',
        ),
        pytest.param(
            Case(
                model_name='anthropic:claude-sonnet-4-5',
                expected_output=snapshot(
                    'The current exchange rate from USD to EUR is **1 USD = 0.92 EUR**. This means that 1 US Dollar is equal to 0.92 Euros.'
                ),
            ),
            id='anthropic',
            marks=pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed'),
        ),
        pytest.param(
            Case(
                model_name='google-gla:gemini-3-flash-preview',
                expected_output=snapshot('The current exchange rate from USD to EUR is 1 USD = 0.92 EUR.'),
            ),
            id='google',
            marks=pytest.mark.skipif(not google_available(), reason='google-genai not installed'),
        ),
    ],
)
async def test_tool_search_discovers_and_uses_tool(allow_model_requests: None, case: Case) -> None:
    """Test that a model naturally discovers and uses a deferred tool via search.

    The prompt asks a question that requires a tool the model can't see.
    The model must figure out on its own that it should use search_tools to find it.
    """
    agent = _build_agent()
    agent.model = case.model_name

    result = await agent.run('What is the current exchange rate from USD to EUR?')

    assert result.output == case.expected_output

    tool_calls: list[str] = []
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    tool_calls.append(part.tool_name)

    assert 'search_tools' in tool_calls
    assert 'get_exchange_rate' in tool_calls
