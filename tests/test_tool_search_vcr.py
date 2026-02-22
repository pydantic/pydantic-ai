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
        if not os.getenv(key):
            monkeypatch.setenv(key, default)


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
                expected_output=snapshot(
                    'I searched for "mortgage" and found the mortgage_calculator tool. Using it for a $400,000 loan at 6.5% interest over 30 years gives a monthly payment of $2,528.27.'
                ),
            ),
            id='openai',
        ),
        pytest.param(
            Case(
                model_name='anthropic:claude-sonnet-4-5',
                expected_output=snapshot("""\
Perfect! Here are the results:

**Search Results:** Found 1 tool matching "mortgage" - the mortgage_calculator tool for calculating monthly mortgage payments.

**Mortgage Calculation:**
- Loan Amount (Principal): $400,000
- Interest Rate: 6.5%
- Loan Term: 30 years
- **Monthly Payment: $2,528.27**

This monthly payment includes principal and interest. Keep in mind that your actual monthly housing payment may be higher when you add property taxes, homeowners insurance, and potentially PMI (Private Mortgage Insurance) if applicable.\
"""),
            ),
            id='anthropic',
            marks=pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed'),
        ),
        pytest.param(
            Case(
                model_name='google-gla:gemini-3-flash-preview',
                expected_output=snapshot(
                    'OK. I searched for "mortgage" and found the `mortgage_calculator` tool. Using that tool for a $400,000 loan at 6.5% interest over 30 years, the calculated monthly payment is $2,528.27.'
                ),
            ),
            id='google',
            marks=pytest.mark.skipif(not google_available(), reason='google-genai not installed'),
        ),
    ],
)
async def test_tool_search_discovers_and_uses_tool(allow_model_requests: None, case: Case) -> None:
    """Test that an agent can discover a deferred tool via search and use it.

    This test verifies the complete tool search flow:
    1. Agent is created with deferred tools (not initially visible to model)
    2. Model uses search_tools to find the mortgage_calculator tool
    3. Model calls the discovered tool with the correct arguments
    4. Tool returns the calculated result
    """
    agent: Agent[None, str] = Agent(case.model_name)

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'The weather in {city} is sunny and 72°F.'

    @agent.tool_plain(defer_loading=True)
    def mortgage_calculator(principal: float, rate: float, years: int) -> str:
        """Mortgage calculator - calculate monthly mortgage payment for a home loan."""
        monthly_rate = rate / 12 / 100
        num_payments = years * 12
        if monthly_rate == 0:
            payment = principal / num_payments  # pragma: no cover
        else:
            payment = (
                principal
                * (monthly_rate * (1 + monthly_rate) ** num_payments)
                / ((1 + monthly_rate) ** num_payments - 1)
            )
        return f'Monthly payment: ${payment:.2f}'

    @agent.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:  # pragma: no cover
        """Look up stock price by ticker symbol."""
        return f'Stock {symbol}: $150.00'

    result = await agent.run(
        'Use search_tools to search for "mortgage", then use the mortgage_calculator tool '
        'to calculate the monthly payment for a $400,000 loan at 6.5% interest for 30 years.'
    )

    assert result.output == case.expected_output

    tool_calls: list[str] = []
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    tool_calls.append(part.tool_name)

    assert 'search_tools' in tool_calls
    assert 'mortgage_calculator' in tool_calls
