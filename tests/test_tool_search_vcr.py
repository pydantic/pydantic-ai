"""VCR integration tests for tool search functionality using pydantic-evals.

NOTE: If you change the search tool description or keyword schema in _searchable.py,
re-record all cassettes with: uv run pytest tests/test_tool_search_vcr.py --record-mode=rewrite
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import Agent
from pydantic_ai._utils import is_str_dict
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.run import AgentRunResult

from .conftest import try_import

with try_import() as evals_available:
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext

with try_import() as anthropic_available:
    import anthropic  # pyright: ignore[reportUnusedImport]  # noqa: F401

with try_import() as google_available:
    import google.genai  # pyright: ignore[reportUnusedImport]  # noqa: F401

pytestmark = [
    pytest.mark.skipif(not evals_available(), reason='pydantic-evals not installed'),
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


# --- Eval types ---


class EvalOutput(BaseModel):
    tool_calls: list[str]
    search_interactions: list[tuple[Any, Any]]


class EvalMetadata(BaseModel):
    expected_tools: list[str]
    scenario: str


# --- Evaluators ---

if evals_available():

    @dataclass(repr=False)
    class UsedSearchTools(Evaluator[str, EvalOutput, EvalMetadata]):
        """Check that the model used search_tools when expected tools exist."""

        evaluation_name: str | None = field(default='used_search_tools')

        def evaluate(self, ctx: EvaluatorContext[str, EvalOutput, EvalMetadata]) -> bool:
            if not ctx.metadata or not ctx.metadata.expected_tools:
                return True
            return 'search_tools' in ctx.output.tool_calls

    @dataclass(repr=False)
    class FoundExpectedTools(Evaluator[str, EvalOutput, EvalMetadata]):
        """Check that the model found and called the expected tools."""

        evaluation_name: str | None = field(default='found_expected_tools')

        def evaluate(self, ctx: EvaluatorContext[str, EvalOutput, EvalMetadata]) -> bool:
            if not ctx.metadata or not ctx.metadata.expected_tools:
                return True
            return all(t in ctx.output.tool_calls for t in ctx.metadata.expected_tools)

    @dataclass(repr=False)
    class ReasonableToolUsage(Evaluator[str, EvalOutput, EvalMetadata]):
        """Check that the model didn't use an excessive number of tool calls."""

        max_calls: int = 10
        evaluation_name: str | None = field(default='reasonable_usage')

        def evaluate(self, ctx: EvaluatorContext[str, EvalOutput, EvalMetadata]) -> bool:
            return len(ctx.output.tool_calls) <= self.max_calls

    @dataclass(repr=False)
    class KeywordCount(Evaluator[str, EvalOutput, EvalMetadata]):
        """Score the number of keywords used in the search query. Best is <= 3."""

        evaluation_name: str | None = field(default='keyword_count')

        def evaluate(self, ctx: EvaluatorContext[str, EvalOutput, EvalMetadata]) -> int | dict[str, int]:
            if not ctx.output.search_interactions:
                return {}
            args = ctx.output.search_interactions[0][0]
            if isinstance(args, str):
                args = json.loads(args)
            if is_str_dict(args):
                keywords_str = args.get('keywords', '')
            else:
                return 0
            return len(keywords_str.split())


# --- Helpers ---


def _extract_tool_calls(result: AgentRunResult[str]) -> list[str]:
    tool_calls: list[str] = []
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    tool_calls.append(part.tool_name)
    return tool_calls


def _extract_search_interactions(result: AgentRunResult[str]) -> list[tuple[Any, Any]]:
    """Extract (keywords_args, return_content) pairs from search_tools calls."""
    calls: dict[str, Any] = {}
    interactions: list[tuple[Any, Any]] = []
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_name == 'search_tools':
                    calls[part.tool_call_id] = part.args
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == 'search_tools':
                    interactions.append((calls.get(part.tool_call_id), part.content))
    return interactions


def _build_agent(model_name: str) -> Agent[None, str]:
    """Build an agent with a visible tool and several deferred tools for testing."""
    agent: Agent[None, str] = Agent(model=model_name)

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
    def stock_lookup(symbol: str) -> str:
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


if evals_available():

    def _build_dataset() -> Dataset[str, EvalOutput, EvalMetadata]:
        return Dataset[str, EvalOutput, EvalMetadata](
            cases=[
                Case(
                    name='exchange_rate',
                    inputs='What is the current exchange rate from USD to EUR?',
                    metadata=EvalMetadata(expected_tools=['get_exchange_rate'], scenario='exchange_rate'),
                ),
                Case(
                    name='stock_price',
                    inputs='What is the current stock price for AAPL?',
                    metadata=EvalMetadata(expected_tools=['stock_lookup'], scenario='stock_price'),
                ),
                Case(
                    name='translation',
                    inputs="Translate 'hello, how are you?' to French.",
                    metadata=EvalMetadata(expected_tools=[], scenario='translation'),
                ),
                Case(
                    name='no_matching_tool',
                    inputs='Book a flight from New York to London for next week.',
                    metadata=EvalMetadata(expected_tools=[], scenario='no_matching_tool'),
                ),
            ],
            evaluators=[
                UsedSearchTools(),
                FoundExpectedTools(),
                ReasonableToolUsage(max_calls=10),
                KeywordCount(),
            ],
        )


def _summarize_report(report: Any) -> dict[str, ScenarioSummary]:
    """Extract a compact summary from eval report for snapshotting."""
    summary: dict[str, ScenarioSummary] = {}
    for case in report.cases:
        output: EvalOutput = case.output
        keywords: str | None = None
        if output.search_interactions:
            args = output.search_interactions[0][0]
            if isinstance(args, str):
                args = json.loads(args)
            if is_str_dict(args):
                keywords = args.get('keywords')
        summary[case.name] = ScenarioSummary(keywords=keywords, tool_calls=output.tool_calls)
    return summary


class ScenarioSummary(TypedDict):
    """The search keywords the model chose and the tools it discovered and called."""

    keywords: str | None
    tool_calls: list[str]


@dataclass
class ModelCase:
    model_name: str
    marks: list[pytest.MarkDecorator] = field(default_factory=list[pytest.MarkDecorator])
    scenario_summary: dict[str, ScenarioSummary] = field(default_factory=dict[str, ScenarioSummary])


_CASES = [
    ModelCase(
        model_name='openai:gpt-5-mini',
        scenario_summary=snapshot(
            {
                'exchange_rate': {
                    'keywords': 'exchange rate currency USD EUR forex rate',
                    'tool_calls': ['search_tools', 'get_exchange_rate'],
                },
                'stock_price': {
                    'keywords': 'stock AAPL price finance market quote',
                    'tool_calls': ['search_tools', 'stock_lookup'],
                },
                'translation': {'keywords': None, 'tool_calls': []},
                'no_matching_tool': {'keywords': None, 'tool_calls': []},
            }
        ),
    ),
    ModelCase(
        model_name='anthropic:claude-sonnet-4-5',
        marks=[pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')],
        scenario_summary=snapshot(
            {
                'exchange_rate': {
                    'keywords': 'exchange rate currency conversion USD EUR',
                    'tool_calls': ['search_tools', 'get_exchange_rate'],
                },
                'stock_price': {'keywords': 'stock price quote ticker', 'tool_calls': ['search_tools', 'stock_lookup']},
                'translation': {'keywords': 'translate translation language French', 'tool_calls': ['search_tools']},
                'no_matching_tool': {'keywords': 'flight booking travel reservation', 'tool_calls': ['search_tools']},
            }
        ),
    ),
    ModelCase(
        model_name='google-gla:gemini-3-flash-preview',
        marks=[pytest.mark.skipif(not google_available(), reason='google-genai not installed')],
        scenario_summary=snapshot(
            {
                'exchange_rate': {
                    'keywords': 'exchange rate USD to EUR',
                    'tool_calls': ['search_tools', 'get_exchange_rate'],
                },
                'stock_price': {
                    'keywords': 'stock price financial data AAPL',
                    'tool_calls': ['search_tools', 'stock_lookup'],
                },
                'translation': {'keywords': None, 'tool_calls': []},
                'no_matching_tool': {'keywords': 'flight booking travel', 'tool_calls': ['search_tools']},
            }
        ),
    ),
]


@pytest.mark.parametrize(
    'case',
    [pytest.param(c, id=c.model_name.split(':')[0], marks=c.marks) for c in _CASES],
)
async def test_tool_search_eval(allow_model_requests: None, case: ModelCase) -> None:
    """Evaluate tool search behavior across scenarios using pydantic-evals.

    Runs 4 scenarios per model: exchange_rate, stock_price, translation, no_matching_tool.
    Evaluators check: used_search_tools, found_expected_tools, reasonable_usage, keyword_count.
    """
    agent = _build_agent(case.model_name)

    async def task(prompt: str) -> EvalOutput:
        try:
            result = await agent.run(prompt)
        except UnexpectedModelBehavior:
            return EvalOutput(tool_calls=[], search_interactions=[])
        return EvalOutput(
            tool_calls=_extract_tool_calls(result),
            search_interactions=_extract_search_interactions(result),
        )

    dataset = _build_dataset()
    report = await dataset.evaluate(task, name='tool_search', progress=False, max_concurrency=1)

    assert not report.failures
    for eval_case in report.cases:
        for name, result in eval_case.assertions.items():
            assert result.value, f'{eval_case.name}/{name} failed'

    assert _summarize_report(report) == case.scenario_summary
