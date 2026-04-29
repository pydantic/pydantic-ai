"""Tests for tool search functionality.

Unit tests for ToolSearchToolset plus VCR integration tests using pydantic-evals.

NOTE: If you change the search tool description or keyword schema in _tool_search.py,
re-record all cassettes with: uv run pytest tests/test_tool_search.py --record-mode=rewrite
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import Agent, FunctionToolset, ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai.builtin_tools import ToolSearchTool
from pydantic_ai.capabilities._ordering import collect_leaves
from pydantic_ai.capabilities._tool_search import ToolSearch
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolReturn, ToolReturnPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._tool_search import (
    _SEARCH_TOOLS_NAME,  # pyright: ignore[reportPrivateUsage]
    ToolSearchToolset,
)
from pydantic_ai.usage import RunUsage

from .conftest import try_import

with try_import() as evals_available:
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext
    from pydantic_evals.reporting import EvaluationReport

with try_import() as anthropic_available:
    import anthropic  # pyright: ignore[reportUnusedImport]  # noqa: F401

with try_import() as google_available:
    import google.genai  # pyright: ignore[reportUnusedImport]  # noqa: F401

pytestmark = pytest.mark.anyio

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
    search_args: list[dict[str, str]]


class EvalMetadata(BaseModel):
    expected_tools: list[str]


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
            if not ctx.output.search_args:
                return {}
            keywords_str = ctx.output.search_args[0].get('keywords', '')
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


def _extract_search_args(result: AgentRunResult[str]) -> list[dict[str, str]]:
    """Extract parsed keyword args dicts from search_tools calls."""
    args_list: list[dict[str, str]] = []
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_name == 'search_tools' and part.args is not None:
                    raw_args: dict[str, Any] = json.loads(part.args) if isinstance(part.args, str) else part.args
                    args_list.append({k: str(v) for k, v in raw_args.items()})
    return args_list


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
            name='tool_search',
            cases=[
                Case(
                    name='exchange_rate',
                    inputs='What is the current exchange rate from USD to EUR?',
                    metadata=EvalMetadata(expected_tools=['get_exchange_rate']),
                ),
                Case(
                    name='stock_price',
                    inputs='What is the current stock price for AAPL?',
                    metadata=EvalMetadata(expected_tools=['stock_lookup']),
                ),
                Case(
                    name='translation',
                    inputs="Translate 'hello, how are you?' to French.",
                    metadata=EvalMetadata(expected_tools=[]),
                ),
                Case(
                    name='no_matching_tool',
                    inputs='Book a flight from New York to London for next week.',
                    metadata=EvalMetadata(expected_tools=[]),
                ),
            ],
            evaluators=[
                UsedSearchTools(),
                FoundExpectedTools(),
                ReasonableToolUsage(max_calls=5),
                KeywordCount(),
            ],
        )


def _summarize_report(report: EvaluationReport[str, EvalOutput, EvalMetadata]) -> dict[str, ScenarioSummary]:
    """Extract a compact summary from eval report for snapshotting."""
    summary: dict[str, ScenarioSummary] = {}
    for case in report.cases:
        output: EvalOutput = case.output
        keywords: str | None = None
        if output.search_args:
            keywords = output.search_args[0].get('keywords')
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
        model_name='openai:gpt-5.4-mini',
        scenario_summary=snapshot(
            {
                'exchange_rate': {
                    'keywords': 'exchange rate currency USD EUR',
                    'tool_calls': ['search_tools', 'get_exchange_rate'],
                },
                'stock_price': {
                    'keywords': 'stock price market quote finance AAPL',
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
                    'keywords': 'exchange rate currency USD EUR',
                    'tool_calls': ['search_tools', 'get_exchange_rate'],
                },
                'stock_price': {'keywords': 'stock price quote ticker', 'tool_calls': ['search_tools', 'stock_lookup']},
                'translation': {'keywords': 'translate translation language French', 'tool_calls': ['search_tools']},
                'no_matching_tool': {
                    'keywords': 'flight booking reservation travel airline',
                    'tool_calls': ['search_tools'],
                },
            }
        ),
    ),
    ModelCase(
        model_name='google-gla:gemini-3-flash-preview',
        marks=[pytest.mark.skipif(not google_available(), reason='google-genai not installed')],
        scenario_summary=snapshot(
            {
                'exchange_rate': {
                    'keywords': 'USD to EUR exchange rate current',
                    'tool_calls': ['search_tools', 'get_exchange_rate'],
                },
                'stock_price': {
                    'keywords': 'stock price finance market',
                    'tool_calls': ['search_tools', 'stock_lookup'],
                },
                'translation': {'keywords': None, 'tool_calls': []},
                'no_matching_tool': {'keywords': 'flight booking search reservation', 'tool_calls': ['search_tools']},
            }
        ),
    ),
]


@pytest.mark.skipif(not evals_available(), reason='pydantic-evals not installed')
@pytest.mark.vcr
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
        except UnexpectedModelBehavior:  # pragma: no cover
            return EvalOutput(tool_calls=[], search_args=[])
        return EvalOutput(
            tool_calls=_extract_tool_calls(result),
            search_args=_extract_search_args(result),
        )

    dataset = _build_dataset()
    report = await dataset.evaluate(task, name='tool_search', progress=False, max_concurrency=1)

    assert not report.failures
    for eval_case in report.cases:
        for name, result in eval_case.assertions.items():
            assert result.value, f'{eval_case.name}/{name} failed'

    assert _summarize_report(report) == case.scenario_summary


# --- Unit tests ---

T = TypeVar('T')


def _build_run_context(
    deps: T,
    run_step: int = 0,
    messages: list[ModelMessage] | None = None,
) -> RunContext[T]:
    """Build a ``RunContext`` for unit tests using ``TestModel``."""
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=messages or [],
        run_step=run_step,
    )


def _create_function_toolset() -> FunctionToolset[None]:
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    @toolset.tool_plain
    def get_time(timezone: str) -> str:  # pragma: no cover
        """Get the current time in a timezone."""
        return f'Time in {timezone}'

    @toolset.tool_plain(defer_loading=True)
    def calculate_mortgage(principal: float, rate: float, years: int) -> str:
        """Calculate monthly mortgage payment for a loan."""
        return 'Mortgage calculated'

    @toolset.tool_plain(defer_loading=True)
    def stock_price(symbol: str) -> str:  # pragma: no cover
        """Get the current stock price for a symbol."""
        return f'Stock price for {symbol}'

    @toolset.tool_plain(defer_loading=True)
    def crypto_price(coin: str) -> str:  # pragma: no cover
        """Get the current cryptocurrency price."""
        return f'Crypto price for {coin}'

    return toolset


async def test_tool_search_toolset_filters_deferred_tools():
    """On the local path, deferred tools stay hidden until discovered — only the
    visible tools and the ``search_tools`` function are exposed up front."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert tool_names == snapshot(
        [
            'get_weather',
            'get_time',
            'calculate_mortgage~managed:tool_search',
            'stock_price~managed:tool_search',
            'crypto_price~managed:tool_search',
            'search_tools',
        ]
    )


async def test_search_tool_def_description_and_schema():
    """Test that the search tool definition includes deferred count and TypeAdapter-generated schema."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    assert search_tool.tool_def.description == snapshot(
        'There are additional tools not yet visible to you. When you need a capability not provided by your current tools, search here by providing specific keywords to discover and activate relevant tools. Each keyword is matched independently against tool names and descriptions. If no tools are found, they do not exist — do not retry.'
    )
    assert search_tool.tool_def.parameters_json_schema == snapshot(
        {
            'properties': {
                'keywords': {
                    'description': 'Space-separated keywords to match against tool names and descriptions. Use specific words likely to appear in tool names or descriptions to narrow down relevant tools.',
                    'title': 'Keywords',
                    'type': 'string',
                }
            },
            'required': ['keywords'],
            'title': 'SearchToolArgs',
            'type': 'object',
        }
    )


async def test_tool_search_toolset_search_returns_matching_tools():
    """Test that search_tools returns matching deferred tools."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'mortgage'}, ctx, search_tool)
    assert result == snapshot(
        ToolReturn(
            return_value={
                'tools': [
                    {'name': 'calculate_mortgage', 'description': 'Calculate monthly mortgage payment for a loan.'}
                ]
            },
        )
    )


async def test_tool_search_toolset_search_is_case_insensitive():
    """Test that search is case insensitive."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'STOCK'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    rv = cast('dict[str, Any]', result.return_value)
    assert len(rv['tools']) == 1
    assert rv['tools'][0]['name'] == 'stock_price'


async def test_tool_search_toolset_search_matches_description():
    """Test that search matches tool descriptions."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'cryptocurrency'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    rv = cast('dict[str, Any]', result.return_value)
    assert len(rv['tools']) == 1
    assert rv['tools'][0]['name'] == 'crypto_price'


async def test_tool_search_toolset_prefers_specific_term_matches():
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def github_get_me() -> str:  # pragma: no cover
        """Get the authenticated GitHub profile."""
        return 'me'

    @toolset.tool_plain(defer_loading=True)
    def github_create_gist() -> str:  # pragma: no cover
        """Create a new GitHub gist."""
        return 'gist'

    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'github profile'}, ctx, search_tool)
    assert result == snapshot(
        ToolReturn(
            return_value={
                'tools': [
                    {'name': 'github_get_me', 'description': 'Get the authenticated GitHub profile.'},
                    {'name': 'github_create_gist', 'description': 'Create a new GitHub gist.'},
                ]
            },
        )
    )


async def test_tool_search_toolset_keeps_lower_scoring_matches_after_top_hits():
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def stock_price() -> str:  # pragma: no cover
        """Get the current stock price."""
        return 'stock'

    @toolset.tool_plain(defer_loading=True)
    def crypto_price() -> str:  # pragma: no cover
        """Get the current cryptocurrency price."""
        return 'crypto'

    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'stock price'}, ctx, search_tool)
    assert result == snapshot(
        ToolReturn(
            return_value={
                'tools': [
                    {'name': 'stock_price', 'description': 'Get the current stock price.'},
                    {'name': 'crypto_price', 'description': 'Get the current cryptocurrency price.'},
                ]
            },
        )
    )


async def test_tool_search_toolset_does_not_match_substrings_inside_words():
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def github_get_me() -> str:  # pragma: no cover
        """Get my GitHub profile."""
        return 'me'

    @toolset.tool_plain(defer_loading=True)
    def github_add_comment_to_pending_review() -> str:  # pragma: no cover
        """Add a pending review comment on GitHub."""
        return 'comment'

    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'get me'}, ctx, search_tool)
    assert result == snapshot(
        ToolReturn(
            return_value={'tools': [{'name': 'github_get_me', 'description': 'Get my GitHub profile.'}]},
        )
    )


async def test_tool_search_toolset_search_returns_no_matches():
    """Test that search returns empty list when no matches."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'nonexistent'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    assert result.return_value == snapshot({'tools': []})


async def test_tool_search_toolset_search_empty_query():
    """Test that search with empty query raises ModelRetry."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    with pytest.raises(ModelRetry, match='Please provide search keywords.'):
        await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': ''}, ctx, search_tool)


@pytest.mark.parametrize('keywords', ['   ', '---', '!!!', '...'])
async def test_tool_search_toolset_search_non_tokenizable_query(keywords: str):
    """Queries that tokenize to an empty set must retry, not match every tool."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    with pytest.raises(ModelRetry, match='Please provide search keywords.'):
        await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': keywords}, ctx, search_tool)


async def test_tool_search_toolset_max_results():
    """Test that results are capped at `_MAX_SEARCH_RESULTS` (10)."""
    toolset: FunctionToolset[None] = FunctionToolset()

    for i in range(15):

        @toolset.tool_plain(defer_loading=True, name=f'tool_{i}')
        def tool_func() -> str:  # pragma: no cover
            """A tool for testing."""
            return 'result'

    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'tool'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    rv = cast('dict[str, Any]', result.return_value)
    assert len(rv['tools']) == 10


async def test_tool_search_toolset_discovered_tools_available():
    """Test that discovered tools become available after search."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'tools': [{'name': 'calculate_mortgage', 'description': None}]},
                ),
            ]
        )
    ]
    ctx = _build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert 'calculate_mortgage' in tool_names
    assert 'stock_price' not in tool_names


async def test_tool_search_toolset_omits_search_tool_once_all_deferred_tools_are_discovered():
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={
                        'tools': [
                            {'name': 'calculate_mortgage', 'description': None},
                            {'name': 'stock_price', 'description': None},
                            {'name': 'crypto_price', 'description': None},
                        ]
                    },
                )
            ]
        )
    ]
    ctx = _build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert tool_names == snapshot(
        [
            'get_weather',
            'get_time',
            'calculate_mortgage~managed:tool_search',
            'calculate_mortgage',
            'stock_price~managed:tool_search',
            'stock_price',
            'crypto_price~managed:tool_search',
            'crypto_price',
        ]
    )


async def test_tool_search_toolset_reserved_name_collision():
    """Test that `UserError` is raised if a tool is named 'search_tools' and deferred tools exist."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool_plain
    def search_tools(query: str) -> str:  # pragma: no cover
        """Search for tools."""
        return 'search result'

    @toolset.tool_plain(defer_loading=True)
    def deferred_tool() -> str:  # pragma: no cover
        """A deferred tool to trigger search injection."""
        return 'deferred'

    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    with pytest.raises(UserError, match="Tool name 'search_tools' is reserved"):
        await searchable.get_tools(ctx)


async def test_tool_search_toolset_no_deferred_tools_returns_all():
    """Test that when there are no deferred tools, all tools are returned without search_tools."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    @toolset.tool_plain
    def get_time(timezone: str) -> str:  # pragma: no cover
        """Get the current time in a timezone."""
        return f'Time in {timezone}'

    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert tool_names == snapshot(['get_weather', 'get_time'])


async def test_agent_auto_injects_tool_search_capability():
    """Test that agent auto-injects ToolSearch capability, with and without deferred tools."""
    agent_no_deferred = Agent('test')

    @agent_no_deferred.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    leaves = collect_leaves(agent_no_deferred.root_capability)
    assert any(isinstance(leaf, ToolSearch) for leaf in leaves)

    agent_with_deferred = Agent('test')

    @agent_with_deferred.tool_plain
    def get_weather2(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    @agent_with_deferred.tool_plain(defer_loading=True)
    def calculate_mortgage(principal: float) -> str:  # pragma: no cover
        """Calculate mortgage payment."""
        return 'Calculated'

    leaves = collect_leaves(agent_with_deferred.root_capability)
    assert any(isinstance(leaf, ToolSearch) for leaf in leaves)


async def test_explicit_tool_search_not_duplicated():
    """Passing ToolSearch explicitly doesn't result in a second auto-injected one."""
    agent = Agent('test', capabilities=[ToolSearch()])

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    leaves = collect_leaves(agent.root_capability)
    tool_search_count = sum(1 for leaf in leaves if isinstance(leaf, ToolSearch))
    assert tool_search_count == 1


def test_tool_search_in_capability_registry():
    """ToolSearch is a public, spec-constructible capability."""
    from pydantic_ai.capabilities import CAPABILITY_TYPES

    assert ToolSearch.get_serialization_name() == 'ToolSearch'
    assert CAPABILITY_TYPES['ToolSearch'] is ToolSearch


async def test_tool_manager_with_tool_search_toolset_exposes_both_variants():
    """The toolset emits both representations of every deferred tool — a managed variant
    carrying ``managed_by_builtin='tool_search'`` and, for discovered tools, a regular
    variant. ``Model.prepare_request`` filters to one based on model support."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tool_manager = ToolManager[None](searchable)
    run_step_toolset = await tool_manager.for_run_step(ctx)

    managed_names = {t.name for t in run_step_toolset.tool_defs if t.managed_by_builtin == 'tool_search'}
    assert managed_names == {'calculate_mortgage', 'stock_price', 'crypto_price'}

    local_names = [t.name for t in run_step_toolset.tool_defs if not t.managed_by_builtin]
    assert 'get_weather' in local_names
    assert 'search_tools' in local_names
    # Undiscovered deferred tools don't appear in regular form.
    assert 'calculate_mortgage' not in local_names

    # Dispatch works under the plain name regardless of whether the entry only
    # exists under the suffixed `~managed:tool_search` key.
    result = await run_step_toolset.handle_call(
        ToolCallPart(tool_name='calculate_mortgage', args={'principal': 100.0, 'rate': 5.0, 'years': 30})
    )
    assert 'Mortgage calculated' in str(result)

    # The local search_tools function is also dispatchable.
    result = await run_step_toolset.handle_call(ToolCallPart(tool_name='search_tools', args={'keywords': 'mortgage'}))
    assert 'calculate_mortgage' in str(result)


async def test_tool_search_toolset_tool_with_none_description():
    """Test that tools with None description are handled correctly in search."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def no_desc_tool() -> str:  # pragma: no cover
        return 'no description'

    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'no_desc'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    assert result.return_value == snapshot({'tools': [{'name': 'no_desc_tool', 'description': None}]})


async def test_tool_search_toolset_multiple_searches_accumulate():
    """Test that tools discovered in multiple searches accumulate correctly."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'tools': [{'name': 'calculate_mortgage', 'description': None}]},
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'tools': [{'name': 'stock_price', 'description': None}]},
                ),
            ]
        ),
    ]
    ctx = _build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert 'calculate_mortgage' in tool_names
    assert 'stock_price' in tool_names
    assert 'crypto_price' not in tool_names


async def test_function_toolset_all_deferred():
    """Test FunctionToolset with all tools having defer_loading=True."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def deferred_tool1() -> str:  # pragma: no cover
        """First deferred tool."""
        return 'result1'

    @toolset.tool_plain(defer_loading=True)
    def deferred_tool2() -> str:  # pragma: no cover
        """Second deferred tool."""
        return 'result2'

    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert tool_names == snapshot(
        ['deferred_tool1~managed:tool_search', 'deferred_tool2~managed:tool_search', 'search_tools']
    )


async def test_tool_search_toolset_ignores_malformed_content_history():
    """Discovery reads ``content`` via ``extract_tool_search_return``; malformed shapes
    (non-dict content, non-list ``tools``, entries missing a string ``name``) are
    ignored, and only well-formed entries are picked up."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        # content is not a dict
        ModelRequest(parts=[ToolReturnPart(tool_name=_SEARCH_TOOLS_NAME, content='not a dict')]),
        # content is a dict but `tools` is missing
        ModelRequest(parts=[ToolReturnPart(tool_name=_SEARCH_TOOLS_NAME, content={'message': 'hi'})]),
        # `tools` is not a list
        ModelRequest(parts=[ToolReturnPart(tool_name=_SEARCH_TOOLS_NAME, content={'tools': 'not a list'})]),
        # `tools` contains malformed entries (non-dict, missing/non-string name)
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'tools': ['not a dict', {'name': 123}, {'description': 'no name'}]},
                ),
            ]
        ),
        # valid content
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'tools': [{'name': 'calculate_mortgage', 'description': None}]},
                ),
            ]
        ),
    ]
    ctx = _build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    assert 'calculate_mortgage' in tools
    assert 'stock_price' not in tools
    assert 'crypto_price' not in tools


async def test_deferred_loading_toolset_marks_all_tools():
    """Test that DeferredLoadingToolset marks all tools for deferred loading when tool_names is None."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool_plain
    def tool_a() -> str:  # pragma: no cover
        """Tool A."""
        return 'a'

    @toolset.tool_plain
    def tool_b() -> str:  # pragma: no cover
        """Tool B."""
        return 'b'

    deferred = toolset.defer_loading()
    searchable = ToolSearchToolset(wrapped=deferred)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    assert 'search_tools' in tools
    assert 'tool_a' not in tools
    assert 'tool_b' not in tools


async def test_deferred_loading_toolset_marks_specific_tools():
    """Test that DeferredLoadingToolset marks only named tools for deferred loading."""
    toolset: FunctionToolset[None] = FunctionToolset()

    @toolset.tool_plain
    def tool_a() -> str:  # pragma: no cover
        """Tool A."""
        return 'a'

    @toolset.tool_plain
    def tool_b() -> str:  # pragma: no cover
        """Tool B."""
        return 'b'

    deferred = toolset.defer_loading(['tool_b'])
    searchable = ToolSearchToolset(wrapped=deferred)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    assert 'search_tools' in tools
    assert 'tool_a' in tools
    assert 'tool_b' not in tools


async def test_tool_search_toolset_emits_managed_variant_under_suffixed_key():
    """Every deferred tool has a managed variant under a ``~managed:tool_search``
    suffixed key regardless of the current model — the adapter's ``prepare_request``
    filters to it when the model supports native tool search, so the toolset can't
    commit early (e.g. under ``FallbackModel``)."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)

    managed = {name: tool.tool_def for name, tool in tools.items() if tool.tool_def.managed_by_builtin}
    assert set(managed) == {
        'calculate_mortgage~managed:tool_search',
        'stock_price~managed:tool_search',
        'crypto_price~managed:tool_search',
    }
    for tool_def in managed.values():
        assert tool_def.managed_by_builtin == 'tool_search'
        assert tool_def.defer_loading
    # The local fallback is still present — dropped by the adapter via ``prefer_builtin``.
    assert _SEARCH_TOOLS_NAME in tools


async def test_tool_search_toolset_dispatches_by_plain_name_via_tool_manager():
    """The provider calls a deferred tool by its plain name. ``ToolManager`` dispatches
    to the same underlying ``ToolsetTool`` even when only the ``~managed:`` variant is
    present in the toolset output (undiscovered tool)."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tool_manager = ToolManager[None](searchable)
    run_step_toolset = await tool_manager.for_run_step(ctx)
    result = await run_step_toolset.handle_call(
        ToolCallPart(tool_name='calculate_mortgage', args={'principal': 100.0, 'rate': 5.0, 'years': 30})
    )
    assert 'Mortgage calculated' in str(result)


async def test_tool_search_toolset_custom_search_fn_is_used():
    """A custom ``search_fn`` replaces the default token-matching algorithm."""
    calls: list[str] = []

    def custom_search(query: str, tools: Sequence[ToolDefinition]) -> list[str]:
        calls.append(query)
        # Pick anything with 'price' in the name, regardless of query tokens.
        return [t.name for t in tools if 'price' in t.name]

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset, search_fn=custom_search)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'anything'}, ctx, tools[_SEARCH_TOOLS_NAME])
    assert isinstance(result, ToolReturn)
    assert result.return_value == {
        'tools': [
            {'name': 'stock_price', 'description': 'Get the current stock price for a symbol.'},
            {'name': 'crypto_price', 'description': 'Get the current cryptocurrency price.'},
        ]
    }
    assert calls == ['anything']


async def test_tool_search_toolset_custom_search_fn_still_emits_managed_variants():
    """A custom ``search_fn`` handles local discovery, but the toolset still emits the
    managed variants of deferred tools — when the model supports native tool search
    (including provider-side custom callable modes like Anthropic's tool_reference
    mechanism or OpenAI's ``execution='client'``), the adapter keeps them and applies
    ``defer_loading`` on the wire. Commitment to native-vs-local happens in
    ``Model.prepare_request``, not here."""

    def custom_search(query: str, tools: Sequence[ToolDefinition]) -> list[str]:  # pragma: no cover
        return []

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset, search_fn=custom_search)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)

    managed = [t.tool_def.name for t in tools.values() if t.tool_def.managed_by_builtin == 'tool_search']
    assert set(managed) == {'calculate_mortgage', 'stock_price', 'crypto_price'}
    assert _SEARCH_TOOLS_NAME in tools


@pytest.mark.vcr
@pytest.mark.filterwarnings(
    'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
)
async def test_anthropic_native_tool_search_round_trip(allow_model_requests: None, anthropic_api_key: str) -> None:
    """End-to-end against live Anthropic: native BM25 server-side tool search
    populates `BuiltinToolCallPart` / `BuiltinToolReturnPart`, the model invokes
    the discovered deferred tool by its plain name, and the wire request carries
    `defer_loading: true` on the corpus tools and the `tool_search_tool_bm25`
    builtin.
    """
    pytest.importorskip('anthropic')
    from pathlib import Path

    import yaml

    from pydantic_ai.messages import BuiltinToolCallPart, BuiltinToolReturnPart
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent: Agent[None, str] = Agent(model=model)

    @agent.tool_plain(defer_loading=True)
    def get_exchange_rate(from_currency: str, to_currency: str) -> str:
        """Look up the current exchange rate between two currencies."""
        return f'1 {from_currency} = 0.92 {to_currency}'

    @agent.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:  # pragma: no cover
        """Look up stock price by ticker symbol."""
        return f'Stock {symbol}: $150.00'

    result = await agent.run('What is the current USD to EUR exchange rate?')

    parts_by_kind: list[type] = [type(p) for m in result.all_messages() for p in m.parts]
    assert BuiltinToolCallPart in parts_by_kind
    assert BuiltinToolReturnPart in parts_by_kind

    # The model's follow-up tool call for the discovered tool dispatches by its plain
    # name — the toolset exposes deferred tools as their regular variant on the native
    # path so the dispatch doesn't fall through to an "unknown tool" retry.
    rate_returns = [
        p
        for m in result.all_messages()
        for p in m.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    # Wire-level checks against the live cassette.
    cassette_path = (
        Path(__file__).parent / 'cassettes' / 'test_tool_search' / 'test_anthropic_native_tool_search_round_trip.yaml'
    )
    cassette = cast(dict[str, Any], yaml.safe_load(cassette_path.read_text(encoding='utf-8')))
    interactions = cast(list[dict[str, Any]], cassette['interactions'])

    # Initial request: deferred tools ship with `defer_loading: true`, and the BM25
    # builtin is registered alongside.
    first_request = cast(dict[str, Any], interactions[0]['request']['parsed_body'])
    deferred_names = {
        cast(str, t['name'])
        for t in cast(list[dict[str, Any]], first_request['tools'])
        if t.get('defer_loading') is True
    }
    assert deferred_names == {'get_exchange_rate', 'stock_lookup'}
    builtin_tool_types = {
        cast(str, t.get('type'))
        for t in cast(list[dict[str, Any]], first_request['tools'])
        if cast(str, t.get('type', '')).startswith('tool_search_tool_')
    }
    assert builtin_tool_types == {'tool_search_tool_bm25_20251119'}

    # Provisional beta header is rejected by the API — confirm we don't send it.
    assert 'tool-search-tool-2025-11-19' not in (first_request.get('betas') or [])

    # First response contains the server-side tool search round trip.
    first_response_blocks = cast(list[dict[str, Any]], interactions[0]['response']['parsed_body']['content'])
    assert any(
        b.get('type') == 'server_tool_use' and b.get('name') == 'tool_search_tool_bm25' for b in first_response_blocks
    )
    assert any(b.get('type') == 'tool_search_tool_result' for b in first_response_blocks)


@pytest.mark.vcr
async def test_anthropic_custom_callable_round_trip(allow_model_requests: None, anthropic_api_key: str) -> None:
    """End-to-end: a custom callable ``ToolSearch`` strategy runs locally but still
    surfaces natively on Anthropic — deferred tools ship with ``defer_loading: true``,
    the model invokes the regular ``search_tools`` function tool, and our
    ``tool_result`` is formatted as ``tool_reference`` blocks so the discovered tool
    gets unlocked for the next turn."""
    pytest.importorskip('anthropic')
    import yaml

    from pydantic_ai.capabilities import ToolSearch
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    def match_exchange_rate(query: str, tools: Sequence[ToolDefinition]) -> list[str]:
        # Deterministic: always point the model at `get_exchange_rate` so the cassette
        # replay doesn't depend on the exact keywords the model picks.
        return ['get_exchange_rate']

    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent: Agent[None, str] = Agent(
        model=model,
        capabilities=[ToolSearch(strategy=match_exchange_rate)],
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city} is sunny.'

    @agent.tool_plain(defer_loading=True)
    def get_exchange_rate(from_currency: str, to_currency: str) -> str:
        """Look up the current exchange rate between two currencies."""
        return f'1 {from_currency} = 0.92 {to_currency}'

    @agent.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:  # pragma: no cover
        """Look up stock price by ticker symbol."""
        return f'Stock {symbol}: $150.00'

    result = await agent.run('What is the USD to EUR exchange rate?')

    # The full sequence: user prompt -> model asks `search_tools` -> our local callable
    # returns discovered tool names -> model follows up with the discovered tool ->
    # we run it -> model replies with final text.
    part_shape = [
        [(type(part).__name__, getattr(part, 'tool_name', None)) for part in msg.parts] for msg in result.all_messages()
    ]
    assert part_shape == snapshot(
        [
            [('UserPromptPart', None)],
            [('TextPart', None), ('ToolCallPart', 'search_tools')],
            [('ToolReturnPart', 'search_tools')],
            [('TextPart', None), ('ToolCallPart', 'get_exchange_rate')],
            [('ToolReturnPart', 'get_exchange_rate')],
            [('TextPart', None)],
        ]
    )

    # The deferred tool dispatched successfully end-to-end.
    rate_returns = [
        part
        for msg in result.all_messages()
        for part in msg.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    # Wire-level checks against the cassette: the deferred corpus ships with
    # `defer_loading: true`, the model's `search_tools` call appears in the response,
    # and our tool result is formatted as `tool_reference` blocks (not plain text).
    from pathlib import Path
    from typing import cast

    cassette_path = (
        Path(__file__).parent / 'cassettes' / 'test_tool_search' / 'test_anthropic_custom_callable_round_trip.yaml'
    )
    cassette = cast(dict[str, Any], yaml.safe_load(cassette_path.read_text(encoding='utf-8')))
    interactions = cast(list[dict[str, Any]], cassette['interactions'])

    first_request_tools = cast(list[dict[str, Any]], interactions[0]['request']['parsed_body']['tools'])
    deferred_names = {t['name'] for t in first_request_tools if t.get('defer_loading') is True}
    assert deferred_names == {'get_exchange_rate', 'stock_lookup'}

    first_response_blocks = cast(list[dict[str, Any]], interactions[0]['response']['parsed_body']['content'])
    assert any(b['type'] == 'tool_use' and b['name'] == 'search_tools' for b in first_response_blocks)

    second_request_messages = cast(list[dict[str, Any]], interactions[1]['request']['parsed_body']['messages'])
    tool_result_blocks: list[dict[str, Any]] = [
        block
        for msg in second_request_messages
        if msg['role'] == 'user' and isinstance(msg.get('content'), list)
        for block in cast(list[dict[str, Any]], msg['content'])
        if isinstance(block, dict) and block.get('type') == 'tool_result'
    ]
    assert tool_result_blocks, 'expected at least one tool_result block in the follow-up turn'
    tool_reference_names: set[str] = {
        cast(str, inner['tool_name'])
        for block in tool_result_blocks
        for inner in cast(list[dict[str, Any]], block.get('content', []))
        if isinstance(inner, dict) and inner.get('type') == 'tool_reference'
    }
    assert tool_reference_names == {'get_exchange_rate'}


@pytest.mark.vcr
async def test_anthropic_native_tool_search_regex_strategy(allow_model_requests: None, anthropic_api_key: str) -> None:
    """`ToolSearch(strategy='regex')` registers the regex variant of Anthropic's
    native tool search tool rather than BM25, and the live API accepts the request.
    """
    pytest.importorskip('anthropic')
    from pathlib import Path

    import yaml

    from pydantic_ai.capabilities import ToolSearch
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent: Agent[None, str] = Agent(model=model, capabilities=[ToolSearch(strategy='regex')])

    @agent.tool_plain(defer_loading=True)
    def get_exchange_rate(from_currency: str, to_currency: str) -> str:  # pragma: no cover
        """Look up the current exchange rate between two currencies."""
        return f'1 {from_currency} = 0.92 {to_currency}'

    await agent.run('hi, just say hello')

    # The live request carries the regex variant — the mock-only assertion here would
    # only validate that we generate the correct parameter shape, not that Anthropic
    # accepts it.
    cassette_path = (
        Path(__file__).parent
        / 'cassettes'
        / 'test_tool_search'
        / 'test_anthropic_native_tool_search_regex_strategy.yaml'
    )
    cassette = cast(dict[str, Any], yaml.safe_load(cassette_path.read_text(encoding='utf-8')))
    interactions = cast(list[dict[str, Any]], cassette['interactions'])
    request_body = cast(dict[str, Any], interactions[0]['request']['parsed_body'])
    tool_types = [
        cast(str, t.get('type')) for t in cast(list[dict[str, Any]], request_body['tools']) if isinstance(t, dict)
    ]
    assert 'tool_search_tool_regex_20251119' in tool_types
    assert 'tool_search_tool_bm25_20251119' not in tool_types
    # Live API returned 2xx — the absence of a 4xx is the strongest signal that the
    # request shape (no beta header, regex variant) is accepted.
    assert interactions[0]['response']['status']['code'] == 200


async def test_anthropic_regex_strategy_replay_preserves_variant(allow_model_requests: None):
    """History replay must re-emit the exact server-tool variant the provider used —
    downgrading ``tool_search_tool_regex`` to ``tool_search_tool_bm25`` on a resend would
    silently run a different algorithm than the earlier turn."""
    pytest.importorskip('anthropic')
    from anthropic.types.beta import BetaServerToolUseBlock, BetaTextBlock, BetaUsage
    from anthropic.types.beta.beta_server_tool_use_block import BetaDirectCaller

    from pydantic_ai.capabilities import ToolSearch
    from pydantic_ai.messages import BuiltinToolCallPart, BuiltinToolReturnPart
    from pydantic_ai.models.anthropic import (
        AnthropicModel,
        _map_server_tool_use_block,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .models.test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

    # Provider-side call used the regex variant; the adapter must round-trip that choice.
    regex_block = BetaServerToolUseBlock(
        id='srv_r',
        name='tool_search_tool_regex',
        input={'query': '.*'},
        type='server_tool_use',
        caller=BetaDirectCaller(type='direct'),
    )
    call_part = _map_server_tool_use_block(regex_block, 'anthropic')
    assert isinstance(call_part, BuiltinToolCallPart)
    assert call_part.provider_details == {'strategy': 'regex'}

    # On replay, the adapter should emit `tool_search_tool_regex` (not bm25).
    response = completion_message(
        [BetaTextBlock(text='done', type='text')],
        BetaUsage(input_tokens=5, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(response)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent: Agent[None, str] = Agent(model=model, capabilities=[ToolSearch(strategy='regex')])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:  # pragma: no cover
        return f'Weather in {city}.'

    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('look it up'),
        ModelResponse(
            parts=[
                call_part,
                BuiltinToolReturnPart(
                    provider_name='anthropic',
                    tool_name='tool_search',
                    tool_call_id='srv_r',
                    content={'tools': [{'name': 'get_weather', 'description': None}]},
                ),
            ],
            provider_name='anthropic',
        ),
    ]
    await agent.run('and again', message_history=history)
    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    # Inspect the replayed Anthropic request. Content blocks are dicts on the request
    # path (params); flatten via comprehension so each replayed call's `name` shows up
    # in `names`.
    blocks = [
        cast('dict[str, Any]', block) for msg in kwargs['messages'] for block in cast('list[Any]', msg['content'])
    ]
    names = [block['name'] for block in blocks if block.get('type') == 'server_tool_use']
    assert 'tool_search_tool_regex' in names
    assert 'tool_search_tool_bm25' not in names


async def test_openai_rejects_anthropic_named_strategy(allow_model_requests: None):
    """OpenAI Responses has no bm25/regex concept — using one must error loudly rather
    than silently falling through to OpenAI's default server-side tool search."""
    pytest.importorskip('openai')
    from pydantic_ai.capabilities import ToolSearch
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAIResponses, response_message

    mock_client = MockOpenAIResponses.create_mock(response_message([]))
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=mock_client))
    agent: Agent[None, str] = Agent(model=model, capabilities=[ToolSearch(strategy='bm25')])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:  # pragma: no cover
        return f'Weather in {city}.'

    with pytest.raises(UserError, match='Anthropic-native'):
        await agent.run('what should I wear?')


async def test_openai_client_tool_search_stamps_envelope_marker():
    """Client-executed tool search calls carry an envelope marker on
    ``provider_details`` so replay doesn't depend on the current request's builtin
    configuration (the user may have reconfigured ``ToolSearch``, or history may be
    handed over from another run)."""
    pytest.importorskip('openai')
    from openai.types.responses import ResponseToolSearchCall

    from pydantic_ai.models.openai import (
        CLIENT_TOOL_SEARCH_ENVELOPE,
        _map_client_tool_search_call,  # pyright: ignore[reportPrivateUsage]
    )

    call = ResponseToolSearchCall(
        id='ts_1',
        arguments={'keywords': 'exchange rate'},
        call_id='call_1',
        execution='client',
        status='completed',
        type='tool_search_call',
    )
    part = _map_client_tool_search_call(call, 'azure')
    assert part.tool_name == _SEARCH_TOOLS_NAME
    # Provider name flows through from the model — important for OpenAI-compatible
    # providers (Azure, gateways) where ``self.system`` differs from ``'openai'``.
    assert part.provider_name == 'azure'
    assert part.provider_details == {'envelope': CLIENT_TOOL_SEARCH_ENVELOPE}


async def test_cross_provider_history_replay_anthropic_to_openai(allow_model_requests: None):
    """A model switch between turns (Anthropic → OpenAI) should replay cleanly: the
    provider-specific Builtin* tool search parts are skipped by the mismatched provider,
    and the agent can still dispatch already-discovered tools by name. This is the
    canonical FallbackModel-style scenario the design calls for."""
    pytest.importorskip('openai')
    pytest.importorskip('anthropic')
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    from pydantic_ai.capabilities import ToolSearch
    from pydantic_ai.messages import BuiltinToolCallPart, BuiltinToolReturnPart
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

    # Prior turn: Anthropic ran a native BM25 search and discovered `get_weather`.
    prior: list[ModelMessage] = [
        ModelRequest.user_text_prompt('weather please'),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    provider_name='anthropic',
                    tool_name='tool_search',
                    tool_call_id='srv_a',
                    args={'query': 'weather'},
                    provider_details={'strategy': 'bm25'},
                ),
                BuiltinToolReturnPart(
                    provider_name='anthropic',
                    tool_name='tool_search',
                    tool_call_id='srv_a',
                    content={'tools': [{'name': 'get_weather', 'description': None}]},
                ),
            ],
            provider_name='anthropic',
        ),
    ]

    # Switch to OpenAI for the follow-up. The Anthropic builtin parts should be silently
    # skipped (`provider_name` mismatch). `get_weather` was discovered in the prior turn,
    # so `ToolSearchToolset._parse_discovered_tools` picks it up and exposes the regular
    # variant on the new provider — the model can call it directly.
    followup = response_message(
        [
            ResponseOutputMessage(
                id='msg_1',
                content=[ResponseOutputText(text='Sunny.', type='output_text', annotations=[])],
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(followup)
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=mock_client))
    agent: Agent[None, str] = Agent(model=model, capabilities=[ToolSearch()])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:  # pragma: no cover
        return f'Weather in {city}.'

    await agent.run('and what about tomorrow?', message_history=prior)
    kwargs = get_mock_responses_kwargs(mock_client)[0]
    # The Anthropic-generated tool search parts are not echoed back to OpenAI (wrong
    # provider) — the replayed input contains only the user message from the prior turn
    # and the new user prompt, plus no `tool_search_call` items.
    item_types = [cast('dict[str, Any]', item).get('type') for item in kwargs['input']]
    assert 'tool_search_call' not in item_types
    # `get_weather` is visible on this turn because it was discovered in the prior turn's
    # history — the local ``ToolSearchToolset`` emits its regular variant in the tool
    # list so the OpenAI request carries `get_weather` as a regular function tool.
    tool_names = [cast('dict[str, Any]', tool).get('name') for tool in kwargs['tools']]
    assert 'get_weather' in tool_names


def test_anthropic_tool_search_result_error_block_mapping():
    """An error result block (no `tool_references`) produces a
    `BuiltinToolReturnPart` without discovered tools in its metadata."""
    pytest.importorskip('anthropic')
    from anthropic.types.beta import BetaToolSearchToolResultBlock
    from anthropic.types.beta.beta_tool_search_tool_result_error import BetaToolSearchToolResultError

    from pydantic_ai.models.anthropic import _map_tool_search_tool_result_block  # pyright: ignore[reportPrivateUsage]

    error_block = BetaToolSearchToolResultBlock(
        tool_use_id='srv_err',
        type='tool_search_tool_result',
        content=BetaToolSearchToolResultError(
            error_code='unavailable',
            error_message='unavailable',
            type='tool_search_tool_result_error',
        ),
    )
    part = _map_tool_search_tool_result_block(error_block, 'anthropic')
    assert part.tool_name == 'tool_search'
    assert part.metadata is None


def test_anthropic_extract_discovered_tool_names_malformed_content():
    """Custom-callable replay must fall through to text formatting when the persisted
    return content doesn't parse as a ``ToolSearchReturn`` — e.g. older history written
    before the typed shape, or a hand-crafted return — rather than crashing or
    fabricating an empty discovery."""
    pytest.importorskip('anthropic')
    from pydantic_ai.messages import ToolReturnPart
    from pydantic_ai.models.anthropic import _extract_discovered_tool_names  # pyright: ignore[reportPrivateUsage]

    malformed = ToolReturnPart(tool_name='search_tools', content='not a typed return', tool_call_id='c1')
    assert _extract_discovered_tool_names(malformed, custom_tool_search_active=True) is None


def test_anthropic_build_tool_search_replay_block_error_branch():
    """Replay reconstruction must round-trip an error result that the parse-time
    mapper stashed on ``provider_details`` back into the ``tool_search_tool_result_error``
    inner block — otherwise a transient provider error on turn N would silently
    flip into a fake successful empty-search on turn N+1's resend."""
    pytest.importorskip('anthropic')
    from pydantic_ai.messages import BuiltinToolReturnPart
    from pydantic_ai.models.anthropic import _build_tool_search_replay_block  # pyright: ignore[reportPrivateUsage]

    return_part = BuiltinToolReturnPart(
        provider_name='anthropic',
        tool_name='tool_search',
        tool_call_id='srv_err',
        content={'tools': []},
        provider_details={'error_code': 'unavailable', 'error_message': 'temporary outage'},
    )
    block = _build_tool_search_replay_block(return_part, 'srv_err')
    assert block == {
        'tool_use_id': 'srv_err',
        'type': 'tool_search_tool_result',
        'content': {
            'type': 'tool_search_tool_result_error',
            'error_code': 'unavailable',
            'error_message': 'temporary outage',
        },
    }


def test_anthropic_build_tool_search_replay_block_missing_error_message():
    """`error_message` is optional on the wire; reconstruction must default to
    an empty string rather than dropping the field (Anthropic rejects partial error
    blocks)."""
    pytest.importorskip('anthropic')
    from pydantic_ai.messages import BuiltinToolReturnPart
    from pydantic_ai.models.anthropic import _build_tool_search_replay_block  # pyright: ignore[reportPrivateUsage]

    return_part = BuiltinToolReturnPart(
        provider_name='anthropic',
        tool_name='tool_search',
        tool_call_id='srv_err',
        content={'tools': []},
        provider_details={'error_code': 'unavailable'},
    )
    block = _build_tool_search_replay_block(return_part, 'srv_err')
    assert block['content'] == {
        'type': 'tool_search_tool_result_error',
        'error_code': 'unavailable',
        'error_message': '',
    }


def test_openai_map_tool_search_call_unit():
    """Unit-level: `_map_tool_search_call` and `_build_tool_search_return_part` produce
    populated metadata for various output shapes — useful as a fast deterministic
    gate without burning a live API call. The end-to-end live cassette in
    `test_openai_native_tool_search_round_trip` exercises the same functions with
    real provider responses."""
    from openai.types.responses import (
        FunctionTool,
        ResponseToolSearchCall,
        ResponseToolSearchOutputItem,
    )

    from pydantic_ai.messages import BuiltinToolCallPart, BuiltinToolReturnPart
    from pydantic_ai.models.openai import (
        _build_tool_search_return_part,  # pyright: ignore[reportPrivateUsage]
        _map_tool_search_call,  # pyright: ignore[reportPrivateUsage]
    )

    call = ResponseToolSearchCall(
        id='ts_1',
        arguments={'query': 'exchange rate'},
        call_id='call_1',
        execution='server',
        status='completed',
        type='tool_search_call',
    )
    output = ResponseToolSearchOutputItem(
        id='tso_1',
        call_id='call_1',
        execution='server',
        status='completed',
        tools=[
            FunctionTool(name='get_exchange_rate', description='', parameters={}, strict=False, type='function'),
        ],
        type='tool_search_output',
    )
    call_part, return_part = _map_tool_search_call(call, output, 'openai')
    assert isinstance(call_part, BuiltinToolCallPart)
    assert call_part.tool_name == 'tool_search'
    assert isinstance(return_part, BuiltinToolReturnPart)
    assert return_part.content == {'tools': [{'name': 'get_exchange_rate', 'description': ''}]}
    assert return_part.provider_details == {'status': 'completed'}

    # No output item → empty discovery (streaming start case).
    empty_return = _build_tool_search_return_part('call_1', 'in_progress', None, 'openai')
    assert empty_return.content == {'tools': []}
    assert empty_return.provider_details == {'status': 'in_progress'}

    # Non-function tools in the output don't have a `name` attribute and are skipped.
    from openai.types.responses.file_search_tool import FileSearchTool

    mixed_output = ResponseToolSearchOutputItem(
        id='tso_mix',
        call_id='call_mix',
        execution='server',
        status='completed',
        tools=[
            FunctionTool(name='real', description='', parameters={}, strict=False, type='function'),
            # FileSearchTool doesn't have a `name` — the loop's `isinstance` guard skips it.
            FileSearchTool(type='file_search', vector_store_ids=['vs_1']),
        ],
        type='tool_search_output',
    )
    mixed = _build_tool_search_return_part('call_mix', 'completed', mixed_output, 'openai')
    assert mixed.content == {'tools': [{'name': 'real', 'description': ''}]}


@pytest.mark.vcr
@pytest.mark.filterwarnings(
    'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
)
async def test_openai_native_tool_search_round_trip(allow_model_requests: None, openai_api_key: str) -> None:
    """End-to-end against live OpenAI Responses: native server-executed `tool_search`
    populates `BuiltinToolCallPart` / `BuiltinToolReturnPart`, the model invokes the
    discovered deferred tool by its plain name, and the second-turn replay carries
    `defer_loading: true` on the corpus function tool plus a `tool_search_call` item.
    """
    from pathlib import Path

    import yaml

    from pydantic_ai.messages import BuiltinToolCallPart, BuiltinToolReturnPart
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
    agent: Agent[None, str] = Agent(model=model)

    @agent.tool_plain(defer_loading=True)
    def get_exchange_rate(from_currency: str, to_currency: str) -> str:
        """Look up the current exchange rate between two currencies."""
        return f'1 {from_currency} = 0.92 {to_currency}'

    @agent.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:  # pragma: no cover
        """Look up stock price by ticker symbol."""
        return f'Stock {symbol}: $150.00'

    result = await agent.run('What is the current USD to EUR exchange rate?')

    assert any(
        isinstance(p, BuiltinToolCallPart) and p.tool_name == 'tool_search'
        for m in result.all_messages()
        for p in m.parts
    )
    assert any(
        isinstance(p, BuiltinToolReturnPart) and p.tool_name == 'tool_search'
        for m in result.all_messages()
        for p in m.parts
    )

    rate_returns = [
        p
        for m in result.all_messages()
        for p in m.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    # Wire-level checks against the live cassette.
    cassette_path = (
        Path(__file__).parent / 'cassettes' / 'test_tool_search' / 'test_openai_native_tool_search_round_trip.yaml'
    )
    cassette = cast(dict[str, Any], yaml.safe_load(cassette_path.read_text(encoding='utf-8')))
    interactions = cast(list[dict[str, Any]], cassette['interactions'])

    # Initial request: deferred tools ship with `defer_loading: true`, and the native
    # `tool_search` builtin is registered alongside.
    first_request = cast(dict[str, Any], interactions[0]['request']['parsed_body'])
    deferred_names = {
        cast(str, t['name'])
        for t in cast(list[dict[str, Any]], first_request['tools'])
        if t.get('defer_loading') is True
    }
    assert deferred_names == {'get_exchange_rate', 'stock_lookup'}
    assert any(t.get('type') == 'tool_search' for t in cast(list[dict[str, Any]], first_request['tools']))
    # Second-turn replay carries the native tool_search_call back; the deferred corpus
    # is preserved with `defer_loading: true`.
    second_request = cast(dict[str, Any], interactions[1]['request']['parsed_body'])
    second_input_types = {
        cast(str, item.get('type'))
        for item in cast(list[dict[str, Any]], second_request['input'])
        if isinstance(item, dict)
    }
    assert 'tool_search_call' in second_input_types
    second_deferred = {
        cast(str, t['name'])
        for t in cast(list[dict[str, Any]], second_request['tools'])
        if t.get('defer_loading') is True
    }
    assert 'get_exchange_rate' in second_deferred


@pytest.mark.vcr
async def test_openai_execution_client_round_trip(allow_model_requests: None, openai_api_key: str) -> None:
    """End-to-end: a custom callable ``ToolSearch`` strategy surfaces natively on OpenAI
    Responses as ``ToolSearchToolParam(execution='client')`` — the provider emits a
    ``tool_search_call`` with ``execution='client'`` whose arguments we dispatch to the
    local ``search_tools`` function, and the resulting ``ToolReturnPart`` is replayed
    as a ``tool_search_output`` (execution='client') carrying the discovered tool defs."""
    from pydantic_ai.capabilities import ToolSearch
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    def match_exchange_rate(query: str, tools: Sequence[ToolDefinition]) -> list[str]:
        # Deterministic: always point the model at `get_exchange_rate` so the cassette
        # replay doesn't depend on the exact keywords the model picks.
        return ['get_exchange_rate']

    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
    agent: Agent[None, str] = Agent(
        model=model,
        instructions=(
            'When you need a capability not provided by your visible tools, call the built-in '
            'tool search first to discover and activate the right one before answering.'
        ),
        capabilities=[ToolSearch(strategy=match_exchange_rate)],
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city} is sunny.'

    @agent.tool_plain(defer_loading=True)
    def get_exchange_rate(from_currency: str, to_currency: str) -> str:
        """Look up the current exchange rate between two currencies."""
        return f'1 {from_currency} = 0.92 {to_currency}'

    @agent.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:  # pragma: no cover
        """Look up stock price by ticker symbol."""
        return f'Stock {symbol}: $150.00'

    result = await agent.run('What is the current exchange rate from USD to EUR?')

    tool_call_names = [
        part.tool_name
        for msg in result.all_messages()
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    # The model called `search_tools` (our local, client-executed search) and then the
    # discovered `get_exchange_rate` — routed through the regular `ToolCallPart` /
    # `ToolReturnPart` path on both sides of the wire.
    assert 'search_tools' in tool_call_names
    assert 'get_exchange_rate' in tool_call_names

    # The local `search_tools` run recorded the discovered tool on `content` as a typed
    # ``ToolSearchReturn`` — this is the same value read back by `ToolSearchToolset` on
    # later turns to unlock the deferred tool on the local path (and round-tripped as
    # `tool_search_output.tools` in the cassette's replay request body).
    search_returns = [
        part
        for msg in result.all_messages()
        for part in msg.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'search_tools'
    ]
    assert len(search_returns) == 1
    assert search_returns[0].content == {
        'tools': [
            {
                'name': 'get_exchange_rate',
                'description': 'Look up the current exchange rate between two currencies.',
            }
        ]
    }

    rate_returns = [
        part
        for msg in result.all_messages()
        for part in msg.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'


async def test_anthropic_native_tool_search_streaming(allow_model_requests: None):
    """Anthropic streaming: `BetaToolSearchToolResultBlock` goes through the part
    manager on `content_block_start` and emits a `BuiltinToolReturnPart`."""
    pytest.importorskip('anthropic')
    from anthropic.types.beta import (
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaToolSearchToolResultBlock,
        BetaUsage,
    )
    from anthropic.types.beta.beta_raw_message_delta_event import Delta
    from anthropic.types.beta.beta_tool_reference_block import BetaToolReferenceBlock
    from anthropic.types.beta.beta_tool_search_tool_search_result_block import (
        BetaToolSearchToolSearchResultBlock,
    )

    from pydantic_ai.messages import BuiltinToolReturnPart
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .models.test_anthropic import MockAnthropic

    result_block = BetaToolSearchToolResultBlock(
        tool_use_id='srv_1',
        type='tool_search_tool_result',
        content=BetaToolSearchToolSearchResultBlock(
            tool_references=[BetaToolReferenceBlock(tool_name='get_exchange_rate', type='tool_reference')],
            type='tool_search_tool_search_result',
        ),
    )
    events = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_1',
                model='claude-sonnet-4-5',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=5, output_tokens=0),
            ),
        ),
        BetaRawContentBlockStartEvent(type='content_block_start', index=0, content_block=result_block),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(input_tokens=5, output_tokens=1),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]
    mock_client = MockAnthropic.create_stream_mock(events)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))

    from pydantic_ai.models import ModelRequestParameters as _ModelRequestParameters

    params = _ModelRequestParameters(function_tools=[], builtin_tools=[], allow_text_output=True)
    async with model.request_stream([], None, params) as streamed:
        async for _ in streamed:
            pass
        response = streamed.get()
    return_parts = [p for p in response.parts if isinstance(p, BuiltinToolReturnPart)]
    assert return_parts and return_parts[0].content == {'tools': [{'name': 'get_exchange_rate', 'description': None}]}


@pytest.mark.filterwarnings(
    'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
)
async def test_openai_native_tool_search_streaming(allow_model_requests: None):
    """OpenAI Responses streaming: `ResponseToolSearchCall` and
    `ResponseToolSearchOutputItem` events produce `BuiltinToolCallPart` and
    `BuiltinToolReturnPart` through the streaming part manager."""
    from openai.types import responses as resp
    from openai.types.responses import (
        FunctionTool,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseToolSearchCall,
        ResponseToolSearchOutputItem,
    )

    from pydantic_ai.messages import BuiltinToolCallPart, BuiltinToolReturnPart
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAIResponses

    base = resp.Response(
        id='123',
        model='gpt-5.4',
        object='response',
        created_at=1704067200,
        output=[],
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
    )
    call = ResponseToolSearchCall(
        id='ts_1',
        arguments={'query': 'rate'},
        call_id='call_1',
        execution='server',
        status='completed',
        type='tool_search_call',
    )
    output = ResponseToolSearchOutputItem(
        id='tso_1',
        call_id='call_1',
        execution='server',
        status='completed',
        tools=[FunctionTool(name='real_tool', description='', parameters={}, strict=False, type='function')],
        type='tool_search_output',
    )
    final = ResponseOutputMessage(
        id='msg_1',
        role='assistant',
        status='completed',
        type='message',
        content=[ResponseOutputText(type='output_text', text='done.', annotations=[])],
    )

    stream: list[resp.ResponseStreamEvent] = [
        resp.ResponseCreatedEvent(response=base, type='response.created', sequence_number=0),
        resp.ResponseInProgressEvent(response=base, type='response.in_progress', sequence_number=1),
        resp.ResponseOutputItemAddedEvent(
            item=call, output_index=0, type='response.output_item.added', sequence_number=2
        ),
        resp.ResponseOutputItemDoneEvent(
            item=call, output_index=0, type='response.output_item.done', sequence_number=3
        ),
        resp.ResponseOutputItemAddedEvent(
            item=output, output_index=1, type='response.output_item.added', sequence_number=4
        ),
        resp.ResponseOutputItemDoneEvent(
            item=output, output_index=1, type='response.output_item.done', sequence_number=5
        ),
        resp.ResponseOutputItemAddedEvent(
            item=final, output_index=2, type='response.output_item.added', sequence_number=6
        ),
        resp.ResponseOutputItemDoneEvent(
            item=final, output_index=2, type='response.output_item.done', sequence_number=7
        ),
        resp.ResponseCompletedEvent(
            response=base.model_copy(update={'status': 'completed'}),
            type='response.completed',
            sequence_number=8,
        ),
    ]

    mock_client = MockOpenAIResponses.create_mock_stream(stream)
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=mock_client))

    # Drive the stream directly via the model's `request_stream` so we exercise the
    # streaming handler without needing the agent graph to accept the output.
    from pydantic_ai.models import ModelRequestParameters as _ModelRequestParameters

    params = _ModelRequestParameters(function_tools=[], builtin_tools=[], allow_text_output=True)
    async with model.request_stream([], None, params) as streamed:
        async for _ in streamed:
            pass
        response = streamed.get()
    parts = response.parts
    assert any(isinstance(p, BuiltinToolCallPart) and p.tool_name == 'tool_search' for p in parts)
    assert any(
        isinstance(p, BuiltinToolReturnPart) and p.content == {'tools': [{'name': 'real_tool', 'description': ''}]}
        for p in parts
    )


async def test_openai_client_tool_search_streaming(allow_model_requests: None):
    """OpenAI Responses streaming for a client-executed `tool_search_call`:
    Added/Done events surface as a regular `ToolCallPart` on ``search_tools`` so the
    standard agent-graph tool-execution path can run the local callable strategy.
    Also covers the per-call ``namespace`` round-trip that the Responses API attaches
    to function calls emitted against discovered deferred tools.
    """
    from openai.types import responses as resp
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseToolSearchCall,
    )

    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAIResponses

    base = resp.Response(
        id='123',
        model='gpt-5.4',
        object='response',
        created_at=1704067200,
        output=[],
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
    )
    client_call = ResponseToolSearchCall(
        id='tsc_1',
        arguments={'keywords': 'exchange rate'},
        call_id='call_client_1',
        execution='client',
        status='completed',
        type='tool_search_call',
    )
    # Discovered deferred tool: OpenAI attaches a per-call `namespace` we preserve on the
    # resulting `ToolCallPart.provider_details` so we can round-trip it on replay.
    namespaced_fn_call = ResponseFunctionToolCall(
        id='fc_1',
        arguments='{"from_currency":"USD","to_currency":"EUR"}',
        call_id='call_fc_1',
        name='get_exchange_rate',
        namespace='get_exchange_rate',
        status='completed',
        type='function_call',
    )
    final = ResponseOutputMessage(
        id='msg_1',
        role='assistant',
        status='completed',
        type='message',
        content=[ResponseOutputText(type='output_text', text='done.', annotations=[])],
    )

    stream: list[resp.ResponseStreamEvent] = [
        resp.ResponseCreatedEvent(response=base, type='response.created', sequence_number=0),
        resp.ResponseInProgressEvent(response=base, type='response.in_progress', sequence_number=1),
        resp.ResponseOutputItemAddedEvent(
            item=client_call, output_index=0, type='response.output_item.added', sequence_number=2
        ),
        resp.ResponseOutputItemDoneEvent(
            item=client_call, output_index=0, type='response.output_item.done', sequence_number=3
        ),
        resp.ResponseOutputItemAddedEvent(
            item=namespaced_fn_call, output_index=1, type='response.output_item.added', sequence_number=4
        ),
        resp.ResponseOutputItemDoneEvent(
            item=namespaced_fn_call, output_index=1, type='response.output_item.done', sequence_number=5
        ),
        resp.ResponseOutputItemAddedEvent(
            item=final, output_index=2, type='response.output_item.added', sequence_number=6
        ),
        resp.ResponseOutputItemDoneEvent(
            item=final, output_index=2, type='response.output_item.done', sequence_number=7
        ),
        resp.ResponseCompletedEvent(
            response=base.model_copy(update={'status': 'completed'}),
            type='response.completed',
            sequence_number=8,
        ),
    ]

    mock_client = MockOpenAIResponses.create_mock_stream(stream)
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=mock_client))

    from pydantic_ai.models import ModelRequestParameters as _ModelRequestParameters

    params = _ModelRequestParameters(function_tools=[], builtin_tools=[], allow_text_output=True)
    async with model.request_stream([], None, params) as streamed:
        async for _ in streamed:
            pass
        response = streamed.get()
    # Route A: the client-executed `tool_search_call` surfaces as a regular
    # `ToolCallPart` against our local `search_tools` function.
    search_calls = [p for p in response.parts if isinstance(p, ToolCallPart) and p.tool_name == 'search_tools']
    assert search_calls and search_calls[0].args_as_dict() == {'keywords': 'exchange rate'}
    # The discovered function call carries its `namespace` in `provider_details` for replay.
    fn_calls = [p for p in response.parts if isinstance(p, ToolCallPart) and p.tool_name == 'get_exchange_rate']
    assert fn_calls and fn_calls[0].provider_details == {'namespace': 'get_exchange_rate'}


async def test_agent_graph_without_builtin_tools(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers `_agent_graph`'s empty `ctx.deps.builtin_tools` branch.

    Auto-inject always adds `ToolSearchTool`, so the only way to exercise the empty
    branch is to disable auto-inject in the test.
    """
    import pydantic_ai.agent as agent_module

    monkeypatch.setattr(agent_module, '_AUTO_INJECT_CAPABILITY_TYPES', ())
    agent: Agent[None, str] = Agent('test')
    result = await agent.run('hi')
    assert isinstance(result.output, str)


async def test_tool_search_toolset_discovers_from_builtin_return_part():
    """Discovery metadata on a `BuiltinToolReturnPart` from a native provider search
    is picked up so the local path recovers state on cross-provider handover."""
    from pydantic_ai.messages import BuiltinToolReturnPart

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                BuiltinToolReturnPart(
                    tool_name='tool_search',
                    content={'tools': [{'name': 'calculate_mortgage', 'description': None}]},
                )
            ]
        )
    ]
    ctx = _build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    assert 'calculate_mortgage' in tools
    assert 'stock_price' not in tools


async def test_tool_search_toolset_custom_search_fn_filters_unknown_names():
    """Names returned by ``search_fn`` that aren't in the deferred set are discarded."""

    def custom_search(query: str, tools: Sequence[ToolDefinition]) -> list[str]:
        return ['stock_price', 'not_a_real_tool', 'crypto_price']

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset, search_fn=custom_search)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'anything'}, ctx, tools[_SEARCH_TOOLS_NAME])
    assert isinstance(result, ToolReturn)
    assert result.return_value == {
        'tools': [
            {'name': 'stock_price', 'description': 'Get the current stock price for a symbol.'},
            {'name': 'crypto_price', 'description': 'Get the current cryptocurrency price.'},
        ]
    }


async def test_tool_search_toolset_custom_search_fn_no_matches():
    """Custom search function returning no names produces the 'no matches' message."""

    def custom_search(query: str, tools: Sequence[ToolDefinition]) -> list[str]:
        return []

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset, search_fn=custom_search)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'anything'}, ctx, tools[_SEARCH_TOOLS_NAME])
    assert isinstance(result, ToolReturn)
    assert result.return_value == {'tools': []}
    assert result.content == 'No matching tools found. The tools you need may not be available.'


async def test_tool_search_capability_strategy_callable_registers_custom_builtin():
    """A callable strategy still registers a ``ToolSearchTool`` builtin with ``custom=True``
    so provider adapters that support a custom-callable native surface (e.g. Anthropic's
    ``tool_reference`` result blocks, OpenAI's ``execution='client'``) can use it; models
    without support drop it as optional and fall back to the local ``search_tools`` tool."""

    def noop(query: str, tools: Sequence[ToolDefinition]) -> list[str]:  # pragma: no cover
        return []

    cap = ToolSearch(strategy=noop)
    builtins = list(cap.get_builtin_tools())
    assert len(builtins) == 1
    tool = builtins[0]
    assert isinstance(tool, ToolSearchTool)
    assert tool.strategy is None
    assert tool.custom is True


async def test_tool_search_capability_strategy_named_registers_builtin():
    """Named native strategies register a non-optional `ToolSearchTool` — the request
    must error on models that can't honor the choice rather than silently substituting
    a local algorithm for bm25/regex."""
    cap = ToolSearch(strategy='regex')
    builtins = list(cap.get_builtin_tools())
    assert len(builtins) == 1
    tool = builtins[0]
    assert isinstance(tool, ToolSearchTool)
    assert tool.strategy == 'regex'
    assert tool.optional is False


async def test_tool_search_capability_strategy_none_optional_builtin():
    """The default (``None``) strategy registers an optional builtin so the local
    token-matching fallback takes over on models without native support."""
    cap = ToolSearch()
    builtins = list(cap.get_builtin_tools())
    assert len(builtins) == 1
    tool = builtins[0]
    assert isinstance(tool, ToolSearchTool)
    assert tool.strategy is None
    assert tool.optional is True


async def test_tool_search_capability_strategy_substring_no_builtin():
    """``strategy='substring'`` is explicitly local — no native builtin is registered,
    the default token-overlap algorithm runs via the local ``search_tools`` function."""
    cap = ToolSearch(strategy='substring')
    builtins = list(cap.get_builtin_tools())
    assert builtins == []


async def test_tool_search_capability_substring_keeps_local_fallback():
    """``strategy='substring'`` still registers the local ``search_tools`` toolset."""
    toolset = _create_function_toolset()
    cap = ToolSearch(strategy='substring')
    wrapped = cap.get_wrapper_toolset(toolset)
    assert isinstance(wrapped, ToolSearchToolset)
    assert wrapped.local_fallback is True


async def test_tool_search_capability_named_strategy_skips_local_fallback():
    """Named native strategies (bm25/regex) must suppress the local ``search_tools``
    tool so ``prepare_request`` raises on unsupported models instead of falling back
    to a different local algorithm."""
    toolset = _create_function_toolset()
    cap = ToolSearch(strategy='bm25')
    wrapped = cap.get_wrapper_toolset(toolset)
    assert isinstance(wrapped, ToolSearchToolset)
    assert wrapped.local_fallback is False


async def test_tool_search_named_strategy_raises_on_unsupported_model():
    """Named native strategies error on models that don't support ``ToolSearchTool``
    — there's no legal fallback for ``strategy='bm25'`` on e.g. GPT-4."""
    from pydantic_ai.models.test import TestModel

    m = TestModel()
    with pytest.raises(UserError, match='not supported by this model'):
        m.prepare_request(
            None,
            ModelRequestParameters(function_tools=[], builtin_tools=[ToolSearchTool(strategy='bm25')]),
        )


async def test_tool_search_substring_ignores_builtin_support():
    """``strategy='substring'`` never tries to use a native builtin — the swap is a
    no-op even on models that support ``ToolSearchTool``."""
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import ToolDefinition

    class ToolSearchTestModel(TestModel):
        @classmethod
        def supported_builtin_tools(cls):
            return frozenset({ToolSearchTool})

    m = ToolSearchTestModel()
    search_tool = ToolDefinition(name=_SEARCH_TOOLS_NAME, description='local', parameters_json_schema={})
    _, prepared = m.prepare_request(
        None,
        ModelRequestParameters(function_tools=[search_tool], builtin_tools=[]),
    )
    assert prepared.builtin_tools == []
    assert [t.name for t in prepared.function_tools] == [_SEARCH_TOOLS_NAME]


def test_managed_by_builtin_swaps_on_support():
    """In `prepare_request`, `managed_by_builtin` tools are kept when the builtin
    is supported and dropped otherwise — mirroring `prefer_builtin` in reverse."""
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import ToolDefinition

    m = TestModel()
    # `optional=True` models the default auto path where the builtin is a best-effort
    # upgrade; on a model that doesn't support it, both the builtin and its corpus drop
    # so the local `ToolSearch` fallback handles discovery.
    search_builtin = ToolSearchTool(optional=True)
    corpus_tool = ToolDefinition(name='deferred_tool', managed_by_builtin='tool_search')

    _, prepared = m.prepare_request(
        None,
        ModelRequestParameters(
            function_tools=[corpus_tool],
            builtin_tools=[search_builtin],
        ),
    )
    assert prepared.builtin_tools == []
    assert prepared.function_tools == []


def test_managed_by_builtin_kept_on_supporting_model():
    """On a supporting model, managed tools are kept so the adapter can emit them
    with provider-specific wire-format tweaks."""
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import ToolDefinition

    class ToolSearchTestModel(TestModel):
        @classmethod
        def supported_builtin_tools(cls):
            return frozenset({ToolSearchTool})

    m = ToolSearchTestModel()
    corpus_tool = ToolDefinition(name='deferred_tool', managed_by_builtin='tool_search')
    _, prepared = m.prepare_request(
        None,
        ModelRequestParameters(
            function_tools=[corpus_tool],
            builtin_tools=[ToolSearchTool()],
        ),
    )
    assert [t.name for t in prepared.function_tools] == ['deferred_tool']
    assert any(isinstance(t, ToolSearchTool) for t in prepared.builtin_tools)


def test_optional_builtin_dropped_with_empty_corpus():
    """An ``optional`` builtin is silently dropped when no managed corpus is in the request."""
    from pydantic_ai.models.test import TestModel

    class ToolSearchTestModel(TestModel):
        @classmethod
        def supported_builtin_tools(cls):
            return frozenset({ToolSearchTool})

    m = ToolSearchTestModel()
    _, prepared = m.prepare_request(
        None,
        ModelRequestParameters(function_tools=[], builtin_tools=[ToolSearchTool(optional=True)]),
    )
    assert prepared.builtin_tools == []
