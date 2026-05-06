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
from pydantic_ai._tool_search_synthetic import synthesize_local_tool_search_messages
from pydantic_ai.builtin_tools.tool_search import ToolSearchTool
from pydantic_ai.capabilities._ordering import collect_leaves
from pydantic_ai.capabilities._tool_search import ToolSearch
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    BuiltinToolSearchCallPart,
    BuiltinToolSearchReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturn,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    UserPromptPart,
)
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
                    parsed = json.loads(part.args) if isinstance(part.args, str) else part.args
                    raw_args = cast('dict[str, Any]', parsed)
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
            'calculate_mortgage',
            'stock_price',
            'crypto_price',
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
                'discovered_tools': [
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
    assert len(rv['discovered_tools']) == 1
    assert rv['discovered_tools'][0]['name'] == 'stock_price'


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
    assert len(rv['discovered_tools']) == 1
    assert rv['discovered_tools'][0]['name'] == 'crypto_price'


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
                'discovered_tools': [
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
                'discovered_tools': [
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
            return_value={'discovered_tools': [{'name': 'github_get_me', 'description': 'Get my GitHub profile.'}]},
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
    assert result.return_value == snapshot(
        {'discovered_tools': [], 'message': 'No matching tools found. The tools you need may not be available.'}
    )


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
    assert len(rv['discovered_tools']) == 10


async def test_tool_search_toolset_discovered_tools_flip_defer_loading():
    """Discovered tools have ``defer_loading=False``; undiscovered ones still have
    ``defer_loading=True``. Both stay in the toolset under their real names — the
    wire-side filter in ``Model.prepare_request`` decides what reaches the model."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage', 'description': None}]},
                ),
            ]
        )
    ]
    ctx = _build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    assert tools['calculate_mortgage'].tool_def.defer_loading is False
    assert tools['stock_price'].tool_def.defer_loading is True
    assert tools['crypto_price'].tool_def.defer_loading is True


async def test_tool_search_toolset_keeps_search_tool_after_all_discovered():
    """``search_tools`` stays in the request even when every deferred tool is discovered.

    Dropping it would invalidate the cached request prefix on the next turn — keeping
    it preserves prompt caching across discovery steps. The local tool's body is a no-op
    branch in `_search_tools` since the index is empty, and on native paths it's dropped
    by the adapter via its `unless_builtin='tool_search'` flag anyway.
    """
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
            'calculate_mortgage',
            'stock_price',
            'crypto_price',
            'search_tools',
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


async def test_tool_manager_with_tool_search_toolset_marks_corpus():
    """Every deferred tool appears once under its real name with
    ``with_builtin='tool_search'``. Visible tools and ``search_tools`` round
    out the dispatch dict. ``Model.prepare_request`` filters per-model to decide what
    actually reaches the wire."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tool_manager = ToolManager[None](searchable)
    run_step_toolset = await tool_manager.for_run_step(ctx)

    managed_names = {t.name for t in run_step_toolset.tool_defs if t.with_builtin == 'tool_search'}
    assert managed_names == {'calculate_mortgage', 'stock_price', 'crypto_price'}

    local_names = [t.name for t in run_step_toolset.tool_defs if not t.with_builtin]
    assert 'get_weather' in local_names
    assert 'search_tools' in local_names

    # Undiscovered deferred tools are still dispatchable through the toolset under their
    # real name — the wire-side filtering in `prepare_request` decides whether the
    # model can see them, but `ToolManager` doesn't gatekeep dispatch on that.
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
    assert result.return_value == snapshot({'discovered_tools': [{'name': 'no_desc_tool', 'description': None}]})


async def test_tool_search_toolset_multiple_searches_accumulate():
    """Discovery accumulates across search turns: tools surfaced in any past
    ``search_tools`` return have `defer_loading=False` on the next step, and
    not-yet-found ones keep `defer_loading=True`."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage', 'description': None}]},
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'stock_price', 'description': None}]},
                ),
            ]
        ),
    ]
    ctx = _build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    assert tools['calculate_mortgage'].tool_def.defer_loading is False
    assert tools['stock_price'].tool_def.defer_loading is False
    assert tools['crypto_price'].tool_def.defer_loading is True


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

    assert tool_names == snapshot(['deferred_tool1', 'deferred_tool2', 'search_tools'])


async def test_tool_search_toolset_ignores_malformed_content_history():
    """Discovery reads `discovered_tools` off the typed return content; malformed
    shapes (non-dict content, missing list, entries missing a string `name`) are
    ignored, and only well-formed entries are picked up."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        # content is not a dict (string fall-through)
        ModelRequest(parts=[ToolSearchReturnPart(content='not a dict')]),
        # content is a dict but `discovered_tools` is missing
        ModelRequest(parts=[ToolSearchReturnPart(content=cast(Any, {'message': 'hi'}))]),
        # `discovered_tools` is not a list
        ModelRequest(parts=[ToolSearchReturnPart(content=cast(Any, {'discovered_tools': 'not a list'}))]),
        # `discovered_tools` contains malformed entries (non-dict, missing/non-string name)
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content=cast(
                        Any,
                        {
                            'discovered_tools': [
                                'not a dict',
                                {'name': 123},
                                {'description': 'no name'},
                            ],
                        },
                    ),
                ),
            ]
        ),
        # valid content
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage', 'description': None}]},
                ),
            ]
        ),
        # Legacy shape (pre-typed-parts): discoveries on `metadata`, content as plain text.
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content='legacy text return',
                    metadata={'discovered_tools': ['stock_price']},
                ),
            ]
        ),
    ]
    ctx = _build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    # Only the well-formed entries flip `defer_loading` off; malformed history is ignored.
    assert tools['calculate_mortgage'].tool_def.defer_loading is False
    assert tools['stock_price'].tool_def.defer_loading is False
    assert tools['crypto_price'].tool_def.defer_loading is True


async def test_tool_search_toolset_reads_legacy_metadata_discovered_tools():
    """Pre-typed-content versions of this toolset wrote discovered tool names to
    ``ToolReturnPart.metadata['discovered_tools']`` instead of the typed
    :class:`ToolSearchReturn` on ``content``. Persisted histories from those versions
    must still surface their discoveries on resume; otherwise an agent reloaded from
    a saved transcript would re-emit ``search_tools`` and the user would see a
    duplicated discovery turn."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content='legacy text return',
                    metadata={'discovered_tools': ['stock_price', 'crypto_price']},
                ),
            ]
        ),
        # Malformed legacy: not a list, ignored.
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content='another',
                    metadata={'discovered_tools': 'not a list'},
                ),
            ]
        ),
        # Malformed legacy: list with non-string entries; the string ones are still picked up.
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content='third',
                    metadata={'discovered_tools': [123, 'calculate_mortgage', None]},
                ),
            ]
        ),
    ]
    ctx = _build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    assert {'stock_price', 'crypto_price', 'calculate_mortgage'} <= set(tools)


async def test_deferred_loading_toolset_marks_all_tools():
    """``DeferredLoadingToolset`` (with `tool_names=None`) flips `defer_loading=True`
    on every tool. After wrapping with `ToolSearchToolset`, all of them appear under
    their real name with `defer_loading=True` (visibility hidden until discovered).
    `search_tools` is the only directly-callable tool up front."""
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
    assert tools['tool_a'].tool_def.defer_loading is True
    assert tools['tool_b'].tool_def.defer_loading is True


async def test_deferred_loading_toolset_marks_specific_tools():
    """``DeferredLoadingToolset`` with explicit names only flips `defer_loading=True`
    on the listed tools; others stay visible."""
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
    assert tools['tool_a'].tool_def.defer_loading is False
    assert tools['tool_b'].tool_def.defer_loading is True


async def test_tool_search_toolset_marks_corpus_with_builtin():
    """Every deferred tool keeps its real name in the toolset output and carries
    ``with_builtin='tool_search'`` regardless of the current model — the adapter's
    ``prepare_request`` decides what reaches the wire so the toolset can't commit early
    (e.g. under ``FallbackModel``)."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)

    managed = {name: tool.tool_def for name, tool in tools.items() if tool.tool_def.with_builtin}
    assert set(managed) == {'calculate_mortgage', 'stock_price', 'crypto_price'}
    for tool_def in managed.values():
        assert tool_def.with_builtin == 'tool_search'
        assert tool_def.defer_loading
    # The local fallback is still present — dropped by the adapter via ``unless_builtin``.
    assert _SEARCH_TOOLS_NAME in tools


async def test_tool_search_toolset_dispatches_by_plain_name_via_tool_manager():
    """The provider calls a deferred tool by its plain name and ``ToolManager``
    dispatches directly via the dict key (also the plain name)."""
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
    """A custom ``search_fn`` replaces the default keyword-matching algorithm."""
    calls: list[str] = []

    def custom_search(ctx: RunContext[None], query: str, tools: Sequence[ToolDefinition]) -> list[str]:
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
        'discovered_tools': [
            {'name': 'stock_price', 'description': 'Get the current stock price for a symbol.'},
            {'name': 'crypto_price', 'description': 'Get the current cryptocurrency price.'},
        ]
    }
    assert calls == ['anything']


async def test_tool_search_toolset_custom_search_fn_still_marks_corpus():
    """A custom ``search_fn`` handles local discovery, but the toolset still flags every
    deferred tool with ``with_builtin='tool_search'`` — when the model supports
    native tool search (including provider-side custom callable modes like Anthropic's
    tool_reference mechanism or OpenAI's ``execution='client'``), the adapter keeps them
    and applies ``defer_loading`` on the wire. Commitment to native-vs-local happens in
    ``Model.prepare_request``, not here."""

    def custom_search(
        ctx: RunContext[None], query: str, tools: Sequence[ToolDefinition]
    ) -> list[str]:  # pragma: no cover
        return []

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset, search_fn=custom_search)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)

    managed = [t.tool_def.name for t in tools.values() if t.tool_def.with_builtin == 'tool_search']
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

    # Native server-side tool search auto-promotes to the typed
    # `BuiltinToolSearchCallPart` / `BuiltinToolSearchReturnPart` subclasses
    # (which still `isinstance`-match the base `BuiltinToolCallPart` /
    # `BuiltinToolReturnPart`).
    builtin_call_parts = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolCallPart)]
    builtin_return_parts = [p for m in result.all_messages() for p in m.parts if isinstance(p, BuiltinToolReturnPart)]
    assert builtin_call_parts and builtin_return_parts

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

    def match_exchange_rate(ctx: RunContext[None], query: str, tools: Sequence[ToolDefinition]) -> list[str]:
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
                    content={'discovered_tools': [{'name': 'get_weather', 'description': None}]},
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


async def test_openai_client_tool_search_maps_to_local_search_call():
    """Client-executed `tool_search_call` items map to a regular `ToolCallPart` against
    the local `search_tools` function. Replay later detects the OpenAI native variant
    via the current request's builtin configuration plus a `provider_name` match."""
    pytest.importorskip('openai')
    from openai.types.responses import ResponseToolSearchCall

    from pydantic_ai.models.openai import (
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
    # No envelope marker any more: replay derives intent from the current request's
    # builtin configuration + a `provider_name` match against `self.system`.
    assert part.provider_details is None


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
                    content={'discovered_tools': [{'name': 'get_weather', 'description': None}]},
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


def test_anthropic_custom_replay_blocks_malformed_content():
    """Custom-callable replay must fall through to text formatting when the persisted
    return content doesn't parse as a `ToolSearchReturnContent` — e.g. older history
    written before the typed shape, or a hand-crafted return — rather than crashing or
    fabricating an empty discovery."""
    pytest.importorskip('anthropic')
    from pydantic_ai.messages import ToolReturnPart
    from pydantic_ai.models.anthropic import (
        _build_custom_tool_search_replay_blocks,  # pyright: ignore[reportPrivateUsage]
    )

    malformed = ToolReturnPart(tool_name='search_tools', content='not a typed return', tool_call_id='c1')
    refs, message = _build_custom_tool_search_replay_blocks(malformed, tool_search_active=True)
    assert refs is None and message is None


def test_anthropic_build_tool_search_replay_block_error_branch():
    """Replay reconstruction must round-trip an error result that the parse-time
    mapper stashed on ``provider_details`` back into the ``tool_search_tool_result_error``
    inner block — otherwise a transient provider error on turn N would silently
    flip into a fake successful empty-search on turn N+1's resend.

    The Anthropic SDK's `BetaToolSearchToolResultErrorParam` carries only `error_code`
    on the wire (no `error_message`), so the message stashed on `provider_details`
    is observability-only — verified separately in
    `test_anthropic_tool_search_result_error_block_mapping`.
    """
    pytest.importorskip('anthropic')
    from pydantic_ai.messages import BuiltinToolSearchReturnPart
    from pydantic_ai.models.anthropic import _build_tool_search_replay_block  # pyright: ignore[reportPrivateUsage]

    return_part = BuiltinToolSearchReturnPart(
        provider_name='anthropic',
        tool_call_id='srv_err',
        content={'discovered_tools': []},
        provider_details={'error_code': 'unavailable', 'error_message': 'temporary outage'},
    )
    block = _build_tool_search_replay_block(return_part, 'srv_err')
    assert block == {
        'tool_use_id': 'srv_err',
        'type': 'tool_search_tool_result',
        'content': {
            'type': 'tool_search_tool_result_error',
            'error_code': 'unavailable',
        },
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

    from pydantic_ai.messages import BuiltinToolSearchCallPart, BuiltinToolSearchReturnPart
    from pydantic_ai.models.openai import (
        _build_tool_search_return_part,  # pyright: ignore[reportPrivateUsage]
        _map_tool_search_call,  # pyright: ignore[reportPrivateUsage]
    )

    call = ResponseToolSearchCall(
        id='ts_1',
        arguments={'paths': ['get_exchange_rate']},
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
    assert isinstance(call_part, BuiltinToolSearchCallPart)
    assert call_part.tool_name == 'tool_search'
    # OpenAI server-executed `tool_search.arguments` carries `paths`; the adapter
    # normalizes that into the cross-provider `queries` slot.
    assert call_part.args == {'queries': ['get_exchange_rate']}
    assert isinstance(return_part, BuiltinToolSearchReturnPart)
    assert return_part.content == {'discovered_tools': [{'name': 'get_exchange_rate', 'description': ''}]}
    assert return_part.provider_details == {'status': 'completed'}

    # No output item → empty discovery (streaming start case).
    empty_return = _build_tool_search_return_part('call_1', 'in_progress', None, 'openai')
    assert empty_return.content == {'discovered_tools': []}
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
    assert mixed.content == {'discovered_tools': [{'name': 'real', 'description': ''}]}


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

    def match_exchange_rate(ctx: RunContext[None], query: str, tools: Sequence[ToolDefinition]) -> list[str]:
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
    # ``ToolSearchReturnContent`` — this is the same value read back by ``ToolSearchToolset``
    # on later turns to unlock the deferred tool on the local path (and round-tripped as
    # `tool_search_output.tools` in the cassette's replay request body).
    search_returns = [
        part
        for msg in result.all_messages()
        for part in msg.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'search_tools'
    ]
    assert len(search_returns) == 1
    assert search_returns[0].content == {
        'discovered_tools': [
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


@pytest.mark.vcr
@pytest.mark.filterwarnings(
    'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
)
async def test_anthropic_native_tool_search_streaming(allow_model_requests: None, anthropic_api_key: str) -> None:
    """End-to-end streaming against live Anthropic: native BM25 server-side tool search
    streams `BuiltinToolSearchCallPart` / `BuiltinToolSearchReturnPart` through the part
    manager during ``agent.iter`` + ``node.stream``, the model invokes the discovered
    deferred tool by its plain name, and the agent loop runs to a final text response."""
    pytest.importorskip('anthropic')

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent: Agent[None, str] = Agent(model=model)

    @agent.tool_plain(defer_loading=True)
    def get_exchange_rate(from_currency: str, to_currency: str) -> str:
        """Look up the current exchange rate between two currencies."""
        return f'1 {from_currency} = 0.92 {to_currency}'

    @agent.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:  # pragma: no cover
        """Look up stock price by ticker symbol."""
        return f'Stock {symbol}: $150.00'

    streamed_events: list[Any] = []
    async with agent.iter(user_prompt='What is the current USD to EUR exchange rate?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        streamed_events.append(event)

    assert agent_run.result is not None

    # The streamed run materializes the same typed builtin parts as the non-streaming
    # round-trip — the part manager promotes them through the discriminator at
    # `content_block_start` time, not just on final response assembly.
    builtin_call_parts = [
        p for m in agent_run.result.all_messages() for p in m.parts if isinstance(p, BuiltinToolSearchCallPart)
    ]
    builtin_return_parts = [
        p for m in agent_run.result.all_messages() for p in m.parts if isinstance(p, BuiltinToolSearchReturnPart)
    ]
    assert builtin_call_parts and builtin_return_parts

    # The discovered deferred tool dispatches by its plain name and produces its
    # ToolReturnPart end-to-end.
    rate_returns = [
        p
        for m in agent_run.result.all_messages()
        for p in m.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    # We received streaming events from both the model-request node and the call-tools
    # node — i.e. the part manager surfaced the builtin tool-search parts as the stream
    # came in (not just on `streamed.get()`).
    assert streamed_events, 'expected streaming events from the request stream'


@pytest.mark.vcr
@pytest.mark.filterwarnings(
    'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
)
@pytest.mark.filterwarnings(
    'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
)
async def test_openai_native_tool_search_streaming(allow_model_requests: None, openai_api_key: str) -> None:
    """End-to-end streaming against live OpenAI Responses: native server-executed
    `tool_search` streams `BuiltinToolSearchCallPart` / `BuiltinToolSearchReturnPart`
    through the part manager during ``agent.iter`` + ``node.stream``, the model invokes
    the discovered deferred tool by its plain name, and the agent loop runs to a final
    text response."""
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

    streamed_events: list[Any] = []
    async with agent.iter(user_prompt='What is the current USD to EUR exchange rate?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        streamed_events.append(event)

    assert agent_run.result is not None

    builtin_call_parts = [
        p for m in agent_run.result.all_messages() for p in m.parts if isinstance(p, BuiltinToolSearchCallPart)
    ]
    builtin_return_parts = [
        p for m in agent_run.result.all_messages() for p in m.parts if isinstance(p, BuiltinToolSearchReturnPart)
    ]
    assert builtin_call_parts and builtin_return_parts

    rate_returns = [
        p
        for m in agent_run.result.all_messages()
        for p in m.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    assert streamed_events, 'expected streaming events from the request stream'


@pytest.mark.vcr
async def test_openai_client_tool_search_streaming(allow_model_requests: None, openai_api_key: str) -> None:
    """End-to-end streaming against live OpenAI Responses with a custom callable
    ``ToolSearch`` strategy. The provider emits a ``tool_search_call`` with
    ``execution='client'`` whose arguments we dispatch to the local ``search_tools``
    function — both events surface through the streaming part manager (the
    ``tool_search_call`` as a regular ``ToolCallPart``), the agent loop runs the
    callable strategy, the model follows up with the discovered deferred tool, and
    the run completes with a final text response."""
    from pydantic_ai.capabilities import ToolSearch
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    def match_exchange_rate(ctx: RunContext[None], query: str, tools: Sequence[ToolDefinition]) -> list[str]:
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

    streamed_events: list[Any] = []
    async with agent.iter(user_prompt='What is the current exchange rate from USD to EUR?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        streamed_events.append(event)

    assert agent_run.result is not None

    tool_call_names = [
        part.tool_name
        for msg in agent_run.result.all_messages()
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    # Client-executed tool search: the `tool_search_call` is routed to the local
    # `search_tools` function, then the model follows up with the discovered tool.
    assert 'search_tools' in tool_call_names
    assert 'get_exchange_rate' in tool_call_names

    search_returns = [
        part
        for msg in agent_run.result.all_messages()
        for part in msg.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'search_tools'
    ]
    assert len(search_returns) == 1
    assert search_returns[0].content == {
        'discovered_tools': [
            {
                'name': 'get_exchange_rate',
                'description': 'Look up the current exchange rate between two currencies.',
            }
        ]
    }

    rate_returns = [
        part
        for msg in agent_run.result.all_messages()
        for part in msg.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    assert streamed_events, 'expected streaming events from the request stream'


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
    """Discovery metadata on a `BuiltinToolSearchReturnPart` from a native provider search
    is picked up so the local path recovers state on cross-provider handover."""
    from pydantic_ai.messages import BuiltinToolSearchReturnPart

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                BuiltinToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage', 'description': None}]},
                )
            ]
        )
    ]
    ctx = _build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    assert tools['calculate_mortgage'].tool_def.defer_loading is False
    assert tools['stock_price'].tool_def.defer_loading is True


async def test_tool_search_toolset_custom_search_fn_filters_unknown_names():
    """Names returned by ``search_fn`` that aren't in the deferred set are discarded."""

    def custom_search(ctx: RunContext[None], query: str, tools: Sequence[ToolDefinition]) -> list[str]:
        return ['stock_price', 'not_a_real_tool', 'crypto_price']

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset, search_fn=custom_search)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'anything'}, ctx, tools[_SEARCH_TOOLS_NAME])
    assert isinstance(result, ToolReturn)
    assert result.return_value == {
        'discovered_tools': [
            {'name': 'stock_price', 'description': 'Get the current stock price for a symbol.'},
            {'name': 'crypto_price', 'description': 'Get the current cryptocurrency price.'},
        ]
    }


async def test_tool_search_toolset_custom_search_fn_no_matches():
    """Custom search function returning no names produces the 'no matches' message."""

    def custom_search(ctx: RunContext[None], query: str, tools: Sequence[ToolDefinition]) -> list[str]:
        return []

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset, search_fn=custom_search)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'anything'}, ctx, tools[_SEARCH_TOOLS_NAME])
    assert isinstance(result, ToolReturn)
    assert result.return_value == {
        'discovered_tools': [],
        'message': 'No matching tools found. The tools you need may not be available.',
    }
    assert result.content == 'No matching tools found. The tools you need may not be available.'


async def test_tool_search_capability_strategy_callable_registers_custom_builtin():
    """A callable strategy still registers a ``ToolSearchTool`` builtin with ``strategy='custom'``
    so provider adapters that support a custom-callable native surface (e.g. Anthropic's
    ``tool_reference`` result blocks, OpenAI's ``execution='client'``) can use it; models
    without support drop it as optional and fall back to the local ``search_tools`` tool."""

    def noop(ctx: RunContext[None], query: str, tools: Sequence[ToolDefinition]) -> list[str]:  # pragma: no cover
        return []

    cap = ToolSearch(strategy=noop)
    builtins = list(cap.get_builtin_tools())
    assert len(builtins) == 1
    tool = builtins[0]
    assert isinstance(tool, ToolSearchTool)
    assert tool.strategy == 'custom'


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


async def test_tool_search_capability_wraps_with_tool_search_toolset():
    """``strategy='keywords'`` wraps with ``ToolSearchToolset`` so the corpus is
    exposed and ``search_tools`` carries the user's customizations. The toolset's
    ``search_fn`` is set to the built-in keyword-overlap algorithm so the local
    dispatch routes through ``_run_search_fn`` (same path as a custom callable),
    enabling client-executed-native wire on supporting providers."""
    toolset = _create_function_toolset()
    cap = ToolSearch(strategy='keywords')
    wrapped = cap.get_wrapper_toolset(toolset)
    assert isinstance(wrapped, ToolSearchToolset)
    assert wrapped.search_fn is not None


async def test_tool_search_capability_named_strategy_wraps_with_tool_search_toolset():
    """Named native strategies (bm25/regex) still wrap with ``ToolSearchToolset`` so
    the corpus is exposed; ``prepare_request`` raises on unsupported models because the
    builtin is registered with ``optional=False``."""
    toolset = _create_function_toolset()
    cap = ToolSearch(strategy='bm25')
    wrapped = cap.get_wrapper_toolset(toolset)
    assert isinstance(wrapped, ToolSearchToolset)
    assert wrapped.search_fn is None


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


async def test_tool_search_keywords_ignores_builtin_support():
    """``strategy='keywords'`` never tries to use a native builtin — the swap is a
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


def test_with_builtin_undiscovered_drops_on_unsupported_model():
    """In `prepare_request`, `with_builtin` corpus members with `defer_loading=True`
    (still undiscovered) drop on a model that doesn't support the builtin — the model has
    no way to call them and the local `search_tools` fallback handles discovery."""
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import ToolDefinition

    m = TestModel()
    # `optional=True` models the default auto path where the builtin is a best-effort
    # upgrade; on a model that doesn't support it, both the builtin and its undiscovered
    # corpus drop so the local `ToolSearch` fallback handles discovery.
    search_builtin = ToolSearchTool(optional=True)
    corpus_tool = ToolDefinition(name='deferred_tool', with_builtin='tool_search', defer_loading=True)

    _, prepared = m.prepare_request(
        None,
        ModelRequestParameters(
            function_tools=[corpus_tool],
            builtin_tools=[search_builtin],
        ),
    )
    assert prepared.builtin_tools == []
    assert prepared.function_tools == []


def test_with_builtin_discovered_kept_on_unsupported_model():
    """A discovered corpus member (`defer_loading=False`) stays in the request even when
    the builtin is unsupported — the model can call it directly by name on the local path."""
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import ToolDefinition

    m = TestModel()
    corpus_tool = ToolDefinition(name='deferred_tool', with_builtin='tool_search', defer_loading=False)

    _, prepared = m.prepare_request(
        None,
        ModelRequestParameters(
            function_tools=[corpus_tool],
            builtin_tools=[ToolSearchTool(optional=True)],
        ),
    )
    assert prepared.builtin_tools == []
    assert [t.name for t in prepared.function_tools] == ['deferred_tool']


def test_with_builtin_kept_on_supporting_model():
    """On a supporting model, managed tools are kept so the adapter can emit them
    with provider-specific wire-format tweaks."""
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import ToolDefinition

    class ToolSearchTestModel(TestModel):
        @classmethod
        def supported_builtin_tools(cls):
            return frozenset({ToolSearchTool})

    m = ToolSearchTestModel()
    corpus_tool = ToolDefinition(name='deferred_tool', with_builtin='tool_search')
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


def test_narrow_type_promotes_builtin_call_to_tool_search() -> None:
    """Direct construction of `BuiltinToolCallPart` with `tool_name='tool_search'`
    promotes to `BuiltinToolSearchCallPart` via the narrowing registry."""
    base = BuiltinToolCallPart(
        tool_name='tool_search',
        args={'queries': ['mortgage']},
        tool_call_id='c1',
        provider_name='anthropic',
        provider_details={'strategy': 'bm25'},
    )
    narrowed = BuiltinToolCallPart.narrow_type(base)
    assert isinstance(narrowed, BuiltinToolSearchCallPart)
    assert narrowed.args == {'queries': ['mortgage']}
    assert narrowed.tool_call_id == 'c1'
    assert narrowed.provider_name == 'anthropic'
    assert narrowed.provider_details == {'strategy': 'bm25'}

    already_narrowed = BuiltinToolSearchCallPart(args={'queries': ['x']}, tool_call_id='c2')
    assert BuiltinToolCallPart.narrow_type(already_narrowed) is already_narrowed


def test_narrow_type_promotes_builtin_return_to_tool_search() -> None:
    """Direct construction of `BuiltinToolReturnPart` with `tool_name='tool_search'`
    promotes to `BuiltinToolSearchReturnPart` via the narrowing registry."""
    base = BuiltinToolReturnPart(
        tool_name='tool_search',
        content={'discovered_tools': [{'name': 'foo', 'description': None}]},
        tool_call_id='c1',
        provider_name='anthropic',
    )
    narrowed = BuiltinToolReturnPart.narrow_type(base)
    assert isinstance(narrowed, BuiltinToolSearchReturnPart)
    assert narrowed.content == {'discovered_tools': [{'name': 'foo', 'description': None}]}

    already_narrowed = BuiltinToolSearchReturnPart(content={'discovered_tools': []}, tool_call_id='c2')
    assert BuiltinToolReturnPart.narrow_type(already_narrowed) is already_narrowed


def test_narrow_type_unknown_tool_name_returns_input_unchanged() -> None:
    """Unknown `tool_name` values aren't promoted (future builtins not yet typed)."""
    base = BuiltinToolCallPart(tool_name='something_unregistered', args={}, tool_call_id='c1')
    assert BuiltinToolCallPart.narrow_type(base) is base


def test_model_response_dict_round_trip_promotes_typed_subclasses() -> None:
    """Pydantic deserialization of a dict-shaped `ModelResponse` promotes
    `tool_search` builtin parts to typed subclasses via the discriminator."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    raw: dict[str, Any] = {
        'kind': 'response',
        'parts': [
            {
                'part_kind': 'builtin-tool-call',
                'tool_name': 'tool_search',
                'args': {'queries': ['mortgage']},
                'tool_call_id': 'c1',
                'provider_name': 'anthropic',
            },
            {
                'part_kind': 'builtin-tool-return',
                'tool_name': 'tool_search',
                'content': {'discovered_tools': [{'name': 'foo', 'description': None}]},
                'tool_call_id': 'c1',
                'provider_name': 'anthropic',
            },
            {
                'part_kind': 'builtin-tool-call',
                'tool_name': 'web_search',
                'args': {'query': 'x'},
                'tool_call_id': 'c2',
            },
        ],
    }
    [resp] = ModelMessagesTypeAdapter.validate_python([raw])
    assert isinstance(resp, ModelResponse)
    assert isinstance(resp.parts[0], BuiltinToolSearchCallPart)
    assert isinstance(resp.parts[1], BuiltinToolSearchReturnPart)
    # Unrecognized `tool_name` falls through to the base class.
    assert isinstance(resp.parts[2], BuiltinToolCallPart)
    assert not isinstance(resp.parts[2], BuiltinToolSearchCallPart)


def test_model_response_instance_round_trip_promotes_typed_subclasses() -> None:
    """Re-validation of a `ModelResponse` instance preserves typed builtin parts."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    resp = ModelResponse(
        parts=[
            BuiltinToolSearchCallPart(args={'queries': ['x']}, tool_call_id='c1'),
            BuiltinToolSearchReturnPart(
                content={'discovered_tools': [{'name': 'foo', 'description': None}]},
                tool_call_id='c1',
            ),
            BuiltinToolCallPart(tool_name='web_search', args={}, tool_call_id='c2'),
        ]
    )
    [revalidated] = ModelMessagesTypeAdapter.validate_python([resp])
    assert isinstance(revalidated, ModelResponse)
    assert isinstance(revalidated.parts[0], BuiltinToolSearchCallPart)
    assert isinstance(revalidated.parts[1], BuiltinToolSearchReturnPart)
    assert isinstance(revalidated.parts[2], BuiltinToolCallPart)


async def test_tool_search_toolset_promotes_base_builtin_return_part() -> None:
    """A `BuiltinToolReturnPart` (base class) carrying `tool_name='tool_search'`
    is promoted by the toolset's parser via `narrow_type` so its discoveries
    surface — covers the fall-through promotion branch."""
    base_toolset = FunctionToolset[None]()

    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                BuiltinToolReturnPart(
                    tool_name='tool_search',
                    content={'discovered_tools': [{'name': 'calculate_mortgage', 'description': None}]},
                    tool_call_id='c1',
                ),
            ],
        ),
    ]
    ts = ToolSearchToolset(wrapped=base_toolset)
    discovered = ts._parse_discovered_tools(  # pyright: ignore[reportPrivateUsage]
        cast(Any, type('_Ctx', (), {'messages': history})()),
    )
    assert 'calculate_mortgage' in discovered


async def test_tool_search_toolset_replays_main_branch_legacy_shape() -> None:
    """Histories serialized on `main` (before this PR's typed-content shape) carry the
    discovered names on `ToolReturnPart.metadata['discovered_tools']` rather than on a
    typed `content`. They must continue to replay cleanly on the typed-parts shape so
    upgrading users don't lose discovered-tool state on the next turn.

    This is the wire shape on the `main` branch as of the merge-base.
    """
    base_toolset = FunctionToolset[None]()

    history: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart(tool_name='search_tools', args={'keywords': 'mortgage'}, tool_call_id='c1')]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='search_tools',
                    # `main`-branch shape: structured discoveries live on `metadata`,
                    # `content` is just the user-visible string the model sees.
                    content='Found 1 tool: calculate_mortgage',
                    tool_call_id='c1',
                    metadata={'discovered_tools': ['calculate_mortgage']},
                ),
            ],
        ),
    ]
    ts = ToolSearchToolset(wrapped=base_toolset)
    discovered = ts._parse_discovered_tools(  # pyright: ignore[reportPrivateUsage]
        cast(Any, type('_Ctx', (), {'messages': history})()),
    )
    assert discovered == {'calculate_mortgage'}


def test_synthetic_injection_translates_builtin_to_local_tool_search_parts() -> None:
    """Cross-provider replay end-to-end: a `BuiltinToolSearch*Part` carried over from
    a prior native turn is translated into the local-shape typed parts so a non-native
    adapter can replay it as a normal `search_tools` function-call exchange. The
    toolset's `_parse_discovered_tools` then surfaces the discoveries via the
    discriminated-union dispatch."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Find me a mortgage tool.')]),
        ModelResponse(
            parts=[
                BuiltinToolSearchCallPart(
                    args={'queries': ['mortgage']},
                    tool_call_id='c1',
                    provider_name='anthropic',
                    provider_details={'strategy': 'bm25'},
                ),
                BuiltinToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage', 'description': None}]},
                    tool_call_id='c1',
                    provider_name='anthropic',
                ),
            ],
        ),
    ]

    translated = synthesize_local_tool_search_messages(history)

    # The user prompt request passes through unchanged.
    assert translated[0] is history[0]

    # The response now carries a local `ToolSearchCallPart` (typed `ToolCallPart` subclass),
    # and the return part has been lifted onto a fresh trailing `ModelRequest`.
    response = translated[1]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 1
    call_part = response.parts[0]
    assert isinstance(call_part, ToolSearchCallPart)
    # Subclass of `ToolCallPart`, NOT `BuiltinToolSearchCallPart`.
    assert isinstance(call_part, ToolCallPart)
    assert not isinstance(call_part, BuiltinToolSearchCallPart)
    assert call_part.tool_name == 'search_tools'
    assert call_part.args == {'queries': ['mortgage']}

    return_request = translated[2]
    assert isinstance(return_request, ModelRequest)
    return_part = return_request.parts[0]
    assert isinstance(return_part, ToolSearchReturnPart)
    assert isinstance(return_part, ToolReturnPart)
    assert not isinstance(return_part, BuiltinToolSearchReturnPart)
    assert return_part.tool_name == 'search_tools'
    assert return_part.content == {'discovered_tools': [{'name': 'calculate_mortgage', 'description': None}]}

    # And the toolset's parser surfaces the discovery off the translated history.
    base_toolset = FunctionToolset[None]()
    ts = ToolSearchToolset(wrapped=base_toolset)
    discovered = ts._parse_discovered_tools(  # pyright: ignore[reportPrivateUsage]
        cast(Any, type('_Ctx', (), {'messages': translated})()),
    )
    assert discovered == {'calculate_mortgage'}


def test_prepare_messages_translates_on_non_native_model() -> None:
    """`Model.prepare_messages` is the centralized hook that runs before the adapter's
    message-prep on every request. On a model whose profile doesn't include
    ``ToolSearchTool`` in ``supported_builtin_tools``, the hook translates any prior
    server-side tool-search exchange into the local-shape typed parts so the adapter
    sees a normal ``search_tools`` function-call exchange.

    The single ``ModelResponse(call+return)`` carrying the inline server-side result
    splits into ``ModelResponse(call) + ModelRequest(return)``."""
    # Default `TestModel` excludes `ToolSearchTool` from `supported_builtin_tools`.
    model = TestModel()
    assert ToolSearchTool not in model.profile.supported_builtin_tools

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Find me a mortgage tool.')]),
        ModelResponse(
            parts=[
                BuiltinToolSearchCallPart(
                    args={'queries': ['mortgage']},
                    tool_call_id='c1',
                    provider_name='anthropic',
                ),
                BuiltinToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage', 'description': None}]},
                    tool_call_id='c1',
                    provider_name='anthropic',
                ),
            ],
        ),
    ]

    prepared = model.prepare_messages(history)

    # Original 2 messages became 3: user prompt, response with local call,
    # request carrying the lifted return.
    assert len(prepared) == 3
    assert prepared[0] is history[0]

    response = prepared[1]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 1
    call_part = response.parts[0]
    assert isinstance(call_part, ToolSearchCallPart)
    assert not isinstance(call_part, BuiltinToolSearchCallPart)
    assert call_part.tool_name == 'search_tools'

    return_request = prepared[2]
    assert isinstance(return_request, ModelRequest)
    [return_part] = return_request.parts
    assert isinstance(return_part, ToolSearchReturnPart)
    assert not isinstance(return_part, BuiltinToolSearchReturnPart)
    assert return_part.tool_name == 'search_tools'
    assert return_part.content == {'discovered_tools': [{'name': 'calculate_mortgage', 'description': None}]}


def test_prepare_messages_passes_through_on_native_model() -> None:
    """A model whose profile *does* include ``ToolSearchTool`` in
    ``supported_builtin_tools`` keeps the prior exchange as-is — the native adapter
    knows how to ship the typed builtin parts back on the wire."""

    class NativeToolSearchTestModel(TestModel):
        @classmethod
        def supported_builtin_tools(cls):
            return frozenset({ToolSearchTool})

    model = NativeToolSearchTestModel()
    assert ToolSearchTool in model.profile.supported_builtin_tools

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Find me a mortgage tool.')]),
        ModelResponse(
            parts=[
                BuiltinToolSearchCallPart(
                    args={'queries': ['mortgage']},
                    tool_call_id='c1',
                    provider_name='anthropic',
                ),
                BuiltinToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage', 'description': None}]},
                    tool_call_id='c1',
                    provider_name='anthropic',
                ),
            ],
        ),
    ]

    prepared = model.prepare_messages(history)

    assert prepared is history


def test_narrow_type_local_promotes_with_well_shaped_args() -> None:
    """A `ToolCallPart` with `tool_name='search_tools'` and matching args promotes
    to `ToolSearchCallPart` via the narrowing registry."""
    from pydantic_ai.messages import ToolCallPart

    part = ToolCallPart(tool_name='search_tools', args={'queries': ['mortgage']}, tool_call_id='c1')
    narrowed = ToolCallPart.narrow_type(part)
    assert isinstance(narrowed, ToolSearchCallPart)
    assert narrowed.args == {'queries': ['mortgage']}


def test_narrow_type_local_passthrough_when_already_narrowed() -> None:
    """Narrowing an already-typed `ToolSearchCallPart` returns the input instance."""
    part = ToolSearchCallPart(args={'queries': ['x']}, tool_call_id='c1')
    from pydantic_ai.messages import ToolCallPart

    assert ToolCallPart.narrow_type(part) is part


def test_pydantic_validation_rejects_search_tools_collision_with_bad_args() -> None:
    """A user-defined function tool that *happens* to be named `search_tools` but
    carries the wrong args shape must fail Pydantic validation when deserialized
    through `ModelRequestPart`. Auto-promotion via the typed-class discriminator
    requires the args to match `ToolSearchArgs`."""
    import pydantic

    from pydantic_ai.messages import ModelMessagesTypeAdapter

    raw = [
        {
            'kind': 'request',
            'parts': [
                {
                    'part_kind': 'tool-return',
                    'tool_name': 'search_tools',
                    # Wrong shape: not `ToolSearchReturnContent`, not `str`, not `None`.
                    'content': {'unrelated': 'data', 'definitely_not_discovered_tools': 42},
                    'tool_call_id': 'c1',
                },
            ],
        },
    ]
    with pytest.raises(pydantic.ValidationError):
        ModelMessagesTypeAdapter.validate_python(raw)


def test_pydantic_validation_accepts_search_tools_string_content_collision() -> None:
    """The `str` variant on `ToolSearchReturnPart.content` covers the case where a
    user-named tool returns plain text — they're collision-tolerant by design.

    Adapters that consume the typed view check `isinstance(content, dict)` before
    keying into `discovered_tools`, so a string content surfaces as "not a real
    tool-search return" and falls through cleanly."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    raw = [
        {
            'kind': 'request',
            'parts': [
                {
                    'part_kind': 'tool-return',
                    'tool_name': 'search_tools',
                    'content': 'hello world',
                    'tool_call_id': 'c1',
                },
            ],
        },
    ]
    [request] = ModelMessagesTypeAdapter.validate_python(raw)
    [part] = request.parts
    assert isinstance(part, ToolSearchReturnPart)
    assert part.content == 'hello world'


def test_synthesize_local_from_builtin_call_str_args_passthrough() -> None:
    """Streaming partial-args (`str`) are passed through unchanged when translating."""
    from pydantic_ai._tool_search_synthetic import synthesize_local_from_builtin_call
    from pydantic_ai.messages import BuiltinToolSearchCallPart

    part = BuiltinToolSearchCallPart(args='{"queries":', tool_call_id='c1')
    result = synthesize_local_from_builtin_call(part)
    assert result.args == '{"queries":'
    assert result.tool_call_id == 'c1'


def test_synthesize_local_from_builtin_call_none_args_falls_through() -> None:
    """`None` args remain `None` after translation."""
    from pydantic_ai._tool_search_synthetic import synthesize_local_from_builtin_call
    from pydantic_ai.messages import BuiltinToolSearchCallPart

    part = BuiltinToolSearchCallPart(args=None, tool_call_id='c1')
    result = synthesize_local_from_builtin_call(part)
    assert result.args is None


def test_synthesize_local_from_builtin_return_none_content_defaults() -> None:
    """A return part with `content=None` (legal per the typed signature) gets a
    sensible empty-discoveries default. Covers the `else` branch in
    `synthesize_local_from_builtin_return`."""
    from pydantic_ai._tool_search_synthetic import synthesize_local_from_builtin_return
    from pydantic_ai.messages import BuiltinToolSearchReturnPart

    part = BuiltinToolSearchReturnPart(content=None, tool_call_id='c1')
    result = synthesize_local_from_builtin_return(part)
    assert result.content == {'discovered_tools': []}


def test_synthesize_messages_response_with_only_call_part_no_lift() -> None:
    """A response with only a `BuiltinToolSearchCallPart` (no return — streaming case)
    translates the call but doesn't synthesize a trailing `ModelRequest`."""
    from pydantic_ai._tool_search_synthetic import synthesize_local_tool_search_messages
    from pydantic_ai.messages import BuiltinToolSearchCallPart

    history: list[ModelMessage] = [
        ModelResponse(parts=[BuiltinToolSearchCallPart(args={'queries': ['x']}, tool_call_id='c1')]),
    ]
    result = synthesize_local_tool_search_messages(history)
    assert len(result) == 1
    response = result[0]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], ToolSearchCallPart)


def test_synthesize_messages_response_with_only_return_part_no_response_kept() -> None:
    """A response with only a `BuiltinToolSearchReturnPart` (no remaining parts) — the
    response is dropped since it'd be empty, and the return is lifted onto a fresh request."""
    from pydantic_ai._tool_search_synthetic import synthesize_local_tool_search_messages
    from pydantic_ai.messages import BuiltinToolSearchReturnPart

    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                BuiltinToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'foo', 'description': None}]},
                    tool_call_id='c1',
                ),
            ],
        ),
    ]
    result = synthesize_local_tool_search_messages(history)
    assert len(result) == 1
    request = result[0]
    assert isinstance(request, ModelRequest)
    assert len(request.parts) == 1
    return_part = request.parts[0]
    assert isinstance(return_part, ToolSearchReturnPart)


def test_synthesize_messages_request_with_unrelated_tool_return_passthrough() -> None:
    """A `ToolReturnPart` with `tool_name != 'search_tools'` doesn't get promoted —
    the request is returned unchanged."""
    from pydantic_ai._tool_search_synthetic import synthesize_local_tool_search_messages

    request = ModelRequest(parts=[ToolReturnPart(tool_name='get_weather', content='sunny', tool_call_id='c1')])
    result = synthesize_local_tool_search_messages([request])
    assert len(result) == 1
    assert result[0] is request


def test_synthesize_messages_response_with_search_then_downstream_tool_call_splits_4_messages() -> None:
    """Native turn with `[Text, BuiltinSearchCall, BuiltinSearchReturn, ToolCall(weather)]`
    must split into a coherent local-shape sequence: response[Text, ToolSearchCall],
    request[ToolSearchReturn], response[ToolCall(weather)], (passthrough) request[ToolReturn(weather)].

    Currently we keep the downstream `ToolCall(weather)` on the same response as the
    `ToolSearchCall`, which is incoherent (model "called weather before seeing search
    results") and produces consecutive `ModelRequest`s after the lifted return —
    Devin's observation. Splitting at every `BuiltinToolSearchReturn` boundary fixes
    both: the timeline reads correctly and the lifted return doesn't collide with the
    next request.
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Look up something then call it.')]),
        ModelResponse(
            parts=[
                TextPart(content='Searching first.'),
                BuiltinToolSearchCallPart(
                    args={'queries': ['weather']},
                    tool_call_id='search1',
                    provider_name='anthropic',
                ),
                BuiltinToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_weather', 'description': None}]},
                    tool_call_id='search1',
                    provider_name='anthropic',
                ),
                ToolCallPart(tool_name='get_weather', args={'city': 'NYC'}, tool_call_id='wx1'),
            ],
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name='get_weather', content='sunny', tool_call_id='wx1')]),
    ]

    result = synthesize_local_tool_search_messages(history)

    # 5 messages: user request, response[Text, ToolSearchCall], request[ToolSearchReturn],
    # response[ToolCall(weather)], request[ToolReturn(weather)] (the original).
    assert len(result) == 5

    assert isinstance(result[0], ModelRequest)
    assert result[0] is history[0]

    # First synthetic response: text + search call only — NOT the downstream weather call.
    first_resp = result[1]
    assert isinstance(first_resp, ModelResponse)
    assert len(first_resp.parts) == 2
    assert isinstance(first_resp.parts[0], TextPart)
    assert isinstance(first_resp.parts[1], ToolSearchCallPart)
    # No `ToolCallPart(weather)` snuck onto this response.
    assert not any(isinstance(p, ToolCallPart) and not isinstance(p, ToolSearchCallPart) for p in first_resp.parts)

    # Lifted search return as a fresh request.
    search_return_req = result[2]
    assert isinstance(search_return_req, ModelRequest)
    assert len(search_return_req.parts) == 1
    assert isinstance(search_return_req.parts[0], ToolSearchReturnPart)

    # Second synthetic response: weather call only.
    second_resp = result[3]
    assert isinstance(second_resp, ModelResponse)
    assert len(second_resp.parts) == 1
    weather_call = second_resp.parts[0]
    assert isinstance(weather_call, ToolCallPart)
    assert weather_call.tool_name == 'get_weather'

    # Original weather-return request flows naturally — no consecutive `ModelRequest`s.
    assert isinstance(result[4], ModelRequest)
    assert result[4] is history[2]


def test_synthesize_messages_devins_consecutive_request_repro() -> None:
    """Regression: synthesis must not produce two consecutive `ModelRequest`s.

    Reproduces Devin's bug report exactly: native search exchange immediately followed
    by a regular tool call within the same `ModelResponse`, then a `ModelRequest` for
    the regular tool's return. The proper splitter inserts a synthetic `ModelResponse`
    between the lifted search return and the original tool-return request, so message
    roles alternate correctly for adapters with strict user/assistant alternation.
    """
    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                BuiltinToolSearchCallPart(args={'queries': ['x']}, tool_call_id='s1'),
                BuiltinToolSearchReturnPart(content={'discovered_tools': []}, tool_call_id='s1'),
                ToolCallPart(tool_name='get_weather', args={}, tool_call_id='w1'),
            ],
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name='get_weather', content='ok', tool_call_id='w1')]),
    ]

    result = synthesize_local_tool_search_messages(history)

    # Walk and verify no two consecutive entries are both `ModelRequest`.
    for i in range(len(result) - 1):
        if isinstance(result[i], ModelRequest):
            assert not isinstance(result[i + 1], ModelRequest), f'Consecutive ModelRequests at index {i}: {result}'


def test_synthesize_messages_multiple_search_rounds_in_one_response() -> None:
    """Two server-side search rounds inside a single native `ModelResponse` split into
    two response/request pairs, preserving order and not bundling them onto one response.
    """
    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                BuiltinToolSearchCallPart(args={'queries': ['a']}, tool_call_id='s1'),
                BuiltinToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'tool_a', 'description': None}]},
                    tool_call_id='s1',
                ),
                BuiltinToolSearchCallPart(args={'queries': ['b']}, tool_call_id='s2'),
                BuiltinToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'tool_b', 'description': None}]},
                    tool_call_id='s2',
                ),
            ],
        ),
    ]

    result = synthesize_local_tool_search_messages(history)

    # 4 messages: response[call_a], request[return_a], response[call_b], request[return_b].
    assert len(result) == 4
    assert isinstance(result[0], ModelResponse)
    assert isinstance(result[0].parts[0], ToolSearchCallPart)
    assert result[0].parts[0].tool_call_id == 's1'

    assert isinstance(result[1], ModelRequest)
    assert isinstance(result[1].parts[0], ToolSearchReturnPart)
    assert result[1].parts[0].tool_call_id == 's1'

    assert isinstance(result[2], ModelResponse)
    assert isinstance(result[2].parts[0], ToolSearchCallPart)
    assert result[2].parts[0].tool_call_id == 's2'

    assert isinstance(result[3], ModelRequest)
    assert isinstance(result[3].parts[0], ToolSearchReturnPart)
    assert result[3].parts[0].tool_call_id == 's2'


def test_synthesize_messages_metadata_kept_on_first_split_only() -> None:
    """Splitting one native `ModelResponse` into multiple responses must not duplicate
    its identity-level metadata (`provider_response_id`, usage). The first split keeps
    the original identity; subsequent splits get fresh/blank fields so downstream
    consumers don't double-count usage or find two responses for the same API call.
    """
    from pydantic_ai.usage import RequestUsage

    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                BuiltinToolSearchCallPart(args={'queries': ['x']}, tool_call_id='s1'),
                BuiltinToolSearchReturnPart(content={'discovered_tools': []}, tool_call_id='s1'),
                ToolCallPart(tool_name='get_weather', args={}, tool_call_id='w1'),
            ],
            usage=RequestUsage(input_tokens=100, output_tokens=50),
            provider_response_id='msg_real_anthropic_id',
            provider_name='anthropic',
            model_name='claude-sonnet-4-5',
        ),
    ]

    result = synthesize_local_tool_search_messages(history)

    # Two responses out (split around the search return).
    responses = [m for m in result if isinstance(m, ModelResponse)]
    assert len(responses) == 2

    # First split keeps full metadata.
    assert responses[0].provider_response_id == 'msg_real_anthropic_id'
    assert responses[0].usage.input_tokens == 100
    assert responses[0].usage.output_tokens == 50

    # Second split gets cleared identity to avoid double-counting / duplicate lookup.
    assert responses[1].provider_response_id is None
    assert responses[1].usage.input_tokens == 0
    assert responses[1].usage.output_tokens == 0


def test_narrow_type_local_return_passthrough_when_already_narrowed() -> None:
    """Narrowing an already-typed `ToolSearchReturnPart` returns the input instance."""
    part = ToolSearchReturnPart(content={'discovered_tools': []}, tool_call_id='c1')
    assert ToolReturnPart.narrow_type(part) is part


def test_model_request_part_discriminator_recognizes_tool_search_return_instance() -> None:
    """The request-part discriminator returns the typed tag when called with a
    `ToolSearchReturnPart` instance.

    Pydantic's discriminated-union fast path bypasses the discriminator when the input
    already matches one of the tagged variants by isinstance, so this exercises the
    function directly rather than via `ModelMessagesTypeAdapter`.
    """
    from pydantic_ai.messages import _model_request_part_discriminator  # pyright: ignore[reportPrivateUsage]

    part = ToolSearchReturnPart(content={'discovered_tools': []}, tool_call_id='c1')
    assert _model_request_part_discriminator(part) == 'tool-search-return'


def test_model_response_part_discriminator_recognizes_local_call_dict_dispatch() -> None:
    """A dict-shaped `ToolCallPart` with `tool_name='search_tools'` gets dispatched to
    `ToolSearchCallPart` via the discriminator (covers the `'tool-call'` branch)."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    raw = [
        {
            'kind': 'response',
            'parts': [
                {
                    'part_kind': 'tool-call',
                    'tool_name': 'search_tools',
                    'args': {'queries': ['x']},
                    'tool_call_id': 'c1',
                },
            ],
        },
    ]
    [resp] = ModelMessagesTypeAdapter.validate_python(raw)
    assert isinstance(resp, ModelResponse)
    [part] = resp.parts
    assert isinstance(part, ToolSearchCallPart)


def test_model_response_part_discriminator_passthrough_for_unknown_part_kind() -> None:
    """Instance dispatch falls through to `getattr(v, 'part_kind', ...)` for other types."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter, TextPart

    resp = ModelResponse(parts=[TextPart(content='hello')])
    [revalidated] = ModelMessagesTypeAdapter.validate_python([resp])
    assert isinstance(revalidated, ModelResponse)
    [part] = revalidated.parts
    assert isinstance(part, TextPart)


def test_model_response_part_discriminator_recognizes_typed_instances() -> None:
    """The response-part discriminator returns the typed tag for each typed-instance branch.

    Pydantic's discriminated-union fast path bypasses the discriminator when the input
    already matches one of the tagged variants by isinstance, so the instance branches
    in `_model_response_part_discriminator` are only reachable by calling the function
    directly. This locks in the contract for any future caller (or pydantic version
    that changes its short-circuit behavior).
    """
    from pydantic_ai.messages import _model_response_part_discriminator  # pyright: ignore[reportPrivateUsage]

    builtin_call = BuiltinToolSearchCallPart(args={'queries': ['x']}, tool_call_id='c1', provider_name='anthropic')
    assert _model_response_part_discriminator(builtin_call) == 'builtin-tool-search-call'

    builtin_return = BuiltinToolSearchReturnPart(
        content={'discovered_tools': []},
        tool_call_id='c1',
        provider_name='anthropic',
    )
    assert _model_response_part_discriminator(builtin_return) == 'builtin-tool-search-return'

    local_call = ToolSearchCallPart(args={'queries': ['x']}, tool_call_id='c1')
    assert _model_response_part_discriminator(local_call) == 'tool-search-call'


async def test_tool_search_toolset_async_search_fn_is_awaited() -> None:
    """Custom search functions can be `async`; the toolset awaits them."""

    async def async_match(_ctx: RunContext[None], _query: str, tools: Sequence[ToolDefinition]) -> Sequence[str]:
        return [t.name for t in tools]

    ts = ToolSearchToolset(wrapped=_create_function_toolset(), search_fn=async_match)
    ctx = _build_run_context(None)
    tools = await ts.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]
    result = await ts.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'weather'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    return_value = cast(dict[str, Any], result.return_value)
    discovered_names = {match['name'] for match in return_value['discovered_tools']}
    # `_create_function_toolset` registers a fixed set of deferred tools; verify the
    # async search function received the corpus and returned discoverable names.
    assert 'calculate_mortgage' in discovered_names


def test_anthropic_custom_replay_blocks_returns_message_on_empty_discovered() -> None:
    """When the typed return carries empty `discovered_tools` and a `message`, the
    helper returns `([], message)`. The `_map_message` flow then renders the message
    as a single text block (Anthropic rejects empty `tool_result.content`)."""
    pytest.importorskip('anthropic')
    from pydantic_ai.messages import ToolReturnPart
    from pydantic_ai.models.anthropic import (
        _build_custom_tool_search_replay_blocks,  # pyright: ignore[reportPrivateUsage]
    )

    empty = ToolReturnPart(
        tool_name='search_tools',
        content={'discovered_tools': [], 'message': 'No matches; try other keywords.'},
        tool_call_id='c1',
    )
    refs, message = _build_custom_tool_search_replay_blocks(empty, tool_search_active=True)
    assert refs == []
    assert message == 'No matches; try other keywords.'


def test_anthropic_custom_replay_blocks_matches_not_a_list() -> None:
    """When `content['discovered_tools']` is the wrong type, fall through to text formatting."""
    pytest.importorskip('anthropic')
    from pydantic_ai.messages import ToolReturnPart
    from pydantic_ai.models.anthropic import (
        _build_custom_tool_search_replay_blocks,  # pyright: ignore[reportPrivateUsage]
    )

    malformed = ToolReturnPart(
        tool_name='search_tools',
        content={'discovered_tools': 'not-a-list'},
        tool_call_id='c1',
    )
    refs, message = _build_custom_tool_search_replay_blocks(malformed, tool_search_active=True)
    assert refs is None and message is None


def test_anthropic_custom_replay_blocks_skips_non_dict_match_entries() -> None:
    """Match entries that aren't dicts are silently skipped (defensive parsing)."""
    pytest.importorskip('anthropic')
    from pydantic_ai.messages import ToolReturnPart
    from pydantic_ai.models.anthropic import (
        _build_custom_tool_search_replay_blocks,  # pyright: ignore[reportPrivateUsage]
    )

    mixed = ToolReturnPart(
        tool_name='search_tools',
        content={
            'discovered_tools': [
                'not-a-dict',
                {'name': 'foo'},
                {'name': 123},  # name not a string — skipped
                {'description': 'no name key'},  # no name — skipped
                {'name': 'bar'},
            ],
        },
        tool_call_id='c1',
    )
    refs, _msg = _build_custom_tool_search_replay_blocks(mixed, tool_search_active=True)
    assert refs is not None
    tool_names = [r['tool_name'] for r in refs]
    assert tool_names == ['foo', 'bar']


def test_anthropic_finalize_streamed_tool_search_call_part_with_canonical_dict_args() -> None:
    """Already-canonical `ToolSearchArgs` dict passes through unchanged — the typed
    contract guarantees `queries`, so re-running normalization would corrupt the data."""
    pytest.importorskip('anthropic')
    from pydantic_ai.messages import BuiltinToolSearchCallPart
    from pydantic_ai.models.anthropic import (
        _finalize_streamed_tool_search_call_part,  # pyright: ignore[reportPrivateUsage]
    )

    part = BuiltinToolSearchCallPart(
        args={'queries': ['mortgage']},
        tool_call_id='c1',
        provider_name='anthropic',
        provider_details={'strategy': 'bm25'},
    )
    result = _finalize_streamed_tool_search_call_part(part)
    assert result.args == {'queries': ['mortgage']}


def test_anthropic_finalize_streamed_tool_search_call_part_with_none_args() -> None:
    """`args=None` finalizes to a normalized empty `queries` list."""
    pytest.importorskip('anthropic')
    from pydantic_ai.messages import BuiltinToolSearchCallPart
    from pydantic_ai.models.anthropic import (
        _finalize_streamed_tool_search_call_part,  # pyright: ignore[reportPrivateUsage]
    )

    part = BuiltinToolSearchCallPart(args=None, tool_call_id='c1', provider_name='anthropic')
    result = _finalize_streamed_tool_search_call_part(part)
    assert isinstance(result.args, dict) and 'queries' in result.args


async def test_anthropic_map_message_empty_search_renders_message_text_block():
    """When custom-callable tool search returns no matches, `_map_message` emits the
    typed return as a single text-content `tool_result` block (not the default text
    fallthrough). Anthropic rejects empty `tool_result.content` arrays — this is the
    spec-compliant path for the custom-search empty-results case."""
    pytest.importorskip('anthropic')
    from pydantic_ai.builtin_tools.tool_search import ToolSearchTool
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .models.test_anthropic import MockAnthropic

    model = AnthropicModel(
        'claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=MockAnthropic.create_mock(()))
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='find me a mortgage tool')]),
        ModelResponse(
            parts=[ToolCallPart(tool_name='search_tools', args={'queries': ['mortgage']}, tool_call_id='c1')]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='search_tools',
                    content={
                        'discovered_tools': [],
                        'message': 'No matching tools found. Try other keywords.',
                    },
                    tool_call_id='c1',
                ),
            ],
        ),
    ]
    params = ModelRequestParameters(
        function_tools=[],
        builtin_tools=[ToolSearchTool(strategy='custom')],
        allow_text_output=True,
    )
    _system, anthropic_messages = await model._map_message(history, params, AnthropicModelSettings())  # pyright: ignore[reportPrivateUsage]

    # Find the tool_result block across all user messages.
    tool_results: list[dict[str, Any]] = [
        c
        for m in anthropic_messages
        if m['role'] == 'user' and isinstance(m['content'], list)
        for c in cast(list[Any], m['content'])
        if isinstance(c, dict) and cast(dict[str, Any], c).get('type') == 'tool_result'
    ]
    [tool_result] = tool_results
    assert tool_result['content'] == [{'text': 'No matching tools found. Try other keywords.', 'type': 'text'}]
    assert tool_result['is_error'] is False


async def test_anthropic_map_message_replays_tool_search_call_without_queries():
    """A `BuiltinToolSearchCallPart` with `args=None` (streaming partial state, or a
    history fragment that never carried args) falls through to forwarding the empty
    `args_as_dict()` to the wire `input`. Covers the `else: wire_input = args_dict`
    branch where the cross-provider `queries` slot isn't populated."""
    pytest.importorskip('anthropic')
    from pydantic_ai.builtin_tools.tool_search import ToolSearchTool
    from pydantic_ai.messages import BuiltinToolSearchCallPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .models.test_anthropic import MockAnthropic

    model = AnthropicModel(
        'claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=MockAnthropic.create_mock(()))
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[
                BuiltinToolSearchCallPart(
                    args=None,
                    tool_call_id='srv_1',
                    provider_name='anthropic',
                    provider_details={'strategy': 'bm25'},
                ),
            ],
        ),
    ]
    params = ModelRequestParameters(
        function_tools=[],
        builtin_tools=[ToolSearchTool(strategy='bm25')],
        allow_text_output=True,
    )
    _system, anthropic_messages = await model._map_message(history, params, AnthropicModelSettings())  # pyright: ignore[reportPrivateUsage]

    [assistant_msg] = [m for m in anthropic_messages if m['role'] == 'assistant']
    assistant_content = cast(list[Any], assistant_msg['content'])
    server_tool_uses: list[dict[str, Any]] = [
        c for c in assistant_content if isinstance(c, dict) and cast(dict[str, Any], c).get('type') == 'server_tool_use'
    ]
    [server_tool_use] = server_tool_uses
    assert server_tool_use['input'] == {}


def test_anthropic_build_tool_search_replay_block_skips_malformed_content_shapes() -> None:
    """`_build_tool_search_replay_block` defensively handles malformed content:
    non-dict content, non-list `discovered_tools`, non-dict entries, and entries
    with missing/non-string names all get skipped without crashing."""
    pytest.importorskip('anthropic')
    from pydantic_ai.messages import BuiltinToolSearchReturnPart
    from pydantic_ai.models.anthropic import _build_tool_search_replay_block  # pyright: ignore[reportPrivateUsage]

    # Non-dict content — falls through with empty `tool_references`.
    part_str_content = BuiltinToolSearchReturnPart(content='not a dict', tool_call_id='c1')
    block_str = _build_tool_search_replay_block(part_str_content, 'srv_1')
    assert block_str['content'] == {'type': 'tool_search_tool_search_result', 'tool_references': []}

    # Dict content with non-list `discovered_tools` — same fall-through.
    part_bad_list = BuiltinToolSearchReturnPart(
        content=cast(Any, {'discovered_tools': 'not a list'}),
        tool_call_id='c2',
    )
    block_bad_list = _build_tool_search_replay_block(part_bad_list, 'srv_2')
    assert block_bad_list['content'] == {'type': 'tool_search_tool_search_result', 'tool_references': []}

    # Mixed entries: non-dict, missing name, non-string name, valid — only valids surface.
    part_mixed = BuiltinToolSearchReturnPart(
        content=cast(
            Any,
            {
                'discovered_tools': [
                    'not a dict',
                    {'description': 'no name'},
                    {'name': 123},
                    {'name': 'ok_tool'},
                ],
            },
        ),
        tool_call_id='c3',
    )
    block_mixed = _build_tool_search_replay_block(part_mixed, 'srv_3')
    inner = cast(dict[str, Any], block_mixed['content'])
    refs = cast(list[dict[str, Any]], inner['tool_references'])
    assert [r['tool_name'] for r in refs] == ['ok_tool']


def test_openai_normalize_tool_search_args_handles_non_dict_input() -> None:
    """OpenAI's `_normalize_tool_search_args` falls through to an empty `queries` list
    for any non-dict input (e.g. raw `None` from an empty server tool-search call)."""
    pytest.importorskip('openai')
    from pydantic_ai.models.openai import _normalize_tool_search_args  # pyright: ignore[reportPrivateUsage]

    assert _normalize_tool_search_args(None) == {'queries': []}
    assert _normalize_tool_search_args('') == {'queries': []}
    # Dict without `paths` key — also empty.
    assert _normalize_tool_search_args({'something_else': 'x'}) == {'queries': []}
    # Dict with non-list `paths` — also empty.
    assert _normalize_tool_search_args({'paths': 'not a list'}) == {'queries': []}


def test_openai_build_client_tool_search_output_param_handles_malformed_content() -> None:
    """`_build_client_tool_search_output_param` defensively handles malformed
    `discovered_tools` shapes — non-dict content, non-list matches, non-dict entries,
    non-string names — without crashing."""
    pytest.importorskip('openai')
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.openai import _build_client_tool_search_output_param  # pyright: ignore[reportPrivateUsage]

    params = ModelRequestParameters(function_tools=[], builtin_tools=[], allow_text_output=True)

    # Non-dict content — no tools surface.
    part_none = ToolReturnPart(tool_name='search_tools', content='not a dict', tool_call_id='c1')
    out_none = _build_client_tool_search_output_param(part_none, 'c1', params)
    assert out_none['tools'] == []

    # Dict content with non-list `discovered_tools` — same fall-through.
    part_bad_list = ToolReturnPart(
        tool_name='search_tools',
        content={'discovered_tools': 'not a list'},
        tool_call_id='c2',
    )
    out_bad_list = _build_client_tool_search_output_param(part_bad_list, 'c2', params)
    assert out_bad_list['tools'] == []

    # Mixed entries: non-dict, missing name, non-string name — all skipped.
    part_mixed = ToolReturnPart(
        tool_name='search_tools',
        content={
            'discovered_tools': [
                'not a dict',
                {'description': 'no name'},
                {'name': 123},
                # No matching `function_tools` for any valid name; output stays empty
                # because `_build_client_tool_search_output_param` looks up by name.
            ],
        },
        tool_call_id='c3',
    )
    out_mixed = _build_client_tool_search_output_param(part_mixed, 'c3', params)
    assert out_mixed['tools'] == []


# --- Cross-provider local→native promotion ---
#
# The local-fallback path emits typed `ToolSearchCallPart` / `ToolSearchReturnPart`
# (subclasses of the regular `ToolCallPart` / `ToolReturnPart`). When a follow-up
# turn runs on a provider that natively supports tool search, the adapter should
# render those local-shape parts back into the provider's native wire format so the
# previously discovered tools get unlocked from `defer_loading=True` without forcing
# the model to re-search. This must work regardless of the current turn's `strategy`
# (default native, named native, or custom callable) — the gate is "current request
# has any tool search active", not "strategy is custom".


async def test_anthropic_promotes_local_search_history_with_default_native_strategy() -> None:
    """Local-shape ``ToolSearch*Part`` from a prior cross-provider turn must render
    into Anthropic's native tool_search wire when the current turn is the default
    server-executed strategy (``ToolSearchTool()`` / `strategy=None`).

    The wire shape uses Anthropic's "client-side flavor" of tool search per empirical
    research: a standard ``tool_use`` for the local ``search_tools`` function tool
    paired with a ``tool_result`` whose ``content`` is a ``tool_reference`` array
    (NOT a string of stringified discoveries). Anthropic's server unlocks the
    discovered tools' schemas from ``defer_loading=true`` once it sees the
    ``tool_reference`` block.

    Currently fails because ``_build_custom_tool_search_replay_blocks`` is gated on
    ``strategy='custom'``, so the default-strategy case falls through and the return
    is rendered as a plain ``tool_result`` carrying stringified content — the
    discovered tools stay hidden and the model has to re-search.
    """
    pytest.importorskip('anthropic')
    from pydantic_ai.builtin_tools.tool_search import ToolSearchTool
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .models.test_anthropic import MockAnthropic

    model = AnthropicModel(
        'claude-sonnet-4-6',
        provider=AnthropicProvider(anthropic_client=MockAnthropic.create_mock(())),
    )

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='find a weather tool')]),
        ModelResponse(
            parts=[
                ToolSearchCallPart(args={'queries': ['weather']}, tool_call_id='c1'),
            ],
        ),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_weather', 'description': None}]},
                    tool_call_id='c1',
                ),
            ],
        ),
    ]
    # Default native strategy (NOT 'custom') — currently the gate that activates the
    # tool_reference replay re-formatting only fires for `strategy='custom'`.
    params = ModelRequestParameters(
        function_tools=[],
        builtin_tools=[ToolSearchTool()],
        allow_text_output=True,
    )

    _system, anthropic_messages = await model._map_message(history, params, AnthropicModelSettings())  # pyright: ignore[reportPrivateUsage]

    tool_results: list[dict[str, Any]] = [
        c
        for m in anthropic_messages
        if m['role'] == 'user' and isinstance(m['content'], list)
        for c in cast(list[Any], m['content'])
        if isinstance(c, dict) and cast(dict[str, Any], c).get('type') == 'tool_result'
    ]
    [tool_result] = tool_results
    # Promotion target: the result content must be a `tool_reference` array, not a
    # stringified discovery JSON. Anthropic uses this shape to unlock deferred tools.
    assert tool_result['content'] == [{'type': 'tool_reference', 'tool_name': 'get_weather'}]


async def test_anthropic_promotes_local_search_history_with_named_native_strategy() -> None:
    """Same promotion as above but with an explicit named native strategy
    (``strategy='bm25'``). Confirms the gate is "any tool search active", not "custom"
    or "default" — whenever the provider supports native tool search and the current
    request carries it, the historical local-shape parts get the native wire.
    """
    pytest.importorskip('anthropic')
    from pydantic_ai.builtin_tools.tool_search import ToolSearchTool
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .models.test_anthropic import MockAnthropic

    model = AnthropicModel(
        'claude-sonnet-4-6',
        provider=AnthropicProvider(anthropic_client=MockAnthropic.create_mock(())),
    )

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='find a calc tool')]),
        ModelResponse(
            parts=[
                ToolSearchCallPart(args={'queries': ['calc']}, tool_call_id='c2'),
            ],
        ),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate', 'description': None}]},
                    tool_call_id='c2',
                ),
            ],
        ),
    ]
    params = ModelRequestParameters(
        function_tools=[],
        builtin_tools=[ToolSearchTool(strategy='bm25')],
        allow_text_output=True,
    )

    _system, anthropic_messages = await model._map_message(history, params, AnthropicModelSettings())  # pyright: ignore[reportPrivateUsage]

    tool_results: list[dict[str, Any]] = [
        c
        for m in anthropic_messages
        if m['role'] == 'user' and isinstance(m['content'], list)
        for c in cast(list[Any], m['content'])
        if isinstance(c, dict) and cast(dict[str, Any], c).get('type') == 'tool_result'
    ]
    [tool_result] = tool_results
    assert tool_result['content'] == [{'type': 'tool_reference', 'tool_name': 'calculate'}]


async def test_openai_promotes_local_search_history_with_default_native_strategy() -> None:
    """Local-shape ``ToolSearch*Part`` from a prior cross-provider turn must render
    into OpenAI's native tool_search wire when the current turn is the default
    server-executed strategy (``ToolSearchTool()`` / `strategy=None`).

    The wire shape uses ``tool_search_call`` + ``tool_search_output`` items with
    ``execution='client'`` per empirical research — even though the current turn is
    server-executed, the historical replay must use ``execution='client'`` because
    the prior turn was framework-executed locally and OpenAI accepts ``'client'`` as
    the historical-replay shape regardless of the current turn's mode.

    Currently fails because both ``_get_tools`` and the ``_map_messages`` replay
    branches are gated on ``_has_custom_tool_search`` (i.e. ``strategy='custom'`` on
    the active builtin), so the default-native case never activates the promotion.
    """
    pytest.importorskip('openai')
    from openai.types.responses import ResponseFunctionToolCallParam

    from pydantic_ai.builtin_tools.tool_search import ToolSearchTool
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.tools import ToolDefinition

    from .models.mock_openai import MockOpenAIResponses

    model = OpenAIResponsesModel(
        'gpt-5.4-mini',
        provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())),
    )

    discovered_tool = ToolDefinition(
        name='get_weather',
        description='Get the weather for a city.',
        parameters_json_schema={
            'type': 'object',
            'properties': {'city': {'type': 'string'}},
            'required': ['city'],
        },
    )

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='find a weather tool')]),
        ModelResponse(
            parts=[
                ToolSearchCallPart(args={'queries': ['weather']}, tool_call_id='oc1'),
            ],
            provider_name='openai',
        ),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_weather', 'description': None}]},
                    tool_call_id='oc1',
                ),
            ],
        ),
    ]
    # Default native strategy — `ToolSearchTool()` with no `strategy='custom'`.
    # Discovered tool needs to be in `function_tools` so the replay can pair the
    # `tool_search_output.tools[]` schema by name.
    params = ModelRequestParameters(
        function_tools=[discovered_tool],
        builtin_tools=[ToolSearchTool()],
        allow_text_output=True,
    )

    from pydantic_ai.models.openai import OpenAIResponsesModelSettings

    _system, openai_messages = await model._map_messages(history, OpenAIResponsesModelSettings(), params)  # pyright: ignore[reportPrivateUsage]

    # The local search call should render as a `tool_search_call` item with
    # `execution='client'`, and the local return should render as a paired
    # `tool_search_output` carrying the `get_weather` schema.
    tool_search_calls = [
        item
        for item in openai_messages
        if isinstance(item, dict) and cast(dict[str, Any], item).get('type') == 'tool_search_call'
    ]
    tool_search_outputs = [
        item
        for item in openai_messages
        if isinstance(item, dict) and cast(dict[str, Any], item).get('type') == 'tool_search_output'
    ]
    function_calls = [
        item
        for item in openai_messages
        if isinstance(item, dict) and cast(dict[str, Any], item).get('type') == 'function_call'
    ]
    function_outputs = [
        item
        for item in openai_messages
        if isinstance(item, dict) and cast(dict[str, Any], item).get('type') == 'function_call_output'
    ]

    assert len(tool_search_calls) == 1, (
        f'expected 1 tool_search_call, got {len(tool_search_calls)}; full output: {openai_messages}'
    )
    assert len(tool_search_outputs) == 1
    assert tool_search_calls[0].get('execution') == 'client'
    assert tool_search_outputs[0].get('execution') == 'client'

    # Output carries the discovered tool's full schema for OpenAI to "rediscover".
    output_tools = cast(list[dict[str, Any]], tool_search_outputs[0].get('tools'))
    assert len(output_tools) == 1
    assert output_tools[0]['name'] == 'get_weather'
    assert output_tools[0]['type'] == 'function'

    # The `search_tools` exchange must NOT also surface as a regular function_call /
    # function_call_output — that would double-count the discovery.
    assert not any(
        cast(dict[str, Any], call).get('name') == _SEARCH_TOOLS_NAME
        for call in cast(list[ResponseFunctionToolCallParam], function_calls)
    )
    assert not function_outputs


# --- `strategy='keywords'` on natively-supporting providers ---
#
# `'keywords'` is a strategy CHOICE: "use the keyword-overlap algorithm". The execution
# mode (server-side / client-executed-native / local fallback) is auto-derived from
# the algorithm's needs and the provider's capabilities. On Anthropic and OpenAI,
# native tool search is available and the keyword algorithm runs LOCALLY but the
# wire ships in the provider's native tool-search shape so the prompt cache stays
# warm across discovery rounds (deferred tools don't get re-added to the request's
# tool definitions on each turn).


def test_tool_search_strategy_keywords_registers_builtin_for_client_execution() -> None:
    """`ToolSearch(strategy='keywords')` must register `ToolSearchTool(strategy='custom',
    optional=True)` so the client-executed native path engages on supporting providers.

    Currently fails because `get_builtin_tools` returns `[]` for `'keywords'`,
    forcing the local-fallback path on every provider — losing the cache benefit
    that the client-executed native path provides on Anthropic and OpenAI.
    """
    cap: ToolSearch[None] = ToolSearch(strategy='keywords')
    builtins = cap.get_builtin_tools()
    assert len(builtins) == 1
    [builtin] = builtins
    assert isinstance(builtin, ToolSearchTool)
    # `strategy='custom'` marks the builtin as "the algorithm runs on our side"; the
    # adapter then wires it as Anthropic's tool_use+tool_reference flavor or OpenAI's
    # `execution='client'`. `optional=True` so it gets dropped on providers that
    # don't support it (toolset's local `search_tools` function tool is the fallback).
    assert builtin.strategy == 'custom'
    assert builtin.optional is True


async def test_openai_promotes_mixed_native_and_local_history_a_b_c_chain() -> None:
    """Multi-hop chain: Anthropic-native turn 1 → local turn 2 (Google etc.) → OpenAI turn 3.

    The persisted history at turn 3 carries BOTH a `BuiltinToolSearch*Part` from the
    Anthropic turn AND a `ToolSearch*Part` from the local turn. OpenAI's adapter must
    promote both into native `tool_search_call`+`tool_search_output` items so the
    discovered tools' schemas stay unlocked across the entire chain — the model
    shouldn't have to re-search anything it discovered earlier.
    """
    pytest.importorskip('openai')
    from pydantic_ai.builtin_tools.tool_search import ToolSearchTool
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.tools import ToolDefinition

    from .models.mock_openai import MockOpenAIResponses

    model = OpenAIResponsesModel(
        'gpt-5.4-mini',
        provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())),
    )

    weather = ToolDefinition(
        name='get_weather',
        description='Get the weather for a city.',
        parameters_json_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']},
    )
    calc = ToolDefinition(
        name='calculate_mortgage',
        description='Calculate monthly mortgage payment.',
        parameters_json_schema={'type': 'object', 'properties': {'p': {'type': 'number'}}, 'required': ['p']},
    )

    # Turn 1 on Anthropic: native bm25, discovers `get_weather`.
    # Turn 2 on Google: local function tool, discovers `calculate_mortgage`.
    # Turn 3 on OpenAI: should promote BOTH discoveries to native wire.
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='find a weather tool')]),
        # Anthropic-native (turn 1) — `BuiltinToolSearch*Part`.
        ModelResponse(
            parts=[
                BuiltinToolSearchCallPart(
                    args={'queries': ['weather']},
                    tool_call_id='ant_1',
                    provider_name='anthropic',
                    provider_details={'strategy': 'bm25'},
                ),
                BuiltinToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_weather', 'description': None}]},
                    tool_call_id='ant_1',
                    provider_name='anthropic',
                ),
            ],
            provider_name='anthropic',
        ),
        ModelRequest(parts=[UserPromptPart(content='now find a mortgage one')]),
        # Local fallback (turn 2 on Google or similar) — `ToolSearch*Part`.
        ModelResponse(parts=[ToolSearchCallPart(args={'queries': ['mortgage']}, tool_call_id='loc_1')]),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage', 'description': None}]},
                    tool_call_id='loc_1',
                ),
            ],
        ),
        ModelRequest(parts=[UserPromptPart(content='now compute both')]),
    ]

    params = ModelRequestParameters(
        function_tools=[weather, calc],
        builtin_tools=[ToolSearchTool()],
        allow_text_output=True,
    )

    _system, openai_messages = await model._map_messages(history, OpenAIResponsesModelSettings(), params)  # pyright: ignore[reportPrivateUsage]

    tool_search_calls = [
        item
        for item in openai_messages
        if isinstance(item, dict) and cast(dict[str, Any], item).get('type') == 'tool_search_call'
    ]
    tool_search_outputs = [
        item
        for item in openai_messages
        if isinstance(item, dict) and cast(dict[str, Any], item).get('type') == 'tool_search_output'
    ]

    # Both prior discoveries should surface as native tool_search exchanges with execution=client.
    # The local-fallback one promotes via the new gating; the Anthropic-native one is left as-is
    # because its provider_name doesn't match self.system (foreign-provider builtin parts are
    # filtered out from the OpenAI wire, but get_weather still needs to be discoverable — that's
    # handled by the toolset re-emitting it as a regular function tool in this turn's `tools[]`).
    assert len(tool_search_calls) >= 1, (
        f'expected at least one promoted tool_search_call (local→native), got {len(tool_search_calls)}; '
        f'output: {openai_messages}'
    )
    assert len(tool_search_outputs) >= 1
    # The local discovery (`calculate_mortgage`) made it into the promoted output.
    output_tools_names = {
        cast(dict[str, Any], t).get('name')
        for output in tool_search_outputs
        for t in cast(list[Any], cast(dict[str, Any], output).get('tools', []))
    }
    assert 'calculate_mortgage' in output_tools_names, (
        f'local-fallback discovery should be promoted; got tools: {output_tools_names}'
    )


async def test_tool_search_strategy_keywords_runs_keyword_algorithm_via_search_fn() -> None:
    """When `strategy='keywords'` activates the client-executed native path, the local
    `search_tools` function (still in `function_tools` for client-execution) must run
    the built-in keyword-overlap algorithm — not error out with no `search_fn` set.

    Verifies end-to-end: the toolset's `search_fn` is wired to a callable that
    matches keywords against the corpus, returning matching tool names.
    """
    cap: ToolSearch[None] = ToolSearch(strategy='keywords')
    base = _create_function_toolset()
    # `get_wrapper_toolset` is what the framework calls when injecting the capability.
    ts = cap.get_wrapper_toolset(base)
    assert isinstance(ts, ToolSearchToolset)
    # Internal `search_fn` is set so `_run_search_fn` (not `_run_keywords_search`) handles
    # the dispatch — but the algorithm is still keyword overlap.
    assert ts.search_fn is not None

    ctx = _build_run_context(None)
    tools = await ts.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]
    result = await ts.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'mortgage'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    return_value = cast(dict[str, Any], result.return_value)
    discovered_names = {match['name'] for match in return_value['discovered_tools']}
    assert 'calculate_mortgage' in discovered_names
