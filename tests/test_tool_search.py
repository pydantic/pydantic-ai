"""Tests for tool search functionality.

Unit tests for ToolSearchToolset plus VCR integration tests using pydantic-evals.

NOTE: If you change the search tool description or keyword schema in _tool_search.py,
re-record all cassettes with: uv run pytest tests/test_tool_search.py --record-mode=rewrite
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, TypeVar

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import Agent, FunctionToolset, ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolReturn, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.toolsets._tool_search import (
    _DISCOVERED_TOOLS_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    _SEARCH_TOOLS_NAME,  # pyright: ignore[reportPrivateUsage]
    ToolSearchToolset,
)
from pydantic_ai.usage import RunUsage

from .conftest import try_import

with try_import() as evals_available:
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext

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


def _summarize_report(report: Any) -> dict[str, ScenarioSummary]:
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


def _build_run_context(deps: T, run_step: int = 0, messages: list[ModelMessage] | None = None) -> RunContext[T]:
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
    def calculate_mortgage(principal: float, rate: float, years: int) -> str:  # pragma: no cover
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
    """Test that deferred tools are not exposed initially."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert tool_names == snapshot(['search_tools', 'get_weather', 'get_time'])
    assert 'calculate_mortgage' not in tool_names
    assert 'stock_price' not in tool_names
    assert 'crypto_price' not in tool_names


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
    assert isinstance(result, ToolReturn)
    assert result.return_value == snapshot(
        [{'name': 'calculate_mortgage', 'description': 'Calculate monthly mortgage payment for a loan.'}]
    )
    assert result.metadata == snapshot({'discovered_tools': ['calculate_mortgage']})


async def test_tool_search_toolset_search_is_case_insensitive():
    """Test that search is case insensitive."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'STOCK'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    rv: list[dict[str, str | None]] = result.return_value  # pyright: ignore[reportAssignmentType]
    assert len(rv) == 1
    assert rv[0]['name'] == 'stock_price'


async def test_tool_search_toolset_search_matches_description():
    """Test that search matches tool descriptions."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'cryptocurrency'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    rv: list[dict[str, str | None]] = result.return_value  # pyright: ignore[reportAssignmentType]
    assert len(rv) == 1
    assert rv[0]['name'] == 'crypto_price'


async def test_tool_search_toolset_search_returns_no_matches():
    """Test that search returns empty list when no matches."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'nonexistent'}, ctx, search_tool)
    assert isinstance(result, ToolReturn)
    assert result.return_value == snapshot('No matching tools found. The tools you need may not be available.')
    assert result.metadata == snapshot({'discovered_tools': []})


async def test_tool_search_toolset_search_empty_query():
    """Test that search with empty query raises ModelRetry."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    with pytest.raises(ModelRetry, match='Please provide search keywords.'):
        await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': ''}, ctx, search_tool)


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
    rv: list[dict[str, str | None]] = result.return_value  # pyright: ignore[reportAssignmentType]
    assert len(rv) == 10


async def test_tool_search_toolset_discovered_tools_available():
    """Test that discovered tools become available after search."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={
                        'message': "Found 1 tool(s) matching 'mortgage'",
                        'tools': [{'name': 'calculate_mortgage'}],
                    },
                    metadata={_DISCOVERED_TOOLS_METADATA_KEY: ['calculate_mortgage']},
                ),
            ]
        )
    ]
    ctx = _build_run_context(None, messages=messages)

    tools = await searchable.get_tools(ctx)
    tool_names = list(tools.keys())

    assert 'calculate_mortgage' in tool_names
    assert 'stock_price' not in tool_names


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
    assert _SEARCH_TOOLS_NAME not in tool_names


async def test_agent_always_wraps_in_tool_search_toolset():
    """Test that agent always wraps toolset in ToolSearchToolset."""
    agent = Agent('test')

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    toolset = agent._get_toolset()  # pyright: ignore[reportPrivateUsage]
    assert isinstance(toolset, ToolSearchToolset)


async def test_agent_wraps_in_tool_search_toolset_with_deferred():
    """Test that agent wraps with ToolSearchToolset when there are deferred tools."""
    agent = Agent('test')

    @agent.tool_plain
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the current weather for a city."""
        return f'Weather in {city}'

    @agent.tool_plain(defer_loading=True)
    def calculate_mortgage(principal: float) -> str:  # pragma: no cover
        """Calculate mortgage payment."""
        return 'Calculated'

    toolset = agent._get_toolset()  # pyright: ignore[reportPrivateUsage]
    assert isinstance(toolset, ToolSearchToolset)


async def test_tool_manager_with_tool_search_toolset():
    """Test that ToolManager works correctly with ToolSearchToolset."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tool_manager = ToolManager[None](searchable)
    run_step_toolset = await tool_manager.for_run_step(ctx)

    tool_names = [t.name for t in run_step_toolset.tool_defs]
    assert 'search_tools' in tool_names
    assert 'get_weather' in tool_names
    assert 'calculate_mortgage' not in tool_names

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
    assert result.return_value == snapshot([{'name': 'no_desc_tool', 'description': None}])


async def test_tool_search_toolset_multiple_searches_accumulate():
    """Test that tools discovered in multiple searches accumulate correctly."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={
                        'message': "Found 1 tool(s) matching 'mortgage'",
                        'tools': [{'name': 'calculate_mortgage'}],
                    },
                    metadata={_DISCOVERED_TOOLS_METADATA_KEY: ['calculate_mortgage']},
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'message': "Found 1 tool(s) matching 'stock'", 'tools': [{'name': 'stock_price'}]},
                    metadata={_DISCOVERED_TOOLS_METADATA_KEY: ['stock_price']},
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

    assert tool_names == snapshot(['search_tools'])
    assert 'deferred_tool1' not in tool_names
    assert 'deferred_tool2' not in tool_names


async def test_tool_search_toolset_ignores_non_metadata_history():
    """Test that discovery only reads metadata, ignoring malformed content."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        # metadata is None (no discovered tools)
        ModelRequest(parts=[ToolReturnPart(tool_name=_SEARCH_TOOLS_NAME, content={'message': 'hi'})]),
        # metadata is not a dict
        ModelRequest(
            parts=[ToolReturnPart(tool_name=_SEARCH_TOOLS_NAME, content={'tools': 'not a list'}, metadata='not a dict')]
        ),
        # metadata is a dict but value is not a list
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'tools': []},
                    metadata={_DISCOVERED_TOOLS_METADATA_KEY: 'not a list'},
                )
            ]
        ),
        # metadata contains non-string items in the list
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'tools': []},
                    metadata={_DISCOVERED_TOOLS_METADATA_KEY: [123, None]},
                )
            ]
        ),
        # valid metadata
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content={'message': 'found', 'tools': [{'name': 'calculate_mortgage'}]},
                    metadata={_DISCOVERED_TOOLS_METADATA_KEY: ['calculate_mortgage']},
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


async def test_call_tool_returns_tool_return_with_metadata():
    """Test that call_tool for search_tools returns a ToolReturn with metadata listing matched tools."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'keywords': 'mortgage'}, ctx, search_tool)
    assert result == snapshot(
        ToolReturn(
            return_value=[
                {'name': 'calculate_mortgage', 'description': 'Calculate monthly mortgage payment for a loan.'}
            ],
            metadata={'discovered_tools': ['calculate_mortgage']},
        )
    )
