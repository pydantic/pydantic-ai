"""Tests for tool search functionality.

Unit tests for ToolSearchToolset plus VCR integration tests using pydantic-evals.

NOTE: If you change the search tool description or keyword schema in _tool_search.py,
re-record all cassettes with: uv run pytest tests/test_tool_search.py --record-mode=rewrite
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import pytest
import yaml
from inline_snapshot import snapshot
from pydantic import BaseModel
from pytest_mock import MockerFixture
from typing_extensions import TypedDict

import pydantic_ai.agent as agent_module
from pydantic_ai import Agent, FunctionToolset, ToolCallPart
from pydantic_ai._agent_graph import _clean_message_history  # pyright: ignore[reportPrivateUsage]
from pydantic_ai._deferred_capabilities import parse_loaded_capabilities
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_search import (
    synthesize_local_from_native_call,
    synthesize_local_tool_search_messages,
)
from pydantic_ai.capabilities import CAPABILITY_TYPES, ProcessHistory, ToolSearch
from pydantic_ai.capabilities._ordering import collect_leaves
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.capability import Capability
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.exceptions import ModelAPIError, ModelRetry, ToolRetryError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    CompactionPart,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    NativeToolSearchCallPart,
    NativeToolSearchReturnPart,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolPartKind,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnContent,
    ToolSearchReturnPart,
    UserPromptPart,
    _model_request_part_discriminator,  # pyright: ignore[reportPrivateUsage]
    _model_response_part_discriminator,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.models import Model, ModelRequestParameters, infer_model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool, WebSearchTool
from pydantic_ai.native_tools._tool_search import ToolSearchMatch, ToolSearchTool
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets._deferred_capability_loader import LOAD_CAPABILITY_TOOL_NAME
from pydantic_ai.toolsets._tool_search import (
    ToolSearchToolset,
    keywords_search_fn,
    parse_discovered_tools,
)
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.usage import RequestUsage, RunUsage

from .conftest import iter_message_parts, message, message_part, try_import

_SEARCH_TOOLS_NAME = ToolSearch.function_tool_name

with try_import() as evals_available:
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext
    from pydantic_evals.reporting import EvaluationReport

with try_import() as ag_ui_available:
    from pydantic_ai.ui.ag_ui import AGUIAdapter


def ag_ui_preserves_tool_kind() -> bool:
    """Whether the installed `ag-ui-protocol` carries a tool call's kind through a round-trip.

    `0.1.10` — the floor the `ag-ui` extra declares — lacks what the adapter needs, so a
    `ToolSearchCallPart` comes back as a plain `ToolCallPart` and the request renders differently.
    `0.1.11` and up round-trip it. Worth recording rather than papering over: on `0.1.10` a
    `ToolAvailabilityDeltaPart` is the only one of the three representations that survives, because
    it doesn't ride the tool-call channel at all.
    """
    # `lax no cover`, not `no cover`: whether this branch runs depends on whether the `ag-ui` extra is
    # installed, so it's covered in one CI job and dead in another.
    if not ag_ui_available():  # pragma: lax no cover
        return False

    from importlib.metadata import version

    from packaging.version import Version

    return Version(version('ag-ui-protocol')) >= Version('0.1.11')


with try_import() as anthropic_available:
    import anthropic  # pyright: ignore[reportUnusedImport]  # noqa: F401
    from anthropic.types.beta import (
        BetaServerToolUseBlock,
        BetaTextBlock,
        BetaToolSearchToolResultBlock,
        BetaUsage,
    )
    from anthropic.types.beta.beta_server_tool_use_block import BetaDirectCaller
    from anthropic.types.beta.beta_tool_search_tool_result_error import BetaToolSearchToolResultError

    from pydantic_ai.models.anthropic import (
        AnthropicModel,
        AnthropicModelSettings,
        _build_custom_tool_search_replay_blocks,  # pyright: ignore[reportPrivateUsage]
        _build_tool_search_replay_block,  # pyright: ignore[reportPrivateUsage]
        _finalize_streamed_tool_search_call_part,  # pyright: ignore[reportPrivateUsage]
        _map_server_tool_use_block,  # pyright: ignore[reportPrivateUsage]
        _map_tool_search_tool_result_block,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .models.test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

with try_import() as openai_available:
    from openai.types.responses import (
        FunctionTool,
        ResponseFunctionToolCall,
        ResponseFunctionToolCallParam,
        ResponseOutputMessage,
        ResponseOutputText,
        ResponseToolSearchCall,
        ResponseToolSearchOutputItem,
    )
    from openai.types.responses.file_search_tool import FileSearchTool

    from pydantic_ai.models.openai import (
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
        _build_tool_search_return_part,  # pyright: ignore[reportPrivateUsage]
        _map_client_tool_search_call,  # pyright: ignore[reportPrivateUsage]
        _map_tool_search_call,  # pyright: ignore[reportPrivateUsage]
        _normalize_tool_search_args,  # pyright: ignore[reportPrivateUsage]
        _tool_search_namespace_for_synthesis,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

with try_import() as google_available:
    import google.genai  # pyright: ignore[reportUnusedImport]  # noqa: F401
    from google.genai.types import Candidate, Content, GenerateContentResponse, Part

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = pytest.mark.anyio

MOCK_API_KEYS: dict[str, str] = {
    'OPENAI_API_KEY': 'mock-api-key',
    'ANTHROPIC_API_KEY': 'mock-api-key',
    # google-gla checks GEMINI_API_KEY only. Mocking GOOGLE_API_KEY would shadow a real
    # GEMINI_API_KEY in .env because the google-genai SDK prefers GOOGLE_API_KEY when both
    # are present, so re-recording against real credentials would silently use the mock.
    'GEMINI_API_KEY': 'mock-api-key',
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

        def evaluate(self, ctx: EvaluatorContext[str, EvalOutput, EvalMetadata]) -> bool:
            if not ctx.metadata or not ctx.metadata.expected_tools:
                return True
            return 'search_tools' in ctx.output.tool_calls

        def get_default_evaluation_name(self) -> str:
            return 'used_search_tools'

    @dataclass(repr=False)
    class FoundExpectedTools(Evaluator[str, EvalOutput, EvalMetadata]):
        """Check that the model found and called the expected tools."""

        def evaluate(self, ctx: EvaluatorContext[str, EvalOutput, EvalMetadata]) -> bool:
            if not ctx.metadata or not ctx.metadata.expected_tools:
                return True
            return all(t in ctx.output.tool_calls for t in ctx.metadata.expected_tools)

        def get_default_evaluation_name(self) -> str:
            return 'found_expected_tools'

    @dataclass(repr=False)
    class ReasonableToolUsage(Evaluator[str, EvalOutput, EvalMetadata]):
        """Check that the model didn't use an excessive number of tool calls."""

        max_calls: int = 10

        def evaluate(self, ctx: EvaluatorContext[str, EvalOutput, EvalMetadata]) -> bool:
            return len(ctx.output.tool_calls) <= self.max_calls

        def get_default_evaluation_name(self) -> str:
            return 'reasonable_usage'

    @dataclass(repr=False)
    class KeywordCount(Evaluator[str, EvalOutput, EvalMetadata]):
        """Score the number of keywords used in the search query. Best is <= 3."""

        def evaluate(self, ctx: EvaluatorContext[str, EvalOutput, EvalMetadata]) -> int | dict[str, int]:
            if not ctx.output.search_args:
                return {}
            raw: Any = ctx.output.search_args[0].get('queries')
            queries = cast('list[str]', raw) if isinstance(raw, list) else ([str(raw)] if raw else [])
            return len(' '.join(queries).split())

        def get_default_evaluation_name(self) -> str:
            return 'keyword_count'


# --- Helpers ---


def _extract_tool_calls(result: AgentRunResult[str]) -> list[str]:
    """Extract tool-call names across both local and native tool-search paths.

    Normalizes native tool-search calls (`NativeToolSearchCallPart`, `tool_name='tool_search'`)
    to `'search_tools'` so the evaluator sees the same name regardless of which path the
    active provider took.
    """
    tool_calls: list[str] = []
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, (ToolCallPart, NativeToolCallPart)):
                    name = 'search_tools' if part.tool_kind == 'tool-search' else part.tool_name
                    tool_calls.append(name)
    return tool_calls


def _extract_search_args(result: AgentRunResult[str]) -> list[dict[str, str]]:
    """Extract parsed args dicts from tool-search calls across local and native paths."""
    args_list: list[dict[str, str]] = []
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if (
                    isinstance(part, (ToolCallPart, NativeToolCallPart))
                    and part.tool_kind == 'tool-search'
                    and part.args is not None
                ):
                    parsed = json.loads(part.args) if isinstance(part.args, str) else part.args
                    args_list.append({k: str(v) for k, v in parsed.items()})
    return args_list


def _build_agent(model_name: str) -> Agent[object, str]:
    """Build an agent with a visible tool and several deferred tools for testing.

    Forces the local `search_tools` function-tool path on every provider by removing
    `ToolSearchTool` from the model profile's `supported_native_tools`. This eval
    exercises OUR search-tool prompts and behavior; providers' native tool-search
    paths use the provider's own prompts and aren't under test here.
    """
    model = infer_model(model_name)
    # Override the cached profile to drop ToolSearchTool — forces the local path
    # uniformly across providers with and without native tool-search support.
    setattr(
        model,
        'profile',
        merge_profile(
            model.profile,
            ModelProfile(
                supported_native_tools=model.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)
                - {ToolSearchTool}
            ),
        ),
    )
    agent = Agent(model=model)

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
            raw: Any = output.search_args[0].get('queries')
            queries = cast('list[str]', raw) if isinstance(raw, list) else ([str(raw)] if raw else [])
            keywords = ' '.join(queries) or None
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
        model_name='openai-chat:gpt-5.4-mini',
        scenario_summary=snapshot(
            {
                'exchange_rate': {
                    'keywords': "['exchange rate currency USD EUR current']",
                    'tool_calls': ['search_tools', 'get_exchange_rate'],
                },
                'stock_price': {
                    'keywords': "['stock price market quote AAPL current']",
                    'tool_calls': ['search_tools', 'stock_lookup'],
                },
                'translation': {'keywords': None, 'tool_calls': []},
                'no_matching_tool': {
                    'keywords': None,
                    'tool_calls': [],
                },
            }
        ),
    ),
    ModelCase(
        model_name='anthropic:claude-sonnet-4-5',
        marks=[
            pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed'),
        ],
        scenario_summary=snapshot(
            {
                'exchange_rate': {
                    'keywords': "['currency exchange rate', 'USD EUR conversion', 'foreign exchange', 'currency converter']",
                    'tool_calls': ['search_tools', 'get_exchange_rate'],
                },
                'stock_price': {
                    'keywords': "['stock price', 'AAPL', 'ticker symbol', 'market data', 'financial data']",
                    'tool_calls': ['search_tools', 'stock_lookup', 'stock_lookup'],
                },
                'translation': {
                    'keywords': "['translate', 'translation', 'French', 'language']",
                    'tool_calls': ['search_tools'],
                },
                'no_matching_tool': {
                    'keywords': "['book flight', 'flight booking', 'airline reservation', 'travel booking']",
                    'tool_calls': ['search_tools'],
                },
            }
        ),
    ),
    ModelCase(
        model_name='google:gemini-3-flash-preview',
        marks=[pytest.mark.skipif(not google_available(), reason='google-genai not installed')],
        scenario_summary=snapshot(
            {
                'exchange_rate': {
                    'keywords': "['exchange rate', 'currency conversion']",
                    'tool_calls': ['search_tools', 'get_exchange_rate'],
                },
                'stock_price': {
                    'keywords': "['stock price', 'financial data', 'market data']",
                    'tool_calls': ['search_tools', 'stock_lookup'],
                },
                'translation': {'keywords': None, 'tool_calls': []},
                'no_matching_tool': {
                    'keywords': "['flight booking', 'search flights', 'book flight']",
                    'tool_calls': ['search_tools', 'search_tools'],
                },
            }
        ),
    ),
]


@pytest.mark.skipif(not evals_available(), reason='pydantic-evals not installed')
@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='dynamic tool disclosure after ToolSearch discovery')
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


class NoNativeToolSearchModel(FunctionModel):
    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        return frozenset(super().supported_native_tools()) - {ToolSearchTool}


def _build_run_context(
    deps: T,
    run_step: int = 0,
    messages: list[ModelMessage] | None = None,
    capabilities: dict[str, AbstractCapability[T]] | None = None,
    discovered_tool_names: set[str] | None = None,
    max_retries: int = 0,
) -> RunContext[T]:
    """Build a `RunContext` for unit tests using `TestModel`."""
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=messages or [],
        run_step=run_step,
        capabilities=capabilities or {},
        discovered_tool_names=discovered_tool_names or set(),
        max_retries=max_retries,
    )


def _create_function_toolset() -> FunctionToolset:
    toolset: FunctionToolset = FunctionToolset()

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
    visible tools and the `search_tools` function are exposed up front."""
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
        'Search first for a standalone deferred tool when current tools and catalog descriptions do not name the requested operation. A capability id used as an ordinary domain word does not request that capability. This cannot find capability-owned tools; load a listed capability by id instead. If no tools are found, do not retry.'
    )
    assert search_tool.tool_def.parameters_json_schema == snapshot(
        {
            'additionalProperties': False,
            'properties': {
                'queries': {
                    'description': 'List of search queries to match against tool names and descriptions. Use specific words likely to appear in tool names or descriptions to narrow down relevant tools. Each query is independently tokenized; matches across queries are unioned.',
                    'items': {'type': 'string'},
                    'type': 'array',
                }
            },
            'required': ['queries'],
            'type': 'object',
        }
    )


@pytest.mark.parametrize('tool_retries', [1, 3])
async def test_search_tools_inherits_agent_tool_retries(tool_retries: int):
    """`search_tools` honors `Agent(retries={'tools': N})` instead of a hardcoded budget.

    Driven through the public API because the budget was previously pinned inside
    `ToolSearchToolset`, where no user could reach it. The model keeps sending `queries` as
    a bare string, so every call fails validation: `N` retries after the first attempt means
    `N + 1` invocations before `UnexpectedModelBehavior`.
    """
    calls = 0

    def malformed_queries(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[ToolCallPart(_SEARCH_TOOLS_NAME, {'queries': 'weather'})])

    agent = Agent(
        NoNativeToolSearchModel(malformed_queries),
        toolsets=[_create_function_toolset()],
        capabilities=[ToolSearch()],
        retries={'tools': tool_retries},
    )

    with pytest.raises(UnexpectedModelBehavior, match=f'exceeded max retries count of {tool_retries}'):
        await agent.run('find me a weather tool')

    assert calls == tool_retries + 1


async def test_tool_search_toolset_max_retries_overrides_agent_budget():
    """An explicit `max_retries` on the toolset wins over the agent's tool retry budget.

    Asserted at `get_tools` rather than through an agent run: the capability that owns tool
    search is auto-injected, so a hand-constructed `ToolSearchToolset` can't be handed to an
    `Agent` without the outer wrapper tripping the reserved-`search_tools` guard.
    """
    toolset = _create_function_toolset()
    ctx = _build_run_context(None, max_retries=5)

    inheriting = await ToolSearchToolset(wrapped=toolset).get_tools(ctx)
    overriding = await ToolSearchToolset(wrapped=toolset, max_retries=2).get_tools(ctx)

    assert inheriting[_SEARCH_TOOLS_NAME].max_retries == 5
    assert overriding[_SEARCH_TOOLS_NAME].max_retries == 2


async def test_tool_search_toolset_search_returns_matching_tools():
    """Test that search_tools returns matching deferred tools."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['mortgage']}, ctx, search_tool)
    assert result == snapshot({'discovered_tools': [{'name': 'calculate_mortgage'}]})


async def test_tool_search_toolset_search_is_case_insensitive():
    """Test that search is case insensitive."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['STOCK']}, ctx, search_tool)
    rv = cast(ToolSearchReturnContent, result)
    assert len(rv['discovered_tools']) == 1
    assert rv['discovered_tools'][0]['name'] == 'stock_price'


async def test_tool_search_toolset_search_matches_description():
    """Test that search matches tool descriptions."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['cryptocurrency']}, ctx, search_tool)
    rv = cast(ToolSearchReturnContent, result)
    assert len(rv['discovered_tools']) == 1
    assert rv['discovered_tools'][0]['name'] == 'crypto_price'


async def test_tool_search_toolset_prefers_specific_term_matches():
    toolset: FunctionToolset = FunctionToolset()

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

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['github profile']}, ctx, search_tool)
    assert result == snapshot(
        {
            'discovered_tools': [
                {'name': 'github_get_me'},
                {'name': 'github_create_gist'},
            ]
        }
    )


async def test_tool_search_toolset_keeps_lower_scoring_matches_after_top_hits():
    toolset: FunctionToolset = FunctionToolset()

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

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['stock price']}, ctx, search_tool)
    assert result == snapshot(
        {
            'discovered_tools': [
                {'name': 'stock_price'},
                {'name': 'crypto_price'},
            ]
        }
    )


async def test_tool_search_toolset_does_not_match_substrings_inside_words():
    toolset: FunctionToolset = FunctionToolset()

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

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['get me']}, ctx, search_tool)
    assert result == snapshot({'discovered_tools': [{'name': 'github_get_me'}]})


async def test_tool_search_toolset_search_returns_no_matches():
    """Test that search returns empty list when no matches."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['nonexistent']}, ctx, search_tool)
    assert result == snapshot(
        {'discovered_tools': [], 'message': 'No matching tools found. The tools you need may not be available.'}
    )


async def test_tool_search_toolset_search_empty_query():
    """Test that search with empty query raises ModelRetry."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    with pytest.raises(ModelRetry, match=re.escape('Please provide at least one non-empty search query.')):
        await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['']}, ctx, search_tool)


@pytest.mark.parametrize('query', ['   ', '---', '!!!', '...'])
async def test_tool_search_toolset_search_non_tokenizable_query(query: str):
    """Queries that tokenize to an empty set must retry, not match every tool."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    with pytest.raises(ModelRetry, match=re.escape('Please provide at least one non-empty search query.')):
        await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': [query]}, ctx, search_tool)


async def test_tool_search_toolset_max_results():
    """Test that results are capped at `_MAX_SEARCH_RESULTS` (10)."""
    toolset: FunctionToolset = FunctionToolset()

    for i in range(15):

        @toolset.tool_plain(defer_loading=True, name=f'tool_{i}')
        def tool_func() -> str:  # pragma: no cover
            """A tool for testing."""
            return 'result'

    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['tool']}, ctx, search_tool)
    rv = cast(ToolSearchReturnContent, result)
    assert len(rv['discovered_tools']) == 10


async def test_tool_search_toolset_ranks_undiscovered_matches_first_when_trimmed() -> None:
    """An already-available match can never displace an undiscovered one when `max_results`
    trims: undiscovered-first is the primary sort key, relevance the tiebreak, so discovered
    tools only fill leftover slots."""
    toolset = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def first_tool() -> str:  # pragma: no cover
        return 'first'

    @toolset.tool_plain(defer_loading=True)
    def second_tool() -> str:  # pragma: no cover
        return 'second'

    searchable = ToolSearchToolset(wrapped=toolset, max_results=1)
    ctx = _build_run_context(None, discovered_tool_names={'first_tool'})
    search_tool = (await searchable.get_tools(ctx))[_SEARCH_TOOLS_NAME]

    # `first_tool` scores higher (matches both terms) but is already discovered — the
    # lower-scoring undiscovered `second_tool` still takes the single slot.
    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['first', 'tool']}, ctx, search_tool)

    assert result == {'discovered_tools': [{'name': 'second_tool'}]}

    # With room for both, the discovered tool is appended after the undiscovered one rather
    # than excluded — the corpus never shrinks with discovery.
    searchable = ToolSearchToolset(wrapped=toolset, max_results=2)
    search_tool = (await searchable.get_tools(ctx))[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['first', 'tool']}, ctx, search_tool)

    assert result == {'discovered_tools': [{'name': 'second_tool'}, {'name': 'first_tool'}]}


async def test_repeated_searches_paginate_through_a_large_corpus() -> None:
    """Repeating the same query enumerates a corpus larger than `max_results`.

    Each page's results become discovered and sink below undiscovered matches, so the next
    identical search surfaces the next tranche — preserving the scan-by-repetition idiom the
    old corpus subtraction enabled. A page that includes already-available tools is the
    signal that enumeration is complete."""
    toolset = FunctionToolset()
    all_names = [f'mcp_tool_{i:02d}' for i in range(25)]
    for tool_name in all_names:

        def tool(name: str = tool_name) -> str:  # pragma: no cover
            return name

        toolset.add_function(tool, name=tool_name, defer_loading=True)

    searchable = ToolSearchToolset(wrapped=toolset)
    discovered: set[str] = set()
    pages: list[list[str]] = []
    for _ in range(3):
        ctx = _build_run_context(None, discovered_tool_names=set(discovered))
        search_tool = (await searchable.get_tools(ctx))[_SEARCH_TOOLS_NAME]
        result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['mcp']}, ctx, search_tool)
        page = [match['name'] for match in result['discovered_tools']]
        pages.append(page)
        discovered.update(page)

    assert len(pages[0]) == len(pages[1]) == 10
    assert not set(pages[0]) & set(pages[1])
    # The final page leads with the 5 still-undiscovered tools; already-available ones
    # fill the leftover slots — the model's signal that it has seen the whole corpus.
    assert set(pages[0]) | set(pages[1]) | set(pages[2][:5]) == set(all_names)
    assert set(pages[2][5:]) <= set(pages[0]) | set(pages[1])


async def test_search_corpus_includes_already_discovered_tools() -> None:
    """The corpus a custom `search_fn` receives never shrinks with discovery: an
    already-discovered tool stays searchable with no compaction boundary in sight."""
    toolset = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def first_tool() -> str:  # pragma: no cover
        return 'first'

    @toolset.tool_plain(defer_loading=True)
    def second_tool() -> str:  # pragma: no cover
        return 'second'

    corpora: list[list[str]] = []

    def search(_ctx: RunContext[None], _queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
        corpora.append(sorted(tool.name for tool in tools))
        return ['first_tool']

    searchable = ToolSearchToolset(wrapped=toolset, search_fn=search)
    ctx = _build_run_context(None, discovered_tool_names={'first_tool'})
    search_tool = (await searchable.get_tools(ctx))[_SEARCH_TOOLS_NAME]

    await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['first']}, ctx, search_tool)

    assert corpora == [['first_tool', 'second_tool']]


async def test_stripped_reveal_exchange_heals_via_re_search() -> None:
    """A history processor that strips a reveal exchange must not strand the revealed tool (#7259).

    The search corpus never subtracts discovered names, so when the processor deletes the first
    search exchange the model can simply search again; the new exchange survives, re-reveals the
    tool, and the run completes. (A corpus that subtracted discovered names left the tool
    simultaneously withheld on the wire and unsearchable — permanently unavailable.)
    """

    class SearchTwiceThenCallModel(TestModel):
        request_number = 0

        def _request(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            type(self).request_number += 1
            if self.request_number <= 2:
                # Step 1 discovers the tool; the processor then strips that exchange, so step 2's
                # history carries no evidence and the model searches again.
                assert model_request_parameters.visibility_of('secret_lookup') == 'withheld'
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=_SEARCH_TOOLS_NAME,
                            args={'queries': ['secret lookup']},
                            tool_call_id=f'search-{self.request_number}',
                        )
                    ]
                )
            # The second exchange survived: the tool is revealed again and callable.
            assert model_request_parameters.visibility_of('secret_lookup') == 'visible'
            if self.request_number == 3:
                return ModelResponse(parts=[ToolCallPart(tool_name='secret_lookup', args={}, tool_call_id='lookup-1')])
            return super()._request(messages, model_settings, model_request_parameters)

    def strip_first_search_exchange(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [
            replace(
                message,
                parts=[
                    part
                    for part in message.parts
                    if not (
                        isinstance(part, ToolCallPart | ToolSearchCallPart | ToolReturnPart | ToolSearchReturnPart)
                        and part.tool_call_id == 'search-1'
                    )
                ],
            )
            for message in messages
        ]

    agent = Agent(SearchTwiceThenCallModel(), capabilities=[ProcessHistory(strip_first_search_exchange)])

    @agent.tool_plain(defer_loading=True)
    def secret_lookup() -> str:
        return 'SECRET'

    result = await agent.run('find and call the secret lookup tool')

    assert 'SECRET' in str(result.output)


async def test_tool_search_toolset_discovered_tools_keep_defer_loading():
    """Discovery does not overwrite the tools' authored `defer_loading=True` value."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage'}]},
                ),
            ]
        )
    ]
    ctx = _build_run_context(None, messages=messages, discovered_tool_names={'calculate_mortgage'})

    tools = await searchable.get_tools(ctx)
    assert tools['calculate_mortgage'].tool_def.defer_loading is True
    assert tools['stock_price'].tool_def.defer_loading is True
    assert tools['crypto_price'].tool_def.defer_loading is True


async def test_tool_search_toolset_keeps_search_tool_after_all_discovered():
    """`search_tools` stays in the request even when every deferred tool is discovered.

    Dropping it would invalidate the cached request prefix on the next turn — keeping
    it preserves prompt caching across discovery steps. The local tool's body is a no-op
    branch in `_search_tools` since the index is empty, and on native paths it's dropped
    by the adapter via its `unless_native='tool_search'` flag anyway.
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
                            {'name': 'calculate_mortgage'},
                            {'name': 'stock_price'},
                            {'name': 'crypto_price'},
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
    toolset: FunctionToolset = FunctionToolset()

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
    toolset: FunctionToolset = FunctionToolset()

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


async def test_tool_search_handles_search_gated_tools_from_eager_capability():
    """Search-gated tools from eager capabilities stay hidden until searched."""
    toolset: FunctionToolset = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def capability_search_tool() -> str:
        """Search-gated tool owned by an eager capability."""
        return 'search-gated-result'

    capability = Capability(
        id='example',
        description='Example capability.',
        toolsets=[toolset],
    )

    seen_tool_names: list[list[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tool_names.append([t.name for t in info.function_tools])
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))

        if not any(part.tool_name == _SEARCH_TOOLS_NAME for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=_SEARCH_TOOLS_NAME,
                        args={'queries': ['eager']},
                        tool_call_id='search-1',
                    )
                ]
            )

        if not any(part.tool_name == 'capability_search_tool' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='capability_search_tool',
                        args={},
                        tool_call_id='call-search-gated',
                    )
                ]
            )

        gated_result = next(part.content for part in tool_returns if part.tool_name == 'capability_search_tool')
        return ModelResponse(parts=[TextPart(content=f'final: {gated_result}')])

    agent = Agent(NoNativeToolSearchModel(model_fn), capabilities=[capability])

    result = await agent.run('find the gated tool')

    assert result.output == 'final: search-gated-result'
    assert seen_tool_names == snapshot(
        [
            ['search_tools'],
            ['capability_search_tool', 'search_tools'],
            ['capability_search_tool', 'search_tools'],
        ]
    )


async def test_tool_search_handles_capability_deferred_and_loaded_tools():
    """Deferred capability tools become visible as a unit after loading."""
    toolset: FunctionToolset = FunctionToolset()

    @toolset.tool_plain
    def inherited_tool() -> str:  # pragma: no cover
        """Tool inheriting cap-level defer."""
        return 'inherited-result'

    @toolset.tool_plain(defer_loading=True)
    def also_deferred_tool() -> str:
        """Tool with its own defer flag — shadowed by cap-level defer."""
        return 'also-deferred-result'

    capability = Capability(
        id='example',
        description='Example capability.',
        defer_loading=True,
        toolsets=[toolset],
    )

    seen_tool_names: list[list[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tool_names.append([t.name for t in info.function_tools])
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))

        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'example'},
                        tool_call_id='load-example',
                    )
                ]
            )

        if not any(part.tool_name == 'also_deferred_tool' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='also_deferred_tool',
                        args={},
                        tool_call_id='call-deferred',
                    )
                ]
            )

        gated_result = next(part.content for part in tool_returns if part.tool_name == 'also_deferred_tool')
        return ModelResponse(parts=[TextPart(content=f'final: {gated_result}')])

    agent = Agent(NoNativeToolSearchModel(model_fn), capabilities=[capability])

    result = await agent.run('use the special tool')

    assert result.output == 'final: also-deferred-result'
    assert seen_tool_names == snapshot(
        [
            ['load_capability'],
            ['load_capability', 'inherited_tool', 'also_deferred_tool'],
            ['load_capability', 'inherited_tool', 'also_deferred_tool'],
        ]
    )


async def test_explicit_tool_search_offers_no_search_surface_for_a_capability_only_corpus():
    """Capability-gated tools are never searchable, so no search surface is offered at all.

    Loading the capability is the only way to reach its tools, before or after — a search over them
    could only ever answer "no matches". So an explicitly configured strategy is left with nothing
    to index and `search_tools` never reaches the model: on a model with no native tool search it
    would otherwise sit in every request, spending a tool slot and cache-prefix bytes to say nothing.
    The capability's tool still shows up on the turn after the load, without any search in between.
    """
    toolset: FunctionToolset = FunctionToolset()

    @toolset.tool_plain
    def capability_tool() -> str:  # pragma: no cover
        """A tool owned by the deferred capability."""
        return 'capability-result'

    capability = Capability(
        id='example',
        description='Example capability.',
        defer_loading=True,
        toolsets=[toolset],
    )

    def search_strategy(
        ctx: RunContext[object], queries: Sequence[str], tool_defs: Sequence[ToolDefinition]
    ) -> list[str]:  # pragma: no cover
        raise AssertionError('nothing is searchable, so the strategy must never run')

    seen_tool_names: list[list[str]] = []
    request_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal request_count
        request_count += 1
        seen_tool_names.append([t.name for t in info.function_tools])
        if request_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'example'},
                        tool_call_id='load-example',
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content='done')])

    agent: Agent[object, str] = Agent(
        NoNativeToolSearchModel(model_fn),
        capabilities=[capability, ToolSearch(strategy=search_strategy)],
    )

    result = await agent.run('load the capability')

    assert result.output == 'done'
    assert seen_tool_names == snapshot(
        [
            ['load_capability'],
            ['load_capability', 'capability_tool'],
        ]
    )


async def test_tool_search_ignores_malformed_loaded_capability_history():
    """Malformed `load_capability` results must not unlock capability-owned tools."""
    toolset: FunctionToolset = FunctionToolset()

    @toolset.tool_plain
    def inherited_tool() -> str:  # pragma: no cover
        """Inherited deferred tool."""
        return 'inherited'

    capability = Capability(
        id='reports',
        description='Report tools.',
        defer_loading=True,
        toolsets=[toolset],
    )
    cap_toolset = CombinedCapability([capability]).get_toolset()
    assert isinstance(cap_toolset, AbstractToolset)
    cap_toolset = cast(AbstractToolset, cap_toolset)
    searchable: ToolSearchToolset = ToolSearchToolset(wrapped=cap_toolset)

    cases: dict[str, object] = {
        'not_a_dict': 'loaded reports',
        'non_string_instructions': {'instructions': ['bad']},
    }
    tool_defer_state: dict[str, list[tuple[str, bool]]] = {}

    for case_name, content in cases.items():
        messages: list[ModelMessage] = [
            ModelRequest(parts=[ToolReturnPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, content=content)])
        ]
        ctx = _build_run_context(None, messages=messages, capabilities={'reports': capability})
        tools = await searchable.get_tools(ctx)
        tool_defer_state[case_name] = [(name, bool(t.tool_def.defer_loading)) for name, t in tools.items()]

    assert tool_defer_state == snapshot(
        {
            'not_a_dict': [('inherited_tool', True)],
            'non_string_instructions': [('inherited_tool', True)],
        }
    )


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

    assert ToolSearch.get_serialization_name() == 'ToolSearch'
    assert CAPABILITY_TYPES['ToolSearch'] is ToolSearch


async def test_tool_manager_with_tool_search_toolset_marks_corpus():
    """Every deferred tool appears once under its real name with
    `with_native='tool_search'`. Visible tools and `search_tools` round
    out the dispatch dict. `Model.prepare_request` filters per-model to decide what
    actually reaches the wire."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tool_manager = ToolManager(searchable)
    run_step_toolset = await tool_manager.for_run_step(ctx)

    managed_names = {t.name for t in run_step_toolset.tool_defs if t.with_native == 'tool_search'}
    assert managed_names == {'calculate_mortgage', 'stock_price', 'crypto_price'}

    local_names = [t.name for t in run_step_toolset.tool_defs if not t.with_native]
    assert 'get_weather' in local_names
    assert 'search_tools' in local_names

    # An undiscovered deferred tool is in the dispatch dict but not callable: `ToolManager`
    # gates on availability, so the model is told to search rather than that it doesn't exist.
    with pytest.raises(ToolRetryError, match='is not available yet'):
        await run_step_toolset.handle_call(
            ToolCallPart(tool_name='calculate_mortgage', args={'principal': 100.0, 'rate': 5.0, 'years': 30})
        )

    # The local search_tools function is also dispatchable.
    result = await run_step_toolset.handle_call(ToolCallPart(tool_name='search_tools', args={'queries': ['mortgage']}))
    assert 'calculate_mortgage' in str(result)


async def test_tool_search_toolset_tool_with_none_description():
    """Test that tools with None description are handled correctly in search."""
    toolset: FunctionToolset = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def no_desc_tool() -> str:  # pragma: no cover
        return 'no description'

    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]

    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['no_desc']}, ctx, search_tool)
    assert result == snapshot({'discovered_tools': [{'name': 'no_desc_tool'}]})


async def test_tool_search_toolset_multiple_searches_accumulate():
    """Discovery accumulates without changing the tools' authored deferred state."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage'}]},
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'stock_price'}]},
                ),
            ]
        ),
    ]
    ctx = _build_run_context(None, messages=messages, discovered_tool_names={'calculate_mortgage', 'stock_price'})

    tools = await searchable.get_tools(ctx)
    assert tools['calculate_mortgage'].tool_def.defer_loading is True
    assert tools['stock_price'].tool_def.defer_loading is True
    assert tools['crypto_price'].tool_def.defer_loading is True


async def test_function_toolset_all_deferred():
    """Test FunctionToolset with all tools having defer_loading=True."""
    toolset: FunctionToolset = FunctionToolset()

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


async def test_tool_search_toolset_reads_legacy_metadata_discovered_tools():
    """Pre-typed-content versions of this toolset wrote discovered tool names to
    `ToolReturnPart.metadata['discovered_tools']` instead of the typed
    `ToolSearchReturn` on `content`. Persisted histories from those versions
    must still surface their discoveries on resume; otherwise an agent reloaded from
    a saved transcript would re-emit `search_tools` and the user would see a
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


def test_parse_discovered_tools_reads_legacy_metadata():
    """`parse_discovered_tools` reads the pre-typed-content legacy sideband directly.

    `get_tools` now consults `ctx.discovered_tool_names` rather than re-parsing history, so the
    legacy `metadata['discovered_tools']` reader is exercised through this classmethod (which
    run preparation calls to populate `discovered_tool_names`). A valid legacy list surfaces its
    names; a malformed one (wrong shape) is silently skipped via the `ValidationError` guard."""
    valid: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content='legacy text return',
                    metadata={'discovered_tools': ['stock_price', 'crypto_price']},
                ),
            ]
        ),
    ]
    assert parse_discovered_tools(valid) == {'stock_price', 'crypto_price'}

    # Malformed legacy metadata (`discovered_tools` is not a `list[str]`) trips
    # `_LEGACY_METADATA_TA.validate_python`, so `_collect_legacy` hits its `except: return`.
    malformed: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=_SEARCH_TOOLS_NAME,
                    content='another',
                    metadata={'discovered_tools': 'not a list'},
                ),
            ]
        ),
    ]
    assert parse_discovered_tools(malformed) == set()


async def test_run_context_seeds_discovered_tool_names_from_history_before_first_step():
    """Pre-first-step hooks see tool-search discoveries replayed from message history."""
    seen_discovered_tool_names: list[set[str]] = []

    @dataclass
    class ObserveDiscoveredTools(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            seen_discovered_tool_names.append(set(ctx.discovered_tool_names))

    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage'}]},
                ),
            ]
        )
    ]
    agent = Agent(TestModel(), capabilities=[ObserveDiscoveredTools()])

    await agent.run('hello', message_history=history)

    assert seen_discovered_tool_names == [{'calculate_mortgage'}]


async def test_deferred_loading_toolset_marks_all_tools():
    """`DeferredLoadingToolset` (with `tool_names=None`) flips `defer_loading=True`
    on every tool. After wrapping with `ToolSearchToolset`, all of them appear under
    their real name with `defer_loading=True` (visibility hidden until discovered).
    `search_tools` is the only directly-callable tool up front."""
    toolset: FunctionToolset = FunctionToolset()

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
    """`DeferredLoadingToolset` with explicit names only flips `defer_loading=True`
    on the listed tools; others stay visible."""
    toolset: FunctionToolset = FunctionToolset()

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


async def test_tool_search_toolset_marks_corpus_with_native():
    """Every deferred tool keeps its real name in the toolset output and carries
    `with_native='tool_search'` regardless of the current model — the adapter's
    `prepare_request` decides what reaches the wire so the toolset can't commit early
    (e.g. under `FallbackModel`)."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)

    managed = {name: tool.tool_def for name, tool in tools.items() if tool.tool_def.with_native}
    assert set(managed) == {'calculate_mortgage', 'stock_price', 'crypto_price'}
    for tool_def in managed.values():
        assert tool_def.with_native == 'tool_search'
        assert tool_def.defer_loading
    # The local fallback is still present — dropped by the adapter via `unless_native`.
    assert _SEARCH_TOOLS_NAME in tools


async def test_tool_search_toolset_dispatches_by_plain_name_via_tool_manager():
    """Once discovered, the provider calls a deferred tool by its plain name and `ToolManager`
    dispatches directly via the dict key (also the plain name)."""
    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)
    ctx = _build_run_context(None, discovered_tool_names={'calculate_mortgage'})

    tool_manager = ToolManager(searchable)
    run_step_toolset = await tool_manager.for_run_step(ctx)
    result = await run_step_toolset.handle_call(
        ToolCallPart(tool_name='calculate_mortgage', args={'principal': 100.0, 'rate': 5.0, 'years': 30})
    )
    assert 'Mortgage calculated' in str(result)


async def test_tool_search_toolset_custom_search_fn_is_used():
    """A custom `search_fn` replaces the default keyword-matching algorithm."""
    calls: list[Sequence[str]] = []

    def custom_search(ctx: RunContext, queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
        calls.append(queries)
        # Pick anything with 'price' in the name, regardless of query tokens.
        return [t.name for t in tools if 'price' in t.name]

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset, search_fn=custom_search)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['anything']}, ctx, tools[_SEARCH_TOOLS_NAME])
    assert result == {
        'discovered_tools': [
            {'name': 'stock_price'},
            {'name': 'crypto_price'},
        ]
    }
    assert calls == [['anything']]


async def test_tool_search_toolset_custom_search_fn_still_marks_corpus():
    """A custom `search_fn` handles local discovery, but the toolset still flags every
    deferred tool with `with_native='tool_search'` — when the model supports
    native tool search (including provider-side custom callable modes like Anthropic's
    tool_reference mechanism or OpenAI's `execution='client'`), the adapter keeps them
    and applies `defer_loading` on the wire. Commitment to native-vs-local happens in
    `Model.prepare_request`, not here."""

    def custom_search(
        ctx: RunContext, queries: Sequence[str], tools: Sequence[ToolDefinition]
    ) -> list[str]:  # pragma: no cover
        return []

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset, search_fn=custom_search)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)

    managed = [t.tool_def.name for t in tools.values() if t.tool_def.with_native == 'tool_search']
    assert set(managed) == {'calculate_mortgage', 'stock_price', 'crypto_price'}
    assert _SEARCH_TOOLS_NAME in tools


@pytest.mark.vcr
async def test_anthropic_native_tool_search_round_trip(allow_model_requests: None, anthropic_api_key: str) -> None:
    """End-to-end against live Anthropic: native BM25 server-side tool search
    populates `NativeToolCallPart` / `NativeToolReturnPart`, the model invokes
    the discovered deferred tool by its plain name, and the wire request carries
    `defer_loading: true` on the corpus tools and the `tool_search_tool_bm25`
    builtin.
    """
    pytest.importorskip('anthropic')

    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model=model)

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
    # `NativeToolSearchCallPart` / `NativeToolSearchReturnPart` subclasses
    # (which still `isinstance`-match the base `NativeToolCallPart` /
    # `NativeToolReturnPart`).
    builtin_call_parts = list(iter_message_parts(result.all_messages(), ModelResponse, NativeToolCallPart))
    builtin_return_parts = list(iter_message_parts(result.all_messages(), ModelResponse, NativeToolReturnPart))
    assert builtin_call_parts and builtin_return_parts

    # The model's follow-up tool call for the discovered tool dispatches by its plain
    # name — the toolset exposes deferred tools as their regular variant on the native
    # path so the dispatch doesn't fall through to an "unknown tool" retry.
    rate_returns = [
        p
        for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if p.tool_name == 'get_exchange_rate'
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
    """End-to-end: a custom callable `ToolSearch` strategy runs locally but still
    surfaces natively on Anthropic — deferred tools ship with `defer_loading: true`,
    the model invokes the regular `search_tools` function tool, and our
    `tool_result` is formatted as `tool_reference` blocks so the discovered tool
    gets unlocked for the next turn."""
    pytest.importorskip('anthropic')

    def match_exchange_rate(ctx: RunContext, queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
        # Deterministic: always point the model at `get_exchange_rate` so the cassette
        # replay doesn't depend on the exact keywords the model picks.
        return ['get_exchange_rate']

    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(
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
    assert part_shape == snapshot(part_shape)

    # The deferred tool dispatched successfully end-to-end.
    rate_returns = [
        part
        for part in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if part.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    # Wire-level checks against the cassette: the deferred corpus ships with
    # `defer_loading: true`, the model's `search_tools` call appears in the response,
    # and our tool result is formatted as `tool_reference` blocks (not plain text).

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
async def test_anthropic_promotes_local_search_history_round_trip(
    allow_model_requests: None, anthropic_api_key: str
) -> None:
    """End-to-end against live Anthropic: a turn with local-shape `ToolSearch*Part`
    history (from a prior cross-provider turn — e.g. on Google) runs cleanly on
    Anthropic. The adapter promotes the local-shape return into a `tool_result` with
    `tool_reference` content so Anthropic unlocks the discovered tool's schema, and
    the model dispatches the discovered tool directly without issuing a fresh
    `tool_search_tool_*` call.
    """
    pytest.importorskip('anthropic')

    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model=model, capabilities=[ToolSearch()])

    @agent.tool_plain(defer_loading=True)
    def get_exchange_rate(from_currency: str, to_currency: str) -> str:
        """Look up the current exchange rate between two currencies."""
        return f'1 {from_currency} = 0.92 {to_currency}'

    @agent.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:  # pragma: no cover
        """Look up stock price by ticker symbol."""
        return f'Stock {symbol}: $150.00'

    # Synthetic history: a prior turn on a non-supporting provider (Google etc.)
    # discovered `get_exchange_rate` via the local `search_tools` function tool.
    # Carries the local-shape typed parts on a `ToolSearchReturnPart` (sub of
    # `ToolReturnPart`) — exactly what the toolset would emit on the local path.
    prior_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='I might want to look up exchange rates later.')]),
        ModelResponse(
            parts=[
                ToolSearchCallPart(args={'queries': ['exchange rate']}, tool_call_id='loc_search_1'),
            ],
        ),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_exchange_rate'}]},
                    tool_call_id='loc_search_1',
                ),
            ],
        ),
    ]

    result = await agent.run('What is the USD to EUR exchange rate?', message_history=prior_history)

    # The model uses the discovered tool directly — no fresh `tool_search_tool_*` call
    # was needed because the prior local-shape return got promoted to native shape on
    # the wire, unlocking `get_exchange_rate` server-side.
    rate_returns = [
        part
        for part in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if part.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    # No fresh native tool_search exchange after the synthetic history.
    fresh_native_search_calls = [
        part for msg in result.all_messages() for part in msg.parts if isinstance(part, NativeToolSearchCallPart)
    ]
    assert fresh_native_search_calls == []

    # Wire-level: cassette confirms the request to Anthropic carried the prior
    # local-shape return as a `tool_result` with `tool_reference` content (NOT a
    # stringified JSON of the discoveries).
    cassette_path = (
        Path(__file__).parent
        / 'cassettes'
        / 'test_tool_search'
        / 'test_anthropic_promotes_local_search_history_round_trip.yaml'
    )
    cassette = cast(dict[str, Any], yaml.safe_load(cassette_path.read_text(encoding='utf-8')))
    interactions = cast(list[dict[str, Any]], cassette['interactions'])

    first_request_messages = cast(list[dict[str, Any]], interactions[0]['request']['parsed_body']['messages'])
    tool_result_contents: list[Any] = [
        block.get('content')
        for msg in first_request_messages
        if msg.get('role') == 'user' and isinstance(msg.get('content'), list)
        for block in cast(list[dict[str, Any]], msg['content'])
        if isinstance(block, dict) and block.get('type') == 'tool_result'
    ]
    # The `tool_reference` array shape proves the promotion fired.
    promoted_names = {
        cast(str, inner.get('tool_name'))
        for content in tool_result_contents
        if isinstance(content, list)
        for inner in cast(list[dict[str, Any]], content)
        if isinstance(inner, dict) and inner.get('type') == 'tool_reference'
    }
    assert promoted_names == {'get_exchange_rate'}


@pytest.mark.vcr
async def test_openai_promotes_local_search_history_round_trip(
    allow_model_requests: None, openai_api_key: str, vcr: Any
) -> None:
    """End-to-end against live OpenAI: a turn with local-shape `ToolSearch*Part`
    history runs cleanly on OpenAI Responses. The adapter promotes the local-shape
    pair into `tool_search_call` + `tool_search_output` items with
    `execution='client'`, and the model dispatches the discovered tool directly.
    """
    pytest.importorskip('openai')

    model = OpenAIResponsesModel('gpt-5.4-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, capabilities=[ToolSearch()])

    @agent.tool_plain(defer_loading=True)
    def get_exchange_rate(from_currency: str, to_currency: str) -> str:
        """Look up the current exchange rate between two currencies."""
        return f'1 {from_currency} = 0.92 {to_currency}'

    @agent.tool_plain(defer_loading=True)
    def stock_lookup(symbol: str) -> str:  # pragma: no cover
        """Look up stock price by ticker symbol."""
        return f'Stock {symbol}: $150.00'

    prior_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='I might want to look up exchange rates later.')]),
        ModelResponse(
            parts=[
                ToolSearchCallPart(args={'queries': ['exchange rate']}, tool_call_id='loc_search_1'),
            ],
        ),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_exchange_rate'}]},
                    tool_call_id='loc_search_1',
                ),
            ],
        ),
    ]

    result = await agent.run('What is the USD to EUR exchange rate?', message_history=prior_history)

    rate_returns = [
        part
        for part in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if part.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    request_inputs = [json.loads(request.body).get('input', []) for request in vcr.requests]

    # Wire-level: cassette confirms the local-shape pair got promoted to
    # `tool_search_call` + `tool_search_output` items with `execution='client'`.
    promoted_calls = [item for item in request_inputs[0] if item.get('type') == 'tool_search_call']
    promoted_outputs = [item for item in request_inputs[0] if item.get('type') == 'tool_search_output']
    assert promoted_calls, 'expected the local-shape call to be promoted to tool_search_call'
    assert promoted_outputs, 'expected the local-shape return to be promoted to tool_search_output'
    assert all(item.get('execution') == 'client' for item in promoted_calls)
    assert all(item.get('execution') == 'client' for item in promoted_outputs)
    promoted_tool_names = {t.get('name') for output in promoted_outputs for t in output.get('tools', [])}
    assert 'get_exchange_rate' in promoted_tool_names

    # Wire-level: once the discovered tool is dispatched, its replayed `function_call` must carry a
    # `namespace` — OpenAI rejects a tool-search-discovered call without one. Guards the namespace
    # emission in `_map_messages` (a regression would only surface as a live 422 otherwise).
    replayed_calls = [
        item
        for request_input in request_inputs
        for item in request_input
        if item.get('type') == 'function_call' and item.get('name') == 'get_exchange_rate'
    ]
    assert replayed_calls, 'expected the discovered tool call to be replayed to OpenAI'
    assert any(item.get('namespace') == 'get_exchange_rate' for item in replayed_calls)


@pytest.mark.vcr
async def test_anthropic_native_tool_search_regex_strategy(allow_model_requests: None, anthropic_api_key: str) -> None:
    """`ToolSearch(strategy='regex')` registers the regex variant of Anthropic's
    native tool search tool rather than BM25, and the live API accepts the request.
    """
    pytest.importorskip('anthropic')

    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model=model, capabilities=[ToolSearch(strategy='regex')])

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
    downgrading `tool_search_tool_regex` to `tool_search_tool_bm25` on a resend would
    silently run a different algorithm than the earlier turn."""
    pytest.importorskip('anthropic')

    # Provider-side call used the regex variant; the adapter must round-trip that choice.
    # Anthropic's regex variant emits `pattern` (not `query`) in the wire input.
    regex_block = BetaServerToolUseBlock(
        id='srv_r',
        name='tool_search_tool_regex',
        input={'pattern': 'weather.*'},
        type='server_tool_use',
        caller=BetaDirectCaller(type='direct'),
    )
    call_part = _map_server_tool_use_block(regex_block, 'anthropic')
    assert isinstance(call_part, NativeToolCallPart)
    assert call_part.provider_details == {'strategy': 'regex'}
    # Cross-provider canonical shape collects the regex into the `queries` slot.
    assert call_part.args == snapshot({'queries': ['weather.*']})

    # On replay, the adapter should emit `tool_search_tool_regex` (not bm25).
    response = completion_message(
        [BetaTextBlock(text='done', type='text')],
        BetaUsage(input_tokens=5, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(response)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model=model, capabilities=[ToolSearch(strategy='regex')])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:  # pragma: no cover
        return f'Weather in {city}.'

    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('look it up'),
        ModelResponse(
            parts=[
                call_part,
                NativeToolSearchReturnPart(
                    provider_name='anthropic',
                    tool_call_id='srv_r',
                    content={'discovered_tools': [{'name': 'get_weather'}]},
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
    server_blocks = [block for block in blocks if block.get('type') == 'server_tool_use']
    names = [block['name'] for block in server_blocks]
    assert 'tool_search_tool_regex' in names
    assert 'tool_search_tool_bm25' not in names
    # Regex variant must replay with `pattern` (not `query`) — Anthropic 400s otherwise.
    regex_inputs = [block['input'] for block in server_blocks if block['name'] == 'tool_search_tool_regex']
    assert regex_inputs == snapshot([{'pattern': 'weather.*'}])


async def test_anthropic_cache_tool_definitions_skips_deferred_tools(allow_model_requests: None) -> None:
    """`anthropic_cache_tool_definitions=True` must apply `cache_control` to the last
    *non-deferred* tool. Anthropic rejects requests with `cache_control` and
    `defer_loading=True` on the same tool: `Tools with defer_loading cannot use prompt
    caching`. Reported by @kclisp on PR #5143.
    """
    pytest.importorskip('anthropic')

    response = completion_message(
        [BetaTextBlock(text='ok', type='text')],
        BetaUsage(input_tokens=5, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(response)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        model=model,
        capabilities=[ToolSearch()],
        model_settings=AnthropicModelSettings(anthropic_cache_tool_definitions=True),
    )

    @agent.tool_plain
    def visible_tool() -> str:  # pragma: no cover
        return 'visible'

    @agent.tool_plain(defer_loading=True)
    def deferred_tool() -> str:  # pragma: no cover
        return 'deferred'

    await agent.run('hi')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = cast('list[dict[str, Any]]', kwargs['tools'])
    by_name = {tool['name']: tool for tool in tools}
    # The deferred tool must NOT have `cache_control` — pairing it with `defer_loading`
    # is what Anthropic rejects.
    assert 'cache_control' not in by_name['deferred_tool']
    assert by_name['deferred_tool'].get('defer_loading') is True
    # The last non-deferred tool gets the cache breakpoint.
    assert by_name['visible_tool']['cache_control'] == snapshot({'type': 'ephemeral', 'ttl': '5m'})


async def test_anthropic_cache_tool_definitions_skips_when_all_tools_deferred(allow_model_requests: None) -> None:
    """When *every* tool is deferred, there's nothing in the cacheable prompt prefix to
    attach `cache_control` to. The loop must fall through without breaking — applying
    `cache_control` to any deferred tool would 400 the request.
    """
    pytest.importorskip('anthropic')

    response = completion_message(
        [BetaTextBlock(text='ok', type='text')],
        BetaUsage(input_tokens=5, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(response)
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(
        model=model,
        capabilities=[ToolSearch()],
        model_settings=AnthropicModelSettings(anthropic_cache_tool_definitions=True),
    )

    @agent.tool_plain(defer_loading=True)
    def deferred_one() -> str:  # pragma: no cover
        return 'one'

    @agent.tool_plain(defer_loading=True)
    def deferred_two() -> str:  # pragma: no cover
        return 'two'

    await agent.run('hi')

    kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = cast('list[dict[str, Any]]', kwargs['tools'])
    function_tools = [tool for tool in tools if 'input_schema' in tool]
    # No tool ends up with `cache_control` — pairing any deferred tool with it 400s.
    for tool in function_tools:
        assert 'cache_control' not in tool


async def test_openai_rejects_anthropic_named_strategy(allow_model_requests: None):
    """OpenAI Responses has no bm25/regex concept — using one must error loudly rather
    than silently falling through to OpenAI's default server-side tool search."""
    pytest.importorskip('openai')

    mock_client = MockOpenAIResponses.create_mock(response_message([]))
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model=model, capabilities=[ToolSearch(strategy='bm25')])

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

    call = ResponseToolSearchCall(
        id='ts_1',
        arguments={'queries': ['exchange rate']},
        call_id='call_1',
        execution='client',
        status='completed',
        type='tool_search_call',
    )
    part = _map_client_tool_search_call(call, 'azure')
    assert part.tool_name == _SEARCH_TOOLS_NAME
    # Provider name flows through from the model — important for OpenAI-compatible
    # providers (Azure, gateways) where `self.system` differs from `'openai'`.
    assert part.provider_name == 'azure'
    # No envelope marker any more: replay derives intent from the current request's
    # builtin configuration + a `provider_name` match against `self.system`.
    assert part.provider_details is None


async def test_openai_deferred_capability_reveal_sends_no_tool_search_surface(allow_model_requests: None):
    """A capability-gated corpus sends no `tool_search` tool and no `defer_loading` on OpenAI.

    Nothing here is searchable, so there's no search surface to send — and without one the Responses
    API won't take `defer_loading` either (`Invalid Value: 'tools.defer_loading'. Deferred tools
    require tools.tool_search.`, verified live on `gpt-5.6`). So the gated tool is simply not
    declared until it's revealed, and `tools` carries `load_capability` alone on the first turn.

    Pinned against an endpoint that doesn't implement `additional_tools`, which is where the reveal
    is still the synthesized tool-search exchange in history — a `search_tools` call naming a tool
    that isn't declared, which every provider measured accepts. First-party OpenAI models take the
    native item and keep `tools` byte-identical instead; they're covered by
    `tests/models/test_openai_tool_availability_delta.py`.
    """
    pytest.importorskip('openai')

    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:  # pragma: no cover
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed'

    capability = Capability(
        id='refunds',
        description='Refund policy tools.',
        instructions='Use the refund policy tool before answering refund questions.',
        defer_loading=True,
        toolsets=[refunds_toolset],
    )
    responses = [
        response_message(
            [
                ResponseFunctionToolCall(
                    id='fc_load',
                    arguments='{"id":"refunds"}',
                    call_id='call_load',
                    name=LOAD_CAPABILITY_TOOL_NAME,
                    status='completed',
                    type='function_call',
                )
            ]
        ),
        response_message(
            [
                ResponseOutputMessage(
                    id='msg_done',
                    content=[ResponseOutputText(text='Loaded.', type='output_text', annotations=[])],
                    role='assistant',
                    status='completed',
                    type='message',
                )
            ]
        ),
    ]
    mock_client = MockOpenAIResponses.create_mock(responses)
    model = OpenAIResponsesModel(
        'gpt-5.4',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=merge_profile(
            openai_model_profile('gpt-5.4'),
            OpenAIModelProfile(tool_addition_mode=None),
        ),
    )
    agent: Agent[None, str] = Agent(model=model, capabilities=[capability])

    result = await agent.run('Can I get a refund on order-123?')

    assert result.output == 'Loaded.'
    assert any(
        isinstance(part, ToolAvailabilityDeltaPart) and part.tools_added == ['lookup_refund_policy']
        for message in result.all_messages()
        for part in message.parts
    )
    [first_request, second_request] = get_mock_responses_kwargs(mock_client)

    first_tools = cast(list[dict[str, Any]], first_request['tools'])
    second_tools = cast(list[dict[str, Any]], second_request['tools'])
    assert [tool.get('name') or tool['type'] for tool in first_tools] == snapshot(['load_capability'])
    assert [tool.get('name') or tool['type'] for tool in second_tools] == snapshot(
        ['load_capability', 'lookup_refund_policy']
    )
    assert not any('defer_loading' in tool for tool in first_tools + second_tools)

    # Nothing tool-search-shaped anywhere on the wire — no native item, and no replayed `search_tools`
    # call either. The reveal is stated as a system instruction instead, so the history never claims
    # the model ran a search it didn't run, and never names a `search_tools` tool that isn't declared.
    second_input = cast(list[dict[str, Any]], second_request['input'])
    assert not [item for item in second_input if str(item.get('type', '')).startswith('tool_search')]
    assert not [item for item in second_input if item.get('name') == _SEARCH_TOOLS_NAME]
    announcements = [
        item for item in second_input if 'tool(s) are now available' in json.dumps(item.get('content', ''))
    ]
    assert len(announcements) == 1


async def test_openai_mixed_corpus_keeps_the_search_surface_and_defers_both_kinds(allow_model_requests: None):
    """A definition-carrying reveal keeps capability tools out of a server-searchable corpus.

    `get_weather` is searchable and advertised as deferred. `lookup_refund_policy` is hidden outside
    the corpus, so it stays off the wire until an `additional_tools` item carries its full definition.
    """
    pytest.importorskip('openai')

    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:  # pragma: no cover
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed'

    capability = Capability(
        id='refunds',
        description='Refund policy tools.',
        defer_loading=True,
        toolsets=[refunds_toolset],
    )
    mock_client = MockOpenAIResponses.create_mock(
        response_message(
            [
                ResponseOutputMessage(
                    id='msg_done',
                    content=[ResponseOutputText(text='Done.', type='output_text', annotations=[])],
                    role='assistant',
                    status='completed',
                    type='message',
                )
            ]
        )
    )
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=mock_client))
    agent: Agent[None, str] = Agent(model=model, capabilities=[capability])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the weather in a city."""
        return f'Weather in {city}.'

    result = await agent.run('Hello')

    assert result.output == 'Done.'
    [request] = get_mock_responses_kwargs(mock_client)
    tools = cast(list[dict[str, Any]], request['tools'])
    assert [(tool.get('name') or tool['type'], tool.get('defer_loading')) for tool in tools] == snapshot(
        [
            ('tool_search', None),
            ('load_capability', None),
            ('get_weather', True),
        ]
    )
    [tool_search] = [tool for tool in tools if tool['type'] == 'tool_search']
    assert 'execution' not in tool_search


async def test_openai_capability_only_corpus_keeps_tools_byte_identical(allow_model_requests: None):
    """A capability load leaves `tools` byte-for-byte alone on OpenAI, with no `tool_search` in sight.

    `tools` is the first cache section, ahead of `instructions` and every input item, so a difference
    there invalidates the whole prefix on the one turn this is supposed to be free. Nothing here is
    searchable, so there's no search tool — and the Responses API won't take `defer_loading` without
    one. So the gated tool is never declared, and the `additional_tools` item is the entire reveal:
    the schema arrives in an appended input item, which the prefix doesn't include.

    Verified live on `gpt-5.6` before being pinned here: the model calls a tool declared only in that
    item, 3/3, against 0/3 with the item removed.
    """
    pytest.importorskip('openai')

    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:  # pragma: no cover
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed'

    capability = Capability(
        id='refunds',
        description='Refund policy tools.',
        defer_loading=True,
        toolsets=[refunds_toolset],
    )
    mock_client = MockOpenAIResponses.create_mock(
        [
            response_message(
                [
                    ResponseFunctionToolCall(
                        id='fc_load',
                        arguments='{"id":"refunds"}',
                        call_id='call_load',
                        name=LOAD_CAPABILITY_TOOL_NAME,
                        status='completed',
                        type='function_call',
                    )
                ]
            ),
            response_message(
                [
                    ResponseOutputMessage(
                        id='msg_done',
                        content=[ResponseOutputText(text='Loaded.', type='output_text', annotations=[])],
                        role='assistant',
                        status='completed',
                        type='message',
                    )
                ]
            ),
        ]
    )
    model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(openai_client=mock_client))
    agent: Agent[None, str] = Agent(model=model, capabilities=[capability])

    result = await agent.run('Can I get a refund on order-123?')

    assert result.output == 'Loaded.'
    [before, after] = get_mock_responses_kwargs(mock_client)
    assert json.dumps(after['tools'], sort_keys=True) == json.dumps(before['tools'], sort_keys=True)
    assert [tool.get('name') or tool['type'] for tool in cast(list[dict[str, Any]], before['tools'])] == snapshot(
        ['load_capability']
    )
    # The reveal rides an appended input item, so it costs nothing the prefix has already cached.
    assert cast(list[dict[str, Any]], after['input'])[-1] == snapshot(
        {
            'type': 'additional_tools',
            'role': 'developer',
            'tools': [
                {
                    'type': 'function',
                    'name': 'lookup_refund_policy',
                    'parameters': {
                        'additionalProperties': False,
                        'properties': {'order_id': {'type': 'string'}},
                        'required': ['order_id'],
                        'type': 'object',
                    },
                    'description': 'Look up the refund policy for an order.',
                    'strict': True,
                }
            ],
        }
    )


async def test_openai_mixed_corpus_keeps_tools_byte_identical(allow_model_requests: None):
    """A capability load appends its definition without changing a native-search request's `tools`."""
    pytest.importorskip('openai')

    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:  # pragma: no cover
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed'

    capability = Capability(
        id='refunds', description='Refund policy tools.', defer_loading=True, toolsets=[refunds_toolset]
    )
    mock_client = MockOpenAIResponses.create_mock(
        [
            response_message(
                [
                    ResponseFunctionToolCall(
                        id='fc_load',
                        arguments='{"id":"refunds"}',
                        call_id='call_load',
                        name=LOAD_CAPABILITY_TOOL_NAME,
                        status='completed',
                        type='function_call',
                    )
                ]
            ),
            response_message(
                [
                    ResponseOutputMessage(
                        id='msg_done',
                        content=[ResponseOutputText(text='Loaded.', type='output_text', annotations=[])],
                        role='assistant',
                        status='completed',
                        type='message',
                    )
                ]
            ),
        ]
    )
    model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(openai_client=mock_client))
    agent: Agent[None, str] = Agent(model=model, capabilities=[capability])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:  # pragma: no cover
        """Get the weather in a city."""
        return f'Weather in {city}.'

    result = await agent.run('Can I get a refund on order-123?')

    assert result.output == 'Loaded.'
    [before, after] = get_mock_responses_kwargs(mock_client)
    assert json.dumps(after['tools'], sort_keys=True) == json.dumps(before['tools'], sort_keys=True)
    before_tools = cast(list[dict[str, Any]], before['tools'])
    assert [(tool.get('name') or tool['type'], tool.get('defer_loading')) for tool in before_tools] == snapshot(
        [('tool_search', None), ('load_capability', None), ('get_weather', True)]
    )
    [tool_search] = [tool for tool in before_tools if tool['type'] == 'tool_search']
    assert 'execution' not in tool_search
    assert cast(list[dict[str, Any]], after['input'])[-1] == snapshot(
        {
            'type': 'additional_tools',
            'role': 'developer',
            'tools': [
                {
                    'type': 'function',
                    'name': 'lookup_refund_policy',
                    'parameters': {
                        'additionalProperties': False,
                        'properties': {'order_id': {'type': 'string'}},
                        'required': ['order_id'],
                        'type': 'object',
                    },
                    'description': 'Look up the refund policy for an order.',
                    'strict': True,
                }
            ],
        }
    )


@pytest.mark.parametrize(
    ('model_name', 'native_tool_search'),
    [('gpt-5', False), ('gpt-4.1', False), ('gpt-5.6', True)],
)
async def test_openai_local_search_keeps_tools_byte_identical(
    allow_model_requests: None, model_name: str, native_tool_search: bool
) -> None:
    """A local discovery appends its revealed schema without changing OpenAI's `tools` cache section.

    This is mocked because the invariant compares two requests from one run, while a cassette records
    each request separately and the default VCR matchers do not include the body. The native-search case
    pins the other side of the model-profile branch so it cannot silently move onto `additional_tools`.
    """
    pytest.importorskip('openai')

    if native_tool_search:
        discovery = [
            ResponseToolSearchCall(
                id='ts_search',
                arguments={'paths': ['lookup_exchange_rate']},
                call_id=None,
                execution='server',
                status='completed',
                type='tool_search_call',
            ),
            ResponseToolSearchOutputItem(
                id='tso_search',
                call_id=None,
                execution='server',
                status='completed',
                tools=[
                    FunctionTool(
                        name='lookup_exchange_rate',
                        description='Look up an exchange rate.',
                        parameters={},
                        strict=False,
                        type='function',
                    )
                ],
                type='tool_search_output',
            ),
        ]
    else:
        discovery = [
            ResponseFunctionToolCall(
                id='fc_search',
                arguments='{"queries":["exchange rate"]}',
                call_id='call_search',
                name='search_tools',
                status='completed',
                type='function_call',
            )
        ]

    mock_client = MockOpenAIResponses.create_mock(
        [
            response_message(discovery),
            response_message(
                [
                    ResponseOutputMessage(
                        id='msg_done',
                        content=[ResponseOutputText(text='Found it.', type='output_text', annotations=[])],
                        role='assistant',
                        status='completed',
                        type='message',
                    )
                ]
            ),
        ]
    )
    model = OpenAIResponsesModel(model_name, provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(model=model, capabilities=[ToolSearch()])

    @agent.tool_plain
    def always_ready() -> str:  # pragma: no cover
        """An always-visible tool."""
        return 'ready'

    @agent.tool_plain(defer_loading=True)
    def lookup_exchange_rate(currency: str) -> str:  # pragma: no cover
        """Look up an exchange rate."""
        return f'1 {currency} = 1 test unit'

    result = await agent.run('Find the exchange-rate tool.')

    assert result.output == 'Found it.'
    [before, after] = get_mock_responses_kwargs(mock_client)
    assert json.dumps(after['tools'], sort_keys=True) == json.dumps(before['tools'], sort_keys=True)
    additional_tools = [
        item for item in cast(list[dict[str, Any]], after['input']) if item.get('type') == 'additional_tools'
    ]
    if native_tool_search:
        assert additional_tools == []
        assert any(tool['type'] == 'tool_search' for tool in cast(list[dict[str, Any]], after['tools']))
    else:
        assert additional_tools == [
            {
                'type': 'additional_tools',
                'role': 'developer',
                'tools': [
                    {
                        'type': 'function',
                        'name': 'lookup_exchange_rate',
                        'parameters': {
                            'additionalProperties': False,
                            'properties': {'currency': {'type': 'string'}},
                            'required': ['currency'],
                            'type': 'object',
                        },
                        'description': 'Look up an exchange rate.',
                        'strict': True,
                    }
                ],
            }
        ]


async def test_openai_stored_delta_keeps_local_search_tools_byte_identical(allow_model_requests: None) -> None:
    """A stored availability delta preserves the same `gpt-5` local-search cache prefix as a live reveal."""
    pytest.importorskip('openai')

    mock_client = MockOpenAIResponses.create_mock([response_message([]), response_message([])])
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))
    search_tool = ToolDefinition(
        name='search_tools',
        parameters_json_schema={'type': 'object'},
        unless_native=ToolSearchTool.kind,
    )
    revealed_tool = ToolDefinition(
        name='lookup_exchange_rate',
        description='Look up an exchange rate.',
        parameters_json_schema={'type': 'object'},
        defer_loading=True,
        with_native=ToolSearchTool.kind,
    )
    parameters = ModelRequestParameters(function_tools=[search_tool, revealed_tool], native_tools=[ToolSearchTool()])
    before: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='Find the exchange-rate tool.')])]
    after = [*before, ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[revealed_tool.name])])]

    model_settings, before_parameters = model.prepare_request(None, parameters)
    _, after_parameters = model.prepare_request(None, replace(parameters, revealed_tool_names={revealed_tool.name}))
    await model.request(before, model_settings, before_parameters)
    await model.request(after, model_settings, after_parameters)

    [before_request, after_request] = get_mock_responses_kwargs(mock_client)
    assert json.dumps(after_request['tools'], sort_keys=True) == json.dumps(before_request['tools'], sort_keys=True)
    assert [item['type'] for item in after_request['input'] if item.get('type') == 'additional_tools'] == [
        'additional_tools'
    ]


async def test_openai_discovered_tool_without_native_tool_search_omits_defer_loading(
    allow_model_requests: None,
):
    """A discovered tool moves to `additional_tools` on a model without native `tool_search`.

    OpenAI's `defer_loading` only travels alongside a native `tool_search` tool; without one the
    provider rejects a lone `defer_loading` (#5938). Once discovered, the schema is appended in an
    `additional_tools` input item and omitted from top-level `tools`, preserving that cache section.

    This is a unit test, not VCR: the cassette matcher keys only on method and path, so a request
    that regained a stale `defer_loading` (or an over-eager native-tool swap) would still match the
    existing recording and pass green. Only a direct assertion on the emitted payload pins the wire
    invariant the fix is responsible for. The end-to-end deferred-capability flow is covered by
    `test_openai_deferred_capability_runs_on_model_without_native_tool_search`.
    """
    pytest.importorskip('openai')

    final = response_message(
        [
            ResponseOutputMessage(
                id='msg',
                content=[ResponseOutputText(text='Sunny.', type='output_text', annotations=[])],
                role='assistant',
                status='completed',
                type='message',
            )
        ]
    )
    mock_client = MockOpenAIResponses.create_mock(final)
    model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(openai_client=mock_client))
    agent: Agent[None, str] = Agent(model=model, capabilities=[ToolSearch()])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:  # pragma: no cover
        return f'Weather in {city}.'

    # `get_weather` was discovered last turn, so its schema now rides an appended input item.
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='I might want the weather later.')]),
        ModelResponse(parts=[ToolSearchCallPart(args={'queries': ['weather']}, tool_call_id='loc_1')]),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_weather'}]},
                    tool_call_id='loc_1',
                )
            ]
        ),
    ]

    await agent.run('Weather in Paris?', message_history=history)

    [request] = get_mock_responses_kwargs(mock_client)
    request_tools = cast(list[dict[str, Any]], request['tools'])
    assert not any(tool['type'] == 'tool_search' for tool in request_tools)
    assert not any(tool.get('name') == 'get_weather' for tool in request_tools)
    [additional_tools] = [
        item for item in cast(list[dict[str, Any]], request['input']) if item.get('type') == 'additional_tools'
    ]
    [weather_tool] = additional_tools['tools']
    assert 'defer_loading' not in weather_tool
    assert weather_tool['name'] == 'get_weather'


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='dynamic tool disclosure after ToolSearch discovery')
async def test_openai_deferred_capability_runs_on_model_without_native_tool_search(
    allow_model_requests: None, openai_api_key: str
) -> None:
    """Loading a deferred `Capability` and calling its tool must complete on a model
    *without* native `tool_search` (#5938).

    `gpt-5.2` predates OpenAI's native `tool_search`, so search falls back to local and no
    `tool_search` builtin is on the wire. The tool revealed by `load_capability` still carries
    `with_native='tool_search'`, which the base-class filter sheds so no adapter emits a
    wire-side `defer_loading` flag with no native `tool_search` tool to pair it with — which
    OpenAI rejects. The invariant is simply that the run completes and `bar` returns.

    The wire-payload shape (no `defer_loading`, no `tool_search` on the wire) is pinned directly by
    `test_openai_discovered_tool_without_native_tool_search_omits_defer_loading`, since the cassette
    matcher isn't body-sensitive and wouldn't catch a regression here on its own.

    This is the tool-search-*unsupported* half of the matrix. OpenAI is the only provider it
    can be recorded against: every non-deprecated Anthropic (and Google) model supports native
    tool search, so there is no equivalent no-native-`tool_search` model to exercise.
    """
    foo = Capability[None](id='foo', description='Use this capability when the user asks for foo.', defer_loading=True)

    @foo.tool_plain
    def bar(x: int) -> int:
        """Return x."""
        return x

    model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key=openai_api_key))
    agent: Agent[None, str] = Agent(
        model,
        deps_type=type(None),
        instructions="First load capability id 'foo', then call bar.",
        capabilities=[foo],
    )

    result = await agent.run('Use foo with x=1.')

    # The capability loaded and `bar` returned, so the follow-up request carrying the revealed
    # `bar` (the one that used to 400) was accepted and the run finished.
    assert any(isinstance(p, LoadCapabilityReturnPart) for m in result.all_messages() for p in m.parts)
    bar_returns = [
        p for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart) if p.tool_name == 'bar'
    ]
    assert [p.content for p in bar_returns] == [1]


def _cross_provider_tool_search_agent(model: AnthropicModel | OpenAIResponsesModel) -> Agent[None, str]:
    agent: Agent[None, str] = Agent(
        model,
        instructions='Always use get_exchange_rate to answer exchange-rate questions.',
    )

    @agent.tool_plain(defer_loading=True)
    def get_exchange_rate(pair: str) -> str:
        """Look up the exchange rate for a currency pair."""
        return f'{pair}: 0.92'

    return agent


def _assert_exchange_rate_called(messages: list[ModelMessage], pair: str) -> None:
    returns = [
        part
        for part in iter_message_parts(messages, ModelRequest, ToolReturnPart)
        if part.tool_name == 'get_exchange_rate'
    ]
    assert [part.content for part in returns] == [f'{pair}: 0.92']


def _recorded_request_bodies(vcr: Any, provider: Literal['anthropic', 'openai']) -> list[dict[str, Any]]:
    uri_fragment = 'api.anthropic.com' if provider == 'anthropic' else 'api.openai.com'
    return [json.loads(request.body) for request in vcr.requests if uri_fragment in request.uri]


def _assert_openai_native_search_replay(body: dict[str, Any]) -> None:
    item_types = [item.get('type') for item in body['input']]
    assert 'tool_search_call' in item_types
    assert 'tool_search_output' in item_types


def _assert_anthropic_native_search_replay(
    body: dict[str, Any], result_type: Literal['tool_result', 'tool_search_tool_result']
) -> None:
    search_results = [
        block for message in body['messages'] for block in message['content'] if block.get('type') == result_type
    ]
    if result_type == 'tool_result':
        references = [nested for result in search_results for nested in result['content']]
    else:
        references = [reference for result in search_results for reference in result['content']['tool_references']]
    assert any(reference.get('type') == 'tool_reference' for reference in references)


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='cross-provider replay rewrites native tool-search history')
@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_live_tool_search_handoff_anthropic_to_openai(
    allow_model_requests: None,
    anthropic_api_key: str,
    openai_api_key: str,
    vcr: Any,
) -> None:
    """A real Anthropic native search remains callable after replay on OpenAI."""
    first_agent = _cross_provider_tool_search_agent(
        AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    )
    first = await first_agent.run('What is the USD/EUR exchange rate?')
    assert any(isinstance(part, NativeToolSearchCallPart) for message in first.all_messages() for part in message.parts)

    second_agent = _cross_provider_tool_search_agent(
        OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
    )
    second = await second_agent.run(
        'Now call get_exchange_rate for GBP/USD without searching again.',
        message_history=first.all_messages(),
    )

    _assert_exchange_rate_called(second.new_messages(), 'GBP/USD')
    _assert_openai_native_search_replay(_recorded_request_bodies(vcr, 'openai')[0])


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='cross-provider replay rewrites native tool-search history')
@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_live_tool_search_handoff_openai_to_anthropic(
    allow_model_requests: None,
    anthropic_api_key: str,
    openai_api_key: str,
    vcr: Any,
) -> None:
    """A real OpenAI native search remains callable after replay on Anthropic."""
    first_agent = _cross_provider_tool_search_agent(
        OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
    )
    first = await first_agent.run('What is the USD/EUR exchange rate?')
    assert any(isinstance(part, NativeToolSearchCallPart) for message in first.all_messages() for part in message.parts)

    second_agent = _cross_provider_tool_search_agent(
        AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    )
    second = await second_agent.run(
        'Now call get_exchange_rate for GBP/USD without searching again.',
        message_history=first.all_messages(),
    )

    _assert_exchange_rate_called(second.new_messages(), 'GBP/USD')
    _assert_anthropic_native_search_replay(_recorded_request_bodies(vcr, 'anthropic')[0], 'tool_result')


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='cross-provider replay rewrites native tool-search history')
@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_live_tool_search_handoff_anthropic_openai_anthropic(
    allow_model_requests: None,
    anthropic_api_key: str,
    openai_api_key: str,
    vcr: Any,
) -> None:
    """Twice-travelled real history remains accepted and callable on its origin vendor."""
    first_agent = _cross_provider_tool_search_agent(
        AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    )
    first = await first_agent.run('What is the USD/EUR exchange rate?')
    assert any(isinstance(part, NativeToolSearchCallPart) for message in first.all_messages() for part in message.parts)

    second_agent = _cross_provider_tool_search_agent(
        OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
    )
    second = await second_agent.run(
        'Now call get_exchange_rate for GBP/USD without searching again.',
        message_history=first.all_messages(),
    )
    _assert_exchange_rate_called(second.new_messages(), 'GBP/USD')

    third_agent = _cross_provider_tool_search_agent(
        AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    )
    third = await third_agent.run(
        'Now call get_exchange_rate for CAD/USD without searching again.',
        message_history=second.all_messages(),
    )

    _assert_exchange_rate_called(third.new_messages(), 'CAD/USD')
    openai_bodies = _recorded_request_bodies(vcr, 'openai')
    anthropic_bodies = _recorded_request_bodies(vcr, 'anthropic')
    _assert_openai_native_search_replay(openai_bodies[0])
    _assert_anthropic_native_search_replay(anthropic_bodies[-1], 'tool_search_tool_result')


def _trace_capability_messages(messages: list[ModelMessage]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Compact one-line-per-part trace of a deferred-capability conversation.

    Used by the cross-provider replay tests to assert the *story* of the run
    (load → availability delta → tool call → answer) without coupling to provider-specific
    wire shapes. There is deliberately no case for the tool-search parts: a capability-owned
    tool is never searchable, so a search exchange appearing in one of these traces is a
    regression, and the catch-all below says so."""
    trace: list[tuple[str, list[dict[str, Any]]]] = []
    for msg in messages:
        part_trace: list[dict[str, Any]] = []
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                part_info: dict[str, Any] = {'type': 'user', 'content': part.content}
            elif isinstance(part, LoadCapabilityCallPart):
                part_info = {'type': 'load_capability_call', 'id': part.capability_id}
            elif isinstance(part, LoadCapabilityReturnPart):
                part_info = {'type': 'load_capability_return', 'instructions': part.instructions}
            elif isinstance(part, ToolAvailabilityDeltaPart):
                part_info = {'type': 'tool_availability_delta', 'added': part.tools_added}
            elif isinstance(part, ToolCallPart):
                # Normalize args from JSON string to dict so per-row snapshots don't
                # pin on provider-specific whitespace or key ordering.
                part_info = {'type': 'tool_call', 'tool_name': part.tool_name, 'args': part.args_as_dict()}
            elif isinstance(part, ToolReturnPart):
                part_info = {'type': 'tool_return', 'tool_name': part.tool_name, 'content': part.content}
            elif isinstance(part, TextPart):
                part_info = {'type': 'text'}
            else:
                raise AssertionError(
                    f'cross-provider replay trace helper saw unexpected part type: {type(part).__name__}'
                )  # pragma: no cover
            part_trace.append(part_info)
        # Use a flat lowercase tag so inline-snapshot writes a plain string instead
        # of "helpfully" resolving the class name to `'request'`.
        tag = 'request' if isinstance(msg, ModelRequest) else 'response'
        trace.append((tag, part_trace))
    return trace


# Both turns vary per provider — first-turn trajectory differs (Anthropic prepends
# prose, Google may run native search), and even the resume turn isn't uniform
# (some providers prepend a text part before the tool_call). A single `snapshot()`
# literal can only hold one row's value, so we key both expected traces by
# `(first_model, resume_model)` and let inline-snapshot fill each entry on
# `--record-mode=once --inline-snapshot=create`.
_TraceShape = list[tuple[str, list[dict[str, Any]]]]

_FIRST_TURN_EXPECTED: dict[tuple[str, str], _TraceShape] = {
    ('anthropic:claude-sonnet-4-5', 'openai-responses:gpt-5.4'): snapshot(
        [
            ('request', [{'type': 'user', 'content': 'Can I get a refund on order-123?'}]),
            ('response', [{'type': 'text'}, {'type': 'load_capability_call', 'id': 'refunds'}]),
            (
                'request',
                [
                    {
                        'type': 'load_capability_return',
                        'instructions': 'Use the refund policy tool before answering refund questions.',
                    },
                    {'type': 'tool_availability_delta', 'added': ['lookup_refund_policy']},
                ],
            ),
            (
                'response',
                [{'type': 'tool_call', 'tool_name': 'lookup_refund_policy', 'args': {'order_id': 'order-123'}}],
            ),
            (
                'request',
                [
                    {
                        'type': 'tool_return',
                        'tool_name': 'lookup_refund_policy',
                        'content': 'order-123: refund allowed for 30 days',
                    }
                ],
            ),
            ('response', [{'type': 'text'}]),
        ]
    ),
    ('openai-responses:gpt-5.4', 'anthropic:claude-sonnet-4-5'): snapshot(
        [
            ('request', [{'type': 'user', 'content': 'Can I get a refund on order-123?'}]),
            ('response', [{'type': 'load_capability_call', 'id': 'refunds'}]),
            (
                'request',
                [
                    {
                        'type': 'load_capability_return',
                        'instructions': 'Use the refund policy tool before answering refund questions.',
                    },
                    {'type': 'tool_availability_delta', 'added': ['lookup_refund_policy']},
                ],
            ),
            (
                'response',
                [{'type': 'tool_call', 'tool_name': 'lookup_refund_policy', 'args': {'order_id': 'order-123'}}],
            ),
            (
                'request',
                [
                    {
                        'type': 'tool_return',
                        'tool_name': 'lookup_refund_policy',
                        'content': 'order-123: refund allowed for 30 days',
                    }
                ],
            ),
            ('response', [{'type': 'text'}]),
        ]
    ),
    ('google:gemini-3-flash-preview', 'openai-responses:gpt-5.4'): snapshot(
        [
            ('request', [{'type': 'user', 'content': 'Can I get a refund on order-123?'}]),
            ('response', [{'type': 'load_capability_call', 'id': 'refunds'}]),
            (
                'request',
                [
                    {
                        'type': 'load_capability_return',
                        'instructions': 'Use the refund policy tool before answering refund questions.',
                    },
                    {'type': 'tool_availability_delta', 'added': ['lookup_refund_policy']},
                ],
            ),
            (
                'response',
                [{'type': 'tool_call', 'tool_name': 'lookup_refund_policy', 'args': {'order_id': 'order-123'}}],
            ),
            (
                'request',
                [
                    {
                        'type': 'tool_return',
                        'tool_name': 'lookup_refund_policy',
                        'content': 'order-123: refund allowed for 30 days',
                    }
                ],
            ),
            ('response', [{'type': 'text'}]),
        ]
    ),
    ('openai-responses:gpt-5.4', 'google:gemini-3-flash-preview'): snapshot(
        [
            ('request', [{'type': 'user', 'content': 'Can I get a refund on order-123?'}]),
            ('response', [{'type': 'load_capability_call', 'id': 'refunds'}]),
            (
                'request',
                [
                    {
                        'type': 'load_capability_return',
                        'instructions': 'Use the refund policy tool before answering refund questions.',
                    },
                    {'type': 'tool_availability_delta', 'added': ['lookup_refund_policy']},
                ],
            ),
            (
                'response',
                [{'type': 'tool_call', 'tool_name': 'lookup_refund_policy', 'args': {'order_id': 'order-123'}}],
            ),
            (
                'request',
                [
                    {
                        'type': 'tool_return',
                        'tool_name': 'lookup_refund_policy',
                        'content': 'order-123: refund allowed for 30 days',
                    }
                ],
            ),
            ('response', [{'type': 'text'}]),
        ]
    ),
    ('google:gemini-3-flash-preview', 'anthropic:claude-sonnet-4-5'): snapshot(
        [
            ('request', [{'type': 'user', 'content': 'Can I get a refund on order-123?'}]),
            ('response', [{'type': 'load_capability_call', 'id': 'refunds'}]),
            (
                'request',
                [
                    {
                        'type': 'load_capability_return',
                        'instructions': 'Use the refund policy tool before answering refund questions.',
                    },
                    {'type': 'tool_availability_delta', 'added': ['lookup_refund_policy']},
                ],
            ),
            (
                'response',
                [{'type': 'tool_call', 'tool_name': 'lookup_refund_policy', 'args': {'order_id': 'order-123'}}],
            ),
            (
                'request',
                [
                    {
                        'type': 'tool_return',
                        'tool_name': 'lookup_refund_policy',
                        'content': 'order-123: refund allowed for 30 days',
                    }
                ],
            ),
            ('response', [{'type': 'text'}]),
        ]
    ),
}


# Resume-turn structure also varies per provider — some prepend a text part before the
# tool_call. Keyed like `_FIRST_TURN_EXPECTED` so each row's record stays distinct.
_RESUME_TURN_EXPECTED: dict[tuple[str, str], _TraceShape] = {
    ('anthropic:claude-sonnet-4-5', 'openai-responses:gpt-5.4'): snapshot(
        [
            ('request', [{'type': 'user', 'content': 'And what about order-456?'}]),
            (
                'response',
                [
                    {'type': 'text'},
                    {'type': 'tool_call', 'tool_name': 'lookup_refund_policy', 'args': {'order_id': 'order-456'}},
                ],
            ),
            (
                'request',
                [
                    {
                        'type': 'tool_return',
                        'tool_name': 'lookup_refund_policy',
                        'content': 'order-456: refund allowed for 30 days',
                    }
                ],
            ),
            ('response', [{'type': 'text'}]),
        ]
    ),
    ('openai-responses:gpt-5.4', 'anthropic:claude-sonnet-4-5'): snapshot(
        [
            ('request', [{'type': 'user', 'content': 'And what about order-456?'}]),
            (
                'response',
                [{'type': 'tool_call', 'tool_name': 'lookup_refund_policy', 'args': {'order_id': 'order-456'}}],
            ),
            (
                'request',
                [
                    {
                        'type': 'tool_return',
                        'tool_name': 'lookup_refund_policy',
                        'content': 'order-456: refund allowed for 30 days',
                    }
                ],
            ),
            ('response', [{'type': 'text'}]),
        ]
    ),
    ('google:gemini-3-flash-preview', 'openai-responses:gpt-5.4'): snapshot(
        [
            ('request', [{'type': 'user', 'content': 'And what about order-456?'}]),
            (
                'response',
                [{'type': 'tool_call', 'tool_name': 'lookup_refund_policy', 'args': {'order_id': 'order-456'}}],
            ),
            (
                'request',
                [
                    {
                        'type': 'tool_return',
                        'tool_name': 'lookup_refund_policy',
                        'content': 'order-456: refund allowed for 30 days',
                    }
                ],
            ),
            ('response', [{'type': 'text'}]),
        ]
    ),
    ('openai-responses:gpt-5.4', 'google:gemini-3-flash-preview'): snapshot(
        [
            ('request', [{'type': 'user', 'content': 'And what about order-456?'}]),
            (
                'response',
                [{'type': 'tool_call', 'tool_name': 'lookup_refund_policy', 'args': {'order_id': 'order-456'}}],
            ),
            (
                'request',
                [
                    {
                        'type': 'tool_return',
                        'tool_name': 'lookup_refund_policy',
                        'content': 'order-456: refund allowed for 30 days',
                    }
                ],
            ),
            ('response', [{'type': 'text'}]),
        ]
    ),
    ('google:gemini-3-flash-preview', 'anthropic:claude-sonnet-4-5'): snapshot(
        [
            ('request', [{'type': 'user', 'content': 'And what about order-456?'}]),
            (
                'response',
                [{'type': 'tool_call', 'tool_name': 'lookup_refund_policy', 'args': {'order_id': 'order-456'}}],
            ),
            (
                'request',
                [
                    {
                        'type': 'tool_return',
                        'tool_name': 'lookup_refund_policy',
                        'content': 'order-456: refund allowed for 30 days',
                    }
                ],
            ),
            ('response', [{'type': 'text'}]),
        ]
    ),
}


# Cassette names depend on the parametrize id, so we keep these stable across the
# matrix. Recording: `pytest --record-mode=once --inline-snapshot=create`.
@pytest.mark.parametrize('first_model_name,resume_model_name', list(_FIRST_TURN_EXPECTED.keys()))
@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='dynamic tool disclosure after ToolSearch discovery')
@pytest.mark.filterwarnings('ignore:`BuiltinToolCallEvent` is deprecated:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:`BuiltinToolResultEvent` is deprecated:DeprecationWarning')
async def test_cross_provider_capability_replay(
    first_model_name: str,
    resume_model_name: str,
    allow_model_requests: None,
    anthropic_api_key: str,
    openai_api_key: str,
    gemini_api_key: str,
) -> None:
    """A deferred capability registered with the same definition on both turns must
    replay cleanly across any provider pair: the resuming provider re-marks the
    capability-owned tools, treats the prior turn's tool-search history as
    already-discovered, and dispatches `lookup_refund_policy` directly.

    Asserting the trace-level story over both turns is enough — per-provider wire
    shape is already covered by the dedicated OpenAI/Anthropic/Google adapter tests."""
    pytest.importorskip('anthropic')
    pytest.importorskip('openai')
    pytest.importorskip('google.genai')

    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    def make_refunds_cap() -> Capability:
        return Capability(
            id='refunds',
            description='Refund policy tools.',
            instructions='Use the refund policy tool before answering refund questions.',
            toolsets=[refunds_toolset],
            defer_loading=True,
        )

    first_agent: Agent[None, str] = Agent(model=first_model_name, capabilities=[make_refunds_cap()])
    first_result = await first_agent.run('Can I get a refund on order-123?')

    resume_agent: Agent[None, str] = Agent(model=resume_model_name, capabilities=[make_refunds_cap()])
    resume_result = await resume_agent.run(
        'And what about order-456?',
        message_history=first_result.all_messages(),
    )

    # Per-row records pin each provider pair's full trajectory. The resume turn must
    # dispatch `lookup_refund_policy` off the replayed history without a fresh
    # load_capability or tool_search — drift in that contract surfaces as a diff here.
    key = (first_model_name, resume_model_name)
    assert _trace_capability_messages(first_result.all_messages()) == _FIRST_TURN_EXPECTED[key]
    assert _trace_capability_messages(resume_result.new_messages()) == _RESUME_TURN_EXPECTED[key]


@pytest.mark.vcr
@pytest.mark.filterwarnings('ignore:`BuiltinToolCallEvent` is deprecated:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:`BuiltinToolResultEvent` is deprecated:DeprecationWarning')
async def test_anthropic_to_google_deferred_capability_history_replay(
    allow_model_requests: None,
    anthropic_api_key: str,
    gemini_api_key: str,
) -> None:
    """Deferred capability loads replay across native and local tool-search paths."""
    pytest.importorskip('anthropic')
    pytest.importorskip('google.genai')

    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    def make_refunds_cap() -> Capability:
        return Capability(
            id='refunds',
            description='Refund policy tools.',
            instructions='Use the refund policy tool before answering refund questions.',
            toolsets=[refunds_toolset],
            defer_loading=True,
        )

    def trace_messages(messages: list[ModelMessage]) -> list[tuple[str, list[dict[str, Any]]]]:
        trace: list[tuple[str, list[dict[str, Any]]]] = []
        for msg in messages:
            part_trace: list[dict[str, Any]] = []
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    part_info: dict[str, Any] = {'type': 'user', 'content': part.content}
                elif isinstance(part, LoadCapabilityCallPart):
                    part_info = {'type': 'load_capability_call', 'id': part.capability_id}
                elif isinstance(part, LoadCapabilityReturnPart):
                    part_info = {'type': 'load_capability_return', 'instructions': part.instructions}
                # No `ToolSearch*Part` branches: a capability load is stored as a
                # `ToolAvailabilityDeltaPart` now, rather than as a synthesized search exchange, which is
                # the whole point of the part. If one ever shows up here again the `else` below names it.
                elif isinstance(part, ToolAvailabilityDeltaPart):
                    part_info = {'type': 'tool_availability_delta', 'added': part.tools_added}
                elif isinstance(part, ToolCallPart):
                    part_info = {'type': 'tool_call', 'tool_name': part.tool_name, 'args': part.args}
                elif isinstance(part, ToolReturnPart):
                    part_info = {'type': 'tool_return', 'tool_name': part.tool_name, 'content': part.content}
                elif isinstance(part, TextPart):
                    part_info = {'type': 'text'}
                else:
                    raise AssertionError(
                        f'anthropic→google replay trace helper saw unexpected part type: {type(part).__name__}'
                    )  # pragma: no cover
                part_trace.append(part_info)
            trace.append((type(msg).__name__, part_trace))
        return trace

    anthropic_agent: Agent[None, str] = Agent(
        model='anthropic:claude-sonnet-4-5',
        capabilities=[make_refunds_cap()],
        model_settings=AnthropicModelSettings(anthropic_cache=True),
    )
    anthropic_result = await anthropic_agent.run('Can I get a refund on order-123?')

    assert trace_messages(anthropic_result.all_messages()) == snapshot(
        [
            (
                'ModelRequest',
                [{'type': 'user', 'content': 'Can I get a refund on order-123?'}],
            ),
            (
                'ModelResponse',
                [
                    {'type': 'text'},
                    {'type': 'load_capability_call', 'id': 'refunds'},
                ],
            ),
            (
                'ModelRequest',
                [
                    {
                        'type': 'load_capability_return',
                        'instructions': 'Use the refund policy tool before answering refund questions.',
                    },
                    {'type': 'tool_availability_delta', 'added': ['lookup_refund_policy']},
                ],
            ),
            (
                'ModelResponse',
                [
                    {
                        'type': 'tool_call',
                        'tool_name': 'lookup_refund_policy',
                        'args': {'order_id': 'order-123'},
                    }
                ],
            ),
            (
                'ModelRequest',
                [
                    {
                        'type': 'tool_return',
                        'tool_name': 'lookup_refund_policy',
                        'content': 'order-123: refund allowed for 30 days',
                    }
                ],
            ),
            ('ModelResponse', [{'type': 'text'}]),
        ]
    )

    google_agent: Agent[None, str] = Agent(
        model='google:gemini-3-flash-preview',
        capabilities=[make_refunds_cap()],
    )
    google_result = await google_agent.run(
        'And what about order-456?',
        message_history=anthropic_result.all_messages(),
    )

    assert trace_messages(google_result.new_messages()) == snapshot(
        [
            (
                'ModelRequest',
                [{'type': 'user', 'content': 'And what about order-456?'}],
            ),
            (
                'ModelResponse',
                [
                    {
                        'type': 'tool_call',
                        'tool_name': 'lookup_refund_policy',
                        'args': {'order_id': 'order-456'},
                    }
                ],
            ),
            (
                'ModelRequest',
                [
                    {
                        'type': 'tool_return',
                        'tool_name': 'lookup_refund_policy',
                        'content': 'order-456: refund allowed for 30 days',
                    }
                ],
            ),
            ('ModelResponse', [{'type': 'text'}]),
        ]
    )
    assert google_result.output == snapshot("""\
For order-456, the policy is the same: **a refund is allowed within 30 days** of your purchase.

Is there anything else I can assist you with?\
""")


class _TransientlyFailingModel(WrapperModel):
    """Fails the first N requests with a retryable API error, then delegates to the live model.

    The failure is synthetic by necessity — a genuinely transient provider outage can't be
    scripted into a recording — but every request that does not fail goes out on the real wire,
    so the cassette captures the true cross-provider handoff and recovery.
    """

    def __init__(self, wrapped: Model, failures: int) -> None:
        super().__init__(wrapped)
        self.failures_remaining = failures

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise ModelAPIError(self.model_name, 'synthetic transient failure')
        return await super().request(messages, model_settings, model_request_parameters)


@pytest.mark.vcr
@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_live_fallback_failover_capability_load_and_recovery(
    allow_model_requests: None,
    anthropic_api_key: str,
    openai_api_key: str,
    vcr: Any,
) -> None:
    """A capability loaded during a failover leg survives the switch back to the recovered primary.

    `FallbackModel` retries the primary on every request, so a transient primary failure means the
    run's history is authored by two providers mid-run: the fallback serves the turn that loads the
    deferred capability, then the recovered primary must project that foreign-authored reveal onto
    its own channel — the `tool_addition` block plus the lazily appended `defer_loading` entry —
    and call the revealed tool off it.
    """
    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    primary = _TransientlyFailingModel(
        AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key=anthropic_api_key)), failures=1
    )
    fallback = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(api_key=openai_api_key))
    agent: Agent[None, str] = Agent(
        model=FallbackModel(primary, fallback),
        capabilities=[
            Capability(
                id='refunds',
                description='Refund policy tools.',
                instructions='Use the refund policy tool before answering refund questions.',
                toolsets=[refunds_toolset],
                defer_loading=True,
            )
        ],
    )

    result = await agent.run('Can I get a refund on order-123? Use your tools.')

    # The failover leg (OpenAI) authored the capability load; the recovered primary (Anthropic)
    # authored the revealed tool's call. `provider_name` on each response pins who served what.
    responses = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    served_by = [
        (
            response.provider_name,
            [part.tool_name for part in response.parts if isinstance(part, ToolCallPart)],
        )
        for response in responses
    ]
    assert served_by == snapshot(
        [
            ('openai', ['load_capability']),
            ('anthropic', ['lookup_refund_policy']),
            ('anthropic', []),
        ]
    )
    assert 'refund' in result.output.lower()

    # One OpenAI request (the failover leg), then two Anthropic requests after recovery.
    openai_bodies = _recorded_request_bodies(vcr, 'openai')
    anthropic_bodies = _recorded_request_bodies(vcr, 'anthropic')
    assert (len(openai_bodies), len(anthropic_bodies)) == (1, 2)

    # Both recovered-primary requests render the reveal on Anthropic's own channel: the
    # `tool_addition` reference plus the lazily appended deferred entry. These bind at record
    # time only — playback returns the recorded bodies regardless of what current code sends —
    # so the regression guards for this rendering are the mocked byte-level tests
    # (`test_fallback_reprojects_openai_delta_to_anthropic_tool_addition` and siblings); what
    # stays live here is the run trajectory, `served_by`, and the cache-prefix checker.
    for body in anthropic_bodies:
        serialized = json.dumps(body, sort_keys=True)
        assert serialized.count('"type": "tool_addition"') == 1
        [revealed] = [tool for tool in body['tools'] if tool.get('name') == 'lookup_refund_policy']
        assert revealed.get('defer_loading') is True


def test_anthropic_tool_search_result_error_block_mapping():
    """An error result block (no `tool_references`) produces a
    `NativeToolReturnPart` without discovered tools in its metadata."""
    pytest.importorskip('anthropic')

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

    malformed = ToolReturnPart(tool_name='search_tools', content='not a typed return', tool_call_id='c1')
    refs, message = _build_custom_tool_search_replay_blocks(
        malformed, deferred_tools_active=True, declared_tool_names=set()
    )
    assert refs is None and message is None


def test_anthropic_build_tool_search_replay_block_error_branch():
    """Replay reconstruction must round-trip an error result that the parse-time
    mapper stashed on `provider_details` back into the `tool_search_tool_result_error`
    inner block — otherwise a transient provider error on turn N would silently
    flip into a fake successful empty-search on turn N+1's resend.

    The Anthropic SDK's `BetaToolSearchToolResultErrorParam` carries only `error_code`
    on the wire (no `error_message`), so the message stashed on `provider_details`
    is observability-only — verified separately in
    `test_anthropic_tool_search_result_error_block_mapping`.
    """
    pytest.importorskip('anthropic')

    return_part = NativeToolSearchReturnPart(
        provider_name='anthropic',
        tool_call_id='srv_err',
        content={'discovered_tools': []},
        provider_details={'error_code': 'unavailable', 'error_message': 'temporary outage'},
    )
    block = _build_tool_search_replay_block(return_part, 'srv_err', declared_tool_names=set())
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
    call_part = _map_tool_search_call(call, 'openai')
    return_part = _build_tool_search_return_part('call_1', output, 'openai')
    assert isinstance(call_part, NativeToolSearchCallPart)
    assert call_part.tool_name == 'tool_search'
    # OpenAI server-executed `tool_search.arguments` carries `paths`; the adapter
    # normalizes that into the cross-provider `queries` slot.
    assert call_part.args == {'queries': ['get_exchange_rate']}
    assert isinstance(return_part, NativeToolSearchReturnPart)
    assert return_part.content == {'discovered_tools': [{'name': 'get_exchange_rate'}]}
    assert return_part.provider_details == {
        'id': 'tso_1',
        'call_id': 'call_1',
        'execution': 'server',
        'status': 'completed',
    }

    # Non-function tools in the output don't have a `name` attribute and are skipped.

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
    mixed = _build_tool_search_return_part('call_mix', mixed_output, 'openai')
    assert mixed.content == {'discovered_tools': [{'name': 'real'}]}


@pytest.mark.parametrize('call_id', [None, 'call_1'], ids=['null-id', 'explicit-id'])
@pytest.mark.parametrize('output_first', [False, True], ids=['call-first', 'output-first'])
def test_openai_processes_hosted_tool_search_call_and_output(call_id: str | None, output_first: bool) -> None:
    """A matched output follows its call regardless of provider item order."""
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())))
    call = ResponseToolSearchCall(
        id='ts_1',
        arguments={'paths': ['get_exchange_rate', 'stock_lookup']},
        call_id=call_id,
        execution='server',
        status='completed',
        type='tool_search_call',
    )
    output = ResponseToolSearchOutputItem(
        id='tso_1',
        call_id=call_id,
        execution='server',
        status='completed',
        tools=[
            FunctionTool(name='get_exchange_rate', description='', parameters={}, strict=False, type='function'),
            FunctionTool(name='stock_lookup', description='', parameters={}, strict=False, type='function'),
        ],
        type='tool_search_output',
    )

    response_items = [output, call] if output_first else [call, output]
    response = model._process_response(  # pyright: ignore[reportPrivateUsage]
        response_message(response_items), OpenAIResponsesModelSettings(), ModelRequestParameters()
    )

    [call_part, return_part] = response.parts
    assert isinstance(call_part, NativeToolSearchCallPart)
    assert isinstance(return_part, NativeToolSearchReturnPart)
    assert call_part.tool_call_id == return_part.tool_call_id == (call_id or 'ts_1')
    assert call_part.provider_details == {'call_id': call_id, 'execution': 'server', 'status': 'completed'}
    assert return_part.content == {'discovered_tools': [{'name': 'get_exchange_rate'}, {'name': 'stock_lookup'}]}
    assert return_part.provider_details == {
        'id': 'tso_1',
        'call_id': call_id,
        'execution': 'server',
        'status': 'completed',
    }


@pytest.mark.parametrize(
    ('call_status', 'output_status'),
    [('completed', 'completed'), ('incomplete', 'in_progress')],
)
def test_openai_preserves_empty_hosted_tool_search_output(
    call_status: Literal['completed', 'incomplete'],
    output_status: Literal['completed', 'in_progress'],
) -> None:
    """Completed emptiness and in-progress partial state retain their actual statuses."""
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())))
    call = ResponseToolSearchCall(
        id='ts_1',
        arguments={},
        call_id=None,
        execution='server',
        status=call_status,
        type='tool_search_call',
    )
    output = ResponseToolSearchOutputItem(
        id='tso_1',
        call_id=None,
        execution='server',
        status=output_status,
        tools=[],
        type='tool_search_output',
    )

    response = model._process_response(  # pyright: ignore[reportPrivateUsage]
        response_message([call, output]), OpenAIResponsesModelSettings(), ModelRequestParameters()
    )

    [call_part, return_part] = response.parts
    assert isinstance(call_part, NativeToolSearchCallPart)
    assert isinstance(return_part, NativeToolSearchReturnPart)
    assert return_part.content == {'discovered_tools': []}
    assert call_part.provider_details == {'call_id': None, 'execution': 'server', 'status': call_status}
    assert return_part.provider_details == {
        'id': 'tso_1',
        'call_id': None,
        'execution': 'server',
        'status': output_status,
    }


def test_openai_does_not_fabricate_missing_hosted_tool_search_output() -> None:
    """A call without an output must stay distinct from a completed empty search."""
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())))
    call = ResponseToolSearchCall(
        id='ts_1',
        arguments={},
        call_id=None,
        execution='server',
        status='incomplete',
        type='tool_search_call',
    )

    response = model._process_response(  # pyright: ignore[reportPrivateUsage]
        response_message([call]), OpenAIResponsesModelSettings(), ModelRequestParameters()
    )

    [call_part] = response.parts
    assert isinstance(call_part, NativeToolSearchCallPart)


def _openai_hosted_tool_search_items() -> tuple[list[ResponseToolSearchCall], list[ResponseToolSearchOutputItem]]:
    calls = [
        ResponseToolSearchCall(
            id=f'ts_{suffix}',
            arguments={'paths': [name]},
            call_id=None,
            execution='server',
            status='completed',
            type='tool_search_call',
        )
        for suffix, name in [('a', 'get_exchange_rate'), ('b', 'stock_lookup')]
    ]
    outputs = [
        ResponseToolSearchOutputItem(
            id=f'tso_{suffix}',
            call_id=None,
            execution='server',
            status='completed',
            tools=[FunctionTool(name=name, description='', parameters={}, strict=False, type='function')],
            type='tool_search_output',
        )
        for suffix, name in [('a', 'get_exchange_rate'), ('b', 'stock_lookup')]
    ]
    return calls, outputs


def _openai_hosted_tool_search_parameters() -> ModelRequestParameters:
    return ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name='get_exchange_rate',
                description='Look up an exchange rate.',
                with_native=ToolSearchTool.kind,
                defer_loading=True,
            ),
            ToolDefinition(
                name='stock_lookup',
                description='Look up a stock price.',
                with_native=ToolSearchTool.kind,
                defer_loading=True,
            ),
        ],
        tool_visibility={'get_exchange_rate': 'deferred', 'stock_lookup': 'deferred'},
        native_tools=[ToolSearchTool()],
    )


@pytest.mark.parametrize('call_id', [None, 'call_a'], ids=['null-id', 'explicit-id'])
def test_openai_preserves_unmatched_hosted_tool_search_output(call_id: str | None) -> None:
    """An actual output is retained even when no matching call is present."""
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())))
    _, outputs = _openai_hosted_tool_search_items()
    output = outputs[0].model_copy(update={'call_id': call_id})

    response = model._process_response(  # pyright: ignore[reportPrivateUsage]
        response_message([output]), OpenAIResponsesModelSettings(), ModelRequestParameters()
    )
    [return_part] = response.parts
    assert isinstance(return_part, NativeToolSearchReturnPart)
    assert return_part.tool_call_id == (call_id or 'tso_a')


async def test_openai_does_not_guess_ambiguous_hosted_tool_search_pairing() -> None:
    """Ambiguous null-ID pairs retain and replay every provider item without guessed correlation."""
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())))
    calls, outputs = _openai_hosted_tool_search_items()

    ambiguous = model._process_response(  # pyright: ignore[reportPrivateUsage]
        response_message([calls[0], outputs[0], calls[1], outputs[1]]),
        OpenAIResponsesModelSettings(),
        ModelRequestParameters(),
    )
    ambiguous_parts = [
        part for part in ambiguous.parts if isinstance(part, NativeToolSearchCallPart | NativeToolSearchReturnPart)
    ]
    assert [part.tool_call_id for part in ambiguous_parts] == ['ts_a', 'tso_a', 'ts_b', 'tso_b']

    _, replayed_items = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ambiguous],
        OpenAIResponsesModelSettings(openai_send_reasoning_ids=True),
        _openai_hosted_tool_search_parameters(),
    )
    assert [(item.get('type'), item.get('id'), item.get('call_id')) for item in replayed_items] == [
        ('tool_search_call', 'ts_a', None),
        ('tool_search_output', 'tso_a', None),
        ('tool_search_call', 'ts_b', None),
        ('tool_search_output', 'tso_b', None),
    ]


def test_openai_ignores_client_tool_search_output() -> None:
    """Client outputs are supplied by Pydantic AI and do not belong in model responses."""
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())))
    _, outputs = _openai_hosted_tool_search_items()
    client_output = outputs[0].model_copy(update={'execution': 'client'})

    response = model._process_response(  # pyright: ignore[reportPrivateUsage]
        response_message([client_output]), OpenAIResponsesModelSettings(), ModelRequestParameters()
    )

    assert response.parts == []


async def test_openai_streaming_ignores_client_tool_search_output(allow_model_requests: None) -> None:
    """Streaming drops client-execution output items, matching `_process_response`."""
    from openai.types import responses as resp

    _, outputs = _openai_hosted_tool_search_items()
    client_output = outputs[0].model_copy(update={'execution': 'client'})
    final_response = response_message([client_output]).model_copy(update={'status': 'completed'})
    created_response = response_message([]).model_copy(update={'status': 'in_progress'})
    stream: list[resp.ResponseStreamEvent] = [
        resp.ResponseCreatedEvent(response=created_response, type='response.created', sequence_number=0),
        resp.ResponseOutputItemAddedEvent(
            item=client_output.model_copy(update={'status': 'in_progress'}),
            output_index=0,
            type='response.output_item.added',
            sequence_number=1,
        ),
        resp.ResponseOutputItemDoneEvent(
            item=client_output, output_index=0, type='response.output_item.done', sequence_number=2
        ),
        resp.ResponseCompletedEvent(response=final_response, type='response.completed', sequence_number=3),
    ]

    mock_client = MockOpenAIResponses.create_mock_stream(stream)
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=mock_client))
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart(content='test')])],
        OpenAIResponsesModelSettings(),
        ModelRequestParameters(),
    ) as streamed_response:
        async for _ in streamed_response:
            pass

    assert streamed_response.get().parts == []


@pytest.mark.parametrize('terminal_status', ['completed', 'failed', 'incomplete'])
@pytest.mark.parametrize(
    ('pair_count', 'expected_ids'),
    [(1, ['ts_a', 'ts_a']), (2, ['ts_a', 'tso_a', 'ts_b', 'tso_b'])],
    ids=['singleton', 'ambiguous'],
)
async def test_openai_hosted_tool_search_null_id_streaming_parity(
    allow_model_requests: None,
    pair_count: int,
    expected_ids: list[str],
    terminal_status: Literal['completed', 'failed', 'incomplete'],
) -> None:
    """Single and ambiguous null-ID responses converge to identical parts in both modes.

    Every terminal event variant runs the singleton backfill, so failed and incomplete
    streams re-key the return part just like completed ones.
    """
    from openai.types import responses as resp

    calls, outputs = _openai_hosted_tool_search_items()
    final_items = [item for pair in zip(calls[:pair_count], outputs[:pair_count]) for item in pair]
    completed_response = response_message(final_items).model_copy(update={'status': terminal_status})
    created_response = response_message([]).model_copy(update={'status': 'in_progress'})
    stream: list[resp.ResponseStreamEvent] = [
        resp.ResponseCreatedEvent(response=created_response, type='response.created', sequence_number=0)
    ]
    sequence_number = 1
    for output_index, item in enumerate(final_items):
        added_item = item.model_copy(update={'status': 'in_progress'})
        stream.extend(
            [
                resp.ResponseOutputItemAddedEvent(
                    item=added_item,
                    output_index=output_index,
                    type='response.output_item.added',
                    sequence_number=sequence_number,
                ),
                resp.ResponseOutputItemDoneEvent(
                    item=item,
                    output_index=output_index,
                    type='response.output_item.done',
                    sequence_number=sequence_number + 1,
                ),
            ]
        )
        sequence_number += 2
    if terminal_status == 'completed':
        terminal: resp.ResponseStreamEvent = resp.ResponseCompletedEvent(
            response=completed_response, type='response.completed', sequence_number=sequence_number
        )
    elif terminal_status == 'failed':
        terminal = resp.ResponseFailedEvent(
            response=completed_response, type='response.failed', sequence_number=sequence_number
        )
    else:
        terminal = resp.ResponseIncompleteEvent(
            response=completed_response, type='response.incomplete', sequence_number=sequence_number
        )
    stream.append(terminal)

    mock_client = MockOpenAIResponses.create_mock_stream(stream)
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=mock_client))
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart(content='test')])],
        OpenAIResponsesModelSettings(),
        ModelRequestParameters(),
    ) as streamed_response:
        assert [event async for event in streamed_response]

    streamed = streamed_response.get()
    non_streamed = model._process_response(  # pyright: ignore[reportPrivateUsage]
        completed_response, OpenAIResponsesModelSettings(), ModelRequestParameters()
    )
    streamed_parts = [
        part for part in streamed.parts if isinstance(part, NativeToolSearchCallPart | NativeToolSearchReturnPart)
    ]
    non_streamed_parts = [
        part for part in non_streamed.parts if isinstance(part, NativeToolSearchCallPart | NativeToolSearchReturnPart)
    ]
    assert [part.tool_call_id for part in streamed_parts] == expected_ids

    def normalized(
        parts: list[NativeToolSearchCallPart | NativeToolSearchReturnPart],
    ) -> list[NativeToolSearchCallPart | NativeToolSearchReturnPart]:
        # Return parts stamp a construction-time timestamp; align it so the equality
        # check covers every other field.
        return [
            replace(part, timestamp=non_streamed.timestamp) if isinstance(part, NativeToolSearchReturnPart) else part
            for part in parts
        ]

    assert normalized(streamed_parts) == normalized(non_streamed_parts)


@pytest.mark.parametrize('send_item_ids', [False, True])
@pytest.mark.parametrize(
    ('call_status', 'output_status', 'discovered_names'),
    [
        ('completed', 'completed', ['get_exchange_rate', 'stock_lookup']),
        ('completed', 'completed', []),
        ('incomplete', 'in_progress', []),
    ],
)
async def test_openai_replays_hosted_tool_search_call_and_output(
    send_item_ids: bool,
    call_status: Literal['completed', 'incomplete'],
    output_status: Literal['in_progress', 'completed'],
    discovered_names: list[str],
) -> None:
    """Replay the authoritative output while keeping hosted `call_id` null.

    This pins the request payload because cassette matching does not compare request bodies.
    """
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())))
    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': ['get_exchange_rate']},
                    tool_call_id='ts_1',
                    id='ts_1',
                    provider_name='openai',
                    provider_details={'call_id': None, 'execution': 'server', 'status': call_status},
                ),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': name} for name in discovered_names]},
                    tool_call_id='ts_1',
                    provider_name='openai',
                    provider_details={
                        'id': 'tso_1',
                        'call_id': None,
                        'execution': 'server',
                        'status': output_status,
                    },
                ),
            ],
            provider_name='openai',
        )
    ]
    _, openai_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        history,
        OpenAIResponsesModelSettings(openai_send_reasoning_ids=send_item_ids),
        _openai_hosted_tool_search_parameters(),
    )

    expected_call: dict[str, Any] = {
        'call_id': None,
        'arguments': {'queries': ['get_exchange_rate']},
        'type': 'tool_search_call',
        'execution': 'server',
        'status': call_status,
    }
    expected_output: dict[str, Any] = {
        'call_id': None,
        'execution': 'server',
        'status': output_status,
        'tools': [
            {
                'name': name,
                'parameters': {'type': 'object', 'properties': {}},
                'type': 'function',
                'description': {
                    'get_exchange_rate': 'Look up an exchange rate.',
                    'stock_lookup': 'Look up a stock price.',
                }[name],
                'strict': False,
                'defer_loading': True,
            }
            for name in discovered_names
        ],
        'type': 'tool_search_output',
    }
    if send_item_ids:
        expected_call['id'] = 'ts_1'
        expected_output['id'] = 'tso_1'
    assert openai_messages == [expected_call, expected_output]


@pytest.mark.parametrize('send_item_ids', [False, True])
@pytest.mark.parametrize(
    'discovered_tools',
    [[], [{'name': 'get_exchange_rate'}]],
    ids=['empty', 'discovered'],
)
@pytest.mark.parametrize(
    'return_details',
    [{'status': 'completed'}, {'id': None, 'status': 'completed'}, {'id': 1, 'status': 'completed'}],
    ids=['status-only', 'null-id', 'non-str-id'],
)
async def test_openai_replays_legacy_tool_search_history_call_only(
    send_item_ids: bool, discovered_tools: list[ToolSearchMatch], return_details: dict[str, Any]
) -> None:
    """Returns without a real preserved output identity replay the call item alone.

    Pre-fix histories stashed only `status`; mangled round-trips can carry a null or
    non-string `id`. An empty such return cannot be told apart from a missing output,
    and a non-empty one references server-side state its call already carries, so all
    degrade to the pre-fix call-only wire shape. This pins the request payload because
    cassette matching does not compare request bodies.
    """
    legacy_history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': ['get_exchange_rate']},
                    tool_call_id='ts_old',
                    id='ts_old',
                    provider_name='openai',
                ),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': discovered_tools},
                    tool_call_id='ts_old',
                    provider_name='openai',
                    provider_details=return_details,
                ),
            ],
            provider_name='openai',
        )
    ]
    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(legacy_history))
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())))

    _, openai_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        history,
        OpenAIResponsesModelSettings(openai_send_reasoning_ids=send_item_ids),
        ModelRequestParameters(native_tools=[ToolSearchTool()]),
    )

    expected_call: dict[str, Any] = {
        'call_id': 'ts_old',
        'arguments': {'queries': ['get_exchange_rate']},
        'type': 'tool_search_call',
        'execution': 'server',
        'status': 'completed',
    }
    if send_item_ids:
        expected_call['id'] = 'ts_old'
    assert openai_messages == [expected_call]


async def test_openai_replay_falls_back_from_invalid_provider_call_id() -> None:
    """Malformed persisted provider metadata must not leak into the request payload."""
    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': []},
                    tool_call_id='ts_1',
                    id='ts_1',
                    provider_name='openai',
                    provider_details={'call_id': 1},
                )
            ],
            provider_name='openai',
        )
    ]
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())))

    _, [tool_search_call] = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        history, OpenAIResponsesModelSettings(), ModelRequestParameters(native_tools=[ToolSearchTool()])
    )

    assert tool_search_call.get('call_id') == 'ts_1'


@pytest.mark.vcr
async def test_openai_native_tool_search_round_trip(allow_model_requests: None, openai_api_key: str) -> None:
    """End-to-end against live OpenAI Responses: native server-executed `tool_search`
    populates `NativeToolCallPart` / `NativeToolReturnPart`, the model invokes the
    discovered deferred tool by its plain name, and stateless replay preserves the
    discovered-tool output.
    """

    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)

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
        isinstance(p, NativeToolCallPart) and p.tool_name == 'tool_search'
        for m in result.all_messages()
        for p in m.parts
    )
    search_returns = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, NativeToolSearchReturnPart)
    ]
    assert len(search_returns) == 1
    assert search_returns[0].content == {'discovered_tools': [{'name': 'get_exchange_rate'}]}
    assert search_returns[0].provider_details == {
        'id': 'tso_0ab095b1426acab50069f17095e2b08196922f617e9b78df21',
        'call_id': None,
        'execution': 'server',
        'status': 'completed',
    }

    rate_returns = [
        p
        for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if p.tool_name == 'get_exchange_rate'
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
    # The direct request-mapping test pins the second-turn call/output body because the
    # cassette matcher does not compare request bodies. Here we only verify the deferred
    # corpus remains present in the recorded integration flow.
    second_request = cast(dict[str, Any], interactions[1]['request']['parsed_body'])
    second_deferred = {
        cast(str, t['name'])
        for t in cast(list[dict[str, Any]], second_request['tools'])
        if t.get('defer_loading') is True
    }
    assert 'get_exchange_rate' in second_deferred


@pytest.mark.vcr
async def test_openai_hosted_tool_search_stateless_continuation(
    allow_model_requests: None, openai_api_key: str
) -> None:
    """A tool loaded without being called remains callable through stateless history replay."""
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)

    @agent.tool_plain(defer_loading=True)
    def get_exchange_rate(pair: str) -> str:
        """Look up an exchange rate."""
        return f'{pair}: 0.92'

    first = await agent.run(
        'Use hosted tool search to load get_exchange_rate. Do not call it yet. Reply only with "loaded".'
    )
    assert first.output == 'loaded'
    first_search_returns = [
        part
        for message in first.all_messages()
        for part in message.parts
        if isinstance(part, NativeToolSearchReturnPart)
    ]
    assert len(first_search_returns) == 1
    assert first_search_returns[0].content == {'discovered_tools': [{'name': 'get_exchange_rate'}]}

    second = await agent.run(
        'Call get_exchange_rate with pair="USD/EUR". Do not search again.',
        message_history=first.all_messages(),
    )
    assert not any(
        isinstance(part, NativeToolSearchCallPart | NativeToolSearchReturnPart)
        for message in second.new_messages()
        for part in message.parts
    )
    rate_returns = [
        part
        for part in iter_message_parts(second.new_messages(), ModelRequest, ToolReturnPart)
        if part.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == 'USD/EUR: 0.92'


@pytest.mark.vcr
async def test_openai_hosted_tool_search_stateless_continuation_without_item_ids(
    allow_model_requests: None, openai_api_key: str
) -> None:
    """The identity-less replay shape (`openai_send_reasoning_ids=False`) stays callable.

    With ids stripped, the replayed `tool_search_call` and `tool_search_output` carry
    `call_id: null` and no `id`; the API accepts the pair and the loaded tool remains
    callable without a fresh search. The recording is what proves API acceptance (the
    cassette assertions below guard it against drifting on re-record); the current
    code's payload is pinned by `test_openai_replays_hosted_tool_search_call_and_output`.
    """
    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)

    @agent.tool_plain(defer_loading=True)
    def get_exchange_rate(pair: str) -> str:
        """Look up an exchange rate."""
        return f'{pair}: 0.92'

    first = await agent.run(
        'Use hosted tool search to load get_exchange_rate. Do not call it yet. Reply only with "loaded".'
    )
    assert first.output == 'loaded'

    second = await agent.run(
        'Call get_exchange_rate with pair="USD/EUR". Do not search again.',
        message_history=first.all_messages(),
        model_settings=OpenAIResponsesModelSettings(openai_send_reasoning_ids=False),
    )
    assert not any(
        isinstance(part, NativeToolSearchCallPart | NativeToolSearchReturnPart)
        for message in second.new_messages()
        for part in message.parts
    )
    rate_returns = [
        part
        for part in iter_message_parts(second.new_messages(), ModelRequest, ToolReturnPart)
        if part.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == 'USD/EUR: 0.92'

    cassette_path = (
        Path(__file__).parent
        / 'cassettes'
        / 'test_tool_search'
        / 'test_openai_hosted_tool_search_stateless_continuation_without_item_ids.yaml'
    )
    cassette = cast(dict[str, Any], yaml.safe_load(cassette_path.read_text(encoding='utf-8')))
    interactions = cast(list[dict[str, Any]], cassette['interactions'])
    replayed = [
        item
        for item in interactions[1]['request']['parsed_body']['input']
        if item.get('type') in ('tool_search_call', 'tool_search_output')
    ]
    assert [(item['type'], 'id' in item, item.get('call_id')) for item in replayed] == [
        ('tool_search_call', False, None),
        ('tool_search_output', False, None),
    ]


@pytest.mark.vcr
async def test_openai_native_tool_search_gpt_5_6(allow_model_requests: None, openai_api_key: str) -> None:
    """End-to-end against live OpenAI Responses: GPT-5.6 supports the native `tool_search`
    tool with `defer_loading`, backing `supports_tool_search` in its model profile — the
    server-executed search discovers the deferred tool and the model dispatches it.
    """
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)

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
        isinstance(p, NativeToolCallPart) and p.tool_name == 'tool_search'
        for m in result.all_messages()
        for p in m.parts
    )
    rate_returns = [
        p
        for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if p.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'


@pytest.mark.vcr
async def test_openai_native_tool_search_with_deferred_capability(
    allow_model_requests: None, openai_api_key: str
) -> None:
    """A mixed corpus uses server-side search and reveals a capability tool by full definition."""
    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed'

    capability = Capability(
        id='refunds', description='Refund policy tools.', defer_loading=True, toolsets=[refunds_toolset]
    )
    model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(api_key=openai_api_key))
    agent: Agent[None, str] = Agent(model=model, capabilities=[capability])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:
        """Get the current weather in a city."""
        return f'Weather in {city}: sunny'

    result = await agent.run(
        'Complete both tasks in order before answering: use tool search to find and call the weather tool '
        'for Paris; then load the refunds capability and call its refund-policy tool for order-123.'
    )

    assert list(iter_message_parts(result.all_messages(), ModelResponse, NativeToolSearchCallPart))
    assert list(iter_message_parts(result.all_messages(), ModelRequest, LoadCapabilityReturnPart))
    tool_returns = {
        part.tool_name: part.content for part in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
    }
    assert tool_returns['get_weather'] == 'Weather in Paris: sunny'
    assert tool_returns['lookup_refund_policy'] == 'order-123: refund allowed'


@pytest.mark.vcr
async def test_openai_native_tool_search_with_deferred_capability_gpt_5_6_sol(
    allow_model_requests: None, openai_api_key: str, vcr: Any
) -> None:
    """GPT-5.6 Sol reveals a capability through `additional_tools` without changing `tools`."""
    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed'

    capability = Capability(
        id='refunds', description='Refund policy tools.', defer_loading=True, toolsets=[refunds_toolset]
    )
    model = OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(api_key=openai_api_key))
    agent: Agent[None, str] = Agent(model=model, capabilities=[capability])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:
        """Get the current weather in a city."""
        return f'Weather in {city}: sunny'

    result = await agent.run(
        'Complete both tasks in order before answering: use tool search to find and call the weather tool '
        'for Paris; then load the refunds capability and call its refund-policy tool for order-123.'
    )

    request_bodies = [json.loads(request.body) for request in vcr.requests]
    assert len(request_bodies) >= 3
    before = request_bodies[0]
    reveal = next(
        body for body in request_bodies[1:] if any(item.get('type') == 'additional_tools' for item in body['input'])
    )
    before_tool_names = [tool.get('name') for tool in before['tools']]
    assert 'lookup_refund_policy' not in before_tool_names
    assert reveal['tools'] == before['tools']
    [addition] = [item for item in reveal['input'] if item.get('type') == 'additional_tools']
    [revealed] = addition['tools']
    assert revealed['name'] == 'lookup_refund_policy'
    assert revealed['description'] == 'Look up the refund policy for an order.'
    assert revealed['parameters']['properties']['order_id']['type'] == 'string'

    assert list(iter_message_parts(result.all_messages(), ModelResponse, NativeToolSearchCallPart))
    assert list(iter_message_parts(result.all_messages(), ModelRequest, LoadCapabilityReturnPart))
    tool_returns = {
        part.tool_name: part.content for part in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
    }
    assert tool_returns['get_weather'] == 'Weather in Paris: sunny'
    assert tool_returns['lookup_refund_policy'] == 'order-123: refund allowed'


@pytest.mark.vcr
@pytest.mark.moves_cache_prefix(reason='Gemini reveals deferred capability tools by changing its tools declaration')
async def test_google_deferred_capability_announcement_live(
    allow_model_requests: None, gemini_api_key: str, vcr: Any
) -> None:
    """Gemini 3.6 Flash reveals a capability through the channel-less announcement path."""
    pytest.importorskip('google.genai')
    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed'

    capability = Capability(
        id='refunds',
        description='Refund policy tools. Load this capability before looking up refund policy.',
        defer_loading=True,
        toolsets=[refunds_toolset],
    )
    model = GoogleModel('gemini-3.6-flash', provider=GoogleProvider(api_key=gemini_api_key))
    agent: Agent[None, str] = Agent(model=model, capabilities=[capability])
    result = await agent.run(
        'First load the refunds capability. Then use its newly available tool for order A-4417. Return only the result.'
    )

    request_bodies = [json.loads(request.body) for request in vcr.requests]
    assert len(request_bodies) >= 3
    before, reveal, *later = request_bodies

    def function_declarations(body: dict[str, Any]) -> list[dict[str, Any]]:
        return [declaration for tool in body.get('tools', []) for declaration in tool.get('functionDeclarations', [])]

    before_declarations = function_declarations(before)
    reveal_declarations = function_declarations(reveal)
    assert 'lookup_refund_policy' not in json.dumps(before)
    assert [declaration['name'] for declaration in reveal_declarations] == [
        *[declaration['name'] for declaration in before_declarations],
        'lookup_refund_policy',
    ]
    assert any('tool(s) are now available' in json.dumps(content) for content in reveal['contents'])
    assert all(function_declarations(body) == reveal_declarations for body in later)
    assert any(
        part.tool_name == 'lookup_refund_policy'
        for part in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
    )


@pytest.mark.vcr
async def test_openai_execution_client_round_trip(allow_model_requests: None, openai_api_key: str, vcr: Any) -> None:
    """End-to-end: a custom callable `ToolSearch` strategy surfaces natively on OpenAI
    Responses as `ToolSearchToolParam(execution='client')` — the provider emits a
    `tool_search_call` with `execution='client'` whose arguments we dispatch to the
    local `search_tools` function, and the resulting `ToolReturnPart` is replayed
    as a `tool_search_output` (execution='client') carrying the discovered tool defs."""

    def match_exchange_rate(ctx: RunContext, queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
        # Deterministic: always point the model at `get_exchange_rate` so the cassette
        # replay doesn't depend on the exact keywords the model picks.
        return ['get_exchange_rate']

    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
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
    # `ToolSearchReturnContent` — this is the same value read back by `ToolSearchToolset`
    # on later turns to unlock the deferred tool on the local path (and round-tripped as
    # `tool_search_output.tools` in the cassette's replay request body).
    search_returns = [
        part
        for part in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if part.tool_name == 'search_tools'
    ]
    assert len(search_returns) == 1
    assert search_returns[0].content == {'discovered_tools': [{'name': 'get_exchange_rate'}]}

    rate_returns = [
        part
        for part in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if part.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    # Wire-level: with a custom (client-executed) strategy, OpenAI rejects the `tool_search` tool
    # without a non-null `description`, so the adapter always sends one. Confirm the recorded
    # request carries it (guards the description default against a silent regression → live 422).
    request_tools = json.loads(vcr.requests[0].body)['tools']
    client_tool_search = [
        tool for tool in request_tools if tool.get('type') == 'tool_search' and tool.get('execution') == 'client'
    ]
    assert client_tool_search, 'expected a client-executed tool_search tool in the request'
    assert client_tool_search[0].get('description'), 'tool_search description must be non-null'


@pytest.mark.vcr
async def test_anthropic_native_tool_search_streaming(allow_model_requests: None, anthropic_api_key: str) -> None:
    """End-to-end streaming against live Anthropic: native BM25 server-side tool search
    streams `NativeToolSearchCallPart` / `NativeToolSearchReturnPart` through the part
    manager during `agent.iter` + `node.stream`, the model invokes the discovered
    deferred tool by its plain name, and the agent loop runs to a final text response."""
    pytest.importorskip('anthropic')

    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=anthropic_api_key))
    agent = Agent(model=model)

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
        p for m in agent_run.result.all_messages() for p in m.parts if isinstance(p, NativeToolSearchCallPart)
    ]
    builtin_return_parts = [
        p for m in agent_run.result.all_messages() for p in m.parts if isinstance(p, NativeToolSearchReturnPart)
    ]
    assert builtin_call_parts and builtin_return_parts

    # The discovered deferred tool dispatches by its plain name and produces its
    # ToolReturnPart end-to-end.
    rate_returns = [
        p
        for p in iter_message_parts(agent_run.result.all_messages(), ModelRequest, ToolReturnPart)
        if p.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    # We received streaming events from both the model-request node and the call-tools
    # node — i.e. the part manager surfaced the builtin tool-search parts as the stream
    # came in (not just on `streamed.get()`).
    assert streamed_events, 'expected streaming events from the request stream'


@pytest.mark.vcr
async def test_openai_native_tool_search_streaming(allow_model_requests: None, openai_api_key: str) -> None:
    """End-to-end streaming against live OpenAI Responses: native server-executed
    `tool_search` streams `NativeToolSearchCallPart` / `NativeToolSearchReturnPart`
    through the part manager during `agent.iter` + `node.stream`, the model invokes
    the discovered deferred tool by its plain name, and the agent loop runs to a final
    text response."""

    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)

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
        p for m in agent_run.result.all_messages() for p in m.parts if isinstance(p, NativeToolSearchCallPart)
    ]
    builtin_return_parts = [
        p for m in agent_run.result.all_messages() for p in m.parts if isinstance(p, NativeToolSearchReturnPart)
    ]
    assert len(builtin_call_parts) == len(builtin_return_parts) == 1
    assert builtin_call_parts[0].tool_call_id == builtin_return_parts[0].tool_call_id
    assert builtin_return_parts[0].content == {'discovered_tools': [{'name': 'get_exchange_rate'}]}
    assert builtin_return_parts[0].provider_details == {
        'id': 'tso_060f468708eb0ff90069f3e2f20c84819385b1d75b9c1ffc7d',
        'call_id': None,
        'execution': 'server',
        'status': 'completed',
    }

    rate_returns = [
        p
        for p in iter_message_parts(agent_run.result.all_messages(), ModelRequest, ToolReturnPart)
        if p.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    assert any(
        isinstance(event, PartStartEvent) and isinstance(event.part, NativeToolSearchCallPart)
        for event in streamed_events
    )
    assert any(
        isinstance(event, PartStartEvent) and isinstance(event.part, NativeToolSearchReturnPart)
        for event in streamed_events
    )


@pytest.mark.vcr
async def test_openai_client_tool_search_streaming(allow_model_requests: None, openai_api_key: str) -> None:
    """End-to-end streaming against live OpenAI Responses with a custom callable
    `ToolSearch` strategy. The provider emits a `tool_search_call` with
    `execution='client'` whose arguments we dispatch to the local `search_tools`
    function — both events surface through the streaming part manager (the
    `tool_search_call` as a regular `ToolCallPart`), the agent loop runs the
    callable strategy, the model follows up with the discovered deferred tool, and
    the run completes with a final text response."""

    def match_exchange_rate(ctx: RunContext, queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
        # Deterministic: always point the model at `get_exchange_rate` so the cassette
        # replay doesn't depend on the exact keywords the model picks.
        return ['get_exchange_rate']

    model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
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
        for part in iter_message_parts(agent_run.result.all_messages(), ModelRequest, ToolReturnPart)
        if part.tool_name == 'search_tools'
    ]
    assert len(search_returns) == 1
    assert search_returns[0].content == {'discovered_tools': [{'name': 'get_exchange_rate'}]}

    rate_returns = [
        part
        for part in iter_message_parts(agent_run.result.all_messages(), ModelRequest, ToolReturnPart)
        if part.tool_name == 'get_exchange_rate'
    ]
    assert len(rate_returns) == 1
    assert rate_returns[0].content == '1 USD = 0.92 EUR'

    assert streamed_events, 'expected streaming events from the request stream'


async def test_agent_graph_without_builtin_tools(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Covers `_agent_graph`'s empty `ctx.deps.native_tools` branch.

    Auto-inject always adds `ToolSearchTool`, so the only way to exercise the empty
    branch is to disable auto-inject in the test.
    """

    monkeypatch.setattr(agent_module, '_AUTO_INJECT_CAPABILITY_TYPES', ())
    agent = Agent('test')
    result = await agent.run('hi')
    assert isinstance(result.output, str)


async def test_tool_search_toolset_discovers_from_builtin_return_part():
    """Discovery metadata on a `NativeToolSearchReturnPart` from a native provider search
    is picked up so the local path recovers state on cross-provider handover."""

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset)

    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage'}]},
                )
            ]
        )
    ]
    # `parse_discovered_tools` extracts discovery from the native return part; mirror that
    # into `discovered_tool_names`, which request preparation uses for visibility.
    assert parse_discovered_tools(messages) == {'calculate_mortgage'}
    ctx = _build_run_context(None, messages=messages, discovered_tool_names={'calculate_mortgage'})

    tools = await searchable.get_tools(ctx)
    assert tools['calculate_mortgage'].tool_def.defer_loading is True
    assert tools['stock_price'].tool_def.defer_loading is True


async def test_tool_search_toolset_custom_search_fn_filters_unknown_names():
    """Names returned by `search_fn` that aren't in the deferred set are discarded."""

    def custom_search(ctx: RunContext, queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
        return ['stock_price', 'not_a_real_tool', 'crypto_price']

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset, search_fn=custom_search)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['anything']}, ctx, tools[_SEARCH_TOOLS_NAME])
    assert result == {
        'discovered_tools': [
            {'name': 'stock_price'},
            {'name': 'crypto_price'},
        ]
    }


async def test_tool_search_toolset_custom_search_fn_no_matches():
    """Custom search function returning no names produces the 'no matches' message."""

    def custom_search(ctx: RunContext, queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
        return []

    toolset = _create_function_toolset()
    searchable = ToolSearchToolset(wrapped=toolset, search_fn=custom_search)
    ctx = _build_run_context(None)

    tools = await searchable.get_tools(ctx)
    result = await searchable.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['anything']}, ctx, tools[_SEARCH_TOOLS_NAME])
    assert result == {
        'discovered_tools': [],
        'message': 'No matching tools found. The tools you need may not be available.',
    }


async def test_tool_search_capability_strategy_callable_registers_custom_builtin():
    """A callable strategy still registers a `ToolSearchTool` builtin with `strategy='custom'`
    so provider adapters that support a custom-callable native surface (e.g. Anthropic's
    `tool_reference` result blocks, OpenAI's `execution='client'`) can use it; models
    without support drop it as optional and fall back to the local `search_tools` tool."""

    def noop(ctx: RunContext, queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:  # pragma: no cover
        return []

    cap = ToolSearch(strategy=noop)
    builtins = list(cap.get_native_tools())
    assert len(builtins) == 1
    tool = builtins[0]
    assert isinstance(tool, ToolSearchTool)
    assert tool.strategy == 'custom'


async def test_tool_search_capability_strategy_named_registers_builtin():
    """Named native strategies register a non-optional `ToolSearchTool` — the request
    must error on models that can't honor the choice rather than silently substituting
    a local algorithm for bm25/regex."""
    cap = ToolSearch(strategy='regex')
    builtins = list(cap.get_native_tools())
    assert len(builtins) == 1
    tool = builtins[0]
    assert isinstance(tool, ToolSearchTool)
    assert tool.strategy == 'regex'
    assert tool.optional is False


async def test_tool_search_capability_strategy_none_optional_builtin():
    """The default (`None`) strategy registers an optional builtin so the local
    token-matching fallback takes over on models without native support."""
    cap = ToolSearch()
    builtins = list(cap.get_native_tools())
    assert len(builtins) == 1
    tool = builtins[0]
    assert isinstance(tool, ToolSearchTool)
    assert tool.strategy is None
    assert tool.optional is True


async def test_tool_search_capability_wraps_with_tool_search_toolset():
    """`strategy='keywords'` wraps with `ToolSearchToolset` so the corpus is
    exposed and `search_tools` carries the user's customizations. The toolset's
    `search_fn` is set to the built-in keyword-overlap algorithm so the local
    dispatch routes through `_run_search_fn` (same path as a custom callable),
    enabling client-executed-native wire on supporting providers."""
    toolset = _create_function_toolset()
    cap = ToolSearch(strategy='keywords')
    wrapped = cap.get_wrapper_toolset(toolset)
    assert isinstance(wrapped, ToolSearchToolset)
    assert wrapped.search_fn is not None


async def test_tool_search_capability_named_strategy_wraps_with_tool_search_toolset():
    """Named native strategies (bm25/regex) still wrap with `ToolSearchToolset` so
    the corpus is exposed; `prepare_request` raises on unsupported models because the
    builtin is registered with `optional=False`."""
    toolset = _create_function_toolset()
    cap = ToolSearch(strategy='bm25')
    wrapped = cap.get_wrapper_toolset(toolset)
    assert isinstance(wrapped, ToolSearchToolset)
    assert wrapped.search_fn is None


async def test_tool_search_named_strategy_raises_on_unsupported_model():
    """Named native strategies error on models that don't support `ToolSearchTool`
    — there's no legal fallback for `strategy='bm25'` on e.g. GPT-4."""

    m = TestModel()
    with pytest.raises(UserError, match='not supported by this model'):
        m.prepare_request(
            None,
            ModelRequestParameters(function_tools=[], native_tools=[ToolSearchTool(strategy='bm25')]),
        )


@pytest.mark.parametrize('strategy', ['bm25', 'regex'])
async def test_tool_search_named_strategy_agent_run_raises_on_unsupported_model(strategy: str):
    """End-to-end: `ToolSearch(strategy='bm25'|'regex')` on a model without native
    tool-search support must raise `UserError` rather than silently substituting the
    local keyword-overlap algorithm. The capability promises that named-native strategies
    error on adapters that can't honor the choice; previously the toolset always
    registered the local `search_tools` function as a fallback, which masked the
    error by letting `_resolve_builtin_tool_swap` drop the optional-False builtin."""
    agent = Agent(TestModel(), capabilities=[ToolSearch(strategy=cast(Any, strategy))])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:  # pragma: no cover
        return f'Weather in {city}'

    with pytest.raises(UserError, match=r'ToolSearchTool.*not supported by this model'):
        await agent.run('what should I wear?')


async def test_tool_search_keywords_agent_run_falls_back_on_unsupported_model():
    """Inverse of the named-strategy test: `strategy='keywords'` has a local
    implementation, so the request must fall back silently on a model without native
    tool-search support — running the agent should not raise."""
    agent = Agent(TestModel(), capabilities=[ToolSearch(strategy='keywords')])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:  # pragma: no cover
        return f'Weather in {city}'

    # `TestModel` doesn't support `ToolSearchTool`; with a local fallback available
    # this should run without error.
    result = await agent.run('hello')
    assert result.output


@pytest.mark.parametrize('strategy', ['bm25', 'regex'])
async def test_tool_search_named_strategy_skips_local_search_tools_emission(strategy: str):
    """Named-native strategies (`'bm25'`/`'regex'`) construct the toolset with
    `enable_fallback=False`; `get_tools` then skips emitting the local `search_tools`
    function tool entirely. Two effects fall out:

    * On *supported* providers (Anthropic), the wire carries only the native
      `tool_search_tool_*` builtin — no redundant local function tool that could
      confuse the model or waste a tool slot.
    * On *unsupported* providers, `_resolve_builtin_tool_swap` has no fallback to count
      against the (non-optional) builtin and raises `UserError` as promised."""
    toolset = _create_function_toolset()
    cap = ToolSearch(strategy=cast(Any, strategy))
    wrapped = cap.get_wrapper_toolset(toolset)
    assert isinstance(wrapped, ToolSearchToolset)
    assert wrapped.enable_fallback is False

    ctx = _build_run_context(None)
    tools = await wrapped.get_tools(ctx)
    # `search_tools` is omitted entirely — the deferred corpus is still exposed by name
    # (carrying `with_native='tool_search'`) so the swap logic can route discovery.
    assert _SEARCH_TOOLS_NAME not in tools
    corpus_names = {name for name, t in tools.items() if t.tool_def.with_native == 'tool_search'}
    assert corpus_names == {'calculate_mortgage', 'stock_price', 'crypto_price'}


async def test_tool_search_keywords_ignores_builtin_support():
    """`strategy='keywords'` never tries to use a native builtin — the swap is a
    no-op even on models that support `ToolSearchTool`."""

    class ToolSearchTestModel(TestModel):
        @classmethod
        def supported_native_tools(cls):
            return frozenset({ToolSearchTool})

    m = ToolSearchTestModel()
    search_tool = ToolDefinition(name=_SEARCH_TOOLS_NAME, description='local', parameters_json_schema={})
    _, prepared = m.prepare_request(
        None,
        ModelRequestParameters(function_tools=[search_tool], native_tools=[]),
    )
    assert prepared.native_tools == []
    assert [t.name for t in prepared.function_tools] == [_SEARCH_TOOLS_NAME]


def test_with_native_undiscovered_drops_on_unsupported_model():
    """In `prepare_request`, `with_native` corpus members with `defer_loading=True`
    (still undiscovered) drop on a model that doesn't support the builtin — the model has
    no way to call them and the local `search_tools` fallback handles discovery."""

    m = TestModel()
    # `optional=True` models the default auto path where the builtin is a best-effort
    # upgrade; on a model that doesn't support it, both the builtin and its undiscovered
    # corpus drop so the local `ToolSearch` fallback handles discovery.
    search_builtin = ToolSearchTool(optional=True)
    corpus_tool = ToolDefinition(name='deferred_tool', with_native='tool_search', defer_loading=True)

    _, prepared = m.prepare_request(
        None,
        ModelRequestParameters(
            function_tools=[corpus_tool],
            native_tools=[search_builtin],
        ),
    )
    assert prepared.native_tools == []
    assert prepared.tool_visibility == {'deferred_tool': 'withheld'}
    assert prepared.declared_function_tools == []


def test_with_native_discovered_kept_on_unsupported_model():
    """A revealed corpus member stays in the request even when
    the builtin is unsupported — the model can call it directly by name on the local path."""

    m = TestModel()
    corpus_tool = ToolDefinition(name='deferred_tool', with_native='tool_search', defer_loading=True)

    _, prepared = m.prepare_request(
        None,
        ModelRequestParameters(
            function_tools=[corpus_tool],
            native_tools=[ToolSearchTool(optional=True)],
            revealed_tool_names={'deferred_tool'},
        ),
    )
    assert prepared.native_tools == []
    assert [t.name for t in prepared.function_tools] == ['deferred_tool']


def test_with_native_kept_on_supporting_model():
    """On a supporting model, managed tools are kept so the adapter can emit them
    with provider-specific wire-format tweaks."""

    class ToolSearchTestModel(TestModel):
        @classmethod
        def supported_native_tools(cls):
            return frozenset({ToolSearchTool})

    m = ToolSearchTestModel()
    corpus_tool = ToolDefinition(name='deferred_tool', with_native='tool_search')
    _, prepared = m.prepare_request(
        None,
        ModelRequestParameters(
            function_tools=[corpus_tool],
            native_tools=[ToolSearchTool()],
        ),
    )
    assert [t.name for t in prepared.function_tools] == ['deferred_tool']
    assert any(isinstance(t, ToolSearchTool) for t in prepared.native_tools)


def test_optional_builtin_dropped_with_empty_corpus():
    """An `optional` builtin is silently dropped when no managed corpus is in the request."""

    class ToolSearchTestModel(TestModel):
        @classmethod
        def supported_native_tools(cls):
            return frozenset({ToolSearchTool})

    m = ToolSearchTestModel()
    _, prepared = m.prepare_request(
        None,
        ModelRequestParameters(function_tools=[], native_tools=[ToolSearchTool(optional=True)]),
    )
    assert prepared.native_tools == []


def test_narrow_type_promotes_builtin_call_to_tool_search() -> None:
    """Direct construction of `NativeToolCallPart` with `tool_kind='tool-search'`
    promotes to `NativeToolSearchCallPart` via the narrowing registry."""
    base = NativeToolCallPart(
        tool_name='tool_search',
        args={'queries': ['mortgage']},
        tool_call_id='c1',
        tool_kind='tool-search',
        provider_name='anthropic',
        provider_details={'strategy': 'bm25'},
    )
    narrowed = NativeToolCallPart.narrow_type(base)
    assert isinstance(narrowed, NativeToolSearchCallPart)
    assert narrowed.args == {'queries': ['mortgage']}
    assert narrowed.tool_call_id == 'c1'
    assert narrowed.provider_name == 'anthropic'
    assert narrowed.provider_details == {'strategy': 'bm25'}

    already_narrowed = NativeToolSearchCallPart(args={'queries': ['x']}, tool_call_id='c2')
    assert NativeToolCallPart.narrow_type(already_narrowed) is already_narrowed


def test_narrow_type_promotes_builtin_return_to_tool_search() -> None:
    """Direct construction of `NativeToolReturnPart` with `tool_kind='tool-search'`
    promotes to `NativeToolSearchReturnPart` via the narrowing registry."""
    base = NativeToolReturnPart(
        tool_name='tool_search',
        content={'discovered_tools': [{'name': 'foo'}]},
        tool_call_id='c1',
        tool_kind='tool-search',
        provider_name='anthropic',
    )
    narrowed = NativeToolReturnPart.narrow_type(base)
    assert isinstance(narrowed, NativeToolSearchReturnPart)
    assert narrowed.content == {'discovered_tools': [{'name': 'foo'}]}

    already_narrowed = NativeToolSearchReturnPart(content={'discovered_tools': []}, tool_call_id='c2')
    assert NativeToolReturnPart.narrow_type(already_narrowed) is already_narrowed


def test_narrow_type_unknown_tool_kind_returns_input_unchanged() -> None:
    """Unknown `tool_kind` values aren't promoted (future builtins not yet typed)."""
    base = NativeToolCallPart(
        tool_name='something_unregistered',
        args={},
        tool_call_id='c1',
        tool_kind=cast('ToolPartKind', 'custom_kind'),  # forward-compat: discriminator unknown to the current registry
    )
    assert NativeToolCallPart.narrow_type(base) is base


def test_narrow_type_no_tool_kind_returns_input_unchanged() -> None:
    """User-defined tools sharing a framework `tool_name` aren't promoted when `tool_kind` is unset.

    Protects users whose own tool happens to be called `tool_search` / `search_tools` from
    having their parts promoted to typed subclasses that would fail shape validation against
    the typed `args` `TypedDict`.
    """
    builtin_collision = NativeToolCallPart(tool_name='tool_search', args={'foo': 'bar'}, tool_call_id='c1')
    assert builtin_collision.tool_kind is None
    assert NativeToolCallPart.narrow_type(builtin_collision) is builtin_collision

    local_collision = ToolCallPart(tool_name='search_tools', args={'query': 'x'}, tool_call_id='c2')
    assert local_collision.tool_kind is None
    assert ToolCallPart.narrow_type(local_collision) is local_collision


def test_model_response_dict_round_trip_promotes_typed_subclasses() -> None:
    """Pydantic deserialization of a dict-shaped `ModelResponse` promotes
    `tool_search` builtin parts to typed subclasses via the discriminator."""

    raw: dict[str, Any] = {
        'kind': 'response',
        'parts': [
            {
                'part_kind': 'builtin-tool-call',
                'tool_name': 'tool_search',
                'tool_kind': 'tool-search',
                'args': {'queries': ['mortgage']},
                'tool_call_id': 'c1',
                'provider_name': 'anthropic',
            },
            {
                'part_kind': 'builtin-tool-return',
                'tool_name': 'tool_search',
                'tool_kind': 'tool-search',
                'content': {'discovered_tools': [{'name': 'foo'}]},
                'tool_call_id': 'c1',
                'provider_name': 'anthropic',
            },
            {
                'part_kind': 'builtin-tool-call',
                'tool_name': 'web_search',
                'args': {'query': 'x'},
                'tool_call_id': 'c2',
            },
            # User-defined builtin call colliding with a framework tool_name. Without
            # `tool_kind`, dispatch should NOT promote — args don't match `ToolSearchArgs`.
            {
                'part_kind': 'builtin-tool-call',
                'tool_name': 'tool_search',
                'args': {'foo': 'bar'},
                'tool_call_id': 'c3',
            },
        ],
    }
    resp = message(ModelMessagesTypeAdapter.validate_python([raw]), ModelResponse)
    assert isinstance(resp.parts[0], NativeToolSearchCallPart)
    assert isinstance(resp.parts[1], NativeToolSearchReturnPart)
    # Unrecognized `tool_name` (and unset `tool_kind`) falls through to the base class.
    assert isinstance(resp.parts[2], NativeToolCallPart)
    assert not isinstance(resp.parts[2], NativeToolSearchCallPart)
    # User-defined collision on `tool_name='tool_search'` without `tool_kind` stays base.
    assert type(resp.parts[3]) is NativeToolCallPart
    assert resp.parts[3].args == {'foo': 'bar'}


def test_model_response_instance_round_trip_promotes_typed_subclasses() -> None:
    """Re-validation of a `ModelResponse` instance preserves typed builtin parts."""

    resp = ModelResponse(
        parts=[
            NativeToolSearchCallPart(args={'queries': ['x']}, tool_call_id='c1'),
            NativeToolSearchReturnPart(
                content={'discovered_tools': [{'name': 'foo'}]},
                tool_call_id='c1',
            ),
            NativeToolCallPart(tool_name='web_search', args={}, tool_call_id='c2'),
        ]
    )
    revalidated = message(ModelMessagesTypeAdapter.validate_python([resp]), ModelResponse)
    assert isinstance(revalidated.parts[0], NativeToolSearchCallPart)
    assert isinstance(revalidated.parts[1], NativeToolSearchReturnPart)
    assert isinstance(revalidated.parts[2], NativeToolCallPart)


async def test_tool_search_toolset_protects_user_collision_on_builtin_tool_name() -> None:
    """A user-emitted `NativeToolReturnPart` with `tool_name='tool_search'` (no typed
    subclass, no `tool_kind`) is left alone — discoveries are only surfaced from typed
    `NativeToolSearchReturnPart` instances. This is the typed-trust contract: the
    framework constructs typed subclasses; user collisions on names alone don't get
    treated as our search payload."""

    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                # Framework-emitted: typed subclass surfaces discoveries.
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage'}]},
                    tool_call_id='c1',
                ),
                # User collision on the name with a base part — `tool_kind=None`, not a typed
                # subclass: NOT surfaced.
                NativeToolReturnPart(
                    tool_name='tool_search',
                    content={'discovered_tools': [{'name': 'should_not_surface'}]},
                    tool_call_id='c2',
                ),
            ],
        ),
    ]

    discovered = parse_discovered_tools(history)
    assert 'calculate_mortgage' in discovered
    assert 'should_not_surface' not in discovered


async def test_local_tool_search_stream_emits_typed_call_part_from_first_event() -> None:
    """Streaming counterpart to the non-streaming typed-parts test. The model streams a
    `search_tools` call name + args delta-by-delta; `ModelResponsePartsManager` materializes
    the call part as the typed `ToolSearchCallPart` from the first `PartStartEvent` rather
    than only after a post-stream pass. This relies on the parts manager receiving
    `model_request_parameters` (set on `StreamedResponse.__post_init__`) so it can look up
    `ToolDefinition.tool_kind` for the called tool name.

    Forces the local-fallback path by using a model that doesn't claim native
    `ToolSearchTool` support — otherwise the swap drops `search_tools` from
    `function_tools` (Rule 1) on the assumption the model handles tool search
    server-side via the native wire shape.
    """

    class NoNativeToolSearchModel(FunctionModel):
        """A `FunctionModel` that drops `ToolSearchTool` from its supported builtins so the
        framework routes through the local `search_tools` function tool rather than the
        native wire shape."""

        @classmethod
        def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
            return frozenset(super().supported_native_tools()) - {ToolSearchTool}

    toolset: FunctionToolset = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def calculate_mortgage(principal: float, rate: float, years: int) -> str:  # pragma: no cover
        return f'${principal * rate * years:.2f}'

    call_count = 0

    async def stream_function(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield {0: DeltaToolCall(name='search_tools', tool_call_id='c1')}
            yield {0: DeltaToolCall(json_args='{"queries":')}
            yield {0: DeltaToolCall(json_args='["mortgage"]}')}
        else:
            yield 'done'

    agent = Agent(
        NoNativeToolSearchModel(stream_function=stream_function), toolsets=[toolset], capabilities=[ToolSearch()]
    )

    typed_at_start: list[bool] = []

    async def event_stream_handler(_ctx: RunContext, stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            if (
                isinstance(event, PartStartEvent)
                and isinstance(event.part, ToolCallPart)
                and event.part.tool_name == 'search_tools'
            ):
                typed_at_start.append(isinstance(event.part, ToolSearchCallPart))

    await agent.run('find a mortgage tool', event_stream_handler=event_stream_handler)

    # The first PartStartEvent for the search_tools call already carries the typed identity.
    assert typed_at_start, 'expected a PartStartEvent for search_tools during streaming'
    assert all(typed_at_start), f'expected typed `ToolSearchCallPart` from first event; got {typed_at_start}'


async def test_local_tool_search_dispatch_produces_typed_parts() -> None:
    """End-to-end typed identity for the local `search_tools` path: the model emits a
    base `ToolCallPart`, the framework promotes it to `ToolSearchCallPart` via the
    declared `ToolDefinition.tool_kind`, dispatches to `ToolSearchToolset`, and constructs
    a typed `ToolSearchReturnPart`. Both halves of the exchange carry the typed identity
    so multi-turn discovery parsing and cross-provider replay see typed parts everywhere.

    Reported by Devin's review of commit 53eb27b06 for the return side: previously the
    framework constructed a base `ToolReturnPart` (no `tool_kind`), and neither
    `parse_discovered_tools`' isinstance check nor the legacy-metadata reader caught
    it, so previously-discovered tools reverted to hidden on every subsequent turn.
    """
    toolset: FunctionToolset = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def calculate_mortgage(principal: float, rate: float, years: int) -> str:  # pragma: no cover
        return f'${principal * rate * years:.2f}'

    call_count = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='search_tools', args={'queries': ['mortgage']})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_function), toolsets=[toolset], capabilities=[ToolSearch()])
    result = await agent.run('find a mortgage tool')

    # The framework-promoted call part is typed (via `_narrow_tool_call_parts` post-hook).
    search_calls = [
        part
        for part in iter_message_parts(result.all_messages(), ModelResponse, ToolCallPart)
        if part.tool_name == 'search_tools'
    ]
    assert len(search_calls) == 1
    assert isinstance(search_calls[0], ToolSearchCallPart)
    assert search_calls[0].tool_kind == 'tool-search'

    # The framework-constructed return part is typed (via `_call_tool` dispatch hook).
    search_returns = [
        part
        for part in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if part.tool_name == 'search_tools'
    ]
    assert len(search_returns) == 1
    assert isinstance(search_returns[0], ToolSearchReturnPart)
    assert search_returns[0].tool_kind == 'tool-search'
    # And the typed content carries the discovery.
    content = search_returns[0].content
    assert {m['name'] for m in content['discovered_tools']} == {'calculate_mortgage'}


async def test_tool_search_toolset_replays_main_branch_legacy_shape() -> None:
    """Histories serialized on `main` (before this PR's typed-content shape) carry the
    discovered names on `ToolReturnPart.metadata['discovered_tools']` rather than on a
    typed `content`. They must continue to replay cleanly on the typed-parts shape so
    upgrading users don't lose discovered-tool state on the next turn.

    This is the wire shape on the `main` branch as of the merge-base.
    """

    history: list[ModelMessage] = [
        ModelResponse(
            parts=[ToolCallPart(tool_name='search_tools', args={'queries': ['mortgage']}, tool_call_id='c1')]
        ),
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
    discovered = parse_discovered_tools(history)
    assert discovered == {'calculate_mortgage'}


def test_synthetic_injection_translates_builtin_to_local_tool_search_parts() -> None:
    """Cross-provider replay end-to-end: a `NativeToolSearch*Part` carried over from
    a prior native turn is translated into the local-shape typed parts so a non-native
    adapter can replay it as a normal `search_tools` function-call exchange. The
    `parse_discovered_tools` then surfaces the discoveries via the
    discriminated-union dispatch."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Find me a mortgage tool.')]),
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': ['mortgage']},
                    tool_call_id='c1',
                    provider_name='anthropic',
                    provider_details={'strategy': 'bm25'},
                ),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage'}]},
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
    response = message(translated, ModelResponse, index=1)
    assert len(response.parts) == 1
    call_part = message_part(translated, ToolSearchCallPart, message_index=1)
    # Subclass of `ToolCallPart`, NOT `NativeToolSearchCallPart`.
    assert isinstance(call_part, ToolCallPart)
    assert not isinstance(call_part, NativeToolSearchCallPart)
    assert call_part.tool_name == 'search_tools'
    assert call_part.args == {'queries': ['mortgage']}

    return_part = message_part(translated, ToolSearchReturnPart, message_index=2)
    assert isinstance(return_part, ToolReturnPart)
    assert not isinstance(return_part, NativeToolSearchReturnPart)
    assert return_part.tool_name == 'search_tools'
    assert return_part.content == {'discovered_tools': [{'name': 'calculate_mortgage'}]}

    # And the toolset's parser surfaces the discovery off the translated history.
    discovered = parse_discovered_tools(translated)
    assert discovered == {'calculate_mortgage'}


def test_synthesize_local_promotes_base_tool_return_with_tool_kind_in_request() -> None:
    """`synthesize_local_tool_search_messages` also reaches into existing `ModelRequest`
    parts: a base `ToolReturnPart` carrying `tool_kind='tool-search'` (e.g. one
    constructed manually before going through the discriminator) is promoted to its
    typed `ToolSearchReturnPart` subclass in place. Mirrors the response-side
    promotion so cross-provider history stays uniformly typed regardless of where
    the parts originated."""

    history: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart(tool_name='search_tools', args={'queries': ['a']}, tool_call_id='c1')]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='search_tools',
                    content={'discovered_tools': [{'name': 'foo'}]},
                    tool_call_id='c1',
                    tool_kind='tool-search',
                ),
            ],
        ),
    ]
    translated = synthesize_local_tool_search_messages(history)
    part = message_part(translated, ToolSearchReturnPart, message_index=1)
    assert part.content == {'discovered_tools': [{'name': 'foo'}]}


async def test_tool_search_toolset_uses_custom_parameter_description() -> None:
    """`ToolSearch(parameter_description=...)` flows through to the local `search_tools`
    function tool's `queries` parameter description on the wire — verifies the
    custom-description branch in `_build_search_args_schema` rebuilds the JSON
    schema rather than reusing the default."""
    cap = ToolSearch(parameter_description='custom queries hint')
    base_toolset = _create_function_toolset()
    wrapped = cap.get_wrapper_toolset(base_toolset)
    ctx = _build_run_context(None)
    tools = await wrapped.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]
    schema = search_tool.tool_def.parameters_json_schema
    assert schema['properties']['queries']['description'] == 'custom queries hint'


def test_prepare_messages_translates_on_non_native_model() -> None:
    """`Model.prepare_messages` is the centralized hook that runs before the adapter's
    message-prep on every request. On a model whose profile doesn't include
    `ToolSearchTool` in `supported_native_tools`, the hook translates any prior
    server-side tool-search exchange into the local-shape typed parts so the adapter
    sees a normal `search_tools` function-call exchange.

    The single `ModelResponse(call+return)` carrying the inline server-side result
    splits into `ModelResponse(call) + ModelRequest(return)`."""
    # Default `TestModel` excludes `ToolSearchTool` from `supported_native_tools`.
    model = TestModel()
    assert ToolSearchTool not in model.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Find me a mortgage tool.')]),
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': ['mortgage']},
                    tool_call_id='c1',
                    provider_name='anthropic',
                ),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage'}]},
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

    response = message(prepared, ModelResponse, index=1)
    assert len(response.parts) == 1
    call_part = message_part(prepared, ToolSearchCallPart, message_index=1)
    assert not isinstance(call_part, NativeToolSearchCallPart)
    assert call_part.tool_name == 'search_tools'

    return_part = message_part(prepared, ToolSearchReturnPart, message_index=2)
    assert not isinstance(return_part, NativeToolSearchReturnPart)
    assert return_part.tool_name == 'search_tools'
    assert return_part.content == {'discovered_tools': [{'name': 'calculate_mortgage'}]}


@pytest.mark.parametrize(
    ('origin_provider_name', 'translated'),
    [('test', False), ('anthropic', True), ('openai', True), (None, True)],
)
def test_prepare_messages_on_native_model(origin_provider_name: str | None, translated: bool) -> None:
    """A native model preserves its own parts and translates foreign or untagged parts."""

    class NativeToolSearchTestModel(TestModel):
        @classmethod
        def supported_native_tools(cls):
            return frozenset({ToolSearchTool})

    model = NativeToolSearchTestModel()
    assert ToolSearchTool in model.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Find me a mortgage tool.')]),
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': ['mortgage']},
                    tool_call_id='c1',
                    provider_name=origin_provider_name,
                ),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'calculate_mortgage'}]},
                    tool_call_id='c1',
                    provider_name=origin_provider_name,
                ),
            ],
        ),
    ]

    prepared = model.prepare_messages(history)

    if translated:
        assert isinstance(message_part(prepared, ToolSearchCallPart, message_index=1), ToolSearchCallPart)
        assert isinstance(message_part(prepared, ToolSearchReturnPart, message_index=2), ToolSearchReturnPart)
    else:
        assert prepared is history


def test_narrow_type_local_promotes_with_tool_kind_set() -> None:
    """A `ToolCallPart` with `tool_kind='tool-search'` promotes to `ToolSearchCallPart`.

    Promotion is keyed on `tool_kind`, not `tool_name` — a framework-emitted call carries
    `tool_kind='tool-search'` so it round-trips as the typed subclass.
    """

    part = ToolCallPart(
        tool_name='search_tools',
        args={'queries': ['mortgage']},
        tool_call_id='c1',
        tool_kind='tool-search',
    )
    narrowed = ToolCallPart.narrow_type(part)
    assert isinstance(narrowed, ToolSearchCallPart)
    assert narrowed.args == {'queries': ['mortgage']}


def test_narrow_type_local_passthrough_when_already_narrowed() -> None:
    """Narrowing an already-typed `ToolSearchCallPart` returns the input instance."""
    part = ToolSearchCallPart(args={'queries': ['x']}, tool_call_id='c1')

    assert ToolCallPart.narrow_type(part) is part


def test_pydantic_validation_accepts_search_tools_collision_when_tool_kind_unset() -> None:
    """A user-defined tool literally named `search_tools` with arbitrary args is safe.

    Dispatch is by `tool_kind`, not `tool_name`, so the absence of `tool_kind` keeps
    the part as a base `ToolReturnPart` regardless of args shape — no accidental
    auto-promotion to `ToolSearchReturnPart`, no spurious shape-validation failure.
    """

    raw = [
        {
            'kind': 'request',
            'parts': [
                {
                    'part_kind': 'tool-return',
                    'tool_name': 'search_tools',
                    # Arbitrary user-tool shape.
                    'content': {'unrelated': 'data', 'definitely_not_discovered_tools': 42},
                    'tool_call_id': 'c1',
                },
            ],
        },
    ]
    [req] = ModelMessagesTypeAdapter.validate_python(raw)
    [part] = req.parts
    assert type(part) is ToolReturnPart
    assert part.tool_kind is None
    assert part.content == {'unrelated': 'data', 'definitely_not_discovered_tools': 42}


def test_pydantic_validation_promotes_local_tool_return_with_tool_kind_set() -> None:
    """A serialized `tool-return` carrying `tool_kind='tool-search'` and a typed-shape
    `discovered_tools` payload is promoted to `ToolSearchReturnPart` by Pydantic's
    discriminated-union dispatch — the discriminator routes (part_kind, tool_kind)
    to the typed tag so deserialization rebuilds the typed subclass directly."""

    raw = [
        {
            'kind': 'request',
            'parts': [
                {
                    'part_kind': 'tool-return',
                    'tool_name': 'search_tools',
                    'tool_kind': 'tool-search',
                    'content': {'discovered_tools': [{'name': 'foo'}]},
                    'tool_call_id': 'c1',
                },
            ],
        },
    ]
    part = message_part(ModelMessagesTypeAdapter.validate_python(raw), ToolSearchReturnPart)
    assert part.content == {'discovered_tools': [{'name': 'foo'}]}


def test_pydantic_validation_accepts_search_tools_string_content_collision() -> None:
    """A user tool literally named `search_tools` returning plain text deserializes cleanly.

    Without `tool_kind`, the part stays a base `ToolReturnPart` — the str content survives
    intact. This is the user-tool-collision-tolerance contract: dispatch never promotes
    based on `tool_name` alone.
    """

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
    assert type(part) is ToolReturnPart
    assert part.tool_kind is None
    assert part.content == 'hello world'


def test_synthesize_local_from_native_call_str_args_passthrough() -> None:
    """Streaming partial-args (`str`) are passed through unchanged when translating."""

    part = NativeToolSearchCallPart(args='{"queries":', tool_call_id='c1')
    result = synthesize_local_from_native_call(part)
    assert result.args == '{"queries":'
    assert result.tool_call_id == 'c1'


def test_synthesize_local_from_native_call_none_args_falls_through() -> None:
    """`None` args remain `None` after translation."""

    part = NativeToolSearchCallPart(args=None, tool_call_id='c1')
    result = synthesize_local_from_native_call(part)
    assert result.args is None


def test_synthesize_messages_response_with_only_call_part_no_lift() -> None:
    """A response with only a `NativeToolSearchCallPart` (no return — streaming case)
    translates the call but doesn't synthesize a trailing `ModelRequest`."""

    history: list[ModelMessage] = [
        ModelResponse(parts=[NativeToolSearchCallPart(args={'queries': ['x']}, tool_call_id='c1')]),
    ]
    result = synthesize_local_tool_search_messages(history)
    assert len(result) == 1
    response = message(result, ModelResponse)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], ToolSearchCallPart)


def test_synthesize_messages_response_with_only_return_part_no_response_kept() -> None:
    """A response with only a `NativeToolSearchReturnPart` (no remaining parts) — the
    response is dropped since it'd be empty, and the return is lifted onto a fresh request."""

    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'foo'}]},
                    tool_call_id='c1',
                ),
            ],
        ),
    ]
    result = synthesize_local_tool_search_messages(history)
    assert len(result) == 1
    request = message(result, ModelRequest)
    assert len(request.parts) == 1
    message_part(result, ToolSearchReturnPart)


def test_synthesize_messages_request_with_unrelated_tool_return_passthrough() -> None:
    """A `ToolReturnPart` with `tool_name != 'search_tools'` doesn't get promoted —
    the request is returned unchanged."""

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
    Devin's observation. Splitting at every `NativeToolSearchReturn` boundary fixes
    both: the timeline reads correctly and the lifted return doesn't collide with the
    next request.
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Look up something then call it.')]),
        ModelResponse(
            parts=[
                TextPart(content='Searching first.'),
                NativeToolSearchCallPart(
                    args={'queries': ['weather']},
                    tool_call_id='search1',
                    provider_name='anthropic',
                ),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_weather'}]},
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

    first_req = message(result, ModelRequest)
    assert first_req is history[0]

    # First synthetic response: text + search call only — NOT the downstream weather call.
    first_resp = message(result, ModelResponse, index=1)
    assert len(first_resp.parts) == 2
    assert isinstance(first_resp.parts[0], TextPart)
    assert isinstance(first_resp.parts[1], ToolSearchCallPart)
    # No `ToolCallPart(weather)` snuck onto this response.
    assert not any(isinstance(p, ToolCallPart) and not isinstance(p, ToolSearchCallPart) for p in first_resp.parts)

    # Lifted search return as a fresh request.
    search_return_req = message(result, ModelRequest, index=2)
    assert len(search_return_req.parts) == 1
    assert isinstance(search_return_req.parts[0], ToolSearchReturnPart)

    # Second synthetic response: weather call only.
    second_resp = message(result, ModelResponse, index=3)
    assert len(second_resp.parts) == 1
    weather_call = message_part(result, ToolCallPart, message_index=3)
    assert weather_call.tool_name == 'get_weather'

    # Original weather-return request flows naturally — no consecutive `ModelRequest`s.
    last_req = message(result, ModelRequest, index=4)
    assert last_req is history[2]


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
                NativeToolSearchCallPart(args={'queries': ['x']}, tool_call_id='s1'),
                NativeToolSearchReturnPart(content={'discovered_tools': []}, tool_call_id='s1'),
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
                NativeToolSearchCallPart(args={'queries': ['a']}, tool_call_id='s1'),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'tool_a'}]},
                    tool_call_id='s1',
                ),
                NativeToolSearchCallPart(args={'queries': ['b']}, tool_call_id='s2'),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'tool_b'}]},
                    tool_call_id='s2',
                ),
            ],
        ),
    ]

    result = synthesize_local_tool_search_messages(history)

    # 4 messages: response[call_a], request[return_a], response[call_b], request[return_b].
    assert len(result) == 4
    call_part_1 = message_part(result, ToolSearchCallPart)
    assert call_part_1.tool_call_id == 's1'

    return_part_1 = message_part(result, ToolSearchReturnPart, message_index=1)
    assert return_part_1.tool_call_id == 's1'

    call_part_2 = message_part(result, ToolSearchCallPart, message_index=2)
    assert call_part_2.tool_call_id == 's2'

    return_part_2 = message_part(result, ToolSearchReturnPart, message_index=3)
    assert return_part_2.tool_call_id == 's2'


def test_synthesize_messages_metadata_kept_on_first_split_only() -> None:
    """Splitting one native `ModelResponse` into multiple responses must not duplicate
    its identity-level metadata (`provider_response_id`, usage). The first split keeps
    the original identity; subsequent splits get fresh/blank fields so downstream
    consumers don't double-count usage or find two responses for the same API call.
    """

    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(args={'queries': ['x']}, tool_call_id='s1'),
                NativeToolSearchReturnPart(content={'discovered_tools': []}, tool_call_id='s1'),
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


def test_prepare_messages_then_clean_history_merges_consecutive_requests() -> None:
    """Regression: the bare `[SearchCall, SearchReturn]` response shape — common when a model
    finishes a turn right after server-side search results — splits into `Response + Request`,
    which collides with the next `ModelRequest` in the history. `_clean_message_history` must
    run *after* `prepare_messages` so the splitter's synthetic `Request([SearchReturn])` and
    the original `Request([UserPromptPart])` merge into a single `ModelRequest`, preserving
    strict user/assistant alternation for adapters that require it.
    """

    history: list[ModelMessage] = [
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(args={'queries': ['x']}, tool_call_id='s1'),
                NativeToolSearchReturnPart(content={'discovered_tools': []}, tool_call_id='s1'),
            ],
        ),
        ModelRequest(parts=[UserPromptPart(content='follow-up')]),
    ]

    # Mirror `_make_request`'s post-fix order: synthesize first, then clean. Without the cleanup
    # pass, the synthesizer produces `Response, Request, Request` (a synthetic request for the
    # search return next to the original request); the clean pass merges those two requests so
    # adapters with strict user/assistant alternation see `Response, Request`.
    after_synthesis = synthesize_local_tool_search_messages(history)
    assert [type(m).__name__ for m in after_synthesis] == ['ModelResponse', 'ModelRequest', 'ModelRequest']

    cleaned = _clean_message_history(after_synthesis)
    assert [type(m).__name__ for m in cleaned] == ['ModelResponse', 'ModelRequest']

    # The merged request carries both the synthetic search return and the original user prompt,
    # with the tool return part sorted ahead of the user prompt.
    last = message(cleaned, ModelRequest, index=-1)
    assert isinstance(last.parts[0], ToolSearchReturnPart)
    assert isinstance(last.parts[1], UserPromptPart)


def test_narrow_type_local_return_passthrough_when_already_narrowed() -> None:
    """Narrowing an already-typed `ToolSearchReturnPart` returns the input instance."""
    part = ToolSearchReturnPart(content={'discovered_tools': []}, tool_call_id='c1')
    assert ToolReturnPart.narrow_type(part) is part


def test_narrow_type_local_return_promotes_with_tool_kind_set() -> None:
    """A base `ToolReturnPart` with `tool_kind='tool-search'` and a valid typed-content
    payload is promoted to `ToolSearchReturnPart` by `narrow_type`. Mirror of the
    builtin-side promotion test, exercising the local (function-tool) variant."""
    base = ToolReturnPart(
        tool_name='search_tools',
        content={'discovered_tools': [{'name': 'foo'}]},
        tool_call_id='c1',
        tool_kind='tool-search',
    )
    narrowed = ToolReturnPart.narrow_type(base)
    assert isinstance(narrowed, ToolSearchReturnPart)
    assert narrowed.content == {'discovered_tools': [{'name': 'foo'}]}


def test_narrow_type_no_tool_kind_returns_input_unchanged_for_local_and_builtin_returns() -> None:
    """`narrow_type` is a no-op when `tool_kind` is `None` — the user-tool default —
    on both `ToolReturnPart` and `NativeToolReturnPart`. This is the early-exit
    branch that keeps user tools untouched without consulting the registry."""
    local = ToolReturnPart(tool_name='foo', content='bar', tool_call_id='c1')
    assert ToolReturnPart.narrow_type(local) is local
    builtin = NativeToolReturnPart(tool_name='foo', content='bar', tool_call_id='c1')
    assert NativeToolReturnPart.narrow_type(builtin) is builtin


def test_model_request_part_discriminator_recognizes_tool_search_return_instance() -> None:
    """The request-part discriminator returns the typed tag when called with a
    `ToolSearchReturnPart` instance.

    Pydantic's discriminated-union fast path bypasses the discriminator when the input
    already matches one of the tagged variants by isinstance, so this exercises the
    function directly rather than via `ModelMessagesTypeAdapter`.
    """

    part = ToolSearchReturnPart(content={'discovered_tools': []}, tool_call_id='c1')
    assert _model_request_part_discriminator(part) == 'tool-search-return'


def test_model_response_part_discriminator_recognizes_local_call_dict_dispatch() -> None:
    """A dict-shaped `ToolCallPart` with `tool_kind='tool-search'` gets dispatched to
    `ToolSearchCallPart` via the discriminator (covers the `'tool-call'` branch)."""

    raw = [
        {
            'kind': 'response',
            'parts': [
                {
                    'part_kind': 'tool-call',
                    'tool_name': 'search_tools',
                    'tool_kind': 'tool-search',
                    'args': {'queries': ['x']},
                    'tool_call_id': 'c1',
                },
            ],
        },
    ]
    message_part(ModelMessagesTypeAdapter.validate_python(raw), ToolSearchCallPart)


def test_model_response_part_discriminator_passthrough_for_unknown_part_kind() -> None:
    """Instance dispatch falls through to `getattr(v, 'part_kind', ...)` for other types."""

    resp = ModelResponse(parts=[TextPart(content='hello')])
    message_part(ModelMessagesTypeAdapter.validate_python([resp]), TextPart)


def test_model_response_part_discriminator_recognizes_typed_instances() -> None:
    """The response-part discriminator returns the typed tag for each typed-instance branch.

    Pydantic's discriminated-union fast path bypasses the discriminator when the input
    already matches one of the tagged variants by isinstance, so the instance branches
    in `_model_response_part_discriminator` are only reachable by calling the function
    directly. This locks in the contract for any future caller (or pydantic version
    that changes its short-circuit behavior).
    """

    builtin_call = NativeToolSearchCallPart(args={'queries': ['x']}, tool_call_id='c1', provider_name='anthropic')
    assert _model_response_part_discriminator(builtin_call) == 'builtin-tool-search-call'

    builtin_return = NativeToolSearchReturnPart(
        content={'discovered_tools': []},
        tool_call_id='c1',
        provider_name='anthropic',
    )
    assert _model_response_part_discriminator(builtin_return) == 'builtin-tool-search-return'

    local_call = ToolSearchCallPart(args={'queries': ['x']}, tool_call_id='c1')
    assert _model_response_part_discriminator(local_call) == 'tool-search-call'


def test_discriminator_unknown_tool_kind_falls_through_to_part_kind() -> None:
    """Dict-form parts with an unregistered `tool_kind` fall through to the bare `part_kind`.

    Exercises the registry-miss branch in both discriminator functions: `_TYPED_PART_TAGS`
    doesn't contain `(part_kind, 'unknown-kind')`, so the discriminator returns the bare
    `part_kind` rather than a typed-subclass tag.

    Calls the discriminator directly because constructing a valid ModelMessage with
    `tool_kind='unknown-kind'` would fail Pydantic's `ToolPartKind` Literal validation
    upstream — the registry-miss branch is internal logic, not a deserialization path
    that any well-formed input would take.
    """

    return_raw = {
        'part_kind': 'tool-return',
        'tool_name': 'something',
        'tool_kind': 'unknown-kind',
        'content': 'hello',
        'tool_call_id': 'c1',
    }
    assert _model_request_part_discriminator(return_raw) == 'tool-return'

    call_raw = {
        'part_kind': 'tool-call',
        'tool_name': 'something',
        'tool_kind': 'unknown-kind',
        'args': {'x': 1},
        'tool_call_id': 'c1',
    }
    assert _model_response_part_discriminator(call_raw) == 'tool-call'


def test_typed_call_part_accessors_return_typed_shapes() -> None:
    """`typed_args` and `queries` on typed call parts read the parsed args.

    Covers both the local-fallback (`ToolSearchCallPart`) and native server-side
    (`NativeToolSearchCallPart`) variants — they're symmetric.
    """

    local_call = ToolSearchCallPart(args={'queries': ['weather', 'github']}, tool_call_id='c1')
    assert local_call.typed_args == {'queries': ['weather', 'github']}
    assert local_call.queries == ['weather', 'github']

    builtin_call = NativeToolSearchCallPart(args={'queries': ['weather']}, tool_call_id='c2', provider_name='anthropic')
    assert builtin_call.typed_args == {'queries': ['weather']}
    assert builtin_call.queries == ['weather']


def test_typed_call_part_typed_args_returns_none_for_unparsed_args() -> None:
    """`typed_args` returns `None` when args haven't been finalized yet.

    Covers the streaming-partial path: `args=None`, partial JSON strings, and
    non-dict JSON values all yield `None` (the contract for streaming-not-yet-ready
    or unexpected shapes). Exercises both typed call part subclasses.
    """

    for cls in (ToolSearchCallPart, NativeToolSearchCallPart):
        kwargs: dict[str, Any] = {'tool_call_id': 'c1'}
        if cls is NativeToolSearchCallPart:
            kwargs['provider_name'] = 'anthropic'

        none_part = cls(args=None, **kwargs)
        assert none_part.typed_args is None
        assert none_part.queries == []

        # Partial JSON string raises during parsing → None.
        partial_part = cls(args='{"queries": ["wea', **kwargs)
        assert partial_part.typed_args is None
        assert partial_part.queries == []

        # Valid JSON that parses to a non-dict (e.g. a bare string) → None.
        scalar_part = cls(args='"just a string"', **kwargs)
        assert scalar_part.typed_args is None
        assert scalar_part.queries == []

        # Valid JSON dict → typed_args populated.
        complete_part = cls(args='{"queries": ["x"]}', **kwargs)
        assert complete_part.typed_args == {'queries': ['x']}
        assert complete_part.queries == ['x']


def test_builtin_tool_search_return_part_message_accessor() -> None:
    """`message` on `NativeToolSearchReturnPart` reads `content.get('message')`.

    The native server-side path doesn't currently populate `message` (Anthropic emits
    its own error/result blocks), so this accessor exists for symmetry with the local
    return part. Exercise it directly to lock in the contract.
    """

    with_message = NativeToolSearchReturnPart(
        content={'discovered_tools': [], 'message': 'no matches'},
        tool_call_id='c1',
        provider_name='anthropic',
    )
    assert with_message.message == 'no matches'

    without_message = NativeToolSearchReturnPart(
        content={'discovered_tools': [{'name': 'foo'}]},
        tool_call_id='c2',
        provider_name='anthropic',
    )
    assert without_message.message is None


async def test_tool_search_toolset_async_search_fn_is_awaited() -> None:
    """Custom search functions can be `async`; the toolset awaits them."""

    async def async_match(_ctx: RunContext, _queries: Sequence[str], tools: Sequence[ToolDefinition]) -> Sequence[str]:
        return [t.name for t in tools]

    ts = ToolSearchToolset(wrapped=_create_function_toolset(), search_fn=async_match)
    ctx = _build_run_context(None)
    tools = await ts.get_tools(ctx)
    search_tool = tools[_SEARCH_TOOLS_NAME]
    result = await ts.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['weather']}, ctx, search_tool)
    return_value = cast(dict[str, Any], result)
    discovered_names = {match['name'] for match in return_value['discovered_tools']}
    # `_create_function_toolset` registers a fixed set of deferred tools; verify the
    # async search function received the corpus and returned discoverable names.
    assert 'calculate_mortgage' in discovered_names


def test_anthropic_custom_replay_blocks_returns_message_on_empty_discovered() -> None:
    """When the typed return carries empty `discovered_tools` and a `message`, the
    helper returns `([], message)`. The `_map_message` flow then renders the message
    as a single text block (Anthropic rejects empty `tool_result.content`)."""
    pytest.importorskip('anthropic')

    empty = ToolSearchReturnPart(
        content={'discovered_tools': [], 'message': 'No matches; try other keywords.'},
        tool_call_id='c1',
    )
    refs, message = _build_custom_tool_search_replay_blocks(
        empty, deferred_tools_active=True, declared_tool_names=set()
    )
    assert refs == []
    assert message == 'No matches; try other keywords.'


def test_anthropic_custom_replay_blocks_skips_non_typed_returns() -> None:
    """A base `ToolReturnPart` (not a typed `ToolSearchReturnPart`) is left alone:
    helper returns `(None, None)` so the caller falls through to default text formatting.
    This is the typed-trust contract — the framework only re-shapes typed parts."""
    pytest.importorskip('anthropic')

    base_part = ToolReturnPart(
        tool_name='search_tools',
        content={'discovered_tools': [{'name': 'foo'}]},
        tool_call_id='c1',
    )
    refs, message = _build_custom_tool_search_replay_blocks(
        base_part, deferred_tools_active=True, declared_tool_names={'foo'}
    )
    assert refs is None and message is None


def test_anthropic_replay_filters_stale_tool_references() -> None:
    """Anthropic rejects `tool_reference` blocks pointing at tools not in the request's
    `tools` list (e.g. an MCP whose connection failed this turn, dropping its tools
    from the corpus). Both replay paths — custom-callable `tool_result.content` and
    native `tool_search_tool_search_result.tool_references` — must filter against the
    set of tools the current turn will actually send."""
    pytest.importorskip('anthropic')

    discovered: list[ToolSearchMatch] = [
        {'name': 'still_here'},
        {'name': 'gone_this_turn'},
    ]
    content: ToolSearchReturnContent = {'discovered_tools': discovered}

    custom_part = ToolSearchReturnPart(content=content, tool_call_id='c1')
    refs, _ = _build_custom_tool_search_replay_blocks(
        custom_part, deferred_tools_active=True, declared_tool_names={'still_here'}
    )
    assert refs == [{'tool_name': 'still_here', 'type': 'tool_reference'}]

    native_part = NativeToolSearchReturnPart(
        provider_name='anthropic',
        tool_call_id='srv_ok',
        content=content,
    )
    block = _build_tool_search_replay_block(native_part, 'srv_ok', declared_tool_names={'still_here'})
    assert block == {
        'tool_use_id': 'srv_ok',
        'type': 'tool_search_tool_result',
        'content': {
            'type': 'tool_search_tool_search_result',
            'tool_references': [{'tool_name': 'still_here', 'type': 'tool_reference'}],
        },
    }


def test_anthropic_finalize_streamed_tool_search_call_part_with_canonical_dict_args() -> None:
    """Already-canonical `ToolSearchArgs` dict passes through unchanged — the typed
    contract guarantees `queries`, so re-running normalization would corrupt the data."""
    pytest.importorskip('anthropic')

    part = NativeToolSearchCallPart(
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

    part = NativeToolSearchCallPart(args=None, tool_call_id='c1', provider_name='anthropic')
    result = _finalize_streamed_tool_search_call_part(part)
    assert isinstance(result.args, dict) and 'queries' in result.args


async def test_anthropic_map_message_empty_search_renders_message_text_block():
    """When custom-callable tool search returns no matches, `_map_message` emits the
    typed return as a single text-content `tool_result` block (not the default text
    fallthrough). Anthropic rejects empty `tool_result.content` arrays — this is the
    spec-compliant path for the custom-search empty-results case."""
    pytest.importorskip('anthropic')

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
                ToolSearchReturnPart(
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
        # A resolved request that withholds a schema — what makes the `tool_reference` reveal
        # legal, and so what puts this exchange on the client-executed replay path at all.
        function_tools=[
            ToolDefinition(
                name='calculate_mortgage',
                parameters_json_schema={'type': 'object'},
                defer_loading=True,
                with_native=ToolSearchTool.kind,
            )
        ],
        tool_visibility={'calculate_mortgage': 'deferred'},
        native_tools=[ToolSearchTool(strategy='custom')],
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
    """A `NativeToolSearchCallPart` with `args=None` (streaming partial state, or a
    history fragment that never carried args) falls through to forwarding the empty
    `args_as_dict()` to the wire `input`. Covers the `else: wire_input = args_dict`
    branch where the cross-provider `queries` slot isn't populated."""
    pytest.importorskip('anthropic')

    model = AnthropicModel(
        'claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=MockAnthropic.create_mock(()))
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args=None,
                    tool_call_id='srv_1',
                    provider_name='anthropic',
                    provider_details={'strategy': 'bm25'},
                ),
                # Pair the call with a return so the orphan-drop pass keeps the call on the wire —
                # this test only exercises the `args=None` code path, not orphan handling.
                NativeToolSearchReturnPart(
                    content={'discovered_tools': []},
                    tool_call_id='srv_1',
                    provider_name='anthropic',
                ),
            ],
        ),
    ]
    params = ModelRequestParameters(
        function_tools=[],
        native_tools=[ToolSearchTool(strategy='bm25')],
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


def test_openai_normalize_tool_search_args_empty_dict_returns_empty_queries() -> None:
    """An empty `arguments={}` payload (the streaming-mid first-event case) normalizes
    to `{'queries': []}` — that's "not yet populated", not "unrecognized"."""
    pytest.importorskip('openai')

    assert _normalize_tool_search_args({}) == {'queries': []}


def test_openai_normalize_tool_search_args_raises_on_unrecognized_shape() -> None:
    """Any non-empty payload that matches neither the `queries: list` nor `paths: list`
    shape raises `UnexpectedModelBehavior` so OpenAI SDK schema drift surfaces loudly
    at the parse boundary rather than silently degrading to an empty result."""
    pytest.importorskip('openai')

    # Non-dict input shouldn't happen given the SDK types arguments as a dict, but if it
    # ever does we want a loud failure rather than a silent fallback.
    with pytest.raises(UnexpectedModelBehavior, match='Unrecognized tool_search arguments shape'):
        _normalize_tool_search_args(None)
    with pytest.raises(UnexpectedModelBehavior, match='Unrecognized tool_search arguments shape'):
        _normalize_tool_search_args('')
    # Dict missing both recognized keys.
    with pytest.raises(UnexpectedModelBehavior, match='Unrecognized tool_search arguments shape'):
        _normalize_tool_search_args({'something_else': 'x'})
    # Dict with `paths` present but of a non-list type.
    with pytest.raises(UnexpectedModelBehavior, match='Unrecognized tool_search arguments shape'):
        _normalize_tool_search_args({'paths': 'not a list'})


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


@pytest.mark.parametrize('model_name', ['claude-sonnet-4-6', 'claude-opus-4-8'])
async def test_anthropic_promotes_local_search_history_with_default_native_strategy(model_name: str) -> None:
    """Local-shape `ToolSearch*Part` from a prior cross-provider turn must render
    into Anthropic's native tool_search wire when the current turn is the default
    server-executed strategy (`ToolSearchTool()` / `strategy=None`).

    The wire shape uses Anthropic's "client-side flavor" of tool search per empirical
    research: a standard `tool_use` for the local `search_tools` function tool
    paired with a `tool_result` whose `content` is a `tool_reference` array
    (NOT a string of stringified discoveries). Anthropic's server unlocks the
    discovered tools' schemas from `defer_loading=true` once it sees the
    `tool_reference` block.

    The model matrix covers both Anthropic replay tiers: native tool search without
    mid-conversation tool deltas, and the newer `tool_addition` path.
    """
    pytest.importorskip('anthropic')

    model = AnthropicModel(
        model_name,
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
                    content={'discovered_tools': [{'name': 'get_weather'}]},
                    tool_call_id='c1',
                ),
            ],
        ),
    ]
    # Default native strategy (NOT 'custom'). The discovered tool ships on the wire
    # with `defer_loading=True`; the replay reference unlocks its schema server-side.
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='get_weather', defer_loading=True)],
        tool_visibility={'get_weather': 'deferred'},
        native_tools=[ToolSearchTool()],
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
    (`strategy='bm25'`). Confirms the gate is "any tool search active", not "custom"
    or "default" — whenever the provider supports native tool search and the current
    request carries it, the historical local-shape parts get the native wire.
    """
    pytest.importorskip('anthropic')

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
                    content={'discovered_tools': [{'name': 'calculate'}]},
                    tool_call_id='c2',
                ),
            ],
        ),
    ]
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='calculate', defer_loading=True)],
        tool_visibility={'calculate': 'deferred'},
        native_tools=[ToolSearchTool(strategy='bm25')],
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


@pytest.mark.parametrize('model_name', ['gpt-5.6', 'gpt-5'])
async def test_openai_promotes_local_search_history_with_default_native_strategy(model_name: str) -> None:
    """Local-shape `ToolSearch*Part` from a prior cross-provider turn must render
    into OpenAI's client-executed tool-search replay items when tool-search replay is active.

    The wire shape uses `tool_search_call` + `tool_search_output` items with
    `execution='client'`; request-building decides when that replay shape also requires
    the current top-level OpenAI `tool_search` registration to use client execution.
    """
    pytest.importorskip('openai')

    model = OpenAIResponsesModel(
        model_name,
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
                    content={'discovered_tools': [{'name': 'get_weather'}]},
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
        native_tools=[ToolSearchTool()],
        allow_text_output=True,
    )

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


async def test_openai_replays_anthropic_native_search_history() -> None:
    """Foreign native history is normalized before OpenAI renders its native replay."""
    pytest.importorskip('openai')

    model = OpenAIResponsesModel(
        'gpt-5.4-mini',
        provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())),
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='find a weather tool')]),
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': ['weather']}, tool_call_id='ant_1', provider_name='anthropic'
                ),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_weather'}]},
                    tool_call_id='ant_1',
                    provider_name='anthropic',
                ),
            ],
            provider_name='anthropic',
        ),
    ]
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='get_weather', defer_loading=True)],
        tool_visibility={'get_weather': 'deferred'},
        native_tools=[ToolSearchTool()],
        allow_text_output=True,
    )

    prepared = model.prepare_messages(history)
    _system, openai_messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        prepared, OpenAIResponsesModelSettings(), params
    )
    calls = [
        item
        for item in openai_messages
        if isinstance(item, dict) and cast(dict[str, Any], item).get('type') == 'tool_search_call'
    ]
    outputs = [
        item
        for item in openai_messages
        if isinstance(item, dict) and cast(dict[str, Any], item).get('type') == 'tool_search_output'
    ]

    assert len(calls) == len(outputs) == 1
    assert calls[0].get('execution') == outputs[0].get('execution') == 'client'
    assert [tool['name'] for tool in cast(list[dict[str, Any]], outputs[0].get('tools'))] == ['get_weather']


async def test_anthropic_replays_openai_native_search_history() -> None:
    """Foreign native history is normalized before Anthropic renders its native replay."""
    pytest.importorskip('anthropic')

    model = AnthropicModel(
        'claude-sonnet-4-6',
        provider=AnthropicProvider(anthropic_client=MockAnthropic.create_mock(())),
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='find a weather tool')]),
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(args={'queries': ['weather']}, tool_call_id='oa_1', provider_name='openai'),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_weather'}]},
                    tool_call_id='oa_1',
                    provider_name='openai',
                ),
            ],
            provider_name='openai',
        ),
    ]
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='get_weather', defer_loading=True)],
        tool_visibility={'get_weather': 'deferred'},
        native_tools=[ToolSearchTool()],
        allow_text_output=True,
    )

    prepared = model.prepare_messages(history)
    _system, anthropic_messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        prepared, params, AnthropicModelSettings()
    )
    tool_uses: list[dict[str, Any]] = [
        cast(dict[str, Any], block)
        for message in anthropic_messages
        if message['role'] == 'assistant' and isinstance(message['content'], list)
        for block in cast(list[Any], message['content'])
        if isinstance(block, dict)
        and cast(dict[str, Any], block).get('type') == 'tool_use'
        and cast(dict[str, Any], block).get('name') == _SEARCH_TOOLS_NAME
    ]
    tool_results: list[dict[str, Any]] = [
        cast(dict[str, Any], block)
        for message in anthropic_messages
        if message['role'] == 'user' and isinstance(message['content'], list)
        for block in cast(list[Any], message['content'])
        if isinstance(block, dict) and cast(dict[str, Any], block).get('type') == 'tool_result'
    ]

    assert len(tool_uses) == len(tool_results) == 1
    assert tool_results[0]['tool_use_id'] == tool_uses[0]['id']
    assert tool_results[0]['content'] == [{'type': 'tool_reference', 'tool_name': 'get_weather'}]


def test_native_search_history_replay_is_stable_across_a_b_a_switch() -> None:
    """Preparing the same stored history for A → B → A never mutates its native shape."""
    pytest.importorskip('openai')
    pytest.importorskip('anthropic')

    anthropic_model = AnthropicModel(
        'claude-sonnet-4-6',
        provider=AnthropicProvider(anthropic_client=MockAnthropic.create_mock(())),
    )
    openai_model = OpenAIResponsesModel(
        'gpt-5.4-mini',
        provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(())),
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='find a weather tool')]),
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': ['weather']}, tool_call_id='ant_1', provider_name='anthropic'
                ),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_weather'}]},
                    tool_call_id='ant_1',
                    provider_name='anthropic',
                ),
            ],
            provider_name='anthropic',
        ),
    ]

    first_anthropic = anthropic_model.prepare_messages(history)
    openai = openai_model.prepare_messages(history)
    second_anthropic = anthropic_model.prepare_messages(history)

    assert first_anthropic is second_anthropic is history
    assert isinstance(message_part(history, NativeToolSearchCallPart, message_index=1), NativeToolSearchCallPart)
    assert isinstance(message_part(openai, ToolSearchCallPart, message_index=1), ToolSearchCallPart)
    assert isinstance(message_part(openai, ToolSearchReturnPart, message_index=2), ToolSearchReturnPart)


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

    Currently fails because `get_native_tools` returns `[]` for `'keywords'`,
    forcing the local-fallback path on every provider — losing the cache benefit
    that the client-executed native path provides on Anthropic and OpenAI.
    """
    cap: ToolSearch = ToolSearch(strategy='keywords')
    builtins = cap.get_native_tools()
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

    The persisted history at turn 3 carries BOTH a `NativeToolSearch*Part` from the
    Anthropic turn AND a `ToolSearch*Part` from the local turn. OpenAI's adapter must
    promote both into native `tool_search_call`+`tool_search_output` items so the
    discovered tools' schemas stay unlocked across the entire chain — the model
    shouldn't have to re-search anything it discovered earlier.
    """
    pytest.importorskip('openai')

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
        # Anthropic-native (turn 1) — `NativeToolSearch*Part`.
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': ['weather']},
                    tool_call_id='ant_1',
                    provider_name='anthropic',
                    provider_details={'strategy': 'bm25'},
                ),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_weather'}]},
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
                    content={'discovered_tools': [{'name': 'calculate_mortgage'}]},
                    tool_call_id='loc_1',
                ),
            ],
        ),
        ModelRequest(parts=[UserPromptPart(content='now compute both')]),
    ]

    params = ModelRequestParameters(
        function_tools=[weather, calc],
        native_tools=[ToolSearchTool()],
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


def test_keywords_search_fn_returns_empty_for_no_tokens() -> None:
    """The shared keyword algorithm returns `[]` when the queries tokenize to nothing
    (whitespace / punctuation only), instead of raising. Callers (`_run_search_fn`
    in the toolset) translate that into the empty-discoveries `_empty_return` shape.
    """

    ctx = _build_run_context(None)
    assert keywords_search_fn(ctx, ['   '], []) == []
    # Punctuation-only queries also produce no tokens — `_SEARCH_TOKEN_RE` matches
    # `[a-z0-9]+` only.
    assert keywords_search_fn(ctx, ['!!!'], []) == []


async def test_tool_search_strategy_keywords_runs_keyword_algorithm_via_search_fn() -> None:
    """When `strategy='keywords'` activates the client-executed native path, the local
    `search_tools` function (still in `function_tools` for client-execution) must run
    the built-in keyword-overlap algorithm — not error out with no `search_fn` set.

    Verifies end-to-end: the toolset's `search_fn` is wired to a callable that
    matches keywords against the corpus, returning matching tool names.
    """
    cap: ToolSearch = ToolSearch(strategy='keywords')
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
    result = await ts.call_tool(_SEARCH_TOOLS_NAME, {'queries': ['mortgage']}, ctx, search_tool)
    return_value = cast(dict[str, Any], result)
    discovered_names = {match['name'] for match in return_value['discovered_tools']}
    assert 'calculate_mortgage' in discovered_names


# --- Pre-advertised hidden non-corpus tools force client-executed promotion ---
#
# This compatibility path remains only for channel-less and `by_reference` profiles that still
# pre-advertise hidden non-corpus tools. Phase 3 removes it with lazy advertisement.


def _hidden_non_corpus_tool() -> ToolDefinition:
    return ToolDefinition(
        name='lookup_refund_policy',
        parameters_json_schema={'type': 'object', 'properties': {}},
        capability_id='refunds',
        defer_loading=True,
    )


def _local_search_tools_def() -> ToolDefinition:
    return ToolDefinition(name=_SEARCH_TOOLS_NAME, parameters_json_schema={}, unless_native=ToolSearchTool.kind)


@pytest.mark.parametrize(
    ('defer_loading', 'corpus_member', 'revealed', 'tool_deferral_mode', 'tool_addition_mode', 'expected'),
    [
        (False, False, False, None, None, 'visible'),
        (True, True, False, 'standalone', None, 'deferred'),
        (True, True, False, None, None, 'withheld'),
        (True, True, False, 'with_tool_search', None, 'deferred'),
        (True, False, False, 'standalone', 'with_definitions', 'withheld'),
        (True, False, True, 'with_tool_search', 'with_definitions', 'via_history'),
        (True, False, True, None, None, 'visible'),
        (True, False, True, 'standalone', None, 'deferred'),
        (True, False, False, 'standalone', 'by_reference', 'deferred'),
        (True, False, True, 'standalone', 'by_reference', 'deferred'),
    ],
)
def test_prepare_request_resolves_tool_visibility(
    defer_loading: bool,
    corpus_member: bool,
    revealed: bool,
    tool_deferral_mode: Literal['standalone', 'with_tool_search'] | None,
    tool_addition_mode: Literal['by_reference', 'with_definitions'] | None,
    expected: Literal['visible', 'deferred', 'withheld', 'via_history'],
) -> None:
    """Pin the resolve table independently of any provider renderer."""

    class M(TestModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
            return frozenset({ToolSearchTool})

    tool = ToolDefinition(
        name='dynamic_tool',
        parameters_json_schema={'type': 'object', 'properties': {}},
        defer_loading=defer_loading,
        with_native=ToolSearchTool.kind if corpus_member else None,
        unless_native='missing' if not defer_loading else None,
    )
    native_tools: list[AbstractNativeTool] = (
        [ToolSearchTool()] if corpus_member or tool_deferral_mode == 'with_tool_search' else []
    )
    params = ModelRequestParameters(
        function_tools=[tool],
        native_tools=native_tools,
        revealed_tool_names={'dynamic_tool'} if revealed else set(),
    )
    profile = ModelProfile(
        tool_deferral_mode=tool_deferral_mode,
        tool_addition_mode=tool_addition_mode,
        supported_native_tools=frozenset({ToolSearchTool}),
    )

    _, prepared = M(profile=profile).prepare_request(None, params)

    [resolved] = prepared.function_tools
    assert resolved.defer_loading is defer_loading
    assert prepared.tool_visibility == {'dynamic_tool': expected}


def test_prepare_request_stamps_visibility_on_the_plain_path() -> None:
    """With nothing native and nothing deferred, `prepare_request` still resolves every tool.

    The consumers happen to treat an absent entry like `'visible'`, so deleting the plain-path
    stamp would fail nothing downstream — this pins the documented invariant directly: authored
    parameters are empty, prepared parameters cover every function tool.
    """
    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='plain_a', parameters_json_schema={'type': 'object', 'properties': {}}),
            ToolDefinition(name='plain_b', parameters_json_schema={'type': 'object', 'properties': {}}),
        ]
    )
    assert params.tool_visibility is None

    _, prepared = TestModel().prepare_request(None, params)

    assert prepared.tool_visibility == {'plain_a': 'visible', 'plain_b': 'visible'}

    # Zero function tools still resolves: `None` means unresolved, `{}` means resolved-and-empty.
    _, empty = TestModel().prepare_request(None, ModelRequestParameters())
    assert empty.tool_visibility == {}


@pytest.mark.parametrize('strategy', ['bm25', 'regex'])
def test_hidden_non_corpus_tool_keeps_named_native_strategy(strategy: str) -> None:
    """Native search stays native because hidden non-corpus tools are withheld from its index."""

    class M(TestModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
            return frozenset({ToolSearchTool})

    params = ModelRequestParameters(
        function_tools=[
            replace(_hidden_non_corpus_tool(), name='searchable_tool', with_native=ToolSearchTool.kind),
            _hidden_non_corpus_tool(),
        ],
        native_tools=[ToolSearchTool(strategy=cast(Any, strategy), optional=True)],
    )
    _, prepared = M(profile=ModelProfile(tool_deferral_mode='standalone')).prepare_request(None, params)

    [native] = prepared.native_tools
    assert isinstance(native, ToolSearchTool) and native.strategy == strategy
    assert prepared.tool_visibility == {
        'searchable_tool': 'deferred',
        'lookup_refund_policy': 'withheld',
    }


def test_hidden_non_corpus_tool_keeps_default_native_strategy() -> None:
    """Default search remains server-executed and its local callback is stripped."""

    class M(TestModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
            return frozenset({ToolSearchTool})

    params = ModelRequestParameters(
        function_tools=[
            _local_search_tools_def(),
            ToolDefinition(
                name='get_weather',
                parameters_json_schema={'type': 'object', 'properties': {}},
                with_native=ToolSearchTool.kind,
                defer_loading=True,
            ),
            _hidden_non_corpus_tool(),
        ],
        native_tools=[ToolSearchTool(strategy=None, optional=True)],
    )
    _, prepared = M(profile=ModelProfile(tool_deferral_mode='standalone')).prepare_request(None, params)

    [native] = prepared.native_tools
    assert isinstance(native, ToolSearchTool) and native.strategy is None
    assert _SEARCH_TOOLS_NAME not in [t.name for t in prepared.function_tools]
    assert prepared.visibility_of('lookup_refund_policy') == 'withheld'


def test_revealed_hidden_tool_keeps_native_search_stable() -> None:
    """The native search surface renders identically before and after a reveal."""

    class M(TestModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
            return frozenset({ToolSearchTool})

    params = ModelRequestParameters(
        function_tools=[
            _local_search_tools_def(),
            replace(_hidden_non_corpus_tool(), with_native=ToolSearchTool.kind),
            _hidden_non_corpus_tool(),
        ],
        native_tools=[ToolSearchTool(strategy=None, optional=True)],
    )
    model = M(profile=ModelProfile(tool_deferral_mode='standalone'))
    _, before = model.prepare_request(None, params)
    _, after = model.prepare_request(None, replace(params, revealed_tool_names={'lookup_refund_policy'}))

    assert before.native_tools == after.native_tools == [ToolSearchTool(strategy=None, optional=True)]
    assert _SEARCH_TOOLS_NAME not in before.tool_defs
    assert _SEARCH_TOOLS_NAME not in after.tool_defs
    assert before.visibility_of('lookup_refund_policy') == 'withheld'
    assert after.visibility_of('lookup_refund_policy') == 'deferred'


def test_hidden_non_corpus_tool_leaves_other_natives_and_custom_search_unchanged() -> None:
    """Other native tools and an explicitly client-executed search strategy are unchanged."""

    class M(TestModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
            return frozenset({ToolSearchTool, WebSearchTool})

    params = ModelRequestParameters(
        function_tools=[
            _local_search_tools_def(),
            replace(_hidden_non_corpus_tool(), with_native=ToolSearchTool.kind),
            _hidden_non_corpus_tool(),
        ],
        # WebSearchTool listed first so the promotion loop must `continue` past it.
        native_tools=[WebSearchTool(), ToolSearchTool(strategy='custom', optional=True)],
    )
    _, prepared = M(profile=ModelProfile(tool_deferral_mode='standalone')).prepare_request(None, params)

    [tool_search] = [t for t in prepared.native_tools if isinstance(t, ToolSearchTool)]
    assert tool_search.strategy == 'custom'
    assert _SEARCH_TOOLS_NAME not in [t.name for t in prepared.function_tools]


def test_capability_gated_tool_search_leaves_non_capability_corpus_alone() -> None:
    """No `capability_id` → no promotion: `strategy=None` reaches the adapter unchanged and
    rule 1 still strips the local `search_tools` since the native is supported."""

    class M(TestModel):
        @classmethod
        def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
            return frozenset({ToolSearchTool})

    plain_deferred = ToolDefinition(
        name='deferred_tool',
        parameters_json_schema={'type': 'object', 'properties': {}},
        with_native=ToolSearchTool.kind,
        defer_loading=True,
    )
    params = ModelRequestParameters(
        function_tools=[_local_search_tools_def(), plain_deferred],
        native_tools=[ToolSearchTool(strategy=None, optional=True)],
    )
    _, prepared = M().prepare_request(None, params)

    [native] = prepared.native_tools
    assert isinstance(native, ToolSearchTool) and native.strategy is None
    assert _SEARCH_TOOLS_NAME not in [t.name for t in prepared.function_tools]


# --- Namespace synthesis for any revealed tool ---
#
# OpenAI rejects a replayed call to a tool that arrived mid-conversation without a `namespace`. For
# cross-provider replay there's no captured namespace, so the adapter synthesizes one from the tool
# name. The gate is `revealed_tool_names` — a searchable corpus member the model discovered and a
# capability-gated tool that was never searchable both land there, by different routes.


@pytest.mark.parametrize(
    'tool_def',
    [
        pytest.param(
            ToolDefinition(
                name='lookup_refund_policy',
                parameters_json_schema={'type': 'object', 'properties': {}},
                with_native=ToolSearchTool.kind,
                defer_loading=True,
            ),
            id='searchable-corpus-member',
        ),
        pytest.param(
            ToolDefinition(
                name='lookup_refund_policy',
                parameters_json_schema={'type': 'object', 'properties': {}},
                capability_id='refunds',
                defer_loading=True,
            ),
            id='capability-gated-tool',
        ),
    ],
)
def test_tool_search_namespace_synthesis_returns_tool_name_for_revealed_tool(tool_def: ToolDefinition) -> None:
    """Either kind of reveal earns a synthesized namespace.

    A capability-gated tool belongs to no corpus, so it carries no `with_native` to key off — but
    OpenAI asks for the namespace on the replayed call just the same.
    """
    pytest.importorskip('openai')

    params = ModelRequestParameters(
        function_tools=[tool_def],
        revealed_tool_names={'lookup_refund_policy'},
    )
    params = replace(params, tool_visibility={tool_def.name: 'via_history'})
    assert _tool_search_namespace_for_synthesis('lookup_refund_policy', params) == 'lookup_refund_policy'


def test_tool_search_namespace_synthesis_returns_none_for_unrelated_function_tool() -> None:
    """A regular function tool, declared up front and never revealed, must not be tagged —
    synthesizing a namespace there would inject a field the API didn't request."""
    pytest.importorskip('openai')

    regular_tool = ToolDefinition(name='get_weather', parameters_json_schema={'type': 'object', 'properties': {}})
    params = ModelRequestParameters(function_tools=[regular_tool])
    assert _tool_search_namespace_for_synthesis('get_weather', params) is None


def test_tool_availability_delta_accumulates_onto_earlier_search_returns():
    """A delta adds to what search already discovered rather than replacing it.

    Additions are the only direction, so reconstructing what the model can see is a union over both
    shapes — the older `ToolSearchReturnPart` histories and the newer delta — and nothing ever leaves
    the set. Withdrawal is tracked in #6985 and would make this an ordered reduction again.
    """
    messages = [
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'old_tool'}, {'name': 'kept_tool'}]},
                    tool_call_id='search-1',
                )
            ]
        ),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['new_tool'])]),
    ]

    assert parse_discovered_tools(messages) == {'old_tool', 'kept_tool', 'new_tool'}


def test_compaction_resets_discovered_tools_at_part_boundary() -> None:
    """Compaction hides every discovery representation before the boundary, including
    earlier parts in the same response, while later visible discoveries still count."""
    before_compaction = ModelRequest(
        parts=[
            ToolSearchReturnPart(
                content={'discovered_tools': [{'name': 'typed_before'}]}, tool_call_id='search-before'
            ),
            ToolAvailabilityDeltaPart(tools_added=['delta_before']),
            ToolReturnPart(
                tool_name='search_tools',
                content='Found legacy_before',
                tool_call_id='legacy-before',
                metadata={'discovered_tools': ['legacy_before']},
            ),
        ]
    )
    compaction = CompactionPart(content='Summary.', provider_name='anthropic')
    assert parse_discovered_tools([before_compaction, ModelResponse(parts=[compaction])]) == set()

    messages = [
        before_compaction,
        ModelResponse(
            parts=[
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'native_before'}]}, tool_call_id='native-before'
                ),
                compaction,
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'native_after'}]}, tool_call_id='native-after'
                ),
            ]
        ),
    ]

    assert parse_discovered_tools(messages) == {'native_after'}


def test_compaction_resets_loaded_capabilities_at_part_boundary() -> None:
    """A capability load is visible only when its complete call/return pair follows the
    latest compaction boundary; an unmatched pre-boundary call cannot complete later."""
    messages = [
        ModelResponse(
            parts=[
                LoadCapabilityCallPart(args={'id': 'before'}, tool_call_id='before'),
                LoadCapabilityCallPart(args={'id': 'split'}, tool_call_id='split'),
            ]
        ),
        ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='before')]),
        ModelResponse(parts=[CompactionPart(content='Summary.', provider_name='anthropic')]),
        ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='split')]),
        ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'after'}, tool_call_id='after')]),
        ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='after')]),
    ]

    assert parse_loaded_capabilities(messages) == {'after'}


async def test_compaction_rehides_capability_tools_until_reloaded() -> None:
    """A capability whose load pair is before compaction starts hidden again, then its
    tool is re-revealed after the model loads the capability in visible history."""
    capability = Capability[None](id='refunds', description='Refund tools.', defer_loading=True)

    @capability.tool_plain
    def issue_refund() -> str:  # pragma: no cover
        return 'refunded'

    history: list[ModelMessage] = [
        ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'refunds'}, tool_call_id='old-load')]),
        ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='old-load')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['issue_refund'], tool_call_id='old-load')]),
        ModelResponse(parts=[CompactionPart(content='Summary.', provider_name='anthropic')]),
    ]
    visible_tools: list[list[str]] = []

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        visible_tools.append([tool.name for tool in info.function_tools])
        if len(visible_tools) == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='load_capability', args={'id': 'refunds'}, tool_call_id='new-load')]
            )
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent[None, str] = Agent(FunctionModel(model_fn), capabilities=[capability], deps_type=type(None))
    await agent.run('refund', message_history=history)

    assert visible_tools == [['load_capability'], ['load_capability', 'issue_refund']]


class _HookObservingCapability(Capability[None]):
    """A deferred capability that records its own tool-execute hook activity."""

    def __init__(self) -> None:
        super().__init__(id='refunds', description='Refund tools.', defer_loading=True)
        self.hook_log: list[str] = []

    async def before_tool_execute(
        self, ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        self.hook_log.append(f'before:{call.tool_name}:available={ctx.capability_active}')  # pragma: no cover
        return args  # pragma: no cover

    async def wrap_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        self.hook_log.append(f'wrap:{call.tool_name}')  # pragma: no cover
        return await handler(args)  # pragma: no cover


async def _call_capability_tool_directly(
    history: list[ModelMessage],
) -> tuple[_HookObservingCapability, list[str], list[str]]:
    """Run an agent whose model calls the capability-owned tool directly, with no (re)load.

    The model gives up once it is told the tool is unavailable, as a real one would after reading
    the retry — otherwise it would just exhaust the retry budget.
    """
    capability = _HookObservingCapability()
    executed: list[str] = []
    refusals: list[str] = []

    @capability.tool_plain
    def issue_refund() -> str:
        executed.append('issue_refund')  # pragma: no cover
        return 'refunded'  # pragma: no cover

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        for part in iter_message_parts(messages, ModelRequest, RetryPromptPart):
            refusals.append(part.content if isinstance(part.content, str) else str(part.content))
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='issue_refund', args={})])

    agent: Agent[None, str] = Agent(FunctionModel(model_fn), capabilities=[capability], deps_type=type(None))
    await agent.run('refund now', message_history=history)
    return capability, executed, refusals


async def test_capability_tool_called_after_compaction_is_refused_until_reloaded() -> None:
    """The boundary reset revokes availability, not just the schema: a capability-owned tool whose
    load pair sits pre-boundary is refused, so the model reloads and the post-compaction history
    ends up carrying the load exchange that justifies the call."""
    history: list[ModelMessage] = [
        ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'refunds'}, tool_call_id='old-load')]),
        ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='old-load')]),
        ModelResponse(parts=[CompactionPart(content='Summary: refund tooling exists.', provider_name='anthropic')]),
    ]
    capability, executed, refusals = await _call_capability_tool_directly(history)

    assert executed == []
    assert capability.hook_log == []
    assert refusals and 'is not available yet' in refusals[0]


async def test_capability_tool_called_without_any_load_is_refused() -> None:
    """The same for the fabricated-history residual: reveal evidence with no load pair does not
    make a capability's tool callable, because a reveal cannot stand in for loading the bundle."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['issue_refund'])]),
        ModelResponse(parts=[TextPart('ok')]),
    ]
    capability, executed, refusals = await _call_capability_tool_directly(history)

    assert executed == []
    assert capability.hook_log == []
    assert refusals and 'is not available yet' in refusals[0]


async def test_searchable_corpus_survives_discovery_and_compaction() -> None:
    """A custom search sees the complete A–E corpus after A–C were discovered, and A can
    be rediscovered after compaction and called without a runtime availability failure."""
    toolset = FunctionToolset()
    executed: list[str] = []
    for tool_name in ['a', 'b', 'c', 'd', 'e']:

        def tool(name: str = tool_name) -> str:
            executed.append(name)
            return name

        toolset.add_function(tool, name=tool_name, defer_loading=True)

    corpora: list[list[str]] = []

    def search(_ctx: RunContext[None], queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
        corpora.append([tool.name for tool in tools])
        return ['a', 'b', 'c'] if queries == ['first'] else ['a']

    call = 0

    def model_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal call
        call += 1
        if call == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='search_tools', args={'queries': ['first']})])
        if call == 2:
            return ModelResponse(
                parts=[CompactionPart(content='Summary.', provider_name='anthropic'), TextPart('compacted')]
            )
        if call == 3:
            return ModelResponse(parts=[ToolCallPart(tool_name='search_tools', args={'queries': ['again']})])
        if call == 4:
            return ModelResponse(parts=[ToolCallPart(tool_name='a', args={})])
        return ModelResponse(parts=[TextPart('done')])

    agent: Agent[None, str] = Agent(
        NoNativeToolSearchModel(model_fn),
        toolsets=[toolset],
        capabilities=[ToolSearch(strategy=search)],
        deps_type=type(None),
    )
    first = await agent.run('discover tools')
    await agent.run('find A again', message_history=first.all_messages())

    assert corpora == [['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'c', 'd', 'e']]
    assert executed == ['a']


async def test_pre_compaction_tool_is_refused_until_rediscovered() -> None:
    """A model that remembers a pre-compaction tool from the summary is asked to search again
    rather than allowed to call it, so the search exchange that justifies the call is regenerated
    on the near side of the boundary."""
    toolset = FunctionToolset()
    executed: list[str] = []

    @toolset.tool_plain(defer_loading=True)
    def issue_refund() -> str:
        executed.append('issue_refund')  # pragma: no cover
        return 'refunded'  # pragma: no cover

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='discover tools')]),
        ModelResponse(parts=[ToolCallPart(tool_name='search_tools', args={'queries': ['refund']}, tool_call_id='s1')]),
        ModelRequest(
            parts=[ToolSearchReturnPart(content={'discovered_tools': [{'name': 'issue_refund'}]}, tool_call_id='s1')]
        ),
        ModelResponse(
            parts=[
                CompactionPart(content='Summary: refund tooling exists.', provider_name='anthropic'),
                TextPart('compacted'),
            ]
        ),
    ]

    refusals: list[str] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        for part in iter_message_parts(messages, ModelRequest, RetryPromptPart):
            refusals.append(part.content if isinstance(part.content, str) else str(part.content))
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='issue_refund', args={})])

    agent: Agent[None, str] = Agent(
        NoNativeToolSearchModel(model_fn), toolsets=[toolset], capabilities=[ToolSearch()], deps_type=type(None)
    )
    await agent.run('refund now', message_history=history)

    assert executed == []
    assert refusals and 'search for it first' in refusals[0]


async def test_delta_in_history_does_not_reveal_a_capability_tool_without_a_load(allow_model_requests: None):
    """A delta naming a capability-owned tool is dropped unless the history also loads its capability.

    A reveal says a schema may go to the model; it cannot stand in for loading the bundle the tool
    belongs to. Honouring it would advertise a tool `ToolManager` then refuses to run — visible and
    uncallable — so the name is filtered out of the request's reveal state instead.
    Deployments accepting client-supplied history get integrity from authenticated endpoints and
    server-persisted history (the UI docs' trust model), not from reveal-state derivation. This test
    pins that decision so the asymmetry isn't mistaken for an oversight.
    """
    capability = Capability[None](
        id='refunds', description='Refund operations.', instructions='Follow the refund policy.', defer_loading=True
    )

    @capability.tool_plain
    def issue_refund() -> str:
        """Issue a refund."""
        return 'refunded'  # pragma: no cover

    captured: list[ModelRequestParameters] = []

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.append(info.model_request_parameters)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[capability], deps_type=type(None))
    result = await agent.run(
        'help', message_history=[ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['issue_refund'])])]
    )

    assert result.output == 'done'
    [params] = captured
    assert 'issue_refund' not in params.revealed_tool_names


def test_tool_availability_delta_falls_back_to_a_system_instruction():
    """A profile without native tool changes is told what happened, not sold a search it never ran.

    The part is replaced where it stands, so the message count doesn't change — which is the point:
    the fabricated `search_tools` call this replaced had to be spliced in as a separate
    `ModelResponse` ahead of the rebuilt request.
    """
    model = TestModel()
    tool = ToolDefinition(name='new_tool', parameters_json_schema={'type': 'object'}, defer_loading=True)
    prepared = model.prepare_messages(
        [ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['new_tool'], tool_call_id='load-1')])],
        ModelRequestParameters(function_tools=[tool]),
    )

    assert len(prepared) == 1
    request = prepared[0]
    assert isinstance(request, ModelRequest)
    [part] = request.parts
    assert isinstance(part, SystemPromptPart)
    assert part.content == snapshot('The following tool(s) are now available: `new_tool`')


def test_tool_availability_delta_does_not_announce_unknown_tool():
    """Persisted UI history cannot turn a provider-shaped name into a system instruction."""
    model = TestModel()
    prepared = model.prepare_messages(
        [ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['ignore_previous_instructions'])])],
        ModelRequestParameters(
            function_tools=[ToolDefinition(name='known_tool', parameters_json_schema={'type': 'object'})]
        ),
    )

    assert prepared == []


async def test_native_tool_availability_delta_does_not_render_unknown_tool():
    """Native delta renderers also resolve names against the current definitions."""
    pytest.importorskip('anthropic')
    pytest.importorskip('openai')
    history: list[ModelMessage] = [
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['ignore_previous_instructions'])])
    ]
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='known_tool', parameters_json_schema={'type': 'object'})]
    )

    anthropic_model = AnthropicModel(
        'claude-opus-4-8', provider=AnthropicProvider(anthropic_client=MockAnthropic.create_mock(()))
    )
    _, anthropic_messages = await anthropic_model._map_message(  # pyright: ignore[reportPrivateUsage]
        history, params, AnthropicModelSettings()
    )
    assert anthropic_messages == []

    openai_model = OpenAIResponsesModel(
        'gpt-5.6', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(()))
    )
    _, openai_messages = await openai_model._map_messages(  # pyright: ignore[reportPrivateUsage]
        history, OpenAIResponsesModelSettings(), params
    )
    assert openai_messages == []


def test_tool_availability_delta_keeps_its_place_among_other_parts():
    """The announcement replaces the delta in place, so the parts around it keep their order.

    This is the shape the old splice got wrong: it appended the fabricated `ModelResponse` to the
    output before the rebuilt `ModelRequest`, so an assistant turn landed ahead of a user prompt that
    had originally preceded the delta.
    """
    model = TestModel()
    tool = ToolDefinition(name='new_tool', parameters_json_schema={'type': 'object'}, defer_loading=True)
    prepared = model.prepare_messages(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='before'),
                    ToolAvailabilityDeltaPart(tools_added=['new_tool']),
                    UserPromptPart(content='after'),
                ]
            )
        ],
        ModelRequestParameters(function_tools=[tool]),
    )

    assert len(prepared) == 1
    request = prepared[0]
    assert isinstance(request, ModelRequest)
    assert [type(part).__name__ for part in request.parts] == snapshot(
        ['UserPromptPart', 'UserPromptPart', 'UserPromptPart']
    )
    announcement = request.parts[1]
    assert isinstance(announcement, UserPromptPart)
    assert announcement.content == '<system>The following tool(s) are now available: `new_tool`</system>'


def test_tool_availability_delta_adding_nothing_leaves_no_empty_request():
    """A delta with nothing to announce drops out rather than reaching an adapter with no parts."""
    model = TestModel()
    prepared = model.prepare_messages([ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[])])])

    assert prepared == snapshot([])


def _vercel_tool_history_roundtrip(messages: list[ModelMessage]) -> list[ModelMessage]:
    return VercelAIAdapter.load_messages(VercelAIAdapter.dump_messages(messages))


def _ag_ui_tool_history_roundtrip(messages: list[ModelMessage]) -> list[ModelMessage]:
    return AGUIAdapter.load_messages(AGUIAdapter.dump_messages(messages, ag_ui_version='0.1.13'))


def _portable_tool_history(representation: Literal['local', 'native', 'delta']) -> list[ModelMessage]:
    if representation == 'local':
        return [
            ModelResponse(parts=[ToolSearchCallPart(args={'queries': ['weather']}, tool_call_id='search-1')]),
            ModelRequest(
                parts=[
                    ToolSearchReturnPart(
                        content={'discovered_tools': [{'name': 'get_weather'}]},
                        tool_call_id='search-1',
                    )
                ]
            ),
        ]
    if representation == 'native':
        return [
            ModelResponse(
                parts=[
                    NativeToolSearchCallPart(
                        args={'queries': ['weather']},
                        tool_call_id='search-1',
                        provider_name='anthropic',
                        provider_details={'strategy': 'bm25'},
                    ),
                    NativeToolSearchReturnPart(
                        content={'discovered_tools': [{'name': 'get_weather'}]},
                        tool_call_id='search-1',
                        provider_name='anthropic',
                    ),
                ],
                provider_name='anthropic',
            )
        ]
    return [ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['get_weather'], tool_call_id='search-1')])]


@pytest.mark.parametrize('representation', ['local', 'native', 'delta'])
@pytest.mark.parametrize(
    'roundtrip',
    [
        pytest.param(_vercel_tool_history_roundtrip, id='vercel'),
        pytest.param(
            _ag_ui_tool_history_roundtrip,
            id='ag-ui',
            marks=pytest.mark.skipif(
                not ag_ui_preserves_tool_kind(),
                reason='ag-ui-protocol not installed, or older than 0.1.11 which drops the tool kind',
            ),
        ),
    ],
)
async def test_tool_history_ui_roundtrip_preserves_anthropic_request(
    representation: Literal['local', 'native', 'delta'],
    roundtrip: Callable[[list[ModelMessage]], list[ModelMessage]],
    allow_model_requests: None,
) -> None:
    """Every persisted tool-discovery representation renders identically after a UI adapter round-trip."""
    pytest.importorskip('anthropic')

    async def render(messages: list[ModelMessage]) -> dict[str, Any]:
        response = completion_message(
            [BetaTextBlock(text='done', type='text')],
            BetaUsage(input_tokens=5, output_tokens=5),
        )
        mock_client = MockAnthropic.create_mock(response)
        model = AnthropicModel(
            'claude-opus-4-8',
            provider=AnthropicProvider(anthropic_client=mock_client),
        )
        tool = ToolDefinition(
            name='get_weather',
            description='Get the weather.',
            parameters_json_schema={'type': 'object', 'properties': {}},
            defer_loading=True,
            with_native=ToolSearchTool.kind,
        )
        await model.request(
            model.prepare_messages(messages),
            None,
            ModelRequestParameters(function_tools=[tool], native_tools=[ToolSearchTool()]),
        )
        return get_mock_chat_completion_kwargs(mock_client)[0]

    history = _portable_tool_history(representation)
    assert await render(roundtrip(history)) == await render(history)


@pytest.mark.parametrize(
    'roundtrip',
    [
        pytest.param(_vercel_tool_history_roundtrip, id='vercel'),
        pytest.param(
            _ag_ui_tool_history_roundtrip,
            id='ag-ui',
            marks=pytest.mark.skipif(not ag_ui_preserves_tool_kind(), reason='ag-ui cannot preserve tool kind'),
        ),
    ],
)
@pytest.mark.parametrize('target', ['openai-responses', 'google'])
async def test_tool_history_ui_roundtrip_delta_renders_once_on_non_anthropic_target(
    roundtrip: Callable[[list[ModelMessage]], list[ModelMessage]],
    target: str,
    allow_model_requests: None,
    mocker: MockerFixture,
) -> None:
    """A persisted delta renders exactly once after adapters rebuild its control representation."""
    tool = ToolDefinition(
        name='get_weather',
        description='Get the weather.',
        parameters_json_schema={'type': 'object', 'properties': {}},
        defer_loading=True,
    )
    history = roundtrip(_portable_tool_history('delta'))
    parameters = ModelRequestParameters(function_tools=[tool], revealed_tool_names={tool.name})

    if target == 'openai-responses':
        pytest.importorskip('openai')
        client = MockOpenAIResponses.create_mock(response_message([]))
        model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(openai_client=client))
        settings, parameters = model.prepare_request(None, parameters)
        await model.request(model.prepare_messages(history, parameters), settings, parameters)
        [request] = get_mock_responses_kwargs(client)
        additional = [item for item in request['input'] if item.get('type') == 'additional_tools']
        assert len(additional) == 1
        assert [wire_tool['name'] for wire_tool in additional[0]['tools']] == ['get_weather']
    else:
        pytest.importorskip('google.genai')
        model = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='test'))
        response = GenerateContentResponse(
            candidates=[Candidate(content=Content(parts=[Part(text='ok')], role='model'))],
            response_id='response-1',
            model_version='gemini-3-flash-preview',
        )
        generate = mocker.patch.object(model.client.aio.models, 'generate_content', return_value=response)
        settings, parameters = model.prepare_request(None, parameters)
        await model.request(model.prepare_messages(history, parameters), settings, parameters)
        announcement = 'The following tool(s) are now available: `get_weather`'
        assert json.dumps(generate.call_args.kwargs, default=str).count(announcement) == 1


def test_tool_availability_delta_adding_nothing_is_dropped_on_the_reveal_path_too():
    """An empty delta has no reveal to render, so it leaves no exchange and no empty request behind.

    The counterpart of `test_tool_availability_delta_adding_nothing_leaves_no_empty_request`, on the
    branch that renders the tool-search exchange because the model withholds schemas. A request whose
    only part is an empty delta has to disappear entirely rather than reach the adapter with no parts.
    """
    pytest.importorskip('anthropic')
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    # `claude-sonnet-4-6` has native tool search and takes `defer_loading` without a search surface,
    # so it renders the reveal as the tool-search exchange rather than announcing it.
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='not-used'))

    assert model.prepare_messages([ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=[])])]) == snapshot([])

    # And an empty delta alongside real content leaves that content untouched.
    prepared = model.prepare_messages(
        [ModelRequest(parts=[UserPromptPart(content='hello'), ToolAvailabilityDeltaPart(tools_added=[])])]
    )
    assert len(prepared) == 1
    request = prepared[0]
    assert isinstance(request, ModelRequest)
    assert [type(part).__name__ for part in request.parts] == snapshot(['UserPromptPart'])


def test_tool_availability_delta_synthesis_deconflicts_duplicate_client_ids():
    """Two persisted deltas cannot emit provider history with duplicate call IDs."""
    pytest.importorskip('anthropic')
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='not-used'))
    messages: list[ModelMessage] = [
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['new_tool'], tool_call_id='duplicate')]),
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['new_tool'], tool_call_id='duplicate')]),
    ]
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='new_tool', defer_loading=True)],
    )
    _, params = model.prepare_request(None, params)

    prepared = model.prepare_messages(messages, params)
    call_ids = [
        part.tool_call_id
        for message in prepared
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolSearchCallPart)
    ]
    assert len(call_ids) == len(set(call_ids)) == 2
