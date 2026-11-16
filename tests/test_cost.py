"""Tests for `RunUsage.cost` accumulation across an agent run.

As each model response is appended to the run, its best-effort USD cost (from
[`genai-prices`](https://github.com/pydantic/genai-prices)) is added to `RunUsage.cost`. These tests pin
that behavior:

- it works for streamed responses too (regression: the cost must be calculated *after* the stream is
  consumed, not while it's still empty);
- models/providers `genai-prices` can't price (including `TestModel`/`FunctionModel`) contribute nothing
  and don't warn;
- an unexpected pricing failure is surfaced as a `CostCalculationFailedWarning` rather than crashing the run.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from genai_prices.data_snapshot import DataSnapshot, get_snapshot, set_custom_snapshot
from genai_prices.types import ClauseEquals, ModelInfo, ModelPrice, Provider
from inline_snapshot import snapshot

from pydantic_ai import Agent, CostCalculationFailedWarning
from pydantic_ai._genai_prices import best_effort_price, lookup_context_window
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage, RunUsage

from .conftest import try_import

with try_import() as openai_imports_successful:
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.completion_usage import CompletionUsage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAI


requires_openai = pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')

# A real model name (not the `gpt-4o-123` the shared mock helpers use) so `genai-prices` can price it.
_USAGE = (
    None
    if not openai_imports_successful()
    else CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
)


def _completion() -> chat.ChatCompletion:
    return chat.ChatCompletion(
        id='123',
        choices=[
            Choice(finish_reason='stop', index=0, message=ChatCompletionMessage(content='world', role='assistant'))
        ],
        created=1704067200,  # 2024-01-01
        model='gpt-4o',
        object='chat.completion',
        usage=_USAGE,
    )


def _chunks() -> list[chat.ChatCompletionChunk]:
    def _chunk(delta: ChoiceDelta, *, usage: CompletionUsage | None = None) -> chat.ChatCompletionChunk:
        return chat.ChatCompletionChunk(
            id='123',
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=None)],
            created=1704067200,
            model='gpt-4o',
            object='chat.completion.chunk',
            usage=usage,
        )

    # Mirror real OpenAI streaming: usage arrives only on the final chunk.
    return [
        _chunk(ChoiceDelta(content='wor', role='assistant')),
        _chunk(ChoiceDelta(content='ld')),
        _chunk(ChoiceDelta(), usage=_USAGE),
    ]


@pytest.mark.parametrize(
    ('provider_name', 'expected_provider_id', 'expected_price'),
    [
        ('openai-decisions', 'openai-decisions', Decimal('0.0001')),
        ('alternate', 'openai', Decimal('0.001')),
    ],
)
async def test_cost_uses_endpoint_specific_provider(
    provider_name: str, expected_provider_id: str, expected_price: Decimal
) -> None:
    """OpenAI Decisions shares OpenAI's URL, so its provider ID must select its distinct prices."""
    base_snapshot = get_snapshot()
    openai_provider = next(provider for provider in base_snapshot.providers if provider.id == 'openai')
    openai_model = ModelInfo(
        id='gpt-6-luna',
        match=ClauseEquals(equals='gpt-6-luna'),
        prices=ModelPrice(input_mtok=Decimal('1'), output_mtok=Decimal('0')),
    )
    decisions_model = ModelInfo(
        id='gpt-6-luna',
        match=ClauseEquals(equals='gpt-6-luna'),
        prices=ModelPrice(input_mtok=Decimal('0.1'), output_mtok=Decimal('0')),
    )
    alternate_model = ModelInfo(
        id='gpt-6-luna',
        match=ClauseEquals(equals='gpt-6-luna'),
        prices=ModelPrice(input_mtok=Decimal('2'), output_mtok=Decimal('0')),
    )
    providers: list[Provider] = [
        replace(openai_provider, models=[openai_model]),
        replace(openai_provider, id='openai-decisions', name='OpenAI Decisions', models=[decisions_model]),
        replace(openai_provider, id='alternate', name='Alternate', models=[alternate_model]),
    ]
    set_custom_snapshot(DataSnapshot(providers=providers, from_auto_update=False))

    def respond(_messages: list[ModelMessage], _agent_info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart('done')],
            usage=RequestUsage(input_tokens=1000, cache_read_tokens=200, output_tokens=100),
            provider_name=provider_name,
            provider_url='https://api.openai.com/v1/',
        )

    try:
        result = await Agent(FunctionModel(respond, model_name='gpt-6-luna')).run('hello')
        response = result.all_messages()[-1]
        assert isinstance(response, ModelResponse)
        response_price = response.cost()
        assert response_price.provider.id == expected_provider_id
        assert response_price.total_price == expected_price
        assert result.usage.cost == expected_price
    finally:
        set_custom_snapshot(None)


@requires_openai
@pytest.mark.parametrize('stream', [False, True])
async def test_cost_matches_response_price(allow_model_requests: None, stream: bool):
    """`RunUsage.cost` equals the priced final response, for both `run` and `run_stream`.

    The streaming case is the regression: before the fix the cost was read off the stream before it was
    consumed (so always zero). Asserting equality with the final response's own `cost()` proves the run
    accumulated the fully-consumed usage.
    """
    if stream:
        model = OpenAIChatModel(
            'gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(_chunks()))
        )
    else:
        model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock(_completion())))
    agent = Agent(model)

    if stream:
        async with agent.run_stream('hello') as result:
            output = await result.get_output()
            usage = result.usage
            messages = result.all_messages()
    else:
        run_result = await agent.run('hello')
        output = run_result.output
        usage = run_result.usage
        messages = run_result.all_messages()

    assert output == 'world'
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    price_calculation = best_effort_price(
        response.usage,
        model_name=response.model_name,
        provider_api_url=response.provider_url,
        provider_name=response.provider_name,
        genai_request_timestamp=response.timestamp,
    )
    assert price_calculation is not None
    assert usage.cost == price_calculation.total_price
    assert usage.cost == response.cost().total_price
    assert usage.cost == snapshot(Decimal('0.00075'))


_TIER_UPDATE = {'created': 1791849600, 'model': 'gpt-6.1-sol'}  # 2026-10-13, after genai-prices' tier rates start


@requires_openai
@pytest.mark.parametrize('stream', [False, True])
async def test_cost_uses_openai_service_tier(allow_model_requests: None, stream: bool):
    """The service tier OpenAI reports selects its tier rates, here priority instead of standard."""
    update = {**_TIER_UPDATE, 'service_tier': 'priority'}
    if stream:
        client = MockOpenAI.create_mock_stream([chunk.model_copy(update=update) for chunk in _chunks()])
    else:
        client = MockOpenAI.create_mock(_completion().model_copy(update=update))
    agent = Agent(OpenAIChatModel('gpt-6.1-sol', provider=OpenAIProvider(openai_client=client)))

    if stream:
        async with agent.run_stream('hello') as result:
            await result.get_output()
            usage = result.usage
            messages = result.all_messages()
    else:
        run_result = await agent.run('hello')
        usage = run_result.usage
        messages = run_result.all_messages()

    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert usage.cost == response.cost().total_price
    # 100 input tokens at $4/MTok + 50 output tokens at $20/MTok; standard would be $0.0007.
    assert usage.cost == snapshot(Decimal('0.0014'))


@requires_openai
async def test_cost_service_tier_without_tier_rates_is_standard(allow_model_requests: None):
    """A tier genai-prices has no rates for (OpenAI's `scale`) is priced at standard rates, without warning."""
    completion = _completion().model_copy(update={**_TIER_UPDATE, 'service_tier': 'scale'})
    agent = Agent(
        OpenAIChatModel('gpt-6.1-sol', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock(completion)))
    )
    result = await agent.run('hello')
    # 100 input tokens at $2/MTok + 50 output tokens at $10/MTok.
    assert result.usage.cost == snapshot(Decimal('0.0007'))


def test_service_tier_only_changes_openai_prices():
    """Each response's recorded tier is passed along; where genai-prices has no tier rates it's ignored, silently."""
    timestamp = datetime(2026, 10, 13, tzinfo=UTC)
    prices: dict[str, Decimal] = {}
    # Google records its tier too (e.g. `standard`), but genai-prices only has OpenAI tier rates.
    for provider_name, model_name, service_tier in [
        ('openai', 'gpt-6.1-sol', 'priority'),
        ('google', 'gemini-2.5-flash', 'standard'),
    ]:
        response = ModelResponse(
            parts=[],
            usage=RequestUsage(input_tokens=100_000),
            model_name=model_name,
            provider_name=provider_name,
            provider_details={'service_tier': service_tier},
            timestamp=timestamp,
        )
        prices[provider_name] = response.cost().total_price
    # OpenAI at its $4/MTok priority rate (standard is $2); Gemini at its standard rate.
    assert prices == snapshot({'openai': Decimal('0.4'), 'google': Decimal('0.03')})


async def test_cost_is_silent_for_unpriceable_model(allow_model_requests: None):
    """`TestModel` isn't in `genai-prices`, so cost stays unknown and no warning is emitted."""
    agent = Agent(TestModel())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CostCalculationFailedWarning)
        result = await agent.run('hello')
    assert result.usage.cost is None


async def test_cost_invalid_usage_is_silent(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch):
    """Usage that `genai-prices` refuses to price (`ValueError`) is expected and doesn't warn or fail.

    Real providers can report token breakdowns `genai-prices` considers inconsistent (e.g. cache counts that
    imply negative uncached input tokens); those must not intrude on an always-on cost calculation.
    """

    def _raise(*args: object, **kwargs: object) -> object:
        raise ValueError('inconsistent usage')

    monkeypatch.setattr('pydantic_ai._genai_prices.calc_price', _raise)
    agent = Agent(TestModel())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CostCalculationFailedWarning)
        result = await agent.run('hello')
    assert result.usage.cost is None


async def test_cost_unexpected_failure_warns(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch):
    """An unexpected pricing error (not `LookupError`/`ValueError`) warns instead of failing the run."""

    def _raise(*args: object, **kwargs: object) -> object:
        raise RuntimeError('boom')

    monkeypatch.setattr('pydantic_ai._genai_prices.calc_price', _raise)
    agent = Agent(TestModel())
    with pytest.warns(CostCalculationFailedWarning, match='RuntimeError: boom'):
        result = await agent.run('hello')
    assert result.usage.cost is None


def test_request_usage_cost_arithmetic():
    """Producer-supplied costs are summed with the rest of a response's usage."""
    combined = RequestUsage(cost=Decimal('1.5')) + RequestUsage(cost=Decimal('2'))
    assert combined.cost == Decimal('3.5')

    usage = RequestUsage(cost=Decimal('1.5'))
    usage.incr(RequestUsage(cost=Decimal('2')))
    assert usage.cost == Decimal('3.5')

    # Numeric costs must not be mistaken for token fields and added a second time.
    numeric_usage = RequestUsage(cost=1)
    numeric_usage.incr(RequestUsage(cost=2))
    assert numeric_usage.cost == 3


def test_run_usage_cost_arithmetic():
    """`RunUsage` sums costs from both complete runs and individual requests."""
    combined = RunUsage(cost=Decimal('1.5')) + RunUsage(cost=Decimal('2'))
    assert combined.cost == Decimal('3.5')

    usage = RunUsage(cost=Decimal('1.5'))
    usage.incr(RunUsage(cost=Decimal('2')))
    assert usage.cost == Decimal('3.5')

    usage.incr(RequestUsage(cost=Decimal('2'), input_tokens=10))
    assert usage.cost == Decimal('5.5')


def test_run_usage_subtraction():
    """`RunUsage` subtraction includes custom counters in details from either side."""
    before = RunUsage(requests=1, input_tokens=2, details={'cached': 3, 'custom_units': 2}, cost=Decimal('1.5'))
    after = RunUsage(
        requests=3,
        tool_calls=1,
        input_tokens=7,
        output_tokens=4,
        details={'cached': 5, 'reasoning': 2, 'custom_units': 7},
        cost=Decimal('4'),
    )

    delta = after - before
    expected = RunUsage(
        requests=2,
        tool_calls=1,
        input_tokens=5,
        output_tokens=4,
        details={'cached': 2, 'reasoning': 2, 'custom_units': 5},
        cost=Decimal('2.5'),
    )
    assert delta == expected
    unchanged = before - before
    assert unchanged.details == {'cached': 0, 'custom_units': 0}
    assert (RunUsage() - before).details['custom_units'] == -2


def test_run_usage_subtraction_keeps_every_counter():
    """Subtraction keeps every numeric counter, declared or extracted by genai-prices, the same as `incr` does."""
    before = RunUsage(cache_write_tokens=4, cache_write_1h_tokens=3, output_reasoning_tokens=1, web_searches=1)
    after = RunUsage(
        cache_write_tokens=15,
        cache_write_1h_tokens=11,
        output_reasoning_tokens=5,
        web_searches=3,
        provider_note='kept',
    )

    delta = after - before

    assert delta == RunUsage(
        cache_write_tokens=11,
        cache_write_1h_tokens=8,
        output_reasoning_tokens=4,
        web_searches=2,
        provider_note='kept',
    )
    # A counter only the baseline has comes out negative, like a declared one would.
    assert getattr(RunUsage() - before, 'web_searches', 0) == -1
    # The operands are left as they were.
    assert after.cache_write_1h_tokens == 11
    assert before.details == {}


def test_model_response_cost_requires_model_name():
    """`ModelResponse.cost()` is the public entry point, so it owns the "can this be priced at all?" guard.

    `calculate_price_for_usage` takes a non-optional `model_name`, so the check can't live there; a response
    without one (e.g. synthetic, from a capability) has nothing to look up.
    """
    with pytest.raises(AssertionError, match='Model name is required to calculate price'):
        ModelResponse(parts=[]).cost()


def test_lookup_context_window_by_url_or_provider_name():
    """Matches on the API URL when genai-prices knows it; an unknown URL falls back to the provider name."""
    by_name = lookup_context_window('gpt-5', provider_name='openai')
    assert by_name is not None
    assert lookup_context_window('gpt-5', provider_api_url='https://api.openai.com/v1') == by_name
    assert (
        lookup_context_window('gpt-5', provider_api_url='https://example.invalid/v1', provider_name='openai') == by_name
    )


def test_lookup_context_window_unknown_model_and_provider():
    """Unknown models and providers degrade to `None`, as do lookups with nothing to match on."""
    assert lookup_context_window('potato-gpt', provider_name='openai') is None
    assert lookup_context_window('gpt-5', provider_name='not-a-provider') is None
    assert lookup_context_window('gpt-5') is None
