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
from decimal import Decimal
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai._cost import best_effort_cost, best_effort_price_calculation, cost_from_provider_details
from pydantic_ai._warnings import CostCalculationFailedWarning
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

pytestmark = pytest.mark.anyio

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
    price_calculation = best_effort_price_calculation(response)
    assert price_calculation is not None
    assert usage.cost == price_calculation.total_price
    assert usage.cost == response.cost().total_price
    assert usage.cost == snapshot(Decimal('0.00075'))


async def test_cost_is_silent_for_unpriceable_model(allow_model_requests: None):
    """`TestModel` isn't in `genai-prices`, so cost stays zero and no warning is emitted."""
    agent = Agent(TestModel())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CostCalculationFailedWarning)
        result = await agent.run('hello')
    assert result.usage.cost == Decimal(0)


async def test_cost_invalid_usage_is_silent(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch):
    """Usage that `genai-prices` refuses to price (`ValueError`) is expected and doesn't warn or fail.

    Real providers can report token breakdowns `genai-prices` considers inconsistent (e.g. cache counts that
    imply negative uncached input tokens); those must not intrude on an always-on cost calculation.
    """

    def _raise(self: ModelResponse):
        raise ValueError('inconsistent usage')

    monkeypatch.setattr(ModelResponse, 'cost', _raise)
    agent = Agent(TestModel())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CostCalculationFailedWarning)
        result = await agent.run('hello')
    assert result.usage.cost == Decimal(0)


async def test_cost_unexpected_failure_warns(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch):
    """An unexpected pricing error (not `LookupError`/`ValueError`) warns instead of failing the run."""

    def _raise(self: ModelResponse):
        raise RuntimeError('boom')

    monkeypatch.setattr(ModelResponse, 'cost', _raise)
    agent = Agent(TestModel())
    with pytest.warns(CostCalculationFailedWarning, match='RuntimeError: boom'):
        result = await agent.run('hello')
    assert result.usage.cost == Decimal(0)


def test_run_usage_cost_arithmetic():
    """`cost` is summed when combining `RunUsage`s, and untouched when incrementing with a `RequestUsage`."""
    combined = RunUsage(cost=Decimal('1.5')) + RunUsage(cost=Decimal('2'))
    assert combined.cost == Decimal('3.5')

    usage = RunUsage(cost=Decimal('1.5'))
    usage.incr(RunUsage(cost=Decimal('2')))
    assert usage.cost == Decimal('3.5')

    # `RequestUsage` carries no cost, so incrementing with one leaves the accumulated cost unchanged.
    usage.incr(RequestUsage(input_tokens=10))
    assert usage.cost == Decimal('3.5')


def _response_with_provider_details(provider_details: dict[str, Any] | None) -> ModelResponse:
    """A minimal priceable `gpt-4o` response so provider-vs-`genai-prices` preference is observable."""
    return ModelResponse(
        parts=[],
        usage=RequestUsage(input_tokens=100, output_tokens=50),
        model_name='gpt-4o',
        provider_name='openai',
        provider_details=provider_details,
    )


def test_cost_from_provider_details_openrouter_cost():
    """OpenRouter reports its own cost under `provider_details['cost']` (a float, via `Decimal(str(...))`).

    The end-to-end path is covered by `test_openrouter.py::test_openrouter_usage`; this pins the parsing and
    the float-to-`Decimal` conversion (which must go through `str` to avoid binary-float noise) in isolation.
    """
    assert cost_from_provider_details(_response_with_provider_details({'cost': 0.00333825})) == Decimal('0.00333825')
    # A reported cost of exactly zero is a value, not "unknown", so it's returned rather than skipped.
    assert cost_from_provider_details(_response_with_provider_details({'cost': 0.0})) == Decimal('0')


def test_cost_from_provider_details_gateway():
    """The Pydantic AI Gateway reports a `cost_estimate` nested under `provider_details['usage']`."""
    response = _response_with_provider_details({'usage': {'pydantic_ai_gateway': {'cost_estimate': 0.00012625}}})
    assert cost_from_provider_details(response) == Decimal('0.00012625')


def test_cost_from_provider_details_openrouter_cost_takes_precedence_over_gateway():
    """A top-level OpenRouter `cost` wins over the nested gateway estimate."""
    response = _response_with_provider_details({'cost': 0.5, 'usage': {'pydantic_ai_gateway': {'cost_estimate': 0.1}}})
    assert cost_from_provider_details(response) == Decimal('0.5')


@pytest.mark.parametrize(
    'provider_details',
    [
        pytest.param(None, id='no_provider_details'),
        pytest.param({}, id='empty'),
        pytest.param({'finish_reason': 'stop'}, id='unrelated_keys'),
        pytest.param({'cost': None}, id='explicit_none_cost'),
        pytest.param({'usage': {}}, id='usage_without_gateway'),
        pytest.param({'usage': 'not-a-dict'}, id='malformed_usage'),
        pytest.param({'usage': {'pydantic_ai_gateway': 'not-a-dict'}}, id='malformed_gateway'),
        pytest.param({'usage': {'pydantic_ai_gateway': {}}}, id='gateway_without_price'),
        pytest.param({'usage': {'pydantic_ai_gateway': {'cost_estimate': None}}}, id='explicit_none_estimate'),
    ],
)
def test_cost_from_provider_details_absent(provider_details: dict[str, Any] | None):
    """When no provider-reported cost is present, `None` is returned so pricing falls back to `genai-prices`."""
    assert cost_from_provider_details(_response_with_provider_details(provider_details)) is None


def test_best_effort_cost_prefers_provider_details():
    """A provider-reported cost overrides `genai-prices`, even for a model `genai-prices` can price itself."""
    response = _response_with_provider_details({'cost': 0.123})

    # `gpt-4o` is priceable, so this is a genuine choice between two available numbers, not a fallback.
    price_calculation = best_effort_price_calculation(response)
    assert price_calculation is not None
    assert price_calculation.total_price == snapshot(Decimal('0.00075'))

    assert best_effort_cost(response) == Decimal('0.123')


async def test_run_usage_cost_prefers_provider_details(allow_model_requests: None):
    """`RunUsage.cost` uses provider-reported cost when it is available on the final model response."""

    def model_function(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart('done')],
            usage=RequestUsage(input_tokens=100, output_tokens=50),
            provider_details={'cost': 0.123},
        )

    agent = Agent(FunctionModel(model_function, model_name='gpt-4o'))
    result = await agent.run('hello')

    assert result.output == 'done'
    assert result.usage.cost == Decimal('0.123')


def test_best_effort_cost_falls_back_to_price_calculation():
    """Without a provider-reported cost, `best_effort_cost` uses the `genai-prices` calculation."""
    response = _response_with_provider_details(None)
    price_calculation = best_effort_price_calculation(response)
    assert price_calculation is not None
    assert best_effort_cost(response) == price_calculation.total_price == snapshot(Decimal('0.00075'))
