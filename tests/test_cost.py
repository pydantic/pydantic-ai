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

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, CostCalculationFailedWarning
from pydantic_ai._cost import best_effort_price
from pydantic_ai.messages import ModelResponse
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

    monkeypatch.setattr('pydantic_ai._cost.calc_price', _raise)
    agent = Agent(TestModel())
    with warnings.catch_warnings():
        warnings.simplefilter('error', CostCalculationFailedWarning)
        result = await agent.run('hello')
    assert result.usage.cost is None


async def test_cost_unexpected_failure_warns(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch):
    """An unexpected pricing error (not `LookupError`/`ValueError`) warns instead of failing the run."""

    def _raise(*args: object, **kwargs: object) -> object:
        raise RuntimeError('boom')

    monkeypatch.setattr('pydantic_ai._cost.calc_price', _raise)
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


def test_model_response_cost_requires_model_name():
    """`ModelResponse.cost()` is the public entry point, so it owns the "can this be priced at all?" guard.

    `calculate_price_for_usage` takes a non-optional `model_name`, so the check can't live there; a response
    without one (e.g. synthetic, from a capability) has nothing to look up.
    """
    with pytest.raises(AssertionError, match='Model name is required to calculate price'):
        ModelResponse(parts=[]).cost()
