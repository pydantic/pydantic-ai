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

from pydantic_ai import Agent
from pydantic_ai._warnings import CostCalculationFailedWarning
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
