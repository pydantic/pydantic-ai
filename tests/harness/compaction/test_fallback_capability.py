"""Threshold-driven fallback chains through real agent requests."""

from dataclasses import dataclass

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import AbstractModel, ModelRequestContext
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.compaction import (
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    compact_now,
    estimate_context_tokens,
    pin,
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def history() -> list[ModelMessage]:
    return [ModelRequest.user_text_prompt('old ' * 100), ModelResponse(parts=[TextPart('reply')])]


@dataclass
class Tail:
    calls: int = 0
    model: AbstractModel | str | None = None

    async def compact(self, messages: list[ModelMessage], ctx: RunContext[None]) -> list[ModelMessage]:
        self.calls += 1
        self.model = ctx.model
        return messages[-1:]


@pytest.mark.parametrize('offset', [-1, 0, 1, None])
async def test_threshold_boundary_and_manual_bypass(offset: int | None) -> None:
    messages = [*history(), ModelRequest.user_text_prompt('new')]
    tokens = estimate_context_tokens(messages)
    tail = Tail()
    chain = FallbackCompaction(fallback_chain=[tail], max_tokens=None if offset is None else tokens + offset)
    result = await Agent(TestModel(), deps_type=type(None), capabilities=[chain]).run(
        'new', message_history=messages[:-1]
    )
    fired = offset == -1
    assert tail.calls == int(fired)
    assert len(result.all_messages()) == (2 if fired else 4)
    assert await compact_now(chain, messages, model=TestModel()) == messages[-1:]
    assert tail.calls == int(fired) + 1


@pytest.mark.parametrize('override', [None, 100])
async def test_fraction_fallback_window(override: int | None) -> None:
    tail = Tail()
    chain = FallbackCompaction(
        fallback_chain=[tail], max_fraction=0.5, context_window=override, fallback_context_window=100
    )
    result = await Agent(TestModel(), deps_type=type(None), capabilities=[chain]).run('new', message_history=history())
    assert tail.calls == 1
    assert len(result.all_messages()) == 2


async def test_fraction_uses_request_model_and_passes_it_to_strategy() -> None:
    class CatalogModel(TestModel):
        @property
        def model_id(self) -> str:
            return 'openai:gpt-4o'

    replacement = CatalogModel()

    class SwitchModel(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            request_context.model = replacement
            return request_context

    tail = Tail()
    chain = FallbackCompaction(fallback_chain=[tail], max_fraction=0.0005, fallback_context_window=1_000_000)
    result = await Agent(TestModel(), deps_type=type(None), capabilities=[SwitchModel(), chain]).run(
        'new', message_history=history()
    )
    assert tail.model is replacement
    assert len(result.all_messages()) == 2


async def test_pins_survive_and_compacted_history_persists() -> None:
    pinned = pin('durable state')
    messages = [ModelRequest(parts=[pinned]), *history()]
    tail = Tail()
    agent = Agent(
        TestModel(), deps_type=type(None), capabilities=[FallbackCompaction(fallback_chain=[tail], max_tokens=1)]
    )
    result = await agent.run('new', message_history=messages)
    persisted = result.all_messages()
    assert len(persisted) == 3
    assert isinstance(persisted[0], ModelRequest) and persisted[0].parts == [pinned]
    assert all(message not in persisted for message in history())
    next_result = await agent.run('next', message_history=persisted)
    assert isinstance(next_result.all_messages()[0], ModelRequest)
    assert next_result.all_messages()[0].parts == [pinned]


async def test_automatic_failure_uses_fallback() -> None:
    class Fail:
        async def compact(self, messages: list[ModelMessage], ctx: RunContext[None]) -> list[ModelMessage]:
            raise ModelAPIError('test', 'unavailable')

    tail = Tail()
    chain = FallbackCompaction(fallback_chain=[Fail(), tail], max_tokens=1)
    result = await Agent(TestModel(), deps_type=type(None), capabilities=[chain]).run('new', message_history=history())
    assert tail.calls == 1
    assert len(result.all_messages()) == 2


async def test_trigger_counts_request_instructions() -> None:
    tail = Tail()
    chain = FallbackCompaction(fallback_chain=[tail], max_tokens=500, tokenizer=len)
    await Agent(TestModel(), deps_type=type(None), instructions='x' * 1000, capabilities=[chain]).run('new')
    assert tail.calls == 1


@pytest.mark.parametrize('fraction', [False, True])
def test_focus_preserves_trigger_configuration(fraction: bool) -> None:
    chain: FallbackCompaction[None] = FallbackCompaction(
        fallback_chain=[SummarizingCompaction(max_messages=1), SlidingWindowCompaction(max_messages=1)],
        max_tokens=None if fraction else 100,
        max_fraction=0.5 if fraction else None,
        context_window=1000,
        fallback_context_window=2000,
        tokenizer=len,
        fallback_on=(ValueError,),
    )
    focused = chain.with_focus('auth')
    assert focused.max_tokens == chain.max_tokens
    assert focused.max_fraction == chain.max_fraction
    assert focused.context_window == 1000
    assert focused.fallback_context_window == 2000
    assert focused.tokenizer is len
    assert focused.fallback_on == (ValueError,)
    summarizer = focused.fallback_chain[0]
    assert isinstance(summarizer, SummarizingCompaction)
    assert 'auth' in summarizer.summary_prompt
    assert focused.fallback_chain[1] is chain.fallback_chain[1]


@pytest.mark.parametrize(
    'tokens,fraction,window,fallback',
    [
        (0, None, None, 100),
        (1, 0.5, None, 100),
        (None, 0.0, None, 100),
        (None, 1.1, None, 100),
        (None, float('nan'), None, 100),
        (None, None, 0, 100),
        (None, None, None, 0),
    ],
)
def test_invalid_triggers(tokens: int | None, fraction: float | None, window: int | None, fallback: int) -> None:
    with pytest.raises(ValueError):
        FallbackCompaction(
            fallback_chain=[Tail()],
            max_tokens=tokens,
            max_fraction=fraction,
            context_window=window,
            fallback_context_window=fallback,
        )
