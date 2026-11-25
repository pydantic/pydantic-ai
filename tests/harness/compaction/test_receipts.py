"""Compaction receipts speak in the application's voice, and are recognized by type, never by their text."""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai._capability_parts import effective_authority
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    OperatorSource,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai_harness.compaction import (
    CompactionReceiptPart,
    CompactionStrategy,
    RelevanceCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    compact_now,
)
from pydantic_ai_harness.compaction._receipts import is_receipt_part
from pydantic_ai_harness.compaction._shared import starts_user_turn, user_turn_starts

_RECEIPT_TEXT = (
    '[History before this point (3 messages, ~10 tokens) was dropped by the harness. That context is no longer '
    'in the window; re-verify critical facts against primary sources.]'
)
_LEGACY_METADATA = 'pydantic-ai-harness.compaction.receipt.v1'


def _user(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(text)])


def _assistant(text: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(text)])


def _legacy_receipt() -> ModelRequest:
    """A receipt the way older versions wrote it: a user prompt marked with model-invisible metadata."""
    return ModelRequest(parts=[UserPromptPart([TextContent(_RECEIPT_TEXT, metadata=_LEGACY_METADATA)])])


def _receipts(messages: list[ModelMessage]) -> list[CompactionReceiptPart]:
    return [part for message in messages for part in message.parts if isinstance(part, CompactionReceiptPart)]


def _summarizer(prompts: list[str] | None = None) -> SummarizingCompaction[Any]:
    def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[-1]
        assert isinstance(prompt, UserPromptPart) and isinstance(prompt.content, str)
        if prompts is not None:
            prompts.append(prompt.content)
        return ModelResponse(parts=[TextPart('the summary')])

    return SummarizingCompaction(
        model=FunctionModel(summarize),
        max_messages=3,
        keep_messages=1,
        receipts=True,
        bridge_prefix=False,
        preserve_first_user_message=False,
    )


def _judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Scores every step as no longer needed, so each one is removed."""
    return ModelResponse(
        parts=[ToolCallPart(info.output_tools[0].name, {'needed_verbatim': 0.05, 'still_matters': 0.05})]
    )


def _step_history() -> list[ModelMessage]:
    messages: list[ModelMessage] = [ModelRequest(parts=[SystemPromptPart('sys'), UserPromptPart('do the task')])]
    for index in range(3):
        messages.append(ModelResponse(parts=[ToolCallPart('read', {'n': index}, tool_call_id=f'c{index}')]))
        messages.append(ModelRequest(parts=[ToolReturnPart('read', 'junk x ' * 400, tool_call_id=f'c{index}')]))
    messages.append(_assistant('done'))
    messages.append(_user('next please'))
    return messages


def _long_history() -> list[ModelMessage]:
    return [_user('a'), _assistant('b' * 4000), _user('c'), _assistant('d' * 4000), _user('e')]


_STRATEGIES: dict[str, tuple[CompactionStrategy[None], list[ModelMessage]]] = {
    'sliding_window': (
        SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True, preserve_first_user_message=False),
        _long_history(),
    ),
    'summarizing': (_summarizer(), _long_history()),
    'relevance': (
        RelevanceCompaction(
            FunctionModel(_judge),
            max_tokens=2_000,
            target_tokens=740,
            keep_tokens=10,
            truncated_result_chars=0,
            receipts=True,
        ),
        _step_history(),
    ),
}


@pytest.mark.parametrize('name', list(_STRATEGIES))
async def test_receipt_is_an_operator_capability_part(name: str) -> None:
    strategy, history = _STRATEGIES[name]

    result = await compact_now(strategy, history, model=TestModel())

    [receipt] = _receipts(result)
    assert receipt.content.startswith('[History before this point')
    assert receipt.source == OperatorSource()
    assert receipt.scope == 'conversation'
    assert receipt.kind == 'compaction.receipt'
    assert effective_authority(receipt) == 'operator'
    # Nothing in the user's voice carries the receipt.
    assert not any(
        isinstance(part, UserPromptPart) and 'History before this point' in str(part.content)
        for message in result
        for part in message.parts
    )
    # It sits at the cut point, in a request of its own, which doesn't start a user turn.
    [receipt_request] = [message for message in result if receipt in message.parts]
    assert receipt_request.parts == [receipt]
    assert not starts_user_turn(receipt_request)
    assert result.index(receipt_request) not in user_turn_starts(result)


async def test_receipt_records_the_strategy_that_wrote_it() -> None:
    agent = Agent(
        FunctionModel(lambda messages, info: _assistant('ok')),
        capabilities=[SlidingWindowCompaction(max_messages=3, keep_messages=1, receipts=True)],
    )

    result = await agent.run('one', message_history=_long_history())

    [receipt] = _receipts(result.all_messages())
    assert receipt.capability_id is not None and 'sliding_window_compaction' in receipt.capability_id


async def test_user_prompt_with_the_receipt_text_is_not_a_receipt() -> None:
    typed = ModelRequest(parts=[UserPromptPart(_RECEIPT_TEXT)])
    typed_item = ModelRequest(parts=[UserPromptPart([TextContent(_RECEIPT_TEXT)])])
    assert not is_receipt_part(typed.parts[0])
    assert not is_receipt_part(typed_item.parts[0])
    assert starts_user_turn(typed) and starts_user_turn(typed_item)

    strategy = SlidingWindowCompaction(
        max_messages=4, keep_messages=5, receipts=True, preserve_first_user_message=False
    )
    history: list[ModelMessage] = [_user('a'), _assistant('b'), typed, _assistant('c'), typed_item]

    result = await compact_now(strategy, history, model=TestModel())

    # The user's messages are kept, not de-accumulated as earlier receipts.
    assert typed in result and typed_item in result
    assert len(_receipts(result)) == 1


async def test_legacy_receipt_is_recognized_and_replaced() -> None:
    legacy = _legacy_receipt()
    assert is_receipt_part(legacy.parts[0])
    assert not starts_user_turn(legacy)

    strategy = SlidingWindowCompaction(
        max_messages=4, keep_messages=5, receipts=True, preserve_first_user_message=False
    )
    history: list[ModelMessage] = [legacy, _user('a'), _assistant('b'), _user('c'), _assistant('d'), _user('e')]

    result = await compact_now(strategy, history, model=TestModel())

    assert legacy not in result
    assert len(_receipts(result)) == 1


@pytest.mark.parametrize('legacy', [False, True])
async def test_resummarizing_drops_the_earlier_receipt(legacy: bool) -> None:
    prompts: list[str] = []
    strategy = _summarizer(prompts)
    first = await compact_now(strategy, _long_history(), model=TestModel())
    if legacy:
        first = [_legacy_receipt() if _receipts([message]) else message for message in first]
    assert sum(is_receipt_part(part) for message in first for part in message.parts) == 1

    second = await compact_now(
        strategy, [*first, _assistant('f' * 4000), _user('g'), _assistant('h' * 4000), _user('i')], model=TestModel()
    )

    # The summarizer never read the earlier receipt, as a system message or otherwise.
    assert len(prompts) == 2
    assert 'History before this point' not in prompts[1]
    # The new receipt replaced it.
    assert sum(is_receipt_part(part) for message in second for part in message.parts) == 1
    assert len(_receipts(second)) == 1


@pytest.mark.parametrize('inline_system_prompts', [True, False])
async def test_receipt_reaches_the_model_right_after_the_summary(inline_system_prompts: bool) -> None:
    """Without a mid-conversation system role, the receipt is a `<system>` block in place, not hoisted to the top."""
    sent: list[ModelMessage] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent[:] = messages
        return _assistant('ok')

    profile = ModelProfile(supports_inline_system_prompts=inline_system_prompts)
    agent = Agent(
        FunctionModel(respond, profile=profile),
        instructions='be brief',
        capabilities=[_summarizer()],
    )

    await agent.run('next', message_history=_long_history())

    parts = [part for message in sent if isinstance(message, ModelRequest) for part in message.parts]
    summary_at = next(index for index, part in enumerate(parts) if isinstance(part, CompactionPart))
    receipt = parts[summary_at + 1]
    if inline_system_prompts:
        assert isinstance(receipt, SystemPromptPart)
        assert receipt.content.startswith('[History before this point')
    else:
        assert isinstance(receipt, UserPromptPart)
        assert str(receipt.content).startswith('<system>[History before this point')
    assert 'The summary above is secondhand' in str(receipt.content)
