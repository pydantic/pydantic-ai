"""Tests for the opt-in `CompactRetryHistory` capability.

These go through the public `Agent` API with `FunctionModel` rather than calling the
compactor directly: the claim is what the model receives (and what `all_messages()`
keeps) after structured-output or tool retries, which a helper unit test would not prove.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities import CompactRetryHistory
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import PromptedOutput

from .conftest import iter_message_parts

pytestmark = [pytest.mark.anyio]


class Item(BaseModel):
    name: str
    qty: int


def _part_kinds(messages: list[ModelMessage]) -> list[tuple[str, list[str]]]:
    return [(m.kind, [p.part_kind for p in m.parts]) for m in messages]


def _retry_count(messages: list[ModelMessage]) -> int:
    return len(list(iter_message_parts(messages, ModelRequest, RetryPromptPart)))


def test_default_keeps_every_structured_output_retry():
    """Without the capability, each failed output attempt stays in history."""
    request_lengths: list[int] = []
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        request_lengths.append(len(messages))
        assert info.output_tools is not None
        if calls < 3:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"name": 1, "qty": "nope"}')])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"name": "widget", "qty": 2}')])

    agent = Agent(FunctionModel(model_fn), output_type=Item, retries=3)
    result = agent.run_sync('Give me an item.')

    assert result.output == Item(name='widget', qty=2)
    assert request_lengths == [1, 3, 5]
    assert _retry_count(result.all_messages()) == 2
    assert _part_kinds(result.all_messages()) == [
        ('request', ['user-prompt']),
        ('response', ['tool-call']),
        ('request', ['retry-prompt']),
        ('response', ['tool-call']),
        ('request', ['retry-prompt']),
        ('response', ['tool-call']),
        ('request', ['tool-return']),
    ]


def test_compacts_structured_output_retries():
    """With `CompactRetryHistory`, the third request carries only the last failed attempt."""
    request_lengths: list[int] = []
    seen: list[list[tuple[str, list[str]]]] = []
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        request_lengths.append(len(messages))
        seen.append(_part_kinds(messages))
        assert info.output_tools is not None
        if calls < 3:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"name": 1, "qty": "nope"}')])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"name": "widget", "qty": 2}')])

    agent = Agent(
        FunctionModel(model_fn),
        output_type=Item,
        retries=3,
        capabilities=[CompactRetryHistory()],
    )
    result = agent.run_sync('Give me an item.')

    assert result.output == Item(name='widget', qty=2)
    assert request_lengths == [1, 3, 3]
    assert seen[2] == [
        ('request', ['user-prompt']),
        ('response', ['tool-call']),
        ('request', ['retry-prompt']),
    ]
    assert _retry_count(result.all_messages()) == 1
    assert _part_kinds(result.all_messages()) == [
        ('request', ['user-prompt']),
        ('response', ['tool-call']),
        ('request', ['retry-prompt']),
        ('response', ['tool-call']),
        ('request', ['tool-return']),
    ]


def test_compacts_prompted_output_retries():
    """Text-path output retries (`RetryPromptPart` with no `tool_name`) compact the same way."""
    request_lengths: list[int] = []
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        request_lengths.append(len(messages))
        assert info.model_request_parameters.output_mode == 'prompted'
        if calls < 3:
            return ModelResponse(parts=[TextPart(content='{"name": 1, "qty": "nope"}')])
        return ModelResponse(parts=[TextPart(content='{"name": "widget", "qty": 2}')])

    agent = Agent(
        FunctionModel(model_fn),
        output_type=PromptedOutput(Item),
        retries=3,
        capabilities=[CompactRetryHistory()],
    )
    result = agent.run_sync('Give me an item.')

    assert result.output == Item(name='widget', qty=2)
    assert request_lengths == [1, 3, 3]
    retry_parts = list(iter_message_parts(result.all_messages(), ModelRequest, RetryPromptPart))
    assert len(retry_parts) == 1
    assert retry_parts[0].tool_name is None


def test_preserves_successful_tool_exchange_before_output_retries():
    """A successful tool turn before output retries is not dropped."""
    request_lengths: list[int] = []
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        request_lengths.append(len(messages))
        assert info.output_tools is not None
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('ping', {})])
        if calls < 4:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"name": 1, "qty": "nope"}')])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"name": "widget", "qty": 2}')])

    agent = Agent(
        FunctionModel(model_fn),
        output_type=Item,
        retries=3,
        capabilities=[CompactRetryHistory()],
    )

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    result = agent.run_sync('Give me an item.')

    assert result.output == Item(name='widget', qty=2)
    # 1: user; 3: user + ping call + return; 5 then 5: last output failure only.
    assert request_lengths == [1, 3, 5, 5]
    assert _part_kinds(result.all_messages()) == [
        ('request', ['user-prompt']),
        ('response', ['tool-call']),
        ('request', ['tool-return']),
        ('response', ['tool-call']),
        ('request', ['retry-prompt']),
        ('response', ['tool-call']),
        ('request', ['tool-return']),
    ]


def test_compacts_function_tool_retry_streak():
    """A streak of retry-only tool failures is compacted the same way as output retries."""
    request_lengths: list[int] = []
    calls = 0

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        request_lengths.append(len(messages))
        if calls < 3:
            return ModelResponse(parts=[ToolCallPart('lookup', {'name': f'bad{calls}'})])
        if calls == 3:
            return ModelResponse(parts=[ToolCallPart('lookup', {'name': 'ok'})])
        return ModelResponse(parts=[TextPart(content='found')])

    agent = Agent(FunctionModel(model_fn), retries=3, capabilities=[CompactRetryHistory()])

    @agent.tool_plain
    def lookup(name: str) -> str:
        if name != 'ok':
            raise ModelRetry('use ok')
        return 'found'

    result = agent.run_sync('Look it up.')

    assert result.output == 'found'
    assert request_lengths == [1, 3, 3, 5]
    assert _retry_count(result.all_messages()) == 1


def test_mixed_tool_return_and_retry_is_not_a_compactable_pair():
    """A request that mixes a tool return with a retry prompt is not a retry-only pair."""
    request_lengths: list[int] = []
    calls = 0

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        request_lengths.append(len(messages))
        if calls == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('ok_tool', {}),
                    ToolCallPart('fail_tool', {}),
                ]
            )
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_fn), retries=3, capabilities=[CompactRetryHistory()])

    @agent.tool_plain
    def ok_tool() -> str:
        return 'ok'

    @agent.tool_plain
    def fail_tool() -> str:
        raise ModelRetry('try again')

    result = agent.run_sync('Do both.')

    assert result.output == 'done'
    # Second request is user + both calls + the mixed return/retry (one request, not two).
    assert request_lengths == [1, 3]
    assert _part_kinds(result.all_messages())[2] == ('request', ['tool-return', 'retry-prompt'])


def test_no_op_when_there_are_no_retries():
    """A successful first attempt is unchanged."""
    request_lengths: list[int] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request_lengths.append(len(messages))
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"name": "widget", "qty": 2}')])

    agent = Agent(
        FunctionModel(model_fn),
        output_type=Item,
        capabilities=[CompactRetryHistory()],
    )
    result = agent.run_sync('Give me an item.')

    assert result.output == Item(name='widget', qty=2)
    assert request_lengths == [1]
    assert _retry_count(result.all_messages()) == 0


def test_from_spec():
    agent = Agent.from_spec({'model': 'test', 'capabilities': ['CompactRetryHistory']})
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    assert any(isinstance(child, CompactRetryHistory) for child in children)
