"""Tests for the repair_tool_orphans history processor."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart, capture_run_messages
from pydantic_ai.history_processors import repair_tool_orphans
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    RetryPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = [pytest.mark.anyio]


# ---------------------------------------------------------------------------
# Unit tests for repair_tool_orphans
# ---------------------------------------------------------------------------


def test_empty_history():
    """Empty input should return empty output."""
    assert repair_tool_orphans([]) == []


def test_clean_history_unchanged():
    """Already-valid history should pass through unchanged."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='tc1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='my_tool', content='result', tool_call_id='tc1')]),
        ModelResponse(parts=[TextPart(content='done')]),
    ]
    result = repair_tool_orphans(messages)
    assert len(result) == 4
    # Parts should be structurally identical
    assert isinstance(result[2], ModelRequest)
    tool_returns = [p for p in result[2].parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 1
    assert tool_returns[0].tool_call_id == 'tc1'


def test_dangling_regular_tool_call_with_following_request():
    """A ToolCallPart with no matching return in the next request should get a synthetic return."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='tc1')]),
        ModelRequest(parts=[UserPromptPart(content='continue')]),
    ]
    result = repair_tool_orphans(messages)
    assert len(result) == 3
    # The second request should now have a synthetic ToolReturnPart
    req = result[2]
    assert isinstance(req, ModelRequest)
    synthetic = [p for p in req.parts if isinstance(p, ToolReturnPart)]
    assert len(synthetic) == 1
    assert synthetic[0].tool_call_id == 'tc1'
    assert synthetic[0].tool_name == 'my_tool'
    assert synthetic[0].content == 'Tool call was not completed.'
    # Original user prompt should still be there
    user_parts = [p for p in req.parts if isinstance(p, UserPromptPart)]
    assert len(user_parts) == 1


def test_dangling_regular_tool_call_trailing():
    """A trailing ModelResponse with dangling tool calls gets a synthetic request appended."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='tc1')]),
    ]
    result = repair_tool_orphans(messages)
    assert len(result) == 3
    req = result[2]
    assert isinstance(req, ModelRequest)
    assert len(req.parts) == 1
    assert isinstance(req.parts[0], ToolReturnPart)
    assert req.parts[0].tool_call_id == 'tc1'


def test_dangling_builtin_tool_call():
    """A BuiltinToolCallPart without matching return gets a synthetic BuiltinToolReturnPart in the same response."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(tool_name='web_search', args='{"q": "test"}', tool_call_id='btc1'),
            ]
        ),
        ModelRequest(parts=[UserPromptPart(content='continue')]),
    ]
    result = repair_tool_orphans(messages)
    resp = result[1]
    assert isinstance(resp, ModelResponse)
    builtin_returns = [p for p in resp.parts if isinstance(p, BuiltinToolReturnPart)]
    assert len(builtin_returns) == 1
    assert builtin_returns[0].tool_call_id == 'btc1'
    assert builtin_returns[0].content == 'Tool call was not completed.'


def test_builtin_tool_call_already_has_return():
    """A BuiltinToolCallPart that already has a matching return should not get a duplicate."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(tool_name='web_search', args='{}', tool_call_id='btc1'),
                BuiltinToolReturnPart(tool_name='web_search', content='found it', tool_call_id='btc1'),
            ]
        ),
    ]
    result = repair_tool_orphans(messages)
    resp = result[1]
    assert isinstance(resp, ModelResponse)
    builtin_returns = [p for p in resp.parts if isinstance(p, BuiltinToolReturnPart)]
    assert len(builtin_returns) == 1  # No duplicate


def test_orphaned_tool_return_removed():
    """A ToolReturnPart whose tool_call_id doesn't match any preceding call should be removed."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[TextPart(content='just text')]),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='ghost_tool', content='orphan', tool_call_id='nonexistent'),
                UserPromptPart(content='continue'),
            ]
        ),
    ]
    result = repair_tool_orphans(messages)
    req = result[2]
    assert isinstance(req, ModelRequest)
    tool_returns = [p for p in req.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 0
    user_parts = [p for p in req.parts if isinstance(p, UserPromptPart)]
    assert len(user_parts) == 1


def test_orphaned_retry_prompt_removed():
    """A RetryPromptPart whose tool_call_id doesn't match should be removed."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[TextPart(content='text')]),
        ModelRequest(
            parts=[
                RetryPromptPart(content='retry', tool_name='ghost', tool_call_id='nonexistent'),
                UserPromptPart(content='continue'),
            ]
        ),
    ]
    result = repair_tool_orphans(messages)
    req = result[2]
    assert isinstance(req, ModelRequest)
    retry_parts = [p for p in req.parts if isinstance(p, RetryPromptPart)]
    assert len(retry_parts) == 0


def test_all_parts_removed_gets_placeholder():
    """If removing orphaned parts empties the request, a placeholder UserPromptPart is added."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[TextPart(content='text')]),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='ghost', content='orphan', tool_call_id='nonexistent'),
            ]
        ),
    ]
    result = repair_tool_orphans(messages)
    req = result[2]
    assert isinstance(req, ModelRequest)
    assert len(req.parts) == 1
    assert isinstance(req.parts[0], UserPromptPart)
    assert req.parts[0].content == '...'


def test_multiple_dangling_calls():
    """Multiple dangling tool calls should all get synthetic returns."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='tool_a', args='{}', tool_call_id='tc1'),
                ToolCallPart(tool_name='tool_b', args='{}', tool_call_id='tc2'),
            ]
        ),
        ModelRequest(parts=[UserPromptPart(content='continue')]),
    ]
    result = repair_tool_orphans(messages)
    req = result[2]
    assert isinstance(req, ModelRequest)
    synthetic = [p for p in req.parts if isinstance(p, ToolReturnPart)]
    assert len(synthetic) == 2
    assert {s.tool_call_id for s in synthetic} == {'tc1', 'tc2'}


def test_partial_dangling_calls():
    """Only unanswered tool calls should get synthetic returns."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='tool_a', args='{}', tool_call_id='tc1'),
                ToolCallPart(tool_name='tool_b', args='{}', tool_call_id='tc2'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='tool_a', content='result_a', tool_call_id='tc1'),
                # tc2 is dangling
            ]
        ),
    ]
    result = repair_tool_orphans(messages)
    req = result[2]
    assert isinstance(req, ModelRequest)
    returns = [p for p in req.parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 2
    ids = {r.tool_call_id for r in returns}
    assert ids == {'tc1', 'tc2'}
    # The synthetic one should have the error message
    for r in returns:
        if r.tool_call_id == 'tc2':
            assert r.content == 'Tool call was not completed.'
        elif r.tool_call_id == 'tc1':
            assert r.content == 'result_a'


def test_mixed_dangling_and_orphaned():
    """Both dangling calls and orphaned returns in the same conversation."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='real_tool', args='{}', tool_call_id='tc1'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='ghost_tool', content='orphan', tool_call_id='wrong_id'),
                UserPromptPart(content='continue'),
            ]
        ),
    ]
    result = repair_tool_orphans(messages)
    req = result[2]
    assert isinstance(req, ModelRequest)
    # Orphaned return should be removed
    orphan_returns = [p for p in req.parts if isinstance(p, ToolReturnPart) and p.tool_call_id == 'wrong_id']
    assert len(orphan_returns) == 0
    # Synthetic return for tc1 should be added
    synthetic = [p for p in req.parts if isinstance(p, ToolReturnPart) and p.tool_call_id == 'tc1']
    assert len(synthetic) == 1
    assert synthetic[0].content == 'Tool call was not completed.'
    # User prompt preserved
    user_parts = [p for p in req.parts if isinstance(p, UserPromptPart)]
    assert len(user_parts) == 1


def test_retry_prompt_counts_as_answer():
    """A RetryPromptPart with matching tool_call_id should count as answering that call."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='tc1')]),
        ModelRequest(parts=[RetryPromptPart(content='try again', tool_name='my_tool', tool_call_id='tc1')]),
    ]
    result = repair_tool_orphans(messages)
    req = result[2]
    assert isinstance(req, ModelRequest)
    # Should not inject a synthetic return since retry answers it
    synthetic = [p for p in req.parts if isinstance(p, ToolReturnPart)]
    assert len(synthetic) == 0
    retries = [p for p in req.parts if isinstance(p, RetryPromptPart)]
    assert len(retries) == 1


def test_text_only_response_no_changes():
    """A ModelResponse with only text parts should pass through unchanged."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[TextPart(content='just text')]),
    ]
    result = repair_tool_orphans(messages)
    assert len(result) == 2


def test_preserves_request_metadata():
    """Repaired requests should preserve timestamp, instructions, run_id, metadata."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[TextPart(content='text')]),
        ModelRequest(
            parts=[ToolReturnPart(tool_name='ghost', content='orphan', tool_call_id='nonexistent')],
            instructions='be nice',
            run_id='run-123',
            metadata={'key': 'value'},
        ),
    ]
    result = repair_tool_orphans(messages)
    req = result[2]
    assert isinstance(req, ModelRequest)
    assert req.instructions == 'be nice'
    assert req.run_id == 'run-123'
    assert req.metadata == {'key': 'value'}


def test_preserves_response_fields():
    """Repaired responses should preserve all original fields."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[BuiltinToolCallPart(tool_name='search', args='{}', tool_call_id='btc1')],
            model_name='test-model',
            provider_name='test-provider',
            run_id='run-456',
        ),
    ]
    result = repair_tool_orphans(messages)
    resp = result[1]
    assert isinstance(resp, ModelResponse)
    assert resp.model_name == 'test-model'
    assert resp.provider_name == 'test-provider'
    assert resp.run_id == 'run-456'


def test_no_preceding_response_for_request():
    """A ModelRequest with tool returns but no preceding ModelResponse -- returns are orphaned."""
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='tool', content='result', tool_call_id='tc1'),
            ]
        ),
    ]
    result = repair_tool_orphans(messages)
    req = result[0]
    assert isinstance(req, ModelRequest)
    # Should be replaced with placeholder since it's orphaned
    assert len(req.parts) == 1
    assert isinstance(req.parts[0], UserPromptPart)


def test_multi_turn_poisoned_conversation():
    """Simulate a multi-turn conversation that becomes poisoned over time."""
    messages: list[ModelMessage] = [
        # Turn 1: normal
        ModelRequest(parts=[UserPromptPart(content='start')]),
        ModelResponse(parts=[ToolCallPart(tool_name='fetch', args='{}', tool_call_id='tc1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='fetch', content='ok', tool_call_id='tc1')]),
        ModelResponse(parts=[TextPart(content='got it')]),
        # Turn 2: dangling tool call (timeout)
        ModelRequest(parts=[UserPromptPart(content='next')]),
        ModelResponse(parts=[ToolCallPart(tool_name='fetch', args='{}', tool_call_id='tc2')]),
        # Turn 3: orphaned return with wrong ID
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='fetch', content='late result', tool_call_id='tc_wrong'),
                UserPromptPart(content='continue'),
            ]
        ),
        ModelResponse(parts=[TextPart(content='confused')]),
    ]
    result = repair_tool_orphans(messages)

    # The request after tc2 should have synthetic return for tc2 and no orphaned tc_wrong
    req = result[6]
    assert isinstance(req, ModelRequest)
    returns = [p for p in req.parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 1
    assert returns[0].tool_call_id == 'tc2'
    assert returns[0].content == 'Tool call was not completed.'
    # The orphaned tc_wrong should be gone
    orphans = [p for p in req.parts if isinstance(p, ToolReturnPart) and p.tool_call_id == 'tc_wrong']
    assert len(orphans) == 0


# ---------------------------------------------------------------------------
# Integration test with Agent
# ---------------------------------------------------------------------------


async def test_integration_with_agent():
    """repair_tool_orphans works when passed as a history_processor to Agent."""
    call_count = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return ModelResponse(parts=[TextPart(content='done')])

    model = FunctionModel(model_function)
    agent = Agent(model, history_processors=[repair_tool_orphans])

    # Provide poisoned message history
    poisoned_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='hi')]),
        ModelResponse(parts=[ToolCallPart(tool_name='tool', args='{}', tool_call_id='tc1')]),
        # Missing tool return for tc1
        ModelRequest(parts=[UserPromptPart(content='next')]),
    ]

    with capture_run_messages() as msgs:
        result = await agent.run('continue', message_history=poisoned_history)

    assert result.output == 'done'
    assert call_count == 1
