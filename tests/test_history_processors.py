"""Tests for built-in history processor functions."""

from __future__ import annotations

import pytest

from pydantic_ai.history_processors import repair_orphaned_tool_parts
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


def test_no_changes_needed():
    """Matched pairs pass through untouched."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[ToolCallPart(tool_name='get_data', tool_call_id='call_1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='get_data', content='result', tool_call_id='call_1')]),
        ModelResponse(parts=[TextPart(content='done')]),
    ]
    result = repair_orphaned_tool_parts(messages)
    assert len(result) == 4
    assert result[0] == messages[0]
    assert result[1] == messages[1]
    assert result[2] == messages[2]
    assert result[3] == messages[3]


def test_orphaned_tool_return_removed():
    """ToolReturnPart with no matching ToolCallPart is removed."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='unknown', content='orphan', tool_call_id='call_missing'),
                UserPromptPart(content='continue'),
            ]
        ),
        ModelResponse(parts=[TextPart(content='ok')]),
    ]
    result = repair_orphaned_tool_parts(messages)
    assert len(result) == 3
    assert len(result[1].parts) == 1
    assert isinstance(result[1].parts[0], UserPromptPart)


def test_orphaned_retry_prompt_removed():
    """RetryPromptPart with no matching ToolCallPart is removed."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelRequest(
            parts=[
                RetryPromptPart(content='try again', tool_name='missing_tool', tool_call_id='call_gone'),
            ]
        ),
        ModelResponse(parts=[TextPart(content='ok')]),
    ]
    result = repair_orphaned_tool_parts(messages)
    assert len(result) == 2


def test_orphaned_tool_call_removed():
    """ToolCallPart with no matching ToolReturnPart or RetryPromptPart is removed."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(
            parts=[
                TextPart(content='Let me call a tool'),
                ToolCallPart(tool_name='timed_out', tool_call_id='call_orphan'),
            ]
        ),
        ModelRequest(parts=[UserPromptPart(content='what happened?')]),
        ModelResponse(parts=[TextPart(content='sorry about that')]),
    ]
    result = repair_orphaned_tool_parts(messages)
    assert len(result) == 4
    response = result[1]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)


def test_empty_message_removed():
    """Messages with all parts removed are dropped entirely."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[ToolCallPart(tool_name='lost', tool_call_id='call_lost')]),
        ModelRequest(
            parts=[ToolReturnPart(tool_name='ghost', content='data', tool_call_id='call_ghost')]
        ),
        ModelResponse(parts=[TextPart(content='end')]),
    ]
    result = repair_orphaned_tool_parts(messages)
    assert len(result) == 2
    assert isinstance(result[0].parts[0], UserPromptPart)
    assert isinstance(result[1], ModelResponse)
    assert isinstance(result[1].parts[0], TextPart)


def test_multiple_matched_pairs():
    """Multiple valid tool call/return pairs are preserved."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='do work')]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='a', tool_call_id='id_a'),
                ToolCallPart(tool_name='b', tool_call_id='id_b'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='a', content='result_a', tool_call_id='id_a'),
                ToolReturnPart(tool_name='b', content='result_b', tool_call_id='id_b'),
            ]
        ),
        ModelResponse(parts=[TextPart(content='all done')]),
    ]
    result = repair_orphaned_tool_parts(messages)
    assert len(result) == 4
    assert result == messages


def test_mixed_orphans_and_valid():
    """Only orphaned parts are removed; valid pairs remain."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='go')]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='valid', tool_call_id='id_ok'),
                ToolCallPart(tool_name='orphan_call', tool_call_id='id_orphan'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='valid', content='good', tool_call_id='id_ok'),
                ToolReturnPart(tool_name='orphan_return', content='bad', tool_call_id='id_no_call'),
            ]
        ),
        ModelResponse(parts=[TextPart(content='done')]),
    ]
    result = repair_orphaned_tool_parts(messages)
    assert len(result) == 4

    response = result[1]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], ToolCallPart)
    assert response.parts[0].tool_call_id == 'id_ok'

    request = result[2]
    assert isinstance(request, ModelRequest)
    assert len(request.parts) == 1
    assert isinstance(request.parts[0], ToolReturnPart)
    assert request.parts[0].tool_call_id == 'id_ok'


def test_retry_prompt_matches_call():
    """RetryPromptPart with matching ToolCallPart is preserved."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='try')]),
        ModelResponse(parts=[ToolCallPart(tool_name='flaky', tool_call_id='id_retry')]),
        ModelRequest(
            parts=[RetryPromptPart(content='bad args', tool_name='flaky', tool_call_id='id_retry')]
        ),
        ModelResponse(parts=[TextPart(content='ok')]),
    ]
    result = repair_orphaned_tool_parts(messages)
    assert len(result) == 4
    assert result == messages


def test_empty_history():
    """Empty input returns empty output."""
    assert repair_orphaned_tool_parts([]) == []


def test_text_only_history():
    """History with no tool parts passes through unchanged."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='hello')]),
        ModelResponse(parts=[TextPart(content='hi')]),
        ModelRequest(parts=[UserPromptPart(content='bye')]),
        ModelResponse(parts=[TextPart(content='goodbye')]),
    ]
    result = repair_orphaned_tool_parts(messages)
    assert result == messages


def test_retry_prompt_without_tool_name_preserved():
    """RetryPromptPart without tool_name (output validation retry) is kept."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='generate')]),
        ModelResponse(parts=[TextPart(content='bad output')]),
        ModelRequest(parts=[RetryPromptPart(content='validation failed')]),
        ModelResponse(parts=[TextPart(content='better output')]),
    ]
    result = repair_orphaned_tool_parts(messages)
    assert len(result) == 4
    assert result == messages
