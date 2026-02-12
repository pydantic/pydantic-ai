"""Tests for checkpoint serialization, deserialization, and round-trip consistency."""

from __future__ import annotations

import pytest

from pydantic_ai.runtime.abstract import (
    DeserializedCheckpoint,
    FunctionCall,
    InterruptedToolCall,
    decode_checkpoint_results,
    deserialize_checkpoint,
    serialize_checkpoint_results,
)

pytestmark = pytest.mark.anyio


def test_serialize_deserialize_roundtrip_simple():
    """Simple values survive a serialize → deserialize round-trip."""
    completed = {1: 'hello', 2: 42, 3: {'key': 'value'}}
    interrupted: list[InterruptedToolCall] = []

    checkpoint = serialize_checkpoint_results(completed, interrupted)
    result = deserialize_checkpoint(checkpoint)

    assert isinstance(result, DeserializedCheckpoint)
    assert result.pending_calls == {}
    assert result.interpreter_state is None

    decoded = decode_checkpoint_results(result.completed_results)
    assert decoded == {'1': 'hello', '2': 42, '3': {'key': 'value'}}


def test_serialize_deserialize_with_interrupted_calls():
    """Interrupted tool calls are preserved through round-trip."""
    from pydantic_ai.exceptions import ApprovalRequired

    completed = {1: 'done'}
    interrupted = [
        InterruptedToolCall(
            reason=ApprovalRequired(),
            call=FunctionCall(call_id='2', function_name='dangerous_tool', args=(), kwargs={'path': '/etc/passwd'}),
        )
    ]

    checkpoint = serialize_checkpoint_results(completed, interrupted)
    result = deserialize_checkpoint(checkpoint)

    decoded = decode_checkpoint_results(result.completed_results)
    assert decoded == {'1': 'done'}

    assert '2' in result.pending_calls
    pending = result.pending_calls['2']
    assert pending['function_name'] == 'dangerous_tool'
    assert pending['kwargs'] == {'path': '/etc/passwd'}


def test_serialize_deserialize_with_interpreter_state():
    """Interpreter state (used by Monty) survives round-trip."""
    completed = {1: 'result'}
    interrupted: list[InterruptedToolCall] = []
    state = b'\x00\x01\x02\x03fake-interpreter-state'

    checkpoint = serialize_checkpoint_results(completed, interrupted, interpreter_state=state)
    result = deserialize_checkpoint(checkpoint)

    assert result.interpreter_state == state


def test_serialize_deserialize_complex_values():
    """Complex nested structures survive round-trip."""
    completed = {
        1: [1, 2, 3],
        2: {'nested': {'deep': True}},
        3: None,
        4: [{'a': 1}, {'b': 2}],
    }
    interrupted: list[InterruptedToolCall] = []

    checkpoint = serialize_checkpoint_results(completed, interrupted)
    result = deserialize_checkpoint(checkpoint)
    decoded = decode_checkpoint_results(result.completed_results)

    assert decoded['1'] == [1, 2, 3]
    assert decoded['2'] == {'nested': {'deep': True}}
    assert decoded['3'] is None
    assert decoded['4'] == [{'a': 1}, {'b': 2}]


def test_serialize_empty_checkpoint():
    """Empty checkpoint (no completed results, no interruptions) round-trips."""
    checkpoint = serialize_checkpoint_results({}, [])
    result = deserialize_checkpoint(checkpoint)

    assert result.completed_results == {}
    assert result.pending_calls == {}
    assert result.interpreter_state is None


def test_decode_checkpoint_results_preserves_types():
    """decode_checkpoint_results produces JSON-compatible (not rich) types."""
    completed = {1: {'name': 'Alice', 'scores': [95, 87, 92]}}
    interrupted: list[InterruptedToolCall] = []

    checkpoint = serialize_checkpoint_results(completed, interrupted)
    result = deserialize_checkpoint(checkpoint)
    decoded = decode_checkpoint_results(result.completed_results)

    # Values should be plain dicts/lists, not Pydantic models
    val = decoded['1']
    assert isinstance(val, dict)
    assert isinstance(val['scores'], list)
