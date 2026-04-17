"""Unit tests for conversation extraction and conversation-level evaluators (no LLM calls)."""

from __future__ import annotations as _annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pytest_mock import MockerFixture

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators import (
        ConversationGoalAchievement,
        ConversationTurn,
        EvaluationReason,
        EvaluatorContext,
        RoleAdherence,
        extract_conversation_turns,
        format_transcript,
    )
    from pydantic_evals.evaluators.llm_as_a_judge import GradingOutput
    from pydantic_evals.otel._errors import SpanTreeRecordingError
    from pydantic_evals.otel.span_tree import SpanNode, SpanTree

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


def _make_span(
    *,
    name: str,
    attributes: dict[str, Any] | None = None,
    span_id: int = 1,
    parent_span_id: int | None = None,
    start_offset_s: float = 0.0,
    duration_s: float = 0.1,
) -> SpanNode:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SpanNode(
        name=name,
        trace_id=0xABCDEF,
        span_id=span_id,
        parent_span_id=parent_span_id,
        start_timestamp=base + timedelta(seconds=start_offset_s),
        end_timestamp=base + timedelta(seconds=start_offset_s + duration_s),
        attributes=attributes or {},
    )


def _tree_with_all_messages(messages: list[Any], new_message_index: int | None = None) -> SpanTree:
    attrs: dict[str, Any] = {'pydantic_ai.all_messages': json.dumps(messages)}
    if new_message_index is not None:
        attrs['pydantic_ai.new_message_index'] = new_message_index
    tree = SpanTree()
    tree.add_spans([_make_span(name='agent run', attributes=attrs)])
    return tree


def _ctx_with_tree(tree: SpanTree, output: Any = '') -> EvaluatorContext[Any, Any, Any]:
    return EvaluatorContext(
        name='test',
        inputs={},
        metadata=None,
        expected_output=None,
        output=output,
        duration=0.0,
        _span_tree=tree,
        attributes={},
        metrics={},
    )


# --------------------------------------------------------------------------- extraction


async def test_extract_basic_conversation():
    """System + user + assistant flatten into three ordered turns."""
    tree = _tree_with_all_messages(
        [
            {'role': 'system', 'parts': [{'type': 'text', 'content': 'You are helpful.'}]},
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hi there'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hello!'}]},
        ]
    )
    turns = extract_conversation_turns(tree)
    assert turns == snapshot(
        [
            ConversationTurn(role='system', content='You are helpful.', turn_index=0),
            ConversationTurn(role='user', content='Hi there', turn_index=1),
            ConversationTurn(role='assistant', content='Hello!', turn_index=2),
        ]
    )


async def test_extract_multi_part_assistant_with_tool_call():
    """An assistant message with a text part plus a tool_call part flattens to two turns."""
    tree = _tree_with_all_messages(
        [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Weather in Paris?'}]},
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'text', 'content': 'Let me check.'},
                    {
                        'type': 'tool_call',
                        'id': 'call_1',
                        'name': 'get_weather',
                        'arguments': {'city': 'Paris'},
                    },
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {
                        'type': 'tool_call_response',
                        'id': 'call_1',
                        'name': 'get_weather',
                        'result': {'temp_c': 18, 'sky': 'clear'},
                    }
                ],
            },
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': '18°C and clear.'}]},
        ]
    )
    turns = extract_conversation_turns(tree)
    assert turns == snapshot(
        [
            ConversationTurn(role='user', content='Weather in Paris?', turn_index=0),
            ConversationTurn(role='assistant', content='Let me check.', turn_index=1),
            ConversationTurn(
                role='assistant',
                content='get_weather',
                turn_index=2,
                tool_name='get_weather',
                tool_arguments='{"city": "Paris"}',
            ),
            ConversationTurn(
                role='tool',
                content='{"temp_c": 18, "sky": "clear"}',
                turn_index=3,
                tool_name='get_weather',
            ),
            ConversationTurn(role='assistant', content='18°C and clear.', turn_index=4),
        ]
    )


async def test_extract_system_only_conversation():
    """A conversation that's just a system prompt extracts a single system turn."""
    tree = _tree_with_all_messages([{'role': 'system', 'parts': [{'type': 'text', 'content': 'be brief'}]}])
    turns = extract_conversation_turns(tree)
    assert turns == [ConversationTurn(role='system', content='be brief', turn_index=0)]


async def test_extract_multimodal_placeholders():
    """Multimodal parts render as `[modality]` placeholders."""
    tree = _tree_with_all_messages(
        [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'What is this?'},
                    {'type': 'image-url', 'url': 'https://example.com/img.png'},
                    {'type': 'uri', 'modality': 'audio', 'uri': 'https://example.com/a.mp3'},
                    {'type': 'binary', 'media_type': 'image/png'},
                    {'type': 'blob', 'modality': 'video', 'mime_type': 'video/mp4'},
                    {'type': 'file', 'modality': 'image', 'mime_type': 'image/jpeg', 'file_id': 'f_1'},
                    # A file part with only a mime_type renders the mime_type.
                    {'type': 'file', 'mime_type': 'application/pdf'},
                    # A bare file part with no modality and no mime_type is just `[file]`.
                    {'type': 'file'},
                    # A uri without modality falls back to 'file' as well.
                    {'type': 'uri', 'uri': 'https://example.com/x.bin'},
                    # A blob without modality falls back to mime_type when present.
                    {'type': 'blob', 'mime_type': 'application/octet-stream'},
                    # A binary part without a media_type falls back to the literal 'binary'.
                    {'type': 'binary'},
                ],
            },
        ]
    )
    turns = extract_conversation_turns(tree)
    assert [t.content for t in turns] == snapshot(
        [
            'What is this?',
            '[image]',
            '[audio]',
            '[image/png]',
            '[video]',
            '[file: image: image/jpeg]',
            '[file: application/pdf]',
            '[file]',
            '[file]',
            '[application/octet-stream]',
            '[binary]',
        ]
    )


async def test_extract_thinking_part_skipped():
    """`thinking` parts are internal reasoning and should not appear in turns."""
    tree = _tree_with_all_messages(
        [
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'thinking', 'content': 'The user wants X...'},
                    {'type': 'text', 'content': 'Here you go.'},
                ],
            },
        ]
    )
    turns = extract_conversation_turns(tree)
    assert turns == [ConversationTurn(role='assistant', content='Here you go.', turn_index=0)]


async def test_extract_malformed_json_skips_and_warns(caplog: pytest.LogCaptureFixture):
    """Invalid JSON in `pydantic_ai.all_messages` falls back to the gen_ai source (or empty), with a warning."""
    tree = SpanTree()
    tree.add_spans([_make_span(name='agent run', attributes={'pydantic_ai.all_messages': 'not-json'})])
    with caplog.at_level(logging.WARNING, logger='pydantic_evals.evaluators.extraction'):
        turns = extract_conversation_turns(tree)
    assert turns == []
    assert any('Failed to parse JSON' in rec.message for rec in caplog.records)


async def test_extract_non_list_all_messages_skips_and_warns(caplog: pytest.LogCaptureFixture):
    """Valid JSON that isn't an array is also ignored with a warning."""
    tree = SpanTree()
    tree.add_spans(
        [_make_span(name='agent run', attributes={'pydantic_ai.all_messages': json.dumps({'not': 'a list'})})]
    )
    with caplog.at_level(logging.WARNING, logger='pydantic_evals.evaluators.extraction'):
        turns = extract_conversation_turns(tree)
    assert turns == []
    assert any('did not decode to a list' in rec.message for rec in caplog.records)


async def test_extract_non_string_attribute_skips_and_warns(caplog: pytest.LogCaptureFixture):
    """A non-string attribute value is ignored with a warning."""
    tree = SpanTree()
    # Pretend the instrumentation emitted a bare int rather than a JSON string.
    tree.add_spans([_make_span(name='agent run', attributes={'pydantic_ai.all_messages': 42})])
    with caplog.at_level(logging.WARNING, logger='pydantic_evals.evaluators.extraction'):
        turns = extract_conversation_turns(tree)
    assert turns == []
    assert any('non-string value' in rec.message for rec in caplog.records)


async def test_extract_missing_role_skips_and_warns(caplog: pytest.LogCaptureFixture):
    """A message missing a valid role is skipped with a warning; others still flatten."""
    # Deliberately build a malformed payload: includes a bare string where a dict is expected.
    malformed: list[Any] = [
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'hi'}]},
        {'parts': [{'type': 'text', 'content': 'no role'}]},
        {'role': 'martian', 'parts': [{'type': 'text', 'content': 'bogus role'}]},
        {'role': 'user', 'parts': 'not-a-list'},
        'not-a-dict',
        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'ok'}]},
    ]
    tree = _tree_with_all_messages(malformed)
    with caplog.at_level(logging.WARNING, logger='pydantic_evals.evaluators.extraction'):
        turns = extract_conversation_turns(tree)
    assert [t.content for t in turns] == ['hi', 'ok']
    messages = [rec.message for rec in caplog.records]
    assert any('invalid role' in m for m in messages)
    assert any('non-list parts' in m for m in messages)
    assert any('non-dict message' in m for m in messages)


async def test_extract_unknown_part_type_skips_and_warns(caplog: pytest.LogCaptureFixture):
    """Unknown part types are skipped with a warning; known ones are kept."""
    tree = _tree_with_all_messages(
        [
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'text', 'content': 'keep me'},
                    {'type': 'laser-beam', 'content': 'skip me'},
                    'not-a-dict-part',
                ],
            },
        ]
    )
    with caplog.at_level(logging.WARNING, logger='pydantic_evals.evaluators.extraction'):
        turns = extract_conversation_turns(tree)
    assert [t.content for t in turns] == ['keep me']
    assert any('unknown part type' in rec.message for rec in caplog.records)
    assert any('non-dict part' in rec.message for rec in caplog.records)


async def test_extract_tool_call_missing_name_skips_and_warns(caplog: pytest.LogCaptureFixture):
    """A tool_call part without a name is skipped with a warning."""
    tree = _tree_with_all_messages(
        [
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'tool_call', 'id': 'x', 'arguments': {}},  # missing name
                    {'type': 'tool_call_response', 'id': 'y', 'result': 'done'},  # missing name
                ],
            },
        ]
    )
    with caplog.at_level(logging.WARNING, logger='pydantic_evals.evaluators.extraction'):
        turns = extract_conversation_turns(tree)
    assert turns == []
    assert any('tool_call part with missing name' in rec.message for rec in caplog.records)
    assert any('tool_call_response part with missing name' in rec.message for rec in caplog.records)


async def test_extract_tool_call_arguments_variants():
    """tool_call `arguments` is preserved as a JSON string (when dict), as-is (when str), or omitted (when absent)."""
    tree = _tree_with_all_messages(
        [
            {
                'role': 'assistant',
                'parts': [
                    {'type': 'tool_call', 'id': '1', 'name': 't1', 'arguments': {'k': 'v'}},
                    {'type': 'tool_call', 'id': '2', 'name': 't2', 'arguments': '{"already": "string"}'},
                    {'type': 'tool_call', 'id': '3', 'name': 't3'},
                ],
            },
        ]
    )
    turns = extract_conversation_turns(tree)
    assert [t.tool_arguments for t in turns] == ['{"k": "v"}', '{"already": "string"}', None]


async def test_extract_tool_call_response_variants():
    """tool_call_response `result` renders as string, JSON, or empty — always recoverable."""
    tree = _tree_with_all_messages(
        [
            {
                'role': 'user',
                'parts': [
                    {'type': 'tool_call_response', 'id': '1', 'name': 't', 'result': 'hello'},
                    {'type': 'tool_call_response', 'id': '2', 'name': 't', 'result': {'a': 1}},
                    {'type': 'tool_call_response', 'id': '3', 'name': 't'},
                ],
            },
        ]
    )
    turns = extract_conversation_turns(tree)
    assert [t.content for t in turns] == ['hello', '{"a": 1}', '']


async def test_extract_text_part_with_non_string_content():
    """A text part whose `content` isn't a string is stringified rather than skipped."""
    tree = _tree_with_all_messages([{'role': 'user', 'parts': [{'type': 'text', 'content': 42}]}])
    turns = extract_conversation_turns(tree)
    assert turns == [ConversationTurn(role='user', content='42', turn_index=0)]


async def test_extract_unrepresentable_json_falls_back(caplog: pytest.LogCaptureFixture):
    """tool_call arguments / tool_call_response result that fail JSON encoding don't crash the evaluator."""

    class NotSerializable:
        pass

    # Mock json.dumps to raise TypeError on our sentinel so we exercise the fallback paths
    # without needing the full extraction pipeline to preserve custom Python objects.
    unserializable = NotSerializable()
    tree = SpanTree()
    # Directly build the attribute bypassing json.dumps to inject a literal that we'll swap
    # during extraction via a monkey-patch. Simpler: put something that json.dumps _can_ round-trip
    # for transport but whose subsequent json.dumps call inside extraction will fail. That's
    # impossible for pure JSON round-trip, so we test the code path via direct dict injection.
    # Instead, build the span attribute manually as a JSON string containing valid JSON for
    # transport, and ensure the code path is exercised via a mocked json.dumps below.
    tree.add_spans(
        [
            _make_span(
                name='agent run',
                attributes={
                    'pydantic_ai.all_messages': json.dumps(
                        [
                            {
                                'role': 'assistant',
                                'parts': [
                                    {'type': 'tool_call', 'id': '1', 'name': 't', 'arguments': {'k': 'v'}},
                                    {'type': 'tool_call_response', 'id': '2', 'name': 't', 'result': {'a': 1}},
                                ],
                            }
                        ]
                    )
                },
            )
        ]
    )

    # Confirm the unserializable sentinel is never referenced (this is a safety/placeholder).
    del unserializable

    # Patch the module-local json.dumps to raise on the _second_ call (the args re-dump);
    # the first call constructs the span attribute above before the patch is applied.
    import pydantic_evals.evaluators.extraction as extraction_mod

    original = extraction_mod.json.dumps
    calls: list[Any] = []

    def failing_dumps(value: Any, *args: Any, **kwargs: Any) -> str:
        calls.append(value)
        raise TypeError('cannot serialize')

    extraction_mod.json.dumps = failing_dumps
    try:
        with caplog.at_level(logging.WARNING, logger='pydantic_evals.evaluators.extraction'):
            turns = extract_conversation_turns(tree)
    finally:
        extraction_mod.json.dumps = original

    # tool_call arguments fall back to None with a warning.
    tool_call = turns[0]
    assert tool_call.tool_name == 't'
    assert tool_call.tool_arguments is None
    assert any('Failed to JSON-encode tool_call arguments' in rec.message for rec in caplog.records)

    # tool_call_response result falls back to repr().
    tool_return = turns[1]
    assert tool_return.content == repr({'a': 1})


async def test_extract_continued_conversation_honors_new_message_index():
    """With `pydantic_ai.new_message_index`, prior history is dropped."""
    tree = _tree_with_all_messages(
        [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'old prompt'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'old reply'}]},
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'new prompt'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'new reply'}]},
        ],
        new_message_index=2,
    )
    turns = extract_conversation_turns(tree)
    assert [t.content for t in turns] == ['new prompt', 'new reply']
    # And the turn indices restart from 0 in the returned slice.
    assert [t.turn_index for t in turns] == [0, 1]


async def test_extract_invalid_new_message_index_falls_back(caplog: pytest.LogCaptureFixture):
    """An out-of-range or non-integer new_message_index logs a warning and falls back to all turns."""
    tree = _tree_with_all_messages(
        [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'a'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'b'}]},
        ],
        new_message_index=99,
    )
    with caplog.at_level(logging.WARNING, logger='pydantic_evals.evaluators.extraction'):
        turns = extract_conversation_turns(tree)
    assert [t.content for t in turns] == ['a', 'b']
    assert any('out of range' in rec.message for rec in caplog.records)

    # Non-integer (string) value
    tree2 = SpanTree()
    tree2.add_spans(
        [
            _make_span(
                name='agent run',
                attributes={
                    'pydantic_ai.all_messages': json.dumps(
                        [{'role': 'user', 'parts': [{'type': 'text', 'content': 'x'}]}]
                    ),
                    'pydantic_ai.new_message_index': 'oops',
                },
            )
        ]
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='pydantic_evals.evaluators.extraction'):
        turns2 = extract_conversation_turns(tree2)
    assert [t.content for t in turns2] == ['x']
    assert any('non-integer value' in rec.message for rec in caplog.records)


async def test_extract_falls_back_to_gen_ai_spans():
    """When no agent-run span has `all_messages`, extraction falls back to per-request spans."""
    input_messages_1 = [
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hi'}]},
    ]
    output_message_1 = [
        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hello'}]},
    ]
    # Second request's input subsumes the earlier conversation; we only take its output.
    input_messages_2 = [
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hi'}]},
        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hello'}]},
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'How are you?'}]},
    ]
    output_message_2 = [
        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Great!'}]},
    ]
    tree = SpanTree()
    tree.add_spans(
        [
            _make_span(
                name='chat request',
                span_id=1,
                start_offset_s=0.0,
                attributes={
                    'gen_ai.input.messages': json.dumps(input_messages_1),
                    'gen_ai.output.messages': json.dumps(output_message_1),
                },
            ),
            _make_span(
                name='chat request',
                span_id=2,
                start_offset_s=0.5,
                attributes={
                    'gen_ai.input.messages': json.dumps(input_messages_2),
                    'gen_ai.output.messages': json.dumps(output_message_2),
                },
            ),
        ]
    )
    turns = extract_conversation_turns(tree)
    assert [t.content for t in turns] == ['Hi', 'Hello', 'Great!']


async def test_extract_gen_ai_fallback_output_only_span():
    """A fallback span with only an output (no input) still contributes a turn."""
    tree = SpanTree()
    tree.add_spans(
        [
            _make_span(
                name='chat request',
                attributes={
                    'gen_ai.output.messages': json.dumps(
                        [{'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Only output'}]}]
                    ),
                },
            )
        ]
    )
    turns = extract_conversation_turns(tree)
    assert [t.content for t in turns] == ['Only output']


async def test_extract_gen_ai_fallback_input_only_span():
    """A fallback span with only an input (no output) still contributes the input turns."""
    tree = SpanTree()
    tree.add_spans(
        [
            _make_span(
                name='chat request',
                attributes={
                    'gen_ai.input.messages': json.dumps(
                        [{'role': 'user', 'parts': [{'type': 'text', 'content': 'Only input'}]}]
                    ),
                },
            )
        ]
    )
    turns = extract_conversation_turns(tree)
    assert [t.content for t in turns] == ['Only input']


async def test_extract_gen_ai_fallback_invalid_payloads_are_skipped(caplog: pytest.LogCaptureFixture):
    """Invalid JSON in either input or output messages is skipped; the other side still contributes."""
    tree = SpanTree()
    tree.add_spans(
        [
            _make_span(
                name='chat request',
                attributes={
                    'gen_ai.input.messages': 'not-json',
                    'gen_ai.output.messages': 'also-not-json',
                },
            )
        ]
    )
    with caplog.at_level(logging.WARNING, logger='pydantic_evals.evaluators.extraction'):
        turns = extract_conversation_turns(tree)
    assert turns == []
    messages = [rec.message for rec in caplog.records]
    assert any('gen_ai.input.messages' in m for m in messages)
    assert any('gen_ai.output.messages' in m for m in messages)


async def test_extract_file_with_only_modality():
    """A file part with just a modality (no mime_type) renders the modality."""
    tree = _tree_with_all_messages(
        [
            {
                'role': 'user',
                'parts': [{'type': 'file', 'modality': 'document'}],
            }
        ]
    )
    turns = extract_conversation_turns(tree)
    assert [t.content for t in turns] == ['[file: document]']


async def test_extract_part_handler_exception_is_caught(caplog: pytest.LogCaptureFixture, mocker: MockerFixture):
    """A handler raising an unexpected `TypeError` is logged and skipped, not propagated."""
    import pydantic_evals.evaluators.extraction as extraction_mod

    mocker.patch.object(extraction_mod, '_part_to_turn', side_effect=TypeError('boom'))
    tree = _tree_with_all_messages([{'role': 'user', 'parts': [{'type': 'text', 'content': 'hi'}]}])
    with caplog.at_level(logging.WARNING, logger='pydantic_evals.evaluators.extraction'):
        turns = extract_conversation_turns(tree)
    assert turns == []
    assert any('Failed to flatten part' in rec.message for rec in caplog.records)


async def test_extract_empty_tree():
    """An empty span tree yields no turns."""
    assert extract_conversation_turns(SpanTree()) == []


# --------------------------------------------------------------------------- format_transcript


async def test_format_transcript_shape():
    turns = [
        ConversationTurn(role='system', content='be helpful', turn_index=0),
        ConversationTurn(role='user', content='hi', turn_index=1),
        ConversationTurn(
            role='assistant',
            content='get_weather',
            turn_index=2,
            tool_name='get_weather',
            tool_arguments='{"city": "Paris"}',
        ),
        ConversationTurn(role='tool', content='18C', turn_index=3, tool_name='get_weather'),
        ConversationTurn(role='assistant', content='18C in Paris', turn_index=4),
    ]
    assert format_transcript(turns) == snapshot("""\
[0] system: be helpful
[1] user: hi
[2] assistant (tool_call 'get_weather' args={"city": "Paris"}): get_weather
[3] tool (tool 'get_weather'): 18C
[4] assistant: 18C in Paris\
""")


async def test_format_transcript_empty_returns_empty_string():
    assert format_transcript([]) == ''


async def test_format_transcript_tool_call_without_arguments():
    """A tool call with no arguments omits the `args=` suffix."""
    turns = [
        ConversationTurn(
            role='assistant',
            content='ping',
            turn_index=0,
            tool_name='ping',
        ),
    ]
    assert format_transcript(turns) == "[0] assistant (tool_call 'ping'): ping"


# --------------------------------------------------------------------------- evaluator assembly


async def test_conversation_goal_achievement_no_turns_returns_failure():
    """When extraction yields no turns, the evaluator returns a clean failure (no judge call)."""
    evaluator = ConversationGoalAchievement(goal='resolve the issue')
    result = await evaluator.evaluate_async(_ctx_with_tree(SpanTree()))
    assert result == snapshot(
        {
            'goal_achieved': EvaluationReason(
                value=False,
                reason='No conversation turns were found in the span tree; cannot judge goal achievement.',
            ),
            'goal_achievement_score': 0.0,
        }
    )


async def test_role_adherence_no_turns_returns_failure():
    evaluator = RoleAdherence(role='helpful assistant')
    result = await evaluator.evaluate_async(_ctx_with_tree(SpanTree()))
    assert result == snapshot(
        {
            'role_adhered': EvaluationReason(
                value=False,
                reason='No conversation turns were found in the span tree; cannot judge role adherence.',
            ),
            'role_adherence_score': 0.0,
        }
    )


async def test_conversation_goal_achievement_propagates_span_tree_recording_error():
    """When spans weren't recorded, accessing `ctx.span_tree` propagates the recording error."""
    ctx: EvaluatorContext[Any, Any, Any] = EvaluatorContext(
        name='test',
        inputs={},
        metadata=None,
        expected_output=None,
        output='',
        duration=0.0,
        _span_tree=SpanTreeRecordingError('spans were not recorded'),
        attributes={},
        metrics={},
    )
    evaluator = ConversationGoalAchievement(goal='x')
    with pytest.raises(SpanTreeRecordingError):
        await evaluator.evaluate_async(ctx)


async def test_role_adherence_propagates_span_tree_recording_error():
    ctx: EvaluatorContext[Any, Any, Any] = EvaluatorContext(
        name='test',
        inputs={},
        metadata=None,
        expected_output=None,
        output='',
        duration=0.0,
        _span_tree=SpanTreeRecordingError('spans were not recorded'),
        attributes={},
        metrics={},
    )
    evaluator = RoleAdherence(role='x')
    with pytest.raises(SpanTreeRecordingError):
        await evaluator.evaluate_async(ctx)


async def test_conversation_goal_achievement_calls_judge_and_returns_result(mocker: MockerFixture):
    """With turns present, the evaluator calls `judge_conversation_goal` and maps the result."""
    mock_judge = mocker.patch(
        'pydantic_evals.evaluators.conversation.judge_conversation_goal',
        return_value=GradingOutput(reason='clear resolution', pass_=True, score=0.9),
    )
    tree = _tree_with_all_messages(
        [
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'Can you reset my password?'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Sure. Done.'}]},
        ]
    )
    evaluator = ConversationGoalAchievement(goal='Reset the password.')
    result = await evaluator.evaluate_async(_ctx_with_tree(tree, output='Done.'))
    assert result == snapshot(
        {
            'goal_achieved': EvaluationReason(value=True, reason='clear resolution'),
            'goal_achievement_score': 0.9,
        }
    )
    mock_judge.assert_called_once()
    kwargs = mock_judge.call_args.kwargs
    assert kwargs['goal'] == 'Reset the password.'
    assert [t.content for t in kwargs['turns']] == ['Can you reset my password?', 'Sure. Done.']
    assert kwargs['final_output'] == 'Done.'


async def test_conversation_goal_achievement_threshold_boundary(mocker: MockerFixture):
    """`score == threshold` counts as achieved (>= is the rule, not >)."""
    mocker.patch(
        'pydantic_evals.evaluators.conversation.judge_conversation_goal',
        return_value=GradingOutput(reason='borderline', pass_=False, score=0.7),
    )
    tree = _tree_with_all_messages([{'role': 'user', 'parts': [{'type': 'text', 'content': 'go'}]}])
    evaluator = ConversationGoalAchievement(goal='x', threshold=0.7)
    result = await evaluator.evaluate_async(_ctx_with_tree(tree))
    # The user-facing `goal_achieved` uses `score >= threshold`, even though the judge flagged `pass_=False`.
    goal_achieved = result['goal_achieved']  # type: ignore[index]
    assert isinstance(goal_achieved, EvaluationReason)
    assert goal_achieved.value is True


async def test_conversation_goal_achievement_score_below_threshold(mocker: MockerFixture):
    mocker.patch(
        'pydantic_evals.evaluators.conversation.judge_conversation_goal',
        return_value=GradingOutput(reason='nope', pass_=False, score=0.3),
    )
    tree = _tree_with_all_messages([{'role': 'user', 'parts': [{'type': 'text', 'content': 'go'}]}])
    evaluator = ConversationGoalAchievement(goal='x')
    result = await evaluator.evaluate_async(_ctx_with_tree(tree))
    assert result == snapshot(
        {
            'goal_achieved': EvaluationReason(value=False, reason='nope'),
            'goal_achievement_score': 0.3,
        }
    )


async def test_conversation_goal_achievement_include_reason_false(mocker: MockerFixture):
    """With `include_reason=False`, the assertion is a bare bool (no `EvaluationReason`)."""
    mocker.patch(
        'pydantic_evals.evaluators.conversation.judge_conversation_goal',
        return_value=GradingOutput(reason='ok', pass_=True, score=0.9),
    )
    tree = _tree_with_all_messages([{'role': 'user', 'parts': [{'type': 'text', 'content': 'go'}]}])
    evaluator = ConversationGoalAchievement(goal='x', include_reason=False)
    result = await evaluator.evaluate_async(_ctx_with_tree(tree))
    assert result == {'goal_achieved': True, 'goal_achievement_score': 0.9}


async def test_role_adherence_calls_judge(mocker: MockerFixture):
    mock_judge = mocker.patch(
        'pydantic_evals.evaluators.conversation.judge_role_adherence',
        return_value=GradingOutput(reason='stayed in role', pass_=True, score=1.0),
    )
    tree = _tree_with_all_messages(
        [
            {'role': 'system', 'parts': [{'type': 'text', 'content': 'You are polite.'}]},
            {'role': 'user', 'parts': [{'type': 'text', 'content': 'hey'}]},
            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hello!'}]},
        ]
    )
    evaluator = RoleAdherence(role='polite helper')
    result = await evaluator.evaluate_async(_ctx_with_tree(tree))
    assert result == snapshot(
        {
            'role_adhered': EvaluationReason(value=True, reason='stayed in role'),
            'role_adherence_score': 1.0,
        }
    )
    mock_judge.assert_called_once()
    assert mock_judge.call_args.kwargs['role'] == 'polite helper'


async def test_role_adherence_include_reason_false(mocker: MockerFixture):
    mocker.patch(
        'pydantic_evals.evaluators.conversation.judge_role_adherence',
        return_value=GradingOutput(reason='ok', pass_=True, score=0.8),
    )
    tree = _tree_with_all_messages([{'role': 'user', 'parts': [{'type': 'text', 'content': 'go'}]}])
    evaluator = RoleAdherence(role='x', include_reason=False)
    result = await evaluator.evaluate_async(_ctx_with_tree(tree))
    assert result == {'role_adhered': True, 'role_adherence_score': 0.8}


# --------------------------------------------------------------------------- serialization


async def test_evaluator_serialization_model_passthrough():
    """A string model is preserved; a `Model` instance is serialized as its `model_id`."""
    evaluator = ConversationGoalAchievement(goal='x', model='openai:gpt-5.2')
    args = evaluator.build_serialization_arguments()
    assert args['model'] == 'openai:gpt-5.2'

    evaluator_role = RoleAdherence(role='y', model='anthropic:claude-opus-4-5')
    args_role = evaluator_role.build_serialization_arguments()
    assert args_role['model'] == 'anthropic:claude-opus-4-5'


async def test_evaluator_serialization_model_instance():
    """A `Model` instance is serialized using its `model_id`."""
    from pydantic_ai.models.test import TestModel

    model = TestModel()
    args = ConversationGoalAchievement(goal='x', model=model).build_serialization_arguments()
    assert args['model'] == model.model_id

    args_role = RoleAdherence(role='y', model=model).build_serialization_arguments()
    assert args_role['model'] == model.model_id


async def test_conversation_evaluators_are_registered_as_defaults():
    """The new evaluators are surfaced in DEFAULT_EVALUATORS so specs can deserialize them."""
    from pydantic_evals.evaluators.common import DEFAULT_EVALUATORS

    assert ConversationGoalAchievement in DEFAULT_EVALUATORS
    assert RoleAdherence in DEFAULT_EVALUATORS
