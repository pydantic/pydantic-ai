"""This file tests the "embedded thinking handling" functionality of the Parts Manager (_parts_manager.py).

It tests each case with both vendor_part_id='content' and vendor_part_id=None to ensure consistent behavior.
"""

from __future__ import annotations as _annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai._parts_manager import ModelResponsePart, ModelResponsePartsManager
from pydantic_ai.messages import ModelResponseStreamEvent


def stream_text_deltas(
    case: Case,
) -> tuple[list[ModelResponseStreamEvent], list[ModelResponseStreamEvent], list[ModelResponsePart]]:
    """Helper to stream chunks through manager and return all events + final parts."""
    manager = ModelResponsePartsManager()
    normal_events: list[ModelResponseStreamEvent] = []

    for chunk in case.chunks:
        for event in manager.handle_text_delta(
            vendor_part_id=case.vendor_part_id,
            content=chunk,
            thinking_tags=case.thinking_tags,
            ignore_leading_whitespace=case.ignore_leading_whitespace,
        ):
            normal_events.append(event)

    flushed_events: list[ModelResponseStreamEvent] = []
    for event in manager.final_flush():
        flushed_events.append(event)

    return normal_events, flushed_events, manager.get_parts()


def init_model_response_stream_event_iterator() -> Sequence[ModelResponseStreamEvent]:
    # both pyright and pre-commit asked for this
    return []


@dataclass
class Case:
    name: str
    chunks: list[str]
    expected_parts: list[ModelResponsePart]  # [TextPart|ThinkingPart('final content')]
    expected_normal_events: Sequence[ModelResponseStreamEvent]
    expected_flushed_events: Sequence[ModelResponseStreamEvent] = field(
        default_factory=init_model_response_stream_event_iterator
    )
    vendor_part_id: Hashable | None = 'content'
    ignore_leading_whitespace: bool = False
    thinking_tags: tuple[str, str] | None = ('<think>', '</think>')


FULL_SPLITS = [
    Case(
        name='full_split_partial_closing',
        chunks=['<th', 'ink>con', 'tent</th'],
        expected_parts=[ThinkingPart('content</th')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('con')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='tent')),
        ],
        expected_flushed_events=[
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='</th')),
        ],
    ),
    Case(
        name='full_split_on_both_sides_clean',
        chunks=['<th', 'ink>con', 'tent</th', 'ink>', 'after'],
        expected_parts=[ThinkingPart('content'), TextPart('after')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('con')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='tent')),
            PartStartEvent(index=1, part=TextPart('after')),
        ],
    ),
    Case(
        name='full_split_on_both_sides_closing_buffer_and_stutter',
        chunks=['<th', 'ink>con', 'tent</th', '</think>', 'after'],
        expected_parts=[ThinkingPart('content</th'), TextPart('after')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('con')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='tent')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='</th')),
            PartStartEvent(index=1, part=TextPart('after')),
        ],
    ),
    Case(
        name='think_to_content_to_partial_opening_gets_flushed',
        chunks=['<th', 'ink>con', 'tent</th', 'ink>', 'after', '<th'],
        expected_parts=[ThinkingPart('content'), TextPart('after<th')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('con')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='tent')),
            PartStartEvent(index=1, part=TextPart('after')),
        ],
        expected_flushed_events=[
            PartDeltaEvent(index=1, delta=TextPartDelta('<th')),
        ],
    ),
]

# Category 1: Opening Tag Handling (partial openings, splits, completes, empties)
OPENING_TAG_CASES: list[Case] = [
    Case(
        name='full_opening_starts_thinking_part',
        chunks=['<think>content'],
        expected_parts=[ThinkingPart('content')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
    ),
]

# Category 2: Delayed Thinking (no event until content after complete opening)
DELAYED_THINKING_CASES: list[Case] = [
    Case(
        name='delayed_thinking_with_content_closes_in_next_chunk',
        chunks=['<think>', 'content</think>'],
        expected_parts=[ThinkingPart('content')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
    ),
    Case(
        name='delayed_thinking_with_leading_whitespace_trimmed',
        chunks=['<think>', '  content', '</think>'],
        expected_parts=[ThinkingPart('content')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
        ignore_leading_whitespace=True,
    ),
    Case(
        name='delayed_empty_thinking_closes_in_separate_chunk_with_after',
        chunks=['<think>', '</think>after'],
        expected_parts=[TextPart('after')],
        expected_normal_events=[
            PartStartEvent(index=0, part=TextPart('after')),
        ],
        # NOTE empty thinking is skipped entirely
        expected_flushed_events=[],
    ),
]

# Category 3: Invalid Opening Tags (prefixes, invalid continuations, flushes)
INVALID_OPENING_CASES: list[Case] = [
    Case(
        name='multiple_partial_openings_buffered_until_invalid_continuation',
        chunks=['<t', 'hi', 'foo'],  # equivalent to ['<thi', 'foo']
        expected_parts=[TextPart('<thifoo')],
        expected_normal_events=[
            PartStartEvent(index=0, part=TextPart(content='<thifoo')),
        ],
    ),
    Case(
        name='new_part_invalid_opening_with_prefix',
        chunks=['pre<think>'],
        expected_parts=[TextPart('pre<think>')],
        expected_normal_events=[
            PartStartEvent(index=0, part=TextPart('pre<think>')),
        ],
    ),
]

# Category 4: Full Thinking Tags (complete cycles: open + content + close, with/without after)
FULL_THINKING_CASES: list[Case] = [
    Case(
        name='new_part_empty_thinking_treated_as_text',
        chunks=['<think></think>'],
        expected_parts=[],  # Empty thinking is now skipped entirely
        expected_normal_events=[],
    ),
    Case(
        name='new_part_empty_thinking_with_after_treated_as_text',
        chunks=['<think></think>more'],
        expected_parts=[TextPart('more')],
        expected_normal_events=[
            PartStartEvent(index=0, part=TextPart('more')),
        ],
    ),
    Case(
        name='new_part_complete_thinking_with_content_no_after',
        chunks=['<think>content</think>'],
        expected_parts=[ThinkingPart('content')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
    ),
    Case(
        name='new_part_complete_thinking_with_content_with_after',
        chunks=['<think>content</think>more'],
        expected_parts=[ThinkingPart('content'), TextPart('more')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartStartEvent(index=1, part=TextPart('more')),
        ],
    ),
]

# Category 5: Closing Tag Handling (clean closings, with before/after, no before)
CLOSING_TAG_CASES: list[Case] = [
    Case(
        name='existing_thinking_clean_closing',
        chunks=['<think>content', '</think>'],
        expected_parts=[ThinkingPart('content')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
    ),
    Case(
        name='existing_thinking_closing_with_before',
        chunks=['<think>content', 'more</think>'],
        expected_parts=[ThinkingPart('contentmore')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
        ],
    ),
    Case(
        name='existing_thinking_closing_with_before_after',
        chunks=['<think>content', 'more</think>after'],
        expected_parts=[ThinkingPart('contentmore'), TextPart('after')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
            PartStartEvent(index=1, part=TextPart('after')),
        ],
    ),
    Case(
        name='existing_thinking_closing_no_before_with_after',
        chunks=['<think>content', '</think>after'],
        expected_parts=[ThinkingPart('content'), TextPart('after')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartStartEvent(index=1, part=TextPart('after')),
        ],
    ),
]

# Category 6: Partial Closing Tags (partials, overlaps, completes, with content)
PARTIAL_CLOSING_CASES: list[Case] = [
    Case(
        name='new_part_opening_with_content_partial_closing',
        chunks=['<think>content</th'],
        expected_parts=[ThinkingPart('content</th')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
        expected_flushed_events=[
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='</th')),
        ],
    ),
    Case(
        name='existing_thinking_partial_closing',
        chunks=['<think>content', '</th'],
        expected_parts=[ThinkingPart('content</th')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
        expected_flushed_events=[
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='</th')),
        ],
    ),
    Case(
        name='existing_thinking_buffer_completes_partial_closing',
        chunks=['<think>content', '</th', 'ink>'],
        expected_parts=[ThinkingPart('content')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
    ),
    Case(
        name='existing_thinking_partial_closing_with_content_to_add',
        chunks=['<think>content', 'more</th'],
        expected_parts=[ThinkingPart('contentmore</th')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
        ],
        expected_flushed_events=[
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='</th')),
        ],
    ),
    Case(
        name='existing_thinking_buffer_multi_partial_closing_completes',
        chunks=['<think>content', 'more</', 'thi', 'nk>'],
        expected_parts=[ThinkingPart('contentmore')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
        ],
    ),
    Case(
        name='new_part_empty_thinking_with_partial_closing_treated_as_text',
        chunks=['<think></th'],
        expected_parts=[TextPart('<think></th')],
        expected_normal_events=[],
        expected_flushed_events=[
            PartStartEvent(index=0, part=TextPart('<think></th')),
        ],
    ),
    # existing_thinking_partial_closing_overlap_non_empty_content_to_add_fake_closing
    Case(
        name='existing_thinking_partial_closing_overlap_non_empty_content_to_add_completes',
        chunks=['<think>content', 'more</th', 'fooink>'],
        expected_parts=[ThinkingPart('contentmore</thfooink>')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='</thfooink>')),
        ],
    ),
]

# Category 7: Adding Content to Existing (updates without closing)
ADDING_CONTENT_CASES: list[Case] = [
    Case(
        name='existing_thinking_add_more_content',
        chunks=['<think>content', 'more'],
        expected_parts=[ThinkingPart('contentmore')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
        ],
    ),
]

# Category 8: Whitespace Handling (ignore leading, mixed, not ignore)
WHITESPACE_CASES: list[Case] = [
    Case(
        name='new_part_ignore_whitespace_empty',
        chunks=['   '],
        expected_parts=[],
        expected_normal_events=[],
        ignore_leading_whitespace=True,
    ),
    Case(
        name='new_part_not_ignore_whitespace',
        chunks=['   '],
        expected_parts=[TextPart('   ')],
        expected_normal_events=[
            PartStartEvent(index=0, part=TextPart('   ')),
        ],
    ),
    Case(
        name='new_part_no_vendor_id_ignore_whitespace_not_empty',
        chunks=['   content'],
        expected_parts=[TextPart('content')],
        expected_normal_events=[
            PartStartEvent(index=0, part=TextPart('content')),
        ],
        ignore_leading_whitespace=True,
    ),
    Case(
        name='new_part_ignore_whitespace_mixed_with_partial_opening',
        chunks=['  <thi'],
        expected_parts=[TextPart('<thi')],
        expected_normal_events=[],
        expected_flushed_events=[
            PartStartEvent(index=0, part=TextPart('<thi')),
        ],
        ignore_leading_whitespace=True,
    ),
    Case(
        name='new_part_ignore_whitespace_mixed_with_full_opening',
        chunks=['  <think>'],
        expected_parts=[TextPart('<think>')],
        expected_normal_events=[],
        expected_flushed_events=[
            PartStartEvent(index=0, part=TextPart('<think>')),
        ],
        ignore_leading_whitespace=True,
    ),
]

# Category 9: No Vendor ID (updates, new after thinking, closings as text)
NO_VENDOR_ID_CASES: list[Case] = []

# Category 10: No Thinking Tags (tags treated as text)
NO_THINKING_TAGS_CASES: list[Case] = [
    Case(
        name='new_part_tags_as_text_when_thinking_tags_none',
        chunks=['<think>content</think>'],
        expected_parts=[TextPart('<think>content</think>')],
        expected_normal_events=[
            PartStartEvent(index=0, part=TextPart('<think>content</think>')),
        ],
        thinking_tags=None,
    )
]

# Category 11: Buffer Management (stutter, flushed)
BUFFER_MANAGEMENT_CASES: list[Case] = [
    Case(
        name='empty_first_chunk_with_buffered_partial_opening_flushed',
        chunks=['', '<thi'],
        expected_parts=[TextPart('<thi')],
        expected_normal_events=[],
        expected_flushed_events=[
            PartStartEvent(index=0, part=TextPart('<thi')),
        ],
    ),
    Case(
        name='existing_text_stutter_buffer_via_tag_prefix_match',
        chunks=['<t', '<thi'],
        expected_parts=[TextPart('<t<thi')],
        expected_normal_events=[
            PartStartEvent(index=0, part=TextPart('<t')),
        ],
        expected_flushed_events=[
            PartDeltaEvent(index=0, delta=TextPartDelta('<thi')),
        ],
    ),
    Case(
        name='existing_text_stutter_buffer_via_replace',
        chunks=['<thi', '<think>content</think>'],
        expected_parts=[TextPart('<thi'), ThinkingPart('content')],
        expected_normal_events=[
            PartStartEvent(index=0, part=TextPart(content='<thi')),
            PartStartEvent(index=1, part=ThinkingPart(content='content')),
        ],
    ),
    Case(
        name='stutter_buffer_in_non_last_part_handled_as_text',
        chunks=['<thi', 'content'],
        expected_parts=[TextPart('<thicontent')],
        expected_normal_events=[
            PartStartEvent(index=0, part=TextPart(content='<thicontent')),
        ],
    ),
    Case(
        name='existing_text_add_partial_opening_flush',
        chunks=['hello', '<thi'],
        expected_parts=[TextPart('hello<thi')],
        expected_normal_events=[
            PartStartEvent(index=0, part=TextPart('hello')),
        ],
        expected_flushed_events=[
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='<thi')),
        ],
    ),
    Case(
        name='existing_text_stutter_via_append_non_empty_content',
        chunks=['hello', '<thi', '<think>content</think>'],
        expected_parts=[TextPart('hello<thi'), ThinkingPart('content')],
        expected_normal_events=[
            PartStartEvent(index=0, part=TextPart('hello')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='<thi')),
            PartStartEvent(index=1, part=ThinkingPart('content')),
        ],
    ),
]

# Category 12: Fake or Invalid Closing (added to content)
FAKE_CLOSING_CASES: list[Case] = [
    Case(
        name='existing_thinking_fake_closing_added_to_thinking',
        chunks=['<think>content', '</', 'fake>'],
        expected_parts=[ThinkingPart('content</fake>')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='</fake>')),
        ],
    ),
    Case(
        name='existing_thinking_fake_partial_closing_added_to_content',
        chunks=['<think>content', '</tx'],
        expected_parts=[ThinkingPart('content</tx')],
        expected_normal_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='</tx')),
        ],
    ),
]

# TOOL_PART_CASES = [
#     Case(
#         name='buffered_closing_with_tool_part',
#         chunks=['<thi', 'nk>foo', 'bar</th', ToolCallPart('dummy_tool_name')],
#         expected_parts=[ThinkingPart('foobar</th'), ToolCallPart('dummy_tool_name')],
#         expected_normal_events=[
#             PartStartEvent(index=0, part=ThinkingPart('foo')),
#             PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='bar')),
#             # PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='</th')),
#             PartStartEvent(index=1, part=ToolCallPart('dummy_tool_name')),
#         ],
#     ),
# ]


ALL_CASES = (
    FULL_SPLITS
    + DELAYED_THINKING_CASES
    + OPENING_TAG_CASES
    + INVALID_OPENING_CASES
    + FULL_THINKING_CASES
    + CLOSING_TAG_CASES
    + PARTIAL_CLOSING_CASES
    + ADDING_CONTENT_CASES
    + WHITESPACE_CASES
    + NO_VENDOR_ID_CASES
    + NO_THINKING_TAGS_CASES
    + BUFFER_MANAGEMENT_CASES
    + FAKE_CLOSING_CASES
)


@pytest.mark.parametrize('case', ALL_CASES, ids=lambda c: c.name)
@pytest.mark.parametrize('vendor_part_id', ['content', None], ids=['with_vendor_id', 'without_vendor_id'])
def test_thinking_parts_parametrized(case: Case, vendor_part_id: str | None) -> None:
    """
    Parametrized coverage for all cases described in the report.
    Tests each case with both vendor_part_id='content' and vendor_part_id=None.
    """
    case.vendor_part_id = vendor_part_id

    normal_events, flushed_events, final_parts = stream_text_deltas(case)

    # Parts observed from final state (after all deltas have been applied)
    assert final_parts == case.expected_parts, f'\nObserved: {final_parts}\nExpected: {case.expected_parts}'

    # Events observed from streaming during normal processing
    assert normal_events == case.expected_normal_events, (
        f'\nObserved: {normal_events}\nExpected: {case.expected_normal_events}'
    )

    # Events observed from final_flush
    assert flushed_events == case.expected_flushed_events, (
        f'\nObserved: {flushed_events}\nExpected: {case.expected_flushed_events}'
    )


def test_final_flush_with_partial_tag_on_non_latest_part():
    """Test that final_flush properly handles partial tags attached to earlier parts."""
    manager = ModelResponsePartsManager()

    # Create ThinkingPart at index 0 with partial closing tag buffered
    for _ in manager.handle_text_delta(
        vendor_part_id='thinking',
        content='<think>content<',
        thinking_tags=('<think>', '</think>'),
    ):
        pass

    # Create new part at index 1 using different vendor_part_id (makes ThinkingPart non-latest)
    # Use tool call to create a different part type
    manager.handle_tool_call_delta(
        vendor_part_id='tool',
        tool_name='my_tool',
        args='{}',
    )

    # final_flush should emit PartDeltaEvent to index 0 (non-latest ThinkingPart with buffered '<')
    events = list(manager.final_flush())
    assert events == snapshot([PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='<'))])
