from __future__ import annotations as _annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import pytest

from pydantic_ai import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta, ThinkingPart, ThinkingPartDelta
from pydantic_ai._parts_manager import ModelResponsePart, ModelResponsePartsManager
from pydantic_ai.messages import ModelResponseStreamEvent


def stream_text_deltas(case: Case) -> tuple[list[ModelResponseStreamEvent], list[ModelResponsePart]]:
    """Helper to stream chunks through manager and return all events + final parts."""
    manager = ModelResponsePartsManager()
    all_events: list[ModelResponseStreamEvent] = []

    for chunk in case.chunks:
        for event in manager.handle_text_delta(
            vendor_part_id=case.vendor_part_id,
            content=chunk,
            thinking_tags=case.thinking_tags,
            ignore_leading_whitespace=case.ignore_leading_whitespace,
        ):
            all_events.append(event)

    for event in manager.final_flush():
        all_events.append(event)

    return all_events, manager.get_parts()


@dataclass
class Case:
    name: str
    chunks: list[str]
    expected_parts: list[ModelResponsePart]  # [TextPart|ThinkingPart('final content')]
    expected_events: Sequence[ModelResponseStreamEvent]
    vendor_part_id: Hashable | None = 'content'
    ignore_leading_whitespace: bool = False
    thinking_tags: tuple[str, str] | None = ('<think>', '</think>')


# Category 1: Opening Tag Handling (partial openings, splits, completes, empties)
OPENING_TAG_CASES: list[Case] = [
    Case(
        name='new_part_with_vendor_id_clean_partial_opening',
        chunks=['<thi'],
        expected_parts=[TextPart('<thi')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('<thi')),
        ],
    ),
    Case(
        name='new_part_with_vendor_id_complete_opening_with_content',
        chunks=['<think>content'],
        expected_parts=[ThinkingPart('content')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
    ),
    Case(
        name='existing_buffer_with_vendor_id_multi_partial_opening_invalid_flush',
        chunks=['<t', 'hi', 'foo'],
        expected_parts=[TextPart('<thifoo')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart(content='<thifoo')),
        ],
    ),
    Case(
        name='no_vendor_id_split_partial_opening_completes_empty_treated_as_text',
        chunks=['<th', 'ink>'],
        expected_parts=[TextPart('<think>')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('<th')),
            PartDeltaEvent(index=0, delta=TextPartDelta('ink>')),
        ],
        vendor_part_id=None,
    ),
]

# Category 2: Invalid Opening Tags (prefixes, invalid continuations, flushes)
INVALID_OPENING_CASES: list[Case] = [
    Case(
        name='existing_buffer_with_vendor_id_invalid_partial_opening_flush',
        chunks=['<thi', 'foo'],
        expected_parts=[TextPart('<thifoo')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart(content='<thifoo')),
        ],
    ),
    Case(
        name='new_part_with_vendor_id_invalid_opening_with_prefix',
        chunks=['pre<think>'],
        expected_parts=[TextPart('pre<think>')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('pre<think>')),
        ],
    ),
]

# Category 3: Full Thinking Tags (complete cycles: open + content + close, with/without after)
FULL_THINKING_CASES: list[Case] = [
    Case(
        name='new_part_with_vendor_id_empty_thinking_treated_as_text',
        chunks=['<think></think>'],
        expected_parts=[TextPart('<think></think>')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('<think></think>')),
        ],
    ),
    Case(
        name='new_part_with_vendor_id_empty_thinking_with_after_treated_as_text',
        chunks=['<think></think>more'],
        expected_parts=[TextPart('<think></think>more')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('<think></think>more')),
        ],
    ),
    Case(
        name='new_part_with_vendor_id_complete_thinking_with_content_no_after',
        chunks=['<think>content</think>'],
        expected_parts=[ThinkingPart('content')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
    ),
    Case(
        name='new_part_with_vendor_id_complete_thinking_with_content_with_after',
        chunks=['<think>content</think>more'],
        expected_parts=[ThinkingPart('content'), TextPart('more')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartStartEvent(index=1, part=TextPart('more')),
        ],
    ),
]

# Category 4: Closing Tag Handling (clean closings, with before/after, no before)
CLOSING_TAG_CASES: list[Case] = [
    Case(
        name='existing_thinking_with_vendor_id_clean_closing',
        chunks=['<think>content', '</think>'],
        expected_parts=[ThinkingPart('content')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
    ),
    Case(
        name='existing_thinking_with_vendor_id_closing_with_before',
        chunks=['<think>content', 'more</think>'],
        expected_parts=[ThinkingPart('contentmore')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
        ],
    ),
    Case(
        name='existing_thinking_with_vendor_id_closing_with_before_after',
        chunks=['<think>content', 'more</think>after'],
        expected_parts=[ThinkingPart('contentmore'), TextPart('after')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
            PartStartEvent(index=1, part=TextPart('after')),
        ],
    ),
    Case(
        name='existing_thinking_with_vendor_id_closing_no_before_with_after',
        chunks=['<think>content', '</think>after'],
        expected_parts=[ThinkingPart('content'), TextPart('after')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartStartEvent(index=1, part=TextPart('after')),
        ],
    ),
]

# Category 5: Partial Closing Tags (partials, overlaps, completes, with content)
PARTIAL_CLOSING_CASES: list[Case] = [
    Case(
        name='new_part_with_vendor_id_opening_with_content_partial_closing',
        chunks=['<think>content</th'],
        expected_parts=[ThinkingPart('content')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
    ),
    Case(
        name='existing_thinking_with_vendor_id_partial_closing',
        chunks=['<think>content', '</th'],
        expected_parts=[ThinkingPart('content')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
    ),
    Case(
        name='existing_thinking_buffer_with_vendor_id_completes_partial_closing',
        chunks=['<think>content', '</th', 'ink>'],
        expected_parts=[ThinkingPart('content')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
        ],
    ),
    Case(
        name='existing_thinking_with_vendor_id_partial_closing_with_content_to_add',
        chunks=['<think>content', 'more</th'],
        expected_parts=[ThinkingPart('contentmore')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
        ],
    ),
    Case(
        name='existing_thinking_buffer_with_vendor_id_multi_partial_closing_completes',
        chunks=['<think>content', 'more</', 'thi', 'nk>'],
        expected_parts=[ThinkingPart('contentmore')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
        ],
    ),
    Case(
        name='new_part_with_vendor_id_empty_thinking_with_partial_closing_treated_as_text',
        chunks=['<think></th'],
        expected_parts=[TextPart('<think></th')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('<think></th')),
        ],
    ),
    # existing_thinking_with_vendor_id_partial_closing_overlap_non_empty_content_to_add_fake_closing
    Case(
        name='existing_thinking_with_vendor_id_partial_closing_overlap_non_empty_content_to_add_completes',
        chunks=['<think>content', 'more</th', 'fooink>'],
        expected_parts=[ThinkingPart('contentmore</thfooink>')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='</thfooink>')),
        ],
    ),
]

# Category 6: Fake or Invalid Closing (added to content)
FAKE_CLOSING_CASES: list[Case] = [
    Case(
        name='existing_thinking_with_vendor_id_fake_closing_added_to_thinking',
        chunks=['<think>content', '</fake>'],
        expected_parts=[ThinkingPart('content</fake>')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='</fake>')),
        ],
    ),
    Case(
        name='existing_thinking_with_vendor_id_fake_partial_closing_added_to_content',
        chunks=['<think>content', '</tx'],
        expected_parts=[ThinkingPart('content</tx')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='</tx')),
        ],
    ),
]

# Category 7: Adding Content to Existing (updates without closing)
ADDING_CONTENT_CASES: list[Case] = [
    Case(
        name='existing_thinking_with_vendor_id_add_more_content',
        chunks=['<think>content', 'more'],
        expected_parts=[ThinkingPart('contentmore')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='more')),
        ],
    ),
]

# Category 8: Whitespace Handling (ignore leading, mixed, not ignore)
WHITESPACE_CASES: list[Case] = [
    Case(
        name='new_part_with_vendor_id_ignore_whitespace_empty',
        chunks=['   '],
        expected_parts=[],
        expected_events=[],
        ignore_leading_whitespace=True,
    ),
    Case(
        name='new_part_with_vendor_id_not_ignore_whitespace',
        chunks=['   '],
        expected_parts=[TextPart('   ')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('   ')),
        ],
    ),
    Case(
        name='new_part_no_vendor_id_ignore_whitespace_not_empty',
        chunks=['   content'],
        expected_parts=[TextPart('   content')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('   content')),
        ],
        vendor_part_id=None,
        ignore_leading_whitespace=True,
    ),
    Case(
        name='new_part_with_vendor_id_ignore_whitespace_mixed_with_partial_opening',
        chunks=['  <thi'],
        expected_parts=[TextPart('  <thi')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('  <thi')),
        ],
        ignore_leading_whitespace=True,
    ),
]

# Category 9: No Vendor ID (updates, new after thinking, closings as text)
NO_VENDOR_ID_CASES: list[Case] = [
    Case(
        name='update_latest_no_vendor_id_text',
        chunks=['hello', ' world'],
        expected_parts=[TextPart('hello world')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('hello')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' world')),
        ],
        vendor_part_id=None,
    ),
    Case(
        name='new_part_no_vendor_id_after_thinking',
        chunks=['<think>content', 'more'],
        expected_parts=[ThinkingPart('content'), TextPart('more')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartStartEvent(index=1, part=TextPart('more')),
        ],
        vendor_part_id=None,
    ),
    Case(
        name='no_vendor_id_closing_treated_as_text',
        chunks=['<think>content', '</think>'],
        expected_parts=[ThinkingPart('content'), TextPart('</think>')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartStartEvent(index=1, part=TextPart('</think>')),
        ],
        vendor_part_id=None,
    ),
    Case(
        name='no_vendor_id_after_thinking_add_partial_closing_treated_as_text',
        chunks=['<think>content', '</th'],
        expected_parts=[ThinkingPart('content'), TextPart('</th')],
        expected_events=[
            PartStartEvent(index=0, part=ThinkingPart('content')),
            PartStartEvent(index=1, part=TextPart('</th')),
        ],
        vendor_part_id=None,
    ),
]

# Category 10: No Thinking Tags (tags treated as text)
NO_THINKING_TAGS_CASES: list[Case] = [
    Case(
        name='new_part_with_vendor_id_tags_as_text_when_thinking_tags_none',
        chunks=['<think>content</think>'],
        expected_parts=[TextPart('<think>content</think>')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('<think>content</think>')),
        ],
        vendor_part_id='content',
        ignore_leading_whitespace=False,
        thinking_tags=None,
    ),
    Case(
        name='new_part_with_vendor_id_partial_closing_as_text_when_thinking_tags_none',
        chunks=['content</th', 'ink>'],
        expected_parts=[TextPart('content</think>')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('content</th')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='ink>')),
        ],
        thinking_tags=None,
    ),
]

# Category 11: Buffer Management (stutter, flushed)
BUFFER_MANAGEMENT_CASES: list[Case] = [
    Case(
        name='existing_text_with_vendor_id_stutter_buffer_via_replace',
        chunks=['<thi', '<think>content</think>'],
        expected_parts=[TextPart('<thi'), ThinkingPart('content')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart(content='<thi')),
            PartStartEvent(index=1, part=ThinkingPart(content='content')),
        ],
    ),
    Case(
        name='stutter_buffer_in_non_last_part_handled_as_text',
        chunks=['<thi', 'content'],
        # E         Observed: [TextPart(content=''), TextPart(content='content')]
        expected_parts=[TextPart('<thicontent')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart(content='<thicontent')),
        ],
    ),
    Case(
        name='existing_text_with_vendor_id_add_partial_opening_flush',
        chunks=['hello', '<thi'],
        expected_parts=[TextPart('hello<thi')],
        expected_events=[
            PartStartEvent(index=0, part=TextPart('hello')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='<thi')),
        ],
    ),
    Case(
        name='existing_text_with_vendor_id_stutter_via_append_non_empty_content',
        chunks=['hello', '<thi', '<think>content</think>'],
        expected_parts=[TextPart('hello<thi'), ThinkingPart('content')],
        # E         Observed: [TextPart(content='hello<thi<think>content</think>')]
        expected_events=[
            PartStartEvent(index=0, part=TextPart('hello')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='<thi')),
            PartStartEvent(index=1, part=ThinkingPart('content')),
        ],
    ),
]

# Category 12: Delayed Thinking (no event until content after complete opening)
# TODO these need to be updated, they currently test only delayed thinkig -> flush
# they should test delayed thinking -> real thinking
# delayed thinking -> false alarm (handle as text)
# delayed thinking -> partial closing -> flush
# etc...
# commented out until fixed
DELAYED_THINKING_CASES: list[Case] = [
    Case(
        name='new_part_with_vendor_id_split_partial_complete_opening_delayed_thinking',
        chunks=['<th', 'ink>'],
        expected_parts=[TextPart('<think>')],
        expected_events=[PartStartEvent(index=0, part=TextPart('<think>'))],
    ),
    Case(
        name='new_part_with_vendor_id_complete_opening_delayed_thinking',
        chunks=['<think>'],
        expected_parts=[TextPart('<think>')],
        expected_events=[PartStartEvent(index=0, part=TextPart('<think>'))],
    ),
    # TODO redundant with first case in this list. Placerholder for new edge cases.
    Case(
        name='new_part_with_vendor_id_multi_partial_complete_opening_delayed_thinking',
        chunks=['<t', 'hi', 'nk>'],
        expected_parts=[TextPart('<think>')],
        expected_events=[PartStartEvent(index=0, part=TextPart('<think>'))],
    ),
]

ALL_CASES = (
    OPENING_TAG_CASES
    + INVALID_OPENING_CASES
    + FULL_THINKING_CASES
    + CLOSING_TAG_CASES
    + PARTIAL_CLOSING_CASES
    + FAKE_CLOSING_CASES
    + ADDING_CONTENT_CASES
    + WHITESPACE_CASES
    + NO_VENDOR_ID_CASES
    + NO_THINKING_TAGS_CASES
    + BUFFER_MANAGEMENT_CASES
    # + DELAYED_THINKING_CASES
)


@pytest.mark.parametrize('case', ALL_CASES, ids=lambda c: c.name)
def test_thinking_parts_parametrized(case: Case) -> None:
    """
    Parametrized coverage for all cases described in the report.
    """
    events, final_parts = stream_text_deltas(case)

    # Parts observed from final state (after all deltas have been applied)
    assert final_parts == case.expected_parts, f'\nObserved: {final_parts}\nExpected: {case.expected_parts}'

    # Events observed from streaming and final_flush
    assert events == case.expected_events, f'\nObserved: {events}\nExpected: {case.expected_events}'
