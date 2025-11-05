from __future__ import annotations as _annotations

from collections.abc import Hashable
from dataclasses import dataclass

import pytest

from pydantic_ai import PartStartEvent, TextPart, ThinkingPart
from pydantic_ai._parts_manager import ModelResponsePart, ModelResponsePartsManager
from pydantic_ai.messages import ModelResponseStreamEvent


def stream_text_deltas(
    chunks: list[str],
    vendor_part_id: Hashable | None = 'content',
    thinking_tags: tuple[str, str] | None = ('<think>', '</think>'),
    ignore_leading_whitespace: bool = False,
) -> tuple[list[ModelResponseStreamEvent], list[ModelResponsePart]]:
    """Helper to stream chunks through manager and return all events + final parts."""
    manager = ModelResponsePartsManager()
    all_events: list[ModelResponseStreamEvent] = []

    for chunk in chunks:
        for event in manager.handle_text_delta(
            vendor_part_id=vendor_part_id,
            content=chunk,
            thinking_tags=thinking_tags,
            ignore_leading_whitespace=ignore_leading_whitespace,
        ):
            all_events.append(event)

    for event in manager.finalize():
        all_events.append(event)

    return all_events, manager.get_parts()


@dataclass
class Case:
    name: str
    chunks: list[str]
    expected_parts: list[ModelResponsePart]  # [TextPart|ThinkingPart('final content')]
    vendor_part_id: Hashable | None = 'content'
    ignore_leading_whitespace: bool = False


CASES: list[Case] = [
    # --- Isolated opening/partial tags -> TextPart (flush via finalize) ---
    Case(
        name='incomplete_opening_tag_only',
        chunks=['<thi'],
        expected_parts=[TextPart('<thi')],
    ),
    Case(
        name='isolated_opening_tag_only',
        chunks=['<think>'],
        expected_parts=[TextPart('<think>')],
    ),
    # --- Isolated opening/partial tags with no vendor id -> TextPart ---
    Case(
        name='incomplete_opening_tag_only_no_vendor_id',
        chunks=['<thi'],
        expected_parts=[TextPart('<thi')],
        vendor_part_id=None,
    ),
    Case(
        name='isolated_opening_tag_only_no_vendor_id',
        chunks=['<think>'],
        expected_parts=[TextPart('<think>')],
        vendor_part_id=None,
    ),
    # --- Split thinking tags -> ThinkingPart ---
    Case(
        name='open_with_content_then_close',
        chunks=['<think>content', '</think>'],
        expected_parts=[ThinkingPart('content')],
    ),
    Case(
        name='open_then_content_and_close',
        chunks=['<think>', 'content</think>'],
        expected_parts=[ThinkingPart('content')],
    ),
    Case(
        name='fully_split_open_and_close',
        chunks=['<th', 'ink>content</th', 'ink>'],
        expected_parts=[ThinkingPart('content')],
    ),
    Case(
        name='split_content_across_chunks',
        chunks=['<think>con', 'tent</think>'],
        expected_parts=[ThinkingPart('content')],
    ),
    # --- Non-closed thinking tag -> ThinkingPart (finalize closes) ---
    Case(
        name='non_closed_thinking_generates_thinking_part',
        chunks=['<think>content'],
        expected_parts=[ThinkingPart('content')],
    ),
    # --- Partial closing tag buffered/then appended if stream ends ---
    Case(
        name='partial_close_appended_on_finalize',
        chunks=['<think>content', '</th'],
        expected_parts=[ThinkingPart('content</th')],
    ),
    # --- Tags not at start of chunk -> TextPart (pretext) ---
    Case(
        name='pretext_then_thinking_tag_same_chunk_textpart',
        chunks=['prethink<think>content</think>'],
        expected_parts=[TextPart('prethink<think>content</think>')],
    ),
    # --- Leading whitespace handling (toggle by ignore_leading_whitespace) ---
    Case(
        name='leading_whitespace_allowed_when_flag_true',
        chunks=['\n<think>content'],
        expected_parts=[ThinkingPart('content')],
        ignore_leading_whitespace=True,
    ),
    Case(
        name='leading_whitespace_not_allowed_when_flag_false',
        chunks=['\n<think>content'],
        expected_parts=[TextPart('\n<think>content')],
        ignore_leading_whitespace=False,
    ),
    Case(
        name='split_with_leading_ws_then_open_tag_flag_true',
        chunks=[' \t\n<th', 'ink>content</think>'],
        expected_parts=[ThinkingPart('content')],
        ignore_leading_whitespace=True,
    ),
    Case(
        name='split_with_leading_ws_then_open_tag_flag_false',
        chunks=[' \t\n<th', 'ink>content</think>'],
        expected_parts=[TextPart(' \t\n<think>content</think>')],
        ignore_leading_whitespace=False,
    ),
    # Test case where whitespace is in separate chunk from tag - this should work with the flag
    Case(
        name='leading_ws_separate_chunk_split_tag_flag_true',
        chunks=[' \t\n', '<th', 'ink>content</think>'],
        expected_parts=[ThinkingPart('content')],
        ignore_leading_whitespace=True,
    ),
    # --- Text after closing tag ---
    Case(
        name='text_after_closing_tag_same_chunk',
        chunks=['<think>content</think>after'],
        expected_parts=[ThinkingPart('content'), TextPart('after')],
    ),
    Case(
        name='text_after_closing_tag_next_chunk',
        chunks=['<think>content</think>', 'after'],
        expected_parts=[ThinkingPart('content'), TextPart('after')],
    ),
    Case(
        name='split_close_tag_then_text',
        chunks=['<think>content</th', 'ink>after'],
        expected_parts=[ThinkingPart('content'), TextPart('after')],
    ),
    Case(
        name='multiple_thinking_parts_with_text_between',
        chunks=['<think>first</think>between<think>second</think>'],
        expected_parts=[ThinkingPart('first'), TextPart('between<think>second</think>')],  # right
        # expected_parts=[ThinkingPart('first'), TextPart('between'), ThinkingPart('second')], # wrong
    ),
]


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.name)
def test_thinking_parts_parametrized(case: Case) -> None:
    """
    Parametrized coverage for all cases described in the report.
    Each case defines:
      - input stream chunks
      - expected list of parts [(type, final_content), ...]
      - optional ignore_leading_whitespace toggle
    """
    events, final_parts = stream_text_deltas(
        chunks=case.chunks,
        vendor_part_id=case.vendor_part_id,
        thinking_tags=('<think>', '</think>'),
        ignore_leading_whitespace=case.ignore_leading_whitespace,
    )

    # Parts observed from final state (after all deltas have been applied)
    assert final_parts == case.expected_parts, f'\nObserved: {final_parts}\nExpected: {case.expected_parts}'

    # 1) For ThinkingPart cases, we should have exactly one PartStartEvent (per ThinkingPart).
    thinking_count = sum(1 for part in final_parts if isinstance(part, ThinkingPart))
    if thinking_count:
        starts = [e for e in events if isinstance(e, PartStartEvent) and isinstance(e.part, ThinkingPart)]
        assert len(starts) == thinking_count, 'Each ThinkingPart should have a single PartStartEvent.'

    # 2) Isolated opening tags should not emit a ThinkingPart start without content.
    if case.name in {'isolated_opening_tag_only', 'incomplete_opening_tag_only'}:
        assert all(not (isinstance(e, PartStartEvent) and isinstance(e.part, ThinkingPart)) for e in events), (
            'No ThinkingPart PartStartEvent should be emitted without content.'
        )
