"""Tests for split thinking tag handling in ModelResponsePartsManager."""

from inline_snapshot import snapshot

from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)


def test_handle_text_deltas_with_split_think_tags_at_chunk_start():
    """Test split thinking tags when tag starts at position 0 of chunk."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Chunk 1: "<thi" - starts at position 0, buffer it  # codespell:ignore thi
    events = list(manager.handle_text_delta(vendor_part_id='content', content='<thi', thinking_tags=thinking_tags))
    assert len(events) == 0  # Buffered, no events yet
    assert manager.get_parts() == []

    # Chunk 2: "nk>" - completes the tag
    events = list(manager.handle_text_delta(vendor_part_id='content', content='nk>', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(index=0, part=ThinkingPart(content='', part_kind='thinking'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([ThinkingPart(content='', part_kind='thinking')])

    # Chunk 3: "reasoning content"
    events = list(
        manager.handle_text_delta(vendor_part_id='content', content='reasoning content', thinking_tags=thinking_tags)
    )
    assert len(events) == 1
    assert events[0] == snapshot(
        PartDeltaEvent(
            index=0,
            delta=ThinkingPartDelta(content_delta='reasoning content', part_delta_kind='thinking'),
            event_kind='part_delta',
        )
    )

    # Chunk 4: "</think>" - end tag
    events = list(manager.handle_text_delta(vendor_part_id='content', content='</think>', thinking_tags=thinking_tags))
    assert len(events) == 0

    # Chunk 5: "after" - text after thinking
    events = list(manager.handle_text_delta(vendor_part_id='content', content='after', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(index=1, part=TextPart(content='after', part_kind='text'), event_kind='part_start')
    )


def test_handle_text_deltas_split_tags_after_text():
    """Test split thinking tags at chunk position 0 after text in previous chunk."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Chunk 1: "pre-" - creates TextPart
    events = list(manager.handle_text_delta(vendor_part_id='content', content='pre-', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(index=0, part=TextPart(content='pre-', part_kind='text'), event_kind='part_start')
    )

    # Chunk 2: "<thi" - starts at position 0 of THIS chunk, buffer it
    events = list(manager.handle_text_delta(vendor_part_id='content', content='<thi', thinking_tags=thinking_tags))
    assert len(events) == 0  # Buffered
    assert manager.get_parts() == snapshot([TextPart(content='pre-', part_kind='text')])

    # Chunk 3: "nk>" - completes the tag
    events = list(manager.handle_text_delta(vendor_part_id='content', content='nk>', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(index=1, part=ThinkingPart(content='', part_kind='thinking'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot(
        [TextPart(content='pre-', part_kind='text'), ThinkingPart(content='', part_kind='thinking')]
    )


def test_handle_text_deltas_split_tags_mid_chunk_treated_as_text():
    """Test that split tags mid-chunk (after other content in same chunk) are treated as text."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Chunk 1: "pre-<thi" - tag does NOT start at position 0 of chunk
    events = list(manager.handle_text_delta(vendor_part_id='content', content='pre-<thi', thinking_tags=thinking_tags))
    assert len(events) == 1  # Treated as text, not buffered
    assert events[0] == snapshot(
        PartStartEvent(index=0, part=TextPart(content='pre-<thi', part_kind='text'), event_kind='part_start')
    )

    # Chunk 2: "nk>" - appends to text (not recognized as completing a tag)
    events = list(manager.handle_text_delta(vendor_part_id='content', content='nk>', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert events[0] == snapshot(
        PartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta='nk>', part_delta_kind='text'), event_kind='part_delta'
        )
    )
    assert manager.get_parts() == snapshot([TextPart(content='pre-<think>', part_kind='text')])


def test_handle_text_deltas_split_tags_no_vendor_id():
    """Test that split tags don't work with vendor_part_id=None (no buffering)."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Chunk 1: "<thi" with no vendor_part_id - can't buffer
    events = list(manager.handle_text_delta(vendor_part_id=None, content='<thi', thinking_tags=thinking_tags))
    assert len(events) == 1  # Treated as text immediately (simple path)
    assert events[0] == snapshot(
        PartStartEvent(index=0, part=TextPart(content='<thi', part_kind='text'), event_kind='part_start')
    )

    # Chunk 2: "nk>" - appends to text
    events = list(manager.handle_text_delta(vendor_part_id=None, content='nk>', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert events[0] == snapshot(
        PartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta='nk>', part_delta_kind='text'), event_kind='part_delta'
        )
    )
    assert manager.get_parts() == snapshot([TextPart(content='<think>', part_kind='text')])


def test_handle_text_deltas_false_start_then_real_tag():
    """Test buffering a false start, then processing real content."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Chunk 1: "<th" - could be tag start, buffer it
    events = list(manager.handle_text_delta(vendor_part_id='content', content='<th', thinking_tags=thinking_tags))
    assert len(events) == 0  # Buffered

    # Chunk 2: "is is text" - proves it's not a tag, flush buffer
    events = list(
        manager.handle_text_delta(vendor_part_id='content', content='is is text', thinking_tags=thinking_tags)
    )
    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(index=0, part=TextPart(content='<this is text', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([TextPart(content='<this is text', part_kind='text')])


def test_buffered_content_exceeds_tag_length():
    """Test that buffered content longer than tag is flushed (covers line 231)."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # To hit line 231, we need:
    # 1. Buffer some content
    # 2. Next chunk starts with '<' (to pass first check)
    # 3. Combined length >= tag length

    # First chunk: exactly 6 chars
    events = list(manager.handle_text_delta(vendor_part_id='content', content='<think', thinking_tags=thinking_tags))
    assert len(events) == 0  # Buffered

    # Second chunk: starts with '<' so it checks _could_be_tag_start
    # Combined will be '<think<' (7 chars) which equals tag length '<think>' (7 chars)
    events = list(manager.handle_text_delta(vendor_part_id='content', content='<', thinking_tags=thinking_tags))
    # 7 >= 7 is True, so line 231 returns False
    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(index=0, part=TextPart(content='<think<', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([TextPart(content='<think<', part_kind='text')])


def test_complete_thinking_tag_no_vendor_id():
    """Test complete thinking tag with vendor_part_id=None (covers lines 161-164)."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Complete start tag with vendor_part_id=None goes through simple path
    # This covers lines 161-164 in _handle_text_delta_simple
    events = list(manager.handle_text_delta(vendor_part_id=None, content='<think>', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(index=0, part=ThinkingPart(content='', part_kind='thinking'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([ThinkingPart(content='', part_kind='thinking')])


def test_exact_tag_length_boundary():
    """Test when buffered content exactly equals tag length."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Send content in one chunk that's exactly tag length
    events = list(manager.handle_text_delta(vendor_part_id='content', content='<think>', thinking_tags=thinking_tags))
    # Exact match creates ThinkingPart
    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(index=0, part=ThinkingPart(content='', part_kind='thinking'), event_kind='part_start')
    )
