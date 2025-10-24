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


def test_buffered_content_flushed_on_finalize():
    """Test that buffered content is flushed when finalize is called."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Buffer partial tag
    events = list(manager.handle_text_delta(vendor_part_id='content', content='<thi', thinking_tags=thinking_tags))
    assert len(events) == 0  # Buffered

    # Finalize should flush buffer
    final_events = list(manager.finalize())
    assert len(final_events) == 1
    assert final_events[0] == snapshot(
        PartStartEvent(index=0, part=TextPart(content='<thi', part_kind='text'), event_kind='part_start')
    )


def test_finalize_flushes_all_buffers():
    """Test that finalize flushes all vendor_part_id buffers."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Buffer for vendor_id_1
    list(manager.handle_text_delta(vendor_part_id='id1', content='<th', thinking_tags=thinking_tags))

    # Buffer for vendor_id_2
    list(manager.handle_text_delta(vendor_part_id='id2', content='<thi', thinking_tags=thinking_tags))

    # Finalize should flush both
    final_events = list(manager.finalize())
    assert len(final_events) == 2

    # Both should become TextParts
    parts = manager.get_parts()
    assert len(parts) == 2
    assert all(isinstance(p, TextPart) for p in parts)
    # Note: order may vary, so check content exists
    text_parts = [p for p in parts if isinstance(p, TextPart)]
    contents = {p.content for p in text_parts}
    assert contents == {'<th', '<thi'}


def test_finalize_with_no_buffer():
    """Test that finalize is safe when buffer is empty."""
    manager = ModelResponsePartsManager()
    events = list(manager.finalize())
    assert len(events) == 0  # No events, no errors


def test_finalize_with_empty_buffered_content():
    """Test that finalize handles empty string in buffer (covers 83->82 branch)."""
    manager = ModelResponsePartsManager()
    # Add both empty and non-empty content to test the branch where buffered_content is falsy
    # This ensures the loop continues after skipping the empty content
    manager._thinking_tag_buffer['id1'] = ''  # Will be skipped  # pyright: ignore[reportPrivateUsage]
    manager._thinking_tag_buffer['id2'] = 'content'  # Will be flushed  # pyright: ignore[reportPrivateUsage]
    events = list(manager.finalize())
    assert len(events) == 1  # Only non-empty content produces events
    assert isinstance(events[0], PartStartEvent)
    assert events[0].part == TextPart(content='content')
    assert manager._thinking_tag_buffer == {}  # Buffer should be cleared  # pyright: ignore[reportPrivateUsage]


def test_get_parts_after_finalize():
    """Test that get_parts returns flushed content after finalize (unit test)."""
    # NOTE: This is a unit test of the manager. Real integration testing with
    # StreamedResponse is done in test_finalize_integration().
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    list(manager.handle_text_delta(vendor_part_id='content', content='<thi', thinking_tags=thinking_tags))

    # Before finalize
    assert manager.get_parts() == []  # Buffer not included

    # Finalize
    list(manager.finalize())

    # After finalize
    assert manager.get_parts() == snapshot([TextPart(content='<thi', part_kind='text')])


def test_end_tag_with_trailing_text_same_chunk():
    """Test that text after end tag in same chunk is handled correctly."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Start thinking
    events = list(manager.handle_text_delta(vendor_part_id='content', content='<think>', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert isinstance(events[0], PartStartEvent)
    assert isinstance(events[0].part, ThinkingPart)

    # Add thinking content
    events = list(manager.handle_text_delta(vendor_part_id='content', content='reasoning', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert events[0] == snapshot(
        PartDeltaEvent(
            index=0,
            delta=ThinkingPartDelta(content_delta='reasoning', part_delta_kind='thinking'),
            event_kind='part_delta',
        )
    )

    # End tag with trailing text in same chunk
    events = list(
        manager.handle_text_delta(vendor_part_id='content', content='</think>post-text', thinking_tags=thinking_tags)
    )

    # Should emit event for new TextPart
    assert len(events) == 1
    assert isinstance(events[0], PartStartEvent)
    assert events[0].part == TextPart(content='post-text')

    # Final state
    assert manager.get_parts() == snapshot(
        [ThinkingPart(content='reasoning', part_kind='thinking'), TextPart(content='post-text', part_kind='text')]
    )


def test_split_end_tag_with_trailing_text():
    """Test split end tag with text after it."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Start thinking (tag at position 0)
    events = list(manager.handle_text_delta(vendor_part_id='content', content='<think>', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert isinstance(events[0], PartStartEvent)
    assert isinstance(events[0].part, ThinkingPart)

    # Add thinking content
    events = list(manager.handle_text_delta(vendor_part_id='content', content='thinking', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert isinstance(events[0], PartDeltaEvent)

    # Split end tag: "</thi"
    events = list(manager.handle_text_delta(vendor_part_id='content', content='</thi', thinking_tags=thinking_tags))
    assert len(events) == 0  # Buffered

    # Complete end tag with trailing text: "nk>post"
    events = list(manager.handle_text_delta(vendor_part_id='content', content='nk>post', thinking_tags=thinking_tags))

    # Should close thinking and start text part
    assert len(events) == 1
    assert isinstance(events[0], PartStartEvent)
    assert events[0].part == TextPart(content='post')

    assert manager.get_parts() == snapshot(
        [ThinkingPart(content='thinking', part_kind='thinking'), TextPart(content='post', part_kind='text')]
    )


def test_thinking_content_before_end_tag_with_trailing():
    """Test thinking content before end tag, with trailing text in same chunk."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Start thinking
    events = list(manager.handle_text_delta(vendor_part_id='content', content='<think>', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert isinstance(events[0], PartStartEvent)
    assert isinstance(events[0].part, ThinkingPart)

    # Send content + end tag + trailing all in one chunk
    events = list(
        manager.handle_text_delta(
            vendor_part_id='content', content='reasoning</think>after', thinking_tags=thinking_tags
        )
    )

    # Should emit thinking delta event, then text start event
    assert len(events) == 2
    assert isinstance(events[0], PartDeltaEvent)
    assert events[0].delta == ThinkingPartDelta(content_delta='reasoning')
    assert isinstance(events[1], PartStartEvent)
    assert events[1].part == TextPart(content='after')

    assert manager.get_parts() == snapshot(
        [ThinkingPart(content='reasoning', part_kind='thinking'), TextPart(content='after', part_kind='text')]
    )


# Issue 3b: START tags with trailing content
# These tests document the broken behavior where start tags with trailing content
# in the same chunk are not handled correctly.


def test_start_tag_with_trailing_content_same_chunk():
    """Test that content after start tag in same chunk is handled correctly."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Start tag with trailing content in same chunk
    events = list(
        manager.handle_text_delta(vendor_part_id='content', content='<think>thinking', thinking_tags=thinking_tags)
    )

    # Should emit event for new ThinkingPart, then delta for content
    assert len(events) >= 1
    assert isinstance(events[0], PartStartEvent)
    assert isinstance(events[0].part, ThinkingPart)

    # If content is included in the same event stream
    if len(events) == 2:
        assert isinstance(events[1], PartDeltaEvent)
        assert events[1].delta == ThinkingPartDelta(content_delta='thinking')

    # Final state
    assert manager.get_parts() == snapshot([ThinkingPart(content='thinking', part_kind='thinking')])


def test_split_start_tag_with_trailing_content():
    """Test split start tag with content after it."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Split start tag: "<thi"
    events = list(manager.handle_text_delta(vendor_part_id='content', content='<thi', thinking_tags=thinking_tags))
    assert len(events) == 0  # Buffered

    # Complete start tag with trailing content: "nk>content"
    events = list(
        manager.handle_text_delta(vendor_part_id='content', content='nk>content', thinking_tags=thinking_tags)
    )

    # Should create ThinkingPart and add content
    assert len(events) >= 1
    assert isinstance(events[0], PartStartEvent)
    assert isinstance(events[0].part, ThinkingPart)

    if len(events) == 2:
        assert isinstance(events[1], PartDeltaEvent)
        assert events[1].delta == ThinkingPartDelta(content_delta='content')

    assert manager.get_parts() == snapshot([ThinkingPart(content='content', part_kind='thinking')])


def test_complete_sequence_start_tag_with_inline_content():
    """Test complete sequence: start tag with inline content and end tag."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # All in one chunk: "<think>content</think>after"
    events = list(
        manager.handle_text_delta(
            vendor_part_id='content', content='<think>content</think>after', thinking_tags=thinking_tags
        )
    )

    # Should create ThinkingPart with content, then TextPart
    # Exact event count may vary based on implementation
    assert len(events) >= 2

    # Final state should have both parts
    assert manager.get_parts() == snapshot(
        [ThinkingPart(content='content', part_kind='thinking'), TextPart(content='after', part_kind='text')]
    )


def test_text_then_start_tag_with_content():
    """Test text part followed by start tag with content."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Chunk 1: "Hello "
    events = list(manager.handle_text_delta(vendor_part_id='content', content='Hello ', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert isinstance(events[0], PartStartEvent)
    assert events[0].part == TextPart(content='Hello ')

    # Chunk 2: "<think>reasoning"
    events = list(
        manager.handle_text_delta(vendor_part_id='content', content='<think>reasoning', thinking_tags=thinking_tags)
    )

    # Should create ThinkingPart and add reasoning content
    assert len(events) >= 1
    assert isinstance(events[0], PartStartEvent)
    assert isinstance(events[0].part, ThinkingPart)

    if len(events) == 2:
        assert isinstance(events[1], PartDeltaEvent)
        assert events[1].delta == ThinkingPartDelta(content_delta='reasoning')

    # Final state
    assert manager.get_parts() == snapshot(
        [TextPart(content='Hello ', part_kind='text'), ThinkingPart(content='reasoning', part_kind='thinking')]
    )


def test_text_and_start_tag_same_chunk():
    """Test text followed by start tag in the same chunk (covers line 297)."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Single chunk with text then start tag: "prefix<think>"
    events = list(
        manager.handle_text_delta(vendor_part_id='content', content='prefix<think>', thinking_tags=thinking_tags)
    )

    # Should create TextPart for "prefix", then ThinkingPart
    assert len(events) == 2
    assert isinstance(events[0], PartStartEvent)
    assert events[0].part == TextPart(content='prefix')
    assert isinstance(events[1], PartStartEvent)
    assert isinstance(events[1].part, ThinkingPart)

    # Final state
    assert manager.get_parts() == snapshot(
        [TextPart(content='prefix', part_kind='text'), ThinkingPart(content='', part_kind='thinking')]
    )


def test_text_and_start_tag_with_content_same_chunk():
    """Test text + start tag + content in the same chunk (covers lines 211, 223, 297)."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Single chunk: "prefix<think>thinking"
    events = list(
        manager.handle_text_delta(
            vendor_part_id='content', content='prefix<think>thinking', thinking_tags=thinking_tags
        )
    )

    # Should create TextPart, ThinkingPart, and add thinking content
    assert len(events) >= 2

    # Final state
    assert manager.get_parts() == snapshot(
        [TextPart(content='prefix', part_kind='text'), ThinkingPart(content='thinking', part_kind='thinking')]
    )


def test_start_tag_with_content_no_vendor_id():
    """Test start tag with trailing content when vendor_part_id=None.

    The content after the start tag should be added to the ThinkingPart, not create a separate TextPart.
    """
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # With vendor_part_id=None and start tag with content
    events = list(
        manager.handle_text_delta(vendor_part_id=None, content='<think>thinking', thinking_tags=thinking_tags)
    )

    # Should create ThinkingPart and add content
    assert len(events) >= 1
    assert isinstance(events[0], PartStartEvent)
    assert isinstance(events[0].part, ThinkingPart)

    # Content should be in the ThinkingPart, not a separate TextPart
    assert manager.get_parts() == snapshot([ThinkingPart(content='thinking')])


def test_text_then_start_tag_no_vendor_id():
    """Test text before start tag when vendor_part_id=None (covers line 211 in _handle_text_delta_simple)."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # With vendor_part_id=None and text before start tag
    events = list(manager.handle_text_delta(vendor_part_id=None, content='text<think>', thinking_tags=thinking_tags))

    # Should create TextPart for "text", then ThinkingPart
    assert len(events) == 2
    assert isinstance(events[0], PartStartEvent)
    assert events[0].part == TextPart(content='text')
    assert isinstance(events[1], PartStartEvent)
    assert isinstance(events[1].part, ThinkingPart)

    # Final state
    assert manager.get_parts() == snapshot([TextPart(content='text'), ThinkingPart(content='')])
