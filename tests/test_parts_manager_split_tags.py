"""Tests for split thinking tag handling in ModelResponsePartsManager."""

from __future__ import annotations as _annotations

from collections.abc import Hashable

from inline_snapshot import snapshot

from pydantic_ai import (
    PartStartEvent,
    TextPart,
    ThinkingPart,
)
from pydantic_ai._parts_manager import ModelResponsePart, ModelResponsePartsManager
from pydantic_ai.messages import ModelResponseStreamEvent


def stream_text_deltas(
    chunks: list[str],
    vendor_part_id: Hashable | None = 'content',
    thinking_tags: tuple[str, str] | None = ('<think>', '</think>'),
    ignore_leading_whitespace: bool = False,
    finalize: bool = True,
) -> tuple[list[ModelResponseStreamEvent], list[ModelResponsePart]]:
    """Helper to stream chunks through manager and return all events + final parts.

    Args:
        chunks: List of text chunks to stream
        vendor_part_id: Vendor ID for part tracking
        thinking_tags: Tuple of (start_tag, end_tag) for thinking detection
        ignore_leading_whitespace: Whether to ignore leading whitespace
        finalize: Whether to call finalize() at the end

    Returns:
        Tuple of (all events, final parts)
    """
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

    if finalize:
        for event in manager.finalize():
            all_events.append(event)

    return all_events, manager.get_parts()


def test_handle_text_deltas_with_split_think_tags_at_chunk_start():
    """Test split thinking tags when tags are split across chunks."""

    # Scenario 1: Split start tag - <thi + nk>content</think>
    events, parts = stream_text_deltas(['<thi', 'nk>', 'reasoning content', '</think>', 'after'])
    assert len(events) == 2
    assert events[0] == snapshot(
        PartStartEvent(
            index=0, part=ThinkingPart(content='reasoning content', part_kind='thinking'), event_kind='part_start'
        )
    )
    assert events[1] == snapshot(
        PartStartEvent(index=1, part=TextPart(content='after', part_kind='text'), event_kind='part_start')
    )
    assert len(parts) == 2
    assert parts[0] == snapshot(ThinkingPart(content='reasoning content', part_kind='thinking'))
    assert parts[1] == snapshot(TextPart(content='after', part_kind='text'))

    # Scenario 2: Split end tag - <think>content</thi + nk>
    events, parts = stream_text_deltas(['<think>', 'more content', '</thi', 'nk>', 'text after'])
    assert len(events) == 2
    assert events[0] == snapshot(
        PartStartEvent(
            index=0, part=ThinkingPart(content='more content', part_kind='thinking'), event_kind='part_start'
        )
    )
    assert events[1] == snapshot(
        PartStartEvent(index=1, part=TextPart(content='text after', part_kind='text'), event_kind='part_start')
    )
    assert len(parts) == 2
    assert parts[0] == snapshot(ThinkingPart(content='more content', part_kind='thinking'))
    assert parts[1] == snapshot(TextPart(content='text after', part_kind='text'))

    # Scenario 3: Both tags split - <thi + nk>foo</thi + nk>
    events, parts = stream_text_deltas(['<thi', 'nk>foo</thi', 'nk>'])
    assert events == snapshot([PartStartEvent(index=0, part=ThinkingPart(content='foo'))])
    assert parts == snapshot([ThinkingPart(content='foo')])


def test_exact_tag_length_boundary():
    """Test when buffered content exactly equals tag length."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Send content in one chunk that's exactly tag length
    events = list(manager.handle_text_delta(vendor_part_id='content', content='<think>', thinking_tags=thinking_tags))
    # An empty ThinkingPart is created but no event is yielded until content arrives
    assert len(events) == 0


def test_buffered_content_flushed_on_finalize():
    """Test that buffered content is flushed when finalize is called."""
    events, parts = stream_text_deltas(['<thi'])
    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(index=0, part=TextPart(content='<thi', part_kind='text'), event_kind='part_start')
    )
    assert parts == snapshot([TextPart(content='<thi', part_kind='text')])


def test_finalize_flushes_all_buffers():
    """Test that finalize flushes all vendor_part_id buffers."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    for _ in manager.handle_text_delta(vendor_part_id='id1', content='<th', thinking_tags=thinking_tags):
        pass
    for _ in manager.handle_text_delta(vendor_part_id='id2', content='<thi', thinking_tags=thinking_tags):
        pass

    final_events = list(manager.finalize())
    assert len(final_events) == 2

    parts = manager.get_parts()
    assert len(parts) == 2
    assert all(isinstance(p, TextPart) for p in parts)
    contents: set[str] = {p.content for p in parts if isinstance(p, TextPart)}
    assert contents == {'<th', '<thi'}


def test_prefixed_thinking_tags_are_text():
    """Test that thinking tags (incomplete or complete) with a prefix are treated as plain text."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    # Case 1: Incomplete tag with prefix
    events = list(manager.handle_text_delta(vendor_part_id='content', content='foo<th', thinking_tags=thinking_tags))
    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(index=0, part=TextPart(content='foo<th', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([TextPart(content='foo<th', part_kind='text')])

    # Reset manager for next case
    manager = ModelResponsePartsManager()

    # Case 2: Complete tag with prefix
    events = list(
        manager.handle_text_delta(vendor_part_id='content', content='bar<think>', thinking_tags=thinking_tags)
    )
    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(index=0, part=TextPart(content='bar<think>', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([TextPart(content='bar<think>', part_kind='text')])

    # Reset manager for next case
    manager = ModelResponsePartsManager()

    # Case 3: Complete tag with content and prefix
    events = list(
        manager.handle_text_delta(
            vendor_part_id='content', content='baz<think>thinking</think>', thinking_tags=thinking_tags
        )
    )
    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(
            index=0, part=TextPart(content='baz<think>thinking</think>', part_kind='text'), event_kind='part_start'
        )
    )
    assert manager.get_parts() == snapshot([TextPart(content='baz<think>thinking</think>', part_kind='text')])


def test_stream_and_finalize():
    """Simulates streaming with complete tags and content."""
    events, parts = stream_text_deltas(['<thi', 'nk>', 'content', '</think>', 'final text'], vendor_part_id='stream1')

    assert len(events) == 2
    assert isinstance(events[0], PartStartEvent)
    assert isinstance(events[0].part, ThinkingPart)
    assert events[0].part.content == 'content'

    assert len(parts) == 2
    assert isinstance(parts[1], TextPart)
    assert parts[1].content == 'final text'

    events_incomplete, parts_incomplete = stream_text_deltas(['<thi'], vendor_part_id='stream2')

    assert len(events_incomplete) == 1
    assert isinstance(events_incomplete[0], PartStartEvent)
    assert events_incomplete[0].part == TextPart(content='<thi')
    assert parts_incomplete == [TextPart(content='<thi')]


def test_whitespace_prefixed_thinking_tags():
    """Test thinking tags prefixed by whitespace when ignore_leading_whitespace=True."""
    events, parts = stream_text_deltas(['\n<think>', 'thinking content'], ignore_leading_whitespace=True)

    assert len(events) == 1
    assert events[0] == snapshot(
        PartStartEvent(
            index=0, part=ThinkingPart(content='thinking content', part_kind='thinking'), event_kind='part_start'
        )
    )
    assert parts == snapshot([ThinkingPart(content='thinking content', part_kind='thinking')])


def test_isolated_think_tag_with_finalize():
    """Test isolated <think> tag converted to TextPart on finalize."""
    events, parts = stream_text_deltas(['<think>'])

    assert len(events) == 1
    assert isinstance(events[0], PartStartEvent)
    assert events[0].part == snapshot(TextPart(content='<think>', part_kind='text'))
    assert parts == snapshot([TextPart(content='<think>', part_kind='text')])


def test_vendor_id_switch_during_thinking():
    """Test that switching vendor_part_id during thinking creates separate parts."""
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    events = list(manager.handle_text_delta(vendor_part_id='id1', content='<think>', thinking_tags=thinking_tags))
    assert len(events) == 0

    events = list(
        manager.handle_text_delta(vendor_part_id='id1', content='thinking content', thinking_tags=thinking_tags)
    )
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PartStartEvent)
    assert isinstance(event.part, ThinkingPart)
    assert event.part.content == 'thinking content'

    events = list(
        manager.handle_text_delta(vendor_part_id='id2', content='different part', thinking_tags=thinking_tags)
    )
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PartStartEvent)
    assert isinstance(event.part, TextPart)
    assert event.part.content == 'different part'

    parts = manager.get_parts()
    assert len(parts) == 2
    assert parts[0] == snapshot(ThinkingPart(content='thinking content', part_kind='thinking'))
    assert parts[1] == snapshot(TextPart(content='different part', part_kind='text'))


def test_thinking_interrupted_by_incomplete_end_tag_and_vendor_switch():
    """Test unclosed thinking tag followed by different vendor_part_id.

    When a vendor_part_id switches and leaves a ThinkingPart with buffered partial end tag,
    the buffered content is auto-closed by appending it to the ThinkingPart during finalize().
    """
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    for _ in manager.handle_text_delta(vendor_part_id='id1', content='<think>', thinking_tags=thinking_tags):
        pass
    for _ in manager.handle_text_delta(vendor_part_id='id1', content='thinking foo</th', thinking_tags=thinking_tags):
        pass

    events = list(manager.handle_text_delta(vendor_part_id='id2', content='new content', thinking_tags=thinking_tags))
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PartStartEvent)
    assert isinstance(event.part, TextPart)

    for _ in manager.finalize():
        pass

    parts = manager.get_parts()
    assert len(parts) == 2
    assert isinstance(parts[0], ThinkingPart)
    assert parts[0].content == 'thinking foo</th'
    assert isinstance(parts[1], TextPart)
    assert parts[1].content == 'new content'


def test_split_end_tag_with_content_before():
    """Test content before split end tag in buffered chunks (line 337)."""
    events, parts = stream_text_deltas(['<think>', 'reasoning content</th', 'ink>'])

    assert len(parts) == 1
    assert isinstance(parts[0], ThinkingPart)
    assert parts[0].content == 'reasoning content'

    # Verify events
    assert any(isinstance(e, PartStartEvent) and isinstance(e.part, ThinkingPart) for e in events)


def test_split_end_tag_with_content_after():
    """Test content after split end tag in buffered chunks (line 343)."""
    events, parts = stream_text_deltas(['<think>', 'reasoning', '</thi', 'nk>after text'])

    assert len(parts) == 2
    assert isinstance(parts[0], ThinkingPart)
    assert parts[0].content == 'reasoning'
    assert isinstance(parts[1], TextPart)
    assert parts[1].content == 'after text'

    # Verify events
    assert any(isinstance(e, PartStartEvent) and isinstance(e.part, ThinkingPart) for e in events)
    assert any(isinstance(e, PartStartEvent) and isinstance(e.part, TextPart) for e in events)


def test_split_end_tag_with_content_before_and_after():
    """Test content both before and after split end tag."""
    _, parts = stream_text_deltas(['<think>', 'reason</th', 'ink>after'])

    assert len(parts) == 2
    assert isinstance(parts[0], ThinkingPart)
    assert parts[0].content == 'reason'
    assert isinstance(parts[1], TextPart)
    assert parts[1].content == 'after'


def test_cross_path_end_tag_handling():
    """Test end tag handling when buffering fallback delegates to simple path (C2 → S5).

    This tests the scenario where buffering creates a ThinkingPart, then non-matching
    content triggers the C2 fallback to simple path, which then handles the end tag.
    """
    _, parts = stream_text_deltas(['<think>initial', 'x', 'more</think>after'])

    assert len(parts) == 2
    assert isinstance(parts[0], ThinkingPart)
    assert parts[0].content == 'initialxmore'
    assert isinstance(parts[1], TextPart)
    assert parts[1].content == 'after'


def test_cross_path_bare_end_tag():
    """Test bare end tag when buffering fallback delegates to simple path (C2 → S5).

    This tests the specific branch where content equals exactly the end tag.
    """
    _, parts = stream_text_deltas(['<think>done', 'x', '</think>'])

    assert len(parts) == 1
    assert isinstance(parts[0], ThinkingPart)
    assert parts[0].content == 'donex'
