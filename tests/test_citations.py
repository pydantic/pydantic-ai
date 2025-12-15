"""Tests for citation models."""

import pytest
from inline_snapshot import snapshot
from pydantic import TypeAdapter

from pydantic_ai import Citation, GroundingCitation, TextPart, ToolResultCitation, URLCitation
from pydantic_ai._citation_utils import (
    map_citation_to_text_part,
    merge_citations,
    normalize_citation,
    validate_citation_indices,
)


def test_url_citation_basic():
    """Test creating a basic URL citation."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=10)
    assert citation.url == 'https://example.com'
    assert citation.title is None
    assert citation.start_index == 0
    assert citation.end_index == 10


def test_url_citation_with_title():
    """Test creating a URL citation with title."""
    citation = URLCitation(url='https://example.com', title='Example Site', start_index=5, end_index=15)
    assert citation.url == 'https://example.com'
    assert citation.title == 'Example Site'
    assert citation.start_index == 5
    assert citation.end_index == 15


def test_url_citation_validation_start_negative():
    """Test that negative start_index raises ValueError."""
    with pytest.raises(ValueError, match='start_index must be non-negative'):
        URLCitation(url='https://example.com', start_index=-1, end_index=10)


def test_url_citation_validation_end_negative():
    """Test that negative end_index raises ValueError."""
    with pytest.raises(ValueError, match='end_index must be non-negative'):
        URLCitation(url='https://example.com', start_index=0, end_index=-1)


def test_url_citation_validation_start_gt_end():
    """Test that start_index > end_index raises ValueError."""
    with pytest.raises(ValueError, match='start_index.*must be <= end_index'):
        URLCitation(url='https://example.com', start_index=10, end_index=5)


def test_url_citation_validation_start_eq_end():
    """Test that start_index == end_index is valid (empty range)."""
    citation = URLCitation(url='https://example.com', start_index=5, end_index=5)
    assert citation.start_index == 5
    assert citation.end_index == 5


def test_url_citation_repr():
    """Test URL citation representation."""
    citation = URLCitation(url='https://example.com', title='Example', start_index=0, end_index=10)
    assert repr(citation) == snapshot(
        "URLCitation(url='https://example.com', title='Example', start_index=0, end_index=10)"
    )


def test_tool_result_citation_basic():
    """Test creating a basic tool result citation."""
    citation = ToolResultCitation(tool_name='search_tool')
    assert citation.tool_name == 'search_tool'
    assert citation.tool_call_id is None
    assert citation.citation_data is None


def test_tool_result_citation_with_all_fields():
    """Test creating a tool result citation with all fields."""
    citation = ToolResultCitation(
        tool_name='search_tool',
        tool_call_id='call_123',
        citation_data={'source': 'example.com', 'confidence': 0.9},
    )
    assert citation.tool_name == 'search_tool'
    assert citation.tool_call_id == 'call_123'
    assert citation.citation_data == {'source': 'example.com', 'confidence': 0.9}


def test_tool_result_citation_repr():
    """Test tool result citation representation."""
    citation = ToolResultCitation(tool_name='search_tool', tool_call_id='call_123')
    assert repr(citation) == snapshot("ToolResultCitation(tool_name='search_tool', tool_call_id='call_123')")


def test_grounding_citation_with_grounding_metadata():
    """Test creating a grounding citation with grounding metadata."""
    citation = GroundingCitation(grounding_metadata={'sources': ['source1', 'source2']})
    assert citation.grounding_metadata == {'sources': ['source1', 'source2']}
    assert citation.citation_metadata is None


def test_grounding_citation_with_citation_metadata():
    """Test creating a grounding citation with citation metadata."""
    citation = GroundingCitation(citation_metadata={'citations': [{'url': 'https://example.com'}]})
    assert citation.grounding_metadata is None
    assert citation.citation_metadata == {'citations': [{'url': 'https://example.com'}]}


def test_grounding_citation_with_both():
    """Test creating a grounding citation with both metadata types."""
    citation = GroundingCitation(
        grounding_metadata={'sources': ['source1']},
        citation_metadata={'citations': [{'url': 'https://example.com'}]},
    )
    assert citation.grounding_metadata == {'sources': ['source1']}
    assert citation.citation_metadata == {'citations': [{'url': 'https://example.com'}]}


def test_grounding_citation_validation_no_metadata():
    """Test that grounding citation requires at least one metadata field."""
    with pytest.raises(ValueError, match='At least one of grounding_metadata or citation_metadata'):
        GroundingCitation()


def test_grounding_citation_repr():
    """Test grounding citation representation."""
    citation = GroundingCitation(grounding_metadata={'sources': ['source1']})
    assert repr(citation) == snapshot("GroundingCitation(grounding_metadata={'sources': ['source1']})")


def test_citation_union_type_url():
    """Test that Citation union type accepts URLCitation."""
    citation: Citation = URLCitation(url='https://example.com', start_index=0, end_index=10)
    assert isinstance(citation, URLCitation)


def test_citation_union_type_tool_result():
    """Test that Citation union type accepts ToolResultCitation."""
    citation: Citation = ToolResultCitation(tool_name='search_tool')
    assert isinstance(citation, ToolResultCitation)


def test_citation_union_type_grounding():
    """Test that Citation union type accepts GroundingCitation."""
    citation: Citation = GroundingCitation(grounding_metadata={'sources': ['source1']})
    assert isinstance(citation, GroundingCitation)


def test_citation_serialization_url():
    """Test serializing URL citation to JSON."""
    citation = URLCitation(url='https://example.com', title='Example', start_index=0, end_index=10)
    ta = TypeAdapter(URLCitation)
    json_str = ta.dump_json(citation)
    # Parse back to verify it's valid JSON
    parsed = ta.validate_json(json_str)
    assert parsed.url == citation.url
    assert parsed.title == citation.title
    assert parsed.start_index == citation.start_index
    assert parsed.end_index == citation.end_index


def test_citation_serialization_tool_result():
    """Test serializing tool result citation to JSON."""
    citation = ToolResultCitation(
        tool_name='search_tool',
        tool_call_id='call_123',
        citation_data={'source': 'example.com'},
    )
    ta = TypeAdapter(ToolResultCitation)
    json_str = ta.dump_json(citation)
    parsed = ta.validate_json(json_str)
    assert parsed.tool_name == citation.tool_name
    assert parsed.tool_call_id == citation.tool_call_id
    assert parsed.citation_data == citation.citation_data


def test_citation_serialization_grounding():
    """Test serializing grounding citation to JSON."""
    citation = GroundingCitation(
        grounding_metadata={'sources': ['source1']},
        citation_metadata={'citations': [{'url': 'https://example.com'}]},
    )
    ta = TypeAdapter(GroundingCitation)
    json_str = ta.dump_json(citation)
    parsed = ta.validate_json(json_str)
    assert parsed.grounding_metadata == citation.grounding_metadata
    assert parsed.citation_metadata == citation.citation_metadata


def test_citation_union_serialization():
    """Test serializing Citation union type."""
    citations: list[Citation] = [
        URLCitation(url='https://example.com', start_index=0, end_index=10),
        ToolResultCitation(tool_name='search_tool'),
        GroundingCitation(grounding_metadata={'sources': ['source1']}),
    ]
    ta = TypeAdapter(list[Citation])
    json_str = ta.dump_json(citations)
    parsed = ta.validate_json(json_str)
    assert len(parsed) == 3
    assert isinstance(parsed[0], URLCitation)
    assert isinstance(parsed[1], ToolResultCitation)
    assert isinstance(parsed[2], GroundingCitation)


# --- Citation Utility Functions Tests ---


def test_merge_citations_empty():
    """Test merging empty citation lists."""
    result = merge_citations()
    assert result == []


def test_merge_citations_none():
    """Test merging None citation lists."""
    result = merge_citations(None, None)
    assert result == []


def test_merge_citations_single_list():
    """Test merging a single citation list."""
    citations = [URLCitation(url='https://example.com', start_index=0, end_index=5)]
    result = merge_citations(citations)
    assert len(result) == 1
    assert result[0] == citations[0]


def test_merge_citations_multiple_lists():
    """Test merging multiple citation lists."""
    citations1 = [URLCitation(url='https://example.com', start_index=0, end_index=5)]
    citations2 = [URLCitation(url='https://example.org', start_index=6, end_index=10)]
    citations3 = [ToolResultCitation(tool_name='search_tool')]
    result = merge_citations(citations1, citations2, citations3)
    assert len(result) == 3
    assert result[0] == citations1[0]
    assert result[1] == citations2[0]
    assert result[2] == citations3[0]


def test_merge_citations_with_none():
    """Test merging citation lists with None values."""
    citations1 = [URLCitation(url='https://example.com', start_index=0, end_index=5)]
    citations2 = None
    citations3 = [URLCitation(url='https://example.org', start_index=6, end_index=10)]
    result = merge_citations(citations1, citations2, citations3)
    assert len(result) == 2
    assert result[0] == citations1[0]
    assert result[1] == citations3[0]


def test_merge_citations_empty_lists():
    """Test merging empty citation lists."""
    result = merge_citations([], [], [])
    assert result == []


def test_validate_citation_indices_valid():
    """Test validating valid citation indices."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=5)
    assert validate_citation_indices(citation, content_length=10) is True


def test_validate_citation_indices_at_boundary():
    """Test validating citation indices at content boundary."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=10)
    assert validate_citation_indices(citation, content_length=10) is True


def test_validate_citation_indices_out_of_bounds():
    """Test validating citation indices that are out of bounds."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=15)
    assert validate_citation_indices(citation, content_length=10) is False


def test_validate_citation_indices_negative():
    """Validating citation indices with negative values."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=5)
    # Test validation function directly by modifying the citation
    citation.start_index = -1
    assert validate_citation_indices(citation, content_length=10) is False


def test_validate_citation_indices_start_gt_end():
    """Validating citation indices where start > end."""
    citation = URLCitation(url='https://example.com', start_index=3, end_index=5)
    # Test validation function directly by modifying the citation
    citation.start_index = 5
    citation.end_index = 3
    assert validate_citation_indices(citation, content_length=10) is False


def test_map_citation_to_text_part_single_part():
    """Test mapping citation to a single TextPart."""
    parts = [TextPart(content='Hello, world!')]
    offsets = [0]
    citation = URLCitation(url='https://example.com', start_index=0, end_index=5)
    result = map_citation_to_text_part(citation, parts, offsets)
    assert result == 0


def test_map_citation_to_text_part_first_part():
    """Test mapping citation to the first TextPart in multiple parts."""
    parts = [
        TextPart(content='Hello'),
        TextPart(content=' world'),
        TextPart(content='!'),
    ]
    offsets = [0, 5, 11]
    citation = URLCitation(url='https://example.com', start_index=2, end_index=4)
    result = map_citation_to_text_part(citation, parts, offsets)
    assert result == 0


def test_map_citation_to_text_part_second_part():
    """Test mapping citation to the second TextPart."""
    parts = [
        TextPart(content='Hello'),
        TextPart(content=' world'),
    ]
    offsets = [0, 5]
    citation = URLCitation(url='https://example.com', start_index=6, end_index=8)
    result = map_citation_to_text_part(citation, parts, offsets)
    assert result == 1


def test_map_citation_to_text_part_last_part_boundary():
    """Test mapping citation at the boundary of the last part."""
    parts = [
        TextPart(content='Hello'),
        TextPart(content=' world'),
    ]
    offsets = [0, 5]
    citation = URLCitation(url='https://example.com', start_index=11, end_index=11)
    result = map_citation_to_text_part(citation, parts, offsets)
    assert result == 1


def test_map_citation_to_text_part_out_of_bounds():
    """Test mapping citation that is out of bounds."""
    parts = [TextPart(content='Hello')]
    offsets = [0]
    citation = URLCitation(url='https://example.com', start_index=20, end_index=25)
    result = map_citation_to_text_part(citation, parts, offsets)
    assert result is None


def test_map_citation_to_text_part_empty_parts():
    """Test mapping citation with empty parts list."""
    parts: list[TextPart] = []
    offsets: list[int] = []
    citation = URLCitation(url='https://example.com', start_index=0, end_index=5)
    result = map_citation_to_text_part(citation, parts, offsets)
    assert result is None


def test_map_citation_to_text_part_mismatched_lengths():
    """Test mapping citation with mismatched parts and offsets lengths."""
    parts = [TextPart(content='Hello')]
    offsets = [0, 5]  # Mismatched length
    citation = URLCitation(url='https://example.com', start_index=0, end_index=5)
    with pytest.raises(ValueError, match='text_parts and content_offsets must have the same length'):
        map_citation_to_text_part(citation, parts, offsets)


def test_normalize_citation_url():
    """Test normalizing a URL citation."""
    citation = URLCitation(url='https://example.com', title='Example', start_index=0, end_index=5)
    normalized = normalize_citation(citation)
    assert normalized == citation
    assert isinstance(normalized, URLCitation)


def test_normalize_citation_tool_result():
    """Test normalizing a tool result citation."""
    citation = ToolResultCitation(tool_name='search_tool', tool_call_id='call_123')
    normalized = normalize_citation(citation)
    assert normalized == citation
    assert isinstance(normalized, ToolResultCitation)


def test_normalize_citation_grounding():
    """Test normalizing a grounding citation."""
    citation = GroundingCitation(grounding_metadata={'sources': ['source1']})
    normalized = normalize_citation(citation)
    assert normalized == citation
    assert isinstance(normalized, GroundingCitation)


# --- TextPart with Citations Tests ---


def test_text_part_without_citations():
    """Test TextPart can be created without citations (backward compatible)."""
    text_part = TextPart(content='Hello, world!')
    assert text_part.content == 'Hello, world!'
    assert text_part.citations is None
    assert text_part.id is None


def test_text_part_with_empty_citations():
    """Test TextPart can be created with empty citations list."""
    text_part = TextPart(content='Hello, world!', citations=[])
    assert text_part.content == 'Hello, world!'
    assert text_part.citations == []


def test_text_part_with_single_citation():
    """Test TextPart can be created with a single citation."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=5)
    text_part = TextPart(content='Hello, world!', citations=[citation])
    assert text_part.content == 'Hello, world!'
    assert len(text_part.citations) == 1
    assert text_part.citations[0] == citation
    assert isinstance(text_part.citations[0], URLCitation)


def test_text_part_with_multiple_citations():
    """Test TextPart can be created with multiple citations."""
    citations = [
        URLCitation(url='https://example.com', start_index=0, end_index=5),
        URLCitation(url='https://example.org', title='Example', start_index=6, end_index=11),
        ToolResultCitation(tool_name='search_tool'),
    ]
    text_part = TextPart(content='Hello, world!', citations=citations)
    assert text_part.content == 'Hello, world!'
    assert len(text_part.citations) == 3
    assert text_part.citations == citations


def test_text_part_citations_repr():
    """Test that citations are included in TextPart repr."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=5)
    text_part = TextPart(content='Hello', citations=[citation])
    repr_str = repr(text_part)
    assert 'Hello' in repr_str
    assert 'citations' in repr_str
    assert 'https://example.com' in repr_str


def test_text_part_citations_repr_none():
    """Test that TextPart repr works when citations is None."""
    text_part = TextPart(content='Hello')
    repr_str = repr(text_part)
    assert 'Hello' in repr_str
    # Citations should not appear in repr when None (following dataclass pattern)


def test_text_part_serialization_with_citations():
    """Test that TextPart with citations can be serialized to JSON."""
    citation = URLCitation(url='https://example.com', title='Example', start_index=0, end_index=5)
    text_part = TextPart(content='Hello', citations=[citation])
    ta = TypeAdapter(TextPart)
    json_str = ta.dump_json(text_part)
    parsed = ta.validate_json(json_str)
    assert parsed.content == text_part.content
    assert len(parsed.citations) == 1
    assert isinstance(parsed.citations[0], URLCitation)
    assert parsed.citations[0].url == citation.url


def test_text_part_serialization_without_citations():
    """Test that TextPart without citations can be serialized (backward compatible)."""
    text_part = TextPart(content='Hello')
    ta = TypeAdapter(TextPart)
    json_str = ta.dump_json(text_part)
    parsed = ta.validate_json(json_str)
    assert parsed.content == text_part.content
    assert parsed.citations is None


def test_text_part_backward_compatibility():
    """Test that existing code using TextPart without citations still works."""
    # This simulates existing code that creates TextPart
    text_part = TextPart(content='Existing content')
    assert text_part.content == 'Existing content'
    assert text_part.citations is None
    assert text_part.id is None
    assert text_part.part_kind == 'text'
    assert text_part.has_content() is True


def test_text_part_with_mixed_citation_types():
    """Test TextPart with different citation types."""
    citations: list[Citation] = [
        URLCitation(url='https://example.com', start_index=0, end_index=5),
        ToolResultCitation(tool_name='search_tool', tool_call_id='call_123'),
        GroundingCitation(grounding_metadata={'sources': ['source1']}),
    ]
    text_part = TextPart(content='Mixed citations', citations=citations)
    assert len(text_part.citations) == 3
    assert isinstance(text_part.citations[0], URLCitation)
    assert isinstance(text_part.citations[1], ToolResultCitation)
    assert isinstance(text_part.citations[2], GroundingCitation)
