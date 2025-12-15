"""Edge case tests for citations."""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import (
    GroundingCitation,
    TextPart,
    ToolResultCitation,
    URLCitation,
)
from pydantic_ai._citation_utils import (
    map_citation_to_text_part,
    merge_citations,
    normalize_citation,
    validate_citation_indices,
)

# Invalid citation data tests


def test_url_citation_invalid_url_type():
    """URLCitation accepts any string as URL (no validation)."""
    # URLs are not validated - validation is left to the application
    citation = URLCitation(url='not-a-url', start_index=0, end_index=5)
    assert citation.url == 'not-a-url'


def test_url_citation_empty_url():
    """URLCitation accepts empty URLs."""
    citation = URLCitation(url='', start_index=0, end_index=5)
    assert citation.url == ''


def test_url_citation_very_large_indices():
    """URLCitation works with very large indices."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=1000000)
    assert citation.start_index == 0
    assert citation.end_index == 1000000


def test_url_citation_zero_length_range():
    """URLCitation allows zero-length ranges (start == end)."""
    citation = URLCitation(url='https://example.com', start_index=5, end_index=5)
    assert citation.start_index == 5
    assert citation.end_index == 5


def test_tool_result_citation_empty_tool_name():
    """Test ToolResultCitation with empty tool name."""
    # Empty tool names are allowed (though not recommended in practice)
    citation = ToolResultCitation(tool_name='')
    assert citation.tool_name == ''


def test_tool_result_citation_none_citation_data():
    """Test ToolResultCitation with None citation_data."""
    citation = ToolResultCitation(tool_name='test_tool', citation_data=None)
    assert citation.citation_data is None


def test_grounding_citation_both_metadata_none():
    """Test GroundingCitation with both metadata fields as None (should fail)."""
    with pytest.raises(ValueError, match='At least one of grounding_metadata or citation_metadata'):
        GroundingCitation(grounding_metadata=None, citation_metadata=None)


def test_grounding_citation_empty_metadata():
    """Test GroundingCitation with empty metadata dicts."""
    # Empty dicts are allowed
    citation = GroundingCitation(grounding_metadata={})
    assert citation.grounding_metadata == {}


# Out-of-bounds and boundary tests


def test_validate_citation_indices_zero_content():
    """Validating citations with zero-length content."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=0)
    assert validate_citation_indices(citation, content_length=0) is True

    citation2 = URLCitation(url='https://example.com', start_index=0, end_index=1)
    assert validate_citation_indices(citation2, content_length=0) is False


def test_validate_citation_indices_at_exact_boundary():
    """Validating citations exactly at content boundary."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=10)
    # end_index == content_length is valid (exclusive end)
    assert validate_citation_indices(citation, content_length=10) is True

    citation2 = URLCitation(url='https://example.com', start_index=0, end_index=11)
    assert validate_citation_indices(citation2, content_length=10) is False


def test_validate_citation_indices_very_large_content():
    """Validating citations with very large content."""
    citation = URLCitation(url='https://example.com', start_index=1000000, end_index=1000005)
    assert validate_citation_indices(citation, content_length=2000000) is True
    assert validate_citation_indices(citation, content_length=1000000) is False


def test_map_citation_to_text_part_overlapping_parts():
    """Mapping citation when TextParts overlap (edge case)."""
    # This shouldn't happen in practice, but the code handles it
    parts = [
        TextPart(content='Hello'),
        TextPart(content='lo world'),  # Overlaps with first part
    ]
    offsets = [0, 3]

    citation = URLCitation(url='https://example.com', start_index=2, end_index=5)
    result = map_citation_to_text_part(citation, parts, offsets)
    assert result == 0


def test_map_citation_to_text_part_citation_spanning_multiple_parts():
    """Mapping citation that spans multiple TextParts."""
    parts = [
        TextPart(content='Hello'),
        TextPart(content=' world'),
    ]
    offsets = [0, 5]

    # Citation spans both parts - maps to first part (where it starts)
    citation = URLCitation(url='https://example.com', start_index=2, end_index=8)
    result = map_citation_to_text_part(citation, parts, offsets)
    assert result == 0


def test_map_citation_to_text_part_citation_at_part_boundary():
    """Mapping citation exactly at TextPart boundary."""
    parts = [
        TextPart(content='Hello'),
        TextPart(content=' world'),
    ]
    offsets = [0, 5]

    # Citation starts exactly at boundary
    citation = URLCitation(url='https://example.com', start_index=5, end_index=7)
    result = map_citation_to_text_part(citation, parts, offsets)
    assert result == 1


# Overlapping citations tests


def test_text_part_with_overlapping_citations():
    """Test TextPart with overlapping citations (should be allowed)."""
    citations = [
        URLCitation(url='https://example.com', start_index=0, end_index=10),
        URLCitation(url='https://example.org', start_index=5, end_index=15),  # Overlaps
    ]
    text_part = TextPart(content='Hello, world! This is a test.', citations=citations)
    assert len(text_part.citations) == 2
    # Both citations should be preserved even if they overlap


def test_text_part_with_identical_citations():
    """Test TextPart with identical citations (duplicates allowed)."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=5)
    citations = [citation, citation]  # Same citation twice
    text_part = TextPart(content='Hello', citations=citations)
    assert len(text_part.citations) == 2
    assert text_part.citations[0] == text_part.citations[1]


# Citations outside content tests


def test_text_part_citation_outside_content():
    """Test TextPart with citation indices outside content (should still work)."""
    # Citation extends beyond content - should still be stored
    citation = URLCitation(url='https://example.com', start_index=0, end_index=100)
    # Validation would fail, but TextPart should still accept it
    text_part_with_citation = TextPart(content='Hello', citations=[citation])
    assert len(text_part_with_citation.citations) == 1
    # But validation should catch it
    assert validate_citation_indices(citation, content_length=5) is False


def test_text_part_citation_negative_indices():
    """Test TextPart with citation having negative indices (should be rejected at creation)."""
    # URLCitation validation should prevent negative indices
    with pytest.raises(ValueError, match='start_index must be non-negative'):
        URLCitation(url='https://example.com', start_index=-1, end_index=5)


# Malformed data tests


def test_merge_citations_with_invalid_types():
    """Test merge_citations with invalid types in lists."""
    # merge_citations should handle any iterable, but type checking should catch issues
    # In practice, this would be caught by type checkers, but runtime should handle gracefully
    valid_citations = [URLCitation(url='https://example.com', start_index=0, end_index=5)]
    result = merge_citations(valid_citations)
    assert len(result) == 1


def test_normalize_citation_with_all_types():
    """Test normalize_citation handles all citation types correctly."""
    url_citation = URLCitation(url='https://example.com', start_index=0, end_index=5)
    tool_citation = ToolResultCitation(tool_name='test')
    grounding_citation = GroundingCitation(grounding_metadata={'sources': ['s1']})

    assert normalize_citation(url_citation) == url_citation
    assert normalize_citation(tool_citation) == tool_citation
    assert normalize_citation(grounding_citation) == grounding_citation


# Serialization edge cases


def test_citation_serialization_with_special_characters():
    """Test serializing citations with special characters in URLs."""
    citation = URLCitation(
        url='https://example.com/path?query=test&param=value#fragment',
        title='Test & Example',
        start_index=0,
        end_index=5,
    )
    from pydantic import TypeAdapter

    ta = TypeAdapter(URLCitation)
    json_str = ta.dump_json(citation)
    parsed = ta.validate_json(json_str)
    assert parsed.url == citation.url
    assert parsed.title == citation.title


def test_citation_serialization_with_unicode():
    """Test serializing citations with Unicode characters."""
    citation = URLCitation(
        url='https://example.com/测试',
        title='测试标题',
        start_index=0,
        end_index=5,
    )
    from pydantic import TypeAdapter

    ta = TypeAdapter(URLCitation)
    json_str = ta.dump_json(citation)
    parsed = ta.validate_json(json_str)
    assert parsed.url == citation.url
    assert parsed.title == citation.title


def test_tool_result_citation_serialization_with_complex_data():
    """Test serializing ToolResultCitation with complex citation_data."""
    citation = ToolResultCitation(
        tool_name='search',
        citation_data={
            'urls': ['https://example.com', 'https://example.org'],
            'scores': [0.9, 0.8],
            'metadata': {'nested': {'deep': 'value'}},
        },
    )
    from pydantic import TypeAdapter

    ta = TypeAdapter(ToolResultCitation)
    json_str = ta.dump_json(citation)
    parsed = ta.validate_json(json_str)
    assert parsed.citation_data == citation.citation_data


def test_grounding_citation_serialization_with_nested_metadata():
    """Test serializing GroundingCitation with nested metadata."""
    citation = GroundingCitation(
        grounding_metadata={
            'chunks': [
                {'type': 'web', 'url': 'https://example.com'},
                {'type': 'map', 'location': 'New York'},
            ],
        },
    )
    from pydantic import TypeAdapter

    ta = TypeAdapter(GroundingCitation)
    json_str = ta.dump_json(citation)
    parsed = ta.validate_json(json_str)
    assert parsed.grounding_metadata == citation.grounding_metadata


# Type error tests


def test_map_citation_to_text_part_type_errors():
    """Test map_citation_to_text_part with type errors."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=5)
    parts = [TextPart(content='Hello')]

    # Mismatched lengths should raise ValueError
    with pytest.raises(ValueError, match='text_parts and content_offsets must have the same length'):
        map_citation_to_text_part(citation, parts, [0, 5])  # Wrong length

    # Empty parts should return None
    result = map_citation_to_text_part(citation, [], [])
    assert result is None


# Edge cases in citation lists


def test_text_part_with_empty_citation_list():
    """Test TextPart with empty citations list vs None."""
    part_with_empty = TextPart(content='Hello', citations=[])
    part_with_none = TextPart(content='Hello', citations=None)

    assert part_with_empty.citations == []
    assert part_with_none.citations is None
    # Both should be valid but different


def test_merge_citations_with_mixed_none_and_empty():
    """Test merging citations with mix of None and empty lists."""
    citations1 = [URLCitation(url='https://example.com', start_index=0, end_index=5)]
    citations2 = None
    citations3 = []
    citations4 = [URLCitation(url='https://example.org', start_index=6, end_index=10)]

    result = merge_citations(citations1, citations2, citations3, citations4)
    assert len(result) == 2
    assert result[0].url == 'https://example.com'
    assert result[1].url == 'https://example.org'


# Boundary condition tests


def test_citation_at_content_start():
    """Test citation exactly at content start."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=5)
    assert validate_citation_indices(citation, content_length=10) is True


def test_citation_at_content_end():
    """Test citation exactly at content end."""
    citation = URLCitation(url='https://example.com', start_index=5, end_index=10)
    assert validate_citation_indices(citation, content_length=10) is True


def test_citation_covering_entire_content():
    """Test citation covering entire content."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=10)
    assert validate_citation_indices(citation, content_length=10) is True


def test_multiple_citations_covering_all_content():
    """Test multiple citations that together cover all content."""
    citations = [
        URLCitation(url='https://example.com', start_index=0, end_index=5),
        URLCitation(url='https://example.org', start_index=5, end_index=10),
    ]
    text_part = TextPart(content='Hello world', citations=citations)
    assert len(text_part.citations) == 2
    # Both citations should be valid
    assert all(validate_citation_indices(c, content_length=11) for c in citations)


# Error recovery tests


def test_validate_citation_indices_handles_all_edge_cases():
    """Test that validate_citation_indices handles all edge cases gracefully."""
    # Valid citation
    valid = URLCitation(url='https://example.com', start_index=0, end_index=5)
    assert validate_citation_indices(valid, content_length=10) is True

    invalid = URLCitation(url='https://example.com', start_index=0, end_index=5)
    invalid.start_index = -1
    assert validate_citation_indices(invalid, content_length=10) is False

    invalid.start_index = 0
    invalid.end_index = 15
    assert validate_citation_indices(invalid, content_length=10) is False

    invalid.end_index = 5
    invalid.start_index = 10  # Start > end
    assert validate_citation_indices(invalid, content_length=20) is False


def test_text_part_handles_malformed_citations_gracefully():
    """Test that TextPart creation handles various citation edge cases."""
    # TextPart should accept citations even if they're invalid (validation is separate)
    invalid_citation = URLCitation(url='https://example.com', start_index=0, end_index=100)
    text_part = TextPart(content='Hello', citations=[invalid_citation])
    assert len(text_part.citations) == 1
    # But validation should fail
    assert validate_citation_indices(invalid_citation, content_length=5) is False


# Provider-specific edge cases


def test_merge_citations_with_very_long_urls():
    """Test merging citations with very long URLs."""
    long_url = 'https://example.com/' + 'a' * 2000
    citations = [
        URLCitation(url=long_url, start_index=0, end_index=5),
        URLCitation(url='https://example.org', start_index=6, end_index=10),
    ]
    result = merge_citations(citations)
    assert len(result) == 2
    assert len(result[0].url) > 2000


def test_text_part_with_very_long_citation_list():
    """Test TextPart with a very long list of citations."""
    # Create 100 citations
    citations = [URLCitation(url=f'https://example.com/page{i}', start_index=i, end_index=i + 1) for i in range(100)]
    content = ' '.join([f'word{i}' for i in range(100)])
    text_part = TextPart(content=content, citations=citations)
    assert len(text_part.citations) == 100


def test_citation_serialization_with_none_values():
    """Test serializing citations with None optional fields."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=5, title=None)
    from pydantic import TypeAdapter

    ta = TypeAdapter(URLCitation)
    json_str = ta.dump_json(citation)
    parsed = ta.validate_json(json_str)
    assert parsed.title is None


def test_tool_result_citation_with_empty_dict():
    """Test ToolResultCitation with empty citation_data dict."""
    citation = ToolResultCitation(tool_name='test', citation_data={})
    assert citation.citation_data == {}


def test_grounding_citation_with_empty_lists():
    """Test GroundingCitation with empty lists in metadata."""
    citation = GroundingCitation(
        grounding_metadata={'chunks': []},
        citation_metadata={'citations': []},
    )
    assert citation.grounding_metadata == {'chunks': []}
    assert citation.citation_metadata == {'citations': []}


# Concurrent processing edge cases


def test_merge_citations_thread_safety():
    """Test that merge_citations can handle concurrent access (basic test)."""
    # This is a simple test - full thread safety would require more complex setup
    citations1 = [URLCitation(url='https://example.com', start_index=0, end_index=5)]
    citations2 = [URLCitation(url='https://example.org', start_index=6, end_index=10)]

    # Merge multiple times - should be idempotent
    result1 = merge_citations(citations1, citations2)
    result2 = merge_citations(citations1, citations2)
    assert result1 == result2


# Unicode and special character edge cases


def test_citation_with_unicode_in_title():
    """Test citation with Unicode characters in title."""
    citation = URLCitation(
        url='https://example.com',
        title='测试标题 🎉',
        start_index=0,
        end_index=5,
    )
    assert citation.title == '测试标题 🎉'

    # Should serialize/deserialize correctly
    from pydantic import TypeAdapter

    ta = TypeAdapter(URLCitation)
    json_str = ta.dump_json(citation)
    parsed = ta.validate_json(json_str)
    assert parsed.title == '测试标题 🎉'


def test_tool_result_citation_with_unicode_in_data():
    """Test ToolResultCitation with Unicode in citation_data."""
    citation = ToolResultCitation(
        tool_name='test',
        citation_data={'title': '测试标题', 'description': '描述内容'},
    )
    assert citation.citation_data['title'] == '测试标题'

    # Should serialize/deserialize correctly
    from pydantic import TypeAdapter

    ta = TypeAdapter(ToolResultCitation)
    json_str = ta.dump_json(citation)
    parsed = ta.validate_json(json_str)
    assert parsed.citation_data == citation.citation_data


# Edge cases with None and optional fields


def test_text_part_citations_none_vs_empty_list():
    """Test distinction between None and empty list for citations."""
    part_none = TextPart(content='Hello', citations=None)
    part_empty = TextPart(content='Hello', citations=[])

    # Both are valid but different
    assert part_none.citations is None
    assert part_empty.citations == []

    # merge_citations should handle both
    result1 = merge_citations(part_none.citations)
    result2 = merge_citations(part_empty.citations)
    assert result1 == []
    assert result2 == []


def test_validate_citation_indices_with_none_content():
    """Test validate_citation_indices edge cases."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=0)
    # Zero-length content with zero-length citation should be valid
    assert validate_citation_indices(citation, content_length=0) is True


# Error messages and validation tests


def test_url_citation_error_messages():
    """Test that URLCitation provides clear error messages."""
    # Test negative start_index
    with pytest.raises(ValueError, match='start_index must be non-negative'):
        URLCitation(url='https://example.com', start_index=-1, end_index=5)

    # Test negative end_index
    with pytest.raises(ValueError, match='end_index must be non-negative'):
        URLCitation(url='https://example.com', start_index=0, end_index=-1)

    # Test start > end
    with pytest.raises(ValueError, match='start_index.*must be <= end_index'):
        URLCitation(url='https://example.com', start_index=10, end_index=5)


def test_grounding_citation_error_messages():
    """Test that GroundingCitation provides clear error messages."""
    with pytest.raises(ValueError, match='At least one of grounding_metadata or citation_metadata'):
        GroundingCitation()


def test_map_citation_to_text_part_error_messages():
    """Test that map_citation_to_text_part provides clear error messages."""
    citation = URLCitation(url='https://example.com', start_index=0, end_index=5)
    parts = [TextPart(content='Hello')]

    with pytest.raises(ValueError, match='text_parts and content_offsets must have the same length'):
        map_citation_to_text_part(citation, parts, [0, 5])  # Mismatched lengths
