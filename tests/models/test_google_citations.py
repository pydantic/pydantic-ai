"""Tests for Google citations."""

from __future__ import annotations as _annotations

from typing import Any, cast

import pytest  # pyright: ignore[reportMissingImports]

from pydantic_ai import GroundingCitation, TextPart

from ..conftest import try_import

with try_import() as imports_successful:
    from google.genai.types import (
        Citation,
        CitationMetadata,
        GenerateContentResponse,
        GroundingChunk,
        GroundingChunkWeb,
        GroundingMetadata,
        GroundingSupport,
        Segment,
    )

    from pydantic_ai.models.google import (
        _parse_google_citation_metadata,
        _parse_google_grounding_metadata,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='Google GenAI SDK not installed')


# Unit tests for _parse_google_citation_metadata


def test_parse_citation_metadata_none():
    """Parsing when citation_metadata is None."""
    citations = _parse_google_citation_metadata(None)
    assert citations == []


def test_parse_citation_metadata_empty_citations():
    """Parsing when citations list is empty."""
    citation_metadata = CitationMetadata(citations=[])
    citations = _parse_google_citation_metadata(citation_metadata)
    assert citations == []


def test_parse_citation_metadata_single():
    """Test parsing a single citation."""
    citation = Citation(
        start_index=0,
        end_index=5,
        uri='https://example.com',
        title='Example Site',
        license=None,
        publication_date=None,
    )
    citation_metadata = CitationMetadata(citations=[citation])

    citations = _parse_google_citation_metadata(citation_metadata)
    assert len(citations) == 1
    assert isinstance(citations[0], GroundingCitation)
    assert citations[0].citation_metadata is not None
    assert 'citations' in citations[0].citation_metadata
    assert len(citations[0].citation_metadata['citations']) == 1
    cit_data = citations[0].citation_metadata['citations'][0]
    assert cit_data['start_index'] == 0
    assert cit_data['end_index'] == 5
    assert cit_data['uri'] == 'https://example.com'
    assert cit_data['title'] == 'Example Site'


def test_parse_citation_metadata_no_title():
    """Test parsing citation with empty title string."""
    citation = Citation(
        start_index=0,
        end_index=5,
        uri='https://example.com',
        title='',  # Empty string should be converted to None
        license=None,
        publication_date=None,
    )
    citation_metadata = CitationMetadata(citations=[citation])

    citations = _parse_google_citation_metadata(citation_metadata)
    assert len(citations) == 1
    cit_data = citations[0].citation_metadata['citations'][0]
    assert cit_data['title'] is None  # Empty string converted to None


def test_parse_citation_metadata_with_optional_fields():
    """Test parsing citation with license and publication_date."""
    # publication_date is optional and may be None or a complex type
    # For simplicity, test with just license
    citation = Citation(
        start_index=0,
        end_index=5,
        uri='https://example.com',
        title='Example Site',
        license='MIT',
        publication_date=None,  # Skip complex date type for now
    )
    citation_metadata = CitationMetadata(citations=[citation])

    citations = _parse_google_citation_metadata(citation_metadata)
    assert len(citations) == 1
    cit_data = citations[0].citation_metadata['citations'][0]
    assert cit_data['license'] == 'MIT'


def test_parse_citation_metadata_invalid_indices_negative():
    """Test parsing citation with negative indices."""
    citation = Citation(
        start_index=-1,  # Invalid
        end_index=5,
        uri='https://example.com',
        title='Example Site',
    )
    citation_metadata = CitationMetadata(citations=[citation])

    citations = _parse_google_citation_metadata(citation_metadata)
    assert citations == []  # Invalid indices should be skipped


def test_parse_citation_metadata_invalid_indices_start_gt_end():
    """Test parsing citation with start_index > end_index."""
    citation = Citation(
        start_index=10,  # Invalid: start > end
        end_index=5,
        uri='https://example.com',
        title='Example Site',
    )
    citation_metadata = CitationMetadata(citations=[citation])

    citations = _parse_google_citation_metadata(citation_metadata)
    assert citations == []  # Invalid indices should be skipped


def test_parse_citation_metadata_missing_uri():
    """Test parsing citation with missing or empty URI."""
    citation = Citation(
        start_index=0,
        end_index=5,
        uri='',  # Empty URI - invalid
        title='Example Site',
    )
    citation_metadata = CitationMetadata(citations=[citation])

    citations = _parse_google_citation_metadata(citation_metadata)
    assert citations == []  # Missing URI should be skipped


def test_parse_citation_metadata_multiple():
    """Test parsing multiple citations."""
    citation1 = Citation(
        start_index=0,
        end_index=5,
        uri='https://example.com',
        title='Example Site',
    )
    citation2 = Citation(
        start_index=10,
        end_index=15,
        uri='https://example.org',
        title='Another Site',
    )
    citation_metadata = CitationMetadata(citations=[citation1, citation2])

    citations = _parse_google_citation_metadata(citation_metadata)
    assert len(citations) == 2
    assert citations[0].citation_metadata['citations'][0]['uri'] == 'https://example.com'
    assert citations[1].citation_metadata['citations'][0]['uri'] == 'https://example.org'


# Unit tests for _parse_google_grounding_metadata


def test_parse_grounding_metadata_none():
    """Test parsing when grounding_metadata is None."""
    citations = _parse_google_grounding_metadata(None)
    assert citations == []


def test_parse_grounding_metadata_no_chunks():
    """Test parsing when grounding_chunks is None or empty."""
    grounding_metadata = GroundingMetadata(grounding_chunks=None, grounding_supports=None)
    citations = _parse_google_grounding_metadata(grounding_metadata)
    assert citations == []


def test_parse_grounding_metadata_no_supports():
    """Test parsing when grounding_supports is None or empty."""
    web_chunk = GroundingChunkWeb(
        uri='https://example.com',
        title='Example Site',
        domain='example.com',
    )
    chunk = GroundingChunk(web=web_chunk)
    grounding_metadata = GroundingMetadata(
        grounding_chunks=[chunk],
        grounding_supports=None,
    )
    citations = _parse_google_grounding_metadata(grounding_metadata)
    assert citations == []  # Need supports to create citations


def test_parse_grounding_metadata_single_web_chunk():
    """Test parsing a single web chunk with grounding support."""
    web_chunk = GroundingChunkWeb(
        uri='https://example.com',
        title='Example Site',
        domain='example.com',
    )
    chunk = GroundingChunk(web=web_chunk)

    # Create a segment
    segment = Segment(start_index=0, end_index=5, text='Hello')

    # Create grounding support linking segment to chunk
    support = GroundingSupport(
        grounding_chunk_indices=[0],  # Reference to first chunk
        segment=segment,
        confidence_scores=None,
    )

    grounding_metadata = GroundingMetadata(
        grounding_chunks=[chunk],
        grounding_supports=[support],
    )

    citations = _parse_google_grounding_metadata(grounding_metadata)
    assert len(citations) == 1
    assert isinstance(citations[0], GroundingCitation)
    assert citations[0].grounding_metadata is not None
    assert 'grounding_chunks' in citations[0].grounding_metadata
    assert len(citations[0].grounding_metadata['grounding_chunks']) == 1
    chunk_data = citations[0].grounding_metadata['grounding_chunks'][0]
    assert 'web' in chunk_data
    assert chunk_data['web']['uri'] == 'https://example.com'
    assert chunk_data['web']['title'] == 'Example Site'
    assert chunk_data['web']['domain'] == 'example.com'

    # Check segment data
    assert 'segment' in citations[0].grounding_metadata
    segment_data = citations[0].grounding_metadata['segment']
    assert segment_data['start_index'] == 0
    assert segment_data['end_index'] == 5
    assert segment_data['text'] == 'Hello'


def test_parse_grounding_metadata_invalid_segment_indices():
    """Test parsing with invalid segment indices."""
    web_chunk = GroundingChunkWeb(uri='https://example.com', title='Example')
    chunk = GroundingChunk(web=web_chunk)

    # Invalid segment: start > end
    segment = Segment(start_index=10, end_index=5, text='Hello')
    support = GroundingSupport(
        grounding_chunk_indices=[0],
        segment=segment,
    )

    grounding_metadata = GroundingMetadata(
        grounding_chunks=[chunk],
        grounding_supports=[support],
    )

    citations = _parse_google_grounding_metadata(grounding_metadata)
    assert citations == []  # Invalid indices should be skipped


def test_parse_grounding_metadata_invalid_chunk_index():
    """Test parsing with invalid chunk index in support."""
    web_chunk = GroundingChunkWeb(uri='https://example.com', title='Example')
    chunk = GroundingChunk(web=web_chunk)

    segment = Segment(start_index=0, end_index=5, text='Hello')
    # Invalid: chunk index 1 doesn't exist (only 0 exists)
    support = GroundingSupport(
        grounding_chunk_indices=[1],  # Invalid index
        segment=segment,
    )

    grounding_metadata = GroundingMetadata(
        grounding_chunks=[chunk],
        grounding_supports=[support],
    )

    citations = _parse_google_grounding_metadata(grounding_metadata)
    assert citations == []  # Invalid chunk index should result in no citations


def test_parse_grounding_metadata_multiple_chunks():
    """Test parsing with multiple chunks and supports."""
    web_chunk1 = GroundingChunkWeb(
        uri='https://example.com',
        title='Example Site',
        domain='example.com',
    )
    web_chunk2 = GroundingChunkWeb(
        uri='https://example.org',
        title='Another Site',
        domain='example.org',
    )
    chunk1 = GroundingChunk(web=web_chunk1)
    chunk2 = GroundingChunk(web=web_chunk2)

    segment1 = Segment(start_index=0, end_index=5, text='Hello')
    segment2 = Segment(start_index=10, end_index=15, text='world')

    support1 = GroundingSupport(
        grounding_chunk_indices=[0],
        segment=segment1,
    )
    support2 = GroundingSupport(
        grounding_chunk_indices=[1],
        segment=segment2,
    )

    grounding_metadata = GroundingMetadata(
        grounding_chunks=[chunk1, chunk2],
        grounding_supports=[support1, support2],
        web_search_queries=['test query'],
    )

    citations = _parse_google_grounding_metadata(grounding_metadata)
    assert len(citations) == 2
    assert citations[0].grounding_metadata['grounding_chunks'][0]['web']['uri'] == 'https://example.com'
    assert citations[1].grounding_metadata['grounding_chunks'][0]['web']['uri'] == 'https://example.org'
    # Check that web_search_queries are included
    assert citations[0].grounding_metadata.get('web_search_queries') == ['test query']


def test_parse_grounding_metadata_chunk_without_web():
    """Test parsing chunk that doesn't have web data."""
    # Chunk with no web/maps/retrieved_context
    chunk = GroundingChunk(web=None, maps=None, retrieved_context=None)

    segment = Segment(start_index=0, end_index=5, text='Hello')
    support = GroundingSupport(
        grounding_chunk_indices=[0],
        segment=segment,
    )

    grounding_metadata = GroundingMetadata(
        grounding_chunks=[chunk],
        grounding_supports=[support],
    )

    citations = _parse_google_grounding_metadata(grounding_metadata)
    assert citations == []  # Chunk with no data should be skipped


# Mock setup for integration tests


class MockGoogleClient:
    """Mock Google GenAI Client for testing."""

    def __init__(
        self,
        response: GenerateContentResponse | None = None,
        stream: list[GenerateContentResponse] | None = None,
    ):
        self.response = response
        self.stream = stream
        self.aio = type('AIO', (), {'models': self})()
        # Create a mock _api_client for provider compatibility
        self._api_client = type('APIClient', (), {'vertexai': False})()

    async def generate_content(self, *args: Any, **kwargs: Any) -> GenerateContentResponse:
        """Mock generate_content for non-streaming."""
        if self.response is None:
            raise ValueError('No response provided to mock')
        return self.response

    async def generate_content_stream(self, *args: Any, **kwargs: Any) -> Any:  # Returns async iterator
        """Mock generate_content_stream for streaming."""
        if self.stream is None:
            raise ValueError('No stream provided to mock')
        from .mock_async_stream import MockAsyncStream

        return MockAsyncStream(iter(self.stream))

    @classmethod
    def create_mock(cls, response: GenerateContentResponse) -> Any:
        """Create a mock client with a non-streaming response."""
        from google.genai import Client

        return cast(Client, cls(response=response))

    @classmethod
    def create_stream_mock(cls, stream: list[GenerateContentResponse]) -> Any:
        """Create a mock client with a streaming response."""
        from google.genai import Client

        return cast(Client, cls(stream=stream))


# Integration tests for non-streaming with citations


@pytest.mark.anyio
async def test_non_streaming_with_citation_metadata(allow_model_requests: None):
    """Test non-streaming response with citation_metadata."""
    from google.genai.types import (
        Candidate,
        Content,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
        Part,
    )

    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    # Create a mock response with citation_metadata
    citation = Citation(
        start_index=0,
        end_index=5,
        uri='https://example.com',
        title='Example Site',
    )
    citation_metadata = CitationMetadata(citations=[citation])

    text_part = Part(text='Hello world!')
    content = Content(parts=[text_part])
    candidate = Candidate(
        content=content,
        citation_metadata=citation_metadata,
        finish_reason='STOP',
    )

    response = GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
        ),
    )

    mock_client = MockGoogleClient.create_mock(response)
    model = GoogleModel('gemini-1.5-flash', provider=GoogleProvider(client=mock_client))

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    result = await model.request(messages, None, ModelRequestParameters())

    # Find TextPart with citations
    text_part_with_citations = None
    for part in result.parts:
        if isinstance(part, TextPart) and part.citations:
            text_part_with_citations = part
            break

    # If no part has citations, check all TextParts
    if text_part_with_citations is None:
        for part in result.parts:
            if isinstance(part, TextPart):
                text_part_with_citations = part
                break

    assert text_part_with_citations is not None
    assert text_part_with_citations.citations is not None
    assert len(text_part_with_citations.citations) == 1
    assert isinstance(text_part_with_citations.citations[0], GroundingCitation)
    cit_data = text_part_with_citations.citations[0].citation_metadata['citations'][0]
    assert cit_data['uri'] == 'https://example.com'
    assert cit_data['title'] == 'Example Site'


@pytest.mark.anyio
async def test_non_streaming_with_grounding_metadata(allow_model_requests: None):
    """Test non-streaming response with grounding_metadata."""
    from google.genai.types import (
        Candidate,
        Content,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
        Part,
    )

    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    # Create a mock response with grounding_metadata
    web_chunk = GroundingChunkWeb(
        uri='https://example.com',
        title='Example Site',
        domain='example.com',
    )
    chunk = GroundingChunk(web=web_chunk)
    segment = Segment(start_index=0, end_index=5, text='Hello')
    support = GroundingSupport(
        grounding_chunk_indices=[0],
        segment=segment,
    )

    grounding_metadata = GroundingMetadata(
        grounding_chunks=[chunk],
        grounding_supports=[support],
    )

    text_part = Part(text='Hello world!')
    content = Content(parts=[text_part])
    candidate = Candidate(
        content=content,
        grounding_metadata=grounding_metadata,
        finish_reason='STOP',
    )

    response = GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
        ),
    )

    mock_client = MockGoogleClient.create_mock(response)
    model = GoogleModel('gemini-1.5-flash', provider=GoogleProvider(client=mock_client))

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    result = await model.request(messages, None, ModelRequestParameters())

    # Find TextPart with citations
    text_part_with_citations = None
    for part in result.parts:
        if isinstance(part, TextPart) and part.citations:
            text_part_with_citations = part
            break

    # If no part has citations, check all TextParts
    if text_part_with_citations is None:
        for part in result.parts:
            if isinstance(part, TextPart):
                text_part_with_citations = part
                break

    assert text_part_with_citations is not None
    assert text_part_with_citations.citations is not None
    assert len(text_part_with_citations.citations) == 1
    assert isinstance(text_part_with_citations.citations[0], GroundingCitation)
    chunk_data = text_part_with_citations.citations[0].grounding_metadata['grounding_chunks'][0]
    assert chunk_data['web']['uri'] == 'https://example.com'
    assert chunk_data['web']['title'] == 'Example Site'


@pytest.mark.anyio
async def test_non_streaming_without_citations(allow_model_requests: None):
    """Test non-streaming response without citations."""
    from google.genai.types import (
        Candidate,
        Content,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
        Part,
    )

    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    text_part = Part(text='Hello world!')
    content = Content(parts=[text_part])
    candidate = Candidate(
        content=content,
        citation_metadata=None,
        grounding_metadata=None,
        finish_reason='STOP',
    )

    response = GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
        ),
    )

    mock_client = MockGoogleClient.create_mock(response)
    model = GoogleModel('gemini-1.5-flash', provider=GoogleProvider(client=mock_client))

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    result = await model.request(messages, None, ModelRequestParameters())

    text_part = None
    for part in result.parts:
        if isinstance(part, TextPart):
            text_part = part
            break

    assert text_part is not None
    assert text_part.citations is None or len(text_part.citations) == 0


@pytest.mark.anyio
async def test_non_streaming_with_both_metadata_types(allow_model_requests: None):
    """Test non-streaming response with both citation_metadata and grounding_metadata."""
    from google.genai.types import (
        Candidate,
        Content,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
        Part,
    )

    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    # Create citation_metadata
    citation = Citation(
        start_index=0,
        end_index=5,
        uri='https://example.com',
        title='Example Site',
    )
    citation_metadata = CitationMetadata(citations=[citation])

    # Create grounding_metadata
    web_chunk = GroundingChunkWeb(
        uri='https://example.org',
        title='Another Site',
        domain='example.org',
    )
    chunk = GroundingChunk(web=web_chunk)
    segment = Segment(start_index=10, end_index=15, text='world')
    support = GroundingSupport(
        grounding_chunk_indices=[0],
        segment=segment,
    )
    grounding_metadata = GroundingMetadata(
        grounding_chunks=[chunk],
        grounding_supports=[support],
    )

    text_part = Part(text='Hello world!')
    content = Content(parts=[text_part])
    candidate = Candidate(
        content=content,
        citation_metadata=citation_metadata,
        grounding_metadata=grounding_metadata,
        finish_reason='STOP',
    )

    response = GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
        ),
    )

    mock_client = MockGoogleClient.create_mock(response)
    model = GoogleModel('gemini-1.5-flash', provider=GoogleProvider(client=mock_client))

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    result = await model.request(messages, None, ModelRequestParameters())

    # Find TextPart with citations
    text_part_with_citations = None
    for part in result.parts:
        if isinstance(part, TextPart) and part.citations:
            text_part_with_citations = part
            break

    assert text_part_with_citations is not None
    assert text_part_with_citations.citations is not None
    # Should have citations from both metadata types
    assert len(text_part_with_citations.citations) >= 1


# Integration tests for streaming with citations


@pytest.mark.anyio
async def test_stream_with_citation_metadata(allow_model_requests: None):
    """Test streaming response with citation_metadata."""
    from google.genai.types import (
        Candidate,
        Content,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
        Part,
    )

    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    # Create streaming chunks
    citation = Citation(
        start_index=0,
        end_index=5,
        uri='https://example.com',
        title='Example Site',
    )
    citation_metadata = CitationMetadata(citations=[citation])

    # First chunk: text content
    text_part1 = Part(text='Hello')
    content1 = Content(parts=[text_part1])
    candidate1 = Candidate(
        content=content1,
        citation_metadata=None,
        finish_reason=None,
    )
    chunk1 = GenerateContentResponse(
        candidates=[candidate1],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=2,
        ),
    )

    # Second chunk: citation metadata arrives
    text_part2 = Part(text=' world!')
    content2 = Content(parts=[text_part2])
    candidate2 = Candidate(
        content=content2,
        citation_metadata=citation_metadata,
        finish_reason='STOP',
    )
    chunk2 = GenerateContentResponse(
        candidates=[candidate2],
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
        ),
    )

    stream_chunks = [chunk1, chunk2]
    mock_client = MockGoogleClient.create_stream_mock(stream_chunks)
    model = GoogleModel('gemini-1.5-flash', provider=GoogleProvider(client=mock_client))

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    async with model.request_stream(messages, None, ModelRequestParameters()) as streamed_response:
        # Consume all events so citations are processed
        async for _event in streamed_response:
            pass

    # Get the final response which should have citations attached
    final_response = streamed_response.get()

    # Find TextPart with citations
    text_part_with_citations = None
    for part in final_response.parts:
        if isinstance(part, TextPart) and part.citations:
            text_part_with_citations = part
            break

    # If no part has citations, check all TextParts
    if text_part_with_citations is None:
        for part in final_response.parts:
            if isinstance(part, TextPart):
                text_part_with_citations = part
                break

    assert text_part_with_citations is not None
    assert text_part_with_citations.citations is not None
    assert len(text_part_with_citations.citations) == 1
    assert isinstance(text_part_with_citations.citations[0], GroundingCitation)
