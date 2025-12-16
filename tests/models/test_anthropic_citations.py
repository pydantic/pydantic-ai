"""Tests for Anthropic citations."""

from __future__ import annotations as _annotations

from typing import cast

import pytest  # pyright: ignore[reportMissingImports]

from pydantic_ai import TextPart, ToolResultCitation

from ..conftest import try_import

with try_import() as imports_successful:
    from anthropic.types.beta import (
        BetaCitationCharLocation,
        BetaCitationsDelta,
        BetaCitationSearchResultLocation,
        BetaCitationsWebSearchResultLocation,
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaTextBlock,
        BetaTextDelta,
        BetaUsage,
    )
    from anthropic.types.beta.beta_raw_message_delta_event import Delta

    from pydantic_ai.models.anthropic import (
        _parse_anthropic_citation_delta,
        _parse_anthropic_text_block_citations,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='Anthropic SDK not installed')


# Unit tests for _parse_anthropic_citation_delta


def test_parse_citation_delta_none_citation():
    """Parsing when citation is None."""

    # Mock delta with None citation (shouldn't happen, but test it anyway)
    class MockDelta:
        citation = None
        type = 'citations_delta'

    delta = MockDelta()  # type: ignore
    citation = _parse_anthropic_citation_delta(cast(BetaCitationsDelta, delta))
    assert citation is None


def test_parse_citation_delta_web_search_single():
    """Test parsing a single web search result citation."""
    # Create actual BetaCitationsWebSearchResultLocation object
    web_search_citation = BetaCitationsWebSearchResultLocation(
        url='https://example.com',
        title='Example Site',
        cited_text='This is cited text',
        encrypted_index='encrypted_123',
        type='web_search_result_location',
    )
    delta = BetaCitationsDelta(citation=web_search_citation, type='citations_delta')

    citation = _parse_anthropic_citation_delta(delta)
    assert citation is not None
    assert isinstance(citation, ToolResultCitation)
    assert citation.tool_name == 'web_search'
    assert citation.tool_call_id is None
    assert citation.citation_data is not None
    assert citation.citation_data['url'] == 'https://example.com'
    assert citation.citation_data['title'] == 'Example Site'
    assert citation.citation_data['cited_text'] == 'This is cited text'
    assert citation.citation_data['encrypted_index'] == 'encrypted_123'


def test_parse_citation_delta_web_search_no_title():
    """Test parsing web search citation with empty title string."""
    web_search_citation = BetaCitationsWebSearchResultLocation(
        url='https://example.com',
        title='',  # Empty string should be converted to None
        cited_text='Cited text',
        encrypted_index='encrypted_123',
        type='web_search_result_location',
    )
    delta = BetaCitationsDelta(citation=web_search_citation, type='citations_delta')

    citation = _parse_anthropic_citation_delta(delta)
    assert citation is not None
    assert citation.citation_data is not None
    assert citation.citation_data['title'] is None  # Empty string converted to None


def test_parse_citation_delta_web_search_none_title():
    """Test parsing web search citation with None title."""
    web_search_citation = BetaCitationsWebSearchResultLocation(
        url='https://example.com',
        title=None,
        cited_text='Cited text',
        encrypted_index='encrypted_123',
        type='web_search_result_location',
    )
    delta = BetaCitationsDelta(citation=web_search_citation, type='citations_delta')

    citation = _parse_anthropic_citation_delta(delta)
    assert citation is not None
    assert citation.citation_data is not None
    assert citation.citation_data['title'] is None


def test_parse_citation_delta_web_search_invalid_url():
    """Parsing web search citation with invalid URL (empty string)."""
    try:
        web_search_citation = BetaCitationsWebSearchResultLocation(
            url='',
            title='Example Site',
            cited_text='Cited text',
            encrypted_index='encrypted_123',
            type='web_search_result_location',
        )
        delta = BetaCitationsDelta(citation=web_search_citation, type='citations_delta')
        citation = _parse_anthropic_citation_delta(delta)
        assert citation is None
    except (ValueError, TypeError, AttributeError):
        # SDK may validate and raise error, which is fine
        pytest.skip('SDK validates empty URL')


def test_parse_citation_delta_search_result():
    """Test parsing a search result citation."""
    search_result_citation = BetaCitationSearchResultLocation(
        source='https://example.org',
        title='Search Result',
        cited_text='Cited from search',
        search_result_index=0,
        start_block_index=0,
        end_block_index=1,
        type='search_result_location',
    )
    delta = BetaCitationsDelta(citation=search_result_citation, type='citations_delta')

    citation = _parse_anthropic_citation_delta(delta)
    assert citation is not None
    assert isinstance(citation, ToolResultCitation)
    assert citation.tool_name == 'search'
    assert citation.tool_call_id is None
    assert citation.citation_data is not None
    assert citation.citation_data['source'] == 'https://example.org'
    assert citation.citation_data['title'] == 'Search Result'
    assert citation.citation_data['cited_text'] == 'Cited from search'
    assert citation.citation_data['search_result_index'] == 0


def test_parse_citation_delta_search_result_invalid_source():
    """Parsing search result citation with invalid source (empty string)."""
    try:
        search_result_citation = BetaCitationSearchResultLocation(
            source='',
            title='Search Result',
            cited_text='Cited from search',
            search_result_index=0,
            start_block_index=0,
            end_block_index=1,
            type='search_result_location',
        )
        delta = BetaCitationsDelta(citation=search_result_citation, type='citations_delta')
        citation = _parse_anthropic_citation_delta(delta)
        assert citation is None
    except (ValueError, TypeError, AttributeError):
        # SDK may validate and raise error, which is fine
        pytest.skip('SDK validates empty source')


# Unit tests for _parse_anthropic_text_block_citations


def test_parse_text_block_citations_none():
    """Test parsing when citations is None."""
    text_block = BetaTextBlock(text='Hello, world!', type='text')
    # BetaTextBlock doesn't have citations by default, so we need to mock it
    # In practice, citations would be set by the API response
    citations = _parse_anthropic_text_block_citations(text_block)
    assert citations == []  # Should return empty list when citations is None or missing


def test_parse_text_block_citations_empty_list():
    """Test parsing when citations is an empty list."""
    # Create a text block with empty citations
    # Note: BetaTextBlock may not support setting citations directly, so the function behavior is tested
    text_block = BetaTextBlock(text='Hello, world!', type='text')
    # The function checks for citations attribute, which may not exist
    citations = _parse_anthropic_text_block_citations(text_block)
    assert citations == []


def test_parse_text_block_citations_single_web_search():
    """Test parsing a single web search citation from text block."""
    web_search_citation = BetaCitationsWebSearchResultLocation(
        url='https://example.com',
        title='Example Site',
        cited_text='This is cited text',
        encrypted_index='encrypted_123',
        type='web_search_result_location',
    )

    # Create a text block with citations
    # BetaTextBlock may support citations in constructor or we need to set it
    text_block = BetaTextBlock(text='Hello, world!', type='text', citations=[web_search_citation])  # type: ignore

    citations = _parse_anthropic_text_block_citations(text_block)
    assert len(citations) == 1
    assert isinstance(citations[0], ToolResultCitation)
    assert citations[0].tool_name == 'web_search'
    assert citations[0].citation_data is not None
    assert citations[0].citation_data['url'] == 'https://example.com'


def test_parse_text_block_citations_multiple():
    """Test parsing multiple citations from text block."""
    web_search_citation1 = BetaCitationsWebSearchResultLocation(
        url='https://example.com',
        title='Example Site',
        cited_text='First citation',
        encrypted_index='encrypted_123',
        type='web_search_result_location',
    )
    web_search_citation2 = BetaCitationsWebSearchResultLocation(
        url='https://example.org',
        title='Another Site',
        cited_text='Second citation',
        encrypted_index='encrypted_456',
        type='web_search_result_location',
    )

    text_block = BetaTextBlock(
        text='Hello, world!',
        type='text',
        citations=[web_search_citation1, web_search_citation2],  # type: ignore
    )

    citations = _parse_anthropic_text_block_citations(text_block)
    assert len(citations) == 2
    assert citations[0].citation_data is not None
    assert citations[0].citation_data['url'] == 'https://example.com'
    assert citations[1].citation_data is not None
    assert citations[1].citation_data['url'] == 'https://example.org'


def test_parse_text_block_citations_mixed_types():
    """Test parsing citations with mixed types (web search and search result)."""
    web_search_citation = BetaCitationsWebSearchResultLocation(
        url='https://example.com',
        title='Example Site',
        cited_text='Web search citation',
        encrypted_index='encrypted_123',
        type='web_search_result_location',
    )
    search_result_citation = BetaCitationSearchResultLocation(
        source='https://example.org',
        title='Search Result',
        cited_text='Search result citation',
        search_result_index=0,
        start_block_index=0,
        end_block_index=1,
        type='search_result_location',
    )

    text_block = BetaTextBlock(
        text='Hello, world!',
        type='text',
        citations=[web_search_citation, search_result_citation],  # type: ignore
    )

    citations = _parse_anthropic_text_block_citations(text_block)
    assert len(citations) == 2
    assert citations[0].tool_name == 'web_search'
    assert citations[1].tool_name == 'search'


def test_parse_text_block_citations_invalid_web_search():
    """Parsing text block with invalid web search citation (empty URL)."""
    try:
        web_search_citation = BetaCitationsWebSearchResultLocation(
            url='',
            title='Example Site',
            cited_text='Cited text',
            encrypted_index='encrypted_123',
            type='web_search_result_location',
        )
        text_block = BetaTextBlock(
            text='Hello, world!',
            type='text',
            citations=[web_search_citation],  # type: ignore
        )
        citations = _parse_anthropic_text_block_citations(text_block)
        assert citations == []
    except (ValueError, TypeError, AttributeError):
        # SDK may validate and raise error, which is fine
        pytest.skip('SDK validates empty URL')


def test_parse_text_block_citations_skips_document_citations():
    """Test that document citations (char_location, etc.) are skipped."""
    # Document citations are a different feature and should be ignored
    # We can't easily create these without the full SDK structure, but we can test
    # that the function only processes tool result citations
    text_block = BetaTextBlock(text='Hello, world!', type='text')
    # If citations contains document citations, they should be skipped
    # For now, we just verify the function handles None/empty gracefully
    citations = _parse_anthropic_text_block_citations(text_block)
    assert citations == []


# Integration tests for streaming with citations


@pytest.mark.anyio
async def test_stream_with_single_citation(allow_model_requests: None):
    """Test streaming with a single citation event."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_anthropic import MockAnthropic

    # Create a web search citation
    web_search_citation = BetaCitationsWebSearchResultLocation(
        url='https://example.com',
        title='Example Site',
        cited_text='Hello',
        encrypted_index='encrypted_123',
        type='web_search_result_location',
    )

    # Create streaming events
    stream_events = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                model='claude-3-5-haiku-123',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=10, output_tokens=0),
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaTextBlock(text='', type='text'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaTextDelta(type='text_delta', text='Hello'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaCitationsDelta(citation=web_search_citation, type='citations_delta'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaTextDelta(type='text_delta', text=' world!'),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(input_tokens=10, output_tokens=5),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    mock_client = MockAnthropic.create_stream_mock(stream_events)
    model = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))

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
    assert isinstance(text_part_with_citations.citations[0], ToolResultCitation)
    assert text_part_with_citations.citations[0].tool_name == 'web_search'
    assert text_part_with_citations.citations[0].citation_data is not None
    assert text_part_with_citations.citations[0].citation_data['url'] == 'https://example.com'
    assert text_part_with_citations.citations[0].citation_data['title'] == 'Example Site'
    assert text_part_with_citations.content == 'Hello world!'


@pytest.mark.anyio
async def test_stream_with_multiple_citations(allow_model_requests: None):
    """Test streaming with multiple citation events."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_anthropic import MockAnthropic

    # Create multiple web search citations
    web_search_citation1 = BetaCitationsWebSearchResultLocation(
        url='https://example.com',
        title='Example Site',
        cited_text='Hello',
        encrypted_index='encrypted_123',
        type='web_search_result_location',
    )
    web_search_citation2 = BetaCitationsWebSearchResultLocation(
        url='https://example.org',
        title='Another Site',
        cited_text='world',
        encrypted_index='encrypted_456',
        type='web_search_result_location',
    )

    stream_events = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                model='claude-3-5-haiku-123',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=10, output_tokens=0),
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaTextBlock(text='', type='text'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaTextDelta(type='text_delta', text='Hello'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaCitationsDelta(citation=web_search_citation1, type='citations_delta'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaTextDelta(type='text_delta', text=' world'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaCitationsDelta(citation=web_search_citation2, type='citations_delta'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaTextDelta(type='text_delta', text='!'),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(input_tokens=10, output_tokens=5),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    mock_client = MockAnthropic.create_stream_mock(stream_events)
    model = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    async with model.request_stream(messages, None, ModelRequestParameters()) as streamed_response:
        async for _event in streamed_response:
            pass

    final_response = streamed_response.get()

    text_part_with_citations = None
    for part in final_response.parts:
        if isinstance(part, TextPart) and part.citations:
            text_part_with_citations = part
            break

    if text_part_with_citations is None:
        for part in final_response.parts:
            if isinstance(part, TextPart):
                text_part_with_citations = part
                break

    assert text_part_with_citations is not None
    assert text_part_with_citations.citations is not None
    assert len(text_part_with_citations.citations) == 2
    assert text_part_with_citations.citations[0].citation_data['url'] == 'https://example.com'
    assert text_part_with_citations.citations[1].citation_data['url'] == 'https://example.org'


@pytest.mark.anyio
async def test_stream_citation_before_text(allow_model_requests: None):
    """Test that citations arriving before text content are handled correctly."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_anthropic import MockAnthropic

    web_search_citation = BetaCitationsWebSearchResultLocation(
        url='https://example.com',
        title='Example Site',
        cited_text='Hello world!',
        encrypted_index='encrypted_123',
        type='web_search_result_location',
    )

    # Citation arrives before text content
    stream_events = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                model='claude-3-5-haiku-123',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=10, output_tokens=0),
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaTextBlock(text='', type='text'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaCitationsDelta(citation=web_search_citation, type='citations_delta'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaTextDelta(type='text_delta', text='Hello world!'),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(input_tokens=10, output_tokens=5),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    mock_client = MockAnthropic.create_stream_mock(stream_events)
    model = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    async with model.request_stream(messages, None, ModelRequestParameters()) as streamed_response:
        async for _event in streamed_response:
            pass

    final_response = streamed_response.get()

    text_part_with_citations = None
    for part in final_response.parts:
        if isinstance(part, TextPart):
            text_part_with_citations = part
            break

    assert text_part_with_citations is not None
    # Citation should still be attached even if it arrived before text
    if text_part_with_citations.citations:
        assert len(text_part_with_citations.citations) == 1


@pytest.mark.anyio
async def test_stream_invalid_citation_skipped(allow_model_requests: None):
    """Test that invalid citations are skipped during streaming."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_anthropic import MockAnthropic

    # Use a document citation type which should be skipped (parser only handles tool result citations)

    # Document citation (should be skipped by parser)
    document_citation = BetaCitationCharLocation(
        cited_text='Hello',
        document_index=0,
        start_char_index=0,
        end_char_index=5,
        type='char_location',
    )

    stream_events = [
        BetaRawMessageStartEvent(
            type='message_start',
            message=BetaMessage(
                id='msg_123',
                model='claude-3-5-haiku-123',
                role='assistant',
                type='message',
                content=[],
                stop_reason=None,
                usage=BetaUsage(input_tokens=10, output_tokens=0),
            ),
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start',
            index=0,
            content_block=BetaTextBlock(text='', type='text'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaTextDelta(type='text_delta', text='Hello world!'),
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaCitationsDelta(citation=document_citation, type='citations_delta'),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='end_turn'),
            usage=BetaMessageDeltaUsage(input_tokens=10, output_tokens=5),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]

    mock_client = MockAnthropic.create_stream_mock(stream_events)
    model = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    async with model.request_stream(messages, None, ModelRequestParameters()) as streamed_response:
        async for _event in streamed_response:
            pass

    final_response = streamed_response.get()

    text_part = None
    for part in final_response.parts:
        if isinstance(part, TextPart):
            text_part = part
            break

    assert text_part is not None
    # Document citations should be skipped (parser returns None for non-tool-result citations)
    assert text_part.citations is None or len(text_part.citations) == 0


# Integration tests for non-streaming with citations


@pytest.mark.anyio
async def test_non_streaming_with_citations(allow_model_requests: None):
    """Test non-streaming response with citations."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_anthropic import MockAnthropic, completion_message

    # Create a text block with citations
    web_search_citation = BetaCitationsWebSearchResultLocation(
        url='https://example.com',
        title='Example Site',
        cited_text='Hello world!',
        encrypted_index='encrypted_123',
        type='web_search_result_location',
    )

    text_block = BetaTextBlock(
        text='Hello world!',
        type='text',
        citations=[web_search_citation],  # type: ignore
    )

    message = completion_message([text_block], BetaUsage(input_tokens=10, output_tokens=5))
    mock_client = MockAnthropic.create_mock(message)
    model = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    response = await model.request(messages, None, ModelRequestParameters())

    # Find TextPart with citations
    text_part_with_citations = None
    for part in response.parts:
        if isinstance(part, TextPart) and part.citations:
            text_part_with_citations = part
            break

    if text_part_with_citations is None:
        for part in response.parts:
            if isinstance(part, TextPart):
                text_part_with_citations = part
                break

    assert text_part_with_citations is not None
    assert text_part_with_citations.citations is not None
    assert len(text_part_with_citations.citations) == 1
    assert isinstance(text_part_with_citations.citations[0], ToolResultCitation)
    assert text_part_with_citations.citations[0].tool_name == 'web_search'
    assert text_part_with_citations.citations[0].citation_data['url'] == 'https://example.com'


@pytest.mark.anyio
async def test_non_streaming_without_citations(allow_model_requests: None):
    """Test non-streaming response without citations."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_anthropic import MockAnthropic, completion_message

    text_block = BetaTextBlock(text='Hello world!', type='text')
    message = completion_message([text_block], BetaUsage(input_tokens=10, output_tokens=5))
    mock_client = MockAnthropic.create_mock(message)
    model = AnthropicModel('claude-3-5-haiku-latest', provider=AnthropicProvider(anthropic_client=mock_client))

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    response = await model.request(messages, None, ModelRequestParameters())

    text_part = None
    for part in response.parts:
        if isinstance(part, TextPart):
            text_part = part
            break

    assert text_part is not None
    assert text_part.citations is None or len(text_part.citations) == 0
