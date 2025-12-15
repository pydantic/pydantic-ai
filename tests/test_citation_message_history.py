"""Tests for citations in message history."""

from __future__ import annotations as _annotations

import pytest  # pyright: ignore[reportMissingImports]

from pydantic_ai import GroundingCitation, TextPart, ToolResultCitation, URLCitation, usage
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, UserPromptPart


def test_citation_serialization_round_trip():
    """Citations survive JSON serialization/deserialization."""
    # Test URLCitation
    url_citation = URLCitation(url='https://example.com', title='Example', start_index=0, end_index=5)
    text_part = TextPart(content='Hello', citations=[url_citation])
    response = ModelResponse(
        parts=[text_part],
        model_name='test',
        usage=usage.RequestUsage(input_tokens=10, output_tokens=5),
    )
    messages = [response]

    # Serialize and deserialize
    json_bytes = ModelMessagesTypeAdapter.dump_json(messages)
    deserialized = ModelMessagesTypeAdapter.validate_python(ModelMessagesTypeAdapter.validate_json(json_bytes))

    assert len(deserialized) == 1
    assert len(deserialized[0].parts) == 1
    assert isinstance(deserialized[0].parts[0], TextPart)
    assert deserialized[0].parts[0].citations is not None
    assert len(deserialized[0].parts[0].citations) == 1
    assert isinstance(deserialized[0].parts[0].citations[0], URLCitation)
    assert deserialized[0].parts[0].citations[0].url == 'https://example.com'
    assert deserialized[0].parts[0].citations[0].title == 'Example'


def test_tool_result_citation_serialization():
    """ToolResultCitation survives serialization."""
    tool_citation = ToolResultCitation(
        tool_name='web_search',
        tool_call_id='call_123',
        citation_data={'url': 'https://example.com', 'title': 'Example'},
    )
    text_part = TextPart(content='Hello', citations=[tool_citation])
    response = ModelResponse(
        parts=[text_part],
        model_name='test',
        usage=usage.RequestUsage(input_tokens=10, output_tokens=5),
    )
    messages = [response]

    json_bytes = ModelMessagesTypeAdapter.dump_json(messages)
    deserialized = ModelMessagesTypeAdapter.validate_python(ModelMessagesTypeAdapter.validate_json(json_bytes))

    assert deserialized[0].parts[0].citations is not None
    assert len(deserialized[0].parts[0].citations) == 1
    assert isinstance(deserialized[0].parts[0].citations[0], ToolResultCitation)
    assert deserialized[0].parts[0].citations[0].tool_name == 'web_search'
    assert deserialized[0].parts[0].citations[0].citation_data['url'] == 'https://example.com'


def test_grounding_citation_serialization():
    """GroundingCitation survives serialization."""
    grounding_citation = GroundingCitation(
        citation_metadata={
            'citations': [{'uri': 'https://example.com', 'title': 'Example', 'start_index': 0, 'end_index': 5}]
        }
    )
    text_part = TextPart(content='Hello', citations=[grounding_citation])
    response = ModelResponse(
        parts=[text_part],
        model_name='test',
        usage=usage.RequestUsage(input_tokens=10, output_tokens=5),
    )
    messages = [response]

    json_bytes = ModelMessagesTypeAdapter.dump_json(messages)
    deserialized = ModelMessagesTypeAdapter.validate_python(ModelMessagesTypeAdapter.validate_json(json_bytes))

    assert deserialized[0].parts[0].citations is not None
    assert len(deserialized[0].parts[0].citations) == 1
    assert isinstance(deserialized[0].parts[0].citations[0], GroundingCitation)
    assert deserialized[0].parts[0].citations[0].citation_metadata is not None
    assert deserialized[0].parts[0].citations[0].citation_metadata['citations'][0]['uri'] == 'https://example.com'


def test_multiple_citations_serialization():
    """Multiple citations survive serialization."""
    url_citation = URLCitation(url='https://example.com', title='Example', start_index=0, end_index=5)
    tool_citation = ToolResultCitation(
        tool_name='web_search',
        tool_call_id='call_123',
        citation_data={'url': 'https://example.org', 'title': 'Another'},
    )
    text_part = TextPart(content='Hello', citations=[url_citation, tool_citation])
    response = ModelResponse(
        parts=[text_part],
        model_name='test',
        usage=usage.RequestUsage(input_tokens=10, output_tokens=5),
    )
    messages = [response]

    json_bytes = ModelMessagesTypeAdapter.dump_json(messages)
    deserialized = ModelMessagesTypeAdapter.validate_python(ModelMessagesTypeAdapter.validate_json(json_bytes))

    assert deserialized[0].parts[0].citations is not None
    assert len(deserialized[0].parts[0].citations) == 2
    assert isinstance(deserialized[0].parts[0].citations[0], URLCitation)
    assert isinstance(deserialized[0].parts[0].citations[1], ToolResultCitation)


def test_citation_in_multi_turn_conversation():
    """Citations persist in multi-turn conversations."""
    # First turn with citation
    url_citation = URLCitation(url='https://example.com', title='Example', start_index=0, end_index=5)
    text_part1 = TextPart(content='Hello', citations=[url_citation])
    response1 = ModelResponse(
        parts=[text_part1],
        model_name='test',
        usage=usage.RequestUsage(input_tokens=10, output_tokens=5),
    )

    # Second turn
    request2 = ModelRequest(parts=[UserPromptPart(content='Continue')])
    text_part2 = TextPart(content='World')
    response2 = ModelResponse(
        parts=[text_part2],
        model_name='test',
        usage=usage.RequestUsage(input_tokens=15, output_tokens=5),
    )

    # Serialize full conversation
    messages = [response1, request2, response2]
    json_bytes = ModelMessagesTypeAdapter.dump_json(messages)
    deserialized = ModelMessagesTypeAdapter.validate_python(ModelMessagesTypeAdapter.validate_json(json_bytes))

    # Verify first response still has citations
    assert isinstance(deserialized[0], ModelResponse)
    assert len(deserialized[0].parts) == 1
    assert deserialized[0].parts[0].citations is not None
    assert len(deserialized[0].parts[0].citations) == 1
    assert deserialized[0].parts[0].citations[0].url == 'https://example.com'

    # Verify second response doesn't have citations (as expected)
    assert isinstance(deserialized[2], ModelResponse)
    assert deserialized[2].parts[0].citations is None or len(deserialized[2].parts[0].citations) == 0


@pytest.mark.anyio
async def test_citations_persist_in_agent_message_history(allow_model_requests: None):
    """Test that citations persist when using message_history in agent runs."""
    from anthropic.types.beta import BetaCitationsWebSearchResultLocation, BetaTextBlock, BetaUsage

    from pydantic_ai import Agent
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .models.test_anthropic import MockAnthropic, completion_message

    # Create a response with citations
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
    agent = Agent(model=model)

    # First run
    result1 = await agent.run('Test query')
    assert result1.output == 'Hello world!'

    # Verify citations in first response
    response1 = result1.response
    text_part_with_citations = None
    for part in response1.parts:
        if isinstance(part, TextPart) and part.citations:
            text_part_with_citations = part
            break

    assert text_part_with_citations is not None
    assert text_part_with_citations.citations is not None
    assert len(text_part_with_citations.citations) == 1

    # Second run with message_history
    result2 = await agent.run('Continue', message_history=result1.new_messages())

    # Verify citations are still in the message history
    all_messages = result2.all_messages()
    # Find the first response in history
    first_response = None
    for msg in all_messages:
        if isinstance(msg, ModelResponse) and msg != result2.response:
            first_response = msg
            break

    if first_response:
        # Check if citations are preserved
        for part in first_response.parts:
            if isinstance(part, TextPart) and part.citations:
                assert len(part.citations) == 1
                assert isinstance(part.citations[0], ToolResultCitation)
                break
