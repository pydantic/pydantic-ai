"""Tests for OpenAI Responses API citation/annotation parsing."""

from __future__ import annotations as _annotations

from typing import cast

import pytest  # pyright: ignore[reportMissingImports]

from pydantic_ai import TextPart, URLCitation

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types.responses import ResponseOutputTextAnnotationAddedEvent

    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesStreamedResponse
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='OpenAI SDK not installed')


def _create_streamed_response():
    """Helper function to create a streamed response instance for testing."""
    from datetime import datetime

    from pydantic_ai.models import ModelRequestParameters

    return OpenAIResponsesStreamedResponse(
        model_request_parameters=ModelRequestParameters(),
        _model_name='gpt-4o',
        _response=iter([]),  # Empty iterator
        _timestamp=datetime.now(),
        _provider_name='openai',
    )


def test_parse_responses_annotation_none():
    """Test parsing when annotation url_citation is None."""
    streamed_response = _create_streamed_response()

    # Create a mock annotation object with None url_citation
    class MockAnnotation:
        type = 'url_citation'
        url_citation = None

    annotation = MockAnnotation()
    event = ResponseOutputTextAnnotationAddedEvent(
        annotation=annotation,  # type: ignore
        annotation_index=0,
        content_index=0,
        item_id='item-1',
        output_index=0,
        sequence_number=1,
        type='response.output_text.annotation.added',
    )

    citation = streamed_response._parse_responses_annotation(event)
    assert citation is None


def test_parse_responses_annotation_single():
    """Test parsing a single annotation."""
    streamed_response = _create_streamed_response()

    # Create a mock url_citation object
    class MockURLCitation:
        url = 'https://example.com'
        title = 'Example Site'
        start_index = 0
        end_index = 5

    url_citation_obj = MockURLCitation()

    # Create annotation object with url_citation
    class MockAnnotation:
        type = 'url_citation'
        url_citation = url_citation_obj

    annotation = MockAnnotation()
    event = ResponseOutputTextAnnotationAddedEvent(
        annotation=annotation,  # type: ignore
        annotation_index=0,
        content_index=0,
        item_id='item-1',
        output_index=0,
        sequence_number=1,
        type='response.output_text.annotation.added',
    )

    citation = streamed_response._parse_responses_annotation(event)
    assert citation is not None
    assert isinstance(citation, URLCitation)
    assert citation.url == 'https://example.com'
    assert citation.title == 'Example Site'
    assert citation.start_index == 0
    assert citation.end_index == 5


def test_parse_responses_annotation_no_title():
    """Test parsing annotation with empty title string."""
    streamed_response = _create_streamed_response()

    # Create a mock url_citation object
    class MockURLCitation:
        url = 'https://example.com'
        title = ''  # Empty title
        start_index = 0
        end_index = 5

    url_citation_obj = MockURLCitation()

    class MockAnnotation:
        type = 'url_citation'
        url_citation = url_citation_obj

    annotation = MockAnnotation()
    event = ResponseOutputTextAnnotationAddedEvent(
        annotation=annotation,  # type: ignore
        annotation_index=0,
        content_index=0,
        item_id='item-1',
        output_index=0,
        sequence_number=1,
        type='response.output_text.annotation.added',
    )

    citation = streamed_response._parse_responses_annotation(event)
    assert citation is not None
    assert citation.url == 'https://example.com'
    # Empty string title should be converted to None in our format
    assert citation.title is None


def test_parse_responses_annotation_invalid_indices_negative():
    """Test parsing annotation with negative indices (should return None)."""
    streamed_response = _create_streamed_response()

    # Create a mock url_citation object
    class MockURLCitation:
        url = 'https://example.com'
        title = 'Test'
        start_index = -1  # Invalid negative index
        end_index = 5

    url_citation_obj = MockURLCitation()

    class MockAnnotation:
        type = 'url_citation'
        url_citation = url_citation_obj

    annotation = MockAnnotation()
    event = ResponseOutputTextAnnotationAddedEvent(
        annotation=annotation,  # type: ignore
        annotation_index=0,
        content_index=0,
        item_id='item-1',
        output_index=0,
        sequence_number=1,
        type='response.output_text.annotation.added',
    )

    citation = streamed_response._parse_responses_annotation(event)
    assert citation is None  # Invalid indices should be skipped


def test_parse_responses_annotation_invalid_indices_start_gt_end():
    """Test parsing annotation with start > end (should return None)."""
    streamed_response = _create_streamed_response()

    # Create a mock url_citation object
    class MockURLCitation:
        url = 'https://example.com'
        title = 'Test'
        start_index = 10  # Start > end
        end_index = 5

    url_citation_obj = MockURLCitation()

    class MockAnnotation:
        type = 'url_citation'
        url_citation = url_citation_obj

    annotation = MockAnnotation()
    event = ResponseOutputTextAnnotationAddedEvent(
        annotation=annotation,  # type: ignore
        annotation_index=0,
        content_index=0,
        item_id='item-1',
        output_index=0,
        sequence_number=1,
        type='response.output_text.annotation.added',
    )

    citation = streamed_response._parse_responses_annotation(event)
    assert citation is None  # Invalid range should be skipped


def test_parse_responses_annotation_invalid_type():
    """Test parsing annotation with non-url_citation type (should return None)."""
    streamed_response = _create_streamed_response()

    # Create annotation with invalid type
    class MockAnnotation:
        type = 'invalid_type'  # Not url_citation
        url_citation = None

    annotation = MockAnnotation()
    event = ResponseOutputTextAnnotationAddedEvent(
        annotation=annotation,  # type: ignore
        annotation_index=0,
        content_index=0,
        item_id='item-1',
        output_index=0,
        sequence_number=1,
        type='response.output_text.annotation.added',
    )

    citation = streamed_response._parse_responses_annotation(event)
    # Should return None for non-url_citation type
    assert citation is None


def test_parse_responses_annotation_missing_url():
    """Test parsing annotation with empty URL (should return None)."""
    streamed_response = _create_streamed_response()

    # Create a mock url_citation object
    class MockURLCitation:
        url = ''  # Empty URL
        title = 'Test'
        start_index = 0
        end_index = 5

    url_citation_obj = MockURLCitation()

    class MockAnnotation:
        type = 'url_citation'
        url_citation = url_citation_obj

    annotation = MockAnnotation()
    event = ResponseOutputTextAnnotationAddedEvent(
        annotation=annotation,  # type: ignore
        annotation_index=0,
        content_index=0,
        item_id='item-1',
        output_index=0,
        sequence_number=1,
        type='response.output_text.annotation.added',
    )

    citation = streamed_response._parse_responses_annotation(event)
    assert citation is None  # Empty URL should be skipped


def test_parse_responses_annotation_malformed():
    """Test parsing malformed annotation (should return None gracefully)."""
    streamed_response = _create_streamed_response()

    # Create annotation with missing required fields
    # We'll use a mock object that doesn't have the required attributes
    class MockAnnotation:
        type = 'url_citation'
        # Missing url_citation attribute

    annotation = MockAnnotation()
    event = ResponseOutputTextAnnotationAddedEvent(
        annotation=annotation,  # type: ignore
        annotation_index=0,
        content_index=0,
        item_id='item-1',
        output_index=0,
        sequence_number=1,
        type='response.output_text.annotation.added',
    )

    citation = streamed_response._parse_responses_annotation(event)
    assert citation is None  # Malformed annotation should be skipped gracefully


# Integration tests for streaming with citations


@pytest.mark.anyio
async def test_stream_with_single_annotation(allow_model_requests: None):
    """Test streaming with a single annotation event."""
    from openai.types.responses import (
        ResponseCompletedEvent,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
    )
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage, ResponseOutputText
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

    from .mock_openai import MockOpenAIResponses, response_message

    # Create a mock stream with text deltas and annotation
    # Create a mock url_citation object
    class MockURLCitation:
        url = 'https://example.com'
        title = 'Example Site'
        start_index = 0
        end_index = 5

    url_citation_obj = MockURLCitation()

    class MockAnnotation:
        type = 'url_citation'
        url_citation = url_citation_obj

    annotation = MockAnnotation()

    from openai.types.responses import ResponseCreatedEvent

    stream_events = [
        ResponseCreatedEvent(
            response=response_message(
                [
                    ResponseOutputMessage(
                        id='item-1',
                        content=cast(list[Content], [ResponseOutputText(text='', type='output_text', annotations=[])]),
                        role='assistant',
                        status='in_progress',
                        type='message',
                    )
                ],
            ),
            sequence_number=0,
            type='response.created',
        ),
        ResponseTextDeltaEvent(
            item_id='item-1',
            delta='Hello',
            output_index=0,
            content_index=0,
            logprobs=[],
            sequence_number=1,
            type='response.output_text.delta',
        ),
        ResponseOutputTextAnnotationAddedEvent(
            annotation=annotation,  # type: ignore
            annotation_index=0,
            content_index=0,
            item_id='item-1',
            output_index=0,
            sequence_number=2,
            type='response.output_text.annotation.added',
        ),
        ResponseTextDeltaEvent(
            item_id='item-1',
            delta=' world!',
            output_index=0,
            content_index=0,
            logprobs=[],
            sequence_number=3,
            type='response.output_text.delta',
        ),
        ResponseTextDoneEvent(
            item_id='item-1',
            output_index=0,
            content_index=0,
            logprobs=[],
            text='Hello world!',
            sequence_number=4,
            type='response.output_text.done',
        ),
        ResponseCompletedEvent(
            response=response_message(
                [
                    ResponseOutputMessage(
                        id='item-1',
                        content=cast(
                            list[Content], [ResponseOutputText(text='Hello world!', type='output_text', annotations=[])]
                        ),
                        role='assistant',
                        status='completed',
                        type='message',
                    )
                ],
                usage=ResponseUsage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    input_tokens_details=InputTokensDetails(cached_tokens=0),
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                ),
            ),
            sequence_number=5,
            type='response.completed',
        ),
    ]

    mock_client = MockOpenAIResponses.create_mock_stream(stream_events)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    # Stream the response using request_stream
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    async with model.request_stream(messages, None, ModelRequestParameters()) as streamed_response:
        # streamed_response is the StreamedResponse object
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
    assert text_part_with_citations.citations[0].url == 'https://example.com'
    assert text_part_with_citations.citations[0].title == 'Example Site'


@pytest.mark.anyio
async def test_stream_with_multiple_annotations(allow_model_requests: None):
    """Test streaming with multiple annotation events."""
    from openai.types.responses import (
        ResponseCompletedEvent,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
    )
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage, ResponseOutputText
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

    from .mock_openai import MockOpenAIResponses, response_message

    # Create multiple annotations
    # Create mock url_citation objects
    class MockURLCitation1:
        url = 'https://example.com'
        title = 'Example Site'
        start_index = 0
        end_index = 5

    class MockURLCitation2:
        url = 'https://example.org'
        title = 'Another Site'
        start_index = 7
        end_index = 12

    url_citation1 = MockURLCitation1()
    url_citation2 = MockURLCitation2()

    class MockAnnotation1:
        type = 'url_citation'
        url_citation = url_citation1

    class MockAnnotation2:
        type = 'url_citation'
        url_citation = url_citation2

    annotation1 = MockAnnotation1()
    annotation2 = MockAnnotation2()

    from openai.types.responses import ResponseCreatedEvent

    stream_events = [
        ResponseCreatedEvent(
            response=response_message(
                [
                    ResponseOutputMessage(
                        id='item-1',
                        content=cast(list[Content], [ResponseOutputText(text='', type='output_text', annotations=[])]),
                        role='assistant',
                        status='in_progress',
                        type='message',
                    )
                ],
            ),
            sequence_number=0,
            type='response.created',
        ),
        ResponseTextDeltaEvent(
            item_id='item-1',
            delta='Hello world!',
            output_index=0,
            content_index=0,
            logprobs=[],
            sequence_number=1,
            type='response.output_text.delta',
        ),
        ResponseOutputTextAnnotationAddedEvent(
            annotation=annotation1,  # type: ignore
            annotation_index=0,
            content_index=0,
            item_id='item-1',
            output_index=0,
            sequence_number=2,
            type='response.output_text.annotation.added',
        ),
        ResponseOutputTextAnnotationAddedEvent(
            annotation=annotation2,  # type: ignore
            annotation_index=1,
            content_index=0,
            item_id='item-1',
            output_index=0,
            sequence_number=3,
            type='response.output_text.annotation.added',
        ),
        ResponseTextDoneEvent(
            item_id='item-1',
            output_index=0,
            content_index=0,
            logprobs=[],
            text='Hello world!',
            sequence_number=4,
            type='response.output_text.done',
        ),
        ResponseCompletedEvent(
            response=response_message(
                [
                    ResponseOutputMessage(
                        id='item-1',
                        content=cast(
                            list[Content], [ResponseOutputText(text='Hello world!', type='output_text', annotations=[])]
                        ),
                        role='assistant',
                        status='completed',
                        type='message',
                    )
                ],
                usage=ResponseUsage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    input_tokens_details=InputTokensDetails(cached_tokens=0),
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                ),
            ),
            sequence_number=5,
            type='response.completed',
        ),
    ]

    mock_client = MockOpenAIResponses.create_mock_stream(stream_events)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    # Stream the response using request_stream
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters

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
    assert len(text_part_with_citations.citations) == 2
    assert text_part_with_citations.citations[0].url == 'https://example.com'
    assert text_part_with_citations.citations[1].url == 'https://example.org'


@pytest.mark.anyio
async def test_stream_annotation_before_textpart(allow_model_requests: None):
    """Test that annotation arriving before TextPart is created is handled gracefully."""
    from openai.types.responses import (
        ResponseCompletedEvent,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
    )
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage, ResponseOutputText
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

    from .mock_openai import MockOpenAIResponses, response_message

    # Create annotation
    # Create a mock url_citation object
    class MockURLCitation:
        url = 'https://example.com'
        title = 'Example Site'
        start_index = 0
        end_index = 5

    url_citation_obj = MockURLCitation()

    class MockAnnotation:
        type = 'url_citation'
        url_citation = url_citation_obj

    annotation = MockAnnotation()

    # Annotation arrives before text delta (edge case)
    from openai.types.responses import ResponseCreatedEvent

    stream_events = [
        ResponseCreatedEvent(
            response=response_message(
                [
                    ResponseOutputMessage(
                        id='item-1',
                        content=cast(list[Content], [ResponseOutputText(text='', type='output_text', annotations=[])]),
                        role='assistant',
                        status='in_progress',
                        type='message',
                    )
                ],
            ),
            sequence_number=0,
            type='response.created',
        ),
        ResponseOutputTextAnnotationAddedEvent(
            annotation=annotation,  # type: ignore
            annotation_index=0,
            content_index=0,
            item_id='item-1',
            output_index=0,
            sequence_number=1,
            type='response.output_text.annotation.added',
        ),
        ResponseTextDeltaEvent(
            item_id='item-1',
            delta='Hello world!',
            output_index=0,
            content_index=0,
            logprobs=[],
            sequence_number=2,
            type='response.output_text.delta',
        ),
        ResponseTextDoneEvent(
            item_id='item-1',
            output_index=0,
            content_index=0,
            logprobs=[],
            text='Hello world!',
            sequence_number=3,
            type='response.output_text.done',
        ),
        ResponseCompletedEvent(
            response=response_message(
                [
                    ResponseOutputMessage(
                        id='item-1',
                        content=cast(
                            list[Content], [ResponseOutputText(text='Hello world!', type='output_text', annotations=[])]
                        ),
                        role='assistant',
                        status='completed',
                        type='message',
                    )
                ],
                usage=ResponseUsage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    input_tokens_details=InputTokensDetails(cached_tokens=0),
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                ),
            ),
            sequence_number=4,
            type='response.completed',
        ),
    ]

    mock_client = MockOpenAIResponses.create_mock_stream(stream_events)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    # Stream the response - should not crash even if annotation arrives before text
    # Stream the response using request_stream
    from pydantic_ai.messages import ModelRequest, PartStartEvent, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
        parts = []
        async for event in stream:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                parts.append(event.part)

    # TextPart should exist, citation may or may not be attached depending on timing
    assert len(parts) > 0
    text_part = parts[-1]
    # If citation was attached, verify it; if not, that's okay (edge case)
    if text_part.citations:
        assert len(text_part.citations) == 1


@pytest.mark.anyio
async def test_stream_invalid_annotation_skipped(allow_model_requests: None):
    """Test that invalid annotations are skipped during streaming."""
    from openai.types.responses import (
        ResponseCompletedEvent,
        ResponseTextDeltaEvent,
        ResponseTextDoneEvent,
    )
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage, ResponseOutputText
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage

    from .mock_openai import MockOpenAIResponses, response_message

    # Create invalid annotation (empty URL)
    class MockURLCitation:
        url = ''  # Empty URL - invalid
        title = 'Test'
        start_index = 0
        end_index = 5

    url_citation_obj = MockURLCitation()

    class MockAnnotation:
        type = 'url_citation'
        url_citation = url_citation_obj

    annotation = MockAnnotation()

    from openai.types.responses import ResponseCreatedEvent

    stream_events = [
        ResponseCreatedEvent(
            response=response_message(
                [
                    ResponseOutputMessage(
                        id='item-1',
                        content=cast(list[Content], [ResponseOutputText(text='', type='output_text', annotations=[])]),
                        role='assistant',
                        status='in_progress',
                        type='message',
                    )
                ],
            ),
            sequence_number=0,
            type='response.created',
        ),
        ResponseTextDeltaEvent(
            item_id='item-1',
            delta='Hello world!',
            output_index=0,
            content_index=0,
            logprobs=[],
            sequence_number=1,
            type='response.output_text.delta',
        ),
        ResponseOutputTextAnnotationAddedEvent(
            annotation=annotation,  # type: ignore
            annotation_index=0,
            content_index=0,
            item_id='item-1',
            output_index=0,
            sequence_number=2,
            type='response.output_text.annotation.added',
        ),
        ResponseTextDoneEvent(
            item_id='item-1',
            output_index=0,
            content_index=0,
            logprobs=[],
            text='Hello world!',
            sequence_number=3,
            type='response.output_text.done',
        ),
        ResponseCompletedEvent(
            response=response_message(
                [
                    ResponseOutputMessage(
                        id='item-1',
                        content=cast(
                            list[Content], [ResponseOutputText(text='Hello world!', type='output_text', annotations=[])]
                        ),
                        role='assistant',
                        status='completed',
                        type='message',
                    )
                ],
                usage=ResponseUsage(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    input_tokens_details=InputTokensDetails(cached_tokens=0),
                    output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                ),
            ),
            sequence_number=4,
            type='response.completed',
        ),
    ]

    mock_client = MockOpenAIResponses.create_mock_stream(stream_events)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    # Stream the response using request_stream
    from pydantic_ai.messages import ModelRequest, PartStartEvent, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters

    messages = [ModelRequest(parts=[UserPromptPart(content='Test')])]
    async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
        parts = []
        async for event in stream:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                parts.append(event.part)

    # Check that invalid annotation was skipped
    assert len(parts) > 0
    text_part = parts[-1]
    # Citations should be None or empty (invalid annotation was skipped)
    assert text_part.citations is None or len(text_part.citations) == 0
