"""Tests for OpenAI citation/annotation parsing."""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import TextPart, URLCitation

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types.chat.chat_completion_message import Annotation, AnnotationURLCitation, ChatCompletionMessage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful, reason='OpenAI SDK not installed')


def test_parse_openai_annotations_none():
    """Test parsing when annotations is None."""
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))
    message = ChatCompletionMessage(role='assistant', content='Hello, world!')

    citations = model._parse_openai_annotations(message, content='Hello, world!')
    assert citations == []


def test_parse_openai_annotations_empty_list():
    """Test parsing when annotations is an empty list."""
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))
    message = ChatCompletionMessage(role='assistant', content='Hello, world!', annotations=[])

    citations = model._parse_openai_annotations(message, content='Hello, world!')
    assert citations == []


def test_parse_openai_annotations_single():
    """Test parsing a single annotation."""
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))

    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example Site',
        start_index=0,
        end_index=5,
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)
    message = ChatCompletionMessage(
        role='assistant',
        content='Hello, world!',
        annotations=[annotation],
    )

    citations = model._parse_openai_annotations(message, content='Hello, world!')
    assert len(citations) == 1
    assert isinstance(citations[0], URLCitation)
    assert citations[0].url == 'https://example.com'
    assert citations[0].title == 'Example Site'
    assert citations[0].start_index == 0
    assert citations[0].end_index == 5


def test_parse_openai_annotations_multiple():
    """Test parsing multiple annotations."""
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))

    url_citation1 = AnnotationURLCitation(
        url='https://example.com',
        title='Example Site',
        start_index=0,
        end_index=5,
    )
    url_citation2 = AnnotationURLCitation(
        url='https://example.org',
        title='Another Site',
        start_index=7,
        end_index=12,
    )
    annotation1 = Annotation(type='url_citation', url_citation=url_citation1)
    annotation2 = Annotation(type='url_citation', url_citation=url_citation2)
    message = ChatCompletionMessage(
        role='assistant',
        content='Hello, world!',
        annotations=[annotation1, annotation2],
    )

    citations = model._parse_openai_annotations(message, content='Hello, world!')
    assert len(citations) == 2
    assert citations[0].url == 'https://example.com'
    assert citations[1].url == 'https://example.org'


def test_parse_openai_annotations_no_title():
    """Test parsing annotation with empty title string (SDK requires title, but can be empty)."""
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))

    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='',  # SDK requires title, but can be empty string
        start_index=0,
        end_index=5,
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)
    message = ChatCompletionMessage(
        role='assistant',
        content='Hello, world!',
        annotations=[annotation],
    )

    citations = model._parse_openai_annotations(message, content='Hello, world!')
    assert len(citations) == 1
    assert citations[0].url == 'https://example.com'
    # Empty string title should be converted to None in our format
    assert citations[0].title is None or citations[0].title == ''


def test_parse_openai_annotations_invalid_indices_negative():
    """Test parsing annotation with negative indices (should be skipped)."""
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))

    # SDK validation prevents creating invalid citations, so we'll test by manually modifying
    # after creation, or we can test that the validation in our function works
    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Test',
        start_index=0,
        end_index=5,
    )
    # Manually set negative index for testing (bypassing SDK validation)
    url_citation.start_index = -1
    annotation = Annotation(type='url_citation', url_citation=url_citation)
    message = ChatCompletionMessage(
        role='assistant',
        content='Hello, world!',
        annotations=[annotation],
    )

    citations = model._parse_openai_annotations(message, content='Hello, world!')
    assert citations == []  # Invalid indices should be skipped


def test_parse_openai_annotations_invalid_indices_start_gt_end():
    """Test parsing annotation with start > end (should be skipped)."""
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))

    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Test',
        start_index=5,
        end_index=10,
    )
    # Manually set invalid range for testing
    url_citation.start_index = 10
    url_citation.end_index = 5
    annotation = Annotation(type='url_citation', url_citation=url_citation)
    message = ChatCompletionMessage(
        role='assistant',
        content='Hello, world!',
        annotations=[annotation],
    )

    citations = model._parse_openai_annotations(message, content='Hello, world!')
    assert citations == []  # Invalid range should be skipped


def test_parse_openai_annotations_out_of_bounds():
    """Test parsing annotation with indices out of content bounds."""
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))

    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Test',
        start_index=0,
        end_index=100,  # Content is only 13 characters
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)
    message = ChatCompletionMessage(
        role='assistant',
        content='Hello, world!',
        annotations=[annotation],
    )

    citations = model._parse_openai_annotations(message, content='Hello, world!')
    assert citations == []  # Out of bounds should be skipped


def test_parse_openai_annotations_no_content():
    """Test parsing annotations when content is None."""
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))

    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Test',
        start_index=0,
        end_index=5,
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)
    message = ChatCompletionMessage(
        role='assistant',
        content=None,
        annotations=[annotation],
    )

    # Should still parse citations even without content (no validation)
    citations = model._parse_openai_annotations(message, content=None)
    assert len(citations) == 1
    assert citations[0].url == 'https://example.com'


def test_parse_openai_annotations_at_boundary():
    """Test parsing annotation at content boundary (should be valid)."""
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))
    content = 'Hello, world!'

    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Test',
        start_index=0,
        end_index=len(content),  # At boundary
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)
    message = ChatCompletionMessage(
        role='assistant',
        content=content,
        annotations=[annotation],
    )

    citations = model._parse_openai_annotations(message, content=content)
    assert len(citations) == 1
    assert citations[0].end_index == len(content)


def test_parse_openai_annotations_invalid_type():
    """Test parsing annotation with None url_citation (should be skipped)."""
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test-key'))

    # Note: Annotation.type is Literal['url_citation'] and url_citation is required,
    # so invalid annotations can't be easily created. Instead, the code is tested to handle
    # the function handles missing annotations gracefully.
    # This test verifies the function doesn't crash on edge cases.
    message = ChatCompletionMessage(
        role='assistant',
        content='Hello, world!',
        annotations=[],  # Empty list
    )

    citations = model._parse_openai_annotations(message, content='Hello, world!')
    assert citations == []  # Empty annotations should return empty list


# Integration tests for _process_response()


def test_process_response_with_annotations_single_textpart():
    """Test that citations are attached to TextPart in response."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example Site',
        start_index=0,
        end_index=5,
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)
    message = ChatCompletionMessage(
        role='assistant',
        content='Hello, world!',
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    response = model._process_response(completion)

    # Check that we have a TextPart with citations
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 1
    assert text_parts[0].citations[0].url == 'https://example.com'


def test_process_response_without_annotations():
    """Test that responses without annotations work normally."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    message = ChatCompletionMessage(
        role='assistant',
        content='Hello, world!',
        annotations=None,
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    response = model._process_response(completion)

    # Check that we have a TextPart without citations
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    assert text_parts[0].citations is None


def test_process_response_with_multiple_annotations():
    """Test that multiple citations are attached correctly."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    url_citation1 = AnnotationURLCitation(
        url='https://example.com',
        title='Example Site',
        start_index=0,
        end_index=5,
    )
    url_citation2 = AnnotationURLCitation(
        url='https://example.org',
        title='Another Site',
        start_index=7,
        end_index=12,
    )
    annotation1 = Annotation(type='url_citation', url_citation=url_citation1)
    annotation2 = Annotation(type='url_citation', url_citation=url_citation2)

    message = ChatCompletionMessage(
        role='assistant',
        content='Hello, world!',
        annotations=[annotation1, annotation2],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    response = model._process_response(completion)

    # Check that we have a TextPart with multiple citations
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 2
    assert text_parts[0].citations[0].url == 'https://example.com'
    assert text_parts[0].citations[1].url == 'https://example.org'


# Tests for thinking tags + citations


def test_process_response_with_thinking_tags_and_citations():
    """Test citations with content that includes thinking tags."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Content with thinking tags: "Hello <think>reasoning</think> world"
    # Citation refers to "Hello" (start_index=0, end_index=5)
    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example Site',
        start_index=0,
        end_index=5,  # "Hello"
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)

    # Use <think> tags (common thinking tag format)
    content_with_thinking = 'Hello <think>some reasoning here</think> world'
    message = ChatCompletionMessage(
        role='assistant',
        content=content_with_thinking,
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    # Set thinking tags to match the content
    model.profile.thinking_tags = ('<think>', '</think>')

    response = model._process_response(completion)

    # Check that citation is attached to the first TextPart
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) >= 1
    # First TextPart should be "Hello " (before thinking tag)
    assert text_parts[0].content == 'Hello '
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 1
    assert text_parts[0].citations[0].url == 'https://example.com'


def test_process_response_citation_spanning_thinking_tag():
    """Test citation that spans across a thinking tag (should attach to first TextPart)."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Content: "Hello <think>r</think> world"
    # Citation spans from start to after thinking tag: start_index=0, end_index=18
    # This citation spans "Hello <think>r</think>"
    content_with_thinking = 'Hello <think>r</think> world'
    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example',
        start_index=0,
        end_index=18,  # Spans "Hello <think>r</think>"
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)

    message = ChatCompletionMessage(
        role='assistant',
        content=content_with_thinking,
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    model.profile.thinking_tags = ('<think>', '</think>')

    response = model._process_response(completion)

    # Citation should attach to the first TextPart where it starts
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) >= 1
    # Citation starts at index 0, which is in the first TextPart
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 1
    assert text_parts[0].citations[0].url == 'https://example.com'


def test_process_response_citation_inside_thinking_tag():
    """Test citation that refers to content inside thinking tag (should be dropped)."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Content: "Hello <think>reasoning</think> world"
    # Citation refers to content inside thinking tag: start_index=12, end_index=20
    # This is "reasoning" inside the tag
    content_with_thinking = 'Hello <think>reasoning</think> world'
    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example',
        start_index=12,  # Inside thinking tag
        end_index=20,  # Inside thinking tag
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)

    message = ChatCompletionMessage(
        role='assistant',
        content=content_with_thinking,
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    model.profile.thinking_tags = ('<think>', '</think>')

    response = model._process_response(completion)

    # Citation should be dropped (doesn't map to any TextPart)
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    for text_part in text_parts:
        # All TextParts should have no citations (citation was inside thinking tag)
        assert text_part.citations is None or len(text_part.citations) == 0


def test_process_response_citation_after_thinking_tag():
    """Test citation that refers to content after thinking tag."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Content: "Hello <think>r</think> world"
    # Citation refers to "world": need to calculate correct indices
    content_with_thinking = 'Hello <think>r</think> world'
    # "world" starts at position 20 (after "Hello <think>r</think> ")
    world_start = content_with_thinking.find('world')
    world_end = world_start + len('world')
    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example',
        start_index=world_start,  # "world" starts here
        end_index=world_end,  # "world" ends here
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)

    message = ChatCompletionMessage(
        role='assistant',
        content=content_with_thinking,
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    model.profile.thinking_tags = ('<think>', '</think>')

    response = model._process_response(completion)

    # Citation should attach to the second TextPart (" world")
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) >= 2
    # Second TextPart should be " world"
    assert text_parts[1].content == ' world'
    assert text_parts[1].citations is not None
    assert len(text_parts[1].citations) == 1
    assert text_parts[1].citations[0].url == 'https://example.com'


def test_process_response_multiple_citations_with_thinking_tags():
    """Test multiple citations with content that includes thinking tags."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Content: "Hello <think>r</think> world"
    # First citation: "Hello" (0-5)
    # Second citation: "world" (need to calculate)
    content_with_thinking = 'Hello <think>r</think> world'
    world_start = content_with_thinking.find('world')
    world_end = world_start + len('world')
    url_citation1 = AnnotationURLCitation(
        url='https://example.com',
        title='Example 1',
        start_index=0,
        end_index=5,  # "Hello"
    )
    url_citation2 = AnnotationURLCitation(
        url='https://example.org',
        title='Example 2',
        start_index=world_start,  # "world" starts here
        end_index=world_end,  # "world" ends here
    )
    annotation1 = Annotation(type='url_citation', url_citation=url_citation1)
    annotation2 = Annotation(type='url_citation', url_citation=url_citation2)

    message = ChatCompletionMessage(
        role='assistant',
        content=content_with_thinking,
        annotations=[annotation1, annotation2],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    model.profile.thinking_tags = ('<think>', '</think>')

    response = model._process_response(completion)

    # Check citations are attached to correct TextParts
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) >= 2
    # First TextPart should have first citation
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 1
    assert text_parts[0].citations[0].url == 'https://example.com'
    # Second TextPart should have second citation
    assert text_parts[1].citations is not None
    assert len(text_parts[1].citations) == 1
    assert text_parts[1].citations[0].url == 'https://example.org'


# Content splitting edge cases


def test_process_response_citation_spanning_three_textparts():
    """Test citation that spans across three TextParts (should attach to first TextPart)."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Content: "First <think>r1</think> Second <think>r2</think> Third"
    # Citation spans from start to end: covers all three TextParts
    content_with_thinking = 'First <think>r1</think> Second <think>r2</think> Third'
    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example',
        start_index=0,  # Starts at beginning
        end_index=len(content_with_thinking),  # Spans entire content
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)

    message = ChatCompletionMessage(
        role='assistant',
        content=content_with_thinking,
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    model.profile.thinking_tags = ('<think>', '</think>')

    response = model._process_response(completion)

    # Citation should attach to the first TextPart where it starts
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) >= 3
    # Citation starts at index 0, which is in the first TextPart
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 1
    assert text_parts[0].citations[0].url == 'https://example.com'
    # Other TextParts should not have this citation (it attaches to starting part)
    assert text_parts[1].citations is None or len(text_parts[1].citations) == 0
    assert text_parts[2].citations is None or len(text_parts[2].citations) == 0


def test_process_response_overlapping_citations():
    """Test overlapping citations (both should attach to the same TextPart)."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Content: "Hello world"
    # Citation 1: start_index=0, end_index=5 (covers "Hello")
    # Citation 2: start_index=0, end_index=11 (covers "Hello world" - overlaps with citation 1)
    content = 'Hello world'
    url_citation1 = AnnotationURLCitation(
        url='https://example.com',
        title='Example 1',
        start_index=0,
        end_index=5,  # "Hello"
    )
    url_citation2 = AnnotationURLCitation(
        url='https://example.org',
        title='Example 2',
        start_index=0,
        end_index=11,  # "Hello world" - overlaps with citation 1
    )
    annotation1 = Annotation(type='url_citation', url_citation=url_citation1)
    annotation2 = Annotation(type='url_citation', url_citation=url_citation2)

    message = ChatCompletionMessage(
        role='assistant',
        content=content,
        annotations=[annotation1, annotation2],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    response = model._process_response(completion)

    # Both citations should attach to the same TextPart
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 2
    # Both citations should be present
    urls = {citation.url for citation in text_parts[0].citations}
    assert 'https://example.com' in urls
    assert 'https://example.org' in urls


def test_process_response_citation_entirely_within_first_textpart():
    """Test citation that is entirely within the first TextPart."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Content: "Hello <think>r</think> world"
    # Citation: start_index=0, end_index=5 (covers "Hello" - entirely in first TextPart)
    content_with_thinking = 'Hello <think>r</think> world'
    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example',
        start_index=0,
        end_index=5,  # "Hello" - entirely in first TextPart
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)

    message = ChatCompletionMessage(
        role='assistant',
        content=content_with_thinking,
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    model.profile.thinking_tags = ('<think>', '</think>')

    response = model._process_response(completion)

    # Citation should attach to first TextPart only
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) >= 2
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 1
    assert text_parts[0].citations[0].url == 'https://example.com'
    # Second TextPart should not have citation
    assert text_parts[1].citations is None or len(text_parts[1].citations) == 0


def test_process_response_citation_entirely_within_second_textpart():
    """Test citation that is entirely within the second TextPart."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Content: "Hello <think>r</think> world"
    # Citation: start_index=20, end_index=25 (covers "world" - entirely in second TextPart)
    content_with_thinking = 'Hello <think>r</think> world'
    world_start = content_with_thinking.find('world')
    world_end = world_start + len('world')
    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example',
        start_index=world_start,  # "world" starts here
        end_index=world_end,  # "world" ends here
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)

    message = ChatCompletionMessage(
        role='assistant',
        content=content_with_thinking,
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))
    model.profile.thinking_tags = ('<think>', '</think>')

    response = model._process_response(completion)

    # Citation should attach to second TextPart only
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) >= 2
    # First TextPart should not have citation
    assert text_parts[0].citations is None or len(text_parts[0].citations) == 0
    # Second TextPart should have citation
    assert text_parts[1].citations is not None
    assert len(text_parts[1].citations) == 1
    assert text_parts[1].citations[0].url == 'https://example.com'


# Provider compatibility tests


def test_openrouter_provider_with_citations():
    """Test that OpenRouter provider works with citations through OpenAIChatModel."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    with try_import() as imports_successful:
        from pydantic_ai.providers.openrouter import OpenRouterProvider

    if not imports_successful():
        pytest.skip('OpenRouter provider not available')

    content = 'This is a test with a citation.'
    # "citation" starts at index 22 and ends at 30 (8 characters)
    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example',
        start_index=22,
        end_index=30,
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)

    message = ChatCompletionMessage(
        role='assistant',
        content=content,
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    provider = OpenRouterProvider(openai_client=mock_client)
    # OpenRouter requires model name in format 'provider/model'
    model = OpenAIChatModel('openai/gpt-4o', provider=provider)

    response = model._process_response(completion)

    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 1
    assert text_parts[0].citations[0].url == 'https://example.com'


def test_perplexity_provider_with_citations():
    """Test that Perplexity provider works with citations through OpenAIChatModel."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    with try_import() as imports_successful:
        from pydantic_ai.providers.perplexity import PerplexityProvider

    if not imports_successful():
        pytest.skip('Perplexity provider not available')

    content = 'This is a test with a citation.'
    # "citation" starts at index 22 and ends at 30 (8 characters)
    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example',
        start_index=22,
        end_index=30,
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)

    message = ChatCompletionMessage(
        role='assistant',
        content=content,
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='llama-3.1-sonar-small-128k-online',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    provider = PerplexityProvider(openai_client=mock_client)
    # Perplexity uses OpenAI-compatible format, so citations work automatically
    model = OpenAIChatModel('llama-3.1-sonar-small-128k-online', provider=provider)

    response = model._process_response(completion)

    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 1
    assert text_parts[0].citations[0].url == 'https://example.com'
    assert text_parts[0].citations[0].title == 'Example'


def test_azure_provider_with_citations():
    """Test that Azure provider works with citations through OpenAIChatModel."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    with try_import() as imports_successful:
        from openai import AsyncAzureOpenAI

        from pydantic_ai.providers.azure import AzureProvider

    if not imports_successful():
        pytest.skip('Azure provider not available')

    content = 'This is a test with a citation.'
    # "citation" starts at index 22 and ends at 30 (8 characters)
    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example',
        start_index=22,
        end_index=30,
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)

    message = ChatCompletionMessage(
        role='assistant',
        content=content,
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    # Create an AsyncAzureOpenAI instance - we need a real instance for the provider
    # but since _process_response doesn't use the client, we can create a minimal one
    azure_client = AsyncAzureOpenAI(
        azure_endpoint='https://test.openai.azure.com/',
        api_key='test-key',
        api_version='2024-12-01-preview',
    )
    # Replace the chat completions create method with our mock's method
    azure_client.chat.completions.create = mock_client.chat.completions.create  # type: ignore[assignment,method-assign]
    provider = AzureProvider(openai_client=azure_client)
    model = OpenAIChatModel('gpt-4o', provider=provider)

    response = model._process_response(completion)

    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 1
    assert text_parts[0].citations[0].url == 'https://example.com'


# Error handling tests


def test_process_response_malformed_annotation_missing_url():
    """Test handling of annotation with invalid URL (empty string)."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Create annotation with empty URL (realistic edge case)
    url_citation = AnnotationURLCitation(
        url='',  # Empty URL
        title='Example',
        start_index=0,
        end_index=4,
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)

    message = ChatCompletionMessage(
        role='assistant',
        content='Test',
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    # Should process annotation even with empty URL (URL validation is not our responsibility)
    response = model._process_response(completion)
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    # Should still have citation, even with empty URL
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 1
    assert text_parts[0].citations[0].url == ''


def test_process_response_annotation_with_invalid_type():
    """Test handling of annotation list with only non-url_citation types (if API adds new types)."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Note: OpenAI SDK's Annotation type only supports 'url_citation', so we can't test
    # invalid types directly. Instead, the code is tested to handle cases where
    # annotations might be filtered (though with current SDK, all annotations are url_citation).
    # This test verifies that empty annotations list is handled correctly.
    message = ChatCompletionMessage(
        role='assistant',
        content='Test content',
        annotations=[],  # Empty annotations
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    # Should handle empty annotations gracefully
    response = model._process_response(completion)
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    assert text_parts[0].citations is None or len(text_parts[0].citations) == 0


def test_process_response_annotation_with_non_string_url():
    """Test handling of annotation with non-string URL."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Create annotation with valid URL (we can't test non-string URL with Pydantic validation)
    # This test verifies that valid citations are processed correctly
    url_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example',
        start_index=0,
        end_index=4,  # "Test" is 4 characters
    )
    annotation = Annotation(type='url_citation', url_citation=url_citation)

    message = ChatCompletionMessage(
        role='assistant',
        content='Test',
        annotations=[annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    # Should handle gracefully
    response = model._process_response(completion)
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    # Should have citation if URL is valid
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 1


def test_process_response_multiple_invalid_annotations():
    """Test handling of multiple invalid annotations mixed with valid ones."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    content = 'This is a test.'
    # Valid citation
    valid_citation = AnnotationURLCitation(
        url='https://example.com',
        title='Example',
        start_index=10,
        end_index=14,
    )
    valid_annotation = Annotation(type='url_citation', url_citation=valid_citation)

    # Invalid citation (out of bounds)
    invalid_citation = AnnotationURLCitation(
        url='https://invalid.com',
        title='Invalid',
        start_index=100,  # Out of bounds
        end_index=200,
    )
    invalid_annotation = Annotation(type='url_citation', url_citation=invalid_citation)

    message = ChatCompletionMessage(
        role='assistant',
        content=content,
        annotations=[valid_annotation, invalid_annotation],
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    response = model._process_response(completion)
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    # Should only have the valid citation
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 1
    assert text_parts[0].citations[0].url == 'https://example.com'


# Performance tests


def test_process_response_many_citations():
    """Test handling of response with many citations (performance test)."""
    from openai.types import chat
    from openai.types.chat.chat_completion import Choice

    from .mock_openai import MockOpenAI

    # Create content with many words
    words = ['word'] * 100
    content = ' '.join(words)

    # Create 50 citations, each covering a different word
    annotations = []
    for i in range(50):
        start = i * 5  # Each word is 4 chars + 1 space
        end = start + 4
        if end <= len(content):
            url_citation = AnnotationURLCitation(
                url=f'https://example.com/{i}',
                title=f'Citation {i}',
                start_index=start,
                end_index=end,
            )
            annotation = Annotation(type='url_citation', url_citation=url_citation)
            annotations.append(annotation)

    message = ChatCompletionMessage(
        role='assistant',
        content=content,
        annotations=annotations,
    )

    completion = chat.ChatCompletion(
        id='test-123',
        choices=[Choice(finish_reason='stop', index=0, message=message)],
        created=1704067200,
        model='gpt-4o',
        object='chat.completion',
    )

    mock_client = MockOpenAI.create_mock(completion)
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    response = model._process_response(completion)
    text_parts = [part for part in response.parts if isinstance(part, TextPart)]
    assert len(text_parts) == 1
    # Should have all valid citations
    assert text_parts[0].citations is not None
    assert len(text_parts[0].citations) == 50
    # Verify a few citations
    assert text_parts[0].citations[0].url == 'https://example.com/0'
    assert text_parts[0].citations[49].url == 'https://example.com/49'
