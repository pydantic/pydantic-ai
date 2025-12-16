"""Tests for citations in OpenTelemetry events."""

from __future__ import annotations as _annotations

from pydantic_ai import GroundingCitation, TextPart, ToolResultCitation, URLCitation, usage
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.instrumented import InstrumentationSettings


def test_otel_events_include_url_citation():
    """URLCitation is included in OTEL events."""
    url_citation = URLCitation(url='https://example.com', title='Example', start_index=0, end_index=5)
    text_part = TextPart(content='Hello', citations=[url_citation])
    response = ModelResponse(
        parts=[text_part],
        model_name='test',
        usage=usage.RequestUsage(input_tokens=10, output_tokens=5),
    )

    settings = InstrumentationSettings(include_content=True)
    events = response.otel_events(settings)

    assert len(events) == 1
    event_body = events[0].body
    assert 'content' in event_body

    content = event_body['content']
    # Content should be a list when citations are present
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]['kind'] == 'text'
    assert 'citations' in content[0]
    assert len(content[0]['citations']) == 1
    assert content[0]['citations'][0]['type'] == 'URLCitation'
    assert content[0]['citations'][0]['url'] == 'https://example.com'
    assert content[0]['citations'][0]['title'] == 'Example'


def test_otel_events_include_tool_result_citation():
    """ToolResultCitation is included in OTEL events."""
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

    settings = InstrumentationSettings(include_content=True)
    events = response.otel_events(settings)

    assert len(events) == 1
    content = events[0].body['content']
    assert isinstance(content, list)
    assert 'citations' in content[0]
    assert content[0]['citations'][0]['type'] == 'ToolResultCitation'
    assert content[0]['citations'][0]['tool_name'] == 'web_search'
    assert content[0]['citations'][0]['citation_data']['url'] == 'https://example.com'


def test_otel_events_include_grounding_citation():
    """GroundingCitation is included in OTEL events."""
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

    settings = InstrumentationSettings(include_content=True)
    events = response.otel_events(settings)

    assert len(events) == 1
    content = events[0].body['content']
    assert isinstance(content, list)
    assert 'citations' in content[0]
    assert content[0]['citations'][0]['type'] == 'GroundingCitation'
    assert 'citation_metadata' in content[0]['citations'][0]


def test_otel_events_without_citations():
    """OTEL events work without citations."""
    text_part = TextPart(content='Hello')
    response = ModelResponse(
        parts=[text_part],
        model_name='test',
        usage=usage.RequestUsage(input_tokens=10, output_tokens=5),
    )

    settings = InstrumentationSettings(include_content=True)
    events = response.otel_events(settings)

    assert len(events) == 1
    content = events[0].body['content']
    # Without citations, content should be simplified to just the text string
    assert content == 'Hello'


def test_otel_message_parts_include_citations():
    """Citations are included in OTEL message parts."""
    url_citation = URLCitation(url='https://example.com', title='Example', start_index=0, end_index=5)
    text_part = TextPart(content='Hello', citations=[url_citation])
    response = ModelResponse(
        parts=[text_part],
        model_name='test',
        usage=usage.RequestUsage(input_tokens=10, output_tokens=5),
    )

    settings = InstrumentationSettings(include_content=True)
    parts = response.otel_message_parts(settings)

    assert len(parts) == 1
    assert parts[0]['type'] == 'text'
    assert 'citations' in parts[0]  # type: ignore[typeddict-item]
    assert len(parts[0]['citations']) == 1  # type: ignore[typeddict-item]
    assert parts[0]['citations'][0]['type'] == 'URLCitation'  # type: ignore[typeddict-item]


def test_otel_message_parts_without_citations():
    """OTEL message parts work without citations."""
    text_part = TextPart(content='Hello')
    response = ModelResponse(
        parts=[text_part],
        model_name='test',
        usage=usage.RequestUsage(input_tokens=10, output_tokens=5),
    )

    settings = InstrumentationSettings(include_content=True)
    parts = response.otel_message_parts(settings)

    assert len(parts) == 1
    assert parts[0]['type'] == 'text'
    assert 'citations' not in parts[0] or parts[0].get('citations') is None
