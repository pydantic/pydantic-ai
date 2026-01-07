"""Unit tests for multimodal tool return functionality.

These tests use FunctionModel (deterministic mock) to verify core functionality
without requiring live API calls or VCR cassettes.

Key behaviors tested:
- ToolReturnPart.multimodal_content and text_or_json_content properties
- Tool returns with BinaryContent and FileUrl types
- Mixed content (JSON data + files in same return)
- Error handling for nested ToolReturn objects
"""

from __future__ import annotations

from datetime import timezone
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AudioUrl,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from ..conftest import IsBytes, IsDatetime, IsNow, IsStr

# =============================================================================
# ToolReturnPart property tests
# =============================================================================


def test_tool_return_part_file_content_methods():
    """Test that ToolReturnPart properly separates files from data content."""
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82'
    binary_content = BinaryContent(png_data, media_type='image/png')

    tool_return = ToolReturnPart(tool_name='test_tool', content=binary_content, tool_call_id='test_call_123')

    # File-only content: text_or_json_content is None, files accessible via multimodal_content
    assert tool_return == snapshot(
        ToolReturnPart(
            tool_name='test_tool',
            content=BinaryContent(
                data=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82',
                media_type='image/png',
            ),
            tool_call_id='test_call_123',
            timestamp=IsDatetime(),
        )
    )
    # Verify the new methods work correctly
    assert tool_return.text_or_json_content is None
    assert tool_return.model_response_object() == {}
    assert tool_return.model_response_str() == ''
    assert tool_return.multimodal_content == [binary_content]


# =============================================================================
# Tool return with BinaryContent tests
# =============================================================================


def test_tool_returning_binary_content_directly():
    """Test that a tool returning BinaryContent directly works correctly."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('get_image', {})])
        else:
            return ModelResponse(parts=[TextPart('Image received')])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def get_image() -> BinaryContent:
        """Return a simple image."""
        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82'
        return BinaryContent(png_data, media_type='image/png')

    result = agent.run_sync('Get an image')
    assert result.output == 'Image received'


def test_tool_returning_binary_content_with_identifier():
    """Test that a tool returning BinaryContent with custom identifier preserves it."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('get_image', {})])
        else:
            return ModelResponse(parts=[TextPart('Image received')])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def get_image() -> BinaryContent:
        """Return a simple image."""
        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82'
        return BinaryContent(png_data, media_type='image/png', identifier='image_id_1')

    result = agent.run_sync('Get an image')
    assert result.all_messages()[2] == snapshot(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='get_image',
                    content=BinaryContent(
                        data=IsBytes(),
                        media_type='image/png',
                        _identifier='image_id_1',
                    ),
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
        )
    )


# =============================================================================
# Tool return with FileUrl tests
# =============================================================================


def test_tool_returning_file_url_with_identifier():
    """Test that a tool returning FileUrl subclasses with identifiers works correctly."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('get_files', {})])
        else:
            return ModelResponse(parts=[TextPart('Files received')])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def get_files():
        """Return various file URLs with custom identifiers."""
        return [
            ImageUrl(url='https://example.com/image.jpg', identifier='img_001'),
            VideoUrl(url='https://example.com/video.mp4', identifier='vid_002'),
            AudioUrl(url='https://example.com/audio.mp3', identifier='aud_003'),
            DocumentUrl(url='https://example.com/document.pdf', identifier='doc_004'),
        ]

    result = agent.run_sync('Get some files')
    assert result.all_messages()[2] == snapshot(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='get_files',
                    content=[
                        ImageUrl(url='https://example.com/image.jpg', _identifier='img_001'),
                        VideoUrl(url='https://example.com/video.mp4', _identifier='vid_002'),
                        AudioUrl(url='https://example.com/audio.mp3', _identifier='aud_003'),
                        DocumentUrl(url='https://example.com/document.pdf', _identifier='doc_004'),
                    ],
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
        )
    )


# =============================================================================
# ToolReturn with content tests
# =============================================================================


def test_multimodal_tool_response():
    """Test ToolReturn with custom content and tool return."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[TextPart('Starting analysis'), ToolCallPart('analyze_data', {})])
        else:
            return ModelResponse(parts=[TextPart('Analysis completed')])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def analyze_data() -> ToolReturn:
        return ToolReturn(
            return_value='Data analysis completed successfully',
            content=[
                'Here are the analysis results:',
                ImageUrl('https://example.com/chart.jpg'),
                'The chart shows positive trends.',
            ],
            metadata={'foo': 'bar'},
        )

    result = agent.run_sync('Please analyze the data')

    assert result.output == 'Analysis completed'

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Please analyze the data', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='Starting analysis'),
                    ToolCallPart(tool_name='analyze_data', args={}, tool_call_id=IsStr()),
                ],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='analyze_data',
                        content='Data analysis completed successfully',
                        tool_call_id=IsStr(),
                        metadata={'foo': 'bar'},
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    UserPromptPart(
                        content=[
                            'Here are the analysis results:',
                            ImageUrl(url='https://example.com/chart.jpg', identifier='672a5c'),
                            'The chart shows positive trends.',
                        ],
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Analysis completed')],
                usage=RequestUsage(input_tokens=70, output_tokens=6),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


def test_multimodal_tool_response_nested():
    """Test ToolReturn with multimodal content directly in `return_value`."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[TextPart('Starting analysis'), ToolCallPart('analyze_data', {})])
        else:
            return ModelResponse(parts=[TextPart('Analysis completed')])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def analyze_data() -> ToolReturn:
        return ToolReturn(
            return_value=ImageUrl('https://example.com/chart.jpg'),
            metadata={'foo': 'bar'},
        )

    result = agent.run_sync('Please analyze the data')
    assert result.output == 'Analysis completed'
    tool_return_part = result.all_messages()[2].parts[0]
    assert tool_return_part == snapshot(
        ToolReturnPart(
            tool_name='analyze_data',
            content=ImageUrl(url='https://example.com/chart.jpg'),
            tool_call_id=IsStr(),
            metadata={'foo': 'bar'},
            timestamp=IsDatetime(),
        )
    )


# =============================================================================
# Mixed content tests
# =============================================================================


def test_tool_return_mixed_list():
    """Test that a tool can return a list of mixed data and files (ToolReturnPart.content as list)."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_calls = [ToolCallPart('get_mixed_content', '{}')]
        if not any(
            isinstance(m, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in m.parts) for m in messages
        ):
            return ModelResponse(parts=tool_calls)
        return ModelResponse(parts=[TextPart('Received mixed content')])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def get_mixed_content() -> list[Any]:
        """Returns a list with mixed data and multimodal content."""
        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        return [
            'Here is some analysis text',
            {'data': 'structured result', 'count': 42},
            BinaryContent(png_data, media_type='image/png'),
            ImageUrl('https://example.com/chart.jpg'),
        ]

    result = agent.run_sync('Get mixed content')
    assert result.output == 'Received mixed content'

    tool_return_part = result.all_messages()[2].parts[0]
    assert isinstance(tool_return_part, ToolReturnPart)
    assert tool_return_part == snapshot(
        ToolReturnPart(
            tool_name='get_mixed_content',
            content=[
                'Here is some analysis text',
                {'data': 'structured result', 'count': 42},
                BinaryContent(data=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR', media_type='image/png'),
                ImageUrl(url='https://example.com/chart.jpg'),
            ],
            tool_call_id=IsStr(),
            timestamp=IsDatetime(),
        )
    )
    # Verify the new methods correctly separate files from data
    assert tool_return_part.multimodal_content == [
        BinaryContent(data=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR', media_type='image/png'),
        ImageUrl(url='https://example.com/chart.jpg'),
    ]
    assert tool_return_part.text_or_json_content == [
        'Here is some analysis text',
        {'data': 'structured result', 'count': 42},
    ]


# =============================================================================
# Error case tests
# =============================================================================


@pytest.mark.anyio
async def test_openai_chat_video_url_raises():
    """Test that OpenAI Chat API raises NotImplementedError for VideoUrl in user prompts.

    OpenAI Chat sends multimodal tool returns via separate user messages,
    so VideoUrl is rejected when mapping the UserPromptPart.
    """
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    user_prompt = UserPromptPart(content=[VideoUrl(url='https://example.com/video.mp4')])

    with pytest.raises(NotImplementedError, match='VideoUrl is not supported for OpenAI'):
        await model._map_user_prompt(user_prompt)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_openai_responses_audio_binary_raises():
    """Test that OpenAI Responses API raises RuntimeError for audio binary in tool returns.

    Note: Audio binary in tool returns falls to the unsupported content type handler.
    The NotImplementedError for audio is only raised for user prompts.
    """
    from pydantic_ai.models.openai import OpenAIResponsesModel

    tool_return = ToolReturnPart(
        tool_name='get_audio',
        content=BinaryContent(data=b'audio data', media_type='audio/mpeg'),
        tool_call_id='test_call',
    )

    with pytest.raises(RuntimeError, match='Unsupported binary content type: audio/mpeg'):
        await OpenAIResponsesModel._map_tool_return_output(tool_return)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_openai_responses_video_url_raises():
    """Test that OpenAI Responses API raises NotImplementedError for VideoUrl in tool returns."""
    from pydantic_ai.models.openai import OpenAIResponsesModel

    tool_return = ToolReturnPart(
        tool_name='get_video',
        content=VideoUrl(url='https://example.com/video.mp4'),
        tool_call_id='test_call',
    )

    with pytest.raises(NotImplementedError, match='VideoUrl is not supported for OpenAI'):
        await OpenAIResponsesModel._map_tool_return_output(tool_return)  # pyright: ignore[reportPrivateUsage]


def test_many_multimodal_tool_response():
    """Test that nested ToolReturn objects in a list raise an error."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[TextPart('Starting analysis'), ToolCallPart('analyze_data', {})])
        else:
            return ModelResponse(  # pragma: no cover
                parts=[TextPart('Analysis completed')]
            )

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def analyze_data() -> list[Any]:
        return [
            ToolReturn(
                return_value='Data analysis completed successfully',
                content=[
                    'Here are the analysis results:',
                    ImageUrl('https://example.com/chart.jpg'),
                    'The chart shows positive trends.',
                ],
                metadata={'foo': 'bar'},
            ),
            'Something else',
        ]

    with pytest.raises(
        UserError,
        match="The return value of tool 'analyze_data' contains invalid nested `ToolReturn` objects. `ToolReturn` should be used directly.",
    ):
        agent.run_sync('Please analyze the data')
