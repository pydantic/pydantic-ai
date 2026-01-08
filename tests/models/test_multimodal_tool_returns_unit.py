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
from pytest_mock import MockerFixture

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
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from ..conftest import IsBytes, IsDatetime, IsNow, IsStr, try_import

with try_import() as openai_imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from tests.models.test_bedrock import _StubBedrockProvider  # pyright: ignore[reportPrivateUsage]

with try_import() as mistral_imports_successful:
    from mistralai.models import (
        AssistantMessage,
        FunctionCall,
        TextChunk,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

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
@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_chat_video_url_raises():
    """Test that OpenAI Chat API raises NotImplementedError for VideoUrl in user prompts.

    OpenAI Chat sends multimodal tool returns via separate user messages,
    so VideoUrl is rejected when mapping the UserPromptPart.
    """
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    user_prompt = UserPromptPart(content=[VideoUrl(url='https://example.com/video.mp4')])

    with pytest.raises(NotImplementedError, match='VideoUrl is not supported for OpenAI'):
        await model._map_user_prompt(user_prompt)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_responses_audio_binary_raises():
    """Test that OpenAI Responses API raises RuntimeError for audio binary in tool returns.

    Note: Audio binary in tool returns falls to the unsupported content type handler.
    The NotImplementedError for audio is only raised for user prompts.
    """
    tool_return = ToolReturnPart(
        tool_name='get_audio',
        content=BinaryContent(data=b'audio data', media_type='audio/mpeg'),
        tool_call_id='test_call',
    )

    with pytest.raises(RuntimeError, match='Unsupported binary content type: audio/mpeg'):
        await OpenAIResponsesModel._map_tool_return_output(tool_return)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_responses_video_url_raises():
    """Test that OpenAI Responses API raises NotImplementedError for VideoUrl in tool returns."""
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


# =============================================================================
# Provider-specific mapping tests
# =============================================================================


@pytest.mark.anyio
@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_image_url_force_download(mocker: MockerFixture):
    """Test that Anthropic handles ImageUrl with force_download=True in tool returns."""
    mocker.patch(
        'pydantic_ai.models.anthropic.download_item',
        return_value={'data': b'fake image bytes', 'data_type': 'image/png'},
    )
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test'))
    messages = [
        ModelRequest(parts=[UserPromptPart(content='Get image')]),
        ModelResponse(parts=[ToolCallPart(tool_name='get_image', args={}, tool_call_id='call1')]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='get_image',
                    content=ImageUrl(url='https://example.com/image.png', force_download=True),
                    tool_call_id='call1',
                )
            ]
        ),
    ]

    _, anthropic_messages = await model._map_message(messages, ModelRequestParameters(), AnthropicModelSettings())  # pyright: ignore[reportPrivateUsage]
    # Not using snapshot() here because Anthropic SDK uses BytesIO objects which don't
    # compare equal after deepcopy (inline-snapshot requirement)
    assert len(anthropic_messages) == 3
    assert anthropic_messages[0] == {'role': 'user', 'content': [{'type': 'text', 'text': 'Get image'}]}
    assert anthropic_messages[1] == {
        'role': 'assistant',
        'content': [{'type': 'tool_use', 'id': 'call1', 'name': 'get_image', 'input': {}}],
    }
    tool_result_msg = anthropic_messages[2]
    assert tool_result_msg['role'] == 'user'
    tool_result: Any = list(tool_result_msg['content'])[0]
    assert tool_result['type'] == 'tool_result'
    assert tool_result['tool_use_id'] == 'call1'
    image_block = tool_result['content'][0]
    assert image_block['type'] == 'image'
    assert image_block['source']['type'] == 'base64'
    assert image_block['source']['media_type'] == 'image/png'
    image_block['source']['data'].seek(0)
    assert image_block['source']['data'].read() == b'fake image bytes'


@pytest.mark.anyio
@pytest.mark.skipif(not bedrock_imports_successful(), reason='bedrock not installed')
async def test_bedrock_s3_url_document():
    """Test that Bedrock handles S3 URLs for documents."""
    s3_url = 's3://my-bucket/path/to/document.pdf?bucketOwner=123456789012'
    doc_count = iter(range(1, 100))

    result = await BedrockConverseModel._map_file_to_content_block(  # pyright: ignore[reportPrivateUsage]
        DocumentUrl(url=s3_url, media_type='application/pdf'), doc_count
    )
    assert result == snapshot(
        {
            'document': {
                'name': 'Document 1',
                'format': 'pdf',
                'source': {'s3Location': {'uri': 's3://my-bucket/path/to/document.pdf', 'bucketOwner': '123456789012'}},
            }
        }
    )


@pytest.mark.anyio
@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_responses_tool_return_with_data_and_image():
    """Test OpenAI Responses API tool return with both text and image content."""
    tool_return = ToolReturnPart(
        tool_name='get_data',
        content=['analysis result', ImageUrl(url='https://example.com/chart.png')],
        tool_call_id='test_call',
    )

    result = await OpenAIResponsesModel._map_tool_return_output(tool_return)  # pyright: ignore[reportPrivateUsage]
    assert result == snapshot(
        [
            {'type': 'input_text', 'text': '["analysis result"]'},
            {'type': 'input_image', 'image_url': 'https://example.com/chart.png', 'detail': 'auto'},
        ]
    )


@pytest.mark.anyio
@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_responses_image_vendor_metadata():
    """Test OpenAI Responses API handles vendor_metadata for images."""
    png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
    binary_image = BinaryContent(data=png_data, media_type='image/png', vendor_metadata={'detail': 'high'})
    tool_return = ToolReturnPart(
        tool_name='get_image',
        content=binary_image,
        tool_call_id='test_call',
    )

    result = await OpenAIResponsesModel._map_tool_return_output(tool_return)  # pyright: ignore[reportPrivateUsage]
    assert result == snapshot(
        [
            {
                'type': 'input_image',
                'image_url': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe',
                'detail': 'high',
            }
        ]
    )

    image_url = ImageUrl(url='https://example.com/image.png', vendor_metadata={'detail': 'low'})
    tool_return2 = ToolReturnPart(
        tool_name='get_image',
        content=image_url,
        tool_call_id='test_call2',
    )

    result2 = await OpenAIResponsesModel._map_tool_return_output(tool_return2)  # pyright: ignore[reportPrivateUsage]
    assert result2 == snapshot([{'type': 'input_image', 'image_url': 'https://example.com/image.png', 'detail': 'low'}])


@pytest.mark.anyio
@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_responses_image_url_force_download(mocker: MockerFixture):
    """Test OpenAI Responses API handles ImageUrl with force_download."""
    mocker.patch(
        'pydantic_ai.models.openai.download_item',
        return_value={'data': 'data:image/png;base64,ZmFrZWltYWdl', 'data_type': 'png'},
    )
    image_url = ImageUrl(url='https://example.com/image.png', force_download=True)
    tool_return = ToolReturnPart(
        tool_name='get_image',
        content=image_url,
        tool_call_id='test_call',
    )

    result = await OpenAIResponsesModel._map_tool_return_output(tool_return)  # pyright: ignore[reportPrivateUsage]
    assert result == snapshot(
        [{'type': 'input_image', 'image_url': 'data:image/png;base64,ZmFrZWltYWdl', 'detail': 'auto'}]
    )


@pytest.mark.anyio
@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_responses_document_url_force_download(mocker: MockerFixture):
    """Test OpenAI Responses API handles DocumentUrl with force_download."""
    mocker.patch(
        'pydantic_ai.models.openai.download_item',
        return_value={'data': 'data:application/pdf;base64,ZmFrZXBkZg==', 'data_type': 'pdf'},
    )
    doc_url = DocumentUrl(url='https://example.com/doc.pdf', force_download=True)
    tool_return = ToolReturnPart(
        tool_name='get_doc',
        content=doc_url,
        tool_call_id='test_call',
    )

    result = await OpenAIResponsesModel._map_tool_return_output(tool_return)  # pyright: ignore[reportPrivateUsage]
    assert result == snapshot(
        [{'type': 'input_file', 'file_data': 'data:application/pdf;base64,ZmFrZXBkZg==', 'filename': 'filename.pdf'}]
    )


@pytest.mark.anyio
@pytest.mark.skipif(not bedrock_imports_successful(), reason='bedrock not installed')
async def test_bedrock_empty_tool_result_json_format():
    """Test Bedrock uses json format for empty tool results with Mistral models.

    When a tool returns only a document (no text/json), the tool_result_content is empty.
    For Mistral models on Bedrock, this should use {'json': {}} instead of {'text': ''}.
    """
    from pydantic_ai.providers.bedrock import BedrockModelProfile

    class _MistralProfileStubProvider(_StubBedrockProvider):
        def model_profile(self, model_name: str):
            return BedrockModelProfile(bedrock_tool_result_format='json')

    model = BedrockConverseModel('mistral.mistral-large-2411-v1:0', provider=_MistralProfileStubProvider(None))  # pyright: ignore[reportArgumentType]
    req = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(parts=[ToolCallPart(tool_name='get_file', args={}, tool_call_id='call1')]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='get_file',
                    content=DocumentUrl(url='s3://bucket/doc.pdf'),
                    tool_call_id='call1',
                )
            ]
        ),
    ]

    _, bedrock_messages = await model._map_messages(req, ModelRequestParameters(), BedrockModelSettings())  # pyright: ignore[reportPrivateUsage]
    assert bedrock_messages == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Hello'}]},
            {'role': 'assistant', 'content': [{'toolUse': {'toolUseId': 'call1', 'name': 'get_file', 'input': {}}}]},
            {
                'role': 'user',
                'content': [
                    {'toolResult': {'toolUseId': 'call1', 'content': [{'json': {}}], 'status': 'success'}},
                    {'text': 'Additional file from tool result:'},
                    {
                        'document': {
                            'name': 'Document 1',
                            'format': 'pdf',
                            'source': {'s3Location': {'uri': 's3://bucket/doc.pdf'}},
                        }
                    },
                ],
            },
        ]
    )


@pytest.mark.anyio
@pytest.mark.skipif(not mistral_imports_successful(), reason='mistral not installed')
async def test_mistral_tool_then_user_sequence():
    """Test Mistral inserts dummy assistant message between tool result and user prompt."""
    model = MistralModel('mistral-large-latest', provider=MistralProvider(api_key='test'))
    messages = [
        ModelRequest(parts=[UserPromptPart(content='Call the tool')]),
        ModelResponse(parts=[ToolCallPart(tool_name='my_tool', args={}, tool_call_id='call1')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='my_tool', content='tool result', tool_call_id='call1')]),
        ModelRequest(parts=[UserPromptPart(content='Now do something else')]),
    ]

    result = await model._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    assert result == snapshot(
        [
            UserMessage(content='Call the tool'),
            AssistantMessage(
                content=[],
                tool_calls=[ToolCall(function=FunctionCall(name='my_tool', arguments={}), id='call1', type='function')],
            ),
            ToolMessage(content='tool result', tool_call_id='call1'),
            AssistantMessage(content=[TextChunk(text='OK')]),
            UserMessage(content='Now do something else'),
        ]
    )
