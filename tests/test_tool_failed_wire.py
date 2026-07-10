"""Provider-wire framing tests for failed tool returns.

These tests map messages directly because VCR cassette matching can be insensitive to request
bodies. Pinning the mapped payload proves the failure signal reaches each provider wire format.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Literal

import pytest

from pydantic_ai.messages import BinaryContent, ImageUrl, ModelRequest, ToolReturnPart
from pydantic_ai.models import ModelRequestParameters

from .conftest import try_import

with try_import() as anthropic_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as bedrock_imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.providers.bedrock import BedrockProvider

with try_import() as cohere_imports_successful:
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

with try_import() as groq_imports_successful:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as huggingface_imports_successful:
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

with try_import() as mistral_imports_successful:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

with try_import() as openai_imports_successful:
    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as xai_imports_successful:
    from google.protobuf.json_format import MessageToDict

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider

pytestmark = pytest.mark.anyio

_TOOL_CONTENT = 'Disk full'
_FAILED_WIRE_CONTENT = '{"error":"Disk full"}'

WireMapper = Callable[[ToolReturnPart], Awaitable[object]]


async def _map_openai_chat(part: ToolReturnPart) -> object:
    model = OpenAIChatModel('gpt-5', provider=OpenAIProvider(api_key='test-key'))
    return await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[part])], ModelRequestParameters()
    )


async def _map_openai_responses(part: ToolReturnPart) -> object:
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='test-key'))
    _, messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[part])], OpenAIResponsesModelSettings(), ModelRequestParameters()
    )
    return messages


async def _map_groq(part: ToolReturnPart) -> object:
    model = GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(api_key='test-key'))
    return await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[part])], ModelRequestParameters()
    )


async def _map_mistral(part: ToolReturnPart) -> object:
    model = MistralModel('mistral-large-latest', provider=MistralProvider(api_key='test-key'))
    messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[part])], ModelRequestParameters()
    )
    return [message.model_dump() for message in messages]


async def _map_xai(part: ToolReturnPart) -> object:
    model = XaiModel('grok-4-fast-non-reasoning', provider=XaiProvider(api_key='test-key'))
    messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[part])], ModelRequestParameters()
    )
    return [MessageToDict(message, preserving_proto_field_name=True) for message in messages]


async def _map_huggingface(part: ToolReturnPart) -> object:
    model = HuggingFaceModel('hf-model', provider=HuggingFaceProvider(api_key='test-key'))
    messages = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[part])], ModelRequestParameters()
    )
    return [{key: value for key, value in asdict(message).items() if value is not None} for message in messages]


async def _map_cohere(part: ToolReturnPart) -> object:
    model = CohereModel('command-r7b-12-2024', provider=CohereProvider(api_key='test-key'))
    messages = model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[part])], ModelRequestParameters()
    )
    return [getattr(message, 'content', None) for message in messages]


@dataclass(frozen=True)
class ChannelLessCase:
    id: str
    mapper: WireMapper
    success_wire: object
    failed_wire: object
    marks: tuple[pytest.MarkDecorator, ...] = ()


_CHAT_SUCCESS_WIRE = [{'role': 'tool', 'tool_call_id': 'call_1', 'content': _TOOL_CONTENT}]
_CHAT_FAILED_WIRE = [{'role': 'tool', 'tool_call_id': 'call_1', 'content': _FAILED_WIRE_CONTENT}]

_CHANNEL_LESS_CASES = [
    ChannelLessCase(
        id='openai-chat',
        mapper=_map_openai_chat,
        success_wire=_CHAT_SUCCESS_WIRE,
        failed_wire=_CHAT_FAILED_WIRE,
        marks=(pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed'),),
    ),
    ChannelLessCase(
        id='openai-responses',
        mapper=_map_openai_responses,
        success_wire=[{'type': 'function_call_output', 'call_id': 'call_1', 'output': _TOOL_CONTENT}],
        failed_wire=[{'type': 'function_call_output', 'call_id': 'call_1', 'output': _FAILED_WIRE_CONTENT}],
        marks=(pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed'),),
    ),
    ChannelLessCase(
        id='groq',
        mapper=_map_groq,
        success_wire=_CHAT_SUCCESS_WIRE,
        failed_wire=_CHAT_FAILED_WIRE,
        marks=(pytest.mark.skipif(not groq_imports_successful(), reason='groq not installed'),),
    ),
    ChannelLessCase(
        id='mistral',
        mapper=_map_mistral,
        success_wire=_CHAT_SUCCESS_WIRE,
        failed_wire=_CHAT_FAILED_WIRE,
        marks=(pytest.mark.skipif(not mistral_imports_successful(), reason='mistral not installed'),),
    ),
    ChannelLessCase(
        id='xai',
        mapper=_map_xai,
        success_wire=[
            {'content': [{'text': _TOOL_CONTENT}], 'role': 'ROLE_TOOL', 'tool_call_id': 'call_1'},
        ],
        failed_wire=[
            {'content': [{'text': _FAILED_WIRE_CONTENT}], 'role': 'ROLE_TOOL', 'tool_call_id': 'call_1'},
        ],
        marks=(pytest.mark.skipif(not xai_imports_successful(), reason='xai-sdk not installed'),),
    ),
    ChannelLessCase(
        id='huggingface',
        mapper=_map_huggingface,
        success_wire=_CHAT_SUCCESS_WIRE,
        failed_wire=_CHAT_FAILED_WIRE,
        marks=(pytest.mark.skipif(not huggingface_imports_successful(), reason='huggingface-hub not installed'),),
    ),
    ChannelLessCase(
        id='cohere',
        mapper=_map_cohere,
        success_wire=[_TOOL_CONTENT],
        failed_wire=[_FAILED_WIRE_CONTENT],
        marks=(pytest.mark.skipif(not cohere_imports_successful(), reason='cohere not installed'),),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id, marks=case.marks) for case in _CHANNEL_LESS_CASES])
@pytest.mark.parametrize('outcome', ['success', 'failed', 'denied'])
async def test_channel_less_tool_return_framing(
    case: ChannelLessCase, outcome: Literal['success', 'failed', 'denied']
) -> None:
    """Direct mapping pins request content that a VCR cassette could fail to distinguish."""
    part = ToolReturnPart(
        tool_name='tool',
        content=_TOOL_CONTENT,
        tool_call_id='call_1',
        outcome=outcome,
    )

    wire = await case.mapper(part)

    assert wire == (case.failed_wire if outcome == 'failed' else case.success_wire)
    assert part.content == _TOOL_CONTENT


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_chat_failed_tool_return_keeps_files_out_of_error() -> None:
    """Direct mapping proves shared text-plus-files framing without relying on VCR body matching."""
    image = ImageUrl(url='https://example.com/image.png', identifier='report')
    part = ToolReturnPart(
        tool_name='tool',
        content=[_TOOL_CONTENT, image],
        tool_call_id='call_1',
        outcome='failed',
    )

    wire = await _map_openai_chat(part)

    assert wire == [
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': _FAILED_WIRE_CONTENT},
        {
            'role': 'user',
            'content': [
                {'text': 'This is file report:', 'type': 'text'},
                {'image_url': {'url': 'https://example.com/image.png'}, 'type': 'image_url'},
            ],
        },
    ]
    assert part.content == [_TOOL_CONTENT, image]


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_responses_failed_tool_return_keeps_files_out_of_error() -> None:
    """Direct mapping covers Responses' separate multimodal list-building path, which VCR may miss."""
    image = ImageUrl(url='https://example.com/image.png', identifier='report')
    part = ToolReturnPart(
        tool_name='tool',
        content=[_TOOL_CONTENT, image],
        tool_call_id='call_1',
        outcome='failed',
    )

    wire = await _map_openai_responses(part)

    assert wire == [
        {
            'type': 'function_call_output',
            'call_id': 'call_1',
            'output': [
                {'type': 'input_text', 'text': _FAILED_WIRE_CONTENT},
                {
                    'image_url': 'https://example.com/image.png',
                    'type': 'input_image',
                    'detail': 'auto',
                },
            ],
        }
    ]
    assert part.content == [_TOOL_CONTENT, image]


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_failed_tool_return_does_not_sniff_error_key() -> None:
    """Direct mapping pins nested framing so legitimate `error` output is never mistaken for prior wrapping."""
    content = {'error': 'legitimate output'}
    success = ToolReturnPart(tool_name='tool', content=content, tool_call_id='call_1')
    failed = ToolReturnPart(tool_name='tool', content=content, tool_call_id='call_1', outcome='failed')

    success_wire = await _map_openai_chat(success)
    failed_wire = await _map_openai_chat(failed)

    assert success_wire == [{'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"error":"legitimate output"}'}]
    assert failed_wire == [
        {
            'role': 'tool',
            'tool_call_id': 'call_1',
            'content': '{"error":"{\\"error\\":\\"legitimate output\\"}"}',
        }
    ]
    assert success.content is content
    assert failed.content is content


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_anthropic_failed_tool_return_uses_native_error_channel() -> None:
    """Direct mapping guards Anthropic's native signal and raw content where VCR matching may be lax."""
    model = AnthropicModel('claude-haiku-4-5', provider=AnthropicProvider(api_key='test-key'))
    part = ToolReturnPart(
        tool_name='tool',
        content=_TOOL_CONTENT,
        tool_call_id='call_1',
        outcome='failed',
    )

    _, wire = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[part])], ModelRequestParameters(), AnthropicModelSettings()
    )

    assert wire == [
        {
            'role': 'user',
            'content': [
                {
                    'tool_use_id': 'call_1',
                    'type': 'tool_result',
                    'content': [{'text': _TOOL_CONTENT, 'type': 'text'}],
                    'is_error': True,
                }
            ],
        }
    ]


@pytest.mark.skipif(not bedrock_imports_successful(), reason='boto3 not installed')
@pytest.mark.parametrize(
    'model_name,outcome,expected_tool_result',
    [
        pytest.param(
            'us.amazon.nova-micro-v1:0',
            'failed',
            {
                'toolUseId': 'call_1',
                'content': [{'text': _TOOL_CONTENT}],
                'status': 'error',
            },
            id='native-status',
        ),
        pytest.param(
            'us.writer.palmyra-x4-v1:0',
            'failed',
            {'toolUseId': 'call_1', 'content': [{'text': _FAILED_WIRE_CONTENT}]},
            id='framed-fallback',
        ),
        pytest.param(
            'us.writer.palmyra-x4-v1:0',
            'success',
            {'toolUseId': 'call_1', 'content': [{'text': _TOOL_CONTENT}]},
            id='fallback-success',
        ),
        pytest.param(
            'us.writer.palmyra-x4-v1:0',
            'denied',
            {'toolUseId': 'call_1', 'content': [{'text': _TOOL_CONTENT}]},
            id='fallback-denied',
        ),
    ],
)
async def test_bedrock_failed_tool_return_signal(
    bedrock_provider: BedrockProvider,
    model_name: str,
    outcome: Literal['success', 'failed', 'denied'],
    expected_tool_result: object,
) -> None:
    """Direct mapping distinguishes native status from the no-status fallback despite VCR matching."""
    model = BedrockConverseModel(model_name, provider=bedrock_provider)
    part = ToolReturnPart(
        tool_name='tool',
        content=_TOOL_CONTENT,
        tool_call_id='call_1',
        outcome=outcome,
    )

    _, wire = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[part])], ModelRequestParameters(), BedrockModelSettings()
    )

    assert wire == [
        {
            'role': 'user',
            'content': [{'toolResult': expected_tool_result}],
        }
    ]


@pytest.mark.skipif(not bedrock_imports_successful(), reason='boto3 not installed')
async def test_bedrock_framed_failure_preserves_file_content(bedrock_provider: BedrockProvider) -> None:
    """Direct mapping proves the no-status fallback keeps files without polluting its framed error."""
    image = BinaryContent(data=b'fake-image', media_type='image/png', identifier='report')
    model = BedrockConverseModel('us.writer.palmyra-x4-v1:0', provider=bedrock_provider)
    part = ToolReturnPart(
        tool_name='tool',
        content=[_TOOL_CONTENT, image],
        tool_call_id='call_1',
        outcome='failed',
    )

    _, wire = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[part])], ModelRequestParameters(), BedrockModelSettings()
    )

    assert wire == [
        {
            'role': 'user',
            'content': [
                {
                    'toolResult': {
                        'toolUseId': 'call_1',
                        'content': [
                            {'text': _FAILED_WIRE_CONTENT},
                            {'image': {'format': 'png', 'source': {'bytes': b'fake-image'}}},
                        ],
                    }
                }
            ],
        }
    ]
    assert part.content == [_TOOL_CONTENT, image]


@pytest.mark.skipif(not bedrock_imports_successful(), reason='boto3 not installed')
async def test_bedrock_framed_failure_defers_unsupported_media(bedrock_provider: BedrockProvider) -> None:
    """A media kind unsupported in `toolResult` still rides a later turn; the framed error drops the file ref."""
    pdf = BinaryContent(data=b'%PDF-1.4', media_type='application/pdf', identifier='report')
    model = BedrockConverseModel('us.writer.palmyra-x4-v1:0', provider=bedrock_provider)
    part = ToolReturnPart(tool_name='tool', content=[_TOOL_CONTENT, pdf], tool_call_id='call_1', outcome='failed')

    _, wire = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[part])], ModelRequestParameters(), BedrockModelSettings()
    )

    assert wire == [
        {
            'role': 'user',
            'content': [{'toolResult': {'toolUseId': 'call_1', 'content': [{'text': _FAILED_WIRE_CONTENT}]}}],
        },
        {'role': 'assistant', 'content': [{'text': '.'}]},
        {
            'role': 'user',
            'content': [
                {'text': 'This is file report:'},
                {'document': {'name': 'Document 1', 'format': 'pdf', 'source': {'bytes': b'%PDF-1.4'}}},
            ],
        },
    ]


@pytest.mark.skipif(not bedrock_imports_successful(), reason='boto3 not installed')
async def test_bedrock_framed_failure_drops_structured_content(bedrock_provider: BedrockProvider) -> None:
    """Structured (non-file) failed content is folded into the framed error, not re-sent as a `json` block."""
    model = BedrockConverseModel('us.writer.palmyra-x4-v1:0', provider=bedrock_provider)
    content = {'code': 500, 'detail': 'Disk full'}
    part = ToolReturnPart(tool_name='tool', content=content, tool_call_id='call_1', outcome='failed')

    _, wire = await model._map_messages(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[part])], ModelRequestParameters(), BedrockModelSettings()
    )

    assert wire == [
        {
            'role': 'user',
            'content': [
                {
                    'toolResult': {
                        'toolUseId': 'call_1',
                        'content': [{'text': '{"error":"{\\"code\\":500,\\"detail\\":\\"Disk full\\"}"}'}],
                    }
                }
            ],
        }
    ]
