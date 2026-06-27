from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock

import pytest

from pydantic_ai import BinaryImage, GeneratedImage, ImageGenerationResult, ImageGenerator
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.images import (
    ImageGenerationSettings,
    TestImageGenerationModel,
    infer_image_generation_model,
    merge_image_generation_settings,
)
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInt, IsStr, try_import

pytestmark = [
    pytest.mark.anyio,
]

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire

with try_import() as openai_imports_successful:
    from openai import AsyncOpenAI
    from openai.types.image import Image
    from openai.types.images_response import ImagesResponse, Usage, UsageInputTokensDetails, UsageOutputTokensDetails

    from pydantic_ai.images.openai import OpenAIImageGenerationModel, OpenAIImageGenerationSettings
    from pydantic_ai.providers.openai import OpenAIProvider

TINY_PNG = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
    b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
    b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
)


async def test_image_generator_with_test_model():
    test_model = TestImageGenerationModel()
    generator = ImageGenerator(test_model)

    result = await generator.generate('tiny robot', settings={'n': 2})

    assert result == snapshot(
        ImageGenerationResult(
            images=[
                GeneratedImage(
                    content=BinaryImage(data=TINY_PNG, media_type='image/png'),
                    size='1x1',
                    output_format='png',
                ),
                GeneratedImage(
                    content=BinaryImage(data=TINY_PNG, media_type='image/png'),
                    size='1x1',
                    output_format='png',
                ),
            ],
            prompt='tiny robot',
            model_name='test',
            provider_name='test',
            timestamp=IsDatetime(),
            usage=RequestUsage(input_tokens=2),
            settings={'n': 2},
            provider_response_id=IsStr(),
        )
    )
    assert test_model.last_settings == {'n': 2}


def test_merge_image_generation_settings():
    base: ImageGenerationSettings = {'n': 1, 'extra_body': {'provider_option': True}}
    overrides: ImageGenerationSettings = {'n': 2, 'output_format': 'webp'}

    assert merge_image_generation_settings(base, overrides) == snapshot(
        {'n': 2, 'extra_body': {'provider_option': True}, 'output_format': 'webp'}
    )
    assert merge_image_generation_settings(None, overrides) == overrides
    assert merge_image_generation_settings(base, None) == base


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_infer_openai_image_generation_model():
    model = infer_image_generation_model(
        'openai:gpt-image-1',
        provider_factory=lambda _: OpenAIProvider(openai_client=AsyncOpenAI(api_key='test-api-key')),
    )

    assert isinstance(model, OpenAIImageGenerationModel)
    assert model.model_name == 'gpt-image-1'
    assert model.system == 'openai'
    assert model.base_url == 'https://api.openai.com/v1/'


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_response_mapping():
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    mock_client.images.generate.return_value = ImagesResponse.model_construct(
        created=123,
        background='opaque',
        data=[Image.model_construct(b64_json='aGVsbG8=', revised_prompt='A tiny friendly robot')],
        output_format='png',
        quality='low',
        size='1024x1024',
        usage=Usage.model_construct(
            input_tokens=3,
            input_tokens_details=UsageInputTokensDetails.model_construct(text_tokens=3, image_tokens=0),
            output_tokens=5,
            total_tokens=8,
            output_tokens_details=UsageOutputTokensDetails.model_construct(text_tokens=0, image_tokens=5),
        ),
    )
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    settings = OpenAIImageGenerationSettings(n=1, openai_size='1024x1024', output_format='png', openai_quality='low')
    result = await model.generate('tiny robot', settings=settings)

    assert result == snapshot(
        ImageGenerationResult(
            images=[
                GeneratedImage(
                    content=BinaryImage(data=b'hello', media_type='image/png'),
                    revised_prompt='A tiny friendly robot',
                    size='1024x1024',
                    quality='low',
                    output_format='png',
                    background='opaque',
                )
            ],
            prompt='tiny robot',
            model_name='gpt-image-1',
            provider_name='openai',
            timestamp=IsDatetime(),
            usage=RequestUsage(
                input_tokens=3,
                output_tokens=5,
                details={
                    'input_text_tokens': 3,
                    'input_image_tokens': 0,
                    'output_text_tokens': 0,
                    'output_image_tokens': 5,
                },
            ),
            settings=settings,
            provider_details={'created': 123},
            provider_url='https://api.openai.com/v1/',
        )
    )
    assert 'response_format' not in mock_client.images.generate.await_args.kwargs
    assert mock_client.images.generate.await_args.kwargs['size'] == '1024x1024'


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_response_without_base64():
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    mock_client.images.generate.return_value = ImagesResponse.model_construct(
        created=123, data=[Image.model_construct(url='https://example.com/a.png')]
    )
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    with pytest.raises(UnexpectedModelBehavior, match='base64 image data'):
        await model.generate('tiny robot')


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation(capfire: CaptureLogfire):
    generator = ImageGenerator(TestImageGenerationModel(), instrument=True)
    await generator.generate('tiny robot', settings={'n': 1})

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(span for span in spans if 'image_generation' in span['name'])

    assert span == snapshot(
        {
            'name': 'image_generation test',
            'context': {'trace_id': 1, 'span_id': 1, 'is_remote': False},
            'parent': None,
            'start_time': IsInt(),
            'end_time': IsInt(),
            'attributes': {
                'gen_ai.operation.name': 'image_generation',
                'gen_ai.provider.name': 'test',
                'gen_ai.request.model': 'test',
                'prompt_length': 10,
                'image_generation_settings': {'n': 1},
                'prompt': 'tiny robot',
                'logfire.json_schema': {
                    'type': 'object',
                    'properties': {
                        'prompt_length': {'type': 'integer'},
                        'image_generation_settings': {'type': 'object'},
                        'image_count': {'type': 'integer'},
                        'prompt': {'type': 'string'},
                    },
                },
                'logfire.span_type': 'span',
                'logfire.msg': 'image_generation test',
                'gen_ai.usage.input_tokens': 2,
                'gen_ai.response.model': 'test',
                'image_count': 1,
                'image.0.size': '1x1',
                'image.0.output_format': 'png',
                'image.0.media_type': 'image/png',
                'gen_ai.response.id': IsStr(),
            },
        }
    )
    assert 'aGVsbG8=' not in str(span)


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation_excludes_generated_image_content_when_include_content_false(capfire: CaptureLogfire):
    generator = ImageGenerator(
        TestImageGenerationModel(),
        instrument=InstrumentationSettings(include_content=False, include_binary_content=False),
    )
    await generator.generate('tiny robot')

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(span for span in spans if 'image_generation' in span['name'])
    attributes = span['attributes']

    assert 'prompt' not in attributes
    assert 'image.0.media_type' in attributes
    assert 'data' not in str(span)
