from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock

import pytest

import pydantic_ai.images as images_module
from pydantic_ai import (
    BinaryImage,
    GeneratedImage,
    ImageGenerationInput,
    ImageGenerationResult,
    ImageGenerator,
    ImageUrl,
    UploadedFile,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
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


async def test_image_generator_forwards_reference_images():
    test_model = TestImageGenerationModel()
    generator = ImageGenerator(test_model)
    images = (
        ImageUrl('https://example.com/reference.png'),
        BinaryImage(data=TINY_PNG, media_type='image/png'),
        UploadedFile(file_id='file-reference', provider_name='openai', media_type='image/webp'),
    )

    await generator.generate('edit these images', images=images)

    assert test_model.last_images == list(images)


async def test_image_generator_override():
    default_model = TestImageGenerationModel(model_name='default')
    override_model = TestImageGenerationModel(model_name='override')
    generator = ImageGenerator(default_model)

    with generator.override(model=override_model):
        result = await generator.generate('tiny robot')
        assert result.model_name == 'override'

    with generator.override():
        result = await generator.generate('tiny robot')
        assert result.model_name == 'default'


async def test_image_generator_eager_and_deferred_model_inference(monkeypatch: pytest.MonkeyPatch):
    resolved_model = TestImageGenerationModel(model_name='resolved')
    inferred_models: list[object] = []

    def infer_model(model: object) -> TestImageGenerationModel:
        if not isinstance(model, TestImageGenerationModel):
            inferred_models.append(model)
        return resolved_model

    monkeypatch.setattr(images_module, 'infer_image_generation_model', infer_model)

    eager_generator = ImageGenerator('test:eager', defer_model_check=False)
    assert eager_generator.model is resolved_model
    assert inferred_models == ['test:eager']

    inferred_models.clear()
    deferred_generator = ImageGenerator('test:deferred')
    assert deferred_generator.model == 'test:deferred'
    assert inferred_models == []

    result = await deferred_generator.generate('tiny robot')
    assert result.model_name == 'resolved'
    assert deferred_generator.model is resolved_model
    assert inferred_models == ['test:deferred']


def test_image_generator_sync_forwards_reference_images():
    test_model = TestImageGenerationModel()
    generator = ImageGenerator(test_model)
    image = BinaryImage(data=TINY_PNG, media_type='image/png')

    generator.generate_sync('edit this image', images=[image])

    assert test_model.last_images == [image]


async def test_image_generation_requires_non_empty_prompt():
    with pytest.raises(UserError, match='non-empty prompt'):
        await TestImageGenerationModel().generate('  ')


async def test_image_generation_rejects_non_image_uploaded_file():
    document = UploadedFile(file_id='file-document', provider_name='openai', media_type='application/pdf')

    with pytest.raises(UserError, match='must have an image media type'):
        await TestImageGenerationModel().generate('edit this image', images=[document])


async def test_image_generation_rejects_invalid_input_type():
    invalid_input = cast(ImageGenerationInput, object())

    with pytest.raises(UserError, match='must be `ImageUrl`, `BinaryImage`, or `UploadedFile`'):
        await TestImageGenerationModel().generate('edit this image', images=[invalid_input])


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


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_rejects_reference_images_before_request():
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    with pytest.raises(UserError, match='Reference images are not supported'):
        await model.generate('edit this image', images=[BinaryImage(data=TINY_PNG, media_type='image/png')])

    mock_client.images.generate.assert_not_awaited()


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation(capfire: CaptureLogfire):
    reference_url = 'https://example.com/private-reference.png'
    generator = ImageGenerator(TestImageGenerationModel(), instrument=True)
    await generator.generate(
        'tiny robot',
        images=[ImageUrl(reference_url), BinaryImage(data=TINY_PNG, media_type='image/png')],
        settings={'n': 1},
    )

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
                'input_image_count': 2,
                'image_generation_settings': {'n': 1},
                'prompt': 'tiny robot',
                'logfire.json_schema': {
                    'type': 'object',
                    'properties': {
                        'prompt_length': {'type': 'integer'},
                        'input_image_count': {'type': 'integer'},
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
    assert reference_url not in str(span)


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrument_all(capfire: CaptureLogfire):
    generator = ImageGenerator(TestImageGenerationModel())
    ImageGenerator.instrument_all()
    try:
        await generator.generate('instrumented globally')
    finally:
        ImageGenerator.instrument_all(False)

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    image_generation_spans = [span for span in spans if 'image_generation' in span['name']]
    assert len(image_generation_spans) == 1
    assert image_generation_spans[0]['attributes']['input_image_count'] == 0
    assert image_generation_spans[0]['attributes']['prompt'] == 'instrumented globally'

    await generator.generate('not instrumented globally')
    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    assert len([span for span in spans if 'image_generation' in span['name']]) == 1


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
