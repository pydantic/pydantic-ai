from __future__ import annotations

import base64
import json
import os
import warnings
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import cast, get_args
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from genai_prices.types import PriceCalculation

import pydantic_ai.images as images_module
import pydantic_ai.images._google_geometry as google_geometry
import pydantic_ai.images._openai_geometry as openai_geometry
from pydantic_ai import (
    BinaryImage,
    GeneratedImage,
    ImageGenerationInput,
    ImageGenerationResult,
    ImageGenerator,
    ImageUrl,
    UploadedFile,
)
from pydantic_ai.exceptions import (
    ContentFilterError,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.images import (
    ImageGenerationSettings,
    InstrumentedImageGenerationModel,
    KnownImageGenerationModelName,
    TestImageGenerationModel,
    WrapperImageGenerationModel,
    infer_image_generation_model,
    merge_image_generation_settings,
)
from pydantic_ai.images.settings import image_generation_tool_settings
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
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI
    from openai.types.image import Image
    from openai.types.images_response import ImagesResponse, Usage, UsageInputTokensDetails, UsageOutputTokensDetails

    import pydantic_ai.images.openai as openai_images
    from pydantic_ai.images.openai import OpenAIImageGenerationModel, OpenAIImageGenerationSettings
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as google_imports_successful:
    from google.genai import Client as GoogleClient, errors as google_errors, types as google_types

    import pydantic_ai.images.google as google_images
    from pydantic_ai.images.google import (
        GoogleImageGenerationModel,
        GoogleImageGenerationSettings,
    )
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as xai_imports_successful:
    import grpc
    from xai_sdk import AsyncClient as XaiAsyncClient
    from xai_sdk.aio.image import ImageResponse as XaiImageResponse
    from xai_sdk.proto import image_pb2 as xai_image_pb2, usage_pb2 as xai_usage_pb2

    import pydantic_ai.images._xai_geometry as xai_geometry
    import pydantic_ai.images.xai as xai_images
    from pydantic_ai.images.xai import XaiImageGenerationModel, XaiImageGenerationSettings
    from pydantic_ai.providers.xai import XaiProvider

with try_import() as image_demo_imports_successful:
    import pydantic_ai.images._demo as images_demo

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


def test_images_module_exports_image_generator():
    assert 'ImageGenerator' in images_module.__all__


async def test_image_generator_settings_precedence():
    test_model = TestImageGenerationModel(settings={'n': 1, 'output_format': 'png', 'quality': 'high'})
    generator = ImageGenerator(test_model, settings={'n': 2, 'quality': 'low'})

    result = await generator.generate('tiny robot', settings={'n': 3, 'output_format': 'webp'})

    expected_settings: ImageGenerationSettings = {'n': 3, 'output_format': 'webp', 'quality': 'low'}
    assert result.settings == expected_settings
    assert test_model.last_settings == expected_settings


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


def test_infer_image_generation_model_requires_provider_prefix():
    with pytest.raises(ValueError, match='provide a provider prefix'):
        infer_image_generation_model('gpt-image-1')


async def test_wrapper_image_generation_model_delegates_properties():
    wrapped = TestImageGenerationModel(settings={'n': 2})
    model = WrapperImageGenerationModel(wrapped)

    result = await model.generate('tiny robot')

    assert result.model_name == 'test'
    assert model.model_name == wrapped.model_name
    assert model.system == wrapped.system
    assert model.settings == {'n': 2}
    assert model.base_url is None


def test_image_generator_sync_forwards_reference_images():
    test_model = TestImageGenerationModel()
    generator = ImageGenerator(test_model)
    image = BinaryImage(data=TINY_PNG, media_type='image/png')

    generator.generate_sync('edit this image', images=[image])

    assert test_model.last_images == [image]


def test_image_generation_cost_is_unavailable():
    result = ImageGenerator(TestImageGenerationModel()).generate_sync('tiny robot')

    with pytest.raises(LookupError, match='until `genai-prices` supports image pricing'):
        result.cost()


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


def test_image_generation_tool_settings_filters_new_geometry():
    filtered, ignored = image_generation_tool_settings(
        {'dimensions': (1024, 1024), 'size': '2048x2048', 'aspect_ratio': '1:2'}
    )

    assert filtered == {}
    assert ignored == ['dimensions', 'size', 'aspect_ratio']

    supported, ignored = image_generation_tool_settings({'size': '1024x1024', 'aspect_ratio': '1:1'})

    assert supported == {'size': '1024x1024', 'aspect_ratio': '1:1'}
    assert ignored == []


@pytest.mark.parametrize(
    'settings',
    [
        {'dimensions': (1024, 1024), 'aspect_ratio': '1:1'},
        {'dimensions': (1024, 1024), 'size': '1024x1024'},
    ],
)
async def test_image_generation_dimensions_are_mutually_exclusive(settings: ImageGenerationSettings):
    with pytest.raises(UserError, match='mutually exclusive'):
        await TestImageGenerationModel().generate('tiny robot', settings=settings)


@pytest.mark.parametrize('n', [0, -1, True, cast(int, '1')])
async def test_image_generation_n_must_be_a_positive_integer(n: int):
    with pytest.raises(UserError, match=r'`n` must be a positive integer'):
        await TestImageGenerationModel().generate('tiny robot', settings={'n': n})


@pytest.mark.parametrize(
    'dimensions',
    [
        (0, 1024),
        (1024, -1),
        cast(tuple[int, int], (1024,)),
        cast(tuple[int, int], (True, 1024)),
        cast(tuple[int, int], [1024, 1024]),
    ],
)
async def test_image_generation_dimensions_must_be_positive_integer_tuple(dimensions: tuple[int, int]):
    with pytest.raises(UserError, match=r'`dimensions` must be a .* tuple of positive integers'):
        await TestImageGenerationModel().generate('tiny robot', settings={'dimensions': dimensions})


def test_known_openai_image_generation_model_names():
    known_names = get_args(KnownImageGenerationModelName.__value__)

    assert {name for name in known_names if name.startswith('openai:')} == {
        'openai:gpt-image-1',
        'openai:gpt-image-1-mini',
        'openai:gpt-image-1.5',
        'openai:gpt-image-2',
    }


def test_known_google_image_generation_model_names():
    known_names = get_args(KnownImageGenerationModelName.__value__)

    assert {name for name in known_names if name.startswith('google:')} == {
        'google:gemini-2.5-flash-image',
        'google:gemini-3-pro-image',
        'google:gemini-3.1-flash-image',
        'google:gemini-3.1-flash-lite-image',
    }


def test_known_xai_image_generation_model_names():
    """Pin the curated xAI image models, including the `grok-imagine-image-pro` tier.

    `grok-imagine-image-pro` is a real, released model (xai_sdk `ImageGenerationModel` literal;
    https://docs.x.ai/developers/models/grok-imagine-image-pro) that was missing from the known set.
    """
    known_names = get_args(KnownImageGenerationModelName.__value__)

    assert {name for name in known_names if name.startswith('xai:')} == {
        'xai:grok-imagine-image',
        'xai:grok-imagine-image-pro',
        'xai:grok-imagine-image-quality',
    }


def test_google_geometry_profiles_conflicts_and_unknown_models():
    geometry = google_geometry.resolve_google_geometry(
        'gemini-3.1-flash-image',
        {'dimensions': (1024, 1024)},
        provider_aspect_ratio='1:1',
        provider_size='2K',
        provider_size_is_set=True,
    )
    assert geometry.aspect_ratio == '1:1'
    assert geometry.image_size == '2K'
    assert geometry.conflicts == ['dimensions']

    ignored_ratio = google_geometry.resolve_google_geometry(
        'gemini-2.5-flash-image',
        {'aspect_ratio': '1:2'},
        provider_aspect_ratio=None,
        provider_size=None,
        provider_size_is_set=False,
    )
    assert ignored_ratio.ignored == ['aspect_ratio']

    assert google_geometry.resolve_google_aspect_ratio('future-image-model', '1:2') == ('1:2', None)
    assert google_geometry.resolve_google_aspect_ratio('gemini-3-pro-image', '1:2') is None
    assert google_geometry.resolve_google_dimensions('gemini-3-pro-image', (1024, 1024)) == ('1:1', '1K')
    assert google_geometry.google_supports_image_size('gemini-3.1-flash-lite-image', '1K')
    assert not google_geometry.google_supports_image_size('gemini-3.1-flash-lite-image', '4K')
    assert google_geometry.google_supports_image_size('gemini-3.1-flash-image', '512')
    assert google_geometry.google_supports_image_size('gemini-3-pro-image', '4K')
    assert not google_geometry.google_supports_image_size('gemini-3-pro-image', '512')
    assert not google_geometry.google_supports_image_size('gemini-2.5-flash-image', '1K')
    assert google_geometry.google_supports_image_size('future-image-model', '4K')
    future_geometry = google_geometry.resolve_google_geometry(
        'future-image-model',
        {},
        provider_aspect_ratio=None,
        provider_size='4K',
        provider_size_is_set=True,
    )
    assert future_geometry.image_size == '4K'
    supported_size = google_geometry.resolve_google_geometry(
        'gemini-3.1-flash-image',
        {'size': '2K'},
        provider_aspect_ratio=None,
        provider_size=None,
        provider_size_is_set=False,
    )
    assert supported_size.image_size == '2K'
    with pytest.raises(UserError, match='does not support'):
        google_geometry.resolve_google_dimensions('future-image-model', (1024, 1024))


@pytest.mark.parametrize(
    ('dimensions', 'error_message'),
    [
        ((4096, 2048), 'longest edge'),
        ((3072, 992), 'aspect ratio'),
        ((800, 800), 'total pixel count'),
    ],
)
def test_openai_gpt_image_2_rejects_out_of_bounds_dimensions(dimensions: tuple[int, int], error_message: str):
    with pytest.raises(UserError, match=error_message):
        openai_geometry.resolve_openai_dimensions('gpt-image-2', dimensions)


def test_openai_geometry_conflicts_and_invalid_compatibility_sizes():
    geometry = openai_geometry.resolve_openai_geometry(
        'gpt-image-2',
        {'dimensions': (1024, 1024)},
        provider_size='1280x720',
    )
    assert geometry.size == '1280x720'
    assert geometry.conflicts == ['dimensions']

    with pytest.raises(UserError, match='Supported exact dimensions'):
        openai_geometry.resolve_openai_dimensions('gpt-image-1', (2048, 2048))
    assert openai_geometry.resolve_openai_dimensions('gpt-image-1', (1024, 1024)) == '1024x1024'

    assert openai_geometry.resolve_openai_compatibility_size('gpt-image-2', 'invalid') is None
    assert not openai_geometry.size_matches_aspect_ratio('invalid', '1:1')
    assert openai_geometry.parse_dimensions('invalidx10') is None
    assert openai_geometry.parse_dimensions('0x10') is None


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
def test_xai_geometry_reports_provider_conflicts_with_dimensions():
    geometry = xai_geometry.resolve_xai_geometry(
        'grok-imagine-image',
        {'dimensions': (1024, 1024)},
        provider_aspect_ratio='16:9',
        provider_resolution='2k',
    )

    assert geometry.aspect_ratio == '16:9'
    assert geometry.resolution == '2k'
    assert geometry.conflicts == ['dimensions', 'dimensions']

    matching_geometry = xai_geometry.resolve_xai_geometry(
        'grok-imagine-image',
        {'dimensions': (1024, 1024)},
        provider_aspect_ratio='1:1',
        provider_resolution='1k',
    )
    assert matching_geometry.conflicts == []


@pytest.mark.skipif(not image_demo_imports_successful(), reason='Image demo dependencies not installed')
async def test_image_demo_section_selection(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []
    parse_selection = images_demo._parse_selection  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(images_demo, '_run_openai_demo', AsyncMock(side_effect=lambda: calls.append('openai')))
    monkeypatch.setattr(images_demo, '_run_google_demo', AsyncMock(side_effect=lambda: calls.append('google')))
    monkeypatch.setattr(images_demo, '_run_xai_demo', AsyncMock(side_effect=lambda: calls.append('xai')))
    monkeypatch.setattr(
        images_demo,
        '_run_image_capability_demo',
        AsyncMock(side_effect=lambda: calls.append('capability')),
    )

    assert parse_selection([]) == images_demo.DemoSelection(
        providers=frozenset({'openai', 'google', 'xai'}),
        capability=True,
    )
    assert parse_selection(['--provider', 'google']) == images_demo.DemoSelection(
        providers=frozenset({'google'}),
        capability=False,
    )
    assert parse_selection(['--capability']) == images_demo.DemoSelection(
        providers=frozenset(),
        capability=True,
    )
    assert parse_selection(['-p', 'google', '-p', 'openai', '--capability']) == images_demo.DemoSelection(
        providers=frozenset({'openai', 'google'}),
        capability=True,
    )

    await images_demo.run_demo(images_demo.DemoSelection(frozenset({'google'}), capability=False))
    assert calls == ['google']

    calls.clear()
    await images_demo.run_demo(images_demo.DemoSelection(frozenset(), capability=True))
    assert calls == ['capability']

    calls.clear()
    await images_demo.run_demo()
    assert calls == ['openai', 'google', 'xai', 'capability']


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_infer_google_image_generation_model():
    model = infer_image_generation_model(
        'google:gemini-2.5-flash-image',
        provider_factory=lambda _: GoogleProvider(api_key='test-api-key'),
    )

    assert isinstance(model, GoogleImageGenerationModel)
    assert model.model_name == 'gemini-2.5-flash-image'
    assert model.system == 'google'
    assert model.base_url == 'https://generativelanguage.googleapis.com/'


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
def test_google_image_generation_model_infers_string_provider(monkeypatch: pytest.MonkeyPatch):
    provider = GoogleProvider(api_key='test-api-key')
    infer_provider = MagicMock(return_value=provider)
    monkeypatch.setattr(google_images, 'infer_provider', infer_provider)

    model = GoogleImageGenerationModel('gemini-2.5-flash-image')

    assert model.system == 'google'
    infer_provider.assert_called_once_with('google')


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_wire_payload_and_response_mapping():
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'candidates': [
                    {
                        'content': {
                            'parts': [
                                {
                                    'inlineData': {'data': 'dGhvdWdodA==', 'mimeType': 'image/png'},
                                    'thought': True,
                                },
                                {
                                    'inlineData': {'data': 'aGVsbG8=', 'mimeType': 'image/png'},
                                    'thoughtSignature': 'c2lnbmF0dXJl',
                                },
                            ],
                            'role': 'model',
                        },
                        'finishReason': 'STOP',
                        'index': 0,
                    },
                    {'finishReason': 'OTHER'},
                ],
                'modelVersion': 'gemini-2.5-flash-image',
                'responseId': 'response-123',
                'usageMetadata': {
                    'candidatesTokenCount': 5,
                    'candidatesTokensDetails': [{'modality': 'IMAGE', 'tokenCount': 5}],
                    'cacheTokensDetails': [{'modality': 'TEXT', 'tokenCount': 2}],
                    'cachedContentTokenCount': 2,
                    'promptTokenCount': 3,
                    'promptTokensDetails': [
                        {'modality': 'TEXT', 'tokenCount': 1},
                        {'modality': 'IMAGE', 'tokenCount': 2},
                        {'modality': 'TEXT'},
                    ],
                    'thoughtsTokenCount': 2,
                    'toolUsePromptTokenCount': 4,
                    'toolUsePromptTokensDetails': [{'modality': 'TEXT', 'tokenCount': 4}],
                    'totalTokenCount': 10,
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = GoogleProvider(api_key='test-api-key', base_url='https://example.com', http_client=http_client)
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)
    settings = GoogleImageGenerationSettings(
        aspect_ratio='1:1',
        size='1K',
        quality='low',
        extra_headers={'x-test-header': 'test-value'},
    )

    try:
        with pytest.warns(UserWarning, match=r'ignored unsupported settings: `quality`, `size`'):
            result = await model.generate(
                'replace the subject',
                images=[
                    BinaryImage(
                        data=b'first-image',
                        media_type='image/png',
                        vendor_metadata={'media_resolution': {'level': 'MEDIA_RESOLUTION_LOW'}},
                    ),
                    UploadedFile(
                        file_id='https://generativelanguage.googleapis.com/v1beta/files/file-123',
                        provider_name='google',
                        media_type='image/webp',
                    ),
                    ImageUrl(
                        'https://generativelanguage.googleapis.com/v1beta/files/file-456',
                        media_type='image/jpeg',
                    ),
                ],
                settings=settings,
            )

        conflicting_settings = GoogleImageGenerationSettings(
            n=2,
            aspect_ratio='16:9',
            size='2K',
            google_image_config={'aspect_ratio': '1:1'},
        )
        with pytest.warns(
            UserWarning,
            match=r'ignored unsupported settings: `n`, `size`.*used provider-specific settings instead of: `aspect_ratio`',
        ):
            await model.generate('conflicting settings', settings=conflicting_settings)

        with pytest.warns(UserWarning, match=r'ignored unsupported settings: `size`'):
            await model.generate(
                'unsupported settings',
                settings=GoogleImageGenerationSettings(size='1024x1024'),
            )

        with pytest.raises(UserError, match=r"does not support `google_image_config.image_size='4K'`"):
            await model.generate(
                'invalid provider size',
                settings=GoogleImageGenerationSettings(google_image_config={'image_size': '4K'}),
            )
    finally:
        await http_client.aclose()

    assert len(requests) == 3
    request = requests[0]
    assert request.method == 'POST'
    assert request.url.path == '/v1beta/models/gemini-2.5-flash-image:generateContent'
    assert request.headers['x-test-header'] == 'test-value'
    assert json.loads(request.content) == snapshot(
        {
            'contents': [
                {
                    'parts': [
                        {'text': 'replace the subject'},
                        {
                            'inlineData': {'data': 'Zmlyc3QtaW1hZ2U=', 'mimeType': 'image/png'},
                            'mediaResolution': {'level': 'MEDIA_RESOLUTION_LOW'},
                        },
                        {
                            'fileData': {
                                'fileUri': 'https://generativelanguage.googleapis.com/v1beta/files/file-123',
                                'mimeType': 'image/webp',
                            }
                        },
                        {
                            'fileData': {
                                'fileUri': 'https://generativelanguage.googleapis.com/v1beta/files/file-456',
                                'mimeType': 'image/jpeg',
                            }
                        },
                    ],
                    'role': 'user',
                }
            ],
            'generationConfig': {
                'imageConfig': {'aspectRatio': '1:1'},
                'responseModalities': ['IMAGE'],
            },
        }
    )
    assert result == snapshot(
        ImageGenerationResult(
            images=[
                GeneratedImage(
                    content=BinaryImage(data=b'hello', media_type='image/png'),
                    output_format='png',
                    provider_details={'has_thought_signature': True},
                )
            ],
            prompt='replace the subject',
            model_name='gemini-2.5-flash-image',
            provider_name='google',
            timestamp=IsDatetime(),
            usage=RequestUsage(
                input_tokens=7,
                cache_read_tokens=2,
                output_tokens=7,
                details={
                    'thoughts_tokens': 2,
                    'cached_content_tokens': 2,
                    'tool_use_prompt_tokens': 4,
                    'text_prompt_tokens': 1,
                    'image_prompt_tokens': 2,
                    'text_cache_tokens': 2,
                    'image_candidates_tokens': 5,
                    'text_tool_use_prompt_tokens': 4,
                },
            ),
            settings=settings,
            provider_details={'finish_reason': 'STOP'},
            provider_response_id='response-123',
            provider_url='https://example.com',
        )
    )


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_resolves_dimensions_and_aspect_ratio():
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'candidates': [
                    {
                        'content': {
                            'parts': [{'inlineData': {'data': 'aGVsbG8=', 'mimeType': 'image/png'}}],
                            'role': 'model',
                        },
                        'finishReason': 'STOP',
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = GoogleProvider(api_key='test-api-key', base_url='https://example.com', http_client=http_client)
    model = GoogleImageGenerationModel('gemini-3.1-flash-image', provider=provider)

    try:
        await model.generate('wide image', settings={'dimensions': (1376, 768)})
        await model.generate('portrait image', settings={'aspect_ratio': '3:4'})
        with pytest.raises(UserError, match=r'does not support `dimensions=\(1920, 1080\)`'):
            await model.generate('unsupported dimensions', settings={'dimensions': (1920, 1080)})
    finally:
        await http_client.aclose()

    assert [json.loads(request.content)['generationConfig']['imageConfig'] for request in requests] == [
        {'aspectRatio': '16:9', 'imageSize': '1K'},
        {'aspectRatio': '3:4', 'imageSize': '1K'},
    ]


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_wires_extra_body():
    """The `extra_body` escape hatch is merged into the outgoing `generateContent` request body.

    `ImageGenerationSettings.extra_body` reached the Google adapter but was silently dropped — the resolver
    forwarded only `extra_headers`. `google.genai`'s `HttpOptions.extra_body` recursively merges its dict into
    the request body, so a caller's extra fields must appear on the wire (and must not raise a spurious
    "ignored unsupported settings" warning, since the setting is now honored).

    - `HttpOptions.extra_body` ("Extra parameters to add to the request body"): python-genai
      `google/genai/types.py` `HttpOptions.extra_body`.
    - Merge site: python-genai `google/genai/_api_client.py`
      `_common.recursive_dict_update(request_dict, patched_http_options.extra_body)`.
    - Review item: `local-notes/review-items.md` §1.3 (`extra_body` silently dropped).
    """
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'candidates': [
                    {
                        'content': {
                            'parts': [{'inlineData': {'data': 'aGVsbG8=', 'mimeType': 'image/png'}}],
                            'role': 'model',
                        },
                        'finishReason': 'STOP',
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = GoogleProvider(api_key='test-api-key', base_url='https://example.com', http_client=http_client)
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            await model.generate('a robot', settings={'extra_body': {'labels': {'team': 'growth'}}})
    finally:
        await http_client.aclose()

    assert json.loads(requests[0].content)['labels'] == snapshot({'team': 'growth'})


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_downloads_image_url(monkeypatch: pytest.MonkeyPatch):
    download_mock = AsyncMock(return_value={'data': b'downloaded', 'data_type': 'image/webp'})
    monkeypatch.setattr(google_images, 'download_item', download_mock)
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'candidates': [
                    {
                        'content': {
                            'parts': [{'inlineData': {'data': 'aGVsbG8=', 'mimeType': 'image/png'}}],
                            'role': 'model',
                        },
                        'finishReason': 'STOP',
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = GoogleProvider(api_key='test-api-key', base_url='https://example.com', http_client=http_client)
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)
    image_url = ImageUrl('https://example.com/reference.png')

    try:
        await model.generate('edit this image', images=[image_url])
    finally:
        await http_client.aclose()

    download_mock.assert_awaited_once_with(image_url, data_format='bytes')
    body = json.loads(requests[0].content)
    assert body['contents'][0]['parts'][1] == {'inlineData': {'data': 'ZG93bmxvYWRlZA==', 'mimeType': 'image/webp'}}


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
@pytest.mark.parametrize(
    ('uploaded_file', 'error_message'),
    [
        (
            UploadedFile(file_id='file-google', provider_name='google', media_type='image/png'),
            'Google Files API URI.*starting with `https://`',
        ),
        (
            UploadedFile(file_id='https://example.com/file.png', provider_name='openai', media_type='image/png'),
            "provider_name='openai'.*Expected `provider_name` to be `'google'`",
        ),
    ],
)
async def test_google_image_generation_rejects_invalid_uploaded_file(uploaded_file: UploadedFile, error_message: str):
    provider = GoogleProvider(api_key='test-api-key')
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

    with pytest.raises(UserError, match=error_message):
        await model.generate('edit this image', images=[uploaded_file])


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_no_image_finish_reason():
    """A benign `NO_IMAGE` no-output raises `UnexpectedModelBehavior` naming the finish reason, not a filter error.

    Gemini can return HTTP 200 with `finishReason=NO_IMAGE` and only text (the "returns text instead of an
    image" soft-failure) when it declines to draw — this is not a safety block. The user must be able to tell
    it apart from a content-moderation block without parsing the raw body, so the finish reason is named in the
    message, and it must NOT surface as `ContentFilterError`.

    - `FinishReason.NO_IMAGE` ("model was expected to generate an image, but none was generated"):
      python-genai `google/genai/types.py` `FinishReason`.
    - ai.google.dev image-generation guide (NO_IMAGE soft failure): https://ai.google.dev/gemini-api/docs/image-generation
    - Research: `local-notes/image-gen-research/google-gemini-image-api.md` §5; gap analysis B3.
    """

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'candidates': [
                    {
                        'content': {'parts': [{'text': 'Here is a description instead.'}], 'role': 'model'},
                        'finishReason': 'NO_IMAGE',
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = GoogleProvider(api_key='test-api-key', base_url='https://example.com', http_client=http_client)
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

    try:
        with pytest.raises(
            UnexpectedModelBehavior, match=r'did not contain any images \(finish_reason: NO_IMAGE\)'
        ) as exc_info:
            await model.generate('tiny robot')
    finally:
        await http_client.aclose()

    assert not isinstance(exc_info.value, ContentFilterError)
    assert exc_info.value.body is not None
    assert "'finish_reason': 'NO_IMAGE'" in exc_info.value.body


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_image_safety_finish_reason():
    """An `IMAGE_SAFETY` finish reason raises a typed `ContentFilterError` naming the reason.

    A candidate returned with `finishReason=IMAGE_SAFETY` and no image part is a content-policy refusal, so we
    raise `ContentFilterError` (the images content-moderation error, consistent with the xAI adapter) rather
    than a generic `UnexpectedModelBehavior`, and name the reason in the message.

    - `FinishReason.IMAGE_SAFETY`: python-genai `google/genai/types.py` `FinishReason`.
    - Research: `local-notes/image-gen-research/google-gemini-image-api.md` §5 (IMAGE_SAFETY silent block); gap B4.
    """

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'candidates': [
                    {
                        'content': {'parts': [{'text': 'I cannot create that image.'}], 'role': 'model'},
                        'finishReason': 'IMAGE_SAFETY',
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = GoogleProvider(api_key='test-api-key', base_url='https://example.com', http_client=http_client)
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

    try:
        with pytest.raises(ContentFilterError, match=r'content moderation \(reason: IMAGE_SAFETY\)') as exc_info:
            await model.generate('tiny robot')
    finally:
        await http_client.aclose()

    assert exc_info.value.body is not None
    assert "'finish_reason': 'IMAGE_SAFETY'" in exc_info.value.body


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_prompt_blocked():
    """A prompt-level block (empty candidates + `promptFeedback.blockReason`) raises `ContentFilterError`.

    When Gemini blocks the prompt itself it returns no candidates and a `promptFeedback.blockReason`
    (e.g. `PROHIBITED_CONTENT`). This is a content-moderation outcome, so it raises `ContentFilterError`
    naming the block reason, with the block details preserved in the body.

    - `BlockedReason.PROHIBITED_CONTENT`: python-genai `google/genai/types.py` `BlockedReason`.
    - Research: `local-notes/image-gen-research/google-gemini-image-api.md` §5 (safety blocks → empty parts, `prompt_feedback.block_reason`).
    """

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'promptFeedback': {
                    'blockReason': 'PROHIBITED_CONTENT',
                    'blockReasonMessage': 'blocked by safety policy',
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = GoogleProvider(api_key='test-api-key', base_url='https://example.com', http_client=http_client)
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

    try:
        with pytest.raises(ContentFilterError, match=r'content moderation \(reason: PROHIBITED_CONTENT\)') as exc_info:
            await model.generate('tiny robot')
    finally:
        await http_client.aclose()

    assert exc_info.value.body is not None
    assert "'block_reason': 'PROHIBITED_CONTENT'" in exc_info.value.body
    assert "'block_reason_message': 'blocked by safety policy'" in exc_info.value.body


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_degenerate_candidates():
    """Degenerate candidates (empty `parts`, or no candidates at all) raise a clean typed error, never `IndexError`.

    A candidate whose `content.parts` is empty, and a response with no candidates, both yield no image. The
    adapter must not index into empty sequences; it raises `UnexpectedModelBehavior` (not `ContentFilterError`,
    since neither carries a moderation signal), naming the finish reason when one is present.

    - Empty `parts` / 200-OK-no-image guard: python-genai response shape `candidates[].content.parts`.
    - Research: `local-notes/image-gen-research/google-gemini-image-api.md` §8.4 (empty `parts` with 200 OK); gap B4.
    """
    degenerate_responses: list[dict[str, object]] = [
        {'candidates': [{'content': {'parts': [], 'role': 'model'}, 'finishReason': 'STOP'}]},
        {'candidates': []},
    ]
    responses = iter(degenerate_responses)

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(responses))

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = GoogleProvider(api_key='test-api-key', base_url='https://example.com', http_client=http_client)
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

    try:
        with pytest.raises(
            UnexpectedModelBehavior, match=r'did not contain any images \(finish_reason: STOP\)'
        ) as empty_parts:
            await model.generate('empty parts')
        with pytest.raises(UnexpectedModelBehavior, match=r'did not contain any images$') as no_candidates:
            await model.generate('no candidates')
    finally:
        await http_client.aclose()

    assert not isinstance(empty_parts.value, ContentFilterError)
    assert no_candidates.value.body is None


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_supported_settings_emit_no_warning():
    """A fully-supported Google settings combination emits no warning.

    Over-warning erodes the signal of the warning channel; warnings are reserved for settings a request
    genuinely ignores or overrides. `n=1`, a `google_image_config` aspect ratio, `extra_headers`, and
    `extra_body` are all honored by the adapter, so the call must be silent.

    - `warn_image_generation_settings` channel: `pydantic_ai/images/settings.py`.
    - Negative-warning coverage gap: `local-notes/review-items.md` §4 (warnings coverage should include the negative).
    """

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'candidates': [
                    {
                        'content': {
                            'parts': [{'inlineData': {'data': 'aGVsbG8=', 'mimeType': 'image/png'}}],
                            'role': 'model',
                        },
                        'finishReason': 'STOP',
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = GoogleProvider(api_key='test-api-key', base_url='https://example.com', http_client=http_client)
    model = GoogleImageGenerationModel('gemini-3-pro-image', provider=provider)
    settings = GoogleImageGenerationSettings(
        n=1,
        extra_headers={'x-team': 'growth'},
        extra_body={'labels': {'team': 'growth'}},
        google_image_config={'aspect_ratio': '16:9'},
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            await model.generate('a robot', settings=settings)
    finally:
        await http_client.aclose()


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_maps_complete_provider_metadata():
    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'candidates': [
                    {
                        'content': {
                            'parts': [{'inlineData': {'data': 'aGVsbG8=', 'mimeType': 'image/png'}}],
                            'role': 'model',
                        },
                        'finishReason': 'STOP',
                        'safetyRatings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE'}],
                    }
                ],
                'createTime': '2025-01-01T00:00:00Z',
                'promptFeedback': {
                    'blockReason': 'OTHER',
                    'blockReasonMessage': 'provider detail',
                    'safetyRatings': [{'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'LOW'}],
                },
                'usageMetadata': {'trafficType': 'ON_DEMAND'},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = GoogleProvider(api_key='test-api-key', base_url='https://example.com', http_client=http_client)
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

    try:
        result = await model.generate('tiny robot')
    finally:
        await http_client.aclose()

    assert result.provider_details is not None
    assert result.provider_details['finish_reason'] == 'STOP'
    assert result.provider_details['block_reason'] == 'OTHER'
    assert result.provider_details['block_reason_message'] == 'provider detail'
    assert result.provider_details['traffic_type'] == 'ON_DEMAND'
    assert result.provider_details['safety_ratings'] == [
        {
            'blocked': None,
            'category': 'HARM_CATEGORY_HARASSMENT',
            'overwrittenThreshold': None,
            'probability': 'LOW',
            'probabilityScore': None,
            'severity': None,
            'severityScore': None,
        }
    ]

    response_with_timestamp = google_types.GenerateContentResponse.model_validate(
        {'createTime': '2025-01-01T00:00:00Z'}
    )
    timestamp_details = google_images._response_provider_details(  # pyright: ignore[reportPrivateUsage]
        response_with_timestamp
    )
    assert timestamp_details['timestamp'] == IsDatetime()

    response_without_block_message = google_types.GenerateContentResponse.model_validate(
        {'promptFeedback': {'blockReason': 'OTHER'}}
    )
    block_details = google_images._response_provider_details(  # pyright: ignore[reportPrivateUsage]
        response_without_block_message
    )
    assert block_details == {'block_reason': 'OTHER'}


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_maps_non_http_api_error():
    client = AsyncMock()
    client.aio.models.generate_content.side_effect = google_errors.APIError(302, {'error': 'redirect'})
    provider = GoogleProvider(client=cast(GoogleClient, client))
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

    with pytest.raises(ModelAPIError, match='redirect'):
        await model.generate('tiny robot')


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
def test_google_image_generation_ignores_non_image_output_format():
    output_format = google_images._output_format_from_media_type(  # pyright: ignore[reportPrivateUsage]
        'application/octet-stream'
    )

    assert output_format is None


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_status_error():
    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={'error': {'code': 400, 'message': 'invalid image request', 'status': 'INVALID_ARGUMENT'}},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    provider = GoogleProvider(api_key='test-api-key', base_url='https://example.com', http_client=http_client)
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

    try:
        with pytest.raises(ModelHTTPError) as exc_info:
            await model.generate('tiny robot')
    finally:
        await http_client.aclose()

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {
        'error': {'code': 400, 'message': 'invalid image request', 'status': 'INVALID_ARGUMENT'}
    }


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
@pytest.mark.vcr
async def test_google_image_generation_vcr():
    provider = GoogleProvider(api_key=os.getenv('GOOGLE_API_KEY', os.getenv('GEMINI_API_KEY', 'mock-api-key')))
    model = GoogleImageGenerationModel('gemini-3.1-flash-lite-image', provider=provider)

    result = await model.generate(
        'A cat with a cowboy hat, dancing in Rome.',
        settings=GoogleImageGenerationSettings(dimensions=(1024, 1024)),
    )

    assert len(result.images) == 1
    generated_image = result.images[0]
    assert generated_image.content.media_type.startswith('image/')
    assert len(generated_image.content.data) > 100
    assert generated_image.output_format
    assert result.model_name == 'gemini-3.1-flash-lite-image'
    assert result.provider_name == 'google'
    assert result.provider_url == 'https://generativelanguage.googleapis.com/'
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.provider_details == {'finish_reason': 'STOP'}
    assert result.provider_response_id


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
@pytest.mark.vcr
async def test_google_image_edit_binary_image_vcr(image_content: BinaryImage):
    provider = GoogleProvider(api_key=os.getenv('GOOGLE_API_KEY', os.getenv('GEMINI_API_KEY', 'mock-api-key')))
    model = GoogleImageGenerationModel('gemini-3.1-flash-lite-image', provider=provider)

    result = await model.generate(
        'Transform the subject into a dog with a cowboy hat, dancing in Rome.',
        images=[image_content],
        settings=GoogleImageGenerationSettings(google_image_config={'aspect_ratio': '1:1'}),
    )

    assert len(result.images) == 1
    edited_image = result.images[0]
    assert edited_image.content.media_type.startswith('image/')
    assert len(edited_image.content.data) > 100
    assert edited_image.output_format
    assert result.model_name == 'gemini-3.1-flash-lite-image'
    assert result.provider_name == 'google'
    assert result.provider_url == 'https://generativelanguage.googleapis.com/'
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.provider_details == {'finish_reason': 'STOP'}
    assert result.provider_response_id


def _xai_image_responses(*data: bytes, respect_moderation: bool = True) -> list[XaiImageResponse]:
    proto = xai_image_pb2.ImageResponse(
        images=[
            xai_image_pb2.GeneratedImage(
                base64=f'data:image/jpeg;base64,{base64.b64encode(image_data).decode()}',
                respect_moderation=respect_moderation,
            )
            for image_data in data
        ],
        model='grok-imagine-image',
        usage=xai_usage_pb2.SamplingUsage(
            prompt_tokens=7,
            completion_tokens=11,
            total_tokens=18,
            reasoning_tokens=3,
            cached_prompt_text_tokens=2,
            prompt_text_tokens=4,
            prompt_image_tokens=3,
            cost_in_usd_ticks=200_000_000,
        ),
    )
    return [XaiImageResponse(proto, index) for index in range(len(data))]


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_infer_xai_image_generation_model():
    model = infer_image_generation_model(
        'xai:grok-imagine-image',
        provider_factory=lambda _: XaiProvider(xai_client=XaiAsyncClient(api_key='test-api-key')),
    )

    assert isinstance(model, XaiImageGenerationModel)
    assert model.model_name == 'grok-imagine-image'
    assert model.system == 'xai'
    assert model.base_url == 'https://api.x.ai/v1'


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
def test_xai_image_generation_model_infers_string_provider(monkeypatch: pytest.MonkeyPatch):
    provider = XaiProvider(xai_client=XaiAsyncClient(api_key='test-api-key'))
    infer_provider = MagicMock(return_value=provider)
    monkeypatch.setattr(xai_images, 'infer_provider', infer_provider)

    model = XaiImageGenerationModel('grok-imagine-image')

    assert model.system == 'xai'
    infer_provider.assert_called_once_with('xai')


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_wire_payload_and_response_mapping():
    mock_client = AsyncMock()
    responses = _xai_image_responses(b'first-image', b'second-image')
    mock_client.image.sample_batch.return_value = responses
    provider = XaiProvider(xai_client=cast(XaiAsyncClient, mock_client))
    model = XaiImageGenerationModel('grok-imagine-image', provider=provider)
    settings = XaiImageGenerationSettings(n=2, xai_user='user-123', aspect_ratio='1:1', size='1K')

    result = await model.generate(
        'replace the subject',
        images=[
            UploadedFile(file_id='file-123', provider_name='xai', media_type='image/jpeg'),
            BinaryImage(data=b'binary-image', media_type='image/png'),
            ImageUrl('https://example.com/reference.webp'),
        ],
        settings=settings,
    )

    mock_client.image.sample_batch.assert_awaited_once_with(
        'replace the subject',
        'grok-imagine-image',
        2,
        image_url=None,
        image_file_id=None,
        image_urls=['data:image/png;base64,YmluYXJ5LWltYWdl', 'https://example.com/reference.webp'],
        image_file_ids=['file-123'],
        user='user-123',
        image_format='base64',
        aspect_ratio='1:1',
        resolution='1k',
    )
    assert result == snapshot(
        ImageGenerationResult(
            images=[
                GeneratedImage(
                    content=BinaryImage(data=b'first-image', media_type='image/jpeg'),
                    output_format='jpeg',
                    provider_details={'respect_moderation': True},
                ),
                GeneratedImage(
                    content=BinaryImage(data=b'second-image', media_type='image/jpeg'),
                    output_format='jpeg',
                    provider_details={'respect_moderation': True},
                ),
            ],
            prompt='replace the subject',
            model_name='grok-imagine-image',
            provider_name='xai',
            timestamp=IsDatetime(),
            usage=RequestUsage(
                input_tokens=7,
                output_tokens=11,
                details={
                    'reasoning_tokens': 3,
                    'cached_prompt_text_tokens': 2,
                    'input_text_tokens': 4,
                    'input_image_tokens': 3,
                },
            ),
            settings=settings,
            provider_details={'cost_in_usd_ticks': 200000000, 'cost_usd': 0.02},
            provider_url='https://api.x.ai/v1',
        )
    )

    mock_client.image.sample.return_value = responses[0]
    conflicting_settings = XaiImageGenerationSettings(
        aspect_ratio='16:9',
        size='2K',
        quality='low',
        xai_aspect_ratio='1:1',
        xai_resolution='1k',
    )
    with pytest.warns(
        UserWarning,
        match=r'ignored unsupported settings: `quality`.*used provider-specific settings instead of: `aspect_ratio`, `size`',
    ):
        await model.generate('conflicting settings', settings=conflicting_settings)

    mock_client.image.sample.assert_awaited_once_with(
        'conflicting settings',
        'grok-imagine-image',
        image_url=None,
        image_file_id=None,
        image_urls=None,
        image_file_ids=None,
        user=None,
        image_format='base64',
        aspect_ratio='1:1',
        resolution='1k',
    )

    with pytest.warns(UserWarning, match=r'ignored unsupported settings: `aspect_ratio`, `size`'):
        await model.generate(
            'unsupported settings',
            settings=XaiImageGenerationSettings(aspect_ratio='4:5', size='4K'),
        )


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
@pytest.mark.parametrize('model_name', ['grok-imagine-image', 'grok-imagine-image-quality'])
@pytest.mark.parametrize(
    ('dimensions', 'aspect_ratio', 'resolution'),
    [
        ((1024, 1024), '1:1', '1k'),
        ((2048, 2048), '1:1', '2k'),
        ((864, 1152), '3:4', '1k'),
        ((1776, 2368), '3:4', '2k'),
        ((1152, 864), '4:3', '1k'),
        ((2368, 1776), '4:3', '2k'),
        ((720, 1280), '9:16', '1k'),
        ((1584, 2816), '9:16', '2k'),
        ((1280, 720), '16:9', '1k'),
        ((2816, 1584), '16:9', '2k'),
        ((832, 1248), '2:3', '1k'),
        ((1664, 2496), '2:3', '2k'),
        ((1248, 832), '3:2', '1k'),
        ((2496, 1664), '3:2', '2k'),
        ((576, 1248), '9:19.5', '1k'),
        ((1344, 2912), '9:19.5', '2k'),
        ((1248, 576), '19.5:9', '1k'),
        ((2912, 1344), '19.5:9', '2k'),
        ((576, 1280), '9:20', '1k'),
        ((1440, 3200), '9:20', '2k'),
        ((1280, 576), '20:9', '1k'),
        ((3200, 1440), '20:9', '2k'),
        ((704, 1408), '1:2', '1k'),
        ((1456, 2912), '1:2', '2k'),
        ((1408, 704), '2:1', '1k'),
        ((2912, 1456), '2:1', '2k'),
    ],
)
async def test_xai_image_generation_resolves_dimensions(
    model_name: str, dimensions: tuple[int, int], aspect_ratio: str, resolution: str
):
    mock_client = AsyncMock()
    mock_client.image.sample.return_value = _xai_image_responses(b'image')[0]
    model = XaiImageGenerationModel(
        model_name,
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    await model.generate('geometric image', settings={'dimensions': dimensions})

    assert mock_client.image.sample.await_args.kwargs['aspect_ratio'] == aspect_ratio
    assert mock_client.image.sample.await_args.kwargs['resolution'] == resolution


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_rejects_unsupported_dimensions():
    mock_client = AsyncMock()
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    with pytest.raises(UserError, match=r"model 'grok-imagine-image' does not support `dimensions=\(1920, 1080\)`"):
        await model.generate('unsupported dimensions', settings={'dimensions': (1920, 1080)})
    mock_client.image.sample.assert_not_awaited()


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_maps_common_aspect_ratio_to_canonical_1k_geometry():
    mock_client = AsyncMock()
    mock_client.image.sample.return_value = _xai_image_responses(b'image')[0]
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    await model.generate('wide image', settings={'aspect_ratio': '16:9'})

    assert mock_client.image.sample.await_args.kwargs['aspect_ratio'] == '16:9'
    assert mock_client.image.sample.await_args.kwargs['resolution'] == '1k'


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_rejects_dimensions_for_unknown_model():
    mock_client = AsyncMock()
    model = XaiImageGenerationModel(
        'future-image-model',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    with pytest.raises(UserError, match='does not have a known exact-dimensions mapping'):
        await model.generate('unknown geometry', settings={'dimensions': (1024, 1024)})
    mock_client.image.sample.assert_not_awaited()


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_single_uploaded_file():
    mock_client = AsyncMock()
    mock_client.image.sample.return_value = _xai_image_responses(b'edited-image')[0]
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    await model.generate(
        'edit this image',
        images=[UploadedFile(file_id='file-123', provider_name='xai', media_type='image/jpeg')],
    )

    mock_client.image.sample.assert_awaited_once_with(
        'edit this image',
        'grok-imagine-image',
        image_url=None,
        image_file_id='file-123',
        image_urls=None,
        image_file_ids=None,
        user=None,
        image_format='base64',
        aspect_ratio=None,
        resolution=None,
    )


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_downloads_forced_image_url(monkeypatch: pytest.MonkeyPatch):
    download_mock = AsyncMock(return_value={'data': b'downloaded-image', 'data_type': 'image/webp'})
    monkeypatch.setattr(xai_images, 'download_item', download_mock)
    mock_client = AsyncMock()
    mock_client.image.sample.return_value = _xai_image_responses(b'edited-image')[0]
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )
    image_url = ImageUrl('https://example.com/reference.png', force_download=True)

    await model.generate('edit this image', images=[image_url])

    download_mock.assert_awaited_once_with(image_url, data_format='bytes')
    assert mock_client.image.sample.await_args.kwargs['image_url'] == (
        'data:image/webp;base64,ZG93bmxvYWRlZC1pbWFnZQ=='
    )


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_rejects_uploaded_file_provider_mismatch():
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, AsyncMock())),
    )

    with pytest.raises(UserError, match="Expected `provider_name` to be `'xai'`"):
        await model.generate(
            'edit this image',
            images=[UploadedFile(file_id='file-123', provider_name='google', media_type='image/jpeg')],
        )


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_rejects_mixed_inputs_that_would_be_reordered():
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, AsyncMock())),
    )

    with pytest.raises(UserError, match='Place all `UploadedFile` inputs first'):
        await model.generate(
            'edit these images',
            images=[
                ImageUrl('https://example.com/reference.png'),
                UploadedFile(file_id='file-123', provider_name='xai', media_type='image/jpeg'),
            ],
        )


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
@pytest.mark.parametrize('base64_value', ['', 'data:text/plain;base64,aGVsbG8='])
async def test_xai_image_generation_invalid_response(base64_value: str):
    """A non-moderated slot with a missing or non-image payload is unexpected behavior.

    Distinct from silent moderation (`respect_moderation=False`, covered separately): here the SDK
    reports the image respects moderation yet the base64 payload is empty or carries a non-image media
    type, so we surface `UnexpectedModelBehavior` rather than dropping it as a flagged slot.

    Reference: `xai_sdk.aio.image.ImageResponse.base64` raises when the payload is empty.
    """
    mock_client = AsyncMock()
    proto = xai_image_pb2.ImageResponse(
        images=[xai_image_pb2.GeneratedImage(base64=base64_value, respect_moderation=True)],
        model='grok-imagine-image',
    )
    mock_client.image.sample.return_value = XaiImageResponse(proto, 0)
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    with pytest.raises(UnexpectedModelBehavior, match='did not contain valid base64 image data'):
        await model.generate('tiny robot')


def _xai_moderated_image_responses(*slots: tuple[bytes | None, bool]) -> list[XaiImageResponse]:
    """Build one batch `ImageResponse` list mirroring xAI's silent per-slot moderation.

    Each slot is `(image_bytes_or_None, respect_moderation)`. A moderated slot (`respect_moderation=False`)
    carries an empty `base64` field, exactly as the SDK exposes a flagged image; accessing its `.base64`
    then raises `ValueError` (`xai_sdk.aio.image.ImageResponse.base64`). All slots share one proto, matching
    the real `sample_batch` wire shape (one RPC, one response, positional `images[]`).
    """
    proto = xai_image_pb2.ImageResponse(
        images=[
            xai_image_pb2.GeneratedImage(
                base64=(f'data:image/jpeg;base64,{base64.b64encode(data).decode()}' if data is not None else ''),
                respect_moderation=respect_moderation,
            )
            for data, respect_moderation in slots
        ],
        model='grok-imagine-image',
        usage=xai_usage_pb2.SamplingUsage(
            prompt_tokens=7,
            completion_tokens=11,
            cost_in_usd_ticks=200_000_000,
        ),
    )
    return [XaiImageResponse(proto, index) for index in range(len(slots))]


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_skips_moderated_batch_slots(monkeypatch: pytest.MonkeyPatch):
    """One silently-moderated slot must not discard the rest of a paid batch.

    xAI moderation is silent: the RPC succeeds and a flagged slot returns `respect_moderation=False`
    with an empty payload (`xai_sdk.aio.image.ImageResponse.base64` raises on access). We skip flagged
    slots, return the clean subset, and surface the flagged indices at result level via
    `ImageGenerationResult.provider_details['moderated_image_indices']`, keeping per-image
    `respect_moderation` too.

    References:
    - `xai_sdk.aio.image.ImageResponse.respect_moderation` / `.base64` (silent-moderation semantics).
    - Research: local-notes/image-gen-research/xai-grok-imagine-api.md section 4 (Error behavior).
    """
    decoded_values: list[str] = []
    real_decode = xai_images._decode_data_url  # pyright: ignore[reportPrivateUsage]

    def spy_decode(value: str) -> BinaryImage:
        decoded_values.append(value)
        return real_decode(value)

    monkeypatch.setattr(xai_images, '_decode_data_url', spy_decode)

    mock_client = AsyncMock()
    mock_client.image.sample_batch.return_value = _xai_moderated_image_responses(
        (b'first-image', True),
        (None, False),
        (b'third-image', True),
    )
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    result = await model.generate('tiny robot', settings={'n': 3})

    assert [image.content.data for image in result.images] == [b'first-image', b'third-image']
    assert all(image.provider_details == {'respect_moderation': True} for image in result.images)
    assert result.provider_details == snapshot(
        {'cost_in_usd_ticks': 200000000, 'cost_usd': 0.02, 'moderated_image_indices': [1]}
    )
    # Decoding is never attempted on the flagged slot (its `.base64` access would raise).
    assert decoded_values == snapshot(
        ['data:image/jpeg;base64,Zmlyc3QtaW1hZ2U=', 'data:image/jpeg;base64,dGhpcmQtaW1hZ2U=']
    )


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_all_slots_moderated_raises_content_filter():
    """When every slot is moderated there is nothing to return, so raise a content-filter error.

    A fully-moderated batch is not `UnexpectedModelBehavior` (the RPC behaved as designed) — it is a
    content-moderation outcome, so we raise the semantically-correct `ContentFilterError`.

    Reference: `xai_sdk.aio.image.ImageResponse.respect_moderation` (silent moderation, per-slot).
    """
    mock_client = AsyncMock()
    mock_client.image.sample_batch.return_value = _xai_moderated_image_responses(
        (None, False),
        (None, False),
    )
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    with pytest.raises(ContentFilterError, match='content moderation'):
        await model.generate('tiny robot', settings={'n': 2})


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_response_without_cost_details():
    mock_client = AsyncMock()
    proto = xai_image_pb2.ImageResponse(
        images=[xai_image_pb2.GeneratedImage(base64='data:image/png;base64,aGVsbG8=', respect_moderation=True)],
        model='grok-imagine-image',
        usage=xai_usage_pb2.SamplingUsage(prompt_tokens=1),
    )
    mock_client.image.sample.return_value = XaiImageResponse(proto, 0)
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    result = await model.generate('tiny robot')

    assert result.provider_details == {}


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_empty_response():
    mock_client = AsyncMock()
    mock_client.image.sample_batch.return_value = []
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    with pytest.raises(UnexpectedModelBehavior, match='did not contain any images'):
        await model.generate('tiny robot', settings={'n': 2})


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_status_error():
    class TestRpcError(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return grpc.StatusCode.RESOURCE_EXHAUSTED

        def details(self) -> str:
            return 'rate limit exceeded'

    mock_client = AsyncMock()
    mock_client.image.sample.side_effect = TestRpcError()
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        await model.generate('tiny robot')

    assert exc_info.value.status_code == 429
    assert exc_info.value.body == 'rate limit exceeded'


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_unknown_status_error():
    class TestRpcError(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return grpc.StatusCode.CANCELLED

        def details(self) -> str:
            return 'request cancelled'

    mock_client = AsyncMock()
    mock_client.image.sample.side_effect = TestRpcError()
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    with pytest.raises(ModelAPIError, match='request cancelled'):
        await model.generate('tiny robot')


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
@pytest.mark.parametrize(
    'status_name,expected_http',
    [
        ('INVALID_ARGUMENT', 400),
        ('UNAUTHENTICATED', 401),
        ('PERMISSION_DENIED', 403),
    ],
)
async def test_xai_image_generation_maps_grpc_status_to_http(status_name: str, expected_http: int):
    """gRPC status codes map to their HTTP-equivalent `ModelHTTPError`.

    xAI's image path is gRPC, so provider errors arrive as `grpc.StatusCode`, not HTTP codes. A bad
    request (`INVALID_ARGUMENT`) must surface as 400 rather than the generic `ModelAPIError`, and the
    already-mapped auth codes (`UNAUTHENTICATED`/`PERMISSION_DENIED`) are pinned here too.
    Parametrized by enum name because `grpc` is an optional import: enum values in the decorator would
    `NameError` at collection time in environments without the xAI extras.

    Reference: `_GRPC_STATUS_TO_HTTP` in `pydantic_ai.images.xai`.
    """
    status_code = grpc.StatusCode[status_name]

    class TestRpcError(grpc.RpcError):
        def code(self) -> grpc.StatusCode:
            return status_code

        def details(self) -> str:
            return 'boom'

    mock_client = AsyncMock()
    mock_client.image.sample.side_effect = TestRpcError()
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        await model.generate('tiny robot')

    assert exc_info.value.status_code == expected_http
    assert exc_info.value.body == 'boom'


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
@pytest.mark.vcr
def test_xai_image_generation_vcr(xai_provider: XaiProvider):
    model = XaiImageGenerationModel('grok-imagine-image', provider=xai_provider)
    generator = ImageGenerator(model)

    result = generator.generate_sync(
        'A cat with a cowboy hat, dancing in Rome.',
        settings=XaiImageGenerationSettings(dimensions=(1024, 1024)),
    )

    assert len(result.images) == 1
    generated_image = result.images[0]
    assert generated_image.content.media_type == 'image/jpeg'
    assert len(generated_image.content.data) > 100
    assert generated_image.output_format == 'jpeg'
    assert generated_image.provider_details == {'respect_moderation': True}
    assert result.model_name == 'grok-imagine-image'
    assert result.provider_name == 'xai'
    assert result.provider_url == 'https://api.x.ai/v1'
    assert result.usage == RequestUsage()
    assert result.provider_details == {'cost_in_usd_ticks': 200000000, 'cost_usd': 0.02}


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
@pytest.mark.vcr
def test_xai_image_generation_unlisted_model_vcr(xai_provider: XaiProvider):
    """`ImageGenerator` completes a real generate() when the model id is supplied as a plain `str`.

    Forward-compat guard: model ids churn faster than `KnownImageGenerationModelName`. Supplying the id
    through a `str`-typed variable — not the `KnownImageGenerationModelName` / `xai_sdk` Literal — proves
    the non-Literal branch of `XaiImageGenerationModelName` runs end-to-end against the live API, so a
    brand-new model id not yet in any Literal still works. Recorded against the real
    `grok-imagine-image-pro` tier (https://docs.x.ai/developers/models/grok-imagine-image-pro); it is in
    the curated set, so the `str` annotation (not the model string) is what exercises the forward path.

    Live-API note, pinned by the `model_name` assertion below: xAI resolves the requested
    `grok-imagine-image-pro` id to `grok-imagine-image-quality` in the response's `model` field, and
    `result.model_name` reflects the resolved response model (not the requested id).
    """
    model_name: str = 'grok-imagine-image-pro'
    generator = ImageGenerator(XaiImageGenerationModel(model_name, provider=xai_provider))

    result = generator.generate_sync('A cat with a cowboy hat, dancing in Rome.')

    assert len(result.images) == 1
    generated_image = result.images[0]
    assert generated_image.content.media_type == 'image/jpeg'
    assert len(generated_image.content.data) > 100
    assert result.model_name == 'grok-imagine-image-quality'
    assert result.provider_name == 'xai'
    assert result.provider_url == 'https://api.x.ai/v1'


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
@pytest.mark.vcr
async def test_xai_image_edit_binary_image_vcr(xai_provider: XaiProvider, image_content: BinaryImage):
    model = XaiImageGenerationModel('grok-imagine-image', provider=xai_provider)

    result = await model.generate(
        'Replace the cat with a dog while preserving the cowboy hat, dancing pose, and Rome setting.',
        images=[image_content],
        settings=XaiImageGenerationSettings(xai_aspect_ratio='1:1', xai_resolution='1k'),
    )

    assert len(result.images) == 1
    edited_image = result.images[0]
    assert edited_image.content.media_type == 'image/jpeg'
    assert len(edited_image.content.data) > 100
    assert edited_image.output_format == 'jpeg'
    assert edited_image.provider_details == {'respect_moderation': True}
    assert result.model_name == 'grok-imagine-image'
    assert result.provider_name == 'xai'
    assert result.provider_url == 'https://api.x.ai/v1'
    assert result.usage == RequestUsage()
    assert result.provider_details == {'cost_in_usd_ticks': 220000000, 'cost_usd': 0.022000000000000002}


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
def test_openai_image_generation_model_infers_string_provider(monkeypatch: pytest.MonkeyPatch):
    provider = OpenAIProvider(openai_client=AsyncOpenAI(api_key='test-api-key'))
    infer_provider = MagicMock(return_value=provider)
    monkeypatch.setattr(openai_images, 'infer_provider', infer_provider)

    model = OpenAIImageGenerationModel('gpt-image-1')

    assert model.system == 'openai'
    infer_provider.assert_called_once_with('openai')


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

    settings = OpenAIImageGenerationSettings(
        n=1,
        aspect_ratio='1:1',
        size='auto',
        background='opaque',
        moderation='low',
        output_format='png',
        output_compression=80,
        quality='low',
    )
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
    assert mock_client.images.generate.await_args.kwargs['background'] == 'opaque'
    assert mock_client.images.generate.await_args.kwargs['moderation'] == 'low'
    assert mock_client.images.generate.await_args.kwargs['quality'] == 'low'
    assert mock_client.images.generate.await_args.kwargs['output_compression'] == 80

    unsupported_settings = OpenAIImageGenerationSettings(
        input_fidelity='high',
        size='1K',
        aspect_ratio='16:9',
    )
    with pytest.warns(
        UserWarning,
        match=r'ignored unsupported settings: `input_fidelity`, `size`, `aspect_ratio`',
    ):
        await model.generate('unsupported settings', settings=unsupported_settings)

    with pytest.warns(UserWarning, match=r'ignored unsupported settings: `aspect_ratio`'):
        await model.generate(
            'conflicting normalized dimensions',
            settings=OpenAIImageGenerationSettings(size='1024x1024', aspect_ratio='3:2'),
        )

    await model.generate(
        'provider-only background',
        settings=OpenAIImageGenerationSettings(openai_background='opaque'),
    )

    await model.generate(
        'valid transparent background',
        settings=OpenAIImageGenerationSettings(background='transparent', output_format='webp'),
    )
    assert mock_client.images.generate.await_args.kwargs['background'] == 'transparent'
    assert mock_client.images.generate.await_args.kwargs['output_format'] == 'webp'


_JPEG_MAGIC_BYTES = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01' + b'\x00' * 8
_WEBP_MAGIC_BYTES = b'RIFF\x00\x00\x00\x00WEBPVP8 '


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.parametrize(
    ('image_bytes', 'echoed_output_format', 'expected_media_type', 'expected_output_format'),
    [
        pytest.param(TINY_PNG, 'webp', 'image/png', 'png', id='webp-echo-lies-png-bytes-win'),
        pytest.param(TINY_PNG, 'png', 'image/png', 'png', id='png-echo-and-bytes-agree'),
        pytest.param(_JPEG_MAGIC_BYTES, 'jpeg', 'image/jpeg', 'jpeg', id='jpeg-sniffed'),
        pytest.param(_WEBP_MAGIC_BYTES, 'webp', 'image/webp', 'webp', id='webp-sniffed'),
        pytest.param(b'not really an image', 'webp', 'image/webp', 'webp', id='unknown-bytes-fall-back-to-echo'),
        pytest.param(b'not really an image', 'jpeg', 'image/jpeg', 'jpeg', id='unknown-bytes-fall-back-to-jpeg-echo'),
    ],
)
async def test_openai_media_type_reflects_actual_bytes(
    image_bytes: bytes,
    echoed_output_format: str,
    expected_media_type: str,
    expected_output_format: str,
):
    """`media_type` and `output_format` come from the returned bytes, not the provider's echo.

    gpt-image-2 silently ignores `output_format='webp'` and returns PNG bytes while the response still
    echoes `output_format: webp`, so trusting the echo makes `GeneratedImage.content.media_type` lie and
    breaks downstream content-type handling. We sniff PNG/JPEG/WebP magic bytes and prefer the sniffed
    type, keeping `content.media_type` and `output_format` consistent with each other; the echo is used
    only as a fallback for bytes we can't recognize.

    https://github.com/openai/openai-node/issues/1850
    See `local-notes/image-gen-research/openai-images-api.md` §1c, §7.
    """
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    mock_client.images.generate.return_value = ImagesResponse.model_construct(
        data=[Image.model_construct(b64_json=base64.b64encode(image_bytes).decode())],
        output_format=echoed_output_format,
    )
    model = OpenAIImageGenerationModel(
        'gpt-image-2',
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)),
    )

    result = await model.generate('a robot')

    image = result.images[0]
    assert image.content.media_type == expected_media_type
    assert image.content.data == image_bytes
    assert image.output_format == expected_output_format


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_usage_falls_back_to_sdk_totals(monkeypatch: pytest.MonkeyPatch):
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    mock_client.images.generate.return_value = ImagesResponse.model_construct(
        data=[Image.model_construct(b64_json='aGVsbG8=')],
        usage=Usage.model_construct(
            input_tokens=3,
            input_tokens_details=UsageInputTokensDetails.model_construct(text_tokens=3, image_tokens=0),
            output_tokens=5,
            output_tokens_details=None,
            total_tokens=8,
        ),
    )
    monkeypatch.setattr(openai_images.RequestUsage, 'extract', MagicMock(return_value=RequestUsage()))
    model = OpenAIImageGenerationModel(
        'gpt-image-1',
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)),
    )

    result = await model.generate('tiny robot')

    assert result.usage == RequestUsage(
        input_tokens=3,
        output_tokens=5,
        details={'input_text_tokens': 3, 'input_image_tokens': 0},
    )

    extracted_usage = RequestUsage(input_tokens=9, output_tokens=7)
    monkeypatch.setattr(openai_images.RequestUsage, 'extract', MagicMock(return_value=extracted_usage))

    extracted_result = await model.generate('another tiny robot')

    assert extracted_result.usage == extracted_usage


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_gpt_image_2_resolves_dimensions_and_aspect_ratio():
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    mock_client.images.generate.return_value = ImagesResponse.model_construct(
        data=[Image.model_construct(b64_json='aGVsbG8=')], output_format='png'
    )
    model = OpenAIImageGenerationModel(
        'gpt-image-2',
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)),
    )

    await model.generate('wide image', settings={'dimensions': (2048, 1152)})
    assert mock_client.images.generate.await_args.kwargs['size'] == '2048x1152'

    await model.generate('wide ratio', settings={'aspect_ratio': '16:9'})
    assert mock_client.images.generate.await_args.kwargs['size'] == '1280x720'

    await model.generate('compatibility size', settings={'size': '2048x1152', 'aspect_ratio': '16:9'})
    assert mock_client.images.generate.await_args.kwargs['size'] == '2048x1152'

    mock_client.images.generate.reset_mock()
    with pytest.raises(UserError, match='height must be multiples of 16'):
        await model.generate('invalid dimensions', settings={'dimensions': (1920, 1080)})
    mock_client.images.generate.assert_not_awaited()

    with pytest.raises(UserError, match='height must be multiples of 16'):
        await model.generate(
            'invalid overridden dimensions',
            settings=OpenAIImageGenerationSettings(dimensions=(1920, 1080), openai_size='1920x1080'),
        )
    mock_client.images.generate.assert_not_awaited()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.parametrize(
    'model_name,settings',
    [
        ('gpt-image-2', OpenAIImageGenerationSettings(background='transparent')),
        (
            'gpt-image-2-2026-04-21',
            OpenAIImageGenerationSettings(background='opaque', openai_background='transparent'),
        ),
    ],
)
async def test_openai_gpt_image_2_rejects_transparent_background(
    model_name: str, settings: OpenAIImageGenerationSettings
):
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    model = OpenAIImageGenerationModel(
        model_name,
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)),
    )

    with pytest.raises(UserError, match='does not support `background="transparent"`'):
        await model.generate('transparent image', settings=settings)

    mock_client.images.generate.assert_not_awaited()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.parametrize(
    'settings',
    [
        OpenAIImageGenerationSettings(input_fidelity='low'),
        OpenAIImageGenerationSettings(openai_input_fidelity='high'),
    ],
)
async def test_openai_gpt_image_2_ignores_input_fidelity_on_edit(settings: OpenAIImageGenerationSettings):
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    mock_client.images.edit.return_value = ImagesResponse.model_construct(
        data=[Image.model_construct(b64_json='ZWRpdGVk')], output_format='png'
    )
    model = OpenAIImageGenerationModel(
        'gpt-image-2',
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)),
    )

    with pytest.warns(UserWarning, match=r'ignored unsupported settings: `input_fidelity`'):
        await model.generate(
            'edit this image',
            images=[BinaryImage(data=b'image', media_type='image/png')],
            settings=settings,
        )

    mock_client.images.edit.assert_awaited_once()
    assert mock_client.images.edit.await_args.kwargs['input_fidelity'] is openai_images.OMIT


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_rejects_transparent_background_with_jpeg_edit():
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    model = OpenAIImageGenerationModel(
        'gpt-image-1.5',
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client)),
    )

    with pytest.raises(UserError, match='require `output_format="png"` or `"webp"`'):
        await model.generate(
            'transparent edit',
            images=[BinaryImage(data=b'image', media_type='image/png')],
            settings=OpenAIImageGenerationSettings(background='transparent', output_format='jpeg'),
        )

    mock_client.images.edit.assert_not_awaited()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.vcr
async def test_openai_gpt_image_2_generation_vcr(openai_api_key: str):
    provider = OpenAIProvider(api_key=openai_api_key)
    model = OpenAIImageGenerationModel('gpt-image-2', provider=provider)

    result = await model.generate(
        'A cat with a cowboy hat, dancing in Rome.',
        settings=OpenAIImageGenerationSettings(
            dimensions=(1280, 720),
            output_format='jpeg',
            output_compression=10,
            quality='low',
        ),
    )

    assert len(result.images) == 1
    generated_image = result.images[0]
    assert generated_image.content.media_type == 'image/jpeg'
    assert len(generated_image.content.data) > 100
    assert generated_image.output_format == 'jpeg'
    assert generated_image.size == '1280x720'
    assert result.model_name == 'gpt-image-2'
    assert result.provider_name == 'openai'
    assert result.provider_url == 'https://api.openai.com/v1/'
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.vcr
async def test_openai_gpt_image_2_webp_generation_vcr(openai_api_key: str):
    """gpt-image-2 honors `output_format='webp'` and the media type is sniffed from the real bytes.

    A live regression guard for the sniff success path against a real provider response: the returned
    payload is real WebP (`RIFF....WEBP`), so `content.media_type` and `output_format` both reflect WebP.
    openai-node#1850 documents a historical case where gpt-image-2 downgraded webp to PNG while still
    echoing `output_format: webp`; sniffing the bytes keeps us correct whichever the provider does.

    https://github.com/openai/openai-node/issues/1850
    """
    provider = OpenAIProvider(api_key=openai_api_key)
    model = OpenAIImageGenerationModel('gpt-image-2', provider=provider)

    result = await model.generate(
        'A small red circle centered on a plain white background.',
        settings=OpenAIImageGenerationSettings(dimensions=(1024, 1024), output_format='webp', quality='low'),
    )

    assert len(result.images) == 1
    generated_image = result.images[0]
    assert generated_image.content.data[:4] == b'RIFF'
    assert generated_image.content.data[8:12] == b'WEBP'
    assert generated_image.content.media_type == 'image/webp'
    assert generated_image.output_format == 'webp'
    assert generated_image.size == '1024x1024'
    assert result.model_name == 'gpt-image-2'


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_response_without_image_data():
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    mock_client.images.generate.return_value = ImagesResponse.model_construct(created=123, data=[])
    with pytest.raises(UnexpectedModelBehavior, match='did not contain any images'):
        await model.generate('tiny robot')

    mock_client.images.generate.return_value = ImagesResponse.model_construct(
        created=123, data=[Image.model_construct(url='https://example.com/a.png')]
    )
    with pytest.raises(UnexpectedModelBehavior, match='base64 image data'):
        await model.generate('tiny robot')

    mock_client.images.generate.return_value = ImagesResponse.model_construct(
        created=123, data=[Image.model_construct(b64_json='!!!!')]
    )
    with pytest.raises(UnexpectedModelBehavior, match='valid base64 image data') as exc_info:
        await model.generate('tiny robot')

    assert exc_info.value.body is not None
    assert '!!!!' in exc_info.value.body


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_edit_request():
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    mock_client.images.edit.return_value = ImagesResponse.model_construct(
        created=456,
        data=[Image.model_construct(b64_json='ZWRpdGVk')],
        output_format='webp',
        quality='high',
        size='1024x1024',
    )
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    settings = OpenAIImageGenerationSettings(
        n=1,
        background='transparent',
        input_fidelity='low',
        moderation='auto',
        output_format='webp',
        output_compression=50,
        quality='low',
        size='1024x1536',
        aspect_ratio='3:2',
        extra_headers={'x-test': 'header'},
        extra_body={'provider_option': True},
        openai_size='1024x1024',
        openai_quality='high',
        openai_background='opaque',
        openai_input_fidelity='high',
        openai_moderation='low',
        openai_output_compression=80,
        openai_user='user-123',
    )
    with pytest.warns(
        UserWarning,
        match=r'ignored unsupported settings: `moderation`.*used provider-specific settings instead',
    ):
        result = await model.generate(
            'turn these into one image',
            images=[
                BinaryImage(data=b'first', media_type='image/png'),
                BinaryImage(data=b'second', media_type='image/jpeg'),
            ],
            settings=settings,
        )

    mock_client.images.generate.assert_not_awaited()
    mock_client.images.edit.assert_awaited_once()
    kwargs = mock_client.images.edit.await_args.kwargs
    assert kwargs['image'] == [
        ('image-0.png', b'first', 'image/png'),
        ('image-1.jpg', b'second', 'image/jpeg'),
    ]
    assert kwargs['prompt'] == 'turn these into one image'
    assert kwargs['model'] == 'gpt-image-1'
    assert kwargs['n'] == 1
    assert kwargs['size'] == '1024x1024'
    assert kwargs['output_format'] == 'webp'
    assert kwargs['quality'] == 'high'
    assert kwargs['background'] == 'opaque'
    assert kwargs['input_fidelity'] == 'high'
    assert kwargs['output_compression'] == 80
    assert kwargs['user'] == 'user-123'
    assert kwargs['extra_headers'] == {'x-test': 'header'}
    assert kwargs['extra_body'] == {'provider_option': True}
    assert 'moderation' not in kwargs
    assert result.images[0].content == BinaryImage(data=b'edited', media_type='image/webp')
    assert result.provider_details == {'created': 456}


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_edit_wire_payload():
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'created': 456,
                'data': [{'b64_json': 'ZWRpdGVk'}],
                'output_format': 'png',
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    openai_client = AsyncOpenAI(api_key='test-api-key', base_url='https://example.com/v1', http_client=http_client)
    provider = OpenAIProvider(openai_client=openai_client)
    model = OpenAIImageGenerationModel('gpt-image-1.5', provider=provider)
    settings = OpenAIImageGenerationSettings(output_format='png', openai_input_fidelity='high', openai_moderation='low')

    try:
        with pytest.warns(UserWarning, match=r'ignored unsupported settings: `moderation`'):
            await model.generate(
                'replace the subject',
                images=[
                    BinaryImage(data=b'first-image', media_type='image/png'),
                    BinaryImage(data=b'second-image', media_type='image/webp'),
                ],
                settings=settings,
            )
    finally:
        await http_client.aclose()

    assert len(requests) == 1
    request = requests[0]
    assert request.method == 'POST'
    assert request.url.path == '/v1/images/edits'
    assert request.headers['content-type'].startswith('multipart/form-data; boundary=')
    body = request.content
    assert b'name="prompt"' in body
    assert b'replace the subject' in body
    assert b'name="model"' in body
    assert b'gpt-image-1.5' in body
    assert b'name="input_fidelity"' in body
    assert b'high' in body
    assert b'name="output_format"' in body
    assert b'filename="image-0.png"' in body
    assert b'Content-Type: image/png' in body
    assert b'filename="image-1.webp"' in body
    assert b'Content-Type: image/webp' in body
    assert body.index(b'first-image') < body.index(b'second-image')
    assert b'name="moderation"' not in body


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.vcr
async def test_openai_image_edit_vcr(openai_api_key: str, assets_path: Path):
    provider = OpenAIProvider(api_key=openai_api_key)
    model = OpenAIImageGenerationModel('gpt-image-1.5', provider=provider)
    reference_image = BinaryImage(data=(assets_path / 'kiwi.jpg').read_bytes(), media_type='image/jpeg')
    settings = OpenAIImageGenerationSettings(
        n=1,
        output_format='jpeg',
        openai_size='1024x1024',
        openai_quality='low',
        openai_input_fidelity='low',
        openai_output_compression=100,
    )

    result = await model.generate(
        'Place this kiwi fruit on a plain white studio background.',
        images=[reference_image],
        settings=settings,
    )

    assert len(result.images) == 1
    generated_image = result.images[0]
    assert generated_image.content.media_type == 'image/jpeg'
    assert len(generated_image.content.data) > 100
    assert generated_image.size == '1024x1024'
    assert generated_image.quality == 'low'
    assert generated_image.output_format == 'jpeg'
    assert result.prompt == 'Place this kiwi fruit on a plain white studio background.'
    assert result.model_name == 'gpt-image-1.5'
    assert result.provider_name == 'openai'
    assert result.provider_url == 'https://api.openai.com/v1/'
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.provider_details is not None
    assert result.provider_details.get('created')


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_edit_downloads_image_url(monkeypatch: pytest.MonkeyPatch):
    download_mock = AsyncMock(return_value={'data': b'downloaded', 'data_type': 'image/webp'})
    monkeypatch.setattr(openai_images, 'download_item', download_mock)

    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    mock_client.images.edit.return_value = ImagesResponse.model_construct(
        data=[Image.model_construct(b64_json='ZWRpdGVk')], output_format='png'
    )
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)
    image_url = ImageUrl('https://example.com/reference.png')

    await model.generate('edit this image', images=[image_url])

    download_mock.assert_awaited_once_with(image_url, data_format='bytes')
    assert mock_client.images.edit.await_args.kwargs['image'] == [('image-0.webp', b'downloaded', 'image/webp')]


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.parametrize(
    ('uploaded_file', 'error_message'),
    [
        (
            UploadedFile(file_id='file-openai', provider_name='openai', media_type='image/png'),
            'requires file content.*does not accept `UploadedFile.file_id`',
        ),
        (
            UploadedFile(file_id='file-anthropic', provider_name='anthropic', media_type='image/png'),
            "provider_name='anthropic'.*Expected `provider_name` to be `'openai'`",
        ),
    ],
)
async def test_openai_image_edit_rejects_uploaded_file(uploaded_file: UploadedFile, error_message: str):
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    with pytest.raises(UserError, match=error_message):
        await model.generate('edit this image', images=[uploaded_file])

    mock_client.images.generate.assert_not_awaited()
    mock_client.images.edit.assert_not_awaited()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_edit_rejects_unsupported_image_format():
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    with pytest.raises(UserError, match=r'only supports PNG, JPEG, or WebP.*image/gif'):
        await model.generate('edit this image', images=[BinaryImage(data=b'gif', media_type='image/gif')])

    mock_client.images.edit.assert_not_awaited()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_edit_status_error():
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    mock_client.images.edit.side_effect = APIStatusError(
        'test error',
        response=httpx.Response(status_code=500, request=httpx.Request('POST', 'https://example.com/v1/images/edits')),
        body={'error': 'test error'},
    )
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    with pytest.raises(ModelHTTPError) as exc_info:
        await model.generate('edit this image', images=[BinaryImage(data=TINY_PNG, media_type='image/png')])

    assert exc_info.value.status_code == 500
    assert exc_info.value.body == {'error': 'test error'}


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_connection_error():
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    mock_client.images.generate.side_effect = APIConnectionError(
        message='connection failed', request=httpx.Request('POST', 'https://example.com/v1/images/generations')
    )
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    with pytest.raises(ModelAPIError, match='connection failed'):
        await model.generate('generate this image')


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_rate_limited():
    """A 429 rate-limit surfaces as `ModelHTTPError` with the status and body preserved.

    Image models are rate-limited by images/min and images/day, and a Tier-1 org (~5 images/min) can hit
    the limit before its first successful generation, so this is a common first-call failure, not an edge case.

    See `local-notes/image-gen-research/openai-images-api.md` §4 (rate limits) and
    https://platform.openai.com/docs/guides/rate-limits.
    """
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    rate_limit_body = {'error': {'code': 'rate_limit_exceeded', 'type': 'requests', 'message': 'Rate limit reached'}}
    mock_client.images.generate.side_effect = APIStatusError(
        'Rate limit reached',
        response=httpx.Response(
            status_code=429, request=httpx.Request('POST', 'https://example.com/v1/images/generations')
        ),
        body=rate_limit_body,
    )
    model = OpenAIImageGenerationModel(
        'gpt-image-1', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        await model.generate('a robot')

    assert exc_info.value.status_code == 429
    assert exc_info.value.body == rate_limit_body
    mock_client.images.generate.assert_awaited_once()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_moderation_blocked():
    """A `moderation_blocked` 400 keeps its structured body and is never auto-retried.

    OpenAI returns HTTP 400 with `error.code == 'moderation_blocked'` and a `moderation_details` object
    (`moderation_stage`, `categories`). The wrapper must preserve that structure as data — so callers can
    branch on the code and inspect the categories — rather than flattening it into a string. A moderation
    block reflects the prompt, so retrying the identical request is wrong; we assert a single attempt.

    See `local-notes/image-gen-research/openai-images-api.md` §4 (content policy / moderation) and
    https://platform.openai.com/docs/guides/image-generation.
    """
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    moderation_body = {
        'error': {
            'code': 'moderation_blocked',
            'type': 'image_generation_user_error',
            'message': 'Your request was rejected as a result of our safety system.',
            'moderation_details': {'moderation_stage': 'input', 'categories': ['violence', 'self-harm']},
        }
    }
    mock_client.images.generate.side_effect = APIStatusError(
        'moderation_blocked',
        response=httpx.Response(
            status_code=400, request=httpx.Request('POST', 'https://example.com/v1/images/generations')
        ),
        body=moderation_body,
    )
    model = OpenAIImageGenerationModel(
        'gpt-image-1', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        await model.generate('a blocked prompt')

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == snapshot(
        {
            'error': {
                'code': 'moderation_blocked',
                'type': 'image_generation_user_error',
                'message': 'Your request was rejected as a result of our safety system.',
                'moderation_details': {'moderation_stage': 'input', 'categories': ['violence', 'self-harm']},
            }
        }
    )
    mock_client.images.generate.assert_awaited_once()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_does_not_override_timeout():
    """The adapter never smuggles its own request timeout into the OpenAI SDK call.

    Generations run 30-130s against a ~180s infra ceiling, and users configure timeouts on the client they
    pass in. Injecting a per-request `timeout` would silently override the user's client and truncate long
    generations, so the contract is: we forward no `timeout` for either generate or edit.

    See `local-notes/image-gen-research/openai-images-api.md` §4 (timeouts / long generations).
    """
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    mock_client.images.generate.return_value = ImagesResponse.model_construct(
        data=[Image.model_construct(b64_json=base64.b64encode(TINY_PNG).decode())], output_format='png'
    )
    mock_client.images.edit.return_value = ImagesResponse.model_construct(
        data=[Image.model_construct(b64_json=base64.b64encode(TINY_PNG).decode())], output_format='png'
    )
    model = OpenAIImageGenerationModel(
        'gpt-image-1', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    )

    await model.generate('a robot')
    assert 'timeout' not in mock_client.images.generate.await_args.kwargs

    await model.generate('edit this', images=[BinaryImage(data=TINY_PNG, media_type='image/png')])
    assert 'timeout' not in mock_client.images.edit.await_args.kwargs


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_tolerates_unknown_response_fields():
    """Unknown top-level and per-image response fields are tolerated (SDK forward compat).

    Providers add response fields without notice; the wrapper must parse successfully and still return the
    images rather than choking on fields it doesn't model. Recorded through the real SDK over a mock
    transport so the SDK's own (extra-allowing) parsing is exercised, not a hand-built response object.
    """
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'created': 123,
                'output_format': 'png',
                'data': [{'b64_json': base64.b64encode(TINY_PNG).decode(), 'unexpected_image_field': 'ignored'}],
                'unexpected_top_level_field': {'nested': True},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    openai_client = AsyncOpenAI(api_key='test-api-key', base_url='https://example.com/v1', http_client=http_client)
    model = OpenAIImageGenerationModel('gpt-image-1', provider=OpenAIProvider(openai_client=openai_client))

    try:
        result = await model.generate('a robot')
    finally:
        await http_client.aclose()

    assert len(requests) == 1
    assert result.images[0].content == BinaryImage(data=TINY_PNG, media_type='image/png')


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_supported_settings_emit_no_warning():
    """A fully-supported settings combination emits no warning.

    Over-warning erodes the signal of the warning channel; warnings are reserved for settings a request
    genuinely ignores or overrides. Every setting here is supported by `gpt-image-1` generation, so the call
    must be silent.
    """
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    mock_client.images.generate.return_value = ImagesResponse.model_construct(
        data=[Image.model_construct(b64_json=base64.b64encode(TINY_PNG).decode())], output_format='png'
    )
    model = OpenAIImageGenerationModel(
        'gpt-image-1', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, mock_client))
    )
    settings = OpenAIImageGenerationSettings(
        n=1,
        size='1024x1024',
        quality='high',
        background='opaque',
        moderation='low',
        output_format='png',
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        await model.generate('a robot', settings=settings)


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation(capfire: CaptureLogfire):
    reference_url = 'https://example.com/private-reference.png'
    provider_file_id = 'private-provider-file-id'
    generator = ImageGenerator(TestImageGenerationModel(), instrument=True)
    await generator.generate(
        'tiny robot',
        images=[
            ImageUrl(reference_url),
            BinaryImage(data=TINY_PNG, media_type='image/png'),
            UploadedFile(file_id=provider_file_id, provider_name='openai', media_type='image/png'),
        ],
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
                'gen_ai.output.type': 'image',
                'gen_ai.provider.name': 'test',
                'gen_ai.request.model': 'test',
                'prompt_length': 10,
                'input_image_count': 3,
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
    assert provider_file_id not in str(span)
    assert 'operation.cost' not in span['attributes']

    metrics = capfire.get_collected_metrics()
    assert [metric['name'] for metric in metrics] == ['gen_ai.client.token.usage']
    data_points = metrics[0]['data']['data_points']
    assert len(data_points) == 1
    assert data_points[0]['attributes'] == {
        'gen_ai.provider.name': 'test',
        'gen_ai.operation.name': 'image_generation',
        'gen_ai.request.model': 'test',
        'gen_ai.response.model': 'test',
        'gen_ai.token.type': 'input',
    }
    assert data_points[0]['sum'] == 2


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation_records_complete_response_metrics(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch
):
    wrapped = TestImageGenerationModel()
    result = ImageGenerationResult(
        images=[
            GeneratedImage(
                content=BinaryImage(data=TINY_PNG, media_type='image/png'),
                size='1024x1024',
                quality='high',
                output_format='png',
                background='transparent',
            )
        ],
        prompt='tiny robot',
        model_name='response-model',
        provider_name='test',
        usage=RequestUsage(output_tokens=3),
    )
    monkeypatch.setattr(wrapped, 'generate', AsyncMock(return_value=result))
    monkeypatch.setattr(type(wrapped), 'base_url', property(lambda _: 'relative/path'))
    assert InstrumentedImageGenerationModel.model_attributes(wrapped) == {
        'gen_ai.provider.name': 'test',
        'gen_ai.request.model': 'test',
    }
    monkeypatch.setattr(type(wrapped), 'base_url', property(lambda _: 'https://example.com/v1'))
    model = InstrumentedImageGenerationModel(wrapped)
    price = cast(PriceCalculation, SimpleNamespace(total_price=Decimal('0.25')))
    monkeypatch.setattr(model, '_price_calculation', MagicMock(return_value=price))

    await model.generate('tiny robot')

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(span for span in spans if 'image_generation' in span['name'])
    attributes = span['attributes']
    assert attributes['server.address'] == 'example.com'
    assert attributes['gen_ai.usage.output_tokens'] == 3
    assert attributes['gen_ai.response.model'] == 'response-model'
    assert attributes['image.0.size'] == '1024x1024'
    assert attributes['image.0.quality'] == 'high'
    assert attributes['image.0.output_format'] == 'png'
    assert attributes['image.0.background'] == 'transparent'
    assert attributes['operation.cost'] == 0.25
    assert 'gen_ai.response.id' not in attributes

    metrics = capfire.get_collected_metrics()
    assert [metric['name'] for metric in metrics] == [
        'gen_ai.client.token.usage',
        'operation.cost',
    ]
    assert metrics[0]['data']['data_points'][0]['attributes']['gen_ai.token.type'] == 'output'
    assert metrics[0]['data']['data_points'][0]['sum'] == 3
    assert metrics[1]['data']['data_points'][0]['sum'] == 0.25

    sparse_result = ImageGenerationResult(
        images=[GeneratedImage(content=BinaryImage(data=TINY_PNG, media_type='image/png'))],
        prompt='tiny robot',
        model_name='response-model',
        provider_name='test',
    )
    sparse_attributes = model._response_attributes(  # pyright: ignore[reportPrivateUsage]
        sparse_result, 'response-model', None
    )
    assert 'image.0.size' not in sparse_attributes
    assert 'image.0.output_format' not in sparse_attributes

    with model._instrument('unfinished request', [], None):  # pyright: ignore[reportPrivateUsage]
        pass


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_instrumentation_does_not_record_metrics_when_generation_fails(capfire: CaptureLogfire):
    wrapped = TestImageGenerationModel()
    wrapped.generate = AsyncMock(side_effect=RuntimeError('generation failed'))
    model = InstrumentedImageGenerationModel(wrapped)

    with pytest.raises(RuntimeError, match='generation failed'):
        await model.generate('tiny robot')

    assert capfire.metrics_reader.get_metrics_data() is None


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
async def test_instrumentation_respects_content_and_request_parameter_flags(capfire: CaptureLogfire):
    generator = ImageGenerator(
        TestImageGenerationModel(),
        instrument=InstrumentationSettings(
            include_content=False,
            include_binary_content=False,
            include_model_request_parameters=False,
        ),
    )
    await generator.generate('tiny robot', settings={'n': 1})

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(span for span in spans if 'image_generation' in span['name'])
    attributes = span['attributes']

    assert 'prompt' not in attributes
    assert 'image_generation_settings' not in attributes
    assert 'image_generation_settings' not in attributes['logfire.json_schema']['properties']
    assert 'image.0.media_type' in attributes
    assert 'data' not in str(span)
