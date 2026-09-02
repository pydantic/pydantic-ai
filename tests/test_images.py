from __future__ import annotations

import base64
import json
import os
import re
import warnings
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast, get_args
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest
from genai_prices.types import PriceCalculation

import pydantic_ai.images as images_module
import pydantic_ai.images._google_geometry as google_geometry
import pydantic_ai.images._openai_geometry as openai_geometry
from pydantic_ai import (
    BinaryImage,
    GeneratedImage,
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
    ImageGenerationAspectRatio,
    ImageGenerationInput,
    ImageGenerationModel,
    ImageGenerationSettings,
    InstrumentedImageGenerationModel,
    KnownImageGenerationModelName,
    TestImageGenerationModel,
    WrapperImageGenerationModel,
    infer_image_generation_model,
    merge_image_generation_settings,
)
from pydantic_ai.messages import UploadedFileProviderName
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInt, IsStr, TestEnv, try_import

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.usefixtures('allow_model_requests'),
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
    from pydantic_ai.providers.google import BaseGoogleProvider, GoogleCloudLocation, GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

with try_import() as xai_imports_successful:
    import grpc
    from xai_sdk import AsyncClient as XaiAsyncClient
    from xai_sdk.aio.image import ImageResponse as XaiImageResponse
    from xai_sdk.proto import image_pb2 as xai_image_pb2, usage_pb2 as xai_usage_pb2

    import pydantic_ai.images._xai_geometry as xai_geometry
    import pydantic_ai.images.xai as xai_images
    from pydantic_ai.images.xai import XaiImageGenerationModel, XaiImageGenerationSettings
    from pydantic_ai.providers.xai import XaiProvider
    from tests.models.xai_proto_cassettes import ImageMethodInteraction, XaiProtoCassette, XaiProtoCassetteClient

TINY_PNG = (
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
    b'\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01'
    b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
)


async def test_image_generator_with_test_model():
    test_model = TestImageGenerationModel()
    generator = ImageGenerator(test_model)

    result = await generator.generate('tiny robot')

    assert result == snapshot(
        ImageGenerationResult(
            images=[
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
            settings={},
            provider_response_id=IsStr(),
        )
    )
    assert test_model.last_settings == {}


async def test_test_image_generation_model_generates_png():
    result = await TestImageGenerationModel().generate('tiny robot')

    generated_image = result.images[0]
    assert generated_image.content.media_type == 'image/png'
    assert generated_image.content.data.startswith(b'\x89PNG\r\n\x1a\n')
    assert generated_image.output_format == 'png'
    assert result.image == generated_image.content


@pytest.mark.parametrize(
    'build_target',
    [
        pytest.param(
            lambda: OpenAIImageGenerationModel('gpt-image-2', provider=OpenAIProvider(api_key='test-key')),
            id='openai-model',
            marks=pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed'),
        ),
        pytest.param(
            lambda: GoogleImageGenerationModel('gemini-2.5-flash-image', provider=GoogleProvider(api_key='test-key')),
            id='google-model',
            marks=pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed'),
        ),
        pytest.param(
            lambda: XaiImageGenerationModel('grok-imagine-image', provider=XaiProvider(api_key='test-key')),
            id='xai-model',
            marks=pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed'),
        ),
        pytest.param(
            lambda: ImageGenerator(
                OpenAIImageGenerationModel('gpt-image-2', provider=OpenAIProvider(api_key='test-key'))
            ),
            id='image-generator',
            marks=pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed'),
        ),
    ],
)
async def test_image_generation_blocks_requests_when_disabled(
    build_target: Callable[[], ImageGenerationModel | ImageGenerator],
):
    """Every entry point guards on `ALLOW_MODEL_REQUESTS` before reaching the provider.

    A guard that fires before the request is made can't be exercised through a VCR recording. The
    per-model cases prove each concrete `generate()` guards; the `ImageGenerator` case proves the
    wrapper chain (`ImageGenerator` -> `InstrumentedImageGenerationModel` / `WrapperImageGenerationModel`
    -> concrete model) surfaces the `RuntimeError` rather than swallowing or wrapping it.
    """
    target = build_target()

    with override_allow_model_requests(False):
        with pytest.raises(RuntimeError, match='Model requests are not allowed'):
            await target.generate('a robot')


async def test_test_image_generation_model_is_exempt_from_request_guard():
    """`ALLOW_MODEL_REQUESTS`'s docstring promises `TestImageGenerationModel` is unaffected; pin that promise.

    Without this, adding the guard to `TestImageGenerationModel.generate` would break every user's
    test suite while this file still passed, since nothing else here runs with the flag off.
    """
    generator = ImageGenerator(TestImageGenerationModel())

    with override_allow_model_requests(False):
        result = await generator.generate('a robot')

    assert result.images[0].content.media_type == 'image/png'


def test_images_module_exports_image_generator():
    assert 'ImageGenerator' in images_module.__all__


async def test_image_generator_settings_precedence():
    test_model = TestImageGenerationModel(settings={'dimensions': (1024, 1024), 'extra_body': {'quality': 'high'}})
    generator = ImageGenerator(test_model, settings={'dimensions': (512, 512), 'extra_body': {'quality': 'low'}})

    result = await generator.generate('tiny robot', settings={'extra_body': {'quality': 'auto'}})

    expected_settings: ImageGenerationSettings = {'dimensions': (512, 512), 'extra_body': {'quality': 'auto'}}
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


@pytest.mark.parametrize('model', ['anthropic:claude-sonnet-4-5', 'cohere:embed-v4.0'])
def test_infer_image_generation_model_rejects_provider_without_image_support(model: str):
    """The provider resolves before dispatch, so the error names it rather than calling the model unknown."""
    provider_name = model.split(':', maxsplit=1)[0]
    with pytest.raises(UserError, match=f"Provider '{provider_name}' does not support direct image generation"):
        infer_image_generation_model(model, provider_factory=lambda name: MagicMock())


def test_infer_image_generation_model_rejects_unsupported_openai_gateway_route():
    """The gateway route is rejected before the provider resolves, so no gateway credentials are needed."""
    with pytest.raises(UserError) as exc_info:
        infer_image_generation_model('gateway/openai:gpt-image-2', provider_factory=lambda name: MagicMock())

    assert str(exc_info.value) == snapshot(
        "Image generation provider 'gateway/openai' cannot be routed through the Pydantic AI Gateway. The supported gateway route is `gateway/google`."
    )


def test_infer_image_generation_model_rejects_unsupported_xai_gateway_route():
    """`gateway_provider('xai')` raises on its own; the route check runs first so the error names the fix."""
    with pytest.raises(UserError) as exc_info:
        infer_image_generation_model('gateway/xai:grok-imagine-image', provider_factory=lambda name: MagicMock())

    assert str(exc_info.value) == snapshot(
        "Image generation provider 'gateway/xai' cannot be routed through the Pydantic AI Gateway. The supported gateway route is `gateway/google`."
    )


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
def test_infer_image_generation_model_google_cloud(env: TestEnv):
    """`google-cloud:` is Vertex AI, and `system` reports the provider's own name."""
    for name in ('GOOGLE_APPLICATION_CREDENTIALS', 'GOOGLE_CLOUD_PROJECT', 'GOOGLE_CLOUD_LOCATION', 'GEMINI_API_KEY'):
        env.remove(name)
    env.set('GOOGLE_API_KEY', 'mock-api-key')

    model = infer_image_generation_model('google-cloud:gemini-3.1-flash-image')

    assert isinstance(model, GoogleImageGenerationModel)
    assert model.model_name == 'gemini-3.1-flash-image'
    assert model.system == 'google-cloud'


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
def test_infer_image_generation_model_gateway_google(env: TestEnv):
    """The gateway serves Gemini over its Vertex route, so `gateway/google` resolves to `google-cloud`."""
    env.set('PYDANTIC_AI_GATEWAY_API_KEY', 'test-api-key')
    env.set('PYDANTIC_AI_GATEWAY_BASE_URL', 'https://gateway.pydantic.dev/proxy')

    model = infer_image_generation_model('gateway/google:gemini-3.1-flash-image')

    assert isinstance(model, GoogleImageGenerationModel)
    assert model.model_name == 'gemini-3.1-flash-image'
    assert model.system == 'google-cloud'
    assert model.base_url == snapshot('https://gateway.pydantic.dev/proxy/google-vertex')


async def test_wrapper_image_generation_model_delegates_properties():
    wrapped = TestImageGenerationModel(settings={'dimensions': (1024, 1024)})
    model = WrapperImageGenerationModel(wrapped)

    result = await model.generate('tiny robot')

    assert result.model_name == 'test'
    assert model.model_name == wrapped.model_name
    assert model.system == wrapped.system
    assert model.settings == {'dimensions': (1024, 1024)}
    assert model.base_url is None


def test_image_generator_sync_forwards_reference_images():
    test_model = TestImageGenerationModel()
    generator = ImageGenerator(test_model)
    image = BinaryImage(data=TINY_PNG, media_type='image/png')

    generator.generate_sync('edit this image', images=[image])

    assert test_model.last_images == [image]


def test_image_generation_cost_is_unavailable_for_unpriced_models():
    """`TestImageGenerationModel` has no pricing data, so `cost()` surfaces `genai-prices`' `LookupError`.

    Models priced per generated image rather than per token are in the same position until
    genai-prices represents that unit; the token-priced case is covered below.
    """
    result = ImageGenerator(TestImageGenerationModel()).generate_sync('tiny robot')

    with pytest.raises(LookupError, match='Unable to find provider'):
        result.cost()


def test_image_generation_cost_is_calculated_for_token_priced_models():
    """Token-priced image models resolve through `genai-prices` like any other result type.

    Pins that `cost()` is wired to `calc_price` rather than raising unconditionally — the
    method shipped disabled while genai-prices lacked image pricing, and that gap has closed
    for the token-priced families.
    """
    result = ImageGenerationResult(
        images=[GeneratedImage(content=BinaryImage(data=TINY_PNG, media_type='image/png'))],
        prompt='tiny robot',
        model_name='gpt-image-1',
        provider_name='openai',
        usage=RequestUsage(input_tokens=100, output_tokens=1500),
    )

    price = result.cost()

    assert price.total_price == snapshot(Decimal('0.0605'))


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
    base: ImageGenerationSettings = {'dimensions': (1024, 1024), 'extra_body': {'provider_option': True}}
    overrides: ImageGenerationSettings = {'aspect_ratio': '16:9'}

    assert merge_image_generation_settings(base, overrides) == snapshot(
        {'dimensions': (1024, 1024), 'extra_body': {'provider_option': True}, 'aspect_ratio': '16:9'}
    )
    assert merge_image_generation_settings(None, overrides) == overrides
    assert merge_image_generation_settings(base, None) == base


@pytest.mark.parametrize(
    'settings',
    [
        {'dimensions': (1024, 1024), 'aspect_ratio': '1:1'},
    ],
)
async def test_image_generation_dimensions_are_mutually_exclusive(settings: ImageGenerationSettings):
    with pytest.raises(UserError, match='mutually exclusive'):
        await TestImageGenerationModel().generate('tiny robot', settings=settings)


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


def test_known_google_cloud_image_generation_model_names():
    """Vertex AI serves the same Gemini image models under the `google-cloud:` prefix."""
    known_names = get_args(KnownImageGenerationModelName.__value__)

    assert {name for name in known_names if name.startswith('google-cloud:')} == {
        'google-cloud:gemini-2.5-flash-image',
        'google-cloud:gemini-3-pro-image',
        'google-cloud:gemini-3.1-flash-image',
        'google-cloud:gemini-3.1-flash-lite-image',
    }


def test_known_xai_image_generation_model_names():
    """Pin the current curated xAI image models."""
    known_names = get_args(KnownImageGenerationModelName.__value__)

    assert {name for name in known_names if name.startswith('xai:')} == {
        'xai:grok-imagine-image',
        'xai:grok-imagine-image-quality',
    }


def _xai_cassette_response_bytes() -> bytes:
    response = xai_image_pb2.ImageResponse()
    response.images.add().base64 = 'dGlueSByb2JvdA=='
    return response.SerializeToString()


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
@pytest.mark.parametrize('method', ['sample', 'sample_batch'])
async def test_xai_proto_cassette_replay_validates_image_request_with_binary_placeholder(
    method: Literal['sample', 'sample_batch'],
):
    """A data URL replays against a placeholder recording the same kind and byte length."""
    cassette = XaiProtoCassette()
    cassette.interactions.append(
        ImageMethodInteraction(
            method=method,
            response_raw=_xai_cassette_response_bytes(),
            response_count=1,
            request_json={
                '_args': ["'tiny robot'", "'grok-imagine-image'"],
                'image_url': '<data URL len=38>',
                'image_format': 'base64',
            },
        )
    )
    client = XaiProtoCassetteClient(cassette)

    if method == 'sample':
        await client.image.sample(
            'tiny robot',
            'grok-imagine-image',
            image_url='data:image/png;base64,dGlueSByb2JvdA==',
            image_format='base64',
        )
    else:
        responses = await client.image.sample_batch(
            'tiny robot',
            'grok-imagine-image',
            image_url='data:image/png;base64,dGlueSByb2JvdA==',
            image_format='base64',
        )
        assert len(responses) == 1

    assert client.interaction_idx == 1


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_proto_cassette_replay_rejects_different_image_request():
    cassette = XaiProtoCassette()
    cassette.interactions.append(
        ImageMethodInteraction(
            method='sample',
            response_raw=_xai_cassette_response_bytes(),
            response_count=1,
            request_json={'_args': ["'tiny robot'", "'grok-imagine-image'"], 'image_format': 'base64'},
        )
    )
    client = XaiProtoCassetteClient(cassette)

    with pytest.raises(RuntimeError, match='Cassette request mismatch'):
        await client.image.sample('different robot', 'grok-imagine-image', image_format='base64')


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_proto_cassette_replay_rejects_binary_placeholder_of_different_length():
    """A swapped reference image is caught: the placeholder pins byte length, not just the kind."""
    cassette = XaiProtoCassette()
    cassette.interactions.append(
        ImageMethodInteraction(
            method='sample',
            response_raw=_xai_cassette_response_bytes(),
            response_count=1,
            request_json={
                '_args': ["'tiny robot'", "'grok-imagine-image'"],
                'image_url': '<data URL len=131455>',
                'image_format': 'base64',
            },
        )
    )
    client = XaiProtoCassetteClient(cassette)

    with pytest.raises(RuntimeError, match='Cassette request mismatch'):
        await client.image.sample(
            'tiny robot',
            'grok-imagine-image',
            image_url='data:image/png;base64,dGlueSByb2JvdA==',
            image_format='base64',
        )


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

    # `1:2` is absent from every Gemini geometry profile, yet still reaches the model: our tables say
    # which shapes we can name for `dimensions`, not what the API accepts.
    unprofiled_ratio = google_geometry.resolve_google_geometry(
        'gemini-2.5-flash-image',
        {'aspect_ratio': '1:2'},
        provider_aspect_ratio=None,
        provider_size=None,
        provider_size_is_set=False,
    )
    assert (unprofiled_ratio.aspect_ratio, unprofiled_ratio.image_size) == ('1:2', None)

    unknown_model_ratio = google_geometry.resolve_google_geometry(
        'future-image-model',
        {'aspect_ratio': '1:2'},
        provider_aspect_ratio=None,
        provider_size=None,
        provider_size_is_set=False,
    )
    assert (unknown_model_ratio.aspect_ratio, unknown_model_ratio.image_size) == ('1:2', None)

    tiered_model_ratio = google_geometry.resolve_google_geometry(
        'gemini-3-pro-image',
        {'aspect_ratio': '1:2'},
        provider_aspect_ratio=None,
        provider_size=None,
        provider_size_is_set=False,
    )
    assert (tiered_model_ratio.aspect_ratio, tiered_model_ratio.image_size) == ('1:2', '1K')

    assert google_geometry.resolve_google_dimensions('gemini-3-pro-image', (1024, 1024)) == ('1:1', '1K')
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
        {},
        provider_aspect_ratio=None,
        provider_size='2K',
        provider_size_is_set=True,
    )
    assert supported_size.image_size == '2K'
    with pytest.raises(UserError, match='does not support'):
        google_geometry.resolve_google_dimensions('future-image-model', (1024, 1024))


@pytest.mark.parametrize(
    ('aspect_ratio', 'tiers'),
    [
        ('1:4', {'512': (256, 1024), '1K': (512, 2064), '2K': (1024, 4128), '4K': (2048, 8256)}),
        ('1:8', {'512': (176, 1456), '1K': (352, 2928), '2K': (704, 5856), '4K': (1408, 11712)}),
        ('4:1', {'512': (1024, 256), '1K': (2064, 512), '2K': (4128, 1024), '4K': (8256, 2048)}),
        ('8:1', {'512': (1456, 176), '1K': (2928, 352), '2K': (5856, 704), '4K': (11712, 1408)}),
    ],
)
def test_google_extended_ratio_dimensions_match_the_live_api(aspect_ratio: str, tiers: dict[str, tuple[int, int]]):
    """The extended Gemini 3.1 ratios resolve to the shapes the models actually return.

    Google publishes different pixel sizes for most of these rows — `384x3072` for `1:8` at `1K`, where
    the model returns `352x2928` — and the rows do not scale uniformly from the `512` tier the way the
    standard ratios do, so every cell was probed against the live API. Flash Lite returns the same `1K`
    shapes as Flash and serves no other tier: the `512` column Google documents for it is rejected.
    """
    for image_size, dimensions in tiers.items():
        assert google_geometry.resolve_google_dimensions('gemini-3.1-flash-image', dimensions) == (
            aspect_ratio,
            image_size,
        )

    assert google_geometry.resolve_google_dimensions('gemini-3.1-flash-lite-image', tiers['1K']) == (
        aspect_ratio,
        '1K',
    )
    with pytest.raises(UserError, match='does not support'):
        google_geometry.resolve_google_dimensions('gemini-3.1-flash-lite-image', tiers['512'])


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


@asynccontextmanager
async def _mock_google_provider(
    handle_request: Callable[[httpx2.Request], httpx2.Response],
    *,
    build_provider: Callable[[google_types.HttpOptions], BaseGoogleProvider] | None = None,
) -> AsyncGenerator[BaseGoogleProvider]:
    """A Google provider whose transport is `handle_request`, closed when the block exits.

    `base_url` is pinned so `provider_url` is asserted against a value the test controls rather than
    the SDK's default endpoint. `build_provider` receives the wired `HttpOptions` and returns a
    provider built from a pre-built `client=`, for the constructions where the provider's name and its
    client's transport disagree.
    """
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle_request))
    try:
        if build_provider is None:
            yield GoogleProvider(api_key='test-api-key', base_url='https://example.com', http_client=http_client)
        else:
            yield build_provider(
                google_types.HttpOptions(base_url='https://example.com', httpx_async_client=http_client)
            )
    finally:
        await http_client.aclose()


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_wire_payload_and_response_mapping():
    requests: list[httpx2.Request] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
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

    settings = GoogleImageGenerationSettings(
        google_image_config={'aspect_ratio': '1:1'},
        extra_headers={'x-test-header': 'test-value'},
    )

    async with _mock_google_provider(handle_request) as provider:
        model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

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
            aspect_ratio='16:9',
            google_image_config={'aspect_ratio': '1:1'},
        )
        with pytest.warns(
            UserWarning,
            match=r'used provider-specific settings instead of: `aspect_ratio`',
        ):
            await model.generate('conflicting settings', settings=conflicting_settings)

        await model.generate(
            'provider-specific size',
            settings=GoogleImageGenerationSettings(google_image_config={'image_size': '4K'}),
        )

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
                cache_text_read_tokens=2,
                input_text_tokens=5,
                input_image_tokens=2,
                output_image_tokens=5,
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
                output_reasoning_tokens=2,
                input_tool_tokens=4,
                input_text_tool_tokens=4,
            ),
            settings=settings,
            provider_details={'finish_reason': 'STOP'},
            provider_response_id='response-123',
            provider_url='https://example.com',
        )
    )


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_requires_media_type_for_files_api_url():
    """A Files API `ImageUrl` without an explicit `media_type` fails with actionable guidance.

    Those URIs carry no file extension, so `ImageUrl.media_type` cannot infer one and raises a bare
    `ValueError` naming neither the branch nor the fix — on exactly the input this branch exists for.
    """
    provider = GoogleProvider(api_key='test-api-key')
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

    with pytest.raises(UserError, match='carry no file extension'):
        await model.generate(
            'edit this',
            images=[ImageUrl('https://generativelanguage.googleapis.com/v1beta/files/file-789')],
        )


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_resolves_dimensions_and_aspect_ratio():
    requests: list[httpx2.Request] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
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

    async with _mock_google_provider(handle_request) as provider:
        model = GoogleImageGenerationModel('gemini-3.1-flash-image', provider=provider)
        unknown_model = GoogleImageGenerationModel('future-image-model', provider=provider)

        await model.generate('wide image', settings={'dimensions': (1376, 768)})
        await model.generate('portrait image', settings={'aspect_ratio': '3:4'})
        # No Gemini geometry profile lists `1:2`, and it reaches the wire anyway: the profiles record
        # the shapes we can name for `dimensions`, and Gemini itself judges the ratio it is sent.
        await model.generate('unprofiled ratio', settings={'aspect_ratio': '1:2'})
        await unknown_model.generate('unknown model', settings={'aspect_ratio': '1:2'})
        await model.generate(
            'explicit tier',
            settings=GoogleImageGenerationSettings(aspect_ratio='16:9', google_image_config={'image_size': '4K'}),
        )
        with pytest.raises(UserError, match=r'does not support `dimensions=\(1920, 1080\)`'):
            await model.generate('unsupported dimensions', settings={'dimensions': (1920, 1080)})

    assert [json.loads(request.content)['generationConfig']['imageConfig'] for request in requests] == [
        {'aspectRatio': '16:9', 'imageSize': '1K'},
        {'aspectRatio': '3:4', 'imageSize': '1K'},
        {'aspectRatio': '1:2', 'imageSize': '1K'},
        {'aspectRatio': '1:2'},
        {'imageSize': '4K', 'aspectRatio': '16:9'},
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
    """
    requests: list[httpx2.Request] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
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

    async with _mock_google_provider(handle_request) as provider:
        model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            await model.generate('a robot', settings={'extra_body': {'labels': {'team': 'growth'}}})

    assert json.loads(requests[0].content)['labels'] == snapshot({'team': 'growth'})


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_downloads_image_url(monkeypatch: pytest.MonkeyPatch):
    download_mock = AsyncMock(return_value={'data': b'downloaded', 'data_type': 'image/webp'})
    monkeypatch.setattr(google_images, 'download_item', download_mock)
    requests: list[httpx2.Request] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
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

    image_url = ImageUrl('https://example.com/reference.png')

    async with _mock_google_provider(handle_request) as provider:
        model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

        await model.generate('edit this image', images=[image_url])

    download_mock.assert_awaited_once_with(image_url, data_format='bytes')
    body = json.loads(requests[0].content)
    assert body['contents'][0]['parts'][1] == {'inlineData': {'data': 'ZG93bmxvYWRlZA==', 'mimeType': 'image/webp'}}


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_cloud_image_generation_downloads_files_api_url(monkeypatch: pytest.MonkeyPatch):
    """On the Vertex transport a Files API URL is downloaded and inlined instead of sent as `fileData`.

    Only the Gemini Developer API resolves `generativelanguage.googleapis.com/v1beta/files` URIs, so the
    `fileData` shortcut is keyed on the transport like the `UploadedFile` rejection: forwarding the URL
    would hand Vertex a reference it cannot fetch.

    Built on the disagreeing pair — a Vertex client stored as-is by `GoogleProvider`, which keeps `name`
    `'google'` — so a name-keyed regression fails here. A `GoogleCloudProvider` would agree with its own
    transport and keep passing either way.

    The blob's key spelling is google-genai's serialization rather than ours, and it varies by version:
    `tests/models/cassettes/test_google/test_google_url_input_force_download.yaml` records a live Vertex
    200 for a request body carrying `inlineData.mimeType` (recorded under google-genai 1.70.0), while the
    pinned 2.18.0 emits `mime_type` for the same construction. The assertion reads either spelling; the
    coverage is that the downloaded `image/webp` wins over the URL's declared `image/png`.
    """
    download_mock = AsyncMock(return_value={'data': b'downloaded', 'data_type': 'image/webp'})
    monkeypatch.setattr(google_images, 'download_item', download_mock)
    requests: list[httpx2.Request] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
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

    image_url = ImageUrl('https://generativelanguage.googleapis.com/v1beta/files/abc123', media_type='image/png')

    async with _mock_google_provider(
        handle_request,
        build_provider=lambda http_options: GoogleProvider(
            client=GoogleClient(vertexai=True, api_key='test-api-key', http_options=http_options)
        ),
    ) as provider:
        model = GoogleImageGenerationModel('gemini-3.1-flash-image', provider=provider)
        assert model.system == 'google'

        await model.generate('edit this image', images=[image_url])

    download_mock.assert_awaited_once_with(image_url, data_format='bytes')
    body = json.loads(requests[0].content)
    blob = body['contents'][0]['parts'][1]['inlineData']
    assert (blob['data'], blob.get('mimeType') or blob.get('mime_type')) == ('ZG93bmxvYWRlZA==', 'image/webp')


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
@pytest.mark.parametrize(
    ('image_bytes', 'expected'),
    [
        pytest.param(b'\xff\xd8\xff\xe0jpeg-bytes', ('image/jpeg', 'jpeg'), id='sniffed-jpeg'),
        pytest.param(b'not-a-recognized-image', ('image/png', 'png'), id='unrecognized-falls-back-to-png'),
    ],
)
async def test_google_image_generation_media_type_without_mime_type(image_bytes: bytes, expected: tuple[str, str]):
    """When `Blob.mime_type` is absent the bytes decide the media type, with PNG only as a last resort.

    `mime_type` is optional on `Blob`, and the Gemini image family is not PNG-only: `gemini-2.5-flash-image`
    returns PNG while `gemini-3-pro-image-preview` returns JPEG (the `test_google_image_generation_vcr`
    cassette records `mimeType: image/jpeg`). An unconditional `image/png` default would therefore hand the
    caller JPEG bytes labelled as PNG. Bytes the sniffer doesn't recognize still fall back to `image/png`.

    Not a VCR test because the live API always sends `mimeType`; the absent-field response can only be
    produced by a stub.
    """

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                'candidates': [
                    {
                        'content': {
                            'parts': [{'inlineData': {'data': base64.b64encode(image_bytes).decode()}}],
                            'role': 'model',
                        },
                        'finishReason': 'STOP',
                    }
                ]
            },
        )

    async with _mock_google_provider(handle_request) as provider:
        model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

        result = await model.generate('a robot')

    generated_image = result.images[0]
    assert generated_image.content.data == image_bytes
    assert (generated_image.content.media_type, generated_image.output_format) == expected


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
            r"provider_name='openai'.*Expected `provider_name` to be one of \['google', 'google-gla'\]",
        ),
    ],
)
async def test_google_image_generation_rejects_invalid_uploaded_file(uploaded_file: UploadedFile, error_message: str):
    provider = GoogleProvider(api_key='test-api-key')
    model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

    with pytest.raises(UserError, match=error_message):
        await model.generate('edit this image', images=[uploaded_file])


_GEMINI_API_NAME_FAMILY_CASES: list[
    tuple[Callable[[google_types.HttpOptions], BaseGoogleProvider], UploadedFileProviderName, str]
] = [
    (
        lambda http_options: GoogleProvider(
            client=GoogleClient(vertexai=False, api_key='test-api-key', http_options=http_options)
        ),
        'google-gla',
        'google',
    ),
    (
        lambda http_options: GoogleCloudProvider(
            client=GoogleClient(vertexai=False, api_key='test-api-key', http_options=http_options)
        ),
        'google',
        'google-cloud',
    ),
]
"""Constructions whose provider name differs from the one the Gemini Files API stamps on a file.

Built lazily so the module still imports without `google-genai`, and annotated so the builders keep
their parameter type — `pytest.param` erases it.
"""


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
@pytest.mark.parametrize(
    ('build_provider', 'file_provider_name', 'expected_system'),
    _GEMINI_API_NAME_FAMILY_CASES,
    ids=['pre-v2-name-on-google-provider', 'gemini-api-client-in-google-cloud-provider'],
)
async def test_google_image_generation_accepts_the_gemini_api_provider_name_family(
    build_provider: Callable[[google_types.HttpOptions], BaseGoogleProvider],
    file_provider_name: UploadedFileProviderName,
    expected_system: str,
):
    """Every Gemini Developer API provider name is accepted, not just `system`.

    Matching on `system` alone rejects two files this transport can serve: `'google-gla'`, the pre-v2
    name still carried by persisted message history, and `'google'` — the name the Files API path
    stamps — on a `GoogleCloudProvider` holding a Gemini API client, whose `name` stays `'google-cloud'`
    while the bytes go to the only transport with a Files API. Mirrors `GoogleModel`'s
    `_matching_provider_names`.
    """
    requests: list[httpx2.Request] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
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

    uploaded_file = UploadedFile(
        file_id='https://generativelanguage.googleapis.com/v1beta/files/abc123',
        provider_name=file_provider_name,
        media_type='image/png',
    )

    async with _mock_google_provider(handle_request, build_provider=build_provider) as provider:
        model = GoogleImageGenerationModel('gemini-3.1-flash-image', provider=provider)
        assert model.system == expected_system

        result = await model.generate('edit this image', images=[uploaded_file])

    assert result.images[0].content == BinaryImage(data=b'hello', media_type='image/png')
    # The `fileData` wire shape is pinned by `test_google_image_generation_wire_payload_and_response_mapping`;
    # here the file id reaching the wire is what proves the name was accepted rather than rejected.
    assert uploaded_file.file_id in requests[0].content.decode()


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
@pytest.mark.parametrize(
    ('provider_factory', 'file_provider_name'),
    [
        pytest.param(
            lambda: GoogleCloudProvider(api_key='test-api-key'),
            'google-cloud',
            id='google-cloud-provider',
        ),
        pytest.param(
            lambda: GoogleCloudProvider(api_key='test-api-key'),
            'google',
            id='gemini-files-api-file-on-vertex',
        ),
        pytest.param(
            lambda: GoogleProvider(client=GoogleClient(vertexai=True, project='test-project', location='us-central1')),
            'google',
            id='vertex-client-in-google-provider',
        ),
    ],
)
async def test_google_cloud_image_generation_rejects_uploaded_file(
    provider_factory: Callable[[], BaseGoogleProvider], file_provider_name: UploadedFileProviderName
):
    """The Files API is unavailable on Vertex AI, so an `UploadedFile` is rejected instead of forwarded.

    The rejection is keyed on the client's transport rather than on `system`: `GoogleProvider` stores a
    pre-built Vertex client as-is and keeps `name` `'google'`, so a name-keyed check would forward the
    file as a `fileData` part Vertex cannot resolve. It also runs before the `provider_name` check, so a
    file uploaded through the Gemini Files API (`provider_name='google'`) gets this error rather than
    provider-mismatch advice that only leads back to it.
    """
    model = GoogleImageGenerationModel('gemini-3.1-flash-image', provider=provider_factory())
    uploaded_file = UploadedFile(
        file_id='https://generativelanguage.googleapis.com/v1beta/files/abc123',
        provider_name=file_provider_name,
        media_type='image/png',
    )

    with pytest.raises(UserError, match='The Gemini Files API is not available on Google Cloud'):
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

    Not a VCR test because a `NO_IMAGE` refusal cannot be provoked on demand, so the response is fixed
    from the finish reason the SDK documents.
    """

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
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

    async with _mock_google_provider(handle_request) as provider:
        model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

        with pytest.raises(
            UnexpectedModelBehavior, match=r'did not contain any images \(finish_reason: NO_IMAGE\)'
        ) as exc_info:
            await model.generate('tiny robot')

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

    Not a VCR test because a moderation block cannot be provoked on demand, so the response is fixed
    from the finish reason the SDK documents.
    """

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
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

    async with _mock_google_provider(handle_request) as provider:
        model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

        with pytest.raises(ContentFilterError, match=r'content moderation \(reason: IMAGE_SAFETY\)') as exc_info:
            await model.generate('tiny robot')

    assert exc_info.value.body is not None
    assert "'finish_reason': 'IMAGE_SAFETY'" in exc_info.value.body


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_prompt_blocked():
    """A prompt-level block (empty candidates + `promptFeedback.blockReason`) raises `ContentFilterError`.

    When Gemini blocks the prompt itself it returns no candidates and a `promptFeedback.blockReason`
    (e.g. `PROHIBITED_CONTENT`). This is a content-moderation outcome, so it raises `ContentFilterError`
    naming the block reason, with the block details preserved in the body.

    - `BlockedReason.PROHIBITED_CONTENT`: python-genai `google/genai/types.py` `BlockedReason`.

    Not a VCR test because a prompt-level moderation block cannot be provoked on demand, so the
    response is fixed from the block reason the SDK documents.
    """

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                'promptFeedback': {
                    'blockReason': 'PROHIBITED_CONTENT',
                    'blockReasonMessage': 'blocked by safety policy',
                },
            },
        )

    async with _mock_google_provider(handle_request) as provider:
        model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

        with pytest.raises(ContentFilterError, match=r'content moderation \(reason: PROHIBITED_CONTENT\)') as exc_info:
            await model.generate('tiny robot')

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

    Not a VCR test because a degenerate response cannot be provoked on demand, so both shapes are
    fixed from the candidate structure the SDK documents.
    """
    degenerate_responses: list[dict[str, object]] = [
        {'candidates': [{'content': {'parts': [], 'role': 'model'}, 'finishReason': 'STOP'}]},
        {'candidates': []},
    ]
    responses = iter(degenerate_responses)

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=next(responses))

    async with _mock_google_provider(handle_request) as provider:
        model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

        with pytest.raises(
            UnexpectedModelBehavior, match=r'did not contain any images \(finish_reason: STOP\)'
        ) as empty_parts:
            await model.generate('empty parts')
        with pytest.raises(UnexpectedModelBehavior, match=r'did not contain any images$') as no_candidates:
            await model.generate('no candidates')

    assert not isinstance(empty_parts.value, ContentFilterError)
    assert no_candidates.value.body is None


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_supported_settings_emit_no_warning():
    """A fully-supported Google settings combination emits no warning.

    Over-warning erodes the signal of the warning channel; warnings are reserved for settings a request
    genuinely ignores or overrides. A `google_image_config` aspect ratio, `extra_headers`, and `extra_body`
    are all honored by the adapter, so the call must be silent.

    - `warn_image_generation_settings` channel: `pydantic_ai/images/_validation.py`.
    """

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
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

    settings = GoogleImageGenerationSettings(
        extra_headers={'x-team': 'growth'},
        extra_body={'labels': {'team': 'growth'}},
        google_image_config={'aspect_ratio': '16:9'},
    )

    async with _mock_google_provider(handle_request) as provider:
        model = GoogleImageGenerationModel('gemini-3-pro-image', provider=provider)

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            await model.generate('a robot', settings=settings)


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
async def test_google_image_generation_maps_complete_provider_metadata():
    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
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

    async with _mock_google_provider(handle_request) as provider:
        model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

        result = await model.generate('tiny robot')

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
    """A Google `APIError` whose code is below 400 surfaces as `ModelAPIError`, not `ModelHTTPError`.

    Not a VCR test because a sub-400 `APIError` cannot be provoked on demand, so the error is fixed
    from the shape the SDK documents.
    """
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
    """A Gemini 4xx surfaces as `ModelHTTPError` keeping the status, body, and `retry-after`.

    Not a VCR test because a 400 is easy to provoke but its `retry-after` header and structured error
    body are not: both depend on Google's live quota state, and a recording would pin whatever that
    happened to be. The response here is fixed from Google's documented error format instead.
    """

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            400,
            headers={'retry-after': '12'},
            json={'error': {'code': 400, 'message': 'invalid image request', 'status': 'INVALID_ARGUMENT'}},
        )

    async with _mock_google_provider(handle_request) as provider:
        model = GoogleImageGenerationModel('gemini-2.5-flash-image', provider=provider)

        with pytest.raises(ModelHTTPError) as exc_info:
            await model.generate('tiny robot')

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {
        'error': {'code': 400, 'message': 'invalid image request', 'status': 'INVALID_ARGUMENT'}
    }
    assert exc_info.value.retry_after == 12


def _body_capturing_http_client() -> tuple[httpx2.AsyncClient, list[dict[str, Any]]]:
    """An `httpx2.AsyncClient` paired with the list of JSON bodies it sends.

    Request event hooks run inside `AsyncClient.send`, above the transport VCR patches, so the hook fires
    on replay too and sees the request the live code actually built. A cassette body is frozen and keeps
    replaying after the code stops sending a field, so it cannot show that drift.

    The Google SDK turns the client's timeout into a server-side deadline, and Gemini rejects anything
    under 10 seconds, so recording against the live API needs more than httpx's 5-second default.
    """
    sent_bodies: list[dict[str, Any]] = []

    async def capture_request(request: httpx2.Request) -> None:
        sent_bodies.append(json.loads(request.read()))

    return httpx2.AsyncClient(timeout=120, event_hooks={'request': [capture_request]}), sent_bodies


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
@pytest.mark.vcr
async def test_google_image_generation_vcr():
    """Live generation with `dimensions=(1024, 1024)`, with the resolved geometry pinned on the wire.

    `dimensions` is a common setting that only `_google_geometry` knows how to turn into Gemini's
    `imageConfig`, so the outbound body is snapshotted from a request hook: the response echoes nothing
    about the requested geometry, and a cassette assertion would keep passing if the config stopped
    being sent at all.
    """
    http_client, sent_bodies = _body_capturing_http_client()
    provider = GoogleProvider(
        api_key=os.getenv('GOOGLE_API_KEY', os.getenv('GEMINI_API_KEY', 'mock-api-key')), http_client=http_client
    )
    model = GoogleImageGenerationModel('gemini-3.1-flash-lite-image', provider=provider)

    try:
        result = await model.generate(
            'A cat with a cowboy hat, dancing in Rome.',
            settings=GoogleImageGenerationSettings(dimensions=(1024, 1024)),
        )
    finally:
        await http_client.aclose()

    assert sent_bodies == snapshot(
        [
            {
                'contents': [{'parts': [{'text': 'A cat with a cowboy hat, dancing in Rome.'}], 'role': 'user'}],
                'generationConfig': {
                    'responseModalities': ['IMAGE'],
                    'imageConfig': {'aspectRatio': '1:1', 'imageSize': '1K'},
                },
            }
        ]
    )

    assert len(result.images) == 1
    generated_image = result.images[0]
    # Gemini returned JPEG for a prompt that never asked for one, so these pin the real format
    # rather than a requested one. A loose `startswith('image/')` would pass even if media-type
    # resolution returned the wrong answer.
    assert generated_image.content.data[:4] == b'\xff\xd8\xff\xe0'
    assert generated_image.content.media_type == 'image/jpeg'
    assert generated_image.output_format == 'jpeg'
    # Gemini image responses carry a ~1.5 MB `thoughtSignature`; asserting it here is what makes
    # that payload load-bearing rather than dead weight in the recording.
    assert generated_image.provider_details == {'has_thought_signature': True}
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
    """Live edit with a provider-native `google_image_config`, pinned on the wire.

    Only `generationConfig` is snapshotted: `contents` carries the reference image as a ~130 KB base64
    blob, which would drown the assertion it belongs to. The hook is what makes the pin meaningful — it
    captures the request the client sent, so dropping `imageConfig` fails here instead of replaying green.
    """
    http_client, sent_bodies = _body_capturing_http_client()
    provider = GoogleProvider(
        api_key=os.getenv('GOOGLE_API_KEY', os.getenv('GEMINI_API_KEY', 'mock-api-key')), http_client=http_client
    )
    model = GoogleImageGenerationModel('gemini-3.1-flash-lite-image', provider=provider)

    try:
        result = await model.generate(
            'Transform the subject into a dog with a cowboy hat, dancing in Rome.',
            images=[image_content],
            settings=GoogleImageGenerationSettings(google_image_config={'aspect_ratio': '1:1'}),
        )
    finally:
        await http_client.aclose()

    assert [body['generationConfig'] for body in sent_bodies] == snapshot(
        [{'responseModalities': ['IMAGE'], 'imageConfig': {'aspectRatio': '1:1'}}]
    )

    assert len(result.images) == 1
    edited_image = result.images[0]
    assert edited_image.content.data[:4] == b'\xff\xd8\xff\xe0'
    assert edited_image.content.media_type == 'image/jpeg'
    assert edited_image.output_format == 'jpeg'
    assert edited_image.provider_details == {'has_thought_signature': True}
    assert result.model_name == 'gemini-3.1-flash-lite-image'
    assert result.provider_name == 'google'
    assert result.provider_url == 'https://generativelanguage.googleapis.com/'
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.provider_details == {'finish_reason': 'STOP'}
    assert result.provider_response_id


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
@pytest.mark.vcr
async def test_google_flash_lite_extended_aspect_ratio_vcr():
    """Live proof that `1:8` on Flash Lite returns `352x2928`, the shape `_google_geometry` records.

    Google publishes `384x3072` for this row and does not list `1:8` for Flash Lite at all, so the
    table contradicts the documentation on purpose. The unit test over that table can only assert it
    against itself; this recording is what makes the contradiction evidence instead of a typo.
    """
    http_client, sent_bodies = _body_capturing_http_client()
    provider = GoogleProvider(
        api_key=os.getenv('GOOGLE_API_KEY', os.getenv('GEMINI_API_KEY', 'mock-api-key')), http_client=http_client
    )
    generator = ImageGenerator(GoogleImageGenerationModel('gemini-3.1-flash-lite-image', provider=provider))

    try:
        result = await generator.generate(
            'A single ripe kiwi fruit on a plain white background, product photo.',
            settings={'aspect_ratio': '1:8'},
        )
    finally:
        await http_client.aclose()

    assert [body['generationConfig'] for body in sent_bodies] == snapshot(
        [{'responseModalities': ['IMAGE'], 'imageConfig': {'aspectRatio': '1:8', 'imageSize': '1K'}}]
    )

    generated_image = result.images[0].content
    assert generated_image.media_type == 'image/jpeg'
    # A JPEG start-of-frame segment carries `length`, `precision`, `height`, `width` after its marker.
    frame = re.search(rb'\xff[\xc0\xc2]', generated_image.data)
    assert frame is not None
    height = int.from_bytes(generated_image.data[frame.end() + 3 : frame.end() + 5], 'big')
    width = int.from_bytes(generated_image.data[frame.end() + 5 : frame.end() + 7], 'big')
    assert (width, height) == (352, 2928)


@pytest.fixture
async def google_cloud_body_capture(
    vertex_provider_auth: None,
) -> AsyncGenerator[tuple[GoogleCloudProvider, list[dict[str, Any]]]]:
    """A Vertex AI provider paired with the list of JSON bodies it sends, as `_body_capturing_http_client` is for the Gemini API.

    NOTE: You need to comment out the skip below to rewrite the cassettes locally. `vertex_provider_auth`
    only stubs credentials when `CI` is set, so recording needs `CI` unset and real application-default
    credentials, while playback needs `CI` set for the stub.
    """
    if not os.getenv('CI', False):  # pragma: lax no cover
        pytest.skip('Requires properly configured local google vertex config to pass')

    http_client, sent_bodies = _body_capturing_http_client()
    provider = GoogleCloudProvider(
        project=os.getenv('GOOGLE_PROJECT', 'pydantic-ai'),
        location=cast('GoogleCloudLocation', os.getenv('GOOGLE_LOCATION', 'global')),
        http_client=http_client,
    )
    try:
        yield provider, sent_bodies
    finally:
        await http_client.aclose()


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
@pytest.mark.vcr
async def test_google_cloud_image_generation_vcr(
    google_cloud_body_capture: tuple[GoogleCloudProvider, list[dict[str, Any]]],
):
    """Live Vertex AI generation, proving the `google-cloud:` route reaches a different API than `google:`.

    The Gemini API recordings can't stand in for this one: Vertex authenticates differently and serves
    `generateContent` from its own host, so only a recording made against it shows the adapter working there.
    """
    provider, sent_bodies = google_cloud_body_capture
    model = GoogleImageGenerationModel('gemini-3.1-flash-image', provider=provider)

    result = await model.generate(
        'A cat with a cowboy hat, dancing in Rome.',
        settings=GoogleImageGenerationSettings(dimensions=(512, 512)),
    )

    assert sent_bodies == snapshot(
        [
            {
                'contents': [{'parts': [{'text': 'A cat with a cowboy hat, dancing in Rome.'}], 'role': 'user'}],
                'generationConfig': {
                    'responseModalities': ['IMAGE'],
                    'imageConfig': {'aspectRatio': '1:1', 'imageSize': '512'},
                },
            }
        ]
    )

    assert len(result.images) == 1
    generated_image = result.images[0]
    # The media type is read off the response's `mimeType`, so the magic bytes are what keep the
    # assertion from certifying the provider's own echo.
    assert generated_image.content.data[:8] == b'\x89PNG\r\n\x1a\n'
    assert generated_image.content.media_type == snapshot('image/png')
    assert generated_image.output_format == snapshot('png')
    assert result.model_name == snapshot('gemini-3.1-flash-image')
    assert result.provider_name == 'google-cloud'
    assert result.provider_url == snapshot('https://aiplatform.googleapis.com/')
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.provider_details == snapshot(
        {'finish_reason': 'STOP', 'timestamp': IsDatetime(), 'traffic_type': 'ON_DEMAND'}
    )
    assert result.provider_response_id


@pytest.mark.skipif(not google_imports_successful(), reason='Google Gen AI SDK not installed')
@pytest.mark.vcr
async def test_google_cloud_image_edit_vcr(
    google_cloud_body_capture: tuple[GoogleCloudProvider, list[dict[str, Any]]],
    image_content: BinaryImage,
):
    """Live Vertex AI reference edit, with the resolved geometry pinned on the wire.

    Only `generationConfig` is snapshotted: `contents` carries the reference image as a ~130 KB base64
    blob, which would drown the assertion it belongs to.
    """
    provider, sent_bodies = google_cloud_body_capture
    model = GoogleImageGenerationModel('gemini-3.1-flash-image', provider=provider)

    result = await model.generate(
        'Transform the subject into a dog with a cowboy hat, dancing in Rome.',
        images=[image_content],
        settings=GoogleImageGenerationSettings(dimensions=(512, 512)),
    )

    assert [body['generationConfig'] for body in sent_bodies] == snapshot(
        [{'responseModalities': ['IMAGE'], 'imageConfig': {'aspectRatio': '1:1', 'imageSize': '512'}}]
    )

    assert len(result.images) == 1
    edited_image = result.images[0]
    assert edited_image.content.data[:8] == b'\x89PNG\r\n\x1a\n'
    assert edited_image.content.media_type == snapshot('image/png')
    assert edited_image.output_format == snapshot('png')
    assert result.model_name == snapshot('gemini-3.1-flash-image')
    assert result.provider_name == 'google-cloud'
    assert result.provider_url == snapshot('https://aiplatform.googleapis.com/')
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert result.provider_details == snapshot(
        {'finish_reason': 'STOP', 'timestamp': IsDatetime(), 'traffic_type': 'ON_DEMAND'}
    )
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
    settings = XaiImageGenerationSettings(xai_n=2, xai_user='user-123', aspect_ratio='1:1', xai_resolution='1k')

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
                cache_read_tokens=2,
                details={
                    'reasoning_tokens': 3,
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
        xai_aspect_ratio='1:1',
        xai_resolution='1k',
    )
    with pytest.warns(
        UserWarning,
        match=r'used provider-specific settings instead of: `aspect_ratio`',
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

    # `4:5` has no member in the SDK's `ImageAspectRatio` enum, but the provider-specific value is what
    # travels, so the common one is a conflict to warn about rather than a request that cannot be built.
    # The resolution tier still pins to `1k` because a common ratio was asked for.
    mock_client.image.sample.reset_mock()
    with pytest.warns(
        UserWarning,
        match=r'used provider-specific settings instead of: `aspect_ratio`',
    ):
        await model.generate(
            'overridden inexpressible ratio',
            settings=XaiImageGenerationSettings(aspect_ratio='4:5', xai_aspect_ratio='1:1'),
        )

    mock_client.image.sample.assert_awaited_once_with(
        'overridden inexpressible ratio',
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

    # Without an override there is nothing to carry `4:5`, so the request cannot be built at all.
    with pytest.raises(
        UserError,
        match=r"xAI image generation does not support `aspect_ratio='4:5'`\. Supported aspect ratios are: `1:1`, ",
    ):
        await model.generate(
            'inexpressible aspect ratio',
            settings=XaiImageGenerationSettings(aspect_ratio='4:5'),
        )


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_warns_for_settings_its_transport_cannot_carry():
    """xAI's gRPC transport has no body or header escape hatch, so these portable settings warn."""
    mock_client = AsyncMock()
    mock_client.image.sample.return_value = _xai_image_responses(b'image')[0]
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=mock_client),
    )

    with pytest.warns(UserWarning, match=r'ignored unsupported settings: `extra_headers`, `extra_body`'):
        await model.generate(
            'tiny robot',
            settings=XaiImageGenerationSettings(extra_headers={'x-trace': '1'}, extra_body={'seed': 1}),
        )


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
@pytest.mark.parametrize('xai_n', [0, True])
async def test_xai_image_generation_rejects_invalid_count(xai_n: int):
    """Only counts the request cannot express are rejected; xAI owns its own upper bound.

    `xai_n=0` would otherwise be coerced to a single image, silently ignoring what was asked for.
    """
    mock_client = AsyncMock()
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    with pytest.raises(UserError, match='count must be a positive integer'):
        await model.generate('tiny robot', settings=XaiImageGenerationSettings(xai_n=xai_n))

    mock_client.image.sample.assert_not_awaited()
    mock_client.image.sample_batch.assert_not_awaited()


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_rejects_more_than_three_reference_images():
    mock_client = AsyncMock()
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )
    images = [BinaryImage(data=TINY_PNG, media_type='image/png') for _ in range(4)]

    with pytest.raises(UserError, match='at most three reference images'):
        await model.generate('edit these images', images=images)

    mock_client.image.sample.assert_not_awaited()


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
@pytest.mark.parametrize(
    'model_name',
    [
        'grok-imagine-image-quality',
        'grok-imagine-image-2026-03-02',
        'grok-imagine-image-quality-20260403',
        'grok-imagine-image-quality-latest',
        'grok-imagine-image-pro',
    ],
)
async def test_xai_image_generation_resolves_dimensions_for_documented_aliases(model_name: str):
    """Every documented alias resolves `dimensions` to the same wire geometry as the canonical model.

    xAI publishes dated, `-latest` and `-pro` ids as aliases of the two canonical image models, and both
    canonical models share a single geometry table — so an alias needs no geometry data of its own, only
    recognition. The canonical `grok-imagine-image-quality` is the first case here as the baseline the
    aliases must match; without recognition an alias would be rejected as having no known mapping, the
    behavior reserved for genuinely unknown ids.

    - https://docs.x.ai/developers/models/grok-imagine-image
    - https://docs.x.ai/developers/models/grok-imagine-image-quality
    """
    mock_client = AsyncMock()
    mock_client.image.sample.return_value = _xai_image_responses(b'image')[0]
    model = XaiImageGenerationModel(
        model_name,
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    await model.generate('wide image', settings={'dimensions': (1280, 720)})

    kwargs = mock_client.image.sample.await_args.kwargs
    assert (kwargs['aspect_ratio'], kwargs['resolution']) == snapshot(('16:9', '1k'))


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
    download_mock = AsyncMock(
        return_value={'data': 'data:image/webp;base64,ZG93bmxvYWRlZC1pbWFnZQ==', 'data_type': 'image/webp'}
    )
    monkeypatch.setattr(xai_images, 'download_item', download_mock)
    mock_client = AsyncMock()
    mock_client.image.sample.return_value = _xai_image_responses(b'edited-image')[0]
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )
    image_url = ImageUrl('https://example.com/reference.png', force_download=True)

    await model.generate('edit this image', images=[image_url])

    download_mock.assert_awaited_once_with(image_url, data_format='base64_uri')
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

    Not a VCR test because a malformed slot cannot be provoked on demand, so the payload is fixed
    from the response shape the SDK documents.
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

    result = await model.generate('tiny robot', settings=XaiImageGenerationSettings(xai_n=3))

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

    Not a VCR test because a moderation block cannot be provoked on demand, so the flagged slots are
    fixed from the response shape the SDK documents.
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
        await model.generate('tiny robot', settings=XaiImageGenerationSettings(xai_n=2))


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
async def test_xai_image_generation_usage_falls_back_to_sdk_totals(monkeypatch: pytest.MonkeyPatch):
    """The SDK's own totals backfill the token counts genai-prices could not derive.

    `extract` is stubbed to the zeroed usage it returns when the model is missing from the pricing
    snapshot, leaving `prompt_tokens`/`completion_tokens` as the only source for the counts.
    """
    monkeypatch.setattr(xai_images.RequestUsage, 'extract', MagicMock(return_value=RequestUsage()))
    mock_client = AsyncMock()
    mock_client.image.sample.return_value = _xai_image_responses(b'first-image')[0]
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    result = await model.generate('tiny robot')

    assert result.usage == RequestUsage(input_tokens=7, output_tokens=11)


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
async def test_xai_image_generation_empty_response():
    """A batch that comes back with no slots at all is unexpected behavior, not a moderation block.

    Not a VCR test because an empty batch cannot be provoked on demand, so the response is fixed
    from the sequence the SDK documents.
    """
    mock_client = AsyncMock()
    mock_client.image.sample_batch.return_value = []
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(xai_client=cast(XaiAsyncClient, mock_client)),
    )

    with pytest.raises(UnexpectedModelBehavior, match='did not contain any images'):
        await model.generate('tiny robot', settings=XaiImageGenerationSettings(xai_n=2))


@pytest.mark.skipif(not xai_imports_successful(), reason='xAI SDK not installed')
@pytest.mark.parametrize(
    'status_name,expected_http',
    [
        ('INVALID_ARGUMENT', 400),
        ('UNAUTHENTICATED', 401),
        ('PERMISSION_DENIED', 403),
        ('RESOURCE_EXHAUSTED', 429),
        ('CANCELLED', None),
    ],
)
async def test_xai_image_generation_maps_grpc_status_to_http(status_name: str, expected_http: int | None):
    """gRPC status codes map to their HTTP-equivalent `ModelHTTPError`, or to `ModelAPIError` when unmapped.

    xAI's image path is gRPC, so provider errors arrive as `grpc.StatusCode`, not HTTP codes. A bad
    request (`INVALID_ARGUMENT`) must surface as 400 rather than the generic `ModelAPIError`, and the
    auth (`UNAUTHENTICATED`/`PERMISSION_DENIED`) and rate-limit (`RESOURCE_EXHAUSTED`) codes are pinned
    here too. A status with no HTTP equivalent (`expected_http=None`) keeps the generic `ModelAPIError`,
    still carrying the provider's own detail string.
    Parametrized by enum name because `grpc` is an optional import: enum values in the decorator would
    `NameError` at collection time in environments without the xAI extras.

    Reference: `_GRPC_STATUS_TO_HTTP` in `pydantic_ai.images.xai`.

    Not a VCR test because gRPC never reaches the HTTP transport VCR patches, and the proto cassette
    recorder in `tests/models/xai_proto_cassettes.py` stores request/response protobuf pairs, so an
    `RpcError` — which carries no response proto — has nothing to record. Each status is fixed from the
    codes xAI documents.
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

    if expected_http is None:
        with pytest.raises(ModelAPIError, match='boom'):
            await model.generate('tiny robot')
        return

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
    `grok-imagine-image-pro` alias, so the `str` annotation and model value both exercise the forward path.

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


@pytest.fixture
def openai_mock_client() -> AsyncMock:
    """An `AsyncOpenAI` stand-in carrying the `base_url` the adapter reports as `provider_url`."""
    mock_client = AsyncMock()
    mock_client.base_url = 'https://api.openai.com/v1/'
    return mock_client


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_response_mapping(openai_mock_client: AsyncMock):
    openai_mock_client.images.generate.return_value = ImagesResponse.model_construct(
        created=123,
        background='opaque',
        data=[
            Image.model_construct(b64_json=base64.b64encode(TINY_PNG).decode(), revised_prompt='A tiny friendly robot')
        ],
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
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    settings = OpenAIImageGenerationSettings(
        openai_n=1,
        openai_size='auto',
        openai_background='opaque',
        openai_moderation='low',
        openai_output_format='png',
        openai_output_compression=80,
        openai_quality='low',
    )
    result = await model.generate('tiny robot', settings=settings)

    assert result == snapshot(
        ImageGenerationResult(
            images=[
                GeneratedImage(
                    content=BinaryImage(data=TINY_PNG, media_type='image/png'),
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
                input_text_tokens=3,
                input_image_tokens=0,
                details={
                    'input_text_tokens': 3,
                    'input_image_tokens': 0,
                    'output_text_tokens': 0,
                    'output_image_tokens': 5,
                },
                output_text_tokens=0,
                output_image_tokens=5,
            ),
            settings=settings,
            provider_details={'created': 123},
            provider_url='https://api.openai.com/v1/',
        )
    )
    assert 'response_format' not in openai_mock_client.images.generate.await_args.kwargs
    assert openai_mock_client.images.generate.await_args.kwargs['size'] == 'auto'
    assert openai_mock_client.images.generate.await_args.kwargs['background'] == 'opaque'
    assert openai_mock_client.images.generate.await_args.kwargs['moderation'] == 'low'
    assert openai_mock_client.images.generate.await_args.kwargs['quality'] == 'low'
    assert openai_mock_client.images.generate.await_args.kwargs['output_compression'] == 80

    with pytest.warns(UserWarning, match=r'ignored unsupported settings: `input_fidelity`'):
        await model.generate(
            'unsupported settings',
            settings=OpenAIImageGenerationSettings(openai_input_fidelity='high'),
        )

    with pytest.warns(UserWarning, match=r'used provider-specific settings instead of: `aspect_ratio`'):
        await model.generate(
            'conflicting normalized dimensions',
            settings=OpenAIImageGenerationSettings(openai_size='1024x1024', aspect_ratio='3:2'),
        )

    await model.generate(
        'provider-only background',
        settings=OpenAIImageGenerationSettings(openai_background='opaque'),
    )

    await model.generate(
        'valid transparent background',
        settings=OpenAIImageGenerationSettings(openai_background='transparent', openai_output_format='webp'),
    )
    assert openai_mock_client.images.generate.await_args.kwargs['background'] == 'transparent'
    assert openai_mock_client.images.generate.await_args.kwargs['output_format'] == 'webp'


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.parametrize('openai_n', [0, True])
async def test_openai_image_generation_rejects_invalid_count(openai_n: int, openai_mock_client: AsyncMock):
    """Only counts the request cannot express are rejected; OpenAI owns its own upper bound.

    `openai_n=0` would otherwise be omitted from the request, silently ignoring what was asked for.
    """
    model = OpenAIImageGenerationModel(
        'gpt-image-1',
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client)),
    )

    with pytest.raises(UserError, match='count must be a positive integer'):
        await model.generate('tiny robot', settings=OpenAIImageGenerationSettings(openai_n=openai_n))

    openai_mock_client.images.generate.assert_not_awaited()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.parametrize(
    ('model_name', 'aspect_ratio', 'expected_error'),
    [
        (
            'gpt-image-1',
            '16:9',
            r"model 'gpt-image-1' does not support `aspect_ratio='16:9'`\. "
            r'Supported aspect ratios are: `1:1`, `2:3`, `3:2`\.',
        ),
        (
            'gpt-image-2',
            '1:4',
            r"model 'gpt-image-2' does not support `aspect_ratio='1:4'`\. "
            r'Supported aspect ratios are: `1:1`, `1:2`, `2:1`, ',
        ),
    ],
)
async def test_openai_image_generation_rejects_unmappable_aspect_ratio(
    model_name: str,
    aspect_ratio: ImageGenerationAspectRatio,
    expected_error: str,
    openai_mock_client: AsyncMock,
):
    """OpenAI takes a size rather than a ratio, so a ratio outside the family's table cannot be sent.

    Pydantic AI performs this mapping itself, so there is no provider left to judge the request:
    dropping the ratio would silently generate — and bill for — the model's default shape instead.
    """
    model = OpenAIImageGenerationModel(
        model_name,
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client)),
    )

    with pytest.raises(UserError, match=expected_error):
        await model.generate('unmappable ratio', settings={'aspect_ratio': aspect_ratio})

    openai_mock_client.images.generate.assert_not_awaited()


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
    ],
)
async def test_openai_media_type_reflects_actual_bytes(
    image_bytes: bytes,
    echoed_output_format: str,
    expected_media_type: str,
    expected_output_format: str,
    openai_mock_client: AsyncMock,
):
    """`media_type` and `output_format` come from the returned bytes, not the provider's echo.

    gpt-image-2 silently ignores `output_format='webp'` and returns PNG bytes while the response still
    echoes `output_format: webp`, so trusting the echo makes `GeneratedImage.content.media_type` lie and
    breaks downstream content-type handling. We sniff PNG/JPEG/WebP magic bytes and prefer the sniffed
    type, keeping `content.media_type` and `output_format` consistent with each other.

    https://github.com/openai/openai-node/issues/1850
    """
    openai_mock_client.images.generate.return_value = ImagesResponse.model_construct(
        data=[Image.model_construct(b64_json=base64.b64encode(image_bytes).decode())],
        output_format=echoed_output_format,
    )
    model = OpenAIImageGenerationModel(
        'gpt-image-2',
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client)),
    )

    result = await model.generate('a robot')

    image = result.images[0]
    assert image.content.media_type == expected_media_type
    assert image.content.data == image_bytes
    assert image.output_format == expected_output_format


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_usage_falls_back_to_sdk_totals(
    monkeypatch: pytest.MonkeyPatch, openai_mock_client: AsyncMock
):
    openai_mock_client.images.generate.return_value = ImagesResponse.model_construct(
        data=[Image.model_construct(b64_json=base64.b64encode(TINY_PNG).decode())],
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
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client)),
    )

    result = await model.generate('tiny robot')

    assert result.usage == RequestUsage(input_tokens=3, output_tokens=5)

    extracted_usage = RequestUsage(input_tokens=9, output_tokens=7)
    monkeypatch.setattr(openai_images.RequestUsage, 'extract', MagicMock(return_value=extracted_usage))

    extracted_result = await model.generate('another tiny robot')

    assert extracted_result.usage == extracted_usage


def _openai_png_response() -> ImagesResponse:
    """A minimal successful response: one PNG image, no usage or echoed settings."""
    return ImagesResponse.model_construct(
        data=[Image.model_construct(b64_json=base64.b64encode(TINY_PNG).decode())], output_format='png'
    )


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_gpt_image_2_resolves_dimensions_and_aspect_ratio(openai_mock_client: AsyncMock):
    openai_mock_client.images.generate.return_value = _openai_png_response()
    model = OpenAIImageGenerationModel(
        'gpt-image-2',
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client)),
    )

    await model.generate('wide image', settings={'dimensions': (2048, 1152)})
    assert openai_mock_client.images.generate.await_args.kwargs['size'] == '2048x1152'

    await model.generate('wide ratio', settings={'aspect_ratio': '16:9'})
    assert openai_mock_client.images.generate.await_args.kwargs['size'] == '1280x720'

    await model.generate(
        'provider size',
        settings=OpenAIImageGenerationSettings(openai_size='2048x1152', aspect_ratio='16:9'),
    )
    assert openai_mock_client.images.generate.await_args.kwargs['size'] == '2048x1152'

    openai_mock_client.images.generate.reset_mock()
    with pytest.raises(UserError, match='height must be multiples of 16'):
        await model.generate('invalid dimensions', settings={'dimensions': (1920, 1080)})
    openai_mock_client.images.generate.assert_not_awaited()

    with pytest.raises(UserError, match='height must be multiples of 16'):
        await model.generate(
            'invalid overridden dimensions',
            settings=OpenAIImageGenerationSettings(dimensions=(1920, 1080), openai_size='1920x1080'),
        )
    openai_mock_client.images.generate.assert_not_awaited()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.parametrize('model_name', ['gpt-image-1', 'gpt-image-1-mini', 'gpt-image-1.5'])
async def test_openai_legacy_resolves_dimensions_and_aspect_ratio(model_name: str, openai_mock_client: AsyncMock):
    """Pins the GPT Image 1.x column of the matrix published in `docs/image-generation.md`.

    The legacy aspect-ratio table was reachable only through its miss path, where an unsupported
    ratio returns `None` and warns — which executes the lookup line, so coverage stayed green while
    every documented hit went unasserted. These go through `generate()` so the resolver, the adapter
    and the outgoing `size` are all on the hook.
    """
    openai_mock_client.images.generate.return_value = _openai_png_response()
    model = OpenAIImageGenerationModel(
        model_name,
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client)),
    )

    ratio_cases: list[tuple[ImageGenerationAspectRatio, str]] = [
        ('1:1', '1024x1024'),
        ('2:3', '1024x1536'),
        ('3:2', '1536x1024'),
    ]
    for aspect_ratio, expected_size in ratio_cases:
        ratio_settings: ImageGenerationSettings = {'aspect_ratio': aspect_ratio}
        await model.generate('ratio', settings=ratio_settings)
        assert openai_mock_client.images.generate.await_args.kwargs['size'] == expected_size

    await model.generate('exact', settings={'dimensions': (1024, 1024)})
    assert openai_mock_client.images.generate.await_args.kwargs['size'] == '1024x1024'

    openai_mock_client.images.generate.reset_mock()
    with pytest.raises(UserError, match='Supported exact dimensions'):
        await model.generate('unsupported', settings={'dimensions': (2048, 2048)})
    openai_mock_client.images.generate.assert_not_awaited()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_unknown_model_forwards_dimensions_but_cannot_map_aspect_ratio(openai_mock_client: AsyncMock):
    """A GPT Image release newer than the installed Pydantic AI keeps `dimensions`, but not `aspect_ratio`.

    `size` is a plain string on the wire, so an unrecognized model's exact shape is OpenAI's to accept
    or reject; measuring it against the frozen GPT Image 1.x table would reject shapes the new model
    supports. A ratio has no wire field at all, so Pydantic AI would have to name a canonical size it
    has no table for, and the request cannot be built.
    """
    openai_mock_client.images.generate.return_value = _openai_png_response()
    model = OpenAIImageGenerationModel(
        'gpt-image-3',
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client)),
    )

    await model.generate('future model', settings={'dimensions': (1280, 720)})
    assert openai_mock_client.images.generate.await_args.kwargs['size'] == '1280x720'

    openai_mock_client.images.generate.reset_mock()
    with pytest.raises(
        UserError,
        match=r"Pydantic AI has no `aspect_ratio` mapping for OpenAI model 'gpt-image-3'\. "
        r'Use `openai_size` or `dimensions` to set the output geometry\.',
    ):
        await model.generate('future model', settings={'aspect_ratio': '16:9'})
    openai_mock_client.images.generate.assert_not_awaited()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.parametrize('model_name', ['dall-e-2', 'dall-e-3'])
def test_openai_image_generation_rejects_dalle_models(model_name: str):
    """DALL·E models are in the SDK's `ImageModel` literal but diverge from the GPT Image contract.

    They default to `response_format='url'` where this adapter requires base64 bytes, carry their own
    size sets and quality vocabulary, and cap `n` at 1 for `dall-e-3`. Constructing one used to
    succeed and then fail deep in response mapping with an opaque `UnexpectedModelBehavior`; reject
    it by name at construction instead. Unrecognized future model names still fall through.
    """
    with pytest.raises(UserError, match='is not supported'):
        OpenAIImageGenerationModel(model_name, provider=OpenAIProvider(api_key='test-key'))


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.parametrize('model_name', ['gpt-image-2', 'gpt-image-2-2026-04-21'])
async def test_openai_forwards_unvalidated_transparent_background(model_name: str, openai_mock_client: AsyncMock):
    """`openai_background` is forwarded verbatim even where OpenAI documents it as unsupported.

    OpenAI's image-generation guide states GPT Image 2 does not support transparent backgrounds.
    We still forward the provider-prefixed setting the user opted into rather than guarding on an
    assumed capability limit, per the `models/` rule that the provider API is the authority on what
    it currently supports. This pins the passthrough, NOT a claim that the request succeeds — the
    mock returns success, so a real 400 would not be caught here.
    """
    openai_mock_client.images.generate.return_value = _openai_png_response()
    model = OpenAIImageGenerationModel(
        model_name,
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client)),
    )

    await model.generate('transparent image', settings=OpenAIImageGenerationSettings(openai_background='transparent'))

    assert openai_mock_client.images.generate.await_args.kwargs['background'] == 'transparent'


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_gpt_image_2_forwards_input_fidelity_on_edit(openai_mock_client: AsyncMock):
    openai_mock_client.images.edit.return_value = _openai_png_response()
    model = OpenAIImageGenerationModel(
        'gpt-image-2',
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client)),
    )

    await model.generate(
        'edit this image',
        images=[BinaryImage(data=TINY_PNG, media_type='image/png')],
        settings=OpenAIImageGenerationSettings(openai_input_fidelity='high'),
    )

    openai_mock_client.images.edit.assert_awaited_once()
    assert openai_mock_client.images.edit.await_args.kwargs['input_fidelity'] == 'high'


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_forwards_unvalidated_transparent_background_with_jpeg_edit(openai_mock_client: AsyncMock):
    """Both halves of a documented-incompatible combination are forwarded unmodified.

    OpenAI's API reference states `background='transparent'` requires an output format that
    supports transparency (`png` or `webp`), so pairing it with `jpeg` is documented as invalid.
    As above, this pins that we forward the user's provider-prefixed settings and let the API
    reject them, not that the combination is accepted.
    """
    openai_mock_client.images.edit.return_value = _openai_png_response()
    model = OpenAIImageGenerationModel(
        'gpt-image-1.5',
        provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client)),
    )

    await model.generate(
        'transparent edit',
        images=[BinaryImage(data=TINY_PNG, media_type='image/png')],
        settings=OpenAIImageGenerationSettings(openai_background='transparent', openai_output_format='jpeg'),
    )

    assert openai_mock_client.images.edit.await_args.kwargs['background'] == 'transparent'
    assert openai_mock_client.images.edit.await_args.kwargs['output_format'] == 'jpeg'


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
@pytest.mark.vcr
async def test_openai_gpt_image_2_generation_vcr(openai_api_key: str):
    """Live generation whose settings are pinned as the client actually sent them.

    The response echoes `size`, `output_format` and `quality` back, so asserting the result alone would
    still pass if those settings stopped reaching the wire — and so would a cassette-body assertion,
    since the recording is frozen. The request hook captures the live outbound body instead.
    """
    http_client, sent_bodies = _body_capturing_http_client()
    provider = OpenAIProvider(api_key=openai_api_key, http_client=http_client)
    model = OpenAIImageGenerationModel('gpt-image-2', provider=provider)

    try:
        result = await model.generate(
            'A cat with a cowboy hat, dancing in Rome.',
            settings=OpenAIImageGenerationSettings(
                dimensions=(1280, 720),
                openai_output_format='jpeg',
                openai_output_compression=10,
                openai_quality='low',
            ),
        )
    finally:
        await http_client.aclose()

    assert sent_bodies == snapshot(
        [
            {
                'model': 'gpt-image-2',
                'output_compression': 10,
                'output_format': 'jpeg',
                'prompt': 'A cat with a cowboy hat, dancing in Rome.',
                'quality': 'low',
                'size': '1280x720',
            }
        ]
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

    The request hook is what makes `output_format='webp'` a claim about our own request rather than about
    the recording — the sniff assertions below are only meaningful if webp was really asked for.

    https://github.com/openai/openai-node/issues/1850
    """
    http_client, sent_bodies = _body_capturing_http_client()
    provider = OpenAIProvider(api_key=openai_api_key, http_client=http_client)
    model = OpenAIImageGenerationModel('gpt-image-2', provider=provider)

    try:
        result = await model.generate(
            'A small red circle centered on a plain white background.',
            settings=OpenAIImageGenerationSettings(
                dimensions=(1024, 1024), openai_output_format='webp', openai_quality='low'
            ),
        )
    finally:
        await http_client.aclose()

    assert sent_bodies == snapshot(
        [
            {
                'model': 'gpt-image-2',
                'output_format': 'webp',
                'prompt': 'A small red circle centered on a plain white background.',
                'quality': 'low',
                'size': '1024x1024',
            }
        ]
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
async def test_openai_response_without_image_data(openai_mock_client: AsyncMock):
    """A response with no image, no base64, undecodable base64, or unrecognized bytes raises cleanly.

    Not a VCR test because a malformed response cannot be provoked on demand, so each shape is fixed
    from OpenAI's documented response format.
    """
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    openai_mock_client.images.generate.return_value = ImagesResponse.model_construct(created=123, data=[])
    with pytest.raises(UnexpectedModelBehavior, match='did not contain any images'):
        await model.generate('tiny robot')

    openai_mock_client.images.generate.return_value = ImagesResponse.model_construct(
        created=123, data=[Image.model_construct(url='https://example.com/a.png')]
    )
    with pytest.raises(UnexpectedModelBehavior, match='base64 image data'):
        await model.generate('tiny robot')

    openai_mock_client.images.generate.return_value = ImagesResponse.model_construct(
        created=123, data=[Image.model_construct(b64_json='!!!!')]
    )
    with pytest.raises(UnexpectedModelBehavior, match='valid base64 image data') as exc_info:
        await model.generate('tiny robot')

    # The error body omits `data` so a failure can't dump megabytes of base64 payload.
    assert exc_info.value.body is not None
    assert '!!!!' not in exc_info.value.body
    assert '"created": 123' in exc_info.value.body

    unrecognized_b64 = base64.b64encode(b'not an image').decode()
    openai_mock_client.images.generate.return_value = ImagesResponse.model_construct(
        created=123, data=[Image.model_construct(b64_json=unrecognized_b64)]
    )
    with pytest.raises(UnexpectedModelBehavior, match='recognized image format') as exc_info:
        await model.generate('tiny robot')

    assert exc_info.value.body is not None
    assert unrecognized_b64 not in exc_info.value.body


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_edit_request(openai_mock_client: AsyncMock):
    openai_mock_client.images.edit.return_value = ImagesResponse.model_construct(
        created=456,
        data=[Image.model_construct(b64_json=base64.b64encode(TINY_PNG).decode())],
        output_format='webp',
        quality='high',
        size='1024x1024',
    )
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    settings = OpenAIImageGenerationSettings(
        openai_n=1,
        extra_headers={'x-test': 'header'},
        extra_body={'provider_option': True},
        openai_size='1024x1024',
        openai_quality='high',
        openai_background='opaque',
        openai_input_fidelity='high',
        openai_moderation='low',
        openai_output_format='webp',
        openai_output_compression=80,
        openai_user='user-123',
    )
    with pytest.warns(UserWarning, match=r'ignored unsupported settings: `moderation`'):
        result = await model.generate(
            'turn these into one image',
            images=[
                BinaryImage(data=TINY_PNG, media_type='image/png'),
                BinaryImage(data=_JPEG_MAGIC_BYTES, media_type='image/jpeg'),
            ],
            settings=settings,
        )

    openai_mock_client.images.generate.assert_not_awaited()
    openai_mock_client.images.edit.assert_awaited_once()
    kwargs = openai_mock_client.images.edit.await_args.kwargs
    assert kwargs['image'] == [
        ('image-0.png', TINY_PNG, 'image/png'),
        ('image-1.jpg', _JPEG_MAGIC_BYTES, 'image/jpeg'),
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
    assert result.images[0].content == BinaryImage(data=TINY_PNG, media_type='image/png')
    assert result.provider_details == {'created': 456}


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_edit_wire_payload():
    requests: list[httpx2.Request] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                'created': 456,
                'data': [{'b64_json': base64.b64encode(TINY_PNG).decode()}],
                'output_format': 'png',
            },
        )

    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle_request))
    openai_client = AsyncOpenAI(api_key='test-api-key', base_url='https://example.com/v1', http_client=http_client)
    provider = OpenAIProvider(openai_client=openai_client)
    model = OpenAIImageGenerationModel('gpt-image-1.5', provider=provider)
    settings = OpenAIImageGenerationSettings(
        openai_output_format='png', openai_input_fidelity='high', openai_moderation='low'
    )

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
        openai_n=1,
        openai_output_format='jpeg',
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
async def test_openai_image_edit_downloads_image_url(monkeypatch: pytest.MonkeyPatch, openai_mock_client: AsyncMock):
    download_mock = AsyncMock(return_value={'data': b'downloaded', 'data_type': 'image/webp'})
    monkeypatch.setattr(openai_images, 'download_item', download_mock)

    openai_mock_client.images.edit.return_value = _openai_png_response()
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)
    image_url = ImageUrl('https://example.com/reference.png')

    await model.generate('edit this image', images=[image_url])

    download_mock.assert_awaited_once_with(image_url, data_format='bytes')
    assert openai_mock_client.images.edit.await_args.kwargs['image'] == [('image-0.webp', b'downloaded', 'image/webp')]


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
async def test_openai_image_edit_rejects_uploaded_file(
    uploaded_file: UploadedFile, error_message: str, openai_mock_client: AsyncMock
):
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    with pytest.raises(UserError, match=error_message):
        await model.generate('edit this image', images=[uploaded_file])

    openai_mock_client.images.generate.assert_not_awaited()
    openai_mock_client.images.edit.assert_not_awaited()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_edit_rejects_unsupported_image_format(openai_mock_client: AsyncMock):
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    with pytest.raises(UserError, match=r'only supports PNG, JPEG, or WebP.*image/gif'):
        await model.generate('edit this image', images=[BinaryImage(data=b'gif', media_type='image/gif')])

    openai_mock_client.images.edit.assert_not_awaited()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_edit_status_error(openai_mock_client: AsyncMock):
    """A 5xx on the edit endpoint surfaces as `ModelHTTPError` with the status and body preserved.

    Not a VCR test because a deliberate 500 from the edit endpoint is not reproducible on demand, and
    the structured error body a recording would capture depends on which OpenAI failure produced it.
    """
    openai_mock_client.images.edit.side_effect = APIStatusError(
        'test error',
        response=httpx2.Response(
            status_code=500, request=httpx2.Request('POST', 'https://example.com/v1/images/edits')
        ),
        body={'error': 'test error'},
    )
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    with pytest.raises(ModelHTTPError) as exc_info:
        await model.generate('edit this image', images=[BinaryImage(data=TINY_PNG, media_type='image/png')])

    assert exc_info.value.status_code == 500
    assert exc_info.value.body == {'error': 'test error'}


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_connection_error(openai_mock_client: AsyncMock):
    """A connection failure surfaces as `ModelAPIError` rather than the SDK's own exception.

    Not a VCR test because a transport failure cannot be provoked on demand, so the error is fixed
    from the shape the SDK documents.
    """
    openai_mock_client.images.generate.side_effect = APIConnectionError(
        message='connection failed', request=httpx2.Request('POST', 'https://example.com/v1/images/generations')
    )
    provider = OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client))
    model = OpenAIImageGenerationModel('gpt-image-1', provider=provider)

    with pytest.raises(ModelAPIError, match='connection failed'):
        await model.generate('generate this image')


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_rate_limited(openai_mock_client: AsyncMock):
    """A 429 rate-limit surfaces as `ModelHTTPError` with the status and body preserved.

    Image models are rate-limited by images/min and images/day, and a Tier-1 org (~5 images/min) can hit
    the limit before its first successful generation, so this is a common first-call failure, not an edge case.

    See https://platform.openai.com/docs/guides/rate-limits.

    Not a VCR test because a rate limit cannot be provoked on demand, so the response is fixed from
    OpenAI's documented error format.
    """
    rate_limit_body = {'error': {'code': 'rate_limit_exceeded', 'type': 'requests', 'message': 'Rate limit reached'}}
    openai_mock_client.images.generate.side_effect = APIStatusError(
        'Rate limit reached',
        response=httpx2.Response(
            status_code=429,
            headers={'retry-after': '30'},
            request=httpx2.Request('POST', 'https://example.com/v1/images/generations'),
        ),
        body=rate_limit_body,
    )
    model = OpenAIImageGenerationModel(
        'gpt-image-1', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client))
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        await model.generate('a robot')

    assert exc_info.value.status_code == 429
    assert exc_info.value.body == rate_limit_body
    assert exc_info.value.retry_after == 30
    openai_mock_client.images.generate.assert_awaited_once()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_moderation_blocked(openai_mock_client: AsyncMock):
    """A `moderation_blocked` 400 keeps its structured body and is never auto-retried.

    OpenAI returns HTTP 400 with `error.code == 'moderation_blocked'` and a `moderation_details` object
    (`moderation_stage`, `categories`). The wrapper must preserve that structure as data — so callers can
    branch on the code and inspect the categories — rather than flattening it into a string. A moderation
    block reflects the prompt, so retrying the identical request is wrong; we assert a single attempt.

    See https://developers.openai.com/api/docs/guides/image-generation#content-moderation.

    Not a VCR test because provoking a real block means committing a policy-violating prompt to the
    repository, which the recorded request body would carry verbatim. The response is fixed from
    OpenAI's documented error format instead.
    """
    moderation_body = {
        'error': {
            'code': 'moderation_blocked',
            'type': 'image_generation_user_error',
            'message': 'Your request was rejected as a result of our safety system.',
            'moderation_details': {'moderation_stage': 'input', 'categories': ['violence', 'self-harm']},
        }
    }
    openai_mock_client.images.generate.side_effect = APIStatusError(
        'moderation_blocked',
        response=httpx2.Response(
            status_code=400, request=httpx2.Request('POST', 'https://example.com/v1/images/generations')
        ),
        body=moderation_body,
    )
    model = OpenAIImageGenerationModel(
        'gpt-image-1', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client))
    )

    with pytest.raises(ContentFilterError) as exc_info:
        await model.generate('a blocked prompt')

    assert json.loads(exc_info.value.body or '') == snapshot(
        {
            'error': {
                'code': 'moderation_blocked',
                'type': 'image_generation_user_error',
                'message': 'Your request was rejected as a result of our safety system.',
                'moderation_details': {'moderation_stage': 'input', 'categories': ['violence', 'self-harm']},
            }
        }
    )
    openai_mock_client.images.generate.assert_awaited_once()


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_does_not_override_timeout(openai_mock_client: AsyncMock):
    """The adapter never smuggles its own request timeout into the OpenAI SDK call.

    Generations run 30-130s against a ~180s infra ceiling, and users configure timeouts on the client they
    pass in. Injecting a per-request `timeout` would silently override the user's client and truncate long
    generations, so the contract is: we forward no `timeout` for either generate or edit.

    See https://developers.openai.com/api/docs/api-reference/images/create.
    """
    openai_mock_client.images.generate.return_value = _openai_png_response()
    openai_mock_client.images.edit.return_value = _openai_png_response()
    model = OpenAIImageGenerationModel(
        'gpt-image-1', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client))
    )

    await model.generate('a robot')
    assert 'timeout' not in openai_mock_client.images.generate.await_args.kwargs

    await model.generate('edit this', images=[BinaryImage(data=TINY_PNG, media_type='image/png')])
    assert 'timeout' not in openai_mock_client.images.edit.await_args.kwargs


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_tolerates_unknown_response_fields():
    """Unknown top-level and per-image response fields are tolerated (SDK forward compat).

    Providers add response fields without notice; the wrapper must parse successfully and still return the
    images rather than choking on fields it doesn't model. Recorded through the real SDK over a mock
    transport so the SDK's own (extra-allowing) parsing is exercised, not a hand-built response object.
    """
    requests: list[httpx2.Request] = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                'created': 123,
                'output_format': 'png',
                'data': [{'b64_json': base64.b64encode(TINY_PNG).decode(), 'unexpected_image_field': 'ignored'}],
                'unexpected_top_level_field': {'nested': True},
            },
        )

    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle_request))
    openai_client = AsyncOpenAI(api_key='test-api-key', base_url='https://example.com/v1', http_client=http_client)
    model = OpenAIImageGenerationModel('gpt-image-1', provider=OpenAIProvider(openai_client=openai_client))

    try:
        result = await model.generate('a robot')
    finally:
        await http_client.aclose()

    assert len(requests) == 1
    assert result.images[0].content == BinaryImage(data=TINY_PNG, media_type='image/png')


@pytest.mark.skipif(not openai_imports_successful(), reason='OpenAI not installed')
async def test_openai_image_generation_supported_settings_emit_no_warning(openai_mock_client: AsyncMock):
    """A fully-supported settings combination emits no warning.

    Over-warning erodes the signal of the warning channel; warnings are reserved for settings a request
    genuinely ignores or overrides. Every setting here is supported by `gpt-image-1` generation, so the call
    must be silent.
    """
    openai_mock_client.images.generate.return_value = _openai_png_response()
    model = OpenAIImageGenerationModel(
        'gpt-image-1', provider=OpenAIProvider(openai_client=cast(AsyncOpenAI, openai_mock_client))
    )
    settings = OpenAIImageGenerationSettings(
        openai_n=1,
        openai_size='1024x1024',
        openai_quality='high',
        openai_background='opaque',
        openai_moderation='low',
        openai_output_format='png',
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
        settings={
            'dimensions': (1024, 1024),
            'extra_headers': {'authorization': 'Bearer test'},
            'extra_body': {'application_data': 'test'},
        },
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
                'image_generation_settings': {'dimensions': [1024, 1024]},
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
    # The generated bytes must never reach the span. Assert against what this model actually
    # returns — a constant from another provider's fixtures would pass no matter what we emit.
    assert base64.b64encode(TINY_PNG).decode() not in str(span)
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
async def test_instrumentation_omits_empty_recorded_settings(capfire: CaptureLogfire):
    generator = ImageGenerator(
        TestImageGenerationModel(),
        instrument=InstrumentationSettings(include_model_request_parameters=True),
    )
    await generator.generate(
        'tiny robot',
        settings={
            'extra_headers': {'authorization': 'Bearer test'},
            'extra_body': {'application_data': 'test'},
        },
    )

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(span for span in spans if 'image_generation' in span['name'])
    assert 'image_generation_settings' not in span['attributes']


@pytest.mark.parametrize(
    'base_url,expected',
    [
        pytest.param('relative/path', {}, id='no_authority'),
        pytest.param('https://example.com/v1', {'server.address': 'example.com'}, id='default_port'),
        pytest.param(
            'https://example.com:8443/v1',
            {'server.address': 'example.com', 'server.port': 8443},
            id='explicit_port',
        ),
        pytest.param('https://example.com:notaport/v1', {}, id='malformed_port'),
    ],
)
def test_instrumentation_server_attributes_tolerate_any_base_url(
    base_url: str, expected: dict[str, str | int], monkeypatch: pytest.MonkeyPatch
):
    """A `base_url` whose authority cannot be interpreted omits the server attributes.

    `ImageGenerationModel.base_url` is an overridable property returning an arbitrary string, and
    `urlparse` defers authority validation to `hostname`/`port`, so a non-numeric port parses fine
    and only raises when the port is read.

    This is the only case-by-case pin on `_instrumentation.server_attributes`, which the embedding
    and chat-model instrumentation share, so deleting it uncovers their authority handling too.
    """
    model = TestImageGenerationModel()
    monkeypatch.setattr(type(model), 'base_url', property(lambda _: base_url))

    assert InstrumentedImageGenerationModel.model_attributes(model) == {
        'gen_ai.provider.name': 'test',
        'gen_ai.request.model': 'test',
        **expected,
    }


async def test_instrumentation_does_not_abort_generation_on_malformed_base_url(monkeypatch: pytest.MonkeyPatch):
    """Attributes are best-effort telemetry, so an uninterpretable `base_url` must not fail the request."""
    model = TestImageGenerationModel()
    monkeypatch.setattr(type(model), 'base_url', property(lambda _: 'https://example.com:notaport/v1'))
    generator = ImageGenerator(model, instrument=True)

    result = await generator.generate('tiny robot')

    assert [image.content.media_type for image in result.images] == ['image/png']


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
    monkeypatch.setattr(type(wrapped), 'base_url', property(lambda _: 'https://example.com/v1'))
    model = InstrumentedImageGenerationModel(wrapped)
    price = cast(PriceCalculation, SimpleNamespace(total_price=Decimal('0.25')))
    monkeypatch.setattr('pydantic_ai.images.instrumented.best_effort_price', MagicMock(return_value=price))

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
    await generator.generate('tiny robot', settings={'dimensions': (1024, 1024)})

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    span = next(span for span in spans if 'image_generation' in span['name'])
    attributes = span['attributes']

    assert 'prompt' not in attributes
    assert 'image_generation_settings' not in attributes
    assert 'image_generation_settings' not in attributes['logfire.json_schema']['properties']
    assert 'image.0.media_type' in attributes
    assert 'data' not in str(span)
