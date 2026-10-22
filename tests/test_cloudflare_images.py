"""Recorded native Workers AI image families and their request formats."""

import os

import pytest

from pydantic_ai import BinaryImage, ImageGenerator
from pydantic_ai.images import infer_image_generation_model
from pydantic_ai.images.cloudflare import CloudflareImageGenerationModel
from pydantic_ai.images.settings import ImageGenerationSettings
from pydantic_ai.providers.cloudflare import CloudflareProvider

from .conftest import RequestCapture

pytestmark = [pytest.mark.vcr, pytest.mark.usefixtures('allow_model_requests')]


@pytest.fixture
def cloudflare_provider(request_capture: RequestCapture) -> CloudflareProvider:
    return CloudflareProvider(
        api_key=os.getenv('CLOUDFLARE_API_TOKEN', 'test-token'),
        account_id=os.getenv('CLOUDFLARE_ACCOUNT_ID', 'test-account'),
        http_client=request_capture.http_client(timeout=120),
    )


@pytest.mark.parametrize(
    'model_name',
    [
        '@cf/black-forest-labs/flux-1-schnell',
        '@cf/black-forest-labs/flux-2-klein-4b',
        '@cf/leonardo/lucid-origin',
        '@cf/leonardo/phoenix-1.0',
        '@cf/stabilityai/stable-diffusion-xl-base-1.0',
    ],
)
async def test_generate(model_name: str, cloudflare_provider: CloudflareProvider, request_capture: RequestCapture):
    model = infer_image_generation_model(f'cloudflare:{model_name}', provider_factory=lambda _: cloudflare_provider)
    assert isinstance(model, CloudflareImageGenerationModel)
    settings: ImageGenerationSettings = {} if model_name.endswith('flux-1-schnell') else {'dimensions': (256, 256)}
    prompt = (
        'A scenic landscape with rolling green hills and a clear blue sky.'
        if model_name.startswith('@cf/leonardo/')
        else 'A simple blue circle on a white background.'
    )
    result = await ImageGenerator(model).generate(prompt, settings=settings)
    assert result.image.media_type in ('image/png', 'image/jpeg', 'image/webp')
    assert result.image.data
    assert result.model_name == model_name
    assert result.provider_name == 'cloudflare'
    if 'flux-2-' in model_name:
        assert request_capture.headers[-1]['content-type'].startswith('multipart/form-data; boundary=')
        assert b'name="prompt"' in request_capture.raw_bodies[-1]
        assert b'name="width"' in request_capture.raw_bodies[-1]
    else:
        assert request_capture.body()['prompt'] == prompt


async def test_edit(
    cloudflare_provider: CloudflareProvider,
    request_capture: RequestCapture,
    image_content: BinaryImage,
):
    model = CloudflareImageGenerationModel('@cf/black-forest-labs/flux-2-klein-4b', provider=cloudflare_provider)
    result = await ImageGenerator(model).generate(
        'Paint the fruit in watercolor.', images=[image_content], settings={'dimensions': (256, 256)}
    )
    assert result.image.data
    assert b'name="input_image_0"' in request_capture.raw_bodies[-1]
