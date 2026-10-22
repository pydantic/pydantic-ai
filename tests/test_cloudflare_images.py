"""Recorded native Workers AI image families and their request formats."""

import os
from dataclasses import dataclass

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


@dataclass
class GenerateCase:
    model_name: str
    settings: ImageGenerationSettings
    prompt: str
    multipart: bool
    raw_fragments: tuple[bytes, ...]


CASES = [
    GenerateCase(
        model_name='@cf/black-forest-labs/flux-1-schnell',
        settings={},
        prompt='A simple blue circle on a white background.',
        multipart=False,
        raw_fragments=(),
    ),
    GenerateCase(
        model_name='@cf/black-forest-labs/flux-2-klein-4b',
        settings={'dimensions': (256, 256)},
        prompt='A simple blue circle on a white background.',
        multipart=True,
        raw_fragments=(b'name="prompt"', b'name="width"'),
    ),
    GenerateCase(
        model_name='@cf/leonardo/lucid-origin',
        settings={'dimensions': (256, 256)},
        prompt='A scenic landscape with rolling green hills and a clear blue sky.',
        multipart=False,
        raw_fragments=(),
    ),
    GenerateCase(
        model_name='@cf/leonardo/phoenix-1.0',
        settings={'dimensions': (256, 256)},
        prompt='A scenic landscape with rolling green hills and a clear blue sky.',
        multipart=False,
        raw_fragments=(),
    ),
    GenerateCase(
        model_name='@cf/stabilityai/stable-diffusion-xl-base-1.0',
        settings={'dimensions': (256, 256)},
        prompt='A simple blue circle on a white background.',
        multipart=False,
        raw_fragments=(),
    ),
]


@pytest.mark.parametrize('case', CASES, ids=lambda case: case.model_name)
async def test_generate(case: GenerateCase, cloudflare_provider: CloudflareProvider, request_capture: RequestCapture):
    model = infer_image_generation_model(
        f'cloudflare:{case.model_name}', provider_factory=lambda _: cloudflare_provider
    )
    assert isinstance(model, CloudflareImageGenerationModel)
    result = await ImageGenerator(model).generate(case.prompt, settings=case.settings)
    assert result.image.media_type in ('image/png', 'image/jpeg', 'image/webp')
    assert result.image.data
    assert result.model_name == case.model_name
    assert result.provider_name == 'cloudflare'
    if case.multipart:
        assert request_capture.headers[-1]['content-type'].startswith('multipart/form-data; boundary=')
        for fragment in case.raw_fragments:
            assert fragment in request_capture.raw_bodies[-1]
    else:
        assert request_capture.body()['prompt'] == case.prompt


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
