"""Synthetic malformed images, errors and request overrides cannot be obtained reliably with VCR."""

import base64
from unittest.mock import AsyncMock, patch

import httpx2
import pytest

from pydantic_ai import BinaryImage, ImageUrl, UploadedFile
from pydantic_ai.exceptions import ContentFilterError, ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.images.cloudflare import CloudflareImageGenerationModel, CloudflareImageGenerationSettings
from pydantic_ai.providers.cloudflare import CloudflareProvider

pytestmark = pytest.mark.usefixtures('allow_model_requests')

PNG = b'\x89PNG\r\n\x1a\nimage-data'


@pytest.mark.parametrize('multipart', [False, True])
async def test_native_image_settings(multipart: bool):
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        requests.append(request)
        return httpx2.Response(
            200, json={'result': {'image': base64.b64encode(PNG).decode()}}, headers={'cf-ai-req-id': 'image-id'}
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model = CloudflareImageGenerationModel(
            '@cf/black-forest-labs/flux-2-dev' if multipart else '@cf/leonardo/lucid-origin',
            provider=CloudflareProvider(api_key='token', account_id='account', http_client=client),
        )
        settings: CloudflareImageGenerationSettings = {
            'dimensions': (512, 512),
            'cloudflare_steps': 8,
            'cloudflare_seed': 0,
            'cloudflare_guidance': 2.0,
            'cloudflare_negative_prompt': 'blur',
            'extra_headers': {'authorization': 'Bearer override'},
            'extra_body': {'prompt': 'overridden'},
        }
        result = await model.generate('Original prompt', settings=settings)
        assert result.provider_response_id == 'image-id'
        assert result.image.data == PNG
        assert result.images[0].output_format == 'png'
        assert requests[0].headers['authorization'] == 'Bearer override'
        assert b'overridden' in requests[0].content
        assert b'num_steps' in requests[0].content if not multipart else b'name="steps"' in requests[0].content


@pytest.mark.parametrize('model_name', ['@cf/black-forest-labs/flux-1-schnell', '@cf/leonardo/lucid-origin'])
async def test_geometry(model_name: str):
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        requests.append(request)
        return httpx2.Response(200, content=PNG, headers={'content-type': 'image/png'})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model = CloudflareImageGenerationModel(
            model_name, provider=CloudflareProvider(api_key='token', account_id='account', http_client=client)
        )
        if model_name.endswith('flux-1-schnell'):
            with pytest.raises(UserError, match='aspect_ratio'):
                await model.generate('A landscape', settings={'aspect_ratio': '1:1'})
            with pytest.raises(UserError, match='dimensions'):
                await model.generate('A landscape', settings={'dimensions': (512, 512)})
            assert not requests
        else:
            await model.generate('A landscape', settings={'aspect_ratio': '16:9'})
            assert b'1024' in requests[0].content
            assert b'576' in requests[0].content


@pytest.mark.parametrize(
    'model_name,count', [('@cf/leonardo/lucid-origin', 1), ('@cf/black-forest-labs/flux-2-dev', 5)]
)
async def test_reference_limit(model_name: str, count: int):
    model = CloudflareImageGenerationModel(
        model_name, provider=CloudflareProvider(api_key='token', account_id='account')
    )
    with pytest.raises(UserError, match='reference images'):
        await model.generate('A landscape', images=[BinaryImage(data=PNG, media_type='image/png')] * count)


async def test_uploaded_reference_rejected():
    model = CloudflareImageGenerationModel(
        '@cf/black-forest-labs/flux-2-dev', provider=CloudflareProvider(api_key='token', account_id='account')
    )
    with pytest.raises(UserError, match='uploaded file IDs'):
        await model.generate(
            'A landscape', images=[UploadedFile(file_id='image-id', provider_name='openai', media_type='image/png')]
        )


@pytest.mark.parametrize('extra_body', [[], {'nested': {'not': 'scalar'}}])
async def test_invalid_image_overrides(extra_body: object):
    model = CloudflareImageGenerationModel(
        '@cf/black-forest-labs/flux-2-dev', provider=CloudflareProvider(api_key='token', account_id='account')
    )
    with pytest.raises(UserError, match=r'extra_body|scalar'):
        await model.generate('A landscape', settings={'extra_body': extra_body})


@pytest.mark.parametrize(
    'body',
    [
        {'result': {}},
        {'result': {'image': 'not base64'}},
        {'result': {'image': ''}},
        {'result': {'image': base64.b64encode(b'not an image').decode()}},
    ],
)
async def test_malformed_image_response(body: dict[str, object]):
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=body))) as client:
        model = CloudflareImageGenerationModel(
            '@cf/black-forest-labs/flux-1-schnell',
            provider=CloudflareProvider(api_key='token', account_id='account', http_client=client),
        )
        with pytest.raises(UnexpectedModelBehavior, match='Cloudflare'):
            await model.generate('A landscape')


@pytest.mark.parametrize('blocked', [False, True])
async def test_image_api_error(blocked: bool):
    body: dict[str, object] = {
        'errors': [
            {'code': 3030, 'message': 'Input prompt contains NSFW content.' if blocked else 'Invalid image size.'}
        ]
    }
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _: httpx2.Response(400, json=body))) as client:
        model = CloudflareImageGenerationModel(
            '@cf/black-forest-labs/flux-1-schnell',
            provider=CloudflareProvider(api_key='token', account_id='account', http_client=client),
        )
        with pytest.raises(ContentFilterError if blocked else ModelHTTPError):
            await model.generate('A landscape')


async def test_url_reference():
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        request.read()
        requests.append(request)
        return httpx2.Response(200, content=PNG, headers={'content-type': 'image/png'})

    # The downloader is separately covered against real URLs in shared media tests.
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model = CloudflareImageGenerationModel(
            '@cf/black-forest-labs/flux-2-klein-4b',
            provider=CloudflareProvider(api_key='token', account_id='account', http_client=client),
        )
        with patch(
            'pydantic_ai.images.cloudflare.download_item',
            AsyncMock(return_value={'data': PNG, 'data_type': 'image/png'}),
        ):
            await model.generate('A landscape', images=[ImageUrl('https://example.com/image.png')])
        assert b'name="input_image_0"' in requests[0].content
        assert PNG in requests[0].content
