from __future__ import annotations

import pytest

from pydantic_ai import BinaryContent, ImageUrl, UserPromptPart
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

pytestmark = pytest.mark.anyio


def _model() -> GroqModel:
    return GroqModel('llama-3.2-11b-vision-preview', provider=GroqProvider(api_key='test'))


async def test_image_url_vendor_metadata_detail():
    model = _model()
    part = UserPromptPart(
        content=[
            ImageUrl(url='https://example.com/image.png', vendor_metadata={'detail': 'high'}),
        ]
    )
    result = await model._map_user_prompt(part)
    content = result['content']
    assert isinstance(content, list)
    image_param = content[0]
    assert image_param['type'] == 'image_url'
    assert image_param['image_url']['url'] == 'https://example.com/image.png'
    assert image_param['image_url']['detail'] == 'high'


async def test_image_url_vendor_metadata_detail_default():
    model = _model()
    part = UserPromptPart(
        content=[
            ImageUrl(url='https://example.com/image.png', vendor_metadata={'other': 'value'}),
        ]
    )
    result = await model._map_user_prompt(part)
    content = result['content']
    image_param = content[0]
    assert image_param['image_url']['detail'] == 'auto'


async def test_image_url_no_vendor_metadata():
    model = _model()
    part = UserPromptPart(
        content=[
            ImageUrl(url='https://example.com/image.png'),
        ]
    )
    result = await model._map_user_prompt(part)
    content = result['content']
    image_param = content[0]
    assert 'detail' not in image_param['image_url']


async def test_binary_content_image_vendor_metadata_detail():
    model = _model()
    image_data = b'\x89PNG\r\n\x1a\n'
    part = UserPromptPart(
        content=[
            BinaryContent(data=image_data, media_type='image/png', vendor_metadata={'detail': 'low'}),
        ]
    )
    result = await model._map_user_prompt(part)
    content = result['content']
    image_param = content[0]
    assert image_param['type'] == 'image_url'
    assert image_param['image_url']['detail'] == 'low'


async def test_binary_content_image_vendor_metadata_detail_default():
    model = _model()
    image_data = b'\x89PNG\r\n\x1a\n'
    part = UserPromptPart(
        content=[
            BinaryContent(data=image_data, media_type='image/png', vendor_metadata={'other': 'value'}),
        ]
    )
    result = await model._map_user_prompt(part)
    content = result['content']
    image_param = content[0]
    assert image_param['image_url']['detail'] == 'auto'


async def test_binary_content_image_no_vendor_metadata():
    model = _model()
    image_data = b'\x89PNG\r\n\x1a\n'
    part = UserPromptPart(
        content=[
            BinaryContent(data=image_data, media_type='image/png'),
        ]
    )
    result = await model._map_user_prompt(part)
    content = result['content']
    image_param = content[0]
    assert 'detail' not in image_param['image_url']
