from __future__ import annotations as _annotations

from typing import Any, cast

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.zai import (
        ZaiModel,
        ZaiModelSettings,
        _zai_settings_to_openai_settings,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.zai import ZaiProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


async def test_zai_settings_transformation_thinking_enabled():
    settings = ZaiModelSettings(zai_thinking=True)
    transformed = _zai_settings_to_openai_settings(settings)
    extra_body = cast(dict[str, Any], transformed.get('extra_body', {}))
    assert extra_body.get('thinking') == {'type': 'enabled'}


async def test_zai_settings_transformation_thinking_disabled():
    settings = ZaiModelSettings(zai_thinking=False)
    transformed = _zai_settings_to_openai_settings(settings)
    extra_body = cast(dict[str, Any], transformed.get('extra_body', {}))
    assert extra_body.get('thinking') == {'type': 'disabled'}


async def test_zai_settings_transformation_preserved_thinking():
    settings = ZaiModelSettings(zai_thinking=True, zai_clear_thinking=False)
    transformed = _zai_settings_to_openai_settings(settings)
    extra_body = cast(dict[str, Any], transformed.get('extra_body', {}))
    assert extra_body.get('thinking') == {'type': 'enabled', 'clear_thinking': False}


async def test_zai_settings_transformation_clear_thinking():
    settings = ZaiModelSettings(zai_thinking=True, zai_clear_thinking=True)
    transformed = _zai_settings_to_openai_settings(settings)
    extra_body = cast(dict[str, Any], transformed.get('extra_body', {}))
    assert extra_body.get('thinking') == {'type': 'enabled', 'clear_thinking': True}


async def test_zai_settings_empty():
    settings = ZaiModelSettings()
    transformed = _zai_settings_to_openai_settings(settings)
    assert transformed.get('extra_body') is None


async def test_zai_settings_preserves_existing_extra_body():
    settings = ZaiModelSettings(zai_thinking=True, extra_body={'custom_key': 'value'})
    transformed = _zai_settings_to_openai_settings(settings)
    extra_body = cast(dict[str, Any], transformed.get('extra_body', {}))
    assert extra_body.get('thinking') == {'type': 'enabled'}
    assert extra_body.get('custom_key') == 'value'


async def test_zai_model_prepare_request(zai_api_key: str):
    provider = ZaiProvider(api_key=zai_api_key)
    model = ZaiModel('glm-4.7', provider=provider)
    settings = ZaiModelSettings(zai_thinking=True, zai_clear_thinking=False)
    params = ModelRequestParameters()
    merged_settings, _ = model.prepare_request(settings, params)
    assert merged_settings is not None
    extra_body = cast(dict[str, Any], merged_settings.get('extra_body', {}))
    assert extra_body.get('thinking') == {'type': 'enabled', 'clear_thinking': False}
