from __future__ import annotations as _annotations

from typing import Any, cast

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.zai import (
        ZaiModelSettings,
        _zai_settings_to_openai_settings,  # pyright: ignore[reportPrivateUsage]
    )


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
