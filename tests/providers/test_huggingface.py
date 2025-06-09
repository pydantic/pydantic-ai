from __future__ import annotations as _annotations

import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from huggingface_hub import AsyncInferenceClient

    from pydantic_ai.providers.huggingface import HuggingFaceProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='huggingface_hub not installed')


def test_huggingface_provider():
    hf_client = AsyncInferenceClient(api_key='api-key')
    provider = HuggingFaceProvider(api_key='api-key', hf_client=hf_client)
    assert provider.name == 'huggingface'
    assert isinstance(provider.client, AsyncInferenceClient)
    assert provider.client.token == 'api-key'


def test_huggingface_provider_need_api_key(env: TestEnv) -> None:
    env.remove('HF_TOKEN')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `HF_TOKEN` environment variable or pass it via `HuggingFaceProvider(api_key=...)`'
            'to use the HuggingFace provider.'
        ),
    ):
        HuggingFaceProvider()


def test_huggingface_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    with pytest.raises(
        ValueError,
        match=re.escape('`http_client` is ignored for HuggingFace provider, please use `hf_client` instead'),
    ):
        HuggingFaceProvider(http_client=http_client, api_key='api-key')


def test_huggingface_provider_pass_hf_client() -> None:
    hf_client = AsyncInferenceClient(api_key='api-key')
    provider = HuggingFaceProvider(hf_client=hf_client)
    assert provider.client == hf_client


def test_hf_provider_with_base_url() -> None:
    # Test with environment variable for base_url
    provider = HuggingFaceProvider(
        hf_client=AsyncInferenceClient(base_url='https://router.huggingface.co/nebius/v1'), api_key='test-api-key'
    )
    assert provider.base_url == 'https://router.huggingface.co/nebius/v1'
