from __future__ import annotations as _annotations

import re
from unittest.mock import MagicMock, Mock, patch

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
        HuggingFaceProvider(http_client=http_client, api_key='api-key')  # type: ignore


def test_huggingface_provider_pass_hf_client() -> None:
    hf_client = AsyncInferenceClient(api_key='api-key')
    provider = HuggingFaceProvider(hf_client=hf_client, api_key='api-key')
    assert provider.client == hf_client


def test_hf_provider_with_base_url() -> None:
    # Test with environment variable for base_url
    provider = HuggingFaceProvider(
        hf_client=AsyncInferenceClient(base_url='https://router.huggingface.co/nebius/v1'), api_key='test-api-key'
    )
    assert provider.base_url == 'https://router.huggingface.co/nebius/v1'


def test_huggingface_provider_properties():
    mock_client = Mock(spec=AsyncInferenceClient)
    mock_client.model = 'test-model'
    provider = HuggingFaceProvider(hf_client=mock_client, api_key='test-api-key')
    assert provider.name == 'huggingface'
    assert provider.client is mock_client


def test_huggingface_provider_init_api_key_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('HF_TOKEN', raising=False)
    with pytest.raises(UserError, match='Set the `HF_TOKEN` environment variable'):
        HuggingFaceProvider()


@patch('pydantic_ai.providers.huggingface.AsyncInferenceClient')
def test_huggingface_provider_init_api_key_from_env(
    MockAsyncInferenceClient: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv('HF_TOKEN', 'env-key')
    HuggingFaceProvider()
    MockAsyncInferenceClient.assert_called_with(api_key='env-key', provider=None, base_url=None)


@patch('pydantic_ai.providers.huggingface.AsyncInferenceClient')
def test_huggingface_provider_init_api_key_from_arg(
    MockAsyncInferenceClient: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv('HF_TOKEN', 'env-key')
    HuggingFaceProvider(api_key='arg-key')
    MockAsyncInferenceClient.assert_called_with(api_key='arg-key', provider=None, base_url=None)


def test_huggingface_provider_init_http_client_error():
    with pytest.raises(ValueError, match='`http_client` is ignored'):
        HuggingFaceProvider(api_key='key', http_client=Mock())  # type: ignore[call-overload]


def test_huggingface_provider_init_base_url_and_provider_name_error():
    with pytest.raises(ValueError, match='Cannot provide both `base_url` and `provider_name`'):
        HuggingFaceProvider(api_key='key', base_url='url', provider_name='provider')  # type: ignore[call-overload]


def test_huggingface_provider_init_with_hf_client():
    mock_client = Mock(spec=AsyncInferenceClient)
    provider = HuggingFaceProvider(hf_client=mock_client, api_key='key')
    assert provider.client is mock_client


@patch('pydantic_ai.providers.huggingface.AsyncInferenceClient')
def test_huggingface_provider_init_without_hf_client(MockAsyncInferenceClient: MagicMock):
    provider = HuggingFaceProvider(api_key='key')
    assert provider.client is MockAsyncInferenceClient.return_value
    MockAsyncInferenceClient.assert_called_with(api_key='key', provider=None, base_url=None)


@patch('pydantic_ai.providers.huggingface.AsyncInferenceClient')
def test_huggingface_provider_init_with_provider_name(MockAsyncInferenceClient: MagicMock):
    HuggingFaceProvider(api_key='key', provider_name='test-provider')
    MockAsyncInferenceClient.assert_called_once_with(api_key='key', provider='test-provider', base_url=None)


@patch('pydantic_ai.providers.huggingface.AsyncInferenceClient')
def test_huggingface_provider_init_with_base_url(MockAsyncInferenceClient: MagicMock):
    HuggingFaceProvider(api_key='key', base_url='test-url')
    MockAsyncInferenceClient.assert_called_once_with(api_key='key', provider=None, base_url='test-url')


def test_huggingface_provider_init_api_key_is_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('HF_TOKEN', raising=False)
    with pytest.raises(UserError):
        HuggingFaceProvider(api_key=None)


def test_huggingface_provider_base_url():
    mock_client = Mock(spec=AsyncInferenceClient)
    mock_client.model = 'test-model'
    provider = HuggingFaceProvider(hf_client=mock_client, api_key='test-api-key')
    assert provider.base_url == 'test-model'
