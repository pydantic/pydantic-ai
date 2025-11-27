from __future__ import annotations as _annotations

import logging
import re
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.providers.huggingface import (
    HfRouterProvider,
    HuggingFaceProvider,
    _get_router_info,  # type: ignore
    select_provider,
)

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from huggingface_hub import AsyncInferenceClient


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


def test_huggingface_provider_model_profile(mocker: MockerFixture):
    from pydantic_ai.providers.huggingface import _get_router_info  # type: ignore

    # Clear lru_cache before mocking to ensure no cached results interfere
    _get_router_info.cache_clear()

    ns = 'pydantic_ai.providers.huggingface'
    # Mock _get_router_info to return None (no network calls)
    mocker.patch(f'{ns}._get_router_info', return_value=None)

    mock_client = Mock(spec=AsyncInferenceClient)
    provider = HuggingFaceProvider(hf_client=mock_client, api_key='test-api-key')

    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    qwen_model_profile_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    google_model_profile_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)

    deepseek_profile = provider.model_profile('deepseek-ai/DeepSeek-R1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.ignore_streamed_leading_whitespace is True

    meta_profile = provider.model_profile('meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8')
    meta_model_profile_mock.assert_called_with('llama-4-maverick-17b-128e-instruct-fp8')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    qwen_profile = provider.model_profile('Qwen/QwQ-32B')
    qwen_model_profile_mock.assert_called_with('qwq-32b')
    assert qwen_profile is not None
    assert qwen_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert qwen_profile.ignore_streamed_leading_whitespace is True

    mistral_profile = provider.model_profile('mistralai/Devstral-Small-2505')
    mistral_model_profile_mock.assert_called_with('devstral-small-2505')
    assert mistral_profile is None

    google_profile = provider.model_profile('google/gemma-3-27b-it')
    google_model_profile_mock.assert_called_with('gemma-3-27b-it')
    assert google_profile is not None
    assert google_profile.json_schema_transformer == GoogleJsonSchemaTransformer

    unknown_profile = provider.model_profile('unknown/model')
    assert unknown_profile is None


def test_select_provider_both_capabilities():
    """Test select_provider prefers providers with both tools and structured output."""
    providers: list[HfRouterProvider] = [
        {'provider': 'p1', 'status': 'live', 'supports_tools': False, 'supports_structured_output': False},
        {'provider': 'p2', 'status': 'live', 'supports_tools': True, 'supports_structured_output': True},
        {'provider': 'p3', 'status': 'live', 'supports_tools': True, 'supports_structured_output': False},
    ]
    result = select_provider(providers)
    assert result is not None
    assert result['provider'] == 'p2'


def test_select_provider_either_capability():
    """Test select_provider falls back to providers with either capability."""
    providers: list[HfRouterProvider] = [
        {'provider': 'p1', 'status': 'live', 'supports_tools': False, 'supports_structured_output': False},
        {'provider': 'p2', 'status': 'live', 'supports_tools': True, 'supports_structured_output': False},
    ]
    result = select_provider(providers)
    assert result is not None
    assert result['provider'] == 'p2'


def test_select_provider_any():
    """Test select_provider falls back to any provider."""
    providers: list[HfRouterProvider] = [
        {'provider': 'p1', 'status': 'live', 'supports_tools': False, 'supports_structured_output': False},
    ]
    result = select_provider(providers)
    assert result is not None
    assert result['provider'] == 'p1'


def test_select_provider_empty():
    """Test select_provider returns None for empty list."""
    result = select_provider([])
    assert result is None


def test_select_provider_no_live_fallback():
    """Test select_provider falls back to non-live providers if no live ones."""
    providers: list[HfRouterProvider] = [
        {'provider': 'p1', 'status': 'pending', 'supports_tools': True, 'supports_structured_output': True},
    ]
    result = select_provider(providers)
    assert result is not None
    assert result['provider'] == 'p1'


def test_get_router_info_success(mocker: MockerFixture):
    """Test _get_router_info successfully parses response."""
    _get_router_info.cache_clear()

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'{"data": {"id": "test/model", "providers": []}}'

    mocker.patch('pydantic_ai.providers.huggingface.httpx.get', return_value=mock_response)

    result = _get_router_info('test/model')
    assert result is not None
    assert result['id'] == 'test/model'


def test_get_router_info_http_error(mocker: MockerFixture):
    """Test _get_router_info handles HTTP errors."""
    _get_router_info.cache_clear()

    mocker.patch('pydantic_ai.providers.huggingface.httpx.get', side_effect=httpx.HTTPError('error'))

    result = _get_router_info('test/model')
    assert result is None


def test_get_router_info_non_200(mocker: MockerFixture):
    """Test _get_router_info handles non-200 status."""
    _get_router_info.cache_clear()

    mock_response = Mock()
    mock_response.status_code = 404

    mocker.patch('pydantic_ai.providers.huggingface.httpx.get', return_value=mock_response)

    result = _get_router_info('test/model')
    assert result is None


def test_model_profile_with_router_info(mocker: MockerFixture):
    """Test model_profile uses router info to select provider and set capabilities.

    This also tests that when base_profile is None (unknown prefix), a new ModelProfile is created.
    """
    _get_router_info.cache_clear()

    ns = 'pydantic_ai.providers.huggingface'
    router_info = {
        'id': 'unknown/model',
        'providers': [
            {'provider': 'test-provider', 'status': 'live', 'supports_tools': True, 'supports_structured_output': True},
        ],
    }
    mocker.patch(f'{ns}._get_router_info', return_value=router_info)
    mock_client_class = mocker.patch(f'{ns}.AsyncInferenceClient')

    mock_client = Mock(spec=AsyncInferenceClient)
    provider = HuggingFaceProvider(hf_client=mock_client, api_key='test-api-key')

    # 'unknown' prefix doesn't match any known provider, so base_profile starts as None
    # Router info is found, so a fresh ModelProfile is created
    profile = provider.model_profile('unknown/model')

    assert profile is not None
    assert profile.supports_tools is True
    assert profile.supports_json_schema_output is True
    assert profile.supports_json_object_output is True
    # Verify the client was updated with the selected provider
    mock_client_class.assert_called_with(token='test-api-key', provider='test-provider')
    # Verify the provider's client was updated
    assert provider.client is mock_client_class.return_value


def test_model_profile_with_provider_name_override(mocker: MockerFixture):
    """Test model_profile respects provider_name override."""
    _get_router_info.cache_clear()

    ns = 'pydantic_ai.providers.huggingface'
    router_info = {
        'id': 'unknown/model',
        'providers': [
            {'provider': 'default', 'status': 'live', 'supports_tools': True, 'supports_structured_output': True},
            {'provider': 'override', 'status': 'live', 'supports_tools': False, 'supports_structured_output': False},
        ],
    }
    mocker.patch(f'{ns}._get_router_info', return_value=router_info)
    mock_client_class = mocker.patch(f'{ns}.AsyncInferenceClient')

    provider = HuggingFaceProvider(api_key='test-api-key', provider_name='override')

    profile = provider.model_profile('unknown/model')

    assert profile is not None
    assert profile.supports_tools is False
    mock_client_class.assert_called_with(token='test-api-key', provider='override')


def test_model_profile_provider_name_not_found_fallback(mocker: MockerFixture):
    """Test model_profile falls back to select_provider when provider_name not found."""
    _get_router_info.cache_clear()

    ns = 'pydantic_ai.providers.huggingface'
    router_info = {
        'id': 'unknown/model',
        'providers': [
            {'provider': 'available', 'status': 'live', 'supports_tools': True, 'supports_structured_output': True},
        ],
    }
    mocker.patch(f'{ns}._get_router_info', return_value=router_info)
    mock_client_class = mocker.patch(f'{ns}.AsyncInferenceClient')

    provider = HuggingFaceProvider(api_key='test-api-key', provider_name='nonexistent')

    profile = provider.model_profile('unknown/model')

    assert profile is not None
    # Falls back to 'available' since 'nonexistent' not found
    mock_client_class.assert_called_with(token='test-api-key', provider='available')


def test_model_profile_logs_warning_no_structured_output(mocker: MockerFixture, caplog: pytest.LogCaptureFixture):
    """Test model_profile logs warning when provider doesn't support structured output."""
    _get_router_info.cache_clear()

    ns = 'pydantic_ai.providers.huggingface'
    router_info = {
        'id': 'unknown/model',
        'providers': [
            {'provider': 'limited', 'status': 'live', 'supports_tools': True, 'supports_structured_output': False},
        ],
    }
    mocker.patch(f'{ns}._get_router_info', return_value=router_info)
    mocker.patch(f'{ns}.AsyncInferenceClient')

    mock_client = Mock(spec=AsyncInferenceClient)
    provider = HuggingFaceProvider(hf_client=mock_client, api_key='test-api-key')

    with caplog.at_level(logging.WARNING):
        provider.model_profile('unknown/model')

    assert 'Provider limited does not support structured output' in caplog.text


def test_model_profile_logs_warning_no_tools(mocker: MockerFixture, caplog: pytest.LogCaptureFixture):
    """Test model_profile logs warning when provider doesn't support tools."""
    _get_router_info.cache_clear()

    ns = 'pydantic_ai.providers.huggingface'
    router_info = {
        'id': 'unknown/model',
        'providers': [
            {'provider': 'limited', 'status': 'live', 'supports_tools': False, 'supports_structured_output': True},
        ],
    }
    mocker.patch(f'{ns}._get_router_info', return_value=router_info)
    mocker.patch(f'{ns}.AsyncInferenceClient')

    mock_client = Mock(spec=AsyncInferenceClient)
    provider = HuggingFaceProvider(hf_client=mock_client, api_key='test-api-key')

    with caplog.at_level(logging.WARNING):
        provider.model_profile('unknown/model')

    assert "Provider 'limited' does not support tools" in caplog.text


def test_huggingface_provider_api_key_from_hf_client(monkeypatch: pytest.MonkeyPatch):
    """Test api_key is extracted from hf_client.token when not provided."""
    monkeypatch.delenv('HF_TOKEN', raising=False)

    mock_client = Mock(spec=AsyncInferenceClient)
    mock_client.token = 'client-token'

    provider = HuggingFaceProvider(hf_client=mock_client)
    assert provider.api_key == 'client-token'
