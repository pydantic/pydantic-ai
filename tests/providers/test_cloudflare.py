import httpx2
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.models.cloudflare_clef import CloudflareClefModel
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.providers.cloudflare import CloudflareProvider


def test_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'env-token')
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'env-account')
    provider = CloudflareProvider()
    assert provider.name == 'cloudflare'
    assert provider.api_key == 'env-token'
    assert provider.base_url == 'https://api.cloudflare.com/client/v4/accounts/env-account'
    assert infer_provider_class('cloudflare') is CloudflareProvider


def test_explicit_configuration(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'env-token')
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'env-account')
    provider = CloudflareProvider(api_key='explicit-token', account_id='explicit-account')
    assert provider.api_key == 'explicit-token'
    assert provider.base_url == 'https://api.cloudflare.com/client/v4/accounts/explicit-account'


def test_account_url_takes_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('CLOUDFLARE_ACCOUNT_ID', raising=False)
    provider = CloudflareProvider(
        api_key='explicit-token', account_id='ignored-account', base_url='https://example.com/accounts/url-account/'
    )
    assert provider.base_url == 'https://example.com/accounts/url-account'


def test_missing_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('CLOUDFLARE_API_TOKEN', raising=False)
    with pytest.raises(UserError, match='CLOUDFLARE_API_TOKEN'):
        CloudflareProvider(account_id='account')


def test_missing_account(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('CLOUDFLARE_ACCOUNT_ID', raising=False)
    with pytest.raises(UserError, match='CLOUDFLARE_ACCOUNT_ID'):
        CloudflareProvider(api_key='token')


def test_default_model_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'env-token')
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'env-account')
    model = CloudflareClefModel('@cf/cloudflare/clef')
    assert model.system == 'cloudflare'
    assert model.model_name == '@cf/cloudflare/clef'
    assert model.base_url == 'https://api.cloudflare.com/client/v4/accounts/env-account'


async def test_provider_recreates_owned_client():
    provider = CloudflareProvider(api_key='token', account_id='account')
    async with provider:
        original = provider.client
        assert not original.is_closed
    assert original.is_closed
    async with provider:
        replacement = provider.client
        assert replacement is not original
        assert not replacement.is_closed
    assert replacement.is_closed


async def test_custom_client_is_not_owned():
    async with httpx2.AsyncClient() as client:
        provider = CloudflareProvider(api_key='token', account_id='account', http_client=client)
        model = CloudflareClefModel('@cf/cloudflare/clef-flash', provider=provider)
        async with model:
            assert provider.client is client
            assert model.client is client
        assert not client.is_closed
