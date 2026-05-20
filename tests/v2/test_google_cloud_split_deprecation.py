"""Tests for the Google + Google Cloud provider split.

Covers the 1.x deprecation paths:

- `'google-gla:'` prefix → `'google:'`
- `'google-vertex:'` prefix → `'google-cloud:'`
- `'gateway/google-vertex:'` prefix → `'gateway/google-cloud:'`
- `GoogleModelSettings(google_vertex_service_tier=...)` → `google_cloud_service_tier`
- `GoogleProvider(vertexai=True, ...)` and the Google Cloud-only kwargs
  (`location`, `project`, `credentials`) on `GoogleProvider` → `GoogleCloudProvider(...)`
- `GoogleProvider(vertexai=False, ...)` → drop the redundant kwarg

Plus the new shapes that should NOT warn:

- `'google:gemini-...'` resolves to `GoogleProvider`
- `'google-cloud:gemini-...'` resolves to `GoogleCloudProvider`
- `'gateway/google-cloud:'` routes through the gateway without warning
- `GoogleCloudProvider(...)` does not re-emit the warning when forwarding internally
- `GoogleProvider(client=...)` (user-supplied client) does not warn
"""

from __future__ import annotations as _annotations

import warnings

import pytest

from pydantic_ai._warnings import PydanticAIDeprecationWarning

from ..conftest import try_import

with try_import() as imports_successful:
    from google.genai.client import Client

    from pydantic_ai.models.google import (
        GoogleModelSettings,
        _resolve_google_cloud_service_tier,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers import infer_provider, infer_provider_class
    from pydantic_ai.providers.gateway import normalize_gateway_provider
    from pydantic_ai.providers.google import BaseGoogleProvider, GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')


@pytest.fixture(autouse=True)
def _set_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """`GoogleProvider()` defaults to the Gemini API and requires an API key on construction."""
    monkeypatch.setenv('GOOGLE_API_KEY', 'mock-api-key')


def test_google_gla_prefix_warns_and_routes_to_google_provider() -> None:
    with pytest.warns(PydanticAIDeprecationWarning, match=r"'google-gla.' prefix is deprecated"):
        assert infer_provider_class('google-gla') is GoogleProvider
    with pytest.warns(PydanticAIDeprecationWarning, match=r"'google-gla.' prefix is deprecated"):
        provider = infer_provider('google-gla')
    assert type(provider) is GoogleProvider


def test_google_prefix_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert infer_provider_class('google') is GoogleProvider
        provider = infer_provider('google')
    assert type(provider) is GoogleProvider
    assert provider.name == 'google'


def test_google_vertex_prefix_warns_and_routes_to_google_cloud_provider() -> None:
    with pytest.warns(PydanticAIDeprecationWarning, match=r"'google-vertex.' prefix is deprecated"):
        assert infer_provider_class('google-vertex') is GoogleCloudProvider
    with pytest.warns(PydanticAIDeprecationWarning, match=r"'google-vertex.' prefix is deprecated"):
        provider = infer_provider('google-vertex')
    assert isinstance(provider, GoogleCloudProvider)
    assert provider.name == 'google-cloud'


def test_google_cloud_prefix_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert infer_provider_class('google-cloud') is GoogleCloudProvider
        provider = infer_provider('google-cloud')
    assert isinstance(provider, GoogleCloudProvider)
    assert provider.name == 'google-cloud'


def test_google_provider_vertexai_true_warns() -> None:
    with pytest.warns(PydanticAIDeprecationWarning, match=r'Google Cloud .* arguments'):
        GoogleProvider(vertexai=True, project='p', location='us-central1')  # pyright: ignore[reportCallIssue]


def test_google_provider_vertex_kwargs_warn() -> None:
    """Vertex-only kwargs (`location`, `project`, `credentials`) trigger the deprecation even without `vertexai=True`.

    The google-genai SDK silently routes to Google Cloud when these kwargs are present, so a user could end
    up on Google Cloud without ever passing `vertexai=True` — we still want to steer them to `GoogleCloudProvider`.
    """
    with pytest.warns(PydanticAIDeprecationWarning, match=r'Google Cloud .* arguments'):
        GoogleProvider(project='p', location='us-central1')


def test_google_provider_vertexai_false_warns() -> None:
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`GoogleProvider\(vertexai=False'):
        GoogleProvider(vertexai=False, api_key='k')


def test_google_provider_custom_client_no_warning() -> None:
    """Passing a custom client overrides everything — we have no signal about whether Google Cloud is in use."""
    client = Client(vertexai=False, api_key='mock-api-key')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        provider = GoogleProvider(client=client)
    assert isinstance(provider, GoogleProvider)


def test_google_cloud_provider_no_warning_on_construction() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        provider = GoogleCloudProvider(project='p', location='us-central1')
    assert isinstance(provider, GoogleCloudProvider)


def test_google_cloud_provider_custom_client_no_warning() -> None:
    """`GoogleCloudProvider(client=...)` short-circuits construction and stores the supplied client."""
    client = Client(vertexai=False, api_key='mock-api-key')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        provider = GoogleCloudProvider(client=client)
    assert isinstance(provider, GoogleCloudProvider)
    assert provider.client is client


def test_google_cloud_provider_shares_base_with_google_provider() -> None:
    """Both classes inherit from `BaseGoogleProvider` (which owns the shared client wiring),
    rather than `GoogleCloudProvider` subclassing `GoogleProvider`. This avoids the trap
    where `GoogleCloudProvider.__init__` would inherit `GoogleProvider`'s deprecation warnings."""
    provider = GoogleCloudProvider(project='p', location='us-central1')
    assert isinstance(provider, BaseGoogleProvider)
    assert not isinstance(provider, GoogleProvider)


def test_gateway_google_vertex_prefix_warns_and_routes_to_google_cloud_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`gateway/google-vertex` is a user-facing alias that warns + routes to the Google Cloud path."""
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_API_KEY', 'mock-key')
    with pytest.warns(PydanticAIDeprecationWarning, match=r"'gateway/google-vertex.' prefix is deprecated"):
        assert infer_provider_class('gateway/google-vertex') is GoogleCloudProvider
    with pytest.warns(PydanticAIDeprecationWarning, match=r"'gateway/google-vertex.' prefix is deprecated"):
        provider = infer_provider('gateway/google-vertex')
    assert isinstance(provider, GoogleCloudProvider)


def test_gateway_gemini_prefix_warns_and_routes_to_google_cloud_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`gateway/gemini` is a user-facing alias that warns + routes to the Google Cloud path."""
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_API_KEY', 'mock-key')
    with pytest.warns(PydanticAIDeprecationWarning, match=r"'gateway/gemini.' prefix is deprecated"):
        assert infer_provider_class('gateway/gemini') is GoogleCloudProvider
    with pytest.warns(PydanticAIDeprecationWarning, match=r"'gateway/gemini.' prefix is deprecated"):
        provider = infer_provider('gateway/gemini')
    assert isinstance(provider, GoogleCloudProvider)


def test_gateway_google_cloud_prefix_no_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """`gateway/google-cloud` is the new canonical user-facing prefix — no deprecation warning."""
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_API_KEY', 'mock-key')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert infer_provider_class('gateway/google-cloud') is GoogleCloudProvider
        provider = infer_provider('gateway/google-cloud')
    assert isinstance(provider, GoogleCloudProvider)


def test_gateway_google_cloud_maps_to_legacy_gateway_wire_value() -> None:
    """Rule 17: the user-facing rename ships ahead of the Gateway API rename.

    Both `gateway/google-cloud` and `gateway/google-vertex` must still serialize to
    `'google-vertex'` on the wire until Daniel renames the Gateway side.
    """
    assert normalize_gateway_provider('gateway/google-cloud') == 'google-vertex'
    with pytest.warns(PydanticAIDeprecationWarning, match=r"'gateway/google-vertex.' prefix is deprecated"):
        assert normalize_gateway_provider('gateway/google-vertex') == 'google-vertex'


def test_google_vertex_service_tier_warns_and_routes_to_google_cloud_service_tier() -> None:
    """Old `google_vertex_service_tier` key still routes correctly + emits the rename warning."""
    settings = GoogleModelSettings(google_vertex_service_tier='pt_only')
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`google_vertex_service_tier` is deprecated'):
        assert _resolve_google_cloud_service_tier(settings) == 'pt_only'


def test_google_cloud_service_tier_no_warning() -> None:
    """New `google_cloud_service_tier` key is the canonical name — must not warn."""
    settings = GoogleModelSettings(google_cloud_service_tier='pt_only')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _resolve_google_cloud_service_tier(settings) == 'pt_only'


def test_google_cloud_service_tier_wins_over_deprecated_google_vertex_service_tier() -> None:
    """When both are set, the new name takes precedence (and the old still warns when read)."""
    settings = GoogleModelSettings(
        google_cloud_service_tier='pt_only',
        google_vertex_service_tier='flex_only',
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _resolve_google_cloud_service_tier(settings) == 'pt_only'
