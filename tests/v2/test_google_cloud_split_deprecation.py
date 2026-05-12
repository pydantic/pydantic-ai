"""Tests for the Google + Google Cloud provider split.

Covers the 1.x deprecation paths:

- `'google-gla:'` prefix → `'google:'`
- `'google-vertex:'` prefix → `'google-cloud:'`
- `GoogleProvider(vertexai=True, ...)` and the Google Cloud-only kwargs
  (`location`, `project`, `credentials`) on `GoogleProvider` → `GoogleCloudProvider(...)`
- `GoogleProvider(vertexai=False, ...)` → drop the redundant kwarg

Plus the new shapes that should NOT warn:

- `'google:gemini-...'` resolves to `GoogleProvider`
- `'google-cloud:gemini-...'` resolves to `GoogleCloudProvider`
- `GoogleCloudProvider(...)` does not re-emit the warning when forwarding internally
- `GoogleProvider(client=...)` (user-supplied client) does not warn
"""

from __future__ import annotations as _annotations

import warnings

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from google.genai.client import Client

    from pydantic_ai.providers import infer_provider, infer_provider_class
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')


@pytest.fixture(autouse=True)
def _set_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """`GoogleProvider()` defaults to the Gemini developer API and requires an API key on construction."""
    monkeypatch.setenv('GOOGLE_API_KEY', 'mock-api-key')


def test_google_gla_prefix_warns_and_routes_to_google_provider() -> None:
    with pytest.warns(DeprecationWarning, match=r"'google-gla.' prefix is deprecated"):
        assert infer_provider_class('google-gla') is GoogleProvider
    with pytest.warns(DeprecationWarning, match=r"'google-gla.' prefix is deprecated"):
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
    with pytest.warns(DeprecationWarning, match=r"'google-vertex.' prefix is deprecated"):
        assert infer_provider_class('google-vertex') is GoogleCloudProvider
    with pytest.warns(DeprecationWarning, match=r"'google-vertex.' prefix is deprecated"):
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
    with pytest.warns(DeprecationWarning, match=r'Google Cloud .* arguments'):
        GoogleProvider(vertexai=True, project='p', location='us-central1')  # pyright: ignore[reportCallIssue]


def test_google_provider_vertex_kwargs_warn() -> None:
    """Vertex-only kwargs (`location`, `project`, `credentials`) trigger the deprecation even without `vertexai=True`.

    The google-genai SDK silently routes to Google Cloud when these kwargs are present, so a user could end
    up on Google Cloud without ever passing `vertexai=True` — we still want to steer them to `GoogleCloudProvider`.
    """
    with pytest.warns(DeprecationWarning, match=r'Google Cloud .* arguments'):
        GoogleProvider(project='p', location='us-central1')


def test_google_provider_vertexai_false_warns() -> None:
    with pytest.warns(DeprecationWarning, match=r'`GoogleProvider\(vertexai=False'):
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


def test_google_cloud_provider_is_google_provider() -> None:
    provider = GoogleCloudProvider(project='p', location='us-central1')
    assert isinstance(provider, GoogleProvider)
