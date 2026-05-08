"""Tests for the Google + GCP provider split (card 08 Phase B).

Covers the four 1.x deprecation paths:

- `'google-gla:'` prefix → `'google:'`
- `'google-vertex:'` prefix → `'gcp:'`
- `GoogleProvider(vertexai=True, ...)` → `GCPProvider(...)`
- `GoogleProvider(vertexai=False, ...)` → drop the explicit kwarg

Plus the new shapes that should NOT warn:

- `'google:gemini-...'` (canonical) resolves to `GoogleProvider`
- `'gcp:gemini-...'` (canonical) resolves to `GCPProvider`
- `GCPProvider(...)` does not re-emit the `vertexai=` warning when forwarding internally
"""

from __future__ import annotations as _annotations

import warnings

import pytest

from .conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers import infer_provider, infer_provider_class
    from pydantic_ai.providers.gcp import GCPProvider
    from pydantic_ai.providers.google import GoogleProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')


@pytest.fixture(autouse=True)
def _set_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """`GoogleProvider()` defaults to GLA mode and requires an API key on construction."""
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


def test_google_vertex_prefix_warns_and_routes_to_gcp_provider() -> None:
    with pytest.warns(DeprecationWarning, match=r"'google-vertex.' prefix is deprecated"):
        assert infer_provider_class('google-vertex') is GCPProvider
    with pytest.warns(DeprecationWarning, match=r"'google-vertex.' prefix is deprecated"):
        provider = infer_provider('google-vertex')
    assert isinstance(provider, GCPProvider)
    assert provider.name == 'gcp'


def test_gcp_prefix_no_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert infer_provider_class('gcp') is GCPProvider
        provider = infer_provider('gcp')
    assert isinstance(provider, GCPProvider)


def test_google_provider_vertexai_true_warns() -> None:
    with pytest.warns(DeprecationWarning, match=r'`GoogleProvider\(vertexai=True'):
        GoogleProvider(vertexai=True, project='p', location='us-central1')  # pyright: ignore[reportCallIssue]


def test_google_provider_vertexai_false_warns() -> None:
    with pytest.warns(DeprecationWarning, match=r'`GoogleProvider\(vertexai=False'):
        GoogleProvider(vertexai=False, api_key='k')


def test_gcp_provider_no_warning_on_construction() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        provider = GCPProvider(project='p', location='us-central1')
    assert isinstance(provider, GCPProvider)


def test_gcp_provider_is_google_provider() -> None:
    provider = GCPProvider(project='p', location='us-central1')
    assert isinstance(provider, GoogleProvider)
