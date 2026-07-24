"""Realtime test configuration."""

from __future__ import annotations as _annotations

import json
import os
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from ..conftest import sanitize_filename, try_import
from .ws_cassettes import ProviderName, RealtimeCassette, patched_ws_connect, realtime_cassette_plan

with try_import() as imports_successful:
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as xai_imports_successful:
    from pydantic_ai.providers.xai import XaiProvider

with try_import() as azure_imports_successful:
    from pydantic_ai.providers.azure import AzureProvider

if TYPE_CHECKING:
    from pydantic_ai.providers import Provider

CASSETTES_DIR = Path(__file__).parent / 'cassettes'


def _scrub_ephemeral_secret(response: dict[str, Any]) -> dict[str, Any]:
    """Redact the short-lived WebRTC client secret from recorded `/realtime/client_secrets` responses.

    The mint response body carries `{"value": "ek_..."}` — the ephemeral browser token. It expires in
    seconds and is useless offline, but replacing it keeps recorded cassettes free of anything
    secret-shaped. (The api-key / Entra bearer used to mint it are filtered out via `filter_headers`.)
    """
    try:
        raw = response['body']['string']
    except (KeyError, TypeError):  # pragma: no cover - non-body responses
        return response
    if not raw:  # pragma: no cover - empty body
        return response
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):  # pragma: no cover - non-JSON body
        return response
    if isinstance(data, dict) and isinstance(data.get('value'), str) and data['value'].startswith('ek_'):
        data['value'] = 'ek_scrubbed'
        body = json.dumps(data)
        response['body']['string'] = body.encode() if isinstance(raw, bytes) else body
    return response


@pytest.fixture(scope='module')
def vcr_config() -> dict[str, Any]:
    """VCR config for realtime HTTP (WebRTC signaling) cassettes.

    Extends the repo default with Azure's `api-key` header (the WebSocket cassettes never record HTTP,
    so the default set omits it) and scrubs the minted ephemeral client secret from response bodies.
    """
    return {
        'ignore_localhost': True,
        'filter_headers': ['authorization', 'x-api-key', 'api-key', 'cookie'],
        'decode_compressed_response': True,
        'before_record_response': _scrub_ephemeral_secret,
    }


@pytest.fixture(autouse=True)
def _realtime_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide placeholder API keys so realtime models can resolve their default providers offline.

    The realtime models resolve their provider (and its API client) eagerly at construction, like
    `OpenAIChatModel` / `GoogleModel`. Network-free tests never hit the network, so a placeholder key
    is enough to let `OpenAIRealtimeModel()` / `GoogleRealtimeModel()` build their default providers.

    The cassette fixtures build their provider from the session-scoped `openai_api_key` /
    `gemini_api_key` fixtures, which are resolved before this (function-scoped) override runs and read
    a real key from the environment when recording, so this placeholder doesn't interfere with them.
    """
    monkeypatch.setenv('OPENAI_API_KEY', 'mock-api-key')
    monkeypatch.setenv('GOOGLE_API_KEY', 'mock-api-key')
    monkeypatch.setenv('XAI_API_KEY', 'mock-api-key')
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://mock.openai.azure.com/openai/v1')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'mock-api-key')


def _record_mode(request: pytest.FixtureRequest) -> str | None:
    try:
        return cast('Any', request.config).getoption('record_mode')
    except (ValueError, AttributeError):  # pragma: no cover - depends on pytest-recording being active
        return None


@contextmanager
def _ws_cassette(
    request: pytest.FixtureRequest, provider: ProviderName, *, skip_if_missing: bool = False, subdir: str | None = None
) -> Generator[RealtimeCassette]:
    """Patch the provider's WebSocket transport to replay from / record into this test's cassette.

    `skip_if_missing` skips (rather than errors) when no cassette exists offline, for providers whose
    cassettes may not have been recorded yet (e.g. xAI, gated on realtime API access for our account).
    `subdir` overrides the cassette subdirectory (default: the test module), so a test that also records
    an HTTP VCR cassette (which uses the module-named subdirectory) doesn't collide with the WS cassette.
    """
    module = cast('str', request.node.fspath.basename).replace('.py', '')  # pyright: ignore[reportUnknownMemberType]
    name = sanitize_filename(cast('str', request.node.name), 240)  # pyright: ignore[reportUnknownMemberType]
    path = CASSETTES_DIR / (subdir or module) / f'{name}.yaml'
    plan = realtime_cassette_plan(cassette_exists=path.exists(), record_mode=_record_mode(request))
    if plan == 'error_missing':  # pragma: no cover - only when a cassette is missing offline
        if skip_if_missing:
            pytest.skip(f'Missing realtime WebSocket cassette (record with `--record-mode=rewrite`): {path}')
        raise RuntimeError(
            f'Missing realtime WebSocket cassette: {path}\n'
            'Record it with: uv run --env-file .env pytest --record-mode=rewrite <test> -q'
        )
    cassette = RealtimeCassette.load(path) if plan == 'replay' else RealtimeCassette()
    try:
        with patched_ws_connect(provider, cassette, plan):
            yield cassette
    finally:
        # Persist recorded frames even if later assertions fail, so cassettes can be recorded first
        # and snapshots filled from replay afterwards (mirroring the VCR workflow).
        if plan == 'record' and cassette.interactions:  # pragma: no cover - only runs while recording
            cassette.dump(path)


@pytest.fixture
def openai_ws_cassette(
    request: pytest.FixtureRequest, openai_api_key: str
) -> Iterator[tuple[Provider[Any], RealtimeCassette]]:
    """An `OpenAIProvider` whose realtime WebSocket is backed by a cassette."""
    if not imports_successful():
        pytest.skip('openai / websockets not installed')
    with _ws_cassette(request, 'openai') as cassette:
        yield OpenAIProvider(api_key=openai_api_key), cassette


@pytest.fixture
def openai_ws_sideband_cassette(
    request: pytest.FixtureRequest, openai_api_key: str
) -> Iterator[tuple[Provider[Any], RealtimeCassette]]:
    """An `OpenAIProvider` whose realtime sideband control WebSocket is cassette-backed.

    Stored under a dedicated subdirectory so the WebSocket cassette doesn't collide with the HTTP VCR
    cassette (SDP offer relay) a WebRTC sideband test records under the module-named subdirectory.
    """
    if not imports_successful():
        pytest.skip('openai / websockets not installed')
    with _ws_cassette(request, 'openai', subdir='test_openai_ws_sideband') as cassette:
        yield OpenAIProvider(api_key=openai_api_key), cassette


@pytest.fixture
def gemini_ws_cassette(
    request: pytest.FixtureRequest, gemini_api_key: str
) -> Iterator[tuple[Provider[Any], RealtimeCassette]]:
    """A `GoogleProvider` whose Gemini Live WebSocket is backed by a cassette."""
    if not imports_successful():  # pragma: no cover
        pytest.skip('google-genai not installed')
    with _ws_cassette(request, 'gemini') as cassette:
        yield GoogleProvider(api_key=gemini_api_key), cassette


@pytest.fixture
def xai_ws_cassette(request: pytest.FixtureRequest, xai_api_key: str) -> Iterator[tuple[XaiProvider, RealtimeCassette]]:
    """An `XaiProvider` whose Grok Voice realtime WebSocket is backed by a cassette.

    Skips (rather than errors) when the cassette is missing offline: recording requires xAI realtime
    API access, which our account may not have, so these cassettes may not be present.
    """
    if not xai_imports_successful():  # pragma: no cover
        pytest.skip('xai-sdk / websockets not installed')
    with _ws_cassette(request, 'xai', skip_if_missing=True) as cassette:
        yield XaiProvider(api_key=xai_api_key), cassette


@pytest.fixture(scope='session')
def azure_config() -> tuple[str, str]:
    """Capture real Azure OpenAI configuration before offline placeholders apply."""
    return (
        os.getenv('AZURE_OPENAI_ENDPOINT', 'https://mock.openai.azure.com'),
        os.getenv('AZURE_OPENAI_API_KEY', 'mock-api-key'),
    )


@pytest.fixture
def azure_ws_cassette(
    request: pytest.FixtureRequest, azure_config: tuple[str, str]
) -> Iterator[tuple[AzureProvider, RealtimeCassette]]:
    """An `AzureProvider` whose Azure OpenAI realtime WebSocket is cassette-backed."""
    if not azure_imports_successful():  # pragma: no cover
        pytest.skip('openai / websockets not installed')
    endpoint, api_key = azure_config
    with _ws_cassette(request, 'openai') as cassette:
        yield AzureProvider(azure_endpoint=f'{endpoint.rstrip("/")}/openai/v1', api_key=api_key), cassette
