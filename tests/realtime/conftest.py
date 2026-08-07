"""Realtime test configuration."""

from __future__ import annotations as _annotations

import os
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from ..conftest import sanitize_filename, try_import
from .ws_cassettes import ProviderName, RealtimeCassette, patched_ws_connect, realtime_cassette_plan

with try_import() as openai_imports_successful:
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as azure_imports_successful:
    from pydantic_ai.providers.azure import AzureProvider

if TYPE_CHECKING:
    from pydantic_ai.providers import Provider

CASSETTES_DIR = Path(__file__).parent / 'cassettes'


@pytest.fixture(autouse=True)
def _realtime_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide placeholder API keys so realtime models can resolve their default providers offline.

    The realtime models resolve their provider (and its API client) eagerly at construction, like
    `OpenAIChatModel`. Network-free tests never hit the network, so a placeholder key is enough to let
    `OpenAIRealtimeModel()` build its default provider.

    The cassette fixtures build their provider from the session-scoped `openai_api_key` fixture, which
    is resolved before this (function-scoped) override runs and reads a real key from the environment
    when recording, so this placeholder doesn't interfere with them.
    """
    monkeypatch.setenv('OPENAI_API_KEY', 'mock-api-key')
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://mock.openai.azure.com/openai/v1')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'mock-api-key')


def _record_mode(request: pytest.FixtureRequest) -> str | None:
    try:
        return cast('Any', request.config).getoption('record_mode')
    # Depends on pytest-recording being active.
    except (ValueError, AttributeError):  # pragma: no cover
        return None


@contextmanager
def _ws_cassette(request: pytest.FixtureRequest, provider: ProviderName) -> Generator[RealtimeCassette]:
    """Patch the provider's WebSocket transport to replay from / record into this test's cassette."""
    module = cast('str', request.node.fspath.basename).replace('.py', '')  # pyright: ignore[reportUnknownMemberType]
    name = sanitize_filename(cast('str', request.node.name), 240)  # pyright: ignore[reportUnknownMemberType]
    path = CASSETTES_DIR / module / f'{name}.yaml'
    plan = realtime_cassette_plan(cassette_exists=path.exists(), record_mode=_record_mode(request))
    if plan == 'error_missing':
        # A cassette we expect to exist has gone missing.
        raise RuntimeError(  # pragma: no cover
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
        # Only runs while recording.
        if plan == 'record' and cassette.interactions:  # pragma: no cover
            cassette.dump(path)


@pytest.fixture
def openai_ws_cassette(
    request: pytest.FixtureRequest, openai_api_key: str
) -> Iterator[tuple[Provider[Any], RealtimeCassette]]:
    """An `OpenAIProvider` whose realtime WebSocket is backed by a cassette."""
    if not openai_imports_successful():  # pragma: no cover
        pytest.skip('openai / websockets not installed')
    with _ws_cassette(request, 'openai') as cassette:
        yield OpenAIProvider(api_key=openai_api_key), cassette


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
    # Mirror `AzureProvider.for_realtime`'s normalization: only append `/openai/v1` when the
    # configured endpoint doesn't already end with it, so an env already set to the GA form
    # doesn't dial `.../openai/v1/openai/v1`. Replay uses a suffix-less placeholder endpoint, so
    # only recording against a GA-form env ever takes the other branch.
    if not endpoint.rstrip('/').endswith('/openai/v1'):  # pragma: no branch
        endpoint = f'{endpoint.rstrip("/")}/openai/v1'
    with _ws_cassette(request, 'openai') as cassette:
        yield AzureProvider(azure_endpoint=endpoint, api_key=api_key), cassette
