"""Realtime test configuration.

`test_finance_demo.py` imports the `pydantic_ai_examples` package, which isn't installed in the main
test environment (examples are import-checked separately via `tests/import_examples.py`). Skip
collecting it unless that package is available, so the unit-test job doesn't error on import.
"""

from __future__ import annotations as _annotations

import importlib.util
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

if TYPE_CHECKING:
    from pydantic_ai.providers import Provider

collect_ignore: list[str] = []

if importlib.util.find_spec('pydantic_ai_examples') is None:
    collect_ignore.append('test_finance_demo.py')

CASSETTES_DIR = Path(__file__).parent / 'cassettes'


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


def _record_mode(request: pytest.FixtureRequest) -> str | None:
    try:
        return cast('Any', request.config).getoption('record_mode')
    except (ValueError, AttributeError):  # pragma: no cover - depends on pytest-recording being active
        return None


@contextmanager
def _ws_cassette(request: pytest.FixtureRequest, provider: ProviderName) -> Generator[RealtimeCassette]:
    """Patch the provider's WebSocket transport to replay from / record into this test's cassette."""
    module = cast('str', request.node.fspath.basename).replace('.py', '')  # pyright: ignore[reportUnknownMemberType]
    name = sanitize_filename(cast('str', request.node.name), 240)  # pyright: ignore[reportUnknownMemberType]
    path = CASSETTES_DIR / module / f'{name}.yaml'
    plan = realtime_cassette_plan(cassette_exists=path.exists(), record_mode=_record_mode(request))
    if plan == 'error_missing':  # pragma: no cover - only when a cassette is missing offline
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
    if not imports_successful():  # pragma: no cover
        pytest.skip('openai / websockets not installed')
    with _ws_cassette(request, 'openai') as cassette:
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
