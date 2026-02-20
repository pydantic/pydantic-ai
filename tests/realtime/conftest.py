"""Fixtures for realtime WebSocket cassette recording/replay."""

from __future__ import annotations as _annotations

import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from ..conftest import sanitize_filename, try_import

with try_import() as imports_successful:
    from websockets.asyncio.client import connect as _real_ws_connect

    from pydantic_ai.realtime.gemini import GeminiRealtimeModel
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel

    from .cassettes import (
        RealtimeCassette,
        RecordingWebSocket,
        ReplayWebSocket,
        realtime_cassette_plan,
    )

CASSETTES_DIR = Path(__file__).parent / 'ws_cassettes'


def _get_record_mode(request: pytest.FixtureRequest) -> str | None:
    try:
        return cast(Any, request.config).getoption('record_mode')
    except (ValueError, AttributeError):
        return None


@pytest.fixture
def openai_realtime_model(
    request: pytest.FixtureRequest,
    openai_api_key: str,
    allow_model_requests: None,
) -> Iterator[OpenAIRealtimeModel]:
    """OpenAI Realtime model backed by WebSocket cassettes."""
    cassette_name = sanitize_filename(request.node.name, 240)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    cassette_path = CASSETTES_DIR / 'test_openai' / f'{cassette_name}.yaml'
    record_mode = _get_record_mode(request)

    plan = realtime_cassette_plan(cassette_exists=cassette_path.exists(), record_mode=record_mode)

    if plan == 'error_missing':
        raise RuntimeError(
            f'Missing realtime cassette: {cassette_path}\n'
            'Record with: OPENAI_API_KEY=... uv run pytest --record-mode=once <test> -v'
        )

    cassette = RealtimeCassette.load(cassette_path) if plan == 'replay' else RealtimeCassette()

    @asynccontextmanager
    async def fake_connect(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield ReplayWebSocket(cassette)

    @asynccontextmanager
    async def recording_connect(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async with _real_ws_connect(*args, **kwargs) as ws:
            yield RecordingWebSocket(ws, cassette)

    mock_connect = fake_connect if plan == 'replay' else recording_connect

    with patch('pydantic_ai.realtime.openai.websockets.connect', mock_connect):
        yield OpenAIRealtimeModel(api_key=openai_api_key)

    if plan == 'record' and any(i.direction == 'received' for i in cassette.interactions):
        cassette.dump(cassette_path)


@pytest.fixture
def gemini_realtime_model(
    request: pytest.FixtureRequest,
    vertex_provider_auth: None,
    allow_model_requests: None,
) -> Iterator[GeminiRealtimeModel]:
    """Gemini Realtime model backed by WebSocket cassettes.

    Gemini Live requires Vertex AI auth. In CI the WebSocket is fully mocked
    so no real auth is needed. Locally the tests are skipped by default.

    To rewrite cassettes locally:
    1. Run ``gcloud auth application-default login``
    2. Comment out the ``pytest.skip(...)`` line below
    3. Run with ``--record-mode=rewrite``
    """
    # NOTE: You need to comment out this line to rewrite the cassettes locally.
    if not os.getenv('CI'):
        pytest.skip('Requires gcloud auth application-default login to record')

    cassette_name = sanitize_filename(request.node.name, 240)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    cassette_path = CASSETTES_DIR / 'test_gemini' / f'{cassette_name}.yaml'
    record_mode = _get_record_mode(request)

    plan = realtime_cassette_plan(cassette_exists=cassette_path.exists(), record_mode=record_mode)

    if plan == 'error_missing':
        raise RuntimeError(
            f'Missing realtime cassette: {cassette_path}\nRecord with: uv run pytest --record-mode=once <test> -v'
        )

    cassette = RealtimeCassette.load(cassette_path) if plan == 'replay' else RealtimeCassette()

    @asynccontextmanager
    async def fake_ws_connect(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        yield ReplayWebSocket(cassette)

    @asynccontextmanager
    async def recording_ws_connect(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async with _real_ws_connect(*args, **kwargs) as ws:
            yield RecordingWebSocket(ws, cassette)

    mock_connect = fake_ws_connect if plan == 'replay' else recording_ws_connect

    project = os.getenv('GOOGLE_CLOUD_PROJECT', 'pydantic-ai')
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    model = os.getenv('GOOGLE_LIVE_MODEL', 'gemini-live-2.5-flash-native-audio')

    with patch('google.genai.live.ws_connect', mock_connect):
        yield GeminiRealtimeModel(model=model, project=project, location=location)

    if plan == 'record' and any(i.direction == 'received' for i in cassette.interactions):
        cassette.dump(cassette_path)
