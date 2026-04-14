"""Fixtures for Responses API WebSocket cassette recording/replay."""

from __future__ import annotations as _annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from ..conftest import sanitize_filename, try_import

with try_import() as imports_successful:
    from websockets.asyncio.client import connect as _real_ws_connect

    from pydantic_ai.models.openai import OpenAIResponsesModel

    from .cassettes import (
        RecordingWebSocket,
        ReplayWebSocket,
        WebSocketCassette,
        ws_cassette_plan,
    )


def _cassette_dir_for(request: pytest.FixtureRequest) -> Path:
    """Derive the cassette directory from the requesting test module, matching VCR convention."""
    test_path = request.path
    return test_path.parent / 'cassettes' / test_path.stem


def _get_record_mode(request: pytest.FixtureRequest) -> str | None:
    try:
        return cast(Any, request.config).getoption('record_mode')
    except (ValueError, AttributeError):  # pragma: no cover - depends on pytest-recording plugin presence
        return None


class _ReplayConnect:
    """Mimics websockets.connect — awaitable (returns ws) and async context manager."""

    def __init__(self, ws: ReplayWebSocket):
        self._ws = ws

    def __await__(self) -> Any:
        async def _resolve() -> ReplayWebSocket:
            return self._ws

        return _resolve().__await__()

    async def __aenter__(self) -> ReplayWebSocket:
        return self._ws

    async def __aexit__(self, *args: Any) -> None:
        pass


class _RecordingConnect:  # pragma: no cover - only used during live cassette recording
    """Mimics websockets.connect — awaitable and async context manager, wrapping real connection."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._args = args
        self._kwargs = kwargs
        self._real: Any = None
        self._recording_ws: RecordingWebSocket | None = None
        self._cassette: WebSocketCassette | None = None

    def with_cassette(self, cassette: WebSocketCassette) -> _RecordingConnect:
        self._cassette = cassette
        return self

    def __await__(self) -> Any:
        async def _resolve() -> RecordingWebSocket:
            assert self._cassette is not None
            self._real = _real_ws_connect(*self._args, **self._kwargs)
            ws = await self._real
            self._recording_ws = RecordingWebSocket(ws, self._cassette)
            return self._recording_ws

        return _resolve().__await__()

    async def __aenter__(self) -> RecordingWebSocket:
        return await self

    async def __aexit__(self, *args: Any) -> None:
        if self._recording_ws is not None:
            await self._recording_ws.close()


@pytest.fixture
def openai_ws_model(
    request: pytest.FixtureRequest,
    openai_api_key: str,
    allow_model_requests: None,
) -> Iterator[OpenAIResponsesModel]:
    """OpenAI Responses model backed by WebSocket cassettes."""
    cassette_name = sanitize_filename(request.node.name, 240)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    cassette_path = _cassette_dir_for(request) / f'{cassette_name}.yaml'
    record_mode = _get_record_mode(request)

    plan = ws_cassette_plan(cassette_exists=cassette_path.exists(), record_mode=record_mode)

    if plan == 'error_missing':  # pragma: no cover - only when cassette files are missing
        raise RuntimeError(
            f'Missing WebSocket cassette: {cassette_path}\n'
            'Record with: OPENAI_API_KEY=... uv run pytest --record-mode=once <test> -v'
        )

    cassette = WebSocketCassette.load(cassette_path) if plan == 'replay' else WebSocketCassette()

    def fake_connect(*args: Any, **kwargs: Any) -> _ReplayConnect:
        return _ReplayConnect(ReplayWebSocket(cassette))

    def recording_connect(*args: Any, **kwargs: Any) -> _RecordingConnect:  # pragma: no cover
        return _RecordingConnect(*args, **kwargs).with_cassette(cassette)

    mock_connect = fake_connect if plan == 'replay' else recording_connect

    with patch('websockets.asyncio.client.connect', mock_connect):
        from pydantic_ai.providers.openai import OpenAIProvider

        yield OpenAIResponsesModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))

    if plan == 'record' and any(i.direction == 'received' for i in cassette.interactions):  # pragma: no cover
        cassette.dump(cassette_path)
