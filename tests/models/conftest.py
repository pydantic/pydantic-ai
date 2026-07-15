from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest

from tests.conftest import sanitize_filename, try_import

if TYPE_CHECKING:
    from vcr.cassette import Cassette

    from tests.cassette_utils import CassetteContext

with try_import() as imports_successful:
    from websockets.asyncio.client import connect as _real_ws_connect

    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .websocket_cassettes import (
        RecordingWebSocket,
        ReplayConnect,
        ReplayWebSocket,
        WebSocketCassette,
        ws_cassette_plan,
    )


@pytest.fixture(scope='function')
def cassette_ctx(request: pytest.FixtureRequest, vcr: Cassette) -> CassetteContext:
    """Unified cassette verification context for model tests.

    Returns a CassetteContext for tests with a 'provider' parameter, or for
    non-parametrized tests (defaulting to 'vcr' provider).
    """
    from tests.cassette_utils import CassetteContext

    provider = 'vcr'
    if callspec := getattr(request.node, 'callspec', None):  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        params = cast(dict[str, object], callspec.params)
        p = params.get('provider')
        if isinstance(p, str):  # pragma: no branch
            provider = p

    test_module: str = request.node.fspath.basename.replace('.py', '')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    test_dir = Path(request.node.fspath).parent  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    return CassetteContext(
        provider=provider,
        vcr=vcr,
        test_name=request.node.name,  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        test_module=test_module,  # pyright: ignore[reportUnknownArgumentType]
        test_dir=test_dir,
    )


def _ws_cassette_dir_for(request: pytest.FixtureRequest) -> Path:
    """Derive the cassette directory from the requesting test module, matching VCR convention."""
    test_path = request.path
    return test_path.parent / 'cassettes' / test_path.stem


def _get_record_mode(request: pytest.FixtureRequest) -> str | None:
    try:
        return cast(Any, request.config).getoption('record_mode')
    except (ValueError, AttributeError):  # pragma: no cover
        return None


class _RecordingConnect:
    """Mimics `websockets.connect` while wrapping a real connection for recording."""

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
    cassette_path = _ws_cassette_dir_for(request) / f'{cassette_name}.yaml'
    record_mode = _get_record_mode(request)

    existing_cassette = WebSocketCassette.load(cassette_path) if cassette_path.exists() else None
    plan = ws_cassette_plan(
        cassette_exists=existing_cassette is not None,
        cassette_synthetic=bool(existing_cassette and existing_cassette.synthetic),
        record_mode=record_mode,
    )

    if plan == 'error_missing':  # pragma: no cover
        raise RuntimeError(
            f'Missing WebSocket cassette: {cassette_path}\n'
            'Record with: OPENAI_API_KEY=... uv run pytest --record-mode=once <test> -v'
        )
    if plan == 'error_unsupported':  # pragma: no cover
        raise RuntimeError(
            'WebSocket cassettes do not support `--record-mode=new_episodes`; '
            'use `--record-mode=rewrite` to replace the cassette.'
        )

    cassette = existing_cassette if plan == 'replay' else WebSocketCassette()
    assert cassette is not None
    replay_websockets: list[ReplayWebSocket] = []

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        ws = ReplayWebSocket(cassette)
        replay_websockets.append(ws)
        return ReplayConnect(ws)

    def recording_connect(*args: Any, **kwargs: Any) -> _RecordingConnect:  # pragma: no cover
        return _RecordingConnect(*args, **kwargs).with_cassette(cassette)

    mock_connect = fake_connect if plan == 'replay' else recording_connect

    with patch('websockets.asyncio.client.connect', mock_connect):
        yield OpenAIResponsesModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))

    if plan == 'replay':
        assert all(ws.sent_frames_consumed for ws in replay_websockets)
        node = cast(pytest.Item, request.node)  # pyright: ignore[reportUnknownMemberType]
        marker = node.get_closest_marker('ws_cassette')
        allow_unconsumed = bool(marker and marker.kwargs.get('allow_unconsumed'))
        if not allow_unconsumed:
            assert all(ws.interactions_consumed for ws in replay_websockets)

    if plan == 'record' and any(i.direction == 'received' for i in cassette.interactions):  # pragma: no cover
        cassette.dump(cassette_path)
