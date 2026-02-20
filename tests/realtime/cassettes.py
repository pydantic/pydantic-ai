"""WebSocket cassette utilities for realtime provider tests.

VCR.py can't record WebSocket traffic. Both OpenAI and Gemini use the `websockets`
library under the hood, so we intercept at that level for a unified cassette format.

Only the actual TCP/TLS connection is replaced - everything else (URL construction,
auth headers, SDK setup, event mapping) runs for real.
"""

from __future__ import annotations as _annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..conftest import try_import

with try_import() as imports_successful:
    import yaml
    from websockets.exceptions import ConnectionClosedOK


# ---------------------------------------------------------------------------
# Cassette data model
# ---------------------------------------------------------------------------


@dataclass
class CassetteInteraction:
    """A single WebSocket frame (sent or received)."""

    direction: Literal['sent', 'received']
    data: dict[str, Any]


@dataclass
class RealtimeCassette:
    """Ordered list of WebSocket interactions."""

    version: int = 1
    interactions: list[CassetteInteraction] = field(default_factory=list[CassetteInteraction])

    @classmethod
    def load(cls, path: Path) -> RealtimeCassette:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding='utf-8'))
        raw_interactions: list[dict[str, Any]] = raw.get('interactions', [])
        interactions: list[CassetteInteraction] = [
            CassetteInteraction(direction=item['direction'], data=item['data']) for item in raw_interactions
        ]
        return cls(version=raw.get('version', 1), interactions=interactions)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            'version': self.version,
            'interactions': [{'direction': i.direction, 'data': i.data} for i in self.interactions],
        }
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding='utf-8',
        )


def realtime_cassette_plan(
    *,
    cassette_exists: bool,
    record_mode: str | None,
) -> Literal['replay', 'record', 'error_missing']:
    """Decide replay vs record behavior, mirroring `_proto_cassette_plan` logic."""
    if record_mode is None:
        record_mode = 'none'

    mode = record_mode.strip().lower()

    if mode == 'none':
        return 'replay' if cassette_exists else 'error_missing'
    if mode == 'once':
        return 'replay' if cassette_exists else 'record'
    if mode in {'rewrite', 'all', 'new_episodes'}:
        return 'record'

    return 'replay' if cassette_exists else 'error_missing'


# ---------------------------------------------------------------------------
# WebSocket wrappers (shared by both providers)
# ---------------------------------------------------------------------------


class ReplayWebSocket:
    """Replays pre-recorded 'received' messages from a cassette."""

    def __init__(self, cassette: RealtimeCassette) -> None:
        self._received = [i.data for i in cassette.interactions if i.direction == 'received']
        self._pos = 0

    async def send(self, message: str) -> None:
        pass  # no-op during replay

    async def recv(self, *, decode: bool | None = None) -> str | bytes:
        if self._pos >= len(self._received):
            raise ConnectionClosedOK(None, None)
        data = self._received[self._pos]
        self._pos += 1
        text = json.dumps(data)
        if decode is False:
            return text.encode('utf-8')
        return text

    def __aiter__(self) -> ReplayWebSocket:
        return self

    async def __anext__(self) -> str:
        try:
            result = await self.recv()
        except ConnectionClosedOK:
            raise StopAsyncIteration
        if isinstance(result, bytes):
            return result.decode('utf-8')
        return result

    async def close(self) -> None:
        pass


class RecordingWebSocket:
    """Wraps a real WebSocket connection, recording all frames to a cassette."""

    def __init__(self, ws: Any, cassette: RealtimeCassette) -> None:
        self._ws = ws
        self._cassette = cassette

    async def send(self, message: str) -> None:
        self._cassette.interactions.append(CassetteInteraction(direction='sent', data=json.loads(message)))
        await self._ws.send(message)

    async def recv(self, **kwargs: Any) -> str | bytes:
        raw = await self._ws.recv(**kwargs)
        text = raw.decode('utf-8') if isinstance(raw, bytes) else raw
        self._cassette.interactions.append(CassetteInteraction(direction='received', data=json.loads(text)))
        return raw

    def __aiter__(self) -> RecordingWebSocket:
        return self

    async def __anext__(self) -> str | bytes:
        try:
            return await self.recv()
        except (ConnectionError, StopAsyncIteration):
            raise StopAsyncIteration

    async def close(self) -> None:
        await self._ws.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ws, name)
