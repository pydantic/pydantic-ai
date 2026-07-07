"""WebSocket cassette utilities for Responses API WebSocket tests.

VCR.py can't record WebSocket traffic. We intercept at the `websockets` level
for a unified cassette format that records sent/received JSON frames.
"""

from __future__ import annotations as _annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..conftest import try_import

with try_import() as imports_successful:
    import yaml
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
    from websockets.frames import Close

CassetteDirection = Literal['sent', 'received', 'closed']


@dataclass
class CassetteInteraction:
    """A single WebSocket frame or close event."""

    direction: CassetteDirection
    data: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class WebSocketCassette:
    """Ordered list of WebSocket interactions."""

    version: int = 1
    synthetic: bool = False
    interactions: list[CassetteInteraction] = field(default_factory=list[CassetteInteraction])

    @classmethod
    def load(cls, path: Path) -> WebSocketCassette:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding='utf-8'))
        raw_interactions: list[dict[str, Any]] = raw.get('interactions', [])
        interactions: list[CassetteInteraction] = [
            CassetteInteraction(direction=item['direction'], data=item.get('data', {})) for item in raw_interactions
        ]
        # Synthetic cassettes exercise protocol paths that cannot be recorded on demand.
        return cls(version=raw.get('version', 1), synthetic=raw.get('synthetic', False), interactions=interactions)

    def dump(self, path: Path) -> None:  # pragma: no cover
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            'version': self.version,
            'interactions': [{'direction': i.direction, 'data': i.data} for i in self.interactions],
        }
        if self.synthetic:
            data['synthetic'] = True
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding='utf-8',
        )


def ws_cassette_plan(
    *,
    cassette_exists: bool,
    record_mode: str | None,
) -> Literal['replay', 'record', 'error_missing']:
    """Decide replay vs record behavior, mirroring VCR cassette plan logic."""
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


class ReplayWebSocket:
    """Replays pre-recorded 'received' messages from a cassette."""

    def __init__(self, cassette: WebSocketCassette) -> None:
        self._sent = [i.data for i in cassette.interactions if i.direction == 'sent']
        self._received = [i for i in cassette.interactions if i.direction in ('received', 'closed')]
        self._sent_pos = 0
        self._received_pos = 0

    async def send(self, message: str | bytes) -> None:
        actual = json.loads(message.decode('utf-8') if isinstance(message, bytes) else message)
        if self._sent_pos >= len(self._sent):
            raise AssertionError(f'Unexpected WebSocket send:\n{json.dumps(actual, indent=2, sort_keys=True)}')

        expected = self._sent[self._sent_pos]
        self._sent_pos += 1
        if actual != expected:
            raise AssertionError(
                'WebSocket sent frame does not match cassette.\n'
                f'Expected:\n{json.dumps(expected, indent=2, sort_keys=True)}\n'
                f'Actual:\n{json.dumps(actual, indent=2, sort_keys=True)}'
            )

    @property
    def sent_frames_consumed(self) -> bool:
        return self._sent_pos == len(self._sent)

    async def recv(self, *, decode: bool | None = False) -> bytes:
        if self._received_pos >= len(self._received):
            raise ConnectionClosedOK(None, None)
        interaction = self._received[self._received_pos]
        self._received_pos += 1
        if interaction.direction == 'closed':
            code = interaction.data.get('code', 1011)
            reason = interaction.data.get('reason', '')
            raise ConnectionClosedError(
                Close(code if isinstance(code, int) else 1011, reason if isinstance(reason, str) else ''),
                None,
            )
        return json.dumps(interaction.data).encode('utf-8')

    async def close(self, *, code: int = 1000, reason: str = '') -> None:
        pass


class ReplayConnect:
    """Mimics `websockets.connect` as an awaitable and async context manager."""

    def __init__(self, ws: ReplayWebSocket):
        self._ws = ws

    def __await__(self) -> Any:
        async def _resolve() -> ReplayWebSocket:
            return self._ws

        return _resolve().__await__()

    async def __aenter__(self) -> ReplayWebSocket:
        return self._ws  # pragma: no cover

    async def __aexit__(self, *args: Any) -> None:
        pass


class RecordingWebSocket:  # pragma: no cover
    """Wraps a real WebSocket connection, recording all frames to a cassette."""

    def __init__(self, ws: Any, cassette: WebSocketCassette) -> None:
        self._ws = ws
        self._cassette = cassette

    async def send(self, message: str | bytes) -> None:
        text = message.decode('utf-8') if isinstance(message, bytes) else message
        self._cassette.interactions.append(CassetteInteraction(direction='sent', data=json.loads(text)))
        await self._ws.send(message)

    async def recv(self, **kwargs: Any) -> str | bytes:
        raw = await self._ws.recv(**kwargs)
        text = raw.decode('utf-8') if isinstance(raw, bytes) else raw
        self._cassette.interactions.append(CassetteInteraction(direction='received', data=json.loads(text)))
        return raw

    async def close(self, *, code: int = 1000, reason: str = '') -> None:
        await self._ws.close(code=code, reason=reason)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ws, name)
