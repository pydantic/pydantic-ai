"""Replay a recorded WebSocket cassette's provider frames through the real connection class, alone.

The conformance check needs only the codec events an adapter makes of a real provider trace, not the
conversation that produced it, so the recorded client frames are ignored and there is no session: the
provider's frames are fed, in order, to a fresh connection, one connection per recorded socket (a
cassette of a reconnect holds several), and whatever it yields is collected.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, Literal

import yaml
from google.genai import _live_converters as live_converters, types as genai_types
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from pydantic_ai.realtime.azure import (
    AzureRealtimeConnection,
    _VoiceLiveRealtimeConnection,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.realtime.codec import RealtimeCodecEvent, RealtimeConnection
from pydantic_ai.realtime.google import GoogleRealtimeConnection
from pydantic_ai.realtime.openai import OpenAIRealtimeConnection
from pydantic_ai.realtime.openai_live import OpenAILiveConnection
from pydantic_ai.realtime.xai import XaiRealtimeConnection

from ..ws_cassettes import CassetteClose, CassetteMessage, RealtimeCassette

Protocol = Literal['openai', 'azure', 'azure-voice-live', 'xai', 'gemini', 'openai-live']

CASSETTES_DIR = Path(__file__).parent.parent / 'cassettes'

_MODULE_PROTOCOLS: dict[str, Protocol] = {
    'test_openai_ws': 'openai',
    'test_openai_ws_sideband': 'openai',
    'test_azure_ws': 'azure',
    'test_azure_ws_sideband': 'azure',
    'test_azure_voice_live_ws': 'azure-voice-live',
    'test_xai_ws': 'xai',
    'test_google_ws': 'gemini',
    'test_openai_live_ws': 'openai-live',
    'test_openai_live_ws_sideband': 'openai-live',
}
_PARITY_PROTOCOLS: dict[str, Protocol] = {
    'openai': 'openai',
    'gateway-openai': 'openai',
    'azure': 'azure',
    'xai': 'xai',
    'google': 'gemini',
    'gateway-google': 'gemini',
    'openai-live': 'openai-live',
}


def cassette_protocol(path: Path) -> Protocol:
    """Which adapter a WebSocket cassette's provider frames are meant for.

    Raises `KeyError` for a cassette directory with no mapping, so a new one fails its conformance case
    instead of silently dropping out.
    """
    module = path.parent.name
    if module == 'test_gateway_ws':
        return 'gemini' if 'gemini' in path.stem else 'openai'
    if module == 'test_parity_ws':
        variant = path.stem.split('[', 1)[1].rstrip(']')
        return next(
            protocol
            for prefix, protocol in sorted(_PARITY_PROTOCOLS.items(), key=lambda item: -len(item[0]))
            if variant.startswith(prefix)
        )
    return _MODULE_PROTOCOLS[module]


def websocket_cassettes() -> list[Path]:
    """Every WebSocket cassette (the same directories also hold HTTP recordings of WebRTC signaling)."""
    return sorted(path for path in CASSETTES_DIR.glob('*/*.yaml') if _is_websocket(path))


def _is_websocket(path: Path) -> bool:
    # Both formats can open with `version:`, so tell them apart by their interactions: HTTP recordings
    # (e.g. WebRTC signaling) hold `request`/`response` pairs, WebSocket ones hold frames and closes.
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding='utf-8'))
    interactions: list[dict[str, Any]] = raw.get('interactions') or [{}]
    return 'request' not in interactions[0]


def _segments(cassette: RealtimeCassette) -> Iterator[tuple[list[dict[str, Any]], CassetteClose | None]]:
    """The provider's frames per recorded socket, and how the socket closed (if the recording says)."""
    frames: list[dict[str, Any]] = []
    for interaction in cassette.interactions:
        if isinstance(interaction, CassetteClose):
            yield frames, interaction
            frames = []
        elif isinstance(interaction, CassetteMessage) and interaction.direction == 'received':
            frames.append(interaction.data)
    if frames:
        yield frames, None


def _closed(close: CassetteClose | None) -> Exception:
    """What reading past the last frame raises: the recorded close, abnormal ones included."""
    if close is None or close.ok:
        return ConnectionClosedOK(None, None)
    return ConnectionClosedError(Close(close.code, close.reason), None)


class _InboundOnlySocket:
    """A socket that yields recorded frames, then closes as recorded (a connection with no session sends nothing)."""

    def __init__(self, frames: list[dict[str, Any]], close: CassetteClose | None) -> None:
        self._frames = [json.dumps(frame) for frame in frames]
        self._close = close
        self.close_code: int | None = close.code if close is not None else 1000
        self.close_reason = close.reason if close is not None else ''

    async def recv(self, decode: bool | None = None) -> str:
        if not self._frames:
            raise _closed(self._close)
        return self._frames.pop(0)

    async def __aiter__(self) -> AsyncIterator[str]:
        while self._frames:
            yield self._frames.pop(0)
        if self._close is not None and not self._close.ok:
            raise _closed(self._close)


class _InboundOnlyGeminiSession:
    """A `google-genai` session that yields recorded messages, one model turn per `receive()`, like the SDK's."""

    def __init__(self, frames: list[dict[str, Any]], close: CassetteClose | None) -> None:
        self._close = close
        # Parsed exactly as the SDK's own `AsyncSession.receive()` parses the wire.
        self._messages = [
            genai_types.LiveServerMessage._from_response(  # pyright: ignore[reportPrivateUsage]
                response=live_converters._LiveServerMessage_from_mldev(frame),  # pyright: ignore[reportPrivateUsage]
                kwargs={},
            )
            for frame in frames
        ]

    async def receive(self) -> AsyncIterator[genai_types.LiveServerMessage]:
        if not self._messages:
            raise _closed(self._close)
        while self._messages:
            message = self._messages.pop(0)
            yield message
            content = message.server_content
            if content is not None and content.turn_complete:
                if content.interaction_status != genai_types.InteractionStatus.IN_PROGRESS:
                    return


def _connection(protocol: Protocol, frames: list[dict[str, Any]], close: CassetteClose | None) -> RealtimeConnection:
    socket: Any = _InboundOnlySocket(frames, close)
    if protocol == 'gemini':
        return GoogleRealtimeConnection(_InboundOnlyGeminiSession(frames, close))  # pyright: ignore[reportArgumentType]
    if protocol == 'openai-live':
        return OpenAILiveConnection(socket)
    if protocol == 'xai':
        return XaiRealtimeConnection(socket)
    if protocol == 'azure':
        return AzureRealtimeConnection(socket)
    if protocol == 'azure-voice-live':
        return _VoiceLiveRealtimeConnection(socket)
    return OpenAIRealtimeConnection(socket)


async def replay_codec_events(path: Path) -> list[list[RealtimeCodecEvent]]:
    """The codec events each recorded socket's provider frames make, per socket."""
    protocol = cassette_protocol(path)
    events: list[list[RealtimeCodecEvent]] = []
    for frames, close in _segments(RealtimeCassette.load(path)):
        connection = _connection(protocol, frames, close)
        events.append([event async for event in connection])
        if isinstance(connection, OpenAILiveConnection):
            await connection.aclose()
    return events
