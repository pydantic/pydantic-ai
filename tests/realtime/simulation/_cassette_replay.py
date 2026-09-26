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

from google.genai import _live_converters as live_converters, types as genai_types
from websockets.exceptions import ConnectionClosedOK

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


def cassette_protocol(path: Path) -> Protocol | None:
    """Which adapter a cassette's provider frames are meant for, or `None` for a non-WebSocket recording."""
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
    return _MODULE_PROTOCOLS.get(module)


def websocket_cassettes() -> list[Path]:
    """Every WebSocket cassette (the same directories also hold HTTP recordings of WebRTC signaling)."""
    return sorted(
        path
        for path in CASSETTES_DIR.glob('*/*.yaml')
        if cassette_protocol(path) is not None and path.read_text(encoding='utf-8').startswith('version:')
    )


def _segments(cassette: RealtimeCassette) -> Iterator[list[dict[str, Any]]]:
    """The provider's frames per recorded socket."""
    frames: list[dict[str, Any]] = []
    for interaction in cassette.interactions:
        if isinstance(interaction, CassetteClose):
            yield frames
            frames = []
        elif isinstance(interaction, CassetteMessage) and interaction.direction == 'received':
            frames.append(interaction.data)
    if frames:
        yield frames


class _InboundOnlySocket:
    """A socket that yields recorded frames (a connection with no session sends nothing)."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self._frames = [json.dumps(frame) for frame in frames]
        self.close_code: int | None = 1000
        self.close_reason = ''

    async def recv(self, decode: bool | None = None) -> str:
        if not self._frames:
            raise ConnectionClosedOK(None, None)
        return self._frames.pop(0)

    async def __aiter__(self) -> AsyncIterator[str]:
        while self._frames:
            yield self._frames.pop(0)


class _InboundOnlyGeminiSession:
    """A `google-genai` session that yields recorded messages, one model turn per `receive()`, like the SDK's."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
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
            raise ConnectionClosedOK(None, None)
        while self._messages:
            message = self._messages.pop(0)
            yield message
            content = message.server_content
            if content is not None and content.turn_complete:
                if content.interaction_status != genai_types.InteractionStatus.IN_PROGRESS:
                    return


def _connection(protocol: Protocol, frames: list[dict[str, Any]]) -> RealtimeConnection:
    socket: Any = _InboundOnlySocket(frames)
    if protocol == 'gemini':
        return GoogleRealtimeConnection(_InboundOnlyGeminiSession(frames))  # pyright: ignore[reportArgumentType]
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
    assert protocol is not None
    events: list[list[RealtimeCodecEvent]] = []
    for frames in _segments(RealtimeCassette.load(path)):
        connection = _connection(protocol, frames)
        events.append([event async for event in connection])
        if isinstance(connection, OpenAILiveConnection):
            await connection.aclose()
    return events
