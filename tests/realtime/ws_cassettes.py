"""WebSocket cassette utilities for realtime provider tests.

Realtime providers talk over a persistent WebSocket rather than the request/response HTTP that
`pytest-recording` / VCR captures, so VCR can't record their traffic. These helpers record and
replay the actual JSON frames exchanged with the provider, letting cassette-backed tests exercise
the *real* protocol offline:

- OpenAI Realtime connects with the `websockets` library directly (patched at
  `pydantic_ai.realtime.openai.websockets.connect`).
- Gemini Live connects through the `google-genai` SDK, which itself uses `websockets` under
  `google.genai.live.ws_connect` (patched there). The SDK calls `.send`, `.recv(decode=False)`, and
  `.close` on the returned object, so the same raw-frame engine serves both providers.

The replay path validates outbound frames as well as replaying inbound ones, so a cassette pins both
provider behaviour *and* the exact wire messages the library sends. Recording scrubs anything
secret-looking and truncates inbound audio payloads so cassettes stay small.
"""

from __future__ import annotations as _annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
from unittest import mock

from ..conftest import try_import

with try_import() as imports_successful:
    import yaml
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
    from websockets.frames import Close


_MessageKind = Literal['message']
_CloseKind = Literal['close']
_Direction = Literal['sent', 'received']

ProviderName = Literal['openai', 'gemini', 'xai']

# Outbound frame fields that carry random client-generated ids, normalized to stable placeholders so
# replay can validate frame *structure* without depending on a fresh random value each run.
_CLIENT_ID_KEYS = frozenset({'id', 'item_id', 'previous_item_id'})
_CLIENT_ID_RE = re.compile(r'^[0-9a-f]{24}$')

# Inbound audio payloads are truncated to this many decoded bytes at record time. The exact audio
# content isn't asserted (tests use transcripts and `IsBytes()`/length checks), so a short prefix
# keeps cassettes tiny without changing the event shapes the session produces.
_MAX_AUDIO_BYTES = 32

# OpenAI names its output-audio delta event differently on the GA vs beta surfaces.
_OPENAI_AUDIO_DELTA_TYPES = frozenset({'response.output_audio.delta', 'response.audio.delta'})

# Value patterns that must never land in a cassette (API keys / bearer tokens). Belt-and-braces:
# keys travel in connection headers / the URL, not in frames, but a provider could echo one back.
_SECRET_RE = re.compile(r'(sk-[A-Za-z0-9_\-]{8,}|AIza[A-Za-z0-9_\-]{10,}|Bearer\s+\S+)')
_SECRET_PLACEHOLDER = '<scrubbed>'


@dataclass
class CassetteMessage:
    """A single JSON WebSocket frame."""

    direction: _Direction
    data: dict[str, Any]
    kind: _MessageKind = 'message'


@dataclass
class CassetteClose:
    """A terminal WebSocket close observed while receiving."""

    code: int
    reason: str
    ok: bool
    kind: _CloseKind = 'close'


RealtimeCassetteInteraction = CassetteMessage | CassetteClose


@dataclass
class RealtimeCassette:
    """An ordered list of normalized WebSocket interactions."""

    version: int = 1
    interactions: list[RealtimeCassetteInteraction] = field(default_factory=list['RealtimeCassetteInteraction'])

    @classmethod
    def load(cls, path: Path) -> RealtimeCassette:
        raw = cast('dict[str, Any]', yaml.safe_load(path.read_text(encoding='utf-8')))
        interactions: list[RealtimeCassetteInteraction] = []
        for item in cast('list[dict[str, Any]]', raw.get('interactions', [])):
            if item.get('kind') == 'close':
                interactions.append(CassetteClose(code=item['code'], reason=item.get('reason', ''), ok=item['ok']))
            else:
                interactions.append(CassetteMessage(direction=item['direction'], data=item['data']))
        return cls(version=raw.get('version', 1), interactions=interactions)

    def dump(self, path: Path) -> None:  # pragma: no cover - only runs while recording
        path.parent.mkdir(parents=True, exist_ok=True)
        interactions: list[dict[str, Any]] = [
            {'kind': 'message', 'direction': i.direction, 'data': i.data}
            if isinstance(i, CassetteMessage)
            else {'kind': 'close', 'code': i.code, 'reason': i.reason, 'ok': i.ok}
            for i in self.interactions
        ]
        path.write_text(
            yaml.safe_dump(
                {'version': self.version, 'interactions': interactions}, sort_keys=False, allow_unicode=True
            ),
            encoding='utf-8',
        )


CassettePlan = Literal['replay', 'record', 'error_missing']


def realtime_cassette_plan(*, cassette_exists: bool, record_mode: str | None) -> CassettePlan:
    """Decide replay vs. record, mirroring the repo's `pytest-recording` record modes."""
    mode = (record_mode or 'none').strip().lower()
    if mode in {'rewrite', 'all'}:
        return 'record'
    if mode == 'once':
        return 'replay' if cassette_exists else 'record'
    # 'none' (and anything else): replay only.
    return 'replay' if cassette_exists else 'error_missing'


def _scrub_secrets(value: Any) -> Any:
    """Recursively replace any secret-looking string in a frame with a placeholder."""
    if isinstance(value, str):
        return _SECRET_RE.sub(_SECRET_PLACEHOLDER, value)
    if isinstance(value, dict):
        return {key: _scrub_secrets(item) for key, item in cast('dict[str, Any]', value).items()}
    if isinstance(value, list):
        return [_scrub_secrets(item) for item in cast('list[Any]', value)]
    return value


def _truncate_b64_audio(payload: str) -> str:
    """Truncate a base64 audio payload to the first `_MAX_AUDIO_BYTES` decoded bytes."""
    # base64 encodes 3 bytes per 4 chars; keep enough chars for the byte budget, on a 4-char boundary.
    keep = ((_MAX_AUDIO_BYTES + 2) // 3) * 4
    return payload[:keep]


def _truncate_audio(frame: dict[str, Any]) -> dict[str, Any]:
    """Shrink inbound audio payloads in place-ish, returning a frame safe to store in a cassette.

    Handles both the OpenAI shape (`{'type': 'response.output_audio.delta', 'delta': <b64>}`) and the
    Gemini shape (`serverContent.modelTurn.parts[].inlineData.data`). Transcript deltas (also keyed
    `delta` on OpenAI, but on non-audio event types) are left untouched.
    """
    if frame.get('type') in _OPENAI_AUDIO_DELTA_TYPES and isinstance(frame.get('delta'), str):
        return {**frame, 'delta': _truncate_b64_audio(frame['delta'])}

    def _walk(value: Any) -> Any:
        if isinstance(value, dict):
            node = cast('dict[str, Any]', value)
            inline = node.get('inlineData')
            if isinstance(inline, dict):
                inline = cast('dict[str, Any]', inline)
                data = inline.get('data')
                if isinstance(data, str):
                    return {**node, 'inlineData': {**inline, 'data': _truncate_b64_audio(data)}}
            return {key: _walk(item) for key, item in node.items()}
        if isinstance(value, list):
            return [_walk(item) for item in cast('list[Any]', value)]
        return value

    return cast('dict[str, Any]', _walk(frame))


class _SentFrameNormalizer:
    """Map random client-generated ids in outbound frames to stable `<client-id-N>` placeholders."""

    def __init__(self) -> None:
        self._ids: dict[str, str] = {}

    def normalize(self, value: Any) -> Any:
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in cast('dict[str, Any]', value).items():
                if key in _CLIENT_ID_KEYS and isinstance(item, str) and _CLIENT_ID_RE.fullmatch(item):
                    result[key] = self._ids.setdefault(item, f'<client-id-{len(self._ids) + 1}>')
                else:
                    result[key] = self.normalize(item)
            return result
        if isinstance(value, list):
            return [self.normalize(item) for item in cast('list[Any]', value)]
        return value


class ReplayWebSocket:
    """Replay a recorded WebSocket conversation, validating outbound frames as they are sent.

    Send/receive interleaving is preserved: the realtime session runs a background reader task, so
    `recv()` must block while the next recorded interaction is an outbound send rather than eagerly
    consuming a future inbound frame.
    """

    def __init__(self, cassette: RealtimeCassette) -> None:
        self._interactions = cassette.interactions
        self._position = 0
        self._normalizer = _SentFrameNormalizer()
        self._condition = asyncio.Condition()

    async def send(self, message: str | bytes) -> None:
        text = message.decode('utf-8') if isinstance(message, bytes) else message
        actual = self._normalizer.normalize(_scrub_secrets(json.loads(text)))
        async with self._condition:
            interaction = self._peek()
            if not isinstance(interaction, CassetteMessage) or interaction.direction != 'sent':
                raise AssertionError(
                    f'Outbound WebSocket frame had no matching recorded send (position {self._position}).\n'
                    f'sent={actual!r}'
                )
            self._position += 1
            self._condition.notify_all()
        assert actual == interaction.data, (
            f'Outbound WebSocket frame did not match cassette at position {self._position - 1}.\n'
            f'expected={interaction.data!r}\nactual={actual!r}'
        )

    async def recv(self, *, decode: bool | None = None) -> str | bytes:
        async with self._condition:
            while True:
                interaction = self._peek()
                if interaction is None:
                    raise ConnectionClosedOK(None, None)
                if isinstance(interaction, CassetteClose):
                    self._position += 1
                    self._condition.notify_all()
                    close = Close(interaction.code, interaction.reason)
                    raise (ConnectionClosedOK if interaction.ok else ConnectionClosedError)(close, None)
                if interaction.direction == 'received':
                    self._position += 1
                    self._condition.notify_all()
                    payload = interaction.data
                    break
                await self._condition.wait()
        text = json.dumps(payload)
        return text.encode('utf-8') if decode is False else text

    async def __aiter__(self):
        while True:
            try:
                yield await self.recv()
            except (ConnectionClosedOK, ConnectionClosedError):
                return

    async def close(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def _peek(self) -> RealtimeCassetteInteraction | None:
        if self._position >= len(self._interactions):
            return None
        return self._interactions[self._position]


class RecordingWebSocket:  # pragma: no cover - only runs while recording
    """Wrap a live WebSocket, recording JSON frames (secrets scrubbed, inbound audio truncated)."""

    def __init__(self, ws: Any, cassette: RealtimeCassette) -> None:
        self._ws = ws
        self._cassette = cassette
        self._normalizer = _SentFrameNormalizer()

    async def send(self, message: str | bytes) -> None:
        text = message.decode('utf-8') if isinstance(message, bytes) else message
        data = self._normalizer.normalize(_scrub_secrets(json.loads(text)))
        self._cassette.interactions.append(CassetteMessage(direction='sent', data=data))
        await self._ws.send(message)

    async def recv(self, **kwargs: Any) -> str | bytes:
        try:
            raw = await self._ws.recv(**kwargs)
        except ConnectionClosedOK as e:
            self._record_close(e, ok=True)
            raise
        except ConnectionClosedError as e:
            self._record_close(e, ok=False)
            raise
        text = raw.decode('utf-8') if isinstance(raw, bytes) else raw
        data = _truncate_audio(_scrub_secrets(json.loads(text)))
        self._cassette.interactions.append(CassetteMessage(direction='received', data=data))
        return raw

    def __aiter__(self) -> RecordingWebSocket:
        return self

    async def __anext__(self) -> str | bytes:
        try:
            return await self.recv()
        except (ConnectionClosedOK, ConnectionClosedError):
            raise StopAsyncIteration

    async def close(self, *args: Any, **kwargs: Any) -> None:
        await self._ws.close(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ws, name)

    def _record_close(self, exc: ConnectionClosedOK | ConnectionClosedError, *, ok: bool) -> None:
        close = exc.rcvd or exc.sent
        self._cassette.interactions.append(
            CassetteClose(
                code=close.code if close is not None else 1000,
                reason=close.reason if close is not None else '',
                ok=ok,
            )
        )


def _connect_target(provider: ProviderName) -> tuple[Any, str]:
    """The module and attribute name of the `connect` callable to patch for `provider`."""
    if provider == 'openai':
        from pydantic_ai.realtime import openai as rt_openai

        return rt_openai.websockets, 'connect'
    if provider == 'xai':
        # xAI clones the OpenAI Realtime protocol and connects with the `websockets` library directly,
        # so the same raw-frame engine serves it (patched at its own module reference).
        from pydantic_ai.realtime import xai as rt_xai

        return rt_xai.websockets, 'connect'
    from google.genai import live

    return live, 'ws_connect'


@contextmanager
def patched_ws_connect(provider: ProviderName, cassette: RealtimeCassette, plan: CassettePlan) -> Generator[None]:
    """Patch the provider's WebSocket `connect` to replay from (or record into) `cassette`."""
    target, attr = _connect_target(provider)
    real_connect = getattr(target, attr)

    @asynccontextmanager
    async def connect(*args: Any, **kwargs: Any) -> AsyncGenerator[ReplayWebSocket | RecordingWebSocket]:
        if plan == 'replay':
            yield ReplayWebSocket(cassette)
        else:  # pragma: no cover - only runs while recording
            async with real_connect(*args, **kwargs) as ws:
                yield RecordingWebSocket(ws, cassette)

    with mock.patch.object(target, attr, connect):
        yield


def ws_cassettes_available() -> bool:
    return imports_successful()
