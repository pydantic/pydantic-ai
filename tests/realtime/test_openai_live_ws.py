"""Cassette-backed GPT-Live WebSocket tests.

Live's wire shape differs from the Realtime API's in ways that only a real conversation exercises:
work is delegated to a Responses backend rather than tool-called directly, neither the user's turn
nor the model's reply has a terminal frame, and output audio is a continuous track rather than a
per-response stream. These record the real frames so the default suite runs offline.
"""

from __future__ import annotations as _annotations

import asyncio
import fractions
import importlib
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest
from genai_prices import calc_price
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryImage,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    SpeechPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.providers import Provider
from pydantic_ai.realtime import RealtimeTurnCompleteEvent

from ..conftest import try_import
from .conftest import REAL_SDP_OFFER
from .ws_cassettes import RealtimeCassette

with try_import() as imports_successful:
    from pydantic_ai.realtime.openai_live import OpenAILiveModel, OpenAILiveModelSettings

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='realtime provider dependencies not installed'),
]

# Live has no end-of-turn frame, so the adapter waits out a quiet stretch. Replay delivers the
# recorded frames back to back, so only the wait after the last one costs real time; keeping the
# default would add two seconds to every test here.
_FAST_TURN = OpenAILiveModelSettings(openai_live_turn_silence_ms=1000)

# The agent's own model is what the Live session delegates to, so naming one here pins the backend these
# recordings were made with rather than whatever `'auto'` resolves to today.
_BACKEND = 'openai:gpt-5.6-sol'


# Live's session timeline advances with the audio it receives, and the model can only speak onto that
# timeline. A clip that simply stops leaves it nothing to speak into; a real microphone keeps sending
# silence, so these tests send a fixed amount of it, which also keeps the outbound frame count
# deterministic for cassette replay. It has to cover the delegated answer, which Live speaks several
# seconds after the question.
_TRAILING_SILENCE_FRAMES = 150


async def _stream(session: Any, pcm: bytes, cassette: RealtimeCassette, *, paced: bool) -> None:
    """Feed a clip, then a fixed tail of silence, in the ~100 ms frames a microphone would produce.

    `paced` sends them at a microphone's pace. Sent in one burst, the whole tail lands at once, Live's
    timeline runs ahead of the conversation and then stops, and the model never gets to voice a
    delegated answer — so a recording made that way captures no reply at all. Replay sends as fast as
    it can instead, letting each frame wait for its recorded turn among the session's own sends.
    """
    frames = [pcm[start : start + 4800] for start in range(0, len(pcm), 4800)]
    frames += [b'\x00' * 4800] * _TRAILING_SILENCE_FRAMES
    for frame in frames:
        await cassette.before_audio_send()
        await session.send_audio(frame)
        if paced:  # pragma: no branch
            await anyio.sleep(0.1)  # pragma: no cover  # only while recording


async def test_audio_in_delegated_tool_round(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path, realtime_recording: bool
) -> None:
    """A spoken request is delegated to the Responses backend, which calls the agent's tool.

    This is the whole point of the adapter: the Live model runs the conversation, but the tool
    executes locally through the ordinary `ToolManager`, so the round lands in history in exactly the
    four-message shape a standard run produces.
    """
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    agent = Agent(_BACKEND, instructions='You answer weather questions. Use the `lookup_forecast` tool.')

    @agent.tool_plain
    async def lookup_forecast(city: str) -> str:
        """Look up tomorrow's forecast for a city."""
        return f'{city}: 14 degrees Celsius, light rain.'

    pcm = assets_path.joinpath('weather_question_24khz.pcm').read_bytes()
    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        await _stream(session, pcm, cassette, paced=realtime_recording)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert [
        type(event).__name__ for event in events if isinstance(event, (FunctionToolCallEvent, FunctionToolResultEvent))
    ] == snapshot(['FunctionToolCallEvent', 'FunctionToolResultEvent'])

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(
        ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    # The user's spoken turn is recorded even though Live never says a transcript is finished.
    user_speech = [part for part in messages[0].parts if isinstance(part, SpeechPart)]
    assert len(user_speech) == 1
    assert user_speech[0].speaker == 'user'
    assert 'Amsterdam' in (user_speech[0].transcript or '')
    # The delegated call and its result are ordinary parts, not provider-specific ones.
    assert any(isinstance(part, ToolCallPart) for part in messages[1].parts)
    assert any(isinstance(part, ToolReturnPart) for part in messages[2].parts)
    # Both meters are recorded: the delegated backend's tokens, and Live's own audio seconds, which it
    # reports on a timer (see the caveat on `docs/realtime/openai.md`).
    assert session.usage.input_tokens > 0
    assert session.usage.output_tokens > 0
    assert session.usage.audio_seconds > 0
    # Live reports how full its own context is; the backend's tokens don't measure it.
    assert session.context_window_used == snapshot(0.01021875)
    # Each backend response's tokens land on the `ModelResponse` it produced, as in a standard run: the
    # one that asked for the tool on the tool-call response, the continuation on the spoken answer.
    tool_call_response, spoken_reply = messages[1], messages[3]
    assert isinstance(tool_call_response, ModelResponse) and isinstance(spoken_reply, ModelResponse)
    assert tool_call_response.usage.input_tokens > 0
    assert spoken_reply.usage.input_tokens > 0
    assert tool_call_response.usage.input_tokens + spoken_reply.usage.input_tokens == session.usage.input_tokens
    # Both carry Live's name, so each also records the backend model that spent its tokens, which is
    # what lets the cost be recalculated from `usage` later without charging it at Live's rate.
    assert tool_call_response.model_name == spoken_reply.model_name == snapshot('gpt-live-1')
    assert tool_call_response.provider_details == snapshot(
        {
            'delegated_model': 'gpt-5.6-sol',
            'delegated_response_id': 'resp_0f173858b0a30685006ab4608a48e487d182bd7ed5d3cf7080',
        }
    )
    assert spoken_reply.provider_details == snapshot(
        {
            'delegated_model': 'gpt-5.6-sol',
            'delegated_response_id': 'resp_0f173858b0a30685006ab4608b987087d192b5641ad71a8236',
        }
    )
    for response in (tool_call_response, spoken_reply):
        assert response.provider_details is not None
        repriced = calc_price(response.usage, response.provider_details['delegated_model'], provider_id='openai')
        assert response.usage.cost == repriced.total_price
    # And Live speaks the backend's answer, which is the point of delegating.
    answer = spoken_reply.parts[0]
    assert isinstance(answer, SpeechPart) and answer.speaker == 'assistant'
    assert 'fourteen' in (answer.transcript or '').lower() or '14' in (answer.transcript or '')


async def test_thinking_sets_the_backends_reasoning_effort(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path, realtime_recording: bool
) -> None:
    """The shared `thinking` setting reaches the delegated backend, which is the model that reasons.

    Left at its default effort (`medium`, as the backend echoes it), the backend spent 75 reasoning tokens
    on this request in a control recording made alongside this one; `thinking=False` is sent as `'none'`
    and it spends none.
    `parallel_tool_calls` reaches the backend the same way. The recording pins both in the session
    config, and the backend echoes both on every response.
    """
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel(
        'gpt-live-1',
        provider=provider,
        settings=OpenAILiveModelSettings(thinking=False, parallel_tool_calls=False, openai_live_turn_silence_ms=1000),
    )
    agent = Agent(
        _BACKEND,
        instructions=(
            'You answer weather questions. Use the `lookup_forecast` tool, then say whether the temperature '
            'is above the yearly average for that city, reasoning it out from what you know.'
        ),
    )

    @agent.tool_plain
    async def lookup_forecast(city: str) -> str:
        """Look up tomorrow's forecast for a city."""
        return f'{city}: 14 degrees Celsius, light rain.'

    pcm = assets_path.joinpath('weather_question_24khz.pcm').read_bytes()
    async with agent.realtime(model).session() as session:
        await _stream(session, pcm, cassette, paced=realtime_recording)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    # Both backend responses (the tool call and the answer after it) reported their usage.
    assert session.usage.requests == 2
    assert session.usage.tool_calls == 1
    assert session.usage.details == {'reasoning_tokens': 0}


async def test_text_reaches_the_model_as_context(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """`send(text)` steers a Live session even though Live has no user-message event.

    Text is injected as speakable context, and lands on the session's audio timeline, so it only
    takes effect while audio is flowing — hence the silence on both sides of it. Nobody speaks here:
    the reply is entirely the result of the injected text.
    """
    provider, _ = openai_live_ws_cassette
    # Relaying injected context takes the model a beat longer than answering, and a turn boundary
    # inferred from silence will cut in if it is too eager — the tradeoff the setting exists for.
    model = OpenAILiveModel(
        'gpt-live-1', provider=provider, settings=OpenAILiveModelSettings(openai_live_turn_silence_ms=2500)
    )
    agent = Agent(_BACKEND, instructions='Relay what you are told, in one short sentence.')

    async def silence(frames: int) -> None:
        for _ in range(frames):
            await session.send_audio(b'\x00' * 4800)

    async with agent.realtime(model).session() as session:
        await silence(20)
        await session.send('Tell the user their package arrives on Friday.')
        # Enough for the relayed sentence, which needs no delegated round-trip first.
        await silence(120)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    spoken = ' '.join(
        part.transcript or ''
        for message in session.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, SpeechPart)
    )
    assert 'Friday' in spoken


async def test_history_seeding(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path, realtime_recording: bool
) -> None:
    """Prior text history is seeded into the session and is still present in `all_messages()`."""
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    agent = Agent(_BACKEND, instructions='Answer in a few words.')
    history = [
        ModelRequest(parts=[UserPromptPart(content='My favorite color is orange.')]),
        ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='Good to know!')]),
    ]

    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes()
    async with agent.realtime(model, message_history=history).session() as session:
        await _stream(session, pcm, cassette, paced=realtime_recording)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    messages = session.all_messages()
    assert messages[:2] == history
    # The live exchange follows the seeded one: the user's words, then a spoken reply to them.
    assert [type(message).__name__ for message in messages[2:]] == snapshot(['ModelRequest', 'ModelResponse'])
    user_speech, reply = messages[2].parts[0], messages[3].parts[0]
    assert isinstance(user_speech, SpeechPart) and 'Marcelo' in (user_speech.transcript or '')
    assert isinstance(reply, SpeechPart) and reply.speaker == 'assistant' and reply.transcript


async def test_an_image_is_described_by_the_backend(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette],
    image_content: BinaryImage,
    realtime_recording: bool,
) -> None:
    """Live's voice model sees no images, but the backend it delegates to does, and Live speaks for it.

    With `respond=True` the image goes to the backend as ordinary Responses input and the backend runs
    on it straight away; the microphone keeps streaming meanwhile, since that is what moves Live's
    timeline along.
    """
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    agent = Agent(_BACKEND, instructions='Say what is in images the user shares, in one short sentence.')

    async def silence(frames: int) -> None:
        for _ in range(frames):
            await cassette.before_audio_send()
            await session.send_audio(b'\x00' * 4800)
            if realtime_recording:  # pragma: no branch
                await anyio.sleep(0.1)  # pragma: no cover  # only while recording

    async with agent.realtime(model).session() as session:
        await silence(10)
        await session.send(image_content, respond=True)
        await silence(_TRAILING_SILENCE_FRAMES)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    spoken = ' '.join(
        part.transcript or ''
        for message in session.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, SpeechPart)
    )
    assert 'kiwi' in spoken.lower()


# `(offer_sdp, connect(answer_sdp), speak())`: the browser side of a WebRTC call.
_BrowserPeer = tuple[str, Callable[[str], Awaitable[None]], Callable[[], None]]


@asynccontextmanager
async def _speaking_browser(pcm: bytes) -> AsyncGenerator[_BrowserPeer]:  # pragma: no cover
    """Negotiate a real WebRTC call with `aiortc`, standing in for a browser whose user asks a question.

    Recording only. Live's timeline moves with the audio it receives, and on a WebRTC call that audio
    comes from the browser, so a canned offer that never connects records a session in which nothing
    happens. The track sends silence until `speak()`, then the clip, then silence again, so the question
    is asked only once the sideband is attached to hear the answer. `aiortc` is not a project dependency:
    record with `uv run --with aiortc --env-file .env pytest ... --record-mode=rewrite`.
    """
    aiortc = importlib.import_module('aiortc')
    av = importlib.import_module('av')
    rate, frame_samples = 24000, 480  # 20 ms frames of the 24 kHz clip
    state = {'position': -1}

    class _Microphone(aiortc.MediaStreamTrack):
        kind = 'audio'

        def __init__(self) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType]
            self._start: float | None = None
            self._timestamp = 0

        async def recv(self) -> Any:
            if self._start is None:
                self._start = time.monotonic()
            else:
                await asyncio.sleep(max(0.0, self._start + self._timestamp / rate - time.monotonic()))
            chunk = b''
            if state['position'] >= 0:
                chunk = pcm[state['position'] : state['position'] + frame_samples * 2]
                state['position'] += frame_samples * 2
            frame = av.AudioFrame(format='s16', layout='mono', samples=frame_samples)
            frame.planes[0].update(chunk.ljust(frame_samples * 2, b'\x00'))
            frame.sample_rate = rate
            frame.pts = self._timestamp
            frame.time_base = fractions.Fraction(1, rate)
            self._timestamp += frame_samples
            return frame

    # No STUN server, as in `test_openai_ws.py`: host candidates reach OpenAI's ICE-lite endpoint.
    pc = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
    pc.addTrack(_Microphone())

    @pc.on('track')
    def _drain_inbound_audio(track: Any) -> None:
        async def pump() -> None:
            while True:
                try:
                    await track.recv()
                except Exception:
                    return

        asyncio.ensure_future(pump())

    await pc.setLocalDescription(await pc.createOffer())
    while pc.iceGatheringState != 'complete':
        await anyio.sleep(0.1)

    async def connect(answer_sdp: str) -> None:
        await pc.setRemoteDescription(aiortc.RTCSessionDescription(sdp=answer_sdp, type='answer'))

    def speak() -> None:
        state['position'] = 0

    try:
        yield pc.localDescription.sdp, connect, speak
    finally:
        await pc.close()


async def _no_browser_to_connect(answer_sdp: str) -> None:
    """Replay has no browser: the recorded sideband frames are the whole call."""


@pytest.mark.vcr
async def test_webrtc_sideband_runs_the_delegated_tool_round(
    openai_live_ws_sideband_cassette: tuple[Provider[Any], RealtimeCassette],
    assets_path: Path,
    realtime_recording: bool,
) -> None:
    """A browser WebRTC call, negotiated by the server, with the agent run over a sideband.

    The server starts the session from the browser's offer, with the agent's instructions and tools
    already on the backend; the browser holds the audio; the sideband runs the backend's tool call and
    records the conversation. The offer is an HTTP VCR cassette, the sideband a WebSocket cassette.
    """
    provider, _ = openai_live_ws_sideband_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    agent = Agent(_BACKEND, instructions='You answer weather questions. Use the `lookup_forecast` tool.')

    @agent.tool_plain
    async def lookup_forecast(city: str) -> str:
        """Look up tomorrow's forecast for a city."""
        return f'{city}: 14 degrees Celsius, light rain.'

    realtime = agent.realtime(model)
    pcm = assets_path.joinpath('weather_question_24khz.pcm').read_bytes()

    @asynccontextmanager
    async def browser() -> AsyncGenerator[_BrowserPeer]:
        if realtime_recording:  # pragma: no cover
            async with _speaking_browser(pcm) as peer:
                yield peer
            return
        yield REAL_SDP_OFFER, _no_browser_to_connect, lambda: None

    async with browser() as (offer, connect, speak):
        answer = await realtime.answer_webrtc_offer(offer)
        assert answer.sdp.startswith('v=0')
        assert answer.session.provider_name == 'openai'
        assert answer.session.session_id.startswith('live_')
        await connect(answer.sdp)

        async with realtime.session(provider_session=answer.session) as session:
            with pytest.raises(UserError, match='does not own the audio transport'):
                await session.send_audio(b'\x00\x00')
            speak()
            events: list[Any] = []
            with anyio.fail_after(60):
                async for event in session:  # pragma: no branch
                    events.append(event)
                    if isinstance(event, RealtimeTurnCompleteEvent) and any(
                        isinstance(e, FunctionToolResultEvent) for e in events
                    ):
                        break

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(
        ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    user_speech = [part for part in messages[0].parts if isinstance(part, SpeechPart)]
    assert 'Amsterdam' in (user_speech[0].transcript or '')
    assert any(isinstance(part, ToolCallPart) for part in messages[1].parts)
    assert any(isinstance(part, ToolReturnPart) for part in messages[2].parts)
    answer_part = messages[3].parts[-1]
    assert isinstance(answer_part, SpeechPart)
    assert '14' in (answer_part.transcript or '') or 'fourteen' in (answer_part.transcript or '').lower()
    # The browser plays the audio, so the sideband records the reply without its bytes.
    assert answer_part.audio is None
    assert session.usage.input_tokens > 0
