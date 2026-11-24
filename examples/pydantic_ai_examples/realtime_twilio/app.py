"""Answer phone calls with a realtime voice agent, over Twilio Media Streams.

When a call comes in, Twilio asks `POST /twiml` what to do, and the TwiML it gets back
connects the call's audio to the `/media-stream` WebSocket. That WebSocket is bridged to
a Pydantic AI realtime session: the caller's 8 kHz mu-law audio is converted to the
session's PCM16, and the model's speech back again.

The bridge also keeps track of what the caller has actually heard. Twilio buffers the
audio it's sent and plays it in order, so the session can't know that by itself: the
bridge sends a `mark` after each chunk, which Twilio echoes once that chunk has played.
When the caller talks over the model, the bridge tells Twilio to drop what it hasn't
played yet (`clear`). On a model that can be interrupted, it also tells the session how
much was heard (`interrupt(played_bytes=...)`), so history records the reply as the
caller heard it.

It needs a Twilio phone number, a public HTTPS address for this server (a tunnel such as
`cloudflared tunnel --url http://localhost:8000` works), and these environment variables:

- `OPENAI_API_KEY`
- `TWILIO_AUTH_TOKEN`: your Twilio account's auth token, to check Twilio's signatures
- `PUBLIC_URL`: the public HTTPS address of this server, e.g. `https://example.com`

Run with:

    uv run --all-packages uvicorn pydantic_ai_examples.realtime_twilio.app:app

then set the phone number's "A call comes in" webhook to `<PUBLIC_URL>/twiml`.
"""

from __future__ import annotations

import base64
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from html import escape
from typing import Annotated, Literal

import anyio
import logfire
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import Response
from pydantic import Base64Bytes, BaseModel, Field, TypeAdapter
from twilio.request_validator import RequestValidator

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    RealtimeInputSpeechStartEvent,
    RealtimeResponseInterruptedEvent,
)
from pydantic_ai.realtime import RealtimeSession

from .audio import mulaw_to_pcm16, pcm16_to_mulaw

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

MODEL = os.getenv('TWILIO_REALTIME_MODEL', 'openai:gpt-realtime')


@dataclass
class Caller:
    """What the session knows about the call, from the stream's custom parameters."""

    number: str


agent = Agent(
    deps_type=Caller,
    instructions=(
        'You answer the phone for The Terrace, a restaurant. '
        'Keep replies short: the caller is on a phone line.'
    ),
)


@agent.instructions
def caller_number(ctx: RunContext[Caller]) -> str:
    return f'The caller is calling from {ctx.deps.number}.'


@agent.tool_plain
async def opening_hours(day: str) -> str:
    """Look up the restaurant's opening hours on a day of the week."""
    return 'Closed.' if day.lower() == 'monday' else '5 pm to 11 pm.'


# Twilio's Media Streams messages, parsed and base64-decoded by Pydantic.
class _Connected(BaseModel):
    event: Literal['connected']


class _StartDetails(BaseModel):
    call_sid: str = Field(alias='callSid')
    custom_parameters: dict[str, str] = Field(alias='customParameters')


class _Start(BaseModel):
    event: Literal['start']
    stream_sid: str = Field(alias='streamSid')
    start: _StartDetails


class _Payload(BaseModel):
    payload: Base64Bytes


class _Media(BaseModel):
    event: Literal['media']
    media: _Payload


class _MarkName(BaseModel):
    name: str


class _Mark(BaseModel):
    event: Literal['mark']
    mark: _MarkName


class _Digit(BaseModel):
    digit: str


class _Dtmf(BaseModel):
    event: Literal['dtmf']
    dtmf: _Digit


class _Stop(BaseModel):
    event: Literal['stop']


TwilioMessage = Annotated[
    _Connected | _Start | _Media | _Mark | _Dtmf | _Stop, Field(discriminator='event')
]
twilio_message: TypeAdapter[TwilioMessage] = TypeAdapter(TwilioMessage)


@dataclass
class PlaybackMarks:
    """How much of the session's audio Twilio has played to the caller.

    Each chunk sent to Twilio is followed by a `mark`, which Twilio echoes once the
    chunk has played. `played_bytes` counts the session's PCM16 bytes Twilio is done
    with, played or dropped by a `clear`: the position `interrupt(played_bytes=...)` takes.
    """

    played_bytes: int = 0
    _pending: dict[str, int] = field(default_factory=dict[str, int])
    _sent: int = 0

    def sent(self, pcm_bytes: int) -> str:
        """Note a chunk sent to Twilio, returning the name of the mark to send after it."""
        self._sent += 1
        name = str(self._sent)
        self._pending[name] = pcm_bytes
        return name

    def played(self, name: str) -> None:
        """Twilio echoed a mark: the chunk before it has played."""
        # A mark dropped by `clear()` was never played, though Twilio echoes it.
        self.played_bytes += self._pending.pop(name, 0)

    def clear(self) -> int:
        """Note a `clear` dropping the chunks Twilio hasn't played, returning the position the caller heard to.

        The echoes `clear` sends for their marks count for nothing. Their bytes still count
        toward later positions: the session handed them over, and has no other way to learn
        the caller skipped them.
        """
        heard = self.played_bytes
        self.played_bytes += sum(self._pending.values())
        self._pending.clear()
        return heard


app = FastAPI()
public_url = os.environ.get('PUBLIC_URL', 'http://localhost:8000')
# The calls `/twiml` answered, whose stream `/media-stream` is waiting for. A production
# server would also forget a call whose stream never arrives.
answered_calls: set[str] = set()


@app.post('/twiml')
async def twiml(request: Request) -> Response:
    # Anyone can sign with an empty key, so without the token every request would pass as Twilio's.
    if not (auth_token := os.environ.get('TWILIO_AUTH_TOKEN')):
        raise RuntimeError(
            'Set `TWILIO_AUTH_TOKEN` to check that requests come from Twilio.'
        )
    form = await request.form()
    params = {key: str(value) for key, value in form.items()}
    # Twilio signs the public URL it called, which a tunnel or proxy hides from the request.
    signature = request.headers.get('X-Twilio-Signature', '')
    validator = RequestValidator(auth_token)
    # The SDK is untyped.
    if not validator.validate(f'{public_url}/twiml', params, signature):  # pyright: ignore[reportUnknownMemberType]
        raise HTTPException(status_code=403)
    answered_calls.add(params['CallSid'])
    stream_url = public_url.replace('https://', 'wss://', 1) + '/media-stream'
    # The stream URL can't carry a query string: per-call context goes in `<Parameter>`s,
    # which arrive in the stream's `start` message.
    caller = escape(params.get('From', 'unknown'))
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
        f'<Stream url="{stream_url}"><Parameter name="caller" value="{caller}"/></Stream>'
        '</Connect></Response>'
    )
    return Response(twiml, media_type='application/xml')


async def twilio_messages(websocket: WebSocket) -> AsyncIterator[TwilioMessage]:
    while True:
        yield twilio_message.validate_json(await websocket.receive_text())


@app.websocket('/media-stream')
async def media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    messages = twilio_messages(websocket)
    start = None
    while start is None:
        message = await anext(messages)
        start = message if isinstance(message, _Start) else None
    if start.start.call_sid not in answered_calls:
        # Only a call the validated webhook answered gets a session.
        await websocket.close(code=1008)
        return
    answered_calls.discard(start.start.call_sid)
    caller = Caller(number=start.start.custom_parameters.get('caller', 'unknown'))

    async with agent.realtime(MODEL, deps=caller).session() as session:
        await bridge(session, websocket, start.stream_sid, messages)
    # Closing the socket ends the call: Twilio moves on to the TwiML after `<Connect>`.
    await websocket.close()


async def bridge(
    session: RealtimeSession,
    websocket: WebSocket,
    stream_sid: str,
    messages: AsyncIterator[TwilioMessage],
) -> None:
    """Carry the call's audio to and from the session until Twilio stops the stream."""
    marks = PlaybackMarks()

    async def caller_audio() -> AsyncIterator[bytes]:
        async for message in messages:
            if isinstance(message, _Media):
                yield mulaw_to_pcm16(
                    message.media.payload, rate=session.audio_input_sample_rate
                )
            elif isinstance(message, _Mark):
                marks.played(message.mark.name)
            elif isinstance(message, _Dtmf):
                # Sessions take no keypad input, so a key press reaches the model as text.
                await session.send(
                    f'The caller pressed {message.dtmf.digit} on their keypad.'
                )
            elif isinstance(message, _Stop):
                # The call ended: returning ends `send_audio()`, and the session with it.
                return

    async def playback() -> None:
        async for chunk in session.stream_audio():
            payload = pcm16_to_mulaw(chunk, rate=session.audio_output_sample_rate)
            # Named before the media goes out: a `clear` sent meanwhile drops the
            # chunk, and must drop its mark with it.
            mark = marks.sent(len(chunk))
            await websocket.send_json(
                {
                    'event': 'media',
                    'streamSid': stream_sid,
                    'media': {'payload': base64.b64encode(payload).decode()},
                }
            )
            await websocket.send_json(
                {'event': 'mark', 'streamSid': stream_sid, 'mark': {'name': mark}}
            )

    async def barge_in() -> None:
        async for event in session:
            # Models that report the caller starting to talk (OpenAI gpt-realtime, Azure,
            # xAI) can be interrupted; Gemini and ElevenLabs Agents interrupt themselves and
            # say so. Either way, Twilio still holds audio the caller shouldn't hear.
            # OpenAI GPT-Live reports neither: it stops talking by itself, and sends its
            # audio at the pace it speaks it, so little of it waits in Twilio's buffer.
            if isinstance(
                event, RealtimeInputSpeechStartEvent | RealtimeResponseInterruptedEvent
            ):
                # `clear` makes Twilio echo every mark still pending, for audio the
                # caller never heard: drop them first, so they don't count as played.
                played = marks.clear()
                await websocket.send_json({'event': 'clear', 'streamSid': stream_sid})
                if isinstance(
                    event, RealtimeInputSpeechStartEvent
                ) and session.profile.get('supports_interruption'):
                    await session.interrupt(played_bytes=played)

    async with anyio.create_task_group() as tg:
        tg.start_soon(playback)
        tg.start_soon(barge_in)
        # Returns when Twilio stops the stream, and raises when the socket drops.
        await session.send_audio(caller_audio())
        tg.cancel_scope.cancel()
