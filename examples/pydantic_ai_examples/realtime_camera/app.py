"""Realtime camera + voice assistant — talk to a Gemini Live model and show it your camera.

The browser streams microphone audio (PCM16, 16kHz) and ~1 fps JPEG camera frames into a Gemini
Live session and plays the model's audio back. Pure conversation — no tools, no supervisor — just
"talk and show": point your camera at something and ask about it.

Requires `GOOGLE_API_KEY` (Gemini Live access) — or, where org policy disallows API keys, set
`GOOGLE_GENAI_USE_VERTEXAI=true` (+ `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION` and
`gcloud auth application-default login`) to use Vertex AI instead. Put the config in a `.env` at the
repo root, then:

    uv run --all-packages uvicorn pydantic_ai_examples.realtime_camera.app:app

Open http://localhost:8000 on the same machine (localhost is a secure context, so camera/mic work).
To use it from your phone you need HTTPS — expose the local server with a Cloudflare quick tunnel
(no account needed):

    cloudflared tunnel --url http://localhost:8000

then open the printed `https://<...>.trycloudflare.com` URL on the phone and allow camera + mic.

`CAMERA_REALTIME_MODEL` (default `gemini-live-2.5-flash`, or `gemini-live-2.5-flash-native-audio` on
Vertex) and `CAMERA_REALTIME_VOICE` (default `Puck`) set the fallback defaults; the UI's settings panel
(model, voice, language, turn coverage, VAD sensitivity, proactive/affective audio) overrides them.

The camera assistant keeps every video frame in context (`turn_coverage='all_input'`) so it has the
live scene to reason about. The browser's **Watch** toggle drives proactive narration: while on, it
periodically nudges the model to report what changed. Set `CAMERA_PROACTIVE=true` (native-audio models
only) so the model stays silent when nothing changed instead of replying to every nudge;
`CAMERA_AFFECTIVE=true` enables emotion-aware delivery.

`CAMERA_TURN_COVERAGE` defaults to `all_input` (works on both the Gemini Developer API and Vertex AI).
The newer `all_video` value keeps *all* video but only audio during speech — but it isn't accepted on
Vertex's `v1beta1` API yet, so it's not the default.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import AsyncGenerator, Mapping
from pathlib import Path
from typing import Literal, cast

import anyio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from pydantic_ai import Agent
from pydantic_ai.realtime import (
    AudioDelta,
    InputTranscript,
    RealtimeSession,
    RealtimeSessionEvent,
    SpeechStarted,
    Transcript,
    TurnComplete,
)
from pydantic_ai.realtime.google import AutomaticVAD, GoogleRealtimeModel

load_dotenv()

# Use Vertex AI (ADC) instead of a Gemini API key when `GOOGLE_GENAI_USE_VERTEXAI` is truthy — handy
# where org policy disallows API keys. Needs `gcloud auth application-default login` + project/location.
USE_VERTEX = os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').lower() in (
    '1',
    'true',
    'yes',
)
# On Vertex the only widely-available Live model is the native-audio one, so default to it there;
# the Developer API exposes `gemini-live-2.5-flash`. (Override with `CAMERA_REALTIME_MODEL`.)
MODEL = os.environ.get('CAMERA_REALTIME_MODEL') or (
    'gemini-live-2.5-flash-native-audio' if USE_VERTEX else 'gemini-live-2.5-flash'
)
VOICE = os.environ.get('CAMERA_REALTIME_VOICE', 'Puck')
GCP_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
GCP_LOCATION = os.environ.get('GOOGLE_CLOUD_LOCATION')
# `all_input` keeps every camera frame in the model's context and works on both the Developer API and
# Vertex; `all_video` is newer and not yet accepted on Vertex. Proactive/affective audio are
# native-audio-only knobs, off by default so the standard model still connects.
TURN_COVERAGE = cast(
    "Literal['activity_only', 'all_input', 'all_video']",
    os.environ.get('CAMERA_TURN_COVERAGE', 'all_input'),
)
PROACTIVE = os.environ.get('CAMERA_PROACTIVE', '').lower() in ('1', 'true', 'yes')
AFFECTIVE = os.environ.get('CAMERA_AFFECTIVE', '').lower() in ('1', 'true', 'yes')
WATCH_PROMPT = os.environ.get(
    'CAMERA_WATCH_PROMPT',
    "Look at the current camera view. In a few words, say what's changed since you last spoke; "
    'if nothing notable changed, stay silent.',
)
_INDEX_PATH = Path(__file__).parent / 'index.html'

INSTRUCTIONS = (
    'You are a friendly, concise voice assistant. The user is talking to you and may show you things '
    'through their camera — when relevant, describe and reason about what you can see. Keep replies '
    'short and natural, like a conversation.'
)

agent = Agent(instructions=INSTRUCTIONS)
app = FastAPI()


def _truthy(value: str | None) -> bool:
    return (value or '').lower() in ('1', 'true', 'yes', 'on')


@app.get('/')
async def index() -> HTMLResponse:
    # Seed the settings panel with the server's env-configured defaults so the UI mirrors them.
    defaults = json.dumps(
        {
            'model': MODEL,
            'voice': VOICE,
            'turn_coverage': TURN_COVERAGE,
            'proactive': PROACTIVE,
            'affective': AFFECTIVE,
        }
    )
    return HTMLResponse(_INDEX_PATH.read_text().replace('__DEFAULTS__', defaults))


def _build_model(params: Mapping[str, str]) -> GoogleRealtimeModel:
    """Build the Gemini model from the UI's settings, falling back to the env-configured defaults."""
    start, end = params.get('start_sensitivity'), params.get('end_sensitivity')
    vad = None
    if start in ('high', 'low') or end in ('high', 'low'):
        vad = AutomaticVAD(
            start_sensitivity=start if start in ('high', 'low') else None,
            end_sensitivity=end if end in ('high', 'low') else None,
        )
    coverage = params.get('turn_coverage') or TURN_COVERAGE
    return GoogleRealtimeModel(
        params.get('model') or MODEL,
        voice=params.get('voice') or VOICE,
        language_code=params.get('language') or None,
        response_modality=cast(
            "Literal['audio', 'text']", params.get('modality', 'audio')
        ),
        proactive_audio=_truthy(params['proactive'])
        if 'proactive' in params
        else PROACTIVE,
        affective_dialog=_truthy(params['affective'])
        if 'affective' in params
        else AFFECTIVE,
        turn_coverage=coverage
        if coverage in ('activity_only', 'all_input', 'all_video')
        else None,
        vad=vad,
        vertexai=USE_VERTEX,
        project=GCP_PROJECT,
        location=GCP_LOCATION,
    )


def _json_message(event: RealtimeSessionEvent) -> dict[str, object] | None:
    """Translate a session event into a JSON message for the browser (audio is sent separately)."""
    if isinstance(event, SpeechStarted):
        return {'type': 'speech_started'}
    if isinstance(event, InputTranscript) and event.is_final and event.text:
        return {'type': 'user', 'text': event.text}
    if isinstance(event, Transcript) and event.is_final and event.text:
        return {'type': 'assistant', 'text': event.text}
    if isinstance(event, TurnComplete):
        return {'type': 'turn_complete'}
    return None


async def _dispatch_text(session: RealtimeSession, text: str) -> None:
    """Route a JSON text frame: a camera frame (`image`), a typed turn (`text`), or a watch `nudge`."""
    try:
        data = json.loads(text)
        if data.get('type') == 'image':
            await session.send_image(
                base64.b64decode(data['data']),
                mime_type=data.get('mime') or 'image/jpeg',
            )
        elif data.get('type') == 'text':
            await session.send_text(data['text'])
        elif data.get('type') == 'nudge':
            # Watch mode: trigger a turn so the model reports visual changes.
            await session.send_text(WATCH_PROMPT)
    except (ValueError, AttributeError, KeyError, TypeError):
        return


@app.websocket('/ws')
async def ws(socket: WebSocket) -> None:
    await socket.accept()

    model = _build_model(socket.query_params)
    async with agent.realtime_session(model=model) as session:
        async with anyio.create_task_group() as tg:

            async def pump_events() -> None:
                events = cast(
                    'AsyncGenerator[RealtimeSessionEvent, None]', session.__aiter__()
                )
                try:
                    async for event in events:
                        if isinstance(event, AudioDelta):
                            await socket.send_bytes(event.data)
                            continue
                        if (message := _json_message(event)) is not None:
                            await socket.send_json(message)
                except Exception:
                    tg.cancel_scope.cancel()
                finally:
                    with anyio.CancelScope(shield=True):
                        await events.aclose()

            async def pump_inbound() -> None:
                try:
                    while True:
                        message = await socket.receive()
                        if message.get('type') == 'websocket.disconnect':
                            break
                        if (chunk := message.get('bytes')) is not None:
                            await session.send_audio(chunk)  # raw PCM16 mic audio
                        elif (text := message.get('text')) is not None:
                            await _dispatch_text(session, text)
                    tg.cancel_scope.cancel()
                except WebSocketDisconnect:
                    tg.cancel_scope.cancel()

            tg.start_soon(pump_events)
            tg.start_soon(pump_inbound)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)
