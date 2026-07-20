"""Realtime camera + voice assistant — talk to a Gemini Live model and show it your camera.

The browser streams microphone audio (PCM16, 16kHz) and ~1 fps JPEG camera frames into a Gemini
Live session and plays the model's audio back: point your camera at something and ask about it.
The assistant can also ground answers with web search and redraw a hand-drawn sketch into a clean
diagram (see the tool and capability notes below).

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

`CAMERA_REALTIME_MODEL` (default `gemini-2.5-flash-native-audio-latest`, or
`gemini-live-2.5-flash-native-audio` on Vertex) and `CAMERA_REALTIME_VOICE` (default `Puck`) set the
fallback defaults; the UI's settings panel
(model, voice, language, turn coverage, VAD sensitivity, proactive/affective audio) overrides them.

The camera assistant keeps every video frame in context (`turn_coverage='all_input'`) so it has the
live scene to reason about. The browser's **Watch** toggle drives proactive narration: while on, it
periodically nudges the model to report what changed. Set `CAMERA_PROACTIVE=true` (native-audio models
only) so the model stays silent when nothing changed instead of replying to every nudge;
`CAMERA_AFFECTIVE=true` enables emotion-aware delivery.

`CAMERA_TURN_COVERAGE` defaults to `all_input` (works on both the Gemini Developer API and Vertex AI).
The newer `all_video` value keeps *all* video but only audio during speech — but it isn't accepted on
Vertex's `v1beta1` API yet, so it's not the default.

The app is instrumented with Logfire: set `LOGFIRE_TOKEN` (e.g. in the same `.env`) to see the
realtime session, model turns, and tool calls as traces; without a token nothing is sent.

Web search (the `WebSearch` capability — Grounding with Google Search) is **on by default** so the
assistant can answer with current facts and cite its sources as chips in the UI; set
`CAMERA_WEB_SEARCH=false` to disable (or if your model/region doesn't support grounding).

**Redraw a sketch.** Show the camera a hand-drawn diagram (a system design, flow chart, wireframe)
and ask the assistant to clean it up: it calls the `redraw_diagram` tool, which hands the current
camera frame to a separate vision agent (Gemini by default — same `GOOGLE_API_KEY`) that recreates the
sketch as a clean, self-contained HTML diagram. The browser renders it in an overlay and can export it
to PNG client-side. Set `CAMERA_DRAW=false` to disable, or `CAMERA_DRAW_MODEL` to any `provider:model`
vision model. Because Gemini Live can't combine function calling with Google Search
grounding in one session, enabling the drawing tool turns web search off.
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

import anyio
import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.messages import NativeToolReturnPart
from pydantic_ai.providers.google_cloud import GoogleCloudProvider
from pydantic_ai.realtime import (
    InputSpeechStartEvent,
    PartDeltaEvent,
    PartEndEvent,
    RealtimeEvent,
    RealtimeSession,
    SpeechPart,
    SpeechPartDelta,
    TurnCompleteEvent,
)
from pydantic_ai.realtime.google import (
    AutomaticVAD,
    GoogleRealtimeModel,
    GoogleRealtimeModelSettings,
)

load_dotenv()

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured.
# Configure after `load_dotenv()` so a `LOGFIRE_TOKEN` in `.env` is picked up.
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

# Use Vertex AI (ADC) instead of a Gemini API key when `GOOGLE_GENAI_USE_VERTEXAI` is truthy — handy
# where org policy disallows API keys. Needs `gcloud auth application-default login` + project/location.
USE_VERTEX = os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').lower() in (
    '1',
    'true',
    'yes',
)
# The Developer API's current Live models are the native-audio family; the `-latest` alias tracks the
# newest one. Vertex still uses the older `gemini-live-*` naming. (Override with `CAMERA_REALTIME_MODEL`.)
MODEL = os.environ.get('CAMERA_REALTIME_MODEL') or (
    'gemini-live-2.5-flash-native-audio'
    if USE_VERTEX
    else 'gemini-2.5-flash-native-audio-latest'
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
# Sketch-to-diagram: a `redraw_diagram` tool hands the current camera frame to a separate vision
# agent that recreates the drawing as clean HTML. On by default; the drawing model defaults to
# Gemini (same `GOOGLE_API_KEY` as the live session — no extra key). `CAMERA_DRAW_MODEL` takes any
# `provider:model` string to use a different vision model.
DRAW = os.environ.get('CAMERA_DRAW', 'true').lower() in ('1', 'true', 'yes')
DRAW_MODEL = os.environ.get('CAMERA_DRAW_MODEL', 'google:gemini-3.5-flash')
# Grounding with Google Search (a native tool) — on by default; the native-audio Live models
# support it. Set `CAMERA_WEB_SEARCH=false` to disable, or if your model/region doesn't support it.
# (We don't also add `WebFetch` here: Gemini 2.5 / native-audio can't combine Google Search grounding
# with function calling in one session, so a fetch tool alongside grounding wouldn't be callable.)
# That same limitation means the `redraw_diagram` tool and grounding are mutually exclusive, so the
# drawing tool forces web search off.
WEB_SEARCH = not DRAW and os.environ.get('CAMERA_WEB_SEARCH', 'true').lower() in (
    '1',
    'true',
    'yes',
)
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
    + (
        ' Search the web when a question needs current or external facts.'
        if WEB_SEARCH
        else ''
    )
    + (
        ' You can redraw a hand-drawn sketch the user shows you — a diagram, system design, flow '
        'chart, or wireframe — into a clean version with the `redraw_diagram` tool. Do NOT call it '
        'the moment you see a drawing. First make sure you understand what they actually want: if '
        "they haven't said, ask one short question — keep it faithful but tidier, turn it into a "
        'flowchart, restructure it, add or label something? Only once their intent is clear, pass it '
        "to the tool as the `instructions`. It takes a moment, so say you're on it, then briefly "
        'describe what you drew once it appears.'
        if DRAW
        else ''
    )
)


@dataclass
class CameraDeps:
    """Per-connection hooks the `redraw_diagram` tool needs.

    `capture_frame` asks the browser for a fresh high-resolution still (so the drawing agent can read
    fine detail like hand-written labels — the frames streamed to Gemini are deliberately small and
    low-detail to keep the live session cheap), and `emit` pushes a JSON message back to that browser.
    """

    capture_frame: Callable[[], Awaitable[bytes | None]]
    emit: Callable[[dict[str, object]], Awaitable[None]]


agent = Agent(
    instructions=INSTRUCTIONS,
    deps_type=CameraDeps,
    capabilities=[WebSearch()] if WEB_SEARCH else [],
)
app = FastAPI()
logfire.instrument_fastapi(app)

DRAW_INSTRUCTIONS = (
    'You turn a photo of a hand-drawn sketch — a diagram, system design, flow chart, or wireframe — '
    'into a clean, modern, self-contained HTML page that recreates and tidies up the drawing. '
    'Faithfully preserve the boxes, labels, arrows, and connections the user drew, fixing obvious '
    'wobbles and typos, and lay everything out neatly with clear typography, generous spacing, and '
    'restrained color on a light background. '
    'Design it to fit comfortably on a phone screen in portrait: prefer a vertical flow over very '
    'wide horizontal layouts, let content wrap, and use relative widths so nothing is cut off. '
    'Respond with a SINGLE complete HTML document and nothing else: inline all CSS in a `<style>` '
    'tag, use no external resources (no images, web fonts, or scripts), and no markdown fences.'
)
DRAW_PROMPT = (
    'Recreate and clean up the diagram in this photo as a self-contained HTML page. '
    'What the user asked for: {instructions}'
)
_FENCE_RE = re.compile(r'^```[a-zA-Z]*\n(.*)\n```$', re.DOTALL)


@lru_cache(maxsize=1)
def _draw_agent() -> Agent[None, str]:
    """Build the vision agent that redraws sketches, lazily so it only needs credentials when used."""
    return Agent(DRAW_MODEL, instructions=DRAW_INSTRUCTIONS)


def _extract_html(text: str) -> str:
    """Strip a ```html ... ``` fence if the model wrapped its output in one."""
    text = text.strip()
    match = _FENCE_RE.match(text)
    return (match.group(1) if match else text).strip()


if DRAW:

    @agent.tool
    async def redraw_diagram(ctx: RunContext[CameraDeps], instructions: str) -> str:
        """Redraw a sketch the user is showing the camera as a clean diagram on their screen.

        Use this for a hand-drawn diagram, system design, flow chart, or wireframe when the user asks
        to clean it up, redraw, digitize, or "make a proper version" of what they're holding up.

        Args:
            ctx: The context.
            instructions: What the user wants, in their words (e.g. "clean up this microservices
                diagram and label the queues").
        """
        frame = await ctx.deps.capture_frame()
        if frame is None:
            return "I can't see the camera yet — ask them to hold the drawing up to it."
        await ctx.deps.emit({'type': 'drawing_started', 'request': instructions})
        try:
            result = await _draw_agent().run(
                [
                    DRAW_PROMPT.format(instructions=instructions),
                    BinaryContent(data=frame, media_type='image/jpeg'),
                ]
            )
        except Exception as exc:
            await ctx.deps.emit({'type': 'drawing_error'})
            return f'The redraw failed: {exc}'
        await ctx.deps.emit({'type': 'drawing', 'html': _extract_html(result.output)})
        return 'Done — the cleaned-up diagram is on their screen now. Briefly tell them what you drew.'


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
    return HTMLResponse(
        _INDEX_PATH.read_text(encoding='utf-8').replace('__DEFAULTS__', defaults)
    )


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
    settings = GoogleRealtimeModelSettings(
        voice=params.get('voice') or VOICE,
        output_modality=cast(
            "Literal['audio', 'text']", params.get('modality', 'audio')
        ),
        google_proactive_audio=_truthy(params['proactive'])
        if 'proactive' in params
        else PROACTIVE,
        google_affective_dialog=_truthy(params['affective'])
        if 'affective' in params
        else AFFECTIVE,
    )
    if language_code := params.get('language'):
        settings['google_language_code'] = language_code
    if coverage in ('activity_only', 'all_input', 'all_video'):
        settings['google_turn_coverage'] = coverage
    if vad is not None:
        settings['google_vad'] = vad
    return GoogleRealtimeModel(
        params.get('model') or MODEL,
        settings=settings,
        provider=GoogleCloudProvider(project=GCP_PROJECT, location=GCP_LOCATION)
        if USE_VERTEX
        else 'google',
    )


def _json_message(event: RealtimeEvent) -> dict[str, object] | None:
    """Translate a session event into a JSON message for the browser (audio is sent separately)."""
    if isinstance(event, InputSpeechStartEvent):
        return {'type': 'speech_started'}
    if (
        isinstance(event, PartEndEvent)
        and isinstance(event.part, SpeechPart)
        and event.part.transcript
    ):
        return {'type': event.part.speaker, 'text': event.part.transcript}
    if isinstance(event, PartEndEvent) and isinstance(event.part, NativeToolReturnPart):
        content = cast(object, event.part.content)
        sources = cast('list[object]', content) if isinstance(content, list) else []
        return {
            'type': 'sources',
            'queries': [],
            'sources': [
                {'url': item.get('uri'), 'title': item.get('title')}
                for source in sources
                if isinstance(source, dict)
                and isinstance(
                    (item := cast('dict[str, object]', source)).get('uri'), str
                )
            ],
        }
    if isinstance(event, TurnCompleteEvent):
        return {'type': 'turn_complete'}
    return None


class _FrameStore:
    """Keeps the latest streamed frame and coordinates on-demand high-res snapshots for redrawing.

    Frames streamed to Gemini are small and low-detail (cheap context). `redraw_diagram` instead asks
    the browser for one sharp snapshot via `capture` and waits briefly for the `frame_hd` reply, so
    the drawing agent can read fine detail like hand-written labels — falling back to the last
    streamed frame if no snapshot arrives.
    """

    def __init__(self) -> None:
        self._streamed: bytes | None = None
        self._hd: bytes | None = None
        self._ready: anyio.Event | None = None

    def store_streamed(self, frame: bytes) -> None:
        self._streamed = frame

    def store_hd(self, frame: bytes) -> None:
        self._hd = frame
        if self._ready is not None:
            self._ready.set()

    async def capture(
        self, emit: Callable[[dict[str, object]], Awaitable[None]]
    ) -> bytes | None:
        self._hd, self._ready = None, anyio.Event()
        await emit({'type': 'capture'})
        with anyio.move_on_after(4):
            await self._ready.wait()
        self._ready = None
        return self._hd or self._streamed


async def _dispatch_text(
    session: RealtimeSession,
    text: str,
    store_frame: Callable[[bytes], None],
    store_hd: Callable[[bytes], None],
) -> None:
    """Route a JSON text frame from the browser.

    Handles a streamed camera frame (`image`), an on-demand high-res snapshot (`frame_hd`), a typed
    turn (`text`), or a watch `nudge`.
    """
    try:
        data = json.loads(text)
        if data.get('type') == 'image':
            frame = base64.b64decode(data['data'])
            store_frame(frame)  # low-res fallback if a high-res capture isn't available
            await session.send(
                BinaryContent(data=frame, media_type=data.get('mime') or 'image/jpeg')
            )
        elif data.get('type') == 'frame_hd':
            # High-res still requested for `redraw_diagram` — keep it for the tool but don't forward
            # it to Gemini (the live session only needs the cheap streamed frames for context).
            store_hd(base64.b64decode(data['data']))
        elif data.get('type') == 'text':
            await session.send(data['text'])
        elif data.get('type') == 'nudge':
            # Watch mode: trigger a turn so the model reports visual changes.
            await session.send(WATCH_PROMPT)
    except (ValueError, AttributeError, KeyError, TypeError):
        return


@app.websocket('/ws')
async def ws(socket: WebSocket) -> None:
    await socket.accept()

    # A lock serializes WebSocket sends, since a tool's `emit` can race the event pump.
    send_lock = anyio.Lock()

    async def emit(message: dict[str, object]) -> None:
        async with send_lock:
            await socket.send_json(message)

    frames = _FrameStore()
    deps = CameraDeps(capture_frame=lambda: frames.capture(emit), emit=emit)

    model = _build_model(socket.query_params)
    async with agent.realtime_session(
        model=model,
        deps=deps,
    ) as session:
        async with anyio.create_task_group() as tg:

            async def pump_events() -> None:
                events = cast(
                    'AsyncGenerator[RealtimeEvent, None]', session.__aiter__()
                )
                try:
                    async for event in events:
                        if (
                            isinstance(event, PartDeltaEvent)
                            and isinstance(event.delta, SpeechPartDelta)
                            and event.delta.audio_chunk is not None
                        ):
                            async with send_lock:
                                await socket.send_bytes(event.delta.audio_chunk)
                            continue
                        if (message := _json_message(event)) is not None:
                            await emit(message)
                except Exception:
                    # Log before tearing down so a pump failure (e.g. a provider error surfaced through
                    # the session) is diagnosable instead of a silent disconnect.
                    logfire.exception('Realtime event pump failed')
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
                            await _dispatch_text(
                                session, text, frames.store_streamed, frames.store_hd
                            )
                    tg.cancel_scope.cancel()
                except WebSocketDisconnect:
                    tg.cancel_scope.cancel()

            tg.start_soon(pump_events)
            tg.start_soon(pump_inbound)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)
