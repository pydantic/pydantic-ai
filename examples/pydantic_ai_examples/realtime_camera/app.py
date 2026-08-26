"""Realtime camera + voice assistant: a browser bridged to a provider-agnostic realtime session.

The browser streams microphone PCM, one camera frame per second, and typed text over a WebSocket;
the server forwards them into a [realtime session](https://pydantic.dev/docs/ai/realtime/overview/) and
streams model audio, transcripts, and tool results back. The spine is the two pumps in
`_run_session`; everything else is configuration and optional demo features: **Watch** (proactive
narration of scene changes), **web search** with citation chips, and **sketch redrawing** through a
second agent.

Set the API key for the model you want to talk to — `GOOGLE_API_KEY` for the default Gemini model,
`OPENAI_API_KEY` for OpenAI, or the `AZURE_OPENAI_*` variables for Azure OpenAI — in a `.env` at the
repo root, then run:

    uv run -m pydantic_ai_examples.realtime_camera.app

and open http://localhost:8000 on the same machine. `CAMERA_REALTIME_MODEL` sets the default model
(the UI's model picker takes any `provider:model` per session); see README.md for the other
`CAMERA_*` settings, Vertex AI credentials, and how the bridge works.

This is a development example: the WebSocket checks browser origins so other sites can't drive your
session, but there is no authentication — don't expose the server to the internet.
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import anyio
import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from pydantic_ai import (
    Agent,
    BinaryContent,
    PartDeltaEvent,
    PartEndEvent,
    RunContext,
    SpeechPartDelta,
)
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.messages import NativeToolReturnPart, TextPartDelta
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.realtime import (
    RealtimeError,
    RealtimeEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeModel,
    RealtimeModelSettings,
    RealtimeResponseInterruptedEvent,
    RealtimeSession,
    RealtimeTurnCompleteEvent,
    ReconnectPolicy,
    TurnDetection,
    infer_realtime_model,
)
from pydantic_ai.realtime.google import (
    AutomaticVAD,
    GoogleRealtimeModel,
    GoogleRealtimeModelSettings,
)
from pydantic_ai.realtime.openai import (
    OpenAIRealtimeModel,
    OpenAIRealtimeModelSettings,
)

load_dotenv()

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured.
# Configure after `load_dotenv()` so a `LOGFIRE_TOKEN` in `.env` is picked up.
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


def _truthy(value: str | None) -> bool:
    """Parse an env/query flag: `'1'`, `'true'`, `'yes'`, or `'on'` (any case) mean enabled."""
    return (value or '').lower() in ('1', 'true', 'yes', 'on')


MODEL = os.environ.get('CAMERA_REALTIME_MODEL', 'google:gemini-3.1-flash-live-preview')
# Empty by default so each provider picks its own default voice — no need to change it when switching
# between Gemini and OpenAI, whose voice names differ (Gemini rejects `alloy`, OpenAI rejects `Puck`).
VOICE = os.environ.get('CAMERA_REALTIME_VOICE', '')
# Use Vertex AI (Application Default Credentials) instead of a Gemini API key — handy where org
# policy disallows API keys. Needs `gcloud auth application-default login` + `GOOGLE_CLOUD_PROJECT`.
USE_VERTEX = _truthy(os.environ.get('GOOGLE_GENAI_USE_VERTEXAI'))
# `all_input` keeps every camera frame in the model's context — the live scene the assistant reasons
# about — and works on both the Gemini Developer API and Vertex AI (the newer `all_video` doesn't yet).
TURN_COVERAGE = os.environ.get('CAMERA_TURN_COVERAGE', 'all_input')
# Gemini native-audio-only knobs, off by default so the default model still connects: proactive audio
# lets the model stay silent when a Watch nudge finds nothing new; affective dialog adapts delivery
# to emotion in the conversation.
PROACTIVE = _truthy(os.environ.get('CAMERA_PROACTIVE'))
AFFECTIVE = _truthy(os.environ.get('CAMERA_AFFECTIVE'))
# Sketch-to-diagram: the `redraw_diagram` tool passes the realtime model's text description of a
# sketch to a separate drawing agent that renders it as clean HTML. The default drawing model reuses
# the `GOOGLE_API_KEY` the default realtime model already needs, and is a fast small model because
# the user is waiting on a live call: output tokens dominate the redraw's latency, and a larger
# model mostly adds thinking time. `CAMERA_DRAW_MODEL` takes any `provider:model` string.
DRAW = _truthy(os.environ.get('CAMERA_DRAW', 'true'))
DRAW_MODEL = os.environ.get('CAMERA_DRAW_MODEL', 'google:gemini-3.5-flash')
# Web search (the `WebSearch` capability) — on by default, but only enabled for a session when the
# selected model supports web search natively (see `_web_search_supported`), so switching models
# drops the capability instead of failing the session.
WEB_SEARCH = _truthy(os.environ.get('CAMERA_WEB_SEARCH', 'true'))
WATCH_PROMPT = os.environ.get(
    'CAMERA_WATCH_PROMPT',
    "Look at the current camera view. In a few words, say what's changed since you last spoke; "
    'if nothing notable changed, stay silent.',
)
_INDEX_PATH = Path(__file__).parent / 'index.html'


def _same_origin(socket: WebSocket) -> bool:
    """Accept browser WebSockets only from the origin serving this development example.

    Any web page can open a WebSocket to this server (which spends your API credits), so the
    browser-reported `Origin` must match the host the request was addressed to. Three ways in: a
    loopback origin matching `Host` (direct local use); an origin matching `X-Forwarded-Host` (a
    reverse proxy such as Codespaces or a dev tunnel — trustworthy because the browser WebSocket API
    cannot send custom headers, so its presence proves a real proxy hop); or an origin listed in
    `CAMERA_ALLOWED_ORIGINS` (comma-separated `scheme://host[:port]`, for proxies that forward
    neither).
    """
    origin = socket.headers.get('origin')
    if not origin:
        return False
    allowed = os.environ.get('CAMERA_ALLOWED_ORIGINS', '')
    if origin in {value.strip() for value in allowed.split(',') if value.strip()}:
        return True
    parsed = urlsplit(origin)
    if parsed.scheme not in ('http', 'https'):
        return False
    if parsed.netloc == socket.headers.get('x-forwarded-host'):
        return True
    return parsed.hostname in (
        'localhost',
        '127.0.0.1',
        '::1',
    ) and parsed.netloc == socket.headers.get('host')


def _instructions(*, web_search: bool) -> str:
    """The assistant's instructions, built per connection.

    The web-search guidance is included only when web search is actually enabled for the selected
    model (see `_web_search_supported`), so the model isn't told about a tool it doesn't have.
    """
    return (
        'You are a friendly, concise voice assistant. The user is talking to you and may show you things '
        'through their camera — when relevant, describe and reason about what you can see. Keep replies '
        'short and natural, like a conversation.'
        + (
            ' Search the web when a question needs current or external facts.'
            if web_search
            else ''
        )
        + (
            ' You can redraw a hand-drawn sketch the user shows you — a diagram, system design, flow '
            'chart, or wireframe — into a clean version with the `redraw_diagram` tool. Do NOT call it '
            'the moment you see a drawing. First make sure you understand what they actually want: if '
            "they haven't said, ask one short question — keep it faithful but tidier, turn it into a "
            'flowchart, restructure it, add or label something? Once their intent is clear, FIRST tell '
            "them out loud that you're about to redraw it and that it takes a few moments (around ten"
            "seconds) — don't leave them waiting in silence — THEN call the tool. The drawing tool "
            'cannot see the camera, so pass it a thorough text description as `instructions`: every box '
            'and its label, every arrow and what it connects, groupings, and the overall layout, plus '
            'what the user asked you to change. Be specific — it can only draw what you describe. '
            'After calling the tool, stop talking until its result arrives — never say the redraw is '
            'done in the same breath as calling it, because the drawing takes several seconds. Once '
            'the result arrives, briefly describe what you drew.'
            if DRAW
            else ''
        )
    )


@dataclass
class CameraDeps:
    """Per-connection hooks the `redraw_diagram` tool needs.

    `emit` pushes a JSON message back to this connection's browser — the tool uses it to show and
    then clear the drawing overlay while the diagram is being generated.
    """

    emit: Callable[[dict[str, object]], Awaitable[None]]


app = FastAPI()
logfire.instrument_fastapi(app)

DRAW_INSTRUCTIONS = (
    'You turn a text description of a hand-drawn sketch — a diagram, system design, flow chart, or '
    'wireframe — into a clean, modern, self-contained HTML page that recreates and tidies up the '
    'drawing. Faithfully render every box, label, arrow, and connection the description mentions, '
    'and lay everything out neatly with clear typography, generous spacing, and restrained color on '
    'a light background. '
    'Design it to fit comfortably on a phone screen in portrait: prefer a vertical flow over very '
    'wide horizontal layouts, let content wrap, and use relative widths so nothing is cut off. '
    # The user is waiting on a live call while this generates, so latency is part of the spec:
    # output tokens dominate the wall-clock time, and a compact page halves it.
    'Keep the page LEAN so it generates fast: one short `<style>` block with a few shared classes, '
    'simple semantic markup (plain divs, or one inline SVG for connector-heavy layouts), and no '
    'decorative gradients, shadows, animations, or per-element styling. Do not restate the '
    'description in comments or prose. '
    'Respond with a SINGLE complete HTML document and nothing else: inline all CSS in a `<style>` '
    'tag, use no external resources (no images, web fonts, or scripts), and no markdown fences.'
)
DRAW_PROMPT = 'Recreate this diagram as a self-contained HTML page:\n\n{instructions}'
_FENCE_RE = re.compile(r'^```[a-zA-Z]*\n(.*)\n```$', re.DOTALL)


@lru_cache(maxsize=1)
def _draw_agent() -> Agent[None, str]:
    """Build the drawing agent that redraws sketches, lazily so it only needs credentials when used.

    A full HTML page for a busy diagram can outgrow a provider's default `max_tokens` (Anthropic's
    default is low enough to cut off mid-page), so the limit is raised explicitly.
    """
    return Agent(
        DRAW_MODEL,
        name='diagram_drawer',
        instructions=DRAW_INSTRUCTIONS,
        model_settings={'max_tokens': 16_384},
    )


def _extract_html(text: str) -> str:
    """Strip a Markdown HTML fence if the model wrapped its output in one."""
    text = text.strip()
    match = _FENCE_RE.match(text)
    return (match.group(1) if match else text).strip()


async def redraw_diagram(ctx: RunContext[CameraDeps], instructions: str) -> str:
    """Redraw a sketch the user is showing the camera as a clean diagram on their screen.

    Use this for a hand-drawn diagram, system design, flow chart, or wireframe when the user asks
    to clean it up, redraw, digitize, or "make a proper version" of what they're holding up.

    The drawing tool cannot see the camera, so describe the sketch in full here — it draws only
    what you describe.

    Args:
        ctx: The context.
        instructions: A thorough text description of the diagram to draw: every box and its
            label, every arrow and what it connects, groupings, overall layout, and any changes
            the user asked for (e.g. "clean up this microservices diagram and label the queues").
    """
    await ctx.deps.emit({'type': 'drawing_started', 'request': instructions})
    try:
        result = await _draw_agent().run(DRAW_PROMPT.format(instructions=instructions))
    except anyio.get_cancelled_exc_class():
        # The realtime model cancelled this call mid-draw — e.g. the user barged in, so the provider
        # abandoned the turn (a `ToolCallCancelled`, which the session maps to task cancellation).
        # Cancellation is a `BaseException`, so it skips the `except Exception` below; clear the
        # browser's loading overlay (shielded, since we're unwinding a cancellation) before re-raising.
        with anyio.CancelScope(shield=True):
            await ctx.deps.emit({'type': 'drawing_error'})
        raise
    except Exception as exc:
        await ctx.deps.emit({'type': 'drawing_error'})
        return f'The redraw failed: {exc}'
    await ctx.deps.emit({'type': 'drawing', 'html': _extract_html(result.output)})
    return 'Done — the cleaned-up diagram is on their screen now. Briefly tell them what you drew.'


def _web_search_supported(model: RealtimeModel) -> bool:
    """Whether `model` supports web search natively, read from its realtime profile."""
    return WebSearchTool in model.profile.get('supported_native_tools', frozenset())


def _build_agent(*, web_search: bool) -> Agent[CameraDeps, str]:
    """Build the camera assistant for one connection.

    The agent is per connection because whether web search is available depends on the selected
    model, so its capabilities and instructions vary. The `redraw_diagram` tool is registered
    whenever drawing is enabled, independently of web search — the two can be active at once.
    """
    agent = Agent(
        # Named so Logfire tells this run apart from the drawing agent's.
        name='camera_assistant',
        instructions=_instructions(web_search=web_search),
        deps_type=CameraDeps,
        capabilities=[WebSearch()] if web_search else [],
    )
    if DRAW:
        agent.tool(redraw_diagram)
    return agent


@app.get('/')
async def index() -> HTMLResponse:
    # Seed the settings panel with the server's env-configured defaults so the UI mirrors them.
    # Angle brackets are JSON-escaped so env-supplied values can't break out of the script tag.
    defaults = (
        json.dumps(
            {
                'model': MODEL,
                'voice': VOICE,
                'turn_coverage': TURN_COVERAGE,
                'proactive': PROACTIVE,
                'affective': AFFECTIVE,
            }
        )
        .replace('<', '\\u003c')
        .replace('>', '\\u003e')
    )
    return HTMLResponse(
        _INDEX_PATH.read_text(encoding='utf-8').replace('__DEFAULTS__', defaults)
    )


def _build_model(params: Mapping[str, str]) -> RealtimeModel:
    """Build the selected realtime model with provider-appropriate UI settings."""
    model_id = params.get('model') or MODEL
    if USE_VERTEX and model_id.startswith('google:'):
        model = GoogleRealtimeModel(
            model_id.removeprefix('google:'), provider='google-cloud'
        )
    else:
        model = infer_realtime_model(model_id)

    modality = params.get('modality', 'audio')
    if modality not in ('audio', 'text'):
        raise ValueError(f'Output modality {modality!r} must be "audio" or "text"')
    common_settings = RealtimeModelSettings(
        output_modality=modality, reconnect=ReconnectPolicy(max_attempts=5)
    )
    voice = params.get('voice') or VOICE
    start, end = params.get('start_sensitivity'), params.get('end_sensitivity')
    if isinstance(model, GoogleRealtimeModel):
        settings = GoogleRealtimeModelSettings(
            **common_settings,
            google_proactive_audio=_truthy(params['proactive'])
            if 'proactive' in params
            else PROACTIVE,
            google_affective_dialog=_truthy(params['affective'])
            if 'affective' in params
            else AFFECTIVE,
            google_enable_session_resumption=True,
        )
        if voice:
            settings['google_voice'] = voice
        if language_code := params.get('language'):
            settings['google_language_code'] = language_code
        coverage = params.get('turn_coverage') or TURN_COVERAGE
        if coverage in ('activity_only', 'all_input', 'all_video'):
            settings['google_turn_coverage'] = coverage
        if start in ('high', 'low') or end in ('high', 'low'):
            vad: AutomaticVAD = {}
            if start in ('high', 'low'):
                vad['start_sensitivity'] = start
            if end in ('high', 'low'):
                vad['end_sensitivity'] = end
            settings['google_vad'] = vad
        model.settings = settings
    elif isinstance(model, OpenAIRealtimeModel):
        settings = OpenAIRealtimeModelSettings(**common_settings)
        if voice:
            settings['openai_voice'] = voice
        # OpenAI has one shared turn-detection sensitivity, so either UI VAD control maps onto it.
        if (sensitivity := start or end) in ('high', 'low'):
            settings['turn_detection'] = TurnDetection(sensitivity=sensitivity)
        model.settings = settings
    else:
        raise ValueError(
            f'Realtime model {model_id!r} does not support camera image input'
        )
    return model


def _grounding_sources(content: object) -> list[dict[str, object]]:
    """Extract `{url, title}` source chips from a grounding `NativeToolReturnPart.content`.

    Google Search grounding returns cited pages as a list of provider-shaped chunks; keep the ones
    with a usable URL, and degrade to no chips on an unexpected shape rather than an error.
    """
    if not isinstance(content, list):
        return []
    sources: list[dict[str, object]] = []
    for raw in cast('list[object]', content):
        if not isinstance(raw, dict):
            continue
        chunk = cast('dict[str, object]', raw)
        if isinstance(url := chunk.get('uri'), str):
            sources.append({'url': url, 'title': chunk.get('title')})
    return sources


def _json_message(event: RealtimeEvent) -> dict[str, object] | None:
    """Translate a session event into a JSON message for the browser.

    Audio and the incrementally streamed transcript are handled directly in `pump_events`; this
    covers the remaining one-shot events (barge-in, grounding sources, end of turn).
    """
    match event:
        case RealtimeInputSpeechStartEvent() | RealtimeResponseInterruptedEvent():
            # A barge-in: the user started talking over the model, or the provider reported the
            # response interrupted — Gemini signals only the latter, without an
            # `RealtimeInputSpeechStartEvent`. The browser flushes buffered audio either way.
            return {'type': 'speech_started'}
        case PartEndEvent(part=NativeToolReturnPart(content=content)):
            # Google Search grounding finished; surface its cited sources as chips.
            return {
                'type': 'sources',
                'queries': [],
                'sources': _grounding_sources(content),
            }
        case RealtimeTurnCompleteEvent():
            return {'type': 'turn_complete'}
        case _:
            return None


async def _dispatch_text(session: RealtimeSession, text: str) -> None:
    """Route a JSON text frame from the browser.

    Handles a streamed camera frame (`image`), a typed turn (`text`), or a watch `nudge`.
    """
    try:
        # Decode and validate the message. A malformed frame is ignored here, but a genuine send
        # failure must surface, so `session.send()` stays outside this guard.
        raw: object = json.loads(text)
        if not isinstance(raw, dict):
            return
        # JSON object keys are strings.
        data = cast('dict[str, object]', raw)
        match data.get('type'):
            case 'image':
                image_data = data.get('data')
                media_type = data.get('mime', 'image/jpeg')
                if not isinstance(image_data, str) or not isinstance(media_type, str):
                    return
                content: str | BinaryContent = BinaryContent(
                    data=base64.b64decode(image_data),
                    media_type=media_type,
                )
            case 'text':
                text_content = data.get('text')
                if not isinstance(text_content, str):
                    return
                content = text_content
            case 'nudge':
                # Watch mode: trigger a turn so the model reports visual changes.
                content = WATCH_PROMPT
            case _:
                return
    except ValueError:
        logfire.exception('Ignoring malformed browser message')
        return
    await session.send(content)


async def _run_session(
    session: RealtimeSession,
    socket: WebSocket,
    emit: Callable[[dict[str, object]], Awaitable[None]],
    send_lock: anyio.Lock,
) -> None:
    """The realtime bridge: model output goes out while browser input goes in.

    Two concurrent pumps run until either side ends; when one stops (a disconnect or a provider drop)
    it cancels the task group so the other unwinds and the session closes.
    """
    async with anyio.create_task_group() as tg:

        async def pump_events() -> None:
            try:
                async for event in session:
                    match event:
                        case PartDeltaEvent(
                            delta=SpeechPartDelta(audio_chunk=chunk)
                        ) if chunk is not None:
                            # Model audio goes back as raw binary frames.
                            async with send_lock:
                                await socket.send_bytes(chunk)
                        case PartDeltaEvent(
                            delta=SpeechPartDelta(
                                speaker=speaker, transcript_delta=delta
                            )
                        ) if delta:
                            # Stream the transcript into the browser bubble as it arrives. Both
                            # speakers' transcripts stream at once, so each delta names its own
                            # speaker rather than needing to be tied back to a `PartStartEvent`.
                            await emit(
                                {
                                    'type': 'transcript',
                                    'speaker': speaker or 'assistant',
                                    'delta': delta,
                                }
                            )
                        case PartDeltaEvent(
                            delta=TextPartDelta(content_delta=delta)
                        ) if delta:
                            # With `output_modality='text'` the assistant's reply arrives as text
                            # part deltas rather than speech; stream it into the same bubble.
                            await emit(
                                {
                                    'type': 'transcript',
                                    'speaker': 'assistant',
                                    'delta': delta,
                                }
                            )
                        case _:
                            if (message := _json_message(event)) is not None:
                                await emit(message)
            except Exception as exc:
                logfire.exception('Realtime event pump failed')
                # Best effort: the socket itself may be what failed.
                with suppress(Exception):
                    await emit(
                        {'type': 'error', 'message': f'Realtime provider failed: {exc}'}
                    )
            finally:
                tg.cancel_scope.cancel()

        async def pump_inbound() -> None:
            try:
                while True:
                    message: Mapping[str, object] = await socket.receive()
                    if message.get('type') == 'websocket.disconnect':
                        break
                    if isinstance(chunk := message.get('bytes'), bytes):
                        await session.send_audio(chunk)
                    elif isinstance(text := message.get('text'), str):
                        await _dispatch_text(session, text)
            except WebSocketDisconnect:
                pass
            except RealtimeError as exc:
                # Send-side recovery is not reconnect-aware yet; a provider drop ends this session and
                # lets the browser reconnect. See https://github.com/pydantic/pydantic-ai/issues/6703.
                logfire.exception('Realtime inbound pump failed')
                with suppress(Exception):
                    await emit(
                        {'type': 'error', 'message': f'Realtime provider failed: {exc}'}
                    )
            finally:
                tg.cancel_scope.cancel()

        tg.start_soon(pump_events)
        tg.start_soon(pump_inbound)


@app.websocket('/ws')
async def ws(socket: WebSocket) -> None:
    if not _same_origin(socket):
        logfire.warn(
            'Rejected WebSocket: origin {origin!r} does not match host {host!r} or forwarded host '
            '{forwarded_host!r}. Behind a proxy that rewrites Host, set CAMERA_ALLOWED_ORIGINS to '
            "the browser-facing origin (e.g. 'https://myapp.example.com').",
            origin=socket.headers.get('origin'),
            host=socket.headers.get('host'),
            forwarded_host=socket.headers.get('x-forwarded-host'),
        )
        await socket.close(code=1008, reason='WebSocket origin does not match Host')
        return
    await socket.accept()

    # A lock serializes WebSocket sends, since a tool's `emit` can race the event pump.
    send_lock = anyio.Lock()

    async def emit(message: dict[str, object]) -> None:
        async with send_lock:
            await socket.send_json(message)

    try:
        model = _build_model(socket.query_params)
    except (UserError, ValueError) as exc:
        logfire.exception('Could not build realtime model')
        await emit({'type': 'error', 'message': str(exc)})
        return

    # This handshake must precede mic capture: raw PCM does not carry its sample rate.
    await emit(
        {
            'type': 'session_config',
            'input_sample_rate': model.audio_input_sample_rate,
            'output_sample_rate': model.audio_output_sample_rate,
        }
    )

    agent = _build_agent(web_search=WEB_SEARCH and _web_search_supported(model))
    try:
        async with agent.realtime(
            model, deps=CameraDeps(emit=emit)
        ).session() as session:
            await _run_session(session, socket, emit, send_lock)
    except ModelAPIError as exc:
        logfire.exception('Realtime session failed to connect')
        await emit({'type': 'error', 'message': str(exc)})


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)
