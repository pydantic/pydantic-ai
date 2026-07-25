"""Realtime camera + voice assistant — talk to a realtime model and show it your camera.

The browser streams microphone audio (PCM16, 16kHz) and low-rate JPEG camera frames (~1 every 2s) into
a Gemini, OpenAI, or Azure OpenAI realtime session and plays the model's audio back: point your camera
at something and ask about it. xAI realtime is not supported because it has no image input.
The assistant can also ground answers with web search and redraw a hand-drawn sketch into a clean
diagram (see the tool and capability notes below).

Set the credentials for the selected provider: `GOOGLE_API_KEY` for Gemini, `OPENAI_API_KEY` for
OpenAI, or the `AZURE_OPENAI_*` variables for Azure OpenAI. Where org policy disallows Google API
keys, set `GOOGLE_GENAI_USE_VERTEXAI=true` (+ `GOOGLE_CLOUD_PROJECT` /
`GOOGLE_CLOUD_LOCATION` and `gcloud auth application-default login`) to use Vertex AI instead. Put
the config in a `.env` at the repo root, then:

    uv run --all-packages uvicorn pydantic_ai_examples.realtime_camera.app:app

Open http://localhost:8000 on the same machine (localhost is a secure context, so camera/mic work).
To use it from your phone you need HTTPS — expose the local server with a Cloudflare quick tunnel
(no account needed):

    cloudflared tunnel --url http://localhost:8000

then open the printed `https://<...>.trycloudflare.com` URL on the phone and allow camera + mic.

`CAMERA_REALTIME_MODEL` (default `google:gemini-3.1-flash-live-preview`) and
`CAMERA_REALTIME_VOICE` (default: the provider's own default voice) set the fallback defaults. Model
IDs must include the provider prefix, for example `google:gemini-3.1-flash-live-preview`,
`openai:gpt-realtime-2.1`, or `azure:<deployment>`. The UI's model, voice, and output modality settings
work across providers. Leaving the voice empty uses each provider's default, so you don't need to
change it when switching between Gemini and OpenAI (whose voice names differ).
Language, turn coverage, start/end VAD sensitivity, proactive audio, and affective dialog are
Gemini-only; OpenAI/Azure map either sensitivity control to cross-provider turn detection instead.

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

Web search (the `WebSearch` capability) is **on by default** so the assistant can answer with current
facts and cite its sources as chips in the UI. It's enabled per session only when the selected model
supports web search natively (checked through the model's profile), so it stays available alongside the
drawing tool on a model that supports both (e.g. Gemini 3.1) and simply drops on a model that doesn't
(e.g. an OpenAI realtime model). Set `CAMERA_WEB_SEARCH=false` to turn it off entirely.

**Redraw a sketch.** Show the camera a hand-drawn diagram (a system design, flow chart, wireframe)
and ask the assistant to clean it up: it calls the `redraw_diagram` tool with a detailed text
description of what it drew (the realtime model already has the live camera in context, so it
describes the diagram rather than re-sending a photo — which keeps the tool fast and captures the
moment the user meant). A separate drawing agent (Claude Sonnet 5 through the gateway by default) turns
that description into a clean, self-contained HTML diagram; the browser renders it in an overlay and
can export it to PNG client-side. Set `CAMERA_DRAW=false` to disable, or `CAMERA_DRAW_MODEL` to any
`provider:model`. Proactive audio lets a Gemini native-audio model decide when to speak and stay
silent; affective dialog enables emotion-aware delivery. Both settings are Gemini native-audio only.
"""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

import anyio
import logfire
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.messages import NativeToolReturnPart
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.providers.google_cloud import GoogleCloudProvider
from pydantic_ai.realtime import (
    InputSpeechStartEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RealtimeError,
    RealtimeEvent,
    RealtimeModel,
    RealtimeModelSettings,
    RealtimeSession,
    ReconnectPolicy,
    SpeechPart,
    SpeechPartDelta,
    TurnCompleteEvent,
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


# Use Vertex AI (ADC) instead of a Gemini API key when `GOOGLE_GENAI_USE_VERTEXAI` is truthy — handy
# where org policy disallows API keys. Needs `gcloud auth application-default login` + project/location.
USE_VERTEX = _truthy(os.environ.get('GOOGLE_GENAI_USE_VERTEXAI'))
MODEL = os.environ.get('CAMERA_REALTIME_MODEL', 'google:gemini-3.1-flash-live-preview')
# Empty by default so each provider picks its own default voice — no need to change it when switching
# between Gemini and OpenAI, whose voice names differ (Gemini rejects `alloy`, OpenAI rejects `Puck`).
VOICE = os.environ.get('CAMERA_REALTIME_VOICE', '')
GCP_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
GCP_LOCATION = os.environ.get('GOOGLE_CLOUD_LOCATION')
# `all_input` keeps every camera frame in the model's context and works on both the Developer API and
# Vertex; `all_video` is newer and not yet accepted on Vertex. Proactive/affective audio are
# native-audio-only knobs, off by default so the standard model still connects.
TURN_COVERAGE = cast(
    "Literal['activity_only', 'all_input', 'all_video']",
    os.environ.get('CAMERA_TURN_COVERAGE', 'all_input'),
)
PROACTIVE = _truthy(os.environ.get('CAMERA_PROACTIVE'))
AFFECTIVE = _truthy(os.environ.get('CAMERA_AFFECTIVE'))
# Sketch-to-diagram: a `redraw_diagram` tool passes the realtime model's text description of the
# sketch to a separate drawing agent that renders it as clean HTML. On by default; the drawing model
# defaults to Claude Sonnet 5 through the Pydantic AI Gateway (`PYDANTIC_AI_GATEWAY_API_KEY`), which is
# strong at self-contained HTML. `CAMERA_DRAW_MODEL` takes any `provider:model` string to use another
# model (e.g. `google:gemini-3.5-flash` to reuse the live session's `GOOGLE_API_KEY`).
DRAW = _truthy(os.environ.get('CAMERA_DRAW', 'true'))
DRAW_MODEL = os.environ.get('CAMERA_DRAW_MODEL', 'gateway/anthropic:claude-sonnet-5')
# Web search (the `WebSearch` capability) — on by default. It's only enabled for a session when the
# selected model supports web search natively, checked per connection through the model's profile (see
# `_web_search_supported`), so it and the drawing tool can be active together on a model that supports
# both (e.g. Gemini 3.1). Set `CAMERA_WEB_SEARCH=false` to turn it off entirely.
WEB_SEARCH = _truthy(os.environ.get('CAMERA_WEB_SEARCH', 'true'))
WATCH_PROMPT = os.environ.get(
    'CAMERA_WATCH_PROMPT',
    "Look at the current camera view. In a few words, say what's changed since you last spoke; "
    'if nothing notable changed, stay silent.',
)
_INDEX_PATH = Path(__file__).parent / 'index.html'


def _instructions(*, web_search: bool) -> str:
    """The assistant's instructions, built per connection.

    The web-search guidance is included only when web search is actually enabled for the selected model
    (see `_web_search_supported`), so the model isn't told about a tool it doesn't have.
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
            "them out loud that you're about to redraw it and that it takes a few moments (around 30 "
            "seconds) — don't leave them waiting in silence — THEN call the tool. The drawing tool "
            'cannot see the camera, so pass it a thorough text description as `instructions`: every box '
            'and its label, every arrow and what it connects, groupings, and the overall layout, plus '
            'what the user asked you to change. Be specific — it can only draw what you describe. Once '
            'the diagram appears, briefly describe what you drew.'
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
    'Respond with a SINGLE complete HTML document and nothing else: inline all CSS in a `<style>` '
    'tag, use no external resources (no images, web fonts, or scripts), and no markdown fences.'
)
DRAW_PROMPT = 'Recreate this diagram as a self-contained HTML page:\n\n{instructions}'
_FENCE_RE = re.compile(r'^```[a-zA-Z]*\n(.*)\n```$', re.DOTALL)


@lru_cache(maxsize=1)
def _draw_agent() -> Agent[None, str]:
    """Build the drawing agent that redraws sketches, lazily so it only needs credentials when used."""
    return Agent(DRAW_MODEL, name='diagram_drawer', instructions=DRAW_INSTRUCTIONS)


def _extract_html(text: str) -> str:
    """Strip a ```html ... ``` fence if the model wrapped its output in one."""
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
    """Whether `model` supports web search natively, read from its realtime profile.

    Web search is enabled per connection only when this is true, so switching the model in the UI to one
    without native web search (e.g. an OpenAI realtime model) simply drops the capability instead of
    failing the session.
    """
    return WebSearchTool in model.profile.get('supported_native_tools', frozenset())


def _build_agent(*, web_search: bool) -> Agent[CameraDeps, str]:
    """Build the camera assistant for one connection.

    The agent is per connection because whether web search is available depends on the selected model
    (see `_web_search_supported`), so its capabilities and instructions vary. The `redraw_diagram` tool
    is registered whenever drawing is enabled, independently of web search — the two can be active at once.
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


def _build_model(params: Mapping[str, str]) -> RealtimeModel:
    """Build the selected realtime model with provider-appropriate UI settings."""
    model_id = params.get('model') or MODEL
    if USE_VERTEX and model_id.startswith('google:'):
        model = GoogleRealtimeModel(
            model_id.removeprefix('google:'),
            provider=GoogleCloudProvider(project=GCP_PROJECT, location=GCP_LOCATION),
        )
    else:
        model = infer_realtime_model(model_id)

    start, end = params.get('start_sensitivity'), params.get('end_sensitivity')
    common_settings = RealtimeModelSettings(
        output_modality=cast(
            "Literal['audio', 'text']", params.get('modality', 'audio')
        ),
    )
    # Only set a voice when one is given; an empty voice lets each provider use its own default, so the
    # same settings work across Gemini and OpenAI without swapping voice names.
    if voice := (params.get('voice') or VOICE):
        common_settings['voice'] = voice
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
        if language_code := params.get('language'):
            settings['google_language_code'] = language_code
        coverage = params.get('turn_coverage') or TURN_COVERAGE
        if coverage in ('activity_only', 'all_input', 'all_video'):
            settings['google_turn_coverage'] = coverage
        if start in ('high', 'low') or end in ('high', 'low'):
            settings['google_vad'] = AutomaticVAD(
                start_sensitivity=start if start in ('high', 'low') else None,
                end_sensitivity=end if end in ('high', 'low') else None,
            )
        model.settings = settings
        model.reconnect = ReconnectPolicy(max_attempts=5)
    elif isinstance(model, OpenAIRealtimeModel):
        settings = OpenAIRealtimeModelSettings(**common_settings)
        if sensitivity := start or end:
            if sensitivity in ('high', 'low'):
                settings['turn_detection'] = TurnDetection(sensitivity=sensitivity)
        model.settings = settings
        model.reconnect = ReconnectPolicy(max_attempts=5)
    else:
        raise ValueError(
            f'Realtime model {model_id!r} does not support camera image input'
        )
    return model


def _grounding_sources(content: object) -> list[dict[str, object]]:
    """Extract `{url, title}` source chips from a grounding `NativeToolReturnPart.content`.

    Google Search grounding returns its cited pages as a list of `{'uri', 'title', ...}` chunks; keep
    the ones with a usable URL. Typed defensively (the content is provider-shaped) so an unexpected
    shape degrades to no chips rather than an error.
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
        case InputSpeechStartEvent():
            # The user started talking over the model — a barge-in; the browser flushes buffered audio.
            # (The realtime session records the barge-in in its telemetry.)
            return {'type': 'speech_started'}
        case PartEndEvent(part=NativeToolReturnPart(content=content)):
            # Google Search grounding finished; surface its cited sources as chips.
            return {
                'type': 'sources',
                'queries': [],
                'sources': _grounding_sources(content),
            }
        case TurnCompleteEvent():
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
        data = json.loads(text)
        match data.get('type'):
            case 'image':
                content: str | BinaryContent = BinaryContent(
                    data=base64.b64decode(data['data']),
                    media_type=data.get('mime') or 'image/jpeg',
                )
            case 'text':
                content = data['text']
            case 'nudge':
                # Watch mode: trigger a turn so the model reports visual changes.
                content = WATCH_PROMPT
            case _:
                return
    except (ValueError, AttributeError, KeyError, TypeError):
        logfire.exception('Ignoring malformed browser message')
        return
    await session.send(content)


async def _run_session(
    session: RealtimeSession,
    socket: WebSocket,
    emit: Callable[[dict[str, object]], Awaitable[None]],
    send_lock: anyio.Lock,
) -> None:
    """Bridge a live session to the browser: model output out, mic audio and camera frames in.

    Two concurrent pumps run until either side ends; when one stops (a disconnect or a provider drop)
    it cancels the task group so the other unwinds and the session closes.
    """
    async with anyio.create_task_group() as tg:

        async def pump_events() -> None:
            # Transcript deltas carry only the part index, not the speaker, so we remember which
            # speaker each index belongs to from its `PartStartEvent`. (A known ergonomic wrinkle of
            # the current realtime event API.)
            speakers: dict[int, str] = {}
            try:
                async for event in session:
                    match event:
                        case PartDeltaEvent(
                            delta=SpeechPartDelta(audio_chunk=chunk)
                        ) if chunk is not None:
                            # Model audio goes back as raw binary frames.
                            async with send_lock:
                                await socket.send_bytes(chunk)
                        case PartStartEvent(
                            index=index, part=SpeechPart(speaker=speaker)
                        ):
                            speakers[index] = speaker
                        case PartDeltaEvent(
                            index=index, delta=SpeechPartDelta(transcript_delta=delta)
                        ) if delta:
                            # Stream the transcript into the browser bubble as it arrives.
                            await emit(
                                {
                                    'type': 'transcript',
                                    'speaker': speakers.get(index, 'assistant'),
                                    'delta': delta,
                                }
                            )
                        case _:
                            if (message := _json_message(event)) is not None:
                                await emit(message)
            except Exception:
                # Log before tearing down so a pump failure (e.g. a provider error surfaced through the
                # session) is diagnosable instead of a silent disconnect.
                logfire.exception('Realtime event pump failed')
            finally:
                tg.cancel_scope.cancel()

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
            except WebSocketDisconnect:
                pass
            except (RealtimeError, websockets.exceptions.ConnectionClosed):
                # Send-side recovery is not reconnect-aware yet; a provider drop ends this session and
                # lets the browser reconnect. See https://github.com/pydantic/pydantic-ai/issues/6703.
                logfire.exception('Realtime inbound pump failed')
            finally:
                tg.cancel_scope.cancel()

        tg.start_soon(pump_events)
        tg.start_soon(pump_inbound)


@app.websocket('/ws')
async def ws(socket: WebSocket) -> None:
    await socket.accept()

    # A lock serializes WebSocket sends, since a tool's `emit` can race the event pump.
    send_lock = anyio.Lock()

    async def emit(message: dict[str, object]) -> None:
        async with send_lock:
            await socket.send_json(message)

    deps = CameraDeps(emit=emit)

    try:
        model = _build_model(socket.query_params)
    except (UserError, ValueError) as exc:
        # Unknown provider prefix, or a model that can't take camera frames.
        logfire.exception('Could not build realtime model')
        await emit({'type': 'error', 'message': str(exc)})
        return

    # Enable web search only when the selected model supports it natively, so it and the drawing tool can
    # both be active on a model that supports both (e.g. Gemini 3.1) without failing on one that doesn't.
    agent = _build_agent(web_search=WEB_SEARCH and _web_search_supported(model))

    try:
        async with agent.realtime(model, deps=deps).session() as session:
            await _run_session(session, socket, emit, send_lock)
    except ModelAPIError as exc:
        # The provider rejected the session at connect time — e.g. a voice this model doesn't support, or
        # an unknown model/deployment. Log the full error (with its chained cause) so the underlying
        # reason is visible in Logfire, and surface a message to the browser instead of a bare disconnect.
        logfire.exception('Realtime session failed to connect')
        await emit({'type': 'error', 'message': str(exc)})


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)
