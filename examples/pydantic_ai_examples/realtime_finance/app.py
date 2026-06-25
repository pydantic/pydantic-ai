"""FastAPI backend for the realtime finance voice demo.

A true speech-to-speech demo: the browser streams microphone audio (PCM16, 24kHz) into an OpenAI
Realtime session and plays the model's generated audio back. When the user asks about their money the
realtime model delegates to a finance supervisor agent (a normal text model), whose tools return
typed widgets rendered as cards in the chat.

The voice provider, model, voice, and tuning are chosen from the UI's settings panel and passed to
`/ws` as query params (`_build_realtime_model` maps them onto the provider). Requires `OPENAI_API_KEY`
(used for the supervisor, and the OpenAI voice) and, to use the Gemini voice, `GOOGLE_API_KEY` (or
Vertex AI via `GOOGLE_GENAI_USE_VERTEXAI=true`). Put them in a `.env` at the repo root, then:

    uv run --all-packages uvicorn pydantic_ai_examples.realtime_finance.app:app

`FINANCE_SUPERVISOR_MODEL` (default `openai:gpt-4o-mini`), `FINANCE_REALTIME_MODEL` (default
`gpt-realtime`), `FINANCE_REALTIME_VOICE` (default `alloy`), `FINANCE_GEMINI_MODEL` (default
`gemini-live-2.5-flash`, or `gemini-live-2.5-flash-native-audio` on Vertex) and `FINANCE_GEMINI_VOICE`
(default `Puck`) set the fallback defaults; the UI panel overrides them per session.

Set `LOGFIRE_WRITE_TOKEN` to stream traces (the realtime session, tool calls, and token usage) to
Logfire; without it telemetry is simply off.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator, Mapping
from pathlib import Path
from typing import Literal, cast

import anyio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from pydantic_ai.realtime import (
    AudioDelta,
    InputTranscript,
    RealtimeModel,
    RealtimeSessionEvent,
    SpeechStarted,
    ToolCallCompleted,
    ToolCallStarted,
    Transcript,
    TurnComplete,
)
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, SemanticVAD, ServerVAD

from .supervisor import create_finance_supervisor
from .voice import BACKGROUND_TOOLS, VoiceDeps, create_voice_front

load_dotenv()

SUPERVISOR_MODEL = os.environ.get('FINANCE_SUPERVISOR_MODEL', 'openai:gpt-4o-mini')
REALTIME_MODEL = os.environ.get('FINANCE_REALTIME_MODEL', 'gpt-realtime')
VOICE = os.environ.get('FINANCE_REALTIME_VOICE', 'alloy')
GEMINI_VOICE = os.environ.get('FINANCE_GEMINI_VOICE', 'Puck')
# Use Vertex AI (ADC) instead of a Gemini API key when `GOOGLE_GENAI_USE_VERTEXAI` is truthy.
USE_VERTEX = os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').lower() in (
    '1',
    'true',
    'yes',
)
# On Vertex the only widely-available Live model is the native-audio one, so default to it there;
# the Developer API exposes `gemini-live-2.5-flash`. (Override with `FINANCE_GEMINI_MODEL`.)
GEMINI_MODEL = os.environ.get('FINANCE_GEMINI_MODEL') or (
    'gemini-live-2.5-flash-native-audio' if USE_VERTEX else 'gemini-live-2.5-flash'
)
_INDEX_PATH = Path(__file__).parent / 'index.html'


def _truthy(value: str | None) -> bool:
    return (value or '').lower() in ('1', 'true', 'yes', 'on')


def _opt_float(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _opt_int(value: str | None) -> int | None:
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _build_openai_model(params: Mapping[str, str]) -> RealtimeModel:
    """Build an OpenAI realtime model from the UI's settings."""
    vad = params.get('vad', 'server')
    turn_detection: ServerVAD | SemanticVAD | None
    if vad == 'off':  # push-to-talk
        turn_detection = None
    elif vad == 'semantic':
        turn_detection = SemanticVAD(
            eagerness=cast(
                "Literal['low', 'medium', 'high', 'auto']",
                params.get('eagerness', 'auto'),
            )
        )
    else:
        turn_detection = ServerVAD(
            silence_duration_ms=_opt_int(params.get('silence_ms'))
        )
    noise = params.get('noise')
    return OpenAIRealtimeModel(
        params.get('model') or REALTIME_MODEL,
        voice=params.get('voice') or VOICE,
        turn_detection=turn_detection,
        input_noise_reduction=noise if noise in ('near_field', 'far_field') else None,
        output_modalities=(
            cast("Literal['audio', 'text']", params.get('modality', 'audio')),
        ),
        output_speed=_opt_float(params.get('speed')),
    )


def _build_gemini_model(params: Mapping[str, str]) -> RealtimeModel:
    """Build a Gemini realtime model from the UI's settings.

    Gemini needs `GOOGLE_API_KEY`, or set `GOOGLE_GENAI_USE_VERTEXAI=true` (+ `GOOGLE_CLOUD_PROJECT` /
    `GOOGLE_CLOUD_LOCATION` and `gcloud auth application-default login`) to use Vertex AI instead.
    """
    from pydantic_ai.realtime.google import AutomaticVAD, GoogleRealtimeModel

    start, end = params.get('start_sensitivity'), params.get('end_sensitivity')
    vad = None
    if start in ('high', 'low') or end in ('high', 'low'):
        vad = AutomaticVAD(
            start_sensitivity=start if start in ('high', 'low') else None,
            end_sensitivity=end if end in ('high', 'low') else None,
        )
    coverage = params.get('turn_coverage')
    return GoogleRealtimeModel(
        params.get('model') or GEMINI_MODEL,
        voice=params.get('voice') or GEMINI_VOICE,
        language_code=params.get('language') or None,
        response_modality=cast(
            "Literal['audio', 'text']", params.get('modality', 'audio')
        ),
        proactive_audio=_truthy(params.get('proactive')),
        affective_dialog=_truthy(params.get('affective')),
        turn_coverage=coverage
        if coverage in ('activity_only', 'all_input', 'all_video')
        else None,
        vad=vad,
        vertexai=USE_VERTEX,
        project=os.environ.get('GOOGLE_CLOUD_PROJECT'),
        location=os.environ.get('GOOGLE_CLOUD_LOCATION'),
    )


def _build_realtime_model(params: Mapping[str, str]) -> RealtimeModel:
    """Build the realtime voice model from the UI's settings (provider + voice + tuning)."""
    if params.get('provider') == 'gemini':
        return _build_gemini_model(params)
    return _build_openai_model(params)


def _configure_logfire() -> None:
    """Send traces to Logfire when `LOGFIRE_WRITE_TOKEN` is set.

    `instrument_pydantic_ai()` turns on instrumentation for every agent, so the realtime session
    shows up in the dashboard as a span tree: the session, each `execute_tool` call (including the
    delegated supervisor run), and cumulative token usage. No-op when the token is absent.
    """
    token = os.environ.get('LOGFIRE_WRITE_TOKEN')
    if not token:
        return
    try:
        import logfire
    except ImportError:  # pragma: no cover
        print(
            'LOGFIRE_WRITE_TOKEN is set but `logfire` is not installed; skipping telemetry (`pip install logfire`).'
        )
        return
    logfire.configure(token=token, service_name='realtime-finance')
    logfire.instrument_pydantic_ai()


_configure_logfire()

app = FastAPI()


@app.get('/')
async def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_PATH.read_text())


def _json_messages(
    event: RealtimeSessionEvent, deps: VoiceDeps
) -> list[dict[str, object]]:
    """Translate a session event into JSON messages for the browser (audio is handled separately)."""
    if isinstance(event, SpeechStarted):
        return [{'type': 'speech_started'}]
    if isinstance(event, InputTranscript) and event.is_final and event.text:
        return [{'type': 'user', 'text': event.text}]
    if isinstance(event, Transcript) and event.is_final and event.text:
        return [{'type': 'assistant', 'text': event.text}]
    if isinstance(event, ToolCallStarted):
        mode = 'async' if event.tool_name in BACKGROUND_TOOLS else 'sync'
        return [
            {
                'type': 'tool',
                'id': event.tool_call_id,
                'tool': event.tool_name,
                'mode': mode,
            }
        ]
    if isinstance(event, ToolCallCompleted):
        messages: list[dict[str, object]] = [
            {'type': 'widget', 'id': event.tool_call_id, 'widget': w.model_dump()}
            for w in deps.widgets.pop(event.tool_call_id, [])
        ]
        messages.append({'type': 'tool_done', 'id': event.tool_call_id})
        return messages
    if isinstance(event, TurnComplete):
        return [{'type': 'turn_complete'}]
    return []


@app.websocket('/ws')
async def ws(socket: WebSocket) -> None:
    await socket.accept()

    supervisor = create_finance_supervisor(SUPERVISOR_MODEL)
    voice = create_voice_front()
    deps = VoiceDeps(supervisor=supervisor)

    async with voice.realtime_session(
        model=_build_realtime_model(socket.query_params),
        deps=deps,
        background_tools=BACKGROUND_TOOLS,
    ) as session:
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
                        for message in _json_messages(event, deps):
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
                            await session.send_audio(chunk)
                        elif (text := message.get('text')) is not None:
                            try:
                                question = (
                                    json.loads(text).get('question') or ''
                                ).strip()
                            except (ValueError, AttributeError):
                                continue
                            if question:
                                await session.send_text(question)
                    tg.cancel_scope.cancel()
                except WebSocketDisconnect:
                    tg.cancel_scope.cancel()

            tg.start_soon(pump_events)
            tg.start_soon(pump_inbound)


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)
