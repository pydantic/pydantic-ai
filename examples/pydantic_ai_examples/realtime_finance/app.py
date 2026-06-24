"""FastAPI backend for the realtime finance voice demo.

A true speech-to-speech demo: the browser streams microphone audio (PCM16, 24kHz) into an OpenAI
Realtime session and plays the model's generated audio back. When the user asks about their money the
realtime model delegates to a finance supervisor agent (a normal text model), whose tools return
typed widgets rendered as cards in the chat.

Requires `OPENAI_API_KEY` (used for both the realtime voice and the supervisor). Put it in a `.env`
at the repo root, then:

    uv run --all-packages uvicorn pydantic_ai_examples.realtime_finance.app:app

`FINANCE_SUPERVISOR_MODEL` (default `openai:gpt-4o-mini`), `FINANCE_REALTIME_MODEL` (default
`gpt-realtime`) and `FINANCE_REALTIME_VOICE` (default `alloy`) can override the defaults.

Set `LOGFIRE_WRITE_TOKEN` to stream traces (the realtime session, tool calls, and token usage) to
Logfire; without it telemetry is simply off.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import cast

import anyio
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from pydantic_ai.realtime import (
    AudioDelta,
    InputTranscript,
    RealtimeSessionEvent,
    SpeechStarted,
    ToolCallCompleted,
    ToolCallStarted,
    Transcript,
    TurnComplete,
)
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

from .supervisor import create_finance_supervisor
from .voice import BACKGROUND_TOOLS, VoiceDeps, create_voice_front

load_dotenv()

SUPERVISOR_MODEL = os.environ.get('FINANCE_SUPERVISOR_MODEL', 'openai:gpt-4o-mini')
REALTIME_MODEL = os.environ.get('FINANCE_REALTIME_MODEL', 'gpt-realtime')
VOICE = os.environ.get('FINANCE_REALTIME_VOICE', 'alloy')
_INDEX_PATH = Path(__file__).parent / 'index.html'


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
        model=OpenAIRealtimeModel(REALTIME_MODEL, voice=VOICE),
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
