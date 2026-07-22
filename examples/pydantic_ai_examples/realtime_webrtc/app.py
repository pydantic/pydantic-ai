"""Browser voice agent over WebRTC, with a Pydantic AI sideband running the tools.

The browser exchanges audio with OpenAI **directly** over WebRTC (lowest latency), while this backend
stays the control plane: it relays the browser's SDP offer to OpenAI (so the API key never reaches the
browser), then attaches a [`realtime_session`][pydantic_ai.Agent.realtime_session] to the same call by
`call_id` and runs the agent's tools server-side. See the [realtime guide](https://ai.pydantic.dev/realtime/#browser--webrtc).

The topology:

    browser ──mic/speaker audio (WebRTC media)──▶  OpenAI Realtime
           ◀─────────────────────────────────────
       │  SDP offer (POST /offer)                    ▲ control WebSocket (call_id)
       ▼                                             │
    FastAPI backend  ──answer_webrtc_offer()──▶ OpenAI  ──realtime_session(provider_session=…)──┘
                     (relays SDP, gets call_id)        (runs tools, builds history)

Requires `OPENAI_API_KEY` (put it in a `.env` at the repo root), then:

    uv run --all-packages uvicorn pydantic_ai_examples.realtime_webrtc.app:app

Open http://localhost:8000 (localhost is a secure context, so the browser allows the microphone) and
click **Start call**. Ask "What time is it in Tokyo?" or "What's your refund policy?" to trigger a
server-side tool.

The app is instrumented with Logfire: set `LOGFIRE_TOKEN` (e.g. in the same `.env`) to see the realtime
session, model turns, and tool calls as traces; without a token nothing is sent.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import logfire
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from pydantic_ai import Agent
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent
from pydantic_ai.realtime import TurnCompleteEvent, WebRTCCall
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings

load_dotenv()

logfire.configure(send_to_logfire='if-token-present', service_name='realtime-webrtc')
logfire.instrument_pydantic_ai()

MODEL_NAME = os.getenv('WEBRTC_REALTIME_MODEL', 'gpt-realtime')
VOICE = os.getenv('WEBRTC_REALTIME_VOICE', 'marin')
INSTRUCTIONS = (
    'You are Roberto, a concise and friendly voice support assistant. '
    'Use `lookup_time` for time questions and `lookup_support_policy` for account or refund questions. '
    'Keep answers short and natural for speech.'
)

INDEX_HTML = (Path(__file__).parent / 'index.html').read_text(encoding='utf-8')

agent = Agent(instructions=INSTRUCTIONS)


@agent.tool_plain
def lookup_time(city: str) -> str:
    """Look up the current local time for a city."""
    timezones = {
        'london': 'Europe/London',
        'new york': 'America/New_York',
        'tokyo': 'Asia/Tokyo',
        'sydney': 'Australia/Sydney',
        'san francisco': 'America/Los_Angeles',
    }
    zone = timezones.get(city.lower())
    if zone is None:
        return f'I only know these example cities: {", ".join(sorted(timezones))}.'
    try:
        now = datetime.now(ZoneInfo(zone))
    except ZoneInfoNotFoundError:  # pragma: no cover - depends on the host tz database
        return f'I could not load timezone data for {city}.'
    return now.strftime(f'It is %A, %I:%M %p in {city}.')


@agent.tool_plain
def lookup_support_policy(topic: str) -> str:
    """Return a short canned support policy answer."""
    policies = {
        'refund': 'Refunds are available within 30 days for billing errors or duplicate charges.',
        'return': 'Physical returns can be started within 14 days of the delivery date.',
        'password': 'Reset your password from the sign-in page using the email verification flow.',
    }
    return policies.get(
        topic.lower(), 'I only have example policies for refund, return, and password.'
    )


def build_model() -> OpenAIRealtimeModel:
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError(
            'Set `OPENAI_API_KEY` (e.g. in a `.env` at the repo root) to run this example.'
        )
    return OpenAIRealtimeModel(
        MODEL_NAME, settings=OpenAIRealtimeModelSettings(voice=VOICE)
    )


@dataclass
class Call:
    """One live WebRTC call and its server-side sideband task."""

    answer_sdp: str
    provider_session: WebRTCCall
    task: asyncio.Task[None] | None = None
    attached: asyncio.Event = field(default_factory=asyncio.Event)


CALLS: dict[str, Call] = {}


async def run_sideband(call: Call) -> None:
    """Attach the sideband session to the WebRTC call and run the agent's tool loop over its events."""
    call_id = call.provider_session.call_id
    try:
        async with agent.realtime_session(
            model=build_model(), provider_session=call.provider_session
        ) as session:
            call.attached.set()
            async for event in session:
                if isinstance(event, FunctionToolCallEvent):
                    logfire.info(
                        'tool call', tool=event.part.tool_name, args=event.part.args
                    )
                elif isinstance(event, FunctionToolResultEvent):
                    logfire.info(
                        'tool result',
                        tool=event.part.tool_name,
                        content=event.part.content,
                    )
                elif isinstance(event, TurnCompleteEvent):
                    logfire.info('turn complete', messages=len(session.all_messages()))
    except asyncio.CancelledError:
        raise
    except Exception:
        logfire.exception('sideband session for {call_id} failed', call_id=call_id)
        call.attached.set()  # unblock `/offer` so it can surface the failure
    finally:
        CALLS.pop(call_id, None)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        for call in list(CALLS.values()):
            if call.task is not None:
                call.task.cancel()
                with suppress(asyncio.CancelledError):
                    await call.task


app = FastAPI(lifespan=lifespan)


@app.get('/')
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.post('/offer')
async def offer(request: Request) -> JSONResponse:
    """Relay the browser's SDP offer to OpenAI, start the sideband, and return the SDP answer."""
    sdp_offer = (await request.body()).decode('utf-8')
    if not sdp_offer.strip():
        raise HTTPException(
            status_code=400, detail='Expected an SDP offer in the request body.'
        )

    answer = await build_model().answer_webrtc_offer(
        sdp_offer, instructions=INSTRUCTIONS
    )
    call = Call(answer_sdp=answer.sdp, provider_session=answer.call)
    CALLS[answer.call.call_id] = call

    # Attach the sideband before returning the answer, so the tools are live before the browser (which
    # only starts sending audio once it has the answer) can speak.
    call.task = asyncio.create_task(run_sideband(call))
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(call.attached.wait(), timeout=10)

    return JSONResponse({'sdp': call.answer_sdp, 'call_id': answer.call.call_id})


@app.post('/hangup/{call_id}')
async def hangup(call_id: str) -> JSONResponse:
    call = CALLS.get(call_id)
    if call is not None and call.task is not None:
        call.task.cancel()
        with suppress(asyncio.CancelledError):
            await call.task
    return JSONResponse({'stopped': call is not None})


def main() -> None:  # pragma: no cover - manual entrypoint
    import uvicorn

    uvicorn.run(app, host='127.0.0.1', port=8000)


if __name__ == '__main__':  # pragma: no cover
    main()
