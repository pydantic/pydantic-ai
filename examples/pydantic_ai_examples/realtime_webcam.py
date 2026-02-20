"""Realtime webcam + voice assistant - talk to Roberto about what your camera sees.

Opens a webcam preview, captures microphone audio, and streams both to
Gemini Live. Roberto sees your camera feed and hears your voice, then
speaks back through your speakers. He can also browse the web for you
via Playwright. Transcripts are shown in the terminal.

Requires:
    - opencv-python: ``pip install opencv-python``
    - sounddevice: ``pip install sounddevice``
    - Playwright MCP: ``npx @playwright/mcp@latest``
    - Vertex AI auth: ``gcloud auth application-default login``

Set your project::

    export GOOGLE_CLOUD_PROJECT=your-project-id

Run with::

    uv run --no-sync -m pydantic_ai_examples.realtime_webcam
"""

from __future__ import annotations as _annotations

import os
import queue
import sys
import threading
from typing import Any

import anyio
import anyio.abc
import anyio.to_thread
import logfire
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.realtime import (
    AudioDelta,
    AudioInput,
    ImageInput,
    InputTranscript,
    RealtimeSession,
    Transcript,
)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

console = Console()

INPUT_SAMPLE_RATE = 16_000  # Gemini expects 16kHz mono PCM16
OUTPUT_SAMPLE_RATE = 24_000  # Gemini outputs 24kHz mono PCM16

playwright_server = MCPServerStdio('npx', args=['@playwright/mcp@latest'])

agent: Agent[None, str] = Agent(
    instructions=(
        'Your name is Roberto. You are our favorite bot on Slack - you help us '
        'by telling us when new customers become paying customers. '
        'You are looking at a live webcam feed and listening to the user. '
        'Answer based on what you see and hear. Be concise and conversational. '
        'You can browse the web using Playwright tools ONLY when the user '
        'explicitly asks you to open a website or search for something. '
        'If a tool call fails, tell the user and move on - never retry.'
    ),
    toolsets=[playwright_server],
)

# Thread-safe queue: main thread produces JPEG frames, async thread consumes them
frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=2)


async def _play_responses(session: RealtimeSession, tg: anyio.abc.TaskGroup) -> None:
    """Play model audio through speakers and show transcripts."""
    out = sd.RawOutputStream(samplerate=OUTPUT_SAMPLE_RATE, channels=1, dtype='int16')
    out.start()
    try:
        async for event in session:
            if isinstance(event, AudioDelta):
                out.write(event.data)
            elif isinstance(event, Transcript) and event.is_final:
                console.print(f'  [magenta]Roberto:[/magenta] {event.text}')
            elif isinstance(event, InputTranscript) and event.is_final:
                console.print(f'  [blue]You:[/blue] {event.text}')
    finally:
        out.stop()
        out.close()
        tg.cancel_scope.cancel()


async def _run_session(model: Any, stop: threading.Event) -> None:
    """Run the realtime session - sends frames + mic audio, plays responses."""

    async def bridge_stop(tg: anyio.abc.TaskGroup) -> None:
        while not stop.is_set():
            await anyio.sleep(0.2)
        tg.cancel_scope.cancel()

    async def send_frames(session: RealtimeSession) -> None:
        while True:
            try:
                jpeg = await anyio.to_thread.run_sync(
                    lambda: frame_queue.get(True, 0.5)
                )
            except queue.Empty:
                continue
            await session.send(ImageInput(data=jpeg, mime_type='image/jpeg'))
            await anyio.sleep(1.0)

    async def send_mic(session: RealtimeSession) -> None:
        audio_q: queue.Queue[bytes] = queue.Queue()

        def callback(data: Any, *_args: Any) -> None:
            audio_q.put_nowait(bytes(data))

        with sd.RawInputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=1,
            dtype='int16',
            blocksize=4000,
            callback=callback,
        ):
            while True:
                try:
                    chunk = await anyio.to_thread.run_sync(
                        lambda: audio_q.get(True, 0.5)
                    )
                except queue.Empty:
                    continue
                await session.send(AudioInput(data=chunk))

    try:
        async with agent.realtime_session(model=model) as session:
            async with anyio.create_task_group() as tg:
                tg.start_soon(bridge_stop, tg)
                tg.start_soon(send_frames, session)
                tg.start_soon(send_mic, session)
                tg.start_soon(_play_responses, session, tg)
    except Exception as e:
        console.print(f'[red]Session error: {e}[/red]')
    finally:
        stop.set()


def main() -> None:
    if cv2 is None:
        print('This example requires opencv-python: pip install opencv-python')
        sys.exit(1)
    if sd is None:
        print('This example requires sounddevice: pip install sounddevice')
        sys.exit(1)

    from pydantic_ai.realtime.gemini import GeminiRealtimeModel

    project = os.environ.get('GOOGLE_CLOUD_PROJECT')
    if not project:
        console.print('[red]Set GOOGLE_CLOUD_PROJECT to your GCP project ID.[/red]')
        return

    model = GeminiRealtimeModel(
        model='gemini-live-2.5-flash-native-audio',
        project=project,
        location=os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1'),
        voice='Puck',
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        console.print('[red]Could not open webcam[/red]')
        return

    console.print(
        f'[bold green]Roberto - Webcam + Voice Assistant[/bold green]\n'
        f'Model: [cyan]{model.model_name}[/cyan] (voice: Puck)\n'
        f'Speak and show things to your camera. Press q or Ctrl+C to exit.\n'
    )

    stop = threading.Event()
    ready = threading.Event()

    def run_in_thread() -> None:
        async def go() -> None:
            ready.set()
            await _run_session(model, stop)

        anyio.run(go)

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    ready.wait()

    # Main thread: OpenCV display loop (required on macOS)
    try:
        while not stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Roberto - press q to quit', frame)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok and not frame_queue.full():
                frame_queue.put_nowait(buf.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        cap.release()
        cv2.destroyAllWindows()
        thread.join(timeout=3)
        console.print('\n[dim]Session closed.[/dim]')


if __name__ == '__main__':
    main()
