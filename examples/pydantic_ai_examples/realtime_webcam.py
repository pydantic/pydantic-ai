"""Realtime webcam + voice assistant - talk to Gemini about what your camera sees.

Opens a webcam preview, captures microphone audio, and streams both to
Gemini Live. The model sees your camera feed and hears your voice, then
speaks back through your speakers. Transcripts are shown in the terminal.

Requires:
    - opencv-python: ``pip install opencv-python``
    - sounddevice: ``pip install sounddevice``
    - Vertex AI auth: ``gcloud auth application-default login``

Set your project:

    export GOOGLE_CLOUD_PROJECT=your-project-id

Run with:

    uv run --no-sync -m pydantic_ai_examples.realtime_webcam
"""

from __future__ import annotations as _annotations

import asyncio
import io
import os
import sys

import logfire
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.realtime import (
    AudioDelta,
    AudioInput,
    ImageInput,
    InputTranscript,
    RealtimeSession,
    Transcript,
    TurnComplete,
)

try:
    import cv2
except ImportError:
    print('This example requires opencv-python: pip install opencv-python')
    sys.exit(1)

try:
    import sounddevice as sd  # type: ignore[import-untyped]
except ImportError:
    print('This example requires sounddevice: pip install sounddevice')
    sys.exit(1)

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

console = Console()

INPUT_SAMPLE_RATE = 16_000  # Gemini expects 16kHz mono PCM16
OUTPUT_SAMPLE_RATE = 24_000  # Gemini outputs 24kHz mono PCM16
CHANNELS = 1

agent: Agent[None, str] = Agent(
    instructions=(
        'You are a visual and voice assistant looking at a live webcam feed. '
        'The user is talking to you. Answer based on what you see and hear. '
        'Be concise and conversational.'
    ),
)


# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------


def _capture_and_show(cap: cv2.VideoCapture) -> bytes | None:  # type: ignore[type-arg]
    """Capture a frame, show it in a window, and return JPEG bytes."""
    ret, frame = cap.read()
    if not ret:
        return None
    cv2.imshow('Webcam - press q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return None
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return io.BytesIO(buf.tobytes()).getvalue()


async def _send_frames(cap: cv2.VideoCapture, session: RealtimeSession, stop: asyncio.Event) -> None:  # type: ignore[type-arg]
    """Send webcam frames at ~1 FPS."""
    while not stop.is_set():
        jpeg = await asyncio.to_thread(_capture_and_show, cap)
        if jpeg is None:
            stop.set()
            break
        await session.send(ImageInput(data=jpeg, mime_type='image/jpeg'))
        await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# Microphone
# ---------------------------------------------------------------------------


async def _send_audio(session: RealtimeSession, stop: asyncio.Event) -> None:
    """Stream microphone audio to the session."""
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _audio_callback(indata: bytes, frames: int, time_info: object, status: object) -> None:
        loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

    with sd.RawInputStream(
        samplerate=INPUT_SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',
        blocksize=4000,  # 250ms chunks
        callback=_audio_callback,
    ):
        while not stop.is_set():
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
            except TimeoutError:
                continue
            await session.send(AudioInput(data=chunk))


# ---------------------------------------------------------------------------
# Speaker
# ---------------------------------------------------------------------------


async def _play_audio(session: RealtimeSession, stop: asyncio.Event) -> None:
    """Play model audio responses through speakers and show transcripts."""
    stream = sd.RawOutputStream(
        samplerate=OUTPUT_SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16',
    )
    stream.start()

    try:
        async for event in session:
            if stop.is_set():
                break

            if isinstance(event, AudioDelta):
                stream.write(event.data)

            elif isinstance(event, Transcript):
                if event.is_final:
                    console.print(f'  [magenta]Model:[/magenta] {event.text}')

            elif isinstance(event, InputTranscript):
                if event.is_final:
                    console.print(f'  [blue]You:[/blue] {event.text}')

            elif isinstance(event, TurnComplete):
                pass
    finally:
        stream.stop()
        stream.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    from pydantic_ai.realtime.gemini import GeminiRealtimeModel

    project = os.environ.get('GOOGLE_CLOUD_PROJECT')
    if not project:
        console.print('[red]Set GOOGLE_CLOUD_PROJECT to your GCP project ID.[/red]')
        return

    location = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
    model = GeminiRealtimeModel(
        model='gemini-live-2.5-flash-native-audio',
        project=project,
        location=location,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        console.print('[red]Could not open webcam[/red]')
        return

    console.print(
        f'[bold green]Realtime Webcam + Voice Assistant[/bold green]\n'
        f'Model: [cyan]{model.model_name}[/cyan]\n\n'
        f'Speak to the assistant and show things to your camera.\n'
        f'Press [bold]q[/bold] in the webcam window or Ctrl+C to exit.\n'
    )

    stop = asyncio.Event()

    try:
        async with agent.realtime_session(model=model) as session:
            tasks = [
                asyncio.create_task(_send_frames(cap, session, stop)),
                asyncio.create_task(_send_audio(session, stop)),
                asyncio.create_task(_play_audio(session, stop)),
            ]

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            stop.set()
            for t in pending:
                t.cancel()
            for t in done:
                if not t.cancelled() and t.exception():
                    raise t.exception()  # type: ignore[misc]
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        console.print('\n[dim]Session closed.[/dim]')


if __name__ == '__main__':
    asyncio.run(main())
