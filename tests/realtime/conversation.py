"""Scripted multi-turn voice conversations for realtime tests.

A real voice app opens the microphone once and streams it for the whole call, silence between
utterances included. `speak_continuously` does the same with speech checked in under `tests/assets/`,
and `assert_conversation_invariants` checks what the history of any such conversation must satisfy,
whatever the provider.
"""

from __future__ import annotations as _annotations

from array import array
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import anyio

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    SpeechPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.realtime import RealtimeSession

FRAME_SECONDS = 0.1
"""How much audio each microphone frame carries, and how often one is sent when paced."""

_ASSET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Utterance:
    """One thing the user says: speech checked in as `tests/assets/<asset>_16khz.pcm`."""

    asset: str
    keyword: str
    """A word the transcript of this utterance must contain, which ties each user turn in history to it."""


def load_utterance(assets_path: Path, utterance: Utterance, sample_rate: int) -> bytes:
    """Load an utterance as mono PCM16 at `sample_rate`, linearly resampled from the 16 kHz asset."""
    pcm = assets_path.joinpath(f'{utterance.asset}_{_ASSET_SAMPLE_RATE // 1000}khz.pcm').read_bytes()
    if sample_rate == _ASSET_SAMPLE_RATE:
        return pcm
    samples = array('h', pcm)
    resampled = array('h')
    step = _ASSET_SAMPLE_RATE / sample_rate
    for index in range(int(len(samples) / step)):
        position = index * step
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        resampled.append(round(samples[left] + (samples[right] - samples[left]) * (position - left)))
    return resampled.tobytes()


async def speak_continuously(
    session: RealtimeSession,
    utterances: Sequence[bytes],
    *,
    sample_rate: int,
    silence_after: float,
    before_send: Callable[[], Awaitable[None]],
    pace: bool,
) -> None:
    """Stream each utterance followed by `silence_after` seconds of silence, without ever pausing the microphone.

    `before_send` runs ahead of every frame: pass the cassette's `before_audio_send`, so replay keeps
    each frame in its recorded place. `pace` sends in real time, as a live microphone does; pass it
    while recording (see the `realtime_recording` fixture).
    """
    frame_bytes = int(sample_rate * FRAME_SECONDS) * 2
    silence = bytes(frame_bytes)
    silent_frames = round(silence_after / FRAME_SECONDS)
    for pcm in utterances:
        frames = [pcm[start : start + frame_bytes] for start in range(0, len(pcm), frame_bytes)]
        for frame in [*frames, *[silence] * silent_frames]:
            await before_send()
            await session.send_audio(frame)
            if pace:  # pragma: no branch
                await anyio.sleep(FRAME_SECONDS)  # pragma: no cover  # only while recording


def assert_conversation_invariants(session: RealtimeSession, keywords: Sequence[str | None]) -> None:
    """Check the history of a finished scripted conversation against what every provider must produce.

    `keywords` holds a word from each utterance, in the order they were spoken, or `None` for an
    utterance spoken with input transcription off.

    - one user request per utterance, in speaking order, each with its whole transcript (never a lone
      fragment, never blank), which is what containing its keyword checks; an untranscribed utterance
      has no transcript at all, so the check there is that no stray (silent) user turn was recorded;
    - requests and responses strictly alternate, from the first user turn to the final answer;
    - every tool call is answered by the request right after its response;
    - `session.usage.requests` counts every recorded `ModelResponse`;
    - the history survives a round trip through `ModelMessagesTypeAdapter`.
    """
    messages = session.all_messages()
    shape = _describe(messages)

    assert [message.kind for message in messages] == ['request', 'response'] * (len(messages) // 2), shape
    assert messages and len(messages) % 2 == 0, shape

    user_turns = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SpeechPart) and part.speaker == 'user'
    ]
    assert len(user_turns) == len(keywords), shape
    for turn, keyword in zip(user_turns, keywords):
        if keyword is None:
            assert turn.transcript is None, shape
        else:
            assert turn.transcript and keyword in turn.transcript.lower(), shape

    for index, message in enumerate(messages):
        calls = [part.tool_call_id for part in message.parts if isinstance(part, ToolCallPart)]
        if calls:
            answer = messages[index + 1]
            returns = [part.tool_call_id for part in answer.parts if isinstance(part, ToolReturnPart)]
            assert sorted(returns) == sorted(calls), shape

    assert session.usage.requests == sum(isinstance(message, ModelResponse) for message in messages), shape
    assert ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages)) == messages


def _describe(messages: Sequence[ModelMessage]) -> str:
    """A one-line-per-message view of a history, so a failed invariant shows the whole conversation."""
    lines: list[str] = []
    for message in messages:
        parts: list[str] = []
        for part in message.parts:
            if isinstance(part, SpeechPart):
                parts.append(f'{part.speaker}: {part.transcript!r}')
            else:
                parts.append(part.part_kind)
        lines.append(f'{type(message).__name__}({", ".join(parts)})')
    return '\n'.join(lines)
