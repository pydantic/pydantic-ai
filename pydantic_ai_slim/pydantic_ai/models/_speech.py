from __future__ import annotations as _annotations

from ..messages import ModelResponse, SpeechPart


def assistant_speech_text(message: ModelResponse, index: int) -> str:
    """The text the assistant `SpeechPart` at `index` of `message` stands for when a model reads it back.

    That is its transcript, followed by a marker when the listener cut it off: `[Interrupted after N ms]`
    where the part records how much was heard, or `[Interrupted]` on the last speech part of an
    interrupted response that doesn't. A barge-in cuts the model off mid-sentence (and generation outruns
    playback, so the transcript can run past what was heard): without the marker, a model reads the
    transcript as an utterance the user heard in full, and may repeat itself or assume what was never said.

    The marker is written on the way to a model and never persisted: history keeps the interruption on
    [`SpeechPart.interrupted_at_ms`][pydantic_ai.messages.SpeechPart.interrupted_at_ms]. Empty when the
    part has neither a transcript nor an interruption to report.
    """
    part = message.parts[index]
    assert isinstance(part, SpeechPart)
    lines = [part.transcript] if part.transcript else []
    if part.interrupted_at_ms is not None:
        lines.append(f'[Interrupted after {part.interrupted_at_ms} ms]')
    elif message.state == 'interrupted' and index == max(
        i for i, candidate in enumerate(message.parts) if isinstance(candidate, SpeechPart)
    ):
        # The provider reported the interruption without an offset.
        lines.append('[Interrupted]')
    return '\n'.join(lines)
