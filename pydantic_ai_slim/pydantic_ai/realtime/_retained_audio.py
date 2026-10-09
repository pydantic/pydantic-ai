"""The budget behind `retain_audio_max_seconds`: how much audio a realtime session keeps in history."""

from __future__ import annotations

import math
from dataclasses import replace

from ..exceptions import UserError
from ..messages import BinaryContent, ModelMessage, SpeechPart


class RetainedAudioBudget:
    """Bounds the audio a realtime session retains (with `audio_retention`), the way `retain_images_max` bounds images.

    The session reports each piece of audio it builds into a `SpeechPart` with `track`. Whenever the audio it
    retains grows, it asks how far over budget that is with `excess`, counting the audio not built into a part
    yet, and frees that excess oldest first: `strip` drops the audio of recorded parts (keeping their
    transcripts), `weight` accounts for a dropped segment that is waiting for its turn, and `trim` keeps only
    the most recent audio of a turn still being spoken.

    Audio is measured in a unit both sample rates divide, so input and output audio add up exactly: a byte of
    input audio weighs the output rate and a byte of output audio the input rate, which makes a second of either
    weigh `2 * input_sample_rate * output_sample_rate`.
    """

    def __init__(self, max_seconds: float | None, *, input_sample_rate: int, output_sample_rate: int) -> None:
        if max_seconds is not None and not (math.isfinite(max_seconds) and max_seconds >= 0):
            raise UserError('`retain_audio_max_seconds` must be a finite number of at least 0, or `None` for no limit.')
        self.retains_audio = max_seconds != 0
        """Whether any audio may be retained at all: `False` for a budget of `0`."""
        self._input_byte_weight = output_sample_rate
        self._output_byte_weight = input_sample_rate
        self._max_weight = (
            None if max_seconds is None else round(max_seconds * 2 * input_sample_rate * output_sample_rate)
        )
        # The weight of each tracked `SpeechPart.audio` still retained, by the `id` of that `BinaryContent`
        # (held by the part, so the id stays its own), and their total.
        self._tracked: dict[int, int] = {}
        self._tracked_weight = 0

    @property
    def tracked_parts(self) -> int:
        """How many tracked pieces of audio are still retained: `strip` has something to free while there are any."""
        return len(self._tracked)

    def weight(self, byte_count: int, *, output: bool) -> int:
        """The weight of `byte_count` bytes of PCM16 input or output audio."""
        return byte_count * (self._output_byte_weight if output else self._input_byte_weight)

    def track(self, audio: BinaryContent, pcm_length: int, *, output: bool) -> BinaryContent:
        """Count a `SpeechPart.audio` built from `pcm_length` bytes of PCM16 against the budget, and return it."""
        if self._max_weight is not None:
            weight = self.weight(pcm_length, output=output)
            self._tracked[id(audio)] = weight
            self._tracked_weight += weight
        return audio

    def excess(self, *, untracked_input: int, untracked_output: int) -> int:
        """How far the retained audio is over budget (within it when not positive).

        `untracked_input` and `untracked_output` are the bytes retained but not built into a part yet: turns
        still being spoken, and segments waiting for their turn.
        """
        if self._max_weight is None:
            return 0
        return (
            self._tracked_weight
            + self.weight(untracked_input, output=False)
            + self.weight(untracked_output, output=True)
            - self._max_weight
        )

    def strip(self, message: ModelMessage, excess: int) -> tuple[ModelMessage, int]:
        """A copy of `message` without the audio of its tracked speech parts, until `excess` is freed, and the weight freed.

        Returns `message` itself, and `0`, when it has no tracked audio. A copy rather than an in-place edit, so a
        snapshot of history already handed out doesn't change; the caller swaps it in.
        """
        parts = list(message.parts)
        freed = 0
        for index, part in enumerate(parts):
            if (
                freed < excess
                and isinstance(part, SpeechPart)
                and part.audio is not None
                and (weight := self._tracked.pop(id(part.audio), None)) is not None
            ):
                parts[index] = replace(part, audio=None)
                freed += weight
        if not freed:
            return message, 0
        self._tracked_weight -= freed
        return replace(message, parts=parts), freed

    def trim(self, buffer: bytearray, excess: int, *, output: bool) -> int:
        """Drop the oldest audio of a live PCM16 `buffer` to free `excess`, returning the excess that remains."""
        if excess <= 0:
            return excess
        byte_weight = self._output_byte_weight if output else self._input_byte_weight
        # Whole samples (2 bytes each), rounded up so the trim covers the excess.
        drop = min(len(buffer), -(-excess // (2 * byte_weight)) * 2)
        del buffer[:drop]
        return excess - drop * byte_weight
