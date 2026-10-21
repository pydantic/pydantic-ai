"""Expressive audio tags in ElevenLabs transcripts, shared by the Eleven Realtime and ElevenLabs Agents models.

With an expressive voice, ElevenLabs' model writes tags such as `[laughs]` into its reply to direct the delivery. They
come back as part of the transcript, though they aren't spoken.
"""

from __future__ import annotations as _annotations

import re
from dataclasses import dataclass

_TAG_LETTER = r'[^\W\d_]'
_AUDIO_TAG = re.compile(rf"\[({_TAG_LETTER}(?:{_TAG_LETTER}|[ '\u2019-]){{1,49}})\](?!\()")
"""An expressive audio tag such as `[laughs]` or `[strong French accent]`: words in square brackets.

Tags are lowercase by convention, so a bracketed phrase starting with a capital (`[Your Name]`) is kept, as are
citations (`[1]`), checkboxes (`[x]`), and Markdown links (`[text](url)`).
"""
_AUDIO_TAG_START = re.compile(rf"\[(?:{_TAG_LETTER}(?:{_TAG_LETTER}|[ '\u2019-]){{0,49}}\]?)?")
"""What the rest of an audio tag could still follow: an opening bracket, part of a tag, or a whole one (which a `(`
could yet turn into a Markdown link)."""


def strip_audio_tags(text: str, *, started: bool = False) -> tuple[str, list[str]]:
    """Remove the audio tags from `text`, returning what is left and the tags removed (without brackets).

    A tag goes with the whitespace before it, or after it at the start of the text (unless the text continues
    output that has already `started`), so `Hi [laughs] there` becomes `Hi there` and `[laughs] Hi` becomes `Hi`.
    """
    tags: list[str] = []
    result = ''
    position = 0
    for match in _AUDIO_TAG.finditer(text):
        if match[1][0].isupper():
            continue
        result = (result + text[position : match.start()]).rstrip()
        tags.append(match[1])
        position = match.end()
    result += text[position:]
    return (result if started else result.lstrip()), tags


@dataclass
class AudioTagStripper:
    """Strips audio tags from streamed text so that the pieces add up to the whole text stripped at once.

    A tag can be split across pieces, so text from an opening bracket that may yet start one is held back until
    it is known, as is trailing whitespace, which a tag that follows would remove.
    """

    held: str = ''
    started: bool = False

    def feed(self, text: str) -> str:
        buffer = self.held + text
        end = buffer.rfind('[')
        if end == -1 or not _AUDIO_TAG_START.fullmatch(buffer, end):
            end = len(buffer)
        head = buffer[:end].rstrip()
        self.held = buffer[len(head) :]
        stripped, _ = strip_audio_tags(head, started=self.started)
        self.started = self.started or bool(stripped)
        return stripped
