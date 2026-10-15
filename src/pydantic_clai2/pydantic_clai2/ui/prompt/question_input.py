"""Paste-aware input for inline questions while the main editor is suspended."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty, Queue
from typing import TypeAlias

from prompt_toolkit.input import create_input

from pydantic_clai2.ui.menus.menu_worker import worker_stopping
from pydantic_clai2.ui.prompt.prompt_keys import PromptKeys
from pydantic_clai2.ui.prompt.prompt_surface import TRANSCRIPT_KEYS


@dataclass(frozen=True, kw_only=True)
class Paste:
    """Literal text, never picker shortcuts or a submit key."""

    text: str


@dataclass(frozen=True, kw_only=True)
class TranscriptKey:
    """A page key or mouse report that scrolls or selects the transcript above the question, never the answer."""

    key: str
    data: str = ''


QuestionKey: TypeAlias = str | Paste | TranscriptKey
"""What the question reader yields: a key name, pasted text, or transcript input."""


@contextmanager
def question_input() -> Generator[Callable[[], QuestionKey]]:
    """Attach on the event loop; let the joined menu worker consume decoded keys."""
    pending: Queue[QuestionKey] = Queue()
    source = create_input()

    def feed(key: str, data: str) -> None:
        # Inline questions keep Ctrl-J as confirmation, unlike the multiline editor.
        if key == 'ctrl-j':
            key = 'enter'
        if key in TRANSCRIPT_KEYS:
            pending.put(TranscriptKey(key=key, data=data))
        else:
            pending.put(Paste(text=data) if key == 'paste' else key)

    def read() -> QuestionKey:
        if worker_stopping():
            return 'ctrl-c'
        try:
            return pending.get(timeout=0.05)
        except Empty:
            return ''

    keys = PromptKeys(source=source, feed=feed, eof=lambda: pending.put('ctrl-c'))
    try:
        keys.start()
        yield read
    finally:
        keys.stop()
        source.close()
