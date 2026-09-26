"""Paste-aware input for inline questions while the main editor is suspended."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Empty, Queue

from prompt_toolkit.input import create_input

from .menu_worker import worker_stopping
from .prompt_keys import PromptKeys


@dataclass(frozen=True, kw_only=True)
class Paste:
    """Literal text, never picker shortcuts or a submit key."""

    text: str


@contextmanager
def question_input() -> Generator[Callable[[], str | Paste]]:
    """Attach on the event loop; let the joined menu worker consume decoded keys."""
    pending: Queue[str | Paste] = Queue()
    source = create_input()

    def feed(key: str, data: str) -> None:
        pending.put(Paste(text=data) if key == 'paste' else key)

    def read() -> str | Paste:
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
