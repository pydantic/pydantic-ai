"""Input queued for a session's next turn: one entry per prompt or `/command`, each editable until it is read."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from pydantic_clai2.cli.shell_passthrough import shell_command
from pydantic_clai2.commands import is_command_input

NOT_QUEUED = 'That message is no longer queued: it was sent or removed.'

InputKind = Literal['prompt', 'command', 'shell']
"""What a line entered at the prompt is: a prompt, the only kind that can steer the running turn, a `/command`, or a `!` shell line."""


def input_kind(text: str) -> InputKind:
    """Whether `text` is a `/command`, a `!` shell line, or a prompt."""
    if is_command_input(text):
        return 'command'
    return 'shell' if shell_command(text) is not None else 'prompt'


def can_steer(text: str) -> bool:
    """Whether `text` may join a running turn: `/commands` and `!` shell lines never steer."""
    return input_kind(text) == 'prompt'


@dataclass(eq=False)
class QueuedPrompt:
    """A queued prompt compared by identity, so an edit finds it even after the queue shifts.

    Its `id` names it to a browser, stable while it waits, also when it moves between the editor and the inbox.
    The terminal's editor serves whichever session it shows, so IDs are unique everywhere rather than counted.
    """

    text: str
    recorded: str = ''
    """The raw draft `accept` saved to history for this prompt, before command expansion."""
    id: str = field(default_factory=lambda: uuid4().hex)


def find_queued(entries: list[QueuedPrompt], input_id: str) -> QueuedPrompt:
    """The entry named `input_id`; `LookupError` once it ran or was removed."""
    found = next((entry for entry in entries if entry.id == input_id), None)
    if found is None:
        raise LookupError(NOT_QUEUED)
    return found


class PromptInbox:
    """Input entered while no editor reads the session's keys, as in `clai2 --web`, read in order."""

    def __init__(self) -> None:
        self._entries: deque[QueuedPrompt] = deque()
        self._ready = asyncio.Event()

    @property
    def entries(self) -> list[QueuedPrompt]:
        """What waits, oldest first."""
        return list(self._entries)

    def put(self, entry: QueuedPrompt) -> None:
        self._entries.append(entry)
        self._ready.set()

    def remove(self, entry: QueuedPrompt) -> None:
        self._entries.remove(entry)

    def take(self) -> list[QueuedPrompt]:
        """Remove and return everything waiting, as when an editor starts reading for the session."""
        taken = list(self._entries)
        self._entries.clear()
        return taken

    async def get(self) -> str:
        """The oldest entry's text, once there is one."""
        while not self._entries:
            self._ready.clear()
            await self._ready.wait()
        return self._entries.popleft().text
