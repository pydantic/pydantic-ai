"""Every message a run appends to its history, kept whatever a capability drops from that history later."""

from __future__ import annotations as _annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from .messages import ModelMessage

if TYPE_CHECKING:
    from ._run_context import RunContext


@dataclasses.dataclass
class AppendedMessages:
    """The history a run was given, followed by every message the run appended to it, in order.

    A capability can rewrite the run's message history, say to drop old messages before a request, so
    the history can lose messages this run was given or added. This keeps them, so the run's
    conversation can still hold all of them: see `AgentRunResult.conversation`.
    """

    messages: list[ModelMessage] | None = None
    """The history as it stood when the run first appended to it, then every message appended since. `None` until then."""

    new_message_index: int = 0
    """Where the messages the run added start in `messages`."""

    def start(self, history: list[ModelMessage]) -> list[ModelMessage]:
        """Record `history` as the history the run was given, unless the run already appended to it."""
        if self.messages is None:
            self.messages = list(history)
            self.new_message_index = len(history)
        return self.messages

    def append(self, history: list[ModelMessage], *messages: ModelMessage, given: bool = False) -> None:
        """Append `messages` to `history`, the run's message history, and record them.

        `given` counts them as part of the history the run was given, like the request a run resuming without
        a new prompt sends again.
        """
        recorded = self.start(history)
        history.extend(messages)
        recorded.extend(messages)
        if given:
            self.new_message_index = len(recorded)

    def replace(self, message: ModelMessage, replacement: ModelMessage | None) -> None:
        """Record that the run replaced `message` in its history with `replacement`, or removed it when that's `None`."""
        recorded = self.messages or []
        index = next((index for index, kept in enumerate(recorded) if kept is message), None)
        if index is None:
            # Not one the run appended or was given, like a copy a capability put in its place.
            return
        if replacement is not None:
            recorded[index] = replacement
        else:
            # A failed step's empty trailing request, normally one the run added. Were it one the run was given,
            # `new_message_index` would point one past them, which only has a store compare the whole history.
            del recorded[index]


def append_to_history(ctx: RunContext[Any], *messages: ModelMessage) -> None:
    """Append `messages` to the history of the run `ctx` belongs to, recording them as messages the run added."""
    ctx._appended_messages.append(ctx.messages, *messages)  # pyright: ignore[reportPrivateUsage]
