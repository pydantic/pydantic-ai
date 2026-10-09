"""Deprecated write-back of messages appended to `request_context.messages` in `before_model_request`."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Self

from ._warnings import PydanticAIDeprecationWarning
from .messages import ModelMessage

if TYPE_CHECKING:
    from .models import ModelRequestContext

_WARNING = (
    'Appending to `request_context.messages` in `before_model_request` to add messages to the history is deprecated. '
    'Use `request_context.messages = [*request_context.messages, msg]` plus `ctx.messages.append(msg)` instead.'
)


class _History:
    """The history that every mirroring list in one before-chain adds to, until the chain finishes."""

    def __init__(self, messages: list[ModelMessage]):
        self.messages: list[ModelMessage] | None = messages


class HistoryMirroringMessages(list[ModelMessage]):
    """The `ModelRequestContext.messages` list that `before_model_request` hooks receive.

    Before hooks used to have their final request list written back to the message history, so code
    that appended to `request_context.messages` also changed history. Messages appended here with
    `append`, `extend` or `+=` are still added to the end of history, with a deprecation warning, until
    the before-chain finishes and `detach()` turns every list of the chain into a plain request-only list.
    """

    def __init__(self, messages: Iterable[ModelMessage], history: list[ModelMessage] | _History):
        super().__init__(messages)
        self._history = history if isinstance(history, _History) else _History(history)

    def detach(self) -> None:
        self._history.messages = None

    def _add_to_history(self, messages: list[ModelMessage]) -> None:
        if self._history.messages is not None and messages:
            # stacklevel points at the hook's `append`/`extend`/`+=` call.
            warnings.warn(_WARNING, PydanticAIDeprecationWarning, stacklevel=3)
            self._history.messages.extend(messages)

    def append(self, message: ModelMessage) -> None:
        super().append(message)
        self._add_to_history([message])

    def extend(self, messages: Iterable[ModelMessage]) -> None:
        messages = list(messages)
        super().extend(messages)
        self._add_to_history(messages)

    def __iadd__(self, messages: Iterable[ModelMessage]) -> Self:
        messages = list(messages)
        super().extend(messages)
        self._add_to_history(messages)
        return self


def keep_mirroring(previous: list[ModelMessage], request_context: ModelRequestContext) -> None:
    """After a hook replaces the mirroring list, wrap the new one so later hooks' appends still reach history."""
    if isinstance(previous, HistoryMirroringMessages) and not isinstance(
        request_context.messages, HistoryMirroringMessages
    ):
        request_context.messages = HistoryMirroringMessages(request_context.messages, previous._history)  # pyright: ignore[reportPrivateUsage]
