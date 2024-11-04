from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from . import messages

__all__ = 'ResultData', 'Cost', 'RunResult'

ResultData = TypeVar('ResultData')


@dataclass
class Cost:
    """Cost of a request or run."""

    request_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    details: dict[str, int] | None = None

    def __add__(self, other: Cost) -> Cost:
        counts: dict[str, int] = {}
        for field in 'request_tokens', 'response_tokens', 'total_tokens':
            self_value = getattr(self, field)
            other_value = getattr(other, field)
            if self_value is not None or other_value is not None:
                counts[field] = (self_value or 0) + (other_value or 0)

        details = self.details.copy() if self.details is not None else None
        if other.details is not None:
            details = details or {}
            for key, value in other.details.items():
                details[key] = details.get(key, 0) + value

        return Cost(**counts, details=details or None)


@dataclass
class RunResult(Generic[ResultData]):
    """Result of a run."""

    response: ResultData
    cost: Cost
    _all_messages: list[messages.Message]
    _new_message_index: int

    def all_messages(self) -> list[messages.Message]:
        """Return the history of messages."""
        # this is a method to be consistent with the other methods
        return self._all_messages

    def all_messages_json(self) -> bytes:
        """Return the history of messages as JSON bytes."""
        return messages.MessagesTypeAdapter.dump_json(self.all_messages())

    def new_messages(self) -> list[messages.Message]:
        """Return new messages associated with this run.

        System prompts and any messages from older runs are excluded.
        """
        return self.all_messages()[self._new_message_index :]

    def new_messages_json(self) -> bytes:
        """Return new messages from [new_messages][] as JSON bytes."""
        return messages.MessagesTypeAdapter.dump_json(self.new_messages())
