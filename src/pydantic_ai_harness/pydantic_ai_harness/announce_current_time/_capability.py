"""`AnnounceCurrentTime`: tell the model the current date, or the current hour, in a time zone you pick."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, tzinfo
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import InstructionPart, ModelRequest
from pydantic_ai.tools import AgentDepsT, RunContext

_ID = 'current_time'
_WEEKDAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
"""Spelled out because `%A` follows the process locale."""


@dataclass(kw_only=True)
class AnnounceCurrentTime(AbstractCapability[AgentDepsT]):
    """Tell the model the current date, or the current hour, in a time zone you pick.

    The time is the timestamp of the request being sent, so the capability reads no clock of its own. When it
    moves on during a conversation, the new value is appended to the message history rather than rewriting the
    instructions, which keeps the cached prompt prefix.
    """

    id: str | None = _ID
    """Keys the instruction block, which appending a changed value to the history needs.

    `None` or `''` means the default, `current_time`.
    """
    timezone: str | tzinfo = UTC
    """An IANA time zone name such as `'Europe/Amsterdam'`, or a `tzinfo`.

    A name is looked up in the system time zone database, or the `tzdata` package where there is none (Windows,
    minimal containers). A `tzinfo` such as `datetime.UTC` needs neither.
    """
    precision: Literal['date', 'hour'] = 'date'
    """`'date'` states the weekday and date. `'hour'` also states the hour, adding an update to the history every
    hour of a conversation."""
    _zone: tzinfo = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.id = self.id or _ID
        # Resolved once here, as reading tzdata on a request would block the event loop.
        self._zone = ZoneInfo(self.timezone) if isinstance(self.timezone, str) else self.timezone

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], InstructionPart | None]:
        """Return the current date or hour, read from the request being sent."""
        return self._current_time

    def _current_time(self, ctx: RunContext[AgentDepsT]) -> InstructionPart | None:
        stamp = next((m.timestamp for m in reversed(ctx.messages) if isinstance(m, ModelRequest) and m.timestamp), None)
        if stamp is None:
            return None
        now = stamp.astimezone(self._zone)
        hour = now.replace(minute=0, second=0, microsecond=0).isoformat(timespec='minutes')
        offset = hour[16:]  # The `+HH:MM` after `YYYY-MM-DDTHH:MM`.
        key = self._zone.key if isinstance(self._zone, ZoneInfo) else None
        name = key or now.tzname() or f'UTC{offset}'
        weekday = _WEEKDAYS[now.weekday()]
        if self.precision == 'date':
            text = f"Today's date is {weekday}, {now.date().isoformat()} ({name})."
        else:
            text = f'The current time is within the hour starting {hour} ({weekday}, {name}).'
        return InstructionPart(text, on_change='append')
