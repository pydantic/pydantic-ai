"""`CurrentTime`: tell the model the current date, or the current hour, in a time zone you pick."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, tzinfo
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import InstructionPart, ModelRequest
from pydantic_ai.tools import AgentDepsT, RunContext


@dataclass(kw_only=True)
class CurrentTime(AbstractCapability[AgentDepsT]):
    """Tell the model the current date, or the current hour, in a time zone you pick.

    The time is the timestamp of the request being sent, so the capability reads no clock of its own. When it
    moves on during a conversation, the new value is appended to the message history rather than rewriting the
    instructions, which keeps the cached prompt prefix.
    """

    id: str | None = 'current_time'
    """Keys the instruction block, which appending a changed value to the history needs."""
    timezone: str | tzinfo = UTC
    """An IANA time zone name such as `'Europe/Amsterdam'`, or a `tzinfo`."""
    precision: Literal['date', 'hour'] = 'date'
    """`'date'` states the weekday and date. `'hour'` also states the hour, adding an update to the history every
    hour of a conversation."""

    def __post_init__(self) -> None:
        self._zone()

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], InstructionPart | None]:
        """Return the current date or hour, read from the request being sent."""
        return self._current_time

    def _zone(self) -> tzinfo:
        return ZoneInfo(self.timezone) if isinstance(self.timezone, str) else self.timezone

    def _current_time(self, ctx: RunContext[AgentDepsT]) -> InstructionPart | None:
        stamp = next((m.timestamp for m in reversed(ctx.messages) if isinstance(m, ModelRequest) and m.timestamp), None)
        if stamp is None:
            return None
        zone = self._zone()
        now = stamp.astimezone(zone)
        name = zone.key if isinstance(zone, ZoneInfo) else now.tzname()
        if self.precision == 'date':
            text = f"Today's date is {now:%A}, {now.date().isoformat()} ({name})."
        else:
            hour = now.replace(minute=0, second=0, microsecond=0).isoformat(timespec='minutes')
            text = f'The current time is within the hour starting {hour} ({now:%A}, {name}).'
        return InstructionPart(text, on_change='append')
