"""A moved event-loop clock, so a test makes a deadline pass exactly when it wants rather than waiting for it."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

LONG_TIMEOUT = 10
"""For a test in which something must get done before the deadline, e.g. a first chunk reaching the consumer. Where the
deadline must then pass, the test moves the `Clock` past it rather than waiting, so a loaded runner can't make the
deadline pass too soon."""


@dataclass
class Clock:
    """The event loop's clock, which a test moves forward to make a deadline pass exactly when it wants."""

    offset: float = 0

    def advance(self, seconds: float = LONG_TIMEOUT) -> None:
        self.offset += seconds

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make the running event loop's clock this one, until the test ends."""
        loop = asyncio.get_running_loop()
        real_time = loop.time
        monkeypatch.setattr(loop, 'time', lambda: real_time() + self.offset)
