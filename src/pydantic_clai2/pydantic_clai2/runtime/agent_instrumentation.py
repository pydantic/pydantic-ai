"""Pydantic AI's process-wide agent instrumentation, the setting `Agent.instrument_all` writes.

Core offers a setter and no getter, so this reads the class default the setter assigns. The
`observability` plugin claims it so agents CLAI does not run itself (summaries, sub-agents, session
naming) are traced too, and `/compact` reads its tracer for the compaction span.

Claims live here rather than on the plugin class: `/plugins reload` re-imports the plugin's module,
and during a turn the old instance releases only after the new one has claimed.
"""

from dataclasses import dataclass, field

from opentelemetry.trace import Tracer

from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings


@dataclass
class _Claims:
    live: list[InstrumentationSettings] = field(default_factory=list[InstrumentationSettings])
    """Claimed settings, newest last; the newest is in effect."""
    outside: InstrumentationSettings | bool = False
    """The setting from before the first live claim, restored after the last."""


_CLAIMS = _Claims()


def current() -> InstrumentationSettings | bool:
    """What agents without instrumentation of their own use, as `Agent.instrument_all` last set it."""
    return Agent._instrument_default  # pyright: ignore[reportPrivateUsage]


def tracer() -> Tracer | None:
    """The tracer those agents trace with, or `None` while instrumentation is off."""
    instrument = current()
    if instrument is False:
        return None
    return (instrument if isinstance(instrument, InstrumentationSettings) else InstrumentationSettings()).tracer


def claim(settings: InstrumentationSettings) -> None:
    """Instrument every agent with `settings` until `release`."""
    if not _CLAIMS.live:
        _CLAIMS.outside = current()
    _CLAIMS.live.append(settings)
    Agent.instrument_all(settings)


def release(settings: InstrumentationSettings) -> None:
    """Hand agents to the newest remaining claim, or back to the outside setting after the last.

    A setting someone else made since the claim is left in place. Releasing an unclaimed setting does nothing.
    """
    if not any(live is settings for live in _CLAIMS.live):
        return
    _CLAIMS.live = [live for live in _CLAIMS.live if live is not settings]
    if current() is settings:
        Agent.instrument_all(_CLAIMS.live[-1] if _CLAIMS.live else _CLAIMS.outside)
