"""Pydantic AI's process-wide agent instrumentation, the setting `Agent.instrument_all` writes.

Core offers a setter and no getter, so this reads the class default the setter assigns. The
`observability` plugin sets it so agents CLAI does not run itself (summaries, sub-agents) are traced
too, and `/compact` reads its tracer for the compaction span.
"""

from opentelemetry.trace import Tracer

from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings


def current() -> InstrumentationSettings | bool:
    """What agents without instrumentation of their own use, as `Agent.instrument_all` last set it."""
    return Agent._instrument_default  # pyright: ignore[reportPrivateUsage]


def tracer() -> Tracer | None:
    """The tracer those agents trace with, or `None` while instrumentation is off."""
    instrument = current()
    if instrument is False:
        return None
    return (instrument if isinstance(instrument, InstrumentationSettings) else InstrumentationSettings()).tracer
