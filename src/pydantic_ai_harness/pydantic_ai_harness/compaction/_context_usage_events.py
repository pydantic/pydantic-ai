"""Events emitted by context usage reporting."""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent

REPORT_CONTEXT_USAGE_EVENTS = 'report_context_usage'


@dataclass(kw_only=True)
class ContextUsageEvent(CapabilityEvent, namespace=REPORT_CONTEXT_USAGE_EVENTS, name='measured'):
    """A context usage reading was recorded before a model request."""

    used_tokens: int
    window_tokens: int
    resolved: bool
    fraction: float
