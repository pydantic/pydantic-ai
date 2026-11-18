"""Events emitted by context usage reporting."""

from dataclasses import dataclass

from pydantic_ai import CapabilityEvent

REPORT_CONTEXT_USAGE_EVENTS = 'report_context_usage'


@dataclass(kw_only=True)
class ContextUsageMeasuredEvent(CapabilityEvent, namespace=REPORT_CONTEXT_USAGE_EVENTS, name='measured'):
    """A context usage reading was recorded before a model request."""

    used_tokens: int
    """Estimated tokens in the message history about to be sent.

    Counted by `estimate_context_tokens`: the provider-reported usage of the most recent model
    response (tool schemas included) plus an estimate of messages and newly revealed tool schemas
    added since. With no reported usage, message text falls back to the character heuristic;
    schemas named by availability deltas are still estimated from the pending request
    parameters, but other tool schemas remain outside that heuristic.
    """

    window_tokens: int
    """Context window the reading is measured against."""

    resolved: bool
    """Whether `window_tokens` is the model's real window or the fallback.

    A gauge can render an unresolved window differently -- the percentage is a guess when
    the model is not in the pricing registry.
    """

    fraction: float
    """`used_tokens` as a fraction of the window."""
