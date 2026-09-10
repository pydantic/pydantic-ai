"""Lifecycle events for history compaction.

One event family shared by everything that compacts history: the provider-native
[`OpenAICompaction`][pydantic_ai.models.openai.OpenAICompaction] and
[`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] capabilities, and the
model-agnostic [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/compaction/) strategies.
Subscribers use one vocabulary regardless of which mechanism is doing the compacting.

The events are only emitted where the compacting code runs client-side and can observe (or decide)
the compaction. Server-side compaction — Anthropic's `context_management`, OpenAI's stateful mode —
happens inside the model response itself, surfacing as a
[`CompactionPart`][pydantic_ai.messages.CompactionPart] in the stream instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.messages import CapabilityEvent

COMPACTION_EVENT_NAMESPACE = 'compaction'
"""The [`CapabilityEvent`][pydantic_ai.messages.CapabilityEvent] namespace of the compaction event family."""


@dataclass(repr=False, kw_only=True)
class CompactionStartEvent(CapabilityEvent, namespace=COMPACTION_EVENT_NAMESPACE, name='start', dispatch='immediate'):
    """Emitted before history is compacted; listeners may [`cancel()`][pydantic_ai.capabilities.CompactionStartEvent.cancel] the attempt.

    Dispatched immediately: the emitter awaits the listener chain before proceeding and honors
    [`cancelled`][pydantic_ai.capabilities.CompactionStartEvent.cancelled], so a
    capability (or app code via `hooks.on.event`) can hold compaction while it's mid-activity.
    Cancelling skips this attempt only — the compacting capability re-attempts the next time its
    trigger condition is met.
    """

    strategy: str
    """A stable identifier for the compacting mechanism, e.g. `'openai'` or `'summarizing'`.

    Emitters must pick an explicit identifier that is safe to persist and branch on — never a
    Python class name, which a refactor could silently change under dashboards and stored records.
    """

    messages_before: int
    """The number of messages subject to this compaction attempt.

    Emitters that compact a slice of the history (e.g. everything except the current request)
    count that slice, not the full message list.
    """

    tokens_before: int | None = None
    """The emitter's (typically estimated) token size of the history being compacted, when it has one."""

    cancelled: bool = False
    """Whether a listener cancelled this compaction attempt."""

    cancel_reason: str | None = None
    """The reason given by the cancelling listener, if any."""

    def cancel(self, reason: str | None = None) -> None:
        """Cancel this compaction attempt.

        Args:
            reason: Optional human-readable reason, stored on
                [`cancel_reason`][pydantic_ai.capabilities.CompactionStartEvent.cancel_reason].
        """
        self.cancelled = True
        self.cancel_reason = reason


@dataclass(repr=False, kw_only=True)
class CompactionEndEvent(CapabilityEvent, namespace=COMPACTION_EVENT_NAMESPACE, name='end'):
    """Emitted after history was actually compacted (not when an attempt was cancelled or a no-op)."""

    strategy: str
    """Identifies the compaction mechanism, matching
    [`CompactionStartEvent.strategy`][pydantic_ai.capabilities.CompactionStartEvent.strategy]."""

    messages_before: int
    """The number of messages the compaction operated on."""

    messages_after: int
    """The number of messages that replaced them."""

    tokens_before: int | None = None
    """The history's token size before compaction, when the emitter knows it."""

    tokens_after: int | None = None
    """The history's token size after compaction, when the emitter knows it."""
