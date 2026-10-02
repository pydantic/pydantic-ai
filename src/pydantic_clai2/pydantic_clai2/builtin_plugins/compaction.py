"""Keep long conversations inside the context window by summarizing older messages; adds /compact.

The built-in `compaction` plugin: Code Puppy's compaction chain from harness, `/compact`, and a context gauge.

The chain is `FallbackCompaction` over `SummarizingCompaction` then `SlidingWindowCompaction`, so a
failed or over-budget summary degrades to truncation. `compact_now` drives the chain for `/compact`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, on_event
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, UsageLimitExceeded
from pydantic_ai_harness.compaction import (
    ContextUsageEvent,
    FallbackCompaction,
    ReportContextUsage,
    SlidingWindowCompaction,
    SummarizingCompaction,
    compact_now,
    estimate_token_count,
)
from pydantic_clai2.commands import Command
from pydantic_clai2.plugins import Plugin, PluginHost, SessionEnd
from pydantic_clai2.ui.rendering.status import Status


class CompactionSettings(BaseModel):
    """What `/plugins add compaction pydantic_clai2.builtin_plugins.compaction '{...}'` may override."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    strategy: Literal['summarization', 'truncation'] = Field(
        default='summarization',
        description='`summarization` summarises older messages and falls back to truncation; `truncation` only drops them.',
    )
    threshold: float = Field(
        default=0.85,
        gt=0,
        le=1,
        allow_inf_nan=False,
        description='Compact once the history exceeds this fraction of the context window.',
    )
    protected_tokens: int = Field(
        default=50_000, ge=0, description='Tokens of the most recent messages that are never compacted.'
    )
    context_window: int | None = Field(
        default=None,
        gt=0,
        description='Context window in tokens, when the catalog is wrong or silent. Unset resolves it from the model.',
    )
    summarization_model: str | None = Field(
        default=None, description='Model that writes the summary. Unset uses the model running the conversation.'
    )


def build_chain(config: CompactionSettings) -> FallbackCompaction[None]:
    """Code Puppy's chain: summarise, and truncate when the summary fails or blows the usage limit."""
    sliding: SlidingWindowCompaction[None] = SlidingWindowCompaction(
        max_messages=1, keep_tokens=config.protected_tokens
    )
    if config.strategy == 'truncation':
        return FallbackCompaction(
            fallback_chain=[sliding], max_fraction=config.threshold, context_window=config.context_window
        )
    summarizer: SummarizingCompaction[None] = SummarizingCompaction(
        model=config.summarization_model, max_messages=1, keep_tokens=config.protected_tokens
    )
    return FallbackCompaction(
        fallback_chain=[summarizer, sliding],
        max_fraction=config.threshold,
        context_window=config.context_window,
        fallback_on=(ModelAPIError, FallbackExceptionGroup, UsageLimitExceeded),
    )


@dataclass
class _ContextGauge(AbstractCapability[None]):
    """Show each request's size in the status row as it goes out; the response's reported usage replaces it."""

    status: Status
    threshold: float

    @on_event(ContextUsageEvent)
    async def _gauge(self, ctx: RunContext[None], event: ContextUsageEvent) -> None:
        self.status.context_tokens = event.used_tokens
        self.status.context_window = event.window_tokens if event.resolved else None
        self.status.context_alert = event.fraction > self.threshold


class CompactionPlugin(Plugin[CompactionSettings]):
    """Automatic compaction, a gauge of the remaining context, and `/compact [focus]`.

    Typed for `None` deps because `compact_now` runs the chain on a context with no deps;
    the strategies never read them, so the plugin works with any agent.
    """

    def __init__(self, host: PluginHost[None], settings: CompactionSettings) -> None:
        super().__init__(host, settings)
        self.chain = build_chain(settings)

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (
            self.chain,
            ReportContextUsage(context_window=self.settings.context_window),
            _ContextGauge(status=self.host.status, threshold=self.settings.threshold),
        )

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='compact',
                description='Compact the conversation so far; add words to say what the summary must keep',
                handler=self._compact,
                raw=True,
            ),
        )

    async def on_session_end(self, event: SessionEnd) -> None:
        self.host.status.context_window = None
        self.host.status.context_alert = False

    async def _compact(self, args: list[str]) -> str:
        conversation = self.host.conversation
        before = conversation.messages
        if not before:
            return 'Nothing to compact: the conversation is empty.'
        model = await conversation.resolved_model()
        if model is None:
            raise ValueError('Choose a model first: /set model <Tab>')
        after = await compact_now(self.chain, before, model=model, focus=' '.join(args) or None)
        if after == before:
            return f'Nothing to compact: the last {self.settings.protected_tokens:,} tokens are always kept.'
        await conversation.commit_messages(after)
        saved = max(estimate_token_count(before) - estimate_token_count(after), 0)
        return f'Compacted {len(before)} messages down to {len(after)}; about {saved:,} tokens saved.'
