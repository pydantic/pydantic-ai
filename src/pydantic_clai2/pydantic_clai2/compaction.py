"""The built-in `compaction` plugin: Code Puppy's compaction chain from harness, `/compact`, and a context gauge.

The chain is `FallbackCompaction` over `SummarizingCompaction` then `SlidingWindowCompaction`, so a
failed or over-budget summary degrades to truncation. `compact_now` drives the chain for `/compact`.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import RunContext
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

from .commands import Command
from .plugins import PluginHost, SessionEnd


class CompactionSettings(BaseModel):
    """What `/plugins add compaction pydantic_clai2.compaction '{...}'` may override."""

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


def activate(host: PluginHost[None]) -> None:
    """Register automatic compaction, gauge the remaining usage, and offer `/compact [focus]`.

    Typed for `None` deps because `compact_now` runs the chain on a context with no deps;
    the strategies never read them, so the plugin works with any agent.
    """
    config = host.settings(CompactionSettings)
    chain = build_chain(config)
    host.add(chain)
    host.add(ReportContextUsage(context_window=config.context_window))

    @host.on(ContextUsageEvent)
    async def gauge(ctx: RunContext[None], event: ContextUsageEvent) -> None:
        """Show the request's size as it goes out; the response's reported usage replaces it on arrival."""
        host.status.context_tokens = event.used_tokens
        host.status.context_alert = event.fraction > config.threshold

    @host.on('session_end')
    async def clear_alert(event: SessionEnd) -> None:
        host.status.context_alert = False

    async def compact(args: list[str]) -> str:
        before = host.conversation.messages
        if not before:
            return 'Nothing to compact: the conversation is empty.'
        model = await host.conversation.resolved_model()
        if model is None:
            raise ValueError('Choose a model first: /set model <Tab>')
        after = await compact_now(chain, before, model=model, focus=' '.join(args) or None)
        if after == before:
            return f'Nothing to compact: the last {config.protected_tokens:,} tokens are always kept.'
        await host.conversation.commit_messages(after)
        saved = max(estimate_token_count(before) - estimate_token_count(after), 0)
        return f'Compacted {len(before)} messages down to {len(after)}; about {saved:,} tokens saved.'

    host.commands.register(
        Command(
            name='compact',
            description='Compact the conversation so far; add words to say what the summary must keep',
            handler=compact,
        )
    )
