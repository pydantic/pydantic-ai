"""Keep long conversations inside the context window by summarizing older messages; adds /compact.

The built-in `compaction` plugin: Code Puppy's compaction chain from harness, `/compact`, and a context gauge.

The chain is `FallbackCompaction` over `SummarizingCompaction` then `SlidingWindowCompaction`, so a
failed or over-budget summary degrades to truncation. `compact_now` drives the chain for `/compact`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
)

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
from pydantic_clai2.ui.menus.field_menu import FieldMenu, FieldRow, first_error, run_flow, shown
from pydantic_clai2.ui.menus.menu_worker import run_worker
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
        default=50_000,
        ge=0,
        description='Tokens of the most recent messages that are never compacted; /compact keeps at most half.',
    )
    context_window: int | None = Field(
        default=None,
        gt=0,
        description='Context window in tokens, when the catalog is wrong or silent. Unset resolves it from the model.',
    )
    summarization_model: str | None = Field(
        default=None, description='Model that writes the summary. Unset uses the model running the conversation.'
    )

    @model_serializer(mode='wrap')
    def _only_chosen(self, handler: SerializerFunctionWrapHandler) -> dict[str, JsonValue]:
        """Save only the settings someone chose, so menu edits never pin the other defaults."""
        dumped: dict[str, JsonValue] = handler(self)
        return {key: value for key, value in dumped.items() if key in self.model_fields_set}


_TEXT_FIELDS = frozenset({'strategy', 'summarization_model'})
"""Settings typed as plain text; the others are typed as JSON numbers."""


class CompactionSource:
    """The compaction settings, one row each, saved through the validated model; empty input resets."""

    title = 'Compaction settings'

    def __init__(self, host: PluginHost[None]) -> None:
        self.host = host

    def rows(self) -> tuple[FieldRow, ...]:
        fields = CompactionSettings.model_fields

        def row(key: str, label: str, choices: tuple[str, ...] = ()) -> FieldRow:
            default: JsonValue = fields[key].default
            return FieldRow(
                key=key,
                label=label,
                description=fields[key].description or '',
                default=shown(default),
                choices=choices,
                allow_custom=not choices,
            )

        return (
            row('strategy', 'Strategy', ('summarization', 'truncation')),
            row('threshold', 'Threshold'),
            row('protected_tokens', 'Protected tokens'),
            row('context_window', 'Context window'),
            row('summarization_model', 'Summarization model'),
        )

    def current(self, row: FieldRow) -> str:
        settings = self.host.settings(CompactionSettings)
        values: dict[str, JsonValue] = {
            'strategy': settings.strategy,
            'threshold': settings.threshold,
            'protected_tokens': settings.protected_tokens,
            'context_window': settings.context_window,
            'summarization_model': settings.summarization_model,
        }
        return shown(values[row.key])

    def _updated(self, row: FieldRow, raw: str) -> CompactionSettings:
        data: dict[str, JsonValue] = self.host.settings(CompactionSettings).model_dump(mode='json')
        data[row.key] = raw if row.key in _TEXT_FIELDS else TypeAdapter(JsonValue).validate_json(raw)
        return CompactionSettings.model_validate(data)

    def problem(self, row: FieldRow, text: str) -> str | None:
        try:
            self._updated(row, text)
        except ValueError as exc:
            return first_error(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        self.host.save_settings(self._updated(row, raw))
        return f'Saved {row.label}.'

    def reset(self, row: FieldRow) -> str:
        data: dict[str, JsonValue] = self.host.settings(CompactionSettings).model_dump(mode='json')
        data.pop(row.key, None)
        self.host.save_settings(CompactionSettings.model_validate(data))
        return f'Reset {row.label}.'


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

    async def configure(self) -> str:
        if not self.host.console.is_terminal:
            return 'Configure compaction from a terminal: /plugins configure compaction'
        messages = await run_worker(lambda: run_flow(FieldMenu(CompactionSource(self.host))))
        return '\n'.join(messages) or 'No compaction settings changed.'

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
        tokens = estimate_token_count(before)
        # Protect at most half the history: a conversation just over the protected tail would
        # otherwise trade a sliver of its head for a summary plus the kept first prompt, and grow.
        tail = min(self.settings.protected_tokens, tokens // 2)
        chain = build_chain(self.settings.model_copy(update={'protected_tokens': tail}))
        after = await compact_now(chain, before, model=model, focus=' '.join(args) or None)
        saved = tokens - estimate_token_count(after)
        if saved <= 0:
            return 'Nothing to compact: compacting would not make the conversation smaller.'
        await conversation.commit_messages(after)
        return f'Compacted {len(before)} messages down to {len(after)}; about {saved:,} of {tokens:,} tokens saved.'
