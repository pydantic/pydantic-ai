"""Keep long conversations inside the context window by summarizing older messages; adds /compact and /auto-compact.

The built-in `compaction` plugin: Code Puppy's compaction chain from harness, `/compact`, `/auto-compact` to
move its trigger, and a context gauge.

The chain is `FallbackCompaction` over `SummarizingCompaction` then `SlidingWindowCompaction`, so a
failed or over-budget summary degrades to truncation. `compact_now` drives the chain for `/compact`,
which raises a summary the model got wrong instead, so the user sees the summarizer's error.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    model_serializer,
)

from pydantic_ai import ModelMessage, ModelResponse, RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, on_event
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai_harness.compaction import (
    ContextUsageMeasuredEvent,
    FallbackCompaction,
    ReportContextUsage,
    SlidingWindowCompaction,
    SummarizingCompaction,
    compact_now,
    estimate_message_tokens,
)
from pydantic_ai_pax.commands import Command
from pydantic_ai_pax.plugins import ConversationChanged, Plugin, PluginHost, SessionEnd, TurnEnd
from pydantic_ai_pax.plugins.menus import FieldRow, SettingsMenu
from pydantic_ai_pax.runtime import agent_instrumentation
from pydantic_ai_pax.ui.menus import dialogs
from pydantic_ai_pax.ui.menus.field_menu import first_error, shown
from pydantic_ai_pax.ui.menus.menu_worker import require_terminal
from pydantic_ai_pax.ui.rendering.status import Status


class CompactionSettings(BaseModel):
    """What `/plugins add compaction pydantic_ai_pax.builtin_plugins.compaction '{...}'` may override."""

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
        description='Compact once the history exceeds this fraction of the context window. '
        'Ignored when threshold_tokens is set.',
    )
    threshold_tokens: int | None = Field(
        default=None,
        gt=0,
        description='Compact once the history exceeds this many tokens, instead of the threshold fraction. '
        'Unset uses the threshold.',
    )
    protected_tokens: int = Field(
        default=50_000,
        ge=0,
        description='Tokens of the most recent messages that are never compacted; /compact caps it at half.',
    )
    context_window: int | None = Field(
        default=None,
        gt=0,
        description='Context window in tokens, when the catalog is wrong or silent. Unset resolves it from the model.',
    )
    summarization_model: str | None = Field(
        default=None, description='Model that writes the summary. Unset uses the model running the conversation.'
    )
    background_compaction: bool = Field(
        default=False,
        description='Experimental: write the summary in the background before the threshold and swap it in when '
        'compaction is due. Only with `summarization`.',
    )
    background_start: float = Field(
        default=0.85,
        gt=0,
        lt=1,
        allow_inf_nan=False,
        description='Experimental: with background compaction on, start the summary at this fraction of the threshold, '
        'or of threshold_tokens when set.',
    )

    @model_serializer(mode='wrap')
    def _only_chosen(self, handler: SerializerFunctionWrapHandler) -> dict[str, JsonValue]:
        """Save only the settings someone chose, so menu edits never pin the other defaults."""
        dumped: dict[str, JsonValue] = handler(self)
        return {key: value for key, value in dumped.items() if key in self.model_fields_set}


BACKGROUND_COMPACTION_FEATURE = 'compaction-background'
"""Builds without it would reject a saved `background_compaction` or `background_start`, so they drop both."""

THRESHOLD_TOKENS_FEATURE = 'compaction-threshold-tokens'
"""Builds without it would reject a saved `threshold_tokens`, so they drop it and compact at `threshold`."""

COMPACT_AT = 'compact_at'
"""The settings menu row that edits `threshold` and `threshold_tokens` together; not a setting itself."""

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
            self._compact_at(),
            row('strategy', 'Strategy', ('summarization', 'truncation')),
            row('protected_tokens', 'Protected tokens'),
            row('context_window', 'Context window'),
            row('summarization_model', 'Summarization model'),
            row('background_compaction', 'Background compaction', ('true', 'false')),
            row('background_start', 'Background start'),
        )

    def _compact_at(self) -> FieldRow:
        """`threshold` and `threshold_tokens` as one row: shares of the window to step through, or a token count."""
        settings = self.host.settings(CompactionSettings)
        # A saved share the presets lack sits among them, in order; a saved token count follows them.
        fractions = sorted({*_PRESETS, *(() if settings.threshold_tokens is not None else (settings.threshold,))})
        shares = [_share(fraction) for fraction in fractions]
        current = _trigger_text(settings)
        window = self.host.status.window
        return FieldRow(
            key=COMPACT_AT,
            label='Compact at',
            description='When compaction runs: a share of the context window, such as 70%, which fits every '
            'model, or a token count, such as 150k, whatever the window.',
            default=_share(_DEFAULT_THRESHOLD),
            choices=(*shares, *(() if current in shares else (current,))),
            choice_labels={
                share: f'{share} - about {_thousands(round(window * fraction))} tokens'
                for share, fraction in zip(shares, fractions, strict=True)
            }
            if window
            else {},
        )

    def current(self, row: FieldRow) -> str:
        settings = self.host.settings(CompactionSettings)
        values: dict[str, JsonValue] = {
            COMPACT_AT: _trigger_text(settings),
            'strategy': settings.strategy,
            'protected_tokens': settings.protected_tokens,
            'context_window': settings.context_window,
            'summarization_model': settings.summarization_model,
            'background_compaction': settings.background_compaction,
            'background_start': settings.background_start,
        }
        return shown(values[row.key])

    def _updated(self, row: FieldRow, raw: str) -> CompactionSettings:
        if row.key == COMPACT_AT:
            return _with_trigger(self.host.settings(CompactionSettings), raw)
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
        settings = self._updated(row, raw)
        self.host.save_settings(settings)
        if row.key == COMPACT_AT and (note := _beyond_window(settings, self.host.status.window)):
            return f'Saved {row.label}{note}.'
        return f'Saved {row.label}.'

    def reset(self, row: FieldRow) -> str:
        if row.key == COMPACT_AT:
            self.host.save_settings(_with_trigger(self.host.settings(CompactionSettings), 'reset'))
            return f'Reset {row.label}.'
        data: dict[str, JsonValue] = self.host.settings(CompactionSettings).model_dump(mode='json')
        data.pop(row.key, None)
        self.host.save_settings(CompactionSettings.model_validate(data))
        return f'Reset {row.label}.'


_TOKEN_COUNT = re.compile(r'(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)([km]?)', re.IGNORECASE)
"""A count, commas only between thousands, and an optional `k` or `m`: `150000`, `150,000`, `150k`, `1.5m`."""

_SUFFIXES = {'': 1, 'k': 1_000, 'm': 1_000_000}

_NOT_A_TRIGGER = 'Type a token count such as 150k, or a share of the window such as 70%.'

_PRESETS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)
"""The shares of the window the Compact at row steps through."""

_DEFAULT_THRESHOLD: float = CompactionSettings.model_fields['threshold'].default


def _token_count(text: str) -> int:
    """`150000`, `150,000`, `150k` or `1.5m` as a token count; `ValueError` otherwise."""
    if (match := _TOKEN_COUNT.fullmatch(text)) is None:
        raise ValueError(_NOT_A_TRIGGER)
    # Decimal, so `4.1m` is exactly 4,100,000 rather than a float just under it.
    count = Decimal(match[1].replace(',', '')) * _SUFFIXES[match[2].lower()]
    if count != count.to_integral_value():
        raise ValueError('Type a whole number of tokens.')
    return int(count)


def _percent(text: str) -> float:
    """`70%` as `0.7`; `ValueError` otherwise."""
    try:
        return float(text[:-1]) / 100
    except ValueError:
        raise ValueError(_NOT_A_TRIGGER) from None


def _with_trigger(settings: CompactionSettings, value: str) -> CompactionSettings:
    """`settings` compacting at `value` (`70%`, `150k`, or `reset`); `ValueError` saying why it is not a trigger.

    Only a share other than the default is saved; the default share and `reset` clear it.
    """
    data: dict[str, JsonValue] = settings.model_dump(mode='json')
    data.pop('threshold_tokens', None)
    share = value.endswith('%')
    # The default share is saved as no choice, as reset is, so a later default still applies.
    if value.lower() == 'reset' or (share and _percent(value) == _DEFAULT_THRESHOLD):
        data.pop('threshold', None)
    elif share:
        data['threshold'] = _percent(value)
    else:
        data['threshold_tokens'] = _token_count(value)
    try:
        return CompactionSettings.model_validate(data)
    except ValueError as exc:
        raise ValueError('Type a share above 0% and up to 100%.' if share else first_error(exc)) from None


def _share(fraction: float) -> str:
    return f'{fraction * 100:g}%'


def _thousands(tokens: int) -> str:
    return f'{round(tokens / 1000):,}k'


def _trigger_text(settings: CompactionSettings) -> str:
    """The trigger as the Compact at row and `/auto-compact` take it typed: `70%`, `150k` or `1.5m`."""
    if (tokens := settings.threshold_tokens) is None:
        return _share(settings.threshold)
    if tokens >= 1_000_000 and tokens % 100_000 == 0:
        return f'{tokens / 1_000_000:g}m'
    return f'{tokens // 1000}k' if tokens % 1000 == 0 else str(tokens)


def _beyond_window(settings: CompactionSettings, window: int | None) -> str:
    """A note when the token count cannot trigger on this model, whose requests fail first; kept for a larger one."""
    tokens = settings.threshold_tokens
    if tokens is None or not window or tokens < window:
        return ''
    return f" (at or above this model's {_thousands(window)} window, so it will not trigger here)"


def _trigger(settings: CompactionSettings) -> str:
    if (tokens := settings.threshold_tokens) is not None:
        return f'above {tokens:,} tokens'
    return f'above {_share(settings.threshold)} of the context window'


def build_chain(config: CompactionSettings, *, explicit: bool = False) -> FallbackCompaction[None]:
    """Code Puppy's chain: summarise, and truncate when the summary fails or blows the usage limit.

    It triggers at `threshold_tokens` when set, else at `threshold` of the window. With `explicit`, for
    `/compact`, a summary the model got wrong (`UnexpectedModelBehavior`) is raised rather than truncated,
    so the user sees why their compaction didn't summarize.
    """
    fallback_on: tuple[type[Exception], ...] = (ModelAPIError, FallbackExceptionGroup, UsageLimitExceeded)
    if not explicit:
        fallback_on = (*fallback_on, UnexpectedModelBehavior)
    sliding: SlidingWindowCompaction[None] = SlidingWindowCompaction(
        max_messages=1, keep_tokens=config.protected_tokens
    )
    max_tokens = config.threshold_tokens
    max_fraction = config.threshold if max_tokens is None else None
    if config.strategy == 'truncation':
        return FallbackCompaction(
            fallback_chain=[sliding],
            max_tokens=max_tokens,
            max_fraction=max_fraction,
            context_window=config.context_window,
        )
    summarizer: SummarizingCompaction[None] = SummarizingCompaction(
        model=config.summarization_model, max_messages=1, keep_tokens=config.protected_tokens
    )
    return FallbackCompaction(
        fallback_chain=[summarizer, sliding],
        max_tokens=max_tokens,
        max_fraction=max_fraction,
        context_window=config.context_window,
        fallback_on=fallback_on,
        background_start=config.background_start if config.background_compaction else None,
    )


@dataclass
class _ContextGauge(AbstractCapability[None]):
    """Show each request's size in the status row as it goes out; the response's reported usage replaces it."""

    status: Status
    threshold: float
    threshold_tokens: int | None = None

    @on_event(ContextUsageMeasuredEvent)
    async def _gauge(self, ctx: RunContext[None], event: ContextUsageMeasuredEvent) -> None:
        self.status.context_tokens = event.used_tokens
        # Without a response's reported usage, the reading counts characters and leaves out most tool
        # definitions, so a session's first request reads well under what the provider then reports.
        self.status.context_estimated = not any(
            isinstance(message, ModelResponse) and message.usage.input_tokens for message in ctx.messages
        )
        self.status.context_window = event.window_tokens if event.resolved else None
        if self.threshold_tokens is None:
            self.status.context_alert = event.fraction > self.threshold
        else:
            self.status.context_alert = event.used_tokens > self.threshold_tokens


class CompactionPlugin(Plugin[CompactionSettings]):
    """Automatic compaction, a gauge of the remaining context, and `/compact [focus]`.

    Typed for `None` deps because `compact_now` runs the chain on a context with no deps;
    the strategies never read them, so the plugin works with any agent.
    """

    @classmethod
    def from_host(cls, host: PluginHost[None]) -> Self:
        # Tags the settings, so a build without a feature uses the defaults instead of failing to load.
        requires = {key: [BACKGROUND_COMPACTION_FEATURE] for key in ('background_compaction', 'background_start')}
        requires['threshold_tokens'] = [THRESHOLD_TOKENS_FEATURE]
        return cls(host, host.settings(CompactionSettings, requires=requires))

    def __init__(self, host: PluginHost[None], settings: CompactionSettings) -> None:
        super().__init__(host, settings)
        self.chain = build_chain(settings)

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (
            self.chain,
            ReportContextUsage(context_window=self.settings.context_window),
            _ContextGauge(
                status=self.host.status,
                threshold=self.settings.threshold,
                threshold_tokens=self.settings.threshold_tokens,
            ),
        )

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='compact',
                description='Compact the conversation so far; add words to say what the summary must keep',
                handler=self._compact,
                raw=True,
                live=True,
            ),
            Command(
                name='auto-compact',
                description='Choose when compaction runs, or set it: a share of the window (70%) or tokens (150k)',
                handler=self._auto_compact,
                complete=lambda args: ('150k', '70%', 'reset') if len(args) <= 1 else (),
            ),
        )

    async def _auto_compact(self, args: list[str]) -> str:
        if len(args) > 1:
            raise ValueError('Give one value: /auto-compact 150k, /auto-compact 70%, or /auto-compact reset.')
        plugins = self.host.plugins
        if not args:
            if plugins is None or not self._terminal():
                return (
                    f'Auto-compaction runs {_trigger(self.settings)}. Change it with /auto-compact 70%, 150k or reset, '
                    'or in the compaction settings.'
                )
            # The settings menu `/plugins configure compaction` opens; the plugin loads again if it changed.
            messages = await plugins.edit_settings(self.host.name, dialogs.run_settings_menu)
            return '\n'.join(messages) or f'Auto-compaction unchanged: it runs {_trigger(self.settings)}.'
        try:
            settings = _with_trigger(self.settings, args[0])
        except ValueError as exc:
            raise ValueError(f'/auto-compact {args[0]}: {exc}') from None
        # Compared as saved, so `150k` over a saved 150000 changes nothing, and reset still drops a pinned default.
        if settings.model_dump(mode='json') == self.settings.model_dump(mode='json'):
            return f'Auto-compaction unchanged: it runs {_trigger(settings)}.'
        if plugins is None:
            raise ValueError('/auto-compact needs a Pax session; use /plugins configure compaction.')
        # Loads the plugin again, so the next turn runs the new chain; this instance is unloaded.
        window = self.host.status.window
        await plugins.save_settings(self.host.name, settings)
        return f'Auto-compaction now runs {_trigger(settings)}{_beyond_window(settings, window)}.'

    def _terminal(self) -> bool:
        """Whether the settings menu can open here: a terminal, and not a surface that refuses one, as the web UI does."""
        if not self.host.console.is_terminal:
            return False
        try:
            require_terminal()
        except ValueError:
            return False
        return True

    def settings_menu(self) -> SettingsMenu:
        return SettingsMenu(source=CompactionSource(self.host))

    async def on_conversation_changed(self, event: ConversationChanged) -> None:
        await self._discard_left_behind()

    async def on_turn_end(self, event: TurnEnd) -> None:
        await self._discard_left_behind()

    async def _discard_left_behind(self) -> None:
        """Drop the background summaries of conversations that may never run again.

        Only the conversation shown and its running delegated tasks, each a conversation of its own
        under the task's ID, can still use theirs; a conversation left with `/new` or `/resume` and a
        finished task would keep theirs, with a copy of their history, until Pax exits.
        """
        keep: set[str | None] = {self.host.conversation.conversation_id}
        if (tasks := self.host.tasks) is not None:
            keep.update(task.id for task in tasks.listing() if task.status == 'running')
        for conversation_id in self.chain.background_conversations - keep:
            await self.chain.discard_background_compaction(conversation_id)

    async def on_session_end(self, event: SessionEnd) -> None:
        self.host.status.context_window = None
        self.host.status.context_alert = False
        await self.chain.aclose()

    async def _compact(self, args: list[str]) -> str:
        conversation = self.host.conversation
        before = conversation.messages
        if not before:
            return 'Nothing to compact: the conversation is empty.'
        model = await conversation.resolved_model()
        if model is None:
            raise ValueError('Choose a model first: /set model <Tab>')
        # Counted as the provider reported them, so thinking returned without text is not missed.
        tokens = sum(estimate_message_tokens(before))
        focus = ' '.join(args) or None
        tracer = agent_instrumentation.tracer()
        conversation_id = conversation.conversation_id
        after: list[ModelMessage] | None = None
        if focus is None:
            after = await self.chain.take_background_compaction(
                before, model=model, conversation_id=conversation_id, tracer=tracer
            )
        else:
            # A summary written in the background has no focus, and the history is about to change.
            await self.chain.discard_background_compaction(conversation_id)
        if after is None:
            # Cap the protected target at half the history: a conversation just over the protected tail
            # would otherwise trade a sliver of its head for a summary plus the kept first prompt, and grow.
            # The kept messages can still exceed it by the whole message that crosses it.
            tail = min(self.settings.protected_tokens, tokens // 2)
            chain = build_chain(self.settings.model_copy(update={'protected_tokens': tail}), explicit=True)
            after = await compact_now(chain, before, model=model, focus=focus, tracer=tracer)
        saved = tokens - sum(estimate_message_tokens(after))
        if saved <= 0:
            return 'Nothing to compact: compacting would not make the conversation smaller.'
        await conversation.commit_messages(after)
        return f'Compacted {len(before)} messages down to {len(after)}; about {saved:,} of {tokens:,} tokens saved.'
