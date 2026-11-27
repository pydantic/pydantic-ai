"""Keep long conversations inside the context window by summarizing older messages; adds /compact and /auto-compact.

The built-in `compaction` plugin: Code Puppy's compaction chain from harness, `/compact`, `/auto-compact` to
move its trigger, and a context gauge.

The chain is `FallbackCompaction` over `SummarizingCompaction` then `SlidingWindowCompaction`, so a
failed or over-budget summary degrades to truncation. `compact_now` drives the chain for `/compact`.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
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
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, UsageLimitExceeded
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
from pydantic_ai_pax.plugins.menus import Dialogs, FieldRow, Option, SettingsMenu
from pydantic_ai_pax.runtime import agent_instrumentation
from pydantic_ai_pax.ui.menus import dialogs
from pydantic_ai_pax.ui.menus.field_menu import first_error, shown
from pydantic_ai_pax.ui.menus.menu_worker import remote_dialogs, require_terminal
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
        'Ignored when Threshold tokens is set.',
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
        'or of Threshold tokens when set.',
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
            row('threshold_tokens', 'Threshold tokens'),
            row('protected_tokens', 'Protected tokens'),
            row('context_window', 'Context window'),
            row('summarization_model', 'Summarization model'),
            row('background_compaction', 'Background compaction', ('true', 'false')),
            row('background_start', 'Background start'),
        )

    def current(self, row: FieldRow) -> str:
        settings = self.host.settings(CompactionSettings)
        values: dict[str, JsonValue] = {
            'strategy': settings.strategy,
            'threshold': settings.threshold,
            'threshold_tokens': settings.threshold_tokens,
            'protected_tokens': settings.protected_tokens,
            'context_window': settings.context_window,
            'summarization_model': settings.summarization_model,
            'background_compaction': settings.background_compaction,
            'background_start': settings.background_start,
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


_TOKEN_COUNT = re.compile(r'(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)([km]?)', re.IGNORECASE)
"""A count, commas only between thousands, and an optional `k` or `m`: `150000`, `150,000`, `150k`, `1.5m`."""

_SUFFIXES = {'': 1, 'k': 1_000, 'm': 1_000_000}


def _token_count(text: str) -> int:
    """`150000`, `150,000`, `150k` or `1.5m` as a token count; `ValueError` otherwise."""
    if (match := _TOKEN_COUNT.fullmatch(text)) is None:
        raise ValueError(f'{text!r} is not a token count such as 150k, a share such as 70%, or reset.')
    # Decimal, so `4.1m` is exactly 4,100,000 rather than a float just under it.
    count = Decimal(match[1].replace(',', '')) * _SUFFIXES[match[2].lower()]
    if count != count.to_integral_value():
        raise ValueError(f'{text!r} is not a whole number of tokens.')
    return int(count)


def _with_trigger(settings: CompactionSettings, value: str) -> CompactionSettings:
    """`settings` with the trigger `/auto-compact VALUE` sets; `ValueError` saying why `value` is not one."""
    data: dict[str, JsonValue] = settings.model_dump(mode='json')
    data.pop('threshold_tokens', None)
    if value.lower() == 'reset':
        data.pop('threshold', None)
    elif value.endswith('%'):
        data['threshold'] = _percent(value)
    else:
        data['threshold_tokens'] = _token_count(value)
    try:
        return CompactionSettings.model_validate(data)
    except ValueError as exc:
        raise ValueError(f'/auto-compact {value}: {first_error(exc)}') from None


def _trigger_problem(settings: CompactionSettings, value: str) -> str | None:
    """Why `value` is not a token count to compact above, for the custom row's input; `None` when it is."""
    if _TOKEN_COUNT.fullmatch(value) is None:
        return 'Type a token count, such as 150k or 1.5m.'
    try:
        CompactionSettings.model_validate({**settings.model_dump(mode='json'), 'threshold_tokens': _token_count(value)})
    except ValueError as exc:
        return first_error(exc)
    return None


_PRESETS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9)
"""The shares of the window the `/auto-compact` list offers."""

_CUSTOM = 'custom'
"""The list row that asks for a token count, an `Option.value` `/auto-compact` never takes typed."""


def _share(fraction: float) -> str:
    return f'{fraction * 100:g}%'


def _thousands(tokens: int) -> str:
    return f'{round(tokens / 1000):,}k'


def _trigger_options(settings: CompactionSettings, window: int | None) -> tuple[list[Option], str]:
    """The `/auto-compact` list, each value what `/auto-compact` takes typed, and the current row's value."""
    default = CompactionSettings.model_fields['threshold'].default
    shares = sorted({*_PRESETS, settings.threshold})
    options = [
        Option(
            value=_share(fraction),
            label=' - '.join(
                (
                    _share(fraction),
                    *((f'about {_thousands(round(window * fraction))} tokens',) if window else ()),
                    *(('default',) if fraction == default else ()),
                )
            ),
            description=(
                f'Compacts at about {_thousands(round(window * fraction))} tokens of the {_thousands(window)} window.'
                if window
                else "Compacts at this share of the model's context window, whichever model you use."
            ),
        )
        for fraction in shares
    ]
    current = _share(settings.threshold)
    if (tokens := settings.threshold_tokens) is not None:
        current = str(tokens)
        options.append(
            Option(
                value=current,
                label=f'{tokens:,} tokens',
                description='Compacts at this many tokens, whatever the window.',
            )
        )
    options += [
        Option(value=_CUSTOM, label='Fixed token count...', description='Type a count, such as 150k.'),
        Option(value='reset', label='Reset to default', description=f'Compacts at {_share(default)} of the window.'),
    ]
    return options, current


def _trigger(settings: CompactionSettings) -> str:
    if (tokens := settings.threshold_tokens) is not None:
        return f'above {tokens:,} tokens'
    return f'above {settings.threshold * 100:g}% of the context window'


def _percent(text: str) -> float:
    """`70%` as `0.7`; `ValueError` otherwise."""
    try:
        return float(text[:-1]) / 100
    except ValueError:
        raise ValueError(f'{text!r} is not a share of the window such as 70%.') from None


def build_chain(config: CompactionSettings) -> FallbackCompaction[None]:
    """Code Puppy's chain: summarise, and truncate when the summary fails or blows the usage limit.

    It triggers at `threshold_tokens` when set, else at `threshold` of the window.
    """
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
        fallback_on=(ModelAPIError, FallbackExceptionGroup, UsageLimitExceeded),
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
                description='Choose when compaction runs, or set it directly: a share of the window (70%) or tokens (150k)',
                handler=self._auto_compact,
                complete=lambda args: ('150k', '70%', 'reset') if len(args) <= 1 else (),
            ),
        )

    async def _auto_compact(self, args: list[str]) -> str:
        if len(args) > 1:
            raise ValueError('Give one value: /auto-compact 150k, /auto-compact 70%, or /auto-compact reset.')
        asking = self._dialogs()
        if args:
            value = args[0]
        elif asking is None:
            return f'Auto-compaction runs {_trigger(self.settings)}. Set it with /auto-compact 150k, 70%, or reset.'
        elif (picked := await self._pick_trigger(asking)) is None:
            return f'Auto-compaction unchanged: it runs {_trigger(self.settings)}.'
        else:
            value = picked
        settings = _with_trigger(self.settings, value)
        if (plugins := self.host.plugins) is None:
            raise ValueError('/auto-compact needs a Pax session; use /plugins configure compaction.')
        # Loads the plugin again, so the next turn runs the new chain; this instance is unloaded.
        await plugins.save_settings(self.host.name, settings)
        return f'Auto-compaction now runs {_trigger(settings)}.'

    def _dialogs(self) -> Dialogs | None:
        """Where to show the trigger list: the web UI's dialogs, the terminal, or `None` with neither, as in `-p`."""
        if (remote := remote_dialogs()) is not None:
            return remote
        if not self.host.console.is_terminal:
            return None
        try:
            require_terminal()
        except ValueError:
            return None
        # The module's `RUNNERS` is read here, not imported, so tests can script the terminal.
        return dialogs.TerminalDialogs(dialogs.RUNNERS)

    async def _pick_trigger(self, asking: Dialogs) -> str | None:
        """What the user picked in the trigger list, as `/auto-compact` takes it typed; `None` to keep the current one."""
        window = self.host.status.window
        options, current = _trigger_options(self.settings, window)
        picked = await asking.choose(
            title='When should Pax compact automatically?',
            options=options,
            current=current,
            description=f'This model has a {_thousands(window)} window.' if window else '',
        )
        if picked == _CUSTOM:
            picked = await self._ask_token_count(asking)
        return None if picked in (None, '', current) else picked

    async def _ask_token_count(self, asking: Dialogs) -> str | None:
        tokens = self.settings.threshold_tokens
        return await asking.text(
            title='Compact above how many tokens?',
            label='Tokens',
            placeholder='150k',
            initial='' if tokens is None else str(tokens),
            check=partial(_trigger_problem, self.settings),
            description="Such as 150000, 150k or 1.5m. Keep it below the model's window.",
        )

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
            chain = build_chain(self.settings.model_copy(update={'protected_tokens': tail}))
            after = await compact_now(chain, before, model=model, focus=focus, tracer=tracer)
        saved = tokens - sum(estimate_message_tokens(after))
        if saved <= 0:
            return 'Nothing to compact: compacting would not make the conversation smaller.'
        await conversation.commit_messages(after)
        return f'Compacted {len(before)} messages down to {len(after)}; about {saved:,} of {tokens:,} tokens saved.'
