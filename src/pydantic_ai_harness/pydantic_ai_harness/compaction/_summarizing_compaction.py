"""`SummarizingCompaction` -- LLM-powered summarization of older messages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from pydantic_ai_harness._usage import reserved_usage_limits
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._pinning import is_pinned, reinject_pinned
from pydantic_ai_harness.compaction._receipts import (
    ReceiptInfo,
    discover_transcript_handle,
    format_receipt,
    is_receipt_part,
    make_receipt_part,
    record_receipt,
)
from pydantic_ai_harness.compaction._shared import (
    compact_with_span,
    context_for_request,
    estimate_token_count,
    exceeds,
    find_first_user_message,
    find_safe_cutoff,
    find_token_cutoff,
    is_realtime_model,
    record_compaction_reclaim,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelRequestPart, UserContent
    from pydantic_ai.models import AbstractModel, ModelRequestContext

_DEFAULT_SUMMARY_PROMPT = """\
You are a context summarization assistant.  The conversation below will be replaced by \
your summary, so it must carry everything needed to continue the task.

Write the summary under these exact section headings, omitting a section only if it has \
no content:

## Intent
The user's overall goal and any standing constraints or preferences.

## Key decisions
Choices made and the reasoning, so they are not relitigated.

## Artifacts
Files, paths, identifiers, commands, and APIs touched -- quote exact names.

## Current state
What is done and what is in progress right now.

## Next steps
The immediate actions still required to finish the task.

## Open questions
Unresolved questions or blockers.

Focus on results, not a replay of completed actions.  Respond ONLY with the summary -- no \
preamble, no markdown fences.

<messages>
{messages}
</messages>\
"""

_DEFAULT_INSTRUCTIONS = (
    'You are a context summarization assistant. Extract the most important information from conversations.'
)

_SUMMARY_PREFIX = 'Summary of previous conversation:\n\n'

# Anchored-incremental update instruction (opencode mechanism): the previous summary is fed
# back as an anchor to update in place rather than a summary to re-summarize, which avoids
# summary-of-summary decay.  Wording is minimal/neutral and flagged pending the eval-rig pass.
_INCREMENTAL_UPDATE_INSTRUCTION = (
    'An anchored summary from earlier compaction is provided below in <previous-summary>. '
    'Update it using the conversation above: preserve still-true details, remove stale '
    'details, and merge in new facts.  Keep the same section structure.'
)

# Cross-model bridge prefix (Codex prior art): when the summarizer's model family differs from
# the family that produced the history, prefix the summary with a neutral one-liner so the
# resuming model treats it as a handoff from a different model.  Wording flagged pending eval-rig.
_BRIDGE_PREFIX = 'This summary was produced by a different model than the one continuing the task.'

_KEPT_USER_MESSAGE_METADATA = 'pydantic-ai-harness.compaction.kept-user-message.v1'
"""Model-request metadata marking a user turn retained by `keep_user_messages`."""


def _model_name(model: str | AbstractModel | None) -> str | None:
    """Best-effort model-name string from a model spec or object.

    Accepts any `AbstractModel` (a realtime model included), not just a request-response
    `Model`: the family heuristic only reads `model_name`, which every model carries.
    """
    if model is None:  # pragma: no cover - Pydantic AI always supplies the running model
        return None
    if isinstance(model, str):
        return model
    return getattr(model, 'model_name', None)


def _history_model_name(messages: Sequence[ModelMessage]) -> str | None:
    """Return the most recent response's ``model_name`` -- the family that produced the history."""
    for msg in reversed(messages):
        if isinstance(msg, ModelResponse) and msg.model_name:
            return msg.model_name
    return None


def _is_receipt_message(msg: ModelMessage) -> bool:
    """Return True if *msg* is a request made up entirely of receipt parts."""
    return isinstance(msg, ModelRequest) and bool(msg.parts) and all(is_receipt_part(p) for p in msg.parts)


def _model_family(model: str | AbstractModel | None) -> str | None:
    """Reduce a model name to a coarse family token (e.g. ``openai:gpt-4o`` -> ``gpt``).

    A neutral structural heuristic: drop any ``provider:`` prefix, then take the leading token
    before the first ``-`` or ``/``.  Good enough to tell ``gpt`` from ``claude``; the exact
    family taxonomy is left to the eval-rig pass.
    """
    if isinstance(model, FallbackModel):
        return _model_family(model.models[0])
    name = _model_name(model)
    if not name:  # pragma: no cover - empty model identifiers fail before compaction
        return None
    tail = name.split(':')[-1]
    for sep in ('/', '-'):
        tail = tail.split(sep)[0]
    return tail or None


def _format_messages(messages: Sequence[ModelMessage], *, skip_previous_summary: bool = False) -> str:
    """Render messages into a human-readable string for summarization."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            if _is_kept_user_message(msg):
                continue
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    lines.append(f'User: {_user_prompt_text(part)}')
                elif isinstance(part, SystemPromptPart) and not (
                    skip_previous_summary and part.content.startswith(_SUMMARY_PREFIX)
                ):
                    lines.append(f'System: {part.content}')
                elif isinstance(part, ToolReturnPart):
                    content_str = str(part.content)[:500]
                    if len(str(part.content)) > 500:
                        content_str += '...'
                    lines.append(f'Tool [{part.tool_name}]: {content_str}')
        else:
            for part in msg.parts:
                if isinstance(part, TextPart):
                    lines.append(f'Assistant: {part.content}')
                elif isinstance(part, ToolCallPart):
                    lines.append(f'Tool Call [{part.tool_name}]: {part.args}')
    return '\n'.join(lines)


def _user_prompt_text(part: UserPromptPart) -> str:
    """Extract text content from a user prompt part."""
    if isinstance(part.content, str):
        return part.content
    texts: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
    return ' '.join(texts) if texts else ''


def _extract_system_prompts(messages: list[ModelMessage]) -> list[SystemPromptPart]:
    """Extract leading system-prompt parts from the conversation."""
    parts: list[SystemPromptPart] = []
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            break
        for part in msg.parts:
            if isinstance(part, SystemPromptPart) and not part.content.startswith(_SUMMARY_PREFIX):
                parts.append(part)
            elif is_pinned(part) or is_receipt_part(part):
                continue
            else:
                return parts
    return parts


def _extract_previous_summary(messages: list[ModelMessage]) -> str | None:
    """Extract the most recent compaction summary from the message history.

    Looks for a ``SystemPromptPart`` whose content starts with the summary prefix,
    which indicates it was produced by a prior compaction pass.
    """
    for msg in reversed(messages):
        if not isinstance(msg, ModelRequest):
            continue
        for part in reversed(msg.parts):
            if isinstance(part, SystemPromptPart) and part.content.startswith(_SUMMARY_PREFIX):
                return _without_bridge_prefix(part.content[len(_SUMMARY_PREFIX) :])
    return None


def _without_bridge_prefix(summary: str) -> str:
    """Remove the bridge notice before an incremental summary becomes the next anchor."""
    return summary.removeprefix(f'{_BRIDGE_PREFIX}\n\n')


def _is_kept_user_message(message: ModelRequest) -> bool:
    """Return whether *message* is an earlier `keep_user_messages` retention copy."""
    return message.metadata is not None and message.metadata.get(_KEPT_USER_MESSAGE_METADATA) is True


@dataclass
class SummarizingCompaction(AbstractCapability[AgentDepsT]):
    """LLM-powered conversation compaction.

    When the conversation exceeds a configurable threshold, older messages are
    summarized using a dedicated model call and replaced with a compact, structured
    summary message, preserving recent context and tool-call integrity.

    This is the expensive tier -- summarization turns input tokens into (pricier) output
    tokens -- so it is best used behind cheaper passes (see `TieredCompaction`).

    The summary call's usage is folded into the parent run's usage (it counts as a real
    request), so cost accounting stays honest; note this also increments the run's request
    count, which a request-count limiter would see.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import SummarizingCompaction

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[SummarizingCompaction(
                model='openai:gpt-4o-mini',
                max_messages=60,
                keep_messages=20,
            )],
        )
        ```
    """

    model: str | Model | None = None
    """Model used to generate summaries.

    When `None`, inherits the model the request being compacted is going to. Core starts
    that as the run's model, so the two differ only where a capability replaced
    `ModelRequestContext.model`; set this explicitly to pin the summarizer regardless.
    """

    model_settings: ModelSettings | None = field(default=None, kw_only=True)
    """Settings for the dedicated summary model call.

    These merge over defaults carried by `model`, allowing the summary call to use a
    policy that differs from the running agent without mutating the model.
    """

    max_messages: int | None = None
    """Trigger compaction when message count exceeds this value."""

    max_tokens: int | None = None
    """Trigger compaction when estimated token count exceeds this value."""

    max_fraction: float | None = field(default=None, kw_only=True)
    """Trigger when estimated tokens exceed this fraction of the model's context window.

    Resolved per request from the request's model, so one setting behaves correctly on any
    model. Mutually exclusive with `max_tokens`."""

    context_window: int | None = field(default=None, kw_only=True)
    """Window override in tokens. `None` resolves it from the request's model.

    Unlike `fallback_context_window`, this applies whether or not resolution succeeds. Reach
    for it when the registry is confidently wrong: a beta- or tier-gated window it records as
    the maximum, or a self-hosted endpoint whose model id describes someone else's
    deployment. Only consulted alongside `max_fraction`."""

    fallback_context_window: int = field(default=DEFAULT_CONTEXT_WINDOW, kw_only=True)
    """Window assumed when the request's model is not in the pricing registry.

    Only consulted alongside `max_fraction`. Supply the real number for a deployment the
    registry cannot resolve."""

    keep_messages: int = 20
    """Number of tail messages to preserve after compaction (message-count trigger)."""

    keep_tokens: int | None = None
    """Target token budget to preserve after compaction (token-count trigger).

    When ``None``, falls back to ``keep_messages``.
    """

    summary_prompt: str = _DEFAULT_SUMMARY_PROMPT
    """Prompt template for generating summaries.

    Must contain a ``{messages}`` placeholder.
    """

    instructions: str = field(default=_DEFAULT_INSTRUCTIONS, kw_only=True)
    """Instructions for the internal agent that writes the summary.

    `summary_prompt` shapes the user turn of the summary request; this sets the internal
    agent's static instructions, which Pydantic AI sends in the request's system prompt.
    Override it when the summarizer endpoint requires a fixed leading instruction.
    """

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When ``None``, uses a ~4 characters-per-token heuristic.
    """

    preserve_first_user_message: bool = True
    """When ``True``, the first ``ModelRequest`` containing a ``UserPromptPart``
    is always kept after compaction, in addition to system prompts.
    """

    incremental: bool = True
    """When ``True``, feed any existing summary from a prior compaction back as an anchored
    ``<previous-summary>`` block with an update instruction (preserve still-true, remove stale,
    merge new) so it is updated in place rather than re-summarized -- avoiding
    summary-of-summary decay.
    """

    bridge_prefix: bool = False
    """When ``True`` and the summarizer's model family differs from the family that produced
    the history, prepend a neutral one-line note marking the summary as a cross-model handoff
    (Codex prior art, anti-confabulation).  Only fires on a genuine family mismatch, so it is
    cheap and off in the common same-model case; the note's wording is flagged pending eval-rig.
    """

    keep_user_messages: bool = False
    """When ``True``, preserve recent summarized user messages (each truncated to
    ``keep_user_messages_max_chars``) alongside the summary. Retained messages consume the
    ``keep_messages`` tail budget, keeping compaction bounded. Supersedes
    ``preserve_first_user_message``.
    """

    keep_user_messages_max_chars: int = 20_000
    """Per-message character cap for ``keep_user_messages``; oversized messages are truncated
    with an explicit marker (the shared truncation-marker convention)."""

    receipts: bool = False
    """When ``True``, append a deterministic compaction receipt after the summary noting how
    much history was summarized, that the summary is secondhand, and -- when a
    ``TranscriptHandleProvider`` capability is attached -- a persisted-run handle.

    Opt-in for now: the receipt text is content, so defaulting it on is deferred to the
    benchmark eval-rig pass.  The mechanism itself is structural.
    """

    # Override the inherited default ID because durable-operation recovery needs a stable identity.
    _: KW_ONLY
    id: str | None = 'summarizing_compaction'

    def __post_init__(self) -> None:
        if self.max_messages is None and self.max_tokens is None and self.max_fraction is None:
            raise ValueError('At least one of max_messages, max_tokens, or max_fraction must be set.')
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError('max_messages must be positive.')
        validate_token_trigger(self.max_tokens, self.max_fraction, self.fallback_context_window, self.context_window)
        if self.keep_messages < 0:
            raise ValueError('keep_messages must be non-negative.')
        if self.keep_tokens is not None and self.keep_tokens < 0:
            raise ValueError('keep_tokens must be non-negative.')
        if self.keep_user_messages_max_chars < 1:
            raise ValueError('keep_user_messages_max_chars must be positive.')

    def with_focus(self, focus: str) -> SummarizingCompaction[AgentDepsT]:
        """Return a copy whose summary prompt prioritizes `focus`.

        Used by `compact_now` so a user-invoked compaction can say what the summary must not
        lose. The prompt is later run through `str.format`, so braces in a user- or
        model-supplied focus are escaped to survive it.
        """
        escaped = focus.replace('{', '{{').replace('}', '}}')
        return replace(self, summary_prompt=f'{self.summary_prompt}\n\nGive particular weight to: {escaped}')

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Summarize older messages, replacing them with a single summary message."""
        if self.keep_tokens is not None:
            cutoff = find_token_cutoff(messages, self.keep_tokens, self.tokenizer)
        else:
            cutoff = find_safe_cutoff(messages, self.keep_messages)

        if cutoff <= 0:
            return messages

        system_parts = _extract_system_prompts(messages)
        to_summarize = messages[:cutoff]
        preserved = messages[cutoff:]

        previous_summary = _extract_previous_summary(messages) if self.incremental else None
        summary = await self._summarize(to_summarize, ctx, previous_summary=previous_summary)
        summary = self._maybe_bridge_prefix(summary, messages, ctx)

        summary_part = SystemPromptPart(content=f'{_SUMMARY_PREFIX}{summary}')
        summary_message = ModelRequest(parts=[*system_parts, summary_part])

        extra: list[ModelMessage] = []
        if self.keep_user_messages:
            extra = self._kept_user_messages(to_summarize)
            extra = extra[-self.keep_messages :] if self.keep_messages else []
            token_tail_budget = self.keep_tokens
            if token_tail_budget is not None:
                retained: list[ModelMessage] = []
                for message in reversed(extra):
                    tokens = estimate_token_count([message], self.tokenizer)
                    if tokens <= token_tail_budget:
                        retained.append(message)
                        token_tail_budget -= tokens
                    else:
                        break
                extra = list(reversed(retained))
            retained_tail_slots = self.keep_messages - len(extra)
            if token_tail_budget is not None:
                if token_tail_budget == 0:
                    preserved = []
                else:
                    token_tail = preserved[find_token_cutoff(preserved, token_tail_budget, self.tokenizer) :]
                    preserved = (
                        token_tail if estimate_token_count(token_tail, self.tokenizer) <= token_tail_budget else []
                    )
            if len(preserved) > retained_tail_slots:
                preserved = preserved[find_safe_cutoff(preserved, retained_tail_slots) :]
        elif self.preserve_first_user_message:
            first_user_msg = find_first_user_message(messages)
            if first_user_msg is not None:
                idx = messages.index(first_user_msg)
                if idx < cutoff and first_user_msg not in preserved:
                    extra = [first_user_msg]

        result: list[ModelMessage] = [summary_message, *extra, *preserved]
        result = reinject_pinned(messages, result)
        if self.receipts:
            result = self._insert_receipt(summary_message, to_summarize, result, ctx)
        return result

    def _kept_user_messages(self, to_summarize: list[ModelMessage]) -> list[ModelMessage]:
        """Rebuild each summarized user turn as a preserved, size-bounded user message."""
        out: list[ModelMessage] = []
        for msg in to_summarize:
            if not isinstance(msg, ModelRequest):
                continue
            if _is_kept_user_message(msg):
                out.append(msg)
                continue
            user_parts = [
                part
                for part in msg.parts
                if isinstance(part, UserPromptPart) and not is_pinned(part) and not is_receipt_part(part)
            ]
            if not user_parts:
                continue
            kept: list[ModelRequestPart] = []
            for part in user_parts:
                if isinstance(part.content, str):
                    truncated = self._truncate(part.content)
                    kept.append(part if truncated == part.content else replace(part, content=truncated))
                else:
                    bounded, changed = self._bound_sequence(part.content)
                    kept.append(replace(part, content=bounded) if changed else part)
            metadata = {**(msg.metadata or {}), _KEPT_USER_MESSAGE_METADATA: True}
            out.append(replace(msg, parts=kept, metadata=metadata))
        return out

    def _truncate(self, text: str, max_chars: int | None = None) -> str:
        from pydantic_ai_harness.tool_output_limits import TruncationStrategy
        from pydantic_ai_harness.tool_output_limits._payload import truncate_text

        limit = self.keep_user_messages_max_chars if max_chars is None else max_chars
        truncated = truncate_text(text, limit, TruncationStrategy.head)
        if len(truncated) <= limit:
            return truncated
        marker = '[...]'
        if limit <= len(marker):
            return marker[:limit]
        return f'{text[: limit - len(marker)]}{marker}'

    def _bound_sequence(self, content: Sequence[UserContent]) -> tuple[list[UserContent], bool]:
        """Apply the same per-part character budget to a sequence-shaped user prompt.

        `keep_user_messages_max_chars` bounds the whole part, so the budget is shared across the
        sequence's text-bearing items rather than granted to each one. Non-text items (images,
        audio, cache points) pass through: they carry no characters to cut, and rewriting them
        would change what the model sees rather than shrink it.
        """
        remaining = self.keep_user_messages_max_chars
        bounded: list[UserContent] = []
        changed = False
        for item in content:
            text = item if isinstance(item, str) else item.content if isinstance(item, TextContent) else None
            if text is None:
                bounded.append(item)
                continue
            if remaining == 0:
                changed = True
                continue
            truncated = self._truncate(text, remaining)
            remaining -= len(truncated)
            if truncated == text:
                bounded.append(item)
                continue
            changed = True
            bounded.append(truncated if isinstance(item, str) else replace(item, content=truncated))
        return bounded, changed

    def _maybe_bridge_prefix(self, summary: str, messages: list[ModelMessage], ctx: RunContext[AgentDepsT]) -> str:
        """Prefix *summary* with a cross-model handoff note when families differ."""
        if not self.bridge_prefix:
            return summary
        run_family = _model_family(_history_model_name(messages) or ctx.model)
        summarizer_family = _model_family(self.model if self.model is not None else ctx.model)
        if run_family and summarizer_family and run_family != summarizer_family:
            return f'{_BRIDGE_PREFIX}\n\n{_without_bridge_prefix(summary)}'
        return summary

    def _insert_receipt(
        self,
        summary_message: ModelRequest,
        to_summarize: list[ModelMessage],
        result: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Insert a deterministic receipt right after the summary, de-accumulating old ones."""
        deduped = [msg for msg in result if not _is_receipt_message(msg)]
        dropped_tokens = estimate_token_count(to_summarize, self.tokenizer)
        handle = discover_transcript_handle(ctx)
        summarizer = _model_family(self.model if self.model is not None else ctx.model)
        by = summarizer or 'the summarizer model'
        record_receipt(
            ReceiptInfo(
                strategy='SummarizingCompaction',
                dropped_messages=len(to_summarize),
                dropped_tokens=dropped_tokens,
                by=by,
                handle=handle,
            )
        )
        text = format_receipt(
            dropped_messages=len(to_summarize),
            dropped_tokens=dropped_tokens,
            by=by,
            handle=handle,
            has_summary=True,
        )
        receipt_message = ModelRequest(parts=[make_receipt_part(text)])
        anchor = deduped.index(summary_message)
        return [*deduped[: anchor + 1], receipt_message, *deduped[anchor + 1 :]]

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Summarize older messages when the threshold is exceeded."""
        messages: list[ModelMessage] = list(request_context.messages)
        request_ctx = context_for_request(ctx, request_context)
        token_trigger = resolve_token_trigger(
            self.max_tokens, self.max_fraction, request_ctx.model, self.fallback_context_window, self.context_window
        )
        if not exceeds(
            messages,
            self.max_messages,
            token_trigger,
            self.tokenizer,
            model_request_parameters=request_context.model_request_parameters,
        ):
            return request_context
        compacted = await compact_with_span(
            request_ctx,
            strategy='SummarizingCompaction',
            messages=messages,
            compact=lambda: self.compact(messages, request_ctx),
            tokenizer=self.tokenizer,
        )
        record_compaction_reclaim(
            request_context,
            estimate_token_count(messages, self.tokenizer),
            estimate_token_count(compacted, self.tokenizer),
        )
        request_context.messages = compacted
        return request_context

    @durable_operation('summarize')
    async def _summarize(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
        *,
        previous_summary: str | None = None,
    ) -> str:
        """Generate a summary for the given messages using the configured model."""
        from pydantic_ai import Agent

        formatted = _format_messages(messages, skip_previous_summary=previous_summary is not None)
        prompt = self.summary_prompt.format(messages=formatted)

        if previous_summary is not None:
            prompt = (
                f'{prompt}\n\n{_INCREMENTAL_UPDATE_INSTRUCTION}\n\n'
                f'<previous-summary>\n{previous_summary}\n</previous-summary>'
            )

        model = self.model if self.model is not None else ctx.model
        # `ctx.model` is an `AbstractModel`; summarization needs a request-response model. A
        # realtime run reaches here only when no summarizer `model=` was configured, so ask for
        # one explicitly rather than handing `Agent` a model it cannot run with.
        if is_realtime_model(model):
            raise UserError(
                'SummarizingCompaction needs a request-response model to write the summary, but '
                f'the run uses {type(model).__name__}, which is not one. Set `model=` on '
                'SummarizingCompaction to the model to summarize with when the run uses a realtime model.'
            )
        # `isinstance` narrows the generic `Model` to `Model[Unknown]`; `cast` recovers
        # `Model[Any]`, mirroring core's own `reinject_system_prompt` idiom.
        agent: Agent[None, str] = Agent(
            cast('Model[Any] | str', model),
            instructions=self.instructions,
            model_settings=self.model_settings,
        )
        result = await agent.run(prompt, usage=ctx.usage, usage_limits=reserved_usage_limits(ctx.usage_limits))
        return result.output.strip()
