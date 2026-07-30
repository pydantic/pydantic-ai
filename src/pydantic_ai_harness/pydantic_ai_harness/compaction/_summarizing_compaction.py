"""`SummarizingCompaction` -- LLM-powered summarization of older messages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._shared import (
    compact_with_span,
    context_for_request,
    exceeds,
    find_first_user_message,
    find_safe_cutoff,
    find_token_cutoff,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model, ModelRequestContext

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

_SUMMARY_PREFIX = 'Summary of previous conversation:\n\n'


def _format_messages(messages: Sequence[ModelMessage]) -> str:
    """Render messages into a human-readable string for summarization."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    lines.append(f'User: {_user_prompt_text(part)}')
                elif isinstance(part, SystemPromptPart):
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
            if isinstance(part, SystemPromptPart):
                parts.append(part)
            else:
                return parts
    return parts


def _extract_previous_summary(messages: list[ModelMessage]) -> str | None:
    """Extract the most recent compaction summary from the message history.

    Looks for a ``SystemPromptPart`` whose content starts with the summary prefix,
    which indicates it was produced by a prior compaction pass.
    """
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if isinstance(part, SystemPromptPart) and part.content.startswith(_SUMMARY_PREFIX):
                return part.content[len(_SUMMARY_PREFIX) :]
    return None


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
    """When ``True``, include any existing summary from a prior compaction in the
    summarization prompt so that it is extended rather than regenerated from scratch.
    """

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

        summary_part = SystemPromptPart(content=f'{_SUMMARY_PREFIX}{summary}')
        summary_message = ModelRequest(parts=[*system_parts, summary_part])

        first_user: list[ModelMessage] = []
        if self.preserve_first_user_message:
            first_user_msg = find_first_user_message(messages)
            if first_user_msg is not None:
                idx = messages.index(first_user_msg)
                if idx < cutoff and first_user_msg not in preserved:
                    first_user = [first_user_msg]

        return [summary_message, *first_user, *preserved]

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
        if not exceeds(messages, self.max_messages, token_trigger, self.tokenizer):
            return request_context
        request_context.messages = await compact_with_span(
            request_ctx,
            strategy='SummarizingCompaction',
            messages=messages,
            compact=lambda: self.compact(messages, request_ctx),
            tokenizer=self.tokenizer,
        )
        return request_context

    async def _summarize(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
        *,
        previous_summary: str | None = None,
    ) -> str:
        """Generate a summary for the given messages using the configured model."""
        from pydantic_ai import Agent

        formatted = _format_messages(messages)
        prompt = self.summary_prompt.format(messages=formatted)

        if previous_summary is not None:
            prompt = f'{prompt}\n\n<previous_summary>\n{previous_summary}\n</previous_summary>'

        model = self.model if self.model is not None else ctx.model
        agent: Agent[None, str] = Agent(
            model,
            instructions='You are a context summarization assistant. Extract the most important information from conversations.',
        )
        result = await agent.run(prompt, usage=ctx.usage)
        return result.output.strip()
