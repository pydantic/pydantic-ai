"""`DeduplicateFileReads` -- zero-cost in-place clearing of superseded file reads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._shared import (
    compact_with_span,
    context_for_request,
    estimate_token_count,
    exceeds,
    iter_tool_pairs,
    rebuild_with_cleared,
    record_compaction_reclaim,
    resolve_token_trigger,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass
class DeduplicateFileReads(AbstractCapability[AgentDepsT]):
    """Zero-cost in-place clearing of superseded file reads.

    When the same file is read more than once, only the latest read keeps its content;
    earlier reads are blanked with a placeholder.  Tool-call pairing is preserved.  No LLM
    calls are made.

    File identity is supplied by the ``file_key`` seam -- given a ``ToolCallPart`` it returns
    a stable key for the file being read, or ``None`` if the call is not a file read.  There
    is no default: file-read identification is agent-specific, and a wrong guess would drop
    live data.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.messages import ToolCallPart
        from pydantic_ai_harness.compaction import DeduplicateFileReads

        def file_key(call: ToolCallPart) -> str | None:
            if call.tool_name != 'read_file':
                return None
            args = call.args_as_dict()
            return args.get('path')

        agent = Agent('openai:gpt-4o', capabilities=[DeduplicateFileReads(file_key=file_key)])
        ```
    """

    file_key: Callable[[ToolCallPart], str | None]
    """Map a tool call to a stable file key, or ``None`` if it is not a file read."""

    placeholder: str = '[superseded file read]'
    """Replacement content for a superseded file read."""

    max_messages: int | None = None
    """Optional message-count trigger. When both triggers are ``None``, runs whenever invoked."""

    max_tokens: int | None = None
    """Optional token-count trigger. When both triggers are ``None``, runs whenever invoked."""

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

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When ``None``, uses a ~4 characters-per-token heuristic.
    """

    def __post_init__(self) -> None:
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError('max_messages must be positive.')
        validate_token_trigger(self.max_tokens, self.max_fraction, self.fallback_context_window, self.context_window)

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Blank every file read that is later superseded by a newer read of the same file."""
        pairs = iter_tool_pairs(messages)
        keys: list[str | None] = []
        latest_order: dict[str, int] = {}
        for pair in pairs:
            key = self.file_key(pair.call_part)
            keys.append(key)
            if key is not None:
                latest_order[key] = pair.order

        clear_return_ids: set[str] = set()
        for pair, key in zip(pairs, keys):
            if key is not None and latest_order[key] != pair.order:
                clear_return_ids.add(pair.tool_call_id)

        if not clear_return_ids:
            return messages
        return rebuild_with_cleared(messages, clear_return_ids, set(), self.placeholder)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Deduplicate file reads, optionally gated on a size threshold."""
        messages: list[ModelMessage] = list(request_context.messages)
        request_ctx = context_for_request(ctx, request_context)
        if self.max_messages is not None or self.max_tokens is not None or self.max_fraction is not None:
            token_trigger = resolve_token_trigger(
                self.max_tokens,
                self.max_fraction,
                request_ctx.model,
                self.fallback_context_window,
                self.context_window,
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
            strategy='DeduplicateFileReads',
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
