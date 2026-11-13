"""`ClearToolResults` -- zero-cost in-place clearing of old tool results."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._shared import (
    ToolPair,
    compact_with_span,
    context_for_request,
    estimate_token_count,
    exceeds,
    is_cleared_by_placeholder_function,
    is_cleared_return,
    iter_tool_pairs,
    rebuild_with_cleared,
    record_compaction_reclaim,
    resolve_token_trigger,
    user_turn_starts,
    validate_token_trigger,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


@dataclass
class ClearToolResults(AbstractCapability[AgentDepsT]):
    """Zero-cost in-place clearing of old tool results.

    Replaces the content of the oldest tool *results* with a short placeholder while
    keeping the most recent `keep_pairs` tool-call / tool-return pairs, or the results of the
    last `keep_turns` user turns, intact.  Tool calls remain paired with their (now-blanked)
    results, so the history stays valid. No LLM calls are made.

    This is the cheap first tier of compaction -- tool results typically dominate
    context, and the agent can re-run a tool if it needs the data again.

    Cache tradeoff: clearing rewrites message content, which invalidates the provider's
    prompt cache from the clear point onward (the next request pays a cache-write).  Use
    `min_clear_tokens` to skip clearing that reclaims too little to be worth busting the
    cache.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import ClearToolResults

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[ClearToolResults(max_tokens=100_000, keep_pairs=3)],
        )
        ```
    """

    max_messages: int | None = None
    """Trigger clearing when the messages since the last cleared tool result exceed this value. `None` disables.

    Clearing blanks content but removes no messages, so a plain message count would stay above
    the threshold and clear one more result on every request, rewriting history (and busting the
    prompt cache) each step. Counting from the last cleared result instead makes clearing happen
    in batches, and the requests between two clears extend the cached prefix. Before the first clear
    the count is the whole history. The kept pairs count toward it: right after a clear about
    `2 * keep_pairs` messages follow the last cleared result, and each tool step adds two more, so
    the next clear comes after about `(max_messages - 2 * keep_pairs) / 2` tool steps. At or below
    `2 * keep_pairs` it clears on every request. With `keep_turns`, the messages of the kept turns
    count toward it the same way."""

    max_tokens: int | None = None
    """Trigger clearing when estimated token count exceeds this value. `None` disables."""

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

    keep_pairs: int = 3
    """Number of most-recent tool-call / tool-return pairs left untouched.

    `keep_turns` replaces the pair count when set; setting both to a `keep_pairs` other than the
    default raises a `UserError`."""

    placeholder: str | Callable[[ToolReturnPart], str] = '[tool result cleared]'
    """Replacement content for a cleared tool result, or a function that builds it from the result.

    A function is called when the result is cleared, and its return value is stored as the result's
    content; an already-cleared result is not passed to it again. With `min_clear_tokens`, it's called
    before the reclaim is measured, so a clear that reclaims too little is dropped and the function
    is called for the same results again on a later request. It must be deterministic:
    the same result has to produce the same text on every run, or a history cleared again from its
    original (a resumed or replayed run) is sent with different content and the provider's prompt
    cache misses from that point on. A result whose `metadata` is something other than `None` or a
    `dict` is not cleared with a function, because the mark that records the clear can't be stored.

    A function can't be expressed in an agent spec, so a spec only accepts a string here."""

    exclude_tools: frozenset[str] = frozenset()
    """Tool names whose results are never cleared. Mutually exclusive with `tools`."""

    clear_tool_inputs: bool = False
    """When `True`, also blank the arguments of the cleared tool calls."""

    min_clear_tokens: int | None = None
    """Only clear if doing so reclaims at least this many estimated tokens.

    Protects the prompt cache from being invalidated for a trivial gain. `None` always clears.
    """

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When `None`, uses a ~4 characters-per-token heuristic.
    """

    tools: Sequence[str] | None = field(default=None, kw_only=True)
    """Tool names whose results are cleared; the results of other tools are kept. `None` clears every tool's.

    When set, `keep_pairs` counts only the listed tools' pairs, so `tools=['retrieve'], keep_pairs=2`
    keeps the newest two `retrieve` results whatever other tools ran in between. With several listed
    tools, the window is shared across them. Mutually exclusive with `exclude_tools`."""

    keep_turns: int | None = field(default=None, kw_only=True)
    """Keep the tool results of the last `keep_turns` completed user turns, plus the current one.

    A user turn starts at a request carrying the user's prompt, and runs until the next one. The
    current turn, the one the model is answering, is never cleared, so `0` clears the results of
    every earlier turn. Prompts added by harness capabilities, such as pinned messages and
    compaction receipts, don't start a turn, and neither does user content sent with tool results,
    such as a `ToolReturn`'s `content`. Replaces the `keep_pairs` count when set."""

    def __post_init__(self) -> None:
        if isinstance(self.tools, str):
            raise UserError(f'`tools` takes a sequence of tool names, not a string: use `tools=[{self.tools!r}]`.')
        if self.tools is not None and self.exclude_tools:
            raise UserError('Set at most one of `tools` or `exclude_tools`.')
        # The field's own default, read off the class, so the two can't drift apart.
        if self.keep_turns is not None and self.keep_pairs != ClearToolResults.keep_pairs:
            raise UserError('Set at most one of `keep_pairs` or `keep_turns`.')
        if self.max_messages is None and self.max_tokens is None and self.max_fraction is None:
            raise ValueError('At least one of max_messages, max_tokens, or max_fraction must be set.')
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError('max_messages must be positive.')
        validate_token_trigger(self.max_tokens, self.max_fraction, self.fallback_context_window, self.context_window)
        if self.keep_pairs < 0:
            raise ValueError('keep_pairs must be non-negative.')
        if self.keep_turns is not None and self.keep_turns < 0:
            raise ValueError('keep_turns must be non-negative.')
        if self.min_clear_tokens is not None and self.min_clear_tokens < 0:
            raise ValueError('min_clear_tokens must be non-negative.')

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Blank the oldest tool results beyond the most recent `keep_pairs`, or before the last `keep_turns`."""
        older, kept = messages, list[ModelMessage]()
        if self.keep_turns is None:
            clearable = self._outside_kept_pairs(messages)
        else:
            # Only the messages before the kept turns are rebuilt, so a tool call id a provider reuses
            # in a kept turn can't reach back into it.
            boundary = self._kept_turns_start(messages, self.keep_turns)
            older, kept = messages[:boundary], messages[boundary:]
            clearable = iter_tool_pairs(older)

        clear_return_ids: set[str] = set()
        clear_input_ids: set[str] = set()
        for pair in clearable:
            if pair.tool_name in self.exclude_tools or (self.tools is not None and pair.tool_name not in self.tools):
                continue
            clear_return_ids.add(pair.tool_call_id)
            if self.clear_tool_inputs:
                clear_input_ids.add(pair.tool_call_id)

        if not clear_return_ids:
            return messages

        cleared = [*rebuild_with_cleared(older, clear_return_ids, clear_input_ids, self.placeholder), *kept]
        if self.min_clear_tokens is not None:
            reclaimed = estimate_token_count(messages, self.tokenizer) - estimate_token_count(cleared, self.tokenizer)
            if reclaimed < self.min_clear_tokens:
                return messages
        return cleared

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Clear old tool results if the conversation exceeds the configured threshold."""
        messages: list[ModelMessage] = list(ctx.messages)
        request_ctx = context_for_request(ctx, request_context)
        token_trigger = resolve_token_trigger(
            self.max_tokens, self.max_fraction, request_ctx.model, self.fallback_context_window, self.context_window
        )
        if not self._exceeds_max_messages(messages) and not exceeds(
            messages,
            None,
            token_trigger,
            self.tokenizer,
            model_request_parameters=request_context.model_request_parameters,
        ):
            return request_context
        compacted = await compact_with_span(
            request_ctx,
            strategy='ClearToolResults',
            messages=messages,
            compact=lambda: self.compact(messages, request_ctx),
            tokenizer=self.tokenizer,
        )
        ctx.messages[:] = compacted
        record_compaction_reclaim(
            ctx,
            estimate_token_count(messages, self.tokenizer),
            estimate_token_count(compacted, self.tokenizer),
        )
        return replace(request_context, messages=compacted)

    def _outside_kept_pairs(self, messages: list[ModelMessage]) -> list[ToolPair]:
        """The pairs older than the newest `keep_pairs`, matched by tool call id across the whole history.

        With `tools`, the window holds only the listed tools' pairs, so other tools' calls can't push a
        listed result out of it.
        """
        pairs = iter_tool_pairs(messages)
        if self.tools is not None:
            pairs = [pair for pair in pairs if pair.tool_name in self.tools]
        return pairs[: max(0, len(pairs) - self.keep_pairs)]

    @staticmethod
    def _kept_turns_start(messages: list[ModelMessage], keep_turns: int) -> int:
        """Index of the first message in the last `keep_turns` completed user turns and the current one.

        A history without a user turn is all current turn, so nothing in it is cleared. See
        `user_turn_starts` for which requests start a turn.
        """
        starts = user_turn_starts(messages)
        if len(starts) <= keep_turns:
            return 0
        return starts[-(keep_turns + 1)]

    def _is_cleared(self, part: ToolReturnPart) -> bool:
        if isinstance(self.placeholder, str):
            return is_cleared_return(part, self.placeholder)
        return is_cleared_by_placeholder_function(part)

    def _exceeds_max_messages(self, messages: list[ModelMessage]) -> bool:
        if self.max_messages is None:
            return False
        for since_last_clear, message in enumerate(reversed(messages)):
            if isinstance(message, ModelRequest) and any(
                isinstance(part, ToolReturnPart) and self._is_cleared(part) for part in message.parts
            ):
                break
        else:
            since_last_clear = len(messages)
        return since_last_clear > self.max_messages
