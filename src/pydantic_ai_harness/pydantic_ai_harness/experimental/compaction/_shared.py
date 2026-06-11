"""Shared utilities for the compaction capabilities.

Token estimation, the `CompactionStrategy` protocol, tool-pair-safe cutoff logic, first-user
preservation, and in-place tool-result clearing -- anything used by more than one capability.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import RunContext

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN = 4
"""Rough approximation: ~4 characters per token on average."""


def _collect_text(messages: Sequence[ModelMessage]) -> list[str]:
    """Collect all text segments from a sequence of messages."""
    segments: list[str] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    segments.append(_user_prompt_text_for_counting(part))
                elif isinstance(part, SystemPromptPart):
                    segments.append(part.content)
                elif isinstance(part, ToolReturnPart):
                    segments.append(str(part.content))
        else:
            for part in msg.parts:
                if isinstance(part, TextPart):
                    segments.append(part.content)
                elif isinstance(part, ToolCallPart):
                    segments.append(part.tool_name)
                    segments.append(str(part.args))
    return segments


def _user_prompt_text_for_counting(part: UserPromptPart) -> str:
    """Extract text content from a user prompt part for counting."""
    if isinstance(part.content, str):
        return part.content
    texts: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
    return ''.join(texts)


def estimate_token_count(
    messages: Sequence[ModelMessage],
    tokenizer: Callable[[str], int] | None = None,
) -> int:
    """Approximate token count for a sequence of messages.

    Args:
        messages: Messages to count tokens for.
        tokenizer: Optional callable that returns the token count for a string.
            When ``None``, falls back to a ~4 characters-per-token heuristic.
    """
    segments = _collect_text(messages)
    if tokenizer is not None:
        return sum(tokenizer(s) for s in segments)
    return sum(len(s) for s in segments) // _CHARS_PER_TOKEN


def exceeds(
    messages: Sequence[ModelMessage],
    max_messages: int | None,
    max_tokens: int | None,
    tokenizer: Callable[[str], int] | None,
) -> bool:
    """Return True if *messages* exceeds either configured size threshold."""
    if max_messages is not None and len(messages) > max_messages:
        return True
    if max_tokens is not None and estimate_token_count(messages, tokenizer) > max_tokens:
        return True
    return False


# ---------------------------------------------------------------------------
# Compaction strategy protocol
# ---------------------------------------------------------------------------


class CompactionStrategy(Protocol[AgentDepsT]):
    """A history transform that can be used standalone or as a `TieredCompaction` tier.

    ``compact`` applies the transform *unconditionally* (the trigger check lives in the
    capability's ``before_model_request``).  Implementations must preserve tool-call /
    tool-return pairing.
    """

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Safe cutoff logic -- preserves tool-call / tool-return pairs
# ---------------------------------------------------------------------------

_TOOL_PAIR_SEARCH_RANGE = 5
"""Number of messages to search around a cutoff point for tool-call pairs."""


def _is_safe_cutoff(
    messages: list[ModelMessage],
    cutoff: int,
    search_range: int = _TOOL_PAIR_SEARCH_RANGE,
) -> bool:
    """Return True if cutting at *cutoff* does not orphan any tool-call pair.

    A tool-call pair is a ``ToolCallPart`` in a ``ModelResponse`` together with
    the corresponding ``ToolReturnPart`` in a subsequent ``ModelRequest``.  Both
    sides must end up on the same side of the cut.
    """
    if cutoff >= len(messages):
        return True

    start = max(0, cutoff - search_range)
    end = min(len(messages), cutoff + search_range)

    for i in range(start, end):
        msg = messages[i]
        if not isinstance(msg, ModelResponse):
            continue

        call_ids: set[str] = set()
        for part in msg.parts:
            if isinstance(part, ToolCallPart) and part.tool_call_id:
                call_ids.add(part.tool_call_id)

        if not call_ids:
            continue

        for j in range(i + 1, len(messages)):
            later = messages[j]
            if not isinstance(later, ModelRequest):
                continue
            for rpart in later.parts:
                if isinstance(rpart, ToolReturnPart) and rpart.tool_call_id in call_ids:
                    call_before = i < cutoff
                    return_before = j < cutoff
                    if call_before != return_before:
                        return False

    return True


def find_safe_cutoff(messages: list[ModelMessage], keep: int) -> int:
    """Find a cutoff index that keeps *keep* tail messages without splitting tool pairs.

    Returns 0 if trimming is unnecessary (fewer messages than *keep*).
    """
    if keep == 0:
        return len(messages)
    if len(messages) <= keep:
        return 0

    target = len(messages) - keep
    for idx in range(target, -1, -1):
        if _is_safe_cutoff(messages, idx):
            return idx
    return 0  # pragma: no cover


def find_token_cutoff(
    messages: list[ModelMessage],
    target_tokens: int,
    tokenizer: Callable[[str], int] | None = None,
) -> int:
    """Binary-search for a cutoff such that ``messages[cutoff:]`` fits in *target_tokens*.

    Adjusts the result so that no tool-call pairs are orphaned.
    """
    if not messages or estimate_token_count(messages, tokenizer) <= target_tokens:
        return 0

    lo, hi = 0, len(messages)
    candidate = len(messages)

    while lo < hi:
        mid = (lo + hi) // 2
        if estimate_token_count(messages[mid:], tokenizer) <= target_tokens:
            candidate = mid
            hi = mid
        else:
            lo = mid + 1

    if candidate >= len(messages):
        candidate = max(0, len(messages) - 1)  # pragma: no cover

    # Walk backward to a safe point.
    for idx in range(candidate, -1, -1):
        if _is_safe_cutoff(messages, idx):
            return idx
    return 0  # pragma: no cover


# ---------------------------------------------------------------------------
# First user message preservation
# ---------------------------------------------------------------------------


def find_first_user_message(messages: list[ModelMessage]) -> ModelRequest | None:
    """Return the first ``ModelRequest`` that contains a ``UserPromptPart``, or ``None``."""
    for msg in messages:
        if isinstance(msg, ModelRequest) and any(isinstance(p, UserPromptPart) for p in msg.parts):
            return msg
    return None


def prepend_first_user_message(
    original: list[ModelMessage],
    cutoff: int,
    trimmed: list[ModelMessage],
) -> list[ModelMessage]:
    """Ensure the first user message from *original* appears in *trimmed*.

    If the first ``ModelRequest`` containing a ``UserPromptPart`` in *original*
    was discarded (its index is before *cutoff*) and is not already in *trimmed*,
    prepend it.
    """
    first = find_first_user_message(original)
    if first is None:
        return trimmed
    idx = original.index(first)
    if idx < cutoff and first not in trimmed:
        return [first, *trimmed]
    return trimmed


# ---------------------------------------------------------------------------
# Tool-pair inspection and in-place clearing
# ---------------------------------------------------------------------------


_CLEARED_TOOL_ARGS = '{}'
"""Replacement for cleared tool-call arguments.

Kept JSON-valid: ``ToolCallPart.args_as_json_str()`` returns a ``str`` arg verbatim, so a
non-JSON placeholder would reach the provider as malformed function arguments.
"""


@dataclass(frozen=True)
class _ToolPair:
    """A matched tool call and its return, with the order the return appeared."""

    tool_call_id: str
    tool_name: str
    call_part: ToolCallPart
    order: int


def iter_tool_pairs(messages: Sequence[ModelMessage]) -> list[_ToolPair]:
    """Return matched tool-call / tool-return pairs in return-appearance order."""
    calls: dict[str, ToolCallPart] = {}
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_call_id:
                    calls[part.tool_call_id] = part

    pairs: list[_ToolPair] = []
    order = 0
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_call_id in calls:
                    call = calls[part.tool_call_id]
                    pairs.append(_ToolPair(part.tool_call_id, call.tool_name, call, order))
                    order += 1
    return pairs


def rebuild_with_cleared(
    messages: Sequence[ModelMessage],
    clear_return_ids: set[str],
    clear_input_ids: set[str],
    placeholder: str,
) -> list[ModelMessage]:
    """Return *messages* with selected tool results (and optionally inputs) blanked.

    The ``ToolReturnPart`` / ``ToolCallPart`` are kept in place with placeholder content,
    so tool-call pairing is never broken.  Already-blanked parts are left untouched.
    """
    out: list[ModelMessage] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            request_parts: list[ModelRequestPart] = []
            changed = False
            for part in msg.parts:
                if (
                    isinstance(part, ToolReturnPart)
                    and part.tool_call_id in clear_return_ids
                    and str(part.content) != placeholder
                ):
                    request_parts.append(replace(part, content=placeholder))
                    changed = True
                else:
                    request_parts.append(part)
            out.append(replace(msg, parts=request_parts) if changed else msg)
        else:
            response_parts: list[ModelResponsePart] = []
            changed = False
            for part in msg.parts:
                if (
                    isinstance(part, ToolCallPart)
                    and part.tool_call_id in clear_input_ids
                    and part.args != _CLEARED_TOOL_ARGS
                ):
                    response_parts.append(replace(part, args=_CLEARED_TOOL_ARGS))
                    changed = True
                else:
                    response_parts.append(part)
            out.append(replace(msg, parts=response_parts) if changed else msg)
    return out
