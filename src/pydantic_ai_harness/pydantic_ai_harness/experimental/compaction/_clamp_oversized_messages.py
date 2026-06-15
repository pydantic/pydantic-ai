"""`ClampOversizedMessages` -- zero-cost head/tail truncation of a single oversized part."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.experimental.compaction._shared import estimate_text_tokens

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


_CLAMP_MARKER = '\n[clamped: removed {removed} of {original} characters]\n'
"""Inserted between the head and tail slices of a clamped part. ``{removed}`` and ``{original}``
are filled with character counts."""

_CLAMP_ARGS_KEY = '_clamped'
"""Key of the single-entry object a clamped `ToolCallPart`'s args are replaced with.

Args stay a JSON object (not a bare marker string) so `args_as_json_str()` emits valid function
arguments for the provider."""


@dataclass
class ClampOversizedMessages(AbstractCapability[AgentDepsT]):
    """Zero-cost head/tail truncation of any single oversized message part.

    A runaway generation -- a model response of repeated whitespace, a giant tool-call
    payload -- can produce one part so large the next request exceeds the provider's context
    cap. The size-based strategies cannot help: `SlidingWindow` drops the *oldest* messages
    (the offender is the newest), `ClearToolResults` only touches tool *results*, and feeding
    the history to `SummarizingCompaction` hits the same cap. This strategy truncates the
    offending part in place: it keeps a head slice and a tail slice and inserts a marker for
    the removed middle. Degenerate generations are low-entropy repetition, so a head/tail
    slice loses little. No LLM calls are made.

    What it clamps, in each `ModelResponse`:

    - `TextPart` content (the critical case -- a runaway model-response text part).
    - `ToolCallPart` args, when `clamp_tool_call_args` is set (the same failure shape for a
      giant tool-call payload). The args are replaced with a small JSON object so they stay
      valid function arguments; the original call already executed, so this only shrinks the
      history copy.

    Request-side parts (user prompts, tool returns, system prompts) are out of scope: user
    input should not be silently rewritten, and oversized tool *returns* are the job of
    `ClearToolResults`.

    Clamping rewrites message content, so it invalidates the provider's prompt cache from the
    clamped message onward. That is unavoidable here -- the alternative is a failed request.

    A part is clamped only when it is oversized *and* the clamp actually shrinks it, so set
    `keep_head_chars` + `keep_tail_chars` well below your per-part threshold.

    Composes as the first tier of a `TieredCompaction` (run it before `ClearToolResults`):
    it is the only zero-LLM way to keep a run alive after a runaway generation.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.experimental.compaction import ClampOversizedMessages

        agent = Agent(
            'openai:gpt-4o',
            capabilities=[ClampOversizedMessages(max_part_tokens=50_000)],
        )
        ```
    """

    max_part_tokens: int | None = None
    """Clamp a part whose estimated token count exceeds this value. ``None`` disables this trigger."""

    max_part_chars: int | None = None
    """Clamp a part whose character count exceeds this value. ``None`` disables this trigger."""

    keep_head_chars: int = 2_000
    """Characters of the part's head to retain."""

    keep_tail_chars: int = 2_000
    """Characters of the part's tail to retain."""

    clamp_tool_call_args: bool = True
    """When ``True``, also clamp oversized `ToolCallPart` args, not just response text."""

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When ``None``, uses a ~4 characters-per-token heuristic.
    """

    def __post_init__(self) -> None:
        if self.max_part_tokens is None and self.max_part_chars is None:
            raise ValueError('At least one of max_part_tokens or max_part_chars must be set.')
        if self.max_part_tokens is not None and self.max_part_tokens < 1:
            raise ValueError('max_part_tokens must be positive.')
        if self.max_part_chars is not None and self.max_part_chars < 1:
            raise ValueError('max_part_chars must be positive.')
        if self.keep_head_chars < 0:
            raise ValueError('keep_head_chars must be non-negative.')
        if self.keep_tail_chars < 0:
            raise ValueError('keep_tail_chars must be non-negative.')

    def _is_oversized(self, text: str) -> bool:
        if self.max_part_chars is not None and len(text) > self.max_part_chars:
            return True
        if self.max_part_tokens is not None and estimate_text_tokens(text, self.tokenizer) > self.max_part_tokens:
            return True
        return False

    def _clamp(self, text: str) -> str | None:
        """Return the head/tail-clamped form of *text*, or ``None`` if it would not shrink."""
        if not self._is_oversized(text):
            return None
        head = text[: self.keep_head_chars]
        tail = text[len(text) - self.keep_tail_chars :] if self.keep_tail_chars else ''
        removed = len(text) - len(head) - len(tail)
        clamped = head + _CLAMP_MARKER.format(removed=removed, original=len(text)) + tail
        if len(clamped) >= len(text):
            return None
        return clamped

    async def compact(
        self,
        messages: list[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> list[ModelMessage]:
        """Clamp every oversized response text part (and tool-call args, if enabled)."""
        out: list[ModelMessage] = []
        for msg in messages:
            if not isinstance(msg, ModelResponse):
                out.append(msg)
                continue

            new_parts: list[ModelResponsePart] = []
            changed = False
            for part in msg.parts:
                if isinstance(part, TextPart):
                    clamped = self._clamp(part.content)
                    if clamped is not None:
                        new_parts.append(replace(part, content=clamped))
                        changed = True
                        continue
                elif isinstance(part, ToolCallPart) and self.clamp_tool_call_args:
                    clamped = self._clamp(part.args_as_json_str())
                    if clamped is not None:
                        new_parts.append(replace(part, args={_CLAMP_ARGS_KEY: clamped}))
                        changed = True
                        continue
                new_parts.append(part)
            out.append(replace(msg, parts=new_parts) if changed else msg)
        return out

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Clamp any oversized response part before the request is sent."""
        request_context.messages = await self.compact(list(request_context.messages), ctx)
        return request_context
