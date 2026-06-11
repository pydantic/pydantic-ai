"""`DeduplicateFileReads` -- zero-cost in-place clearing of superseded file reads."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai._run_context import AgentDepsT
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.experimental.compaction._shared import exceeds, iter_tool_pairs, rebuild_with_cleared

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
        from pydantic_ai_harness.experimental.compaction import DeduplicateFileReads


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

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for accurate token counting.

    A callable that returns the token count for a given string.
    When ``None``, uses a ~4 characters-per-token heuristic.
    """

    def __post_init__(self) -> None:
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError('max_messages must be positive.')
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError('max_tokens must be positive.')

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
        if self.max_messages is not None or self.max_tokens is not None:
            if not exceeds(messages, self.max_messages, self.max_tokens, self.tokenizer):
                return request_context
        request_context.messages = await self.compact(messages, ctx)
        return request_context
