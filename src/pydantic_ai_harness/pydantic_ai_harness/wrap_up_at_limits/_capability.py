"""End a run with a final answer instead of `UsageLimitExceeded` when a usage limit is reached."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness._message_text import CHARS_PER_TOKEN
from pydantic_ai_harness._usage import reserved_usage_limits

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

LimitName: TypeAlias = Literal[
    'request_limit', 'tool_calls_limit', 'input_tokens_limit', 'output_tokens_limit', 'total_tokens_limit'
]
"""The [`UsageLimits`][pydantic_ai.usage.UsageLimits] field that made a request the run's last."""

DEFAULT_WRAP_UP_PROMPT = (
    'A run limit has been reached, so this is your last response and tools are unavailable. '
    'Give your final answer now with what you have, and say what is left unfinished.'
)
"""What `WrapUpAtLimits` tells the model on the final request, unless `prompt` is set."""

_ORIGIN = 'pydantic-ai-harness.wrap_up_at_limits.v1'
"""The `origin` in the `metadata` of the final prompt."""


@dataclass
class WrapUpAtLimits(AbstractCapability[AgentDepsT]):
    """End the run with a final answer instead of raising when a usage limit would stop it.

    When the run's [`UsageLimits`][pydantic_ai.usage.UsageLimits] leave room for only one more model
    request, that request is made the last: it keeps the tool definitions with `tool_choice='none'`, so
    the cached tools and instructions still match, and gets a request-scoped system prompt asking the
    model to answer now. The prompt's `metadata` holds the `'origin'`
    `'pydantic-ai-harness.wrap_up_at_limits.v1'` and the `'limit'` that triggered it, which
    [`wrapped_up_by`][pydantic_ai_harness.wrap_up_at_limits.wrapped_up_by] reads back.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import WrapUpAtLimits

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[WrapUpAtLimits()])
    ```
    """

    prompt: str = DEFAULT_WRAP_UP_PROMPT
    """What the model is told on the final request."""

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Make this request the last when the limits leave no room for another one."""
        limits = ctx.usage_limits
        messages = request_context.messages
        # A provider-resume turn ends in a suspended `ModelResponse` and continues the request before it.
        if limits is None or not messages or not isinstance(messages[-1], ModelRequest):
            return request_context
        prompt_tokens = len(self.prompt) // CHARS_PER_TOKEN
        limit = _limit_reached(ctx.usage, limits, _run_responses(messages, ctx.run_id), prompt_tokens)
        if limit is None:
            return request_context
        ctx.tracer.start_span('run wrapped up', attributes={'wrap_up_at_limits.limit': limit}).end()
        ctx.enqueue(SystemPromptPart(self.prompt, scope='request', metadata={'origin': _ORIGIN, 'limit': limit}))
        settings = ModelSettings(**(request_context.model_settings or {}))
        settings['tool_choice'] = 'none'
        return replace(request_context, model_settings=settings)


def wrapped_up_by(messages: Sequence[ModelMessage]) -> LimitName | None:
    """The limit that made `WrapUpAtLimits` cut a run short, given the run's `new_messages()`, or `None`."""
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if (limit := _wrap_up_limit(part)) is not None:
                    return limit
    return None


def _wrap_up_limit(part: ModelRequestPart) -> LimitName | None:
    """The limit on a final prompt `WrapUpAtLimits` added, or `None` for any other part."""
    if isinstance(part, SystemPromptPart) and part.metadata and part.metadata.get('origin') == _ORIGIN:
        return part.metadata['limit']
    return None


def _run_responses(messages: list[ModelMessage], run_id: str | None) -> list[ModelResponse]:
    return [message for message in messages if isinstance(message, ModelResponse) and message.run_id == run_id]


def _limit_reached(
    usage: RunUsage, limits: UsageLimits, responses: list[ModelResponse], prompt_tokens: int
) -> LimitName | None:
    """The limit that leaves room for this request but not for a final one after it, if any.

    The token use of both is projected from `responses`, the run's responses so far, plus the final
    request's `prompt_tokens`. Input grows with the history; each response's output is its own.
    """
    reserved = reserved_usage_limits(limits)
    if reserved is not None and reserved.request_limit is not None and usage.requests >= reserved.request_limit:
        return 'request_limit'
    if limits.tool_calls_limit is not None and usage.tool_calls >= limits.tool_calls_limit:
        return 'tool_calls_limit'
    projections: tuple[tuple[LimitName, int | None, int, int], ...] = (
        (
            'input_tokens_limit',
            limits.input_tokens_limit,
            usage.input_tokens,
            _next_two_requests([r.usage.input_tokens for r in responses]) + prompt_tokens,
        ),
        (
            'output_tokens_limit',
            limits.output_tokens_limit,
            usage.output_tokens,
            2 * max((r.usage.output_tokens for r in responses), default=0),
        ),
        (
            'total_tokens_limit',
            limits.total_tokens_limit,
            usage.total_tokens,
            _next_two_requests([r.usage.total_tokens for r in responses]) + prompt_tokens,
        ),
    )
    for name, limit, used, projected in projections:
        if limit is not None and used + projected > limit:
            return name
    return None


def _next_two_requests(counts: list[int]) -> int:
    """The tokens of two more requests, each growing by as much as the last response grew over the one before."""
    if not counts:
        return 0
    growth = max(0, counts[-1] - counts[-2]) if len(counts) > 1 else 0
    return 2 * counts[-1] + 3 * growth
