"""End a run with a final answer instead of `UsageLimitExceeded` when a usage limit is reached."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeVar

from pydantic_ai._capability_parts import check_operator_text
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    InstructionCapabilityPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    OperatorSource,
    OperatorText,
)
from pydantic_ai.settings import ModelSettings, ToolOrOutput
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness._message_text import CHARS_PER_TOKEN
from pydantic_ai_harness._usage import reserved_usage_limits

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

LimitName: TypeAlias = Literal[
    'request_limit',
    'tool_calls_limit',
    'input_tokens_limit',
    'output_tokens_limit',
    'total_tokens_limit',
    'cost_limit',
    'timeout',
]
"""What made a request the run's last: a [`UsageLimits`][pydantic_ai.usage.UsageLimits] field, or `'timeout'` for the
run's `timeout=` deadline."""

DEFAULT_WRAP_UP_PROMPT = (
    "This run's budget allows only this one last response, so you can't call any more tools to do further work. "
    'Give your final answer now with what you have, and say what is left unfinished.'
)
"""What `WrapUpAtLimits` tells the model on the final request, unless `prompt` is set."""

_Amount = TypeVar('_Amount', int, Decimal)


@dataclass(kw_only=True)
class WrapUpPart(InstructionCapabilityPart, namespace='wrap_up_at_limits', name='wrap_up'):
    """The request-scoped prompt `WrapUpAtLimits` adds to the final request.

    Its `content` is the capability's `prompt`, in the application's voice. Find it in history by this type.
    """

    limit: LimitName
    """The limit that made the request the last. Never sent to the model."""


@dataclass
class WrapUpAtLimits(AbstractCapability[AgentDepsT]):
    """End the run with a final answer instead of raising when a usage limit or the run's timeout would stop it.

    When the run's [`UsageLimits`][pydantic_ai.usage.UsageLimits] leave room for only one more model
    request, or the time left before the run's `timeout=` deadline is projected to fit this request's step
    but not a final response after it, that request is made the last: it keeps the tool definitions, so the cached tools and instructions still
    match, but lets the model call none of the function tools, only the run's output tools if it has
    any, and gets a request-scoped instruction asking the model to answer now. That prompt is a
    [`WrapUpPart`][pydantic_ai_harness.wrap_up_at_limits.WrapUpPart] carrying the `limit` that triggered it, which
    [`wrapped_up_by`][pydantic_ai_harness.wrap_up_at_limits.wrapped_up_by] reads back.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness import WrapUpAtLimits

    agent = Agent('anthropic:claude-opus-5-5', capabilities=[WrapUpAtLimits()])
    ```
    """

    prompt: str = DEFAULT_WRAP_UP_PROMPT
    """What the model is told on the final request."""

    timeout_margin: float | None = 0.0
    """Seconds to keep in reserve before the run's `timeout=` deadline, on top of the projected time.

    `None` doesn't wrap up for time. A negative margin is refused, as it would wrap up too late.
    """

    _last_remaining: float | None = field(default=None, init=False, repr=False, compare=False)
    """The seconds left before the deadline at this run's previous request."""

    _longest_step: float = field(default=0.0, init=False, repr=False, compare=False)
    """The seconds the run's longest step took, from one request to the next."""

    _longest_response: float = field(default=0.0, init=False, repr=False, compare=False)
    """The seconds the run's slowest model response took, counted from its request."""

    def __post_init__(self) -> None:
        check_operator_text(self.prompt, what='`WrapUpAtLimits.prompt`')
        if self.timeout_margin is not None and self.timeout_margin < 0:
            raise UserError(f'`WrapUpAtLimits.timeout_margin` must not be negative, got {self.timeout_margin}.')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> WrapUpAtLimits[AgentDepsT]:
        """Return a fresh capability, so each run measures its own steps."""
        return replace(self)

    async def after_model_request(
        self, ctx: RunContext[AgentDepsT], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        """Measure how long the model took to respond, to project the final response by."""
        remaining = ctx.remaining_time()
        if remaining is not None and self._last_remaining is not None:
            self._longest_response = max(self._longest_response, self._last_remaining - remaining)
        return response

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Make this request the last when the limits leave no room for another one."""
        messages = request_context.messages
        # A provider-resume turn ends in a suspended `ModelResponse` and continues the request before it.
        if not messages or not isinstance(messages[-1], ModelRequest):
            return request_context
        limit = self._wrap_up_reason(ctx, messages)
        if limit is None:
            return request_context
        ctx.tracer.start_span('run wrapped up', attributes={'wrap_up_at_limits.limit': limit}).end()
        ctx.enqueue(
            WrapUpPart(content=OperatorText(self.prompt), source=OperatorSource(), scope='request', limit=limit)
        )
        settings = ModelSettings(**(request_context.model_settings or {}))
        # With output tools, only they stay callable, so a structured output can still be given; with none, the
        # output is text, which native and prompted output parse against the schema.
        if request_context.model_request_parameters.output_tools:
            settings['tool_choice'] = ToolOrOutput(function_tools=[])
        else:
            settings['tool_choice'] = 'none'
        return replace(request_context, model_settings=settings)

    def _wrap_up_reason(self, ctx: RunContext[AgentDepsT], messages: list[ModelMessage]) -> LimitName | None:
        # Measured before the usage limits are checked, so this request's time is recorded even when one fires.
        time_up = self._record_time_and_check(ctx)
        if ctx.usage_limits is not None:
            prompt_tokens = len(self.prompt) // CHARS_PER_TOKEN
            limit = _limit_reached(ctx.usage, ctx.usage_limits, _run_responses(messages, ctx.run_id), prompt_tokens)
            if limit is not None:
                return limit
        return 'timeout' if time_up else None

    def _record_time_and_check(self, ctx: RunContext[AgentDepsT]) -> bool:
        """Record this request's time left, and say whether it fits this request's step but not a final response.

        A step, from one request to the next with its tool calls, is assumed to take as long as the run's
        longest so far, and the final response as long as its slowest response, since it makes no tool calls.
        """
        remaining = ctx.remaining_time()
        if self.timeout_margin is None or remaining is None:
            return False
        if self._last_remaining is not None:
            self._longest_step = max(self._longest_step, self._last_remaining - remaining)
        self._last_remaining = remaining
        return remaining < self._longest_step + self._longest_response + self.timeout_margin


def wrapped_up_by(messages: Sequence[ModelMessage]) -> LimitName | None:
    """The limit that made `WrapUpAtLimits` cut a run short, given the run's `new_messages()`, or `None`."""
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, WrapUpPart):
                    return part.limit
    return None


def _run_responses(messages: list[ModelMessage], run_id: str | None) -> list[ModelResponse]:
    return [message for message in messages if isinstance(message, ModelResponse) and message.run_id == run_id]


def _limit_reached(
    usage: RunUsage, limits: UsageLimits, responses: list[ModelResponse], prompt_tokens: int
) -> LimitName | None:
    """The limit that leaves room for this request but not for a final one after it, if any.

    A batch of tool calls is assumed to be as large as the largest so far, and the token use and cost
    of both requests are projected from `responses`, the run's responses so far, plus the final
    request's `prompt_tokens`. Input grows with the history; each response's output is its own.
    """
    reserved = reserved_usage_limits(limits)
    if reserved is not None and reserved.request_limit is not None and usage.requests >= reserved.request_limit:
        return 'request_limit'
    largest_batch = max((len(r.tool_calls) for r in responses), default=0)
    if limits.tool_calls_limit is not None and limits.tool_calls_limit - usage.tool_calls < max(1, largest_batch):
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
    if _cost_projected_over(usage, limits, responses):
        return 'cost_limit'
    return None


def _cost_projected_over(usage: RunUsage, limits: UsageLimits, responses: list[ModelResponse]) -> bool:
    """Whether the cost of this request and a final one is projected to go over `cost_limit`."""
    if limits.cost_limit is None or usage.cost is None:
        return False
    costs = [r.usage.cost for r in responses if r.usage.cost is not None]
    return usage.cost + _next_two_requests(costs) > limits.cost_limit


def _next_two_requests(amounts: list[_Amount]) -> _Amount | int:
    """The tokens or cost of two more requests, each growing by as much as the last response grew over the one before."""
    if not amounts:
        return 0
    growth = max(amounts[-1] - amounts[-2], 0) if len(amounts) > 1 else 0
    return 2 * amounts[-1] + 3 * growth
