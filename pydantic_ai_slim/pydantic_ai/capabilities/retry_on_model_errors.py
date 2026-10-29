from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

from pydantic_ai._agent_graph import _agent_graph_sleep  # pyright: ignore[reportPrivateUsage]
from pydantic_ai._fallback import (
    FallbackOn,
    FallbackPredicates,
    SpecModelError,
    spec_error_types,
    warn_response_handlers_not_applied,
)
from pydantic_ai._utils import replace_no_init
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelHTTPError,
    ModelRateLimitError,
    ModelServerError,
    ModelUnavailableError,
    RetryModelRequest,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability, CapabilityOrdering
from .model_fallback import ModelFallback

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

__all__ = ('RetryOnModelErrors',)

_DEFAULT_RETRY_ON: tuple[type[ModelAPIError], ...] = (
    ModelRateLimitError,
    ModelUnavailableError,
    ModelServerError,
    ModelConnectionError,
)
_DEFAULT_SPEC_RETRY_ON: tuple[SpecModelError, ...] = (
    'ModelRateLimitError',
    'ModelUnavailableError',
    'ModelServerError',
    'ModelConnectionError',
)


@dataclass(init=False)
class RetryOnModelErrors(AbstractCapability[AgentDepsT]):
    """Attempt the same model again when a request to it fails with a transient error.

    By default, a request is retried when it fails with an error that waiting may fix: a
    [`ModelRateLimitError`][pydantic_ai.exceptions.ModelRateLimitError], a
    [`ModelUnavailableError`][pydantic_ai.exceptions.ModelUnavailableError] (including an overloaded model), a
    [`ModelServerError`][pydantic_ai.exceptions.ModelServerError], or a
    [`ModelConnectionError`][pydantic_ai.exceptions.ModelConnectionError] (including a transport timeout):

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import RetryOnModelErrors

    agent = Agent('openai:gpt-6.1-sol', capabilities=[RetryOnModelErrors(max_attempts=4)])
    ```

    Before each retry it waits as long as the error's
    [`retry_after`][pydantic_ai.exceptions.ModelAPIError.retry_after] asks, or else an exponentially growing,
    jittered delay. A request is not retried when the wait would be longer than `max_delay` or would end after the
    run's [deadline][pydantic_ai.tools.RunContext.deadline], or when the provider said with an `x-should-retry: false`
    header that retrying won't help (see
    [`ModelHTTPError.provider_retry_hint`][pydantic_ai.exceptions.ModelHTTPError.provider_retry_hint]). The error is
    then raised as is, so a [`ModelFallback`][pydantic_ai.capabilities.ModelFallback] can move on to another model.

    With a `ModelFallback`, each model in the chain gets `max_attempts` attempts of its own, whatever order the two
    capabilities are listed in: the request is retried on one model, and falls back to the next once its attempts
    are used up.

    Each attempt is an attempt within the same request step, like a `ModelFallback` attempt: the model never sees
    a retry prompt, and the answering response lists the attempts before it in
    [`failed_attempts`][pydantic_ai.messages.ModelResponse.failed_attempts], each retried one with
    `kind='retry'`. A streamed request is retried when it fails before any of its output has reached you, but not
    after.
    """

    max_attempts: int
    """How many times a request is attempted on one model, the first attempt included."""

    initial_delay: float
    """Seconds to wait before the first retry when the error doesn't say how long; doubled for each retry after it."""

    max_delay: float
    """The longest wait before a retry, in seconds.

    The exponential backoff stops growing here. A provider that asks for a longer wait in `retry_after` isn't
    retried at all: the error is raised, so a `ModelFallback` can move on to another model at once.
    """

    _predicates: FallbackPredicates = field(repr=False)

    # Per-run state, on the `for_run` copy so concurrent runs of the same agent don't share a budget.
    _step: int | None = field(repr=False)
    _model: Model | None = field(repr=False)
    """The model the current attempt is at, without the wrappers applied to it."""
    _attempt: int = field(repr=False)
    """Which attempt in a row at `_model` in this step the current one is, counting from 1."""

    def __init__(
        self,
        *,
        retry_on: FallbackOn = _DEFAULT_RETRY_ON,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
    ):
        """Initialize the retry capability.

        Args:
            retry_on: Conditions that trigger a retry, in the same forms as `ModelFallback`'s `fallback_on`:

                - A tuple of exception types: `(ModelRateLimitError, ModelUnavailableError)`
                - An exception handler (sync or async): `lambda exc: isinstance(exc, MyError)`
                - A response handler (sync or async): `def check(r: ModelResponse) -> bool`
                - A sequence mixing all of the above: `[ModelRateLimitError, exc_handler, response_handler]`

                Defaults to the transient errors listed above. Handler type is auto-detected by inspecting type
                hints on the first parameter: a handler whose first parameter is hinted as `ModelResponse` is a
                response handler, and any other is an exception handler. Response handlers only apply to
                non-streamed requests: a streamed response has already reached the consumer by the time it
                could be judged, so it's accepted as is, with a `UserWarning`.
            max_attempts: How many times a request is attempted on one model, the first attempt included. `1`
                never retries.
            initial_delay: Seconds to wait before the first retry when the error doesn't say how long. The wait
                doubles for each retry after it, up to `max_delay`, and a random part of up to half of it is taken
                off, so that runs failing at the same moment don't all retry at the same moment too.
            max_delay: The longest wait before a retry, in seconds. An error whose `retry_after` asks for longer is
                raised instead of retried.
        """
        super().__init__()
        if max_attempts < 1:
            raise UserError(f'`RetryOnModelErrors` `max_attempts` must be at least 1, got {max_attempts!r}.')
        if initial_delay < 0 or max_delay < 0:
            raise UserError('`RetryOnModelErrors` `initial_delay` and `max_delay` must not be negative.')
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self._predicates = FallbackPredicates.parse(
            retry_on,
            owner='RetryOnModelErrors',
            param='retry_on',
            default='(ModelRateLimitError, ModelUnavailableError, ModelServerError, ModelConnectionError)',
        )
        self._step = None
        self._model = None
        self._attempt = 1

    def get_ordering(self) -> CapabilityOrdering:
        # Inside `ModelFallback`, which is innermost too: the error and after hooks run innermost-first and the
        # first `RetryModelRequest` answers for the chain, so this one retries the current model before
        # `ModelFallback` would move on from it, and only lets the error or response through once its attempts
        # at that model are used up.
        return CapabilityOrdering(position='innermost', wrapped_by=(ModelFallback,))

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> RetryOnModelErrors[AgentDepsT]:
        """Return a fresh copy so each run keeps its own attempt count."""
        return replace_no_init(self, _step=None, _model=None, _attempt=1)

    async def prepare_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Count the attempt about to be made: one more at the same model, or the first at another one.

        Whichever capability asked for the attempt, so a `ModelFallback` moving on to its next model starts a fresh
        budget, and two of these capabilities share one per model.
        """
        model = request_context.model
        # The model may be wrapped by now, e.g. by a durable-execution capability, and wasn't on the previous attempt.
        while isinstance(model, WrapperModel):
            model = model.wrapped
        if self._step == ctx.run_step and model is self._model:
            self._attempt += 1
        else:
            self._step = ctx.run_step
            self._model = model
            self._attempt = 1
        return request_context

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        if isinstance(error, UsageLimitExceeded) or self._attempt >= self.max_attempts:
            raise error
        if isinstance(error, ModelHTTPError) and error.provider_retry_hint is False:
            # The provider said retrying won't help, as the OpenAI and Anthropic SDKs would have heeded.
            raise error
        if not await self._predicates.matches(error):
            raise error
        retry_after = error.retry_after if isinstance(error, ModelAPIError) else None
        if (delay := self._delay(ctx, retry_after)) is None:
            raise error
        await self._retry_later(delay)

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if not self._predicates.response_handlers:
            return response
        if request_context.streaming:
            # A streamed response has already reached the consumer, so it's accepted without consulting
            # the response handlers, as `ModelFallback` does.
            warn_response_handlers_not_applied('RetryOnModelErrors', 'retry_on')
            return response
        if self._attempt >= self.max_attempts or not await self._predicates.matches(response):
            return response
        if (delay := self._delay(ctx, None)) is None:
            return response
        # The core records the rejected response's tokens and cost before the next attempt, and lists the attempt
        # on the response that answers.
        await self._retry_later(delay)

    def _delay(self, ctx: RunContext[AgentDepsT], retry_after: float | None) -> float | None:
        """How long to wait before the next attempt, or `None` if the request shouldn't be retried."""
        if retry_after is not None:
            if retry_after > self.max_delay:
                return None
            delay = max(retry_after, 0.0)
        else:
            backoff = min(self.initial_delay * 2 ** (self._attempt - 1), self.max_delay)
            delay = backoff * (1 - _jitter(ctx, self._attempt) / 2)
        if (remaining := ctx.remaining_time()) is not None and delay >= remaining:
            # The run would time out while waiting; raising now lets a `ModelFallback` try another model instead.
            return None
        return delay

    async def _retry_later(self, delay: float) -> NoReturn:
        # Through the agent graph's sleep, which a durable execution engine replaces with its own durable timer.
        await _agent_graph_sleep(delay)
        raise RetryModelRequest()

    @classmethod
    def from_spec(
        cls,
        *,
        retry_on: SpecModelError | Sequence[SpecModelError] = _DEFAULT_SPEC_RETRY_ON,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> RetryOnModelErrors[Any]:
        """Build from an agent spec, which names the exceptions to retry on.

        `retry_on` takes the names of exception classes in [`pydantic_ai.exceptions`][pydantic_ai.exceptions] that
        subclass [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError]; exception and response handlers can only
        be passed to `RetryOnModelErrors(...)` directly.

        YAML form:

            capabilities:
              - RetryOnModelErrors: {}
              - RetryOnModelErrors:
                  retry_on: [ModelRateLimitError, ModelOverloadedError]
                  max_attempts: 5
        """
        error_types = spec_error_types(retry_on, owner='RetryOnModelErrors', param='retry_on')
        return cls(retry_on=error_types, max_attempts=max_attempts, initial_delay=initial_delay, max_delay=max_delay)


def _jitter(ctx: RunContext[Any], attempt: int) -> float:
    """A number in `[0, 1)` that differs between runs, steps and attempts, and is the same when a run is replayed.

    Derived from the run's ID rather than drawn at random, so a durable execution engine replaying the run waits the
    same delay it recorded, while runs that fail at the same moment still spread out their retries.
    """
    digest = hashlib.blake2b(f'{ctx.run_id}:{ctx.run_step}:{attempt}'.encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'big') / 2**64
