from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai._fallback import FallbackOn, FallbackPredicates, raise_fallback_exception_group
from pydantic_ai._utils import replace_no_init
from pydantic_ai.exceptions import ModelAPIError, RetryModelRequest, UserError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability, AgentModel, CapabilityOrdering

if TYPE_CHECKING:
    from typing import Any

    from pydantic_ai.agent.abstract import AbstractAgent
    from pydantic_ai.models import ModelRequestContext

__all__ = ('Fallback',)


@dataclass(init=False)
class Fallback(AbstractCapability[AgentDepsT]):
    """Attempt other models when the one serving a request fails or returns a rejected response.

    The chain is the model selected for the step followed by this capability's models, so it
    composes with an agent-level model and with
    [`SelectModel`][pydantic_ai.capabilities.SelectModel] rather than replacing them:

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Fallback

    # Try gpt-5.6-sol, then Fable.
    agent = Agent('openai:gpt-5.6-sol', capabilities=[Fallback('anthropic:claude-fable-5')])

    # No agent model: the first candidate is used for the first attempt.
    agent = Agent(capabilities=[Fallback('openai:gpt-5.6-sol', 'anthropic:claude-fable-5')])
    ```

    A model already attempted this step is skipped, so listing the agent's own model as the first
    candidate is harmless rather than a wasted duplicate attempt.

    Each attempt re-runs
    [`prepare_model_request`][pydantic_ai.capabilities.AbstractCapability.prepare_model_request] for
    the candidate about to serve it, so provider-specific preparation — message translation,
    compaction, context-window fitting — is redone per model instead of inherited. The request step
    itself is not repeated: `before_model_request` and `wrap_model_request` run once, and the model
    never sees a retry prompt.

    Prefer this over [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] inside an agent.
    `FallbackModel` remains the way to get the same behavior from a bare
    [`Model`][pydantic_ai.models.Model], outside an agent run.
    """

    models: list[Model | KnownModelName | str]
    """The models to attempt, in order, after the one selected for the step."""

    _predicates: FallbackPredicates = field(repr=False)
    _supplies_model: bool = field(repr=False)

    # Per-step state, reset whenever a new request step starts. Lives on the `for_run` copy so
    # concurrent runs of the same agent don't share a cursor.
    _step: int | None = field(repr=False)
    _cursor: int = field(repr=False)
    _attempted_models: list[Model] = field(repr=False)
    _attempted_ids: set[str] = field(repr=False)
    _exceptions: list[Exception] = field(repr=False)
    _rejected_responses: list[ModelResponse] = field(repr=False)

    def __init__(
        self,
        *models: Model | KnownModelName | str,
        fallback_on: FallbackOn = (ModelAPIError,),
    ):
        """Initialize the fallback capability.

        Args:
            models: The models to attempt after the one selected for the request step. When the
                agent has no model of its own, the first of these is used for the first attempt.
            fallback_on: Conditions that trigger fallback to the next model. Accepts:

                - A tuple of exception types: `(ModelAPIError, RateLimitError)`
                - An exception handler (sync or async): `lambda exc: isinstance(exc, MyError)`
                - A response handler (sync or async): `def check(r: ModelResponse) -> bool`
                - A sequence mixing all of the above: `[ModelAPIError, exc_handler, response_handler]`

                Handler type is auto-detected by inspecting type hints on the first parameter.
                If the first parameter is hinted as `ModelResponse`, it's a response handler.
                Otherwise (including untyped handlers and lambdas), it's an exception handler.

                Response handlers only apply to non-streamed requests: a streamed response has
                already reached the consumer by the time it could be judged. Stream-opening
                failures still fall back.
        """
        super().__init__()
        if not models:
            raise UserError('`Fallback` requires at least one model to fall back to.')
        self.models = list(models)
        self._predicates = FallbackPredicates.parse(fallback_on, owner='Fallback')
        self._supplies_model = False
        self._step = None
        self._cursor = 0
        self._attempted_models = []
        self._attempted_ids = set()
        self._exceptions = []
        self._rejected_responses = []

    def get_ordering(self) -> CapabilityOrdering:
        # Innermost so a rejected attempt is judged before any other capability's
        # `after_model_request` sees it: the chain there runs innermost-first and a raise
        # short-circuits it, so outer hooks — instrumentation included — only ever observe the
        # response that was finally accepted.
        return CapabilityOrdering(position='innermost')

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> Fallback[AgentDepsT]:
        """Decide whether this capability also has to supply the agent's model.

        A capability's model contribution outranks the agent's own default, so contributing
        unconditionally would silently redirect `Agent(model, capabilities=[Fallback(...)])` to the
        first fallback candidate instead of falling back *to* it. Only step in when the agent has no
        model of its own — a run-level `model=` or `override(model=)` outranks a contribution
        anyway, so those keep working either way.
        """
        return replace_no_init(self, _supplies_model=agent.model is None)

    def get_model(self) -> AgentModel[AgentDepsT] | None:
        if not self._supplies_model:
            return None
        # Returned unresolved so `resolve_model_id` capabilities get their say, exactly as they
        # would for a model passed to `Agent`.
        return self.models[0]

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Fallback[AgentDepsT]:
        """Return a fresh copy so each run tracks its own chain position."""
        return replace_no_init(
            self,
            _step=None,
            _cursor=0,
            _attempted_models=[],
            _attempted_ids=set(),
            _exceptions=[],
            _rejected_responses=[],
        )

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        if not await self._predicates.should_fallback(error):
            raise error
        self._start_of_attempt(ctx, request_context)
        self._exceptions.append(error)
        raise self._advance()

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if not await self._predicates.should_fallback(response):
            return response
        self._start_of_attempt(ctx, request_context)
        # The core records a rejected response's tokens and cost before the next attempt, so a
        # rejected generation is still paid for in `RunUsage` even though it never enters history.
        self._rejected_responses.append(response)
        raise self._advance()

    def _start_of_attempt(self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext) -> None:
        """Note which model just failed, resetting the chain when a new request step begins.

        Keyed on `run_step` rather than `request_context.attempt` so the state is still reset
        correctly when some other capability drove the first extra attempt.
        """
        if self._step != ctx.run_step:
            self._step = ctx.run_step
            self._cursor = 0
            self._attempted_models = []
            self._attempted_ids = set()
            self._exceptions = []
            self._rejected_responses = []
        self._attempted_models.append(request_context.model)
        if request_context.model_id is not None:
            self._attempted_ids.add(request_context.model_id)

    def _advance(self) -> RetryModelRequest:
        """Return the exception that moves to the next untried candidate, or the aggregated failure."""
        while self._cursor < len(self.models):
            candidate = self.models[self._cursor]
            self._cursor += 1
            if not self._already_attempted(candidate):
                return RetryModelRequest(candidate)
        raise_fallback_exception_group(self._exceptions, self._rejected_responses, owner='Fallback')

    def _already_attempted(self, candidate: Model | KnownModelName | str) -> bool:
        if isinstance(candidate, Model):
            return any(candidate is attempted for attempted in self._attempted_models)
        return candidate in self._attempted_ids

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (may hold `Model` instances and callables)
