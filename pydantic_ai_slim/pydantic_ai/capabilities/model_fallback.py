from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai._fallback import (
    FALLBACK_CAPABILITY_PIN_KEY,
    FallbackChain,
    FallbackOn,
    FallbackPredicates,
    continuation_pin,
    stamp_continuation_pin,
)
from pydantic_ai._utils import replace_no_init
from pydantic_ai.exceptions import ModelAPIError, RetryModelRequest, UsageLimitExceeded, UserError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability, AgentModel, CapabilityOrdering

if TYPE_CHECKING:
    from typing import Any

    from pydantic_ai.agent.abstract import AbstractAgent
    from pydantic_ai.models import ModelRequestContext

__all__ = ('ModelFallback',)


@dataclass(init=False)
class ModelFallback(AbstractCapability[AgentDepsT]):
    """Attempt other models when the one serving a request fails or returns a rejected response.

    The chain is the model selected for the step followed by this capability's models, so it
    composes with an agent-level model and with
    [`SelectModel`][pydantic_ai.capabilities.SelectModel] rather than replacing them:

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import ModelFallback

    # Try gpt-6.1-sol, then Fable 5.1.
    agent = Agent('openai:gpt-6.1-sol', capabilities=[ModelFallback('anthropic:claude-fable-5-1')])

    # No agent model: the first candidate is used for the first attempt.
    agent = Agent(capabilities=[ModelFallback('openai:gpt-6.1-sol', 'anthropic:claude-fable-5-1')])
    ```

    A model already attempted this step is skipped, so listing the agent's own model as the first
    candidate is harmless rather than a wasted duplicate attempt.

    Each attempt re-runs
    [`prepare_model_request`][pydantic_ai.capabilities.AbstractCapability.prepare_model_request] for
    the candidate about to serve it, so the model's own message preparation and token counting, and
    any preparation a capability does there, are redone per model instead of inherited. (The built-in
    compaction capabilities still run once per step, in `before_model_request`.) The request step
    itself is not repeated: `before_model_request` and `wrap_model_request` run once, and the model
    never sees a retry prompt.

    With `sticky=True`, a conversation that fell back stays on the model that answered, in later
    steps and later runs continuing its message history, so the provider's prompt cache keeps
    being hit and the failing model isn't asked first every step.

    Prefer this over [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] inside an agent.
    `FallbackModel` remains the way to get the same behavior from a bare
    [`Model`][pydantic_ai.models.Model], outside an agent run.
    """

    models: list[Model | KnownModelName | str]
    """The models to attempt, in order, after the one selected for the step."""

    sticky: bool
    """Whether a conversation stays on the model that last answered it after falling back."""

    _predicates: FallbackPredicates = field(repr=False)
    _supplies_model: bool = field(repr=False)
    _default_model: Model | KnownModelName | str | None = field(repr=False)
    """The model a step uses when nothing selects one, as of the run's start: the agent's own, or this capability's first."""

    # Per-step state, reset whenever a new request step starts. Lives on the `for_run` copy so
    # concurrent runs of the same agent don't share a cursor.
    _step: int | None = field(repr=False)
    _step_model: Model | None = field(repr=False)
    """The model the step started with, noted at its first attempt."""
    _remaining: list[Model | KnownModelName | str] = field(repr=False)
    """The candidates left to move on to, in order."""
    _attempted_models: list[Model] = field(repr=False)
    _attempted_ids: set[str] = field(repr=False)
    _chain: FallbackChain = field(repr=False)
    """This step's way down the chain. Its attempt log stays empty: the agent records the attempts."""
    _moved: bool = field(repr=False)
    """Whether the next attempt is at a candidate this capability moved the request to."""

    def __init__(
        self,
        *models: Model | KnownModelName | str,
        fallback_on: FallbackOn = (ModelAPIError,),
        sticky: bool = False,
    ):
        """Initialize the fallback capability.

        Args:
            models: The models to attempt after the one selected for the request step. When the
                agent has no model of its own, the first of these is used for the first attempt.
            fallback_on: Conditions that trigger fallback to the next model. Accepts:

                - A tuple of exception types: `(ModelRateLimitError, ModelUnavailableError)`
                - An exception handler (sync or async): `lambda exc: isinstance(exc, MyError)`
                - A response handler (sync or async): `def check(r: ModelResponse) -> bool`
                - A sequence mixing all of the above: `[ModelAPIError, exc_handler, response_handler]`

                Handler type is auto-detected by inspecting type hints on the first parameter.
                If the first parameter is hinted as `ModelResponse`, it's a response handler.
                Otherwise (including untyped handlers and lambdas), it's an exception handler.

                Response handlers only apply to non-streamed requests: a streamed response has
                already reached the consumer by the time it could be judged, so it's accepted as
                is, with a `UserWarning`. A stream that fails to open, or whose provider reports
                an error before its first event, still falls back.
            sticky: Whether a conversation that fell back keeps using the candidate that answered,
                instead of starting each request step from the model selected for it. The candidate is
                recorded on the response in the message history, so this also holds for a later run
                continuing that history. It only replaces the agent's default model: a model passed to
                `run(model=...)` or `override(model=...)`, or selected by another capability, such as
                `SelectModel`, is still attempted first. If the sticky candidate fails, the chain moves on
                to the models after it, then the step's own model and the models before it.
        """
        super().__init__()
        if not models:
            raise UserError('`ModelFallback` requires at least one model to fall back to.')
        self.models = list(models)
        self.sticky = sticky
        self._predicates = FallbackPredicates.parse(fallback_on, owner='ModelFallback')
        self._supplies_model = False
        self._default_model = None
        self._step = None
        self._step_model = None
        self._remaining = list(self.models)
        self._attempted_models = []
        self._attempted_ids = set()
        self._chain = self._new_chain()
        self._moved = False

    def get_ordering(self) -> CapabilityOrdering:
        # Innermost so a rejected attempt is judged before any other capability's
        # `after_model_request` sees it: the chain there runs innermost-first and a raise
        # short-circuits it, so outer hooks — instrumentation included — only ever observe the
        # response that was finally accepted.
        return CapabilityOrdering(position='innermost')

    def for_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> ModelFallback[AgentDepsT]:
        """Decide whether this capability also has to supply the agent's model.

        A capability's model contribution outranks the agent's own default, so contributing
        unconditionally would silently redirect `Agent(model, capabilities=[ModelFallback(...)])` to the
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

    def _model_is_default(self) -> bool:
        # A stand-in for the agent's missing model: another capability's selection, such as
        # `SelectModel`'s, outranks it rather than being overridden by it.
        return True

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> ModelFallback[AgentDepsT]:
        """Return a fresh copy so each run tracks its own chain position."""
        # The agent's model as the run sees it: since this capability was attached, the agent may have inferred it
        # from a name, or it may have been reassigned.
        agent_model = ctx.agent.model if ctx.agent is not None else None
        default_model = self.models[0] if agent_model is None else agent_model
        return replace_no_init(
            self,
            _default_model=default_model,
            _step=None,
            _step_model=None,
            _remaining=list(self.models),
            _attempted_models=[],
            _attempted_ids=set(),
            _chain=self._new_chain(),
            _moved=False,
        )

    async def prepare_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Note the model about to be attempted, routing the step to the model the conversation is pinned to.

        That's the model that started a suspended response being resumed, or, with `sticky`, the
        candidate that last answered. Every attempt is noted here, whichever capability asked for it,
        so a model that was already attempted this step is never picked again.
        """
        self._reset_for_step(ctx)
        if self._step_model is None:
            self._step_model = request_context.model
            if (pinned := self._pinned_candidate(request_context)) is not None:
                self._remaining, candidate = pinned
                if candidate is not None:
                    # Send the step to the pinned model instead, before anything is sent.
                    self._moved = True
                    raise RetryModelRequest(candidate)
        moved, self._moved = self._moved, False
        if moved and self._already_attempted(request_context.model):
            # A candidate given as a string, which the agent resolved only now, turned out to be a model already
            # attempted this step, like the step's own model under an alias.
            raise self._advance()
        self._attempted_models.append(request_context.model)
        if request_context.model_id is not None:
            self._attempted_ids.add(request_context.model_id)
        return request_context

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        if isinstance(error, UsageLimitExceeded):
            # The run's own limit stopped the request, which no other model would change, whatever `fallback_on` says.
            raise error
        if not await self._chain.falls_back(error):
            raise error
        raise self._advance()

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if request_context.streaming:
            # A streamed response has already reached the consumer, so it's accepted without consulting
            # the response handlers, with a warning when there are any, as `FallbackModel` does.
            self._chain.warn_if_streamed_responses_unjudged()
        if request_context.streaming or not await self._chain.falls_back(response):
            pin = _pin_id(request_context.model_id or request_context.model)
            if response.state == 'suspended' or (self.sticky and self._answered_by_candidate(request_context, pin)):
                # A suspended response's continuation, in this run or a later one that resumes it, has to go
                # back to the model that started the provider-side job, not to the top of the chain. With
                # `sticky`, the conversation's next request goes to the candidate that answered it too.
                stamp_continuation_pin(response, pin, key=FALLBACK_CAPABILITY_PIN_KEY)
            return response
        # The core records a rejected response's tokens and cost before the next attempt, so a
        # rejected generation is still paid for in `RunUsage` even though it never enters history,
        # and records the attempt on the response that answers, as `FallbackModel` does.
        raise self._advance()

    def _new_chain(self) -> FallbackChain:
        return FallbackChain(self._predicates, owner='ModelFallback')

    def _reset_for_step(self, ctx: RunContext[AgentDepsT]) -> None:
        """Reset the chain when a new request step begins.

        Keyed on `run_step` rather than `request_context.attempt` so the state is still reset
        correctly when some other capability drove the first extra attempt.
        """
        if self._step != ctx.run_step:
            self._step = ctx.run_step
            self._step_model = None
            self._remaining = list(self.models)
            self._attempted_models = []
            self._attempted_ids = set()
            self._chain = self._new_chain()
            self._moved = False

    def _pinned_candidate(
        self, request_context: ModelRequestContext
    ) -> tuple[list[Model | KnownModelName | str], Model | str | None] | None:
        """Where in the chain this step is pinned, if the conversation is pinned to one of this capability's models.

        Returns the candidates to move on to should the pinned model fail, and the model to move the request to, or
        `None` when the step's model already is the pinned one.
        """
        messages = request_context.messages
        step_pin = _pin_id(request_context.model_id or request_context.model)
        if messages and isinstance(last := messages[-1], ModelResponse) and last.state == 'suspended':
            # Resuming a suspended response: its continuation has to go to the model that started the job.
            pin = continuation_pin(last, key=FALLBACK_CAPABILITY_PIN_KEY)
            if pin is None:
                return None
            index = self._index_of(pin)
            if pin == step_pin:
                # Should the continuation fail, the chain moves on from the model it's pinned to, like it would
                # have when the job was started, instead of starting over with a model that would just start
                # another. That holds whether or not the step's model already is that model.
                return self.models[(-1 if index is None else index) + 1 :], None
            if index is None:
                raise UserError(
                    f'Cannot resume this suspended response: it was started by {pin!r}, which is neither the '
                    "model selected for this step nor one of `ModelFallback`'s models. Pass that model explicitly "
                    'to `run(model=...)` when resuming.'
                )
            return self.models[index + 1 :], self.models[index]
        if not (self.sticky and self._is_default_model(request_context)):
            return None
        latest = next((message for message in reversed(messages) if isinstance(message, ModelResponse)), None)
        pin = None if latest is None else continuation_pin(latest, key=FALLBACK_CAPABILITY_PIN_KEY)
        # A pin to a model that's no longer a candidate is ignored: the step starts from its own model.
        if pin is None or pin == step_pin or (index := self._index_of(pin)) is None:
            return None
        # Should the sticky model fail, the chain goes on from there and comes back around to the step's model, so
        # sticking to a candidate never leaves a model untried that the chain would otherwise have attempted.
        return [*self.models[index + 1 :], request_context.model, *self.models[:index]], self.models[index]

    def _answered_by_candidate(self, request_context: ModelRequestContext, pin: str) -> bool:
        """Whether the attempt is at one of this capability's models other than the step's, whoever moved it there.

        An answer from the step's own model needs no stamp: the next step starts with it anyway.
        """
        assert self._step_model is not None, '`prepare_model_request` notes the step model before any attempt'
        return not _is_or_wraps(request_context.model, self._step_model) and self._index_of(pin) is not None

    def _index_of(self, pin: str) -> int | None:
        return next((index for index, candidate in enumerate(self.models) if _pin_id(candidate) == pin), None)

    def _is_default_model(self, request_context: ModelRequestContext) -> bool:
        """Whether the step's model is the one used when nothing selects one, which `sticky` may replace."""
        default = self._default_model
        if isinstance(default, Model):
            return _is_or_wraps(request_context.model, default)
        return default is not None and request_context.model_id == default

    def _advance(self) -> RetryModelRequest:
        """Return the exception that moves to the next untried candidate, or the aggregated failure."""
        while self._remaining:
            candidate = self._remaining.pop(0)
            if not self._already_attempted(candidate):
                self._moved = True
                return RetryModelRequest(candidate)
        # The attempts are the agent's to list, as it made them: it records each failed one, and
        # lists them on this group once it ends the request.
        self._chain.exhausted()

    def _already_attempted(self, candidate: Model | KnownModelName | str) -> bool:
        if isinstance(candidate, Model):
            # The attempted model may be wrapped by now, e.g. by a durable-execution capability.
            return any(_is_or_wraps(attempted, candidate) for attempted in self._attempted_models)
        return candidate in self._attempted_ids

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (may hold `Model` instances and callables)


def _pin_id(model: Model | str) -> str:
    """How a candidate is identified in a continuation pin: the string it was given as, or its `model_id`."""
    return model if isinstance(model, str) else model.model_id


def _is_or_wraps(model: Model, candidate: Model) -> bool:
    while True:
        if model is candidate:
            return True
        if not isinstance(model, WrapperModel):
            return False
        model = model.wrapped
