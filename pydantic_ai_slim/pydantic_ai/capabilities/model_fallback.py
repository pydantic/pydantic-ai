from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from pydantic_ai import exceptions
from pydantic_ai._fallback import FallbackChain, FallbackOn, FallbackPredicates
from pydantic_ai._utils import replace_no_init
from pydantic_ai.exceptions import ModelAPIError, RetryModelRequest, UsageLimitExceeded, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability, AgentModel, CapabilityOrdering

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent
    from pydantic_ai.models import ModelRequestContext

__all__ = ('ModelFallback',)

_SpecFallbackError = Literal[
    'ModelAPIError',
    'ModelHTTPError',
    'ModelRateLimitError',
    'ModelQuotaExceededError',
    'ModelUnavailableError',
    'ModelOverloadedError',
    'ModelServerError',
    'ModelConnectionError',
    'ModelTimeoutError',
    'ModelRequestRejectedError',
    'ModelContextWindowExceededError',
    'ModelAuthenticationError',
    'ModelNotFoundError',
    'ModelRequestTimeout',
]
"""The exceptions an agent spec can name in `ModelFallback`'s `fallback_on`: every public `ModelAPIError` class."""

_SPEC_FALLBACK_ERRORS: dict[str, type[ModelAPIError]] = {
    name: getattr(exceptions, name) for name in get_args(_SpecFallbackError)
}


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
    _remaining: list[Model | KnownModelName | str] = field(repr=False)
    """The candidates left to move on to, in order."""
    _attempted_models: list[Model] = field(repr=False)
    _attempted_ids: set[str] = field(repr=False)
    _chain: FallbackChain = field(repr=False)
    """This step's way down the chain. Its attempt log stays empty: the agent records the attempts."""
    _pin_checked: bool = field(repr=False)
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
                instead of starting each request step from the model selected for it. The agent records
                the model that answered as the response's `model_id`, and the attempts that failed before
                it as `failed_attempts`, so this also holds for a later run continuing that history. Only a
                fallback makes the conversation sticky: an answer from a candidate chosen for its run, with
                no attempt failing before it, doesn't. It only replaces the agent's default model: a model passed to
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
        self._remaining = list(self.models)
        self._attempted_models = []
        self._attempted_ids = set()
        self._chain = self._new_chain()
        self._pin_checked = False
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
        return replace_no_init(
            self,
            _default_model=self.models[0] if agent_model is None else agent_model,
            _step=None,
            _remaining=list(self.models),
            _attempted_models=[],
            _attempted_ids=set(),
            _chain=self._new_chain(),
            _pin_checked=False,
            _moved=False,
        )

    async def prepare_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Note the model about to be attempted, routing the step to the candidate the conversation is pinned to.

        That's the candidate that started a suspended response being resumed, or, with `sticky`, the one
        that last answered. Every attempt is noted here, whichever capability asked for it, so a model
        that was already attempted this step is never picked again.
        """
        # Keyed on `run_step` rather than `request_context.attempt`, so the chain still starts over for a new step
        # when another capability asked for its first extra attempt.
        if self._step != ctx.run_step:
            self._step = ctx.run_step
            self._remaining = list(self.models)
            self._attempted_models = []
            self._attempted_ids = set()
            self._chain = self._new_chain()
            self._pin_checked = False
            self._moved = False
        if not self._pin_checked:
            self._pin_checked = True
            if (pinned := self._pinned_candidate(request_context)) is not None:
                self._remaining, candidate = pinned
                if candidate is not None:
                    # Send the step to the pinned candidate instead, before anything is sent.
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
            # A suspended response's continuation, in this run or a later one that resumes it, goes back to the
            # model that started the provider-side job, which the agent records as the response's `model_id`.
            return response
        # The core records a rejected response's tokens and cost before the next attempt, so a
        # rejected generation is still paid for in `RunUsage` even though it never enters history,
        # and records the attempt on the response that answers, as `FallbackModel` does.
        raise self._advance()

    def _new_chain(self) -> FallbackChain:
        return FallbackChain(self._predicates, owner='ModelFallback')

    def _pinned_candidate(
        self, request_context: ModelRequestContext
    ) -> tuple[list[Model | KnownModelName | str], Model | str | None] | None:
        """Where in the chain this step is pinned, if the conversation is pinned to one of this capability's models.

        Returns the candidates to move on to should the pinned one fail, and the candidate to move the request to, or
        `None` when the step's model already is the pinned one.
        """
        messages = request_context.messages
        step_id = _pin_id(request_context.model_id or request_context.model)
        if messages and isinstance(last := messages[-1], ModelResponse) and last.state == 'suspended':
            # A resumed suspended response started by one of this capability's models. Should its continuation fail,
            # the chain moves on from that model, like it would have when the job was started, instead of starting
            # over with a model that would just start another. That holds whether or not the step's model already is
            # that model. A response started by a model outside the chain continues on the step's model.
            if (index := self._index_of(last.model_id)) is None:
                return None
            # Otherwise the step would continue a provider-side job on a model other than the one that started it.
            return self.models[index + 1 :], None if last.model_id == step_id else self.models[index]
        if not (self.sticky and self._is_default_model(request_context)):
            return None
        sticky_id = _fallen_back_to(messages)
        # A conversation the step's own model answered last, or a model that's no longer a candidate, starts as usual.
        if sticky_id is None or sticky_id == step_id or (index := self._index_of(sticky_id)) is None:
            return None
        # Should the sticky candidate fail, the chain goes on from there and comes back around to the step's model, so
        # sticking to a candidate never leaves a model untried that the chain would otherwise have attempted.
        return [*self.models[index + 1 :], request_context.model, *self.models[:index]], self.models[index]

    def _index_of(self, model_id: str | None) -> int | None:
        """The index among `models` of the candidate a response's `model_id` identifies, if any."""
        return next((index for index, candidate in enumerate(self.models) if _pin_id(candidate) == model_id), None)

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
    def from_spec(
        cls,
        models: KnownModelName | str | Sequence[KnownModelName | str],
        *,
        fallback_on: _SpecFallbackError | Sequence[_SpecFallbackError] = 'ModelAPIError',
        sticky: bool = False,
    ) -> ModelFallback[Any]:
        """Build from an agent spec, which names the models and the exceptions to fall back on.

        Model names are kept as given and resolved when the agent attempts them, as a model name passed to
        `ModelFallback(...)` is, so a [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] capability's
        aliases work in a spec too. `fallback_on` takes the names of exception classes in
        [`pydantic_ai.exceptions`][pydantic_ai.exceptions] that subclass
        [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError]; exception and response handlers can only be
        passed to `ModelFallback(...)` directly.

        YAML form:

            capabilities:
              - ModelFallback: [anthropic:claude-opus-5-5, openai:gpt-6.1-sol]
              - ModelFallback:
                  models: [anthropic:claude-opus-5-5]
                  fallback_on: [ModelRateLimitError, ModelUnavailableError]
                  sticky: true
        """
        # Spec arguments reach here unvalidated: the annotations above are what the published spec schema
        # advertises, not something that rejects other values on the way in.
        given_models = _spec_list(models, 'models')
        model_names = [model for model in given_models if isinstance(model, str)]
        if len(model_names) != len(given_models):
            invalid = [model for model in given_models if not isinstance(model, str)]
            raise UserError(f'`ModelFallback` in an agent spec takes model names, got {invalid!r}.')
        given_errors = _spec_list(fallback_on, 'fallback_on')
        error_types = [
            _SPEC_FALLBACK_ERRORS[name]
            for name in given_errors
            if isinstance(name, str) and name in _SPEC_FALLBACK_ERRORS
        ]
        if len(error_types) != len(given_errors):
            unknown = [name for name in given_errors if not (isinstance(name, str) and name in _SPEC_FALLBACK_ERRORS)]
            raise UserError(
                f'`ModelFallback` `fallback_on` in an agent spec takes names of `ModelAPIError` classes, '
                f'got {unknown!r}. Valid names: {", ".join(_SPEC_FALLBACK_ERRORS)}.'
            )
        return cls(*model_names, fallback_on=tuple(error_types), sticky=sticky)


def _spec_list(value: object, name: str) -> Sequence[object]:
    """A `ModelFallback` spec argument given as one name or a list of them, as a list.

    Anything else, like a mapping, whose keys iterating would silently take as the names, is rejected.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return cast(Sequence[object], value)
    raise UserError(f'`ModelFallback` `{name}` in an agent spec must be a name or a list of names, got {value!r}.')


def _pin_id(model: Model | str) -> str:
    """How a candidate is identified in a response's `model_id`: the string it was given as, or its `model_id`."""
    return model if isinstance(model, str) else model.model_id


def _fallen_back_to(messages: Sequence[ModelMessage]) -> str | None:
    """The model the conversation fell back to, if the latest answers all came from it since it was fallen back to.

    That's the `model_id` of the latest responses, as long as one of them was reached by falling back, as the
    attempts that failed before it record. Answers from a model chosen for its run, like with `run(model=...)`, with
    no attempt failing before them, don't make the conversation stick to it.
    """
    model_id: str | None = None
    for message in reversed(messages):
        if not isinstance(message, ModelResponse):
            continue
        if model_id is None:
            model_id = message.model_id
        if message.model_id is None or message.model_id != model_id:
            return None
        if message.failed_attempts:
            return model_id
    return None


def _is_or_wraps(model: Model, candidate: Model) -> bool:
    while True:
        if model is candidate:
            return True
        if not isinstance(model, WrapperModel):
            return False
        model = model.wrapped
