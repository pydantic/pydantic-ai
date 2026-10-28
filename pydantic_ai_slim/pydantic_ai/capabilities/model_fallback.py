from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from pydantic_ai import exceptions
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
    _chain: FallbackChain = field(repr=False)
    """This step's way down the chain. Its attempt log stays empty: the agent records the attempts."""
    _pin_checked: bool = field(repr=False)
    _moved: bool = field(repr=False)
    """Whether the next attempt is at a candidate this capability moved the request to."""

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
        """
        super().__init__()
        if not models:
            raise UserError('`ModelFallback` requires at least one model to fall back to.')
        self.models = list(models)
        self._predicates = FallbackPredicates.parse(fallback_on, owner='ModelFallback')
        self._supplies_model = False
        self._step = None
        self._cursor = 0
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
        return replace_no_init(
            self,
            _step=None,
            _cursor=0,
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
        """Note the model about to be attempted, routing a resumed continuation to the model it's pinned to.

        Every attempt is noted here, whichever capability asked for it, so a model that was already
        attempted this step is never picked again.
        """
        # Keyed on `run_step` rather than `request_context.attempt`, so the chain still starts over for a new step
        # when another capability asked for its first extra attempt.
        if self._step != ctx.run_step:
            self._step = ctx.run_step
            self._cursor = 0
            self._attempted_models = []
            self._attempted_ids = set()
            self._chain = self._new_chain()
            self._pin_checked = False
            self._moved = False
        messages = request_context.messages
        if (
            not self._pin_checked
            and messages
            and isinstance(last := messages[-1], ModelResponse)
            and last.state == 'suspended'
            and (pin := continuation_pin(last, key=FALLBACK_CAPABILITY_PIN_KEY)) is not None
        ):
            # A resumed suspended response pinned to one of this capability's models, or to the step's model alone.
            # Should its continuation fail, the chain moves on from the model it's pinned to, like it would have when
            # the job was started, instead of starting over with a model that would just start another. That holds
            # whether or not the step's model already is that model.
            index = next((index for index, candidate in enumerate(self.models) if _pin_id(candidate) == pin), None)
            if pin == _pin_id(request_context.model_id or request_context.model):
                self._cursor = (-1 if index is None else index) + 1
            elif index is None:
                raise UserError(
                    f'Cannot resume this suspended response: it was started by {pin!r}, which is neither the '
                    "model selected for this step nor one of `ModelFallback`'s models. Pass that model explicitly to "
                    '`run(model=...)` when resuming.'
                )
            else:
                self._cursor = index + 1
                self._pin_checked = True
                # The step would continue a provider-side job on a model other than the one that started it; send
                # it to that model instead, before anything is sent.
                raise RetryModelRequest(self.models[index])
        self._pin_checked = True
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
            if response.state == 'suspended':
                # Its continuation, in this run or a later one that resumes it, has to go back to the
                # model that started the provider-side job, not to the top of the chain.
                stamp_continuation_pin(
                    response,
                    _pin_id(request_context.model_id or request_context.model),
                    key=FALLBACK_CAPABILITY_PIN_KEY,
                )
            return response
        # The core records a rejected response's tokens and cost before the next attempt, so a
        # rejected generation is still paid for in `RunUsage` even though it never enters history,
        # and records the attempt on the response that answers, as `FallbackModel` does.
        raise self._advance()

    def _new_chain(self) -> FallbackChain:
        return FallbackChain(self._predicates, owner='ModelFallback')

    def _advance(self) -> RetryModelRequest:
        """Return the exception that moves to the next untried candidate, or the aggregated failure."""
        while self._cursor < len(self.models):
            candidate = self.models[self._cursor]
            self._cursor += 1
            if not self._already_attempted(candidate):
                self._moved = True
                return RetryModelRequest(candidate)
        # The attempts are the agent's to list, as it made them: it records each failed one, and
        # lists them on this group once it ends the request.
        self._chain.exhausted()

    def _already_attempted(self, candidate: Model | KnownModelName | str) -> bool:
        if not isinstance(candidate, Model):
            return candidate in self._attempted_ids
        # The attempted model may be wrapped by now, e.g. by a durable-execution capability.
        for attempted in self._attempted_models:
            while attempted is not candidate and isinstance(attempted, WrapperModel):
                attempted = attempted.wrapped
            if attempted is candidate:
                return True
        return False

    @classmethod
    def from_spec(
        cls,
        models: KnownModelName | str | Sequence[KnownModelName | str],
        *,
        fallback_on: _SpecFallbackError | Sequence[_SpecFallbackError] = 'ModelAPIError',
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
        return cls(*model_names, fallback_on=tuple(error_types))


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
    """How a candidate is identified in a continuation pin: the string it was given as, or its `model_id`."""
    return model if isinstance(model, str) else model.model_id
