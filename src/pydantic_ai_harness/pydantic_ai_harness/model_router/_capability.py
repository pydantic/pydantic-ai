"""Model routing through Pydantic AI's model-selection hook."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from functools import cache
from typing import Annotated, Literal, TypeGuard, Union

from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.trace import NoOpTracerProvider, Status, StatusCode
from pydantic import BaseModel, Field, create_model

from pydantic_ai import Agent
from pydantic_ai.capabilities import (
    AbstractCapability,
    Instrumentation,
    ModelSelection,
    ModelSelector,
    durable_operation,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelSelectionContext
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness._usage import reserved_usage_limits

_SPAN_NAME = 'model_router.select'
_INSTRUCTIONS = 'Which model should take the next step of this conversation?'


@dataclass(frozen=True)
class ModelChoice:
    """A selectable model and the guidance for choosing it."""

    model: ModelSelection
    """Model name or `Model` instance selected for this choice."""

    description: str
    """When the router should select this choice."""


class _Route(BaseModel):
    """The router agent's output. Each router narrows `choice` to its own keys."""

    choice: str


@dataclass
class ModelRouter(AbstractCapability[AgentDepsT]):
    """Pick the model for a run from a named menu using another model.

    The router is itself a Pydantic AI agent. Its output is a single `choice` field that takes
    one of the configured keys, with each key's description attached to that option in the
    output schema. This works with language models and with models that only produce typed
    output.

    In `once` mode, the capability makes one router request and keeps that choice for the run.
    In `per_step` mode it routes before every logical model request. If the router request
    raises, the declared `default` choice is used and the main run continues.
    """

    choices: Mapping[str, ModelChoice]
    """Named models available to the router."""

    router_model: ModelSelection
    """Model name or `Model` instance that chooses from `choices`."""

    default: str
    """Choice used when routing fails, cannot run, or the pick's probability is too low."""

    mode: Literal['once', 'per_step'] = 'once'
    """Whether to route once for the run or before every request step."""

    probability_threshold: float | None = None
    """Minimum probability, from 0 to 1, the router must give its pick before it is accepted.

    Read from `provider_details['probabilities']['choice']`, which decision models such as
    TypeSafe report. A router that reports no probability keeps its pick.
    """

    # A stable default `id` lets durable execution recover the capability worker-side without configuration.
    _: KW_ONLY
    id: str | None = 'model_router'

    _router_agent: Agent[None, _Route] = field(init=False, repr=False, compare=False)
    _run_ctx: RunContext[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    _cached_choice: str | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the routing menu and build the internal router agent."""
        self.choices = dict(self.choices)
        if not self.choices:
            raise UserError('ModelRouter.choices must not be empty')
        if any(not name for name in self.choices):
            raise UserError('ModelRouter choice names must not be empty')
        if any(not choice.description for choice in self.choices.values()):
            raise UserError('ModelRouter choice descriptions must not be empty')
        if self.default not in self.choices:
            raise UserError(f'ModelRouter.default must name a configured choice, got {self.default!r}')
        if self.mode not in {'once', 'per_step'}:
            raise UserError("ModelRouter.mode must be 'once' or 'per_step'")
        if self.probability_threshold is not None and not 0 <= self.probability_threshold <= 1:
            raise UserError('ModelRouter.probability_threshold must be between 0 and 1')
        # Built here rather than per selection: `per_step` would otherwise re-infer the router
        # model on every step, and an unresolvable `router_model` would be swallowed by the
        # fallback and silently route every request to `default` for the life of the agent.
        self._router_agent = Agent[None, _Route](
            self.router_model,
            name='model_router',
            deps_type=type(None),
            output_type=self._output_type(),
            instructions=_INSTRUCTIONS,
        )

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge configuration and rebuild the router agent from the combined menu."""
        merged = super().combine(capabilities)
        assert isinstance(merged, cls)
        return replace(merged)

    def _output_type(self) -> type[_Route]:
        # One field whose options each carry their description, so a decision model asks one
        # pick-one question with per-option criteria. A union of output types would instead
        # become several output tools, which a decision model refuses to fill.
        options = tuple(
            Annotated[Literal[name], Field(description=choice.description)] for name, choice in self.choices.items()
        )
        choice_type = Union[options]  # noqa: UP007
        return create_model('ModelRoute', __base__=_Route, choice=(choice_type, ...))

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> ModelRouter[AgentDepsT]:
        """Return a run-scoped router with its own selection cache and the run's context."""
        router = replace(self)
        router._run_ctx = ctx
        return router

    def get_model(self) -> ModelSelector[AgentDepsT]:
        """Return the selector used by Pydantic AI before each request step."""
        return self._select_model

    async def _select_model(self, ctx: ModelSelectionContext[AgentDepsT]) -> ModelSelection:
        run_ctx = self._run_ctx
        if run_ctx is None:
            # Pydantic AI asks the agent-level capability for a bootstrap model before `for_run`,
            # then asks the run-scoped copy again for the first step. Routing here would send a
            # second router request for the same step. This is also the only call that can see a
            # history ending in a response, when a run resumes with tool calls still to run: by the
            # run-scoped call, their results are the request being routed.
            return self.choices[self.default].model
        if self.mode == 'once' and self._cached_choice is not None:
            return self.choices[self._cached_choice].model

        with run_ctx.tracer.start_as_current_span(_SPAN_NAME) as span:
            picked = self.default
            probability: float | None = None
            fallback_reason = 'none'
            try:
                candidate, probability, error_type = await self._route(
                    replace(run_ctx, run_step=ctx.run_step), ctx.messages
                )
            except UserError as error:
                # A request the router can never make, such as files sent to a decision model,
                # is a configuration problem, so it surfaces rather than routing to `default`.
                span.set_attribute('model_router.error.type', type(error).__name__)
                raise
            except Exception as error:
                candidate, error_type = None, type(error).__name__
            if candidate is None:
                fallback_reason = 'error'
                if span.is_recording():
                    span.set_attribute('model_router.error.type', str(error_type))
                    span.set_status(Status(StatusCode.ERROR, 'Router request failed; used the default choice.'))
            elif (
                probability is not None
                and self.probability_threshold is not None
                and probability < self.probability_threshold
            ):
                fallback_reason = 'low_probability'
            else:
                picked = candidate

            if span.is_recording():
                attributes: dict[str, str | int | float] = {
                    'model_router.choice': picked,
                    'model_router.fallback_reason': fallback_reason,
                    'model_router.mode': self.mode,
                    'model_router.run_step': ctx.run_step,
                }
                if probability is not None:
                    attributes['model_router.probability'] = probability
                span.set_attributes(attributes)

        if self.mode == 'once':
            self._cached_choice = picked
        return self.choices[picked].model

    @durable_operation('route')
    async def _route(
        self, ctx: RunContext[AgentDepsT], messages: list[ModelMessage]
    ) -> tuple[str | None, float | None, str | None]:
        """Ask the router agent for a choice key, its probability, and the error type if it failed.

        A durable operation, so replay restores the recorded pick instead of asking again and
        possibly choosing another model. A failure is recorded as the fallback rather than
        inheriting the engine's retry policy, which could stall the run on a best-effort choice.
        """
        try:
            result = await self._router_agent.run(
                message_history=messages,
                usage=ctx.usage,
                usage_limits=reserved_usage_limits(ctx.usage_limits),
                capabilities=[Instrumentation(settings=_instrumentation_settings(ctx))],
            )
        except UserError:
            raise
        except Exception as error:
            return None, None, type(error).__name__
        choice = result.output.choice
        return choice, _pick_probability(result.response.provider_details, choice), None

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable because choices can hold live `Model` instances."""
        return None


def _instrumentation_settings(ctx: RunContext[AgentDepsT]) -> InstrumentationSettings:
    """The parent run's instrumentation, so the router run is traced exactly when the parent is.

    An uninstrumented parent gets settings that record nothing rather than none at all: an agent
    run without instrumentation settings shows Pydantic AI's first-run banner, which would then
    describe the internal router instead of the agent the user wrote.
    """
    for capability in (ctx.capabilities or {}).values():
        if isinstance(capability, Instrumentation):
            return capability.settings
    return _uninstrumented()


@cache
def _uninstrumented() -> InstrumentationSettings:
    return InstrumentationSettings(tracer_provider=NoOpTracerProvider(), meter_provider=NoOpMeterProvider())


def _pick_probability(provider_details: Mapping[str, object] | None, pick: str) -> float | None:
    probabilities = (provider_details or {}).get('probabilities')
    if not _is_object_mapping(probabilities):
        return None
    options = probabilities.get('choice')
    if not _is_object_mapping(options):
        return None
    probability = options.get(pick)
    return float(probability) if isinstance(probability, (int, float)) and not isinstance(probability, bool) else None


def _is_object_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)
