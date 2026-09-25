from __future__ import annotations as _annotations

import dataclasses
import json
from abc import abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import Any, ClassVar, Literal, TypeAlias, cast

from opentelemetry.trace import INVALID_SPAN, Span, SpanKind
from opentelemetry.util.types import AttributeValue
from pydantic import JsonValue
from typing_extensions import assert_never, deprecated

from .. import _utils, usage
from .._instrumentation import model_attributes, open_request_policy, record_uncaught_errors, safe_to_json
from .._output import DEFAULT_OUTPUT_TOOL_DESCRIPTION, DEFAULT_OUTPUT_TOOL_NAME
from .._run_context import RunContext
from .._warnings import PydanticAIDeprecationWarning
from ..exceptions import ModelAPIError, UnexpectedModelBehavior, UserError
from ..messages import (
    BaseToolReturnPart,
    CachePoint,
    CompactionPart,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponseStreamEvent,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..profiles import ModelProfile, merge_profile
from ..providers import InterfaceClient
from ..settings import ModelSettings
from ..tools import ToolDefinition
from ..usage import RequestUsage
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    _unconverted_speech_part_error,  # pyright: ignore[reportPrivateUsage]
    _unsynthesized_tool_availability_delta_error,  # pyright: ignore[reportPrivateUsage]
    check_allow_model_requests,
)

__all__ = (
    'ChoiceAnswer',
    'ChoiceQuestion',
    'DecisionAnswer',
    'DecisionHandOff',
    'DecisionModel',
    'DecisionModelSettings',
    'DecisionQuestion',
    'DecisionRequest',
    'DecisionResponse',
    'DecisionStreamedResponse',
    'NoulAnswer',
    'NoulCriteria',
    'NoulQuestion',
    'ScoreAnswer',
    'ScoreQuestion',
    'UnfillableRoute',
    'UnsureRoute',
)


@dataclass(kw_only=True)
class NoulCriteria:
    """Descriptions of the two outcomes of a yes/no question."""

    true: JsonValue = None
    """Description of the yes outcome."""
    false: JsonValue = None
    """Description of the no outcome."""


@dataclass(kw_only=True)
class NoulQuestion:
    """A yes/no question whose answer is the probability of yes."""

    instructions: JsonValue = None
    """What to decide about the state."""
    criteria: NoulCriteria | None = None
    """Descriptions of the yes and no outcomes."""
    type: Literal['noul'] = 'noul'
    """The Decisions protocol question type."""


@dataclass(kw_only=True)
class ChoiceQuestion:
    """A question that selects one named option."""

    criteria: dict[str, JsonValue]
    """Option labels mapped to their descriptions."""
    instructions: JsonValue = None
    """What to decide about the state."""
    type: Literal['choice'] = 'choice'
    """The Decisions protocol question type."""


@dataclass(kw_only=True)
class ScoreQuestion:
    """A question that scores the state against ordered levels."""

    criteria: list[JsonValue]
    """One description per level, starting at zero."""
    instructions: JsonValue = None
    """What to decide about the state."""
    type: Literal['score'] = 'score'
    """The Decisions protocol question type."""


DecisionQuestion: TypeAlias = NoulQuestion | ChoiceQuestion | ScoreQuestion
"""A question supported by the Decisions protocol."""


@dataclass(kw_only=True)
class NoulAnswer:
    """The probability of yes for a yes/no question."""

    noul: float
    """The probability of yes, from 0 to 1."""
    type: Literal['noul'] = 'noul'
    """The Decisions protocol answer type."""


@dataclass(kw_only=True)
class ChoiceAnswer:
    """The selected option and its probability distribution."""

    choice: str
    """The selected option label."""
    confidence: float
    """Confidence in the selected option."""
    probabilities: dict[str, float]
    """Probability for each option label."""
    type: Literal['choice'] = 'choice'
    """The Decisions protocol answer type."""


@dataclass(kw_only=True)
class ScoreAnswer:
    """A score and its probability distribution across levels."""

    score: float
    """The expected score along the ordered levels."""
    confidence: float
    """Confidence in the score."""
    probabilities: dict[int, float]
    """Probability for each level."""
    legend: dict[int, JsonValue] = field(default_factory=dict[int, JsonValue])
    """The descriptions of the levels, as the backend echoes them back, if it does.

    Not read to build the output, which comes from `score` alone; kept so the answer is recorded as it was received.
    """
    type: Literal['score'] = 'score'
    """The Decisions protocol answer type."""


DecisionAnswer: TypeAlias = NoulAnswer | ChoiceAnswer | ScoreAnswer
"""An answer returned by the Decisions protocol."""


@dataclass(kw_only=True)
class DecisionRequest:
    """A request to decide typed questions about a state."""

    state: JsonValue
    """The text or JSON value to decide about."""
    questions: dict[str, DecisionQuestion]
    """Named questions to answer about the state."""


@dataclass(kw_only=True)
class DecisionResponse:
    """The answers to a [`DecisionRequest`][pydantic_ai.models.decision.DecisionRequest], and what produced them."""

    answers: dict[str, DecisionAnswer]
    """Answers keyed by question name."""
    model_name: str
    """The model that produced the answers."""
    usage: RequestUsage = field(default_factory=RequestUsage)
    """Usage for this request."""
    provider_response_id: str | None = None
    """The backend's identifier for this request, if it returned one.

    Recorded as `gen_ai.response.id` on the request's `decide` span. It is not copied to the
    `ModelResponse`, which can be built from two Decisions requests.
    """


def _wire(value: DecisionQuestion | DecisionAnswer) -> dict[str, Any]:
    """A question or answer in the Decisions protocol's JSON shape: `type` first, and unset fields left out.

    What a `decide` span records, so it reads as what was sent and received.
    Read field by field rather than through `asdict`, which would deep-copy whatever a backend put in an answer.
    """
    wire: dict[str, Any] = {'type': value.type}
    for name in (f.name for f in dataclasses.fields(value) if f.name != 'type'):
        item = getattr(value, name)
        if isinstance(item, NoulCriteria):
            # A question's own dataclass, which is the protocol's to shape, unlike what a backend answers.
            item = dataclasses.asdict(item)
        if isinstance(value, NoulQuestion) and name == 'criteria' and item is not None:
            # Either side of a yes/no can be left undescribed, and is then left out, like the question's own fields.
            item = {outcome: meaning for outcome, meaning in item.items() if meaning is not None}
        if item is not None:
            wire[name] = item
    return wire


_UNSUPPORTED_FIELD_HINT = (
    'Use `bool`, a `Literal` or `Enum` of two or more strings or whole numbers, a `float` bounded with `ge=0` and '
    '`le=1`, a `list` of a `Literal` or `Enum`, a rubric of whole numbers from 0 with a description per level in its '
    'schema, or a model of these.'
)

# A pick-one or a rubric still says what it is asking through its options; a yes/no may have nothing else.
_ASKS_NOTHING = (
    'Output field {name!r} asks the model nothing. A question is not part of the text being judged: give the field '
    'a description, or the agent `instructions`, and leave the prompt to the material the question is about. '
    'A `system_prompt` will not do: a decision model is told what was said, not what to ask.'
)


class DecisionModelSettings(ModelSettings, total=False):
    """Settings used for a decision model request."""

    decision_boolean_threshold: float
    """How likely a yes has to be before a `bool` field is `True`, from 0 to 1. Default: 0.5.

    A decision model answers a yes/no with the probability of yes, and the default rounds it: what the framework
    cannot know is what `True` has to mean for you. Raise it where a false positive is the expensive mistake and a
    `True` should be earned, lower it where a false negative is. It applies to every `bool` field and to each option
    of a `list` of a `Literal` or `Enum`, which is one yes/no per option; a `float` bounded with `ge=0` and `le=1`
    returns the probability itself and is not thresholded.

    Reported confidence is the distance from the threshold rather than from the probability, scaled to run from 0
    at the threshold to 1 at certainty, so a yes at 0.8 under a threshold of 0.75 reports the narrow margin it is.
    """

    decision_route_threshold: float
    """How likely the picked route has to be before it is taken, from 0 to 1. Default: unset, so the pick always is.

    With tools attached, or a union of output types, one more question asks which route the text calls for: a
    tool, an output type, an output function or `None`. The likeliest route is taken. With this set, a pick whose
    own probability is below it raises [`UnsureRoute`][pydantic_ai.models.decision.UnsureRoute] instead, before
    any request to fill it. That is a [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a
    [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] with a language model behind the decision model
    hands that model the step; without one, the run raises it.

    A route taken without a pick is not held to it: the one route left when every other has returned this turn, or
    a single output type with nothing else on offer. A higher threshold hands off more steps and gets more of the
    rest right; tune it on labelled examples of your own.

    This is not a guard for a tool with side effects, such as a refund or an account change: require approval
    for that tool instead.
    """


class DecisionHandOff(ModelAPIError):
    """A decision model handed the step off instead of answering it: the base of the hand-offs it raises.

    A [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a
    [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] with a language model behind the decision model
    hands that model the whole step by default, tools and all, and only the steps the decision model hands off cost
    a language model call. Pass `fallback_on=DecisionHandOff` to hand off only these, and let an error from the
    decision model's backend fail the run rather than go to the language model.
    """

    route: str
    """The route the model picked, by the label the route question offered it under."""

    probability: float
    """How likely the model found the picked route, from 0 to 1."""

    def __init__(self, model_name: str, route: str, probability: float, message: str):
        self.route = route
        self.probability = probability
        super().__init__(model_name, message)


class UnfillableRoute(DecisionHandOff):
    """A decision model picked a route whose fields or arguments it cannot fill.

    A tool with an argument the model cannot express, such as a free-form `str`, or an output type with such a
    field. See [`DecisionHandOff`][pydantic_ai.models.decision.DecisionHandOff] for how a
    [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] takes the step.
    """

    def __init__(self, model_name: str, route: str, probability: float):
        super().__init__(
            model_name,
            route,
            probability,
            f'{model_name} picked {route!r} (probability {probability:.2f}) but cannot fill it. Put a model that '
            'can behind it: `FallbackModel(decision_model, language_model)` hands `language_model` this step.',
        )

    @property
    @deprecated('`tool_name` is deprecated, use `route` instead.', category=PydanticAIDeprecationWarning)
    def tool_name(self) -> str:
        """Deprecated alias for [`route`][pydantic_ai.models.decision.DecisionHandOff.route].

        For a tool, the tool's name. For an output type, the name the route question offered it under, as `Reply`,
        rather than the name of the output tool Pydantic AI made for it.
        """
        return self.route

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.model_name, self.route, self.probability)


class UnsureRoute(DecisionHandOff):
    """A decision model picked a route less likely than `decision_route_threshold`.

    Raised before any request to fill the route. See
    [`DecisionHandOff`][pydantic_ai.models.decision.DecisionHandOff] for how a
    [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] takes the step.
    """

    probabilities: dict[str, float]
    """The probability the model gave every route, by label."""

    threshold: float
    """The `decision_route_threshold` the pick fell below."""

    def __init__(self, model_name: str, route: str, probabilities: dict[str, float], threshold: float):
        self.probabilities = probabilities
        self.threshold = threshold
        probability = probabilities[route]
        super().__init__(
            model_name,
            route,
            probability,
            f'{model_name} picked {route!r} with probability {probability:.2f}, below '
            f'`decision_route_threshold` ({threshold:.2f}). Put a model behind it to take the steps it is unsure '
            'of: `FallbackModel(decision_model, language_model)` hands `language_model` this step.',
        )

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.model_name, self.route, self.probabilities, self.threshold)


@dataclass(frozen=True)
class _Limits:
    """How many options a pick-one and how many levels a rubric can have on this model, `None` for no limit.

    Read from the model's `max_choice_options` and `max_score_levels` once per request, so that turning fields into
    questions can refuse a pick-one the backend would reject before anything is sent, and ask whole numbers with
    more levels than a rubric can have as a pick-one instead.
    """

    choice_options: int | None
    score_levels: int | None


@dataclass(init=False)
class DecisionModel(Model[InterfaceClient]):
    """Base class for decision models: models that answer typed questions about a text rather than write text.

    A decision model is sent a *state*, the text or JSON value to judge, and a set of named questions of three
    kinds: a yes/no ([`NoulQuestion`][pydantic_ai.models.decision.NoulQuestion]), a pick-one
    ([`ChoiceQuestion`][pydantic_ai.models.decision.ChoiceQuestion]), and a score against an ordered rubric
    ([`ScoreQuestion`][pydantic_ai.models.decision.ScoreQuestion]). It answers each one with a probability or a
    distribution. That exchange is the Decisions protocol, and this class maps an agent run onto it, so that an
    agent whose job is to decide something runs on a decision model like on any other model:

    - Each field of the `output_type` is one question, and its type picks the kind: a `bool` is a yes/no, a
      `Literal` or `Enum` of strings is a pick-one, and whole numbers from 0 with a description per level are a
      rubric. A `list` or `dict` of options is one yes/no per option, and a nested model is its fields. A field of
      any other type is a [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent, unless there
      is another route to take, as below.
    - The field's description is the question, the output type's docstring its goal, and the agent's
      `instructions` framing shared by every question; a nested field's question also carries what it sits in.
      The latest user prompt is the text to judge, the message history before it goes along beside it, and once a
      tool has returned, what was done since goes along apart from both.
    - With tools attached, or a union of output types, one more pick-one asks which route the text calls for, and
      the likeliest is taken. The fields of every route the model can fill are asked beside it, each on the premise
      of its route, and only the taken route's answers are read; past a size cutoff, a picked route with fields is
      filled in a second request instead. A route whose fields the model cannot express, a single output type's
      included, is raised as [`UnfillableRoute`][pydantic_ai.models.decision.UnfillableRoute], for a
      [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] to hand to a language model. A pick below
      `decision_route_threshold`, when set, is raised as [`UnsureRoute`][pydantic_ai.models.decision.UnsureRoute]
      the same way.
    - Each field's confidence, the full distribution of each pick-one and rubric, and the route pick are reported
      in [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details].

    The answers arrive in one piece, so a streamed run gets the whole answer as one event.

    To support a backend, subclass this, implement [`decide`][pydantic_ai.models.decision.DecisionModel.decide]
    along with `model_name`, `system` and `base_url`, and set `max_choice_options` and `max_score_levels` to the
    backend's limits. See [Decision models](https://pydantic.dev/docs/ai/models/decision/) for the full rules
    and an example.
    """

    max_choice_options: ClassVar[int | None] = None
    """The most options the backend accepts in one pick-one question, or `None` for no limit.

    A pick-one field with more options, or more routes than this on the route question, is a
    [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent.
    """

    max_score_levels: ClassVar[int | None] = None
    """The most levels the backend accepts in one rubric, or `None` for no limit.

    Whole numbers from 0 with more levels than this are not a rubric, so a field of them is asked as a pick-one
    instead, and counts against `max_choice_options`.
    """

    @cached_property
    def profile(self) -> ModelProfile:
        """The model profile, with text output off whatever the provider or `profile=` says.

        A decision model answers questions and has no way to write text, so this is a fact about the class
        rather than a default to override: with text output left on, an `output_type` like `[Ticket, str]`
        would pass the shared request preparation and have its `str` branch silently never taken.
        """
        return merge_profile(super().profile, ModelProfile(supports_text_output=False))

    @abstractmethod
    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        """Send one request to the backend and return its answers.

        This is called once per request the model makes: once per step, or twice when a route is picked in one
        request and, past the size cutoff for asking every route's fields up front, filled in a second. Every
        question in `request.questions` needs an answer of the matching kind under the same name.

        Raise [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] or
        [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] when the backend fails, so a
        [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] can take over;
        [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] when it returns something that
        cannot be read; and [`UserError`][pydantic_ai.exceptions.UserError] when the request cannot be sent as
        given. Forward `timeout`, `extra_headers` and `extra_body` from `model_settings` where the backend
        supports them.
        """
        raise NotImplementedError()

    @asynccontextmanager
    async def _decide(
        self,
        request: DecisionRequest,
        model_settings: DecisionModelSettings,
        *,
        route: str | None = None,
        fields: bool = False,
        route_question: str | None = None,
        route_questions: Mapping[str, list[str]] | None = None,
    ) -> AsyncGenerator[tuple[DecisionResponse, Span]]:
        """Send one request through `decide()`, in a `decide` span when an instrumented request is open.

        Yields the response and the span, which stays open for the caller to record what the run made of the
        answers before it closes. Nothing more is sent inside it: a request that fills a picked route is a sibling
        of the one that picked it, not its child. Outside an instrumented request the span is a non-recording one.

        `route` is the label of the route whose fields this request asks, as the route question offers it, when it
        was or is being picked from others; `fields` says the request asks field questions at all; `route_question`
        is the key of the question that picks between routes, and `route_questions` the keys of the fields
        asked beside it, by the label of the route they belong to.

        A pick the step can't take is raised inside the span, as [`UnsureRoute`][pydantic_ai.models.decision.UnsureRoute]
        or [`UnfillableRoute`][pydantic_ai.models.decision.UnfillableRoute], and recorded on it like any other
        error, with the picked route's label on the exception event.
        """
        policy = open_request_policy()
        if policy is None:
            yield await self.decide(request, model_settings), INVALID_SPAN
            return
        include_content = policy.include_content
        with (
            policy.tracer.start_as_current_span(
                f'decide {self.model_name}',
                attributes=_decide_span_attributes(
                    self,
                    request,
                    model_settings,
                    route=route,
                    fields=fields,
                    route_question=route_question,
                    route_questions=route_questions,
                    include_content=include_content,
                ),
                kind=SpanKind.CLIENT,
                record_exception=False,
                set_status_on_exception=False,
            ) as span,
            record_uncaught_errors(span, include_content=include_content, event_attributes=_hand_off_attributes),
        ):
            response = await self.decide(request, model_settings)
            if span.is_recording():
                # A span that isn't recording, as a sampler can make it, keeps nothing, so nothing is built for it.
                span.set_attributes(_decide_response_attributes(response, request, route_question, include_content))
            yield response, span

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        # The one place a single output type's own name survives, since its tool is `final_result`. Read before
        # preparing, which drops `output_object` outside native and prompted output.
        output_name = output_object.name if (output_object := model_request_parameters.output_object) else None
        model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
        output_tools, hand_offs = _output_tools(model_request_parameters)
        # A withheld tool is not on any wire; one revealed through the history is, and the model sees the history.
        function_tools = [
            tool
            for tool in model_request_parameters.function_tools
            if model_request_parameters.visibility_of(tool.name) != 'withheld'
        ]
        offered = [*hand_offs, *function_tools]
        tools, done = _tools_left(messages, offered)
        routes = _route_labels(output_tools, tools, output_name)
        forced_tool = tools[0] if not output_tools and len(tools) == 1 and len(offered) > 1 else None
        if forced_tool is not None and (
            _none_route(forced_tool) or not _properties(forced_tool.parameters_json_schema)
        ):
            # Preserve the no-request path: there is no state or question to build when no arguments need filling.
            return self._forced(forced_tool, next(iter(routes)))
        state = _map_messages(messages, turn=done)
        instruction_parts = self._get_instruction_parts(messages, model_request_parameters) or []
        instructions = '\n\n'.join(part.content for part in instruction_parts) or None
        settings = cast(DecisionModelSettings, model_settings or {})
        # The bars are read before anything is sent: a setting outside 0 to 1 is a coding error, and finding
        # out from a rejected answer would mean paying for the request that carried the prompt and history.
        # An unset route bar is 0, which no probability is below, so every pick is taken.
        route_threshold = _threshold(settings, 'decision_route_threshold', 0.0)
        boolean_threshold = _threshold(settings, 'decision_boolean_threshold', _DEFAULT_BOOLEAN_THRESHOLD)
        limits = _Limits(choice_options=self.max_choice_options, score_levels=self.max_score_levels)
        if forced_tool is not None:
            # Every other route has returned this turn, so the one left is taken without a choice question.
            return await self._forced_with_arguments(
                forced_tool, next(iter(routes)), state, instructions, settings, boolean_threshold, limits
            )
        fillable = [tool for tool in output_tools if _expressible(tool, instructions, limits)]
        if (
            output_tools
            and not fillable
            and not any(_none_route(tool) or _expressible(tool, instructions, limits) for tool in offered)
        ):
            # A route the model cannot fill is a hand-off, but only while some other route is a real alternative:
            # an output type it can fill, a route with nothing to fill, or a tool whose arguments it can. With
            # none of them the choice is decided before it is asked: every answer hands off, so the request that
            # asks it buys nothing, and every run pays for the decision model on top of the model behind it.
            if len(output_tools) == 1:
                # Alone, the output type's own unsupported field is the error, as it names what to change.
                _Ask.about(output_tools[0], instructions, limits, label=None)
            raise UserError(
                'None of the output types can be filled by this model, so every answer would be handed off and '
                'the request asking which would be wasted. Give the agent an `output_type` it can fill, or drop '
                'it from this model.'
            )
        if not tools and len(output_tools) == 1:
            # One output type and nothing else on offer: there is no route to pick, only fields to fill.
            output_tool = output_tools[0]
            if not fillable:
                # Every other route has returned this turn, and the one left cannot be filled: it is taken without
                # a choice question, like the last tool left, and handing it off needs no request either.
                raise UnfillableRoute(self.model_name, next(iter(routes)), 1.0)
            ask = _Ask.about(output_tool, instructions, limits, label=None)
            async with self._decide(DecisionRequest(state=state, questions=ask.questions), settings, fields=True) as (
                response,
                span,
            ):
                args, provider_details, confidence = ask.answers(response, boolean_threshold)
                _record_outcome(span, confidence)
            return self._response(output_tool, args, response.usage, response.model_name, provider_details)

        speculation = _Speculation.about(routes, output_tools, state, instructions, limits)
        questions = speculation.questions()
        route_key = _route_question(questions, routes, output_tools, tools, instructions, limits)

        # A picked route whose fields are asked in a second request, past the size cutoff: the route, its label and
        # its questions.
        to_fill: tuple[ToolDefinition, str, _Ask] | None = None
        async with self._decide(
            DecisionRequest(state=state, questions=questions),
            settings,
            fields=bool(speculation.asks),
            route_question=route_key,
            route_questions={label: list(keys) for label, keys in speculation.keys.items()},
        ) as (response, span):
            route_details = _route_picked(response.answers.get(route_key), routes)
            label = route_details['choice']
            probability = route_details['probabilities'][label]
            # A pick the step can't take is raised here, inside the span of the request that picked it, so the span
            # records it. `UnsureRoute` and `UnfillableRoute` are `ModelAPIError`s, and a `FallbackModel` that gives
            # a language model the step ends the model request span without an error.
            if probability < route_threshold:
                raise UnsureRoute(self.model_name, label, route_details['probabilities'], route_threshold)
            route = routes[label]
            confidence = None
            if label in speculation.asks:
                # The fields were asked beside the route question, speculatively, and only the taken route's answers
                # are read: answers to a route not taken describe nothing in this response.
                args, provider_details, confidence = speculation.answers(label, response, boolean_threshold)
            elif not _none_route(route) and _properties(route.parameters_json_schema):
                fill = _Ask.to_fill(route, instructions, limits, label=label)
                if fill is None:
                    # A route whose fields the model cannot express is handed off before a request to fill it.
                    raise UnfillableRoute(self.model_name, label, probability)
                # Filled below, once this request's span has closed: the fill is its sibling, not its child.
                to_fill = route, label, fill
                args, provider_details = {}, {}
            else:
                # Nothing to write, so the call is made on the pick alone, and no answer built it.
                args, provider_details = _route_args(route), _unanswered()
            _record_outcome(span, confidence)
        response_usage = response.usage

        if to_fill is not None:
            route, label, fill = to_fill
            response, args, provider_details = await self._fill(label, fill, state, settings, boolean_threshold)
            response_usage += response.usage
            # `RequestUsage.requests` is fixed at 1, so usage cannot say that this turn asked twice: the
            # choice and the fill are two requests inside one step. The count is reported here, and only
            # here, so it appears exactly when it differs from what usage reports. See #8498.
            provider_details['requests'] = 2
        provider_details['route'] = route_details
        return self._response(route, args, response_usage, response.model_name, provider_details)

    def _response(
        self,
        route: ToolDefinition,
        args: dict[str, Any],
        response_usage: RequestUsage,
        model_name: str,
        provider_details: dict[str, Any],
    ) -> ModelResponse:
        """The one tool call a decision model's step answers with."""
        return ModelResponse(
            parts=[ToolCallPart(route.name, args, _utils.generate_tool_call_id())],
            usage=response_usage,
            model_name=model_name,
            provider_name=self.system,
            provider_url=self.base_url,
            provider_details=provider_details,
            finish_reason='tool_call',
        )

    async def _fill(
        self,
        label: str,
        ask: _Ask,
        state: JsonValue,
        settings: DecisionModelSettings,
        boolean_threshold: float,
    ) -> tuple[DecisionResponse, dict[str, Any], dict[str, Any]]:
        """Ask a selected route's fields in a second request.

        One helper for both routes the model picks and then fills: a tool's arguments, and a union member's fields.
        They are the same two steps, and a route whose fields the model cannot express is the same hand-off either
        way, decided by the caller before this is reached, with `_Ask.to_fill`, and raised as
        [`UnfillableRoute`][pydantic_ai.models.decision.UnfillableRoute] inside the span of the request that picked
        it. This is also how a single output type the model cannot express is taken when tools or a `None` are on
        offer beside it: it is a route like any other.

        `label` is the route's name on the route question, from `_route_labels`, which the fill's questions carry in
        their premise, so that the model sees one name for one route across the two requests.
        """
        try:
            async with self._decide(
                DecisionRequest(state=state, questions=ask.questions),
                settings,
                route=label,
                fields=True,
            ) as (response, span):
                args, provider_details, confidence = ask.answers(response, boolean_threshold)
                _record_outcome(span, confidence)
        except (ModelAPIError, UnexpectedModelBehavior) as e:
            # The first request committed to this route. Letting a fallback model rerun the whole original step
            # could silently choose another route, so a failure while filling is terminal and names that route.
            raise UnexpectedModelBehavior(
                f'{self.model_name} selected {label!r}, but failed while filling its fields: {e}'
            ) from e
        return response, args, provider_details

    async def _forced_with_arguments(
        self,
        tool: ToolDefinition,
        label: str,
        state: JsonValue,
        instructions: str | None,
        settings: DecisionModelSettings,
        boolean_threshold: float,
        limits: _Limits,
    ) -> ModelResponse:
        """Fill the arguments of the one route left, without a choice request."""
        fill = _Ask.to_fill(tool, instructions, limits, label=label)
        if fill is None:
            raise UnfillableRoute(self.model_name, label, 1.0)
        response, args, details = await self._fill(label, fill, state, settings, boolean_threshold)
        details['route'] = _forced_route(label)
        return self._response(tool, args, response.usage, response.model_name, details)

    def _forced(self, tool: ToolDefinition, label: str) -> ModelResponse:
        """Call the one argumentless route left, without asking the model."""
        details = {**_unanswered(), 'route': _forced_route(label)}
        return ModelResponse(
            parts=[ToolCallPart(tool.name, _route_args(tool), _utils.generate_tool_call_id())],
            usage=usage.RequestUsage(),
            model_name=self.model_name,
            provider_name=self.system,
            provider_url=self.base_url,
            provider_details=details,
            finish_reason='tool_call',
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        # The answers arrive in one piece, so the whole answer is the one event; a streamed run keeps working.
        response = await self.request(messages, model_settings, model_request_parameters)
        yield DecisionStreamedResponse(model_request_parameters, response)


@dataclass
class DecisionStreamedResponse(StreamedResponse):
    """A decision model's whole answer as one event, so that a streamed run works on a model that cannot stream."""

    _response: ModelResponse

    def __post_init__(self):
        self._usage = self._response.usage
        self.provider_details = self._response.provider_details
        self.finish_reason = self._response.finish_reason

    async def close_stream(self) -> None:
        """No live stream to close: the whole answer was in hand before the first event."""

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for i, part in enumerate(self._response.parts):
            assert isinstance(part, ToolCallPart)  # `request` builds nothing else
            # `ToolCallPart` subclasses narrow `args` to a `TypedDict`; the parts manager takes the plain union.
            yield self._parts_manager.handle_tool_call_part(
                vendor_part_id=i,
                tool_name=part.tool_name,
                args=cast('str | dict[str, Any] | None', part.args),
                tool_call_id=part.tool_call_id,
            )

    @property
    def model_name(self) -> str:
        return self._response.model_name or ''

    @property
    def provider_name(self) -> str | None:
        return self._response.provider_name

    @property
    def provider_url(self) -> str | None:
        return self._response.provider_url

    @property
    def timestamp(self) -> datetime:
        return self._response.timestamp


_DEFAULT_BOOLEAN_THRESHOLD = 0.5


def _hand_off_attributes(error: Exception) -> dict[str, AttributeValue]:
    """The label of the picked route a hand-off names, for its exception event.

    The exception's message names it too, but a message is only recorded with content, and a label is not content.
    """
    return {'pydantic_ai.decision.route': error.route} if isinstance(error, DecisionHandOff) else {}


# What each kind of answer keeps without content: its numbers, and what they are keyed by only where that is not
# content. A score's probabilities are keyed by level numbers; a choice's are keyed by option labels, which can quote
# the text being judged, as can the picked choice itself and a rubric's legend.
_NUMERIC_ANSWER_FIELDS: dict[str, tuple[str, ...]] = {
    'noul': ('noul',),
    'choice': ('confidence',),
    'score': ('score', 'confidence', 'probabilities'),
}


def _decide_span_attributes(
    model: DecisionModel[Any],
    request: DecisionRequest,
    settings: DecisionModelSettings,
    *,
    route: str | None,
    fields: bool,
    route_question: str | None,
    route_questions: Mapping[str, list[str]] | None,
    include_content: bool,
) -> dict[str, AttributeValue]:
    """A `decide` span's attributes from the request, before it is sent.

    The questions are always recorded by type, keyed like the answers, so a trace shows what kind of decision was
    asked even without content; their instructions and criteria are the user's words, and need `include_content`.
    The routes are recorded by their labels, the names of the user's tools and output types, which are not
    content, so they are recorded either way.
    """
    questions = {
        name: _wire(question) if include_content else {'type': question.type}
        for name, question in request.questions.items()
    }
    thresholds = {'boolean': _threshold(settings, 'decision_boolean_threshold', _DEFAULT_BOOLEAN_THRESHOLD)}
    if 'decision_route_threshold' in settings:
        # Unset, every pick is taken, which no number says better than its absence.
        thresholds['route'] = _threshold(settings, 'decision_route_threshold', 0.0)
    attributes: dict[str, AttributeValue] = {
        'gen_ai.operation.name': 'decide',
        **model_attributes(model),
        'pydantic_ai.decision.questions': safe_to_json(questions).decode(),
        'pydantic_ai.decision.thresholds': safe_to_json(thresholds).decode(),
    }
    json_attributes = ['pydantic_ai.decision.questions', 'pydantic_ai.decision.thresholds']
    if route is not None:
        attributes['pydantic_ai.decision.route'] = route
    if fields:
        # Set once the response is in, and declared here with the rest.
        json_attributes.append('pydantic_ai.decision.confidence')
    if route_question is not None:
        question = request.questions[route_question]
        assert isinstance(question, ChoiceQuestion)  # `_tool_question` asks nothing else
        attributes['pydantic_ai.decision.route_question'] = route_question
        attributes['pydantic_ai.decision.route_options'] = safe_to_json(list(question.criteria)).decode()
        json_attributes.append('pydantic_ai.decision.route_options')
    if route_questions:
        # Which route each field question belongs to. The keys can't be split to tell: a label and a nested field's
        # name can both contain dots.
        attributes['pydantic_ai.decision.route_questions'] = safe_to_json(route_questions).decode()
        json_attributes.append('pydantic_ai.decision.route_questions')
    if include_content:
        # The state is text or JSON, as it is on the wire, and only JSON is declared as such.
        if isinstance(request.state, str):
            attributes['pydantic_ai.decision.state'] = request.state
        else:
            attributes['pydantic_ai.decision.state'] = safe_to_json(request.state).decode()
            json_attributes.append('pydantic_ai.decision.state')
    json_attributes.append('pydantic_ai.decision.answers')
    attributes['logfire.json_schema'] = safe_to_json(
        {
            'type': 'object',
            'properties': {
                name: {'type': 'array' if name == 'pydantic_ai.decision.route_options' else 'object'}
                for name in json_attributes
            },
        }
    ).decode()
    return attributes


def _decide_response_attributes(
    response: DecisionResponse, request: DecisionRequest, route_question: str | None, include_content: bool
) -> dict[str, AttributeValue]:
    """A `decide` span's attributes from the response.

    Usage is this request's alone, and deliberately not `gen_ai.usage.*`: the `chat` span above reports the sum of its
    requests there, and a backend that adds up usage across spans would count it twice.

    The answers are what the backend sent, so recording them must not be able to fail the request: whatever goes
    wrong reading them for the span leaves them out of it, and the request goes on as it would uninstrumented.
    """
    attributes: dict[str, AttributeValue] = {
        'gen_ai.response.model': response.model_name,
        'pydantic_ai.decision.usage.input_tokens': response.usage.input_tokens,
        'pydantic_ai.decision.usage.output_tokens': response.usage.output_tokens,
    }
    if response.provider_response_id is not None:
        attributes['gen_ai.response.id'] = response.provider_response_id
    offered: frozenset[str] = frozenset()
    if route_question is not None and isinstance(question := request.questions.get(route_question), ChoiceQuestion):
        offered = frozenset(question.criteria)
    try:
        answers = {
            name: _answer_attribute(
                answer, include_content=include_content, offered=offered if name == route_question else None
            )
            for name, answer in response.answers.items()
            # Not an answer at all, from a backend breaking its contract, is left out. The run rejects it as it would
            # without instrumentation, with `UnexpectedModelBehavior`, which the span records.
            if isinstance(answer, NoulAnswer | ChoiceAnswer | ScoreAnswer)
        }
        attributes['pydantic_ai.decision.answers'] = safe_to_json(answers).decode()
    except Exception:
        # Instrumentation must not fail an otherwise-successful request, as a backend's answer nested too deeply to
        # serialize would; the span is left without the answers rather than with a partial or misleading set.
        pass
    return attributes


def _answer_attribute(
    answer: DecisionAnswer, *, include_content: bool, offered: frozenset[str] | None
) -> dict[str, Any]:
    """One answer as a `decide` span records it.

    With content, it's the answer as received. Without content, it keeps its type and numbers, built from those
    fields alone: a score its level probabilities too, which are keyed by number, but not a choice its option
    probabilities, nor the picked option or a rubric's legend, which can quote the state. A `type` the protocol
    doesn't have keeps nothing else.

    The route question's answer, `offered` being the labels its request offered, keeps its pick and probabilities
    without content, as they are route labels: the names of the user's tools and output types. Only those labels
    are kept, as a backend could answer with any string, and a pick or probability under a label that wasn't
    offered is left out.
    """
    if include_content:
        return _wire(answer)
    attribute: dict[str, Any] = {'type': answer.type}
    if offered is not None and isinstance(answer, ChoiceAnswer):
        if answer.choice in offered:
            attribute['choice'] = answer.choice
        if _is_number(answer.confidence):
            attribute['confidence'] = answer.confidence
        attribute['probabilities'] = {
            label: p for label, p in answer.probabilities.items() if label in offered and _is_number(p)
        }
        return attribute
    for name in _NUMERIC_ANSWER_FIELDS.get(answer.type, ()):
        value = getattr(answer, name, None)
        if name == 'probabilities' and isinstance(value, Mapping):
            attribute[name] = {
                level: p
                for level, p in cast(Mapping[object, object], value).items()
                if _is_number(level) and _is_number(p)
            }
        elif _is_number(value):
            attribute[name] = value
    return attribute


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _record_outcome(span: Span, confidence: dict[str, float] | None) -> None:
    """Record on a `decide` span the confidence of the answers the step used, which is not content.

    `confidence` is the thresholded confidence per field question, keyed like the questions, or `None` when the
    step used none of the span's field answers.
    """
    if confidence is not None and span.is_recording():
        # Always recorded, so only numbers: a backend could put anything where its confidence belongs.
        numbers = {key: value for key, value in confidence.items() if _is_number(value)}
        span.set_attribute('pydantic_ai.decision.confidence', safe_to_json(numbers).decode())


def _threshold(settings: DecisionModelSettings, name: str, default: float) -> float:
    """A probability setting, which is only meaningful inside the range the model answers in."""
    threshold = cast(float, settings.get(name, default))
    if not 0 <= threshold <= 1:
        raise UserError(f'`{name}` must be between 0 and 1; got {threshold!r}.')
    return threshold


def _fanned_in(
    name: str, options: Mapping[Any, Any], answers: Mapping[str, object], boolean_threshold: float
) -> tuple[dict[str, tuple[bool, float]], dict[str, float]]:
    """Read back the yes/no per option that `_fan_out` asked, as verdicts and as the raw probabilities.

    A list and a mapping ask the same questions and differ only in how the verdicts are shaped afterwards.
    """
    labelled: dict[str, float] = {}
    for option in options:
        answer = answers.get(f'{name}.{option}')
        if not isinstance(answer, NoulAnswer):
            raise UnexpectedModelBehavior(
                f'Unexpected answer from the model for output field {name!r}, option {option!r}: {answer!r}'
            )
        labelled[option] = answer.noul
    return {option: _verdict(p, boolean_threshold) for option, p in labelled.items()}, labelled


def _verdict(probability: float, threshold: float) -> tuple[bool, float]:
    """Whether the probability of yes clears the bar, and how far from the bar it landed.

    A yes/no is answered with the probability of yes and no confidence of its own, so what is lost in rounding the
    probability to an answer is how sure that answer is. That is the distance from the bar, scaled to run 0 to 1 on
    whichever side of it the answer fell — like the confidence reported for the other two kinds of question. Under
    the default bar of 0.5 this is the distance from the coin flip, doubled: a no returned at 0.01 reports 0.98.
    """
    if not 0 <= probability <= 1:
        # Both scalings divide by the room left on their side of the bar, which a probability outside 0 to 1
        # can make zero. A malformed answer is the model's to report, not a crash.
        raise UnexpectedModelBehavior(f'Unexpected probability from the model: {probability!r}')
    if probability >= threshold:
        # An answer exactly at the bar is the least sure one there is, including when the bar is certainty.
        sureness = (probability - threshold) / (1 - threshold) if threshold < 1 else 0.0
    else:
        sureness = (threshold - probability) / threshold
    # The arithmetic leaves float noise, as 0.8600000000000001 for 0.86, which six places remove without losing
    # anything a probability from the model carries. A fanned-out field's confidence is the least of these, so it
    # is rounded too.
    return probability >= threshold, round(sureness, 6)


def _answers(
    answers: Mapping[str, object],
    properties: dict[str, dict[str, Any]],
    questions: dict[str, DecisionQuestion],
    boolean_threshold: float,
    defaulted: frozenset[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    """The output's arguments and `provider_details` from the model's answers to the field questions.

    Also each question's confidence, keyed like the questions: `provider_details['confidence']` is keyed by field,
    and gives a field that fans out the least sure of its options, where this keeps each option's own.
    """
    args: dict[str, Any] = {}
    confidence: dict[str, float] = {}
    by_question: dict[str, float] = {}
    probabilities: dict[str, dict[str, float]] = {}
    scores: dict[str, float] = {}
    for name, prop in properties.items():
        prop, none_option = _optional(prop)
        if keys := _mapping_options(prop):
            # A mapping keeps every option with the answer it got, unlike a list.
            verdicts, labelled = _fanned_in(name, keys, answers, boolean_threshold)
            _set(args, name, {key: chosen for key, (chosen, _) in verdicts.items()})
            confidence[name] = min(sureness for _, sureness in verdicts.values())
            by_question.update({f'{name}.{option}': sureness for option, (_, sureness) in verdicts.items()})
            probabilities[name] = labelled
            continue
        if prop.get('type') == 'array':
            # A list is the options that came back yes, in their order.
            verdicts, labelled = _fanned_in(name, _options(prop['items']) or {}, answers, boolean_threshold)
            _set(args, name, [option for option, (chosen, _) in verdicts.items() if chosen])
            confidence[name] = min(sureness for _, sureness in verdicts.values())
            by_question.update({f'{name}.{option}': sureness for option, (_, sureness) in verdicts.items()})
            probabilities[name] = labelled
            continue
        answer = answers.get(name)
        if isinstance(questions[name], NoulQuestion) and isinstance(answer, NoulAnswer):
            if (bound := _bounded(prop)) is not None:
                # The probability is the answer, so there is no separate confidence to report: a field
                # that asks for the number would otherwise get it back twice under two names. The bound is
                # the units it is asked in, so the same answer comes back as 0.42 or as 42.
                _set(args, name, answer.noul * bound)
            else:
                chosen, sureness = _verdict(answer.noul, boolean_threshold)
                _set(args, name, chosen)
                confidence[name] = by_question[name] = sureness
        elif isinstance(questions[name], ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
            if answer.choice not in none_option:
                # The option itself is written back, looked up by its label rather than parsed out of it: `1`
                # and `'1'` are two options, and only the lookup knows which one the label stood for.
                labelled = _labelled(_options(prop) or {})
                _set(args, name, labelled.get(answer.choice, answer.choice))
            elif 'default' in properties[name]:
                # "None of these" is the absence of an answer, and a default says what to use when there is
                # none, so the field is left out for Pydantic to fill in. The model it belongs to is still put
                # in place, to apply the default in, unless that model has a default of its own to apply instead.
                _slot(args, name)
            else:
                _set(args, name, None)
            confidence[name] = by_question[name] = answer.confidence
            probabilities[name] = answer.probabilities
        elif isinstance(questions[name], ScoreQuestion) and isinstance(answer, ScoreAnswer):
            # `score` is a position along the rubric and falls between levels. The answer has to be one of them,
            # so it is rounded to the nearest; the likeliest level would throw away the ordering that makes a
            # rubric a rubric. A half goes up, unlike `round`.
            _set(args, name, min(int(answer.score + 0.5), max(answer.probabilities)))
            confidence[name] = by_question[name] = answer.confidence
            probabilities[name] = {str(level): p for level, p in answer.probabilities.items()}
            scores[name] = answer.score
        else:
            raise UnexpectedModelBehavior(f'Unexpected answer from the model for output field {name!r}: {answer!r}')
    _leave_out_unanswered(args, properties, defaulted)
    return args, {'confidence': confidence, 'probabilities': probabilities, 'scores': scores}, by_question


def _route_picked(answer: object, routes: dict[str, ToolDefinition]) -> dict[str, Any]:
    """The model's answer to the route question, as `provider_details['route']` reports it.

    Everything in it is a route label, the name the route question offered each route under, looked up in
    `routes`. The pick is reported with its probabilities and what was on offer, so the rate at which picks fall
    below a `decision_route_threshold` can be watched, and a tool that was withheld this turn can be seen to have been.
    """
    if (
        not isinstance(answer, ChoiceAnswer)
        or answer.choice not in answer.probabilities
        or not all(0 <= p <= 1 for p in answer.probabilities.values())
    ):
        raise UnexpectedModelBehavior(f'Unexpected answer from the model for the route question: {answer!r}')
    if answer.choice not in routes:
        raise UnexpectedModelBehavior(f'The model picked a route it was not offered: {answer.choice!r}')
    return {'choice': answer.choice, 'probabilities': dict(answer.probabilities), 'offered': list(routes)}


def _forced_route(label: str) -> dict[str, Any]:
    """`provider_details['route']` for the one route left, taken without asking: certain, because it was alone."""
    return {'choice': label, 'probabilities': {label: 1.0}, 'offered': [label]}


def _unanswered() -> dict[str, Any]:
    """`provider_details` for a response no answer built: a route taken on the pick alone has no fields to report."""
    return {'confidence': {}, 'probabilities': {}, 'scores': {}}


def _expressible(tool: ToolDefinition, instructions: str | None, limits: _Limits) -> bool:
    """Whether the model could fill this route's fields, decided without sending anything."""
    try:
        _Ask.about(tool, instructions, limits, label=None)
    except UserError:
        return False
    return True


_REQUEST_TOKENS = 260
"""About how many tokens a request costs before its state and questions: what a second request adds besides them."""

_SPECULATION_TOKENS = 16_000
"""The most tokens a request that asks every route's fields up front is allowed to grow to."""

_STATE_CHARS_PER_TOKEN = 6
_QUESTION_CHARS_PER_TOKEN = 4
"""How many characters of JSON make a token, near enough to size a request before it is sent.

Prose, as a state mostly is, packs more characters into a token than a question's labels and punctuation do; both
were fitted on requests to `jev-latest`, where they recover each request's input tokens to within a few percent.
"""


def _state_tokens(state: JsonValue) -> float:
    return len(json.dumps(state, ensure_ascii=False)) / _STATE_CHARS_PER_TOKEN


def _question_tokens(questions: Mapping[str, DecisionQuestion]) -> float:
    return (
        sum(len(json.dumps(dataclasses.asdict(question), ensure_ascii=False)) for question in questions.values())
        / _QUESTION_CHARS_PER_TOKEN
    )


@dataclass(frozen=True)
class _Speculation:
    """The routes whose fields are asked beside the route question, before the pick, so the pick needs no fill.

    Asking a route's fields only once it is picked costs a second request, with the state sent again. Asking every
    route's fields up front costs the questions of the routes that are not taken, which the model answers and the
    run throws away. So every route the model can fill is asked up front while those questions cost no more than the
    request they save, and the request stays small enough to answer as quickly as a small one; past that, a route is
    picked first and filled after, and only a single output type beside tools is still asked up front, as it always
    has been. A route the model cannot fill is never asked, a single output type included: picking it hands the step
    off, so for the cost it is a route with nothing to ask, whose pick would throw every question away.

    Each route's questions are keyed `'<label>.<field>'`, so the same field name on two routes is two questions. A
    question's key is for reading its answer back, not something the model sees, so a key that collides, as a label
    with a dot in it can make one, just gets `_` appended; the answers are read back by the keys each route was
    given, never by parsing one.
    """

    asks: dict[str, _Ask]
    keys: dict[str, dict[str, str]]
    """Per route label, each of its own question keys by the key it was asked under."""

    @classmethod
    def about(
        cls,
        routes: dict[str, ToolDefinition],
        output_tools: list[ToolDefinition],
        state: JsonValue,
        instructions: str | None,
        limits: _Limits,
    ) -> _Speculation:
        asks: dict[str, _Ask] = {}
        for label, route in routes.items():
            if not _none_route(route) and _properties(route.parameters_json_schema):
                try:
                    asks[label] = _Ask.about(route, instructions, limits, label=label)
                except UserError:
                    # Picking a route the model cannot fill hands the step off, a single output type's included,
                    # so there is nothing to ask about it; it still counts as a route that could be taken.
                    continue
        sizes = {label: _question_tokens(ask.questions) for label, ask in asks.items()}
        state_tokens = _state_tokens(state)
        # The route taken is not known yet, so the questions thrown away are all but the smallest route's, at most,
        # and all of them when the route taken could be one with nothing asked about it.
        smallest = min(sizes.values()) if len(sizes) == len(routes) else 0
        unpicked = sum(sizes.values()) - smallest
        if unpicked > _REQUEST_TOKENS + state_tokens or state_tokens + sum(sizes.values()) > _SPECULATION_TOKENS:
            asks = {
                label: ask for label, ask in asks.items() if routes[label] in output_tools and len(output_tools) == 1
            }
        keys: dict[str, dict[str, str]] = {}
        taken: set[str] = set()
        for label, ask in asks.items():
            keys[label] = {}
            for key in ask.questions:
                scoped = f'{label}.{key}'
                while scoped in taken:
                    scoped += '_'
                taken.add(scoped)
                keys[label][scoped] = key
        return cls(asks, keys)

    def questions(self) -> dict[str, DecisionQuestion]:
        """Every speculated route's questions, under their scoped keys."""
        return {
            scoped: self.asks[label].questions[key] for label, keys in self.keys.items() for scoped, key in keys.items()
        }

    def answers(
        self, label: str, response: DecisionResponse, boolean_threshold: float
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
        """The taken route's arguments and details, from the answers to its own questions alone.

        Also each of its questions' confidence, keyed as it was asked, under its scoped key.
        """
        keys = self.keys[label]
        own = {key: response.answers.get(scoped) for scoped, key in keys.items()}
        args, details, confidence = self.asks[label].read(own, boolean_threshold)
        # A question of a field that fans out gets its option's key appended, so the scoped key is found the same way.
        scoped_by_key = {key: scoped for scoped, key in keys.items()}
        return args, details, {scoped_by_key[key]: value for key, value in confidence.items()}


_NONE_OF_THESE = 'None of these.'


def _null(schema: dict[str, Any] | bool) -> bool:
    """Whether a schema is `None`, whatever else the user wrote about it.

    Pydantic renders a bare `None` as `{'type': 'null'}`, but `Annotated[None, Field(description=...)]` — how a
    user says what returning nothing means — renders the description beside it. The type is what makes it `None`;
    the rest is the user's own words, and a route or a field is no less `None` for having some.
    """
    return isinstance(schema, dict) and schema.get('type') == 'null'


def _none_route(tool: ToolDefinition) -> bool:
    """Whether this output tool is the `None` member of a union: a route that returns nothing.

    Pydantic AI wraps a bare `None` output type in an object with one `null` property, so the route has a field
    in its schema and yet only one value that field could ever take. There is nothing to ask about it: it is one
    more option to pick, the same thing `_optional` makes of an `X | None` field one level down.
    """
    properties = list(_properties(tool.parameters_json_schema).values())
    return tool.kind == 'output' and len(properties) == 1 and _null(properties[0])


def _route_args(tool: ToolDefinition) -> dict[str, Any]:
    """The arguments to call a route with when nothing is filled: none, or the `None` a `None` route wraps."""
    return {name: None for name in _properties(tool.parameters_json_schema)} if _none_route(tool) else {}


def _wrapped(tool: ToolDefinition) -> dict[str, Any] | None:
    """The single property Pydantic AI wraps an output type that is not object-like in, if this is one.

    A `Literal`, an `Enum`, a `Choices` set and a bare `None` all reach a model as one property of an object,
    because only an object can be a tool's arguments. What such a type says about itself is written on that
    property, since there is no class for it to be a docstring on. `outer_typed_dict_key` is the wrapping,
    named by whoever did it, so it is read rather than guessed back out of the schema's shape.
    """
    key = tool.outer_typed_dict_key
    return _properties(tool.parameters_json_schema).get(key) if key else None


def _purpose(tool: ToolDefinition) -> str | None:
    """What a route says about itself, wherever it managed to say it.

    An output type says this in its docstring, and `ToolOutput(description=...)` says it for a type that has
    no docstring to write it in. A type that is not object-like has neither: `Choices(description=...)`
    describes the set it wraps, which lands on the wrapped property rather than on the tool. It is the same
    sentence either way, so the route question reads it from there too.
    """
    if described := _described(tool):
        return described
    wrapped = _wrapped(tool)
    if wrapped is None:
        return None
    # An `Enum` is wrapped as a `$ref` to its own definition, which is where its docstring is.
    return _resolved(wrapped, tool.parameters_json_schema).get('description')


def _route_description(tool: ToolDefinition) -> str | None:
    """What a route says about itself on the route question.

    A `None` route can say nothing anywhere, so the library supplies the phrase — unless the user named the
    route themselves, which says more about what declining means on this agent than the stock phrase does.
    """
    if _none_route(tool):
        return _purpose(tool) or _NONE_OF_THESE
    return _purpose(tool) or tool.description


def _output_tools(
    model_request_parameters: ModelRequestParameters,
) -> tuple[list[ToolDefinition], list[ToolDefinition]]:
    """The output tools with fields for the model to fill, and the ones that take no arguments.

    An output function that takes nothing, or only the run context, is a hand-off the model can pick without
    filling anything. Several output types with fields are a union: the model picks which one the text calls for,
    then fills that one's fields in a second request, the same two steps a selected tool's arguments take. Text
    output is refused earlier, by the shared request preparation, on the profile's `supports_text_output`.
    """
    with_fields: list[ToolDefinition] = []
    hand_offs: list[ToolDefinition] = []
    for tool in model_request_parameters.output_tools:
        (hand_offs if _none_route(tool) or not _properties(tool.parameters_json_schema) else with_fields).append(tool)
    return with_fields, hand_offs


def _schema(node: dict[str, Any] | bool) -> dict[str, Any]:
    """A schema as an object. JSON Schema allows `true` and `false` anywhere a schema goes, for anything and nothing.

    Pydantic never renders them, but a hand-written schema, as `Tool.from_schema` takes, may (see #8621). As objects
    they are `{}` and `{'not': {}}`, which ask the model nothing it can answer, so a field of either is unsupported.
    """
    if isinstance(node, bool):
        return {} if node else {'not': {}}
    return node


def _resolved(schema: dict[str, Any], root: dict[str, Any] | None = None) -> dict[str, Any]:
    """A schema through its `$ref` to `root`'s `$defs`, with what the referring schema says beside it taking precedence.

    `root` is the schema the `$defs` are on, which is `schema` itself for the top-level `$ref` Pydantic renders a
    model that refers to itself as.
    """
    if not (ref := schema.get('$ref')):
        return schema
    definition = _schema((root or schema)['$defs'][ref.removeprefix('#/$defs/')])
    return {**definition, **{k: v for k, v in schema.items() if k not in ('$ref', '$defs')}}


def _properties(schema: dict[str, Any]) -> dict[str, Any]:
    """A schema's properties, through the top-level `$ref` Pydantic renders a model that refers to itself as."""
    return _resolved(schema).get('properties', {})


@dataclass(frozen=True)
class _Fields:
    """A route's fields, flattened, and what each one sits inside.

    `properties` are the leaf fields by their dotted names, `defaulted` the nested models with a default in the
    schema, by the same names, for `_leave_out_unanswered`, and `context` each leaf's ancestor chain, for its
    question to carry.
    """

    properties: dict[str, dict[str, Any]]
    defaulted: frozenset[str]
    context: dict[str, list[str]]


def _enum_description(prop: dict[str, Any] | bool, defs: dict[str, Any]) -> str | None:
    """What an `Enum` field's own description hides: the class docstring, as `'<EnumName>: <docstring>'`.

    A field's description takes precedence over the docstring of the `Enum` it refers to, so the docstring is not
    the question; but it still says what the options are for, which the field's question rarely repeats. The
    `Enum` is reached as Pydantic renders it: a `$ref`, one beside `None` for an optional field, or the items of a
    `list`.
    """
    prop = _schema(prop)
    if not (own := prop.get('description')):
        return None
    refs = [prop.get('$ref'), _schema(prop.get('items', {})).get('$ref')]
    refs += [_schema(option).get('$ref') for option in prop.get('anyOf', [])]
    for ref in refs:
        if not ref:
            continue
        definition = _schema(defs.get(ref.removeprefix('#/$defs/'), {}))
        doc = definition.get('description')
        if 'enum' in definition and isinstance(doc, str) and doc and doc != own:
            return f'{definition.get("title") or ref.rsplit("/", 1)[-1]}: {doc}'
    return None


def _ancestor(path: str, raw: dict[str, Any], defs: dict[str, Any]) -> list[str]:
    """What a nested model field says about itself, as entries of its leaves' ancestor chain.

    The field's own description, beside its `$ref`, as `'<dotted path>: <description>'`, and the model's docstring,
    from its definition, as `'<ModelName>: <docstring>'`: the first says which of its kind this one is, and the second
    what that kind is.
    """
    entries: list[str] = []
    if isinstance(own := raw.get('description'), str) and own:
        entries.append(f'{path}: {own}')
    if ref := raw.get('$ref'):
        definition = _schema(defs[ref.removeprefix('#/$defs/')])
        if isinstance(doc := definition.get('description'), str) and doc:
            entries.append(f'{definition.get("title") or ref.rsplit("/", 1)[-1]}: {doc}')
    return entries


def _fields(output_tool: ToolDefinition) -> _Fields:
    """The output schema's fields, flattened, with `$ref`s to `$defs` (how Pydantic renders an `Enum` or a model) resolved.

    A nested model is its fields, named `outer.inner`: each question is about one value, and a field of a field is
    still one value. The answers are nested back into place by `_set`. The nested models with a default in the
    schema come back alongside, by the same names, for `_leave_out_unanswered`.

    Flattening drops what the models on the way say about themselves, which is often all that tells two leaves of
    the same name apart: `origin.city` and `destination.city` both ask "Which city?". So each leaf also gets its
    ancestor chain, root to leaf and without the route itself, which its question carries as `context`: each
    ancestor field's own description, as `'<dotted path>: <description>'`, and each nested model's docstring, as
    `'<ModelName>: <docstring>'`. An `Enum` leaf whose own description hides the `Enum`'s docstring gets that
    docstring as a last entry.
    """
    schema = output_tool.parameters_json_schema
    defs: dict[str, Any] = schema.get('$defs', {})
    context: dict[str, list[str]] = {}

    def resolve(prop: dict[str, Any] | bool) -> dict[str, Any]:
        prop = _schema(prop)
        if ref := prop.get('$ref'):
            prop = {**resolve(defs[ref.removeprefix('#/$defs/')]), **{k: v for k, v in prop.items() if k != '$ref'}}
        if 'items' in prop:
            prop = {**prop, 'items': resolve(prop['items'])}
        if 'anyOf' in prop:
            prop = {**prop, 'anyOf': [resolve(option) for option in prop['anyOf']]}
        if 'propertyNames' in prop:
            # How an `Enum`-keyed mapping names its keys, as a `$ref` to the enum's own schema.
            prop = {**prop, 'propertyNames': resolve(prop['propertyNames'])}
        return prop

    def flatten(
        properties: dict[str, Any], prefix: str, seen: frozenset[str], chain: list[str]
    ) -> dict[str, dict[str, Any]]:
        fields: dict[str, dict[str, Any]] = {}
        for name, prop in properties.items():
            if '.' in name:
                raise UserError(
                    f'Output field {prefix + name!r} is not supported by this model: a dot in a field name is how '
                    'a nested field is named. Rename it.'
                )
            raw = _schema(prop)
            ref = raw.get('$ref')
            prop = resolve(prop)
            if prop.get('type') == 'object' and prop.get('properties'):
                if ref is not None and ref in seen:
                    # A model that contains itself with no way out is infinitely many questions, so there is no
                    # depth at which to stop asking. An optional or list self-reference is refused on the field
                    # itself before the walk gets here.
                    raise UserError(
                        f'Output field {prefix + name!r} is not supported by this model: a model that contains '
                        'itself has no end to fill, and every question is asked up front. Give the field a type '
                        'that does not contain itself.'
                    )
                if 'default' in prop:
                    defaulted.add(f'{prefix}{name}')
                ancestors = [*chain, *_ancestor(f'{prefix}{name}', raw, defs)]
                fields.update(flatten(prop['properties'], f'{prefix}{name}.', seen | {ref} if ref else seen, ancestors))
            else:
                fields[f'{prefix}{name}'] = prop
                leaf_context = [*chain]
                if enum := _enum_description(raw, defs):
                    leaf_context.append(enum)
                if leaf_context:
                    context[f'{prefix}{name}'] = leaf_context
        return fields

    defaulted: set[str] = set()
    properties = flatten(_properties(schema), '', frozenset(), [])
    return _Fields(properties, frozenset(defaulted), context)


def _set(args: dict[str, Any], name: str, value: Any) -> None:
    """Put a flattened field's answer back where it belongs, `outer.inner` under `outer`."""
    parent, leaf = _slot(args, name)
    parent[leaf] = value


def _leave_out_unanswered(
    args: dict[str, Any], properties: dict[str, dict[str, Any]], defaulted: frozenset[str], prefix: str = ''
) -> bool:
    """Leave out a nested model with a default when nothing under it was answered, and say if `args` holds an answer.

    Every field under such a model was left out for its own default to apply, and the model's default says what to
    use when there is nothing to fill it with. An empty model put in place would apply the defaults inside it
    instead. One without a default of its own stays, empty, for those inner defaults to apply in.
    """
    answered = False
    for name, value in list(args.items()):
        path = f'{prefix}{name}'
        if path in properties:
            answered = True
        elif _leave_out_unanswered(value, properties, defaulted, f'{path}.'):
            answered = True
        elif path in defaulted:
            del args[name]
    return answered


def _slot(args: dict[str, Any], name: str) -> tuple[dict[str, Any], str]:
    """Where a flattened field's answer goes: the arguments of the model it belongs to, put in place, and its name there."""
    *path, leaf = name.split('.')
    for part in path:
        args = args.setdefault(part, {})
    return args, leaf


def _optional(prop: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """An `X | None` field as `X` plus one more option, "none of these", named and described; any other as it is.

    An explicit option lets the model say that nothing fits, which is a different thing from being unsure: reading
    `None` off low confidence would conflate the two, and the field's confidence already reports the second. What
    the user wrote about the `None` branch, `Annotated[None, Field(description=...)]`, says what picking it means
    on this field better than the stock phrase does, so it describes the option when there is one.
    """
    options = prop.get('anyOf')
    # Exactly one of the two has to be `None`, and the other one has to be something: a union of nothing but
    # `None`s has no `X` to ask about, and is refused as the unsupported field it is rather than crashing here.
    if not options or len(options) != 2 or sum(_null(option) for option in options) != 1:
        return prop, {}
    none = next(option for option in options if _null(option))
    inner = next(option for option in options if not _null(option))
    prop = {**inner, **{k: v for k, v in prop.items() if k not in ('anyOf', 'default')}}
    key = 'none'
    while key in (_options(prop) or {}):
        key += '_'
    meaning = none.get('description')
    return prop, {key: meaning if isinstance(meaning, str) and meaning else _NONE_OF_THESE}


def _list_options(name: str, prop: dict[str, Any]) -> dict[Any, str | None]:
    """The options a `list` field fans out over, or a `UserError` saying why it is not a list of them."""
    labels = _options(prop['items']) if 'items' in prop else None
    if not labels or len(labels) < 2 or not all(isinstance(label, str) for label in labels):
        raise UserError(
            f'Output field {name!r} is not supported by this model: a list must be of two or more string '
            f'options. {_UNSUPPORTED_FIELD_HINT}'
        )
    if 'maxItems' in prop or 'minItems' in prop:
        # Every option is asked about and every yes is kept, so how many come back is the model's answer rather
        # than something that can be held to a limit -- the same reason a mapping carrying one is refused. Checked
        # after the shape, so a `tuple`, which carries both intrinsically, is still told it is not a list.
        raise UserError(f'Output field {name!r} is not supported by this model. {_UNSUPPORTED_FIELD_HINT}')
    return labels


def _fan_out(name: str, ask: dict[str, JsonValue], labels: dict[Any, str | None]) -> dict[str, NoulQuestion]:
    """One yes/no per option, asked with the field's question and that option's description.

    Both a list of options and a mapping keyed by them ask this; what differs is how the answers are read back.
    """
    questions: dict[str, NoulQuestion] = {}
    for label, meaning in labels.items():
        option = f'{label}: {meaning}' if meaning else label
        questions[f'{name}.{label}'] = NoulQuestion(instructions={**ask, 'option': option})
    return questions


def _bounded(prop: dict[str, Any]) -> float | None:
    """The upper bound of a number field that asks for a probability, or `None` if it is not one.

    A probability is what a yes/no is answered with, so a number field has to be bounded to be one. The bound itself
    is only a scale: `ge=0, le=1` is the probability as it comes back, and `ge=0, le=100` the same answer as a
    percentage. What the field asks is unchanged; only the units it is written in differ.
    """
    if prop.get('type') != 'number' or prop.get('minimum') != 0:
        return None
    if 'multipleOf' in prop:
        # A probability can land anywhere between the bounds, so a field that only accepts steps along them is not
        # a probability in different units; it is a set of levels, which is what a rubric is for.
        return None
    maximum = prop.get('maximum')
    return maximum if isinstance(maximum, (int, float)) and not isinstance(maximum, bool) and maximum > 0 else None


def _mapping_options(prop: dict[str, Any]) -> dict[Any, str | None] | None:
    """The options a `dict[Literal, bool]` field is keyed by, or `None` if the field is not one.

    A mapping keyed by options and valued by yes/no is a fan-out like a list of those options. A mapping of
    anything else -- `dict[str, str]`, `dict[str, int]` -- has no options to fan out over and is not one.
    """
    if prop.get('type') != 'object' or prop.get('properties'):
        return None
    if 'maxProperties' in prop or 'minProperties' in prop:
        # Every option is asked about and every answer is kept, so there is no way to return fewer or more
        # keys than the mapping declares, and a limit on how many there may be could only be broken.
        return None
    # A plain yes/no per key and nothing else. Anything narrower -- a `const`, an `enum`, a `Literal[True]` --
    # would forbid an answer the model is free to give, and `True` rather than a schema is how `dict[str, Any]`
    # says its values are unconstrained. Matching the whole schema rather than one key of it refuses both, and
    # whatever else is put there next.
    if prop.get('additionalProperties') != {'type': 'boolean'}:
        return None
    names = prop.get('propertyNames')
    if not isinstance(names, dict):
        # `propertyNames: true` says the keys are unconstrained, so there are no options to fan out over.
        return None
    return _options(cast('dict[str, Any]', names))


def _options(prop: dict[str, Any]) -> dict[Any, str | None] | None:
    """The options of a pick-one schema, each with its description, or `None` when the schema is not one."""
    if 'enum' in prop:
        return dict.fromkeys(prop['enum'])
    if 'anyOf' in prop and all('const' in option for option in prop['anyOf']):
        return {option['const']: option.get('description') for option in prop['anyOf']}
    return None


def _pickable(options: dict[Any, str | None]) -> bool:
    """Whether every option is a string or a whole number, which is what a pick-one can offer by name."""
    # `bool` is an `int` in Python, but `True` is not a label, and a pick-one of booleans is a yes/no.
    return all(
        isinstance(option, str) or (isinstance(option, int) and not isinstance(option, bool)) for option in options
    )


def _labelled(options: dict[Any, str | None]) -> dict[str, Any]:
    """Each option of a pick-one under the label the model picks it by, in the order they are declared.

    A string is its own label, so a pick-one of strings asks exactly what it always has. A whole number is labelled
    by its digits, unless a string option already is those digits: `Literal['1', 1]` is two options, and the answer
    has to say which of them was picked, so the number becomes `1 (number)`. The answer is read back through this
    mapping rather than by converting the label, so the argument gets the option itself, with its type.
    """
    taken = {option for option in options if isinstance(option, str)}
    labelled: dict[str, Any] = {}
    for option in options:
        label = option
        if not isinstance(option, str):
            label = str(option)
            while label in taken:
                label = f'{label} (number)'
            taken.add(label)
        labelled[label] = option
    return labelled


def _rubric(options: dict[Any, str | None], limits: _Limits) -> list[str] | None:
    """A rubric's level descriptions in level order, or `None` when the options are not a rubric.

    A rubric is ordered levels starting at zero, so the levels have to be exactly that, no more of them than the
    model scores against, and every one of them has to say what it means: a rubric whose levels are unexplained is
    not a rubric. Any other whole numbers -- status codes, levels with nothing said about them, more levels than the
    model takes -- are labels, and are a pick-one instead.
    """
    if not all(isinstance(option, int) and not isinstance(option, bool) for option in options):
        return None
    levels = sorted(options)
    if levels != list(range(len(levels))) or len(levels) < 2:
        return None
    if limits.score_levels is not None and len(levels) > limits.score_levels:
        return None
    criteria = [meaning for level in levels if (meaning := options[level])]
    return criteria if len(criteria) == len(levels) else None


def _with_none(
    name: str, options: dict[Any, str | None] | None, none_option: dict[str, str], limits: _Limits
) -> dict[Any, str | None]:
    """An optional field's options with "none of these" as one more, or a `UserError` saying why it cannot be one."""
    if options is not None and _rubric(options, limits) is not None:
        raise UserError(
            f'Output field {name!r} is a rubric, and a rubric cannot be optional: its levels are ordered and '
            f'`None` is not one of them.'
        )
    if options is None or not _pickable(options):
        raise UserError(
            f'Output field {name!r} is not supported by this model: only a pick-one of strings or whole '
            f'numbers can be optional, since `None` is one more option to pick. {_UNSUPPORTED_FIELD_HINT}'
        )
    return {**options, **none_option}


_INSTRUCTIONS_KEY = 'background'
"""The label the agent's `instructions` go under inside a question's labelled parts.

Not `instructions`, which is the wire field those parts are sent in, and not `context`, which is a field's ancestor
chain: three things, three names.
"""

_CONTEXT_KEY = 'context'
"""The label a nested field's ancestor chain goes under, from `_fields`."""

_PREMISE = "If the user's request calls for {label}"


def _premise(tool: ToolDefinition, label: str) -> str:
    """The route a field belongs to, as the condition its question is asked under.

    A field is asked about a route before the route is taken, or in the request that fills it once it is: either
    way the answer only means something if that route is the one the text calls for, so the question says so,
    naming the route by its label on the route question and saying what the route is for.
    """
    described = _described(tool)
    return f'{_PREMISE.format(label=label)}: {described}' if described else f'{_PREMISE.format(label=label)}.'


def _ask(
    name: str,
    prop: dict[str, Any],
    output_tool: ToolDefinition,
    instructions: str | None,
    *,
    label: str | None,
    context: list[str],
) -> dict[str, JsonValue]:
    """What a field asks, as labelled parts: its field, route, ancestors, question, and shared framing.

    `label` is the route's name on the route question when there is one, which makes the route a premise the
    question is asked under; without a route question, the route's docstring is the goal of every question.
    """
    # Only what the user wrote goes to the model. A bare `bool` output is wrapped in a field named `response`
    # by Pydantic AI, and the output tool has a stock description; neither says anything about the question.
    ask: dict[str, JsonValue] = {}
    # A field's name says what is being asked about, which is not the same as asking something, so it
    # goes under `field` and leaves `question` for a question. The wrapper field Pydantic AI puts around
    # a bare output is named `response` and says nothing about anything, so it is not sent at all.
    if name != output_tool.outer_typed_dict_key:
        ask['field'] = name
    if label is not None:
        ask['premise'] = _premise(output_tool, label)
    if context:
        ask[_CONTEXT_KEY] = cast('list[JsonValue]', context)
    if description := prop.get('description'):
        ask['question'] = description
    if label is None and (described := _described(output_tool)):
        ask['goal'] = described
    if instructions:
        # With no field to describe, a bare output's whole question is what the agent was instructed to
        # ask, so it goes where a question goes. Alongside fields of its own it is shared framing.
        ask['question' if 'question' not in ask and 'field' not in ask else _INSTRUCTIONS_KEY] = instructions

    return ask


@dataclass(frozen=True)
class _Ask:
    """One route's fields as questions, and how to read their answers back.

    `_fields` and `_questions` are derived from the same route, and `_answers` needs both of them again to
    make sense of what comes back, so the three travel together rather than being rebuilt side by side at
    every call site. A turn that picks a route and then fills it builds one of these per request, and what
    a question carries about its route is decided in one place instead of once per caller.
    """

    properties: dict[str, dict[str, Any]]
    questions: dict[str, DecisionQuestion]
    defaulted: frozenset[str]

    @classmethod
    def about(cls, tool: ToolDefinition, instructions: str | None, limits: _Limits, *, label: str | None) -> _Ask:
        """The questions this route's fields become, or a `UserError` if the model cannot express one of them.

        `label` is the route's name on the route question, for a route asked about beside that question or filled
        after it; `None` for the one output type asked when there is no route question.
        """
        fields = _fields(tool)
        questions = _questions(fields, tool, instructions, limits, label=label)
        return cls(fields.properties, questions, fields.defaulted)

    @classmethod
    def to_fill(cls, tool: ToolDefinition, instructions: str | None, limits: _Limits, *, label: str) -> _Ask | None:
        """The questions a picked route's fields become in the request that fills them, or `None` to hand it off.

        `None` means the model cannot express one of the fields, so the route is raised as
        [`UnfillableRoute`][pydantic_ai.models.decision.UnfillableRoute] before anything is sent to fill it.
        """
        try:
            return cls.about(tool, instructions, limits, label=label)
        except UserError:
            return None

    def answers(
        self, response: DecisionResponse, boolean_threshold: float
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
        return self.read(response.answers, boolean_threshold)

    def read(
        self, answers: Mapping[str, object], boolean_threshold: float
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
        return _answers(answers, self.properties, self.questions, boolean_threshold, self.defaulted)


def _questions(
    fields: _Fields,
    output_tool: ToolDefinition,
    instructions: str | None,
    limits: _Limits,
    *,
    label: str | None,
) -> dict[str, DecisionQuestion]:
    """One question per output field, or one per option for a field that fans out."""
    questions: dict[str, DecisionQuestion] = {}
    for name, prop in fields.properties.items():
        ask = _ask(name, prop, output_tool, instructions, label=label, context=fields.context.get(name, []))
        prop, none_option = _optional(prop)
        options = _options(prop)
        if options and all(isinstance(option, bool) for option in options) and not any(options.values()):
            # `Literal[True, False]` spells out the two values a `bool` already has and says nothing about
            # either, so there is nothing to pick between that a yes/no does not ask, and the schema says
            # `boolean` too. Two booleans that *are* described are that same yes/no with criteria, below.
            options = None
        if none_option:
            options = _with_none(name, options, none_option, limits)

        # A single value needs no label, so it is sent bare; the object form earns its keys only once there is
        # more than one thing in it.
        asked: JsonValue | None = next(iter(ask.values())) if len(ask) == 1 else (ask or None)

        if prop.get('type') == 'array':
            # Several options at once is one yes/no per option, all in the same request: does this option apply,
            # asked with the field's question and the option's description.
            questions.update(_fan_out(name, ask, _list_options(name, prop)))
        elif options is not None:
            if (criteria := _rubric(options, limits)) is not None:
                questions[name] = ScoreQuestion(instructions=asked, criteria=cast('list[JsonValue]', criteria))
            elif len(options) == 2 and all(isinstance(option, bool) for option in options):
                # `True` and `False` are the two options a yes/no already has, so an `Enum` or `Literal` of
                # exactly those is that same question, with somewhere to say what each answer means.
                questions[name] = _noul_question(cast('dict[bool, str | None]', options), asked)
            elif len(options) < 2 or not _pickable(options):
                raise UserError(
                    f'Output field {name!r} is not supported by this model: its options are not two or more strings '
                    f'or whole numbers. {_UNSUPPORTED_FIELD_HINT}'
                )
            elif limits.choice_options is not None and len(options) > limits.choice_options:
                raise UserError(
                    f'Output field {name!r} is not supported by this model: it picks from at most '
                    f'{limits.choice_options} options, and this one has {len(options)}.'
                )
            else:
                questions[name] = ChoiceQuestion(
                    instructions=asked,
                    criteria={label: options[option] for label, option in _labelled(options).items()},
                )
        elif keys := _mapping_options(prop):
            # A mapping from options to yes/no asks the same thing per option a list of them does; what
            # differs is the answer, which keeps every option rather than only the ones that came back yes.
            if len(keys) < 2:
                # A key of a JSON object is a string by construction, so only how many there are is in doubt.
                raise UserError(
                    f'Output field {name!r} is not supported by this model: a mapping must be keyed by two or '
                    f'more options. {_UNSUPPORTED_FIELD_HINT}'
                )
            questions.update(_fan_out(name, ask, keys))
        elif prop.get('type') == 'boolean' or _bounded(prop) is not None:
            if not ask:
                raise UserError(_ASKS_NOTHING.format(name=name))
            questions[name] = NoulQuestion(instructions=asked)
        else:
            raise UserError(f'Output field {name!r} is not supported by this model. {_UNSUPPORTED_FIELD_HINT}')
    return questions


_OUTPUT_ROUTE_LABEL = 'output'
"""The label of a single output type that has no name of its own: a bare `bool` or `Literal` that Pydantic AI wraps."""

_OUTPUT_LABEL_SUFFIX = ' (output)'


def _output_route_label(tool: ToolDefinition, output_name: str | None) -> str:
    """The name an output route goes by on the route question, before collisions with other routes are settled.

    The output tool's name is the library's, not the user's: `final_result` for a single output type and
    `final_result_<Member>` for a member of a union, neither of which says anything to the model. So a route is
    labelled by the name the user gave it, wherever that is:

    - a name given with `ToolOutput(name=...)`, which is the tool's name as it stands;
    - a union member's own name, which follows the prefix: `Refund`, `None`, or an output function's name;
    - a single output type's own name, `output_name`: the name of a model or an `Enum`, or of an output function,
      which its tool has nowhere to carry. It comes from the request's `output_object`, which Pydantic AI fills in for
      an output type given without `ToolOutput` and a model prepares away in tool output mode;
    - failing that, a single output type's schema `title`, which is how `ToolOutput(Ticket)` without a name still
      goes by `Ticket`; or `output` for a type that has no name worth offering, a bare `bool`, `Literal` or `list`
      that Pydantic AI wraps in an object, since `bool` says nothing about the route. A wrapped type with a class
      of its own, an `Enum`, is a `$ref` to its definition, and that is how it is told apart.
    """
    if tool.name == DEFAULT_OUTPUT_TOOL_NAME:
        wrapped = _wrapped(tool)
        if output_name and (wrapped is None or '$ref' in wrapped):
            return output_name
        title = _resolved(tool.parameters_json_schema).get('title')
        return title if isinstance(title, str) and title else _OUTPUT_ROUTE_LABEL
    return tool.name.removeprefix(f'{DEFAULT_OUTPUT_TOOL_NAME}_') or _OUTPUT_ROUTE_LABEL


def _route_labels(
    output_tools: list[ToolDefinition], tools: list[ToolDefinition], output_name: str | None
) -> dict[str, ToolDefinition]:
    """Every route on offer under the one label the model knows it by, in the order the route question offers them.

    The label is the option key on the route question and the `chosen` of the fill that follows, so one route has
    one name across both requests. A function tool is labelled by its own name. An output route, including an
    output function that takes no arguments and the `None` member of a union, is labelled by
    `_output_route_label`. Two routes can come out with the same label, as a tool named `Refund` beside an output
    type `Refund` does: the function tool keeps its name, and the output route gets ` (output)` appended until the
    label is free. Function tool names are unique among themselves, so only output routes are ever renamed, and
    in the order they are offered, so the same routes always get the same labels.

    The model's answer is read back through this mapping, never by parsing a label.
    """
    routes = [*output_tools, *tools]
    labels = {tool.name: tool.name for tool in routes if tool.kind != 'output'}
    taken = set(labels.values())
    for tool in routes:
        if tool.kind != 'output':
            continue
        label = _output_route_label(tool, output_name)
        while label in taken:
            label += _OUTPUT_LABEL_SUFFIX
        taken.add(label)
        labels[tool.name] = label
    return {labels[tool.name]: tool for tool in routes}


def _described(tool: ToolDefinition) -> str | None:
    """What the user wrote about a tool, without the stock description generated for an output tool."""
    description = tool.description
    if not description or (tool.kind == 'output' and description.endswith(DEFAULT_OUTPUT_TOOL_DESCRIPTION)):
        return None
    return description


def _tools_left(messages: list[ModelMessage], tools: list[ToolDefinition]) -> tuple[list[ToolDefinition], bool]:
    """The tools still on offer, and whether any tool has returned this turn.

    A tool whose result is already in the turn is not offered again. A decision model judges the text in front of it
    and has no notion of having made a call: with a call and its result in view, the text still calls for the tool, so
    left on offer it is picked again until the usage limit, even with the result set apart under `done`. That goes for
    a call made by a model behind this one too, since this model would propose it again on the same text. A call that
    produced no result, because the tool asked for a retry, leaves the tool on offer. The turn is everything since the
    last user prompt, which is the nearest thing to a run boundary the history has: a result from an earlier turn does
    not withhold the tool, but a judged history that ends in another agent's call to a tool of the same name does.
    """
    returned: set[str] = set()
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                # A new prompt starts a turn, and a result that arrived before it in the same request is the
                # previous turn's.
                returned.clear()
            elif isinstance(part, ToolReturnPart):
                returned.add(part.tool_name)
    return [tool for tool in tools if tool.name not in returned], bool(returned)


_ROUTE_QUESTION = 'Which of these does this call for?'


def _route_question(
    questions: dict[str, DecisionQuestion],
    routes: dict[str, ToolDefinition],
    output_tools: list[ToolDefinition],
    tools: list[ToolDefinition],
    instructions: str | None,
    limits: _Limits,
) -> str:
    """With tools attached, one more question: which route the text calls for, the output types among them.

    The question is keyed `route`, which no field can take, since every field asked beside it is keyed under its
    route's label as `'<label>.<field>'`; each option is a route's label from `routes`. It asks what the situation
    calls for, not what the user asked for: naming the user's request tilts the pick toward doing what was literally
    asked, and away from a route like an escalation that nobody asks for. It carries the agent's instructions beside
    it like a field's question does.

    The fields of the routes the model can fill are asked beside it, or, past `_Speculation`'s size cutoff, a picked
    route's fields are asked in a second request; an unsupported argument leaves the call to a model behind it. The
    output types are the first options, described by what the agent is for, so that filling the output is an action
    weighed against the others. Asking instead whether the model *can* answer would be a question about the question
    rather than about the text, and invites a hand-off on everything. Only what the user wrote describes an output: the
    output type's docstring, or failing that the agent's instructions; the stock output tool description says nothing a
    tool could be weighed against.
    """
    if not output_tools and len(tools) < 2:
        raise UserError(
            'An `output_type` with no fields is not supported by this model; there is nothing to ask the model. '
            'Give it fields, or more than one tool to pick between.'
        )
    criteria: dict[str, JsonValue] = {}
    for label, route in routes.items():
        if route not in output_tools:
            criteria[label] = _route_description(route)
            continue
        described = _purpose(route)
        if not (described or (instructions and len(output_tools) == 1)):
            # With one output type the agent's instructions can say what filling it is for. With several, only
            # each type's own docstring can tell them apart: one instruction cannot describe two different routes.
            raise UserError(
                f'A decision model weighs each route by what it is for, and {label!r} says nothing about itself. '
                'Give the output type a docstring that says what filling it does'
                + ('.' if len(output_tools) > 1 else ', or the agent `instructions`.')
            )
        criteria[label] = described or instructions
    if limits.choice_options is not None and len(criteria) > limits.choice_options:
        raise UserError(
            f'This model picks from at most {limits.choice_options} options, and it is being offered '
            f'{len(criteria)} routes: each output type counts as one beside the tools. Attach fewer tools, or '
            'withhold some of them until they are needed.'
        )
    # The agent's instructions frame the pick as they frame every field, under the same label.
    asked: JsonValue = (
        {'question': _ROUTE_QUESTION, _INSTRUCTIONS_KEY: instructions} if instructions else _ROUTE_QUESTION
    )
    questions['route'] = ChoiceQuestion(instructions=asked, criteria=criteria)
    return 'route'


def _noul_question(options: dict[bool, str | None], asked: JsonValue | None) -> NoulQuestion:
    """A yes/no from an `Enum` or `Literal` of `True` and `False`, with what each answer means.

    A bare `bool` asks the same question and says nothing about its answers, because a `bool` has nowhere to
    write it down. Two described options do, and they become the yes/no's criteria. Only one of the two need be
    described; what is written is sent, and a pair that describes neither never gets here — it is collapsed to a
    plain yes/no by `_questions`, which is also what raises when there is nothing to ask.
    """
    return NoulQuestion(
        instructions=asked,
        criteria=NoulCriteria(true=options[True], false=options[False]),
    )


def _prompt_text(part: UserPromptPart) -> str:
    texts: list[str] = []
    for item in [part.content] if isinstance(part.content, str) else part.content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
        elif isinstance(item, CachePoint):
            pass  # A marker for models that cache a prompt prefix; there is nothing in it to send.
        else:
            raise UserError(
                'Files are not supported by this model: it judges text, so images, audio, video and documents '
                'cannot be sent to it.'
            )
    return '\n\n'.join(texts)


def _tool_return_entry(part: BaseToolReturnPart) -> JsonValue:
    """A tool result as history, or a `UserError` when it carries a file: `model_response_str` would leave it out."""
    if part.files:
        raise UserError('Files are not supported by this model: a file in a tool result cannot be sent to it.')
    return {'tool_return': {'name': part.tool_name, 'content': part.model_response_str()}}


def _request_entry(part: ModelRequestPart) -> JsonValue:
    """A request part as a history entry: a user prompt as what the user said, the rest as what they are."""
    if isinstance(part, SystemPromptPart):
        # Whoever wrote it, a system prompt is something that was said in the conversation, so it is
        # material to judge and not a question to ask. What the model is asked comes from `instructions`.
        return {'system': part.content}
    elif isinstance(part, UserPromptPart):
        return {'user': _prompt_text(part)}
    elif isinstance(part, ToolReturnPart):
        return _tool_return_entry(part)
    elif isinstance(part, RetryPromptPart):
        return {'retry': part.model_response()}
    elif isinstance(part, ToolAvailabilityDeltaPart):  # pragma: no cover
        raise _unsynthesized_tool_availability_delta_error()
    elif isinstance(part, SpeechPart):  # pragma: no cover
        # `Model.prepare_messages` turns realtime speech into `UserPromptPart`s before this runs.
        raise _unconverted_speech_part_error()
    else:
        assert_never(part)


def _response_entries(message: ModelResponse) -> list[JsonValue]:
    """Map a response to history entries, in the order the model produced them."""
    entries: list[JsonValue] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            entries.append({'assistant': part.content})
        elif isinstance(part, ThinkingPart):
            # What a model thought is part of what it did: a judgment can be about the reasoning itself, and a
            # conversation continued from the history should see it as the model that wrote it would.
            # Thinking a provider only returned encrypted, as a `signature` with no text, has nothing to show.
            if part.content:
                entries.append({'thinking': part.content})
        elif isinstance(part, ToolCallPart | NativeToolCallPart):
            entries.append({'tool_call': {'name': part.tool_name, 'args': part.args_as_dict()}})
        elif isinstance(part, NativeToolReturnPart):
            entries.append(_tool_return_entry(part))
        elif isinstance(part, CompactionPart):
            if part.content:
                entries.append({'summary': part.content})
        elif isinstance(part, FilePart):
            raise UserError(
                'Files are not supported by this model: a file in the message history cannot be sent to it.'
            )
        elif isinstance(part, SpeechPart):  # pragma: no cover
            raise _unconverted_speech_part_error()
        else:
            assert_never(part)
    return entries


def _map_messages(messages: list[ModelMessage], *, turn: bool) -> JsonValue:
    """The state to judge.

    The latest user text on its own is the whole state, sent as the plain text it is. With a conversation behind
    it there are two parts to keep apart, so they get named: the text under judgement and the `history` before it.

    With `turn`, a tool has returned since the latest user prompt, and the state splits at that prompt into three:
    the `history` before it, the prompt itself as `text`, and what has been done since under `done` — the calls, their
    results, and anything else in the turn — so the request stays the text being judged while the steps taken for it
    are told apart from it. Every entry lands in exactly one of the three, however the messages arrived: a run's own,
    or a `message_history` passed in that ends partway through a turn.
    """
    if turn and any(isinstance(part, UserPromptPart) for message in messages for part in message.parts):
        return _map_turn(messages)
    history: list[JsonValue] = []
    prompt_parts: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and message is messages[-1]:
                    prompt_parts.append(_prompt_text(part))
                else:
                    history.append(_request_entry(part))
        elif isinstance(message, ModelResponse):
            history.extend(_response_entries(message))
        else:
            assert_never(message)

    text = '\n\n'.join(prompt_parts)
    if not (text or history):
        raise UserError('A request without user text is not supported by this model: it needs text to judge.')
    if not history:
        return text
    state: dict[str, JsonValue] = {'history': history}
    if text:
        state['text'] = text
    return state


def _map_turn(messages: list[ModelMessage]) -> JsonValue:
    """The state split at the latest user prompt, for `_map_messages`."""
    latest = max(
        index
        for index, message in enumerate(messages)
        if isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart) for part in message.parts)
    )
    history: list[JsonValue] = []
    prompt_parts: list[str] = []
    done: list[JsonValue] = []
    for index, message in enumerate(messages):
        if isinstance(message, ModelResponse):
            (history if index < latest else done).extend(_response_entries(message))
            continue
        assert isinstance(message, ModelRequest)
        if index != latest:
            (history if index < latest else done).extend(_request_entry(part) for part in message.parts)
            continue
        # The request holding the latest prompt: what came before its last prompt is the previous turn's, and what
        # came after it is this one's. Its prompts are the text, as they are when it is the last message.
        last_prompt = max(i for i, part in enumerate(message.parts) if isinstance(part, UserPromptPart))
        for i, part in enumerate(message.parts):
            if isinstance(part, UserPromptPart):
                prompt_parts.append(_prompt_text(part))
            else:
                (history if i < last_prompt else done).append(_request_entry(part))
    state: dict[str, JsonValue] = {'history': history} if history else {}
    return {**state, 'text': '\n\n'.join(prompt_parts), 'done': done}
