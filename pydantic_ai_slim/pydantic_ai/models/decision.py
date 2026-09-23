from __future__ import annotations as _annotations

from abc import abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import Any, ClassVar, Literal, TypeAlias, cast

from pydantic import JsonValue
from typing_extensions import assert_never

from .. import _utils, usage
from .._output import DEFAULT_OUTPUT_TOOL_DESCRIPTION, DEFAULT_OUTPUT_TOOL_NAME
from .._run_context import RunContext
from ..exceptions import ModelAPIError, UnexpectedModelBehavior, UserError
from ..messages import (
    BaseToolReturnPart,
    CachePoint,
    CompactionPart,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
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
    'ToolCallProposed',
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
    legend: dict[int, JsonValue]
    """Descriptions of the score levels."""
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

    decision_tool_call_threshold: float
    """How likely a tool call has to be before it is taken, from 0 to 1. Default: 0.6.

    With tools attached, one more question asks which tool the text calls for, the output tool among them. A tool
    picked below this probability is a lean, and the output is filled as usual; one at or above it is called, after
    the model fills any supported arguments, or raised as
    [`ToolCallProposed`][pydantic_ai.models.decision.ToolCallProposed] when its arguments are unsupported.
    """


class ToolCallProposed(ModelAPIError):
    """A decision model picked a tool whose arguments it cannot fill.

    A [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a
    [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] with a language model behind the decision model
    hands that model the whole step by default, tools and all, and only the steps the decision model hands off cost
    a language model call.
    """

    tool_name: str
    """The tool the model proposed."""

    probability: float
    """How likely the model found the call, from 0 to 1."""

    def __init__(self, model_name: str, tool_name: str, probability: float):
        self.tool_name = tool_name
        self.probability = probability
        super().__init__(
            model_name,
            f'{model_name} proposed calling {tool_name!r} (probability {probability:.2f}) but cannot fill its '
            'arguments. Put a model that can behind it: `FallbackModel(decision_model, llm)` hands `llm` this step.',
        )

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.model_name, self.tool_name, self.probability)


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
      any other type is a [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent.
    - The field's description is the question, the output type's docstring its goal, and the agent's
      `instructions` framing shared by every question. The latest user prompt is the text to judge, and the
      message history before it goes along beside it.
    - With tools attached, or a union of output types, one more pick-one asks which route the text calls for. A
      picked route with fields is filled in a second request, and one whose fields the model cannot express is
      raised as [`ToolCallProposed`][pydantic_ai.models.decision.ToolCallProposed], for a
      [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] to hand to a language model.
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
        request and filled in a second. Every question in `request.questions` needs an answer of the matching kind
        under the same name.

        Raise [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] or
        [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] when the backend fails, so a
        [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] can take over;
        [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] when it returns something that
        cannot be read; and [`UserError`][pydantic_ai.exceptions.UserError] when the request cannot be sent as
        given. Forward `timeout`, `extra_headers` and `extra_body` from `model_settings` where the backend
        supports them.
        """
        raise NotImplementedError()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(model_settings, model_request_parameters)
        output_tools, hand_offs = _output_tools(model_request_parameters)
        # One output type is filled in the same request that picks a route; several are a union, so the first
        # request only picks, and the chosen type's fields are asked in the second — the same two steps a
        # selected tool's arguments take, through the same helper.
        output_tool = output_tools[0] if len(output_tools) == 1 else None
        # A withheld tool is not on any wire; one revealed through the history is, and the model sees the history.
        function_tools = [
            tool
            for tool in model_request_parameters.function_tools
            if model_request_parameters.visibility_of(tool.name) != 'withheld'
        ]
        offered = [*hand_offs, *function_tools]
        tools = _tools_left(messages, offered)
        forced_tool = tools[0] if not output_tools and len(tools) == 1 and len(offered) > 1 else None
        if forced_tool is not None and (
            _none_route(forced_tool) or not _properties(forced_tool.parameters_json_schema)
        ):
            # Preserve the no-request path: there is no state or question to build when no arguments need filling.
            return self._forced(forced_tool)
        state = _map_messages(messages)
        instruction_parts = self._get_instruction_parts(messages, model_request_parameters) or []
        instructions = '\n\n'.join(part.content for part in instruction_parts) or None
        settings = cast(DecisionModelSettings, model_settings or {})
        # Both bars are read before anything is sent: a setting outside 0 to 1 is a coding error, and finding
        # out from a rejected answer would mean paying for the request that carried the prompt and history.
        threshold = _threshold(settings, 'decision_tool_call_threshold', 0.6)
        boolean_threshold = _threshold(settings, 'decision_boolean_threshold', 0.5)
        limits = _Limits(choice_options=self.max_choice_options, score_levels=self.max_score_levels)
        if forced_tool is not None:
            # Every other route has returned this turn, so the one left is taken without a choice question.
            return await self._forced_with_arguments(
                forced_tool, state, instructions, settings, boolean_threshold, limits
            )
        if len(output_tools) > 1 and not any(_expressible(tool, instructions, limits) for tool in output_tools):
            # A member the model cannot fill is a hand-off, but only while some other member is a real alternative.
            # With none of them fillable the choice is decided before it is asked: every answer hands off, so the
            # request that asks it buys nothing, and every run pays for the decision model on top of the model
            # behind it.
            raise UserError(
                'None of the output types can be filled by this model, so every answer would be handed off and '
                'the request asking which would be wasted. Give the agent an `output_type` it can fill, or drop '
                'it from this model.'
            )
        ask = _Ask.about(output_tool, instructions, limits) if output_tool else _Ask.nothing()
        tool_key = _tool_question(ask.questions, output_tools, tools, instructions, limits)

        response = await self.decide(DecisionRequest(state=state, questions=ask.questions), settings)
        response_usage = response.usage
        args, provider_details = ask.answers(response, boolean_threshold)
        parts: list[ModelResponsePart] = []
        if output_tool:
            parts.append(ToolCallPart(output_tool.name, args, _utils.generate_tool_call_id()))
        if tool_key is not None:
            picked = _tool_call(
                response.answers.get(tool_key),
                output_tools,
                tools,
                {tool.name for tool in hand_offs},
                threshold,
                provider_details,
            )
            if isinstance(picked, ToolCallPart):
                parts = [picked]
            elif picked is not None and picked is not output_tool:
                # `_tool_call` tolerates an offered route missing from `probabilities` when it falls back to
                # the likeliest one, so the route it returns is not necessarily one the model priced.
                probability = provider_details['tool']['probabilities'].get(picked.name, 0.0)
                response, args, argument_details = await self._fill(
                    picked, probability, state, instructions, settings, boolean_threshold, limits
                )
                response_usage += response.usage
                provider_details.update(argument_details)
                # `RequestUsage.requests` is fixed at 1, so usage cannot say that this turn asked twice: the
                # choice and the fill are two requests inside one step. The count is reported here, and only
                # here, so it appears exactly when it differs from what usage reports. See #8498.
                provider_details['requests'] = 2
                parts = [ToolCallPart(picked.name, args, _utils.generate_tool_call_id())]

        return ModelResponse(
            parts=parts,
            usage=response_usage,
            model_name=response.model_name,
            provider_name=self.system,
            provider_url=self.base_url,
            provider_details=provider_details,
            finish_reason='tool_call',
        )

    async def _fill(
        self,
        tool: ToolDefinition,
        probability: float,
        state: JsonValue,
        instructions: str | None,
        settings: DecisionModelSettings,
        boolean_threshold: float,
        limits: _Limits,
    ) -> tuple[DecisionResponse, dict[str, Any], dict[str, Any]]:
        """Ask a selected route's fields in a second request, or hand it off when the model cannot express them.

        One helper for both routes the model picks and then fills: a tool's arguments, and a union member's fields.
        They are the same two steps, and a route whose fields the model cannot express is the same hand-off either
        way — [`ToolCallProposed`][pydantic_ai.models.decision.ToolCallProposed] is a `ModelAPIError`, so a
        [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] gives a language model the whole step.

        This is why a union may hold a member the model cannot express while a lone `output_type` may not: with one
        output type there is no other route the run could have taken, so an unfillable one can only ever fail,
        and it is refused before any request. Offered beside others, it is a route like any other.
        """
        try:
            ask = _Ask.about(tool, instructions, limits, picked=True)
        except UserError:
            raise ToolCallProposed(self.model_name, tool.name, probability) from None

        try:
            response = await self.decide(DecisionRequest(state=state, questions=ask.questions), settings)
            args, provider_details = ask.answers(response, boolean_threshold)
        except (ModelAPIError, UnexpectedModelBehavior) as e:
            # The first request committed to this route. Letting a fallback model rerun the whole original step
            # could silently choose another route, so a failure while filling is terminal and names that route.
            raise UnexpectedModelBehavior(
                f'{self.model_name} selected {tool.name!r}, but failed while filling its fields: {e}'
            ) from e
        return response, args, provider_details

    async def _forced_with_arguments(
        self,
        tool: ToolDefinition,
        state: JsonValue,
        instructions: str | None,
        settings: DecisionModelSettings,
        boolean_threshold: float,
        limits: _Limits,
    ) -> ModelResponse:
        """Fill the arguments of the one route left, without a choice request."""
        details = {'tool': {'choice': tool.name, 'probabilities': {tool.name: 1.0}, 'offered': [tool.name]}}
        response, args, argument_details = await self._fill(
            tool, 1.0, state, instructions, settings, boolean_threshold, limits
        )
        details.update(argument_details)
        return ModelResponse(
            parts=[ToolCallPart(tool.name, args, _utils.generate_tool_call_id())],
            usage=response.usage,
            model_name=response.model_name,
            provider_name=self.system,
            provider_url=self.base_url,
            provider_details=details,
            finish_reason='tool_call',
        )

    def _forced(self, tool: ToolDefinition) -> ModelResponse:
        """Call the one argumentless route left, without asking the model."""
        details = {'tool': {'choice': tool.name, 'probabilities': {tool.name: 1.0}, 'offered': [tool.name]}}
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
        return True, (probability - threshold) / (1 - threshold) if threshold < 1 else 0.0
    return False, (threshold - probability) / threshold


def _answers(
    answers: Mapping[str, object],
    properties: dict[str, dict[str, Any]],
    questions: dict[str, DecisionQuestion],
    boolean_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The output's arguments and `provider_details` from the model's answers to the field questions."""
    args: dict[str, Any] = {}
    confidence: dict[str, float] = {}
    probabilities: dict[str, dict[str, float]] = {}
    scores: dict[str, float] = {}
    for name, prop in properties.items():
        prop, none_option = _optional(prop)
        if keys := _mapping_options(prop):
            # A mapping keeps every option with the answer it got, unlike a list.
            verdicts, labelled = _fanned_in(name, keys, answers, boolean_threshold)
            _set(args, name, {key: chosen for key, (chosen, _) in verdicts.items()})
            confidence[name] = min(sureness for _, sureness in verdicts.values())
            probabilities[name] = labelled
            continue
        if prop.get('type') == 'array':
            # A list is the options that came back yes, in their order.
            verdicts, labelled = _fanned_in(name, _options(prop['items']) or {}, answers, boolean_threshold)
            _set(args, name, [option for option, (chosen, _) in verdicts.items() if chosen])
            confidence[name] = min(sureness for _, sureness in verdicts.values())
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
                confidence[name] = sureness
        elif isinstance(questions[name], ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
            if answer.choice not in none_option:
                # The option itself is written back, looked up by its label rather than parsed out of it: `1`
                # and `'1'` are two options, and only the lookup knows which one the label stood for.
                labelled = _labelled(_options(prop) or {})
                _set(args, name, labelled.get(answer.choice, answer.choice))
            elif 'default' in properties[name]:
                # "None of these" is the absence of an answer, and a default says what to use when there is
                # none, so the field is left out for Pydantic to fill in. The model it belongs to is still put
                # in place, to apply the default in.
                _slot(args, name)
            else:
                _set(args, name, None)
            confidence[name] = answer.confidence
            probabilities[name] = answer.probabilities
        elif isinstance(questions[name], ScoreQuestion) and isinstance(answer, ScoreAnswer):
            # `score` is a position along the rubric and falls between levels. The answer has to be one of them,
            # so it is rounded to the nearest; the likeliest level would throw away the ordering that makes a
            # rubric a rubric. A half goes up, unlike `round`.
            _set(args, name, min(int(answer.score + 0.5), max(answer.probabilities)))
            confidence[name] = answer.confidence
            probabilities[name] = {str(level): p for level, p in answer.probabilities.items()}
            scores[name] = answer.score
        else:
            raise UnexpectedModelBehavior(f'Unexpected answer from the model for output field {name!r}: {answer!r}')
    return args, {'confidence': confidence, 'probabilities': probabilities, 'scores': scores}


def _tool_call(
    answer: object,
    output_tools: list[ToolDefinition],
    tools: list[ToolDefinition],
    hand_offs: set[str],
    threshold: float,
    provider_details: dict[str, Any],
) -> ToolDefinition | ToolCallPart | None:
    """The output or tool call to take from the model's answer to the route question.

    A tool picked below the threshold is a lean: the likeliest output type is filled, or with none to fill, the
    likeliest output function is taken instead. A selected route with fields is returned for a second request.

    The threshold gates tools, not output types. Picking an output type says which result to fill, not that
    something else should be done; there is nothing to hand off to and nothing to be unsure about beyond the pick
    itself, whose confidence is reported either way.
    """
    if (
        not isinstance(answer, ChoiceAnswer)
        or answer.choice not in answer.probabilities
        or not all(0 <= p <= 1 for p in answer.probabilities.values())
    ):
        raise UnexpectedModelBehavior(f'Unexpected answer from the model for the route question: {answer!r}')
    probability = answer.probabilities[answer.choice]
    # The pick, its probabilities and what was on offer are reported either way, so the hand-off rate can be
    # watched, and a tool that was withheld this turn can be seen to have been.
    provider_details['tool'] = {
        'choice': answer.choice,
        'probabilities': answer.probabilities,
        'offered': [tool.name for tool in tools],
    }
    picked_output = next((tool for tool in output_tools if tool.name == answer.choice), None)
    if picked_output is not None:
        return picked_output
    tool = next((tool for tool in tools if tool.name == answer.choice), None)
    if tool is None:
        raise UnexpectedModelBehavior(f'The model picked a route it was not offered: {answer.choice!r}')
    if tool.name not in hand_offs and probability < threshold:
        # A `None` route is a result to take, not something else to be done, so it is weighed here with the
        # output types rather than below with the hand-offs, even though it is offered as one of those.
        results = [*output_tools, *(candidate for candidate in tools if _none_route(candidate))]
        if likeliest_output := max(
            results, key=lambda candidate: answer.probabilities.get(candidate.name, 0.0), default=None
        ):
            if _none_route(likeliest_output):
                # Nothing to fill, so it is called on the fallback itself rather than asked about again.
                return ToolCallPart(
                    likeliest_output.name, _route_args(likeliest_output), _utils.generate_tool_call_id()
                )
            return likeliest_output
        likeliest = max(
            (candidate for candidate in tools if candidate.name in hand_offs),
            key=lambda candidate: answer.probabilities.get(candidate.name, 0.0),
            default=None,
        )
        if likeliest is not None:
            provider_details['tool']['taken'] = likeliest.name
            tool = likeliest
    if not _none_route(tool) and tool.parameters_json_schema.get('properties'):
        return tool
    # Nothing to write, so the call is made on the pick alone.
    return ToolCallPart(tool.name, _route_args(tool), _utils.generate_tool_call_id())


def _expressible(tool: ToolDefinition, instructions: str | None, limits: _Limits) -> bool:
    """Whether the model could fill this route's fields, decided without sending anything."""
    try:
        _Ask.about(tool, instructions, limits)
    except UserError:
        return False
    return True


_NONE_OF_THESE = 'None of these.'


def _null(schema: dict[str, Any]) -> bool:
    """Whether a schema is `None`, whatever else the user wrote about it.

    Pydantic renders a bare `None` as `{'type': 'null'}`, but `Annotated[None, Field(description=...)]` — how a
    user says what returning nothing means — renders the description beside it. The type is what makes it `None`;
    the rest is the user's own words, and a route or a field is no less `None` for having some.
    """
    return schema.get('type') == 'null'


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
    return wrapped.get('description') if wrapped else None


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


def _properties(schema: dict[str, Any]) -> dict[str, Any]:
    """A schema's properties, through the top-level `$ref` Pydantic renders a model that refers to itself as."""
    if ref := schema.get('$ref'):
        schema = schema['$defs'][ref.removeprefix('#/$defs/')]
    return schema.get('properties', {})


def _fields(output_tool: ToolDefinition) -> dict[str, dict[str, Any]]:
    """The output schema's fields, flattened, with `$ref`s to `$defs` (how Pydantic renders an `Enum` or a model) resolved.

    A nested model is its fields, named `outer.inner`: each question is about one value, and a field of a field is
    still one value. The answers are nested back into place by `_set`.
    """
    schema = output_tool.parameters_json_schema
    defs: dict[str, Any] = schema.get('$defs', {})

    def resolve(prop: dict[str, Any]) -> dict[str, Any]:
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

    def flatten(properties: dict[str, Any], prefix: str, seen: frozenset[str]) -> dict[str, dict[str, Any]]:
        fields: dict[str, dict[str, Any]] = {}
        for name, prop in properties.items():
            if '.' in name:
                raise UserError(
                    f'Output field {prefix + name!r} is not supported by this model: a dot in a field name is how '
                    'a nested field is named. Rename it.'
                )
            ref = prop.get('$ref')
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
                fields.update(flatten(prop['properties'], f'{prefix}{name}.', seen | {ref} if ref else seen))
            else:
                fields[f'{prefix}{name}'] = prop
        return fields

    return flatten(_properties(schema), '', frozenset())


def _set(args: dict[str, Any], name: str, value: Any) -> None:
    """Put a flattened field's answer back where it belongs, `outer.inner` under `outer`."""
    parent, leaf = _slot(args, name)
    parent[leaf] = value


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


def _ask(
    name: str, prop: dict[str, Any], output_tool: ToolDefinition, instructions: str | None, *, picked: bool = False
) -> dict[str, JsonValue]:
    """What a field asks, as labelled parts: the field, its question, the route it belongs to, and shared framing."""
    # Only what the user wrote goes to the model. A bare `bool` output is wrapped in a field named `response`
    # by Pydantic AI, and the output tool has a stock description; neither says anything about the question.
    ask: dict[str, JsonValue] = {}
    # A field's name says what is being asked about, which is not the same as asking something, so it
    # goes under `field` and leaves `question` for a question. The wrapper field Pydantic AI puts around
    # a bare output is named `response` and says nothing about anything, so it is not sent at all.
    if name != output_tool.outer_typed_dict_key:
        ask['field'] = name
    if description := prop.get('description'):
        ask['question'] = description
    if picked and (route := _route_name(output_tool)):
        # A fill is a second request about the same text, so nothing in it says a route was already picked.
        # Its name is what the choice question offered and what the answer named, and a field of that route
        # reads differently once you know which one you are filling.
        ask['chosen'] = route
    if described := _described(output_tool):
        ask['goal'] = described
    if instructions:
        # With no field to describe, a bare output's whole question is what the agent was instructed to
        # ask, so it goes where a question goes. Alongside fields of its own it is shared framing.
        ask['question' if 'question' not in ask and 'field' not in ask else 'instructions'] = instructions

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

    @classmethod
    def about(cls, tool: ToolDefinition, instructions: str | None, limits: _Limits, *, picked: bool = False) -> _Ask:
        """The questions this route's fields become, or a `UserError` if the model cannot express one of them.

        `picked` is for the second request of a turn that chose a route first: those questions name the
        route they belong to, which the first request's questions have no reason to.
        """
        properties = _fields(tool)
        return cls(properties, _questions(properties, tool, instructions, limits, picked=picked))

    @classmethod
    def nothing(cls) -> _Ask:
        """No fields to fill: a turn that only picks a route still reports the same empty details."""
        return cls({}, {})

    def answers(self, response: DecisionResponse, boolean_threshold: float) -> tuple[dict[str, Any], dict[str, Any]]:
        return _answers(response.answers, self.properties, self.questions, boolean_threshold)


def _questions(
    properties: dict[str, dict[str, Any]],
    output_tool: ToolDefinition,
    instructions: str | None,
    limits: _Limits,
    *,
    picked: bool = False,
) -> dict[str, DecisionQuestion]:
    """One question per output field, or one per option for a field that fans out."""
    questions: dict[str, DecisionQuestion] = {}
    for name, prop in properties.items():
        ask = _ask(name, prop, output_tool, instructions, picked=picked)
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


def _route_name(tool: ToolDefinition) -> str | None:
    """What to call a picked route when telling the model which one it picked.

    Only what the user wrote means anything: an output route is named for the library's own tool, either
    `final_result` on its own or `final_result_<Member>` for a union member, and the bare prefix says no
    more than "the answer" does. A function tool's name is the user's.
    """
    if tool.kind != 'output':
        return tool.name
    return tool.name.removeprefix(DEFAULT_OUTPUT_TOOL_NAME).lstrip('_') or None


def _described(tool: ToolDefinition) -> str | None:
    """What the user wrote about a tool, without the stock description generated for an output tool."""
    description = tool.description
    if not description or (tool.kind == 'output' and description.endswith(DEFAULT_OUTPUT_TOOL_DESCRIPTION)):
        return None
    return description


def _tools_left(messages: list[ModelMessage], tools: list[ToolDefinition]) -> list[ToolDefinition]:
    """The tools still on offer: one whose result is already in the turn is not offered again.

    A decision model judges the text in front of it and has no notion of having made a call. With a call and its
    result in view, the text still calls for the tool, so left on offer it is picked again until the usage limit.
    That goes for a call made by a model behind this one too, since this model would propose it again on the same
    text. A call that produced no result, because the tool asked for a retry, leaves the tool on offer. The turn is
    everything since the last user prompt, which is the nearest thing to a run boundary the history has: a result
    from an earlier turn does not withhold the tool, but a judged history that ends in another agent's call to a
    tool of the same name does.
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
    return [tool for tool in tools if tool.name not in returned]


def _tool_question(
    questions: dict[str, DecisionQuestion],
    output_tools: list[ToolDefinition],
    tools: list[ToolDefinition],
    instructions: str | None,
    limits: _Limits,
) -> str | None:
    """With tools attached, one more question: which route the text calls for, the output types among them.

    The model first picks the route, then fills a selected tool's arguments in a separate request when their schema
    maps to questions; an unsupported argument leaves the call to a model behind it. The output types are the first
    options, described by what the agent is for, so that filling the output is an action weighed against the
    others. Asking instead whether the model *can* answer would be a question about the question rather than about
    the text, and invites a hand-off on everything. Only what the user wrote describes an output: the output type's
    docstring, or failing that the agent's instructions; the stock output tool description says nothing a tool
    could be weighed against.
    """
    if not output_tools and len(tools) < 2:
        raise UserError(
            'An `output_type` with no fields is not supported by this model; there is nothing to ask the model. '
            'Give it fields, or more than one tool to pick between.'
        )
    if not tools and len(output_tools) < 2:
        return None
    key = 'tool'
    while key in questions:
        key += '_'
    criteria: dict[str, JsonValue] = {}
    for output_tool in output_tools:
        described = _purpose(output_tool)
        if not (described or (instructions and len(output_tools) == 1)):
            # With one output type the agent's instructions can say what filling it is for. With several, only
            # each type's own docstring can tell them apart: one instruction cannot describe two different routes.
            raise UserError(
                'A decision model weighs each route by what it is for, and '
                f'{output_tool.name!r} says nothing about itself. Give the output type a docstring that says what '
                'filling it does' + ('.' if len(output_tools) > 1 else ', or the agent `instructions`.')
            )
        criteria[output_tool.name] = described or instructions
    criteria.update((tool.name, _route_description(tool)) for tool in tools)
    if limits.choice_options is not None and len(criteria) > limits.choice_options:
        raise UserError(
            f'This model picks from at most {limits.choice_options} options, and it is being offered '
            f'{len(criteria)} routes: each output type counts as one beside the tools. Attach fewer tools, or '
            'withhold some of them until they are needed.'
        )
    questions[key] = ChoiceQuestion(instructions='Which of these does this call for?', criteria=criteria)
    return key


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


def _map_request(message: ModelRequest, *, latest: bool) -> tuple[list[JsonValue], list[str]]:
    """Map a request to history entries and the text to judge."""
    history: list[JsonValue] = []
    prompt_parts: list[str] = []
    for part in message.parts:
        if isinstance(part, SystemPromptPart):
            # Whoever wrote it, a system prompt is something that was said in the conversation, so it is
            # material to judge and not a question to ask. What the model is asked comes from `instructions`.
            history.append({'system': part.content})
        elif isinstance(part, UserPromptPart):
            text = _prompt_text(part)
            if latest:
                prompt_parts.append(text)
            else:
                history.append({'user': text})
        elif isinstance(part, ToolReturnPart):
            history.append(_tool_return_entry(part))
        elif isinstance(part, RetryPromptPart):
            history.append({'retry': part.model_response()})
        elif isinstance(part, ToolAvailabilityDeltaPart):  # pragma: no cover
            raise _unsynthesized_tool_availability_delta_error()
        elif isinstance(part, SpeechPart):  # pragma: no cover
            # `Model.prepare_messages` turns realtime speech into `UserPromptPart`s before this runs.
            raise _unconverted_speech_part_error()
        else:
            assert_never(part)
    return history, prompt_parts


def _response_entries(message: ModelResponse) -> list[JsonValue]:
    """Map a response to history entries, excluding the model's private thinking."""
    entries: list[JsonValue] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            entries.append({'assistant': part.content})
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
        elif isinstance(part, ThinkingPart):
            pass  # The model's own reasoning, not part of the conversation.
        else:
            assert_never(part)
    return entries


def _map_messages(messages: list[ModelMessage]) -> JsonValue:
    """The state to judge.

    The latest user text on its own is the whole state, sent as the plain text it is. With a conversation behind
    it there are two parts to keep apart, so they get named: the text under judgement and the `history` before it.
    """
    history: list[JsonValue] = []
    prompt_parts: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            entries, latest_prompt_parts = _map_request(message, latest=message is messages[-1])
            history.extend(entries)
            prompt_parts.extend(latest_prompt_parts)
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
