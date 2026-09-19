from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, cast

from typing_extensions import assert_never

from .. import _utils, usage
from .._http import to_httpx2_timeout
from .._output import DEFAULT_OUTPUT_TOOL_DESCRIPTION
from .._run_context import RunContext
from ..exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
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
from ..profiles import ModelProfileSpec
from ..providers import Provider, infer_provider
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    _unconverted_speech_part_error,  # pyright: ignore[reportPrivateUsage]
    _unsynthesized_tool_availability_delta_error,  # pyright: ignore[reportPrivateUsage]
    check_allow_model_requests,
)

try:
    from typesafe_sdk import (
        AsyncTypeSafeClient,
        Choice,
        ChoiceAnswer,
        JSONContent,
        Noul,
        NoulAnswer,
        Score,
        ScoreAnswer,
        SystemOneResponse,
        TypeSafeAPIConnectionError,
        TypeSafeAPIError,
        TypeSafeAPIResponseValidationError,
        TypeSafeError,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install the `typesafe-sdk` package to use the TypeSafe model, '
        'you can use the `typesafe` optional group — `pip install "pydantic-ai-slim[typesafe]"`'
    ) from _import_error

__all__ = (
    'TypeSafeModel',
    'TypeSafeModelName',
    'TypeSafeModelSettings',
    'TypeSafeStreamedResponse',
    'LatestTypeSafeModelNames',
    'ToolCallProposed',
)

LatestTypeSafeModelNames = Literal['jev-latest', 'jev-preview']
"""TypeSafe aliases, which move when a release ships. `jev-preview` runs ahead of `jev-latest` when there is a
preview build. A versioned id such as `jev-1.13.0` is accepted too, and is what to use once a confidence
threshold has been tuned against one. https://docs.typesafe.ai/models"""

TypeSafeModelName = str | LatestTypeSafeModelNames
"""Possible TypeSafe model names."""

# Jev picks from at most this many options in one question; a 256th is a 400 from the API.
# https://docs.typesafe.ai/model-jaggedness/jev-1.13
_MAX_CHOICE_OPTIONS = 255

_UNSUPPORTED_FIELD_HINT = (
    'Use `bool`, a `Literal` or `Enum` of two or more strings, a `float` bounded with `ge=0` and `le=1`, a `list` of '
    'a `Literal` or `Enum`, a rubric of whole numbers from 0 with a description per level in its schema, or a model '
    'of these.'
)


class TypeSafeModelSettings(ModelSettings, total=False):
    """Settings used for a TypeSafe model request."""

    # ALL FIELDS MUST BE `typesafe_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    typesafe_boolean_threshold: float
    """How likely a yes has to be before a `bool` field is `True`, from 0 to 1. Default: 0.5.

    Jev answers a yes/no with the probability of yes, and the default rounds it: what the framework cannot know is
    what `True` has to mean for you. Raise it where a false positive is the expensive mistake and a `True` should
    be earned, lower it where a false negative is. It applies to every `bool` field and to each option of a `list`
    of a `Literal` or `Enum`, which is one yes/no per option; a `float` bounded with `ge=0` and `le=1` returns the
    probability itself and is not thresholded.

    Reported confidence is the distance from the threshold rather than from the probability, scaled to run from 0
    at the threshold to 1 at certainty, so a yes at 0.8 under a threshold of 0.75 reports the narrow margin it is.
    """

    typesafe_tool_call_threshold: float
    """How likely Jev has to find a tool call before it is proposed, from 0 to 1. Default: 0.6.

    With tools attached, one more question asks which tool the text calls for, the output tool among them. A tool
    picked below this probability is a lean, and the output is filled as usual; one at or above it is called, after
    Jev fills any supported arguments, or raised as
    [`ToolCallProposed`][pydantic_ai.models.typesafe.ToolCallProposed] when its arguments are unsupported. At 0.6,
    on labelled support tickets, Jev's picks agree with a frontier model as often as two frontier models agree with
    each other; higher takes fewer tools, and is right more often when it does. Tune it on labelled examples of your
    own.
    """


class ToolCallProposed(ModelAPIError):
    """Jev found that the text calls for a tool whose arguments it cannot fill.

    A [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a
    [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] with a language model behind Jev hands it the
    whole step by default, tools and all, and only the requests Jev hands off cost a language model call.
    """

    tool_name: str
    """The tool Jev proposed."""

    probability: float
    """How likely Jev found the call, from 0 to 1."""

    def __init__(self, model_name: str, tool_name: str, probability: float):
        self.tool_name = tool_name
        self.probability = probability
        super().__init__(
            model_name,
            f'Jev proposed calling {tool_name!r} (probability {probability:.2f}) and cannot call tools itself. '
            f'Put a model that can behind it: `FallbackModel(jev, llm)` hands it this request.',
        )

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.model_name, self.tool_name, self.probability)


@dataclass(init=False)
class TypeSafeModel(Model[AsyncTypeSafeClient]):
    """A model that fills a structured `output_type`, and supported tool arguments after choosing a tool.

    Jev does not generate text. It answers typed questions about a text, each with a confidence. This model
    turns the output type's fields into those questions and the user prompt into the text, so an agent whose
    job is to classify runs on it like on any other model:

    ```python
    from typing import Literal

    from pydantic import BaseModel, Field

    from pydantic_ai import Agent


    class Handling(BaseModel):
        verdict: Literal['run', 'reject', 'ask'] = Field(description='How to handle this command.')
        irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


    agent = Agent('typesafe:jev-latest', output_type=Handling)
    ...
    ```

    Each field is one question, all sent in one request:

    | Field type | Question | Answer |
    |---|---|---|
    | `bool` | yes or no | `True` when Jev's probability is at least 0.5 |
    | `Literal[...]` or `Enum` of strings | pick one | the chosen option |
    | `float` with `ge=0` and `le=1` | yes or no | Jev's probability |
    | whole numbers 0, 1, 2, … with a description per level in the schema | score against a rubric | the nearest level |
    | `list` of a `Literal` or `Enum` | one yes or no per option | the options Jev said yes to |
    | a nested model of these | its fields, named `outer.inner` | the model |
    | `Literal[...]` or `Enum`, or `None` | pick one, or none of these | the option, or `None` |

    The field description is the question. The output type's docstring and the agent's instructions go along
    as context. An option is described by a description on its value in the schema, and by its name without one.
    A bare `bool`, `Literal` or `float` output has no field to describe, so there the agent's instructions are the
    question.
    Confidence per field, from 0 for undecided to 1, is in
    [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details] under `confidence`,
    the full distribution of each pick-one and rubric field under `probabilities`, and each rubric field's
    unrounded position along its levels under `scores`.

    The latest user prompt is the text Jev judges, and is the whole state on its own. Everything before it in
    the message history, from any model, goes along beside it as `history`: user prompts, answers, tool calls
    and their results, and retry prompts.

    An `output_type` of several structured types is a union, and a route rather than a field: one question picks
    which type the text calls for, described by each type's own docstring, and a second request asks only that
    type's fields. A member whose fields Jev cannot express is still offered, and picking it raises
    [`ToolCallProposed`][pydantic_ai.models.typesafe.ToolCallProposed]; a lone `output_type` it cannot express is
    refused before any request instead, since no other route could have been taken. A model *field* typed as a
    union of structured types is not supported.

    With tools attached, one more question asks which, the output types first among the options, described by their
    docstrings or the agent's instructions. A tool that takes no arguments, or an output function that takes nothing
    but the run context, is called on Jev's pick. When a picked tool's arguments use the field types above, a second
    request asks only those arguments and Jev returns the filled call. If any argument is unsupported, the call is
    raised as [`ToolCallProposed`][pydantic_ai.models.typesafe.ToolCallProposed], which a
    [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] with a language model behind Jev hands that model,
    tools and all.

    Jev answers in one piece, so a streamed run gets the whole answer as one event rather than failing.

    Anything else Jev cannot do is refused with a [`UserError`][pydantic_ai.exceptions.UserError] before a
    request is sent: text output, other field types, native tools, and files in the prompt or history.

    Sampling settings like `temperature` do not apply and are ignored. `timeout`, `extra_headers` and
    `extra_body` are forwarded.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    _model_name: TypeSafeModelName = field(repr=False)
    _provider: Provider[AsyncTypeSafeClient] = field(repr=False)

    def __init__(
        self,
        model_name: TypeSafeModelName,
        *,
        provider: Literal['typesafe'] | Provider[AsyncTypeSafeClient] = 'typesafe',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a TypeSafe model.

        Args:
            model_name: The name of the TypeSafe model to use, such as `jev-latest`.
            provider: The provider to use for authentication and API access. Can be either the string
                'typesafe' or an instance of `Provider[AsyncTypeSafeClient]`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider

        super().__init__(settings=settings, profile=profile)

    @property
    def client(self) -> AsyncTypeSafeClient:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> TypeSafeModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._provider.name

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
        # A withheld tool is not on any wire; one revealed through the history is, and Jev sees the whole history.
        function_tools = [
            tool
            for tool in model_request_parameters.function_tools
            if model_request_parameters.visibility_of(tool.name) != 'withheld'
        ]
        offered = [*hand_offs, *function_tools]
        tools = _tools_left(messages, offered)
        forced_tool = tools[0] if not output_tools and len(tools) == 1 and len(offered) > 1 else None
        if forced_tool is not None and not forced_tool.parameters_json_schema.get('properties'):
            # Preserve the no-request path: there is no state or question to build when no arguments need filling.
            return self._forced(forced_tool)
        properties = _fields(output_tool) if output_tool else {}
        state = _map_messages(messages)
        instruction_parts = self._get_instruction_parts(messages, model_request_parameters) or []
        instructions = '\n\n'.join(part.content for part in instruction_parts) or None
        settings = cast(TypeSafeModelSettings, model_settings or {})
        # Both bars are read before anything is sent: a setting outside 0 to 1 is a coding error, and finding
        # out from a rejected answer would mean paying for the request that carried the prompt and history.
        threshold = _threshold(settings, 'typesafe_tool_call_threshold', 0.6)
        boolean_threshold = _threshold(settings, 'typesafe_boolean_threshold', 0.5)
        if forced_tool is not None:
            # Every other route has returned this turn, so the one left is taken without a choice question.
            return await self._forced_with_arguments(forced_tool, state, instructions, settings, boolean_threshold)
        if len(output_tools) > 1 and not any(_expressible(tool, instructions) for tool in output_tools):
            # A member Jev cannot fill is a hand-off, but only while some other member is a real alternative.
            # With none of them fillable the choice is decided before it is asked: every answer hands off, so
            # the request that asks it buys nothing and every run pays for Jev on top of the model behind it.
            raise UserError(
                'None of the output types can be filled by this model, so every answer would be handed off and '
                'the request asking which would be wasted. Give the agent an `output_type` it can fill, or drop '
                'it from this model.'
            )
        questions = _questions(properties, output_tool, instructions) if output_tool else {}
        tool_key = _tool_question(questions, output_tools, tools, instructions)

        response = await self._system_one(state, questions, settings)
        response_usage = _request_usage(response)
        args, provider_details = _answers(response.answers, properties, questions, boolean_threshold)
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
                # the likeliest one, so the route it returns is not necessarily one Jev priced.
                probability = provider_details['tool']['probabilities'].get(picked.name, 0.0)
                response, args, argument_details = await self._fill(
                    picked, probability, state, instructions, settings, boolean_threshold
                )
                response_usage += _request_usage(response)
                provider_details.update(argument_details)
                # `RequestUsage.requests` is fixed at 1, so usage cannot say that this turn asked twice: the
                # choice and the fill are two requests inside one step. The count is reported here, and only
                # here, so it appears exactly when it differs from what usage reports. See #8498.
                provider_details['requests'] = 2
                parts = [ToolCallPart(picked.name, args, _utils.generate_tool_call_id())]

        return ModelResponse(
            parts=parts,
            usage=response_usage,
            model_name=response.model,
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            provider_details=provider_details,
            finish_reason='tool_call',
        )

    async def _system_one(
        self,
        state: JSONContent,
        questions: dict[str, Noul | Choice | Score],
        settings: TypeSafeModelSettings,
    ) -> SystemOneResponse:
        """Send one Jev request, translating SDK failures into model errors."""
        timeout = settings.get('timeout')
        try:
            return await self.client.system_one(
                state,
                questions,
                model=self._model_name,
                timeout=None if timeout is None else to_httpx2_timeout(timeout),
                extra_headers=settings.get('extra_headers'),
                extra_body=cast('Mapping[str, JSONContent] | None', settings.get('extra_body')),
            )
        except TypeSafeAPIResponseValidationError as e:
            raise UnexpectedModelBehavior(f'Invalid response from TypeSafe: {e}', str(e.body)) from e
        except TypeSafeAPIError as e:
            raise ModelHTTPError(
                status_code=e.status, model_name=self._model_name, body=e.body, headers=dict(e.headers)
            ) from e
        except TypeSafeAPIConnectionError as e:
            raise ModelAPIError(model_name=self._model_name, message=str(e)) from e
        except TypeSafeError as e:
            # What is left is the SDK refusing to send what it was given, such as an `extra_body` that will
            # not encode as JSON. That is the caller's to fix, not the model's.
            raise UserError(f'TypeSafe could not send this request: {e}') from e

    async def _fill(
        self,
        tool: ToolDefinition,
        probability: float,
        state: JSONContent,
        instructions: str | None,
        settings: TypeSafeModelSettings,
        boolean_threshold: float,
    ) -> tuple[SystemOneResponse, dict[str, Any], dict[str, Any]]:
        """Ask a selected route's fields in a second request, or hand it off when Jev cannot express them.

        One helper for both routes Jev picks and then fills: a tool's arguments, and a union member's fields.
        They are the same two steps, and a route whose fields Jev cannot express is the same hand-off either
        way — [`ToolCallProposed`][pydantic_ai.models.typesafe.ToolCallProposed] is a `ModelAPIError`, so a
        [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] gives a language model the whole step.

        This is why a union may hold a member Jev cannot express while a lone `output_type` may not: with one
        output type there is no other route the run could have taken, so an unfillable one can only ever fail,
        and it is refused before any request. Offered beside others, it is a route like any other.
        """
        try:
            properties = _fields(tool)
            questions = _questions(properties, tool, instructions)
        except UserError:
            raise ToolCallProposed(self._model_name, tool.name, probability) from None

        try:
            response = await self._system_one(state, questions, settings)
            args, provider_details = _answers(response.answers, properties, questions, boolean_threshold)
        except (ModelAPIError, UnexpectedModelBehavior) as e:
            # The first request committed to this route. Letting a fallback model rerun the whole original step
            # could silently choose another route, so a failure while filling is terminal and names that route.
            raise UnexpectedModelBehavior(
                f'TypeSafe selected {tool.name!r}, but failed while filling its fields: {e}'
            ) from e
        return response, args, provider_details

    async def _forced_with_arguments(
        self,
        tool: ToolDefinition,
        state: JSONContent,
        instructions: str | None,
        settings: TypeSafeModelSettings,
        boolean_threshold: float,
    ) -> ModelResponse:
        """Fill the arguments of the one route left, without a choice request."""
        details = {'tool': {'choice': tool.name, 'probabilities': {tool.name: 1.0}, 'offered': [tool.name]}}
        response, args, argument_details = await self._fill(tool, 1.0, state, instructions, settings, boolean_threshold)
        details.update(argument_details)
        return ModelResponse(
            parts=[ToolCallPart(tool.name, args, _utils.generate_tool_call_id())],
            usage=_request_usage(response),
            model_name=response.model,
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            provider_details=details,
            finish_reason='tool_call',
        )

    def _forced(self, tool: ToolDefinition) -> ModelResponse:
        """Call the one argumentless route left, without asking Jev."""
        details = {'tool': {'choice': tool.name, 'probabilities': {tool.name: 1.0}, 'offered': [tool.name]}}
        return ModelResponse(
            parts=[ToolCallPart(tool.name, {}, _utils.generate_tool_call_id())],
            usage=usage.RequestUsage(),
            model_name=self._model_name,
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
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
        # Jev answers in one piece, so the whole answer is the one event; a streamed run keeps working.
        response = await self.request(messages, model_settings, model_request_parameters)
        yield TypeSafeStreamedResponse(model_request_parameters, response)


@dataclass
class TypeSafeStreamedResponse(StreamedResponse):
    """A whole answer from Jev as a stream of one event, so that a streamed run works on a model that cannot stream."""

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


def _threshold(settings: TypeSafeModelSettings, name: str, default: float) -> float:
    """A probability setting, which is only meaningful inside the range Jev answers in."""
    threshold = cast(float, settings.get(name, default))
    if not 0 <= threshold <= 1:
        raise UserError(f'`{name}` must be between 0 and 1; got {threshold!r}.')
    return threshold


def _verdict(probability: float, threshold: float) -> tuple[bool, float]:
    """Whether Jev's probability of yes clears the bar, and how far from the bar it landed.

    Jev reports no confidence for a yes/no: `noul` is the probability of yes, and what is lost in rounding it to
    an answer is how sure that answer is. That is the distance from the bar, scaled to run 0 to 1 on whichever
    side of it the answer fell — like the confidence Jev reports for the other two kinds of question. Under the
    default bar of 0.5 this is the distance from the coin flip, doubled: a no returned at 0.01 reports 0.98.
    """
    if not 0 <= probability <= 1:
        # Both scalings divide by the room left on their side of the bar, which a probability outside the
        # range Jev answers in can make zero. A malformed answer is the model's to report, not a crash.
        raise UnexpectedModelBehavior(f'Unexpected probability from TypeSafe: {probability!r}')
    if probability >= threshold:
        # An answer exactly at the bar is the least sure one there is, including when the bar is certainty.
        return True, (probability - threshold) / (1 - threshold) if threshold < 1 else 0.0
    return False, (threshold - probability) / threshold


def _answers(
    answers: Mapping[str, object],
    properties: dict[str, dict[str, Any]],
    questions: dict[str, Noul | Choice | Score],
    boolean_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The output's arguments and `provider_details` from Jev's answers to the field questions."""
    args: dict[str, Any] = {}
    confidence: dict[str, float] = {}
    probabilities: dict[str, dict[str, float]] = {}
    scores: dict[str, float] = {}
    for name, prop in properties.items():
        prop, none_key = _optional(prop)
        if prop.get('type') == 'array':
            # One yes/no went out per option; the answer is the options that came back yes, in their order.
            labelled: dict[str, float] = {}
            for option in _options(prop['items']) or {}:
                answer = answers.get(f'{name}.{option}')
                if not isinstance(answer, NoulAnswer):
                    raise UnexpectedModelBehavior(
                        f'Unexpected answer from TypeSafe for output field {name!r}, option {option!r}: {answer!r}'
                    )
                labelled[option] = answer.noul
            verdicts = {option: _verdict(p, boolean_threshold) for option, p in labelled.items()}
            _set(args, name, [option for option, (chosen, _) in verdicts.items() if chosen])
            confidence[name] = min(sureness for _, sureness in verdicts.values())
            probabilities[name] = labelled
            continue
        answer = answers.get(name)
        if isinstance(questions[name], Noul) and isinstance(answer, NoulAnswer):
            if prop.get('type') == 'number':
                # The probability is the answer, so there is no separate confidence to report: a field
                # that asks for the number would otherwise get it back twice under two names.
                _set(args, name, answer.noul)
            else:
                chosen, sureness = _verdict(answer.noul, boolean_threshold)
                _set(args, name, chosen)
                confidence[name] = sureness
        elif isinstance(questions[name], Choice) and isinstance(answer, ChoiceAnswer):
            _set(args, name, None if answer.choice == none_key else answer.choice)
            confidence[name] = answer.confidence
            probabilities[name] = answer.probabilities
        elif isinstance(questions[name], Score) and isinstance(answer, ScoreAnswer):
            # `score` is a position along the rubric and falls between levels. The answer has to be one
            # of them, and TypeSafe's way to get one is to "round it to the nearest level"; the mode
            # would throw away the ordering that makes a rubric a rubric. A half goes up, unlike `round`.
            _set(args, name, min(int(answer.score + 0.5), max(answer.probabilities)))
            confidence[name] = answer.confidence
            probabilities[name] = {str(level): p for level, p in answer.probabilities.items()}
            scores[name] = answer.score
        else:
            raise UnexpectedModelBehavior(f'Unexpected answer from TypeSafe for output field {name!r}: {answer!r}')
    return args, {'confidence': confidence, 'probabilities': probabilities, 'scores': scores}


def _request_usage(response: SystemOneResponse) -> usage.RequestUsage:
    """Usage for one Jev request."""
    return usage.RequestUsage(
        input_tokens=response.usage.input_tokens or 0, output_tokens=response.usage.output_tokens or 0
    )


def _tool_call(
    answer: object,
    output_tools: list[ToolDefinition],
    tools: list[ToolDefinition],
    hand_offs: set[str],
    threshold: float,
    provider_details: dict[str, Any],
) -> ToolDefinition | ToolCallPart | None:
    """The output or tool call Jev takes from its answer to the tool question.

    A tool picked below the threshold is a lean: the likeliest output type is filled, or with none to fill, the
    likeliest output function is taken instead. A selected route with fields is returned for a second request.

    The threshold gates tools, not output types. Picking an output type is Jev saying which result to fill, not
    proposing that something else be done; there is nothing to hand off to and nothing to be unsure about beyond
    the pick itself, whose confidence is reported either way.
    """
    if (
        not isinstance(answer, ChoiceAnswer)
        or answer.choice not in answer.probabilities
        or not all(0 <= p <= 1 for p in answer.probabilities.values())
    ):
        raise UnexpectedModelBehavior(f'Unexpected answer from TypeSafe for the tool question: {answer!r}')
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
        raise UnexpectedModelBehavior(f'TypeSafe picked a tool it was not offered: {answer.choice!r}')
    if tool.name not in hand_offs and probability < threshold:
        if likeliest_output := max(
            output_tools, key=lambda candidate: answer.probabilities.get(candidate.name, 0.0), default=None
        ):
            return likeliest_output
        likeliest = max(
            (candidate for candidate in tools if candidate.name in hand_offs),
            key=lambda candidate: answer.probabilities.get(candidate.name, 0.0),
            default=None,
        )
        if likeliest is not None:
            provider_details['tool']['taken'] = likeliest.name
            tool = likeliest
    if tool.parameters_json_schema.get('properties'):
        return tool
    # Nothing to write, so the call is made on Jev's pick.
    return ToolCallPart(tool.name, {}, _utils.generate_tool_call_id())


def _expressible(tool: ToolDefinition, instructions: str | None) -> bool:
    """Whether Jev could fill this route's fields, asked without sending anything."""
    try:
        _questions(_fields(tool), tool, instructions)
    except UserError:
        return False
    return True


def _output_tools(
    model_request_parameters: ModelRequestParameters,
) -> tuple[list[ToolDefinition], list[ToolDefinition]]:
    """The output tools with fields for Jev to fill, and the ones that take no arguments.

    An output function that takes nothing, or only the run context, is a hand-off Jev can pick without writing
    anything. Several output types with fields are a union: Jev picks which one the text calls for, then fills
    that one's fields in a second request, the same two steps a selected tool's arguments take. Text output is
    refused earlier, by the shared request preparation, on the profile's `supports_text_output`.
    """
    with_fields: list[ToolDefinition] = []
    hand_offs: list[ToolDefinition] = []
    for tool in model_request_parameters.output_tools:
        (with_fields if _properties(tool.parameters_json_schema) else hand_offs).append(tool)
    return with_fields, hand_offs


def _properties(schema: dict[str, Any]) -> dict[str, Any]:
    """A schema's properties, through the top-level `$ref` Pydantic renders a model that refers to itself as."""
    if ref := schema.get('$ref'):
        schema = schema['$defs'][ref.removeprefix('#/$defs/')]
    return schema.get('properties', {})


def _fields(output_tool: ToolDefinition) -> dict[str, dict[str, Any]]:
    """The output schema's fields, flattened, with `$ref`s to `$defs` (how Pydantic renders an `Enum` or a model) resolved.

    A nested model is its fields, named `outer.inner`: Jev answers questions, and a field of a field is still one
    question. The answers are nested back into place by `_set`.
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
        return prop

    def flatten(properties: dict[str, Any], prefix: str) -> dict[str, dict[str, Any]]:
        fields: dict[str, dict[str, Any]] = {}
        for name, prop in properties.items():
            if '.' in name:
                raise UserError(
                    f'Output field {prefix + name!r} is not supported by this model: a dot in a field name is how '
                    'a nested field is named. Rename it.'
                )
            prop = resolve(prop)
            if prop.get('type') == 'object' and prop.get('properties'):
                fields.update(flatten(prop['properties'], f'{prefix}{name}.'))
            else:
                fields[f'{prefix}{name}'] = prop
        return fields

    return flatten(_properties(schema), '')


def _set(args: dict[str, Any], name: str, value: Any) -> None:
    """Put a flattened field's answer back where it belongs, `outer.inner` under `outer`."""
    *path, leaf = name.split('.')
    for part in path:
        args = args.setdefault(part, {})
    args[leaf] = value


def _optional(prop: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """An `X | None` field as `X` plus the name of one more option, "none of these"; any other field as it is.

    Measured on labelled tickets, an explicit option is as accurate as an `other` member the user wrote and more
    accurate than reading `None` off low confidence, which is what the field's confidence is for.
    """
    if 'anyOf' not in prop or len(prop['anyOf']) != 2 or {'type': 'null'} not in prop['anyOf']:
        return prop, None
    inner = next(option for option in prop['anyOf'] if option != {'type': 'null'})
    prop = {**inner, **{k: v for k, v in prop.items() if k not in ('anyOf', 'default')}}
    key = 'none'
    while key in (_options(prop) or {}):
        key += '_'
    return prop, key


def _options(prop: dict[str, Any]) -> dict[Any, str | None] | None:
    """The options of a pick-one schema, each with its description, or `None` when the schema is not one."""
    if 'enum' in prop:
        return dict.fromkeys(prop['enum'])
    if 'anyOf' in prop and all('const' in option for option in prop['anyOf']):
        return {option['const']: option.get('description') for option in prop['anyOf']}
    return None


def _ask(
    name: str, prop: dict[str, Any], output_tool: ToolDefinition, instructions: str | None
) -> dict[str, JSONContent]:
    """What a field asks, as the labelled parts TypeSafe's own examples use."""
    # Only what the user wrote goes to Jev. A bare `bool` output is wrapped in a field named `response`
    # by Pydantic AI, and the output tool has a stock description; neither says anything about the question.
    ask: dict[str, JSONContent] = {}
    # A field's name says what is being asked about, which is not the same as asking something, so it
    # goes under `field` and leaves `question` for a question. The wrapper field Pydantic AI puts around
    # a bare output is named `response` and says nothing about anything, so it is not sent at all.
    if name != output_tool.outer_typed_dict_key:
        ask['field'] = name
    if description := prop.get('description'):
        ask['question'] = description
    if described := _described(output_tool):
        ask['goal'] = described
    if instructions:
        # With no field to describe, a bare output's whole question is what the agent was instructed to
        # ask, so it goes where a question goes. Alongside fields of its own it is shared framing.
        ask['question' if 'question' not in ask and 'field' not in ask else 'instructions'] = instructions

    return ask


def _questions(
    properties: dict[str, dict[str, Any]], output_tool: ToolDefinition, instructions: str | None
) -> dict[str, Noul | Choice | Score]:
    """One Jev question per output field."""
    questions: dict[str, Noul | Choice | Score] = {}
    for name, prop in properties.items():
        ask = _ask(name, prop, output_tool, instructions)
        prop, none_key = _optional(prop)
        options = _options(prop)
        if none_key is not None:
            if options is None or not all(isinstance(option, str) for option in options):
                raise UserError(
                    f'Output field {name!r} is not supported by this model: only a `Literal` or `Enum` of strings can '
                    f'be optional, since `None` is one more option to pick. {_UNSUPPORTED_FIELD_HINT}'
                )
            options = {**options, none_key: 'None of these.'}

        # A single value needs no labelling, and TypeSafe's advice is to start with a string; the object
        # form earns its keys only once there is more than one thing in it.
        asked: JSONContent | None = next(iter(ask.values())) if len(ask) == 1 else (ask or None)

        if prop.get('type') == 'array':
            # Several options at once is one yes/no per option, all in the same request, which TypeSafe call
            # fanning out: does this option apply, asked with the field's question and the option's description.
            labels = _options(prop['items'])
            if not labels or len(labels) < 2 or not all(isinstance(label, str) for label in labels):
                raise UserError(
                    f'Output field {name!r} is not supported by this model: a list must be of two or more string '
                    f'options. {_UNSUPPORTED_FIELD_HINT}'
                )
            for label, meaning in labels.items():
                option = f'{label}: {meaning}' if meaning else label
                questions[f'{name}.{label}'] = Noul(instructions={**ask, 'option': option})
        elif options is not None:
            # `bool` is an `int` in Python but never a rubric level, and it is handled as a yes/no below.
            if options and all(isinstance(option, int) and not isinstance(option, bool) for option in options):
                questions[name] = _score_question(name, cast('dict[int, str | None]', options), asked)
            elif len(options) < 2 or not all(isinstance(option, str) for option in options):
                raise UserError(
                    f'Output field {name!r} is not supported by this model: its options are not two or more strings. '
                    f'{_UNSUPPORTED_FIELD_HINT}'
                )
            elif len(options) > _MAX_CHOICE_OPTIONS:
                raise UserError(
                    f'Output field {name!r} is not supported by this model: Jev picks from at most '
                    f'{_MAX_CHOICE_OPTIONS} options, and this one has {len(options)}.'
                )
            else:
                questions[name] = Choice(instructions=asked, criteria=cast('dict[str, str | None]', options))
        elif prop.get('type') == 'boolean' or (
            prop.get('type') == 'number' and prop.get('minimum') == 0 and prop.get('maximum') == 1
        ):
            if not ask:
                # A pick-one or a rubric still says what it is asking through its options; a yes/no has
                # nothing else, and Jev rejects a question with neither instructions nor criteria.
                raise UserError(
                    f'Output field {name!r} asks Jev nothing. A question is not part of the text being judged: '
                    f'give the field a description, or the agent `instructions`, and leave the prompt to the '
                    f'material the question is about. A `system_prompt` will not do: Jev is told what was said, '
                    f'not what to ask.'
                )
            questions[name] = Noul(instructions=asked)
        else:
            raise UserError(f'Output field {name!r} is not supported by this model. {_UNSUPPORTED_FIELD_HINT}')
    return questions


def _described(tool: ToolDefinition) -> str | None:
    """What the user wrote about a tool, without the stock description generated for an output tool."""
    description = tool.description
    if not description or (tool.kind == 'output' and description.endswith(DEFAULT_OUTPUT_TOOL_DESCRIPTION)):
        return None
    return description


def _tools_left(messages: list[ModelMessage], tools: list[ToolDefinition]) -> list[ToolDefinition]:
    """The tools still on offer: one whose result is already in the turn is not offered again.

    Jev has no notion of having made a call. With a call and its result in view, the text still calls for the
    tool, so left on offer it is picked again until the usage limit; that goes for a tool a model behind Jev
    called too, since Jev would propose it again on the same text. A call that produced no result, because the
    tool asked for a retry, leaves the tool on offer. The turn is everything since the last user prompt, which
    is the nearest thing to a run boundary the history has: a result from an earlier turn does not withhold the
    tool, but a judged history that ends in another agent's call to a tool of the same name does.
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
    questions: dict[str, Noul | Choice | Score],
    output_tools: list[ToolDefinition],
    tools: list[ToolDefinition],
    instructions: str | None,
) -> str | None:
    """With tools attached, one more question: which tool the text calls for, the output tool among them.

    Jev first tells which tool the text calls for, then fills a selected tool's arguments in a separate request when
    their schema maps to questions. An unsupported argument leaves the call to a model behind it. The output tool is the first option,
    described by what the agent is for, so that filling the output is an action weighed against the others. Asked
    instead whether it *can* answer, Jev hands off nearly everything: that is a question about the question, not
    about the text. Only what the user wrote describes the output: the output type's docstring, or failing that the
    agent's instructions; the stock output tool description says nothing Jev could weigh a tool against.
    """
    if not output_tools and len(tools) < 2:
        raise UserError(
            'An `output_type` with no fields is not supported by this model; there is nothing to ask Jev. '
            'Give it fields, or more than one tool to pick between.'
        )
    if not tools and len(output_tools) < 2:
        return None
    key = 'tool'
    while key in questions:
        key += '_'
    criteria: dict[str, str | None] = {}
    for output_tool in output_tools:
        described = _described(output_tool)
        if not (described or (instructions and len(output_tools) == 1)):
            # With one output type the agent's instructions can say what filling it is for. With several, only
            # each type's own docstring can tell them apart: one instruction cannot describe two different routes.
            raise UserError(
                'Jev weighs each route by what it is for, and '
                f'{output_tool.name!r} says nothing about itself. Give the output type a docstring that says what '
                'filling it does' + ('.' if len(output_tools) > 1 else ', or the agent `instructions`.')
            )
        criteria[output_tool.name] = described or instructions
    criteria.update((tool.name, tool.description) for tool in tools)
    if len(criteria) > _MAX_CHOICE_OPTIONS:
        raise UserError(
            f'Jev picks from at most {_MAX_CHOICE_OPTIONS} options, and it is being offered {len(criteria)} routes: '
            f'each output type counts as one beside the tools. Attach fewer tools, or withhold some of them until '
            f'they are needed.'
        )
    questions[key] = Choice(instructions='Which of these does this call for?', criteria=criteria)
    return key


def _score_question(name: str, options: dict[int, str | None], asked: JSONContent | None) -> Score:
    """A rubric question from an `IntEnum` or `Literal` of whole numbers, one description per level.

    Jev scores against an ordered rubric that starts at zero, so the levels have to be exactly that, and
    every one of them needs saying what it means: a rubric whose levels are unexplained is not a rubric.
    """
    levels = sorted(options)
    if levels != list(range(len(levels))) or len(levels) < 2:
        raise UserError(
            f'Output field {name!r} is not supported by this model: a rubric must be the whole numbers from 0 '
            f'upwards, in order, and there must be at least two of them. {_UNSUPPORTED_FIELD_HINT}'
        )
    criteria = [options[level] for level in levels]
    if not all(criteria):
        missing = ', '.join(str(level) for level in levels if not options[level])
        raise UserError(
            f'Output field {name!r} is a rubric, so every level needs to say what it means, and {missing} does not. '
            f'Give each level a description in the schema.'
        )
    return Score(instructions=asked, criteria=cast('list[JSONContent]', criteria))


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
                'Files are not supported by this model; images, audio, video and documents cannot be sent to Jev.'
            )
    return '\n\n'.join(texts)


def _tool_return_entry(part: BaseToolReturnPart) -> JSONContent:
    """A tool result as history, or a `UserError` when it carries a file: `model_response_str` would leave it out."""
    if part.files:
        raise UserError('Files are not supported by this model; a file in a tool result cannot be sent to Jev.')
    return {'tool_return': {'name': part.tool_name, 'content': part.model_response_str()}}


def _map_request(message: ModelRequest, *, latest: bool) -> tuple[list[JSONContent], list[str]]:
    """Map a request to history entries and the text to judge."""
    history: list[JSONContent] = []
    prompt_parts: list[str] = []
    for part in message.parts:
        if isinstance(part, SystemPromptPart):
            # Whoever wrote it, a system prompt is something that was said in the conversation, so it is
            # material to judge and not a question to ask. What Jev is asked comes from `instructions`.
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


def _response_entries(message: ModelResponse) -> list[JSONContent]:
    """Map a response to history entries, excluding the model's private thinking."""
    entries: list[JSONContent] = []
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
                'Files are not supported by this model; a file in the message history cannot be sent to Jev.'
            )
        elif isinstance(part, SpeechPart):  # pragma: no cover
            raise _unconverted_speech_part_error()
        elif isinstance(part, ThinkingPart):
            pass  # The model's own reasoning, not part of the conversation.
        else:
            assert_never(part)
    return entries


def _map_messages(messages: list[ModelMessage]) -> JSONContent:
    """The state to judge.

    The latest user text on its own is the whole state, as the text TypeSafe's own examples pass. With a
    conversation behind it there are two parts to keep apart, so they get named: the text under judgement
    and the `history` before it.
    """
    history: list[JSONContent] = []
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
        raise UserError('A request without user text is not supported by this model; Jev needs text to judge.')
    if not history:
        return text
    state: dict[str, JSONContent] = {'history': history}
    if text:
        state['text'] = text
    return state
