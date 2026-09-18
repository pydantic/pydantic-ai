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

_UNSUPPORTED_FIELD_HINT = (
    'Use `bool`, a `Literal` or `Enum` of two or more strings, an `IntEnum` whose members are 0 upwards with a '
    'docstring each, a `float` bounded with `ge=0` and `le=1`, a `list` of a `Literal` or `Enum`, or a model of these.'
)


class TypeSafeModelSettings(ModelSettings, total=False):
    """Settings used for a TypeSafe model request."""

    # ALL FIELDS MUST BE `typesafe_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    typesafe_tool_call_threshold: float
    """How likely Jev has to find a tool call before it is proposed, from 0 to 1. Default: 0.8.

    With tools attached, one more question asks which tool the text calls for, the output tool among them. A tool
    picked below this probability is a lean, and the output is filled as usual; one at or above it is raised as
    [`ToolCallProposed`][pydantic_ai.models.typesafe.ToolCallProposed] for a model behind Jev to call. Tune it on
    labelled examples of your own: higher hands off less, and is right more often when it does.
    """


class ToolCallProposed(ModelAPIError):
    """Jev found that the text calls for a tool, which it cannot call itself.

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
    """A model that fills a structured `output_type` with one request to a TypeSafe Jev model.

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
    | `IntEnum` of 0, 1, 2, … with a docstring each | score against a rubric | the score rounded to a level |
    | `list` of a `Literal` or `Enum` | one yes or no per option | the options Jev said yes to |
    | a nested model of these | its fields, named `outer.inner` | the model |
    | `Literal[...] | None` or `Enum | None` | pick one, or none of these | the option, or `None` |

    The field description is the question. The output type's docstring and the agent's instructions go along
    as context. A docstring under an `Enum` member describes that option, see the [docs](../../models/typesafe.md);
    without one Jev only sees its name. A bare `bool`, `Literal` or `float` output has no field to describe, so
    there the agent's instructions are the question.
    Confidence per field, from 0 for undecided to 1, is in
    [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details] under `confidence`,
    the full distribution of each pick-one and rubric field under `probabilities`, and each rubric field's
    unrounded position along its levels under `scores`.

    The latest user prompt is the text Jev judges, and is the whole state on its own. Everything before it in
    the message history, from any model, goes along beside it as `history`: user prompts, answers, tool calls
    and their results, and retry prompts.

    Jev cannot write a tool's arguments, but it can tell which tool the text calls for. With tools attached, one
    more question asks which, the output type first among the options. A tool that takes no arguments, or an
    output function that takes nothing but the run context, Jev calls itself, so it can run a loop of such tools
    and hand a run off to an output function on its own. A tool with arguments, picked at or above
    `typesafe_tool_call_threshold`, is raised as [`ToolCallProposed`][pydantic_ai.models.typesafe.ToolCallProposed],
    which a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] with a language model behind Jev hands
    that model, tools and all.

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
        output_tool, hand_offs = _output_tools(model_request_parameters)
        tools = _tools_left(messages, [*hand_offs, *model_request_parameters.function_tools])
        properties = _fields(output_tool) if output_tool else {}
        state = _map_messages(messages)
        instruction_parts = self._get_instruction_parts(messages, model_request_parameters) or []
        instructions = '\n\n'.join(part.content for part in instruction_parts) or None
        questions = _questions(properties, output_tool, instructions) if output_tool else {}
        tool_key = _tool_question(questions, output_tool, tools)
        settings = cast(TypeSafeModelSettings, model_settings or {})

        timeout = settings.get('timeout')
        try:
            response = await self.client.system_one(
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

        args, provider_details = _answers(response.answers, properties, questions)
        parts: list[ModelResponsePart] = []
        if output_tool:
            parts.append(ToolCallPart(output_tool.name, args, _utils.generate_tool_call_id()))
        if tool_key is not None:
            threshold = settings.get('typesafe_tool_call_threshold', 0.8)
            if call := _tool_call(
                self._model_name, response.answers.get(tool_key), output_tool, tools, threshold, provider_details
            ):
                parts = [call]

        return ModelResponse(
            parts=parts,
            usage=usage.RequestUsage(
                input_tokens=response.usage.input_tokens or 0, output_tokens=response.usage.output_tokens or 0
            ),
            model_name=response.model,
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            provider_details=provider_details,
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

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for i, part in enumerate(self._response.parts):
            assert isinstance(part, ToolCallPart)  # `request` builds nothing else
            yield self._parts_manager.handle_tool_call_part(
                vendor_part_id=i, tool_name=part.tool_name, args=part.args, tool_call_id=part.tool_call_id
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


def _answers(
    answers: Mapping[str, object], properties: dict[str, dict[str, Any]], questions: dict[str, Noul | Choice | Score]
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
            _set(args, name, [option for option, p in labelled.items() if p >= 0.5])
            confidence[name] = min(abs(p - 0.5) * 2 for p in labelled.values())
            probabilities[name] = labelled
            continue
        answer = answers.get(name)
        if isinstance(questions[name], Noul) and isinstance(answer, NoulAnswer):
            if prop.get('type') == 'number':
                # The probability is the answer, so there is no separate confidence to report: a field
                # that asks for the number would otherwise get it back twice under two names.
                _set(args, name, answer.noul)
            else:
                # Jev reports no confidence for a yes/no: `noul` is the probability of yes, and what is
                # lost in rounding it to an answer is how sure that answer is. That is the distance from
                # the coin flip, doubled so it runs 0 to 1 like the confidence Jev reports for the other
                # two kinds of question — a no returned at 0.01 is a confident no, and reports 0.98.
                _set(args, name, answer.noul >= 0.5)
                confidence[name] = abs(answer.noul - 0.5) * 2
        elif isinstance(questions[name], Choice) and isinstance(answer, ChoiceAnswer):
            _set(args, name, None if answer.choice == none_key else answer.choice)
            confidence[name] = answer.confidence
            probabilities[name] = answer.probabilities
        elif isinstance(questions[name], Score) and isinstance(answer, ScoreAnswer):
            # `score` is a position along the rubric and falls between levels. The answer has to be one
            # of them, and TypeSafe's way to get one is to "round it to the nearest level"; the mode
            # would throw away the ordering that makes a rubric a rubric.
            _set(args, name, round(answer.score))
            confidence[name] = answer.confidence
            probabilities[name] = {str(level): p for level, p in answer.probabilities.items()}
            scores[name] = answer.score
        else:
            raise UnexpectedModelBehavior(f'Unexpected answer from TypeSafe for output field {name!r}: {answer!r}')
    return args, {'confidence': confidence, 'probabilities': probabilities, 'scores': scores}


def _tool_call(
    model_name: str,
    answer: object,
    output_tool: ToolDefinition | None,
    tools: list[ToolDefinition],
    threshold: float,
    provider_details: dict[str, Any],
) -> ToolCallPart | None:
    """The tool call Jev makes itself from its answer to the tool question, if it takes one it can make.

    With nothing to fill, the pick is the answer. Otherwise a tool picked below the threshold is a lean, and the
    output is filled. A tool taken that needs arguments is raised as `ToolCallProposed` for a model behind Jev.
    """
    if not isinstance(answer, ChoiceAnswer):
        raise UnexpectedModelBehavior(f'Unexpected answer from TypeSafe for the tool question: {answer!r}')
    probability = answer.probabilities[answer.choice]
    # The pick and its probabilities are reported either way, so the hand-off rate can be watched.
    provider_details['tool'] = {'choice': answer.choice, 'probabilities': answer.probabilities}
    if output_tool is not None and (answer.choice == output_tool.name or probability < threshold):
        return None
    tool = next(tool for tool in tools if tool.name == answer.choice)
    if tool.parameters_json_schema.get('properties'):
        raise ToolCallProposed(model_name, answer.choice, probability)
    # Nothing to write, so Jev makes the call itself.
    return ToolCallPart(answer.choice, {}, _utils.generate_tool_call_id())


def _output_tools(
    model_request_parameters: ModelRequestParameters,
) -> tuple[ToolDefinition | None, list[ToolDefinition]]:
    """The output tool with fields for Jev to fill, if there is one, and the ones that take no arguments.

    An output function that takes nothing, or only the run context, is a hand-off Jev can pick without writing
    anything, so any number of them can sit beside the one output type it fills. A second output type with fields
    would be a second set of questions with no way to choose between them, and is a `UserError`, like everything
    else this agent could ask for that Jev cannot do.
    """
    if model_request_parameters.allow_text_output:
        raise UserError(
            'Text output is not supported by this model. Give the agent one structured `output_type`, '
            'such as a `BaseModel`, without `str`, `NativeOutput` or `PromptedOutput`.'
        )
    with_fields: list[ToolDefinition] = []
    hand_offs: list[ToolDefinition] = []
    for tool in model_request_parameters.output_tools:
        (with_fields if tool.parameters_json_schema.get('properties') else hand_offs).append(tool)
    if len(with_fields) > 1:
        raise UserError(
            f'Multiple output types with fields are not supported by this model; got {len(with_fields)}. '
            'Give the agent one structured `output_type`, beside any output functions that take no arguments.'
        )
    return (with_fields[0] if with_fields else None), hand_offs


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
            prop = resolve(prop)
            if prop.get('type') == 'object' and prop.get('properties'):
                fields.update(flatten(prop['properties'], f'{prefix}{name}.'))
            else:
                fields[f'{prefix}{name}'] = prop
        return fields

    return flatten(schema['properties'], '')


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
    if output_tool.description and output_tool.description != DEFAULT_OUTPUT_TOOL_DESCRIPTION:
        ask['goal'] = output_tool.description
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


def _tools_left(messages: list[ModelMessage], tools: list[ToolDefinition]) -> list[ToolDefinition]:
    """The tools still on offer: one with no arguments is offered once per run.

    Once it has been called, its result is in the history, and the same call could only return the same result;
    left on offer, Jev keeps picking it, since the text still calls for it. A tool with arguments stays, because
    the model that makes that call can vary them.
    """
    called = {
        part.tool_name
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    }
    return [tool for tool in tools if tool.parameters_json_schema.get('properties') or tool.name not in called]


def _tool_question(
    questions: dict[str, Noul | Choice | Score], output_tool: ToolDefinition | None, tools: list[ToolDefinition]
) -> str | None:
    """With tools attached, one more question: which tool the text calls for, the output tool among them.

    Jev cannot write a tool's arguments, but it can tell which tool the text calls for, and either call it itself
    when there are none to write or leave the call to a model behind it. The output tool is the first option,
    described by what the agent is for, so that filling the output is an action weighed against the others. Asked
    instead whether it *can* answer, Jev hands off nearly everything: that is a question about the question, not
    about the text.
    """
    if output_tool is None and len(tools) < 2:
        raise UserError(
            'An `output_type` with no fields is not supported by this model; there is nothing to ask Jev. '
            'Give it fields, or more than one tool to pick between.'
        )
    if not tools:
        return None
    key = 'tool'
    while key in questions:
        key += '_'
    criteria: dict[str, str | None] = {output_tool.name: output_tool.description} if output_tool else {}
    criteria.update((tool.name, tool.description) for tool in tools)
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
            f'Give each member of the `IntEnum` a docstring describing that score.'
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
