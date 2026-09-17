from __future__ import annotations as _annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from typing_extensions import assert_never

from .. import _utils, usage
from .._http import to_httpx2_timeout
from .._output import DEFAULT_OUTPUT_TOOL_DESCRIPTION
from ..exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from ..messages import (
    CompactionPart,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
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

__all__ = ('TypeSafeModel', 'TypeSafeModelName', 'TypeSafeModelSettings', 'LatestTypeSafeModelNames')

LatestTypeSafeModelNames = Literal['jev-latest', 'jev-preview']
"""TypeSafe aliases, which move when a release ships. `jev-preview` runs ahead of `jev-latest` when there is a
preview build. A versioned id such as `jev-1.13.0` is accepted too, and is what to use once a confidence
threshold has been tuned against one. https://docs.typesafe.ai/models"""

TypeSafeModelName = str | LatestTypeSafeModelNames
"""Possible TypeSafe model names."""

_UNSUPPORTED_FIELD_HINT = (
    'Use `bool`, a `Literal` or `Enum` of two or more strings, an `IntEnum` whose members are 0 upwards with a '
    'docstring each, or a `float` bounded with `ge=0` and `le=1`.'
)


class TypeSafeModelSettings(ModelSettings, total=False):
    """Settings used for a TypeSafe model request."""

    # ALL FIELDS MUST BE `typesafe_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    # This class is a placeholder for any future TypeSafe-specific settings


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

    Anything Jev cannot do is refused with a [`UserError`][pydantic_ai.exceptions.UserError] before a request
    is sent: text output, other field types, tools, files in the prompt or history, and streaming.

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
        output_tool = _output_tool(model_request_parameters)
        properties = _properties(output_tool)
        system_prompts, state = _map_messages(messages)
        instruction_parts = self._get_instruction_parts(messages, model_request_parameters) or []
        instructions = '\n\n'.join([*system_prompts, *(part.content for part in instruction_parts)]) or None
        questions = _questions(properties, output_tool, instructions)
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

        args: dict[str, Any] = {}
        confidence: dict[str, float] = {}
        probabilities: dict[str, dict[str, float]] = {}
        scores: dict[str, float] = {}
        for name, prop in properties.items():
            answer = response.answers.get(name)
            if isinstance(questions[name], Noul) and isinstance(answer, NoulAnswer):
                if prop.get('type') == 'number':
                    # The probability is the answer, so there is no separate confidence to report: a field
                    # that asks for the number would otherwise get it back twice under two names.
                    args[name] = answer.noul
                else:
                    # Jev reports no confidence for a yes/no: `noul` is the probability of yes, and what is
                    # lost in rounding it to an answer is how sure that answer is. That is the distance from
                    # the coin flip, doubled so it runs 0 to 1 like the confidence Jev reports for the other
                    # two kinds of question — a no returned at 0.01 is a confident no, and reports 0.98.
                    args[name] = answer.noul >= 0.5
                    confidence[name] = abs(answer.noul - 0.5) * 2
            elif isinstance(questions[name], Choice) and isinstance(answer, ChoiceAnswer):
                args[name] = answer.choice
                confidence[name] = answer.confidence
                probabilities[name] = answer.probabilities
            elif isinstance(questions[name], Score) and isinstance(answer, ScoreAnswer):
                # `score` is a position along the rubric and falls between levels. The answer has to be one
                # of them, and TypeSafe's way to get one is to "round it to the nearest level"; the mode
                # would throw away the ordering that makes a rubric a rubric.
                args[name] = round(answer.score)
                confidence[name] = answer.confidence
                probabilities[name] = {str(level): p for level, p in answer.probabilities.items()}
                scores[name] = answer.score
            else:
                raise UnexpectedModelBehavior(f'Unexpected answer from TypeSafe for output field {name!r}: {answer!r}')

        return ModelResponse(
            parts=[ToolCallPart(output_tool.name, args, _utils.generate_tool_call_id())],
            usage=usage.RequestUsage(
                input_tokens=response.usage.input_tokens or 0, output_tokens=response.usage.output_tokens or 0
            ),
            model_name=response.model,
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            provider_details={'confidence': confidence, 'probabilities': probabilities, 'scores': scores},
            finish_reason='tool_call',
        )


def _output_tool(model_request_parameters: ModelRequestParameters) -> ToolDefinition:
    """The one output tool Jev answers, or a `UserError` saying why this agent cannot run on Jev."""
    if model_request_parameters.function_tools:
        raise UserError('Function tools are not supported by this model. Give the agent an `output_type` and no tools.')
    if model_request_parameters.allow_text_output:
        raise UserError(
            'Text output is not supported by this model. Give the agent one structured `output_type`, '
            'such as a `BaseModel`, without `str`, `NativeOutput` or `PromptedOutput`.'
        )
    output_tools = model_request_parameters.output_tools
    if len(output_tools) != 1:
        raise UserError(
            f'Multiple output types are not supported by this model; got {len(output_tools)}. '
            'Give the agent one structured `output_type`.'
        )
    return output_tools[0]


def _properties(output_tool: ToolDefinition) -> dict[str, dict[str, Any]]:
    """The output schema's fields, with `$ref`s to `$defs` (how Pydantic renders an `Enum`) resolved."""
    schema = output_tool.parameters_json_schema
    defs: dict[str, Any] = schema.get('$defs', {})
    properties: dict[str, dict[str, Any]] = {}
    for name, prop in schema['properties'].items():
        if ref := prop.get('$ref'):
            prop = {**defs[ref.removeprefix('#/$defs/')], **{k: v for k, v in prop.items() if k != '$ref'}}
        properties[name] = prop
    return properties


def _questions(
    properties: dict[str, dict[str, Any]], output_tool: ToolDefinition, instructions: str | None
) -> dict[str, Noul | Choice | Score]:
    """One Jev question per output field."""
    questions: dict[str, Noul | Choice | Score] = {}
    if not properties:
        raise UserError('An `output_type` with no fields is not supported by this model; there is nothing to ask Jev.')
    for name, prop in properties.items():
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

        options: dict[Any, str | None] | None = None
        if 'enum' in prop:
            options = dict.fromkeys(prop['enum'])
        elif 'anyOf' in prop and all('const' in option for option in prop['anyOf']):
            options = {option['const']: option.get('description') for option in prop['anyOf']}

        if options is not None:
            # `bool` is an `int` in Python but never a rubric level, and it is handled as a yes/no below.
            if options and all(isinstance(option, int) and not isinstance(option, bool) for option in options):
                questions[name] = _score_question(name, cast('dict[int, str | None]', options), ask)
            elif len(options) < 2 or not all(isinstance(option, str) for option in options):
                raise UserError(
                    f'Output field {name!r} is not supported by this model: its options are not two or more strings. '
                    f'{_UNSUPPORTED_FIELD_HINT}'
                )
            else:
                questions[name] = Choice(instructions=ask or None, criteria=cast('dict[str, str | None]', options))
        elif prop.get('type') == 'boolean' or (
            prop.get('type') == 'number' and prop.get('minimum') == 0 and prop.get('maximum') == 1
        ):
            if not ask:
                # A pick-one or a rubric still says what it is asking through its options; a yes/no has
                # nothing else, and Jev rejects a question with neither instructions nor criteria.
                raise UserError(
                    f'Output field {name!r} asks Jev nothing. A question is not part of the text being judged: '
                    f'give the field a description, or the agent `instructions`, and leave the prompt to the '
                    f'material the question is about.'
                )
            questions[name] = Noul(instructions=ask)
        else:
            raise UserError(f'Output field {name!r} is not supported by this model. {_UNSUPPORTED_FIELD_HINT}')
    return questions


def _score_question(name: str, options: dict[int, str | None], ask: dict[str, JSONContent]) -> Score:
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
    return Score(instructions=ask or None, criteria=cast('list[JSONContent]', criteria))


def _prompt_text(part: UserPromptPart) -> str:
    items = [part.content] if isinstance(part.content, str) else list(part.content)
    if not all(isinstance(item, str) for item in items):
        raise UserError(
            'Files are not supported by this model; images, audio, video and documents cannot be sent to Jev.'
        )
    return '\n\n'.join(cast(list[str], items))


def _map_request(message: ModelRequest, *, latest: bool) -> tuple[list[str], list[JSONContent], list[str]]:
    """Map a request to system prompts, history entries, and the text to judge."""
    system_prompts: list[str] = []
    history: list[JSONContent] = []
    prompt_parts: list[str] = []
    for part in message.parts:
        if isinstance(part, SystemPromptPart):
            system_prompts.append(part.content)
        elif isinstance(part, UserPromptPart):
            text = _prompt_text(part)
            if latest:
                prompt_parts.append(text)
            else:
                history.append({'user': text})
        elif isinstance(part, ToolReturnPart):
            history.append({'tool_return': {'name': part.tool_name, 'content': part.model_response_str()}})
        elif isinstance(part, RetryPromptPart):
            history.append({'retry': part.model_response()})
        elif isinstance(part, ToolAvailabilityDeltaPart):  # pragma: no cover
            raise _unsynthesized_tool_availability_delta_error()
        elif isinstance(part, SpeechPart):  # pragma: no cover
            # `Model.prepare_messages` turns realtime speech into `UserPromptPart`s before this runs.
            raise _unconverted_speech_part_error()
        else:
            assert_never(part)
    return system_prompts, history, prompt_parts


def _response_entries(message: ModelResponse) -> list[JSONContent]:
    """Map a response to history entries, excluding the model's private thinking."""
    entries: list[JSONContent] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            entries.append({'assistant': part.content})
        elif isinstance(part, ToolCallPart | NativeToolCallPart):
            entries.append({'tool_call': {'name': part.tool_name, 'args': part.args_as_dict()}})
        elif isinstance(part, NativeToolReturnPart):
            entries.append({'tool_return': {'name': part.tool_name, 'content': part.model_response_str()}})
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


def _map_messages(messages: list[ModelMessage]) -> tuple[list[str], JSONContent]:
    """The system prompts, and the state to judge.

    The latest user text on its own is the whole state, as the text TypeSafe's own examples pass. With a
    conversation behind it there are two parts to keep apart, so they get named: the text under judgement
    and the `history` before it.
    """
    system_prompts: list[str] = []
    history: list[JSONContent] = []
    prompt_parts: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            request_system_prompts, entries, latest_prompt_parts = _map_request(message, latest=message is messages[-1])
            system_prompts.extend(request_system_prompts)
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
        return system_prompts, text
    state: dict[str, JSONContent] = {'history': history}
    if text:
        state['text'] = text
    return system_prompts, state
