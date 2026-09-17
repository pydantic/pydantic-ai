from __future__ import annotations as _annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from typing_extensions import assert_never

from .. import _utils, usage
from .._http import to_httpx2_timeout
from ..exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from ..messages import (
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchReturnPart,
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
        TypeSafeAPIConnectionError,
        TypeSafeAPIError,
        TypeSafeAPIResponseValidationError,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install the `typesafe-sdk` package to use the TypeSafe model, '
        'you can use the `typesafe` optional group — `pip install "pydantic-ai-slim[typesafe]"`'
    ) from _import_error

__all__ = ('TypeSafeModel', 'TypeSafeModelName', 'TypeSafeModelSettings', 'LatestTypeSafeModelNames')

LatestTypeSafeModelNames = Literal['jev-latest']
"""Latest TypeSafe model names."""

TypeSafeModelName = str | LatestTypeSafeModelNames
"""Possible TypeSafe model names."""

_UNSUPPORTED_FIELD_HINT = 'Use `bool`, a `Literal` or `Enum` of strings, or a `float` bounded with `ge=0` and `le=1`.'


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
        verdict: Literal['run', 'reject', 'ask'] = Field(
            description='How to handle this command.',
            json_schema_extra={
                'typesafe_criteria': {
                    'run': 'Reads, builds, tests or edits inside the project. Reversible.',
                    'reject': 'Destroys data, rewrites shared history, or sends secrets over the network.',
                    'ask': 'Legitimate but consequential enough that a human should confirm.',
                }
            },
        )
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

    The field description is the question. The output type's docstring and the agent's instructions go along
    as context. Describe each option of a `Literal` or `Enum` with
    `json_schema_extra={'typesafe_criteria': {option: description}}`, or Jev only sees the option names.
    Jev's confidence per field is in
    [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details] under `confidence`,
    and the full distribution of each pick-one field under `probabilities`.

    Anything Jev cannot do is refused with a [`UserError`][pydantic_ai.exceptions.UserError] before a request
    is sent: text output, other field types, tools, non-text prompts, tool calls in the history, retries, and
    streaming. Earlier user prompts are sent along; earlier answers are not.

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
                'typesafe' or an instance of `Provider[AsyncTypeSafeClient]`. If not provided, a new provider will
                be created using the other parameters.
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
        instructions, state = _map_messages(messages, output_tool.name)
        questions = _questions(output_tool, instructions)
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

        args: dict[str, Any] = {}
        confidence: dict[str, float] = {}
        probabilities: dict[str, dict[str, float]] = {}
        for name, prop in _properties(output_tool).items():
            answer = response.answers.get(name)
            if answer is None:
                raise UnexpectedModelBehavior(f'TypeSafe returned no answer for output field {name!r}.')
            if isinstance(answer, NoulAnswer):
                args[name] = answer.noul if prop.get('type') == 'number' else answer.noul >= 0.5
                confidence[name] = answer.noul
            elif isinstance(answer, ChoiceAnswer):
                args[name] = answer.choice
                confidence[name] = answer.confidence
                probabilities[name] = answer.probabilities
            else:
                raise UnexpectedModelBehavior(f'Unexpected answer type from TypeSafe for field {name!r}: {answer!r}')

        return ModelResponse(
            parts=[ToolCallPart(output_tool.name, args, _utils.generate_tool_call_id())],
            usage=usage.RequestUsage(
                input_tokens=response.usage.input_tokens or 0, output_tokens=response.usage.output_tokens or 0
            ),
            model_name=response.model,
            provider_name=self._provider.name,
            provider_url=self._provider.base_url,
            provider_details={'confidence': confidence, 'probabilities': probabilities},
            finish_reason='tool_call',
        )


def _output_tool(model_request_parameters: ModelRequestParameters) -> ToolDefinition:
    """The one output tool Jev answers, or a `UserError` saying why this agent cannot run on Jev."""
    if model_request_parameters.function_tools:
        raise UserError('Tools are not supported by this model. Give the agent an `output_type` and no tools.')
    output_tools = model_request_parameters.output_tools
    if model_request_parameters.allow_text_output or len(output_tools) != 1:
        raise UserError(
            'Text output is not supported by this model. Give the agent exactly one structured `output_type`, '
            'such as a `BaseModel`, without `str`, `NativeOutput` or `PromptedOutput`.'
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


def _questions(output_tool: ToolDefinition, instructions: str | None) -> dict[str, Noul | Choice]:
    """One Jev question per output field."""
    questions: dict[str, Noul | Choice] = {}
    properties = _properties(output_tool)
    if not properties:
        raise UserError('An `output_type` with no fields is not supported by this model; there is nothing to ask Jev.')
    for name, prop in properties.items():
        ask: dict[str, JSONContent] = {'question': prop.get('description') or name}
        if output_tool.description:
            ask['goal'] = output_tool.description
        if instructions:
            ask['instructions'] = instructions

        if options := prop.get('enum'):
            if not all(isinstance(option, str) for option in options):
                raise UserError(
                    f'Output field {name!r} is not supported by this model: its options are not all strings. {_UNSUPPORTED_FIELD_HINT}'
                )
            criteria: dict[str, JSONContent] = prop.get('typesafe_criteria') or {option: option for option in options}
            if set(criteria) != set(options):
                raise UserError(
                    f'`typesafe_criteria` for output field {name!r} must describe exactly its options {sorted(options)}, '
                    f'got {sorted(criteria)}.'
                )
            questions[name] = Choice(instructions=ask, criteria=criteria)
        elif prop.get('type') == 'boolean':
            questions[name] = Noul(instructions=ask)
        elif prop.get('type') == 'number' and prop.get('minimum') == 0 and prop.get('maximum') == 1:
            questions[name] = Noul(instructions=ask)
        else:
            raise UserError(f'Output field {name!r} is not supported by this model. {_UNSUPPORTED_FIELD_HINT}')
    return questions


def _prompt_text(part: UserPromptPart) -> list[str]:
    items = [part.content] if isinstance(part.content, str) else list(part.content)
    if not all(isinstance(item, str) for item in items):
        raise UserError(
            'Non-text prompts are not supported by this model; images, audio, video and documents cannot be sent to Jev.'
        )
    return cast(list[str], items)


def _map_messages(messages: list[ModelMessage], output_tool_name: str) -> tuple[str | None, dict[str, JSONContent]]:
    """The instructions to judge by, and the state to judge: the latest user text, plus earlier turns' text."""
    instructions: list[str] = []
    prompts: list[str] = []
    latest: list[str] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            if message.instructions:
                instructions.append(message.instructions)
            prompts.extend(latest)
            latest = []
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    instructions.append(part.content)
                elif isinstance(part, UserPromptPart):
                    latest.extend(_prompt_text(part))
                elif isinstance(part, RetryPromptPart):
                    raise UserError(
                        'Retries are not supported by this model; Jev cannot revise an answer. '
                        f'This retry asked: {part.model_response()}'
                    )
                elif isinstance(part, ToolReturnPart | ToolSearchReturnPart | LoadCapabilityReturnPart):
                    # The agent's own "Final result processed." return for an earlier answer is fine; nothing else is.
                    if not (isinstance(part, ToolReturnPart) and part.tool_name == output_tool_name):
                        raise UserError('Tool results are not supported by this model, which cannot call tools.')
                elif isinstance(part, ToolAvailabilityDeltaPart):
                    raise _unsynthesized_tool_availability_delta_error()
                elif isinstance(part, SpeechPart):
                    raise _unconverted_speech_part_error()
                else:
                    assert_never(part)
        elif isinstance(message, ModelResponse):
            pass  # Earlier answers are not context for Jev; the questions are about the user's text.
        else:
            assert_never(message)

    if not any(latest):
        raise UserError('A request without user text is not supported by this model; Jev needs text to judge.')
    state: dict[str, JSONContent] = {'prompt': '\n\n'.join(latest)}
    if prompts:
        state['previous_prompts'] = prompts
    return '\n\n'.join(instructions) or None, state
