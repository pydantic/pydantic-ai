from __future__ import annotations as _annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import ClassVar, Literal, TypeVar, assert_never

from pydantic import JsonValue

from .._http import to_httpx2_timeout
from ..exceptions import ContentFilterError, UnexpectedModelBehavior, UserError
from ..messages import (
    BaseToolReturnPart,
    BinaryContent,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolReturnPart,
    TextPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    is_multi_modal_content,
)
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import get_user_agent
from .decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionModel,
    DecisionModelSettings,
    DecisionQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    _answer_fits,  # pyright: ignore[reportPrivateUsage]
    _score_matches_probabilities,  # pyright: ignore[reportPrivateUsage]
)

try:
    from openai import NOT_GIVEN, AsyncOpenAI
    from openai.types import Decision
    from openai.types.decision import (
        AnswerAnswerResourceChoice,
        AnswerAnswerResourcePredicate,
        AnswerAnswerResourceRefusal,
        AnswerAnswerResourceScore,
    )
    from openai.types.decision_create_params import (
        Question,
        QuestionQuestionParamChoice,
        QuestionQuestionParamChoiceChoice,
        QuestionQuestionParamPredicate,
        QuestionQuestionParamScore,
        QuestionQuestionParamScoreLevel,
    )
    from openai.types.decision_input_image_param import DecisionInputImageParam
    from openai.types.decision_input_message_param import DecisionInputMessageParam
    from openai.types.decision_input_part_union_param import DecisionInputPartUnionParam
    from openai.types.decision_input_text_param import DecisionInputTextParam

    from ..providers.openai_decisions import OpenAIDecisionsProvider
    from . import download_item
    from .openai import _map_api_errors  # pyright: ignore[reportPrivateUsage]
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the OpenAI Decisions model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = (
    'OpenAIDecisionsModel',
    'OpenAIDecisionsModelName',
    'OpenAIDecisionsModelSettings',
)

OpenAIDecisionsModelName = str
"""The ID of the model to ask, such as `gpt-6-luna`, which the Responses API also serves."""


class OpenAIDecisionsModelSettings(DecisionModelSettings, total=False):
    """Settings used for an OpenAI Decisions API request."""

    # ALL FIELDS MUST BE `openai_decisions_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    # This class is a placeholder for any future Decisions API-specific settings.


@dataclass(kw_only=True)
class _OpenAIDecisionRequest(DecisionRequest):
    input: list[DecisionInputMessageParam]


_ToolReturnPartT = TypeVar('_ToolReturnPartT', bound=BaseToolReturnPart)


@dataclass
class _DecisionImagePreparer:
    images: list[DecisionInputImageParam] = field(default_factory=list[DecisionInputImageParam])

    @staticmethod
    def _unsupported_file(context: str) -> UserError:
        return UserError(
            f'OpenAI Decisions supports text and inline images only; {context} contains an unsupported file.'
        )

    async def _add_image(self, item: BinaryContent | ImageUrl) -> str:
        image_url = (
            item.data_uri
            if isinstance(item, BinaryContent)
            else (await download_item(item, data_format='base64_uri'))['data']
        )
        label = f'<image {len(self.images) + 1}>'
        self.images.append(DecisionInputImageParam(type='input_image', image_url=image_url))
        return label

    async def _prepare_tool_return(self, part: _ToolReturnPartT) -> _ToolReturnPartT:
        if not part.files:
            return part
        content: list[str] = []
        for item in part.content_items(mode='str', wrap_if_error=False):
            if isinstance(item, str):
                content.append(item)
            elif isinstance(item, BinaryContent):
                if not item.is_image:
                    raise self._unsupported_file('A tool result')
                content.append(await self._add_image(item))
            elif isinstance(item, ImageUrl):
                content.append(await self._add_image(item))
            else:
                raise self._unsupported_file('A tool result')
        return replace(part, content=content)

    async def prepare(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        prepared: list[ModelMessage] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                request_parts: list[ModelRequestPart] = []
                for part in message.parts:
                    if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                        content: list[UserContent] = []
                        for item in part.content:
                            if isinstance(item, (BinaryContent, ImageUrl)):
                                if isinstance(item, BinaryContent) and not item.is_image:
                                    raise self._unsupported_file('A user prompt')
                                content.append(await self._add_image(item))
                            elif is_multi_modal_content(item):
                                raise self._unsupported_file('A user prompt')
                            else:
                                content.append(item)
                        request_parts.append(replace(part, content=content))
                    elif isinstance(part, ToolReturnPart):
                        request_parts.append(await self._prepare_tool_return(part))
                    else:
                        request_parts.append(part)
                prepared.append(replace(message, parts=request_parts))
            elif isinstance(message, ModelResponse):
                response_parts: list[ModelResponsePart] = []
                for part in message.parts:
                    if isinstance(part, FilePart):
                        if not part.content.is_image:
                            raise self._unsupported_file('An assistant response')
                        response_parts.append(TextPart(content=await self._add_image(part.content)))
                    elif isinstance(part, NativeToolReturnPart):
                        response_parts.append(await self._prepare_tool_return(part))
                    else:
                        response_parts.append(part)
                prepared.append(replace(message, parts=response_parts))
            else:
                assert_never(message)
        return prepared


@dataclass(init=False)
class OpenAIDecisionsModel(DecisionModel[AsyncOpenAI]):
    """The model class for OpenAI's Decisions API, which runs a GPT model as a [decision model][pydantic_ai.models.decision.DecisionModel].

    The Decisions API answers typed questions about text or images, each with a probability or a distribution over the
    options, rather than writing text. An agent whose job is to decide something runs on it like on any other model,
    with the `output_type` as the questions:

    ```python
    from typing import Literal

    from pydantic import BaseModel, Field

    from pydantic_ai import Agent


    class Handling(BaseModel):
        verdict: Literal['run', 'reject', 'ask'] = Field(description='How to handle this command.')
        irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


    agent = Agent('openai-decisions:gpt-6-luna', output_type=Handling)
    ...
    ```

    See [Decision models](../../models/decision.md) for how an agent's output type and tools
    become questions, and [OpenAI](../../models/openai.md#decisions-api) for setup.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    max_choice_options: ClassVar[int | None] = 255
    """The API takes at most this many options in one pick-one; a 256th is a 400.

    `QuestionParamChoice` in https://github.com/openai/openai-openapi/blob/main/openapi.yaml
    """

    max_score_levels: ClassVar[int | None] = 10
    """The API takes at most this many levels in one rubric; an 11th is a 400.

    `QuestionParamScore` in https://github.com/openai/openai-openapi/blob/main/openapi.yaml
    """

    _model_name: OpenAIDecisionsModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: OpenAIDecisionsModelName,
        *,
        provider: Literal['openai-decisions'] | OpenAIDecisionsProvider = 'openai-decisions',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI Decisions model.

        Args:
            model_name: The name of the OpenAI model to use, such as `gpt-6-luna`.
            provider: The provider to use for the API's URL and key.
            profile: The model profile to use. Defaults to one selected by the provider.
            settings: Model-specific settings used as defaults for this model.
        """
        self._model_name = model_name
        if isinstance(provider, str):
            provider = OpenAIDecisionsProvider()
        self._provider = provider
        super().__init__(settings=settings, profile=profile)

    @property
    def client(self) -> AsyncOpenAI:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> OpenAIDecisionsModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._provider.name

    async def _prepare_decision_request(
        self, messages: list[ModelMessage], model_settings: DecisionModelSettings, *, turn: bool
    ) -> DecisionRequest:
        _validate_extra_body(model_settings.get('extra_body'))
        preparer = _DecisionImagePreparer()
        prepared_messages = await preparer.prepare(messages)
        if not preparer.images:
            return await super()._prepare_decision_request(messages, model_settings, turn=turn)

        template = await super()._prepare_decision_request(prepared_messages, model_settings, turn=turn)
        content: list[DecisionInputPartUnionParam] = [
            DecisionInputTextParam(type='input_text', text=_text(template.state))
        ]
        for index, image in enumerate(preparer.images, start=1):
            content.extend(
                (
                    DecisionInputTextParam(type='input_text', text=f'<image {index}>:'),
                    image,
                )
            )
        return _OpenAIDecisionRequest(
            state=template.state,
            questions={},
            input=[DecisionInputMessageParam(role='user', content=content)],
        )

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        """Send one request to the `/v1/decisions` endpoint."""
        extra_headers: dict[str, str] = dict(model_settings.get('extra_headers', {}))
        if all(name.lower() != 'user-agent' for name in extra_headers):
            extra_headers['User-Agent'] = get_user_agent()
        _validate_extra_body(model_settings.get('extra_body'))
        with _map_api_errors(self._model_name, self._provider.model_id_namespace):
            response = await self.client.decisions.with_raw_response.create(
                model=self._model_name,
                input=request.input if isinstance(request, _OpenAIDecisionRequest) else _text(request.state),
                questions=[_question(name, question) for name, question in request.questions.items()],
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
                timeout=to_httpx2_timeout(model_settings.get('timeout', NOT_GIVEN)),
            )

        # The SDK builds its response models without validating them, so the body is validated here.
        try:
            data = json.loads(response.content)
            decision = Decision.model_validate(data)
        except ValueError as e:
            raise UnexpectedModelBehavior(f'Invalid response from the OpenAI Decisions API: {e}', response.text) from e
        by_name = {answer.name: answer for answer in decision.answers if answer.name is not None}
        if len(by_name) != len(decision.answers) or by_name.keys() != request.questions.keys():
            raise UnexpectedModelBehavior(
                'Invalid response from the OpenAI Decisions API: answer names do not match the questions', response.text
            )
        answers: dict[str, DecisionAnswer] = {}
        refused: list[str] = []
        for name, question in request.questions.items():
            answer = by_name[name]
            if isinstance(answer, AnswerAnswerResourceRefusal):
                refused.append(name)
            elif (
                (converted := _answer(answer)) is not None
                and _answer_fits(question, converted)
                and (
                    not isinstance(question, ScoreQuestion)
                    or not isinstance(converted, ScoreAnswer)
                    or _score_matches_probabilities(question, converted)
                )
            ):
                answers[name] = converted
            else:
                raise UnexpectedModelBehavior(
                    f'Invalid response from the OpenAI Decisions API: answer {name!r} does not match its question: {answer!r}',
                    response.text,
                )
        if refused:
            raise ContentFilterError(
                f'Content filter triggered. The OpenAI Decisions API declined to answer: {", ".join(map(repr, refused))}',
                response.text,
            )
        return DecisionResponse(
            answers=answers,
            model_name=decision.model,
            usage=RequestUsage.extract(
                data,
                provider=self.system,
                provider_url=self.base_url,
                provider_fallback='openai',
                api_flavor='responses',
            ),
            # The body carries no ID of its own.
            provider_response_id=response.request_id,
        )


def _validate_extra_body(extra_body: object) -> None:
    if extra_body is not None and not isinstance(extra_body, Mapping):
        raise UserError(f'`extra_body` must be a mapping to send it to the OpenAI Decisions API; got {extra_body!r}.')


def _text(value: JsonValue) -> str:
    """A protocol value as the text the API takes: a string as it is, anything else as JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _question(name: str, question: DecisionQuestion) -> Question:
    """A protocol question as a Decisions API question, whose `predicate` is the protocol's yes/no."""
    if isinstance(question, NoulQuestion):
        instructions = _instructions(_with_meanings(question.instructions, question.criteria or NoulCriteria()))
        return QuestionQuestionParamPredicate(type='predicate', name=name, instructions=instructions)
    elif isinstance(question, ChoiceQuestion):
        choices = [_option(label, meaning) for label, meaning in question.criteria.items()]
        return QuestionQuestionParamChoice(
            type='choice', name=name, instructions=_instructions(question.instructions), choices=choices
        )
    elif isinstance(question, ScoreQuestion):
        levels = [_level(str(level), meaning) for level, meaning in enumerate(question.criteria)]
        return QuestionQuestionParamScore(
            type='score', name=name, instructions=_instructions(question.instructions), levels=levels
        )
    else:
        assert_never(question)


def _instructions(instructions: JsonValue) -> str:
    """A question's instructions as the text the API requires on every question: empty where there are none."""
    return '' if instructions is None else _text(instructions)


def _with_meanings(instructions: JsonValue, criteria: NoulCriteria) -> JsonValue:
    """A yes/no question's instructions, with what yes and no mean, which a `predicate` has no field of its own for."""
    meanings: dict[str, JsonValue] = {
        answer: meaning for answer, meaning in (('yes', criteria.true), ('no', criteria.false)) if meaning is not None
    }
    if not meanings:
        return instructions
    if instructions is None:
        return meanings
    if isinstance(instructions, dict):
        return {**instructions, **meanings}
    return {'question': instructions, **meanings}


def _option(value: str, meaning: JsonValue) -> QuestionQuestionParamChoiceChoice:
    if meaning is None:
        return QuestionQuestionParamChoiceChoice(value=value)
    return QuestionQuestionParamChoiceChoice(value=value, description=_text(meaning))


def _level(label: str, meaning: JsonValue) -> QuestionQuestionParamScoreLevel:
    if meaning is None:
        return QuestionQuestionParamScoreLevel(label=label)
    return QuestionQuestionParamScoreLevel(label=label, description=_text(meaning))


def _answer(
    answer: AnswerAnswerResourcePredicate | AnswerAnswerResourceChoice | AnswerAnswerResourceScore,
) -> DecisionAnswer | None:
    """A Decisions API answer as the protocol's, or `None` for one no question here allows.

    That is a boolean option, which no question here offers, or an option or level given twice.
    """
    if isinstance(answer, AnswerAnswerResourcePredicate):
        return NoulAnswer(noul=answer.probability)
    elif isinstance(answer, AnswerAnswerResourceChoice):
        probabilities = {
            option.value: option.probability for option in answer.probabilities if isinstance(option.value, str)
        }
        if not isinstance(answer.choice, str) or len(probabilities) != len(answer.probabilities):
            return None
        return ChoiceAnswer(choice=answer.choice, confidence=answer.confidence, probabilities=probabilities)
    elif isinstance(answer, AnswerAnswerResourceScore):
        probabilities = {level.value: level.probability for level in answer.probabilities}
        if len(probabilities) != len(answer.probabilities):
            return None
        return ScoreAnswer(score=answer.score, confidence=answer.confidence, probabilities=probabilities)
    else:
        assert_never(answer)
