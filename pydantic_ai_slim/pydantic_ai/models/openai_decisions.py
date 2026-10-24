from __future__ import annotations as _annotations

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import ClassVar, Literal, assert_never, cast

from pydantic import JsonValue

from .. import _model_errors
from .._framing import TOOL_RESULT_TAG, close_tag, neutralize_tags, open_tag
from .._http import to_httpx2_timeout
from .._thinking_part import render_replayed_thinking
from ..exceptions import ContentFilterError, ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from ..messages import (
    ERROR_OUTCOMES,
    BaseToolReturnPart,
    BinaryContent,
    CachePoint,
    CompactionPart,
    FilePart,
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    MultiModalContent,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryFeedbackPart,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]  # TODO(v3): remove RetryPromptPart
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    _translate_legacy_retry_part,  # pyright: ignore[reportPrivateUsage]
)
from ..profiles import ModelProfile, ModelProfileSpec
from ..profiles.decision import decision_model_profile
from ..providers import Provider
from ..settings import ModelSettings
from ..usage import RequestUsage
from . import (
    _unconverted_speech_part_error,  # pyright: ignore[reportPrivateUsage]
    _unprepared_part_error,  # pyright: ignore[reportPrivateUsage]
    _UnpreparedPart,  # pyright: ignore[reportPrivateUsage]
    _unprojected_instruction_delta_error,  # pyright: ignore[reportPrivateUsage]
    get_user_agent,
)
from ._sdk_retries import with_max_retries
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
    _probability_bounds,  # pyright: ignore[reportPrivateUsage]
    _score_fits,  # pyright: ignore[reportPrivateUsage]
)

try:
    from openai import NOT_GIVEN, APIConnectionError, APIError, APIStatusError, APITimeoutError, AsyncOpenAI
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
    from openai.types.responses import (
        EasyInputMessageParam,
        ResponseFunctionCallOutputItemListParam,
        ResponseFunctionToolCallParam,
        ResponseInputImageParam,
        ResponseInputItemParam,
        ResponseInputMessageContentListParam,
        ResponseInputTextParam,
    )
    from openai.types.responses.image_detail import ImageDetail
    from openai.types.responses.response_input_image_content_param import ResponseInputImageContentParam
    from openai.types.responses.response_input_param import FunctionCallOutput
    from openai.types.responses.response_input_text_content_param import ResponseInputTextContentParam

    from ..providers.openai import OpenAIProvider
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

OpenAIDecisionsModelName = str | Literal['gpt-6-luna']
"""The ID of the model to ask, such as `gpt-6-luna`, which the Responses API also serves."""


class OpenAIDecisionsModelSettings(DecisionModelSettings, total=False):
    """Settings used for an OpenAI Decisions API request."""

    # ALL FIELDS MUST BE `openai_decisions_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    # This class is a placeholder for any future Decisions API-specific settings.


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

    supports_image_input: ClassVar[bool] = True

    max_images: ClassVar[int | None] = 128
    """The API accepts at most 128 image parts across all input messages; a 129th returns HTTP 400.

    `DecisionInput` in https://github.com/openai/openai-openapi/blob/main/openapi.yaml.
    An oversized request raises `ModelAPIError` before image URLs are downloaded or the SDK is called.
    """

    max_questions: ClassVar[int | None] = 200
    """The API accepts at most 200 questions per request; a 201st returns HTTP 400.

    `DecisionRequest.questions.maxItems` in https://github.com/openai/openai-openapi/blob/main/openapi.yaml.
    The shared planner splits speculative fields into a second request when needed; a request that still
    exceeds the cap raises `ModelAPIError` before image URLs are downloaded or the SDK is called.
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
        provider: Literal['openai', 'openai-decisions'] | OpenAIProvider = 'openai-decisions',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an OpenAI Decisions model.

        Args:
            model_name: The name of the OpenAI model to use, such as `gpt-6-luna`.
            provider: The provider to use for the API's URL and key: an
                [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider], as for the other OpenAI models, or
                `'openai'` or `'openai-decisions'` (the default), which both create one.
            profile: The model profile to use. Defaults to a decision model's profile for the model name.
            settings: Model-specific settings used as defaults for this model.
        """
        self._model_name = model_name
        if isinstance(provider, str):
            provider = OpenAIProvider()
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
        """The system / model provider: `'openai-decisions'`, whatever the provider is called.

        The Decisions API shares OpenAI's URL with the Responses API, so this name is what tells its responses apart:
        genai-prices prices the endpoint as its own provider, and an
        [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] continuing the conversation must not
        take a decision response, which has no Responses ID to continue from, for one of its own.
        """
        return 'openai-decisions'

    def _provider_model_profile(self) -> ModelProfile:
        # The provider profiles a model name such as `gpt-6-luna` as a Responses API model, whose structured output,
        # thinking and caching this endpoint has none of.
        return decision_model_profile(self.model_name)

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        """Send one request to the `/v1/decisions` endpoint."""
        extra_headers: dict[str, str] = dict(model_settings.get('extra_headers', {}))
        if all(name.lower() != 'user-agent' for name in extra_headers):
            extra_headers['User-Agent'] = get_user_agent()
        self._check_request_limits(image_count=len(request.images), question_count=len(request.questions))
        if any(not image.is_image for image in request.images):
            raise UserError('OpenAI Decisions supports image evidence only; `request.images` contains a non-image.')
        decision_input = (
            _Conversation(request.images, self.profile).items(request.messages)
            if request.messages
            else _state_input(request)
        )
        try:
            response = await with_max_retries(self.client, model_settings).decisions.with_raw_response.create(
                model=self._model_name,
                input=decision_input,
                questions=[_question(name, question) for name, question in request.questions.items()],
                extra_headers=extra_headers,
                extra_body=model_settings.get('extra_body'),
                timeout=to_httpx2_timeout(model_settings.get('timeout', NOT_GIVEN)),
            )
        except APIError as e:
            raise _api_error(e, self._model_name) from e

        # The SDK builds its response models without validating them, so the body is validated here.
        try:
            data = json.loads(response.content)
            decision = Decision.model_validate(data, strict=True)
        except ValueError as e:
            raise UnexpectedModelBehavior('Invalid response from the OpenAI Decisions API', response.text) from e
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
            elif (converted := _answer(answer)) is not None and _answer_fits(question, converted):
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


def _state_input(request: DecisionRequest) -> str | list[DecisionInputMessageParam]:
    """A request without a conversation, as its `state` in one user message with each image after its label."""
    text = _text(request.state)
    if not request.images:
        return text
    content: list[DecisionInputPartUnionParam] = [DecisionInputTextParam(type='input_text', text=text)]
    for index, image in enumerate(request.images, start=1):
        image_input = DecisionInputImageParam(type='input_image', image_url=image.data_uri)
        if metadata := image.vendor_metadata:
            image_input['detail'] = metadata.get('detail', 'auto')
        content.extend((DecisionInputTextParam(type='input_text', text=f'<image {index}>:'), image_input))
    return [DecisionInputMessageParam(role='user', content=content)]


class _Conversation:
    """A conversation as the Responses API input items the Decisions API reads.

    The API takes `message`, `function_call`, `function_call_output`, `custom_tool_call` and
    `custom_tool_call_output` items, and refuses `reasoning` items, an assistant message with anything but text,
    and a tool output whose call is not in the same input. Its SDK type and guide still describe user messages
    only; OpenAI confirmed the native items in https://github.com/pydantic/pydantic-ai/issues/10950.

    No item IDs are sent: the API keeps no state between requests, so an ID only ties an item to a response the
    Decisions API never produced.
    """

    def __init__(self, images: Sequence[BinaryContent], profile: ModelProfile):
        self._images = iter(enumerate(images, start=1))
        self._profile = profile
        self._items: list[ResponseInputItemParam] = []

    def items(self, messages: Sequence[ModelMessage]) -> str | list[DecisionInputMessageParam]:
        calls = {
            part.tool_call_id
            for message in messages
            if isinstance(message, ModelResponse)
            for part in message.parts
            if isinstance(part, ToolCallPart | NativeToolCallPart)
        }
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    self._request_part(part, calls)
            else:
                for part in message.parts:
                    self._response_part(part, calls)
        if next(self._images, None) is not None:
            raise UserError('`request.images` holds more images than `request.messages` refers to.')
        match self._items:
            case [{'role': 'user', 'content': str(text)}]:
                # A lone text prompt goes as the plain text the API documents.
                return text
            case _:
                # The SDK still types `input` as user messages only; see the class docstring.
                return cast(list[DecisionInputMessageParam], self._items)

    def _image(self) -> tuple[int, BinaryContent, ImageDetail]:
        """The next image the messages refer to, its `<image N>` index, and its detail level."""
        try:
            index, image = next(self._images)
        except StopIteration:
            raise UserError('`request.messages` refers to more images than `request.images` holds.') from None
        return index, image, (image.vendor_metadata or {}).get('detail', 'auto')

    def _output(self, part: BaseToolReturnPart) -> ResponseFunctionCallOutputItemListParam:
        """A tool output that carries files, as the text and images it is made of, in order.

        A failed one's text goes in one error wrapper, as without files, so each image is referenced inside it by
        its `<image N>` label and follows it with that label, as the `state` does.
        """
        output: ResponseFunctionCallOutputItemListParam = []
        if part.outcome in ERROR_OUTCOMES:
            labelled: list[str] = []
            images: ResponseFunctionCallOutputItemListParam = []
            for item in part.content_items(mode='str', wrap_if_error=False):
                if isinstance(item, str):
                    labelled.append(item)
                else:
                    index, image, detail = self._image()
                    labelled.append(f'<image {index}>')
                    images.append(ResponseInputTextContentParam(type='input_text', text=f'<image {index}>:'))
                    images.append(
                        ResponseInputImageContentParam(type='input_image', image_url=image.data_uri, detail=detail)
                    )
            text = dataclasses.replace(part, content=labelled).model_response_str()
            return [ResponseInputTextContentParam(type='input_text', text=text), *images]
        for item in part.content_items(mode='str'):
            if isinstance(item, str):
                output.append(ResponseInputTextContentParam(type='input_text', text=item))
            else:
                _, image, detail = self._image()
                output.append(
                    ResponseInputImageContentParam(type='input_image', image_url=image.data_uri, detail=detail)
                )
        return output

    def _user(self, items: Sequence[str | MultiModalContent]) -> None:
        if len(items) == 1 and isinstance(text := items[0], str):
            self._items.append(EasyInputMessageParam(role='user', content=text))
            return
        content: ResponseInputMessageContentListParam = []
        for item in items:
            if isinstance(item, str):
                content.append(ResponseInputTextParam(type='input_text', text=item))
            else:
                _, image, detail = self._image()
                content.append(ResponseInputImageParam(type='input_image', image_url=image.data_uri, detail=detail))
        self._items.append(EasyInputMessageParam(role='user', content=content))

    def _message(self, role: Literal['assistant', 'system', 'developer'], content: str) -> None:
        self._items.append(EasyInputMessageParam(role=role, content=content))

    def _request_part(self, part: ModelRequestPart, calls: set[str]) -> None:
        if isinstance(part, SystemPromptPart):
            self._message('system', part.content)
        elif isinstance(part, UserPromptPart):
            prompt: list[str | MultiModalContent] = []
            for item in [part.content] if isinstance(part.content, str) else part.content:
                if isinstance(item, str):
                    prompt.append(item)
                elif isinstance(item, TextContent):
                    prompt.append(item.content)
                elif isinstance(item, CachePoint):
                    pass  # A marker for models that cache a prompt prefix; there is nothing in it to send.
                else:
                    prompt.append(item)
            self._user(prompt)
        elif isinstance(part, ToolReturnPart):
            self._tool_return(part, calls)
        elif isinstance(part, RetryFeedbackPart):
            # The application speaking, after the latest request: as a user message it would read as a newer
            # request, the one under judgement.
            self._message('developer', part.model_response())
        # TODO(v3): remove `RetryPromptPart`
        elif isinstance(part, RetryPromptPart):  # pyright: ignore[reportDeprecated]
            # A direct `request()` skips `prepare_messages`; send the deprecated part as what it always meant.
            self._request_part(_translate_legacy_retry_part(part), calls)
        elif isinstance(part, (InstructionBaselinePart, InstructionDeltaPart)):  # pragma: no cover
            raise _unprojected_instruction_delta_error()
        elif isinstance(part, _UnpreparedPart):  # pragma: no cover
            raise _unprepared_part_error(part)
        elif isinstance(part, SpeechPart):  # pragma: no cover
            raise _unconverted_speech_part_error()
        else:
            assert_never(part)

    def _tool_return(self, part: BaseToolReturnPart, calls: set[str]) -> None:
        if part.tool_call_id in calls:
            output = self._output(part) if part.files else part.model_response_str()
            self._items.append(
                FunctionCallOutput(type='function_call_output', call_id=part.tool_call_id, output=output)
            )
        else:
            # A trimmed or compacted history can keep a result without its call, which the API refuses natively. It
            # goes on the user channel instead, framed as the tool's so it doesn't read as the user's own words.
            framed: list[str | MultiModalContent] = [
                open_tag(TOOL_RESULT_TAG, tool_name=part.tool_name, tool_call_id=part.tool_call_id),
                *(neutralize_tags(item) if isinstance(item, str) else item for item in part.content_items(mode='str')),
                close_tag(TOOL_RESULT_TAG),
            ]
            self._user(framed)

    def _response_part(self, part: ModelResponsePart, calls: set[str]) -> None:
        if isinstance(part, TextPart):
            self._message('assistant', part.content)
        elif isinstance(part, ThinkingPart):
            # Thinking a provider only returned encrypted, as a `signature` with no text, has nothing to show.
            if part.content:
                self._message('assistant', render_replayed_thinking(part.content, self._profile)[0])
        elif isinstance(part, ToolCallPart | NativeToolCallPart):
            self._items.append(
                ResponseFunctionToolCallParam(
                    type='function_call',
                    call_id=part.tool_call_id,
                    name=part.tool_name,
                    arguments=part.args_as_json_str(),
                )
            )
        elif isinstance(part, NativeToolReturnPart):
            self._tool_return(part, calls)
        elif isinstance(part, CompactionPart):
            if part.content:
                self._message('assistant', part.content)
        elif isinstance(part, FilePart):
            # An assistant message takes only text, so the file goes in a user message after the reference to it.
            index, image, detail = self._image()
            self._message('assistant', f'<image {index}>')
            self._items.append(
                EasyInputMessageParam(
                    role='user',
                    content=[
                        ResponseInputTextParam(type='input_text', text=f'<image {index}>:'),
                        ResponseInputImageParam(type='input_image', image_url=image.data_uri, detail=detail),
                    ],
                )
            )
        elif isinstance(part, SpeechPart):  # pragma: no cover
            raise _unconverted_speech_part_error()
        else:
            assert_never(part)


def _text(value: JsonValue) -> str:
    """A protocol value as the text the API takes: a string as it is, anything else as JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _api_error(e: APIError, model_name: str) -> ModelAPIError:
    """The model error for an `openai` SDK error, in the error category its status and code imply."""
    if isinstance(e, APIStatusError) and e.status_code >= 400:
        return ModelHTTPError.for_category(
            _model_errors.openai_compatible_category(e.status_code, e.code, e.type, e.message),
            status_code=e.status_code,
            model_name=model_name,
            body=e.body,
            headers=dict(e.response.headers),
            provider_error_code=e.code,
            provider_error_type=e.type,
        )
    if isinstance(e, APIConnectionError):
        return _model_errors.connection_error(
            model_name, 'OpenAI Decisions request failed', e, timeout=isinstance(e, APITimeoutError)
        )
    return ModelAPIError(model_name=model_name, message='OpenAI Decisions request failed')


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
        return ScoreAnswer(
            score=answer.score,
            confidence=answer.confidence,
            probabilities=probabilities,
            legend={level.value: level.label for level in answer.probabilities},
        )
    else:
        assert_never(answer)


def _answer_fits(question: DecisionQuestion, answer: DecisionAnswer) -> bool:
    """Whether an OpenAI Decisions answer matches its question and its probabilities are valid.

    A pick-one or a score gives a probability for exactly the options or levels offered, every probability and
    confidence is from 0 to 1, the probabilities sum to one within rounding, and a score is within the rubric and
    one its rounded probabilities can produce.
    """
    # Each range check is a chained comparison, which is false for NaN.
    if isinstance(question, NoulQuestion):
        return isinstance(answer, NoulAnswer) and 0 <= answer.noul <= 1
    elif isinstance(question, ChoiceQuestion):
        if not (
            isinstance(answer, ChoiceAnswer)
            and answer.choice in question.criteria
            and answer.probabilities.keys() == question.criteria.keys()
            and all(0 <= p <= 1 for p in (answer.confidence, *answer.probabilities.values()))
            and sum(answer.probabilities.values()) > 0
        ):
            return False
        lower, upper = _probability_bounds(answer.probabilities.values())
        # A small tolerance accommodates floating-point normalization.
        return sum(lower, Decimal(0)) <= Decimal('1.000001') and sum(upper, Decimal(0)) >= Decimal('0.999999')
    elif isinstance(question, ScoreQuestion):
        if not (
            isinstance(answer, ScoreAnswer)
            and answer.probabilities.keys() == set(range(len(question.criteria)))
            and 0 <= answer.score <= len(question.criteria) - 1
            and all(0 <= p <= 1 for p in (answer.confidence, *answer.probabilities.values()))
            and sum(answer.probabilities.values()) > 0
        ):
            return False
        # A displayed score and its probabilities may each be rounded. Check whether any distribution
        # within their rounding intervals could produce that score, using at least two decimals.
        probabilities: list[float] = [answer.probabilities[level] for level in range(len(question.criteria))]
        lower, upper = _probability_bounds(probabilities)
        return _score_fits(lower, upper, answer.score)
    else:
        assert_never(question)
