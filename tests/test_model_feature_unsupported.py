"""A model whose profile can't do what a request asks raises `ModelFeatureUnsupportedError` before sending it.

These are offline tests with `FunctionModel`, plus an `OpenAIChatModel` on a mock client for a forced `tool_choice`,
rather than recorded requests: the check refuses the request before anything is sent, so there's no request to record,
and what they pin is which model the fallback moves to and that the incapable one is never called.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    BinaryImage,
    FilePart,
    ModelMessage,
    ModelResponse,
    NativeOutput,
    TextPart,
    ToolCallPart,
    ToolOutput,
)
from pydantic_ai.capabilities import ModelFallback, RetryOnModelErrors
from pydantic_ai.exceptions import (
    FallbackExceptionGroup,
    ModelFeature,
    ModelFeatureUnsupportedError,
    ModelRequestRejectedError,
    UserError,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import OutputSpec
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings

from .conftest import try_import

with try_import() as openai_available:
    from openai.types.chat import ChatCompletionMessage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAI, completion_message, get_mock_chat_completion_kwargs

# The forced `tool_choice` case sends to a mock OpenAI client, which never reaches the network.
pytestmark = pytest.mark.usefixtures('allow_model_requests')


class City(BaseModel):
    city: str


def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """What the capable model answers, in whichever output mode the request asks for."""
    params = info.model_request_parameters
    if params.output_tools:
        return ModelResponse(parts=[ToolCallPart(params.output_tools[0].name, {'city': 'Paris'})])
    if params.allow_image_output:
        return ModelResponse(parts=[FilePart(content=BinaryImage(data=b'png', media_type='image/png'))])
    if params.output_object is not None:
        return ModelResponse(parts=[TextPart('{"city": "Paris"}')])
    return ModelResponse(parts=[TextPart('Paris')])


@dataclass
class Incapable:
    """A model that can't serve the case's request, and the requests that reached it."""

    model: Model
    calls: Callable[[], Sequence[object]]


def incapable_function_model(profile: ModelProfile) -> Incapable:
    calls: list[ModelMessage] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
        calls.extend(messages)
        return answer(messages, info)

    return Incapable(FunctionModel(respond, model_name='incapable', profile=profile), lambda: calls)


def incapable_forcing_model() -> Incapable:
    mock_client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='unused', role='assistant')))
    model = OpenAIChatModel(
        'incapable',
        provider=OpenAIProvider(openai_client=mock_client),
        profile=OpenAIModelProfile(supports_forced_tool_choice=False),
    )
    return Incapable(model, lambda: get_mock_chat_completion_kwargs(mock_client))


@dataclass
class Case:
    feature: ModelFeature
    output_type: OutputSpec[object]
    incapable: Callable[[], Incapable]
    message: str
    output: object
    capable_profile: ModelProfile | None = None
    model_settings: ModelSettings = field(default_factory=lambda: ModelSettings())


CASES = [
    pytest.param(
        Case(
            feature='native_output',
            output_type=NativeOutput(City),
            incapable=lambda: incapable_function_model(ModelProfile(supports_json_schema_output=False)),
            capable_profile=ModelProfile(supports_json_schema_output=True),
            message='Native structured output is not supported by this model.',
            output=City(city='Paris'),
        ),
        id='native_output',
    ),
    pytest.param(
        Case(
            feature='tool_output',
            output_type=ToolOutput(City),
            incapable=lambda: incapable_function_model(ModelProfile(supports_tools=False)),
            message='Tool output is not supported by this model.',
            output=City(city='Paris'),
        ),
        id='tool_output',
    ),
    pytest.param(
        Case(
            # An `output_type` that doesn't pick a mode resolves to the model's default, here Tool Output.
            feature='tool_output',
            output_type=City,
            incapable=lambda: incapable_function_model(ModelProfile(supports_tools=False)),
            message='Tool output is not supported by this model.',
            output=City(city='Paris'),
        ),
        id='inferred_tool_output',
    ),
    pytest.param(
        Case(
            feature='text_output',
            output_type=str,
            incapable=lambda: incapable_function_model(ModelProfile(supports_text_output=False)),
            message=(
                'Text output is not supported by this model. Give the agent one structured `output_type`, '
                'such as a `BaseModel`, without `str`, `NativeOutput` or `PromptedOutput`.'
            ),
            output='Paris',
        ),
        id='text_output',
    ),
    pytest.param(
        Case(
            feature='image_output',
            output_type=BinaryImage,
            incapable=lambda: incapable_function_model(ModelProfile(supports_image_output=False)),
            capable_profile=ModelProfile(supports_image_output=True),
            message='Image output is not supported by this model.',
            output=BinaryImage(data=b'png', media_type='image/png'),
        ),
        id='image_output',
    ),
    pytest.param(
        Case(
            feature='tool_choice',
            output_type=ToolOutput(City, name='final_result'),
            incapable=incapable_forcing_model,
            model_settings=ModelSettings(tool_choice=['final_result']),
            message=(
                "tool_choice=['final_result'] is not supported by model 'incapable'. "
                "This model does not support forcing tool use. Use `tool_choice='auto'` instead."
            ),
            output=City(city='Paris'),
        ),
        id='tool_choice',
        marks=pytest.mark.skipif(not openai_available(), reason='openai not installed'),
    ),
]


def capable(case: Case) -> FunctionModel:
    return FunctionModel(answer, model_name='capable', profile=case.capable_profile)


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('chain', ['ModelFallback', 'FallbackModel'])
async def test_fallback_moves_past_a_model_that_cant_serve_the_request(case: Case, chain: str):
    """Both fallbacks move on to the next model without sending the request to the one that can't serve it."""
    incapable = case.incapable()
    if chain == 'ModelFallback':
        agent = Agent(incapable.model, output_type=case.output_type, capabilities=[ModelFallback(capable(case))])
    else:
        agent = Agent(FallbackModel(incapable.model, capable(case)), output_type=case.output_type)

    result = await agent.run('Where?', model_settings=case.model_settings)

    assert result.output == case.output
    assert result.response.model_name == 'capable'
    [attempt] = result.response.failed_attempts or []
    assert (attempt.model_name, attempt.outcome, attempt.error) == (
        'incapable',
        'error',
        f'ModelFeatureUnsupportedError: {case.message}',
    )
    assert incapable.calls() == []


@pytest.mark.parametrize('case', CASES)
async def test_without_a_fallback_the_error_is_raised(case: Case):
    """The error keeps the message it had as a plain `UserError`, and is also a model error naming the feature."""
    incapable = case.incapable()
    agent = Agent(incapable.model, output_type=case.output_type)

    with pytest.raises(ModelFeatureUnsupportedError) as exc_info:
        await agent.run('Where?', model_settings=case.model_settings)

    error = exc_info.value
    assert isinstance(error, UserError)
    assert isinstance(error, ModelRequestRejectedError)
    assert str(error) == error.message == case.message
    assert (error.model_name, error.feature) == ('incapable', case.feature)
    assert incapable.calls() == []


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('chain', ['ModelFallback', 'FallbackModel'])
async def test_a_chain_of_models_that_cant_serve_the_request_raises_their_errors(case: Case, chain: str):
    first, second = case.incapable(), case.incapable()
    if chain == 'ModelFallback':
        agent = Agent(first.model, output_type=case.output_type, capabilities=[ModelFallback(second.model)])
    else:
        agent = Agent(FallbackModel(first.model, second.model), output_type=case.output_type)

    with pytest.raises(FallbackExceptionGroup) as exc_info:
        await agent.run('Where?', model_settings=case.model_settings)

    errors = exc_info.value.exceptions
    assert len(errors) == 2
    assert all(isinstance(error, ModelFeatureUnsupportedError) and error.feature == case.feature for error in errors)
    assert first.calls() == [] and second.calls() == []


class CountingFunctionModel(FunctionModel):
    """A `FunctionModel` that counts the requests made to it, including ones it refuses before calling its function."""

    requests: int = 0

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.requests += 1
        return await super().request(messages, model_settings, model_request_parameters)


async def test_retry_on_model_errors_does_not_retry_it():
    """Asking the same model again can't make it support the feature, so `RetryOnModelErrors` raises it at once."""
    model = CountingFunctionModel(answer, model_name='incapable', profile=ModelProfile(supports_tools=False))
    agent = Agent(model, output_type=ToolOutput(City), capabilities=[RetryOnModelErrors(max_attempts=3)])

    with pytest.raises(ModelFeatureUnsupportedError, match=r'^Tool output is not supported by this model\.$'):
        await agent.run('Where?')
    assert model.requests == 1


# --- The probes the agent graph makes with `prepare_request`, which catch the `UserError` it raises ---


class SwitchingProfileModel(FunctionModel):
    """A model that supports Native Output on the run's first request and only Tool Output after it.

    The kept-mode probe sends the second request the first one's Native Output, which the model's check refuses.
    """

    later_requests: bool = False

    @property
    def profile(self) -> ModelProfile:  # pyright: ignore[reportIncompatibleVariableOverride]
        if self.later_requests:
            return ModelProfile(supports_json_schema_output=False, default_structured_output_mode='tool')
        return ModelProfile(supports_json_schema_output=True, default_structured_output_mode='native')


@pytest.mark.moves_cache_prefix(
    reason='the second request sends the output tool the first one, in Native Output, did not'
)
async def test_a_kept_output_mode_the_model_refuses_resolves_again():
    """When the output mode the run kept from its first request is refused, the request resolves its own mode."""
    modes: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        params = info.model_request_parameters
        modes.append(params.output_mode)
        if len(modes) == 1:
            return ModelResponse(parts=[ToolCallPart('switch', {})])
        return answer(messages, info)

    model = SwitchingProfileModel(respond, model_name='switching')
    agent = Agent(model, output_type=City)

    @agent.tool_plain
    def switch() -> str:
        model.later_requests = True
        return 'switched'

    result = await agent.run('Where?')

    assert result.output == City(city='Paris')
    assert modes == ['native', 'tool']
