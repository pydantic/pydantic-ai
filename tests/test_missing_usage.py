"""Responses whose provider omitted usage are counted in `unmeasured_requests`, unlike ones that reported zero.

The adapter and agent tests use mocked SDK clients rather than cassettes: a response without usage is
a provider malfunction no recording reliably reproduces.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import httpx2
import pytest
from pydantic import TypeAdapter

from pydantic_ai import Agent, UsageLimitUnavailableWarning
from pydantic_ai._genai_prices import best_effort_price
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from ._inline_snapshot import snapshot
from .conftest import try_import

with try_import() as openai_imports:
    from openai import APIConnectionError
    from openai.types import chat, responses
    from openai.types.chat.chat_completion_chunk import Choice as OpenAIChunkChoice, ChoiceDelta as OpenAIChoiceDelta
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
    from openai.types.completion_usage import CompletionUsage
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText, ResponseUsage

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import MockOpenAI, MockOpenAIResponses, completion_message, response_message

with try_import() as groq_imports:
    from groq.types import chat as groq_chat
    from groq.types.chat.chat_completion import Choice as GroqChoice
    from groq.types.chat.chat_completion_message import ChatCompletionMessage as GroqChatCompletionMessage
    from groq.types.completion_usage import CompletionUsage as GroqCompletionUsage

    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

    from .models.test_groq import MockGroq

with try_import() as mistral_imports:
    from mistralai.client.models import (
        AssistantMessage as MistralAssistantMessage,
        ChatCompletionChoice as MistralChatCompletionChoice,
        ChatCompletionResponse as MistralChatCompletionResponse,
        CompletionChunk as MistralCompletionChunk,
        CompletionEvent as MistralCompletionEvent,
        CompletionResponseStreamChoice as MistralCompletionResponseStreamChoice,
        DeltaMessage as MistralDeltaMessage,
        UsageInfo as MistralUsageInfo,
    )

    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

    from .models.test_mistral import MockMistralAI

with try_import() as huggingface_imports:
    from huggingface_hub import (
        ChatCompletionOutput,
        ChatCompletionOutputComplete,
        ChatCompletionOutputMessage,
        ChatCompletionOutputUsage,
        ChatCompletionStreamOutput,
        ChatCompletionStreamOutputChoice,
        ChatCompletionStreamOutputDelta,
        ChatCompletionStreamOutputUsage,
    )

    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

    from .models.test_huggingface import MockHuggingFace

with try_import() as google_imports:
    from google.genai.types import (
        Candidate,
        Content,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
        Part,
    )

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as cohere_imports:
    import cohere

    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

    from .models.test_cohere import MockAsyncClientV2

with try_import() as bedrock_imports:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider


def test_unmeasured_requests_accumulate_and_subtract_exactly() -> None:
    before = RunUsage(requests=1, input_tokens=20, unmeasured_requests=1)
    after = before + RunUsage(requests=1, input_tokens=200, output_tokens=20, unmeasured_requests=2)

    assert after == snapshot(RunUsage(requests=2, input_tokens=220, output_tokens=20, unmeasured_requests=3))
    assert (after - before) == RunUsage(requests=1, input_tokens=200, output_tokens=20, unmeasured_requests=2)
    assert RequestUsage(unmeasured_requests=1) + RequestUsage(unmeasured_requests=2) == RequestUsage(
        unmeasured_requests=3
    )


def test_unmeasured_requests_roundtrip_and_default_is_omitted() -> None:
    request_adapter = TypeAdapter(RequestUsage)
    run_adapter = TypeAdapter(RunUsage)

    assert 'unmeasured_requests' not in request_adapter.dump_python(RequestUsage())
    assert 'unmeasured_requests' not in run_adapter.dump_python(RunUsage())

    usage = RunUsage(requests=2, input_tokens=20, unmeasured_requests=1)
    serialized = run_adapter.dump_json(usage)
    assert b'"unmeasured_requests":1' in serialized
    assert run_adapter.validate_json(serialized) == usage

    default_messages: list[ModelMessage] = [ModelResponse(parts=[], usage=RequestUsage())]
    default_serialized_messages = ModelMessagesTypeAdapter.dump_json(default_messages)
    assert b'"unmeasured_requests"' not in default_serialized_messages

    messages: list[ModelMessage] = [ModelResponse(parts=[], usage=RequestUsage(unmeasured_requests=2))]
    serialized_messages = ModelMessagesTypeAdapter.dump_json(messages)
    restored_messages = ModelMessagesTypeAdapter.validate_json(serialized_messages)
    assert isinstance(restored_messages[0], ModelResponse)
    assert restored_messages[0].usage.unmeasured_requests == 2


def test_unmeasured_requests_are_not_values_or_token_details() -> None:
    usage = RunUsage(unmeasured_requests=1)

    assert usage.has_values() is False
    assert 'gen_ai.usage.details.unmeasured_requests' not in usage.opentelemetry_attributes()


@pytest.mark.parametrize('usage_type', [RequestUsage, RunUsage], ids=['request', 'run'])
def test_missing_usage_is_not_priced_as_zero(usage_type: type[RequestUsage]) -> None:
    usage = usage_type(unmeasured_requests=1)

    assert best_effort_price(usage, model_name='gpt-4o', provider_name='openai') is None
    assert usage.unmeasured_requests == 1


@pytest.mark.parametrize('usage_type', [RequestUsage, RunUsage], ids=['request', 'run'])
def test_best_effort_price_keeps_known_partial_cost(usage_type: type[RequestUsage]) -> None:
    details = {'web_search_requests': 1}
    web_search_usage = usage_type(web_searches=1, unmeasured_requests=1, details=details)
    web_search_price = best_effort_price(web_search_usage, model_name='gpt-4o', provider_name='openai')
    assert web_search_price is not None
    assert web_search_price.total_price == Decimal('0.01')

    token_usage = usage_type(input_tokens=20, unmeasured_requests=1)
    token_price = best_effort_price(token_usage, model_name='gpt-4o', provider_name='openai')
    assert token_price is not None
    assert token_price.total_price > Decimal('0')

    assert web_search_usage.unmeasured_requests == 1
    assert web_search_usage.details == details


def test_response_cost_rejects_partial_cost() -> None:
    response = ModelResponse(
        parts=[],
        usage=RequestUsage(web_searches=1, unmeasured_requests=1, cost=Decimal('0.01')),
        model_name='gpt-4o',
        provider_name='openai',
    )

    with pytest.raises(ValueError, match='response that omitted usage information'):
        response.cost()


def test_known_zero_usage_remains_priceable() -> None:
    usage = RequestUsage(unmeasured_requests=0)

    price = best_effort_price(usage, model_name='gpt-4o', provider_name='openai')
    assert price is not None
    assert price.total_price == Decimal('0')


def test_configured_token_limit_warns_when_usage_is_unavailable() -> None:
    limits = UsageLimits(input_tokens_limit=100)
    usage = RunUsage(unmeasured_requests=1)

    with pytest.warns(UsageLimitUnavailableWarning, match='known token and cost totals are lower bounds'):
        limits.check_tokens(usage)


def test_configured_cost_limit_warns_when_usage_is_unavailable() -> None:
    """`check_tokens` owns the omitted-usage warning, so `check_cost` adds neither it nor a `CostNotFoundWarning`."""
    limits = UsageLimits(cost_limit=Decimal('0.01'))
    usage = RunUsage(unmeasured_requests=1)

    with pytest.warns(UsageLimitUnavailableWarning, match='known token and cost totals are lower bounds'):
        limits.check_tokens(usage)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        limits.check_cost(usage)


def test_unavailable_usage_warning_requires_token_or_cost_limit() -> None:
    usage = RunUsage(unmeasured_requests=1)
    limits_without_token_or_cost = [UsageLimits(), UsageLimits(request_limit=2), UsageLimits(tool_calls_limit=2)]

    with warnings.catch_warnings():
        warnings.simplefilter('error', UsageLimitUnavailableWarning)
        for limits in limits_without_token_or_cost:
            limits.check_tokens(usage)


async def test_usage_unavailable_warning_is_shown_once_per_run() -> None:
    """Every response omits usage, but the default warning filter shows the warning once, not after every response."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        part = ToolCallPart('step') if len(messages) < 7 else TextPart('done')
        return ModelResponse(parts=[part], usage=RequestUsage(unmeasured_requests=1))

    agent = Agent(FunctionModel(respond))

    @agent.tool_plain
    def step() -> str:
        return 'ok'

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('default', UsageLimitUnavailableWarning)
        result = await agent.run('hello', usage_limits=UsageLimits(request_limit=10, total_tokens_limit=1000))

    assert [item.category for item in caught] == [UsageLimitUnavailableWarning]
    assert result.usage == snapshot(RunUsage(requests=4, tool_calls=3, unmeasured_requests=4))


@pytest.mark.parametrize(
    'usage, expected_usage, expected_warnings',
    [
        pytest.param(
            RequestUsage(unmeasured_requests=1),
            snapshot(RunUsage(requests=1, unmeasured_requests=1)),
            1,
            id='explicitly-unmeasured',
        ),
        pytest.param(RequestUsage(), snapshot(RunUsage(output_tokens=1, input_tokens=51, requests=1)), 0, id='default'),
    ],
)
async def test_agent_function_model_preserves_explicit_missing_usage(
    usage: RequestUsage, expected_usage: RunUsage, expected_warnings: int
) -> None:
    """`FunctionModel` estimates usage it wasn't given, but keeps a response's explicit omission marker."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='world')], usage=usage)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', UsageLimitUnavailableWarning)
        result = await Agent(FunctionModel(respond)).run('hello', usage_limits=UsageLimits(input_tokens_limit=100))

    assert len(caught) == expected_warnings
    assert result.usage == expected_usage


# Every adapter maps a response that omitted usage to `unmeasured_requests=1`, and one that reported zero to plain
# zero usage. A streamed response only knows it omitted usage once it has completed.


@dataclass(frozen=True)
class Case:
    id: str
    model: Callable[[bool, pytest.MonkeyPatch], Model]
    """Builds a model whose mocked response omits usage when passed `True`, and reports zero usage otherwise."""
    stream: bool
    reported_zero_usage: RequestUsage = field(default_factory=RequestUsage)
    marks: tuple[pytest.MarkDecorator, ...] = ()


def openai_chat_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    usage = None if omit_usage else CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    completion = completion_message(ChatCompletionMessage(content='hello', role='assistant'), usage=usage)
    return OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock(completion)))


def openai_chunk(text: str | None, usage: CompletionUsage | None = None) -> chat.ChatCompletionChunk:
    choices = [OpenAIChunkChoice(index=0, delta=OpenAIChoiceDelta(content=text, role='assistant'))] if text else []
    return chat.ChatCompletionChunk(
        id='123', choices=choices, created=1704067200, model='gpt-4o', object='chat.completion.chunk', usage=usage
    )


def openai_chat_stream_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    usage = None if omit_usage else CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    stream = [openai_chunk('hello'), openai_chunk(None, usage)]
    return OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(stream)))


def openai_responses_usage(omit_usage: bool) -> ResponseUsage | None:
    return None if omit_usage else ResponseUsage.model_construct(input_tokens=0, output_tokens=0, total_tokens=0)


def openai_responses_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    message = ResponseOutputMessage(
        id='msg_001',
        content=[ResponseOutputText(text='hello', type='output_text', annotations=[])],
        role='assistant',
        status='completed',
        type='message',
    )
    response = response_message([message], usage=openai_responses_usage(omit_usage))
    return OpenAIResponsesModel(
        'gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock(response))
    )


def openai_responses_stream_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    response = response_message([])
    stream: list[responses.ResponseStreamEvent] = [
        responses.ResponseCreatedEvent(response=response, type='response.created', sequence_number=0),
        responses.ResponseCompletedEvent(
            response=response.model_copy(update={'status': 'completed', 'usage': openai_responses_usage(omit_usage)}),
            type='response.completed',
            sequence_number=1,
        ),
    ]
    return OpenAIResponsesModel(
        'gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock_stream(stream))
    )


def groq_usage(omit_usage: bool) -> GroqCompletionUsage | None:
    return None if omit_usage else GroqCompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def groq_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    completion = groq_chat.ChatCompletion(
        id='123',
        choices=[
            GroqChoice(
                finish_reason='stop', index=0, message=GroqChatCompletionMessage(content='hello', role='assistant')
            )
        ],
        created=1704067200,
        model='llama-3.3-70b-versatile',
        object='chat.completion',
        usage=groq_usage(omit_usage),
    )
    return GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=MockGroq.create_mock(completion)))


def groq_stream_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    usage = groq_usage(omit_usage)
    chunk = groq_chat.ChatCompletionChunk.model_validate(
        {
            'id': '123',
            'choices': [{'index': 0, 'delta': {'content': 'hello', 'role': 'assistant'}, 'finish_reason': 'stop'}],
            'created': 1704067200,
            'model': 'llama-3.3-70b-versatile',
            'object': 'chat.completion.chunk',
            'x_groq': usage and {'id': 'x', 'error': None, 'usage': usage.model_dump()},
        }
    )
    return GroqModel('llama-3.3-70b-versatile', provider=GroqProvider(groq_client=MockGroq.create_mock_stream([chunk])))


def mistral_usage(omit_usage: bool) -> MistralUsageInfo | None:
    return None if omit_usage else MistralUsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def mistral_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    completion = MistralChatCompletionResponse(
        id='123',
        choices=[
            MistralChatCompletionChoice(finish_reason='stop', index=0, message=MistralAssistantMessage(content='hello'))
        ],
        created=1704067200,
        model='mistral-large-latest',
        object='chat.completion',
        usage=MistralUsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    if omit_usage:
        completion = completion.model_copy(update={'usage': None})
    return MistralModel(
        'mistral-large-latest', provider=MistralProvider(mistral_client=MockMistralAI.create_mock(completion))
    )


def mistral_stream_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    event = MistralCompletionEvent(
        data=MistralCompletionChunk(
            id='123',
            choices=[
                MistralCompletionResponseStreamChoice(
                    index=0, delta=MistralDeltaMessage(content='hello', role='assistant'), finish_reason='stop'
                )
            ],
            created=1704067200,
            model='mistral-large-latest',
            object='chat.completion.chunk',
            usage=mistral_usage(omit_usage),
        )
    )
    return MistralModel(
        'mistral-large-latest', provider=MistralProvider(mistral_client=MockMistralAI.create_stream_mock([event]))
    )


def huggingface_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    completion = ChatCompletionOutput.parse_obj_as_instance(  # pyright: ignore[reportUnknownMemberType]
        {
            'id': '123',
            'choices': [
                ChatCompletionOutputComplete(
                    finish_reason='stop',
                    index=0,
                    message=ChatCompletionOutputMessage(role='assistant', content='hello'),
                )
            ],
            'created': 1704067200,
            'model': 'hf-model',
            'object': 'chat.completion',
            'usage': None
            if omit_usage
            else ChatCompletionOutputUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
        }
    )
    return HuggingFaceModel(
        'hf-model', provider=HuggingFaceProvider(hf_client=MockHuggingFace.create_mock(completion), api_key='x')
    )


def huggingface_stream_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    chunk = ChatCompletionStreamOutput(
        id='123',
        choices=[
            ChatCompletionStreamOutputChoice(
                index=0, delta=ChatCompletionStreamOutputDelta(role='assistant', content='hello')
            )
        ],
        created=1704067200,
        model='hf-model',
        system_fingerprint='fp',
        usage=None
        if omit_usage
        else ChatCompletionStreamOutputUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
    )
    return HuggingFaceModel(
        'hf-model', provider=HuggingFaceProvider(hf_client=MockHuggingFace.create_stream_mock([chunk]), api_key='x')
    )


def google_response(omit_usage: bool) -> GenerateContentResponse:
    return GenerateContentResponse(
        candidates=[Candidate(content=Content(parts=[Part(text='hello')], role='model'))],
        response_id='123',
        model_version='gemini-2.5-flash',
        usage_metadata=None
        if omit_usage
        else GenerateContentResponseUsageMetadata(prompt_token_count=0, candidates_token_count=0),
    )


def google_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='x'))

    async def generate_content(**_: Any) -> GenerateContentResponse:
        return google_response(omit_usage)

    monkeypatch.setattr(model.client.aio.models, 'generate_content', generate_content)
    return model


def google_stream_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='x'))

    async def chunks():
        yield google_response(omit_usage)

    async def generate_content_stream(**_: Any):
        return chunks()

    monkeypatch.setattr(model.client.aio.models, 'generate_content_stream', generate_content_stream)
    return model


def cohere_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    usage = None if omit_usage else cohere.Usage(tokens=cohere.UsageTokens(input_tokens=0, output_tokens=0))
    completion = cohere.ChatResponse(
        id='123',
        finish_reason='COMPLETE',
        message=cohere.AssistantMessageResponse(content=[cohere.TextAssistantMessageResponseContentItem(text='hello')]),
        usage=usage,
    )
    return CohereModel(
        'command-r7b-12-2024', provider=CohereProvider(cohere_client=MockAsyncClientV2.create_mock(completion))
    )


def bedrock_stream_model(omit_usage: bool, monkeypatch: pytest.MonkeyPatch) -> Model:
    """Bedrock's non-streamed `converse` response always carries usage, so only its stream can omit it."""
    model = BedrockConverseModel(
        'us.amazon.nova-micro-v1:0',
        provider=BedrockProvider(region_name='us-east-1', aws_access_key_id='x', aws_secret_access_key='x'),
    )
    events: list[dict[str, Any]] = [
        {'messageStart': {'role': 'assistant'}},
        {'contentBlockDelta': {'contentBlockIndex': 0, 'delta': {'text': 'hello'}}},
        {'contentBlockStop': {'contentBlockIndex': 0}},
        {'messageStop': {'stopReason': 'end_turn'}},
    ]
    if not omit_usage:
        events.append({'metadata': {'usage': {'inputTokens': 0, 'outputTokens': 0, 'totalTokens': 0}}})

    def converse_stream(**_: Any) -> dict[str, Any]:
        # A generator, as it can be closed like botocore's `EventStream`.
        return {'stream': (event for event in events), 'ResponseMetadata': {'RequestId': '123'}}

    monkeypatch.setattr(model.client, 'converse_stream', converse_stream)
    return model


def skip_unless(imports_successful: Callable[[], bool], package: str) -> tuple[pytest.MarkDecorator, ...]:
    return (pytest.mark.skipif(not imports_successful(), reason=f'{package} not installed'),)


CASES = [
    Case('openai-chat', openai_chat_model, stream=False, marks=skip_unless(openai_imports, 'openai')),
    Case('openai-chat-stream', openai_chat_stream_model, stream=True, marks=skip_unless(openai_imports, 'openai')),
    Case(
        'openai-responses',
        openai_responses_model,
        stream=False,
        reported_zero_usage=snapshot(RequestUsage(details={'reasoning_tokens': 0})),
        marks=skip_unless(openai_imports, 'openai'),
    ),
    Case(
        'openai-responses-stream',
        openai_responses_stream_model,
        stream=True,
        reported_zero_usage=snapshot(RequestUsage(details={'reasoning_tokens': 0})),
        marks=skip_unless(openai_imports, 'openai'),
    ),
    Case('groq', groq_model, stream=False, marks=skip_unless(groq_imports, 'groq')),
    Case('groq-stream', groq_stream_model, stream=True, marks=skip_unless(groq_imports, 'groq')),
    Case('mistral', mistral_model, stream=False, marks=skip_unless(mistral_imports, 'mistral')),
    Case('mistral-stream', mistral_stream_model, stream=True, marks=skip_unless(mistral_imports, 'mistral')),
    Case('huggingface', huggingface_model, stream=False, marks=skip_unless(huggingface_imports, 'huggingface_hub')),
    Case(
        'huggingface-stream',
        huggingface_stream_model,
        stream=True,
        marks=skip_unless(huggingface_imports, 'huggingface_hub'),
    ),
    Case('google', google_model, stream=False, marks=skip_unless(google_imports, 'google-genai')),
    Case('google-stream', google_stream_model, stream=True, marks=skip_unless(google_imports, 'google-genai')),
    Case('cohere', cohere_model, stream=False, marks=skip_unless(cohere_imports, 'cohere')),
    Case('bedrock-stream', bedrock_stream_model, stream=True, marks=skip_unless(bedrock_imports, 'boto3')),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
@pytest.mark.parametrize('omit_usage', [True, False], ids=['omitted', 'reported-zero'])
async def test_adapter_distinguishes_omitted_usage_from_reported_zero(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch, case: Case, omit_usage: bool
) -> None:
    model = case.model(omit_usage, monkeypatch)
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('hello')]

    if case.stream:
        async with model_request_stream(model, messages) as stream:
            async for _ in stream:
                pass
        response = stream.get()
    else:
        response = await model_request(model, messages)

    assert response.usage == (RequestUsage(unmeasured_requests=1) if omit_usage else case.reported_zero_usage)


# Agent-level behavior, driven through the OpenAI Chat adapter.


def openai_completion(usage: CompletionUsage | None = None) -> chat.ChatCompletion:
    return completion_message(ChatCompletionMessage(content='hello', role='assistant'), usage=usage).model_copy(
        update={'model': 'gpt-4o'}
    )


def openai_model(completion: chat.ChatCompletion | list[chat.ChatCompletion] | APIConnectionError) -> OpenAIChatModel:
    return OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock(completion)))


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
@pytest.mark.parametrize(
    'usage_limits',
    [UsageLimits(input_tokens_limit=100), UsageLimits(cost_limit=Decimal('0.01'))],
    ids=['token-limit', 'cost-limit'],
)
async def test_limits_warn_when_response_omits_usage(allow_model_requests: None, usage_limits: UsageLimits) -> None:
    """The model is priceable, but omitted usage must leave the cost unknown rather than priced as zero."""
    with pytest.warns(UsageLimitUnavailableWarning, match='Some responses omitted usage information'):
        result = await Agent(openai_model(openai_completion())).run('hello', usage_limits=usage_limits)

    assert result.usage == snapshot(RunUsage(requests=1, unmeasured_requests=1))


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_omitted_usage_does_not_warn_without_token_or_cost_limit(allow_model_requests: None) -> None:
    """The default `UsageLimits` only caps requests, which pydantic-ai counts itself, so there is nothing to warn about."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', UsageLimitUnavailableWarning)
        result = await Agent(openai_model(openai_completion())).run('hello')

    assert result.usage == snapshot(RunUsage(requests=1, unmeasured_requests=1))


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
@pytest.mark.parametrize(
    'missing_first, expected_warnings', [(True, 2), (False, 1)], ids=['missing-then-known', 'known-then-missing']
)
async def test_mixed_known_and_missing_usage_keeps_known_totals(
    allow_model_requests: None, missing_first: bool, expected_warnings: int
) -> None:
    """Two fully-reported requests would exceed the limit; with one omitted, only the known totals count."""
    known_usage = CompletionUsage(prompt_tokens=100, completion_tokens=100, total_tokens=200)
    tool_call = ChatCompletionMessageToolCall(
        id='call_1', function=Function(arguments='{}', name='get_value'), type='function'
    )
    tool_response = completion_message(
        ChatCompletionMessage(content=None, role='assistant', tool_calls=[tool_call]),
        usage=None if missing_first else known_usage,
    )
    final_response = completion_message(
        ChatCompletionMessage(content='done', role='assistant'), usage=known_usage if missing_first else None
    )
    agent = Agent(openai_model([tool_response, final_response]))

    @agent.tool_plain
    def get_value() -> str:
        return 'value'

    with pytest.warns(UsageLimitUnavailableWarning, match='Some responses omitted usage information') as caught:
        result = await agent.run('hello', usage_limits=UsageLimits(total_tokens_limit=300))

    assert len(caught) == expected_warnings
    assert result.output == 'done'
    assert result.usage == snapshot(
        RunUsage(requests=2, input_tokens=100, output_tokens=100, tool_calls=1, unmeasured_requests=1)
    )


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_usage_unavailable_warning_can_be_promoted_to_error(allow_model_requests: None) -> None:
    """A promoted warning escapes the run instead of being recovered like a model request error."""

    @dataclass
    class RecoveringCapability(AbstractCapability[None]):
        recovery_called: bool = field(default=False, init=False)

        async def on_model_request_error(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            self.recovery_called = True
            return ModelResponse(parts=[TextPart(content='recovered')])

    # The capability does recover a genuine model request error.
    connection_error = APIConnectionError(
        message='test connection failure', request=httpx2.Request('POST', 'https://api.openai.com/v1')
    )
    recovering = RecoveringCapability()
    recovered = await Agent[None, str](
        openai_model(connection_error), capabilities=[recovering], deps_type=type(None)
    ).run('hello')
    assert recovered.output == 'recovered'
    assert recovering.recovery_called is True

    capability = RecoveringCapability()
    agent = Agent[None, str](openai_model(openai_completion()), capabilities=[capability], deps_type=type(None))
    with warnings.catch_warnings():
        warnings.simplefilter('error', UsageLimitUnavailableWarning)
        with pytest.raises(UsageLimitUnavailableWarning, match='Some responses omitted usage information'):
            await agent.run('hello', usage_limits=UsageLimits(input_tokens_limit=100))

    assert capability.recovery_called is False


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
@pytest.mark.parametrize(
    'final_usage, expected_usage_per_step, expected_warnings',
    [
        pytest.param(
            CompletionUsage(prompt_tokens=6, completion_tokens=3, total_tokens=9),
            snapshot(
                [
                    RunUsage(requests=1, cost=Decimal('0.00')),
                    RunUsage(requests=1, cost=Decimal('0.00')),
                    RunUsage(requests=1, output_tokens=3, input_tokens=6, cost=Decimal('0.000045')),
                    RunUsage(requests=1, output_tokens=3, input_tokens=6, cost=Decimal('0.000045')),
                ]
            ),
            0,
            id='reported',
        ),
        pytest.param(
            CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            snapshot(
                [
                    RunUsage(requests=1, cost=Decimal('0.00')),
                    RunUsage(requests=1, cost=Decimal('0.00')),
                    RunUsage(requests=1, cost=Decimal('0.00')),
                    RunUsage(requests=1, cost=Decimal('0.00')),
                ]
            ),
            0,
            id='reported-zero',
        ),
        pytest.param(
            None,
            snapshot(
                [
                    RunUsage(requests=1, cost=Decimal('0.00')),
                    RunUsage(requests=1, cost=Decimal('0.00')),
                    RunUsage(requests=1, cost=Decimal('0.00')),
                    RunUsage(requests=1, unmeasured_requests=1),
                ]
            ),
            1,
            id='omitted',
        ),
    ],
)
async def test_stream_marks_omitted_usage_only_once_complete(
    allow_model_requests: None,
    final_usage: CompletionUsage | None,
    expected_usage_per_step: list[RunUsage],
    expected_warnings: int,
) -> None:
    """OpenAI Chat streams report usage only on their last chunk, so earlier chunks without it mean nothing yet.

    Usage read mid-stream carries no omission marker; only a stream that completes without reporting usage gets one.
    """
    stream = [openai_chunk('hello '), openai_chunk('world'), openai_chunk(None, final_usage)]
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream(stream)))

    usage_per_step: list[RunUsage] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', UsageLimitUnavailableWarning)
        async with Agent(model).run_stream('hello', usage_limits=UsageLimits(input_tokens_limit=100)) as result:
            async for _ in result.stream_response(debounce_by=None):
                usage_per_step.append(result.usage)

    assert len(caught) == expected_warnings
    assert usage_per_step == expected_usage_per_step
    assert result.usage == usage_per_step[-1]


# `FallbackModel`: the usage of a response it rejected counts in `RunUsage`, alongside omitted usage.


def openai_text_model(content: str, *, reports_usage: bool) -> OpenAIChatModel:
    usage = CompletionUsage(prompt_tokens=6, completion_tokens=3, total_tokens=9) if reports_usage else None
    completion = completion_message(ChatCompletionMessage(content=content, role='assistant'), usage=usage)
    return openai_model(completion.model_copy(update={'model': 'gpt-4o'}))


def reject_text(response: ModelResponse) -> bool:
    return response.text == 'rejected'


@dataclass(kw_only=True)
class CostReportingModel(TestModel):
    """Reports explicit token and cost usage, including zero, without `FunctionModel`'s token estimation."""

    reported_cost: Decimal
    input_tokens: int = 0
    output_tokens: int = 0

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await super().request(messages, model_settings, model_request_parameters)
        response.usage = RequestUsage(
            input_tokens=self.input_tokens, output_tokens=self.output_tokens, cost=self.reported_cost
        )
        return response


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
@pytest.mark.parametrize('token_limit', [None, 100], ids=['cost-limit', 'cost-and-token-limits'])
async def test_fallback_counts_rejected_usage_alongside_omitted_usage(
    allow_model_requests: None, token_limit: int | None
) -> None:
    model = FallbackModel(
        openai_text_model('rejected', reports_usage=True),
        openai_text_model('accepted', reports_usage=False),
        fallback_on=reject_text,
    )
    limits = UsageLimits(cost_limit=Decimal('1'), input_tokens_limit=token_limit)

    with pytest.warns(UsageLimitUnavailableWarning, match='Some responses omitted usage information') as caught:
        result = await Agent(model).run('hello', usage_limits=limits)

    assert len(caught) == 1
    assert result.output == 'accepted'
    assert result.usage == snapshot(
        RunUsage(requests=1, input_tokens=6, output_tokens=3, cost=Decimal('0.000045'), unmeasured_requests=1)
    )


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_fallback_omitted_usage_without_limits_does_not_warn(allow_model_requests: None) -> None:
    model = FallbackModel(
        openai_text_model('rejected', reports_usage=True),
        openai_text_model('accepted', reports_usage=False),
        fallback_on=reject_text,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error', UsageLimitUnavailableWarning)
        result = await Agent(model).run('hello')

    assert result.output == 'accepted'
    assert result.usage == snapshot(
        RunUsage(requests=1, input_tokens=6, output_tokens=3, cost=Decimal('0.000045'), unmeasured_requests=1)
    )


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
@pytest.mark.parametrize('rejected_count', [1, 2], ids=['one-omitted-rejection', 'two-omitted-rejections'])
@pytest.mark.parametrize(
    'selected_model, expected_usage',
    [
        pytest.param(
            lambda: openai_text_model('accepted', reports_usage=True),
            RunUsage(requests=1, input_tokens=6, output_tokens=3, cost=Decimal('0.000045')),
            id='computed-cost',
        ),
        pytest.param(
            lambda: CostReportingModel(
                reported_cost=Decimal('0.1'), input_tokens=120, output_tokens=8, custom_output_text='accepted'
            ),
            RunUsage(requests=1, input_tokens=120, output_tokens=8, cost=Decimal('0.1')),
            id='explicit-cost',
        ),
    ],
)
async def test_fallback_counts_each_omitted_rejection(
    allow_model_requests: None, rejected_count: int, selected_model: Callable[[], Model], expected_usage: RunUsage
) -> None:
    rejected = [openai_text_model('rejected', reports_usage=False) for _ in range(rejected_count)]
    model = FallbackModel(rejected[0], *rejected[1:], selected_model(), fallback_on=reject_text)
    limits = UsageLimits(cost_limit=Decimal('1'), input_tokens_limit=200)

    with pytest.warns(UsageLimitUnavailableWarning, match='Some responses omitted usage information') as caught:
        result = await Agent(model).run('hello', usage_limits=limits)

    assert len(caught) == 1
    assert result.output == 'accepted'
    assert result.usage == expected_usage + RunUsage(unmeasured_requests=rejected_count)


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
@pytest.mark.parametrize('reported_cost', [Decimal('0'), Decimal('0.1')], ids=['zero', 'nonzero'])
@pytest.mark.parametrize('token_limit', [None, 100], ids=['cost-limit', 'cost-and-token-limits'])
@pytest.mark.parametrize(
    'fallback, rejected_usage',
    [
        pytest.param(False, RunUsage(), id='direct'),
        pytest.param(True, RunUsage(input_tokens=6, output_tokens=3, cost=Decimal('0.000045')), id='fallback'),
    ],
)
async def test_reported_cost_without_tokens_is_measured(
    allow_model_requests: None,
    reported_cost: Decimal,
    token_limit: int | None,
    fallback: bool,
    rejected_usage: RunUsage,
) -> None:
    """A response that reports only a cost, including zero, reported its usage: zero tokens, not omitted ones."""
    model: Model = CostReportingModel(reported_cost=reported_cost, custom_output_text='accepted')
    if fallback:
        model = FallbackModel(openai_text_model('rejected', reports_usage=True), model, fallback_on=reject_text)

    with warnings.catch_warnings():
        warnings.simplefilter('error', UsageLimitUnavailableWarning)
        result = await Agent(model).run(
            'hello', usage_limits=UsageLimits(cost_limit=Decimal('1'), input_tokens_limit=token_limit)
        )

    assert result.output == 'accepted'
    assert result.usage == rejected_usage + RunUsage(requests=1, cost=reported_cost)
