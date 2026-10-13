"""Tests for Meta's Muse models, recorded against `api.meta.ai`.

Each test pins one fact `MetaProvider.model_profile` encodes: the recordings show Meta accepting the request the
profile shapes, and `request_capture` shows the profile actually shaped it.
"""

from __future__ import annotations as _annotations

import warnings
from collections.abc import AsyncIterable
from typing import Literal

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, AgentStreamEvent, ModelSettings, NativeOutput, RunContext, WebSearchTool
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.messages import ModelResponse, NativeToolCallPart, TextPart, ThinkingPart, ToolCallPart

from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from openai.types import responses

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.meta import MetaProvider

    from .mock_openai import MockOpenAIResponses, get_mock_responses_kwargs, response_message

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]

# Muse reasons before every answer, so recordings need longer than httpx's 5 second default.
_TIMEOUT = 120


def _responses_model(meta_api_key: str, request_capture: RequestCapture) -> OpenAIResponsesModel:
    provider = MetaProvider(api_key=meta_api_key, http_client=request_capture.http_client(timeout=_TIMEOUT))
    return OpenAIResponsesModel('muse-spark-1.3', provider=provider)


class CityInfo(BaseModel):
    city: str
    country: str


async def test_meta_tool_call_replays_phase_and_encrypted_reasoning(
    allow_model_requests: None, meta_api_key: str, request_capture: RequestCapture
):
    """Muse's `commentary` preamble and encrypted reasoning are sent back on the request after the tool call.

    Without the phase, a replayed preamble reads as a final answer, and long runs start ending on preambles.
    """
    agent = Agent(
        _responses_model(meta_api_key, request_capture),
        instructions='Before calling a tool, tell the user in one short sentence what you are about to do.',
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'Sunny and 21C in {city}.'

    result = await agent.run('What is the weather in Paris?')
    assert result.output == snapshot('The weather in Paris is sunny and 21°C.')

    first_response = result.all_messages()[1]
    assert isinstance(first_response, ModelResponse)
    assert [
        (type(part).__name__, (part.provider_details or {}).get('phase') if isinstance(part, TextPart) else None)
        for part in first_response.parts
    ] == snapshot([('ThinkingPart', None), ('TextPart', 'commentary'), ('ToolCallPart', None)])

    first_request, second_request = request_capture.bodies('/responses')
    assert first_request['include'] == ['reasoning.encrypted_content']
    replayed = second_request['input']
    assert isinstance(replayed, list)
    assert [
        (item.get('type'), item.get('phase'), bool(item.get('encrypted_content')))
        for item in replayed
        if isinstance(item, dict) and item.get('type') in ('reasoning', 'message', 'function_call')
    ] == snapshot([('reasoning', None, True), ('message', 'commentary', False), ('function_call', None, False)])


async def test_meta_tool_output_falls_back_to_auto_tool_choice(
    allow_model_requests: None, meta_api_key: str, request_capture: RequestCapture
):
    """`api.meta.ai` rejects `tool_choice='required'`, so the output tool is offered with `tool_choice='auto'`."""
    agent = Agent(_responses_model(meta_api_key, request_capture), output_type=CityInfo)

    result = await agent.run('Which city is the Eiffel Tower in?')
    assert result.output == snapshot(CityInfo(city='Paris', country='France'))
    assert request_capture.body('/responses')['tool_choice'] == 'auto'


async def test_meta_native_output(allow_model_requests: None, meta_api_key: str, request_capture: RequestCapture):
    agent = Agent(_responses_model(meta_api_key, request_capture), output_type=NativeOutput(CityInfo))

    result = await agent.run('Which city is the Colosseum in?')
    assert result.output == snapshot(CityInfo(city='Rome', country='Italy'))
    text_format = request_capture.body('/responses')['text']
    assert isinstance(text_format, dict)
    assert text_format['format'] == snapshot(
        {
            'type': 'json_schema',
            'name': 'CityInfo',
            'schema': {
                'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                'required': ['city', 'country'],
                'type': 'object',
                'additionalProperties': False,
            },
            'strict': True,
        }
    )


async def test_meta_keeps_sampling_params_while_reasoning(
    allow_model_requests: None, meta_api_key: str, request_capture: RequestCapture
):
    """Muse accepts `temperature` while reasoning, so it isn't dropped the way it is for OpenAI reasoning models."""
    agent = Agent(_responses_model(meta_api_key, request_capture))

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = await agent.run('Reply with exactly: OK', model_settings=ModelSettings(temperature=0.2))
    assert result.output == snapshot('OK')
    assert request_capture.body('/responses')['temperature'] == 0.2


async def test_meta_thinking_false_keeps_reasoning_on(
    allow_model_requests: None, meta_api_key: str, request_capture: RequestCapture
):
    """Muse can't turn reasoning off (`effort='none'` is rejected), so `thinking=False` sends no effort at all."""
    agent = Agent(_responses_model(meta_api_key, request_capture))

    result = await agent.run('Reply with exactly: OK', model_settings=ModelSettings(thinking=False))
    assert result.output == snapshot('OK')
    assert 'reasoning' not in request_capture.body('/responses')
    assert any(isinstance(part, ThinkingPart) for part in result.all_messages()[-1].parts)


async def test_meta_unified_thinking_effort(
    allow_model_requests: None, meta_api_key: str, request_capture: RequestCapture
):
    agent = Agent(_responses_model(meta_api_key, request_capture))

    result = await agent.run(
        'Reply with exactly: OK',
        model_settings=OpenAIResponsesModelSettings(thinking='minimal', openai_reasoning_summary='auto'),
    )
    assert result.output == snapshot('OK')
    assert request_capture.body('/responses')['reasoning'] == snapshot({'effort': 'minimal', 'summary': 'auto'})


async def test_meta_web_search(allow_model_requests: None, meta_api_key: str, request_capture: RequestCapture):
    agent = Agent(_responses_model(meta_api_key, request_capture), capabilities=[NativeTool(WebSearchTool())])

    result = await agent.run('Search the web: what city hosted the 2024 Summer Olympics? Reply with just the city.')
    assert result.output == snapshot('Paris')
    assert request_capture.body('/responses')['tools'] == snapshot(
        [{'type': 'web_search', 'search_context_size': 'medium'}]
    )
    assert any(isinstance(part, NativeToolCallPart) for part in result.all_messages()[-1].parts)


async def test_meta_stream(allow_model_requests: None, meta_api_key: str, request_capture: RequestCapture):
    agent = Agent(_responses_model(meta_api_key, request_capture))

    async with agent.run_stream('Count from 1 to 5, separated by commas.') as result:
        chunks = [chunk async for chunk in result.stream_text(delta=True)]
    assert ''.join(chunks) == snapshot('1, 2, 3, 4, 5')


def _stream(items: list[responses.ResponseOutputItem]) -> list[responses.ResponseStreamEvent]:
    """Stream `items` the way `api.meta.ai` does: each item is added empty, then done complete."""
    response = response_message([]).model_copy(update={'model': 'muse-spark-1.3'})
    events: list[responses.ResponseStreamEvent] = [
        responses.ResponseCreatedEvent(response=response, type='response.created', sequence_number=0)
    ]
    for index, item in enumerate(items):
        if isinstance(item, responses.ResponseReasoningItem):
            added = item.model_copy(update={'encrypted_content': None, 'status': 'in_progress'})
        elif isinstance(item, responses.ResponseOutputMessage):
            added = item.model_copy(update={'content': [], 'status': 'in_progress'})
        else:
            assert isinstance(item, responses.ResponseFunctionToolCall)
            added = item.model_copy(update={'arguments': '', 'status': 'in_progress'})
        events.append(
            responses.ResponseOutputItemAddedEvent(
                item=added, output_index=index, type='response.output_item.added', sequence_number=0
            )
        )
        if isinstance(item, responses.ResponseOutputMessage):
            for content in item.content:
                assert isinstance(content, responses.ResponseOutputText)
                events += [
                    responses.ResponseTextDeltaEvent(
                        item_id=item.id,
                        output_index=index,
                        content_index=0,
                        delta=content.text,
                        logprobs=[],
                        type='response.output_text.delta',
                        sequence_number=0,
                    ),
                    responses.ResponseTextDoneEvent(
                        item_id=item.id,
                        output_index=index,
                        content_index=0,
                        text=content.text,
                        logprobs=[],
                        type='response.output_text.done',
                        sequence_number=0,
                    ),
                ]
        events.append(
            responses.ResponseOutputItemDoneEvent(
                item=item, output_index=index, type='response.output_item.done', sequence_number=0
            )
        )
    events.append(
        responses.ResponseCompletedEvent(
            response=response.model_copy(update={'output': items, 'status': 'completed'}),
            type='response.completed',
            sequence_number=0,
        )
    )
    return [event.model_copy(update={'sequence_number': number}) for number, event in enumerate(events)]


def _text_message(
    item_id: str, text: str, phase: Literal['commentary'] | None = None
) -> responses.ResponseOutputMessage:
    return responses.ResponseOutputMessage(
        id=item_id,
        content=[responses.ResponseOutputText(text=text, type='output_text', annotations=[], logprobs=[])],
        role='assistant',
        status='completed',
        type='message',
        phase=phase,
    )


async def test_meta_stream_tool_call_replays_phase_and_encrypted_reasoning(allow_model_requests: None):
    """The streamed counterpart of `test_meta_tool_call_replays_phase_and_encrypted_reasoning`.

    Mocked rather than recorded: the stream is built from the items `api.meta.ai` returned in that test's recording,
    in the event shape of `test_meta_stream`'s recording, until a streamed tool-calling run is recorded.
    """
    first_stream = _stream(
        [
            responses.ResponseReasoningItem(
                id='rs_6ace34a51d298430167541c6:rs_42e3e9c9-ada0-43df-98f3-10467a757ddd',
                encrypted_content='Q-PaDgG-GA-eJMi_IYYLSt',
                summary=[],
                status='completed',
                type='reasoning',
            ),
            _text_message(
                'rs_dac4fda8-a72d-4669-8375-d203ce7e6abf',
                "I'll check the current weather in Paris for you.",
                phase='commentary',
            ),
            responses.ResponseFunctionToolCall(
                id='fc_f14a3a5c-f7b1-4e30-a0b1-581c840e3796',
                call_id='call_f14a3a5c-f7b1-4e30-a0b1-581c840e3796',
                name='get_weather',
                arguments='{"city":"Paris"}',
                status='completed',
                type='function_call',
            ),
        ]
    )
    second_stream = _stream(
        [_text_message('msg_01a1357e291270419dd7fc96dc684fe6', 'The weather in Paris is sunny and 21°C.')]
    )
    client = MockOpenAIResponses.create_mock_stream([first_stream, second_stream])
    agent = Agent(OpenAIResponsesModel('muse-spark-1.3', provider=MetaProvider(openai_client=client)))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'Sunny and 21C in {city}.'

    async def consume(_ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass

    # `event_stream_handler` streams every model request; `run_stream` would end the run on the commentary text.
    result = await agent.run('What is the weather in Paris?', event_stream_handler=consume)
    assert result.output == snapshot('The weather in Paris is sunny and 21°C.')

    first_response = result.all_messages()[1]
    assert isinstance(first_response, ModelResponse)
    assert [
        (
            type(part).__name__,
            (part.provider_details or {}).get('phase') if isinstance(part, TextPart) else None,
            part.signature if isinstance(part, ThinkingPart) else None,
        )
        for part in first_response.parts
    ] == snapshot(
        [
            ('ThinkingPart', None, 'Q-PaDgG-GA-eJMi_IYYLSt'),
            ('TextPart', 'commentary', None),
            ('ToolCallPart', None, None),
        ]
    )

    first_request, second_request = get_mock_responses_kwargs(client)
    assert first_request['include'] == ['reasoning.encrypted_content']
    replayed: list[dict[str, object]] = second_request['input']
    assert [
        (item.get('type'), item.get('phase'), item.get('encrypted_content'))
        for item in replayed
        if item.get('type') in ('reasoning', 'message', 'function_call')
    ] == snapshot(
        [
            ('reasoning', None, 'Q-PaDgG-GA-eJMi_IYYLSt'),
            ('message', 'commentary', None),
            ('function_call', None, None),
        ]
    )


async def test_meta_chat_completions(allow_model_requests: None, meta_api_key: str, request_capture: RequestCapture):
    """Meta's Chat Completions API works too, but returns no reasoning (only a `reasoning_tokens` count)."""
    provider = MetaProvider(api_key=meta_api_key, http_client=request_capture.http_client(timeout=_TIMEOUT))
    agent = Agent(OpenAIChatModel('muse-spark-1.3', provider=provider))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'Sunny and 21C in {city}.'

    result = await agent.run('What is the weather in Paris? Use the tool.')
    assert result.output == snapshot("It's Sunny and 21°C in Paris.")
    assert any(isinstance(part, ToolCallPart) for part in result.all_messages()[1].parts)
    assert not any(isinstance(part, ThinkingPart) for message in result.all_messages() for part in message.parts)
    assert result.usage.details == snapshot({'reasoning_tokens': 517})
