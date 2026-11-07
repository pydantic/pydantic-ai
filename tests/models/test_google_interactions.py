from __future__ import annotations

import io
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from pydantic_ai import (
    Agent,
    BinaryContent,
    BinaryImage,
    DeferredToolRequests,
    DeferredToolResults,
    ImageUrl,
    ModelRetry,
    VideoUrl,
)
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.direct import model_request, model_request_stream
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    Citation,
    ContentCitationAnchor,
    FilePart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    WebCitationSource,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.native_tools import (
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    MCPServerTool,
    WebFetchTool,
    WebSearchTool,
)
from pydantic_ai.output import NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.settings import ThinkingEffort
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsSameStr, IsStr, RequestCapture, try_import

with try_import() as imports_successful:
    from google.genai.types import MediaResolution

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.google_interactions import GoogleInteractionsModel, GoogleInteractionsModelSettings
    from pydantic_ai.profiles.google import GoogleModelProfile
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'), pytest.mark.vcr]


@pytest.fixture
def interactions_model(gemini_api_key: str, request_capture: RequestCapture) -> GoogleInteractionsModel:
    return GoogleInteractionsModel(
        'gemini-3.8-flash',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
    )


def conversation_and_tools(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """The conversation and tools a request sends, on either Google API, without its settings."""
    return {key: body[key] for key in ('input', 'contents', 'tools') if key in body}


@dataclass(frozen=True)
class ReplayExpectation:
    """The messages a replay test ends with, and the conversation and tools each of its requests sent."""

    messages: list[ModelMessage]
    requests: list[dict[str, JsonValue]]


STATELESS_TOOL_HISTORY: dict[bool, ReplayExpectation] = {
    False: ReplayExpectation(
        messages=snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Use city_temperature to find the temperature in Paris.', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            signature=IsStr(),
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 0,
                                'google_interactions_content_index': 0,
                            },
                        ),
                        ToolCallPart(
                            tool_name='city_temperature',
                            args={'city': 'Paris'},
                            tool_call_id='call_693348',
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 1,
                                'google_interactions_content_index': 0,
                            },
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=59,
                        output_tokens=54,
                        output_reasoning_tokens=38,
                        details={'thoughts_tokens': 38, 'tool_use_tokens': 0, 'input_text_tokens': 59},
                        cost=Decimal('0.00024675'),
                    ),
                    model_name='gemini-3.8-flash',
                    model_id='google:gemini-3.8-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_details={
                        'status': 'requires_action',
                        'google_interactions_steps': [
                            {'signature': IsStr(), 'type': 'thought'},
                            {'id': 'call_693348', 'name': 'city_temperature', 'type': 'function_call'},
                        ],
                        'usage': {
                            'input_tokens_by_modality': [{'modality': 'text', 'tokens': 59}],
                            'total_cached_tokens': 0,
                            'total_input_tokens': 59,
                            'total_output_tokens': 16,
                            'total_thought_tokens': 38,
                            'total_tokens': 113,
                            'total_tool_use_tokens': 0,
                            'initial_input_tokens': 38,
                        },
                        'service_tier': 'standard',
                    },
                    finish_reason='tool_call',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='city_temperature', content=21, tool_call_id='call_693348', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            signature=IsStr(),
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 0,
                                'google_interactions_content_index': 0,
                            },
                        ),
                        TextPart(
                            content='The current temperature in Paris is 21°C.',
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 1,
                                'google_interactions_content_index': 0,
                                'google_interactions_text_sha256': 'faf1b9798e9c2858114b84dce9d05c5e7a0519a99f6ce62ea3ff86a7701a68be',
                            },
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=121,
                        output_tokens=28,
                        output_reasoning_tokens=16,
                        details={'thoughts_tokens': 16, 'tool_use_tokens': 0, 'input_text_tokens': 121},
                        cost=Decimal('0.00019575'),
                    ),
                    model_name='gemini-3.8-flash',
                    model_id='google:gemini-3.8-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_details={
                        'status': 'completed',
                        'google_interactions_steps': [
                            {'signature': IsStr(), 'type': 'thought'},
                            {'content': [{'type': 'text'}], 'type': 'model_output'},
                        ],
                        'usage': {
                            'input_tokens_by_modality': [{'modality': 'text', 'tokens': 121}],
                            'total_cached_tokens': 0,
                            'total_input_tokens': 121,
                            'total_output_tokens': 12,
                            'total_thought_tokens': 16,
                            'total_tokens': 149,
                            'total_tool_use_tokens': 0,
                            'initial_input_tokens': 97,
                        },
                        'service_tier': 'standard',
                    },
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Repeat that temperature without calling a tool.', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            signature=IsStr(),
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 0,
                                'google_interactions_content_index': 0,
                            },
                        ),
                        TextPart(
                            content='The temperature in Paris is 21°C.',
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 1,
                                'google_interactions_content_index': 0,
                                'google_interactions_text_sha256': '779c5810f2282b7b4647bbb86a77fa6a765dbfd8d4f32a126f578182a0e59a63',
                            },
                        ),
                    ],
                    usage=RequestUsage(
                        details={'thoughts_tokens': 26, 'tool_use_tokens': 0, 'input_text_tokens': 159},
                        input_tokens=159,
                        output_tokens=37,
                        output_reasoning_tokens=26,
                        cost=Decimal('0.00025800'),
                    ),
                    model_name='gemini-3.8-flash',
                    model_id='google:gemini-3.8-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_details={
                        'status': 'completed',
                        'google_interactions_steps': [
                            {'signature': IsStr(), 'type': 'thought'},
                            {'content': [{'type': 'text'}], 'type': 'model_output'},
                        ],
                        'usage': {
                            'input_tokens_by_modality': [{'modality': 'text', 'tokens': 159}],
                            'total_cached_tokens': 0,
                            'total_input_tokens': 159,
                            'total_output_tokens': 11,
                            'total_thought_tokens': 26,
                            'total_tokens': 196,
                            'total_tool_use_tokens': 0,
                            'initial_input_tokens': 133,
                        },
                        'service_tier': 'standard',
                    },
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        requests=snapshot(
            [
                {
                    'input': [
                        {
                            'type': 'user_input',
                            'content': [
                                {'type': 'text', 'text': 'Use city_temperature to find the temperature in Paris.'}
                            ],
                        }
                    ],
                    'tools': [
                        {
                            'description': 'Look up the current temperature in the requested city.',
                            'name': 'city_temperature',
                            'parameters': {
                                'additionalProperties': False,
                                'properties': {'city': {'type': 'string'}},
                                'required': ['city'],
                                'type': 'object',
                            },
                            'type': 'function',
                        }
                    ],
                },
                {
                    'input': [
                        {
                            'type': 'user_input',
                            'content': [
                                {'type': 'text', 'text': 'Use city_temperature to find the temperature in Paris.'}
                            ],
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {
                            'id': 'call_693348',
                            'name': 'city_temperature',
                            'type': 'function_call',
                            'arguments': {'city': 'Paris'},
                        },
                        {
                            'type': 'function_result',
                            'name': 'city_temperature',
                            'call_id': 'call_693348',
                            'is_error': False,
                            'result': [{'type': 'text', 'text': '21'}],
                        },
                    ],
                    'tools': [
                        {
                            'description': 'Look up the current temperature in the requested city.',
                            'name': 'city_temperature',
                            'parameters': {
                                'additionalProperties': False,
                                'properties': {'city': {'type': 'string'}},
                                'required': ['city'],
                                'type': 'object',
                            },
                            'type': 'function',
                        }
                    ],
                },
                {
                    'input': [
                        {
                            'type': 'user_input',
                            'content': [
                                {'type': 'text', 'text': 'Use city_temperature to find the temperature in Paris.'}
                            ],
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {
                            'id': 'call_693348',
                            'name': 'city_temperature',
                            'type': 'function_call',
                            'arguments': {'city': 'Paris'},
                        },
                        {
                            'type': 'function_result',
                            'name': 'city_temperature',
                            'call_id': 'call_693348',
                            'is_error': False,
                            'result': [{'type': 'text', 'text': '21'}],
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {
                            'content': [{'type': 'text', 'text': 'The current temperature in Paris is 21°C.'}],
                            'type': 'model_output',
                        },
                        {
                            'type': 'user_input',
                            'content': [{'type': 'text', 'text': 'Repeat that temperature without calling a tool.'}],
                        },
                    ],
                    'tools': [
                        {
                            'description': 'Look up the current temperature in the requested city.',
                            'name': 'city_temperature',
                            'parameters': {
                                'additionalProperties': False,
                                'properties': {'city': {'type': 'string'}},
                                'required': ['city'],
                                'type': 'object',
                            },
                            'type': 'function',
                        }
                    ],
                },
            ]
        ),
    ),
    True: ReplayExpectation(
        messages=snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Use city_temperature to find the temperature in Paris.', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            signature=IsStr(),
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 0,
                                'google_interactions_content_index': 0,
                            },
                        ),
                        ToolCallPart(
                            tool_name='city_temperature',
                            args='{"city":"Paris"}',
                            tool_call_id='call_391632',
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 1,
                                'google_interactions_content_index': 0,
                            },
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=59,
                        output_tokens=54,
                        output_reasoning_tokens=38,
                        details={'thoughts_tokens': 38, 'tool_use_tokens': 0, 'input_text_tokens': 59},
                        cost=Decimal('0.00024675'),
                    ),
                    model_name='gemini-3.8-flash',
                    model_id='google:gemini-3.8-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_details={
                        'status': 'requires_action',
                        'google_interactions_steps': [
                            {'signature': IsStr(), 'type': 'thought'},
                            {'id': 'call_391632', 'name': 'city_temperature', 'type': 'function_call'},
                        ],
                        'usage': {
                            'input_tokens_by_modality': [{'modality': 'text', 'tokens': 59}],
                            'total_cached_tokens': 0,
                            'total_input_tokens': 59,
                            'total_output_tokens': 16,
                            'total_thought_tokens': 38,
                            'total_tokens': 113,
                            'total_tool_use_tokens': 0,
                        },
                        'service_tier': 'standard',
                    },
                    finish_reason='tool_call',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='city_temperature', content=21, tool_call_id='call_391632', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            signature=IsStr(),
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 0,
                                'google_interactions_content_index': 0,
                            },
                        ),
                        TextPart(
                            content='The current temperature in Paris is 21°C.',
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 1,
                                'google_interactions_content_index': 0,
                                'google_interactions_text_sha256': 'faf1b9798e9c2858114b84dce9d05c5e7a0519a99f6ce62ea3ff86a7701a68be',
                            },
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=121,
                        output_tokens=30,
                        output_reasoning_tokens=18,
                        details={'thoughts_tokens': 18, 'tool_use_tokens': 0, 'input_text_tokens': 121},
                        cost=Decimal('0.00020325'),
                    ),
                    model_name='gemini-3.8-flash',
                    model_id='google:gemini-3.8-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_details={
                        'status': 'completed',
                        'google_interactions_steps': [
                            {'signature': IsStr(), 'type': 'thought'},
                            {'content': [{'type': 'text'}], 'type': 'model_output'},
                        ],
                        'usage': {
                            'input_tokens_by_modality': [{'modality': 'text', 'tokens': 121}],
                            'total_cached_tokens': 0,
                            'total_input_tokens': 121,
                            'total_output_tokens': 12,
                            'total_thought_tokens': 18,
                            'total_tokens': 151,
                            'total_tool_use_tokens': 0,
                        },
                        'service_tier': 'standard',
                    },
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Repeat that temperature without calling a tool.', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            signature=IsStr(),
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 0,
                                'google_interactions_content_index': 0,
                            },
                        ),
                        TextPart(
                            content='The temperature in Paris is 21°C.',
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 1,
                                'google_interactions_content_index': 0,
                                'google_interactions_text_sha256': '779c5810f2282b7b4647bbb86a77fa6a765dbfd8d4f32a126f578182a0e59a63',
                            },
                        ),
                    ],
                    usage=RequestUsage(
                        details={'thoughts_tokens': 28, 'tool_use_tokens': 0, 'input_text_tokens': 161},
                        input_tokens=161,
                        output_tokens=39,
                        output_reasoning_tokens=28,
                        cost=Decimal('0.00026700'),
                    ),
                    model_name='gemini-3.8-flash',
                    model_id='google:gemini-3.8-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_details={
                        'status': 'completed',
                        'google_interactions_steps': [
                            {'signature': IsStr(), 'type': 'thought'},
                            {'content': [{'type': 'text'}], 'type': 'model_output'},
                        ],
                        'usage': {
                            'input_tokens_by_modality': [{'modality': 'text', 'tokens': 161}],
                            'total_cached_tokens': 0,
                            'total_input_tokens': 161,
                            'total_output_tokens': 11,
                            'total_thought_tokens': 28,
                            'total_tokens': 200,
                            'total_tool_use_tokens': 0,
                            'initial_input_tokens': 135,
                        },
                        'service_tier': 'standard',
                    },
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        requests=snapshot(
            [
                {
                    'input': [
                        {
                            'type': 'user_input',
                            'content': [
                                {'type': 'text', 'text': 'Use city_temperature to find the temperature in Paris.'}
                            ],
                        }
                    ],
                    'tools': [
                        {
                            'description': 'Look up the current temperature in the requested city.',
                            'name': 'city_temperature',
                            'parameters': {
                                'additionalProperties': False,
                                'properties': {'city': {'type': 'string'}},
                                'required': ['city'],
                                'type': 'object',
                            },
                            'type': 'function',
                        }
                    ],
                },
                {
                    'input': [
                        {
                            'type': 'user_input',
                            'content': [
                                {'type': 'text', 'text': 'Use city_temperature to find the temperature in Paris.'}
                            ],
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {
                            'id': 'call_391632',
                            'name': 'city_temperature',
                            'type': 'function_call',
                            'arguments': {'city': 'Paris'},
                        },
                        {
                            'type': 'function_result',
                            'name': 'city_temperature',
                            'call_id': 'call_391632',
                            'is_error': False,
                            'result': [{'type': 'text', 'text': '21'}],
                        },
                    ],
                    'tools': [
                        {
                            'description': 'Look up the current temperature in the requested city.',
                            'name': 'city_temperature',
                            'parameters': {
                                'additionalProperties': False,
                                'properties': {'city': {'type': 'string'}},
                                'required': ['city'],
                                'type': 'object',
                            },
                            'type': 'function',
                        }
                    ],
                },
                {
                    'input': [
                        {
                            'type': 'user_input',
                            'content': [
                                {'type': 'text', 'text': 'Use city_temperature to find the temperature in Paris.'}
                            ],
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {
                            'id': 'call_391632',
                            'name': 'city_temperature',
                            'type': 'function_call',
                            'arguments': {'city': 'Paris'},
                        },
                        {
                            'type': 'function_result',
                            'name': 'city_temperature',
                            'call_id': 'call_391632',
                            'is_error': False,
                            'result': [{'type': 'text', 'text': '21'}],
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {
                            'content': [{'type': 'text', 'text': 'The current temperature in Paris is 21°C.'}],
                            'type': 'model_output',
                        },
                        {
                            'type': 'user_input',
                            'content': [{'type': 'text', 'text': 'Repeat that temperature without calling a tool.'}],
                        },
                    ],
                    'tools': [
                        {
                            'description': 'Look up the current temperature in the requested city.',
                            'name': 'city_temperature',
                            'parameters': {
                                'additionalProperties': False,
                                'properties': {'city': {'type': 'string'}},
                                'required': ['city'],
                                'type': 'object',
                            },
                            'type': 'function',
                        }
                    ],
                },
            ]
        ),
    ),
}


@pytest.mark.parametrize('stream', [False, True])
async def test_stateless_tool_history(
    interactions_model: GoogleInteractionsModel,
    request_capture: RequestCapture,
    stream: bool,
    allow_model_requests: None,
):
    """Replay the complete signed step history after serialization, without stored interaction state."""
    agent = Agent(interactions_model)

    @agent.tool_plain
    def city_temperature(city: str) -> int:
        """Look up the current temperature in the requested city."""
        assert city == 'Paris'
        return 21

    if stream:
        async with agent.run_stream('Use city_temperature to find the temperature in Paris.') as result:
            await result.get_output()
            history = result.all_messages()
    else:
        history = (await agent.run('Use city_temperature to find the temperature in Paris.')).all_messages()
    history = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))
    final = await agent.run('Repeat that temperature without calling a tool.', message_history=history)
    expected = STATELESS_TOOL_HISTORY[stream]
    assert final.all_messages() == expected.messages
    bodies = request_capture.bodies()
    assert [conversation_and_tools(body) for body in bodies] == expected.requests
    assert all(body['store'] is False and 'previous_interaction_id' not in body for body in bodies)
    # The snapshots elide opaque signatures, so check that the final request replays every persisted step verbatim.
    sent_steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(bodies[-1]['input'])
    for response in history:
        if isinstance(response, ModelResponse):
            assert response.provider_details is not None
            steps = response.provider_details['google_interactions_steps']
            for step in TypeAdapter(list[dict[str, JsonValue]]).validate_python(steps):
                assert any(
                    all(key == 'content' or sent.get(key) == value for key, value in step.items())
                    for sent in sent_steps
                )


class City(BaseModel):
    name: str
    country: str


STRUCTURED_OUTPUT_MESSAGES: dict[tuple[str, bool], list[ModelMessage]] = {
    ('native', False): snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Name the capital city of France and its country.', timestamp=IsDatetime())
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='{"name":"Paris","country":"France"}',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': '9bca2005e9c5f0c3aab2256ab96d6fcfadb26666083cde836426013934bae3dc',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 133, 'tool_use_tokens': 0, 'input_text_tokens': 11},
                    input_tokens=11,
                    output_tokens=142,
                    output_reasoning_tokens=133,
                    cost=Decimal('0.00054075'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 11}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 11,
                        'total_output_tokens': 9,
                        'total_thought_tokens': 133,
                        'total_tokens': 153,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 75,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
    ('native', True): snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Name the capital city of France and its country.', timestamp=IsDatetime())
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='{"name":"Paris","country":"France"}',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': '9bca2005e9c5f0c3aab2256ab96d6fcfadb26666083cde836426013934bae3dc',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 247, 'tool_use_tokens': 0, 'input_text_tokens': 11},
                    input_tokens=11,
                    output_tokens=256,
                    output_reasoning_tokens=247,
                    cost=Decimal('0.00096825'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 11}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 11,
                        'total_output_tokens': 9,
                        'total_thought_tokens': 247,
                        'total_tokens': 267,
                        'total_tool_use_tokens': 0,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
    ('tool', False): snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Name the capital city of France and its country.', timestamp=IsDatetime())
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args={'country': 'France', 'name': 'Paris'},
                        tool_call_id='call_322247',
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 1, 'google_interactions_content_index': 0},
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 93, 'tool_use_tokens': 0, 'input_text_tokens': 67},
                    input_tokens=67,
                    output_tokens=115,
                    output_reasoning_tokens=93,
                    cost=Decimal('0.00048150'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'requires_action',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'id': 'call_322247', 'name': 'final_result', 'type': 'function_call'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 67}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 67,
                        'total_output_tokens': 22,
                        'total_thought_tokens': 93,
                        'total_tokens': 182,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 38,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_322247',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
    ('tool', True): snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Name the capital city of France and its country.', timestamp=IsDatetime())
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"country":"France","name":"Paris"}',
                        tool_call_id='call_352375',
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 1, 'google_interactions_content_index': 0},
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 51, 'tool_use_tokens': 0, 'input_text_tokens': 67},
                    input_tokens=67,
                    output_tokens=73,
                    output_reasoning_tokens=51,
                    cost=Decimal('0.00032400'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'requires_action',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'id': 'call_352375', 'name': 'final_result', 'type': 'function_call'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 67}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 67,
                        'total_output_tokens': 22,
                        'total_thought_tokens': 51,
                        'total_tokens': 140,
                        'total_tool_use_tokens': 0,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_352375',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
    ('prompted', False): snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Name the capital city of France and its country.', timestamp=IsDatetime())
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='{"name": "Paris", "country": "France"}',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': '459092f51cef6d5b912fb2aa53603e6778b45bb8dde09a71916fd606b2738968',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 222, 'tool_use_tokens': 0, 'input_text_tokens': 86},
                    input_tokens=86,
                    output_tokens=234,
                    output_reasoning_tokens=222,
                    cost=Decimal('0.0009420'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 86}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 86,
                        'total_output_tokens': 12,
                        'total_thought_tokens': 222,
                        'total_tokens': 320,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 83,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
    ('prompted', True): snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Name the capital city of France and its country.', timestamp=IsDatetime())
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='{"name": "Paris", "country": "France"}',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': '459092f51cef6d5b912fb2aa53603e6778b45bb8dde09a71916fd606b2738968',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 170, 'tool_use_tokens': 0, 'input_text_tokens': 86},
                    input_tokens=86,
                    output_tokens=182,
                    output_reasoning_tokens=170,
                    cost=Decimal('0.0007470'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 86}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 86,
                        'total_output_tokens': 12,
                        'total_thought_tokens': 170,
                        'total_tokens': 268,
                        'total_tool_use_tokens': 0,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
}


@pytest.mark.parametrize('mode', ['native', 'tool', 'prompted'])
@pytest.mark.parametrize('stream', [False, True])
async def test_structured_output(
    interactions_model: GoogleInteractionsModel, mode: str, stream: bool, allow_model_requests: None
):
    output_type = (
        NativeOutput(City) if mode == 'native' else ToolOutput(City) if mode == 'tool' else PromptedOutput(City)
    )
    agent = Agent(interactions_model, output_type=output_type)
    if stream:
        async with agent.run_stream('Name the capital city of France and its country.') as result:
            output = await result.get_output()
            messages = result.all_messages()
    else:
        result = await agent.run('Name the capital city of France and its country.')
        output, messages = result.output, result.all_messages()
    assert output == City(name='Paris', country='France')
    assert messages == STRUCTURED_OUTPUT_MESSAGES[mode, stream]


@pytest.mark.parametrize('stream', [False, True])
async def test_retry(interactions_model: GoogleInteractionsModel, stream: bool, allow_model_requests: None):
    agent = Agent(interactions_model)
    calls = 0

    @agent.tool_plain
    def get_password() -> str:
        """Get the password. Retry if the service is temporarily unavailable."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelRetry('Temporarily unavailable. Call get_password again.')
        return 'silver fox'

    if stream:
        async with agent.run_stream('Use get_password to tell me the password.') as result:
            output = await result.get_output()
    else:
        output = (await agent.run('Use get_password to tell me the password.')).output
    assert calls == 2
    assert 'silver fox' in output.lower()


NATIVE_TOOL_MESSAGES: dict[tuple[str, bool], list[ModelMessage]] = {
    ('code', False): snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Use code_execution to calculate the sum of the first 137 prime numbers. Include the numerical answer.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    NativeToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
from sympy import primerange, prime

# Calculate the sum of the first 137 prime numbers
# prime(n) gives the n-th prime number (1-indexed)
primes = [prime(i) for i in range(1, 138)]
sum_primes = sum(primes)
print(f"Number of primes: {len(primes)}")
print(f"137th prime: {primes[-1]}")
print(f"Sum of the first 137 prime numbers: {sum_primes}")
""",
                            'language': 'python',
                        },
                        tool_call_id='call_472499',
                        provider_name='google',
                        provider_details={
                            'id': 'call_472499',
                            'type': 'code_execution_call',
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    NativeToolReturnPart(
                        tool_name='code_execution',
                        content="""\
Number of primes: 137
137th prime: 773
Sum of the first 137 prime numbers: 48494
""",
                        tool_call_id='call_472499',
                        timestamp=IsDatetime(),
                        provider_name='google',
                        provider_details={
                            'call_id': 'call_472499',
                            'is_error': False,
                            'type': 'code_execution_result',
                            'google_interactions_step_index': 2,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 3, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='The sum of the first 137 prime numbers (from 2 up to 773) is **48,494**.',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 4,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': 'b25a1dd15a12d50f2b4b44f622cef1cd0f801487b8a522e434d3583ec8e5d6d3',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={
                        'thoughts_tokens': 53,
                        'tool_use_tokens': 247,
                        'input_text_tokens': 77,
                        'tool_use_text_tokens': 247,
                    },
                    input_tokens=77,
                    output_tokens=218,
                    output_reasoning_tokens=53,
                    cost=Decimal('0.00087525'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'id': 'call_472499', 'type': 'code_execution_call'},
                        {'call_id': 'call_472499', 'is_error': False, 'type': 'code_execution_result'},
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 77}],
                        'tool_use_tokens_by_modality': [{'modality': 'text', 'tokens': 247}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 77,
                        'total_output_tokens': 165,
                        'total_thought_tokens': 53,
                        'total_tokens': 542,
                        'total_tool_use_tokens': 247,
                        'initial_input_tokens': 23,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Summarize the answer briefly using only the preceding information.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='The sum of the first 137 prime numbers is **48,494**.',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': 'b92006805f4cd0b24a7f763cd1649f5f0b41f0797386b82ad5c30b7a7764f75e',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 0, 'tool_use_tokens': 0, 'input_text_tokens': 123},
                    input_tokens=123,
                    output_tokens=20,
                    cost=Decimal('0.00016725'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 123}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 123,
                        'total_output_tokens': 20,
                        'total_thought_tokens': 0,
                        'total_tokens': 143,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 289,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
    ('code', True): snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Use code_execution to calculate the sum of the first 137 prime numbers. Include the numerical answer.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    NativeToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
import sympy

# Generate the first 137 prime numbers and find their sum
primes = [sympy.prime(i) for i in range(1, 138)]
print(f"Number of primes: {len(primes)}")
print(f"137th prime: {primes[-1]}")
print(f"Sum: {sum(primes)}")
""",
                            'language': 'python',
                        },
                        tool_call_id='call_15801776122159094243',
                        provider_name='google',
                        provider_details={
                            'id': 'call_15801776122159094243',
                            'signature': IsStr(),
                            'type': 'code_execution_call',
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    NativeToolReturnPart(
                        tool_name='code_execution',
                        content="""\
Number of primes: 137
137th prime: 773
Sum: 48494
""",
                        tool_call_id='call_15801776122159094243',
                        timestamp=IsDatetime(),
                        provider_name='google',
                        provider_details={
                            'call_id': 'call_15801776122159094243',
                            'is_error': False,
                            'signature': IsStr(),
                            'type': 'code_execution_result',
                            'google_interactions_step_index': 2,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 3, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='The sum of the first 137 prime numbers (from 2 up to 773) is **48,494**.',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 4,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': 'b25a1dd15a12d50f2b4b44f622cef1cd0f801487b8a522e434d3583ec8e5d6d3',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={
                        'thoughts_tokens': 32,
                        'tool_use_tokens': 179,
                        'input_text_tokens': 56,
                        'tool_use_text_tokens': 179,
                    },
                    input_tokens=56,
                    output_tokens=159,
                    output_reasoning_tokens=32,
                    cost=Decimal('0.00063825'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'id': 'call_15801776122159094243', 'signature': IsStr(), 'type': 'code_execution_call'},
                        {
                            'call_id': 'call_15801776122159094243',
                            'is_error': False,
                            'signature': IsStr(),
                            'type': 'code_execution_result',
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 56}],
                        'tool_use_tokens_by_modality': [{'modality': 'text', 'tokens': 179}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 56,
                        'total_output_tokens': 127,
                        'total_thought_tokens': 32,
                        'total_tokens': 394,
                        'total_tool_use_tokens': 179,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Summarize the answer briefly using only the preceding information.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='The sum of the first 137 prime numbers is **48,494**.',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': 'b92006805f4cd0b24a7f763cd1649f5f0b41f0797386b82ad5c30b7a7764f75e',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 0, 'tool_use_tokens': 0, 'input_text_tokens': 102},
                    input_tokens=102,
                    output_tokens=20,
                    cost=Decimal('0.0001515'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 102}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 102,
                        'total_output_tokens': 20,
                        'total_thought_tokens': 0,
                        'total_tokens': 122,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 221,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
    ('search', False): snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Use google_search to find the current stable Python version and cite a source.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['python releases site python org', 'latest stable python version']},
                        tool_call_id='call_454387',
                        provider_name='google',
                        provider_details={
                            'id': 'call_454387',
                            'search_type': 'web_search',
                            'signature': IsStr(),
                            'type': 'google_search_call',
                            'google_interactions_step_index': 0,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content=[{'search_suggestions': IsStr()}],
                        tool_call_id='call_454387',
                        timestamp=IsDatetime(),
                        provider_name='google',
                        provider_details={
                            'call_id': 'call_454387',
                            'is_error': False,
                            'signature': IsStr(),
                            'type': 'google_search_result',
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['"Python 3.15.2" site:python.org']},
                        tool_call_id='call_454451',
                        provider_name='google',
                        provider_details={
                            'id': 'call_454451',
                            'search_type': 'web_search',
                            'signature': IsStr(),
                            'type': 'google_search_call',
                            'google_interactions_step_index': 2,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content=[{'search_suggestions': IsStr()}],
                        tool_call_id='call_454451',
                        timestamp=IsDatetime(),
                        provider_name='google',
                        provider_details={
                            'call_id': 'call_454451',
                            'is_error': False,
                            'signature': IsStr(),
                            'type': 'google_search_result',
                            'google_interactions_step_index': 3,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 4, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content="""\
The latest stable version of Python is **Python 3.15.2**, released on October 25, 2026. It is the most recent maintenance and security update in the Python 3.15 series.

**Source:** [Python.org Downloads](https://www.python.org/downloads/) / [Python 3.15.2 Release Notes](https://www.python.org/downloads/release/python-3152/)\
""",
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 5,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': 'd4995280c28fe7b0633e5bb1007744680a65f12941728dbcfcf0f8b4462be931',
                        },
                        citations=[
                            Citation(
                                sources=[
                                    WebCitationSource(
                                        url='https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5c6lAkrItejhy0SgHGKH58U5qjkjHiZ-YLg-5PsZ0R-jhWsCVP7D10rPjO3LdlvFm5ykKcy4-Lgmwz8mY5XEU5RjrEFFVMEEvOCpOhQuielb9yNFT5EdLMIuXOKupmnt9QTNjsWLt6KC6XQ==',
                                        title='python.org',
                                    )
                                ],
                                anchor=ContentCitationAnchor(start=0, end=86),
                            ),
                            Citation(
                                sources=[
                                    WebCitationSource(
                                        url='https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEG9nGyNjX0hEzv3pGi8EZJ13DqqgPpYc77Ioy1Yxi5qNDJqzTChNmm6j0y0J42OHr5xaRjxYegghDC6yGH9RknnHkSkRzvyn2iDZtniv9KLYLIA4sA2pE=',
                                        title='python.org',
                                    )
                                ],
                                anchor=ContentCitationAnchor(start=0, end=86),
                            ),
                            Citation(
                                sources=[
                                    WebCitationSource(
                                        url='https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5c6lAkrItejhy0SgHGKH58U5qjkjHiZ-YLg-5PsZ0R-jhWsCVP7D10rPjO3LdlvFm5ykKcy4-Lgmwz8mY5XEU5RjrEFFVMEEvOCpOhQuielb9yNFT5EdLMIuXOKupmnt9QTNjsWLt6KC6XQ==',
                                        title='python.org',
                                    )
                                ],
                                anchor=ContentCitationAnchor(start=88, end=167),
                            ),
                            Citation(
                                sources=[
                                    WebCitationSource(
                                        url='https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEdsZ-g-cb_wtoig_fwx42SrCkYn8tCPRmbP1isIjQZNz_5lArL1Ec1slnYiGMT55_KZpzEnVbdI0Cxxg9Absh4UNi8wcPESUlH0s4BeSCjnJ1-tb76mZcUPZynzJRczWhqgDz_Zr-a7tvl3I1hQ3GLaOQcHNPzu8zBqA==',
                                        title='python.org',
                                    )
                                ],
                                anchor=ContentCitationAnchor(start=88, end=167),
                            ),
                            Citation(
                                sources=[
                                    WebCitationSource(
                                        url='https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEG9nGyNjX0hEzv3pGi8EZJ13DqqgPpYc77Ioy1Yxi5qNDJqzTChNmm6j0y0J42OHr5xaRjxYegghDC6yGH9RknnHkSkRzvyn2iDZtniv9KLYLIA4sA2pE=',
                                        title='python.org',
                                    )
                                ],
                                anchor=ContentCitationAnchor(start=170, end=239),
                            ),
                            Citation(
                                sources=[
                                    WebCitationSource(
                                        url='https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5c6lAkrItejhy0SgHGKH58U5qjkjHiZ-YLg-5PsZ0R-jhWsCVP7D10rPjO3LdlvFm5ykKcy4-Lgmwz8mY5XEU5RjrEFFVMEEvOCpOhQuielb9yNFT5EdLMIuXOKupmnt9QTNjsWLt6KC6XQ==',
                                        title='python.org',
                                    )
                                ],
                                anchor=ContentCitationAnchor(start=170, end=326),
                            ),
                        ],
                    ),
                ],
                usage=RequestUsage(
                    details={
                        'thoughts_tokens': 359,
                        'tool_use_tokens': 0,
                        'google_search_count': 3,
                        'input_text_tokens': 234,
                        'cached_text_tokens': 87,
                    },
                    input_tokens=234,
                    output_tokens=500,
                    cache_read_tokens=87,
                    output_reasoning_tokens=359,
                    web_searches=3,
                    cost=Decimal('0.043991775'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {
                            'id': 'call_454387',
                            'search_type': 'web_search',
                            'signature': IsStr(),
                            'type': 'google_search_call',
                        },
                        {
                            'call_id': 'call_454387',
                            'is_error': False,
                            'signature': IsStr(),
                            'type': 'google_search_result',
                        },
                        {
                            'id': 'call_454451',
                            'search_type': 'web_search',
                            'signature': IsStr(),
                            'type': 'google_search_call',
                        },
                        {
                            'call_id': 'call_454451',
                            'is_error': False,
                            'signature': IsStr(),
                            'type': 'google_search_result',
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {
                            'content': [
                                {
                                    'annotations': [
                                        {
                                            'end_index': 86,
                                            'start_index': 0,
                                            'title': 'python.org',
                                            'type': 'url_citation',
                                            'url': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5c6lAkrItejhy0SgHGKH58U5qjkjHiZ-YLg-5PsZ0R-jhWsCVP7D10rPjO3LdlvFm5ykKcy4-Lgmwz8mY5XEU5RjrEFFVMEEvOCpOhQuielb9yNFT5EdLMIuXOKupmnt9QTNjsWLt6KC6XQ==',
                                        },
                                        {
                                            'end_index': 86,
                                            'start_index': 0,
                                            'title': 'python.org',
                                            'type': 'url_citation',
                                            'url': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEG9nGyNjX0hEzv3pGi8EZJ13DqqgPpYc77Ioy1Yxi5qNDJqzTChNmm6j0y0J42OHr5xaRjxYegghDC6yGH9RknnHkSkRzvyn2iDZtniv9KLYLIA4sA2pE=',
                                        },
                                        {
                                            'end_index': 167,
                                            'start_index': 88,
                                            'title': 'python.org',
                                            'type': 'url_citation',
                                            'url': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5c6lAkrItejhy0SgHGKH58U5qjkjHiZ-YLg-5PsZ0R-jhWsCVP7D10rPjO3LdlvFm5ykKcy4-Lgmwz8mY5XEU5RjrEFFVMEEvOCpOhQuielb9yNFT5EdLMIuXOKupmnt9QTNjsWLt6KC6XQ==',
                                        },
                                        {
                                            'end_index': 167,
                                            'start_index': 88,
                                            'title': 'python.org',
                                            'type': 'url_citation',
                                            'url': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEdsZ-g-cb_wtoig_fwx42SrCkYn8tCPRmbP1isIjQZNz_5lArL1Ec1slnYiGMT55_KZpzEnVbdI0Cxxg9Absh4UNi8wcPESUlH0s4BeSCjnJ1-tb76mZcUPZynzJRczWhqgDz_Zr-a7tvl3I1hQ3GLaOQcHNPzu8zBqA==',
                                        },
                                        {
                                            'end_index': 239,
                                            'start_index': 170,
                                            'title': 'python.org',
                                            'type': 'url_citation',
                                            'url': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEG9nGyNjX0hEzv3pGi8EZJ13DqqgPpYc77Ioy1Yxi5qNDJqzTChNmm6j0y0J42OHr5xaRjxYegghDC6yGH9RknnHkSkRzvyn2iDZtniv9KLYLIA4sA2pE=',
                                        },
                                        {
                                            'end_index': 326,
                                            'start_index': 170,
                                            'title': 'python.org',
                                            'type': 'url_citation',
                                            'url': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5c6lAkrItejhy0SgHGKH58U5qjkjHiZ-YLg-5PsZ0R-jhWsCVP7D10rPjO3LdlvFm5ykKcy4-Lgmwz8mY5XEU5RjrEFFVMEEvOCpOhQuielb9yNFT5EdLMIuXOKupmnt9QTNjsWLt6KC6XQ==',
                                        },
                                    ],
                                    'type': 'text',
                                }
                            ],
                            'type': 'model_output',
                        },
                    ],
                    'usage': {
                        'cached_tokens_by_modality': [{'modality': 'text', 'tokens': 87}],
                        'grounding_tool_count': [{'count': 3, 'type': 'google_search'}],
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 234}],
                        'total_cached_tokens': 87,
                        'total_input_tokens': 234,
                        'total_output_tokens': 141,
                        'total_thought_tokens': 359,
                        'total_tokens': 734,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 48,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Summarize the answer briefly using only the preceding information.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='The current stable release is **Python 3.15.2**, released on October 25, 2026, as part of the Python 3.15 series (Source: Python.org).',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': '665ef808a5f133d769d682e579b4729eae1247ddf7c73faadfc2cdce5b8858ff',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 251, 'tool_use_tokens': 0, 'input_text_tokens': 273},
                    input_tokens=273,
                    output_tokens=297,
                    output_reasoning_tokens=251,
                    cost=Decimal('0.00131850'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 273}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 273,
                        'total_output_tokens': 46,
                        'total_thought_tokens': 251,
                        'total_tokens': 570,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 302,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
    ('search', True): snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Use google_search to find the current stable Python version and cite a source.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='web_search',
                        args={
                            'queries': [
                                'latest stable Python release python org',
                                'Python stable version python org downloads',
                            ]
                        },
                        tool_call_id='call_373351',
                        provider_name='google',
                        provider_details={
                            'id': 'call_373351',
                            'search_type': 'web_search',
                            'signature': IsStr(),
                            'type': 'google_search_call',
                            'google_interactions_step_index': 0,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    NativeToolReturnPart(
                        tool_name='web_search',
                        content=[{'search_suggestions': IsStr()}],
                        tool_call_id='call_373351',
                        timestamp=IsDatetime(),
                        provider_name='google',
                        provider_details={
                            'call_id': 'call_373351',
                            'is_error': False,
                            'signature': IsStr(),
                            'type': 'google_search_result',
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 2, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content="""\
The current stable version of Python is **Python 3.15.2**, released on October 25, 2026.

**Source:**
* [Python.org Downloads](https://www.python.org/downloads/) / [Python Release Python 3.15.2](https://www.python.org/downloads/release/python-3152/)\
""",
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 3,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': '72899ae3169e6b85e5f6776c2cfd494fc7428c9370307fd082a218b0989282fd',
                        },
                        citations=[
                            Citation(
                                sources=[
                                    WebCitationSource(
                                        url='https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE6LMDPHKaVGWSNq-MLZutYl_kZfTw8pVMs_jMfZZWcCdiQuiwG_jZdZAtp4vqCMWq7kcaT-rIbZ50Z3ZlTek4t-AO_q_OnsWR02SDwEetFNqQAyURL1dc=',
                                        title='python.org',
                                    )
                                ],
                                anchor=ContentCitationAnchor(start=0, end=87),
                            ),
                            Citation(
                                sources=[
                                    WebCitationSource(
                                        url='https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE6LMDPHKaVGWSNq-MLZutYl_kZfTw8pVMs_jMfZZWcCdiQuiwG_jZdZAtp4vqCMWq7kcaT-rIbZ50Z3ZlTek4t-AO_q_OnsWR02SDwEetFNqQAyURL1dc=',
                                        title='python.org',
                                    )
                                ],
                                anchor=ContentCitationAnchor(start=104, end=161),
                            ),
                            Citation(
                                sources=[
                                    WebCitationSource(
                                        url='https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFAGYytNmfoUhtReYwAi4Xb4x82jlgYIRMKOtrRic68rGyMzQxICTtVdWwg9ISwIaVymGETDlAh_HGiCBW1UTUgG1DmRrMfiv0Iuem5qUQXixf7-3pohQZojkbNse653ArnFH-ijHhQEMELDQ==',
                                        title='python.org',
                                    )
                                ],
                                anchor=ContentCitationAnchor(start=104, end=249),
                            ),
                        ],
                    ),
                ],
                usage=RequestUsage(
                    details={
                        'thoughts_tokens': 131,
                        'tool_use_tokens': 0,
                        'google_search_count': 2,
                        'input_text_tokens': 66,
                    },
                    input_tokens=66,
                    output_tokens=235,
                    output_reasoning_tokens=131,
                    web_searches=2,
                    cost=Decimal('0.02893075'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {
                            'id': 'call_373351',
                            'search_type': 'web_search',
                            'signature': IsStr(),
                            'type': 'google_search_call',
                        },
                        {
                            'call_id': 'call_373351',
                            'is_error': False,
                            'signature': IsStr(),
                            'type': 'google_search_result',
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {
                            'content': [
                                {
                                    'annotations': [
                                        {
                                            'end_index': 87,
                                            'start_index': 0,
                                            'title': 'python.org',
                                            'type': 'url_citation',
                                            'url': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE6LMDPHKaVGWSNq-MLZutYl_kZfTw8pVMs_jMfZZWcCdiQuiwG_jZdZAtp4vqCMWq7kcaT-rIbZ50Z3ZlTek4t-AO_q_OnsWR02SDwEetFNqQAyURL1dc=',
                                        },
                                        {
                                            'end_index': 161,
                                            'start_index': 104,
                                            'title': 'python.org',
                                            'type': 'url_citation',
                                            'url': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE6LMDPHKaVGWSNq-MLZutYl_kZfTw8pVMs_jMfZZWcCdiQuiwG_jZdZAtp4vqCMWq7kcaT-rIbZ50Z3ZlTek4t-AO_q_OnsWR02SDwEetFNqQAyURL1dc=',
                                        },
                                        {
                                            'end_index': 249,
                                            'start_index': 104,
                                            'title': 'python.org',
                                            'type': 'url_citation',
                                            'url': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFAGYytNmfoUhtReYwAi4Xb4x82jlgYIRMKOtrRic68rGyMzQxICTtVdWwg9ISwIaVymGETDlAh_HGiCBW1UTUgG1DmRrMfiv0Iuem5qUQXixf7-3pohQZojkbNse653ArnFH-ijHhQEMELDQ==',
                                        },
                                    ],
                                    'type': 'text',
                                }
                            ],
                            'type': 'model_output',
                        },
                    ],
                    'usage': {
                        'grounding_tool_count': [{'count': 2, 'type': 'google_search'}],
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 66}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 66,
                        'total_output_tokens': 104,
                        'total_thought_tokens': 131,
                        'total_tokens': 301,
                        'total_tool_use_tokens': 0,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Summarize the answer briefly using only the preceding information.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='The current stable version of Python is **3.15.2**, released on October 25, 2026, according to Python.org.',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': 'ce708b144ba0d8eef484a16dd71de73745a9c06e068137b41fdbce3db5e62210',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 0, 'tool_use_tokens': 0, 'input_text_tokens': 196},
                    input_tokens=196,
                    output_tokens=34,
                    cost=Decimal('0.0002745'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 196}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 196,
                        'total_output_tokens': 34,
                        'total_thought_tokens': 0,
                        'total_tokens': 230,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 225,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
    ('fetch', False): snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Use url_context to read https://ai.google.dev/gemini-api/docs/interactions-overview and summarize it in a sentence.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='web_fetch',
                        args={'urls': ['https://ai.google.dev/gemini-api/docs/interactions-overview']},
                        tool_call_id='call_712375774064598247',
                        provider_name='google',
                        provider_details={
                            'id': 'call_712375774064598247',
                            'signature': IsStr(),
                            'type': 'url_context_call',
                            'google_interactions_step_index': 0,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    NativeToolReturnPart(
                        tool_name='web_fetch',
                        content=[
                            {'status': 'success', 'url': 'https://aistudio.google.com/docs/interactions-overview'},
                            {'status': 'error', 'url': 'https://ai.google.dev/gemini-api/docs/interactions-overview'},
                        ],
                        tool_call_id='call_712375774064598247',
                        timestamp=IsDatetime(),
                        provider_name='google',
                        provider_details={
                            'call_id': 'call_712375774064598247',
                            'is_error': False,
                            'signature': IsStr(),
                            'type': 'url_context_result',
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 2, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content="The Interactions API is Google's unified interface for building with Gemini models and agents, offering features such as server-side conversation state management, background execution, and observable multi-step reasoning and tool workflows.",
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 3,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': '2dcd429e57be2438297411f79196ac2b7a495c19ea18596852d450651ead857c',
                        },
                        citations=[
                            Citation(
                                sources=[
                                    WebCitationSource(
                                        url='https://aistudio.google.com/docs/interactions-overview',
                                        title='Interactions API - Gemini API | Google AI Studio',
                                    )
                                ],
                                anchor=ContentCitationAnchor(start=0, end=240),
                            )
                        ],
                    ),
                ],
                usage=RequestUsage(
                    details={
                        'thoughts_tokens': 288,
                        'tool_use_tokens': 63,
                        'input_text_tokens': 32,
                        'tool_use_text_tokens': 63,
                    },
                    input_tokens=32,
                    output_tokens=362,
                    output_reasoning_tokens=288,
                    cost=Decimal('0.0013815'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {
                            'id': 'call_712375774064598247',
                            'signature': IsStr(),
                            'type': 'url_context_call',
                        },
                        {
                            'call_id': 'call_712375774064598247',
                            'is_error': False,
                            'signature': IsStr(),
                            'type': 'url_context_result',
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {
                            'content': [
                                {
                                    'annotations': [
                                        {
                                            'end_index': 240,
                                            'start_index': 0,
                                            'title': 'Interactions API - Gemini API | Google AI Studio',
                                            'type': 'url_citation',
                                            'url': 'https://aistudio.google.com/docs/interactions-overview',
                                        }
                                    ],
                                    'type': 'text',
                                }
                            ],
                            'type': 'model_output',
                        },
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 32}],
                        'tool_use_tokens_by_modality': [{'modality': 'text', 'tokens': 63}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 32,
                        'total_output_tokens': 74,
                        'total_thought_tokens': 288,
                        'total_tokens': 457,
                        'total_tool_use_tokens': 63,
                        'initial_input_tokens': 63,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Summarize the answer briefly using only the preceding information.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content="Google's Interactions API provides a unified interface for Gemini models and agents, supporting server-side state management, background execution, and observable multi-step workflows.",
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': 'd8a61b4307f86a393fda0e81293e01f118f06a2457050e4df8b90f8bc38930bd',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 235, 'tool_use_tokens': 0, 'input_text_tokens': 415},
                    input_tokens=415,
                    output_tokens=267,
                    output_reasoning_tokens=235,
                    cost=Decimal('0.00131250'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 415}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 415,
                        'total_output_tokens': 32,
                        'total_thought_tokens': 235,
                        'total_tokens': 682,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 7971,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
    ('fetch', True): snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Use url_context to read https://ai.google.dev/gemini-api/docs/interactions-overview and summarize it in a sentence.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='web_fetch',
                        args={'urls': ['https://ai.google.dev/gemini-api/docs/interactions-overview']},
                        tool_call_id='call_12260913456803845438',
                        provider_name='google',
                        provider_details={
                            'id': 'call_12260913456803845438',
                            'signature': IsStr(),
                            'type': 'url_context_call',
                            'google_interactions_step_index': 0,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    NativeToolReturnPart(
                        tool_name='web_fetch',
                        content=[
                            {'status': 'success', 'url': 'https://aistudio.google.com/docs/interactions-overview'},
                            {'status': 'error', 'url': 'https://ai.google.dev/gemini-api/docs/interactions-overview'},
                        ],
                        tool_call_id='call_12260913456803845438',
                        timestamp=IsDatetime(),
                        provider_name='google',
                        provider_details={
                            'call_id': 'call_12260913456803845438',
                            'is_error': False,
                            'signature': IsStr(),
                            'type': 'url_context_result',
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 2, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content="The Interactions API is Google's unified interface for Gemini models and agents, designed to streamline agentic workflows with features like server-side conversation state management, observable execution steps, and background execution.",
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 3,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': 'a519487e6b0bf31b051a0084c086c0d2cf6135b743b8014ef4956f3323bdf04b',
                        },
                        citations=[
                            Citation(
                                sources=[
                                    WebCitationSource(
                                        url='https://aistudio.google.com/docs/interactions-overview',
                                        title='Interactions API - Gemini API | Google AI Studio',
                                    )
                                ],
                                anchor=ContentCitationAnchor(start=0, end=236),
                            )
                        ],
                    ),
                ],
                usage=RequestUsage(
                    details={
                        'thoughts_tokens': 227,
                        'tool_use_tokens': 63,
                        'input_text_tokens': 32,
                        'tool_use_text_tokens': 63,
                    },
                    input_tokens=32,
                    output_tokens=299,
                    output_reasoning_tokens=227,
                    cost=Decimal('0.00114525'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {
                            'id': 'call_12260913456803845438',
                            'signature': IsStr(),
                            'type': 'url_context_call',
                        },
                        {
                            'call_id': 'call_12260913456803845438',
                            'is_error': False,
                            'signature': IsStr(),
                            'type': 'url_context_result',
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {
                            'content': [
                                {
                                    'annotations': [
                                        {
                                            'end_index': 236,
                                            'start_index': 0,
                                            'title': 'Interactions API - Gemini API | Google AI Studio',
                                            'type': 'url_citation',
                                            'url': 'https://aistudio.google.com/docs/interactions-overview',
                                        }
                                    ],
                                    'type': 'text',
                                }
                            ],
                            'type': 'model_output',
                        },
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 32}],
                        'tool_use_tokens_by_modality': [{'modality': 'text', 'tokens': 63}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 32,
                        'total_output_tokens': 72,
                        'total_thought_tokens': 227,
                        'total_tokens': 394,
                        'total_tool_use_tokens': 63,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Summarize the answer briefly using only the preceding information.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content="Google's Interactions API provides a unified interface for Gemini models and agents, featuring server-side state management, observable execution steps, and background execution to simplify agentic workflows.",
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': '9f309d99c969bf00e6e7c3ed358a476f5bbcf2f809464c9187e1310d51c9119f',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 175, 'tool_use_tokens': 0, 'input_text_tokens': 352},
                    input_tokens=352,
                    output_tokens=210,
                    output_reasoning_tokens=175,
                    cost=Decimal('0.0010515'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 352}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 352,
                        'total_output_tokens': 35,
                        'total_thought_tokens': 175,
                        'total_tokens': 562,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 7908,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
}


@pytest.mark.parametrize('native_tool', ['code', 'search', 'fetch'])
@pytest.mark.parametrize('stream', [False, True])
async def test_native_tools(
    interactions_model: GoogleInteractionsModel, native_tool: str, stream: bool, allow_model_requests: None
):
    tool = (
        CodeExecutionTool() if native_tool == 'code' else WebSearchTool() if native_tool == 'search' else WebFetchTool()
    )
    prompts = {
        'code': 'Use code_execution to calculate the sum of the first 137 prime numbers. Include the numerical answer.',
        'search': 'Use google_search to find the current stable Python version and cite a source.',
        'fetch': 'Use url_context to read https://ai.google.dev/gemini-api/docs/interactions-overview and summarize it in a sentence.',
    }
    agent = Agent(interactions_model, capabilities=[NativeTool(tool)])
    if stream:
        async with agent.run_stream(prompts[native_tool]) as result:
            await result.get_output()
            history = result.all_messages()
    else:
        history = (await agent.run(prompts[native_tool])).all_messages()
    final = await agent.run(
        'Summarize the answer briefly using only the preceding information.', message_history=history
    )
    assert final.all_messages() == NATIVE_TOOL_MESSAGES[native_tool, stream]


IMAGE_DESCRIPTIONS: dict[bool, str] = {
    False: snapshot(
        'A close-up view of a freshly sliced kiwi fruit, highlighting its vibrant green flesh, creamy center, and ring of tiny black seeds against a plain white background.'
    ),
    True: snapshot(
        'A close-up shot of a sliced ripe kiwi fruit, displaying its vibrant green flesh, pale core, and ring of tiny black seeds against a plain white background.'
    ),
}


@pytest.mark.parametrize('stream', [False, True])
async def test_image_input(
    interactions_model: GoogleInteractionsModel, image_content: BinaryContent, stream: bool, allow_model_requests: None
):
    agent = Agent(interactions_model)
    prompt = ['Describe this image in a sentence.', image_content]
    if stream:
        async with agent.run_stream(prompt) as result:
            output = await result.get_output()
    else:
        output = (await agent.run(prompt)).output
    assert output == IMAGE_DESCRIPTIONS[stream]


async def test_deferred_approval(
    interactions_model: GoogleInteractionsModel, request_capture: RequestCapture, allow_model_requests: None
):

    agent = Agent(interactions_model, output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def send_invoice(customer: str) -> str:
        """Send the customer's invoice after approval."""
        assert customer == 'Alice'
        return 'Invoice INV-123 sent to Alice.'

    pending = await agent.run('Send the invoice to Alice using send_invoice.')
    assert isinstance(pending.output, DeferredToolRequests)
    call = pending.output.approvals[0]
    history = ModelMessagesTypeAdapter.validate_json(pending.all_messages_json())
    completed = await agent.run(
        message_history=history, deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: True})
    )
    assert completed.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Send the invoice to Alice using send_invoice.', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=(signature := IsSameStr()),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    ToolCallPart(
                        tool_name='send_invoice',
                        args={'customer': 'Alice'},
                        tool_call_id='call_383271',
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 1, 'google_interactions_content_index': 0},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=57,
                    output_tokens=68,
                    output_reasoning_tokens=52,
                    details={'thoughts_tokens': 52, 'tool_use_tokens': 0, 'input_text_tokens': 57},
                    cost=Decimal('0.00029775'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'requires_action',
                    'google_interactions_steps': [
                        {'signature': signature, 'type': 'thought'},
                        {'id': 'call_383271', 'name': 'send_invoice', 'type': 'function_call'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 57}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 57,
                        'total_output_tokens': 16,
                        'total_thought_tokens': 52,
                        'total_tokens': 125,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 36,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='send_invoice',
                        content='Invoice INV-123 sent to Alice.',
                        tool_call_id='call_383271',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='The invoice (INV-123) has been successfully sent to Alice.',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': 'bf646b9e06430a16a255bfe5e88f5ac9faf26554adcf8f52f6986b4e931907a3',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 11, 'tool_use_tokens': 0, 'input_text_tokens': 133},
                    input_tokens=133,
                    output_tokens=27,
                    output_reasoning_tokens=11,
                    cost=Decimal('0.00020100'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 133}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 133,
                        'total_output_tokens': 16,
                        'total_thought_tokens': 11,
                        'total_tokens': 160,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 117,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert [conversation_and_tools(body) for body in request_capture.bodies()] == snapshot(
        [
            {
                'input': [
                    {
                        'type': 'user_input',
                        'content': [{'type': 'text', 'text': 'Send the invoice to Alice using send_invoice.'}],
                    }
                ],
                'tools': [
                    {
                        'description': "Send the customer's invoice after approval.",
                        'name': 'send_invoice',
                        'parameters': {
                            'additionalProperties': False,
                            'properties': {'customer': {'type': 'string'}},
                            'required': ['customer'],
                            'type': 'object',
                        },
                        'type': 'function',
                    }
                ],
            },
            {
                'input': [
                    {
                        'type': 'user_input',
                        'content': [{'type': 'text', 'text': 'Send the invoice to Alice using send_invoice.'}],
                    },
                    {'signature': signature, 'type': 'thought'},
                    {
                        'id': 'call_383271',
                        'name': 'send_invoice',
                        'type': 'function_call',
                        'arguments': {'customer': 'Alice'},
                    },
                    {
                        'type': 'function_result',
                        'name': 'send_invoice',
                        'call_id': 'call_383271',
                        'is_error': False,
                        'result': [{'type': 'text', 'text': 'Invoice INV-123 sent to Alice.'}],
                    },
                ],
                'tools': [
                    {
                        'description': "Send the customer's invoice after approval.",
                        'name': 'send_invoice',
                        'parameters': {
                            'additionalProperties': False,
                            'properties': {'customer': {'type': 'string'}},
                            'required': ['customer'],
                            'type': 'object',
                        },
                        'type': 'function',
                    }
                ],
            },
        ]
    )


@pytest.mark.parametrize('stream', [False, True])
async def test_image_output(
    gemini_api_key: str, request_capture: RequestCapture, stream: bool, allow_model_requests: None
):

    model = GoogleInteractionsModel(
        'gemini-3.1-flash-image',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
    )
    agent = Agent(model, output_type=BinaryImage)
    if stream:
        async with agent.run_stream('Draw one small red square on a white background.') as result:
            image = await result.get_output()
    else:
        image = (await agent.run('Draw one small red square on a white background.')).output
    assert image.media_type.startswith('image/')
    assert image.data


MCP_MESSAGES: dict[bool, list[ModelMessage]] = {
    False: snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Call read_wiki_structure exactly once for pydantic/pydantic, then name one documentation topic from its result.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'read_wiki_structure',
                            'tool_args': {'repoName': 'pydantic/pydantic'},
                        },
                        tool_call_id='call_422424',
                        provider_name='google',
                        provider_details={
                            'id': 'call_422424',
                            'name': 'deepwiki:read_wiki_structure',
                            'type': 'function_call',
                            'signature': IsStr(),
                            'google_interactions_step_index': 0,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    NativeToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            '_meta': {'fastmcp': {'wrap_result': True}},
                            'content': [
                                {
                                    'type': 'text',
                                    'text': """\
Available pages for pydantic/pydantic:

- 1 Overview
  - 1.1 Getting Started
  - 1.2 Why Pydantic
- 2 Core Model System
  - 2.1 BaseModel
  - 2.2 Field System
  - 2.3 Model Configuration
  - 2.4 Internal Architecture
- 3 Type System
  - 3.1 Built-in and Constrained Types
  - 3.2 Network Types
  - 3.3 TypeAdapter
  - 3.4 Generics and Forward References
  - 3.5 Annotated Metadata and Custom Types
- 4 Validation and Serialization
  - 4.1 Validators
  - 4.2 Serializers
  - 4.3 JSON Conversion
  - 4.4 Error Handling
- 5 Schema Generation
  - 5.1 Core Schema Generation
  - 5.2 JSON Schema Generation
  - 5.3 Discriminated Unions
- 6 Advanced Features
  - 6.1 Dataclass Support
  - 6.2 Function Validation
  - 6.3 RootModel and Computed Fields
  - 6.4 Plugin System
  - 6.5 Mypy Integration
  - 6.6 Experimental Features
- 7 Development and Deployment
  - 7.1 Testing Framework
  - 7.2 CI/CD Pipeline
  - 7.3 Documentation System
  - 7.4 Versioning and Dependencies
  - 7.5 Release Process
- 8 Migration and Compatibility
  - 8.1 V1 to V2 Migration
  - 8.2 Backported Modules
- 9 Type System Internals
  - 9.1 Typing Utilities
  - 9.2 pydantic-core Rust Engine
- 10 Glossary\
""",
                                }
                            ],
                            'structuredContent': {'result': IsStr()},
                            'isError': False,
                        },
                        tool_call_id='call_422424',
                        timestamp=IsDatetime(),
                        provider_name='google',
                        provider_details={
                            'call_id': 'call_422424',
                            'is_error': False,
                            'name': 'deepwiki:read_wiki_structure',
                            'type': 'function_result',
                            'signature': IsStr(),
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 2, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='One documentation topic from the result is **Core Model System** (or specifically **2.1 BaseModel**).',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 3,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': 'bff86a9c24d92a5e3ad222e6a9d576338868a9489b2d9c407a68406cc5df598e',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={
                        'thoughts_tokens': 48,
                        'tool_use_tokens': 1930,
                        'input_text_tokens': 75,
                        'tool_use_text_tokens': 1930,
                    },
                    input_tokens=75,
                    output_tokens=94,
                    output_reasoning_tokens=48,
                    cost=Decimal('0.00040875'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {
                            'id': 'call_422424',
                            'name': 'deepwiki:read_wiki_structure',
                            'type': 'function_call',
                            'signature': IsStr(),
                        },
                        {
                            'call_id': 'call_422424',
                            'is_error': False,
                            'name': 'deepwiki:read_wiki_structure',
                            'type': 'function_result',
                            'signature': IsStr(),
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 75}],
                        'tool_use_tokens_by_modality': [{'modality': 'text', 'tokens': 1930}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 75,
                        'total_output_tokens': 46,
                        'total_thought_tokens': 48,
                        'total_tokens': 2099,
                        'total_tool_use_tokens': 1930,
                        'initial_input_tokens': 84,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Repeat that documentation topic without calling a tool.', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='**BaseModel** (under **Core Model System**)',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': '5c58ca8bae9fd6399b57187382b238877a0f668487b378c0a85e415c0b46ab52',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 0, 'tool_use_tokens': 0, 'input_text_tokens': 149},
                    input_tokens=149,
                    output_tokens=11,
                    cost=Decimal('0.00015300'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 149}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 149,
                        'total_output_tokens': 11,
                        'total_thought_tokens': 0,
                        'total_tokens': 160,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 1961,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
    True: snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Call read_wiki_structure exactly once for pydantic/pydantic, then name one documentation topic from its result.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='mcp_server:deepwiki',
                        args={
                            'action': 'call_tool',
                            'tool_name': 'read_wiki_structure',
                            'tool_args': {'repoName': 'pydantic/pydantic'},
                        },
                        tool_call_id='call_7506469182458035022',
                        provider_name='google',
                        provider_details={
                            'id': 'call_7506469182458035022',
                            'name': 'deepwiki:read_wiki_structure',
                            'server_name': 'deepwiki',
                            'type': 'mcp_server_tool_call',
                            'signature': IsStr(),
                            'google_interactions_step_index': 0,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    NativeToolReturnPart(
                        tool_name='mcp_server:deepwiki',
                        content={
                            '_meta': {'fastmcp': {'wrap_result': True}},
                            'content': [
                                {
                                    'type': 'text',
                                    'text': """\
Available pages for pydantic/pydantic:

- 1 Overview
  - 1.1 Getting Started
  - 1.2 Why Pydantic
- 2 Core Model System
  - 2.1 BaseModel
  - 2.2 Field System
  - 2.3 Model Configuration
  - 2.4 Internal Architecture
- 3 Type System
  - 3.1 Built-in and Constrained Types
  - 3.2 Network Types
  - 3.3 TypeAdapter
  - 3.4 Generics and Forward References
  - 3.5 Annotated Metadata and Custom Types
- 4 Validation and Serialization
  - 4.1 Validators
  - 4.2 Serializers
  - 4.3 JSON Conversion
  - 4.4 Error Handling
- 5 Schema Generation
  - 5.1 Core Schema Generation
  - 5.2 JSON Schema Generation
  - 5.3 Discriminated Unions
- 6 Advanced Features
  - 6.1 Dataclass Support
  - 6.2 Function Validation
  - 6.3 RootModel and Computed Fields
  - 6.4 Plugin System
  - 6.5 Mypy Integration
  - 6.6 Experimental Features
- 7 Development and Deployment
  - 7.1 Testing Framework
  - 7.2 CI/CD Pipeline
  - 7.3 Documentation System
  - 7.4 Versioning and Dependencies
  - 7.5 Release Process
- 8 Migration and Compatibility
  - 8.1 V1 to V2 Migration
  - 8.2 Backported Modules
- 9 Type System Internals
  - 9.1 Typing Utilities
  - 9.2 pydantic-core Rust Engine
- 10 Glossary\
""",
                                }
                            ],
                            'structuredContent': {'result': IsStr()},
                            'isError': False,
                        },
                        tool_call_id='call_7506469182458035022',
                        timestamp=IsDatetime(),
                        provider_name='google',
                        provider_details={
                            'call_id': 'call_7506469182458035022',
                            'name': 'deepwiki:read_wiki_structure',
                            'server_name': 'deepwiki',
                            'type': 'mcp_server_tool_result',
                            'signature': IsStr(),
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                        },
                    ),
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 2, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='One documentation topic from the result is **Core Model System** (specifically, **2.1 BaseModel**).',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 3,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': 'df9e13e07411667ca21829b368e12c0568932c68f2e569502b9d25fe2e399137',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={
                        'thoughts_tokens': 80,
                        'tool_use_tokens': 1931,
                        'input_text_tokens': 76,
                        'tool_use_text_tokens': 1931,
                    },
                    input_tokens=76,
                    output_tokens=126,
                    output_reasoning_tokens=80,
                    cost=Decimal('0.0005295'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {
                            'id': 'call_7506469182458035022',
                            'name': 'deepwiki:read_wiki_structure',
                            'server_name': 'deepwiki',
                            'type': 'mcp_server_tool_call',
                            'signature': IsStr(),
                        },
                        {
                            'call_id': 'call_7506469182458035022',
                            'name': 'deepwiki:read_wiki_structure',
                            'server_name': 'deepwiki',
                            'type': 'mcp_server_tool_result',
                            'signature': IsStr(),
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 76}],
                        'tool_use_tokens_by_modality': [{'modality': 'text', 'tokens': 1931}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 76,
                        'total_output_tokens': 46,
                        'total_thought_tokens': 80,
                        'total_tokens': 2133,
                        'total_tool_use_tokens': 1931,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Repeat that documentation topic without calling a tool.', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature=IsStr(),
                        provider_name='google',
                        provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
                    ),
                    TextPart(
                        content='**Core Model System** (specifically, **2.1 BaseModel**)',
                        provider_name='google',
                        provider_details={
                            'google_interactions_step_index': 1,
                            'google_interactions_content_index': 0,
                            'google_interactions_text_sha256': 'ed7c91c7e33a27ea371c93f9e49ad949c940ce4a4732fee6ab2d47a82ea9e62a',
                        },
                    ),
                ],
                usage=RequestUsage(
                    details={'thoughts_tokens': 36, 'tool_use_tokens': 0, 'input_text_tokens': 181},
                    input_tokens=181,
                    output_tokens=50,
                    output_reasoning_tokens=36,
                    cost=Decimal('0.00032325'),
                ),
                model_name='gemini-3.8-flash',
                model_id='google:gemini-3.8-flash',
                timestamp=IsDatetime(),
                provider_name='google',
                provider_url='https://generativelanguage.googleapis.com/',
                canonical_system='google',
                provider_details={
                    'status': 'completed',
                    'google_interactions_steps': [
                        {'signature': IsStr(), 'type': 'thought'},
                        {'content': [{'type': 'text'}], 'type': 'model_output'},
                    ],
                    'usage': {
                        'input_tokens_by_modality': [{'modality': 'text', 'tokens': 181}],
                        'total_cached_tokens': 0,
                        'total_input_tokens': 181,
                        'total_output_tokens': 14,
                        'total_thought_tokens': 36,
                        'total_tokens': 231,
                        'total_tool_use_tokens': 0,
                        'initial_input_tokens': 1993,
                    },
                    'service_tier': 'standard',
                },
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    ),
}


@pytest.mark.parametrize('stream', [False, True])
async def test_mcp(interactions_model: GoogleInteractionsModel, stream: bool, allow_model_requests: None):

    agent = Agent(
        interactions_model,
        capabilities=[
            NativeTool(
                MCPServerTool(id='deepwiki', url='https://mcp.deepwiki.com/mcp', allowed_tools=['read_wiki_structure'])
            )
        ],
    )
    prompt = 'Call read_wiki_structure exactly once for pydantic/pydantic, then name one documentation topic from its result.'
    if stream:
        async with agent.run_stream(prompt) as result:
            await result.get_output()
            history = result.all_messages()
    else:
        history = (await agent.run(prompt)).all_messages()
    final = await agent.run('Repeat that documentation topic without calling a tool.', message_history=history)
    assert final.all_messages() == MCP_MESSAGES[stream]


# A temporary FastMCP server, reachable only while recording. Its tools return an MCP image block (with
# `structuredContent`, which Google answers with a `$ref`), an audio block and an embedded PDF resource.
MCP_MEDIA_SERVER_URL = 'https://fool-inline-occupational-reed.trycloudflare.com/mcp'


@pytest.mark.parametrize('stream', [False, True])
async def test_mcp_media(interactions_model: GoogleInteractionsModel, stream: bool, allow_model_requests: None):
    agent = Agent(interactions_model, capabilities=[NativeTool(MCPServerTool(id='probe', url=MCP_MEDIA_SERVER_URL))])
    prompt = 'Call get_audio, get_pdf and get_image on the probe server, then say done.'
    if stream:
        async with agent.run_stream(prompt) as result:
            await result.get_output()
            history = result.all_messages()
    else:
        history = (await agent.run(prompt)).all_messages()
    returns = [
        part
        for message in history
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, NativeToolReturnPart)
    ]
    objects = TypeAdapter(dict[str, object])
    media: list[str] = []
    for part in returns:
        for block in TypeAdapter(list[object]).validate_python(objects.validate_python(part.content)['content']):
            item = objects.validate_python(block).get('resource') if isinstance(block, dict) else block
            if isinstance(item, BinaryContent):
                media.append(item.media_type)
    assert sorted(media) == snapshot(['application/pdf', 'audio/wav', 'image/png'])
    hidden = InstrumentationSettings(include_content=True, include_binary_content=False)
    telemetry = repr([message.otel_message_parts(hidden) for message in history if isinstance(message, ModelResponse)])
    assert not any(prefix in telemetry for prefix in ('UklGR', 'JVBERi0', 'iVBORw0KGgo'))
    # Google rejects its own `$ref` result on replay (googleapis/python-genai#3313); the rebuilt result is accepted.
    final = await agent.run('Which tools returned media? Answer without calling a tool.', message_history=history)
    assert final.output


@pytest.mark.parametrize('stream', [False, True])
async def test_file_search(interactions_model: GoogleInteractionsModel, stream: bool, allow_model_requests: None):

    client = interactions_model.client
    store = await client.aio.file_search_stores.create(config={'display_name': 'interactions-test'})
    assert store.name is not None
    try:
        await client.aio.file_search_stores.upload_to_file_search_store(
            file_search_store_name=store.name,
            file=io.BytesIO(b'The secret project code is emerald tiger.'),
            config={'mime_type': 'text/plain', 'display_name': 'secret.txt'},
        )
        agent = Agent(interactions_model, capabilities=[NativeTool(FileSearchTool(file_store_ids=[store.name]))])
        prompt = 'Use file_search to find the secret project code.'
        if stream:
            async with agent.run_stream(prompt) as result:
                assert 'emerald tiger' in (await result.get_output()).lower()
        else:
            assert 'emerald tiger' in (await agent.run(prompt)).output.lower()
    finally:
        await client.aio.file_search_stores.delete(name=store.name, config={'force': True})


@pytest.mark.parametrize('kind', ['document', 'audio', 'video'])
async def test_multimodal_input(
    interactions_model: GoogleInteractionsModel, kind: str, request: pytest.FixtureRequest, allow_model_requests: None
):
    content: BinaryContent = request.getfixturevalue(f'{kind}_content')
    assert (await Agent(interactions_model).run(['Summarize this file in one sentence.', content])).output


@pytest.mark.parametrize('source', ['binary', 'youtube'])
async def test_video_processing_controls(
    interactions_model: GoogleInteractionsModel,
    video_content: BinaryContent,
    request_capture: RequestCapture,
    source: str,
    allow_model_requests: None,
):
    video = VideoUrl('https://www.youtube.com/watch?v=dQw4w9WgXcQ') if source == 'youtube' else video_content
    video.vendor_metadata = {'start_offset': '0s', 'end_offset': '1s', 'fps': 1}
    settings: GoogleInteractionsModelSettings = {'google_video_resolution': MediaResolution.MEDIA_RESOLUTION_LOW}
    result = await Agent(interactions_model, model_settings=settings).run(
        ['Describe this video in one sentence.', video]
    )
    assert result.output
    steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(request_capture.body()['input'])
    content = TypeAdapter(list[dict[str, JsonValue]]).validate_python(steps[0]['content'])
    assert content[1]['processing'] == {'type': 'static', 'start_offset': '0s', 'end_offset': '1s', 'fps': 1}
    assert content[1]['resolution'] == 'low'
    if source == 'youtube':
        assert isinstance(video, VideoUrl)
        assert content[1]['uri'] == video.url
        assert 'data' not in content[1]


@pytest.mark.parametrize('stream', [False, True])
async def test_audio_output(
    gemini_api_key: str, request_capture: RequestCapture, stream: bool, allow_model_requests: None
):

    model = GoogleInteractionsModel(
        'gemini-3.8-flash-tts',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
    )
    settings: GoogleInteractionsModelSettings = {'google_interactions_response_format': {'type': 'audio'}}
    messages = [ModelRequest(parts=[UserPromptPart('Say hello in a friendly voice.')])]
    if stream:
        async with model_request_stream(model, messages, model_settings=settings) as result:
            async for _ in result:
                pass
            response = result.get()
    else:
        response = await model_request(model, messages, model_settings=settings)
    assert any(
        isinstance(part, FilePart) and part.content.media_type.startswith('audio/') and part.content.data
        for part in response.parts
    )


async def test_native_voice_configuration(
    gemini_api_key: str, request_capture: RequestCapture, allow_model_requests: None
):

    model = GoogleInteractionsModel(
        'gemini-3.8-flash-tts',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
    )
    settings: GoogleInteractionsModelSettings = {
        'google_interactions_response_format': {'type': 'audio'},
        'google_interactions_generation_config': {'speech_config': [{'voice': 'Kore', 'language': 'en-US'}]},
    }
    response = await model_request(model, [ModelRequest.user_text_prompt('Say hello.')], model_settings=settings)
    assert any(isinstance(part, FilePart) and part.content.data for part in response.parts)
    config = TypeAdapter(dict[str, JsonValue]).validate_python(request_capture.body()['generation_config'])
    assert config['speech_config'] == [{'voice': 'Kore', 'language': 'en-US'}]


@pytest.mark.parametrize(
    'model_name,kind,native_result',
    [pytest.param('gemini-3.8-flash', kind, True, id=kind) for kind in ('image', 'document', 'audio', 'video')]
    + [
        pytest.param('gemini-4-argon', kind, True, id=f'argon-{kind}')
        for kind in ('image', 'document', 'audio', 'video')
    ]
    + [pytest.param('gemini-3.8-flash', 'document', False, id='fallback-document')],
)
async def test_multimodal_tool_return(
    interactions_model: GoogleInteractionsModel,
    model_name: str,
    kind: str,
    native_result: bool,
    request: pytest.FixtureRequest,
    request_capture: RequestCapture,
    allow_model_requests: None,
):
    content: BinaryContent = request.getfixturevalue(f'{kind}_content')
    provider = interactions_model.provider
    assert provider is not None
    profile: GoogleModelProfile | None = (
        None if native_result else GoogleModelProfile(google_supported_mime_types_in_tool_returns=())
    )
    agent = Agent(GoogleInteractionsModel(model_name, provider=provider, profile=profile))

    @agent.tool_plain
    def get_file() -> BinaryContent:
        """Fetch the file to be summarized."""
        return content

    assert (await agent.run('Use get_file and summarize the file in one sentence.')).output
    steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(request_capture.body(index=-1)['input'])
    result = next(step for step in reversed(steps) if step.get('type') == 'function_result')
    returned = TypeAdapter(list[dict[str, JsonValue]]).validate_python(result['result'])
    if native_result:
        assert any(item.get('type') == kind and item.get('mime_type') == content.media_type for item in returned)
    else:
        assert all(item.get('type') == 'text' for item in returned)
        assert steps[-1]['type'] == 'user_input'
        files = TypeAdapter(list[dict[str, JsonValue]]).validate_python(steps[-1]['content'])
        assert any(item.get('type') == kind and item.get('mime_type') == content.media_type for item in files)
        assert 'get_file' in str(files[0]['text'])


CROSS_API_HISTORY: dict[str, ReplayExpectation] = {
    'to_interactions': ReplayExpectation(
        messages=snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Use code execution to compute 17 * 23, then call get_secret to get the secret phrase.',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        NativeToolCallPart(
                            tool_name='code_execution',
                            args={'code': 'print(17 * 23)\n', 'language': 'PYTHON', 'id': 'call_14053763028014284866'},
                            tool_call_id=IsStr(),
                            provider_name='google',
                            provider_details={'thought_signature': IsStr()},
                        ),
                        ToolCallPart(tool_name='get_secret', args={}, tool_call_id='call_14053763028014287047'),
                        NativeToolReturnPart(
                            tool_name='code_execution',
                            content={'outcome': 'OUTCOME_OK', 'output': '391\n', 'id': 'call_14053763028014284866'},
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='google',
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=52,
                        output_tokens=151,
                        output_reasoning_tokens=118,
                        details={'thoughts_tokens': 118, 'text_prompt_tokens': 52},
                        cost=Decimal('0.00060525'),
                        input_text_tokens=52,
                    ),
                    model_name='gemini-3.8-flash',
                    model_id='google:gemini-3.8-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='LQnuarOVG4e-ytMP_IrUiQ4',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_secret',
                            content='emerald tiger',
                            tool_call_id='call_14053763028014287047',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
The calculation $17 \\times 23$ yields **391**.

The secret phrase retrieved is **"emerald tiger"**.\
""",
                            provider_name='google',
                            provider_details={'thought_signature': IsStr()},
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=197,
                        output_tokens=61,
                        output_reasoning_tokens=32,
                        details={'thoughts_tokens': 32, 'text_prompt_tokens': 197},
                        cost=Decimal('0.00037650'),
                        input_text_tokens=197,
                    ),
                    model_name='gemini-3.8-flash',
                    model_id='google:gemini-3.8-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='MQnuauuwJa6rytMPgpr2iQ4',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[UserPromptPart(content='Repeat the secret phrase.', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            signature=IsStr(),
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 0,
                                'google_interactions_content_index': 0,
                            },
                        ),
                        TextPart(
                            content='The secret phrase is **emerald tiger**.',
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 1,
                                'google_interactions_content_index': 0,
                                'google_interactions_text_sha256': 'b190fc3d927af64b5a41654ba9934c431857c4d36dd16a94301fdf77b1a7a16d',
                            },
                        ),
                    ],
                    usage=RequestUsage(
                        details={'thoughts_tokens': 18, 'tool_use_tokens': 0, 'input_text_tokens': 258},
                        input_tokens=258,
                        output_tokens=26,
                        output_reasoning_tokens=18,
                        cost=Decimal('0.0002910'),
                    ),
                    model_name='gemini-3.8-flash',
                    model_id='google:gemini-3.8-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_details={
                        'status': 'completed',
                        'google_interactions_steps': [
                            {'signature': IsStr(), 'type': 'thought'},
                            {'content': [{'type': 'text'}], 'type': 'model_output'},
                        ],
                        'usage': {
                            'input_tokens_by_modality': [{'modality': 'text', 'tokens': 258}],
                            'total_cached_tokens': 0,
                            'total_input_tokens': 258,
                            'total_output_tokens': 8,
                            'total_thought_tokens': 18,
                            'total_tokens': 284,
                            'total_tool_use_tokens': 0,
                            'initial_input_tokens': 244,
                        },
                        'service_tier': 'standard',
                    },
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        requests=snapshot(
            [
                {
                    'contents': [
                        {
                            'parts': [
                                {
                                    'text': 'Use code execution to compute 17 * 23, then call get_secret to get the secret phrase.'
                                }
                            ],
                            'role': 'user',
                        }
                    ],
                    'tools': [
                        {'codeExecution': {}},
                        {
                            'functionDeclarations': [
                                {
                                    'description': 'Look up the secret phrase.',
                                    'name': 'get_secret',
                                    'parameters_json_schema': {
                                        'additionalProperties': False,
                                        'properties': {},
                                        'type': 'object',
                                    },
                                }
                            ]
                        },
                    ],
                },
                {
                    'contents': [
                        {
                            'parts': [
                                {
                                    'text': 'Use code execution to compute 17 * 23, then call get_secret to get the secret phrase.'
                                }
                            ],
                            'role': 'user',
                        },
                        {
                            'parts': [
                                {
                                    'executableCode': {
                                        'code': 'print(17 * 23)\n',
                                        'language': 'PYTHON',
                                        'id': 'call_14053763028014284866',
                                    },
                                    'thoughtSignature': IsStr(),
                                },
                                {
                                    'functionCall': {
                                        'id': 'call_14053763028014287047',
                                        'args': {},
                                        'name': 'get_secret',
                                    },
                                    'thoughtSignature': IsStr(),
                                },
                                {
                                    'codeExecutionResult': {
                                        'outcome': 'OUTCOME_OK',
                                        'output': '391\n',
                                        'id': 'call_14053763028014284866',
                                    }
                                },
                            ],
                            'role': 'model',
                        },
                        {
                            'parts': [
                                {
                                    'functionResponse': {
                                        'id': 'call_14053763028014287047',
                                        'name': 'get_secret',
                                        'response': {'return_value': 'emerald tiger'},
                                    }
                                }
                            ],
                            'role': 'user',
                        },
                    ],
                    'tools': [
                        {'codeExecution': {}},
                        {
                            'functionDeclarations': [
                                {
                                    'description': 'Look up the secret phrase.',
                                    'name': 'get_secret',
                                    'parameters_json_schema': {
                                        'additionalProperties': False,
                                        'properties': {},
                                        'type': 'object',
                                    },
                                }
                            ]
                        },
                    ],
                },
                {
                    'input': [
                        {
                            'type': 'user_input',
                            'content': [
                                {
                                    'type': 'text',
                                    'text': 'Use code execution to compute 17 * 23, then call get_secret to get the secret phrase.',
                                }
                            ],
                        },
                        {'type': 'thought', 'signature': IsStr()},
                        {
                            'type': 'function_call',
                            'id': 'call_14053763028014287047',
                            'name': 'get_secret',
                            'arguments': {},
                        },
                        {
                            'type': 'function_result',
                            'name': 'get_secret',
                            'call_id': 'call_14053763028014287047',
                            'is_error': False,
                            'result': [{'type': 'text', 'text': 'emerald tiger'}],
                        },
                        {'type': 'thought', 'signature': IsStr()},
                        {
                            'type': 'model_output',
                            'content': [
                                {
                                    'type': 'text',
                                    'text': """\
The calculation $17 \\times 23$ yields **391**.

The secret phrase retrieved is **"emerald tiger"**.\
""",
                                }
                            ],
                        },
                        {'type': 'user_input', 'content': [{'type': 'text', 'text': 'Repeat the secret phrase.'}]},
                    ],
                    'tools': [
                        {
                            'description': 'Look up the secret phrase.',
                            'name': 'get_secret',
                            'parameters': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                            'type': 'function',
                        },
                        {'type': 'code_execution'},
                    ],
                },
            ]
        ),
    ),
    'to_generate_content': ReplayExpectation(
        messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Call get_secret to get the secret phrase.', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            signature=IsStr(),
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 0,
                                'google_interactions_content_index': 0,
                            },
                        ),
                        ToolCallPart(
                            tool_name='get_secret',
                            args={},
                            tool_call_id='call_5558020211210808372',
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 1,
                                'google_interactions_content_index': 0,
                            },
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=38,
                        output_tokens=202,
                        output_reasoning_tokens=192,
                        details={'thoughts_tokens': 192, 'tool_use_tokens': 0, 'input_text_tokens': 38},
                        cost=Decimal('0.0007860'),
                    ),
                    model_name='gemini-3.8-flash',
                    model_id='google:gemini-3.8-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_details={
                        'status': 'requires_action',
                        'google_interactions_steps': [
                            {'signature': IsStr(), 'type': 'thought'},
                            {'id': 'call_5558020211210808372', 'name': 'get_secret', 'type': 'function_call'},
                        ],
                        'usage': {
                            'input_tokens_by_modality': [{'modality': 'text', 'tokens': 38}],
                            'total_cached_tokens': 0,
                            'total_input_tokens': 38,
                            'total_output_tokens': 10,
                            'total_thought_tokens': 192,
                            'total_tokens': 240,
                            'total_tool_use_tokens': 0,
                            'initial_input_tokens': 27,
                        },
                        'service_tier': 'standard',
                    },
                    finish_reason='tool_call',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_secret',
                            content='emerald tiger',
                            tool_call_id='call_5558020211210808372',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            signature=IsStr(),
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 0,
                                'google_interactions_content_index': 0,
                            },
                        ),
                        TextPart(
                            content='The secret phrase is **emerald tiger**.',
                            provider_name='google',
                            provider_details={
                                'google_interactions_step_index': 1,
                                'google_interactions_content_index': 0,
                                'google_interactions_text_sha256': 'b190fc3d927af64b5a41654ba9934c431857c4d36dd16a94301fdf77b1a7a16d',
                            },
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=248,
                        output_tokens=28,
                        output_reasoning_tokens=20,
                        details={'thoughts_tokens': 20, 'tool_use_tokens': 0, 'input_text_tokens': 248},
                        cost=Decimal('0.000291'),
                    ),
                    model_name='gemini-3.8-flash',
                    model_id='google:gemini-3.8-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_details={
                        'status': 'completed',
                        'google_interactions_steps': [
                            {'signature': IsStr(), 'type': 'thought'},
                            {'content': [{'type': 'text'}], 'type': 'model_output'},
                        ],
                        'usage': {
                            'input_tokens_by_modality': [{'modality': 'text', 'tokens': 248}],
                            'total_cached_tokens': 0,
                            'total_input_tokens': 248,
                            'total_output_tokens': 8,
                            'total_thought_tokens': 20,
                            'total_tokens': 276,
                            'total_tool_use_tokens': 0,
                            'initial_input_tokens': 238,
                        },
                        'service_tier': 'standard',
                    },
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[UserPromptPart(content='Repeat the secret phrase.', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The secret phrase is **emerald tiger**.',
                            provider_name='google',
                            provider_details={'thought_signature': IsStr()},
                        )
                    ],
                    usage=RequestUsage(
                        details={'thoughts_tokens': 24, 'text_prompt_tokens': 292},
                        input_tokens=292,
                        input_text_tokens=292,
                        output_tokens=32,
                        output_reasoning_tokens=24,
                        cost=Decimal('0.000339'),
                    ),
                    model_name='gemini-3.8-flash',
                    model_id='google:gemini-3.8-flash',
                    timestamp=IsDatetime(),
                    provider_name='google',
                    provider_url='https://generativelanguage.googleapis.com/',
                    canonical_system='google',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='MKTsavHfGvva0_gP4-uU-Qk',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        requests=snapshot(
            [
                {
                    'input': [
                        {
                            'type': 'user_input',
                            'content': [{'type': 'text', 'text': 'Call get_secret to get the secret phrase.'}],
                        }
                    ],
                    'tools': [
                        {
                            'description': 'Look up the secret phrase.',
                            'name': 'get_secret',
                            'parameters': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                            'type': 'function',
                        }
                    ],
                },
                {
                    'input': [
                        {
                            'type': 'user_input',
                            'content': [{'type': 'text', 'text': 'Call get_secret to get the secret phrase.'}],
                        },
                        {'signature': IsStr(), 'type': 'thought'},
                        {
                            'id': 'call_5558020211210808372',
                            'name': 'get_secret',
                            'type': 'function_call',
                            'arguments': {},
                        },
                        {
                            'type': 'function_result',
                            'name': 'get_secret',
                            'call_id': 'call_5558020211210808372',
                            'is_error': False,
                            'result': [{'type': 'text', 'text': 'emerald tiger'}],
                        },
                    ],
                    'tools': [
                        {
                            'description': 'Look up the secret phrase.',
                            'name': 'get_secret',
                            'parameters': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                            'type': 'function',
                        }
                    ],
                },
                {
                    'contents': [
                        {'parts': [{'text': 'Call get_secret to get the secret phrase.'}], 'role': 'user'},
                        {
                            'parts': [
                                {
                                    'functionCall': {
                                        'id': 'call_5558020211210808372',
                                        'args': {},
                                        'name': 'get_secret',
                                    },
                                    'thoughtSignature': IsStr(),
                                }
                            ],
                            'role': 'model',
                        },
                        {
                            'parts': [
                                {
                                    'functionResponse': {
                                        'id': 'call_5558020211210808372',
                                        'name': 'get_secret',
                                        'response': {'return_value': 'emerald tiger'},
                                    }
                                }
                            ],
                            'role': 'user',
                        },
                        {
                            'parts': [
                                {
                                    'text': 'The secret phrase is **emerald tiger**.',
                                    'thoughtSignature': IsStr(),
                                }
                            ],
                            'role': 'model',
                        },
                        {'parts': [{'text': 'Repeat the secret phrase.'}], 'role': 'user'},
                    ],
                    'tools': [
                        {
                            'functionDeclarations': [
                                {
                                    'description': 'Look up the secret phrase.',
                                    'name': 'get_secret',
                                    'parameters_json_schema': {
                                        'additionalProperties': False,
                                        'properties': {},
                                        'type': 'object',
                                    },
                                }
                            ]
                        }
                    ],
                },
            ]
        ),
    ),
}


async def test_foreign_tool_call_history(
    interactions_model: GoogleInteractionsModel, request_capture: RequestCapture, allow_model_requests: None
):
    """Continue another provider's unsigned tool loop: Gemini 3 rejects it unless the first call carries the placeholder."""
    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Use city_temperature to find the temperature in Paris.'),
        ModelResponse(parts=[ToolCallPart('city_temperature', {'city': 'Paris'}, 'call_1')], provider_name='openai'),
        ModelRequest(parts=[ToolReturnPart('city_temperature', '21', 'call_1')]),
    ]
    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name='city_temperature',
                description='Look up the current temperature in the requested city.',
                parameters_json_schema={
                    'type': 'object',
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                },
            )
        ]
    )
    response = await model_request(interactions_model, history, model_request_parameters=params)
    assert request_capture.body(index=-1)['input'] == snapshot(
        [
            {
                'type': 'user_input',
                'content': [{'type': 'text', 'text': 'Use city_temperature to find the temperature in Paris.'}],
            },
            {
                'type': 'function_call',
                'id': 'call_1',
                'name': 'city_temperature',
                'arguments': {'city': 'Paris'},
                'signature': 'c2tpcF90aG91Z2h0X3NpZ25hdHVyZV92YWxpZGF0b3I=',
            },
            {
                'type': 'function_result',
                'name': 'city_temperature',
                'call_id': 'call_1',
                'is_error': False,
                'result': [{'type': 'text', 'text': '21'}],
            },
        ]
    )
    assert response.parts == snapshot(
        [
            ThinkingPart(
                content='',
                signature=IsStr(),
                provider_name='google',
                provider_details={'google_interactions_step_index': 0, 'google_interactions_content_index': 0},
            ),
            TextPart(
                content='The current temperature in Paris is 21°C.',
                provider_name='google',
                provider_details={
                    'google_interactions_step_index': 1,
                    'google_interactions_content_index': 0,
                    'google_interactions_text_sha256': 'faf1b9798e9c2858114b84dce9d05c5e7a0519a99f6ce62ea3ff86a7701a68be',
                },
            ),
        ]
    )


@pytest.mark.parametrize('direction', ['to_interactions', 'to_generate_content'])
async def test_cross_api_history(
    interactions_model: GoogleInteractionsModel,
    request_capture: RequestCapture,
    direction: str,
    allow_model_requests: None,
):
    """GenerateContent native parts carry no Interactions steps, so continuing on Interactions omits them."""

    provider = interactions_model.provider
    assert provider is not None
    generate_content = GoogleModel(interactions_model.model_name, provider=provider)
    source, target = (
        (generate_content, interactions_model)
        if direction == 'to_interactions'
        else (interactions_model, generate_content)
    )
    native = direction == 'to_interactions'
    agent = Agent(source, capabilities=[NativeTool(CodeExecutionTool())] if native else [])

    @agent.tool_plain
    def get_secret() -> str:
        """Look up the secret phrase."""
        return 'emerald tiger'

    first = await agent.run(
        'Use code execution to compute 17 * 23, then call get_secret to get the secret phrase.'
        if native
        else 'Call get_secret to get the secret phrase.'
    )
    history = ModelMessagesTypeAdapter.validate_json(first.all_messages_json())
    final = await agent.run('Repeat the secret phrase.', model=target, message_history=history)
    expected = CROSS_API_HISTORY[direction]
    assert final.all_messages() == expected.messages
    assert [conversation_and_tools(body) for body in request_capture.bodies()] == expected.requests


async def test_vertex(vertex_provider: GoogleProvider, allow_model_requests: None):
    agent = Agent(GoogleInteractionsModel('gemini-3.8-flash', provider=vertex_provider))

    @agent.tool_plain
    def get_secret() -> str:
        """Look up the secret phrase."""
        return 'emerald tiger'

    result = await agent.run('Use get_secret and tell me the secret phrase.')
    assert 'emerald tiger' in result.output.lower()
    history = ModelMessagesTypeAdapter.validate_json(result.all_messages_json())
    async with agent.run_stream('Repeat the secret phrase.', message_history=history) as stream:
        assert 'emerald tiger' in (await stream.get_output()).lower()


async def test_image_uri_output(gemini_api_key: str, request_capture: RequestCapture, allow_model_requests: None):

    settings: GoogleInteractionsModelSettings = {
        'google_interactions_response_format': {'type': 'image', 'delivery': 'uri'}
    }
    model = GoogleInteractionsModel(
        'gemini-3.1-flash-image',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
        settings=settings,
    )

    with pytest.raises(ModelHTTPError, match='Image delivery mode is not supported'):
        await Agent(model, output_type=BinaryImage).run('Draw one small red square on a white background.')


async def test_native_url_input(
    interactions_model: GoogleInteractionsModel, request_capture: RequestCapture, allow_model_requests: None
):
    image = ImageUrl('https://storage.googleapis.com/generativeai-downloads/images/scones.jpg')
    result = await Agent(interactions_model).run(['Describe the baked goods in this picture.', image])
    assert any(word in result.output.lower() for word in ('scone', 'pastr', 'biscuit', 'bread'))
    steps = TypeAdapter(list[dict[str, JsonValue]]).validate_python(request_capture.body()['input'])
    content = TypeAdapter(list[dict[str, JsonValue]]).validate_python(steps[0]['content'])
    assert content[-1] == {'type': 'image', 'uri': image.url, 'mime_type': 'image/jpeg'}


async def test_video_uri_output_requires_storage(
    gemini_api_key: str, request_capture: RequestCapture, allow_model_requests: None
):

    settings: GoogleInteractionsModelSettings = {
        'google_interactions_response_format': {'type': 'video', 'delivery': 'uri'}
    }
    model = GoogleInteractionsModel(
        'gemini-omni-1.2-flash',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
        settings=settings,
    )
    with pytest.raises(ModelHTTPError, match='store=true is required'):
        await Agent(model).run('Create a short video of a red ball.')


async def test_native_settings(
    interactions_model: GoogleInteractionsModel, request_capture: RequestCapture, allow_model_requests: None
):

    settings: GoogleInteractionsModelSettings = {
        'thinking': 'low',
        'google_interactions_safety_settings': [{'type': 'harassment', 'threshold': 'block_only_high'}],
        'google_labels': {'purpose': 'test'},
        'service_tier': 'default',
    }
    result = await Agent(interactions_model, model_settings=settings).run('Reply hello.')
    assert result.output
    body = request_capture.body()
    assert body['generation_config'] == {'thinking_level': 'low', 'thinking_summaries': 'auto'}
    assert body['safety_settings'] == settings['google_interactions_safety_settings']
    assert body['labels'] == {'purpose': 'test'}
    assert body['service_tier'] == 'standard'


async def test_google_maps(interactions_model: GoogleInteractionsModel, allow_model_requests: None):

    settings: GoogleInteractionsModelSettings = {'google_interactions_native_tools': [{'type': 'google_maps'}]}
    result = await Agent(interactions_model, model_settings=settings).run(
        'Use Google Maps to find the street address of the Eiffel Tower in Paris.'
    )
    assert result.output
    assert any(
        isinstance(message, ModelResponse)
        and any(isinstance(part, NativeToolCallPart) and part.tool_name == 'google_maps' for part in message.parts)
        for message in result.all_messages()
    )
    places = [
        item.get('places')
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, NativeToolReturnPart) and part.tool_name == 'google_maps'
        for item in TypeAdapter(list[dict[str, JsonValue]]).validate_python(part.content)
    ]
    assert any(places)


async def test_image_generation_tool_configuration(
    gemini_api_key: str, request_capture: RequestCapture, allow_model_requests: None
):

    model = GoogleInteractionsModel(
        'gemini-3.1-flash-image',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=120)),
    )
    agent = Agent(
        model,
        output_type=BinaryImage,
        capabilities=[NativeTool(ImageGenerationTool(aspect_ratio='16:9', size='1K', output_format='jpeg'))],
    )
    result = await agent.run('Draw a simple red circle on a white background.')
    assert result.output.data and result.output.media_type == 'image/jpeg'
    assert request_capture.body()['response_format'] == {
        'type': 'image',
        'aspect_ratio': '16:9',
        'image_size': '1K',
        'mime_type': 'image/jpeg',
    }


@pytest.fixture
def pptx_content(assets_path: Path) -> BinaryContent:
    return BinaryContent(
        (assets_path / 'dummy.pptx').read_bytes(),
        media_type='application/vnd.openxmlformats-officedocument.presentationml.presentation',
    )


async def test_pptx_input(
    interactions_model: GoogleInteractionsModel, pptx_content: BinaryContent, allow_model_requests: None
):
    result = await Agent(interactions_model).run(['Describe the text on this slide.', pptx_content])
    assert 'dummy' in result.output.lower()


@pytest.mark.parametrize('kind', ['music', 'video'])
async def test_generated_media(
    gemini_api_key: str, request_capture: RequestCapture, kind: str, allow_model_requests: None
):

    model = GoogleInteractionsModel(
        'lyria-3-clip-preview' if kind == 'music' else 'gemini-omni-1.1-flash',
        provider=GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=180)),
    )
    settings: GoogleInteractionsModelSettings = {
        'google_interactions_response_format': {'type': 'audio'}
        if kind == 'music'
        else {'type': 'video', 'resolution': '360p'}
    }
    prompt = (
        'An instrumental acoustic guitar piece.'
        if kind == 'music'
        else 'A red ball rolling across a plain white floor, fixed camera.'
    )
    response = await model_request(model, [ModelRequest.user_text_prompt(prompt)], model_settings=settings)
    files = [part for part in response.parts if isinstance(part, FilePart)]
    assert files and files[0].content.data
    assert files[0].content.media_type.startswith('audio/' if kind == 'music' else 'video/')
    assert request_capture.body()['store'] is False


@pytest.mark.parametrize('thinking', [False, 'minimal'])
@pytest.mark.parametrize('stream', [False, True])
async def test_minimal_thinking_resolves_to_supported_level(
    interactions_model: GoogleInteractionsModel,
    request_capture: RequestCapture,
    thinking: ThinkingEffort | bool,
    stream: bool,
    allow_model_requests: None,
):
    # Verified against the public Interactions API: this model rejects MINIMAL and accepts LOW.

    settings: GoogleInteractionsModelSettings = {'thinking': thinking}
    agent = Agent(interactions_model, model_settings=settings)
    if stream:
        async with agent.run_stream('Reply exactly OK.') as result:
            assert await result.get_output()
    else:
        assert (await agent.run('Reply exactly OK.')).output
    assert request_capture.body()['generation_config'] == {
        'thinking_level': 'low',
        'thinking_summaries': 'none' if thinking is False else 'auto',
    }
