from __future__ import annotations as _annotations

import base64
import datetime
import json
import os
import re
import tempfile
from collections.abc import AsyncIterator
from datetime import date, timezone
from typing import Any

import pytest
from httpx import AsyncClient as HttpxAsyncClient, Timeout
from inline_snapshot import Is, snapshot
from pydantic import BaseModel, Field
from pytest_mock import MockerFixture
from typing_extensions import TypedDict

from pydantic_ai import (
    AgentRunResult,
    AgentRunResultEvent,
    AgentStreamEvent,
    AudioUrl,
    BinaryContent,
    BinaryImage,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UsageLimitExceeded,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import (
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    UrlContextTool,  # pyright: ignore[reportDeprecated]
    WebFetchTool,
    WebSearchTool,
)
from pydantic_ai.exceptions import (
    ContentFilterError,
    ModelAPIError,
    ModelHTTPError,
    ModelRetry,
    UserError,
)
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
)
from pydantic_ai.models import DEFAULT_HTTP_TIMEOUT, ModelRequestParameters
from pydantic_ai.output import NativeOutput, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from ..conftest import IsDatetime, IsInstance, IsNow, IsStr, try_import
from ..parts_from_messages import part_types_from_messages

with try_import() as imports_successful:
    from google.genai import errors
    from google.genai.types import (
        FinishReason as GoogleFinishReason,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
        HarmBlockThreshold,
        HarmCategory,
        MediaModality,
        ModalityTokenCount,
    )

    from pydantic_ai.models.google import (
        GeminiStreamedResponse,
        GoogleModel,
        GoogleModelSettings,
        _content_model_response,  # pyright: ignore[reportPrivateUsage]
        _metadata_as_usage,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

if not imports_successful():
    # Define placeholder errors module so parametrize decorators can be parsed
    from types import SimpleNamespace

    errors = SimpleNamespace(ServerError=Exception, ClientError=Exception, APIError=Exception)

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
]


@pytest.fixture()
def google_provider(gemini_api_key: str) -> GoogleProvider:
    return GoogleProvider(api_key=gemini_api_key)


async def test_google_model(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-flash', provider=google_provider)
    assert model.base_url == 'https://generativelanguage.googleapis.com/'
    assert model.system == 'google-gla'
    agent = Agent(model=model, instructions='You are a chatbot.')

    result = await agent.run('Hello!')
    assert result.output == snapshot('Hello! How can I help you today?')
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=9,
            output_tokens=43,
            details={'thoughts_tokens': 34, 'text_prompt_tokens': 9},
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello!',
                        timestamp=IsDatetime(),
                    ),
                ],
                instructions='You are a chatbot.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Hello! How can I help you today?')],
                usage=RequestUsage(
                    input_tokens=9, output_tokens=43, details={'thoughts_tokens': 34, 'text_prompt_tokens': 9}
                ),
                model_name='gemini-2.5-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_model_structured_output(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', retries=5)

    class Response(TypedDict):
        temperature: str
        date: date
        city: str

    @agent.tool_plain
    async def temperature(city: str, date: date) -> str:
        """Get the temperature in a city on a specific date.

        Args:
            city: The city name.
            date: The date.

        Returns:
            The temperature in degrees Celsius.
        """
        return '30°C'

    result = await agent.run('What was the temperature in London 1st January 2022?', output_type=Response)
    assert result.output == snapshot({'temperature': '30°C', 'date': date(2022, 1, 1), 'city': 'London'})
    assert result.usage() == snapshot(
        RunUsage(
            requests=2,
            input_tokens=160,
            output_tokens=35,
            tool_calls=1,
            details={'text_prompt_tokens': 160, 'text_candidates_tokens': 35},
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What was the temperature in London 1st January 2022?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful chatbot.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='temperature', args={'date': '2022-01-01', 'city': 'London'}, tool_call_id=IsStr()
                    )
                ],
                usage=RequestUsage(
                    input_tokens=69,
                    output_tokens=14,
                    details={'text_candidates_tokens': 14, 'text_prompt_tokens': 69},
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='temperature', content='30°C', tool_call_id=IsStr(), timestamp=IsDatetime()
                    )
                ],
                instructions='You are a helpful chatbot.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'temperature': '30°C', 'date': '2022-01-01', 'city': 'London'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(
                    input_tokens=91,
                    output_tokens=21,
                    details={'text_candidates_tokens': 21, 'text_prompt_tokens': 91},
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_model_stream(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.0-flash-exp', provider=google_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'temperature': 0.0})
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_output()
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                assert response == snapshot(
                    ModelResponse(
                        parts=[TextPart(content='The capital of France is Paris.\n')],
                        usage=RequestUsage(
                            input_tokens=13,
                            output_tokens=8,
                            details={'text_prompt_tokens': 13, 'text_candidates_tokens': 8},
                        ),
                        model_name='gemini-2.0-flash-exp',
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                        provider_url='https://generativelanguage.googleapis.com/',
                        provider_details={'finish_reason': 'STOP'},
                        provider_response_id=IsStr(),
                        finish_reason='stop',
                    )
                )
    assert data == snapshot('The capital of France is Paris.\n')


async def test_google_model_builtin_code_execution_stream(
    allow_model_requests: None,
    google_provider: GoogleProvider,
):
    """Test Gemini streaming only code execution result or executable_code."""
    model = GoogleModel('gemini-2.5-pro', provider=google_provider)
    agent = Agent(
        model=model,
        instructions='Be concise and always use Python to do calculations no matter how small.',
        builtin_tools=[CodeExecutionTool()],
    )

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='what is 65465-6544 * 65464-6+1.02255') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='what is 65465-6544 * 65464-6+1.02255',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='Be concise and always use Python to do calculations no matter how small.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
    result = 65465 - 6544 * 65464 - 6 + 1.02255
    print(result)
    \
""",
                            'language': 'PYTHON',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
# Calculate the expression 65465-6544 * 65464-6+1.02255
result = 65465 - 6544 * 65464 - 6 + 1.02255
print(result)\
""",
                            'language': 'PYTHON',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(content='The result is -428,330,955.97745.'),
                ],
                usage=RequestUsage(
                    input_tokens=46,
                    output_tokens=1429,
                    details={
                        'thoughts_tokens': 396,
                        'tool_use_prompt_tokens': 901,
                        'text_prompt_tokens': 46,
                        'text_tool_use_prompt_tokens': 901,
                    },
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={
                        'code': """\
    result = 65465 - 6544 * 65464 - 6 + 1.02255
    print(result)
    \
""",
                        'language': 'PYTHON',
                    },
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                ),
            ),
            PartEndEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={
                        'code': """\
    result = 65465 - 6544 * 65464 - 6 + 1.02255
    print(result)
    \
""",
                        'language': 'PYTHON',
                    },
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n'},
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=2,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={
                        'code': """\
# Calculate the expression 65465-6544 * 65464-6+1.02255
result = 65465 - 6544 * 65464 - 6 + 1.02255
print(result)\
""",
                        'language': 'PYTHON',
                    },
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                ),
                previous_part_kind='builtin-tool-return',
            ),
            PartEndEvent(
                index=2,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={
                        'code': """\
# Calculate the expression 65465-6544 * 65464-6+1.02255
result = 65465 - 6544 * 65464 - 6 + 1.02255
print(result)\
""",
                        'language': 'PYTHON',
                    },
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n'},
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=4, part=TextPart(content='The result is'), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta=' -428,330,955.977')),
            PartDeltaEvent(index=4, delta=TextPartDelta(content_delta='45.')),
            PartEndEvent(index=4, part=TextPart(content='The result is -428,330,955.97745.')),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={
                        'code': """\
    result = 65465 - 6544 * 65464 - 6 + 1.02255
    print(result)
    \
""",
                        'language': 'PYTHON',
                    },
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n'},
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                )
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={
                        'code': """\
# Calculate the expression 65465-6544 * 65464-6+1.02255
result = 65465 - 6544 * 65464 - 6 + 1.02255
print(result)\
""",
                        'language': 'PYTHON',
                    },
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n'},
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                )
            ),
        ]
    )


async def test_google_model_retry(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-pro', provider=google_provider)
    agent = Agent(
        model=model, system_prompt='You are a helpful chatbot.', model_settings={'temperature': 0.0}, retries=2
    )

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        """Get the capital of a country.

        Args:
            country: The country name.
        """
        if country == 'La France':
            return 'Paris'
        else:
            raise ModelRetry('The country is not supported. Use "La France" instead.')

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a helpful chatbot.', timestamp=IsDatetime()),
                    UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime()),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_capital',
                        args={'country': 'France'},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=57, output_tokens=139, details={'thoughts_tokens': 124, 'text_prompt_tokens': 57}
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='The country is not supported. Use "La France" instead.',
                        tool_name='get_capital',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_capital',
                        args={'country': 'La France'},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=109, output_tokens=215, details={'thoughts_tokens': 199, 'text_prompt_tokens': 109}
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_capital',
                        content='Paris',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Paris',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=142, output_tokens=98, details={'thoughts_tokens': 97, 'text_prompt_tokens': 142}
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_model_max_tokens(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'max_tokens': 5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is')


async def test_google_model_top_p(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'top_p': 0.5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.\n')


async def test_google_model_thinking_config(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-pro-preview-03-25', provider=google_provider)
    settings = GoogleModelSettings(google_thinking_config={'include_thoughts': False})
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')


async def test_google_model_gla_labels_raises_value_error(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)
    settings = GoogleModelSettings(google_labels={'environment': 'test', 'team': 'analytics'})
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=settings)

    # Raises before any request is made.
    with pytest.raises(ValueError, match='labels parameter is not supported in Gemini API.'):
        await agent.run('What is the capital of France?')


async def test_google_model_vertex_provider(
    allow_model_requests: None, vertex_provider: GoogleProvider
):  # pragma: lax no cover
    model = GoogleModel('gemini-2.0-flash', provider=vertex_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.')
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.\n')


async def test_google_model_vertex_labels(
    allow_model_requests: None, vertex_provider: GoogleProvider
):  # pragma: lax no cover
    model = GoogleModel('gemini-2.0-flash', provider=vertex_provider)
    settings = GoogleModelSettings(google_labels={'environment': 'test', 'team': 'analytics'})
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is Paris.\n')


async def test_google_model_iter_stream(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.')

    @agent.tool_plain
    async def get_capital(country: str) -> str:
        """Get the capital of a country.

        Args:
            country: The country name.
        """
        return 'Paris'  # pragma: lax no cover

    @agent.tool_plain
    async def get_temperature(city: str) -> str:
        """Get the temperature in a city.

        Args:
            city: The city name.
        """
        return '30°C'

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the temperature of the capital of France?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='get_capital', args={'country': 'France'}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='get_capital', args={'country': 'France'}, tool_call_id=IsStr()),
            ),
            IsInstance(FunctionToolCallEvent),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_capital', content='Paris', tool_call_id=IsStr(), timestamp=IsDatetime()
                )
            ),
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='get_temperature', args={'city': 'Paris'}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='get_temperature', args={'city': 'Paris'}, tool_call_id=IsStr()),
            ),
            IsInstance(FunctionToolCallEvent),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_temperature', content='30°C', tool_call_id=IsStr(), timestamp=IsDatetime()
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='The temperature in Paris')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' is 30°C.\n')),
            PartEndEvent(index=0, part=TextPart(content='The temperature in Paris is 30°C.\n')),
        ]
    )


async def test_google_model_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.output == snapshot('The fruit in the image is a kiwi.')


async def test_google_model_video_as_binary_content_input(
    allow_model_requests: None, video_content: BinaryContent, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(['Explain me this video', video_content])
    assert result.output == snapshot("""\
Okay! It looks like the image shows a camera monitor, likely used for professional or semi-professional video recording. \n\

Here's what I can gather from the image:

*   **Camera Monitor:** The central element is a small screen attached to a camera rig (tripod and probably camera body). These monitors are used to provide a larger, clearer view of what the camera is recording, aiding in focus, composition, and exposure adjustments.
*   **Scene on Monitor:** The screen shows an image of what appears to be a rocky mountain path or canyon with a snow capped mountain in the distance.
*   **Background:** The background is blurred, likely the same scene as on the camera monitor.

Let me know if you want me to focus on any specific aspect or detail!\
""")


async def test_google_model_video_as_binary_content_input_with_vendor_metadata(
    allow_model_requests: None, video_content: BinaryContent, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')
    video_content.vendor_metadata = {'start_offset': '2s', 'end_offset': '10s'}

    result = await agent.run(['Explain me this video', video_content])
    assert result.output == snapshot("""\
Okay, I can describe what is visible in the image.

The image shows a camera setup in an outdoor setting. The camera is mounted on a tripod and has an external monitor attached to it. The monitor is displaying a scene that appears to be a desert landscape with rocky formations and mountains in the background. The foreground and background of the overall image, outside of the camera monitor, is also a blurry, desert landscape. The colors in the background are warm and suggest either sunrise, sunset, or reflected light off the rock formations.

It looks like someone is either reviewing footage on the monitor, or using it as an aid for framing the shot.\
""")


async def test_google_model_image_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(
        [
            'What is this vegetable?',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == snapshot('That is a potato.')


async def test_google_model_video_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(
        [
            'Explain me this video',
            VideoUrl(url='https://github.com/pydantic/pydantic-ai/raw/refs/heads/main/tests/assets/small_video.mp4'),
        ]
    )
    assert result.output == snapshot("""\
Certainly! Based on the image you sent, it appears to be a setup for filming or photography. \n\

Here's what I can observe:

*   **Camera Monitor:** There is a monitor mounted on a tripod, displaying a shot of a canyon or mountain landscape.
*   **Camera/Recording Device:** Below the monitor, there is a camera or some other kind of recording device.
*   **Landscape Backdrop:** In the background, there is a similar-looking landscape to what's being displayed on the screen.

In summary, it looks like the image shows a camera setup, perhaps in the process of filming, with a monitor to review the footage.\
""")


async def test_google_model_youtube_video_url_input_with_vendor_metadata(
    allow_model_requests: None, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(
        [
            'Explain me this video',
            VideoUrl(
                url='https://youtu.be/lCdaVNyHtjU',
                vendor_metadata={'fps': 0.2},
            ),
        ]
    )
    assert result.output == snapshot("""\
Okay, based on the image, here's what I can infer:

*   **A camera monitor is mounted on top of a camera.**
*   **The monitor's screen is on, displaying a view of the rocky mountains.**
*   **This setting suggests a professional video shoot.**

If you'd like a more detailed explanation, please provide additional information about the video.\
""")


async def test_google_model_document_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot('The document appears to be a dummy PDF file.\n')


async def test_google_model_text_document_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot(
        'The main content of the TXT file is an explanation of the placeholder name "John Doe" (and related variations) and its usage in legal contexts, popular culture, and other situations where the identity of a person is unknown or needs to be withheld. The document also includes the purpose of the file and other file type information.\n'
    )


async def test_google_model_text_as_binary_content_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    text_content = BinaryContent(data=b'This is a test document.', media_type='text/plain')

    result = await agent.run(['What is the main content on this document?', text_content])
    assert result.output == snapshot('The main content of the document is that it is a test document.\n')


async def test_google_model_instructions(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    def instructions() -> str:
        return 'You are a helpful assistant.'

    agent = Agent(m, instructions=instructions)

    result = await agent.run('What is the capital of France?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.\n')],
                usage=RequestUsage(
                    input_tokens=13, output_tokens=8, details={'text_candidates_tokens': 8, 'text_prompt_tokens': 13}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_model_multiple_documents_in_history(
    allow_model_requests: None, google_provider: GoogleProvider, document_content: BinaryContent
):
    m = GoogleModel(model_name='gemini-2.0-flash', provider=google_provider)
    agent = Agent(model=m)

    result = await agent.run(
        'What is in the documents?',
        message_history=[
            ModelRequest(
                parts=[UserPromptPart(content=['Here is a PDF document: ', document_content])], timestamp=IsDatetime()
            ),
            ModelResponse(parts=[TextPart(content='foo bar')]),
            ModelRequest(
                parts=[UserPromptPart(content=['Here is another PDF document: ', document_content])],
                timestamp=IsDatetime(),
            ),
            ModelResponse(parts=[TextPart(content='foo bar 2')]),
        ],
    )

    assert result.output == snapshot('Both documents contain the text "Dummy PDF file" at the top of the page.')


async def test_google_model_safety_settings(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-1.5-flash', provider=google_provider)
    settings = GoogleModelSettings(
        google_safety_settings=[
            {
                'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                'threshold': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            }
        ]
    )
    agent = Agent(m, instructions='You hate the world!', model_settings=settings)

    with pytest.raises(
        ContentFilterError,
        match="Content filter triggered. Finish reason: 'SAFETY'",
    ) as exc_info:
        await agent.run('Tell me a joke about a Brazilians.')

    assert exc_info.value.body is not None
    body_json = json.loads(exc_info.value.body)

    assert body_json == snapshot(
        [
            {
                'parts': [],
                'usage': {
                    'input_tokens': 14,
                    'cache_write_tokens': 0,
                    'cache_read_tokens': 0,
                    'output_tokens': 0,
                    'input_audio_tokens': 0,
                    'cache_audio_read_tokens': 0,
                    'output_audio_tokens': 0,
                    'details': {'text_prompt_tokens': 14},
                },
                'model_name': 'gemini-1.5-flash',
                'timestamp': IsStr(),
                'kind': 'response',
                'provider_name': 'google-gla',
                'provider_url': 'https://generativelanguage.googleapis.com/',
                'provider_details': {
                    'finish_reason': 'SAFETY',
                    'safety_ratings': [
                        {
                            'blocked': True,
                            'category': 'HARM_CATEGORY_HATE_SPEECH',
                            'overwrittenThreshold': None,
                            'probability': 'LOW',
                            'probabilityScore': None,
                            'severity': None,
                            'severityScore': None,
                        },
                        {
                            'blocked': None,
                            'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                            'overwrittenThreshold': None,
                            'probability': 'NEGLIGIBLE',
                            'probabilityScore': None,
                            'severity': None,
                            'severityScore': None,
                        },
                        {
                            'blocked': None,
                            'category': 'HARM_CATEGORY_HARASSMENT',
                            'overwrittenThreshold': None,
                            'probability': 'NEGLIGIBLE',
                            'probabilityScore': None,
                            'severity': None,
                            'severityScore': None,
                        },
                        {
                            'blocked': None,
                            'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                            'overwrittenThreshold': None,
                            'probability': 'NEGLIGIBLE',
                            'probabilityScore': None,
                            'severity': None,
                            'severityScore': None,
                        },
                    ],
                },
                'provider_response_id': IsStr(),
                'finish_reason': 'content_filter',
                'run_id': IsStr(),
                'metadata': None,
            }
        ]
    )


async def test_google_model_web_search_tool(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-pro', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.', builtin_tools=[WebSearchTool()])

    result = await agent.run('What is the weather in San Francisco today?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in San Francisco today?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful chatbot.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['weather in San Francisco today']},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'domain': None,
                                'title': 'Weather information for San Francisco, CA, US',
                                'uri': 'https://www.google.com/search?q=weather+in+San Francisco, CA,+US',
                            },
                            {
                                'domain': None,
                                'title': 'weather.gov',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF_uqo2G5Goeww8iF1L_dYa2sqWGhzu_UnxEZd1gQ7ZNuXEVVVYEEYcx_La3kuODFm0dPUhHeF4qGP1c6kJ86i4SKfvRqFitMCvNiDx07eC5iM7axwepoTv3FeUdIRC-ou1P-6DDykZ4QzcxcrKISa_1Q==',
                            },
                            {
                                'domain': None,
                                'title': 'wunderground.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFywixFZicmDjijfhfLNw8ya7XdqWR31aJp8CHyULLelG8bujH1TuqeP9RAhK6Pcm1qz11ujm2yM7gM5bJXDFsZwbsubub4cnUp5ixRaloJcjVrHkyd5RHblhkDDxHGiREV9BcuqeJovdr8qhtrCKMcvJk=',
                            },
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(
                        content="""\
## Weather in San Francisco is Mild and Partly Cloudy

**San Francisco, CA** - Residents and visitors in San Francisco are experiencing a mild Tuesday, with partly cloudy skies and temperatures hovering around 69°F. There is a very low chance of rain throughout the day.

According to the latest weather reports, the forecast for the remainder of the day is expected to be sunny, with highs ranging from the mid-60s to the lower 80s. Winds are predicted to come from the west at 10 to 15 mph.

As the evening approaches, the skies are expected to remain partly cloudy, with temperatures dropping to the upper 50s. There is a slight increase in the chance of rain overnight, but it remains low at 20%.

Overall, today's weather in San Francisco is pleasant, with a mix of sun and clouds and comfortable temperatures.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=17,
                    output_tokens=533,
                    details={
                        'thoughts_tokens': 213,
                        'tool_use_prompt_tokens': 119,
                        'text_prompt_tokens': 17,
                        'text_tool_use_prompt_tokens': 119,
                    },
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    messages = result.all_messages()
    result = await agent.run(user_prompt='how about Mexico City?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='how about Mexico City?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful chatbot.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['current weather in Mexico City']},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'domain': None,
                                'title': 'theweathernetwork.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEvigSUuLwtMoqPNq2bvqCduH6yYQLKmhzoj0-SQbxBb2rs_ow380KClss6yfKqxmQ-3HIrmzasviLVdO2FhQ_uEIGfpv6-_r4XOSSLu57LKZgAFYTsswd5Q--VkuO2eEr4Vh8b0aK4KFi3Rt3k_r99frmOa-8mCHzWrXI_HeS58IvIpda0XNtWVEjg',
                            },
                            {
                                'domain': None,
                                'title': 'wunderground.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFEXnJiWubQ1I2xMumZnSwxzZzhO_s2AdGg1yFakgO7GqJXU25aq3-Zl5xFEsUk9KpDtKUsS0NrBQxRNYCTkbKMknHSD5n8Yps9aAYvLOvyKgKPDFt4SkBkt1RO1nyPOweAzOzjPmnnd8AqBqOq',
                            },
                            {
                                'domain': None,
                                'title': 'wunderground.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEDXOJgWay-hTPi0eqxph51YPv_mX15kug_vYdV3Ybx19gm4XsIFdbDN3OhP8tHbKJDheVySvDaxmXZK2lsEJlHITYidz_uKAiY38_peXIPv0Kw4LvBYLWUh4SPwHBLgHAR3CsLQo3293ZbIXZ_3A==',
                            },
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(
                        content="""\
In Mexico City today, you can expect a day of mixed sun and clouds with a high likelihood of showers and thunderstorms, particularly in the afternoon and evening.

Currently, the weather is partly cloudy with temperatures in the mid-60s Fahrenheit (around 17-18°C). As the day progresses, the temperature is expected to rise, reaching a high of around 73-75°F (approximately 23°C).

There is a significant chance of rain, with forecasts indicating a 60% to 100% probability of precipitation, especially from mid-afternoon into the evening. Winds are generally light, coming from the north-northeast at 10 to 15 mph.

Tonight, the skies will remain cloudy with a continued chance of showers, and the temperature will drop to a low of around 57°F (about 14°C).\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=209,
                    output_tokens=623,
                    details={
                        'thoughts_tokens': 131,
                        'tool_use_prompt_tokens': 286,
                        'text_prompt_tokens': 209,
                        'text_tool_use_prompt_tokens': 286,
                    },
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_model_web_search_tool_stream(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-pro', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.', builtin_tools=[WebSearchTool()])

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the weather in San Francisco today?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in San Francisco today?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful chatbot.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
### Weather in San Francisco is Mild and Partly Cloudy Today

**San Francisco, CA** - Today's weather in San Francisco is partly cloudy with temperatures ranging from the high 50s to the low 80s, according to various weather reports.

As of Tuesday afternoon, the temperature is around 69°F (21°C), with a real feel of about 76°F (24°C) and humidity at approximately 68%. Another report indicates a temperature of 68°F with passing clouds. There is a very low chance of rain throughout the day.

The forecast for the remainder of the day predicts sunny skies with highs ranging from the mid-60s to the lower 80s. Some sources suggest the high could reach up to 85°F. Tonight, the weather is expected to be partly cloudy with lows in the upper 50s.

Hourly forecasts show temperatures remaining in the low 70s during the afternoon before gradually cooling down in the evening. The chance of rain remains low throughout the day.\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=17,
                    output_tokens=755,
                    details={
                        'thoughts_tokens': 412,
                        'tool_use_prompt_tokens': 102,
                        'text_prompt_tokens': 17,
                        'text_tool_use_prompt_tokens': 102,
                    },
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=TextPart(
                    content="""\
### Weather in San Francisco is Mild and Partly Cloudy Today

**San Francisco, CA** - Today's weather in San\
"""
                ),
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta=' Francisco is partly cloudy with temperatures ranging from the high 50s to the low 80s, according to various weather'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta="""\
 reports.

As of Tuesday afternoon, the temperature is around 69°F (21°C), with a real\
"""
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta=' feel of about 76°F (24°C) and humidity at approximately 68%. Another'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta=' report indicates a temperature of 68°F with passing clouds. There is a very low chance of'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta="""\
 rain throughout the day.

The forecast for the remainder of the day predicts sunny skies with highs ranging from the mid\
"""
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta='-60s to the lower 80s. Some sources suggest the high could reach up to 85'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta='°F. Tonight, the weather is expected to be partly cloudy with lows in the upper 50s'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta="""\
.

Hourly forecasts show temperatures remaining in the low 70s during the afternoon before gradually cooling down in\
"""
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(content_delta=' the evening. The chance of rain remains low throughout the day.'),
            ),
            PartEndEvent(
                index=0,
                part=TextPart(
                    content="""\
### Weather in San Francisco is Mild and Partly Cloudy Today

**San Francisco, CA** - Today's weather in San Francisco is partly cloudy with temperatures ranging from the high 50s to the low 80s, according to various weather reports.

As of Tuesday afternoon, the temperature is around 69°F (21°C), with a real feel of about 76°F (24°C) and humidity at approximately 68%. Another report indicates a temperature of 68°F with passing clouds. There is a very low chance of rain throughout the day.

The forecast for the remainder of the day predicts sunny skies with highs ranging from the mid-60s to the lower 80s. Some sources suggest the high could reach up to 85°F. Tonight, the weather is expected to be partly cloudy with lows in the upper 50s.

Hourly forecasts show temperatures remaining in the low 70s during the afternoon before gradually cooling down in the evening. The chance of rain remains low throughout the day.\
"""
                ),
            ),
        ]
    )

    result = await agent.run(user_prompt='how about Mexico City?', message_history=messages)
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='how about Mexico City?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful chatbot.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['weather in Mexico City today']},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'domain': None,
                                'title': 'wunderground.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEQC0SXLaLGgcMFH_tEWkajsUbbqi5e41d5DCbU7UYn-07hCucenSJSG81JCNJHvCmvBBNLToqgi9ekV5gIRMRxWyuGtmwk6_mm9PkCXkma14WNA77Mop53-RlMrNGA0Pv1cWWsfjT2eO0TzYw=',
                            },
                            {
                                'domain': None,
                                'title': 'wunderground.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHvVca9OLivHL55Skj5zYB3_Tz-N5Fqhjbq3NA61blVTqN54YtDSleJ9UIx6wsIAcCih6MGTG2GGnqXbcinemBrd66vI4a93SqCUUenrG2M9mzjdVShhGaW3hLtx8jGnNGiGVbg3i6EiHJWExkG',
                            },
                            {
                                'domain': None,
                                'title': 'yahoo.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFTqbIT6r826Xu2U3cET_KtlwQe82Sf_LNSKFQKayYaymtY3qAbz6iIkbQxccEiSnFv-HmDVkk_ie97DIp9d3iw-PapYXUKqV3OA720KCi6KmqZ98zJkAxg-egXxD-PyHIkyaK5eBlCo5JLKDff_EhJchxZ',
                            },
                            {
                                'domain': None,
                                'title': 'theweathernetwork.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGfewQ5Ayt0L90iNqoh_TfbKWfmLEfxHK2StObAJayvxDyyZnZN9RQce45e_lWWThsK4AqsqSRcHabKkQK8YMa1owQR8Bn6-ma7jiWhx8NN2d7Cu5diJcujVwyEbvTLS3ZlavVz8J6lXmUvDTVVDrVA4pKBYkz96YMy76lT1IJJzo4quSaVFhXjk1Y=',
                            },
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(
                        content="""\
### Scattered Thunderstorms and Mild Temperatures in Mexico City Today

**Mexico City, Mexico** - The weather in Mexico City today is generally cloudy with scattered thunderstorms expected to develop, particularly this afternoon. Temperatures are mild, with highs forecasted to be in the mid-70s and lows in the upper 50s.

Currently, the temperature is approximately 78°F (26°C), but it feels like 77°F (25°C). The forecast for the rest of the day indicates a high of around 73°F to 75°F (23°C to 24°C). Tonight, the temperature is expected to drop to a low of about 57°F (14°C).

There is a high chance of rain throughout the day, with some reports stating a 60% to 85% probability of precipitation. Hourly forecasts indicate that the likelihood of rain increases significantly in the late afternoon and evening. Winds are coming from the north-northeast at 10 to 15 mph.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=249,
                    output_tokens=860,
                    details={
                        'thoughts_tokens': 301,
                        'tool_use_prompt_tokens': 319,
                        'text_prompt_tokens': 249,
                        'text_tool_use_prompt_tokens': 319,
                    },
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.parametrize('use_deprecated_url_context_tool', [False, True])
async def test_google_model_web_fetch_tool(
    allow_model_requests: None, google_provider: GoogleProvider, use_deprecated_url_context_tool: bool
):
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    if use_deprecated_url_context_tool:
        with pytest.warns(DeprecationWarning, match='Use `WebFetchTool` instead.'):
            tool = UrlContextTool()  # pyright: ignore[reportDeprecated]
    else:
        tool = WebFetchTool()

    agent = Agent(m, instructions='You are a helpful chatbot.', builtin_tools=[tool])

    result = await agent.run(
        'What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.'
    )

    assert result.output == snapshot(
        'Pydantic AI is a Python agent framework designed to make it less painful to build production grade applications with Generative AI.'
    )

    # Check that BuiltinToolCallPart and BuiltinToolReturnPart are generated
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful chatbot.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_fetch',
                        args={'urls': ['https://ai.pydantic.dev']},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_fetch',
                        content=[
                            {
                                'retrieved_url': 'https://ai.pydantic.dev',
                                'url_retrieval_status': 'URL_RETRIEVAL_STATUS_SUCCESS',
                            }
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(
                        content='Pydantic AI is a Python agent framework designed to make it less painful to build production grade applications with Generative AI.'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=32,
                    output_tokens=2483,
                    details={
                        'thoughts_tokens': 47,
                        'tool_use_prompt_tokens': 2395,
                        'text_prompt_tokens': 32,
                        'text_tool_use_prompt_tokens': 2395,
                    },
                ),
                model_name='gemini-2.5-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_model_web_fetch_tool_stream(allow_model_requests: None, google_provider: GoogleProvider):
    """Test WebFetchTool streaming to ensure BuiltinToolCallPart and BuiltinToolReturnPart are generated."""
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    tool = WebFetchTool()
    agent = Agent(m, instructions='You are a helpful chatbot.', builtin_tools=[tool])

    event_parts: list[Any] = []
    async with agent.iter(
        user_prompt='What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.'
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    messages = agent_run.result.all_messages()

    # Check that BuiltinToolCallPart and BuiltinToolReturnPart are generated in messages
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the first sentence on the page https://ai.pydantic.dev? Reply with only the sentence.',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful chatbot.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_fetch',
                        args={'urls': ['https://ai.pydantic.dev']},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_fetch',
                        content=[
                            {
                                'retrieved_url': 'https://ai.pydantic.dev',
                                'url_retrieval_status': 'URL_RETRIEVAL_STATUS_SUCCESS',
                            }
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=IsInstance(int),
                    output_tokens=IsInstance(int),
                    details={
                        'thoughts_tokens': IsInstance(int),
                        'tool_use_prompt_tokens': IsInstance(int),
                        'text_prompt_tokens': IsInstance(int),
                        'text_tool_use_prompt_tokens': IsInstance(int),
                    },
                ),
                model_name='gemini-2.5-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    # Check that streaming events include BuiltinToolCallPart and BuiltinToolReturnPart
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='web_fetch',
                    args={'urls': ['https://ai.pydantic.dev']},
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                ),
            ),
            PartEndEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='web_fetch',
                    args={'urls': ['https://ai.pydantic.dev']},
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='web_fetch',
                    content=[
                        {
                            'retrieved_url': 'https://ai.pydantic.dev',
                            'url_retrieval_status': 'URL_RETRIEVAL_STATUS_SUCCESS',
                        }
                    ],
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=2,
                part=TextPart(content=IsStr()),
                previous_part_kind='builtin-tool-return',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=IsStr())),
            PartEndEvent(index=2, part=TextPart(content=IsStr())),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='web_fetch',
                    args={'urls': ['https://ai.pydantic.dev']},
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_fetch',
                    content=[
                        {
                            'retrieved_url': 'https://ai.pydantic.dev',
                            'url_retrieval_status': 'URL_RETRIEVAL_STATUS_SUCCESS',
                        }
                    ],
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                )
            ),
        ]
    )


async def test_google_model_code_execution_tool(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-pro', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.', builtin_tools=[CodeExecutionTool()])

    result = await agent.run('What day is today in Utrecht?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What day is today in Utrecht?', timestamp=IsDatetime())],
                instructions='You are a helpful chatbot.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
from datetime import datetime
import pytz

# Get the current time in UTC
utc_now = datetime.now(pytz.utc)

# Get the timezone for Utrecht (which is in the Netherlands, using Europe/Amsterdam)
utrecht_tz = pytz.timezone('Europe/Amsterdam')

# Convert the current UTC time to Utrecht's local time
utrecht_now = utc_now.astimezone(utrecht_tz)

# Format the date to be easily readable (e.g., "Tuesday, May 21, 2024")
formatted_date = utrecht_now.strftime("%A, %B %d, %Y")

print(f"Today in Utrecht is {formatted_date}.")
""",
                            'language': 'PYTHON',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'outcome': 'OUTCOME_OK',
                            'output': 'Today in Utrecht is Tuesday, September 16, 2025.\n',
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(content='Today in Utrecht is Tuesday, September 16, 2025.'),
                ],
                usage=RequestUsage(
                    input_tokens=15,
                    output_tokens=1335,
                    details={
                        'thoughts_tokens': 483,
                        'tool_use_prompt_tokens': 675,
                        'text_prompt_tokens': 15,
                        'text_tool_use_prompt_tokens': 675,
                    },
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run('What day is tomorrow?', message_history=result.all_messages())
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What day is tomorrow?', timestamp=IsDatetime())],
                instructions='You are a helpful chatbot.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
from datetime import date, timedelta

tomorrow = date.today() + timedelta(days=1)
print(f"Tomorrow is {tomorrow.strftime('%A, %B %d, %Y')}.")
""",
                            'language': 'PYTHON',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'outcome': 'OUTCOME_OK',
                            'output': 'Tomorrow is Wednesday, September 17, 2025.\n',
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(content='Tomorrow is Wednesday, September 17, 2025.'),
                ],
                usage=RequestUsage(
                    input_tokens=39,
                    output_tokens=1235,
                    details={
                        'thoughts_tokens': 540,
                        'tool_use_prompt_tokens': 637,
                        'text_prompt_tokens': 39,
                        'text_tool_use_prompt_tokens': 637,
                    },
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_model_server_tool_receive_history_from_another_provider(
    allow_model_requests: None, anthropic_api_key: str, gemini_api_key: str
):
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    anthropic_model = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    google_model = GoogleModel('gemini-2.0-flash', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(builtin_tools=[CodeExecutionTool()])

    result = await agent.run('How much is 3 * 12390?', model=anthropic_model)
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [[UserPromptPart], [TextPart, BuiltinToolCallPart, BuiltinToolReturnPart, TextPart]]
    )

    result = await agent.run('Multiplied by 12390', model=google_model, message_history=result.all_messages())
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [
            [UserPromptPart],
            [TextPart, BuiltinToolCallPart, BuiltinToolReturnPart, TextPart],
            [UserPromptPart],
            [TextPart, BuiltinToolCallPart, BuiltinToolReturnPart, TextPart],
        ]
    )


async def test_google_model_receive_web_search_history_from_another_provider(
    allow_model_requests: None, anthropic_api_key: str, gemini_api_key: str
):
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    anthropic_model = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    anthropic_agent = Agent(model=anthropic_model, builtin_tools=[WebSearchTool()])

    result = await anthropic_agent.run('What are the latest news in the Netherlands?')
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [
            [UserPromptPart],
            [
                BuiltinToolCallPart,
                BuiltinToolReturnPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
            ],
        ]
    )

    google_model = GoogleModel('gemini-2.0-flash', provider=GoogleProvider(api_key=gemini_api_key))
    google_agent = Agent(model=google_model)
    result = await google_agent.run('What day is tomorrow?', message_history=result.all_messages())
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [
            [UserPromptPart],
            [
                BuiltinToolCallPart,
                BuiltinToolReturnPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
                TextPart,
            ],
            [UserPromptPart],
            [TextPart],
        ]
    )


async def test_google_model_empty_user_prompt(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run()
    assert result.output == snapshot("""\
Hello! That's correct. I am designed to be a helpful assistant.

I'm ready to assist you with a wide range of tasks, from answering questions and providing information to brainstorming ideas and generating creative content.

How can I help you today?\
""")


async def test_google_model_empty_assistant_response(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(m)

    result = await agent.run(
        'Was your previous response empty?',
        message_history=[
            ModelRequest(parts=[UserPromptPart(content='Hi')], timestamp=IsDatetime()),
            ModelResponse(parts=[TextPart(content='')]),
        ],
    )

    assert result.output == snapshot("""\
As an AI, I don't retain memory of past interactions or specific conversational history in the way a human does. Each response I generate is based on the current prompt I receive.

Therefore, I cannot directly recall if my specific previous response to you was empty.

However, I am designed to always provide a response with content. If you received an empty response, it would likely indicate a technical issue or an error in the system, rather than an intentional empty output from me.

Could you please tell me what you were expecting or if you'd like me to try again?\
""")


async def test_google_model_thinking_part(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-3-pro-preview', provider=google_provider)
    settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True})
    agent = Agent(m, instructions='You are a helpful assistant.', model_settings=settings)

    # Google only emits thought signatures when there are tools: https://ai.google.dev/gemini-api/docs/thinking#signatures
    @agent.tool_plain
    def dummy() -> None: ...  # pragma: no cover

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr()),
                    TextPart(
                        content=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=29, output_tokens=1737, details={'thoughts_tokens': 1001, 'text_prompt_tokens': 29}
                ),
                model_name='gemini-3-pro-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr()),
                    TextPart(
                        content=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=1280, output_tokens=2073, details={'thoughts_tokens': 1115, 'text_prompt_tokens': 1280}
                ),
                model_name='gemini-3-pro-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_model_thinking_part_from_other_model(
    allow_model_requests: None, google_provider: GoogleProvider, openai_api_key: str
):
    provider = OpenAIProvider(api_key=openai_api_key)
    m = OpenAIResponsesModel('gpt-5', provider=provider)
    settings = OpenAIResponsesModelSettings(openai_reasoning_effort='high', openai_reasoning_summary='detailed')
    agent = Agent(m, instructions='You are a helpful assistant.', model_settings=settings)

    # Google only emits thought signatures when there are tools: https://ai.google.dev/gemini-api/docs/thinking#signatures
    @agent.tool_plain
    def dummy() -> None: ...  # pragma: no cover

    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fb6c15c48196b964881266a03c8e0c14a8a9087e8689',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fb6c15c48196b964881266a03c8e0c14a8a9087e8689',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fb6c15c48196b964881266a03c8e0c14a8a9087e8689',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fb6c15c48196b964881266a03c8e0c14a8a9087e8689',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content=IsStr(),
                        id='rs_68c1fb6c15c48196b964881266a03c8e0c14a8a9087e8689',
                        provider_name='openai',
                    ),
                    TextPart(
                        content=IsStr(),
                        id='msg_68c1fb814fdc8196aec1a46164ddf7680c14a8a9087e8689',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=45, output_tokens=1719, details={'reasoning_tokens': 1408}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime.datetime(2025, 9, 10, 22, 27, 55, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run(
        'Considering the way to cross the street, analogously, how do I cross the river?',
        model=GoogleModel(
            'gemini-2.5-pro',
            provider=google_provider,
            settings=GoogleModelSettings(google_thinking_config={'include_thoughts': True}),
        ),
        message_history=result.all_messages(),
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Considering the way to cross the street, analogously, how do I cross the river?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr()),
                    TextPart(
                        content=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=1106, output_tokens=1867, details={'thoughts_tokens': 1089, 'text_prompt_tokens': 1106}
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_model_thinking_part_iter(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-pro', provider=google_provider)
    settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True})
    agent = Agent(m, instructions='You are a helpful assistant.', model_settings=settings)

    # Google only emits thought signatures when there are tools: https://ai.google.dev/gemini-api/docs/thinking#signatures
    @agent.tool_plain
    def dummy() -> None: ...  # pragma: no cover

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='How do I cross the street?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='You are a helpful assistant.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr()),
                    TextPart(
                        content=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=34, output_tokens=1256, details={'thoughts_tokens': 787, 'text_prompt_tokens': 34}
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=IsStr())),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content="""\
**Clarifying User Goals**

I'm currently focused on defining the user's ultimate goal: ensuring their safety while crossing the street. I've pinpointed that this is a real-world scenario with significant safety considerations. However, I'm also mindful of my limitations as an AI and my inability to physically assist or visually assess the situation.


**Developing a Safety Protocol**

I'm now formulating a comprehensive safety procedure. I've pinpointed the essential first step: finding a safe crossing location, such as marked crosswalks or intersections. Stopping at the curb, and looking and listening for traffic are vital too. The rationale behind "look left, right, then left again" now needs further exploration. I'm focusing on crafting universally applicable and secure steps.


**Prioritizing Safe Crossing**

I've revised the procedure's initial step, emphasizing safe crossing zones (crosswalks, intersections). Next, I'm integrating the "look left, right, then left" sequence, considering why it's repeated. I'm focusing on crafting universal, safety-focused instructions that suit diverse situations and address my inherent limitations.


**Crafting Safe Instructions**

I've identified the core user intent: to learn safe street-crossing. Now, I'm focusing on crafting universally applicable steps. Finding safe crossing locations and looking-listening for traffic remain paramount. I'm prioritizing direct, clear language, addressing my limitations as an AI. I'm crafting advice that works generally, regardless of specific circumstances or locations.


"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=1,
                part=TextPart(
                    content='This is a great question! Safely crossing the street is all about being aware and predictable. Here is a step-by-step',
                    provider_name='google-gla',
                    provider_details={'thought_signature': IsStr()},
                ),
                previous_part_kind='thinking',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=IsStr())),
            PartEndEvent(
                index=1,
                part=TextPart(
                    content="""\
This is a great question! Safely crossing the street is all about being aware and predictable. Here is a step-by-step guide that is widely taught for safety:

### 1. Find a Safe Place to Cross
The best place is always at a designated **crosswalk** or a **street corner/intersection**. These are places where drivers expect to see pedestrians. Avoid crossing in the middle of the block or from between parked cars.

### 2. Stop at the Edge of the Curb
Stand on the sidewalk, a safe distance from the edge of the street. This gives you a clear view of the traffic without putting you in danger.

### 3. Look and Listen for Traffic
Follow the "Left-Right-Left" rule:
*   **Look left** for the traffic that will be closest to you first.
*   **Look right** for oncoming traffic in the other lane.
*   **Look left again** to make sure nothing has changed.
*   **Listen** for the sound of approaching vehicles that you might not be able to see.

### 4. Wait for a Safe Gap
Wait until there is a large enough gap in traffic for you to walk all the way across. Don't assume a driver will stop for you. If you can, try to **make eye contact** with drivers to ensure they have seen you.

### 5. Walk, Don't Run
Once it's safe:
*   Walk straight across the street.
*   **Keep looking and listening** for traffic as you cross. The situation can change quickly.
*   **Don't use your phone** or wear headphones that block out the sound of traffic.

---

### Special Situations:

*   **At a Traffic Light:** Wait for the pedestrian signal to show the "Walk" sign (often a symbol of a person walking). Even when the sign says to walk, you should still look left and right before crossing.
*   **At a Stop Sign:** Wait for the car to come to a complete stop. Make eye contact with the driver before you step into the street to be sure they see you.

The most important rule is to **stay alert and be predictable**. Always assume a driver might not see you.\
""",
                    provider_name='google-gla',
                    provider_details={'thought_signature': IsStr()},
                ),
            ),
        ]
    )


@pytest.mark.parametrize(
    'url,expected_output',
    [
        pytest.param(
            AudioUrl(url='https://cdn.openai.com/API/docs/audio/alloy.wav'),
            'The URL discusses the sunrise in the east and sunset in the west, a phenomenon known to humans for millennia.',
            id='AudioUrl',
        ),
        pytest.param(
            DocumentUrl(url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf'),
            "The URL points to a technical report from Google DeepMind introducing Gemini 1.5 Pro, a multimodal AI model designed for understanding and reasoning over extremely large contexts (millions of tokens). It details the model's architecture, training, performance across a range of tasks, and responsible deployment considerations. Key highlights include near-perfect recall on long-context retrieval tasks, state-of-the-art performance in areas like long-document question answering, and surprising new capabilities like in-context learning of new languages.",
            id='DocumentUrl',
        ),
        pytest.param(
            ImageUrl(url='https://upload.wikimedia.org/wikipedia/commons/6/6a/Www.wikipedia_screenshot_%282021%29.png'),
            "The URL's main content is the landing page of Wikipedia, showcasing the available language editions with article counts, a search bar, and links to other Wikimedia projects.",
            id='ImageUrl',
        ),
        pytest.param(
            VideoUrl(url='https://upload.wikimedia.org/wikipedia/commons/8/8f/Panda_at_Smithsonian_zoo.webm'),
            """The main content of the image is a panda eating bamboo in a zoo enclosure. The enclosure is designed to mimic the panda's natural habitat, with rocks, bamboo, and a painted backdrop of mountains. There is also a large, smooth, tan-colored ball-shaped object in the enclosure.""",
            id='VideoUrl',
        ),
        pytest.param(
            VideoUrl(url='https://youtu.be/lCdaVNyHtjU'),
            'The main content of the URL is an analysis of recent 404 HTTP responses. The analysis identifies several patterns including the most common endpoints with 404 errors, request patterns, timeline-related issues, organization/project access, and configuration and authentication. The analysis also provides some recommendations.',
            id='VideoUrl (YouTube)',
        ),
        pytest.param(
            AudioUrl(url='gs://pydantic-ai-dev/openai-alloy.wav'),
            'The content describes the basic concept of the sun rising in the east and setting in the west.',
            id='AudioUrl (gs)',
        ),
        pytest.param(
            DocumentUrl(url='gs://pydantic-ai-dev/Gemini_1_5_Pro_Technical_Report_Arxiv_1805.pdf'),
            "The URL leads to a research paper titled \"Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context\".  \n\nThe paper introduces Gemini 1.5 Pro, a new model in the Gemini family. It's described as a highly compute-efficient multimodal mixture-of-experts model.  A key feature is its ability to recall and reason over fine-grained information from millions of tokens of context, including long documents and hours of video and audio.  The paper presents experimental results showcasing the model's capabilities on long-context retrieval tasks, QA, ASR, and its performance compared to Gemini 1.0 models. It covers the model's architecture, training data, and evaluations on both synthetic and real-world tasks.  A notable highlight is its ability to learn to translate from English to Kalamang, a low-resource language, from just a grammar manual and dictionary provided in context.  The paper also discusses responsible deployment considerations, including impact assessments and mitigation efforts.\n",
            id='DocumentUrl (gs)',
        ),
        pytest.param(
            ImageUrl(url='gs://pydantic-ai-dev/wikipedia_screenshot.png'),
            "The main content of the URL is the Wikipedia homepage, featuring options to access Wikipedia in different languages and information about the number of articles in each language. It also includes links to other Wikimedia projects and information about Wikipedia's host, the Wikimedia Foundation.\n",
            id='ImageUrl (gs)',
        ),
        pytest.param(
            VideoUrl(url='gs://pydantic-ai-dev/grepit-tiny-video.mp4'),
            'The image shows a charming outdoor cafe in a Greek coastal town. The cafe is nestled between traditional whitewashed buildings, with tables and chairs set along a narrow cobblestone pathway. The sea is visible in the distance, adding to the picturesque and relaxing atmosphere.',
            id='VideoUrl (gs)',
        ),
    ],
)
async def test_google_url_input(
    url: AudioUrl | DocumentUrl | ImageUrl | VideoUrl,
    expected_output: str,
    allow_model_requests: None,
    vertex_provider: GoogleProvider,
) -> None:  # pragma: lax no cover
    m = GoogleModel('gemini-2.0-flash', provider=vertex_provider)
    agent = Agent(m)
    result = await agent.run(['What is the main content of this URL?', url])

    assert result.output == snapshot(Is(expected_output))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What is the main content of this URL?', Is(url)],
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content=Is(expected_output))],
                usage=IsInstance(RequestUsage),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-vertex',
                provider_url='https://aiplatform.googleapis.com/',
                provider_details={'finish_reason': 'STOP', 'timestamp': IsDatetime()},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.skipif(
    not os.getenv('CI', False), reason='Requires properly configured local google vertex config to pass'
)
@pytest.mark.vcr()
async def test_google_url_input_force_download(
    allow_model_requests: None, vertex_provider: GoogleProvider
) -> None:  # pragma: lax no cover
    m = GoogleModel('gemini-2.0-flash', provider=vertex_provider)
    agent = Agent(m)

    video_url = VideoUrl(url='https://data.grepit.app/assets/tiny_video.mp4', force_download=True)
    result = await agent.run(['What is the main content of this URL?', video_url])

    output = 'The image shows a picturesque scene in what appears to be a Greek island town. The focus is on an outdoor dining area with tables and chairs, situated in a narrow alleyway between whitewashed buildings. The ocean is visible at the end of the alley, creating a beautiful and inviting atmosphere.'

    assert result.output == output
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What is the main content of this URL?', Is(video_url)],
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content=Is(output))],
                usage=IsInstance(RequestUsage),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-vertex',
                provider_url='https://aiplatform.googleapis.com/',
                provider_details={'finish_reason': 'STOP', 'timestamp': IsDatetime()},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_gs_url_force_download_raises_user_error(allow_model_requests: None) -> None:
    provider = GoogleProvider(project='pydantic-ai', location='us-central1')
    m = GoogleModel('gemini-2.0-flash', provider=provider)
    agent = Agent(m)

    url = ImageUrl(url='gs://pydantic-ai-dev/wikipedia_screenshot.png', force_download=True)
    with pytest.raises(UserError, match='Downloading from protocol "gs://" is not supported.'):
        _ = await agent.run(['What is the main content of this URL?', url])


async def test_google_tool_config_any_with_tool_without_args(
    allow_model_requests: None, google_provider: GoogleProvider
):
    class Foo(TypedDict):
        bar: str

    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, output_type=Foo)

    @agent.tool_plain
    async def bar() -> str:
        return 'hello'

    result = await agent.run('run bar for me please')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='run bar for me please',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='bar', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=21, output_tokens=1, details={'text_candidates_tokens': 1, 'text_prompt_tokens': 21}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='bar',
                        content='hello',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args={'bar': 'hello'}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=27, output_tokens=5, details={'text_candidates_tokens': 5, 'text_prompt_tokens': 27}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_timeout(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    agent = Agent(model=model)

    result = await agent.run('Hello!', model_settings={'timeout': 10})
    assert result.output == snapshot('Hello there! How can I help you today?\n')

    with pytest.raises(UserError, match='Google does not support setting ModelSettings.timeout to a httpx.Timeout'):
        await agent.run('Hello!', model_settings={'timeout': Timeout(10)})


async def test_google_tool_output(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=ToolOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=33, output_tokens=5, details={'text_candidates_tokens': 5, 'text_prompt_tokens': 33}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'city': 'Mexico City', 'country': 'Mexico'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(
                    input_tokens=47, output_tokens=8, details={'text_candidates_tokens': 8, 'text_prompt_tokens': 47}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_text_output_function(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-pro-preview-05-06', provider=google_provider)

    def upcase(text: str) -> str:
        return text.upper()

    agent = Agent(m, output_type=TextOutput(upcase))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run(
        'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.'
    )
    assert result.output == snapshot('THE LARGEST CITY IN MEXICO IS MEXICO CITY.')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=49, output_tokens=276, details={'thoughts_tokens': 264, 'text_prompt_tokens': 49}
                ),
                model_name='models/gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The largest city in Mexico is Mexico City.')],
                usage=RequestUsage(
                    input_tokens=80, output_tokens=159, details={'thoughts_tokens': 150, 'text_prompt_tokens': 80}
                ),
                model_name='models/gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_native_output_with_tools(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'  # pragma: no cover

    with pytest.raises(
        UserError,
        match=re.escape(
            'Google does not support `NativeOutput` and function tools at the same time. Use `output_type=ToolOutput(...)` instead.'
        ),
    ):
        await agent.run('What is the largest city in the user country?')


async def test_google_native_output(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "city": "Mexico City",
  "country": "Mexico"
}\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=8, output_tokens=20, details={'text_candidates_tokens': 20, 'text_prompt_tokens': 8}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_native_output_multiple(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=NativeOutput([CityLocation, CountryLanguage]))

    result = await agent.run('What is the primarily language spoken in Mexico?')
    assert result.output == snapshot(CountryLanguage(country='Mexico', language='Spanish'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the primarily language spoken in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "result": {
    "kind": "CountryLanguage",
    "data": {
      "country": "Mexico",
      "language": "Spanish"
    }
  }
}\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=50, output_tokens=46, details={'text_candidates_tokens': 46, 'text_prompt_tokens': 50}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_prompted_output(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City", "country": "Mexico"}')],
                usage=RequestUsage(
                    input_tokens=80, output_tokens=13, details={'text_candidates_tokens': 13, 'text_prompt_tokens': 80}
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_prompted_output_with_tools(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-pro-preview-05-06', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run(
        'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.'
    )
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(
                    input_tokens=123, output_tokens=144, details={'thoughts_tokens': 132, 'text_prompt_tokens': 123}
                ),
                model_name='models/gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City", "country": "Mexico"}')],
                usage=RequestUsage(
                    input_tokens=154, output_tokens=166, details={'thoughts_tokens': 153, 'text_prompt_tokens': 154}
                ),
                model_name='models/gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_prompted_output_multiple(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    class CountryLanguage(BaseModel):
        country: str
        language: str

    agent = Agent(m, output_type=PromptedOutput([CityLocation, CountryLanguage]))

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result": {"kind": "CityLocation", "data": {"city": "Mexico City", "country": "Mexico"}}}'
                    )
                ],
                usage=RequestUsage(
                    input_tokens=240,
                    output_tokens=27,
                    details={'text_candidates_tokens': 27, 'text_prompt_tokens': 240},
                ),
                model_name='gemini-2.0-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_model_usage_limit_exceeded(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(model=model)

    with pytest.raises(
        UsageLimitExceeded,
        match='The next request would exceed the input_tokens_limit of 9 \\(input_tokens=12\\)',
    ):
        await agent.run(
            'The quick brown fox jumps over the lazydog.',
            usage_limits=UsageLimits(input_tokens_limit=9, count_tokens_before_request=True),
        )


async def test_google_model_usage_limit_not_exceeded(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(model=model)

    result = await agent.run(
        'The quick brown fox jumps over the lazydog.',
        usage_limits=UsageLimits(input_tokens_limit=15, count_tokens_before_request=True),
    )
    assert result.output == snapshot("""\
That's a classic! It's famously known as a **pangram**, which means it's a sentence that contains every letter of the alphabet.

It's often used for:
*   **Typing practice:** To ensure all keys are hit.
*   **Displaying font samples:** Because it showcases every character.

Just a small note, it's typically written as "lazy dog" (two words) and usually ends with a period:

**The quick brown fox jumps over the lazy dog.**\
""")


async def test_google_vertexai_model_usage_limit_exceeded(
    allow_model_requests: None, vertex_provider: GoogleProvider
):  # pragma: lax no cover
    model = GoogleModel('gemini-2.0-flash', provider=vertex_provider, settings=ModelSettings(max_tokens=100))

    agent = Agent(model, instructions='You are a chatbot.')

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'  # pragma: no cover

    with pytest.raises(
        UsageLimitExceeded, match='The next request would exceed the total_tokens_limit of 9 \\(total_tokens=36\\)'
    ):
        await agent.run(
            'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.',
            usage_limits=UsageLimits(total_tokens_limit=9, count_tokens_before_request=True),
        )


def test_map_usage():
    assert (
        _metadata_as_usage(
            GenerateContentResponse(),
            # Test the 'google' provider fallback
            provider='',
            provider_url='',
        )
        == RequestUsage()
    )

    response = GenerateContentResponse(
        usage_metadata=GenerateContentResponseUsageMetadata(
            prompt_token_count=1,
            candidates_token_count=2,
            cached_content_token_count=9100,
            thoughts_token_count=9500,
            prompt_tokens_details=[ModalityTokenCount(modality=MediaModality.AUDIO, token_count=9200)],
            cache_tokens_details=[ModalityTokenCount(modality=MediaModality.AUDIO, token_count=9300)],
            candidates_tokens_details=[ModalityTokenCount(modality=MediaModality.AUDIO, token_count=9400)],
        )
    )
    assert _metadata_as_usage(response, provider='', provider_url='') == snapshot(
        RequestUsage(
            input_tokens=1,
            cache_read_tokens=9100,
            output_tokens=9502,
            input_audio_tokens=9200,
            cache_audio_read_tokens=9300,
            output_audio_tokens=9400,
            details={
                'cached_content_tokens': 9100,
                'thoughts_tokens': 9500,
                'audio_prompt_tokens': 9200,
                'audio_cache_tokens': 9300,
                'audio_candidates_tokens': 9400,
            },
        )
    )


async def test_google_builtin_tools_with_other_tools(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    agent = Agent(m, builtin_tools=[WebFetchTool()])

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'  # pragma: no cover

    with pytest.raises(
        UserError,
        match=re.escape('Google does not support function tools and built-in tools at the same time.'),
    ):
        await agent.run('What is the largest city in the user country?')

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=ToolOutput(CityLocation), builtin_tools=[WebFetchTool()])

    with pytest.raises(
        UserError,
        match=re.escape(
            'Google does not support output tools and built-in tools at the same time. Use `output_type=PromptedOutput(...)` instead.'
        ),
    ):
        await agent.run('What is the largest city in Mexico?')

    # Will default to prompted output
    agent = Agent(m, output_type=CityLocation, builtin_tools=[WebFetchTool()])

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))


async def test_google_native_output_with_builtin_tools_gemini_3(
    allow_model_requests: None, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-3-pro-preview', provider=google_provider)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=ToolOutput(CityLocation), builtin_tools=[WebFetchTool()])

    with pytest.raises(
        UserError,
        match=re.escape(
            'Google does not support output tools and built-in tools at the same time. Use `output_type=NativeOutput(...)` instead.'
        ),
    ):
        await agent.run('What is the largest city in Mexico?')

    agent = Agent(m, output_type=NativeOutput(CityLocation), builtin_tools=[WebFetchTool()])

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))

    # Will default to native output
    agent = Agent(m, output_type=CityLocation, builtin_tools=[WebFetchTool()])

    result = await agent.run('What is the largest city in Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))


async def test_google_image_generation(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    agent = Agent(m, output_type=BinaryImage)

    result = await agent.run('Generate an image of an axolotl.')
    messages = result.all_messages()

    assert result.output == snapshot(IsInstance(BinaryImage))
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    FilePart(
                        content=IsInstance(BinaryImage),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=10,
                    output_tokens=1304,
                    details={'thoughts_tokens': 115, 'text_prompt_tokens': 10, 'image_candidates_tokens': 1120},
                ),
                model_name='gemini-3-pro-image-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run('Now give it a sombrero.', message_history=messages)
    assert result.output == snapshot(IsInstance(BinaryImage))
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Now give it a sombrero.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    FilePart(
                        content=IsInstance(BinaryImage),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=276,
                    output_tokens=1374,
                    details={
                        'thoughts_tokens': 149,
                        'text_prompt_tokens': 18,
                        'image_prompt_tokens': 258,
                        'image_candidates_tokens': 1120,
                    },
                ),
                model_name='gemini-3-pro-image-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_image_generation_stream(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-flash-image', provider=google_provider)
    agent = Agent(m, output_type=BinaryImage)

    async with agent.run_stream('Generate an image of an axolotl') as result:
        assert await result.get_output() == snapshot(IsInstance(BinaryImage))

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Generate an image of an axolotl.') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot(IsInstance(BinaryImage))
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='Here you go! '),
                    FilePart(content=IsInstance(BinaryImage)),
                ],
                usage=RequestUsage(
                    input_tokens=10,
                    output_tokens=1295,
                    details={'text_prompt_tokens': 10, 'image_candidates_tokens': 1290},
                ),
                model_name='gemini-2.5-flash-image',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartStartEvent(index=0, part=TextPart(content='Here you go!')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' ')),
            PartEndEvent(index=0, part=TextPart(content='Here you go! '), next_part_kind='file'),
            PartStartEvent(
                index=1,
                part=FilePart(content=IsInstance(BinaryImage)),
                previous_part_kind='text',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
        ]
    )


async def test_google_image_generation_with_text(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    agent = Agent(m)

    result = await agent.run('Generate an illustrated two-sentence story about an axolotl.')
    messages = result.all_messages()

    assert result.output == snapshot(
        """\
A little axolotl named Archie lived in a beautiful glass tank, but he always wondered what was beyond the clear walls. One day, he bravely peeked over the edge and discovered a whole new world of sunshine and potted plants.

"""
    )
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an illustrated two-sentence story about an axolotl.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
A little axolotl named Archie lived in a beautiful glass tank, but he always wondered what was beyond the clear walls. One day, he bravely peeked over the edge and discovered a whole new world of sunshine and potted plants.

""",
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=14,
                    output_tokens=1457,
                    details={'thoughts_tokens': 174, 'text_prompt_tokens': 14, 'image_candidates_tokens': 1120},
                ),
                model_name='gemini-3-pro-image-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_image_or_text_output(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-flash-image', provider=google_provider)
    # ImageGenerationTool is listed here to indicate just that it doesn't cause any issues, even though it's not necessary with an image model.
    agent = Agent(m, output_type=str | BinaryImage, builtin_tools=[ImageGenerationTool(size='1K')])

    result = await agent.run('Tell me a two-sentence story about an axolotl, no image please.')
    assert result.output == snapshot(
        'In a hidden cave, a shy axolotl named Pip spent its days dreaming of the world beyond its murky pond. One evening, a glimmering portal appeared, offering Pip a chance to explore the vibrant, unknown depths of the ocean.'
    )

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(IsInstance(BinaryImage))


async def test_google_image_and_text_output(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-flash-image', provider=google_provider)
    agent = Agent(m)

    result = await agent.run('Tell me a two-sentence story about an axolotl with an illustration.')
    assert result.output == snapshot(
        'Once, in a hidden cenote, lived an axolotl named Pip who loved to collect shiny pebbles. One day, Pip found a pebble that glowed, illuminating his entire underwater world with a soft, warm light. '
    )
    assert result.response.files == snapshot([IsInstance(BinaryImage)])


async def test_google_image_generation_with_tool_output(allow_model_requests: None, google_provider: GoogleProvider):
    class Animal(BaseModel):
        species: str
        name: str

    model = GoogleModel('gemini-2.5-flash-image', provider=google_provider)
    agent = Agent(model=model, output_type=Animal)

    with pytest.raises(UserError, match='Tool output is not supported by this model.'):
        await agent.run('Generate an image of an axolotl.')


async def test_google_image_generation_with_native_output(allow_model_requests: None, google_provider: GoogleProvider):
    class Animal(BaseModel):
        species: str
        name: str

    model = GoogleModel('gemini-2.5-flash-image', provider=google_provider)
    agent = Agent(model=model, output_type=NativeOutput(Animal))

    with pytest.raises(UserError, match='Native structured output is not supported by this model.'):
        await agent.run('Generate an image of an axolotl.')

    model = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    agent = Agent(model=model, output_type=NativeOutput(Animal))

    result = await agent.run('Generate an image of an axolotl and then return its details.')
    assert result.output == snapshot(Animal(species='Ambystoma mexicanum', name='Axolotl'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl and then return its details.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    FilePart(
                        content=IsInstance(BinaryImage),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=15,
                    output_tokens=1334,
                    details={'thoughts_tokens': 131, 'text_prompt_tokens': 15, 'image_candidates_tokens': 1120},
                ),
                model_name='gemini-3-pro-image-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please return text or call a tool.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "species": "Ambystoma mexicanum",
  "name": "Axolotl"
} \
""",
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=295,
                    output_tokens=222,
                    details={'thoughts_tokens': 196, 'text_prompt_tokens': 37, 'image_prompt_tokens': 258},
                ),
                model_name='gemini-3-pro-image-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_image_generation_with_prompted_output(
    allow_model_requests: None, google_provider: GoogleProvider
):
    class Animal(BaseModel):
        species: str
        name: str

    model = GoogleModel('gemini-2.5-flash-image', provider=google_provider)
    agent = Agent(model=model, output_type=PromptedOutput(Animal))

    with pytest.raises(UserError, match='JSON output is not supported by this model.'):
        await agent.run('Generate an image of an axolotl.')


async def test_google_image_generation_with_tools(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-flash-image', provider=google_provider)
    agent = Agent(model=model, output_type=BinaryImage)

    @agent.tool_plain
    async def get_animal() -> str:
        return 'axolotl'  # pragma: no cover

    with pytest.raises(UserError, match='Tools are not supported by this model.'):
        await agent.run('Generate an image of an animal returned by the get_animal tool.')


async def test_google_image_generation_with_web_search(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    agent = Agent(model=model, output_type=BinaryImage, builtin_tools=[WebSearchTool()])

    result = await agent.run(
        'Visualize the current weather forecast for the next 5 days in Mexico City as a clean, modern weather chart. Add a visual on what I should wear each day'
    )
    assert result.output == snapshot(IsInstance(BinaryImage))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Visualize the current weather forecast for the next 5 days in Mexico City as a clean, modern weather chart. Add a visual on what I should wear each day',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['', 'current 5-day weather forecast for Mexico City and what to wear']},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'domain': None,
                                'title': 'accuweather.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQElsvx97FT3Kr__tvs8zIgS3C1znKqEOvuHdjyLe2WZZsJpbDDqn9gdF6rKV8KMZytsiWXCDcNwD5m0WvZzGWY6eVbnz0lxftYNTSNdXTiv1AtLrmw-NUcnITjEScK_JHJgnr9xmFapH9DXMGWWYKRSfcT3iy96J1gZeWjCBph5Sci23DAhzA==',
                            },
                            {
                                'domain': None,
                                'title': 'weather-and-climate.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGlGJX9f12rrKOYrY71rszTFf5KghgToVKZckqRWzT-cjW-mYE_PV3xRbk0JxQxJS18rkCt-y8qwpB41BMYEuxLnkCSBapX5s-4-0pwPUimTjHK4W65OdkVtjTU5-wlHsAppBwdwXNDSmzXZNUYLE1N0R9SKhLeHVVj-2BYYeoO9GPH',
                            },
                            {
                                'domain': None,
                                'title': '',
                                'uri': 'https://www.google.com/search?q=time+in+Mexico+City,+MX',
                            },
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    FilePart(
                        content=IsInstance(BinaryImage),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=33,
                    output_tokens=2309,
                    details={'thoughts_tokens': 529, 'text_prompt_tokens': 33, 'image_candidates_tokens': 1120},
                ),
                model_name='gemini-3-pro-image-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_image_generation_tool(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()])

    with pytest.raises(
        UserError,
        match="`ImageGenerationTool` is not supported by this model. Use a model with 'image' in the name instead.",
    ):
        await agent.run('Generate an image of an axolotl.')


async def test_google_image_generation_tool_aspect_ratio(google_provider: GoogleProvider) -> None:
    model = GoogleModel('gemini-2.5-flash-image', provider=google_provider)
    params = ModelRequestParameters(builtin_tools=[ImageGenerationTool(aspect_ratio='16:9')])

    tools, image_config = model._get_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert tools is None
    assert image_config == {'aspect_ratio': '16:9'}


async def test_google_image_generation_resolution(google_provider: GoogleProvider) -> None:
    """Test that resolution parameter from ImageGenerationTool is added to image_config."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    params = ModelRequestParameters(builtin_tools=[ImageGenerationTool(size='2K')])

    tools, image_config = model._get_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert tools is None
    assert image_config == {'image_size': '2K'}


async def test_google_image_generation_resolution_with_aspect_ratio(google_provider: GoogleProvider) -> None:
    """Test that resolution and aspect_ratio from ImageGenerationTool work together."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    params = ModelRequestParameters(builtin_tools=[ImageGenerationTool(aspect_ratio='16:9', size='4K')])

    tools, image_config = model._get_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert tools is None
    assert image_config == {'aspect_ratio': '16:9', 'image_size': '4K'}


async def test_google_image_generation_unsupported_size_raises_error(google_provider: GoogleProvider) -> None:
    """Test that unsupported size values raise an error."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    params = ModelRequestParameters(builtin_tools=[ImageGenerationTool(size='1024x1024')])

    with pytest.raises(UserError, match='Google image generation only supports `size` values'):
        model._get_tools(params)  # pyright: ignore[reportPrivateUsage]


async def test_google_image_generation_auto_size_raises_error(google_provider: GoogleProvider) -> None:
    """Test that 'auto' size raises an error for Google since it doesn't support intelligent size selection."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    params = ModelRequestParameters(builtin_tools=[ImageGenerationTool(size='auto')])

    with pytest.raises(UserError, match='Google image generation only supports `size` values'):
        model._get_tools(params)  # pyright: ignore[reportPrivateUsage]


async def test_google_image_generation_tool_output_format(
    mocker: MockerFixture, google_provider: GoogleProvider
) -> None:
    """Test that ImageGenerationTool.output_format is mapped to ImageConfigDict.output_mime_type on Vertex AI."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value='google-vertex')
    params = ModelRequestParameters(builtin_tools=[ImageGenerationTool(output_format='png')])

    tools, image_config = model._get_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert tools is None
    assert image_config == {'output_mime_type': 'image/png'}


async def test_google_image_generation_tool_unsupported_format_raises_error(
    mocker: MockerFixture, google_provider: GoogleProvider
) -> None:
    """Test that unsupported output_format values raise an error on Vertex AI."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value='google-vertex')
    # 'gif' is not supported by Google
    params = ModelRequestParameters(builtin_tools=[ImageGenerationTool(output_format='gif')])  # type: ignore

    with pytest.raises(UserError, match='Google image generation only supports `output_format` values'):
        model._get_tools(params)  # pyright: ignore[reportPrivateUsage]


async def test_google_image_generation_tool_output_compression(
    mocker: MockerFixture, google_provider: GoogleProvider
) -> None:
    """Test that ImageGenerationTool.output_compression is mapped to ImageConfigDict.output_compression_quality on Vertex AI."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value='google-vertex')

    # Test explicit value
    params = ModelRequestParameters(builtin_tools=[ImageGenerationTool(output_compression=85)])
    tools, image_config = model._get_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert tools is None
    assert image_config == {'output_compression_quality': 85, 'output_mime_type': 'image/jpeg'}

    # Test None (omitted)
    params = ModelRequestParameters(builtin_tools=[ImageGenerationTool(output_compression=None)])
    tools, image_config = model._get_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert image_config == {}


async def test_google_image_generation_tool_compression_validation(
    mocker: MockerFixture, google_provider: GoogleProvider
) -> None:
    """Test compression validation on Vertex AI: range and JPEG-only."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value='google-vertex')

    # Invalid range: > 100
    with pytest.raises(UserError, match='`output_compression` must be between 0 and 100'):
        model._get_tools(ModelRequestParameters(builtin_tools=[ImageGenerationTool(output_compression=101)]))  # pyright: ignore[reportPrivateUsage]

    # Invalid range: < 0
    with pytest.raises(UserError, match='`output_compression` must be between 0 and 100'):
        model._get_tools(ModelRequestParameters(builtin_tools=[ImageGenerationTool(output_compression=-1)]))  # pyright: ignore[reportPrivateUsage]

    # Non-JPEG format (PNG)
    with pytest.raises(UserError, match='`output_compression` is only supported for JPEG format'):
        model._get_tools(  # pyright: ignore[reportPrivateUsage]
            ModelRequestParameters(builtin_tools=[ImageGenerationTool(output_format='png', output_compression=90)])
        )

    # Non-JPEG format (WebP)
    with pytest.raises(UserError, match='`output_compression` is only supported for JPEG format'):
        model._get_tools(  # pyright: ignore[reportPrivateUsage]
            ModelRequestParameters(builtin_tools=[ImageGenerationTool(output_format='webp', output_compression=90)])
        )


async def test_google_image_generation_silently_ignored_by_gemini_api(google_provider: GoogleProvider) -> None:
    """Test that output_format and compression are silently ignored by Gemini API (google-gla)."""
    model = GoogleModel('gemini-2.5-flash-image', provider=google_provider)

    # Test output_format ignored
    params = ModelRequestParameters(builtin_tools=[ImageGenerationTool(output_format='png')])
    _, image_config = model._get_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert image_config == {}

    # Test output_compression ignored
    params = ModelRequestParameters(builtin_tools=[ImageGenerationTool(output_compression=90)])
    _, image_config = model._get_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert image_config == {}

    # Test both ignored when None
    params = ModelRequestParameters(builtin_tools=[ImageGenerationTool()])
    _, image_config = model._get_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert image_config == {}


async def test_google_vertexai_image_generation_with_output_format(
    allow_model_requests: None, vertex_provider: GoogleProvider
):  # pragma: lax no cover
    """Test that output_format works with Vertex AI."""
    model = GoogleModel('gemini-2.5-flash-image', provider=vertex_provider)
    agent = Agent(
        model,
        builtin_tools=[ImageGenerationTool(output_format='jpeg', output_compression=85)],
        output_type=BinaryImage,
    )

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output.media_type == 'image/jpeg'


async def test_google_image_generation_tool_all_fields(mocker: MockerFixture, google_provider: GoogleProvider) -> None:
    """Test that all ImageGenerationTool fields are mapped correctly on Vertex AI."""
    model = GoogleModel('gemini-3-pro-image-preview', provider=google_provider)
    mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value='google-vertex')
    params = ModelRequestParameters(
        builtin_tools=[ImageGenerationTool(aspect_ratio='16:9', size='2K', output_format='jpeg', output_compression=90)]
    )

    tools, image_config = model._get_tools(params)  # pyright: ignore[reportPrivateUsage]
    assert tools is None
    assert image_config == {
        'aspect_ratio': '16:9',
        'image_size': '2K',
        'output_mime_type': 'image/jpeg',
        'output_compression_quality': 90,
    }


async def test_google_vertexai_image_generation(allow_model_requests: None, vertex_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-flash-image', provider=vertex_provider)

    agent = Agent(model, output_type=BinaryImage)

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(IsInstance(BinaryImage))


async def test_google_httpx_client_is_not_closed(allow_model_requests: None, gemini_api_key: str):
    # This should not raise any errors, see https://github.com/pydantic/pydantic-ai/issues/3242.
    agent = Agent(GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key=gemini_api_key)))
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')

    agent = Agent(GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key=gemini_api_key)))
    result = await agent.run('What is the capital of Mexico?')
    assert result.output == snapshot('The capital of Mexico is **Mexico City**.')


async def test_google_discriminated_union_native_output(allow_model_requests: None, google_provider: GoogleProvider):
    """Test discriminated unions with oneOf and discriminator field using gemini-2.5-flash."""
    from typing import Literal

    from pydantic import Field

    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    class Cat(BaseModel):
        pet_type: Literal['cat'] = 'cat'
        meow_volume: int

    class Dog(BaseModel):
        pet_type: Literal['dog'] = 'dog'
        bark_volume: int

    class PetResponse(BaseModel):
        """A response containing a pet."""

        pet: Cat | Dog = Field(discriminator='pet_type')

    agent = Agent(m, output_type=NativeOutput(PetResponse))

    result = await agent.run('Tell me about a cat with a meow volume of 5')
    assert result.output.pet.pet_type == 'cat'
    assert isinstance(result.output.pet, Cat)
    assert result.output.pet.meow_volume == snapshot(5)


async def test_google_discriminated_union_native_output_gemini_2_0(
    allow_model_requests: None, google_provider: GoogleProvider
):
    """Test discriminated unions with oneOf and discriminator field using gemini-2.0-flash."""
    from typing import Literal

    from pydantic import Field

    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class Cat(BaseModel):
        pet_type: Literal['cat'] = 'cat'
        meow_volume: int

    class Dog(BaseModel):
        pet_type: Literal['dog'] = 'dog'
        bark_volume: int

    class PetResponse(BaseModel):
        """A response containing a pet."""

        pet: Cat | Dog = Field(discriminator='pet_type')

    agent = Agent(m, output_type=NativeOutput(PetResponse))

    result = await agent.run('Tell me about a cat with a meow volume of 5')
    assert result.output.pet.pet_type == 'cat'
    assert isinstance(result.output.pet, Cat)
    assert result.output.pet.meow_volume == snapshot(5)


async def test_google_recursive_schema_native_output(allow_model_requests: None, google_provider: GoogleProvider):
    """Test recursive schemas with $ref and $defs."""
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class TreeNode(BaseModel):
        """A node in a tree structure."""

        value: str
        children: list[TreeNode] = []

    agent = Agent(m, output_type=NativeOutput(TreeNode))

    result = await agent.run('Create a simple tree with root "A" and two children "B" and "C"')
    assert result.output.value == snapshot('A')
    assert len(result.output.children) == snapshot(2)
    assert {child.value for child in result.output.children} == snapshot({'B', 'C'})


async def test_google_recursive_schema_native_output_gemini_2_5(
    allow_model_requests: None, google_provider: GoogleProvider
):
    """Test recursive schemas with $ref and $defs using gemini-2.5-flash."""
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    class TreeNode(BaseModel):
        """A node in a tree structure."""

        value: str
        children: list[TreeNode] = []

    agent = Agent(m, output_type=NativeOutput(TreeNode))

    result = await agent.run('Create a simple tree with root "A" and two children "B" and "C"')
    assert result.output.value == snapshot('A')
    assert len(result.output.children) == snapshot(2)
    assert {child.value for child in result.output.children} == snapshot({'B', 'C'})


async def test_google_dict_with_additional_properties_native_output(
    allow_model_requests: None, google_provider: GoogleProvider
):
    """Test dicts with additionalProperties using gemini-2.5-flash."""
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    class ConfigResponse(BaseModel):
        """A response with configuration metadata."""

        name: str
        metadata: dict[str, str]

    agent = Agent(m, output_type=NativeOutput(ConfigResponse))

    result = await agent.run('Create a config named "api-config" with metadata author="Alice" and version="1.0"')
    assert result.output.name == snapshot('api-config')
    assert result.output.metadata == snapshot({'author': 'Alice', 'version': '1.0'})


async def test_google_dict_with_additional_properties_native_output_gemini_2_0(
    allow_model_requests: None, google_provider: GoogleProvider
):
    """Test dicts with additionalProperties using gemini-2.0-flash."""
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class ConfigResponse(BaseModel):
        """A response with configuration metadata."""

        name: str
        metadata: dict[str, str]

    agent = Agent(m, output_type=NativeOutput(ConfigResponse))

    result = await agent.run('Create a config named "api-config" with metadata author="Alice" and version="1.0"')
    assert result.output.name == snapshot('api-config')
    assert result.output.metadata == snapshot({'author': 'Alice', 'version': '1.0'})


async def test_google_optional_fields_native_output(allow_model_requests: None, google_provider: GoogleProvider):
    """Test optional/nullable fields with type: 'null' using gemini-2.5-flash."""
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str | None = None
        population: int | None = None

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    # Test with all fields provided
    result = await agent.run('Tell me about London, UK with population 9 million')
    assert result.output.city == snapshot('London')
    assert result.output.country == snapshot('UK')
    assert result.output.population is not None

    # Test with optional fields as None
    result2 = await agent.run('Just tell me a city: Paris')
    assert result2.output.city == snapshot('Paris')


async def test_google_optional_fields_native_output_gemini_2_0(
    allow_model_requests: None, google_provider: GoogleProvider
):
    """Test optional/nullable fields with type: 'null' using gemini-2.0-flash."""
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str | None = None
        population: int | None = None

    agent = Agent(m, output_type=NativeOutput(CityLocation))

    # Test with all fields provided
    result = await agent.run('Tell me about London, UK with population 9 million')
    assert result.output.city == snapshot('London')
    assert result.output.country == snapshot('UK')
    assert result.output.population is not None

    # Test with optional fields as None
    result2 = await agent.run('Just tell me a city: Paris')
    assert result2.output.city == snapshot('Paris')


async def test_google_integer_enum_native_output(allow_model_requests: None, google_provider: GoogleProvider):
    """Test integer enums work natively without string conversion using gemini-2.5-flash."""
    from enum import IntEnum

    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    class Priority(IntEnum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3

    class Task(BaseModel):
        """A task with a priority level."""

        name: str
        priority: Priority

    agent = Agent(m, output_type=NativeOutput(Task))

    result = await agent.run('Create a task named "Fix bug" with a priority')
    assert result.output.name == snapshot('Fix bug')
    # Verify it returns a valid Priority enum (any value is fine, we're testing schema support)
    assert isinstance(result.output.priority, Priority)
    assert result.output.priority in {Priority.LOW, Priority.MEDIUM, Priority.HIGH}
    # Verify it's an actual integer value
    assert isinstance(result.output.priority.value, int)


async def test_google_integer_enum_native_output_gemini_2_0(
    allow_model_requests: None, google_provider: GoogleProvider
):
    """Test integer enums work natively without string conversion using gemini-2.0-flash."""
    from enum import IntEnum

    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class Priority(IntEnum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3

    class Task(BaseModel):
        """A task with a priority level."""

        name: str
        priority: Priority

    agent = Agent(m, output_type=NativeOutput(Task))

    result = await agent.run('Create a task named "Fix bug" with a priority')
    assert result.output.name == snapshot('Fix bug')
    # Verify it returns a valid Priority enum (any value is fine, we're testing schema support)
    assert isinstance(result.output.priority, Priority)
    assert result.output.priority in {Priority.LOW, Priority.MEDIUM, Priority.HIGH}
    # Verify it's an actual integer value
    assert isinstance(result.output.priority.value, int)


async def test_google_prefix_items_native_output(allow_model_requests: None, google_provider: GoogleProvider):
    """Test prefixItems (tuple types) work natively without conversion to items using gemini-2.5-flash."""
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    class Coordinate(BaseModel):
        """A 2D coordinate with latitude and longitude."""

        point: tuple[float, float]  # This generates prefixItems in JSON schema

    agent = Agent(m, output_type=NativeOutput(Coordinate))

    result = await agent.run('Give me coordinates for New York City: latitude 40.7128, longitude -74.0060')
    assert len(result.output.point) == snapshot(2)
    # Verify both values are floats
    assert isinstance(result.output.point[0], float)
    assert isinstance(result.output.point[1], float)
    # Rough check for NYC coordinates (latitude ~40, longitude ~-74)
    assert 40 <= result.output.point[0] <= 41
    assert -75 <= result.output.point[1] <= -73


async def test_google_prefix_items_native_output_gemini_2_0(
    allow_model_requests: None, google_provider: GoogleProvider
):
    """Test prefixItems (tuple types) work natively without conversion to items using gemini-2.0-flash."""
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class Coordinate(BaseModel):
        """A 2D coordinate with latitude and longitude."""

        point: tuple[float, float]  # This generates prefixItems in JSON schema

    agent = Agent(m, output_type=NativeOutput(Coordinate))

    result = await agent.run('Give me coordinates for New York City: latitude 40.7128, longitude -74.0060')
    assert len(result.output.point) == snapshot(2)
    # Verify both values are floats
    assert isinstance(result.output.point[0], float)
    assert isinstance(result.output.point[1], float)
    # Rough check for NYC coordinates (latitude ~40, longitude ~-74)
    assert 40 <= result.output.point[0] <= 41
    assert -75 <= result.output.point[1] <= -73


async def test_google_nested_models_without_native_output(allow_model_requests: None, google_provider: GoogleProvider):
    """
    Test that deeply nested Pydantic models work correctly WITHOUT NativeOutput.

    This is a regression test for issue #3483 where nested models were incorrectly
    treated as tool calls instead of structured output schema in v1.20.0.

    When NOT using NativeOutput, the agent should still handle nested models correctly
    by using the OutputToolset approach rather than treating nested models as separate tools.
    """
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    class NestedModel(BaseModel):
        """Represents the deepest nested level."""

        name: str = Field(..., description='Name of the item')
        value: int = Field(..., description='Value of the item')

    class MiddleModel(BaseModel):
        """Represents the middle nested level."""

        title: str = Field(..., description='Title of the page')
        items: list[NestedModel] = Field(..., description='List of nested items')

    class TopModel(BaseModel):
        """Represents the top-level structure."""

        name: str = Field(..., description='Name of the collection')
        pages: list[MiddleModel] = Field(..., description='List of pages')

    # This should work WITHOUT NativeOutput - the agent should use OutputToolset
    # and NOT treat NestedModel/MiddleModel as separate tool calls
    agent = Agent(
        m,
        output_type=TopModel,
        instructions='You are a helpful assistant that creates structured data.',
        retries=5,
    )

    result = await agent.run('Create a simple example with 2 pages, each with 2 items')

    # Verify the structure is correct
    assert isinstance(result.output, TopModel)
    assert result.output.name is not None
    assert len(result.output.pages) == snapshot(2)
    assert all(isinstance(page, MiddleModel) for page in result.output.pages)
    assert all(len(page.items) == 2 for page in result.output.pages)
    assert all(isinstance(item, NestedModel) for page in result.output.pages for item in page.items)


async def test_google_nested_models_with_native_output(allow_model_requests: None, google_provider: GoogleProvider):
    """
    Test that deeply nested Pydantic models work correctly WITH NativeOutput.

    This is the workaround for issue #3483 - using NativeOutput should always work.
    """
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    class NestedModel(BaseModel):
        """Represents the deepest nested level."""

        name: str = Field(..., description='Name of the item')
        value: int = Field(..., description='Value of the item')

    class MiddleModel(BaseModel):
        """Represents the middle nested level."""

        title: str = Field(..., description='Title of the page')
        items: list[NestedModel] = Field(..., description='List of nested items')

    class TopModel(BaseModel):
        """Represents the top-level structure."""

        name: str = Field(..., description='Name of the collection')
        pages: list[MiddleModel] = Field(..., description='List of pages')

    # This should work WITH NativeOutput - uses native JSON schema structured output
    agent = Agent(
        m,
        output_type=NativeOutput(TopModel),
        instructions='You are a helpful assistant that creates structured data.',
    )

    result = await agent.run('Create a simple example with 2 pages, each with 2 items')

    # Verify the structure is correct
    assert isinstance(result.output, TopModel)
    assert result.output.name is not None
    assert len(result.output.pages) == snapshot(2)
    assert all(isinstance(page, MiddleModel) for page in result.output.pages)
    assert all(len(page.items) == 2 for page in result.output.pages)
    assert all(isinstance(item, NestedModel) for page in result.output.pages for item in page.items)


def test_google_process_response_filters_empty_text_parts(google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-pro', provider=google_provider)
    response = _generate_response_with_texts(response_id='resp-123', texts=['', 'first', '', 'second'])

    result = model._process_response(response)  # pyright: ignore[reportPrivateUsage]

    assert result.parts == snapshot([TextPart(content='first'), TextPart(content='second')])


def test_google_process_response_empty_candidates(google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-pro', provider=google_provider)
    response = GenerateContentResponse.model_validate(
        {
            'response_id': 'resp-456',
            'candidates': [],
        }
    )
    result = model._process_response(response)  # pyright: ignore[reportPrivateUsage]

    assert result == snapshot(
        ModelResponse(
            parts=[],
            model_name='gemini-2.5-pro',
            timestamp=IsDatetime(),
            provider_name='google-gla',
            provider_url='https://generativelanguage.googleapis.com/',
            provider_response_id='resp-456',
        )
    )


async def test_gemini_streamed_response_emits_text_events_for_non_empty_parts():
    chunk = _generate_response_with_texts('stream-1', ['', 'streamed text'])

    async def response_iterator() -> AsyncIterator[GenerateContentResponse]:
        yield chunk

    streamed_response = GeminiStreamedResponse(
        model_request_parameters=ModelRequestParameters(),
        _model_name='gemini-test',
        _response=response_iterator(),
        _timestamp=IsDatetime(),
        _provider_name='test-provider',
        _provider_url='',
    )

    events = [event async for event in streamed_response._get_event_iterator()]  # pyright: ignore[reportPrivateUsage]
    assert events == snapshot([PartStartEvent(index=0, part=TextPart(content='streamed text'))])


async def _cleanup_file_search_store(store: Any, client: Any) -> None:  # pragma: lax no cover
    """Helper function to clean up a file search store if it exists."""
    if store is not None and store.name is not None:
        await client.aio.file_search_stores.delete(name=store.name, config={'force': True})


def _generate_response_with_texts(response_id: str, texts: list[str]) -> GenerateContentResponse:
    return GenerateContentResponse.model_validate(
        {
            'response_id': response_id,
            'model_version': 'gemini-test',
            'usage_metadata': GenerateContentResponseUsageMetadata(
                prompt_token_count=0,
                candidates_token_count=0,
            ),
            'candidates': [
                {
                    'finish_reason': GoogleFinishReason.STOP,
                    'content': {
                        'role': 'model',
                        'parts': [{'text': text} for text in texts],
                    },
                }
            ],
        }
    )


@pytest.mark.vcr()
async def test_google_model_file_search_tool(allow_model_requests: None, google_provider: GoogleProvider):
    client = google_provider.client

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write('Paris is the capital of France. The Eiffel Tower is a famous landmark in Paris.')
        test_file_path = f.name

    store = None
    try:
        store = await client.aio.file_search_stores.create(config={'display_name': 'test-file-search-store'})
        assert store.name is not None

        with open(test_file_path, 'rb') as f:
            await client.aio.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store.name, file=f, config={'mime_type': 'text/plain'}
            )

        m = GoogleModel('gemini-2.5-pro', provider=google_provider)
        agent = Agent(
            m,
            system_prompt='You are a helpful assistant.',
            builtin_tools=[FileSearchTool(file_store_ids=[store.name])],
        )

        result = await agent.run('What is the capital of France?')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        SystemPromptPart(
                            content='You are a helpful assistant.',
                            timestamp=IsDatetime(),
                        ),
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=IsDatetime(),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        BuiltinToolCallPart(
                            tool_name='file_search',
                            args={},
                            tool_call_id=IsStr(),
                            provider_name='google-gla',
                        ),
                        BuiltinToolReturnPart(
                            tool_name='file_search',
                            content=[
                                {
                                    'text': 'Paris is the capital of France. The Eiffel Tower is a famous landmark in Paris.'
                                }
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='google-gla',
                        ),
                        TextPart(
                            content='The capital of France is Paris. Paris is also known for its famous landmarks, such as the Eiffel Tower.'
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=15,
                        output_tokens=585,
                        details={
                            'thoughts_tokens': 257,
                            'tool_use_prompt_tokens': 288,
                            'text_prompt_tokens': 15,
                            'text_tool_use_prompt_tokens': 288,
                        },
                    ),
                    model_name='gemini-2.5-pro',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

        messages = result.all_messages()
        result = await agent.run(user_prompt='Tell me about the Eiffel Tower.', message_history=messages)
        assert result.new_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Tell me about the Eiffel Tower.',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        BuiltinToolCallPart(
                            tool_name='file_search',
                            args={},
                            tool_call_id=IsStr(),
                            provider_name='google-gla',
                        ),
                        BuiltinToolReturnPart(
                            tool_name='file_search',
                            content=[
                                {
                                    'text': 'Paris is the capital of France. The Eiffel Tower is a famous landmark in Paris.'
                                },
                                {
                                    'text': 'Paris is the capital of France. The Eiffel Tower is a famous landmark in Paris.'
                                },
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='google-gla',
                        ),
                        TextPart(
                            content="""\
The Eiffel Tower is a world-renowned landmark located in Paris, the capital of France. It is a wrought-iron lattice tower situated on the Champ de Mars.

Here are some key facts about the Eiffel Tower:
*   **Creator:** The tower was designed and built by the company of French civil engineer Gustave Eiffel, and it is named after him.
*   **Construction:** It was constructed from 1887 to 1889 to serve as the entrance arch for the 1889 World's Fair.
*   **Height:** The tower is 330 meters (1,083 feet) tall, which is about the same height as an 81-story building. It was the tallest man-made structure in the world for 41 years until the Chrysler Building in New York City was completed in 1930.
*   **Tourism:** It is one of the most visited paid monuments in the world, attracting millions of visitors each year. The tower has three levels for visitors, with restaurants on the first and second levels. The top level's upper platform is 276 meters (906 feet) above the ground, making it the highest observation deck accessible to the public in the European Union.\
"""
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=46,
                        output_tokens=2709,
                        details={
                            'thoughts_tokens': 980,
                            'tool_use_prompt_tokens': 1436,
                            'text_prompt_tokens': 46,
                            'text_tool_use_prompt_tokens': 1436,
                        },
                    ),
                    model_name='gemini-2.5-pro',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    finally:
        os.unlink(test_file_path)
        await _cleanup_file_search_store(store, client)


@pytest.mark.vcr()
async def test_google_model_file_search_tool_stream(allow_model_requests: None, google_provider: GoogleProvider):
    client = google_provider.client

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write('Paris is the capital of France. The Eiffel Tower is a famous landmark in Paris.')
        test_file_path = f.name

    store = None
    try:
        store = await client.aio.file_search_stores.create(config={'display_name': 'test-file-search-stream'})
        assert store.name is not None

        with open(test_file_path, 'rb') as f:
            await client.aio.file_search_stores.upload_to_file_search_store(
                file_search_store_name=store.name, file=f, config={'mime_type': 'text/plain'}
            )

        m = GoogleModel('gemini-2.5-pro', provider=google_provider)
        agent = Agent(
            m,
            system_prompt='You are a helpful assistant.',
            builtin_tools=[FileSearchTool(file_store_ids=[store.name])],
        )

        event_parts: list[Any] = []
        async with agent.iter(user_prompt='What is the capital of France?') as agent_run:
            async for node in agent_run:
                if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                    async with node.stream(agent_run.ctx) as request_stream:
                        async for event in request_stream:
                            event_parts.append(event)

        assert agent_run.result is not None
        messages = agent_run.result.all_messages()
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        SystemPromptPart(
                            content='You are a helpful assistant.',
                            timestamp=IsDatetime(),
                        ),
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=IsDatetime(),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        BuiltinToolCallPart(
                            tool_name='file_search',
                            args={'query': 'Capital of France'},
                            tool_call_id=IsStr(),
                            provider_name='google-gla',
                        ),
                        TextPart(
                            content='The capital of France is Paris. The city is well-known for its famous landmarks, including the Eiffel Tower.'
                        ),
                        BuiltinToolReturnPart(
                            tool_name='file_search',
                            content=[
                                {
                                    'text': 'Paris is the capital of France. The Eiffel Tower is a famous landmark in Paris.'
                                }
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='google-gla',
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=15,
                        output_tokens=1549,
                        details={
                            'thoughts_tokens': 742,
                            'tool_use_prompt_tokens': 770,
                            'text_prompt_tokens': 15,
                            'text_tool_use_prompt_tokens': 770,
                        },
                    ),
                    model_name='gemini-2.5-pro',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_url='https://generativelanguage.googleapis.com/',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

        assert event_parts == snapshot(
            [
                PartStartEvent(
                    index=0,
                    part=BuiltinToolCallPart(
                        tool_name='file_search',
                        args={'query': 'Capital of France'},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                ),
                PartEndEvent(
                    index=0,
                    part=BuiltinToolCallPart(
                        tool_name='file_search',
                        args={'query': 'Capital of France'},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    next_part_kind='text',
                ),
                PartStartEvent(
                    index=1,
                    part=TextPart(content='The capital of France'),
                    previous_part_kind='builtin-tool-call',
                ),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(
                    index=1,
                    delta=TextPartDelta(content_delta=' is Paris. The city is well-known for its'),
                ),
                PartDeltaEvent(
                    index=1,
                    delta=TextPartDelta(content_delta=' famous landmarks, including the Eiffel Tower.'),
                ),
                PartEndEvent(
                    index=1,
                    part=TextPart(
                        content='The capital of France is Paris. The city is well-known for its famous landmarks, including the Eiffel Tower.'
                    ),
                    next_part_kind='builtin-tool-return',
                ),
                PartStartEvent(
                    index=2,
                    part=BuiltinToolReturnPart(
                        tool_name='file_search',
                        content=[
                            {'text': 'Paris is the capital of France. The Eiffel Tower is a famous landmark in Paris.'}
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    previous_part_kind='text',
                ),
                BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                    part=BuiltinToolCallPart(
                        tool_name='file_search',
                        args={'query': 'Capital of France'},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    )
                ),
                BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                    result=BuiltinToolReturnPart(
                        tool_name='file_search',
                        content=[
                            {'text': 'Paris is the capital of France. The Eiffel Tower is a famous landmark in Paris.'}
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    )
                ),
            ]
        )

    finally:
        os.unlink(test_file_path)
        await _cleanup_file_search_store(store, client)


async def test_cache_point_filtering():
    """Test that CachePoint is filtered out in Google internal method."""
    from pydantic_ai import CachePoint

    # Create a minimal GoogleModel instance to test _map_user_prompt
    model = GoogleModel('gemini-1.5-flash', provider=GoogleProvider(api_key='test-key'))

    # Test that CachePoint in a list is handled (triggers line 606)
    content = await model._map_user_prompt(UserPromptPart(content=['text before', CachePoint(), 'text after']))  # pyright: ignore[reportPrivateUsage]

    # CachePoint should be filtered out, only text content should remain
    assert len(content) == 2
    assert content[0] == {'text': 'text before'}
    assert content[1] == {'text': 'text after'}


# =============================================================================
# GCS VideoUrl tests for google-vertex
#
# GCS URIs (gs://...) with vendor_metadata (video offsets) only work on
# google-vertex because Vertex AI can access GCS buckets directly.
#
# Regression test for https://github.com/pydantic/pydantic-ai/issues/3805
# =============================================================================


async def test_gcs_video_url_with_vendor_metadata_on_google_vertex(mocker: MockerFixture):
    """GCS URIs use file_uri with video_metadata on google-vertex.

    This is the main fix - GCS URIs were previously falling through to FileUrl
    handling which doesn't pass vendor_metadata as video_metadata.
    """
    model = GoogleModel('gemini-1.5-flash', provider=GoogleProvider(api_key='test-key'))
    mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value='google-vertex')

    video = VideoUrl(
        url='gs://bucket/video.mp4',
        vendor_metadata={'start_offset': '300s', 'end_offset': '330s'},
    )
    content = await model._map_user_prompt(UserPromptPart(content=[video]))  # pyright: ignore[reportPrivateUsage]

    assert len(content) == 1
    assert content[0] == {
        'file_data': {'file_uri': 'gs://bucket/video.mp4', 'mime_type': 'video/mp4'},
        'video_metadata': {'start_offset': '300s', 'end_offset': '330s'},
    }


async def test_gcs_video_url_raises_error_on_google_gla():
    """GCS URIs on google-gla fall through to FileUrl and raise a clear error.

    google-gla cannot access GCS buckets, so attempting to use gs:// URLs
    should fail with a helpful error message rather than a cryptic API error.
    """
    model = GoogleModel('gemini-1.5-flash', provider=GoogleProvider(api_key='test-key'))
    # google-gla is the default for GoogleProvider with api_key, but be explicit
    assert model.system == 'google-gla'

    video = VideoUrl(url='gs://bucket/video.mp4')

    with pytest.raises(UserError, match='Downloading from protocol "gs://" is not supported'):
        await model._map_user_prompt(UserPromptPart(content=[video]))  # pyright: ignore[reportPrivateUsage]


# =============================================================================
# HTTP VideoUrl fallback tests (not YouTube, not GCS)
#
# HTTP VideoUrls fall through to FileUrl handling, which is provider-specific:
# - google-gla: downloads the video and sends inline_data
# - google-vertex: uses file_uri directly (no download)
# =============================================================================


async def test_http_video_url_downloads_on_google_gla(mocker: MockerFixture):
    """HTTP VideoUrls are downloaded on google-gla with video_metadata preserved."""
    model = GoogleModel('gemini-1.5-flash', provider=GoogleProvider(api_key='test-key'))

    mock_download = mocker.patch(
        'pydantic_ai.models.google.download_item',
        return_value={'data': b'fake video data', 'data_type': 'video/mp4'},
    )

    video = VideoUrl(
        url='https://example.com/video.mp4',
        vendor_metadata={'start_offset': '10s', 'end_offset': '20s'},
    )
    content = await model._map_user_prompt(UserPromptPart(content=[video]))  # pyright: ignore[reportPrivateUsage]

    mock_download.assert_called_once()
    assert len(content) == 1
    assert 'inline_data' in content[0]
    assert 'file_data' not in content[0]
    # video_metadata is preserved even when video is downloaded
    assert content[0].get('video_metadata') == {'start_offset': '10s', 'end_offset': '20s'}


async def test_http_video_url_uses_file_uri_on_google_vertex(mocker: MockerFixture):
    """HTTP VideoUrls use file_uri directly on google-vertex with video_metadata."""
    model = GoogleModel('gemini-1.5-flash', provider=GoogleProvider(api_key='test-key'))
    mocker.patch.object(GoogleModel, 'system', new_callable=mocker.PropertyMock, return_value='google-vertex')

    video = VideoUrl(
        url='https://example.com/video.mp4',
        vendor_metadata={'start_offset': '10s', 'end_offset': '20s'},
    )
    content = await model._map_user_prompt(UserPromptPart(content=[video]))  # pyright: ignore[reportPrivateUsage]

    assert len(content) == 1
    assert content[0] == {
        'file_data': {'file_uri': 'https://example.com/video.mp4', 'mime_type': 'video/mp4'},
        'video_metadata': {'start_offset': '10s', 'end_offset': '20s'},
    }


async def test_thinking_with_tool_calls_from_other_model(
    allow_model_requests: None, google_provider: GoogleProvider, openai_api_key: str
):
    openai_model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent()

    @agent.tool_plain
    def get_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the capital of the country?', model=openai_model)
    assert result.output == snapshot('Mexico City (Ciudad de México).')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of the country?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id=IsStr(),
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='get_country', args='{}', tool_call_id=IsStr(), id=IsStr(), provider_name='openai'
                    ),
                ],
                usage=RequestUsage(input_tokens=37, output_tokens=272, details={'reasoning_tokens': 256}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime.datetime(2025, 11, 21, 21, 57, 19, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id=IsStr(),
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    TextPart(content='Mexico City (Ciudad de México).', id=IsStr(), provider_name='openai'),
                ],
                usage=RequestUsage(input_tokens=379, output_tokens=77, details={'reasoning_tokens': 64}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime.datetime(2025, 11, 21, 21, 57, 25, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    model = GoogleModel('gemini-3-pro-preview', provider=google_provider)

    result = await agent.run(model=model, message_history=messages[:-1], output_type=CityLocation)
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.new_messages() == snapshot(
        [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'city': 'Mexico City', 'country': 'Mexico'},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=107, output_tokens=146, details={'thoughts_tokens': 123, 'text_prompt_tokens': 107}
                ),
                model_name='gemini-3-pro-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.parametrize(
    'error_class,error_response,expected_status',
    [
        (
            errors.ServerError,
            {'error': {'code': 503, 'message': 'The service is currently unavailable.', 'status': 'UNAVAILABLE'}},
            503,
        ),
        (
            errors.ClientError,
            {'error': {'code': 400, 'message': 'Invalid request parameters', 'status': 'INVALID_ARGUMENT'}},
            400,
        ),
        (
            errors.ClientError,
            {'error': {'code': 429, 'message': 'Rate limit exceeded', 'status': 'RESOURCE_EXHAUSTED'}},
            429,
        ),
    ],
)
async def test_google_api_errors_are_handled(
    allow_model_requests: None,
    google_provider: GoogleProvider,
    mocker: MockerFixture,
    error_class: Any,
    error_response: dict[str, Any],
    expected_status: int,
):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    mocked_error = error_class(expected_status, error_response)
    mocker.patch.object(model.client.aio.models, 'generate_content', side_effect=mocked_error)

    agent = Agent(model=model)

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('This prompt will trigger the mocked error.')

    assert exc_info.value.status_code == expected_status
    assert error_response['error']['message'] in str(exc_info.value.body)


async def test_google_api_non_http_error(
    allow_model_requests: None,
    google_provider: GoogleProvider,
    mocker: MockerFixture,
):
    model = GoogleModel('gemini-1.5-flash', provider=google_provider)
    mocked_error = errors.APIError(302, {'error': {'code': 302, 'message': 'Redirect', 'status': 'REDIRECT'}})
    mocker.patch.object(model.client.aio.models, 'generate_content', side_effect=mocked_error)

    agent = Agent(model=model)

    with pytest.raises(ModelAPIError) as exc_info:
        await agent.run('This prompt will trigger the mocked error.')

    assert exc_info.value.model_name == 'gemini-1.5-flash'


async def test_google_model_retrying_after_empty_response(allow_model_requests: None, google_provider: GoogleProvider):
    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Hi')], timestamp=IsDatetime()),
        ModelResponse(parts=[]),
    ]

    model = GoogleModel('gemini-3-pro-preview', provider=google_provider)

    agent = Agent(model=model)

    result = await agent.run(message_history=message_history)
    assert result.output == snapshot('Hello! How can I help you today?')
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Hello! How can I help you today?',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=2, output_tokens=222, details={'thoughts_tokens': 213, 'text_prompt_tokens': 2}
                ),
                model_name='gemini-3-pro-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


def test_google_thought_signature_on_thinking_part():
    """Verify that "legacy" thought signatures stored on preceding thinking parts are handled identically
    to those stored on provider details."""

    signature = base64.b64encode(b'signature').decode('utf-8')

    old_google_response = _content_model_response(
        ModelResponse(
            parts=[
                TextPart(content='text1'),
                ThinkingPart(content='', signature=signature, provider_name='google-gla'),
                TextPart(content='text2'),
                TextPart(content='text3'),
            ],
            provider_name='google-gla',
        ),
        'google-gla',
    )
    new_google_response = _content_model_response(
        ModelResponse(
            parts=[
                TextPart(content='text1'),
                TextPart(content='text2', provider_details={'thought_signature': signature}),
                TextPart(content='text3'),
            ],
            provider_name='google-gla',
        ),
        'google-gla',
    )
    assert old_google_response == snapshot(
        {
            'role': 'model',
            'parts': [{'text': 'text1'}, {'thought_signature': b'signature', 'text': 'text2'}, {'text': 'text3'}],
        }
    )
    assert new_google_response == snapshot(
        {
            'role': 'model',
            'parts': [{'text': 'text1'}, {'thought_signature': b'signature', 'text': 'text2'}, {'text': 'text3'}],
        }
    )
    assert old_google_response == new_google_response

    old_google_response = _content_model_response(
        ModelResponse(
            parts=[
                ThinkingPart(content='thought', signature=signature, provider_name='google-gla'),
                TextPart(content='text'),
            ],
            provider_name='google-gla',
        ),
        'google-gla',
    )
    new_google_response = _content_model_response(
        ModelResponse(
            parts=[
                ThinkingPart(content='thought'),
                TextPart(content='text', provider_details={'thought_signature': signature}),
            ],
            provider_name='google-gla',
        ),
        'google-gla',
    )
    assert old_google_response == snapshot(
        {
            'role': 'model',
            'parts': [{'text': 'thought', 'thought': True}, {'thought_signature': b'signature', 'text': 'text'}],
        }
    )
    assert new_google_response == snapshot(
        {
            'role': 'model',
            'parts': [{'text': 'thought', 'thought': True}, {'thought_signature': b'signature', 'text': 'text'}],
        }
    )
    assert old_google_response == new_google_response

    old_google_response = _content_model_response(
        ModelResponse(
            parts=[
                ThinkingPart(content='thought', signature=signature, provider_name='google-gla'),
                TextPart(content='text'),
            ],
            provider_name='google-gla',
        ),
        'google-gla',
    )
    new_google_response = _content_model_response(
        ModelResponse(
            parts=[
                ThinkingPart(content='thought'),
                TextPart(content='text', provider_details={'thought_signature': signature}),
            ],
            provider_name='google-gla',
        ),
        'google-gla',
    )
    assert old_google_response == snapshot(
        {
            'role': 'model',
            'parts': [{'text': 'thought', 'thought': True}, {'thought_signature': b'signature', 'text': 'text'}],
        }
    )
    assert new_google_response == snapshot(
        {
            'role': 'model',
            'parts': [{'text': 'thought', 'thought': True}, {'thought_signature': b'signature', 'text': 'text'}],
        }
    )
    assert old_google_response == new_google_response

    # Test that thought_signature is used when item.provider_name matches even if ModelResponse.provider_name doesn't
    response_with_item_provider_name = _content_model_response(
        ModelResponse(
            parts=[
                TextPart(
                    content='text',
                    provider_name='google-gla',
                    provider_details={'thought_signature': signature},
                ),
            ],
            provider_name=None,  # ModelResponse doesn't have provider_name set
        ),
        'google-gla',
    )
    assert response_with_item_provider_name == snapshot(
        {'role': 'model', 'parts': [{'thought_signature': b'signature', 'text': 'text'}]}
    )

    # Also test when ModelResponse has a different provider_name (e.g., from another provider)
    response_with_different_provider = _content_model_response(
        ModelResponse(
            parts=[
                TextPart(
                    content='text',
                    provider_name='google-gla',
                    provider_details={'thought_signature': signature},
                ),
            ],
            provider_name='openai',  # Different provider on ModelResponse
        ),
        'google-gla',
    )
    assert response_with_different_provider == snapshot(
        {'role': 'model', 'parts': [{'thought_signature': b'signature', 'text': 'text'}]}
    )


def test_google_missing_tool_call_thought_signature():
    google_response = _content_model_response(
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='tool', args={}, tool_call_id='tool_call_id'),
                ToolCallPart(tool_name='tool2', args={}, tool_call_id='tool_call_id2'),
            ],
            provider_name='openai',
        ),
        'google-gla',
    )
    assert google_response == snapshot(
        {
            'role': 'model',
            'parts': [
                {
                    'function_call': {'name': 'tool', 'args': {}, 'id': 'tool_call_id'},
                    'thought_signature': b'skip_thought_signature_validator',
                },
                {'function_call': {'name': 'tool2', 'args': {}, 'id': 'tool_call_id2'}},
            ],
        }
    )


async def test_google_streaming_tool_call_thought_signature(
    allow_model_requests: None, google_provider: GoogleProvider
):
    model = GoogleModel('gemini-3-pro-preview', provider=google_provider)
    agent = Agent(model=model)

    @agent.tool_plain
    def get_country() -> str:
        return 'Mexico'

    events: list[AgentStreamEvent] = []
    result: AgentRunResult | None = None
    async for event in agent.run_stream_events('What is the capital of the user country? Call the tool'):
        if isinstance(event, AgentRunResultEvent):
            result = event.result
        else:
            events.append(event)

    assert result is not None
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of the user country? Call the tool',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_country',
                        args={},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=29, output_tokens=212, details={'thoughts_tokens': 202, 'text_prompt_tokens': 29}
                ),
                model_name='gemini-3-pro-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_country',
                        content='Mexico',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of Mexico is Mexico City.')],
                usage=RequestUsage(input_tokens=257, output_tokens=8, details={'text_prompt_tokens': 257}),
                model_name='gemini-3-pro-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(
                    tool_name='get_country',
                    args={},
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                    provider_details={'thought_signature': IsStr()},
                ),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(
                    tool_name='get_country',
                    args={},
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                    provider_details={'thought_signature': IsStr()},
                ),
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='get_country',
                    args={},
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                    provider_details={'thought_signature': IsStr()},
                )
            ),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_country',
                    content='Mexico',
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='The capital of Mexico')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(content_delta=' is Mexico City.'),
            ),
            PartEndEvent(
                index=0,
                part=TextPart(content='The capital of Mexico is Mexico City.'),
            ),
        ]
    )


async def test_google_system_prompts_and_instructions_ordering(google_provider: GoogleProvider):
    """Test that instructions are appended after all system prompts in the system instruction."""
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System prompt 1'),
                SystemPromptPart(content='System prompt 2'),
                UserPromptPart(content='Hello'),
            ],
            instructions='Instructions content',
        ),
    ]

    system_instruction, contents = await m._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

    # Verify system parts are in order: system1, system2, instructions
    assert system_instruction == snapshot(
        {
            'role': 'user',
            'parts': [{'text': 'System prompt 1'}, {'text': 'System prompt 2'}, {'text': 'Instructions content'}],
        }
    )
    assert contents == snapshot([{'role': 'user', 'parts': [{'text': 'Hello'}]}])


async def test_google_stream_safety_filter(
    allow_model_requests: None, google_provider: GoogleProvider, mocker: MockerFixture
):
    """Test that safety ratings are captured in the exception body when streaming."""
    model_name = 'gemini-2.5-flash'
    model = GoogleModel(model_name, provider=google_provider)

    safety_rating = mocker.Mock(category='HARM_CATEGORY_HATE_SPEECH', probability='HIGH', blocked=True)

    safety_rating.model_dump.return_value = {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'probability': 'HIGH',
        'blocked': True,
    }

    candidate = mocker.Mock(
        finish_reason=GoogleFinishReason.SAFETY,
        content=None,
        safety_ratings=[safety_rating],
        grounding_metadata=None,
        url_context_metadata=None,
    )

    chunk = mocker.Mock(
        candidates=[candidate],
        model_version=model_name,
        usage_metadata=None,
        create_time=datetime.datetime.now(),
        response_id='resp_123',
    )
    chunk.model_dump_json.return_value = '{"mock": "json"}'

    async def stream_iterator():
        yield chunk

    mocker.patch.object(model.client.aio.models, 'generate_content_stream', return_value=stream_iterator())

    agent = Agent(model=model)

    with pytest.raises(ContentFilterError) as exc_info:
        async with agent.run_stream('bad content'):
            pass

    # Verify exception message
    assert 'Content filter triggered' in str(exc_info.value)

    # Verify safety ratings are present in the body (serialized ModelResponse)
    assert exc_info.value.body is not None
    body_json = json.loads(exc_info.value.body)

    # body_json is a list of messages, check the first one
    response_msg = body_json[0]
    assert response_msg['provider_details']['finish_reason'] == 'SAFETY'
    assert response_msg['provider_details']['safety_ratings'][0]['category'] == 'HARM_CATEGORY_HATE_SPEECH'


def test_google_provider_sets_http_options_timeout(google_provider: GoogleProvider):
    """Test that GoogleProvider sets HttpOptions.timeout to prevent requests hanging indefinitely.

    The google-genai SDK's HttpOptions.timeout defaults to None, which causes the SDK to
    explicitly pass timeout=None to httpx, overriding any timeout configured on the httpx
    client. This would cause requests to hang indefinitely.

    See https://github.com/pydantic/pydantic-ai/issues/4031
    """
    http_options = google_provider._client._api_client._http_options  # pyright: ignore[reportPrivateUsage]
    assert http_options.timeout == DEFAULT_HTTP_TIMEOUT * 1000


def test_google_provider_respects_custom_http_client_timeout(gemini_api_key: str):
    """Test that GoogleProvider respects a custom timeout from a user-provided http_client.

    See https://github.com/pydantic/pydantic-ai/pull/4032#discussion_r2709797127
    """
    custom_timeout = 120
    custom_http_client = HttpxAsyncClient(timeout=Timeout(custom_timeout))
    provider = GoogleProvider(api_key=gemini_api_key, http_client=custom_http_client)

    http_options = provider._client._api_client._http_options  # pyright: ignore[reportPrivateUsage]
    assert http_options.timeout == custom_timeout * 1000


async def test_google_splits_tool_return_from_user_prompt(google_provider: GoogleProvider):
    """Test that ToolReturnPart and UserPromptPart are split into separate content objects.

    TODO: Remove workaround when https://github.com/pydantic/pydantic-ai/issues/3763 is resolved
    """
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    # ToolReturn + UserPrompt
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='final_result', content='Final result processed.', tool_call_id='test_id'),
                UserPromptPart(content="What's 2 + 2?"),
            ]
        )
    ]

    _, contents = await m._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

    assert contents == snapshot(
        [
            {
                'role': 'user',
                'parts': [
                    {
                        'function_response': {
                            'name': 'final_result',
                            'response': {'return_value': 'Final result processed.'},
                            'id': 'test_id',
                        }
                    }
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {
                        'text': "What's 2 + 2?",
                    }
                ],
            },
        ]
    )

    # ToolReturn + Retry + UserPrompts
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='final_result', content='Final result processed.', tool_call_id='test_id_1'),
                RetryPromptPart(content='Tool error occurred', tool_name='another_tool', tool_call_id='test_id_2'),
                UserPromptPart(content="What's 2 + 2?"),
                UserPromptPart(content="What's 3 + 3?"),
            ]
        )
    ]

    _, contents = await m._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

    assert contents == snapshot(
        [
            {
                'role': 'user',
                'parts': [
                    {
                        'function_response': {
                            'name': 'final_result',
                            'response': {'return_value': 'Final result processed.'},
                            'id': 'test_id_1',
                        }
                    },
                    {
                        'function_response': {
                            'name': 'another_tool',
                            'response': {'error': 'Tool error occurred\n\nFix the errors and try again.'},
                            'id': 'test_id_2',
                        }
                    },
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {
                        'text': "What's 2 + 2?",
                    },
                    {
                        'text': "What's 3 + 3?",
                    },
                ],
            },
        ]
    )

    # ToolReturn only
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='final_result', content='Final result processed.', tool_call_id='test_id'),
            ]
        )
    ]

    _, contents = await m._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

    assert contents == snapshot(
        [
            {
                'role': 'user',
                'parts': [
                    {
                        'function_response': {
                            'name': 'final_result',
                            'response': {'return_value': 'Final result processed.'},
                            'id': 'test_id',
                        }
                    },
                ],
            }
        ]
    )
