from __future__ import annotations as _annotations

import base64
import datetime
import json
import os
import random
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

if not imports_successful():  # pragma: lax no cover
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
    model = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'temperature': 0.0})
    async with agent.run_stream('What is the capital of France?') as result:
        data = await result.get_output()
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                assert response == snapshot(
                    ModelResponse(
                        parts=[TextPart(content='The capital of France is **Paris**.')],
                        usage=RequestUsage(
                            input_tokens=15,
                            output_tokens=24,
                            details={'thoughts_tokens': 16, 'text_prompt_tokens': 15},
                        ),
                        model_name='gemini-2.5-flash',
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                        provider_url='https://generativelanguage.googleapis.com/',
                        provider_details={'finish_reason': 'STOP'},
                        provider_response_id=IsStr(),
                        finish_reason='stop',
                    )
                )
    assert data == snapshot('The capital of France is **Paris**.')


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
    print(65465 - 6544 * 65464 - 6 + 1.02255)
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
                    TextPart(
                        content="""\
Following the order of operations (PEMDAS/BODMAS), the calculation is performed as follows:

1.  Multiplication: `6544 * 65464`
2.  Subtraction: `65465 - (the result from step 1)`
3.  Subtraction: `(the result from step 2) - 6`
4.  Addition: `(the result from step 3) + 1.02255`

Here is the calculation performed using Python:
"""
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={'code': 'print(65465 - 6544 * 65464 - 6 + 1.02255)\n', 'language': 'PYTHON'},
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
                    TextPart(
                        content='The result of the expression `65465-6544 * 65464-6+1.02255` is **-428,330,955.97745**.'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=46,
                    output_tokens=2066,
                    details={
                        'thoughts_tokens': 563,
                        'tool_use_prompt_tokens': 1268,
                        'text_prompt_tokens': 46,
                        'text_tool_use_prompt_tokens': 1268,
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
    print(65465 - 6544 * 65464 - 6 + 1.02255)
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
    print(65465 - 6544 * 65464 - 6 + 1.02255)
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
                part=TextPart(
                    content='Following the order of operations (PEMDAS/BODMAS), the calculation is performed as'
                ),
                previous_part_kind='builtin-tool-return',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=2,
                delta=TextPartDelta(
                    content_delta="""\
 follows:

1.  Multiplication: `6544 * 65464`
2.  Subtraction\
"""
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=TextPartDelta(
                    content_delta="""\
: `65465 - (the result from step 1)`
3.  Subtraction: `(the\
"""
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=TextPartDelta(
                    content_delta="""\
 result from step 2) - 6`
4.  Addition: `(the result from step 3\
"""
                ),
            ),
            PartDeltaEvent(
                index=2,
                delta=TextPartDelta(
                    content_delta="""\
) + 1.02255`

Here is the calculation performed using Python:
"""
                ),
            ),
            PartEndEvent(
                index=2,
                part=TextPart(
                    content="""\
Following the order of operations (PEMDAS/BODMAS), the calculation is performed as follows:

1.  Multiplication: `6544 * 65464`
2.  Subtraction: `65465 - (the result from step 1)`
3.  Subtraction: `(the result from step 2) - 6`
4.  Addition: `(the result from step 3) + 1.02255`

Here is the calculation performed using Python:
"""
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={'code': 'print(65465 - 6544 * 65464 - 6 + 1.02255)\n', 'language': 'PYTHON'},
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                ),
                previous_part_kind='text',
            ),
            PartEndEvent(
                index=3,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={'code': 'print(65465 - 6544 * 65464 - 6 + 1.02255)\n', 'language': 'PYTHON'},
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=4,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n'},
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(index=5, part=TextPart(content='The'), previous_part_kind='builtin-tool-return'),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=' result of the expression `65465-6544 * 65')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta='464-6+1.02255` is **-428,330')),
            PartDeltaEvent(index=5, delta=TextPartDelta(content_delta=',955.97745**.')),
            PartEndEvent(
                index=5,
                part=TextPart(
                    content='The result of the expression `65465-6544 * 65464-6+1.02255` is **-428,330,955.97745**.'
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={
                        'code': """\
    print(65465 - 6544 * 65464 - 6 + 1.02255)
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
                    args={'code': 'print(65465 - 6544 * 65464 - 6 + 1.02255)\n', 'language': 'PYTHON'},
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
                    input_tokens=57, output_tokens=126, details={'thoughts_tokens': 111, 'text_prompt_tokens': 57}
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
                        provider_details={
                            'thought_signature': 'CtUFAXLI2nxgt22EiUCfSwcU1wo3pJK6Bowq5Xk//KMUvV1rQrJVEucURHd7XVEAAxoGXfbMjNzrrAzSftCyBS7351SA9/xu5/odnjjFizPSn5gWYVMgON/qobXE7CXBvHwI5e8fR+BApvZKQsQC+70AmLFdAHgoV6aX2lo7wU7zxaAee0zTRDh9QjSKngUZ1/+H7a/8MxFSsnpoJ3uHwsE6ogbZE/C5Guk/S/5CqeozTJ4ZbWz6HuFgBHhLy8etMH2QAHpdd8MWnoIOHuLTiQ4+8c0op0fI8OQZfoix21MWjjSEAwqta5eKEd+qHMCgI2zodld57b7DaSdht/b2Q43fFriTsEJ15aIXlkCPMrA6ed4abDtUuLYl26aGnWmUYZqDkGR78vcJccThxYgEwM4Y8OlKg/kPuNstt5gMnENKpjXrBQaLQ/tAnXhNyJUSHJw9GvcVv9sus2MOL8TdWOhleKs8vu270fU/jGv3cVkK/SLzkXMlvjKvoDrXXJ5FaXLJBuE6J8cWexIamV02tE3DBYXcN5gNCHb5Rf532uBR/na1m/cHNerxOSxOpDdXy0lcPhyyWGVVB5Vdu1d7skVcRWN4Z37Yy6nXnFiixbPcUH61l2OxErWXWYNwfdQF+nKquJYEIroDAEbVof9luS6ITwH+ffe6bV1zY6NeUyilKwCqu1X6ioE+qO7RwRvQJT936BQqIsDDzdHF+GSQ2qwUOsHxbDZagmJHDwzqQEwdDK43srgxRXfi5JvfU6G/IExnlBaemR18fcqTuLCBo5ov5mvGyoexuxetJMm+3b/RYlju1xx81Sxz0l47wFTkLLhr8EtXEa6E0w0XioTRYaHsz+oLKsjdVYIMo5x6wTpoNxDq/+si4p+iNd0WwxgmBMEKvmFxr+WzAqb3QG1OcjgwfxC634THId/dqU/A9f6YLXZfE5gSgi4PGcAFC8cVXXgkS/gff60='
                        },
                    )
                ],
                usage=RequestUsage(
                    input_tokens=109, output_tokens=163, details={'thoughts_tokens': 147, 'text_prompt_tokens': 109}
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='TFd8aajpJNifz7IPuJq08AE',
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
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='The capital of France is Paris.',
                        provider_name='google-gla',
                        provider_details={
                            'thought_signature': 'CtMDAXLI2nwxeQ/v3e6EHAjLin/y91gLUDFWAYiTSofXvoMXaMjeSxHVJLqYBifB671o7/CllKzdKYP5bIlGr3z7tIlQ498NPcLdZ+tSZ5cH3FT/Yf6hza8mSxD/T5vbOZ8lLh9Yum1G7SdMvpA256cDkYrqJrc3JmSoSxfPavPedKkMHdyQ0T48MCZvDYNmKW+BjALuO7Lehba4EJvxH6FrBLuv73lqfepAydBPWhMFeE/hprw/Csu9rGlVwTjzH8D8v80rBVLQx0fTfoP4wZejtoArNe+NtWt+SAbI/Gi1oZctJI5sxP3D0HII5XpFisZcO7jt1lfwFAedu6h0ULpCeSa/8SMbZg1XvyZ8J+as43qdoll1YBEPhJZF7dS0A2LsQAVV8g+bGBs0flo5Z+OyTp/8b3S+HWiYml7Cg3ftHcv3ZTGNyWNjQiXD9FTfPARUG9lYW2vqd1l0FAVY5tqAOKhCrb83QVaAImGZWfV5Gt/b4rnhD9G+yljY3Nlr2zhEAiUvvvTtCQv9CusN6NSL5x/FA7dbJdLpjkVG9vBc1uvohGxIiDgN7a43j8nxeVyaYaV+yIc/gRiDghxABSo82S/shAa3WbEEfzKbuhEF9GM7O4M='
                        },
                    )
                ],
                usage=RequestUsage(
                    input_tokens=142, output_tokens=108, details={'thoughts_tokens': 101, 'text_prompt_tokens': 142}
                ),
                model_name='gemini-2.5-pro',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='T1d8aYMZ_tfPsg-V08HBDw',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_google_model_max_tokens(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'max_tokens': 5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is')


async def test_google_model_top_p(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings={'top_p': 0.5})
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')


async def test_google_model_thinking_config(allow_model_requests: None, google_provider: GoogleProvider):
    model = GoogleModel('gemini-2.5-pro', provider=google_provider)
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
    model = GoogleModel('gemini-3-flash-preview', provider=vertex_provider)
    agent = Agent(model=model, instructions='You are a helpful chatbot.')
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')


async def test_google_model_vertex_labels(
    allow_model_requests: None, vertex_provider: GoogleProvider
):  # pragma: lax no cover
    model = GoogleModel('gemini-3-flash-preview', provider=vertex_provider)
    settings = GoogleModelSettings(google_labels={'environment': 'test', 'team': 'analytics'})
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')


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
            PartStartEvent(index=0, part=TextPart(content='The temperature in')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' Paris is 30°C.')),
            PartEndEvent(index=0, part=TextPart(content='The temperature in Paris is 30°C.')),
        ]
    )


async def test_google_model_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(['What fruit is in the image?', image_content])
    assert result.output == snapshot('The fruit in the image is a kiwi.\n')


async def test_google_model_video_as_binary_content_input(
    allow_model_requests: None, video_content: BinaryContent, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(['Explain me this video', video_content])
    assert result.output == snapshot("""\
Certainly! Based on the image, here's what I can tell you:

The image shows a camera setup, likely for photography or videography, in an outdoor setting. A camera is mounted on a tripod and connected to an external monitor. The monitor is displaying a scene that is very similar to the background, suggesting that the camera is recording or previewing the landscape. The landscape appears to be a canyon or desert environment, with rocky formations and mountains in the background.
""")


async def test_google_model_video_as_binary_content_input_with_vendor_metadata(
    allow_model_requests: None, video_content: BinaryContent, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')
    video_content.vendor_metadata = {'fps': 2}

    result = await agent.run(['Explain me this video', video_content])
    assert result.output == snapshot("""\
This video shows a camera setup filming a picturesque outdoor landscape. Here's a breakdown:

1.  **Foreground/Subject:** A professional camera rig is prominently displayed. It consists of:
    *   A camera mounted on a tripod.
    *   An external monitor attached to the camera, displaying a live feed or a recently captured shot.

2.  **What's on the Monitor:** The monitor shows a stunning view of a rugged, natural environment. It appears to be a canyon or a mountainous desert landscape. There's a path winding through the scene, with dramatic rock formations on the left and a brightly lit, possibly distant mountain under a blue sky on the right. The lighting is warm, suggesting either sunrise or sunset ("golden hour").

3.  **Background:** The area behind the camera and monitor is blurred, but it clearly shows the *same* natural landscape that the camera is pointed at. This indicates that the camera is set up *in* the very environment it is filming, providing a "behind-the-scenes" look at the filming process. The blurred background emphasizes the crisp image on the monitor, highlighting the camera's focus.

In essence, the video captures a moment of filming a beautiful, warm-toned natural landscape, showcasing the equipment used and the stunning result displayed on the external monitor.\
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
    assert result.output == snapshot('That vegetable is a potato.')


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
Okay! Based on the image, here's what I can tell you:

**Main Elements:**

*   **Camera Setup:** The primary focus is a camera setup, likely for photography or videography. It includes a camera mounted on a tripod and connected to an external monitor.
*   **External Monitor:** The external monitor shows a scene that resembles a desert or canyon landscape. It appears to be displaying the camera's live view or a previously recorded image.
*   **Background:** The background is intentionally blurred, showing a similar desert or canyon environment, which helps to create depth and emphasizes the camera setup as the subject.

**In summary:**

The image likely showcases the equipment used for filming or photographing in a scenic outdoor location. The external monitor helps preview or review the footage.\
""")


async def test_google_model_youtube_video_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(
        [
            'Explain me this video in a few sentences',
            VideoUrl(url='https://youtu.be/lCdaVNyHtjU'),
        ]
    )
    assert result.output == snapshot(
        'This video demonstrates using an AI agent to analyze recent 404 HTTP responses from a service. The user asks the agent, "Logfire," to identify patterns in these errors. The agent then queries a Logfire database, extracts relevant information like URL paths, HTTP methods, and timestamps, and presents a detailed analysis covering common error-prone endpoints, request patterns, timeline-related issues, and potential configuration or authentication problems. Finally, it offers a list of actionable recommendations to address these issues.'
    )


async def test_google_model_youtube_video_url_input_with_vendor_metadata(
    allow_model_requests: None, google_provider: GoogleProvider
):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(
        [
            'Explain me this video in a few sentences',
            VideoUrl(
                url='https://youtu.be/lCdaVNyHtjU',
                vendor_metadata={'fps': 0.2},
            ),
        ]
    )
    assert result.output == snapshot("""\
Here's a breakdown of what's happening in the video:

**Context:**

*   The user is in a development environment, likely using VS Code, examining TypeScript code related to a browser router.
*   The user is interacting with a chat interface within the IDE, specifically using a tool called "logfile" (indicated by the bot's opening lines).

**The Task:**

*   The user asks the bot to analyze recent 404 HTTP responses from log files.  The goal is to identify any patterns that could explain why these "Not Found" errors are occurring.

**The Process:**

1.  **Logfile Schema Check:** The bot first checks the structure or schema of the log files to ensure it can accurately query the data.
2.  **Query Creation:**  Using the schema, the bot formulates a query to search for 404 responses specifically within the last 30 minutes. The query will focus on relevant information like the URL path, HTTP method (GET, POST, etc.), and timestamp.
3.  **Analysis and Pattern Identification:** The bot analyzes the query results to identify common patterns, which will include the following:

    *   **Most Common Endpoints with 404s:**
    *   **Request Patterns:**
    *   **Timeline-Related Issues:**
    *   **Organization/Project Access:**
    *   **Configuration and Authentication:**

**Key takeaways:**

*   The bot is using log data analysis to automatically identify patterns and issues related to 404 errors.
*   The analysis provided by the bot is then displayed in the chat interface to assist the user to find the errors.
*   By identifying frequent 404's the tool also recommends monitoring these 404's to track expected behavior or indicate actual issues.\
""")


async def test_google_model_youtube_video_url_input_with_start_end_offset(
    allow_model_requests: None, google_provider: GoogleProvider
):
    """Test video_metadata with start_offset/end_offset on YouTube URLs.

    Note: start_offset/end_offset only work with file_data (VideoUrl for YouTube/GCS),
    not with inline_data (BinaryContent). See test_google_model_video_as_binary_content_input_with_vendor_metadata
    for the fps-only test with BinaryContent.
    """
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    result = await agent.run(
        [
            'What happens in this specific part of the video? Be brief.',
            VideoUrl(
                url='https://youtu.be/lCdaVNyHtjU',
                vendor_metadata={'start_offset': '40s', 'end_offset': '80s'},
            ),
        ]
    )
    assert result.output == snapshot(
        'At 00:40, the user scrolls through the "Recommendations" section in the chat interface, highlighting various points related to configuration and authentication.'
    )


async def test_google_model_document_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output == snapshot('The main content of the document is the phrase "Dummy PDF file".\n')


async def test_google_model_text_document_url_input(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.0-flash', provider=google_provider)
    agent = Agent(m, instructions='You are a helpful chatbot.')

    text_document_url = DocumentUrl(url='https://example-files.online-convert.com/document/txt/example.txt')

    result = await agent.run(['What is the main content on this document?', text_document_url])
    assert result.output == snapshot(
        'The main content of the document is an explanation of the placeholder name "John Doe" (and related names like Jane Doe, Baby Doe, etc.) and its usage, primarily in legal and other contexts in the United States and Canada, when the true identity of a person is unknown or must be withheld. It also mentions alternative names used in other English-speaking countries like the UK.\n'
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

    assert result.output == snapshot(
        'Both documents contain the text "Dummy PDF file". They appear to be placeholder or example PDF files.\n'
    )


async def test_google_model_safety_settings(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)
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
                'model_name': 'gemini-2.5-flash',
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
                                'title': 'wunderground.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEtIMuE-qRD5sZj_LCu_5McCA8JLJOk7maet5sQ9DHIOSHCup7I3X2p-ERs77Ri1HmhoCKKRi31xHLo5yJKtRQz5aJz408iaDnkjYIkjX2RsaNk1g8nnzq13Qme6hVQhfXa38l-2zMctvbmAhJpaMqOtg==',
                            },
                            {
                                'domain': None,
                                'title': 'Weather information for San Francisco, CA, US',
                                'uri': 'https://www.google.com/search?q=weather+in+San Francisco, CA,+US',
                            },
                            {
                                'domain': None,
                                'title': 'accuweather.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFj_Qm-tG4ohXlBnuw_2KkZkt-9gOaSZpFQ3tOCvrZeAgo3mpoobp0f_tMfbVPptdKoLFB-V1Pj9Xt9HpYPrxUPk9-VyQpZEM2qeyG_yJ6Y6vReEk-4XfrcBVEaEiWeH5BXXsdWK75kHpfSeqFTqQRYnRBa1F9PQy8U76gISO_p5ATCADVFvpY=',
                            },
                            {
                                'domain': None,
                                'title': 'weather.gov',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG_QDYAlEKqk0Pz0unIB2ZaI8LdvAgYTlbPqmBsZZFtnV09N1BO1Ufu1m7OrNxrCspdumvT0790s7vPRYDJ7BvBtKGTUq4S46085nyPJkcK_6cED1Thcw==',
                            },
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(
                        content="""\
Here's a look at the weather in San Francisco for Friday, January 30, 2026.

**Today's forecast calls for a mix of clouds and sun, with cloudy skies in the morning giving way to partly cloudy conditions in the afternoon.** The chance of rain is low, at around 10%.

The high temperature is expected to be in the low 60s, with different sources predicting between 61°F (16°C) and 64°F. The low temperature for tonight is anticipated to be around 48°F. You can expect winds from the north-northeast at 5 to 10 mph.

**Hazardous Conditions:** It's important to be aware of a Coastal Flood Advisory in effect for bayshore locations through Sunday. Additionally, a Beach Hazards Statement is active until Monday, so caution is advised near the water.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=17,
                    output_tokens=745,
                    details={
                        'thoughts_tokens': 437,
                        'tool_use_prompt_tokens': 103,
                        'text_prompt_tokens': 17,
                        'text_tool_use_prompt_tokens': 103,
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
                        args={'queries': ['weather in Mexico City today']},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=None,
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(
                        content="""\
Here is the weather forecast for Mexico City on Friday, January 30, 2026.

You can expect a day with variable cloudiness and a slight chance of rain. The morning will begin with overcast skies and temperatures around 53°F (12°C).

As the day progresses, there is a 10% chance of passing showers in the afternoon, with the high temperature reaching approximately 70°F (21°C). The evening will be overcast with temperatures around 60°F (16°C), dropping to a low of about 50°F (10°C) overnight.

Winds are expected to be light, ranging from 2 to 8 mph.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=213,
                    output_tokens=853,
                    details={
                        'thoughts_tokens': 375,
                        'tool_use_prompt_tokens': 308,
                        'text_prompt_tokens': 213,
                        'text_tool_use_prompt_tokens': 308,
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
### Cloudy Skies with Afternoon Sun Expected in San Francisco Today

**San Francisco, CA** - Residents of San Francisco can expect a day of mixed clouds and sun today, with a high temperature hovering around the low 60s. The morning will begin with mostly cloudy skies, which are expected to give way to partly cloudy conditions and some sunshine in the afternoon. The overnight low is anticipated to be in the mid to upper 40s.

There is a low chance of rain throughout the day, with precipitation chances pegged at around 10%. Winds are forecast to be from the north-northeast at a gentle 5 to 10 miles per hour.

A Coastal Flood Advisory is currently in effect for bayshore locations along the San Francisco Bay. This advisory warns of minor coastal flooding during the highest tides of the day.

Looking ahead, the weekend is expected to bring a similar mild and dry weather pattern.\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=17,
                    output_tokens=628,
                    details={
                        'thoughts_tokens': 319,
                        'tool_use_prompt_tokens': 104,
                        'text_prompt_tokens': 17,
                        'text_tool_use_prompt_tokens': 104,
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
### Cloudy Skies with Afternoon Sun Expected in San Francisco Today

**\
"""
                ),
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta='San Francisco, CA** - Residents of San Francisco can expect a day of mixed clouds and sun today, with a high temperature'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta=' hovering around the low 60s. The morning will begin with mostly cloudy skies, which are expected to give way to partly cloudy'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta=' conditions and some sunshine in the afternoon. The overnight low is anticipated to be in'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta="""\
 the mid to upper 40s.

There is a low\
"""
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta=' chance of rain throughout the day, with precipitation chances pegged at around 10%. Winds are forecast'
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta="""\
 to be from the north-northeast at a gentle 5 to 10 miles per hour.

A Coastal Flood Advisory is currently in effect for bayshore locations along the San Francisco Bay. This advisory warns of minor coastal flooding during the highest tides of the day.

Looking\
"""
                ),
            ),
            PartDeltaEvent(
                index=0,
                delta=TextPartDelta(
                    content_delta=' ahead, the weekend is expected to bring a similar mild and dry weather pattern.'
                ),
            ),
            PartEndEvent(
                index=0,
                part=TextPart(
                    content="""\
### Cloudy Skies with Afternoon Sun Expected in San Francisco Today

**San Francisco, CA** - Residents of San Francisco can expect a day of mixed clouds and sun today, with a high temperature hovering around the low 60s. The morning will begin with mostly cloudy skies, which are expected to give way to partly cloudy conditions and some sunshine in the afternoon. The overnight low is anticipated to be in the mid to upper 40s.

There is a low chance of rain throughout the day, with precipitation chances pegged at around 10%. Winds are forecast to be from the north-northeast at a gentle 5 to 10 miles per hour.

A Coastal Flood Advisory is currently in effect for bayshore locations along the San Francisco Bay. This advisory warns of minor coastal flooding during the highest tides of the day.

Looking ahead, the weekend is expected to bring a similar mild and dry weather pattern.\
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
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHqT9N1t8LIG7wvA12kuHGFiXVVycOQWw1-6QD9MP4gIGPb2hSBro-2vCVrdGAL0BmJ0oiubiM3HgrZKcJcZULwlnO1nrH__pzNh5XnaQ9VmiuJ_1xdfvVQKmocUSSkAHYjAGrzqKVTe8Vy0BJa',
                            },
                            {
                                'domain': None,
                                'title': 'accuweather.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGtaYs8V0c3_QKMVpLlTHuKWAVT_IU79tzw3s6V4uS84oOOR2TN5lGta0JTjQUPFPVgW4W-JanN6uqT6Kta2-Y84iqM8fKY1uxgyxN4aFk-m1abZP0tRXTsVdQR4M3XaRYUSLH3QaQh9d-BQunhhsUp0YMx9q35mKdcz_I1c0blmq6wfYaOuBk=',
                            },
                            {
                                'domain': None,
                                'title': 'wunderground.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGWpQJY3pY95r0iTR7awgFhFwZ-17VxVS1DA_WhAn4E-m3XvMys6pc_zOslQ1enpCaPi4rvWG_3ES3jdhUgwYOW04fZ43KVfNZXpQLfFhcDno8iRPEQM0ERbjBmjqVC5oqS5IRp3k9L13dDlITT0w==',
                            },
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(
                        content="""\
### Mild Temperatures and Variable Clouds in Mexico City Today

**Mexico City, Mexico** - A mild day is in store for Mexico City, with a high temperature expected to reach the mid to upper 70s. The forecast indicates a mix of clouds and sun throughout the day.

This morning, skies will be mostly cloudy, gradually becoming partly cloudy as the day progresses. There is a very low chance of precipitation, with some forecasts suggesting a slight possibility of passing showers in the afternoon.

Winds are anticipated to be from the north and northeast at a speed of 5 to 15 miles per hour.

Tonight, the skies will be partly cloudy early on before becoming more overcast. The overnight low is expected to be in the upper 40s.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=212,
                    output_tokens=704,
                    details={
                        'thoughts_tokens': 239,
                        'tool_use_prompt_tokens': 293,
                        'text_prompt_tokens': 212,
                        'text_tool_use_prompt_tokens': 293,
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

    assert result.output == snapshot('Join us at the inaugural PyAI Conf in San Francisco on March 10th!')

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
                    TextPart(content='Join us at the inaugural PyAI Conf in San Francisco on March 10th!'),
                ],
                usage=IsInstance(RequestUsage),
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
            PartDeltaEvent(
                index=2,
                delta=TextPartDelta(content_delta='Join us at the inaugural PyAI Conf in San Francisco on March 10th!'),
            ),
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

utrecht_timezone = pytz.timezone('Europe/Amsterdam')
utrecht_time = datetime.now(utrecht_timezone)
print(utrecht_time.strftime('%A, %B %d, %Y'))\
""",
                            'language': 'PYTHON',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'outcome': 'OUTCOME_OK', 'output': 'Friday, January 30, 2026\n'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(content='Today in Utrecht is Friday, January 30, 2026.'),
                ],
                usage=RequestUsage(
                    input_tokens=15,
                    output_tokens=847,
                    details={
                        'thoughts_tokens': 343,
                        'tool_use_prompt_tokens': 429,
                        'text_prompt_tokens': 15,
                        'text_tool_use_prompt_tokens': 429,
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
                parts=[TextPart(content='Based on the previous answer, tomorrow will be Saturday, January 31, 2026.')],
                usage=RequestUsage(
                    input_tokens=39,
                    output_tokens=277,
                    details={'thoughts_tokens': 255, 'text_prompt_tokens': 39},
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
            [
                TextPart,
                BuiltinToolCallPart,
                BuiltinToolReturnPart,
                TextPart,
                BuiltinToolCallPart,
                BuiltinToolReturnPart,
                TextPart,
                BuiltinToolCallPart,
                BuiltinToolReturnPart,
                TextPart,
            ],
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


async def test_google_instructions_only_with_tool_calls(allow_model_requests: None, google_provider: GoogleProvider):
    """Test that tools work when using instructions-only without a user prompt.

    This tests the fix for https://github.com/pydantic/pydantic-ai/issues/3692 where the second
    request (after tool results) would fail because contents started with role=model instead of
    role=user. The fix prepends an empty user turn when the first content is a model response.
    """
    m = GoogleModel('gemini-3-flash-preview', provider=google_provider)
    agent: Agent[None, list[str]] = Agent(m, output_type=list[str])

    @agent.instructions
    def agent_instructions() -> str:
        return 'Tell three jokes. Generate topics with the generate_topic tool.'

    @agent.tool_plain
    def generate_topic() -> str:
        return random.choice(('cars', 'penguins', 'golf'))

    result = await agent.run()
    assert result.output == snapshot(
        [
            'Why did the car get a flat tire? Because there was a fork in the road!',
            'Why do golfers carry an extra pair of trousers? In case they get a hole in one!',
            "Why don't you ever see penguins in Great Britain? Because they're afraid of Wales!",
        ]
    )


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
                        content="""\
**Providing street crossing instructions**

I'm looking to craft a response for the user who asked how to cross the street safely. This is about general safety rather than legal advice. I should offer step-by-step instructions, like using crosswalks, looking both ways, and waiting for the walk signal. It's important to mention eye contact with drivers, avoiding distractions, and keeping kids close. For those with mobility challenges, I can include accessible tips like using curb ramps and audible signals.\
""",
                        id='rs_0d3e7d19e1d3306100697cc0a5b06081979f3164a57ea04a97',
                        signature='gAAAAABpfMDLUiax62V-zYQoJHemQlL-lVGvVJ7QQu9oBnz99qirSeOTFgHoBDvYOU98naLPHaDD4N98A-Ms7zZtwceexXUSwfbpuJU7bxs5l_yUUdBxYDwG8hCLUoxpXJhwJPTNBVbpck_9algTm2RzQl38Wo2dTBpO8UfTwwRMMChgxV529wgHcCXFqrBWu6nykwwZ6AmunT_lTzCEXiiluahl-8cEdOtBaxOI4-FPhcKDA8ye-z3UHWc-OJxHQf78e3Bv76oyp-ENJj03r2GOvjNyyeGZoijz2zr6_5H2mRT_nZeqwr5AAXCl5QT6Fqf8r2Ybr8Z7FlmgZdtifDHHoIC_YjU0sFBNlRmOsqJgnp5r41S4hEsM7RuDGdMP71Tmi0Uh_tIXHctoox2CQq3rDQzuuaoFm4WEl_Gr8_texfUAQ795iSxfvBAC46g4EmpHb4aUWD-rsldUB4sqm4nMLF_9IGwSqxCUecZyKP7a8V_oMr_sAAMOJAXAhR0jxaXO-NWGfY13Y3teX0DUUI6ixv7aJshdjNVO0kRO_zBPIInoWBGfX7CQzslumSfjpv__g9yF_RsEBqjxU5c1sdP14jR2NTfKxNlD2B_CMPiyKy56HUflvqRElSNTjSuvYIOCbOKS3JcYWuu135m3q_YjFyO3x8cGl2NBL6TYXvWZJZM6W6DFdLBD_EzGD_mY8FIRD7UePRFVId4esMjabK65Kv0C7YRHCYKzehvBGd7Yf1tFm6-t-NhaJNiUkPXeehcvVG0_KU2GMbLdljO1v9_4duNAYv-mRIr9r7HhX8RO7r5VzVWdJnwPVHv-9dWv3YB0giXHUePzMcsGbra_A7gcb3U30CE4I6EdZ9Ps0VFbvt2hgId_qTTbQG7KPsWRiaEWVQIs-LyEQx6HBYfu0ntcOii73erWSoU9I53Dz2NVsNHUHgqL-ilfjF3e2abj7y_SSCkZEgaz3kRsylBR7WMsVlRgk5ffKo88haVZGIU_zKLfyYHWa_EEOwVJmw2fCo7ck7FZdN7PMbX4IuJaibF4tZNoXskgkF692IstAG2bnZM5tKrCO7odNEVryBYzgtfzs6LWQfU4UXMjUrsv6G-2lWU5PMD_A7TgnnX0FgBmJh9MD5pjPTqJfQubtcn-JuLM95MTfcef5zhTNNqUs_lyL503FRWzWoVjcc2XhATsetRMOIWq4cq_A4h8DxtqKjelnV5zTevGrYBM834a7mANd1tecMOAAyAy9mbau52xuL11x8Id3UfQoSEPmX3ouVmGhsMk1V6KJAyxYjkacR2ilCz6rAHNRcLNzFBINqykxIyvEBv31odSJngkpGSsCRLgk1nLpe9qHEpQC4wG5THqu_-b4QXX4KR3q2g373WmyHSjvGNmHM0YhJvage0z6oVrdFiCk-gMy4qq2HsfkuUsFdtZFcS6pl3P-7omma4WsrghdVfb9jbWLCbqP-jWBJmLNi7uGbXxuQfdxqGlNtd6zMpBL3IQWVoT4HJBfdv2oVYeZGflr0YbMhxp5pbwDu7LEuo6YDSwjehnHRc0Psd3N0ZU4M7N_601_uTuqiQiQ_rOLnk6k5XKQtAwIfry4XeoIQOfhOWI1owXMqb38n0ZmTBIysj7qxAt0USsW4z1wn0qNtGd-kT9eYkfQjZ5PmO5aBxCCwQeFTqDjKrH4eQkwT4MfdBtFaZUYiIh0eT8smOv7fq8J7sz11IJOExSl8oSgQtCMxYHv108EIS4W4qHb0uBekS5bI9_--y6Vk0Gq5YhC8JXblsZ5I0bnihMuT4xljyikjvNXFzWfYZMvTpAkPGVpj59tERk2MV7v3z4z863bNWOPPXgT7kT1YfzFc77mdbHKBZ32jrrp1EljXf-rIxLN4ZpV6u2m6kDUwiHtFynvZGPT_ltRKrnTov0dkgpgXj9-RrihyiZFE2PQq2z5_DsYLX_a4Xtvc5YYgjQyXGcxUmREh_b7zCobRYc-T8SU-wsy2dYHrDMW0HuCz0fpD7DZWhIrg7Rf1FuU_jy3uarDamMUx3hhCb_59vEpG6JgK85-wLFmC7YL7E4K1HcJfig6euyMBMyHVDmYeKVXX47Urjw1Q2L-w14jC-0wjF9k3tYB-QC5cpHpSOLd7mD2xOigMOQKakXg4_FoJKiywanJ79qc2SpH3S3kO8uHZ4V7pxMgI_rTF_XsyOEaqHzRRuAxTsNAhvTYONwm4yuCCGWrA_9N3Vbt3w1jHojNkWmzfY52B4k85r9kWl7TAncEK0zGQw0E3ez0mBwssI0LlKbsX-jzv9l1rYjjiMmXexfX8oylnkZZHQ_9FuQ9bCzYAjA-W72FEqq24FM2JbfTZfBH9Wopsp91z85vrPRTJ4QlJSepBexC-3zoeLJpyLQDa618A1ueSTKiBahSbaiUzbqxvvIiycmAsVUSYqc8RdpnGcLKgDHmNzZ3ldbYN5xZiMpUJtuttYcEucc48LlXt7zf-_OWBQIzIhAH3e8dRf0nmpW2UaRaL1xe16uMPMFNPWJOkT5z2Gh6Qogag4il9sznmtvYAJWpPg3sAgiPCLESkiHlguPuH94lTpIjfDOoT-I9fb1F3c_wQFIHGBQxKcfoinSgyHwvRpMxglP4iBOqja-Fqf8JPDLc13TCSl0cumpFZmHDY2dw6W5eePb58dtrXWkpZz2HVBl4uIxDc3mR4zbcEfVjLiGr3NPKiJwXunzig2wGzfAj1mZZKBV8lEGksiCacFF0UfMqan9GAdBkMp73KL_W6fCBxowQ_keieJY15p_iRQ3hZb67SrXA0nBdSOLxUk8yQ_g1HtjJWN2jTVJXZeHzDxFjdAqxaaeFewtC1CuZWNMnLdhQklm5zgmJX_sk6PhY_vdvoZrNn-oedRe7pMko2h6WM8uNa4PBD81p0wVm05PYLmxHStFiF6RxZdxY6iVzsm9nldRmWtINSPKw47aHH5lsLngcvPAcvBAp3tuuYxQ1fG-42h5wFjjTHdOkE4ZRSaVG7pTK_GkIxJ7lYmEUjyn6Zy-tmQ-ukfMlDZswlsKlX3Phze4p0sc6V7hLFXXjPySzkAxOx7DEJ94nibDdi05v0VBpbMEGz9zDrzg3veQ_mtxaZHQdsRW2l9zZqYX9-hHlvmAFyg7UiJVe61H2IabOdtcP9BJ42KO663NaH77NIC-i2bthG4HWnqlOO2FReFK0R6dGq2tc09kQBKAoLYwvNA-9L4l9u6wDTEiFQOz8ynE8FN6GS9rVy_Nx9GgL4URPq9MtfXWd15_65wgZvV39p6fXVxwzvsvBiBMJUv0AyTCE8R8_rPzge00E-TcznuRapGxHDypmRHnVe9pQIB7s2tPAsnad209LBAFWMYObS2-a_wKYE5UOM3-zNrdxRRN-tQ2OhIANdxoynhk91Xzj1R42hRR4ITsC5wjAtSoj1jLWy-oFDIP9oKSeVi9J0d_pmDDLa3V367Fw6VNnrEZFM5hEYyYU1eLcji7bI-XDb44YsPnYj1nGRoI5-4LMBkbXEsP9xK7c87bmS2BhQ1BJxW_SIIzNvi_LmCW2g-o4SboBaKFDVJ-Sd2-Bp9uSo6DIZ4vC28reB4ydj9hSmk1mBh4w-uVXhCl5XoKqHEVXm9-OG8iWGgd_iNWnXgL0vqP2tD8T6fXGjNxl-3TJg7ifpfBdgmWrkb7aXcvUsFLer1F2YtEa4bNZC8hILeRD7rjkoFMyKOUuQSVv4JyWtpVCxmCj0qwWpMmhxjXO1xaokESZyBE1v6HxffdEHUsSIgpv-mz90LiMTyf1fbt6wGHMcJMw7PtLuImgRMXjwBaMoumzQBSkWlXwCkJP0scf2S4JLhSDysf3Bgg5uv2LfsZalZ1IMtQmAu8bq2LkQl2k41WmSAyXHsuD2XRU2d-gsGoDGha6s795xoTma_zhx7fUfxXEJINUCgu0V79EZuXFN9DbrB5nEX_O5E30_5P4oU46C3I_m3x8_6hOzQMT7d95UiJJgfCpPqWlOcULpDHHyur-XaFRO2yLqKy05FmiGNlkw8r_z2BmztJkBaVPuJzwOxae0FIyksWYJZWeSRjEtQwQM96TutJzvv8c0raiB2E1vwYq9HNO-8jctexwyJMbvsEUiOUCTazRTLEuFXQ1NoTzXRYMVoGHj0EoVi3LtJpuUA9xv4b9UlJJlimvfSO-_NDsWdzQoNwKX430nzRfUwemL4LROFovma8Vu6hUHywaC30Vl7FFt9G993i7oCbWP58Q3XvS65hEOyYHpD2d8LV3fE6_nwtH5t9hapKiGjup3GY5HkyLhWKB2jvgJJceUi4VaQMbbZlaLXQS-66mLCqszHINHhdRduw_7Cfv2asHzleYmOYsaIlQ5DXUPtQjcAALK5JfEhI60bW06sb4ZiYirRQGTWFX6MfZccFy0PenZLn1lpyvhMOxXVoKQ9MjkD8c1Zk3rtBIzD1mD_bli-yQxcrgHgROdBMKVh2ODQvXhMGjrlCQVDqA5VKEi2pZiRQlG1DvhzA3Ab5OvPubOvJ96JiVXHaY-m63HXf5lw4IxNQ6x-Lsp9L67jncTOaGYCQw2isej9tmjm95oG0g3Gv1zL4wK0Dmup_MbDEHRhuyDAYKOJCqA5Wrnh4DP3oxQBZRMfJDgjvplPlBYlDBv2AcKiTEGy_ZZbVcv3NsVnAJpGoCkbA--SKtes9oA0qOA7YZkRIQYe7d0NwIzMEMZSkV23ejeGJdki3fHfBvOurAg1axGk6A7EvKm8SjBbx7pibFAOWwDMf_AV9SKwhCYo2EyRsB1ntzdUM29e6gBqpWtwu1bceOACzhMKtlrXg7COuvfjX62k9R2-aadhef6qKK50v6Av1a9IvEX_TEtylcm6zVx8yRQTRIWKUd6rVlf7VwCQ0u4lTjspdUUjYDQR-l7PuDjehzGdKduiBY6x6YVKhSrL8j6RuXuerx3-xviPE6GXyhl7nWn8awZ1adBdgQ-fqXOF92wh9DSqgCM0vek5g3nCIOxWOwqPcIYP82wJ8E9YupD3TiRcmZen8kNTuzhbqkfYuF3AsptWyu5v263VzFWHRJOhlG4NOfJTCVppqWSVBaJ9Mlv8Qc4BPDin153F3BalFCkfiGqKyHCxPMV8BUkf9kk6Nvquv7l5RqbQMuNmLKWx2hnRDFxcDuIyOnHAZipfn2pVNLwDfJg7uct8W4hvpKDoRQexZkoR0sl5i1784nbV1DbH5AJyP_7y53LNidJhGdkoEBm5bbNP8mUhLNGCXOJNyM2lqWSbc3XyuD43Pv1LDZafWYoZ8q9v7kSack-IS_LUL-p31F8OU3lsE2gfwqpXAYUN74wDl7sB6Otw_hHllDTyJcClAtj-Y7W1oVxIWXxCd3zFGed2iOD1tF6oXMVJM0aDSQX1KmtTW85Ugk45yBOLWQE_lmecNWG-91AxvRLwhObxPmyX9BEAnRJCqhAyG-7zx6mDA9CISvO6kLaDgbt1Vu0qkbe7fN46N2MBButKJXgVxXmunOfovgAGLNigbe24xPozBmGv8JIyb6OgBAQFpbpXHeWkCWD8BdCtvkwgS0pdG0cQ4hNtEEU2SwJ64lZugNsR7oTurvuBCWonbQBLi5PnDlo1DhCTv_-avkVrEZV_Zou3wPZ11p-earhjb0nEtYIkP3V7Pg4qYDHD0y-uYUJaq0tKtBFbfOJk-ugGKdBBujXRm9-cXrgpfL4twkB32YmW7ijyo5PVIVXNs6hsBUknYiCbWSa2XiJ3WFRFMgKgAYIPDxfoeGFbdd-7_vxn0C7shcF6C02e6BZj6uEIXG9iLHCKoy_nhiQOhSb4EXIKRl4c5K9rfaoyRmbeV84GlmzZYtEbSKmD_RONPQOhfcOw0Rl796-vmeuLJmk_EKdup5MTr9WWP8W5dNGhQdEDGTTFT7M_IN3B72rcVFpVVk8j0ZfYHXoVMMngP7uRJ2N8VRim_n_1BMXuMWQZLzL-SIKJKnAk4dBfJhRnPNOEY8rQEfopp7k5zu6pCeY3Himhh8GZ9OX2HgDcQEWTtp7S0ZDfMnFB5YPlLqkl5R1uLVxDgzPgx6d673TRAFa-aHh3d5RrxwYEebU1p6sUUcC_5oYmgpU3gxRXlFSxPbJDWvwJobbJXmb6XeyjZkHznz21xz6R4YOY-hZ5oZdcASjm0Ao9092EV3zKzCLmOzlHQX72qtyYj3OaBisePaO9znlzjwgdX0SUC69P6-DuENrj_7cDIdPrSDZ53MYaweMuRxUBSPvYDbYptR5j7m2kC7ftsw0ZFjsvYLt73DMJzuGSIUpmhHnl6aKH12rY0YXnfu1leEpNhkLB4A8JWiykcJ9C-Oq6FMlpBNlqNaVGFGhAapZG6SGinhdxFFSZD3PUND5FvITKjhGDVnUvGjP1WPB54XbOlZvQR0c4xB4bv41qHbgeEFTNn4LN9PbNqBI-3hwQQRfN76nf_WmCnrg92OW3hWZbPFSlyTD145k3m9vavjUBueD1UpQxVrV3r-w9RhCy1bp1jxz0lIICPbs6MZ01BtTsJJ2f3-TRa3eZideKofJZY4ZgUSxhLxlP1an-wz3vvhPQ6Mhw1CAtK8o6BbGt-6wZpHY_ZvpR-E9Ev6kLu_ia26QW7F6cFiM-a2pdLbgfSfwT9PLGlTxLZpEBOKZegOJ-z_hL5ACP7RvPNxCttfXYTKWUXZL9hAeH8JlIgPd967kgoBHrqqF4_47Od0RF6GMYoM0jHCzMFDdHLK3iA5rKQKn8yJXqc7x0KUJaT2ijvZdk1hFqFV6aYg9yZtJii0bP_HYbCQNO_xfDE2CuS89y-9rKsZL9I_HJ2A7GxdUjpWT_e44HMsAyjEXDy1Mxnx0Tpqi67YvJQD5zQWGT20i_6h0M73logR9UHEVZ6b-HSb9wvaBEdNozIDcm3zW-c1vTh-PKU43lTQ3NhRfW6FwamzN0l0fihiRYUMHwxuJxEzl5w1zZSQL5a7BBmBdkvwwOWZ5_0qZDK1so30jBkL8goL3fC5n6u9ytq6tYZRwjAUjb3W2rpmDljd2MtnFWn6TzHpN11jOCOnqqAhuU16x-j6SG5m3VWgfk09Fi5ZRW4ZfkitUzZVmFLtwkY-3je8FbNrsVkjUk-3UBW-AMN1nzTDv7wqcmKEAiUpmhkLFCr1kqkG67qNdicu_6-FRiIb9vqubm2uOpLFYao-S_EeoK-V2ayMht3nHqNYlpeplcYJUvZ-QZaFhapJDwRwzPKk12x8LDLv0UCd79Z-OlrSCX-b5oY4NUe-ao-jS9lqccfJ33FPpOUAOhZprPDk2kZZ6yz6_7zgTeAEZ0KMXlG0bx4UcqAuMFnaIYYLCMWg0oZ3HufvwTEcu9LgyY2aHu6zqINl-WOMrEWf4TC4XO-ujuqat15oZXkxaNUg8Z_eajg6vENMwbqRbVOwVCI6Vgu7g-sSu48vCCsAVOrSeKXFfzKiw5G-f1pPjwwu666MTxAtSu4YFWXQC_0J2QOkfkFfcezeuLbBAwbQB_pleSP1TljA4TeB1DdI--iCLYHTBtDOpgdzrC3yDHFlW3TehfifZ-PPMfMyIcinm8jZ95sVgPDWPbtguPFrJg8qXCYNRBFSKMe_plylblZuZGZlJ0UrarZ-j4LwVTNlHl92Ws3tgOmC_5ljMezv9pNx2d8jiQIJ0LX7P7L6ZyEZRx69HkHknlWfbKepiAY23FiU2ir1yd-3gdLi-tMdELBY9l7yayOo9XnEWXAXwaL9vGmD2tukbsM9tpN7LgquaqTRXFGRKPg2KkBu6O5n6_B81pngaS-8vBCYjvdXCUq7O5l6_8KgNMi2AwBxqPNtnNaIBIvQ7AZTVVe5sxE6qqxjXsrx_OGi8Vs579THKtne6rjZ3bTgTNjdOPdKxgBaTQh_yB6agIHj3nw9jFY4d45gTcnKRQeVHGQ9anUq46FamgnnLk476BBX1YFpo3ndibpqv2nkz6V4wQ9ZSkAgvL33L--wpzz7_mCQSW9SVj4ZcyV8eG5LXDpqJr_4NISOYvIhaOlFf6stUfvAWtnqG_RnBEElHPlGjiEC-KQPDx7v9nacBOGssFe4io56zZye4Gi2yLhDB3f9d0d8Ogs0NVC40p2yEN72kbHNv-4486U_bAkzwu78gU4Myx2nMgmOi46NKDx6iyx16VKuVlcPBlgpxpA_mfs-iPO7tey1zgejzH3YC-lfK3wfsQHsnVVs0KFnda9QMwmM34RhCzDTcf4YDDWxWnvh-IYdotTDzCEOsi-Nl9SY-gSfGkKA69Y22R42B2kRlpsalLZB-uj71W0C4YW1nixuRIsYZCjHZkBL9I0QtM9EuwkQhojbwu42UpwpNLdWWTbT_kfau5XB2bsueCZVYjbw4cd24kLnq5ZKgmVajM1Qv-dNduCcKiNEJ30sfGjTHKHr4dEGAaoAcloBock1HDUaX6PfTjPHM4c9IdTPgml2wBnOkN1XW7o9X6mnPDmLYldEPKzedl_9UYLYSFQLKaYQA9Mlf50fOBWZgi1RNk-OQxAbxuJlOUj671Wvh_a2ySBMbFYAeFIYJtkzkshqXt_qI-8kQqrEpJ9nbR59E6cQ2QkvloIarpkdAs_m6_9QQ0sw3PuY_8vAa2CYkCJdJOLHoX6mIHId8Czdtgp17xJDI8xRtnLZ2yxagJ8mU4jbtizllPI4hIMIi9x7Gtv-dMy_kMZflReTuiy-_Mya_Y3yJNKc8yBkr6yXUAx5drPKY5wTy_AFrAdaLWDHPc-ZLdRgx6Jp2VazSZQw-RMA4jt5Lb3lrrnorPI-CCQrzwqsP-fOzCZ3h8sXonoY7YCJ6bhL_XXmDomIC1IcBtfCkx70hqGyLxNT77T4qOo9APJeNmJttcJ8Sbi09R1kLIJdIr_03o4kUinCjwY5_utntioURY_ZRcki9wHEaZ4DUvPmc1R9i4cC8Yi0TsXEYuzTTMpCFRcPyHgM_ZgM-cAUrL9TeyIAfMBfJWoq7mUDqQtrslJQt7791HdGxrH-6WSpV8lq5bOUcaAIYH_Gwej4lO9cvAxI3IW-i6RLibrRfwcuDYoXuwccYxWUx9UD-NMMsBSX8zIneim3Zd7dHRLSgwgjAGod-GchNV-J4gqoxccSJqRFrSqOto9JGvgK2irfRi4PwSzU4LBVPLIywGhp2mPjAfGjCv7fm5AAE9vEPxeTIbw4B1F6vQT9VjdFk7ldW6ut8DmOmuJhnvQMPGKRg2avCZp6f-v6GYT186NiXjKzP6gIbn1qXIcfVETNyrFWeMTiLCVna73BxTlG-_FSTSEYq-lL9-mCv-lLCqCczYdH1cbrUif8POLDZluDEIyNtDoxl7fjh8HN6P5FVZVUx6z_qTGAaA18LYARZ_o_rUydnPyKdSzqSEcDvrQ7ZQ4rYo5e-2W5iUggwJ9WIap5o2k5gxbJ_QkRw3HBuxaXi84s3gs8HM5Xwne3NP48VMtc42qvRGd00hKHuJ8Yz70yr5llDJb62ZtF39mNMvuaeF1bMstGIHYVR4l9c939hZilFkYrJ58hr1AEdcBcEcZNXxWEoRBjiGjnYZBtEf6I4Ir_X54_Y9_pcdX5NctxHZbfQ87HXrKbmUm4KognMPyxfbcKn8x9R6AcJMPZU9YGhNaOLPpei1cB0_Or8lf6398lSdUKBaAQ7sKiJcaSQkjmU8LHsJTEdUjKuexTtVf85wKsSUrXJMAhZFz-JtzO2CAWjEVFUSN9DZ_bR7cwlNH5Wu37rYoSfECbJ1iTNUaYK3TmhXCFPrPYK_bGTG_gp47omErksH9q5n8wxbOQ_5N2g73PnE0ywdwUXZqZ4cjfrDwM9GT_F6mcLdih4yUED-LNavBSxOuctKJaCK4KyqtEg3MrYWQL1XmAnxC-kFtW0trNcroTx2jEi8kj0BZ4ehXpxHlSBnL-h11mBYaEoprhgQBSIXgBkt5of3XMJwlAdy4rD8jCfRJwHCuL5GQW5J7bUlIGILGj-jdOstGBsvaD2OGri3G4j5HLi-eZUeBr_114cFInDcqwezkjaTQa7b2L-IPEfeJ8Z_SW--yyfu1adhyLDIm0dhSEHirTk6l_gzdOphFKkPi5HqDD75GLjbOVTMc6N_fPefUg4jLEaptfcvf1UFh9wbUzVb_BRRXnyKy1NbEvfOhKCOv1GbA-eYowXcpRyubqDGL-si8ZTmMt8BUj2fj_lD7b2COjBpouLPlKukOWxz1HFXpCjJ9IHyiSWivf-LcxtQsY2zlXkBRpYJuAixC55xRWeJ4z_YDYLwJXO6OBzHml5MknOK9354P9QuLtkhpF429Vw6S8gKcDU9vv_daWYV_jyny3vzS-0XifocjJSt0Rll25bOT_vGcEMH-AbI2M8sH--pJ5DZqDadT2jsesRoVUtMRs66McV38t47n-hsNUgb1ys-UIq2LnhjldgArpcbNd3Kmn85Asrh4E0RA_kz2Q1QOgzaEqLF21xIjo43xLPYZ_bRDNPlrdVd3zg4vr7xs3UWk2i2R9PbHYJFDpVF0Bgs3zL0pmZXGr0caZeWJXhLPqAld2GjOMPxgkj0-S_QvcmWKF2aO78NgNSSMQOpc6zpE-qt8DWZ16Wscwtmg6uL1HvnSH4Mn_0hdYh1o7glhcVEbvC9TkFIoYcge_Gy4GuSfeJlmGoIE27qz264yZUmpEfaipFPHiPzZlS9tKOlDVyTcWmvhS2GHAmfjU_AV9Fdn_THKZ_ijbz73eOMQLmWRPdrqdYfkLNTmNK4F-pkWraGeQ88k-6GCHkgVirO4rT0C9sM6WqpJVSvikU7rB4nIME1SAp26o07nFzxP0hp3GzjawFoX_WgmKvvS2JUgHNnL8FjSZCcfHbHB-R9r66N89JNTP91uD82fIJg0sSKRScljpSmv_DF2Z5P_5YVAhBMwvrOtsjVkLrB-mPTMxKUYmIjBS-BkOMo5sSL76kJQ9xz2XrYYf2kGWLLEFJZmDU0OHLqCoNmTSQS-F1q9TZFsn2qMu4i8PA1n9I_C4aFsCeADroy6YiflwceYL6TzKN4JKICYFFpJE99CIfYa1NPIXraA1GzcXu9maV0oEmwYvaXAcQeGvR20C1uvvTb8uCJU1g7p0Yi51WTZiAcSu4vjrHcS-AI-6cE5aZbmQTFHhI8Od2-05eD4s7t73rPT3pr8_VhtH6YX-JkdsPeojZ8H4852lXraC1t0kOWX0gaBbLQSEVLwN_iQhxIqKTl4fQwSiRMzTypDbrW9C2QfxuGra4wyRaTBenq_x5_xe0rp5xGA_uXXZ0EnugtB-FwXagIQBhWfjbII_r8SDnfV8NKDoV0V5pTygOd_8j9HSKOrkCq0YREJMSNCnsDSUTe5WXmO-k-EenBRpbTk3Sh6ga_H_w29hoRkewO73Fcn1Wr9CjCza6qBbNkfNIcEGsf_Ha-QWkzyBrvPTGKTC0HYJj9RBTVW7A_KDTw7H2s1s5CBlrlKYJ9LCRZ_bbbxs1Dw_FyMBsy8yVoegLRLEy1UDCQUY-ZbXE8Gm4ycFxvOFkxLjHbCxF5pVRAhus24uYQIfz4hhEvHNKCSe0hSmxErEJ1IDEAQwyYkp88AKIsMxd7ucJIWjkOl_fSSTnMnxhQP_05OeDWksnMJhvHf1-_7olpWfWIy6YSOnj0wJV2jnRu7oqqJBIUX0oxWT2OFpH4nu-_7P9PzMmzVmp4TWH0fptrfhPeQSfB5u8GSWvcZyqQYqufosGr_59hG5T5V-J-Ld2-r4zocNHhhzHzRopcZjlyJvRjF2BwAv86wpDhMHOf19bziMwWsT1fKbaai2MjZYjuo_X1_ZWxYngYOsfk0J3d9tqqzI5zsxJnfr4C7Ryyd-SQwMNWbBfeGoywmBvk3EF4HKZ9ol9dNYF59q040IZRoP97j3ndAn8SfzVpHxxc5b3AQVfGzREktkqMc2Shj3driZQ1W-8kOO9j2spYSjuIm6NdTPtkht8wdj34nEeyvGN-UveXR5kGu29WvqgQl-DVnCAr4ryvJvs8ZASEuhe6eThsJkTVSALZPtLl3uLgFwHfZ6NTLsZR5-9pBo_2GmkASRuwvjfk72s1ax7NxOVDGgbdycM147aLU2qq-uRC-rko61G5QfFfUozwNjauwVl6oYB5BfZtvCCLdxbf2OghoP6eoXuuP_3qCNeqEvj2YVX9KitaY18lbU-_rn8hyDrWHoP6SaGHm20bEoPgKE0uuYu2qhZ7IRvRvq8mqOXIuxDCi_6L-4gH8eBxRtXSzAB48spAmKB9uERdXs0XKRXUyYUODvqIsjw==',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Considering rural and urban crossing safety**

I'm thinking about additional contexts for crossing streets, especially in rural areas without crosswalks. It's important to find straight stretches with good sight lines and cross perpendicularly. I should mention to look for gaps in traffic and be aware of hills or curves. For high-speed roads, use pedestrian overpasses if available. Also, I'll remind to check local driving directions, always looking left-right-left for safety. I might end with a brief, actionable list of tips for crossing at intersections.\
""",
                        id='rs_0d3e7d19e1d3306100697cc0a5b06081979f3164a57ea04a97',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Providing general safety guidelines**

I want to give general safety advice for crossing streets while also asking if the user needs specifics. I need to avoid coming off as patronizing, so I'll focus on being helpful. It's important to provide clear disclaimers about safety and aim for minimal formatting, maybe using bullet points. I'll include tips for kids to hold hands and look both ways. For people with mobility challenges, I should emphasize proper timing with crossing signals and watch for turning vehicles, especially in low visibility.\
""",
                        id='rs_0d3e7d19e1d3306100697cc0a5b06081979f3164a57ea04a97',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Detailing safe crossing procedures**

I want to ensure that the user knows to step out only when it's clear. I should highlight edge cases, like the "double threat" in multi-lane situations, where one car stops but others might not. It's important to peek past parked cars and listen for approaching vehicles. If using a stroller, keep it back until there's a safe gap. I need to emphasize clear viewing points, especially on rural roads or roundabouts, and also remind about the dangers of alcohol or drug impairment. I'll keep it concise while outlining core steps with optional variations.\
""",
                        id='rs_0d3e7d19e1d3306100697cc0a5b06081979f3164a57ea04a97',
                        provider_name='openai',
                    ),
                    ThinkingPart(
                        content="""\
**Creating safe crossing instructions**

I'm looking to create clear and concise instructions for the user about how to safely cross the street. First, I want to know if they're at a signalized intersection, a stop sign, or without a crosswalk. Then, I should present step-by-step guidance without heavy formatting, using bullet points when helpful.

I need to emphasize things like stopping at the edge, putting away phones, and making eye contact with drivers. It's important to discuss special situations, such as crossing near parked vehicles or in low visibility conditions. I'll encourage them to tell me about their specific crossing situation so I can tailor my guidance.\
""",
                        id='rs_0d3e7d19e1d3306100697cc0a5b06081979f3164a57ea04a97',
                        provider_name='openai',
                    ),
                    TextPart(
                        content="""\
Here's a simple, safe way to cross a street:

- Go to a crosswalk or intersection if you can. Avoid crossing mid‐block.
- Stop at the curb; put your phone away and remove earbuds.
- Press the pedestrian button if there is one.
- When the signal shows WALK/the walking person, or when there's no signal but traffic is clear:
  - Look for traffic in all lanes. Check the lane nearest you first.
    - In right‐hand traffic countries: look left → right → left.
    - In left‐hand traffic countries: look right → left → right.
  - Watch for turning cars, bikes, and e‐scooters. Make eye contact with drivers.
- Step off only when vehicles have stopped or there's a big gap. Keep scanning as you cross; walk straight, don't run.
- If the signal starts flashing "Don't Walk" while you're already in the crosswalk, continue to the other side. Don't start crossing on a solid "Don't Walk."
- On multi‐lane roads, be sure each lane is clear or stopped--don't rely on one car waving you through.
- If there's no crosswalk: choose a well‐lit spot with clear views (not near a curve, hill, or parked vehicles), wait for a large gap in both directions, then cross directly across.
- Extra tips:
  - At night or in rain, wear something bright/reflective and take extra time.
  - With kids or pets, hold hands/short leash and cross together.
  - Don't step out from in front of a bus or large vehicle that blocks your view.
  - At roundabouts, cross one direction at a time using the refuge island.

Tell me what kind of road and signals you see, and I can give step‐by‐step guidance for your exact situation.\
""",
                        id='msg_0d3e7d19e1d3306100697cc0c8c61c8197b88aa4803ed780f5',
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=45, output_tokens=2214, details={'reasoning_tokens': 1792}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
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
                    input_tokens=1053, output_tokens=2233, details={'thoughts_tokens': 1341, 'text_prompt_tokens': 1053}
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
                    input_tokens=34, output_tokens=1146, details={'thoughts_tokens': 759, 'text_prompt_tokens': 34}
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
**Defining the Goal**

I've homed in on the user's need: safe street crossing guidance. It's a straightforward task with clear safety parameters. My focus is now on extracting core safety principles for a successful outcome.


**Refining the Steps**

I'm now detailing each step, starting with finding a safe crossing point. I've broken down safe and unsafe locations, explaining the reasoning behind each. Next, I'm defining "edge" in the second step, clarifying the vantage point. The goal is to provide specific, actionable advice.


**Structuring the Guidelines**

I've organized the safety instructions into a logical, step-by-step format, and I'm adding crucial details to each point. I've covered safe crossing points, and I am now expanding the second step, clarifying the vantage point as the edge of the road, giving reasoning as to why it gives a clear advantage. I am working on the third step, which entails looking and listening for traffic. The mantra is "Look left, right, then left again", and I will add what to do about headphones.


**Developing the Structure**

I am finalizing the safety instructions with clear, concise language. I've incorporated the core safety principles into a step-by-step list, fleshing out each point with details like the importance of looking left, right, then left again. I'm focusing on ensuring clarity and addressing potential scenarios, and have added extra safety tips, such as wearing bright clothing.


"""
                ),
                next_part_kind='text',
            ),
            PartStartEvent(
                index=1,
                part=TextPart(
                    content="""\
Here is a step-by-step guide to crossing the street safely:

### 1. Find\
""",
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
            PartEndEvent(
                index=1,
                part=TextPart(
                    content="""\
Here is a step-by-step guide to crossing the street safely:

### 1. Find a Safe Place to Cross
Always use a designated crosswalk or a street corner with traffic signals. These are the safest places because drivers expect pedestrians to be there. Avoid crossing in the middle of the block or from between parked cars.

### 2. Stop at the Edge
Stand on the curb or the edge of the street. This gives you a clear view of the road without putting you in the path of traffic.

### 3. Look and Listen for Traffic
Follow this simple but crucial rule: **Look left, look right, and then look left again.**
*   **Look Left:** For traffic that will be in the lane closest to you.
*   **Look Right:** For traffic in the farther lanes.
*   **Look Left Again:** To make sure nothing has changed in the closest lane before you step off the curb.
*   **Listen:** Sometimes you can hear a vehicle before you can see it. It's a good idea to remove headphones.

### 4. Wait for a Safe Gap
If there are cars coming, wait for a large enough gap in traffic for you to cross safely. If you are at a crosswalk with a signal, press the button and wait for the "Walk" or walking person symbol to appear.

### 5. Make Eye Contact
If possible, make eye contact with the drivers of any stopped or approaching vehicles. This is the best way to be sure they have seen you.

### 6. Cross Quickly but Safely
Walk, do not run, across the street. Keep looking for traffic as you cross, as the situation can change quickly. Once you have started crossing, do not stop in the middle of the road.

By following these steps, you can greatly reduce the risk of an accident. Stay alert and stay safe\
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
                    input_tokens=16, output_tokens=1, details={'text_candidates_tokens': 1, 'text_prompt_tokens': 16}
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
                    input_tokens=22, output_tokens=5, details={'text_candidates_tokens': 5, 'text_prompt_tokens': 22}
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
    model = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(model=model)

    result = await agent.run('Hello!', model_settings={'timeout': 10})
    assert result.output == snapshot('Hello there! How can I help you today?')

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
                    input_tokens=25, output_tokens=5, details={'text_candidates_tokens': 5, 'text_prompt_tokens': 25}
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
                    input_tokens=39, output_tokens=8, details={'text_candidates_tokens': 8, 'text_prompt_tokens': 39}
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
    m = GoogleModel('gemini-2.5-pro', provider=google_provider)

    def upcase(text: str) -> str:
        return text.upper()

    agent = Agent(m, output_type=TextOutput(upcase))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run(
        'What is the largest city in the user country? Use the get_user_country tool and then your own world knowledge.'
    )
    assert result.output == snapshot(
        "I CAN'T PROVIDE THE LARGEST CITY WITHOUT KNOWING YOUR LOCATION. HOWEVER, IF I ASSUME YOU ARE IN MEXICO, THE LARGEST CITY IS MEXICO CITY."
    )

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
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args={},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={
                            'thought_signature': 'Cu8FAXLI2nwOxI+G9/9Z3jRnajoJapA7KjidIMHk8OKBiR76nML4Nn50WqYgxcStX00dDeu/1TARhkhemTHJibXzzq37701VyTTKcSO4dKEqOk0uCtG/eyIEU0tbNLYWqYTajhCAPgZFyRHWfclItjEf8J5gAIMrofx0Tg0dwF6YwmSCSe6m/K9LrAqwhXVzFfy1TUm5zBIOFpBlS5mu7DvdP/F4xBc53AlYJ5FRB+SG0UWT0n3ETnREgNyU9yE04dIdImVlVOuG/CN+jVCCxyTxj5PaoU3IFqW/rg1jNgHHE9MRVmSAgO31iCkejYDBL7ckvGg7H8MZ3dvKz2CJHCo5acM0XmOIBFTqaM0jFcR4P0I2v37maY1/MrjFVj3+Cdi7QN1ype+3xk1e7aQSec9THmxjRoPPvGY01SjPZB/SD5JECbR90NEdDrNy09Y1+eVhr5NaBOY0nobngZ7QtcBVtjPQnpXyeQ942ygYiL899MphdcODv1EVMKKTDU4a7rDPs+x69o1BlStafBWUfkzG2L3azJSXDYX9rk3tIV0vJSVQZfQr/e9uGXGHw7pTudyHCK96bSjp6J/RppdkZB/TEPyemUXhbEdtPnqo36i2UatCmfQoU31PSRqXkdEig8drnmwuceAV4Fxadnkt6OwZ5MonjJN+4CK9S0r/rMtHkqv0RdF1HhR/MnvtZP9TGP5Pu9IIR8ToYF5EVbeAIOkiP1HjMJTbXIM89drkWxZGyTKw1JnBGKcA69xj55fUPNfse3FQuto5JlmX2cO5ax/Cwza1/qVR0sSM/2B87RmjZSpbF5ihRDikkRkEM4DkaJCakpJQ45+yKJoq9zHR1eza0Z89Vfbwgp24YZWKLucLtBt8ENeHlsl4Sthjhks3Zf4oufDFtGrpQ6v+3GyZFAEQFKevdlfGs7LlMlx1Rn/Mkwww8Osa4XNjlo4FXLt6dmmRzMN8/a04sKGaYEsPN9VFg00FmdN74Usa4zVyBZ5DYA=='
                        },
                    )
                ],
                usage=RequestUsage(
                    input_tokens=49, output_tokens=181, details={'thoughts_tokens': 169, 'text_prompt_tokens': 49}
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
                    TextPart(
                        content="I can't provide the largest city without knowing your location. However, if I assume you are in Mexico, the largest city is Mexico City."
                    )
                ],
                usage=RequestUsage(input_tokens=80, output_tokens=30, details={'text_prompt_tokens': 80}),
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
                    input_tokens=9, output_tokens=46, details={'text_candidates_tokens': 46, 'text_prompt_tokens': 9}
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
                    input_tokens=83, output_tokens=13, details={'text_candidates_tokens': 13, 'text_prompt_tokens': 83}
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
    m = GoogleModel('gemini-2.5-pro', provider=google_provider)

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
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args={},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={
                            'thought_signature': 'Co4UAXLI2nzudxgaUXrUC1zevWlZ4jkiOkS8QVwNRwOY3mVz6GrGtg7b3uLuTqCH95cyoN4jE3HECl/mJbjLU92UQVsq7kZ2Xez4FUc3ClRH3u03AZgVyVVUN43ktJqkLqIsX8r3nEbm0jfC+z5Xw5GTbIpyUUHaM0BL391nsK0Y0ZSPB9G7QHzDHN/Nq7+K/QUqtLSxVqLkDxQ5pU8A8FiDuQCiXE6qmqnh9XMCte572qZmJazCnHFW42IQhvOfSUMH9wEabsimgwswiScg1pDgSBhIPMn/SbUPwLlS5BMhYVEsBrgzCI6wwcES5a8CZCsk1ZwjlrMjzOOannQ3D35powwuaTZsNPD2MbfaQv7KR1KGNQc4QPWMf+D/b/ZTYqwfMwBfWVe9SZPe5OD/IvCJpfEuMbupVhrpraY9+J1uaxJ9U9eDMiiZ3WVjwIOJGHxJ1nBibd0dqRcf6ZbHGUaT4ej4mzJwVDcys4/J5Ol54qHA12R3rYGykUdSKEA95S9lDYvJCrHSavIzUK0BjD8C5mNQlnllUFNYKII0j9D+Wt0sYDRegedtC2+SNPBvyRxbR7lSdJGmZUaq7S2k8fEqVh1bujrRG6dkDFlrgOs0NH/vZA9rFW0OvSM1Z68rMYkAhzaxhWz8YdYUYqc02V8o4311huwDNt4EjBTZ9nINLaObZIz6jbZN2l0MUA0uuunGkydRzUj5X1nmrb1mEGEOOYqtYugEZtRLerqj3XWiKmpWtBeEDX/hoVHQ7etKohKVGXPjFRmM5lzFCGH380SVwcvvKmVcMDkzaqtvSgRDMfg0+WuxFdxJ7Sz2ixwxoeAWKtK/TahvYrWJBMBmGnOfT7052jSXuY1r4wpC0CcupLS64P98TRtHr04IzuYpcyKRtOX/GjuABmpd2t4Ha9cFb6flPkp484sHd8xaT5uMJY0w/IoxDVDYafRUXBeeAIqRQb7sWk/5nifY6dfNLJf7jHwE5xD/sTx2XxnB5oBU06ERjPeUCMfLdZYWkbx1ooeLZsYTnwwMKqMi56wQtXIWpGlG8au6OjScEO3skN1RT2WnJMvfYohzQQ+SPISNRdXRb59T7s/+OB2PBHgW2xWYInLHrk0thFiOEJ2N8Pl9hhOzcj1+gaIitAMOaXrF5Ix9bspENdYdh+tnHYg/MQ3S2x4LG4qRJ3cQLXR4nn0nG54RcDmC30JhbPWaL0H72gbLrVGnMIRd1U2VT0l2e6fTrA+FOXEjwV5/BoRnA2UrEzw8DlJubJFyjfpz+I3tAPnWdIQ7u2md6fqnTV8lnpFGedqngmQ2My5/r4bKh/L8/RbmNNVixIb6GLtMNycOS/C0XyUUsJ/liMyicgGzt8AbTam/ytzZmInKFnUdel5WmCF3tCGCbB7WWFA6E2eT5pvMGzcgyKHmSBLGqTdAyvch+8FP0FFD1XrfUpbL3FQFU31wDsglCITbeiWh/+qXNIQEgaVHquD2YdLc4ngJWpL5+Vl+OcFjBhrv1QpgjcfOcc/9Ku7LFB/b4LdkF7bWlSFPasIBGeQfwPM+QJsPlmXUn2mv/bt0i3iA+adWulIdVTzaO+dm9vfJEDNyzHXblUNe4dLD2Ani78Ek5krcmVnw7fFPigh4qszWtHaia69FsLaIqfyA0J/2adbYPATztPuZhA+hlTPejlGkAUQB8mTt1uFpFodQ9+GxxORBfnoRzMrBLu/oL7vxE+UwUALVNEAul6x+EjHSCMDlAMI2rvUgSWq+yQOIQcrD/L+Uy247U+TUyr5huabz/RiAk/QWbviAbAHGE4lgG3ZEJlXvHtcENqZEbuqu0XbTWEuy+7H40Uxf4/1HKt8g+6nRowLxA0Ioyy0uNJXzY9xcE3uQulVC2TDmPZyg8DKaJcPvVZx1Qd0M+bpaKBZYOjDJ6N1wlHUrsdqNQZ+Vem0BvxzkzZFCbG8jS06hnOpBTdJrkFxFp4jhyJIbMX8Q9Y3z/ci225F6es5RtG9Ifr8gpNUgLy2Lc/o/uFO6lKcscJJ7L5kz6jijiEWedWTULmCzYKDh3xGlwUSN9lmIJfry0PqZUKcIVJGfFZ2pU5wifqzpWJaPZOXPfu0OgbL2eZu7jv+MbR/bP3vmSt6eWO57bxuMxwrT07lbydw+4nb2mzLzzhnxSad7kQ771IY4rpDcLXB/nOtdfjaQmT/Nzmw43ofwndPuqgs5banJgy2G/y7dOcd16Mr1K7w5wc5ypEW+sTqFTSUEy1gSUauq5ycji45l52RNAcYgqBzmAVvAHxy1MJlF6z02NnmVclgf654lwy130L+IducvuDGs0t/S3Iwi5ZAGxXzkXm0r1N30ALsstRuAhMDor40smkT/W7dzTkiEA1+ftSLU8E7bcck2sca4pLSiMnl9YIao+uhRQi3yYrcrQeKRDDNNwGy1eV5gV3M6eWu+Oz+Giaxc3brFESm9qq5NBZIYIIseFV9EV/BhsoUxVQ/ASAfBV4BqIAzHrhpvHSjExVGd/kkhdd3Gltg1s3oWm5Y++BXRFeU/wlGuBklnvHtzLMN4KUOoC5N+PMab6SsjA6nTMJh5Sc+v9q3RSLW1s9V3pOFCiuQGGPJi1rNYFXIfn7363sqnMOG1DdhYDUqSQoHoAmwOzE6xEN76PAATl8G7I+4EvBx30R/W/thI8zgrV2wRzG0x/MPqUGDQpVvFC2asr6+ONMGN0RQXMtoja66Avg7O4drouNd+weR3tU8Ht0LaDGs5wyYS11lKEosoOAOGs5X49pY+Ifgv5WHHb8j2uFgQb/TzvLL2xi+HfNlcabMLN3/hNGSzfN7RdPHKl7a3cK8AxrKFEOofbm0wWO/mcNt8vVr0QwhbLSfkV3koDI+LkeWIggps+RbsR3OKaljnkSPQWfYTgLiG8XtjO/kHcUUrwbYCOYN1lFDbcwKdlBR/PCQDsreGX8SZUdndLkRG54ArJfv6a5HbJE9Rhb/lYFrh83gelHmMeg92KhBuPtMbesqF2izeW9b+kA2oxUkh9VfVEQ0cpFTDQvsjQzSZ5LKzJSdPz/JquPpSS8ylXOjJnEHRN3X/NiX3hGdDpEI93b9gCDeWCLyMdQj9z/kCJHUa+XnJGzR0EEjHTy7TSw7CXHCrr2RXCvDYT3vUIcTH9E8O6JYg+D3kyalN5Z7RPGF4Ls+4gZ4o7n2JNrxQ9vvlW0XUAU48CLS3Y0l4mpEebc9zjLVEj9N7UEEm5rHcND7ZurBQLWQLyW4NPsvtG5C/ESFp3GD/AwomTKnFVCRsbSybm6r2HtFe2qxZ1cvmHycwye2eTYQuaPy3IeFbswQSW2/KSOX0BXavTYCqk/DAXqCCyblQPc2l350e/mgmCJ9aU+NPZeUfPzIyq1QhtyubpWU6j2FX0WX0qfPzIXyJbRbR1fUXewXucPUsl2/ycfvuXuPQrkMr4luI'
                        },
                    )
                ],
                usage=RequestUsage(
                    input_tokens=125, output_tokens=658, details={'thoughts_tokens': 646, 'text_prompt_tokens': 125}
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
                    TextPart(
                        content="""\
```json
{
  "city": "Mexico City",
  "country": "Mexico"
}
```\
""",
                        provider_name='google-gla',
                        provider_details={
                            'thought_signature': 'CuAFAXLI2nzl76CfdYfKfaLJTtS0Am7Y1rPsjgFrlVsQpVL6QAci8+TGiYL4KFiowiVketMX1UQukxIszlkC+pRtihhSMiFpkbeOfMdmZOtzgqgAliWS5522OE9aOHiS/eGjLI6Utn4qfe+Y6FEBR5AIL2JQqb1wYv9ef9Qjl92dKdDaCjUsxizigT1r/SnPZPK5IXTj8nc8zDq5n5P1wOoos7v3FG4r5R7wqaDmDumzH4mo5kVp7VmVbhhUFgcvvA/F7DQR50rnzpXTqKkCyV06EhiN68M/eVvUoKNmUAZWO3/bs69AcLiQ3FmhRoJL423hsQQFgKkncJ+7ytTHYHkTWToHnlP9jcxbJIAReq4CNz5p7je+BBFeLYWZJ4xHjlRd+JKr1fOf0k74oh6o1PeotDyGrpPsmwqzTySN/JKmPXBzRt8jtHuwwrxXdV9bXIqLqgr5/r3WX6I86mmNqcGk0E9NUuMCnHTJ5ydrX4ZeJ2btkHxq6uJ4m6jpZ18fenhAd6MnHsJK1QrUhJ11tXAIYUQDtMYOMsswVp236gj2LWoz8w1smlnIhuDZlSBkenEPpT8styL1lafkwsvYWdcygoax0FYZTR6RGLBorrwHujIolWeF1DQsFWKt7GIv+Uwj6krP/FoqC8Bgv3M1eYZXKs/51JtQVTPt5iwAuQ6R+iAv6aU2wvPYLZqTyJ5e/kIAO9pBKK9yacmHpLhAxSONSTm/Hbvc9ei7MEZlFwByMdYc5HOQqzqWCSAAFfPP1ZIY7XURHBkLcOHYznouCsrDB1wPOBTG5WEA2jBvsgTPMJdizZr6oBAjk+Up2NaEV85qZpGTGyJsPC1wtl2QaibbsoEUMpqPnKzlMMsh+gvq3+paj19EI78L3HZLdrBJ3ncje7TnGWciUYrTmVlLksXmAlQJC1WUjTbNYC6XrqiD4hlvwLpbYmpyPi35Wcdrxb3w+6rsd1Z1BimA6o+CbGlpSA=='
                        },
                    )
                ],
                usage=RequestUsage(
                    input_tokens=156, output_tokens=203, details={'thoughts_tokens': 178, 'text_prompt_tokens': 156}
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
                    input_tokens=230,
                    output_tokens=27,
                    details={'text_candidates_tokens': 27, 'text_prompt_tokens': 230},
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
That's a classic! "The quick brown fox jumps over the lazy dog" is one of the most famous **pangrams**.

A pangram is a sentence that contains every letter of the alphabet at least once. It's widely used for:
*   **Typing practice:** To ensure you hit every key.
*   **Displaying typefaces (fonts):** To show how all the letters look in a particular font.
*   **Testing equipment:** Like typewriters or keyboards.

It's quite efficient for that purpose, using only 35 letters (if "lazy dog" is treated as two words, which is more common). You even combined "lazy dog" into "lazydog" which is often done to make it even shorter or for specific software testing!\
""")


async def test_google_vertexai_model_usage_limit_exceeded(
    allow_model_requests: None, vertex_provider: GoogleProvider
):  # pragma: lax no cover
    model = GoogleModel('gemini-3-flash-preview', provider=vertex_provider, settings=ModelSettings(max_tokens=100))

    agent = Agent(model, instructions='You are a chatbot.')

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'  # pragma: no cover

    with pytest.raises(UsageLimitExceeded, match='The next request would exceed the total_tokens_limit of 9'):
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
                    TextPart(content="Sure! Here's an image of an axolotl for you. "),
                    FilePart(content=IsInstance(BinaryImage)),
                ],
                usage=RequestUsage(
                    input_tokens=10,
                    output_tokens=1306,
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
            PartStartEvent(index=0, part=TextPart(content="Sure! Here'")),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='s an image of')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' an axolotl')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' for')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' you. ')),
            PartEndEvent(
                index=0, part=TextPart(content="Sure! Here's an image of an axolotl for you. "), next_part_kind='file'
            ),
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
    agent = Agent(m, output_type=str | BinaryImage)

    result = await agent.run('Tell me a two-sentence story about an axolotl, no image please.')
    assert result.output == snapshot(
        'In a hidden cenote, a shy axolotl named Pipkin loved to collect shiny pebbles. One day, he found a pearl so luminous, it lit up his entire underwater world.'
    )

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(IsInstance(BinaryImage))


async def test_google_image_and_text_output(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-2.5-flash-image', provider=google_provider)
    agent = Agent(m)

    result = await agent.run('Tell me a two-sentence story about an axolotl with an illustration.')
    assert result.output == snapshot(
        'Once, in a hidden cenote, lived a shy axolotl named Pip who dreamed of seeing the surface world. One day, a curious diver gently scooped him up, allowing Pip to briefly witness the sun-dappled world above before returning him to his serene, underwater home. '
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


async def test_google_vertexai_image_generation(
    allow_model_requests: None, vertex_provider: GoogleProvider
):  # pragma: lax no cover
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
    model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test-key'))

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
    model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test-key'))
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
    model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test-key'))
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
    model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test-key'))

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
    model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test-key'))
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
    assert result.output == snapshot('Mexico City.')
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
                usage=RequestUsage(input_tokens=37, output_tokens=168, details={'reasoning_tokens': 128}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
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
                    TextPart(
                        content='Mexico City.',
                        id=IsStr(),
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(input_tokens=217, output_tokens=82, details={'reasoning_tokens': 64}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': IsDatetime(),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    model = GoogleModel('gemini-3-pro-preview', provider=google_provider)

    usage_limits = UsageLimits(request_limit=10)
    result = await agent.run(
        model=model, message_history=messages[:-1], output_type=CityLocation, usage_limits=usage_limits
    )
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
                    input_tokens=107, output_tokens=96, details={'thoughts_tokens': 73, 'text_prompt_tokens': 107}
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
    model = GoogleModel('gemini-2.5-flash', provider=google_provider)
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
    model = GoogleModel('gemini-2.5-flash', provider=google_provider)
    mocked_error = errors.APIError(302, {'error': {'code': 302, 'message': 'Redirect', 'status': 'REDIRECT'}})
    mocker.patch.object(model.client.aio.models, 'generate_content', side_effect=mocked_error)

    agent = Agent(model=model)

    with pytest.raises(ModelAPIError) as exc_info:
        await agent.run('This prompt will trigger the mocked error.')

    assert exc_info.value.model_name == 'gemini-2.5-flash'


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
                    input_tokens=29, output_tokens=120, details={'thoughts_tokens': 110, 'text_prompt_tokens': 29}
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
                usage=RequestUsage(input_tokens=165, output_tokens=8, details={'text_prompt_tokens': 165}),
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


async def test_google_prepends_empty_user_turn_when_first_content_is_model(google_provider: GoogleProvider):
    """Test that an empty user turn is prepended when contents start with a model response.

    This happens when there's a conversation history with a model response (containing tool calls)
    followed by tool results, but no initial user prompt. The Gemini API requires that function
    call turns come immediately after a user turn or function response turn.

    See https://github.com/pydantic/pydantic-ai/issues/3692
    """
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)

    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='generate_topic', args={}, tool_call_id='test_id'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='generate_topic', content='penguins', tool_call_id='test_id'),
            ]
        ),
    ]

    _, contents = await m._map_messages(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]

    assert contents == snapshot(
        [
            {'role': 'user', 'parts': [{'text': ''}]},
            {
                'role': 'model',
                'parts': [
                    {
                        'function_call': {'name': 'generate_topic', 'args': {}, 'id': 'test_id'},
                        'thought_signature': b'skip_thought_signature_validator',
                    }
                ],
            },
            {
                'role': 'user',
                'parts': [
                    {
                        'function_response': {
                            'name': 'generate_topic',
                            'response': {'return_value': 'penguins'},
                            'id': 'test_id',
                        }
                    },
                ],
            },
        ]
    )
