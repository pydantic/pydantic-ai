"""Tests for the MCP (Model Context Protocol) server implementation."""

from __future__ import annotations

import base64
import re
from datetime import timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.mcp import MCPServerStreamableHTTP, load_mcp_servers
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage

from .conftest import IsDatetime, IsNow, IsStr, try_import

with try_import() as imports_successful:
    from mcp import ErrorData, McpError, SamplingMessage
    from mcp.client.session import ClientSession
    from mcp.shared.context import RequestContext
    from mcp.types import CreateMessageRequestParams, ElicitRequestParams, ElicitResult, ImageContent, TextContent

    from pydantic_ai._mcp import map_from_mcp_params, map_from_model_response
    from pydantic_ai.mcp import CallToolFunc, MCPServerSSE, MCPServerStdio, ToolResult
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mcp and openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.fixture
def mcp_server() -> MCPServerStdio:
    return MCPServerStdio('python', ['-m', 'tests.mcp_server'])


@pytest.fixture
def model(openai_api_key: str) -> Model:
    return OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))


@pytest.fixture
def agent(model: Model, mcp_server: MCPServerStdio) -> Agent:
    return Agent(model, toolsets=[mcp_server])


@pytest.fixture
def run_context(model: Model) -> RunContext[int]:
    return RunContext(deps=0, model=model, usage=RunUsage())


async def test_stdio_server(run_context: RunContext[int]):
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    async with server:
        tools = [tool.tool_def for tool in (await server.get_tools(run_context)).values()]
        assert len(tools) == snapshot(18)
        assert tools[0].name == 'celsius_to_fahrenheit'
        assert isinstance(tools[0].description, str)
        assert tools[0].description.startswith('Convert Celsius to Fahrenheit.')

        # Test calling the temperature conversion tool
        result = await server.direct_call_tool('celsius_to_fahrenheit', {'celsius': 0})
        assert result == snapshot(32.0)


async def test_reentrant_context_manager():
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    async with server:
        async with server:
            pass


async def test_context_manager_initialization_error() -> None:
    """Test if streams are closed if client fails to initialize."""
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    from mcp.client.session import ClientSession

    with patch.object(ClientSession, 'initialize', side_effect=Exception):
        with pytest.raises(Exception):
            async with server:
                pass

    assert server._read_stream._closed  # pyright: ignore[reportPrivateUsage]
    assert server._write_stream._closed  # pyright: ignore[reportPrivateUsage]


async def test_aexit_called_more_times_than_aenter():
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'])

    with pytest.raises(ValueError, match='MCPServer.__aexit__ called more times than __aenter__'):
        await server.__aexit__(None, None, None)

    async with server:
        pass  # This will call __aenter__ and __aexit__ once each

    with pytest.raises(ValueError, match='MCPServer.__aexit__ called more times than __aenter__'):
        await server.__aexit__(None, None, None)


async def test_stdio_server_with_tool_prefix(run_context: RunContext[int]):
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'], tool_prefix='foo')
    async with server:
        tools = await server.get_tools(run_context)
        assert all(name.startswith('foo_') for name in tools.keys())

        result = await server.call_tool(
            'foo_celsius_to_fahrenheit', {'celsius': 0}, run_context, tools['foo_celsius_to_fahrenheit']
        )
        assert result == snapshot(32.0)


async def test_stdio_server_with_cwd(run_context: RunContext[int]):
    test_dir = Path(__file__).parent
    server = MCPServerStdio('python', ['mcp_server.py'], cwd=test_dir)
    async with server:
        tools = await server.get_tools(run_context)
        assert len(tools) == snapshot(18)


async def test_process_tool_call(run_context: RunContext[int]) -> int:
    called: bool = False

    async def process_tool_call(
        ctx: RunContext[int],
        call_tool: CallToolFunc,
        name: str,
        tool_args: dict[str, Any],
    ) -> ToolResult:
        """A process_tool_call that sets a flag and sends deps as metadata."""
        nonlocal called
        called = True
        return await call_tool(name, tool_args, {'deps': ctx.deps})

    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'], process_tool_call=process_tool_call)
    async with server:
        agent = Agent(deps_type=int, model=TestModel(call_tools=['echo_deps']), toolsets=[server])
        result = await agent.run('Echo with deps set to 42', deps=42)
        assert result.output == snapshot('{"echo_deps":{"echo":"This is an echo message","deps":42}}')
        assert called, 'process_tool_call should have been called'


def test_sse_server():
    sse_server = MCPServerSSE(url='http://localhost:8000/sse')
    assert sse_server.url == 'http://localhost:8000/sse'
    assert sse_server.log_level is None


def test_sse_server_with_header_and_timeout():
    with pytest.warns(DeprecationWarning, match="'sse_read_timeout' is deprecated, use 'read_timeout' instead."):
        sse_server = MCPServerSSE(
            url='http://localhost:8000/sse',
            headers={'my-custom-header': 'my-header-value'},
            timeout=10,
            sse_read_timeout=100,
            log_level='info',
        )
    assert sse_server.url == 'http://localhost:8000/sse'
    assert sse_server.headers is not None and sse_server.headers['my-custom-header'] == 'my-header-value'
    assert sse_server.timeout == 10
    assert sse_server.read_timeout == 100
    assert sse_server.log_level == 'info'


def test_sse_server_conflicting_timeout_params():
    with pytest.raises(TypeError, match="'read_timeout' and 'sse_read_timeout' cannot be set at the same time."):
        MCPServerSSE(
            url='http://localhost:8000/sse',
            read_timeout=50,
            sse_read_timeout=100,
        )


async def test_agent_with_stdio_server(allow_model_requests: None, agent: Agent):
    async with agent:
        result = await agent.run('What is 0 degrees Celsius in Fahrenheit?')
        assert result.output == snapshot('0 degrees Celsius is equal to 32 degrees Fahrenheit.')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is 0 degrees Celsius in Fahrenheit?',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='celsius_to_fahrenheit',
                            args='{"celsius":0}',
                            tool_call_id='call_QssdxTGkPblTYHmyVES1tKBj',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=195,
                        output_tokens=19,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-BRlnvvqIPFofAtKqtQKMWZkgXhzlT',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='celsius_to_fahrenheit',
                            content=32.0,
                            tool_call_id='call_QssdxTGkPblTYHmyVES1tKBj',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='0 degrees Celsius is equal to 32 degrees Fahrenheit.')],
                    usage=RequestUsage(
                        input_tokens=227,
                        output_tokens=13,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-BRlnyjUo5wlyqvdNdM5I8vIWjo1qF',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_agent_with_conflict_tool_name(agent: Agent):
    @agent.tool_plain
    def get_none() -> None:  # pragma: no cover
        """Return nothing"""
        return None

    async with agent:
        with pytest.raises(
            UserError,
            match=re.escape(
                "MCPServerStdio(command='python', args=['-m', 'tests.mcp_server']) defines a tool whose name conflicts with existing tool from the agent: 'get_none'. Set the `tool_prefix` attribute to avoid name conflicts."
            ),
        ):
            await agent.run('Get me a conflict')


async def test_agent_with_prefix_tool_name(openai_api_key: str):
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'], tool_prefix='foo')
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(
        model,
        toolsets=[server],
    )

    @agent.tool_plain
    def get_none() -> None:  # pragma: no cover
        """Return nothing"""
        return None

    async with agent:
        # This means that we passed the _prepare_request_parameters check and there is no conflict in the tool name
        with pytest.raises(RuntimeError, match='Model requests are not allowed, since ALLOW_MODEL_REQUESTS is False'):
            await agent.run('No conflict')


async def test_agent_with_server_not_running(agent: Agent, allow_model_requests: None):
    result = await agent.run('What is 0 degrees Celsius in Fahrenheit?')
    assert result.output == snapshot('0 degrees Celsius is 32.0 degrees Fahrenheit.')


async def test_log_level_unset(run_context: RunContext[int]):
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    assert server.log_level is None
    async with server:
        result = await server.direct_call_tool('get_log_level', {})
        assert result == snapshot('unset')


async def test_log_level_set(run_context: RunContext[int]):
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'], log_level='info')
    assert server.log_level == 'info'
    async with server:
        result = await server.direct_call_tool('get_log_level', {})
        assert result == snapshot('info')


async def test_tool_returning_str(allow_model_requests: None, agent: Agent):
    async with agent:
        result = await agent.run('What is the weather in Mexico City?')
        assert result.output == snapshot(
            'The weather in Mexico City is currently sunny with a temperature of 26 degrees Celsius.'
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the weather in Mexico City?',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_weather_forecast',
                            args='{"location":"Mexico City"}',
                            tool_call_id='call_m9goNwaHBbU926w47V7RtWPt',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=194,
                        output_tokens=18,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-BRlo3e1Ud2lnvkddMilmwC7LAemiy',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_weather_forecast',
                            content='The weather in Mexico City is sunny and 26 degrees Celsius.',
                            tool_call_id='call_m9goNwaHBbU926w47V7RtWPt',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The weather in Mexico City is currently sunny with a temperature of 26 degrees Celsius.'
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=234,
                        output_tokens=19,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-BRlo41LxqBYgGKWgGrQn67fQacOLp',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_returning_text_resource(allow_model_requests: None, agent: Agent):
    async with agent:
        result = await agent.run('Get me the product name')
        assert result.output == snapshot('The product name is "Pydantic AI".')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me the product name',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_product_name',
                            args='{}',
                            tool_call_id='call_LaiWltzI39sdquflqeuF0EyE',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=200,
                        output_tokens=12,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-BRmhyweJVYonarb7s9ckIMSHf2vHo',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_product_name',
                            content='Pydantic AI',
                            tool_call_id='call_LaiWltzI39sdquflqeuF0EyE',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='The product name is "Pydantic AI".')],
                    usage=RequestUsage(
                        input_tokens=224,
                        output_tokens=12,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-BRmhzqXFObpYwSzREMpJvX9kbDikR',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_returning_text_resource_link(allow_model_requests: None, agent: Agent):
    async with agent:
        result = await agent.run('Get me the product name via get_product_name_link')
        assert result.output == snapshot('The product name is "Pydantic AI".')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me the product name via get_product_name_link',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_product_name_link',
                            args='{}',
                            tool_call_id='call_qi5GtBeIEyT7Y3yJvVFIi062',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=305,
                        output_tokens=12,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-BwdHSFe0EykAOpf0LWZzsWAodIQzb',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_product_name_link',
                            content='Pydantic AI\n',
                            tool_call_id='call_qi5GtBeIEyT7Y3yJvVFIi062',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='The product name is "Pydantic AI".')],
                    usage=RequestUsage(
                        input_tokens=332,
                        output_tokens=11,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-BwdHTIlBZWzXJPBR8VTOdC4O57ZQA',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_returning_image_resource(allow_model_requests: None, agent: Agent, image_content: BinaryContent):
    async with agent:
        result = await agent.run('Get me the image resource')
        assert result.output == snapshot(
            'This is an image of a sliced kiwi with a vibrant green interior and black seeds.'
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me the image resource',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_image_resource',
                            args='{}',
                            tool_call_id='call_nFsDHYDZigO0rOHqmChZ3pmt',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=191,
                        output_tokens=12,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-BRlo7KYJVXuNZ5lLLdYcKZDsX2CHb',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_image_resource',
                            content='See file 1c8566',
                            tool_call_id='call_nFsDHYDZigO0rOHqmChZ3pmt',
                            timestamp=IsDatetime(),
                        ),
                        UserPromptPart(content=['This is file 1c8566:', image_content], timestamp=IsDatetime()),
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='This is an image of a sliced kiwi with a vibrant green interior and black seeds.'
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=1332,
                        output_tokens=19,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-BRloBGHh27w3fQKwxq4fX2cPuZJa9',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_returning_image_resource_link(
    allow_model_requests: None, agent: Agent, image_content: BinaryContent
):
    async with agent:
        result = await agent.run('Get me the image resource via get_image_resource_link')
        assert result.output == snapshot(
            'This is an image of a sliced kiwi fruit. It shows the green, seed-speckled interior with fuzzy brown skin around the edges.'
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me the image resource via get_image_resource_link',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_image_resource_link',
                            args='{}',
                            tool_call_id='call_eVFgn54V9Nuh8Y4zvuzkYjUp',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=305,
                        output_tokens=12,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-BwdHygYePH1mZgHo2Xxzib0Y7sId7',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_image_resource_link',
                            content='See file 1c8566',
                            tool_call_id='call_eVFgn54V9Nuh8Y4zvuzkYjUp',
                            timestamp=IsDatetime(),
                        ),
                        UserPromptPart(content=['This is file 1c8566:', image_content], timestamp=IsDatetime()),
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='This is an image of a sliced kiwi fruit. It shows the green, seed-speckled interior with fuzzy brown skin around the edges.'
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=1452,
                        output_tokens=29,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-BwdI2D2r9dvqq3pbsA0qgwKDEdTtD',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_returning_audio_resource(
    allow_model_requests: None, agent: Agent, audio_content: BinaryContent, gemini_api_key: str
):
    model = GoogleModel('gemini-2.5-pro-preview-03-25', provider=GoogleProvider(api_key=gemini_api_key))
    async with agent:
        result = await agent.run("What's the content of the audio resource?", model=model)
        assert result.output == snapshot('The audio resource contains a voice saying "Hello, my name is Marcelo."')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="What's the content of the audio resource?", timestamp=IsDatetime())],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_audio_resource', args={}, tool_call_id=IsStr())],
                    usage=RequestUsage(
                        input_tokens=383, output_tokens=137, details={'thoughts_tokens': 125, 'text_prompt_tokens': 383}
                    ),
                    model_name='models/gemini-2.5-pro-preview-05-06',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_audio_resource',
                            content='See file 2d36ae',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        ),
                        UserPromptPart(content=['This is file 2d36ae:', audio_content], timestamp=IsDatetime()),
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='The audio resource contains a voice saying "Hello, my name is Marcelo."')],
                    usage=RequestUsage(
                        input_tokens=575,
                        output_tokens=15,
                        input_audio_tokens=144,
                        details={'text_prompt_tokens': 431, 'audio_prompt_tokens': 144},
                    ),
                    model_name='models/gemini-2.5-pro-preview-05-06',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_returning_audio_resource_link(
    allow_model_requests: None, agent: Agent, audio_content: BinaryContent, gemini_api_key: str
):
    model = GoogleModel('gemini-2.5-pro', provider=GoogleProvider(api_key=gemini_api_key))
    async with agent:
        result = await agent.run("What's the content of the audio resource via get_audio_resource_link?", model=model)
        assert result.output == snapshot('00:05')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="What's the content of the audio resource via get_audio_resource_link?",
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            signature=IsStr(),
                            provider_name='google-gla',
                        ),
                        ToolCallPart(
                            tool_name='get_audio_resource_link',
                            args={},
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=605, output_tokens=168, details={'thoughts_tokens': 154, 'text_prompt_tokens': 605}
                    ),
                    model_name='gemini-2.5-pro',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='Pe_BaJGqOKSdz7IP0NqogA8',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_audio_resource_link',
                            content='See file 2d36ae',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        ),
                        UserPromptPart(
                            content=[
                                'This is file 2d36ae:',
                                audio_content,
                            ],
                            timestamp=IsDatetime(),
                        ),
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='00:05')],
                    usage=RequestUsage(
                        input_tokens=801,
                        output_tokens=5,
                        input_audio_tokens=144,
                        details={'text_prompt_tokens': 657, 'audio_prompt_tokens': 144},
                    ),
                    model_name='gemini-2.5-pro',
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                    provider_details={'finish_reason': 'STOP'},
                    provider_response_id='QO_BaLC6AozQz7IPh5Kj4Q4',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_returning_image(allow_model_requests: None, agent: Agent, image_content: BinaryContent):
    async with agent:
        result = await agent.run('Get me an image')
        assert result.output == snapshot('Here is an image of a sliced kiwi on a white background.')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me an image',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_image',
                            args='{}',
                            tool_call_id='call_Q7xG8CCG0dyevVfUS0ubsDdN',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=190,
                        output_tokens=11,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-BRloGQJWIX0Qk7gtNzF4s2Fez0O29',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_image',
                            content='See file 1c8566',
                            tool_call_id='call_Q7xG8CCG0dyevVfUS0ubsDdN',
                            timestamp=IsDatetime(),
                        ),
                        UserPromptPart(
                            content=[
                                'This is file 1c8566:',
                                image_content,
                            ],
                            timestamp=IsDatetime(),
                        ),
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Here is an image of a sliced kiwi on a white background.')],
                    usage=RequestUsage(
                        input_tokens=1329,
                        output_tokens=15,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-BRloJHR654fSD0fcvLWZxtKtn0pag',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_returning_dict(allow_model_requests: None, agent: Agent):
    async with agent:
        result = await agent.run('Get me a dict, respond on one line')
        assert result.output == snapshot('{"foo":"bar","baz":123}')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me a dict, respond on one line',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_dict', args='{}', tool_call_id='call_oqKviITBj8PwpQjGyUu4Zu5x')],
                    usage=RequestUsage(
                        input_tokens=195,
                        output_tokens=11,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-BRloOs7Bb2tq8wJyy9Rv7SQ7L65a7',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_dict',
                            content={'foo': 'bar', 'baz': 123},
                            tool_call_id='call_oqKviITBj8PwpQjGyUu4Zu5x',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"foo":"bar","baz":123}')],
                    usage=RequestUsage(
                        input_tokens=222,
                        output_tokens=11,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-BRloPczU1HSCWnreyo21DdNtdOM7L',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_returning_unstructured_dict(allow_model_requests: None, agent: Agent):
    async with agent:
        result = await agent.run('Get me an unstructured dict, respond on one line')
        assert result.output == snapshot('{"foo":"bar","baz":123}')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me an unstructured dict, respond on one line',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_unstructured_dict', args='{}', tool_call_id='call_R0n2R7S9vL2aZOX25T9jahTd'
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=343,
                        output_tokens=12,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-CLbP82ODQMEznhobUKdq6Rjn9Aa12',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_unstructured_dict',
                            content={'foo': 'bar', 'baz': 123},
                            tool_call_id='call_R0n2R7S9vL2aZOX25T9jahTd',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"foo":"bar","baz":123}')],
                    usage=RequestUsage(
                        input_tokens=374,
                        output_tokens=10,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-CLbPAOYN3jPYdvYeD8JNOOXF5N554',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_returning_error(allow_model_requests: None, agent: Agent):
    async with agent:
        result = await agent.run('Get me an error, pass False as a value, unless the tool tells you otherwise')
        assert result.output == snapshot(
            'I called the tool with the correct parameter, and it returned: "This is not an error."'
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me an error, pass False as a value, unless the tool tells you otherwise',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_error',
                            args='{"value":false}',
                            tool_call_id='call_rETXZWddAGZSHyVHAxptPGgc',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=203,
                        output_tokens=15,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-BRloSNg7aGSp1rXDkhInjMIUHKd7A',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Error executing tool get_error: This is an error. Call the tool with True instead',
                            tool_name='get_error',
                            tool_call_id='call_rETXZWddAGZSHyVHAxptPGgc',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_error',
                            args='{"value":true}',
                            tool_call_id='call_4xGyvdghYKHN8x19KWkRtA5N',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=250,
                        output_tokens=15,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-BRloTvSkFeX4DZKQLqfH9KbQkWlpt',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_error',
                            content='This is not an error',
                            tool_call_id='call_4xGyvdghYKHN8x19KWkRtA5N',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='I called the tool with the correct parameter, and it returned: "This is not an error."'
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=277,
                        output_tokens=22,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-BRloU3MhnqNEqujs28a3ofRbs7VPF',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_returning_none(allow_model_requests: None, agent: Agent):
    async with agent:
        result = await agent.run('Call the none tool and say Hello')
        assert result.output == snapshot('Hello! How can I assist you today?')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Call the none tool and say Hello',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='get_none', args='{}', tool_call_id='call_mJTuQ2Cl5SaHPTJbIILEUhJC')],
                    usage=RequestUsage(
                        input_tokens=193,
                        output_tokens=11,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-BRloX2RokWc9j9PAXAuNXGR73WNqY',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_none',
                            content=[],
                            tool_call_id='call_mJTuQ2Cl5SaHPTJbIILEUhJC',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Hello! How can I assist you today?')],
                    usage=RequestUsage(
                        input_tokens=212,
                        output_tokens=11,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-BRloYWGujk8yE94gfVSsM1T1Ol2Ej',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_returning_multiple_items(allow_model_requests: None, agent: Agent, image_content: BinaryContent):
    async with agent:
        result = await agent.run('Get me multiple items and summarize in one sentence')
        assert result.output == snapshot(
            'The data includes two strings, a dictionary with a key-value pair, and an image of a sliced kiwi.'
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Get me multiple items and summarize in one sentence',
                            timestamp=IsDatetime(),
                        )
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='get_multiple_items',
                            args='{}',
                            tool_call_id='call_kL0TvjEVQBDGZrn1Zv7iNYOW',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=195,
                        output_tokens=12,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'tool_calls'},
                    provider_response_id='chatcmpl-BRlobKLgm6vf79c9O8sloZaYx3coC',
                    finish_reason='tool_call',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='get_multiple_items',
                            content=[
                                'This is a string',
                                'Another string',
                                {'foo': 'bar', 'baz': 123},
                                'See file 1c8566',
                            ],
                            tool_call_id='call_kL0TvjEVQBDGZrn1Zv7iNYOW',
                            timestamp=IsDatetime(),
                        ),
                        UserPromptPart(
                            content=[
                                'This is file 1c8566:',
                                image_content,
                            ],
                            timestamp=IsDatetime(),
                        ),
                    ],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The data includes two strings, a dictionary with a key-value pair, and an image of a sliced kiwi.'
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=1355,
                        output_tokens=24,
                        details={
                            'accepted_prediction_tokens': 0,
                            'audio_tokens': 0,
                            'reasoning_tokens': 0,
                            'rejected_prediction_tokens': 0,
                        },
                    ),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_details={'finish_reason': 'stop'},
                    provider_response_id='chatcmpl-BRloepWR5NJpTgSqFBGTSPeM1SWm8',
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )


async def test_tool_metadata_extraction():
    """Test that MCP tool metadata is properly extracted into ToolDefinition."""

    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    async with server:
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        tools = [tool.tool_def for tool in (await server.get_tools(ctx)).values()]
        # find `celsius_to_fahrenheit`
        celsius_to_fahrenheit = next(tool for tool in tools if tool.name == 'celsius_to_fahrenheit')
        assert celsius_to_fahrenheit.metadata is not None
        assert celsius_to_fahrenheit.metadata.get('annotations') is not None
        assert celsius_to_fahrenheit.metadata.get('annotations', {}).get('title', None) == 'Celsius to Fahrenheit'
        assert celsius_to_fahrenheit.metadata.get('output_schema') is not None
        assert celsius_to_fahrenheit.metadata.get('output_schema', {}).get('type', None) == 'object'


async def test_client_sampling(run_context: RunContext[int]):
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    server.sampling_model = TestModel(custom_output_text='sampling model response')
    async with server:
        result = await server.direct_call_tool('use_sampling', {'foo': 'bar'})
        assert result == snapshot(
            {
                '_meta': None,
                'role': 'assistant',
                'content': {'type': 'text', 'text': 'sampling model response', 'annotations': None, '_meta': None},
                'model': 'test',
                'stopReason': None,
            }
        )


async def test_client_sampling_disabled(run_context: RunContext[int]):
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'], allow_sampling=False)
    server.sampling_model = TestModel(custom_output_text='sampling model response')
    async with server:
        with pytest.raises(ModelRetry, match='Error executing tool use_sampling: Sampling not supported'):
            await server.direct_call_tool('use_sampling', {'foo': 'bar'})


async def test_mcp_server_raises_mcp_error(
    allow_model_requests: None, mcp_server: MCPServerStdio, agent: Agent, run_context: RunContext[int]
) -> None:
    mcp_error = McpError(error=ErrorData(code=400, message='Test MCP error conversion'))

    async with agent:
        with patch.object(
            mcp_server._client,  # pyright: ignore[reportPrivateUsage]
            'send_request',
            new=AsyncMock(side_effect=mcp_error),
        ):
            with pytest.raises(ModelRetry, match='Test MCP error conversion'):
                await mcp_server.direct_call_tool('test_tool', {})


def test_map_from_mcp_params_model_request():
    params = CreateMessageRequestParams(
        messages=[
            SamplingMessage(role='user', content=TextContent(type='text', text='xx')),
            SamplingMessage(
                role='user',
                content=ImageContent(type='image', data=base64.b64encode(b'img').decode(), mimeType='image/png'),
            ),
        ],
        maxTokens=8,
    )
    pai_messages = map_from_mcp_params(params)
    assert pai_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='xx', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(
                        content=[BinaryContent(data=b'img', media_type='image/png', identifier='978ea7')],
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ]
            )
        ]
    )


def test_map_from_mcp_params_model_response():
    params = CreateMessageRequestParams(
        messages=[
            SamplingMessage(role='assistant', content=TextContent(type='text', text='xx')),
        ],
        maxTokens=8,
    )
    pai_messages = map_from_mcp_params(params)
    assert pai_messages == snapshot(
        [
            ModelResponse(
                parts=[TextPart(content='xx')],
                timestamp=IsNow(tz=timezone.utc),
            )
        ]
    )


def test_map_from_model_response():
    with pytest.raises(UnexpectedModelBehavior, match='Unexpected part type: ThinkingPart, expected TextPart'):
        map_from_model_response(ModelResponse(parts=[ThinkingPart(content='Thinking...')]))


async def test_elicitation_callback_functionality(run_context: RunContext[int]):
    """Test that elicitation callback is actually called and works."""
    # Track callback execution
    callback_called = False
    callback_message = None
    callback_response = 'Yes, proceed with the action'

    async def mock_elicitation_callback(
        context: RequestContext[ClientSession, Any, Any], params: ElicitRequestParams
    ) -> ElicitResult:
        nonlocal callback_called, callback_message
        callback_called = True
        callback_message = params.message
        return ElicitResult(action='accept', content={'response': callback_response})

    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'], elicitation_callback=mock_elicitation_callback)

    async with server:
        # Call the tool that uses elicitation
        result = await server.direct_call_tool('use_elicitation', {'question': 'Should I continue?'})

        # Verify the callback was called
        assert callback_called, 'Elicitation callback should have been called'
        assert callback_message == 'Should I continue?', 'Callback should receive the question'
        assert result == f'User responded: {callback_response}', 'Tool should return the callback response'


async def test_elicitation_callback_not_set(run_context: RunContext[int]):
    """Test that elicitation fails when no callback is set."""
    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'])

    async with server:
        # Should raise an error when elicitation is attempted without callback
        with pytest.raises(ModelRetry, match='Elicitation not supported'):
            await server.direct_call_tool('use_elicitation', {'question': 'Should I continue?'})


def test_load_mcp_servers(tmp_path: Path):
    config = tmp_path / 'mcp.json'

    config.write_text('{"mcpServers": {"potato": {"url": "https://example.com/mcp"}}}')
    server = load_mcp_servers(config)[0]
    assert server == MCPServerStreamableHTTP(url='https://example.com/mcp', id='potato', tool_prefix='potato')

    config.write_text('{"mcpServers": {"potato": {"command": "python", "args": ["-m", "tests.mcp_server"]}}}')
    server = load_mcp_servers(config)[0]
    assert server == MCPServerStdio(
        command='python', args=['-m', 'tests.mcp_server'], id='potato', tool_prefix='potato'
    )

    config.write_text('{"mcpServers": {"potato": {"url": "https://example.com/sse"}}}')
    server = load_mcp_servers(config)[0]
    assert server == MCPServerSSE(url='https://example.com/sse', id='potato', tool_prefix='potato')

    with pytest.raises(FileNotFoundError):
        load_mcp_servers(tmp_path / 'does_not_exist.json')


def test_load_mcp_servers_with_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test environment variable expansion in config files."""
    config = tmp_path / 'mcp.json'

    # Test with environment variables in command
    monkeypatch.setenv('PYTHON_CMD', 'python3')
    monkeypatch.setenv('MCP_MODULE', 'tests.mcp_server')
    config.write_text('{"mcpServers": {"my_server": {"command": "${PYTHON_CMD}", "args": ["-m", "${MCP_MODULE}"]}}}')

    servers = load_mcp_servers(config)

    assert len(servers) == 1
    server = servers[0]
    assert isinstance(server, MCPServerStdio)
    assert server.command == 'python3'
    assert server.args == ['-m', 'tests.mcp_server']
    assert server.id == 'my_server'
    assert server.tool_prefix == 'my_server'


def test_load_mcp_servers_env_var_in_env_dict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test environment variable expansion in env dict."""
    config = tmp_path / 'mcp.json'

    # Test with environment variables in env dict
    monkeypatch.setenv('API_KEY', 'secret123')
    config.write_text(
        '{"mcpServers": {"my_server": {"command": "python", "args": ["-m", "tests.mcp_server"], '
        '"env": {"API_KEY": "${API_KEY}"}}}}'
    )

    servers = load_mcp_servers(config)

    assert len(servers) == 1
    server = servers[0]
    assert isinstance(server, MCPServerStdio)
    assert server.env == {'API_KEY': 'secret123'}


def test_load_mcp_servers_env_var_expansion_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test environment variable expansion in URL."""
    config = tmp_path / 'mcp.json'

    # Test with environment variables in URL
    monkeypatch.setenv('SERVER_HOST', 'example.com')
    monkeypatch.setenv('SERVER_PORT', '8080')
    config.write_text('{"mcpServers": {"web_server": {"url": "https://${SERVER_HOST}:${SERVER_PORT}/mcp"}}}')

    servers = load_mcp_servers(config)

    assert len(servers) == 1
    server = servers[0]
    assert isinstance(server, MCPServerStreamableHTTP)
    assert server.url == 'https://example.com:8080/mcp'


def test_load_mcp_servers_undefined_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that undefined environment variables raise an error."""
    config = tmp_path / 'mcp.json'

    # Make sure the environment variable is not set
    monkeypatch.delenv('UNDEFINED_VAR', raising=False)

    config.write_text('{"mcpServers": {"my_server": {"command": "${UNDEFINED_VAR}", "args": []}}}')

    with pytest.raises(ValueError, match='Environment variable \\$\\{UNDEFINED_VAR\\} is not defined'):
        load_mcp_servers(config)


def test_load_mcp_servers_partial_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test environment variables in partial strings."""
    config = tmp_path / 'mcp.json'

    monkeypatch.setenv('HOST', 'example.com')
    monkeypatch.setenv('PATH_SUFFIX', 'mcp')
    config.write_text('{"mcpServers": {"server": {"url": "https://${HOST}/api/${PATH_SUFFIX}"}}}')

    servers = load_mcp_servers(config)

    assert len(servers) == 1
    server = servers[0]
    assert isinstance(server, MCPServerStreamableHTTP)
    assert server.url == 'https://example.com/api/mcp'


def test_load_mcp_servers_with_non_string_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that non-string primitive values (int, bool, null) in nested structures are passed through unchanged."""
    config = tmp_path / 'mcp.json'

    # Create a config with environment variables and extra fields containing primitives
    # The extra fields will be ignored during validation but go through _expand_env_vars
    monkeypatch.setenv('PYTHON_CMD', 'python')
    config.write_text(
        '{"mcpServers": {"my_server": {"command": "${PYTHON_CMD}", "args": ["-m", "tests.mcp_server"], '
        '"metadata": {"count": 42, "enabled": true, "value": null}}}}'
    )

    # This should successfully expand env vars and ignore the metadata field
    servers = load_mcp_servers(config)

    assert len(servers) == 1
    server = servers[0]
    assert isinstance(server, MCPServerStdio)
    assert server.command == 'python'


def test_load_mcp_servers_with_default_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test ${VAR:-default} syntax for environment variable expansion."""
    config = tmp_path / 'mcp.json'

    # Test with undefined variable using default
    monkeypatch.delenv('UNDEFINED_VAR', raising=False)
    config.write_text('{"mcpServers": {"server": {"command": "${UNDEFINED_VAR:-python3}", "args": []}}}')

    servers = load_mcp_servers(config)
    assert len(servers) == 1
    server = servers[0]
    assert isinstance(server, MCPServerStdio)
    assert server.command == 'python3'

    # Test with defined variable (should use actual value, not default)
    monkeypatch.setenv('DEFINED_VAR', 'actual_value')
    config.write_text('{"mcpServers": {"server": {"command": "${DEFINED_VAR:-default_value}", "args": []}}}')

    servers = load_mcp_servers(config)
    assert len(servers) == 1
    server = servers[0]
    assert isinstance(server, MCPServerStdio)
    assert server.command == 'actual_value'

    # Test with empty string as default
    monkeypatch.delenv('UNDEFINED_VAR', raising=False)
    config.write_text('{"mcpServers": {"server": {"command": "${UNDEFINED_VAR:-}", "args": []}}}')

    servers = load_mcp_servers(config)
    assert len(servers) == 1
    server = servers[0]
    assert isinstance(server, MCPServerStdio)
    assert server.command == ''


def test_load_mcp_servers_with_default_values_in_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test ${VAR:-default} syntax in URLs."""
    config = tmp_path / 'mcp.json'

    # Test with default values in URL
    monkeypatch.delenv('HOST', raising=False)
    monkeypatch.setenv('PROTOCOL', 'https')
    config.write_text('{"mcpServers": {"server": {"url": "${PROTOCOL:-http}://${HOST:-localhost}:${PORT:-8080}/mcp"}}}')

    servers = load_mcp_servers(config)
    assert len(servers) == 1
    server = servers[0]
    assert isinstance(server, MCPServerStreamableHTTP)
    assert server.url == 'https://localhost:8080/mcp'


def test_load_mcp_servers_with_default_values_in_env_dict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test ${VAR:-default} syntax in env dictionary."""
    config = tmp_path / 'mcp.json'

    monkeypatch.delenv('API_KEY', raising=False)
    monkeypatch.setenv('CUSTOM_VAR', 'custom_value')
    config.write_text(
        '{"mcpServers": {"server": {"command": "python", "args": [], '
        '"env": {"API_KEY": "${API_KEY:-default_key}", "CUSTOM": "${CUSTOM_VAR:-fallback}"}}}}'
    )

    servers = load_mcp_servers(config)
    assert len(servers) == 1
    server = servers[0]
    assert isinstance(server, MCPServerStdio)
    assert server.env == {'API_KEY': 'default_key', 'CUSTOM': 'custom_value'}


def test_load_mcp_servers_with_complex_default_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test ${VAR:-default} syntax with special characters in default."""
    config = tmp_path / 'mcp.json'

    monkeypatch.delenv('PATH_VAR', raising=False)
    # Test default with slashes, dots, and dashes
    config.write_text('{"mcpServers": {"server": {"command": "${PATH_VAR:-/usr/local/bin/python-3.10}", "args": []}}}')

    servers = load_mcp_servers(config)
    assert len(servers) == 1
    server = servers[0]
    assert isinstance(server, MCPServerStdio)
    assert server.command == '/usr/local/bin/python-3.10'


def test_load_mcp_servers_with_mixed_syntax(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test mixing ${VAR} and ${VAR:-default} syntax in the same config."""
    config = tmp_path / 'mcp.json'

    monkeypatch.setenv('REQUIRED_VAR', 'required_value')
    monkeypatch.delenv('OPTIONAL_VAR', raising=False)
    config.write_text(
        '{"mcpServers": {"server": {"command": "${REQUIRED_VAR}", "args": ["${OPTIONAL_VAR:-default_arg}"]}}}'
    )

    servers = load_mcp_servers(config)
    assert len(servers) == 1
    server = servers[0]
    assert isinstance(server, MCPServerStdio)
    assert server.command == 'required_value'
    assert server.args == ['default_arg']


async def test_server_info(mcp_server: MCPServerStdio) -> None:
    with pytest.raises(
        AttributeError, match='The `MCPServerStdio.server_info` is only instantiated after initialization.'
    ):
        mcp_server.server_info
    async with mcp_server:
        assert mcp_server.server_info is not None
        assert mcp_server.server_info.name == 'Pydantic AI MCP Server'


async def test_agent_run_stream_with_mcp_server_http(allow_model_requests: None, model: Model):
    server = MCPServerStreamableHTTP(url='https://mcp.deepwiki.com/mcp', timeout=30)
    agent = Agent(model, toolsets=[server], instructions='Be concise.')

    # This should not raise an error.
    # See https://github.com/pydantic/pydantic-ai/issues/2818#issuecomment-3476480829
    async with agent.run_stream('Summarize the pydantic/pydantic-ai repo in one sentence') as result:
        output = await result.get_output()
    assert output == snapshot(
        'The `pydantic/pydantic-ai` repository is a Python agent framework designed to facilitate the development of production-grade Generative AI applications and workflows with a focus on type-safety and an ergonomic developer experience.'
    )
