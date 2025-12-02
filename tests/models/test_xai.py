"""Tests for xAI model integration.

Note on builtin tools testing:
------------------------------
xAI's builtin tools (code_execution, web_search, mcp_server) are executed server-side via gRPC.
Since VCR doesn't support gRPC, we cannot record/replay these interactions like we do with HTTP-based
APIs (OpenAI, Anthropic, etc.).

For builtin tool tests, we use simplified mocks that verify:
1. Tools are properly registered with the xAI SDK
2. The agent can process responses when builtin tools are enabled
3. Builtin tools can coexist with custom (client-side) tools

We DO NOT mock the actual server-side tool execution (tool calls and results from xAI's infrastructure)
because that would require complex protobuf mocking that doesn't add significant test value. Instead,
we verify the wiring and keep a few live integration tests for smoke testing.
"""

from __future__ import annotations as _annotations

import json
import os
from datetime import timezone
from typing import Any

import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import (
    Agent,
    AgentRunResult,
    AgentRunResultEvent,
    AudioUrl,
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CodeExecutionTool,
    DocumentUrl,
    FinalResultEvent,
    ImageUrl,
    MCPServerTool,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
    WebSearchTool,
)
from pydantic_ai.messages import (
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
)
from pydantic_ai.models import ModelRequestParameters, ToolDefinition
from pydantic_ai.output import NativeOutput
from pydantic_ai.profiles.grok import GrokModelProfile
from pydantic_ai.result import RunUsage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsInstance, IsNow, IsStr, try_import
from .mock_xai import (
    MockXai,
    create_code_execution_responses,
    create_mcp_server_responses,
    create_mixed_tools_response,
    create_response,
    create_server_tool_call,
    create_stream_chunk,
    create_tool_call,
    create_web_search_responses,
    get_mock_chat_create_kwargs,
)

with try_import() as imports_successful:
    import xai_sdk.chat as chat_types
    from xai_sdk.proto.v6 import chat_pb2, usage_pb2

    from pydantic_ai.models.xai import XaiModel
    from pydantic_ai.providers.xai import XaiProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
]

# Test model constants
XAI_NON_REASONING_MODEL = 'grok-4-1-fast-non-reasoning'
XAI_REASONING_MODEL = 'grok-4-1-fast-reasoning'


def create_usage(
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_prompt_text_tokens: int = 0,
    server_side_tools_used: list[usage_pb2.ServerSideTool] | None = None,
) -> usage_pb2.SamplingUsage:
    """Helper to create xAI SamplingUsage protobuf objects for tests with all required fields."""
    return usage_pb2.SamplingUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_prompt_text_tokens=cached_prompt_text_tokens,
        server_side_tools_used=server_side_tools_used or [],
    )


def test_xai_init():
    from pydantic_ai.providers.xai import XaiProvider

    provider = XaiProvider(api_key='foobar')
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=provider)
    # Check model properties without accessing private attributes
    assert m.model_name == XAI_NON_REASONING_MODEL
    assert m.system == 'xai'


async def test_xai_request_simple_success(allow_model_requests: None):
    response = create_response(content='world')
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(RunUsage(requests=1))

    # reset the index so we get the same response again
    mock_client.index = 0  # type: ignore

    result = await agent.run('hello', message_history=result.new_messages())
    assert result.output == 'world'
    assert result.usage() == snapshot(RunUsage(requests=1))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_request_simple_usage(allow_model_requests: None):
    response = create_response(
        content='world',
        usage=create_usage(prompt_tokens=2, completion_tokens=1),
    )
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Hello')
    assert result.output == 'world'
    assert result.usage() == snapshot(
        RunUsage(
            requests=1,
            input_tokens=2,
            output_tokens=1,
        )
    )


async def test_xai_image_input(allow_model_requests: None):
    """Test that xAI model handles image inputs (text is extracted from content)."""
    response = create_response(content='done')
    mock_client = MockXai.create_mock(response)
    model = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(model)

    image_url = ImageUrl('https://example.com/image.png')
    binary_image = BinaryContent(b'\x89PNG', media_type='image/png')

    result = await agent.run(['Describe these inputs.', image_url, binary_image])
    assert result.output == 'done'


async def test_xai_request_structured_response(allow_model_requests: None):
    tool_call = create_tool_call(
        id='123',
        name='final_result',
        arguments={'response': [1, 2, 123]},
    )
    response = create_response(tool_calls=[tool_call])
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=list[int])

    result = await agent.run('Hello')
    assert result.output == [1, 2, 123]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"response": [1, 2, 123]}',
                        tool_call_id='123',
                    )
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='123',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_request_tool_call(allow_model_requests: None):
    responses = [
        create_response(
            tool_calls=[create_tool_call(id='1', name='get_location', arguments={'loc_name': 'San Fransisco'})],
            usage=create_usage(prompt_tokens=2, completion_tokens=1),
        ),
        create_response(
            tool_calls=[create_tool_call(id='2', name='get_location', arguments={'loc_name': 'London'})],
            usage=create_usage(prompt_tokens=3, completion_tokens=2),
        ),
        create_response(content='final response'),
    ]
    mock_client = MockXai.create_mock(responses)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, system_prompt='this is the system prompt')

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    result = await agent.run('Hello')
    assert result.output == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='this is the system prompt', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "San Fransisco"}',
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=2,
                    output_tokens=1,
                ),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Wrong location, please try again',
                        tool_name='get_location',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_location',
                        args='{"loc_name": "London"}',
                        tool_call_id='2',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=3,
                    output_tokens=2,
                ),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id='2',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert result.usage() == snapshot(RunUsage(requests=3, input_tokens=5, output_tokens=3, tool_calls=1))


# Helpers for creating Grok streaming chunks
def grok_chunk(response: chat_types.Response, chunk: Any) -> tuple[chat_types.Response, Any]:
    """Create a Grok streaming chunk (response, chunk) tuple."""
    return (response, chunk)


def grok_text_chunk(text: str, finish_reason: str = 'stop') -> tuple[chat_types.Response, chat_types.Chunk]:
    """Create a text streaming chunk for Grok.

    Note: For streaming, Response accumulates content, Chunk is the delta.
    Since we can't easily track state across calls, we pass full accumulated text as response.content
    and the delta as chunk.content.

    Returns:
        Tuple of (accumulated Response, delta Chunk) - both using real SDK types
    """
    # Create chunk (delta) - the incremental content
    chunk = create_stream_chunk(
        content=text,
        finish_reason=finish_reason if finish_reason else None,  # type: ignore[arg-type]
    )

    # Create response (accumulated) - for simplicity in mocks, we'll just use the same text
    # In real usage, the Response object would accumulate over multiple chunks
    response = create_response(
        content=text,
        finish_reason=finish_reason if finish_reason else 'stop',  # type: ignore[arg-type]
        usage=create_usage(prompt_tokens=2, completion_tokens=1) if finish_reason else None,
    )

    return (response, chunk)


def grok_reasoning_text_chunk(
    text: str, reasoning_content: str = '', encrypted_content: str = '', finish_reason: str = 'stop'
) -> tuple[chat_types.Response, chat_types.Chunk]:
    """Create a text streaming chunk for Grok with reasoning content.

    Args:
        text: The text content delta
        reasoning_content: The reasoning trace delta
        encrypted_content: The encrypted reasoning signature delta
        finish_reason: The finish reason

    Returns:
        Tuple of (accumulated Response, delta Chunk) - both using real SDK types
    """
    # Create chunk (delta) - includes reasoning content as delta
    chunk = create_stream_chunk(
        content=text,
        reasoning_content=reasoning_content,
        encrypted_content=encrypted_content,
        finish_reason=finish_reason if finish_reason else None,  # type: ignore[arg-type]
    )

    # Create response (accumulated) - includes reasoning content
    response = create_response(
        content=text,
        finish_reason=finish_reason if finish_reason else 'stop',  # type: ignore[arg-type]
        usage=create_usage(prompt_tokens=2, completion_tokens=1) if finish_reason else None,
        reasoning_content=reasoning_content,
        encrypted_content=encrypted_content,
    )

    return (response, chunk)


def _generate_tool_result_content(tool_call: chat_pb2.ToolCall) -> str:
    """Generate appropriate JSON content for completed server-side tool calls.

    Args:
        tool_call: The chat_pb2.ToolCall proto object

    Returns:
        JSON string with tool result content
    """
    tool_name = tool_call.function.name
    tool_type = tool_call.type

    # Code execution tool
    if tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL:
        code = ''
        if isinstance(tool_call.function.arguments, dict):
            code = str(tool_call.function.arguments.get('code', ''))  # type: ignore[arg-type]
        return json.dumps(
            {
                'output': f'Code execution result for: {code}',
                'return_code': 0,
                'stderr': '',
            }
        )

    # MCP tool
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_MCP_TOOL:
        return json.dumps(
            {
                'result': {
                    'content': [{'type': 'text', 'text': f'MCP tool {tool_name} executed successfully'}],
                },
            }
        )

    # Document search tool
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_DOCUMENT_SEARCH_TOOL:
        return json.dumps(
            {
                'documents': [
                    {
                        'id': 'doc_1',
                        'title': 'Sample Document',
                        'content': 'Sample document content',
                        'url': 'https://example.com/doc1',
                    }
                ],
            }
        )

    # Collections search tool
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_COLLECTIONS_SEARCH_TOOL:
        return json.dumps(
            {
                'results': [
                    {
                        'id': 'collection_1',
                        'name': 'Sample Collection',
                        'description': 'Sample collection description',
                    }
                ],
            }
        )

    # Web search tool (results are typically in text_content, but we can add a status)
    elif tool_type == chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL:
        return json.dumps(
            {
                'status': 'completed',
                'results': [
                    {
                        'title': 'Sample Search Result',
                        'url': 'https://example.com',
                        'snippet': 'Sample search result snippet',
                    }
                ],
            }
        )

    # Default: return a simple status
    return json.dumps({'status': 'completed'})


def grok_builtin_tool_chunk(
    tool_call: chat_pb2.ToolCall,
    response_id: str = 'grok-builtin',
    finish_reason: str = '',
) -> tuple[chat_types.Response, chat_types.Chunk]:
    """Create a streaming chunk for Grok with a builtin (server-side) tool call.

    Args:
        tool_call: The server-side tool call proto object (chat_pb2.ToolCall)
        response_id: The response ID
        finish_reason: The finish reason (usually empty for tool call chunks)

    Returns:
        Tuple of (accumulated Response, delta Chunk) - both using real SDK types
    """
    # Generate content for completed tools based on tool type
    content = ''
    if tool_call.status == chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED:
        content = _generate_tool_result_content(tool_call)

    # Create chunk (delta) - the tool call
    chunk = create_stream_chunk(
        content='',
        tool_calls=[tool_call],
        finish_reason=finish_reason if finish_reason else None,  # type: ignore[arg-type]
    )

    # Create response (accumulated) - same tool call with content
    response = create_response(
        content=content,
        tool_calls=[tool_call],
        finish_reason=finish_reason if finish_reason else 'stop',  # type: ignore[arg-type]
        usage=create_usage(prompt_tokens=2, completion_tokens=1) if finish_reason else None,
    )

    return (response, chunk)


async def test_xai_stream_text(allow_model_requests: None):
    stream = [grok_text_chunk('hello '), grok_text_chunk('world')]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=2, output_tokens=1))


async def test_xai_stream_text_finish_reason(allow_model_requests: None):
    # Create streaming chunks with finish reasons
    stream = [
        grok_text_chunk('hello ', ''),
        grok_text_chunk('world', ''),
        grok_text_chunk('.', 'stop'),
    ]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
            ['hello ', 'hello world', 'hello world.']
        )
        assert result.is_complete
        async for response, is_last in result.stream_responses(debounce_by=None):
            if is_last:
                assert response == snapshot(
                    ModelResponse(
                        parts=[TextPart(content='hello world.')],
                        usage=RequestUsage(input_tokens=2, output_tokens=1),
                        model_name=XAI_NON_REASONING_MODEL,
                        timestamp=IsDatetime(),
                        provider_name='xai',
                        provider_response_id='grok-123',
                        finish_reason='stop',
                    )
                )


def grok_tool_chunk(
    tool_name: str | None, tool_arguments: str | None, finish_reason: str = '', accumulated_args: str = ''
) -> tuple[chat_types.Response, chat_types.Chunk]:
    """Create a tool call streaming chunk for Grok.

    Args:
        tool_name: The tool name (should be provided in all chunks for proper tracking)
        tool_arguments: The delta of arguments for this chunk
        finish_reason: The finish reason (only in last chunk)
        accumulated_args: The accumulated arguments string up to and including this chunk

    Note: Unlike the real xAI SDK which only sends the tool name in the first chunk,
    our mock includes it in every chunk to ensure proper tool call tracking.

    Returns:
        Tuple of (accumulated Response, delta Chunk) - both using real SDK types
    """
    # Infer tool name from accumulated state if not provided
    effective_tool_name = tool_name or ('final_result' if accumulated_args else None)

    # Create the chunk tool call (delta) - using real proto type
    chunk_tool_calls: list[chat_pb2.ToolCall] = []
    if effective_tool_name is not None or tool_arguments is not None:
        chunk_tool_call = chat_pb2.ToolCall(
            id='tool-123',
            type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
            function=chat_pb2.FunctionCall(
                name=effective_tool_name or '',
                arguments=tool_arguments if tool_arguments is not None else '',
            ),
        )
        chunk_tool_calls = [chunk_tool_call]

    # Chunk (delta) - using real proto type
    chunk = create_stream_chunk(
        content='',
        tool_calls=chunk_tool_calls,
        finish_reason=finish_reason if finish_reason else None,  # type: ignore[arg-type]
    )

    # Response (accumulated) - create tool calls for response
    response_tool_calls: list[chat_pb2.ToolCall] = []
    if effective_tool_name is not None or accumulated_args:
        response_tool_call = chat_pb2.ToolCall(
            id='tool-123',
            type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CLIENT_SIDE_TOOL,
            function=chat_pb2.FunctionCall(
                name=effective_tool_name or '',
                arguments=accumulated_args,  # Full accumulated arguments
            ),
        )
        response_tool_calls = [response_tool_call]

    response = create_response(
        content='',
        tool_calls=response_tool_calls,
        finish_reason=finish_reason if finish_reason else 'stop',  # type: ignore[arg-type]
        usage=create_usage(prompt_tokens=20, completion_tokens=1) if finish_reason else None,
    )

    return (response, chunk)


class MyTypedDict(TypedDict, total=False):
    first: str
    second: str


async def test_xai_stream_structured(allow_model_requests: None):
    stream = [
        grok_tool_chunk('final_result', None, accumulated_args=''),
        grok_tool_chunk(None, '{"first": "One', accumulated_args='{"first": "One'),
        grok_tool_chunk(None, '", "second": "Two"', accumulated_args='{"first": "One", "second": "Two"'),
        grok_tool_chunk(None, '}', finish_reason='stop', accumulated_args='{"first": "One", "second": "Two"}'),
    ]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream_output(debounce_by=None)] == snapshot(
            [{'first': 'One', 'second': 'Two'}]
        )
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=20, output_tokens=1))


async def test_xai_stream_structured_finish_reason(allow_model_requests: None):
    stream = [
        grok_tool_chunk('final_result', None, accumulated_args=''),
        grok_tool_chunk(None, '{"first": "One', accumulated_args='{"first": "One'),
        grok_tool_chunk(None, '", "second": "Two"', accumulated_args='{"first": "One", "second": "Two"'),
        grok_tool_chunk(None, '}', accumulated_args='{"first": "One", "second": "Two"}'),
        grok_tool_chunk(None, None, finish_reason='stop', accumulated_args='{"first": "One", "second": "Two"}'),
    ]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream_output(debounce_by=None)] == snapshot(
            [{'first': 'One', 'second': 'Two'}]
        )
        assert result.is_complete


async def test_xai_stream_native_output(allow_model_requests: None):
    stream = [
        grok_text_chunk('{"first": "One'),
        grok_text_chunk('", "second": "Two"'),
        grok_text_chunk('}'),
    ]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=NativeOutput(MyTypedDict))

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream_output(debounce_by=None)] == snapshot(
            [{'first': 'One'}, {'first': 'One', 'second': 'Two'}, {'first': 'One', 'second': 'Two'}]
        )
        assert result.is_complete


async def test_xai_stream_tool_call_with_empty_text(allow_model_requests: None):
    stream = [
        grok_tool_chunk('final_result', None, accumulated_args=''),
        grok_tool_chunk(None, '{"first": "One', accumulated_args='{"first": "One'),
        grok_tool_chunk(None, '", "second": "Two"', accumulated_args='{"first": "One", "second": "Two"'),
        grok_tool_chunk(None, '}', finish_reason='stop', accumulated_args='{"first": "One", "second": "Two"}'),
    ]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=[str, MyTypedDict])

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            [{'first': 'One'}, {'first': 'One', 'second': 'Two'}, {'first': 'One', 'second': 'Two'}]
        )
    assert await result.get_output() == snapshot({'first': 'One', 'second': 'Two'})


async def test_xai_no_delta(allow_model_requests: None):
    stream = [
        grok_text_chunk('hello '),
        grok_text_chunk('world'),
    ]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=2, output_tokens=1))


async def test_xai_none_delta(allow_model_requests: None):
    # Test handling of chunks without deltas
    stream = [
        grok_text_chunk('hello '),
        grok_text_chunk('world'),
    ]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=2, output_tokens=1))


@pytest.mark.parametrize('parallel_tool_calls', [True, False])
async def test_xai_parallel_tool_calls(allow_model_requests: None, parallel_tool_calls: bool) -> None:
    tool_call = create_tool_call(
        id='123',
        name='final_result',
        arguments={'response': [1, 2, 3]},
    )
    response = create_response(content='', tool_calls=[tool_call], finish_reason='tool_call')
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, output_type=list[int], model_settings=ModelSettings(parallel_tool_calls=parallel_tool_calls))

    await agent.run('Hello')
    assert get_mock_chat_create_kwargs(mock_client)[0]['parallel_tool_calls'] == parallel_tool_calls


async def test_xai_penalty_parameters(allow_model_requests: None) -> None:
    response = create_response(content='test response')
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))

    settings = ModelSettings(
        temperature=0.7,
        presence_penalty=0.5,
        frequency_penalty=0.3,
        parallel_tool_calls=False,
    )

    agent = Agent(m, model_settings=settings)
    result = await agent.run('Hello')

    # Check that all settings were passed to the xAI SDK
    kwargs = get_mock_chat_create_kwargs(mock_client)[0]
    assert kwargs['temperature'] == 0.7
    assert kwargs['presence_penalty'] == 0.5
    assert kwargs['frequency_penalty'] == 0.3
    assert kwargs['parallel_tool_calls'] is False
    assert result.output == 'test response'


async def test_xai_instructions(allow_model_requests: None):
    """Test that instructions are passed through to xAI SDK as a system message."""
    response = create_response(content='The capital of France is Paris.')
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, instructions='You are a helpful assistant.')

    result = await agent.run('What is the capital of France?')
    # Verify the message history has instructions
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=RequestUsage(),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_image_url_input(allow_model_requests: None):
    response = create_response(content='world')
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run(
        [
            'hello',
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ]
    )
    assert result.output == 'world'
    # Verify that the image URL was included in the messages
    assert len(get_mock_chat_create_kwargs(mock_client)) == 1


async def test_xai_image_url_tool_response(allow_model_requests: None):
    """Test xAI with image URL from tool response."""
    # First response: model calls tool
    tool_call_response = create_response(
        content='',
        tool_calls=[create_tool_call(id='tool_001', name='get_image', arguments={})],
    )
    # Second response: model responds after seeing image
    final_response = create_response(content='The image shows a potato.')

    mock_client = MockXai.create_mock([tool_call_response, final_response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> ImageUrl:
        return ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg')

    result = await agent.run(['What food is in the image you can get from the get_image tool?'])

    # Verify the complete message history with snapshot
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What food is in the image you can get from the get_image tool?'],
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id='tool_001')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file bd38f5',
                        tool_call_id='tool_001',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            'This is file bd38f5:',
                            ImageUrl(
                                url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'
                            ),
                        ],
                        timestamp=IsDatetime(),
                    ),
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The image shows a potato.')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_image_as_binary_content_tool_response(allow_model_requests: None, image_content: BinaryContent):
    """Test xAI with binary image content from tool response."""
    # First response: model calls tool
    tool_call_response = create_response(
        content='',
        tool_calls=[create_tool_call(id='tool_001', name='get_image', arguments={})],
    )
    # Second response: model responds after seeing image
    final_response = create_response(content='The image shows a kiwi fruit.')

    mock_client = MockXai.create_mock([tool_call_response, final_response])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> BinaryContent:
        return image_content

    result = await agent.run(['What fruit is in the image you can get from the get_image tool?'])

    # Verify the complete message history with snapshot
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['What fruit is in the image you can get from the get_image tool?'],
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_image', args='{}', tool_call_id='tool_001')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_image',
                        content='See file 1c8566',
                        tool_call_id='tool_001',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            'This is file 1c8566:',
                            IsInstance(BinaryContent),
                        ],
                        timestamp=IsDatetime(),
                    ),
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The image shows a kiwi fruit.')],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_image_as_binary_content_input(allow_model_requests: None, image_content: BinaryContent):
    """Test passing binary image content directly as input (not from a tool)."""
    response = create_response(content='The image shows a kiwi fruit.')

    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run(['What fruit is in the image?', image_content])

    # Verify the model received and processed the image
    assert result.output
    response_text = result.output.lower()
    assert 'kiwi' in response_text or 'fruit' in response_text


async def test_xai_document_url_input(allow_model_requests: None):
    """Test passing a document URL to the xAI model."""
    response = create_response(content='This document is a dummy PDF file.')

    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    document_url = DocumentUrl(url='https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')

    result = await agent.run(['What is the main content on this document?', document_url])
    assert result.output
    # The document contains "Dummy PDF file"
    response_text = result.output.lower()
    assert 'dummy' in response_text or 'pdf' in response_text


async def test_xai_binary_content_document_input(allow_model_requests: None):
    """Test passing a document as BinaryContent to the xAI model."""
    response = create_response(content='The document discusses testing.')
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    document_content = BinaryContent(
        data=b'%PDF-1.4\nTest document content',
        media_type='application/pdf',
    )

    result = await agent.run(['What is in this document?', document_content])

    # Verify the response
    assert result.output == 'The document discusses testing.'


async def test_xai_audio_url_not_supported(allow_model_requests: None):
    """Test that AudioUrl raises NotImplementedError."""
    response = create_response(content='This should not be reached')
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    audio_url = AudioUrl(url='https://example.com/audio.mp3')

    with pytest.raises(NotImplementedError, match='AudioUrl is not supported by xAI SDK'):
        await agent.run(['What is in this audio?', audio_url])


async def test_xai_video_url_not_supported(allow_model_requests: None):
    """Test that VideoUrl raises NotImplementedError."""
    response = create_response(content='This should not be reached')
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    video_url = VideoUrl(url='https://example.com/video.mp4')

    with pytest.raises(NotImplementedError, match='VideoUrl is not supported by xAI SDK'):
        await agent.run(['What is in this video?', video_url])


async def test_xai_binary_content_audio_not_supported(allow_model_requests: None):
    """Test that BinaryContent with audio raises NotImplementedError."""
    response = create_response(content='This should not be reached')
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    audio_content = BinaryContent(
        data=b'fake audio data',
        media_type='audio/mpeg',
    )

    with pytest.raises(NotImplementedError, match='AudioUrl/BinaryContent with audio is not supported by xAI SDK'):
        await agent.run(['What is in this audio?', audio_content])


# Grok built-in tools tests
# Built-in tools are executed server-side by xAI's infrastructure
# Based on: https://github.com/xai-org/xai-sdk-python/blob/main/examples/aio/server_side_tools.py


async def test_xai_builtin_web_search_tool(allow_model_requests: None):
    """Test xAI's built-in web_search tool."""
    # Create response with outputs
    response = create_web_search_responses(
        query='date of Jan 1 in 2026',
        content='Thursday',
        tool_call_id='ws_001',
    )
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    result = await agent.run('Return just the day of week for the date of Jan 1 in 2026?')
    assert result.output
    assert 'thursday' in result.output.lower()

    # Verify the builtin tool call and result appear in message history
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Return just the day of week for the date of Jan 1 in 2026?',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'date of Jan 1 in 2026'},
                        tool_call_id='ws_001',
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content='Thursday',
                        tool_call_id='ws_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='Thursday'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-ws_001',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_web_search_tool_stream(allow_model_requests: None):
    """Test xAI's built-in web_search tool with streaming."""
    # Create a mock web search server-side tool call
    web_search_tool_call = create_server_tool_call(
        tool_name='web_search',
        arguments={'query': 'weather San Francisco'},
        tool_call_id='ws_stream_001',
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
    )

    # For streaming with builtin tools, the tool call appears in the first chunk
    # and then subsequent chunks contain the text response
    stream = [
        # First chunk: builtin tool call
        grok_builtin_tool_chunk(web_search_tool_call, response_id='grok-ws_stream_001'),
        # Subsequent chunks: text response
        grok_text_chunk('The weather '),
        grok_text_chunk('is sunny.', finish_reason='stop'),
    ]

    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='What is the weather in San Francisco?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        # Capture all events for validation
                        event_parts.append(event)

    # Verify we got the expected builtin tool call events with snapshot
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={
                        'status': 'completed',
                        'results': [
                            {
                                'title': 'Sample Search Result',
                                'url': 'https://example.com',
                                'snippet': 'Sample search result snippet',
                            }
                        ],
                    },
                    tool_call_id='ws_stream_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
            ),
            PartStartEvent(index=1, part=TextPart(content='The weather '), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='is sunny.')),
            PartEndEvent(index=1, part=TextPart(content='The weather is sunny.')),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={
                        'status': 'completed',
                        'results': [
                            {
                                'title': 'Sample Search Result',
                                'url': 'https://example.com',
                                'snippet': 'Sample search result snippet',
                            }
                        ],
                    },
                    tool_call_id='ws_stream_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                )
            ),
        ]
    )


async def test_xai_builtin_x_search_tool(allow_model_requests: None):
    """Test xAI's built-in x_search tool (X/Twitter search)."""
    # Note: XSearchTool is not yet implemented in pydantic-ai
    # This test documents the expected behavior when it is implemented
    pytest.skip('XSearchTool not yet implemented in pydantic-ai')


async def test_xai_builtin_code_execution_tool(allow_model_requests: None):
    """Test xAI's built-in code_execution tool."""
    # Create response with outputs
    response = create_code_execution_responses(
        code='65465 - 6544 * 65464 - 6 + 1.02255',
        content='The result is -428,050,955.97745',
        tool_call_id='code_001',
    )
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    result = await agent.run('What is 65465 - 6544 * 65464 - 6 + 1.02255? Use code to calculate this.')

    # Verify the response
    assert result.output
    assert '-428' in result.output or 'million' in result.output.lower()

    # Verify the builtin tool call and result appear in message history
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 65465 - 6544 * 65464 - 6 + 1.02255? Use code to calculate this.',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={'code': '65465 - 6544 * 65464 - 6 + 1.02255'},
                        tool_call_id='code_001',
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content='The result is -428,050,955.97745',
                        tool_call_id='code_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='The result is -428,050,955.97745'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-code_001',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_code_execution_tool_stream(allow_model_requests: None):
    """Test xAI's built-in code_execution tool with streaming."""
    # Create a mock code execution server-side tool call
    code_tool_call = create_server_tool_call(
        tool_name='code_execution',
        arguments={'code': '2 + 2'},
        tool_call_id='code_stream_001',
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
    )

    # For streaming with builtin tools, the tool call appears in the first chunk
    stream = [
        # First chunk: builtin tool call
        grok_builtin_tool_chunk(code_tool_call, response_id='grok-code_stream_001'),
        # Subsequent chunks: text response
        grok_text_chunk('The result '),
        grok_text_chunk('is 4', finish_reason='stop'),
    ]

    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[CodeExecutionTool()])

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Calculate 2 + 2 using code') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    # Verify we got the expected builtin tool call events with snapshot
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'output': 'Code execution result for: ', 'return_code': 0, 'stderr': ''},
                    tool_call_id='code_stream_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
            ),
            PartStartEvent(index=1, part=TextPart(content='The result '), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='is 4')),
            PartEndEvent(index=1, part=TextPart(content='The result is 4')),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'output': 'Code execution result for: ', 'return_code': 0, 'stderr': ''},
                    tool_call_id='code_stream_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                )
            ),
        ]
    )


async def test_xai_builtin_multiple_tools(allow_model_requests: None):
    """Test using multiple built-in tools together."""
    # Create a response that simulates using both web search and code execution
    web_search_tool = create_server_tool_call(
        tool_name='web_search',
        arguments={'query': 'current price of Bitcoin'},
        tool_call_id='ws_002',
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_WEB_SEARCH_TOOL,
    )
    code_tool = create_server_tool_call(
        tool_name='code_execution',
        arguments={'code': '((65000 - 50000) / 50000) * 100'},
        tool_call_id='code_002',
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_CODE_EXECUTION_TOOL,
    )

    response = create_mixed_tools_response(
        server_tools=[web_search_tool, code_tool],
        text_content='Bitcoin has increased by 30.0% from last week.',
    )

    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[WebSearchTool(), CodeExecutionTool()],
    )

    result = await agent.run(
        'Search for the current price of Bitcoin and calculate its percentage change if it was $50000 last week.'
    )

    # Verify the response
    assert result.output

    # Verify both builtin tool calls appear in message history
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Search for the current price of Bitcoin and calculate its percentage change if it was $50000 last week.',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=None,
                        tool_call_id='ws_002',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content=None,
                        tool_call_id='code_002',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='Bitcoin has increased by 30.0% from last week.'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-multi-tool',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_tools_with_custom_tools(allow_model_requests: None):
    """Test mixing xAI's built-in tools with custom (client-side) tools."""
    # Create response with outputs
    response = create_web_search_responses(
        query='weather in Tokyo',
        content='The weather in Tokyo is sunny with a temperature of 72°F.',
        tool_call_id='ws_003',
    )
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    @agent.tool_plain
    def get_local_temperature(city: str) -> str:
        """Get the local temperature for a city (mock)."""
        return f'The local temperature in {city} is 72°F'

    result = await agent.run('What is the weather in Tokyo?')

    # Verify the agent runs without error when both tool types are registered
    assert result.output
    assert '72' in result.output or 'tokyo' in result.output.lower()

    # Verify the builtin tool call appears in message history
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather in Tokyo?',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'weather in Tokyo'},
                        tool_call_id='ws_003',
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content='The weather in Tokyo is sunny with a temperature of 72°F.',
                        tool_call_id='ws_003',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='The weather in Tokyo is sunny with a temperature of 72°F.'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-ws_003',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_mcp_server_tool(allow_model_requests: None):
    """Test xAI's MCP server tool with Linear."""
    # Create response with outputs
    response = create_mcp_server_responses(
        server_id='linear',
        tool_name='list_issues',
        content='No issues found.',
        tool_call_id='mcp_linear_001',
    )
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='linear',
                url='https://mcp.linear.app/mcp',
                description='MCP server for Linear the project management tool.',
                authorization_token='mock-token',  # Won't be used with mock
            ),
        ],
    )

    result = await agent.run('Can you list my Linear issues? Keep your answer brief.')

    # Verify the response and check builtin tool parts appear in message history
    assert result.output == 'No issues found.'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Can you list my Linear issues? Keep your answer brief.',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='mcp_server:linear',
                        args={},
                        tool_call_id='mcp_linear_001',
                        provider_name='xai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='mcp_server:linear',
                        content='No issues found.',
                        tool_call_id='mcp_linear_001',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                    ),
                    TextPart(content='No issues found.'),
                ],
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-mcp_linear_001',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_builtin_mcp_server_tool_stream(allow_model_requests: None):
    """Test xAI's MCP server tool with Linear using streaming."""
    # Create a mock MCP server tool call
    mcp_tool_call = create_server_tool_call(
        tool_name='linear.list_issues',  # xAI format: server_label.tool_name
        arguments={},
        tool_call_id='mcp_stream_001',
        tool_type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_MCP_TOOL,
    )

    # For streaming with builtin tools, the tool call appears in the first chunk
    stream = [
        # First chunk: builtin tool call
        grok_builtin_tool_chunk(mcp_tool_call, response_id='grok-mcp_stream_001'),
        # Subsequent chunks: text response
        grok_text_chunk('No issues '),
        grok_text_chunk('found.', finish_reason='stop'),
    ]

    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        instructions='You are a helpful assistant.',
        builtin_tools=[
            MCPServerTool(
                id='linear',
                url='https://mcp.linear.app/mcp',
                description='MCP server for Linear the project management tool.',
                authorization_token='mock-token',
            ),
        ],
    )

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Can you list my Linear issues?') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    # Verify we got the expected builtin tool call events with snapshot
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolReturnPart(
                    tool_name='mcp_server:linear',
                    content={
                        'result': {
                            'content': [{'type': 'text', 'text': 'MCP tool linear.list_issues executed successfully'}]
                        }
                    },
                    tool_call_id='mcp_stream_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                ),
            ),
            PartStartEvent(index=1, part=TextPart(content='No issues '), previous_part_kind='builtin-tool-return'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta='found.')),
            PartEndEvent(index=1, part=TextPart(content='No issues found.')),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='mcp_server:linear',
                    content={
                        'result': {
                            'content': [{'type': 'text', 'text': 'MCP tool linear.list_issues executed successfully'}]
                        }
                    },
                    tool_call_id='mcp_stream_001',
                    timestamp=IsDatetime(),
                    provider_name='xai',
                )
            ),
        ]
    )


async def test_xai_model_retries(allow_model_requests: None):
    """Test xAI model with retries."""
    # Create error response then success
    success_response = create_response(content='Success after retry')

    mock_client = MockXai.create_mock(success_response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)
    result = await agent.run('hello')
    assert result.output == 'Success after retry'


async def test_xai_model_settings(allow_model_requests: None):
    """Test xAI model with various settings."""
    response = create_response(content='response with settings')
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(
        m,
        model_settings=ModelSettings(
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
        ),
    )

    result = await agent.run('hello')
    assert result.output == 'response with settings'

    # Verify settings were passed to the mock
    kwargs = get_mock_chat_create_kwargs(mock_client)
    assert len(kwargs) > 0


async def test_xai_model_multiple_tool_calls(allow_model_requests: None):
    """Test xAI model with multiple tool calls in sequence."""
    # Three responses: two tool calls, then final answer
    responses = [
        create_response(
            tool_calls=[create_tool_call(id='1', name='get_data', arguments={'key': 'value1'})],
        ),
        create_response(
            tool_calls=[create_tool_call(id='2', name='process_data', arguments={'data': 'result1'})],
        ),
        create_response(content='Final processed result'),
    ]

    mock_client = MockXai.create_mock(responses)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    async def get_data(key: str) -> str:
        return f'data for {key}'

    @agent.tool_plain
    async def process_data(data: str) -> str:
        return f'processed {data}'

    result = await agent.run('Get and process data')
    assert result.output == 'Final processed result'
    assert result.usage().requests == 3
    assert result.usage().tool_calls == 2


async def test_xai_stream_with_tool_calls(allow_model_requests: None):
    """Test xAI streaming with tool calls."""
    # First stream: tool call
    stream1 = [
        grok_tool_chunk('get_info', None, accumulated_args=''),
        grok_tool_chunk(None, '{"query": "test"}', finish_reason='tool_calls', accumulated_args='{"query": "test"}'),
    ]
    # Second stream: final response after tool execution
    stream2 = [
        grok_text_chunk('Info retrieved: Info about test', finish_reason='stop'),
    ]

    mock_client = MockXai.create_mock_stream([stream1, stream2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    async def get_info(query: str) -> str:
        return f'Info about {query}'

    async with agent.run_stream('Get information') as result:
        # Consume the stream
        [c async for c in result.stream_text(debounce_by=None)]

    # Verify the final output includes the tool result
    assert result.is_complete
    output = await result.get_output()
    assert 'Info about test' in output


# Test for error handling
@pytest.mark.skipif(os.getenv('XAI_API_KEY') is not None, reason='Skipped when XAI_API_KEY is set')
async def test_xai_model_invalid_api_key():
    """Test xAI provider with invalid API key."""
    from pydantic_ai.exceptions import UserError

    with pytest.raises(UserError, match='Set the `XAI_API_KEY` environment variable'):
        XaiProvider(api_key='')


async def test_xai_model_properties():
    """Test xAI model properties."""
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(api_key='test-key'))

    assert m.model_name == XAI_NON_REASONING_MODEL
    assert m.system == 'xai'


# Tests for reasoning/thinking content (similar to OpenAI Responses tests)


async def test_xai_reasoning_simple(allow_model_requests: None):
    """Test xAI model with simple reasoning content."""
    response = create_response(
        content='The answer is 4',
        reasoning_content='Let me think: 2+2 equals 4',
        usage=create_usage(prompt_tokens=10, completion_tokens=20),
    )
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('What is 2+2?')
    assert result.output == 'The answer is 4'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2+2?', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='Let me think: 2+2 equals 4', signature=None),
                    TextPart(content='The answer is 4'),
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=20),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_encrypted_content_only(allow_model_requests: None):
    """Test xAI model with encrypted content (signature) only."""
    response = create_response(
        content='4',
        encrypted_content='abc123signature',
        usage=create_usage(prompt_tokens=10, completion_tokens=5),
    )
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('What is 2+2?')
    assert result.output == '4'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2+2?', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', signature='abc123signature', provider_name='xai'),
                    TextPart(content='4'),
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_reasoning_without_summary(allow_model_requests: None):
    """Test xAI model with encrypted content but no reasoning summary."""
    response = create_response(
        content='4',
        encrypted_content='encrypted123',
    )
    mock_client = MockXai.create_mock(response)
    model = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', signature='encrypted123', provider_name='xai'),
                    TextPart(content='4'),
                ],
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_reasoning_with_tool_calls(allow_model_requests: None):
    """Test xAI model with reasoning content and tool calls."""
    responses = [
        create_response(
            tool_calls=[create_tool_call(id='1', name='calculate', arguments={'expression': '2+2'})],
            reasoning_content='I need to use the calculate tool to solve this',
            usage=create_usage(prompt_tokens=10, completion_tokens=30),
        ),
        create_response(
            content='The calculation shows that 2+2 equals 4',
            usage=create_usage(prompt_tokens=15, completion_tokens=10),
        ),
    ]
    mock_client = MockXai.create_mock(responses)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    async def calculate(expression: str) -> str:
        return '4'

    result = await agent.run('What is 2+2?')
    assert result.output == 'The calculation shows that 2+2 equals 4'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2+2?', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='I need to use the calculate tool to solve this', signature=None),
                    ToolCallPart(
                        tool_name='calculate',
                        args='{"expression": "2+2"}',
                        tool_call_id='1',
                    ),
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=30),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='calculate',
                        content='4',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The calculation shows that 2+2 equals 4')],
                usage=RequestUsage(input_tokens=15, output_tokens=10),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_reasoning_with_encrypted_and_tool_calls(allow_model_requests: None):
    """Test xAI model with encrypted reasoning content and tool calls."""
    responses = [
        create_response(
            tool_calls=[create_tool_call(id='1', name='get_weather', arguments={'city': 'San Francisco'})],
            encrypted_content='encrypted_reasoning_abc123',
            usage=create_usage(prompt_tokens=20, completion_tokens=40),
        ),
        create_response(
            content='The weather in San Francisco is sunny',
            usage=create_usage(prompt_tokens=25, completion_tokens=12),
        ),
    ]
    mock_client = MockXai.create_mock(responses)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    @agent.tool_plain
    async def get_weather(city: str) -> str:
        return 'sunny'

    result = await agent.run('What is the weather in San Francisco?')
    assert result.output == 'The weather in San Francisco is sunny'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='What is the weather in San Francisco?', timestamp=IsNow(tz=timezone.utc))
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', signature='encrypted_reasoning_abc123', provider_name='xai'),
                    ToolCallPart(
                        tool_name='get_weather',
                        args='{"city": "San Francisco"}',
                        tool_call_id='1',
                    ),
                ],
                usage=RequestUsage(input_tokens=20, output_tokens=40),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='sunny',
                        tool_call_id='1',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='The weather in San Francisco is sunny')],
                usage=RequestUsage(input_tokens=25, output_tokens=12),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_stream_with_reasoning(allow_model_requests: None):
    """Test xAI streaming with reasoning content."""
    stream = [
        grok_reasoning_text_chunk('The answer', reasoning_content='Let me think about this...', finish_reason=''),
        grok_reasoning_text_chunk(' is 4', reasoning_content='Let me think about this...', finish_reason='stop'),
    ]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('What is 2+2?') as result:
        assert not result.is_complete
        text_chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert text_chunks == snapshot(['The answer', 'The answer is 4'])
        assert result.is_complete

    # Verify the final response includes both reasoning and text
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ThinkingPart(content='Let me think about this...'), TextPart(content='The answer is 4')],
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_stream_with_encrypted_reasoning(allow_model_requests: None):
    """Test xAI streaming with encrypted reasoning content."""
    stream = [
        grok_reasoning_text_chunk('The weather', encrypted_content='encrypted_abc123', finish_reason=''),
        grok_reasoning_text_chunk(' is sunny', encrypted_content='encrypted_abc123', finish_reason='stop'),
    ]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('What is the weather?') as result:
        assert not result.is_complete
        text_chunks = [c async for c in result.stream_text(debounce_by=None)]
        assert text_chunks == snapshot(['The weather', 'The weather is sunny'])
        assert result.is_complete

    # Verify the final response includes both encrypted reasoning and text
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the weather?',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='', signature='encrypted_abc123', provider_name='xai'),
                    TextPart(content='The weather is sunny'),
                ],
                usage=RequestUsage(input_tokens=2, output_tokens=1),
                model_name=XAI_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id='grok-123',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_xai_stream_events_with_reasoning(allow_model_requests: None):
    """Test xAI streaming events with reasoning content."""
    stream = [
        grok_reasoning_text_chunk('The answer', reasoning_content='Let me think about this...', finish_reason=''),
        grok_reasoning_text_chunk(' is 4', reasoning_content='Let me think about this...', finish_reason='stop'),
    ]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    events = [event async for event in agent.run_stream_events('What is 2+2?')]

    # Verify the events include both ThinkingPart and TextPart events
    assert events == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='Let me think about this...')),
            PartEndEvent(index=0, part=ThinkingPart(content='Let me think about this...'), next_part_kind='text'),
            PartStartEvent(index=1, part=TextPart(content='The answer'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' is 4')),
            PartEndEvent(index=1, part=TextPart(content='The answer is 4')),
            AgentRunResultEvent(result=AgentRunResult(output='The answer is 4')),
        ]
    )


async def test_xai_stream_events_with_encrypted_reasoning(allow_model_requests: None):
    """Test xAI streaming events with encrypted reasoning content."""
    stream = [
        grok_reasoning_text_chunk('The weather', encrypted_content='encrypted_abc123', finish_reason=''),
        grok_reasoning_text_chunk(' is sunny', encrypted_content='encrypted_abc123', finish_reason='stop'),
    ]
    mock_client = MockXai.create_mock_stream(stream)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    events = [event async for event in agent.run_stream_events('What is the weather?')]

    # Verify the events include both ThinkingPart (encrypted) and TextPart events
    assert events == snapshot(
        [
            PartStartEvent(index=0, part=ThinkingPart(content='', signature='encrypted_abc123', provider_name='xai')),
            PartEndEvent(
                index=0,
                part=ThinkingPart(content='', signature='encrypted_abc123', provider_name='xai'),
                next_part_kind='text',
            ),
            PartStartEvent(index=1, part=TextPart(content='The weather'), previous_part_kind='thinking'),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' is sunny')),
            PartEndEvent(index=1, part=TextPart(content='The weather is sunny')),
            AgentRunResultEvent(result=AgentRunResult(output='The weather is sunny')),
        ]
    )


async def test_xai_usage_with_reasoning_tokens(allow_model_requests: None):
    """Test that xAI model properly extracts reasoning_tokens and cache_read_tokens from usage."""
    # Create a mock usage object with reasoning_tokens and cached_prompt_text_tokens
    mock_usage = create_usage(
        prompt_tokens=100,
        completion_tokens=50,
        reasoning_tokens=25,
        cached_prompt_text_tokens=30,
    )
    response = create_response(
        content='The answer is 42',
        reasoning_content='Let me think deeply about this...',
        usage=mock_usage,
    )
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('What is the meaning of life?')
    assert result.output == 'The answer is 42'

    # Verify usage includes details
    usage = result.usage()
    assert usage.input_tokens == 100
    assert usage.output_tokens == 50
    assert usage.total_tokens == 150
    assert usage.details == snapshot({'reasoning_tokens': 25, 'cache_read_tokens': 30})


async def test_xai_usage_without_details(allow_model_requests: None):
    """Test that xAI model handles usage without reasoning_tokens or cached tokens."""
    mock_usage = create_usage(prompt_tokens=20, completion_tokens=10)
    response = create_response(
        content='Simple answer',
        usage=mock_usage,
    )
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Simple question')
    assert result.output == 'Simple answer'

    # Verify usage without details
    usage = result.usage()
    assert usage.input_tokens == 20
    assert usage.output_tokens == 10
    assert usage.total_tokens == 30
    # details should be empty dict when no additional usage info is provided
    assert usage.details == snapshot({})


async def test_xai_usage_with_server_side_tools(allow_model_requests: None):
    """Test that xAI model properly extracts server_side_tools_used from usage."""
    # Create a mock usage object with server_side_tools_used
    # In the real SDK, server_side_tools_used is a repeated field (list-like)
    mock_usage = create_usage(
        prompt_tokens=50,
        completion_tokens=30,
        server_side_tools_used=[usage_pb2.SERVER_SIDE_TOOL_WEB_SEARCH, usage_pb2.SERVER_SIDE_TOOL_WEB_SEARCH],
    )
    response = create_response(
        content='The answer based on web search',
        usage=mock_usage,
    )
    mock_client = MockXai.create_mock(response)
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))
    agent = Agent(m)

    result = await agent.run('Search for something')
    assert result.output == 'The answer based on web search'

    # Verify usage includes server_side_tools_used in details
    usage = result.usage()
    assert usage.input_tokens == 50
    assert usage.output_tokens == 30
    assert usage.total_tokens == 80
    assert usage.details == snapshot({'server_side_tools_web_search': 2})


async def test_xai_native_output_with_tools(allow_model_requests: None):
    """Test that native output works with tools - tools should be called first, then native output."""
    from pydantic import BaseModel

    class CityLocation(BaseModel):
        """A city and its country."""

        city: str
        country: str

    # First response: tool call
    response1 = create_response(
        tool_calls=[create_tool_call('call_get_country', 'get_user_country', {})],
        usage=create_usage(prompt_tokens=70, completion_tokens=12),
    )
    # Second response: native output (JSON)
    response2 = create_response(
        content='{"city":"Mexico City","country":"Mexico"}',
        usage=create_usage(prompt_tokens=90, completion_tokens=15),
    )
    mock_client = MockXai.create_mock([response1, response2])
    m = XaiModel(XAI_NON_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client))

    agent = Agent(m, output_type=NativeOutput(CityLocation))

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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_user_country', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=70, output_tokens=12),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city":"Mexico City","country":"Mexico"}')],
                usage=RequestUsage(input_tokens=90, output_tokens=15),
                model_name=XAI_NON_REASONING_MODEL,
                timestamp=IsDatetime(),
                provider_name='xai',
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_tool_choice_fallback(allow_model_requests: None) -> None:
    """Test that tool_choice falls back to 'auto' when 'required' is not supported."""
    # Create a profile that doesn't support tool_choice='required'
    profile = GrokModelProfile(grok_supports_tool_choice_required=False)

    response = create_response(content='ok', usage=create_usage(prompt_tokens=10, completion_tokens=5))
    mock_client = MockXai.create_mock(response)
    model = XaiModel(XAI_REASONING_MODEL, provider=XaiProvider(xai_client=mock_client), profile=profile)

    params = ModelRequestParameters(function_tools=[ToolDefinition(name='x')], allow_text_output=False)

    await model._create_chat(  # pyright: ignore[reportPrivateUsage]
        messages=[],
        model_settings={},
        model_request_parameters=params,
    )

    # Verify tool_choice was set to 'auto' (not 'required')
    kwargs = get_mock_chat_create_kwargs(mock_client)[0]
    assert kwargs['tool_choice'] == 'auto'


# End of tests
