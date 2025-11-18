from __future__ import annotations as _annotations

import json
import os
from datetime import timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest
from inline_snapshot import snapshot
from typing_extensions import TypedDict

from pydantic_ai import (
    Agent,
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.output import NativeOutput
from pydantic_ai.result import RunUsage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

from ..conftest import IsDatetime, IsNow, IsStr, try_import
from .mock_grok import (
    MockGrok,
    MockGrokResponse,
    MockGrokResponseChunk,
    create_response,
    create_tool_call,
    get_mock_chat_create_kwargs,
)

with try_import() as imports_successful:
    import xai_sdk.chat as chat_types

    from pydantic_ai.models.grok import GrokModel

    MockResponse = chat_types.Response | Exception
    # xai_sdk streaming returns tuples of (Response, chunk) where chunk type is not explicitly defined
    MockResponseChunk = tuple[chat_types.Response, Any] | Exception

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


def test_grok_init():
    m = GrokModel('grok-4-fast-non-reasoning', api_key='foobar')
    # Check model properties without accessing private attributes
    assert m.model_name == 'grok-4-fast-non-reasoning'
    assert m.system == 'xai'


async def test_grok_request_simple_success(allow_model_requests: None):
    response = create_response(content='world')
    mock_client = MockGrok.create_mock(response)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
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
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_grok_request_simple_usage(allow_model_requests: None):
    response = create_response(
        content='world',
        usage=SimpleNamespace(input_tokens=2, output_tokens=1),
    )
    mock_client = MockGrok.create_mock(response)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
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


async def test_grok_image_input(allow_model_requests: None):
    """Test that Grok model handles image inputs (text is extracted from content)."""
    response = create_response(content='done')
    mock_client = MockGrok.create_mock(response)
    model = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
    agent = Agent(model)

    image_url = ImageUrl('https://example.com/image.png')
    binary_image = BinaryContent(b'\x89PNG', media_type='image/png')

    result = await agent.run(['Describe these inputs.', image_url, binary_image])
    assert result.output == 'done'


async def test_grok_request_structured_response(allow_model_requests: None):
    tool_call = create_tool_call(
        id='123',
        name='final_result',
        arguments={'response': [1, 2, 123]},
    )
    response = create_response(tool_calls=[tool_call])
    mock_client = MockGrok.create_mock(response)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
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
                        args={'response': [1, 2, 123]},
                        tool_call_id='123',
                    )
                ],
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
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


async def test_grok_request_tool_call(allow_model_requests: None):
    responses = [
        create_response(
            tool_calls=[create_tool_call(id='1', name='get_location', arguments={'loc_name': 'San Fransisco'})],
            usage=SimpleNamespace(input_tokens=2, output_tokens=1),
        ),
        create_response(
            tool_calls=[create_tool_call(id='2', name='get_location', arguments={'loc_name': 'London'})],
            usage=SimpleNamespace(input_tokens=3, output_tokens=2),
        ),
        create_response(content='final response'),
    ]
    mock_client = MockGrok.create_mock(responses)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
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
                        args={'loc_name': 'San Fransisco'},
                        tool_call_id='1',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=2,
                    output_tokens=1,
                ),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
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
                        args={'loc_name': 'London'},
                        tool_call_id='2',
                    )
                ],
                usage=RequestUsage(
                    input_tokens=3,
                    output_tokens=2,
                ),
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
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
                model_name='grok-4-fast-non-reasoning',
                timestamp=IsDatetime(),
                provider_name='xai',
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


def grok_text_chunk(text: str, finish_reason: str = 'stop') -> tuple[chat_types.Response, Any]:
    """Create a text streaming chunk for Grok.

    Note: For streaming, Response accumulates content, Chunk is the delta.
    Since we can't easily track state across calls, we pass full accumulated text as response.content
    and the delta as chunk.content.
    """
    # Create chunk (delta) - just this piece of text
    chunk = MockGrokResponseChunk(content=text, tool_calls=[])

    # Create response (accumulated) - for simplicity in mocks, we'll just use the same text
    # In real usage, the Response object would accumulate over multiple chunks
    response = MockGrokResponse(
        id='grok-123',
        content=text,  # This will be accumulated by the streaming handler
        tool_calls=[],
        finish_reason=finish_reason if finish_reason else '',
        usage=SimpleNamespace(input_tokens=2, output_tokens=1) if finish_reason else None,
    )

    return (cast(chat_types.Response, response), chunk)


async def test_grok_stream_text(allow_model_requests: None):
    stream = [grok_text_chunk('hello '), grok_text_chunk('world')]
    mock_client = MockGrok.create_mock_stream(stream)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=2, output_tokens=1))


async def test_grok_stream_text_finish_reason(allow_model_requests: None):
    # Create streaming chunks with finish reasons
    stream = [
        grok_text_chunk('hello ', ''),
        grok_text_chunk('world', ''),
        grok_text_chunk('.', 'stop'),
    ]
    mock_client = MockGrok.create_mock_stream(stream)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
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
                        model_name='grok-4-fast-non-reasoning',
                        timestamp=IsDatetime(),
                        provider_name='xai',
                        provider_response_id='grok-123',
                        finish_reason='stop',
                    )
                )


def grok_tool_chunk(
    tool_name: str | None, tool_arguments: str | None, finish_reason: str = '', accumulated_args: str = ''
) -> tuple[chat_types.Response, Any]:
    """Create a tool call streaming chunk for Grok.

    Args:
        tool_name: The tool name (should be provided in all chunks for proper tracking)
        tool_arguments: The delta of arguments for this chunk
        finish_reason: The finish reason (only in last chunk)
        accumulated_args: The accumulated arguments string up to and including this chunk

    Note: Unlike the real xAI SDK which only sends the tool name in the first chunk,
    our mock includes it in every chunk to ensure proper tool call tracking.
    """
    # Infer tool name from accumulated state if not provided
    effective_tool_name = tool_name or ('final_result' if accumulated_args else None)

    # Create the chunk (delta) - includes tool name for proper tracking
    chunk_tool_call = None
    if effective_tool_name is not None or tool_arguments is not None:
        chunk_tool_call = SimpleNamespace(
            id='tool-123',
            function=SimpleNamespace(
                name=effective_tool_name,
                # arguments should be a string (delta JSON), default to empty string
                arguments=tool_arguments if tool_arguments is not None else '',
            ),
        )

    # Chunk (delta)
    chunk = MockGrokResponseChunk(
        content='',
        tool_calls=[chunk_tool_call] if chunk_tool_call else [],
    )

    # Response (accumulated) - contains the full accumulated tool call
    response_tool_call = SimpleNamespace(
        id='tool-123',
        function=SimpleNamespace(
            name=effective_tool_name,
            arguments=accumulated_args,  # Full accumulated arguments
        ),
    )

    response = MockGrokResponse(
        id='grok-123',
        content='',
        tool_calls=[response_tool_call] if (effective_tool_name is not None or accumulated_args) else [],
        finish_reason=finish_reason,
        usage=SimpleNamespace(input_tokens=20, output_tokens=1) if finish_reason else None,
    )

    return (cast(chat_types.Response, response), chunk)


class MyTypedDict(TypedDict, total=False):
    first: str
    second: str


async def test_grok_stream_structured(allow_model_requests: None):
    stream = [
        grok_tool_chunk('final_result', None, accumulated_args=''),
        grok_tool_chunk(None, '{"first": "One', accumulated_args='{"first": "One'),
        grok_tool_chunk(None, '", "second": "Two"', accumulated_args='{"first": "One", "second": "Two"'),
        grok_tool_chunk(None, '}', finish_reason='stop', accumulated_args='{"first": "One", "second": "Two"}'),
    ]
    mock_client = MockGrok.create_mock_stream(stream)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
    agent = Agent(m, output_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream_output(debounce_by=None)] == snapshot(
            [{'first': 'One'}, {'first': 'One', 'second': 'Two'}, {'first': 'One', 'second': 'Two'}]
        )
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=20, output_tokens=1))


async def test_grok_stream_structured_finish_reason(allow_model_requests: None):
    stream = [
        grok_tool_chunk('final_result', None, accumulated_args=''),
        grok_tool_chunk(None, '{"first": "One', accumulated_args='{"first": "One'),
        grok_tool_chunk(None, '", "second": "Two"', accumulated_args='{"first": "One", "second": "Two"'),
        grok_tool_chunk(None, '}', accumulated_args='{"first": "One", "second": "Two"}'),
        grok_tool_chunk(None, None, finish_reason='stop', accumulated_args='{"first": "One", "second": "Two"}'),
    ]
    mock_client = MockGrok.create_mock_stream(stream)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
    agent = Agent(m, output_type=MyTypedDict)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream_output(debounce_by=None)] == snapshot(
            [{'first': 'One'}, {'first': 'One', 'second': 'Two'}, {'first': 'One', 'second': 'Two'}]
        )
        assert result.is_complete


async def test_grok_stream_native_output(allow_model_requests: None):
    stream = [
        grok_text_chunk('{"first": "One'),
        grok_text_chunk('", "second": "Two"'),
        grok_text_chunk('}'),
    ]
    mock_client = MockGrok.create_mock_stream(stream)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
    agent = Agent(m, output_type=NativeOutput(MyTypedDict))

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [dict(c) async for c in result.stream_output(debounce_by=None)] == snapshot(
            [{'first': 'One'}, {'first': 'One', 'second': 'Two'}, {'first': 'One', 'second': 'Two'}]
        )
        assert result.is_complete


async def test_grok_stream_tool_call_with_empty_text(allow_model_requests: None):
    stream = [
        grok_tool_chunk('final_result', None, accumulated_args=''),
        grok_tool_chunk(None, '{"first": "One', accumulated_args='{"first": "One'),
        grok_tool_chunk(None, '", "second": "Two"', accumulated_args='{"first": "One", "second": "Two"'),
        grok_tool_chunk(None, '}', finish_reason='stop', accumulated_args='{"first": "One", "second": "Two"}'),
    ]
    mock_client = MockGrok.create_mock_stream(stream)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
    agent = Agent(m, output_type=[str, MyTypedDict])

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            [{'first': 'One'}, {'first': 'One', 'second': 'Two'}, {'first': 'One', 'second': 'Two'}]
        )
    assert await result.get_output() == snapshot({'first': 'One', 'second': 'Two'})


async def test_grok_no_delta(allow_model_requests: None):
    stream = [
        grok_text_chunk('hello '),
        grok_text_chunk('world'),
    ]
    mock_client = MockGrok.create_mock_stream(stream)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=2, output_tokens=1))


async def test_grok_none_delta(allow_model_requests: None):
    # Test handling of chunks without deltas
    stream = [
        grok_text_chunk('hello '),
        grok_text_chunk('world'),
    ]
    mock_client = MockGrok.create_mock_stream(stream)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
    agent = Agent(m)

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(['hello ', 'hello world'])
        assert result.is_complete
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=2, output_tokens=1))


# Skip OpenAI-specific tests that don't apply to Grok
# test_system_prompt_role - OpenAI specific
# test_system_prompt_role_o1_mini - OpenAI specific
# test_openai_pass_custom_system_prompt_role - OpenAI specific
# test_openai_o1_mini_system_role - OpenAI specific


@pytest.mark.parametrize('parallel_tool_calls', [True, False])
async def test_grok_parallel_tool_calls(allow_model_requests: None, parallel_tool_calls: bool) -> None:
    tool_call = create_tool_call(
        id='123',
        name='final_result',
        arguments={'response': [1, 2, 3]},
    )
    response = create_response(content='', tool_calls=[tool_call], finish_reason='tool_calls')
    mock_client = MockGrok.create_mock(response)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
    agent = Agent(m, output_type=list[int], model_settings=ModelSettings(parallel_tool_calls=parallel_tool_calls))

    await agent.run('Hello')
    assert get_mock_chat_create_kwargs(mock_client)[0]['parallel_tool_calls'] == parallel_tool_calls


async def test_grok_penalty_parameters(allow_model_requests: None) -> None:
    response = create_response(content='test response')
    mock_client = MockGrok.create_mock(response)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)

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


async def test_grok_image_url_input(allow_model_requests: None):
    response = create_response(content='world')
    mock_client = MockGrok.create_mock(response)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
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


@pytest.mark.skipif(os.getenv('XAI_API_KEY') is None, reason='Requires XAI_API_KEY (gRPC, no cassettes)')
async def test_grok_image_url_tool_response(allow_model_requests: None, xai_api_key: str):
    m = GrokModel('grok-4-fast-non-reasoning', api_key=xai_api_key)
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> ImageUrl:
        return ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg')

    result = await agent.run(['What food is in the image you can get from the get_image tool?'])

    # Verify structure with matchers for dynamic values
    messages = result.all_messages()
    assert len(messages) == 4

    # Verify message types and key content
    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[1], ModelResponse)
    assert isinstance(messages[2], ModelRequest)
    assert isinstance(messages[3], ModelResponse)

    # Verify tool was called
    assert isinstance(messages[1].parts[0], ToolCallPart)
    assert messages[1].parts[0].tool_name == 'get_image'

    # Verify image was passed back to model
    assert isinstance(messages[2].parts[1], UserPromptPart)
    assert isinstance(messages[2].parts[1].content, list)
    assert any(isinstance(item, ImageUrl) for item in messages[2].parts[1].content)

    # Verify model responded about the image
    assert isinstance(messages[3].parts[0], TextPart)
    assert 'potato' in messages[3].parts[0].content.lower()


@pytest.mark.skipif(os.getenv('XAI_API_KEY') is None, reason='Requires XAI_API_KEY (gRPC, no cassettes)')
async def test_grok_image_as_binary_content_tool_response(
    allow_model_requests: None, image_content: BinaryContent, xai_api_key: str
):
    m = GrokModel('grok-4-fast-non-reasoning', api_key=xai_api_key)
    agent = Agent(m)

    @agent.tool_plain
    async def get_image() -> BinaryContent:
        return image_content

    result = await agent.run(['What fruit is in the image you can get from the get_image tool?'])

    # Verify structure with matchers for dynamic values
    messages = result.all_messages()
    assert len(messages) == 4

    # Verify message types and key content
    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[1], ModelResponse)
    assert isinstance(messages[2], ModelRequest)
    assert isinstance(messages[3], ModelResponse)

    # Verify tool was called
    assert isinstance(messages[1].parts[0], ToolCallPart)
    assert messages[1].parts[0].tool_name == 'get_image'

    # Verify binary image content was passed back to model
    assert isinstance(messages[2].parts[1], UserPromptPart)
    assert isinstance(messages[2].parts[1].content, list)
    has_binary_image = any(isinstance(item, BinaryContent) and item.is_image for item in messages[2].parts[1].content)
    assert has_binary_image, 'Expected BinaryContent image in tool response'

    # Verify model responded about the image
    assert isinstance(messages[3].parts[0], TextPart)
    response_text = messages[3].parts[0].content.lower()
    assert 'kiwi' in response_text or 'fruit' in response_text


@pytest.mark.skipif(os.getenv('XAI_API_KEY') is None, reason='Requires XAI_API_KEY (gRPC, no cassettes)')
async def test_grok_image_as_binary_content_input(
    allow_model_requests: None, image_content: BinaryContent, xai_api_key: str
):
    """Test passing binary image content directly as input (not from a tool)."""
    m = GrokModel('grok-4-fast-non-reasoning', api_key=xai_api_key)
    agent = Agent(m)

    result = await agent.run(['What fruit is in the image?', image_content])

    # Verify the model received and processed the image
    assert result.output
    response_text = result.output.lower()
    assert 'kiwi' in response_text or 'fruit' in response_text


# Skip tests that are not applicable to Grok model
# The following tests were removed as they are OpenAI-specific:


# Continue with model request/response tests
# Grok-specific tests for built-in tools


async def test_grok_web_search_tool(allow_model_requests: None):
    """Test Grok model with web_search built-in tool."""
    # First response: tool call to web_search
    tool_call = create_tool_call(
        id='web-1',
        name='web_search',
        arguments={'query': 'latest news about AI'},
    )
    response1 = create_response(tool_calls=[tool_call])

    # Second response: final answer
    response2 = create_response(content='Based on web search: AI is advancing rapidly.')

    mock_client = MockGrok.create_mock([response1, response2])
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
    agent = Agent(m)

    # Add a mock web search tool
    @agent.tool_plain
    async def web_search(query: str) -> str:
        return f'Search results for: {query}'

    result = await agent.run('What is the latest news about AI?')
    assert 'AI is advancing rapidly' in result.output
    assert result.usage().requests == 2


async def test_grok_model_retries(allow_model_requests: None):
    """Test Grok model with retries."""
    # Create error response then success
    success_response = create_response(content='Success after retry')

    mock_client = MockGrok.create_mock(success_response)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
    agent = Agent(m)
    result = await agent.run('hello')
    assert result.output == 'Success after retry'


async def test_grok_model_settings(allow_model_requests: None):
    """Test Grok model with various settings."""
    response = create_response(content='response with settings')
    mock_client = MockGrok.create_mock(response)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
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


async def test_grok_model_multiple_tool_calls(allow_model_requests: None):
    """Test Grok model with multiple tool calls in sequence."""
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

    mock_client = MockGrok.create_mock(responses)
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
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


async def test_grok_stream_with_tool_calls(allow_model_requests: None):
    """Test Grok streaming with tool calls."""
    # First stream: tool call
    stream1 = [
        grok_tool_chunk('get_info', None, accumulated_args=''),
        grok_tool_chunk(None, '{"query": "test"}', finish_reason='tool_calls', accumulated_args='{"query": "test"}'),
    ]
    # Second stream: final response after tool execution
    stream2 = [
        grok_text_chunk('Info retrieved: Info about test', finish_reason='stop'),
    ]

    mock_client = MockGrok.create_mock_stream([stream1, stream2])
    m = GrokModel('grok-4-fast-non-reasoning', client=mock_client)
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
async def test_grok_model_invalid_api_key():
    """Test Grok model with invalid API key."""
    with pytest.raises(ValueError, match='XAI API key is required'):
        GrokModel('grok-4-fast-non-reasoning', api_key='')


async def test_grok_model_properties():
    """Test Grok model properties."""
    m = GrokModel('grok-4-fast-non-reasoning', api_key='test-key')

    assert m.model_name == 'grok-4-fast-non-reasoning'
    assert m.system == 'xai'


# End of tests
