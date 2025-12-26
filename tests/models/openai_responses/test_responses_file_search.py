"""Tests for OpenAI Responses model file search functionality."""

from __future__ import annotations

from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FinalResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPartDelta,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
)
from pydantic_ai.usage import RequestUsage

from ...conftest import IsDatetime, IsFloat, IsInt, IsNow, IsStr, try_import
from .conftest import cleanup_openai_resources

with try_import() as imports_successful:
    from pydantic_ai.models.openai import (
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.mark.vcr()
async def test_openai_responses_model_file_search_tool(allow_model_requests: None, openai_api_key: str):
    import asyncio
    import os
    import tempfile

    from openai import AsyncOpenAI

    from pydantic_ai.builtin_tools import FileSearchTool
    from pydantic_ai.providers.openai import OpenAIProvider

    async_client = AsyncOpenAI(api_key=openai_api_key)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write('Paris is the capital of France. It is known for the Eiffel Tower.')
        test_file_path = f.name

    file = None
    vector_store = None
    try:
        with open(test_file_path, 'rb') as f:
            file = await async_client.files.create(file=f, purpose='assistants')

        vector_store = await async_client.vector_stores.create(name='test-file-search')
        await async_client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file.id)

        await asyncio.sleep(2)

        m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=async_client))
        agent = Agent(
            m,
            instructions='You are a helpful assistant.',
            builtin_tools=[FileSearchTool(file_store_ids=[vector_store.id])],
        )

        result = await agent.run('What is the capital of France?')
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    instructions='You are a helpful assistant.',
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        BuiltinToolCallPart(
                            tool_name='file_search',
                            args={'queries': ['What is the capital of France?']},
                            tool_call_id=IsStr(),
                            provider_name='openai',
                        ),
                        BuiltinToolReturnPart(
                            tool_name='file_search',
                            content={'status': 'completed'},
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='openai',
                        ),
                        TextPart(
                            content='The capital of France is Paris.',
                            id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=870, output_tokens=30, details={'reasoning_tokens': 0}),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
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
                    instructions='You are a helpful assistant.',
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        BuiltinToolCallPart(
                            tool_name='file_search',
                            args={'queries': ['Eiffel Tower']},
                            tool_call_id=IsStr(),
                            provider_name='openai',
                        ),
                        BuiltinToolReturnPart(
                            tool_name='file_search',
                            content={'status': 'completed'},
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='openai',
                        ),
                        TextPart(
                            content='The Eiffel Tower is a famous landmark in Paris, the capital of France. It is widely recognized and serves as an iconic symbol of the city.',
                            id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=1188, output_tokens=55, details={'reasoning_tokens': 0}),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    finally:
        os.unlink(test_file_path)
        await cleanup_openai_resources(file, vector_store, async_client)


def test_map_file_search_tool_call():
    from openai.types.responses.response_file_search_tool_call import ResponseFileSearchToolCall

    from pydantic_ai.models.openai.responses import _map_file_search_tool_call  # pyright: ignore[reportPrivateUsage]

    item = ResponseFileSearchToolCall.model_validate(
        {
            'id': 'test-id',
            'queries': ['test query'],
            'status': 'completed',
            'results': [
                {
                    'id': 'result-1',
                    'title': 'Test Result',
                    'url': 'https://example.com',
                    'score': 0.9,
                }
            ],
            'type': 'file_search_call',
        }
    )

    call_part, return_part = _map_file_search_tool_call(item, 'openai')
    assert (call_part, return_part) == snapshot(
        (
            BuiltinToolCallPart(
                tool_name='file_search',
                args={'queries': ['test query']},
                tool_call_id='test-id',
                provider_name='openai',
            ),
            BuiltinToolReturnPart(
                tool_name='file_search',
                content={
                    'status': 'completed',
                    'results': [
                        {
                            'attributes': None,
                            'file_id': None,
                            'filename': None,
                            'id': 'result-1',
                            'text': None,
                            'title': 'Test Result',
                            'url': 'https://example.com',
                            'score': 0.9,
                        }
                    ],
                },
                tool_call_id='test-id',
                timestamp=IsDatetime(),
                provider_name='openai',
            ),
        )
    )


@pytest.mark.vcr()
async def test_openai_responses_model_file_search_tool_stream(allow_model_requests: None, openai_api_key: str):
    import asyncio
    import os
    import tempfile
    from typing import Any

    from openai import AsyncOpenAI

    from pydantic_ai.builtin_tools import FileSearchTool
    from pydantic_ai.providers.openai import OpenAIProvider

    async_client = AsyncOpenAI(api_key=openai_api_key)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write('Paris is the capital of France. It is known for the Eiffel Tower.')
        test_file_path = f.name

    file = None
    vector_store = None
    try:
        with open(test_file_path, 'rb') as f:
            file = await async_client.files.create(file=f, purpose='assistants')

        vector_store = await async_client.vector_stores.create(name='test-file-search-stream')
        await async_client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file.id)

        await asyncio.sleep(2)

        m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=async_client))
        agent = Agent(
            m,
            instructions='You are a helpful assistant.',
            builtin_tools=[FileSearchTool(file_store_ids=[vector_store.id])],
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
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    instructions='You are a helpful assistant.',
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        BuiltinToolCallPart(
                            tool_name='file_search',
                            args={'queries': ['What is the capital of France?']},
                            tool_call_id=IsStr(),
                            provider_name='openai',
                        ),
                        BuiltinToolReturnPart(
                            tool_name='file_search',
                            content={'status': 'completed'},
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='openai',
                        ),
                        TextPart(
                            content='The capital of France is Paris.',
                            id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=1177, output_tokens=37, details={'reasoning_tokens': 0}),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
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
                        tool_call_id=IsStr(),
                        provider_name='openai',
                    ),
                ),
                PartDeltaEvent(
                    index=0,
                    delta=ToolCallPartDelta(
                        args_delta={'queries': ['What is the capital of France?']},
                        tool_call_id=IsStr(),
                    ),
                ),
                PartEndEvent(
                    index=0,
                    part=BuiltinToolCallPart(
                        tool_name='file_search',
                        args={'queries': ['What is the capital of France?']},
                        tool_call_id=IsStr(),
                        provider_name='openai',
                    ),
                    next_part_kind='builtin-tool-return',
                ),
                PartStartEvent(
                    index=1,
                    part=BuiltinToolReturnPart(
                        tool_name='file_search',
                        content={'status': 'completed'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    previous_part_kind='builtin-tool-call',
                ),
                PartStartEvent(
                    index=2,
                    part=TextPart(content='The', id=IsStr()),
                    previous_part_kind='builtin-tool-return',
                ),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' capital')),
                PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' of')),
                PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' France')),
                PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' is')),
                PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' Paris')),
                PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='.')),
                PartEndEvent(
                    index=2,
                    part=TextPart(
                        content='The capital of France is Paris.',
                        id=IsStr(),
                    ),
                ),
                BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                    part=BuiltinToolCallPart(
                        tool_name='file_search',
                        args={'queries': ['What is the capital of France?']},
                        tool_call_id=IsStr(),
                        provider_name='openai',
                    )
                ),
                BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                    result=BuiltinToolReturnPart(
                        tool_name='file_search',
                        content={'status': 'completed'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    )
                ),
            ]
        )

    finally:
        os.unlink(test_file_path)
        await cleanup_openai_resources(file, vector_store, async_client)


@pytest.mark.vcr()
async def test_openai_responses_model_file_search_tool_with_results(allow_model_requests: None, openai_api_key: str):
    """Test that openai_include_file_search_results setting includes file search results in the response."""
    import asyncio
    import os
    import tempfile

    from openai import AsyncOpenAI

    from pydantic_ai.builtin_tools import FileSearchTool
    from pydantic_ai.providers.openai import OpenAIProvider

    async_client = AsyncOpenAI(api_key=openai_api_key)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write('Paris is the capital of France. It is known for the Eiffel Tower.')
        test_file_path = f.name

    file = None
    vector_store = None
    try:
        with open(test_file_path, 'rb') as f:
            file = await async_client.files.create(file=f, purpose='assistants')

        vector_store = await async_client.vector_stores.create(name='test-file-search-with-results')
        await async_client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=file.id)

        await asyncio.sleep(2)

        m = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=async_client))
        agent = Agent(
            m,
            instructions='You are a helpful assistant.',
            builtin_tools=[FileSearchTool(file_store_ids=[vector_store.id])],
        )

        result = await agent.run(
            'What is the capital of France?',
            model_settings=OpenAIResponsesModelSettings(openai_include_file_search_results=True),
        )

        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    instructions='You are a helpful assistant.',
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        BuiltinToolCallPart(
                            tool_name='file_search',
                            args={'queries': ['What is the capital of France?']},
                            tool_call_id=IsStr(),
                            provider_name='openai',
                        ),
                        BuiltinToolReturnPart(
                            tool_name='file_search',
                            content={
                                'status': 'completed',
                                'results': [
                                    {
                                        'attributes': {},
                                        'file_id': IsStr(),
                                        'filename': IsStr(),
                                        'score': IsFloat(),
                                        'text': 'Paris is the capital of France. It is known for the Eiffel Tower.',
                                        'vector_store_id': IsStr(),
                                    }
                                ],
                            },
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                            provider_name='openai',
                        ),
                        TextPart(
                            content=IsStr(),
                            id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt(), details={'reasoning_tokens': 0}),
                    model_name='gpt-4o-2024-08-06',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                    provider_url='https://api.openai.com/v1/',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id=IsStr(),
                    finish_reason='stop',
                    run_id=IsStr(),
                ),
            ]
        )

    finally:
        os.unlink(test_file_path)
        await cleanup_openai_resources(file, vector_store, async_client)


async def test_openai_responses_runs_with_instructions_only(
    allow_model_requests: None,
    openai_api_key: str,
):
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, instructions='Generate a short article about artificial intelligence in 3 sentences.')

    result = await agent.run()

    assert result.output
    assert isinstance(result.output, str)
    assert len(result.output) > 0
