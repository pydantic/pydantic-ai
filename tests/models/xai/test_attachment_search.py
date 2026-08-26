"""Tests for xAI's implicit attachment-search lifecycle."""

from __future__ import annotations as _annotations

from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    BinaryContent,
    FinalResultEvent,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartStartEvent,
    UserPromptPart,
)

from ..._inline_snapshot import snapshot
from ...conftest import try_import
from ..mock_xai import MockXai, create_mixed_tools_response, create_response, get_mock_chat_create_kwargs

with try_import() as imports_successful:
    from xai_sdk.proto import chat_pb2

    from pydantic_ai.models.xai import XaiModel, XaiModelSettings
    from pydantic_ai.providers.xai import XaiProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

XAI_MODEL = 'grok-4.6'


class BrowsedPage(BaseModel):
    """One page of xAI's `pdf_browse` output, without the rendered page image."""

    page_num_one_indexed: int
    text: str


class BrowsedPages(BaseModel):
    pages: list[BrowsedPage]


async def test_xai_attachment_search_with_output(
    allow_model_requests: None, document_content: BinaryContent, xai_provider: XaiProvider
):
    """Attachment-search output is exposed and accepted when replayed in history."""
    agent = Agent(
        XaiModel(XAI_MODEL, provider=xai_provider),
        model_settings=XaiModelSettings(xai_include_attachment_search_output=True),
    )

    result = await agent.run(
        [
            'Summarize every distinct claim made in the attached PDF. Do not use quotation marks.',
            document_content,
        ]
    )
    assert result.output == snapshot(
        'The attached PDF is a placeholder document containing only the phrase Dummy PDF file and makes no claims.'
    )

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)

    ((call_part, return_part),) = response.native_tool_calls
    assert call_part == snapshot(
        NativeToolCallPart(
            tool_name='attachment_search',
            args={'document_id': 'gNF6i', 'pages': '1'},
            tool_call_id='call-f2c77db6-d914-43cb-ae41-95765438ef68-0',
            provider_name='xai',
            provider_details={'function_name': 'pdf_browse'},
        )
    )

    assert return_part.tool_name == 'attachment_search'
    browsed = BrowsedPages.model_validate(return_part.content)
    assert browsed.pages == snapshot([BrowsedPage(page_num_one_indexed=1, text='Dummy PDF file\n\n')])
    assert response.usage.details == snapshot({'reasoning_tokens': 428, 'server_side_tools_attachment_search': 1})

    follow_up = await agent.run('Which page did you read that from?', message_history=result.new_messages())
    assert follow_up.output == snapshot('Page 1 of the attached PDF.')


async def test_xai_attachment_search_history_payload(allow_model_requests: None):
    """Assert history serialization directly because cassette replay does not match request bodies."""
    attachment_search_call = chat_pb2.ToolCall(
        id='attachment_001',
        type=chat_pb2.ToolCallType.TOOL_CALL_TYPE_ATTACHMENT_SEARCH_TOOL,
        status=chat_pb2.ToolCallStatus.TOOL_CALL_STATUS_COMPLETED,
        function=chat_pb2.FunctionCall(
            name='pdf_browse',
            arguments='{"document_id": "s2pXx", "pages": "1"}',
        ),
    )
    mock_client = MockXai.create_mock(
        [
            create_mixed_tools_response([attachment_search_call], text_content='Found the attachment.'),
            create_response(content='Page 1.'),
        ]
    )
    agent = Agent(XaiModel(XAI_MODEL, provider=XaiProvider(xai_client=mock_client)))

    result = await agent.run('Search my attachment')
    await agent.run('Which page?', message_history=result.new_messages())

    assert get_mock_chat_create_kwargs(mock_client)[1]['messages'][1] == snapshot(
        {
            'content': [{'text': ''}],
            'role': 'ROLE_ASSISTANT',
            'tool_calls': [
                {
                    'id': 'attachment_001',
                    'type': 'TOOL_CALL_TYPE_ATTACHMENT_SEARCH_TOOL',
                    'status': 'TOOL_CALL_STATUS_COMPLETED',
                    'function': {
                        'name': 'pdf_browse',
                        'arguments': '{"document_id":"s2pXx","pages":"1"}',
                    },
                }
            ],
        }
    )


async def test_xai_attachment_search_history_without_provider_details(allow_model_requests: None):
    """History `attachment_search` calls without `provider_details` default the outgoing function name."""
    mock_client = MockXai.create_mock([create_response(content='Page 1.')])
    agent = Agent(XaiModel(XAI_MODEL, provider=XaiProvider(xai_client=mock_client)))

    await agent.run(
        'Which page?',
        message_history=[
            ModelRequest(parts=[UserPromptPart(content='Search my attachment')]),
            ModelResponse(
                parts=[
                    NativeToolCallPart(
                        tool_name='attachment_search',
                        args={'document_id': 's2pXx', 'pages': '1'},
                        tool_call_id='attachment_002',
                        provider_name='xai',
                    )
                ],
                model_name=XAI_MODEL,
            ),
        ],
    )

    assert get_mock_chat_create_kwargs(mock_client)[0]['messages'][1] == snapshot(
        {
            'content': [{'text': ''}],
            'role': 'ROLE_ASSISTANT',
            'tool_calls': [
                {
                    'id': 'attachment_002',
                    'type': 'TOOL_CALL_TYPE_ATTACHMENT_SEARCH_TOOL',
                    'status': 'TOOL_CALL_STATUS_COMPLETED',
                    'function': {
                        'name': 'attachment_search',
                        'arguments': '{"document_id":"s2pXx","pages":"1"}',
                    },
                }
            ],
        }
    )


async def test_xai_attachment_search_stream(
    allow_model_requests: None, document_content: BinaryContent, xai_provider: XaiProvider
):
    """Streaming emits the attachment-search lifecycle returned by xAI."""
    agent = Agent(
        XaiModel(XAI_MODEL, provider=xai_provider),
        model_settings=XaiModelSettings(xai_include_attachment_search_output=True),
    )

    events: list[Any] = []
    async with agent.iter(
        user_prompt=[
            'Summarize every distinct claim made in the attached PDF. Do not use quotation marks.',
            document_content,
        ]
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        events.append(event)

    assert agent_run.result is not None
    result = agent_run.result
    assert result.output == snapshot(
        'The attached PDF is a placeholder document containing no assertions or claims of any kind.'
    )

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    ((call_part, return_part),) = response.native_tool_calls
    assert call_part.tool_name == 'attachment_search'
    assert call_part.provider_details == {'function_name': 'pdf_browse'}
    assert return_part.tool_name == 'attachment_search'
    browsed = BrowsedPages.model_validate(return_part.content)
    assert browsed.pages == snapshot([BrowsedPage(page_num_one_indexed=1, text='Dummy PDF file\n\n')])
    assert response.usage.details == snapshot({'reasoning_tokens': 402, 'server_side_tools_attachment_search': 1})

    assert [
        (event.part.part_kind, event.part.tool_name)
        for event in events
        if isinstance(event, PartStartEvent) and isinstance(event.part, (NativeToolCallPart, NativeToolReturnPart))
    ] == snapshot([('builtin-tool-call', 'attachment_search'), ('builtin-tool-return', 'attachment_search')])
    assert any(isinstance(event, FinalResultEvent) for event in events)
