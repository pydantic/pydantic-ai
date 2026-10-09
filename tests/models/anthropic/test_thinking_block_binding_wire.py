"""SDK-level wire checks for Anthropic thinking-block recovery transports."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx2
import pytest

from pydantic_ai import (
    Agent,
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.messages import UploadedFile
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import CodeExecutionTool

from ..._inline_snapshot import snapshot
from ...conftest import try_import

with try_import() as anthropic_imports_successful:
    from anthropic import AsyncAnthropic, AsyncAnthropicVertex

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicStaleThinkingBlockWarning
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_thinking_block_binding import recovered_thinking_history

pytestmark = [
    pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed'),
]

_THINKING_BINDING_BETA = 'thinking-binding-controls-2026-08-01'


_STALE_THINKING_BLOCK_MESSAGE = (
    'messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a '
    'different conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` '
    'to "drop_block". The `system` prompt differs from the one this block was created with.'
)
_IMAGE_BYTES = b'fake-png-payload-for-the-anthropic-wire-tests'
_PDF_BYTES = b'fake-pdf-payload-for-the-anthropic-wire-tests'


def _message_response() -> dict[str, Any]:
    """A minimal valid `messages.create` response body."""
    return {
        'id': 'msg_01',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-fable-5-1',
        'content': [{'type': 'text', 'text': 'ok'}],
        'stop_reason': 'end_turn',
        'stop_sequence': None,
        'usage': {'input_tokens': 10, 'output_tokens': 1},
    }


def _binary_source_datas(body: dict[str, Any]) -> list[str]:
    """The base64 payload of every image/document source in an outbound request body."""
    return [
        block['source']['data']
        for message in body['messages']
        for block in message['content']
        if block['type'] in ('image', 'document')
    ]


async def test_drop_block_retry_resends_identical_image_bytes(allow_model_requests: None):
    """The stale-thinking-block retry re-encodes history images instead of replaying a drained stream."""
    bodies: list[dict[str, Any]] = []
    expected_data = base64.b64encode(_IMAGE_BYTES).decode()

    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        bodies.append(body)
        assert all(_binary_source_datas(body)), 'request carried empty base64 source data'
        if len(bodies) == 1:
            return httpx2.Response(
                400,
                json={
                    'type': 'error',
                    'error': {'type': 'invalid_request_error', 'message': _STALE_THINKING_BLOCK_MESSAGE},
                },
            )
        return httpx2.Response(200, json=_message_response())

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        client = AsyncAnthropic(
            api_key='test',
            base_url='https://example.com',
            http_client=http_client,
            max_retries=0,
        )
        model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=client))
        history: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=['Describe this image.', BinaryContent(data=_IMAGE_BYTES, media_type='image/png')]
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='reasoning', signature='signature', provider_name='anthropic'),
                    TextPart(content='A red square.'),
                ],
                provider_name='anthropic',
            ),
        ]

        with pytest.warns(AnthropicStaleThinkingBlockWarning):
            result = await Agent(model).run('And again?', message_history=history, model_settings={'max_tokens': 1024})

    assert result.output == 'ok'
    assert len(bodies) == 2
    assert [_binary_source_datas(body) for body in bodies] == [[expected_data], [expected_data]]


async def test_expired_container_fallback_resends_identical_image_bytes(allow_model_requests: None):
    """The expired-container fallback re-encodes history images instead of replaying a drained stream."""
    bodies: list[dict[str, Any]] = []
    expected_data = base64.b64encode(_IMAGE_BYTES).decode()

    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        bodies.append(body)
        assert all(_binary_source_datas(body)), 'request carried empty base64 source data'
        if len(bodies) == 1:
            return httpx2.Response(
                404,
                json={
                    'type': 'error',
                    'error': {'type': 'not_found_error', 'message': 'Container not found: container_from_history'},
                },
            )
        return httpx2.Response(200, json=_message_response())

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        client = AsyncAnthropic(
            api_key='test',
            base_url='https://example.com',
            http_client=http_client,
            max_retries=0,
        )
        model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=client))
        agent = Agent(
            model,
            capabilities=[
                NativeTool(CodeExecutionTool(files=[UploadedFile(file_id='file_x', provider_name='anthropic')]))
            ],
        )
        history: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Summarize the uploaded file against this image.',
                            BinaryContent(data=_IMAGE_BYTES, media_type='image/png'),
                        ]
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Working on it.')],
                provider_name='anthropic',
                provider_details={'container_id': 'container_from_history'},
            ),
        ]

        result = await agent.run('And the final answer?', message_history=history, model_settings={'max_tokens': 1024})

    assert result.output == 'ok'
    assert len(bodies) == 2
    assert [_binary_source_datas(body) for body in bodies] == [[expected_data], [expected_data]]


async def test_drop_block_retry_resends_identical_pdf_bytes(allow_model_requests: None):
    """The stale-thinking-block retry re-encodes history PDFs instead of replaying a drained stream."""
    bodies: list[dict[str, Any]] = []
    expected_data = base64.b64encode(_PDF_BYTES).decode()

    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        bodies.append(body)
        assert all(_binary_source_datas(body)), 'request carried empty base64 source data'
        if len(bodies) == 1:
            return httpx2.Response(
                400,
                json={
                    'type': 'error',
                    'error': {'type': 'invalid_request_error', 'message': _STALE_THINKING_BLOCK_MESSAGE},
                },
            )
        return httpx2.Response(200, json=_message_response())

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        client = AsyncAnthropic(
            api_key='test',
            base_url='https://example.com',
            http_client=http_client,
            max_retries=0,
        )
        model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=client))
        history: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            'Summarize this document.',
                            BinaryContent(data=_PDF_BYTES, media_type='application/pdf'),
                        ]
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='reasoning', signature='signature', provider_name='anthropic'),
                    TextPart(content='It is a contract.'),
                ],
                provider_name='anthropic',
            ),
        ]

        with pytest.warns(AnthropicStaleThinkingBlockWarning):
            result = await Agent(model).run('And again?', message_history=history, model_settings={'max_tokens': 1024})

    assert result.output == 'ok'
    assert len(bodies) == 2
    assert [_binary_source_datas(body) for body in bodies] == [[expected_data], [expected_data]]


async def test_anthropic_vertex_count_tokens_sends_persisted_binding_on_the_wire(allow_model_requests: None):
    """Vertex's beta count route carries the binding header and request body unchanged."""
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json={'input_tokens': 10})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        client = AsyncAnthropicVertex(
            project_id='project', region='us-central1', access_token='token', http_client=http_client
        )
        model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=client))

        await model.count_tokens(recovered_thinking_history(), None, ModelRequestParameters())

    [request] = requests
    assert request.url.path.endswith('/publishers/anthropic/models/count-tokens:rawPredict')
    assert _THINKING_BINDING_BETA in request.headers['anthropic-beta']
    assert json.loads(request.content)['thinking'] == snapshot(
        {'type': 'adaptive', 'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}
    )
