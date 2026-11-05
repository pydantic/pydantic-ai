"""Regression tests for Anthropic binary content retries."""

from __future__ import annotations

import base64
import json
from contextlib import nullcontext
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
from pydantic_ai.native_tools import CodeExecutionTool

from ...conftest import try_import

with try_import() as anthropic_imports_successful:
    from anthropic import AsyncAnthropic

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicStaleThinkingBlockWarning
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed'),
]

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


@pytest.mark.parametrize(
    ('error_status', 'error_body', 'binary_content', 'has_code_execution', 'has_thinking_part'),
    [
        pytest.param(
            400,
            {
                'type': 'error',
                'error': {'type': 'invalid_request_error', 'message': _STALE_THINKING_BLOCK_MESSAGE},
            },
            BinaryContent(data=_IMAGE_BYTES, media_type='image/png'),
            False,
            True,
            id='stale-thinking-image',
        ),
        pytest.param(
            404,
            {
                'type': 'error',
                'error': {'type': 'not_found_error', 'message': 'Container not found: container_from_history'},
            },
            BinaryContent(data=_IMAGE_BYTES, media_type='image/png'),
            True,
            False,
            id='expired-container-image',
        ),
        pytest.param(
            400,
            {
                'type': 'error',
                'error': {'type': 'invalid_request_error', 'message': _STALE_THINKING_BLOCK_MESSAGE},
            },
            BinaryContent(data=_PDF_BYTES, media_type='application/pdf'),
            False,
            True,
            id='stale-thinking-pdf',
        ),
    ],
)
async def test_binary_content_retry_resends_identical_bytes(
    allow_model_requests: None,
    error_status: int,
    error_body: dict[str, Any],
    binary_content: BinaryContent,
    has_code_execution: bool,
    has_thinking_part: bool,
):
    """Retry recovery re-encodes binary history content instead of replaying a drained stream."""
    bodies: list[dict[str, Any]] = []
    expected_data = base64.b64encode(binary_content.data).decode()

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx2.Response(error_status, json=error_body)
        return httpx2.Response(200, json=_message_response())

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        client = AsyncAnthropic(
            api_key='test',
            base_url='https://example.com',
            http_client=http_client,
            max_retries=0,
        )
        model = AnthropicModel('claude-fable-5-1', provider=AnthropicProvider(anthropic_client=client))
        capabilities = (
            [NativeTool(CodeExecutionTool(files=[UploadedFile(file_id='file_x', provider_name='anthropic')]))]
            if has_code_execution
            else None
        )
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=['Describe this content.', binary_content])]),
            ModelResponse(
                parts=(
                    [
                        ThinkingPart(content='reasoning', signature='signature', provider_name='anthropic'),
                        TextPart(content='Working on it.'),
                    ]
                    if has_thinking_part
                    else [TextPart(content='Working on it.')]
                ),
                provider_name='anthropic',
                provider_details=None if has_thinking_part else {'container_id': 'container_from_history'},
            ),
        ]
        warning_context = pytest.warns(AnthropicStaleThinkingBlockWarning) if has_thinking_part else nullcontext()

        with warning_context:
            result = await Agent(model, capabilities=capabilities).run(
                'And again?', message_history=history, model_settings={'max_tokens': 1024}
            )

    assert result.output == 'ok'
    assert len(bodies) == 2
    assert [_binary_source_datas(body) for body in bodies] == [[expected_data], [expected_data]]
