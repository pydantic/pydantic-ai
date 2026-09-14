"""SDK-level wire checks for Anthropic thinking-block recovery transports."""

from __future__ import annotations

import json

import httpx2
import pytest

from pydantic_ai.models import ModelRequestParameters

from ..._inline_snapshot import snapshot
from ...conftest import try_import

with try_import() as anthropic_imports_successful:
    from anthropic import AsyncAnthropicVertex

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_thinking_block_binding import recovered_thinking_history

pytestmark = [
    pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]

_THINKING_BINDING_BETA = 'thinking-binding-controls-2026-08-01'


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
        {'block_binding': {'prefix_mismatch_behavior': 'drop_block'}}
    )
