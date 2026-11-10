"""Tests for what an Anthropic fast mode (`anthropic_speed='fast'`) response records.

Anthropic reports the speed each request ran at in `usage.speed`. A fast-mode response records it in
`provider_details['speed']`, which `ModelResponse.cost` passes to genai-prices as the price context, so
its `speed: fast` price variants apply. Every other response reports `usage.speed: 'standard'`, which
matches no variant, and its `provider_details` stay as they were.
"""

from __future__ import annotations as _annotations

from typing import Literal

import pytest

from pydantic_ai import Agent, ModelResponse

from ...conftest import RequestCapture, try_import
from ..conftest import AnthropicModelFactory

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModelSettings

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]


@pytest.mark.vcr
@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
@pytest.mark.parametrize('speed', ['fast', None], ids=['fast', 'standard'])
async def test_fast_mode_provider_details(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
    speed: Literal['fast'] | None,
    stream: bool,
):
    """A fast-mode response records `speed: 'fast'`, streamed or not; a standard one records no speed."""
    # An explicit `max_tokens` keeps the non-streaming request from being streamed behind the scenes.
    settings = AnthropicModelSettings(max_tokens=256)
    if speed is not None:
        settings['anthropic_speed'] = speed
    agent = Agent(anthropic_model('claude-fable-5-2', capture=True), model_settings=settings)

    if stream:
        async with agent.run_stream('Reply with exactly: OK') as streamed:
            await streamed.get_output()
        response = streamed.all_messages()[-1]
    else:
        result = await agent.run('Reply with exactly: OK')
        response = result.all_messages()[-1]

    assert request_capture.body('/v1/messages').get('speed') == speed
    assert isinstance(response, ModelResponse)
    assert response.provider_details is not None
    assert response.provider_details.get('speed') == speed
