"""Tests for what an Anthropic fast mode (`anthropic_speed='fast'`) response records.

Anthropic reports the speed each request ran at in `usage.speed`. A fast-mode response records it in
`provider_details['speed']`, which `ModelResponse.cost` passes to genai-prices as the price context, so
its `speed: fast` price variants apply. Every other response reports `usage.speed: 'standard'`, which
matches no variant, and its `provider_details` stay as they were.

All four cassettes recorded the same usage (18 input, 17 output tokens), so the fast cases cost twice the
standard ones: Claude Fable 5.2 costs $16/$80 per MTok in fast mode and $8/$40 otherwise.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import pytest

from pydantic_ai import Agent, ModelResponse

from ..._inline_snapshot import snapshot
from ...conftest import RequestCapture, try_import
from ..conftest import AnthropicModelFactory

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModelSettings

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]


@dataclass(frozen=True)
class Case:
    id: str
    speed: Literal['fast'] | None
    stream: bool
    provider_details: dict[str, str]
    total_price: Decimal


CASES = [
    Case(
        id='fast-request',
        speed='fast',
        stream=False,
        provider_details=snapshot({'finish_reason': 'end_turn', 'speed': 'fast'}),
        total_price=snapshot(Decimal('0.001648')),
    ),
    Case(
        id='fast-stream',
        speed='fast',
        stream=True,
        provider_details=snapshot({'speed': 'fast', 'finish_reason': 'end_turn'}),
        total_price=snapshot(Decimal('0.001648')),
    ),
    Case(
        id='standard-request',
        speed=None,
        stream=False,
        provider_details=snapshot({'finish_reason': 'end_turn'}),
        total_price=snapshot(Decimal('0.000824')),
    ),
    Case(
        id='standard-stream',
        speed=None,
        stream=True,
        provider_details=snapshot({'finish_reason': 'end_turn'}),
        total_price=snapshot(Decimal('0.000824')),
    ),
]


@pytest.mark.vcr
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
async def test_fast_mode_provider_details(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
    case: Case,
):
    """A fast-mode response records `speed: 'fast'` and prices at fast mode rates, streamed or not.

    A standard one records no speed and prices at standard rates.
    """
    # An explicit `max_tokens` keeps the non-streaming request from being streamed behind the scenes.
    settings = AnthropicModelSettings(max_tokens=256)
    if case.speed is not None:
        settings['anthropic_speed'] = case.speed
    agent = Agent(anthropic_model('claude-fable-5-2', capture=True), model_settings=settings)

    if case.stream:
        async with agent.run_stream('Reply with exactly: OK') as streamed:
            await streamed.get_output()
        response = streamed.all_messages()[-1]
    else:
        result = await agent.run('Reply with exactly: OK')
        response = result.all_messages()[-1]

    assert request_capture.body('/v1/messages').get('speed') == case.speed
    assert isinstance(response, ModelResponse)
    assert response.provider_details == case.provider_details
    assert response.cost().total_price == case.total_price
