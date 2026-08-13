from __future__ import annotations as _annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from decimal import Decimal
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import Capability
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.deepseek import DeepSeekProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.fixture
def freeze_deepseek_off_peak_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make DeepSeek's time-of-day pricing deterministic for snapshot assertions."""
    timestamp = datetime(2026, 8, 13, tzinfo=timezone.utc)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return timestamp if tz is not None else timestamp.replace(tzinfo=None)

    monkeypatch.setattr('pydantic_ai._utils.datetime', FrozenDatetime)


@pytest.mark.moves_cache_prefix(reason='dynamic tool disclosure after ToolSearch discovery')
async def test_deepseek_deferred_capability_with_thinking(allow_model_requests: None, deepseek_api_key: str):
    """Regression test for #5829: real-API check that deferred capabilities work on a DeepSeek thinking model.

    Loading a deferred capability injects a framework-synthesized `search_tools` assistant turn with
    tool calls but no thinking; before the fix DeepSeek rejected it with a 400. A successful
    recording confirms DeepSeek accepts the empty `reasoning_content` the fix sends. The
    deterministic mapping guard is in
    `test_openai.py::test_field_mode_thinking_backfill_on_synthetic_tool_search_turn`.
    """
    model = OpenAIChatModel('deepseek-v4-flash', provider=DeepSeekProvider(api_key=deepseek_api_key))

    def roll_dice() -> str:
        """Roll a six-sided die and return the result."""
        return '4'

    def get_player_name(ctx: RunContext[str]) -> str:
        """Get the player's name."""
        return ctx.deps

    agent = Agent(
        model,
        deps_type=str,
        instructions=(
            "You're a dice game, you should roll the die and see if the number you get back "
            "matches the user's guess. If so, tell them they're a winner. Use the player's name "
            'in the response.'
        ),
        capabilities=[Capability[str](id='DICE_ROLL', tools=[get_player_name, roll_dice], defer_loading=True)],
    )

    result = await agent.run('My guess is 4', deps='Anne')

    # The run completing at all is the core regression signal — it 400'd before the fix. The
    # structural checks make sure the recording exercised the deferred + thinking path rather than
    # the model answering directly (which would leave the bug untested).
    assert isinstance(result.output, str) and result.output
    messages = result.all_messages()
    assert any(
        isinstance(part, ToolCallPart) and part.tool_name == 'load_capability'
        for message in messages
        for part in message.parts
    ), 'expected the model to call `load_capability`; the deferred path was not exercised'
    assert any(isinstance(part, ThinkingPart) for message in messages for part in message.parts), (
        'expected a `ThinkingPart`; thinking was not exercised, so the reasoning_content round-trip is untested'
    )


async def test_deepseek_model_thinking_part(
    allow_model_requests: None, deepseek_api_key: str, freeze_deepseek_off_peak_pricing: None
):
    deepseek_model = OpenAIChatModel('deepseek-v4-flash', provider=DeepSeekProvider(api_key=deepseek_api_key))
    agent = Agent(model=deepseek_model)
    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='deepseek'),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=90,
                    output_tokens=166,
                    details={
                        'prompt_cache_hit_tokens': 0,
                        'prompt_cache_miss_tokens': 90,
                        'reasoning_tokens': 59,
                    },
                    cost=Decimal('0.00005908'),
                ),
                model_name='deepseek-v4-flash',
                provider_name='deepseek',
                provider_url='https://api.deepseek.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='32beb152-2946-409a-8986-7f0e1e351ed2',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_deepseek_model_thinking_stream(
    allow_model_requests: None, deepseek_api_key: str, freeze_deepseek_off_peak_pricing: None
):
    deepseek_model = OpenAIChatModel('deepseek-v4-flash', provider=DeepSeekProvider(api_key=deepseek_api_key))
    agent = Agent(model=deepseek_model)

    result: AgentRunResult | None = None
    async with agent.run_stream_events(user_prompt='How do I cross the street?') as event_stream:
        async for event in event_stream:
            if isinstance(event, AgentRunResultEvent):
                result = event.result

    assert result is not None
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='reasoning_content',
                        provider_name='deepseek',
                    ),
                    TextPart(
                        content="""\
Here's how to cross the street safely:

1. **Find a safe place** – Use a crosswalk, intersection, or traffic signal if available.
2. **Stop at the curb/edge** – Don't step into the road until you're ready.
3. **Look left, right, then left again** – Check for cars, bikes, and motorcycles in all directions.
4. **Wait for the signal** – If there's a pedestrian light, wait for the "Walk" symbol. Always make sure traffic has actually stopped.
5. **Make eye contact with drivers** – If possible, confirm they see you before stepping out.
6. **Stay alert** – Don't use your phone or wear headphones. Keep looking as you cross.
7. **Keep walking at a steady pace** – Don't run unless necessary, but don't linger.

If there's no crosswalk, choose a spot with good visibility, wait for a gap in traffic, and cross straight across—not diagonally.\
"""
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=90,
                    output_tokens=303,
                    details={'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 90, 'reasoning_tokens': 89},
                    cost=Decimal('0.00009744'),
                ),
                model_name='deepseek-v4-flash',
                provider_name='deepseek',
                provider_url='https://api.deepseek.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='fa23191e-e1bf-4fdc-9ba2-c33d951d7e32',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


@dataclass(frozen=True)
class ToolChoiceCase:
    id: str
    model_name: str
    thinking_disabled: bool
    expected_tool_choice: str
    expected_reasoning_effort: str | None


TOOL_CHOICE_CASES = [
    ToolChoiceCase(
        id='thinking_on_downgrades',
        model_name='deepseek-v4-flash',
        thinking_disabled=False,
        expected_tool_choice='auto',
        expected_reasoning_effort=None,
    ),
    ToolChoiceCase(
        id='thinking_off_forces',
        model_name='deepseek-v4-flash',
        thinking_disabled=True,
        expected_tool_choice='required',
        expected_reasoning_effort='none',
    ),
    ToolChoiceCase(
        id='thinking_off_forces_pro',
        model_name='deepseek-v4-pro',
        thinking_disabled=True,
        expected_tool_choice='required',
        expected_reasoning_effort='none',
    ),
]


class Answer(BaseModel):
    text: str


@pytest.mark.parametrize('case', TOOL_CHOICE_CASES, ids=lambda c: c.id)
async def test_deepseek_tool_choice_follows_thinking(
    case: ToolChoiceCase, allow_model_requests: None, deepseek_api_key: str
):
    """DeepSeek rejects a forced tool choice only while thinking is on, so the downgrade is per request.

    The wire assertion is what matters here: with thinking on, `tool_choice` must stay `'auto'` or the
    API answers `Thinking mode does not support this tool_choice`; with thinking off, forcing must
    survive, which is what makes tool-based structured output reliable on `deepseek-v4-pro`.
    """
    sent_bodies: list[dict[str, Any]] = []

    async def capture_request(request: httpx.Request) -> None:
        sent_bodies.append(json.loads(request.read()))

    http_client = httpx.AsyncClient(event_hooks={'request': [capture_request]})
    model = OpenAIChatModel(
        case.model_name, provider=DeepSeekProvider(api_key=deepseek_api_key, http_client=http_client)
    )
    settings = OpenAIChatModelSettings(thinking=False) if case.thinking_disabled else None
    agent = Agent(model, output_type=Answer, model_settings=settings)

    result = await agent.run('Say hi.')

    assert isinstance(result.output, Answer)
    assert sent_bodies[0]['tool_choice'] == case.expected_tool_choice
    assert sent_bodies[0].get('reasoning_effort') == case.expected_reasoning_effort
