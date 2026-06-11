from __future__ import annotations as _annotations

from collections.abc import Callable
from datetime import datetime, timezone

import pytest

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
    from pydantic_ai.models.openai import OpenAIChatModel

    DeepSeekModelFactory = Callable[..., OpenAIChatModel]


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_deepseek_deferred_capability_with_thinking(
    allow_model_requests: None, deepseek_model: DeepSeekModelFactory
):
    """Regression test for #5829: real-API check that deferred capabilities work on a DeepSeek thinking model.

    Loading a deferred capability injects a framework-synthesized `search_tools` assistant turn with
    tool calls but no thinking; before the fix DeepSeek rejected it with a 400. A successful
    recording confirms DeepSeek accepts the empty `reasoning_content` the fix sends. The
    deterministic mapping guard is in
    `test_openai.py::test_field_mode_thinking_backfill_on_synthetic_tool_search_turn`.
    """
    model = deepseek_model('deepseek-reasoner')

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


async def test_deepseek_model_thinking_part(allow_model_requests: None, deepseek_model: DeepSeekModelFactory):
    agent = Agent(model=deepseek_model('deepseek-reasoner'))
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
                    input_tokens=12,
                    output_tokens=789,
                    details={
                        'prompt_cache_hit_tokens': 0,
                        'prompt_cache_miss_tokens': 12,
                        'reasoning_tokens': 415,
                    },
                ),
                model_name='deepseek-reasoner',
                timestamp=IsDatetime(),
                provider_name='deepseek',
                provider_url='https://api.deepseek.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 4, 22, 14, 9, 11, tzinfo=timezone.utc),
                },
                provider_response_id='181d9669-2b3a-445e-bd13-2ebff2c378f6',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_deepseek_model_thinking_stream(allow_model_requests: None, deepseek_model: DeepSeekModelFactory):
    agent = Agent(model=deepseek_model('deepseek-reasoner'))

    result: AgentRunResult | None = None
    async for event in agent.run_stream_events(user_prompt='How do I cross the street?'):
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
                    TextPart(content='Hello there! 😊 How can I help you today?'),
                ],
                usage=RequestUsage(
                    input_tokens=6,
                    output_tokens=212,
                    details={'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 6, 'reasoning_tokens': 198},
                ),
                model_name='deepseek-reasoner',
                timestamp=IsDatetime(),
                provider_name='deepseek',
                provider_url='https://api.deepseek.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': datetime(2025, 7, 10, 17, 41, 44, tzinfo=timezone.utc),
                },
                provider_response_id='33be18fc-3842-486c-8c29-dd8e578f7f20',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
