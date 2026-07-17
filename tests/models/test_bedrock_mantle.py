from __future__ import annotations

import os

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import infer_model

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
]


def _provider() -> BedrockMantleProvider:
    return BedrockMantleProvider(region_name='us-east-1', api_key=os.getenv('AWS_BEARER_TOKEN_BEDROCK', 'mock-api-key'))


@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
async def test_reused_tool_call_ids(stream: bool, allow_model_requests: None) -> None:
    """Mantle GPT-5.6 resets Responses tool-call IDs per response; pydantic-ai must re-qualify them."""
    model = infer_model('bedrock-mantle:openai.gpt-5.6-luna', lambda _: _provider())
    agent = Agent(
        model,
        instructions=(
            'Call first_tool. After receiving its result, call second_tool in a new model response. '
            'After receiving that result, answer with both results. Never call both tools in one response.'
        ),
    )

    @agent.tool_plain
    def first_tool() -> str:
        return 'first result'

    @agent.tool_plain
    def second_tool() -> str:
        return 'second result'

    if stream:
        async with agent.run_stream('Follow the tool instructions.') as result:
            await result.get_output()
            messages = result.all_messages()
    else:
        result = await agent.run('Follow the tool instructions.')
        messages = result.all_messages()

    tool_calls = [
        (message.provider_response_id, tool_call_part)
        for message in messages
        if isinstance(message, ModelResponse)
        for tool_call_part in message.tool_calls
    ]
    assert [call.tool_name for _, call in tool_calls] == ['first_tool', 'second_tool']
    assert len({call.tool_call_id for _, call in tool_calls}) == len(tool_calls)
    assert all(call.tool_call_id.startswith(f'{response_id}:') for response_id, call in tool_calls)

    if not stream:
        # Non-streaming Mantle reuses the raw `call_0` id across separate responses; the qualified ids
        # stay unique, and replaying the full (normalized) history back to Mantle succeeds.
        assert all(call.tool_call_id.endswith(':call_0') for _, call in tool_calls)
        replay_result = await Agent(model).run('Reply with exactly OK.', message_history=messages)
        assert replay_result.output == 'OK'


async def test_gpt_oss_responses(allow_model_requests: None) -> None:
    """GPT-OSS is served on the Responses API at `/v1/responses` (not GPT-5.x's `/openai/v1`)."""
    model = infer_model('bedrock-mantle:openai.gpt-oss-120b', lambda _: _provider())

    result = await Agent(model).run('Reply with exactly OSS.')

    assert result.output == 'OSS'


async def test_safeguard_chat_routing(allow_model_requests: None) -> None:
    """GPT-OSS Safeguard is served on the Chat Completions endpoint, not Responses."""
    model = infer_model('bedrock-mantle:openai.gpt-oss-safeguard-20b', lambda _: _provider())

    result = await Agent(model).run('Reply with exactly SAFE.')

    assert result.output == 'SAFE'
