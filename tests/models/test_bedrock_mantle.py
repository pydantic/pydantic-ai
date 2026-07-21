from __future__ import annotations

import os

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, BinaryImage, RequestUsage, UserError
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import infer_model
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.output import NativeOutput

from ..conftest import IsDatetime, IsStr, try_import

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
        # Pin the full non-streaming message shape (reasoning parts, part ids, response-qualified
        # tool-call ids) so a regression in Mantle GPT-5.x reasoning/tool-call handling can't slip
        # past the id-uniqueness assertions above.
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Follow the tool instructions.', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    instructions='Call first_tool. After receiving its result, call second_tool in a new model response. After receiving that result, answer with both results. Never call both tools in one response.',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='first_tool',
                            args='{}',
                            tool_call_id='resp_43amfn3g3uar3i4sa5b7sz35cukuufaisldkwel5v6o47xzpr5va:call_0',
                            id='fc_ccdc17fdac4f5c7e863e8d8fd3812a13',
                            provider_name='bedrock-mantle',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=88, cache_write_tokens=86, output_tokens=14, details={'reasoning_tokens': 0}
                    ),
                    model_name='openai.gpt-5.6-luna',
                    timestamp=IsDatetime(),
                    provider_name='bedrock-mantle',
                    provider_url='https://bedrock-mantle.us-east-1.api.aws/openai/v1/',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id='resp_43amfn3g3uar3i4sa5b7sz35cukuufaisldkwel5v6o47xzpr5va',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_tool',
                            content='first result',
                            tool_call_id='resp_43amfn3g3uar3i4sa5b7sz35cukuufaisldkwel5v6o47xzpr5va:call_0',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    instructions='Call first_tool. After receiving its result, call second_tool in a new model response. After receiving that result, answer with both results. Never call both tools in one response.',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='second_tool',
                            args='{}',
                            tool_call_id='resp_cizhrpixixp4ylkykezegmnkbgixq2qteaqq2gigeu5aluj6dhva:call_0',
                            id='fc_a0cc01696625559fbb1e15fde9c518f5',
                            provider_name='bedrock-mantle',
                        )
                    ],
                    usage=RequestUsage(
                        input_tokens=115, cache_write_tokens=27, output_tokens=14, details={'reasoning_tokens': 0}
                    ),
                    model_name='openai.gpt-5.6-luna',
                    timestamp=IsDatetime(),
                    provider_name='bedrock-mantle',
                    provider_url='https://bedrock-mantle.us-east-1.api.aws/openai/v1/',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id='resp_cizhrpixixp4ylkykezegmnkbgixq2qteaqq2gigeu5aluj6dhva',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='second_tool',
                            content='second result',
                            tool_call_id='resp_cizhrpixixp4ylkykezegmnkbgixq2qteaqq2gigeu5aluj6dhva:call_0',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    instructions='Call first_tool. After receiving its result, call second_tool in a new model response. After receiving that result, answer with both results. Never call both tools in one response.',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='rs_5204a68ecc8b5dd2a32bce236c06c18d',
                            signature=IsStr(),
                            provider_name='bedrock-mantle',
                        ),
                        TextPart(
                            content="""\
First tool result: `first result`

Second tool result: `second result`\
""",
                            id='msg_0d2ae1802b2b5d25b45dbe411f126a01',
                            provider_name='bedrock-mantle',
                            provider_details={'phase': 'final_answer'},
                        ),
                    ],
                    usage=RequestUsage(
                        input_tokens=142, cache_write_tokens=27, output_tokens=33, details={'reasoning_tokens': 11}
                    ),
                    model_name='openai.gpt-5.6-luna',
                    timestamp=IsDatetime(),
                    provider_name='bedrock-mantle',
                    provider_url='https://bedrock-mantle.us-east-1.api.aws/openai/v1/',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id='resp_m2q7figv7bk4ec5owiamz4wtafpld75shl3qo4mkcnrzfeghzetq',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )
        # Non-streaming Mantle reuses the raw `call_0` id across separate responses; the qualified ids
        # stay unique, and replaying the full (normalized) history back to Mantle succeeds.
        assert all(call.tool_call_id.endswith(':call_0') for _, call in tool_calls)
        replay_result = await Agent(model).run('Reply with exactly OK.', message_history=messages)
        assert replay_result.output == 'OK'


async def test_reused_tool_call_ids_gpt_5_5(allow_model_requests: None) -> None:
    """The Responses ID reset is a property of the `/openai/v1` endpoint, not just gpt-5.6.

    gpt-5.5 resets its tool-call IDs to `call_0` across separate responses too, so pydantic-ai must
    re-qualify them here as well — this covers the broadened response-scoped gate on a non-5.6 model.
    """
    model = infer_model('bedrock-mantle:openai.gpt-5.5', lambda _: _provider())
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
    # gpt-5.5 reuses the raw `call_0` id across separate responses (like gpt-5.6); the qualified ids
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


async def test_native_output(allow_model_requests: None) -> None:
    """Mantle inherits `supports_json_schema_output` from the direct-OpenAI profile; verify it live.

    Inheritance isn't automatically accurate for Mantle — `supports_image_output` had to be overridden to
    `False` per the AWS model cards — so the structured-output claim is confirmed against the real endpoint
    rather than trusted.
    """

    class City(BaseModel):
        city: str
        country: str

    model = infer_model('bedrock-mantle:openai.gpt-5.6-luna', lambda _: _provider())
    agent = Agent(model, output_type=NativeOutput(City))

    result = await agent.run('The Eiffel Tower is in which city and country?')

    assert result.output == City(city='Paris', country='France')


async def test_image_output_unsupported(allow_model_requests: None) -> None:
    """Mantle disables image output (per the AWS model cards), so requesting it fails with a clean
    `UserError` before any request rather than an opaque provider error. No cassette: the profile guard
    raises during request preparation.
    """
    model = infer_model('bedrock-mantle:openai.gpt-5.6-luna', lambda _: _provider())
    agent = Agent(model, output_type=BinaryImage)

    with pytest.raises(UserError, match='Image output is not supported by this model'):
        await agent.run('Draw a kiwi.')


async def test_native_tool_unsupported(allow_model_requests: None) -> None:
    """Mantle proxies the OpenAI models but not OpenAI's server-hosted tools, so a native tool fails with a
    clean `UserError` before any request. No cassette: the profile guard raises during request preparation.
    """
    model = infer_model('bedrock-mantle:openai.gpt-5.6-luna', lambda _: _provider())
    agent = Agent(model, capabilities=[NativeTool(WebSearchTool())])

    with pytest.raises(UserError, match=r"Native tool\(s\) \['WebSearchTool'\] not supported by this model"):
        await agent.run('What is the weather in Paris?')
