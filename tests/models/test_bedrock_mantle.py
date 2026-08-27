from __future__ import annotations

import os
from decimal import Decimal

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
    from openai.types import chat

    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
]


def _provider() -> BedrockMantleProvider:
    return BedrockMantleProvider(region_name='us-east-1', api_key=os.getenv('AWS_BEARER_TOKEN_BEDROCK', 'mock-api-key'))


@pytest.mark.parametrize('stream', [False, True], ids=['request', 'stream'])
@pytest.mark.moves_cache_prefix(reason='replay uses a fresh agent without the original instructions and tools')
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
                        input_tokens=88,
                        cache_write_tokens=86,
                        output_tokens=14,
                        output_reasoning_tokens=0,
                        details={'reasoning_tokens': 0},
                        cost=Decimal('0.00004257'),
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
                        input_tokens=115,
                        cache_write_tokens=27,
                        output_tokens=14,
                        output_reasoning_tokens=0,
                        details={'reasoning_tokens': 0},
                        cost=Decimal('0.000045265'),
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
                        input_tokens=142,
                        cache_write_tokens=27,
                        output_tokens=33,
                        output_reasoning_tokens=11,
                        details={'reasoning_tokens': 11},
                        cost=Decimal('0.000076285'),
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


@pytest.mark.moves_cache_prefix(reason='replay uses a fresh agent without the original instructions and tools')
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
    """Mantle's Lambda/MCP server-side tools differ from Pydantic AI's OpenAI-native tools, so a native
    tool fails with a clean `UserError` before any request. No cassette: the profile guard raises during
    request preparation.
    """
    model = infer_model('bedrock-mantle:openai.gpt-5.6-luna', lambda _: _provider())
    agent = Agent(model, capabilities=[NativeTool(WebSearchTool())])

    with pytest.raises(UserError, match=r"Native tool\(s\) \['WebSearchTool'\] not supported by this model"):
        await agent.run('What is the weather in Paris?')


# ---------------------------------------------------------------------------
# Bedrock-native finish_reason handling (#7816)
#
# `BedrockMantleChatModel` rides the generic OpenAI re-validation path, whose strict
# 5-value `finish_reason` Literal rejects platform-native values such as
# `guardrail_intervened` (litellm documents that value as Bedrock-specific in
# BerriAI/litellm#22138). The tests below pin the same contract set the `ZaiModel`
# fix (#7685) established:
#
#   * widened validation accepts the non-standard literal (raw value survives),
#   * normalisation maps it onto the standard `FinishReason` enum,
#   * the un-normalised raw string is preserved for forensics in
#     `provider_details['finish_reason']`,
#   * both non-streaming and streaming paths behave identically,
#   * genuinely malformed payloads still fail loudly as `UnexpectedModelBehavior`.
#
# Setup follows the mock-construct pattern the repo uses for provider-level override
# tests (see `test_zai.py` / `test_openrouter.py`): `.model_construct()` bypasses
# Pydantic field validation because the strict public SDK constructors would reject
# the dialect values before pydantic-ai sees them.
# ---------------------------------------------------------------------------


def _mantle_chat_model():
    from openai import AsyncOpenAI

    from pydantic_ai.models.bedrock_mantle import BedrockMantleChatModel
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider

    provider = BedrockMantleProvider(openai_client=AsyncOpenAI(api_key='offline', base_url='http://127.0.0.1:9/v1'))
    return BedrockMantleChatModel('openai.gpt-oss-safeguard-120b', provider=provider)


@pytest.mark.parametrize(
    ('raw_finish_reason', 'mapped_finish_reason'),
    [
        ('guardrail_intervened', 'content_filter'),
    ],
)
def test_bedrock_mantle_nonstd_finish_reason_nonstream(raw_finish_reason: str, mapped_finish_reason: str) -> None:
    """Non-streaming path: Bedrock-native finish reasons survive widened validation."""
    from typing import cast

    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.messages import FinishReason

    msg = ChatCompletionMessage(role='assistant', content='blocked')
    bad_choice = Choice.model_construct(finish_reason=raw_finish_reason, index=0, message=msg)
    bad_completion = chat.ChatCompletion.model_construct(
        id='123',
        choices=[bad_choice],
        created=1704067200,  # 2024-01-01
        model='openai.gpt-oss-safeguard-120b',
        object='chat.completion',
    )

    model = _mantle_chat_model()

    # 1. Widened validation accepts the non-standard literal without raising.
    validated = model._validate_completion(bad_completion)  # type: ignore[reportPrivateUsage]
    v_choice = validated.choices[0]
    assert v_choice.finish_reason == raw_finish_reason, 'Raw finish_reason must survive widened validation'

    # 2. Normalisation maps to the standard pydantic-ai FinishReason str-Literal.
    mapped: FinishReason = model._map_finish_reason(v_choice.finish_reason)  # type: ignore[reportPrivateUsage, reportArgumentType, assignment]
    assert mapped is not None
    assert mapped == mapped_finish_reason

    # 3. The inherited `_process_provider_details` helper stores the un-normalised raw
    #    value for downstream forensics.
    provider_details = model._process_provider_details(validated)  # type: ignore[reportPrivateUsage, arg-type]
    assert provider_details is not None
    raw_from_provider_details = cast(dict[str, object], provider_details).get('finish_reason')
    assert raw_from_provider_details == raw_finish_reason


@pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')
def test_bedrock_mantle_standard_finish_reasons_still_map_nonstream() -> None:
    """Regression guard: the widened Literal leaves standard `stop` mapping untouched.

    Unlike the non-standard cases above we intentionally use the strict public
    constructors for `Choice` / `ChatCompletion` — the standard literals are in the
    SDK's strict Literal, so they *should* survive public-ctor validation, and building
    them this way guards against the widened schema accidentally altering behaviour for
    normal values.
    """
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.messages import FinishReason

    msg = ChatCompletionMessage(role='assistant', content='hello')
    good_choice = Choice(finish_reason='stop', index=0, message=msg)
    good_completion = chat.ChatCompletion(
        id='789',
        choices=[good_choice],
        created=1704067200,
        model='openai.gpt-oss-safeguard-120b',
        object='chat.completion',
    )

    model = _mantle_chat_model()
    validated = model._validate_completion(good_completion)  # type: ignore[reportPrivateUsage]
    v_choice = validated.choices[0]
    mapped: FinishReason = model._map_finish_reason(v_choice.finish_reason)  # type: ignore[reportPrivateUsage, assignment]
    assert mapped is not None
    assert mapped == 'stop'


@pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')
async def test_bedrock_mantle_nonstd_finish_reason_stream() -> None:
    """Streaming path: Bedrock-native terminal finish reason passes widened validation.

    Exercises the same contracts as the non-stream test against the
    `BedrockMantleStreamedResponse` overrides on the stream consumer side:

      * `_validate_response` re-validates every chunk through the widened
        `_BedrockMantleChatCompletionChunk`; the non-standard terminal chunk is
        accepted rather than raising `ValidationError`.
      * `_map_finish_reason` normalises the raw value onto the standard
        `FinishReason` enum.
      * Text deltas from intermediate chunks continue to stream through
        (`'A' + 'B' == 'AB'` guard).
    """
    from collections.abc import AsyncIterator
    from typing import cast

    from pydantic_ai.messages import FinishReason
    from pydantic_ai.models.bedrock_mantle import BedrockMantleStreamedResponse

    chunk_a = chat.ChatCompletionChunk.model_construct(
        id='c1',
        choices=[
            chat.chat_completion_chunk.Choice.model_construct(
                finish_reason=None,
                index=0,
                delta=chat.chat_completion_chunk.ChoiceDelta.model_construct(content='A'),
            )
        ],
        created=1704067200,
        model='openai.gpt-oss-safeguard-120b',
        object='chat.completion.chunk',
    )
    chunk_b = chat.ChatCompletionChunk.model_construct(
        id='c2',
        choices=[
            chat.chat_completion_chunk.Choice.model_construct(
                finish_reason='guardrail_intervened',
                index=0,
                delta=chat.chat_completion_chunk.ChoiceDelta.model_construct(content='B'),
            )
        ],
        created=1704067200,
        model='openai.gpt-oss-safeguard-120b',
        object='chat.completion.chunk',
    )

    async def chunk_source() -> AsyncIterator[chat.ChatCompletionChunk]:
        yield chunk_a
        yield chunk_b

    # Bypass the public dataclass constructor: `_validate_response` only consumes the
    # `_response` async iterator, and the strict inherited field type (`PeekableAsyncStream`)
    # does not matter for this narrow `async for` usage. Mirrors the `test_zai.py` approach.
    resp = object.__new__(BedrockMantleStreamedResponse)
    object.__setattr__(resp, '_response', chunk_source())

    text_parts: list[str] = []
    last_chunk: chat.ChatCompletionChunk | None = None
    async for validated_chunk in resp._validate_response():  # type: ignore[reportPrivateUsage]
        last_chunk = validated_chunk
        if validated_chunk.choices:
            delta = validated_chunk.choices[0].delta
            content: str = cast(str, getattr(delta, 'content', None)) or ''
            if content:
                text_parts.append(content)

    assert ''.join(text_parts) == 'AB', 'Streamed text deltas must concatenate correctly'
    assert last_chunk is not None
    terminal_choice = last_chunk.choices[0]

    # Raw non-standard value survives widened chunk validation.
    assert terminal_choice.finish_reason == 'guardrail_intervened'

    # Normalisation maps to the standard pydantic-ai FinishReason str-Literal.
    mapped: FinishReason = BedrockMantleStreamedResponse._map_finish_reason(  # type: ignore[reportPrivateUsage, reportArgumentType, assignment]
        resp,
        terminal_choice.finish_reason,  # type: ignore[reportArgumentType]
    )
    assert mapped is not None
    assert mapped == 'content_filter'


@pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')
def test_bedrock_mantle_validate_completion_raises_unexpected_behavior_on_malformed_payload() -> None:
    """Malformed SDK completions are normalised to `UnexpectedModelBehavior` on the non-stream path.

    Coverage contract: `BedrockMantleChatModel._validate_completion` has an
    `except ValidationError` branch whose sole responsibility is to wrap Pydantic
    validation failures raised by the widened completion type, then re-raise as
    `UnexpectedModelBehavior`. Without this test the exception-only branch stays
    un-exercised and the repo-wide `fail_under = 100` coverage rule aborts CI.
    """
    from unittest.mock import MagicMock

    from pydantic_ai.exceptions import UnexpectedModelBehavior

    # Deliberately malformed payload: missing required top-level keys (`id`, `created`,
    # `object`, `model`) and a choice with the wrong shape, so the widened
    # `_BedrockMantleChatCompletion.model_validate` MUST raise `ValidationError`.
    bad = MagicMock(name='malformed-chat-completion')
    bad.model_dump.return_value = {
        'choices': [{'finish_reason': 'stop'}],
    }

    model = _mantle_chat_model()
    with pytest.raises(UnexpectedModelBehavior, match=r'Invalid response from') as exc_info:
        model._validate_completion(bad)  # type: ignore[reportPrivateUsage, arg-type]
    assert exc_info.value.__cause__ is not None, 'The chain must preserve the original ValidationError'


@pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')
def test_bedrock_mantle_unknown_finish_reason_still_fails_loudly() -> None:
    """Widening is per-provider vocabulary, not a catch-all: unknown values still abort.

    A made-up termination cause must keep the base-class semantics —
    `UnexpectedModelBehavior`-wrapped validation failure — so schema drift is caught
    rather than silently mapped.
    """
    from unittest.mock import MagicMock

    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from pydantic_ai.exceptions import UnexpectedModelBehavior

    msg = ChatCompletionMessage(role='assistant', content='...')
    bad_choice = Choice.model_construct(finish_reason='totally_not_a_real_cause', index=0, message=msg)
    completion = chat.ChatCompletion.model_construct(
        id='456',
        choices=[bad_choice],
        created=1704067200,
        model='openai.gpt-oss-safeguard-120b',
        object='chat.completion',
    )

    # The fake value must not be in the widened literal; sanity-check via a spy dump so we
    # really exercise the model_validate path of `_BedrockMantleChatCompletion`.
    spy = MagicMock(name='completion-spy')
    spy.model_dump.return_value = completion.model_dump()

    model = _mantle_chat_model()
    with pytest.raises(UnexpectedModelBehavior, match=r"Input should be 'stop'"):
        model._validate_completion(spy)  # type: ignore[reportPrivateUsage, arg-type]


@pytest.mark.skipif(not imports_successful(), reason='bedrock not installed')
async def test_bedrock_mantle_malformed_chunk_fails_loudly_in_stream() -> None:
    """Streaming path keeps fail-loud semantics for genuinely malformed chunks.

    Coverage contract for the `except ValidationError` branch inside
    `BedrockMantleStreamedResponse._validate_response`.
    """
    from collections.abc import AsyncIterator
    from unittest.mock import MagicMock

    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from pydantic_ai.models.bedrock_mantle import BedrockMantleStreamedResponse

    malformed = MagicMock(name='malformed-chunk')
    malformed.model_dump.return_value = {
        # missing: id, created, object, model, index, delta
        'choices': [{'finish_reason': 'stop'}],
    }

    async def chunk_source() -> AsyncIterator[chat.ChatCompletionChunk]:
        yield malformed  # type: ignore[misc]

    resp = object.__new__(BedrockMantleStreamedResponse)
    object.__setattr__(resp, '_response', chunk_source())
    object.__setattr__(resp, '_model_name', 'openai.gpt-oss-safeguard-120b')

    with pytest.raises(UnexpectedModelBehavior, match=r'chat completions stream'):
        async for _chunk in resp._validate_response():  # type: ignore[reportPrivateUsage]
            pass
