"""Tests for OpenAI Responses model: chain-of-thought (CoT) reasoning functionality."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.usage import RequestUsage

from ...conftest import IsDatetime, IsNow, IsStr, try_import
from ..mock_openai import MockOpenAIResponses, response_message

with try_import() as imports_successful:
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage, ResponseOutputText
    from openai.types.responses.response_reasoning_item import (
        Content as ReasoningContent,
        ResponseReasoningItem,
        Summary,
    )

    from pydantic_ai.models.openai import (
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_openai_responses_requires_function_call_status_none(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel(
        'gpt-5',
        provider=OpenAIProvider(api_key=openai_api_key),
        profile=replace(openai_model_profile('gpt-5'), openai_responses_requires_function_call_status_none=True),
    )
    agent = Agent(model)

    @agent.tool_plain
    def get_meaning_of_life() -> int:
        return 42

    result = await agent.run('What is the meaning of life?')
    messages = result.all_messages()

    _, openai_messages = await model._map_messages(  # type: ignore[reportPrivateUsage]
        messages,
        model_settings=cast(OpenAIResponsesModelSettings, model.settings or {}),
        model_request_parameters=ModelRequestParameters(),
    )
    assert openai_messages == snapshot(
        [
            {'role': 'user', 'content': 'What is the meaning of life?'},
            {
                'id': 'rs_01d311e2633707df0068fbac0050ec81a2ad76fd9256abcaf7',
                'summary': [],
                'encrypted_content': 'gAAAAABo-6wE6H4S9A886ZkwXcvvHqZ6Vx5BtpYvvNAJV5Ijq7pz-mTBJxfdjilNSzBj0ruy7NOsMRMhWzNahRf-n3KDQ2x1p-PjVCHM5IAGqHqae8A-aAUn_FDRiTbAT5N5FXTrZ80DAtdDv17z2HlODmTTYRvBU2-rX7opysjc4rf7-rvy6j4cUcNbM0ntT5DH8UHxC9LCM_s7Cb2unEV0jaDt7NzFxgfWN2u24Avs2EnjPoxOjd6BR-PWHJk_7kGGkVBub8NU7ZOyHsci3T8DAq_eX38DgkHJBJCPT4EqvlNP-VjPdecYEFUCw5G_Pye6h55-77g8LjkrFO43f8p6wscQ0iM601i1Ugmqbzxyv1ogPIN-YuSk2tkCw-D7xBD7I4fum2AmvyN-fR58lWcn-Z0WTqACA4baTJiCtW5b7uVeAp8vm8-gWzFR5BdDHVdQqu1TAKVWl_1P8NauDtd5M24MjVZd6WC0WrbTDPY9i2gieMMjFek2M8aoQFO0CG7r3JHn2zxfFB3THWCpl4VqZAQp6Ok7rymeY0Oayj--OLpNMBXIYUWc51eyYeurwQ943BSkf-m6PPVKO8T5U__Bx-biCNCePSlFKp7V0Du6h7UgYoqqonH2S3Jrg87c6dk7VJ7ca2i8sZqhy0rG6Kb7ENDVvwkMOdpnaFgdWd3VINp6P8j69kBQg-qwWP-YHPC9LnsjT2j1ktMowVO97eOpV4j2BhiThxunmu_SOIAEbghmjJEkLuRxLxBUPFRIajke2CvvFeIuReJr53isPKOxOjVzsc6oG5ZeykDlfz_mfEap7AByPNY0987zwG58tGueNxXjdpd7NQFcn_6DKj60SvUg0sk49V_QrDY3cAhSRvZoEeqA8XR97pEe7CByYMl80b9fzgyahc4NCdUwK8es2ll-lsJwEx1ZGdC8cB45QOrTnw8tJAUsSM44rLKwAQY-KsuN4UygO99d1CQZEm2YWtnPAvA9I-EhY87UIDx0CpPsEyxxFu2GZCTy7ceSnpcmQbAFWXzfBSpM7k42xVV8G8IK_bHpoF1enF5Vbc37_L_aWd4AgzuAwF_RVyd8exVh3NVJtO3BqPv72kTukr2Fok3KEaSeU0whP_dxr-thP2exS0F2Jdn13ZtB_pqxwKVWEsvzdbN92Q9qs10BAgYs2SA4cq66semwRl-1n-dr7XJyZzPOEiA9TQYgUCw0ueIc0ciMOZ0Waaj094bKIylw_TD5Bu1diXpzbTma_AVO-NZn7INhAZN3guSme-zIUEMrh66w0VJP-DbDA-ecSD41eMRSadyV4g86wLL4NOBE5NwSiSkwd2xJ9NqG7YohFM8BlPdEV4zhmqHcIKpVwAitFItqnAaUSU42Aebdritt9oNVnpKCeeA4QQv_8W7rOXJlLfGXRJUBCrh3Rv7KCVC3yncAOIU8FWu3jyaAqhLrWHLW958wjF8ka7lw80YZbToPjIuiii0UXu2w3Tv5EGVdkhf05A3Yj6M_LXStns8iBMzcU4-mJ1649FnnImLnW5AeohoWPBB6WYhW9gfwjuxejTI3Q5R0mo9jUSP3_tFiawlC2zFgvkNFufC6Kry8-Burjf8l6rpAX7_sjtCu1AlAbI6PEFtxcKhNWHfQp4mUATR6P4k68jk_Kl-FpRBtNOf8YOlLGrKE-WbwCoIV7VAgK2CTZJOxaslxVZRCLObNrA3XuEtc3jo8pMzqx8GJWshIgmF4XiQcmgh65U_kjB07adlgnbCZvGUXdIIQiA2vqIWC6Qu8SSO20nOOR65hGXyIgf4aOolU0Ljbi4slXnJKjbcPaX5O3cXvKHbkVFwXmHK2Ymaqb6fZcap78_On8jLK_GRlw3jV18SLeOcJiG2LqtHzcUawY4K7bPDNY2QX89yL5d4qxRF577QgzalmdQDsKyC_N-wk',
                'type': 'reasoning',
            },
            {
                'name': 'get_meaning_of_life',
                'arguments': '{}',
                'call_id': 'call_cp3x6W9eeyMIryJUNhgMaP5w',
                'type': 'function_call',
                'status': None,
                'id': 'fc_01d311e2633707df0068fbac038f1c81a29847e80d6a1a3f60',
            },
            {'type': 'function_call_output', 'call_id': 'call_cp3x6W9eeyMIryJUNhgMaP5w', 'output': '42'},
            {
                'role': 'assistant',
                'id': 'msg_01d311e2633707df0068fbac094ff481a297b1f4fdafb6ebd9',
                'content': [{'text': '42', 'type': 'output_text', 'annotations': []}],
                'type': 'message',
                'status': 'completed',
            },
        ]
    )


async def test_openai_responses_raw_cot_only(allow_model_requests: None):
    """Test raw CoT content from gpt-oss models (no summary, only raw content)."""
    c = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[],
                type='reasoning',
                content=[
                    ReasoningContent(text='Let me think about this...', type='reasoning_text'),
                    ReasoningContent(text='The answer is 4.', type='reasoning_text'),
                ],
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_123',
                        provider_name='openai',
                        provider_details={'raw_content': ['Let me think about this...', 'The answer is 4.']},
                    ),
                    TextPart(content='4', id='msg_123'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_raw_cot_with_summary(allow_model_requests: None):
    """Test raw CoT content with summary from gpt-oss models.

    When both summary and raw content exist, raw content is stored in provider_details
    while summary goes in content.
    """
    c = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[Summary(text='Summary of thinking', type='summary_text')],
                type='reasoning',
                encrypted_content='encrypted_sig',
                content=[
                    ReasoningContent(text='Raw thinking step 1', type='reasoning_text'),
                    ReasoningContent(text='Raw thinking step 2', type='reasoning_text'),
                ],
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('What is 2+2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='Summary of thinking',
                        id='rs_123',
                        signature='encrypted_sig',
                        provider_name='openai',
                        provider_details={'raw_content': ['Raw thinking step 1', 'Raw thinking step 2']},
                    ),
                    TextPart(content='4', id='msg_123'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_multiple_summaries(allow_model_requests: None):
    """Test reasoning item with multiple summaries.

    When a reasoning item has multiple summary texts, each should become a separate ThinkingPart.
    """
    c = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[
                    Summary(text='First summary', type='summary_text'),
                    Summary(text='Second summary', type='summary_text'),
                    Summary(text='Third summary', type='summary_text'),
                ],
                type='reasoning',
                encrypted_content='encrypted_sig',
                content=[
                    ReasoningContent(text='Raw thinking step 1', type='reasoning_text'),
                ],
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='Done', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    agent = Agent(model=model)
    result = await agent.run('Test multiple summaries')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Test multiple summaries',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='First summary',
                        id='rs_123',
                        signature='encrypted_sig',
                        provider_name='openai',
                        provider_details={'raw_content': ['Raw thinking step 1']},
                    ),
                    ThinkingPart(content='Second summary', id='rs_123'),
                    ThinkingPart(content='Third summary', id='rs_123'),
                    TextPart(content='Done', id='msg_123'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )


@pytest.mark.vcr()
async def test_openai_responses_raw_cot_stream_openrouter(allow_model_requests: None, openrouter_api_key: str):
    """Test streaming raw CoT content from gpt-oss via OpenRouter.

    This is a live test (with cassette) that verifies the streaming raw CoT implementation
    works end-to-end with a real gpt-oss model response.
    """
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    model = OpenAIResponsesModel('openai/gpt-oss-20b', provider=OpenRouterProvider(api_key=openrouter_api_key))
    agent = Agent(model=model)
    async with agent.run_stream('What is 2+2?') as result:
        output = await result.get_output()
    assert output == snapshot('4')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_tmp_2kbe7x16sax',
                        provider_details={
                            'raw_content': [
                                'The user asks: "What is 2+2?" They expect a straightforward answer: 4. Just answer 4.'
                            ]
                        },
                    ),
                    TextPart(content='4', id='msg_tmp_8cjof4f6zpw'),
                ],
                usage=RequestUsage(
                    input_tokens=78,
                    output_tokens=37,
                    details={'is_byok': 0, 'reasoning_tokens': 22},
                ),
                model_name='openai/gpt-oss-20b',
                timestamp=IsDatetime(),
                provider_name='openrouter',
                provider_url='https://openrouter.ai/api/v1',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 11, 27, 17, 43, 31, tzinfo=timezone.utc),
                },
                provider_response_id='gen-1764265411-Fu1iEX7h5MRWiL79lb94',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_raw_cot_sent_in_multiturn(allow_model_requests: None):
    """Test that raw CoT and summaries are sent back correctly in multi-turn conversations.

    Tests three distinct cases across turns:
    - Turn 1: Only raw content (no summary) - gpt-oss style
    - Turn 2: Summary AND raw content - hybrid case
    - Turn 3: Only summary (no raw content) - official OpenAI style
    """
    sent_openai_messages: list[Any] = []

    c1 = response_message(
        [
            ResponseReasoningItem(
                id='rs_123',
                summary=[],
                type='reasoning',
                content=[
                    ReasoningContent(text='Raw CoT step 1', type='reasoning_text'),
                    ReasoningContent(text='Raw CoT step 2', type='reasoning_text'),
                ],
            ),
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )

    c2 = response_message(
        [
            ResponseReasoningItem(
                id='rs_456',
                summary=[
                    Summary(text='First summary', type='summary_text'),
                    Summary(text='Second summary', type='summary_text'),
                ],
                type='reasoning',
                encrypted_content='encrypted_sig_abc',
                content=[
                    ReasoningContent(text='More raw thinking', type='reasoning_text'),
                ],
            ),
            ResponseOutputMessage(
                id='msg_456',
                content=cast(list[Content], [ResponseOutputText(text='9', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )

    c3 = response_message(
        [
            ResponseReasoningItem(
                id='rs_789',
                summary=[
                    Summary(text='Final summary', type='summary_text'),
                ],
                type='reasoning',
                encrypted_content='encrypted_sig_xyz',
                content=[],
            ),
            ResponseOutputMessage(
                id='msg_789',
                content=cast(list[Content], [ResponseOutputText(text='42', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )

    mock_client = MockOpenAIResponses.create_mock([c1, c2, c3])
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(openai_client=mock_client))

    original_map_messages = model._map_messages  # pyright: ignore[reportPrivateUsage]

    async def capture_messages(*args: Any, **kwargs: Any) -> Any:
        result = await original_map_messages(*args, **kwargs)
        sent_openai_messages.append(result[1])
        return result

    model._map_messages = capture_messages  # type: ignore[method-assign]

    agent = Agent(model=model)

    result1 = await agent.run('What is 2+2?')
    assert result1.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_123',
                        provider_name='openai',
                        provider_details={'raw_content': ['Raw CoT step 1', 'Raw CoT step 2']},
                    ),
                    TextPart(content='4', id='msg_123'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )

    result2 = await agent.run('Add 5 to that', message_history=result1.all_messages())
    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_123',
                        provider_name='openai',
                        provider_details={'raw_content': ['Raw CoT step 1', 'Raw CoT step 2']},
                    ),
                    TextPart(content='4', id='msg_123'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Add 5 to that',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='First summary',
                        id='rs_456',
                        signature='encrypted_sig_abc',
                        provider_name='openai',
                        provider_details={'raw_content': ['More raw thinking']},
                    ),
                    ThinkingPart(content='Second summary', id='rs_456'),
                    TextPart(content='9', id='msg_456'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )

    result3 = await agent.run('What next?', message_history=result2.all_messages())
    assert result3.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_123',
                        provider_name='openai',
                        provider_details={'raw_content': ['Raw CoT step 1', 'Raw CoT step 2']},
                    ),
                    TextPart(content='4', id='msg_123'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Add 5 to that',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='First summary',
                        id='rs_456',
                        signature='encrypted_sig_abc',
                        provider_name='openai',
                        provider_details={'raw_content': ['More raw thinking']},
                    ),
                    ThinkingPart(content='Second summary', id='rs_456'),
                    TextPart(content='9', id='msg_456'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What next?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='Final summary',
                        id='rs_789',
                        signature='encrypted_sig_xyz',
                        provider_name='openai',
                    ),
                    TextPart(content='42', id='msg_789'),
                ],
                model_name='gpt-4o-123',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1',
                provider_details={'timestamp': datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)},
                provider_response_id='123',
                run_id=IsStr(),
            ),
        ]
    )

    assert len(sent_openai_messages) == 3

    turn2_messages = sent_openai_messages[1]
    turn2_reasoning = [msg for msg in turn2_messages if msg.get('type') == 'reasoning']
    assert len(turn2_reasoning) == 1
    assert turn2_reasoning[0] == snapshot(
        {
            'type': 'reasoning',
            'summary': [],
            'encrypted_content': None,
            'id': 'rs_123',
            'content': [
                {'type': 'reasoning_text', 'text': 'Raw CoT step 1'},
                {'type': 'reasoning_text', 'text': 'Raw CoT step 2'},
            ],
        }
    )

    turn3_messages = sent_openai_messages[2]
    turn3_reasoning = [msg for msg in turn3_messages if msg.get('type') == 'reasoning']
    assert len(turn3_reasoning) == 2
    assert turn3_reasoning[0] == snapshot(
        {
            'type': 'reasoning',
            'summary': [],
            'encrypted_content': None,
            'id': 'rs_123',
            'content': [
                {'type': 'reasoning_text', 'text': 'Raw CoT step 1'},
                {'type': 'reasoning_text', 'text': 'Raw CoT step 2'},
            ],
        }
    )
    assert turn3_reasoning[1] == snapshot(
        {
            'type': 'reasoning',
            'summary': [
                {'type': 'summary_text', 'text': 'First summary'},
                {'type': 'summary_text', 'text': 'Second summary'},
            ],
            'encrypted_content': 'encrypted_sig_abc',
            'id': 'rs_456',
            'content': [
                {'type': 'reasoning_text', 'text': 'More raw thinking'},
            ],
        }
    )


async def test_openai_responses_system_prompts_ordering(allow_model_requests: None):
    """Test that system prompts are correctly ordered in mapped messages."""
    c = response_message(
        [
            ResponseOutputMessage(
                id='msg_123',
                content=cast(list[Content], [ResponseOutputText(text='ok', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            ),
        ],
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

    messages: list[ModelRequest | ModelResponse] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System prompt 1'),
                SystemPromptPart(content='System prompt 2'),
                UserPromptPart(content='Hello'),
            ],
            instructions='Instructions content',
        ),
    ]

    instructions, openai_messages = await model._map_messages(  # type: ignore[reportPrivateUsage]
        messages,
        model_settings=cast(OpenAIResponsesModelSettings, {}),
        model_request_parameters=ModelRequestParameters(),
    )

    assert instructions == 'Instructions content'

    assert openai_messages == snapshot(
        [
            {'role': 'system', 'content': 'System prompt 1'},
            {'role': 'system', 'content': 'System prompt 2'},
            {'role': 'user', 'content': 'Hello'},
        ]
    )
