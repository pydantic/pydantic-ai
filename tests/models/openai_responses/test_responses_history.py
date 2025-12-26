"""Tests for OpenAI Responses model: message history and previous_response_id."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.output import ToolOutput
from pydantic_ai.usage import RequestUsage, RunUsage

from ...conftest import IsDatetime, IsNow, IsStr, try_import
from ..mock_openai import MockOpenAIResponses, response_message

with try_import() as imports_successful:
    from openai.types.responses.response_output_message import Content, ResponseOutputMessage, ResponseOutputText
    from openai.types.responses.response_usage import ResponseUsage

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


@pytest.mark.vcr()
async def test_openai_responses_verbosity(allow_model_requests: None, openai_api_key: str):
    """Test that verbosity setting is properly passed to the OpenAI API"""
    provider = OpenAIProvider(
        api_key=openai_api_key,
        base_url='https://api.openai.com/v1',
    )
    model = OpenAIResponsesModel('gpt-5', provider=provider)
    agent = Agent(model=model, model_settings=OpenAIResponsesModelSettings(openai_text_verbosity='low'))
    result = await agent.run('What is 2+2?')
    assert result.output == snapshot('4')


@pytest.mark.vcr()
async def test_openai_previous_response_id(allow_model_requests: None, openai_api_key: str):
    """Test if previous responses are detected via previous_response_id in settings"""
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)
    result = await agent.run('The secret key is sesame')
    settings = OpenAIResponsesModelSettings(openai_previous_response_id=result.all_messages()[-1].provider_response_id)  # type: ignore
    result = await agent.run('What is the secret code?', model_settings=settings)
    assert result.output == snapshot('sesame')


@pytest.mark.vcr()
async def test_openai_previous_response_id_auto_mode(allow_model_requests: None, openai_api_key: str):
    """Test if invalid previous response id is ignored when history contains non-OpenAI responses"""
    history = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The first secret key is sesame',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Open sesame! What would you like to unlock?'),
            ],
            model_name='gpt-5',
            provider_name='openai',
            provider_response_id='resp_68b9bd97025c8195b443af591ca2345c08cb6072affe6099',
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The second secret key is olives',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Understood'),
            ],
            model_name='gpt-5',
            provider_name='openai',
            provider_response_id='resp_68b9bda81f5c8197a5a51a20a9f4150a000497db2a4c777b',
        ),
    ]

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model)
    settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
    result = await agent.run('what is the first secret key', message_history=history, model_settings=settings)
    assert result.output == snapshot('sesame')


async def test_openai_previous_response_id_mixed_model_history(allow_model_requests: None, openai_api_key: str):
    """Test if invalid previous response id is ignored when history contains non-OpenAI responses"""
    history = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The first secret key is sesame',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Open sesame! What would you like to unlock?'),
            ],
            model_name='claude-sonnet-4-5',
            provider_name='anthropic',
            provider_response_id='msg_01XUQuedGz9gusk4xZm4gWJj',
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='what is the first secret key?',
                ),
            ],
        ),
    ]

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    previous_response_id, messages = model._get_previous_response_id_and_new_messages(history)  # type: ignore
    assert not previous_response_id
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='The first secret key is sesame', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[TextPart(content='Open sesame! What would you like to unlock?')],
                usage=RequestUsage(),
                model_name='claude-sonnet-4-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_response_id='msg_01XUQuedGz9gusk4xZm4gWJj',
            ),
            ModelRequest(parts=[UserPromptPart(content='what is the first secret key?', timestamp=IsDatetime())]),
        ]
    )


async def test_openai_previous_response_id_same_model_history(allow_model_requests: None, openai_api_key: str):
    """Test if message history is trimmed when model responses are from same model"""
    history = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The first secret key is sesame',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Open sesame! What would you like to unlock?'),
            ],
            model_name='gpt-5',
            provider_name='openai',
            provider_response_id='resp_68b9bd97025c8195b443af591ca2345c08cb6072affe6099',
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='The second secret key is olives',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='Understood'),
            ],
            model_name='gpt-5',
            provider_name='openai',
            provider_response_id='resp_68b9bda81f5c8197a5a51a20a9f4150a000497db2a4c777b',
        ),
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='what is the first secret key?',
                ),
            ],
        ),
    ]

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    previous_response_id, messages = model._get_previous_response_id_and_new_messages(history)  # type: ignore
    assert previous_response_id == 'resp_68b9bda81f5c8197a5a51a20a9f4150a000497db2a4c777b'
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='what is the first secret key?', timestamp=IsDatetime())]),
        ]
    )


async def test_openai_responses_usage_without_tokens_details(allow_model_requests: None):
    c = response_message(
        [
            ResponseOutputMessage(
                id='123',
                content=cast(list[Content], [ResponseOutputText(text='4', type='output_text', annotations=[])]),
                role='assistant',
                status='completed',
                type='message',
            )
        ],
        usage=ResponseUsage.model_construct(input_tokens=14, output_tokens=1, total_tokens=15),
    )
    mock_client = MockOpenAIResponses.create_mock(c)
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=mock_client))

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
                parts=[TextPart(content='4', id='123')],
                usage=RequestUsage(input_tokens=14, output_tokens=1, details={'reasoning_tokens': 0}),
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

    assert result.usage() == snapshot(
        RunUsage(input_tokens=14, output_tokens=1, details={'reasoning_tokens': 0}, requests=1)
    )


async def test_openai_responses_history_with_combined_tool_call_id(allow_model_requests: None, openai_api_key: str):
    m = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(m, output_type=ToolOutput(CityLocation))

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='What is the largest city in the user country?',
                )
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='get_user_country',
                    args='{}',
                    tool_call_id='call_ZWkVhdUjupo528U9dqgFeRkH|fc_68477f0bb8e4819cba6d781e174d77f8001fd29e2d5573f7',
                )
            ],
            model_name='gpt-4o-2024-08-06',
            provider_name='openai',
            provider_response_id='resp_68477f0b40a8819cb8d55594bc2c232a001fd29e2d5573f7',
            finish_reason='stop',
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='get_user_country',
                    content='Mexico',
                    tool_call_id='call_ZWkVhdUjupo528U9dqgFeRkH|fc_68477f0bb8e4819cba6d781e174d77f8001fd29e2d5573f7',
                )
            ]
        ),
    ]

    result = await agent.run('What is the largest city in the user country?', message_history=messages)
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country?',
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
                        id='rs_001fd29e2d5573f70068ece2e816fc819c82755f049c987ea4',
                        signature='gAAAAABo7OLt_-yMcMz15n_JkwU0selGH2vqiwJDNU86YIjY_jQLXid4usIFjjCppiyOnJjtU_C6e7jUIKnfZRBt1DHVFMGpAVvTBZBVdJhXl0ypGjkAj3Wv_3ecAG9oU3DoUMKrbwEMqL0LaSfNSN1qgCTt-RL2sgeEDgFeiOpX40BWgS8tVMfR4_qBxJcp8KeYvw5niPgwcMF3UPIEjHlaVpglJH2SzZtTOdxeFDfYbnvdWTMvwYFIc0jKOREG_-hZE4AznhHdSLV2-I5nGlxuxqaI4GQCk-Fp8Cvcy15_NYYP62ii50VlR6HPp_gQZEetwgC5pThsiuuG7-n1hGOnsj8gZyjSKsMe2KpzlYzhT7ighmArDVEx8Utvp1FXikqGkEzt4RTqqPInp9kuvqQTSyd8JZ6BEetRl1EuZXT7zXrzLwFN7Vm_gqixmf6mLXZUw6vg6LqGkhSh5fo6C7akPTwwJXjVJ37Dzfejo6RiVKOT-_9sdYCHW2kZ9XfQAmRQfB97UpSZ8QrVfaKy_uRIHLexs8QrQvKuw-uHDQBAL3OEmSTzHzCQ-q7b0FHr514Z29l9etavHNVdpeleWGo6VEtLWGQyblIdIBtf946YnQvr6NYIR8uATn9Z91rr8FsFJTpJh_v5iGA2f8rfPRu27nmw-q8XnPVc_FYCZDk08r_YhdEJZn1INBi8wYSWmpib8VxNpkFO7FFRuK-F8rh3MTpYgIOqPQYbf3LCRvKukTwv1b3mjSKVpHQSm_s6s7djdD-rLuc22-3_MLd0ii4_oOT8w51TQIM61LtonGvxUqf4oKHSUFCVnrWWiT-0ttdpwpJ_iB5frnEeY2mWyU1u7sd38BI3dOzoM82IFaIm98g9fa99bmoA7Z7gI60tzyF8YbJmWF-PCwyKHJ7B1MbCBonO36NmeEM-SplrR54fGykxTmwvtbYGhd5f0cdYzD0zulRDj-AhOd96rrUB_fIgoQGTXey8L_w0whcnVTWdG6is-rx8373Sz8ZRoE5RiLWW1mfHzVXxwslphx4BedRVF0tL-1YO7sg5MXhHCf6hpw8dOht-21NMrb1F1DQadFE_fhySFl-TgOD5BlhAuupLMsqcCIa4lcXP_loyA4ERP6WSdz2Bybz7_1eOiflfVodRrNqvr_DnL0NEXD_JkYTeIn84ziarFV7U7ZnkMvRiA_p1fWdbHTsE_8lu1rsf8fcJ1e76_6ycPkOc4TrOZw8gVRb7gIbMMVrv72BT_sFhW7GkXrzCQpQaeybmRw-bjFhkMMjMDYGXkA_H0q2Zfyh3zCOoa40hl2cqRWp7n1XuafmtKG_F8e9hyWox0q7AhZr5HOOaHz8r3O3-dmNl1KP52bqA8S72rLDslAOQlDupmAQgAmkm5ApYeYcEBredN78jHQ1pviUEI2-3qr4ClXZFHPa54AJ_q4HQ-EcKXEcYQglG21mSUy_tFQF-m4X46Qu8yYWcBVW4E0CG3wbvYx0BCdbc5RhIDkJo1elxLK8XS64lpFkCWy62xLVeMuVuCj8q84-Kk7tZ7gtMtLV9PHQCdbl3s2pAzMfuNIBJog6-HPmwha2n9T0Md5qF7OqCtnYWOWUfIMmQVcdW-ECGsQy9uIUmpsOjdtH31hrX3MUEhIOUB5xErLwfp-_s22ciAY_ap3JlYAiTKGlMCxKxTzK7wWEG_nYhDXC1Afj2z-tgvYhtn9MyDf2v0aIpDM9BoTOLEO-ButzylJ06pJlrJhpdvklvwJxUiuhlwy0bHNilb4Zv4QwnUv3DCrIeKe1ne90vEXe6YlDwSMeWJcz1DZIQBvVcNlN8q2y8Rae3lMWzsvD0YXrcXp02ckYoLSOQZgNYviGYLsgRgPGiIkncjSDt7WWV6td3l-zTrP6MT_hKigmg5F5_F6tS1bKb0jlQBZd0NP-_L_TPqMGRjCYG8johd6VyMiagslDjxG39Dh2wyTI19ZW7h_AOuOpnfkt2armqiq6iGfevA3malqkNakb6mFAS04J9O0butWVAw4yiPCEcLuDNAzzi_qrqLee4gkjh0NplvfGCaE6qqYms61GJbJC4wge6vjyTakurbqWEV3YoR3y_dn-0pjQ7TOx9kkruDwg0nZIV5O6yYxaulmbuvo3fs5CZb9ptZPD0MzGZj7CZU2MDCa4a4gr0McOx2MricxSzIu6emuRUzZuC6C1JxPRC00M0TrZNMIe_WVa9fXDLV1ULEAIMwMXzNT9zV6yiYQCwhkp30Wqde3W0LlIRpSbDuJXcvT8OCbXkdPNIScccdT9LvUQQ--hU2P45kisOev3TYn7yv-pdxM3u1KFNwuFxedSArMBPg7GDz1BOxDQRzv0mfwbf_CcoFbuyj7Tf4zWO46HVdHeRNbvIE--bnaSYD-UFaKknp8ZsBQQhBU_2TEca3fKwmg81-g7Vdb28QUZEuPzgE4ekxZejkKpiKqlLC5nJYgvXrqk2H35D51mYdzPs0ST05Mc41x9MFm_YOLxSFyA0yGAKVINmD5wT6kvRflPkgoksd2ryIvo4KMw3oZQKodv5By0mSJ8iX2vhTGylxiM8wj-ICyNuOsaRFrcMSpX7tZbXcDyysApdmx217BSADoQiNZBLngF7ptxc2QGyo3CwuDjaljwmSgL9KeGthd1RJFd826M287IPpCjLM4WRquCL_E0pQryNqOMn-ZEOCAlBjE37290EhkjKbhiGBEnHUvSbhoH4nL47AmunP_Q5aqh5173VfyoyaybuS3fXjQ5WO0kyFjMdD-a7C6PVdwToCTP-TljoF2YnQKCiqUGs9gNHS9mYhQSXzY4uuGlTHLfKB4JKS5_MQHvwI9zCbTvVG854fPuo_2mzSh-y8TSzBWPokhYWI_q095Sh6tOqDIJNMGyjI2GDFRSyKpKhIFCLyU2JEo9B6l91jPlir0XI8ZOQfBd9J0I4JIqnyoj40_1bF1zUDGc014bdGfxazxwlGph_ysKAP39wV7X9DBFS3ZmeSIn-r3s-sci0HmwnJUb2r03m40rFuNTV1cJMAFP7ZY7PQQQ0TtlO_al0uedaOWylLauap_eoRqc6xGJ2rSz1e7cOevksUlAqzK5xknYKHlsW970xuDGHKOZnKPg8O9nb2PKrcjwEQF5RFPc3l8TtOUXPhhvTERZFGoEuGuSuSp1cJhzba06yPnL-wE3CstYUm3jvkaUme6kKqM4tWBCQDg-_2PYf24xXYlmkIklylskqId826Y3pVVUd7e0vQO0POPeVYU1qwtTp7Ln-MhYEWexxptdNkVQ-kWx63w6HXF6_kefSxaf0UcvL8tOV73u7w_udle9MC_TXgwJZpoW2tSi5HETjQ_i28FAP2iJmclWOm3gP08cMiXvgpTpjzh6meBdvKepnifl_ivPzRnyjz3mYCZH-UJ4LmOHIonv-8arnckhCwHoFIpaIX7eSZyY0JcbBETKImtUwrlTSlbD8l02KDtqw2FJURtEWI5dC1sTS8c2HcyjXyQDA9A25a0M1yIgZyaadODGQ1zoa9xXB',
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city":"Mexico City","country":"Mexico"}',
                        tool_call_id='call_LIXPi261Xx3dGYzlDsOoyHGk',
                        id='fc_001fd29e2d5573f70068ece2ecc140819c97ca83bd4647a717',
                    ),
                ],
                usage=RequestUsage(input_tokens=103, output_tokens=409, details={'reasoning_tokens': 384}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 13, 11, 30, 47, tzinfo=timezone.utc),
                },
                provider_response_id='resp_001fd29e2d5573f70068ece2e6dfbc819c96557f0de72802be',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_LIXPi261Xx3dGYzlDsOoyHGk',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )
