"""Tests for image generation."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    BinaryImage,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
    FinalResultEvent,
    ModelRequest,
    ModelResponse,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    capture_run_messages,
)
from pydantic_ai.agent import Agent
from pydantic_ai.builtin_tools import ImageGenerationTool
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
)
from pydantic_ai.output import NativeOutput, PromptedOutput
from pydantic_ai.usage import RequestUsage

from ...conftest import IsBytes, IsDatetime, IsNow, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import (
        OpenAIResponsesModel,
    )
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_openai_responses_image_generation(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, output_type=BinaryImage)

    result = await agent.run('Generate an image of an axolotl.')
    messages = result.all_messages()

    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='68b13f',
        )
    )
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
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
                        id='rs_68cdc3d72da88191a5af3bc08ac54aad08537600f5445fc6',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_68cdc3ed36dc8191b543d16151961f8e08537600f5445fc6',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='68b13f',
                        ),
                        id='ig_68cdc3ed36dc8191b543d16151961f8e08537600f5445fc6',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1536x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_68cdc3ed36dc8191b543d16151961f8e08537600f5445fc6',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(content='', id='msg_68cdc42eae2c81918eeacdbceb60d7fa08537600f5445fc6'),
                ],
                usage=RequestUsage(
                    input_tokens=2746,
                    cache_read_tokens=1664,
                    output_tokens=1106,
                    details={'reasoning_tokens': 960},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 19, 20, 57, 58, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )

    result = await agent.run('Now give it a sombrero.', message_history=messages)
    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='2b4fea',
        )
    )
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Now give it a sombrero.',
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
                        id='rs_68cdc4311c948191a7fb4cb3e04f12f508537600f5445fc6',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_68cdc46a3bc881919771488b1795a68908537600f5445fc6',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='2b4fea',
                        ),
                        id='ig_68cdc46a3bc881919771488b1795a68908537600f5445fc6',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1536x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_68cdc46a3bc881919771488b1795a68908537600f5445fc6',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(content='', id='msg_68cdc4c5951c8191ace8044f1e89571508537600f5445fc6'),
                ],
                usage=RequestUsage(
                    input_tokens=2804,
                    cache_read_tokens=1280,
                    output_tokens=792,
                    details={'reasoning_tokens': 576},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 19, 20, 59, 28, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_image_generation_stream(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model, output_type=BinaryImage)

    async with agent.run_stream('Generate an image of an axolotl') as result:
        assert await result.get_output() == snapshot(
            BinaryImage(
                data=IsBytes(),
                media_type='image/png',
                identifier='be46a2',
            )
        )

    event_parts: list[Any] = []
    async with agent.iter(user_prompt='Generate an image of an axolotl.') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='69eaa4',
        )
    )
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
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
                        id='rs_00d13c4dbac420df0068dd91a321d8819faab4a11031f79355',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='69eaa4',
                        ),
                        id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1024x1536',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=1588,
                    output_tokens=1114,
                    details={'reasoning_tokens': 960},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 20, 40, 2, tzinfo=timezone.utc),
                },
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_00d13c4dbac420df0068dd91a321d8819faab4a11031f79355',
                    signature=IsStr(),
                    provider_name='openai',
                ),
            ),
            PartEndEvent(
                index=0,
                part=ThinkingPart(
                    content='',
                    id='rs_00d13c4dbac420df0068dd91a321d8819faab4a11031f79355',
                    signature='gAAAAABo3ZGveBi351h31WQM2aG_dbN1N74J4X3Lf1SbUrUhElKaT5odbh4N1liwG5Hjip3Ci1illQSsd4n035fOOIV3sZzAMvV3ypncux4WDBpQ9NbeuFMNSyNOPTxJLg4j66UbW2ptw3u1VP3j0vCHvV5MoDhErheYZsWKhYVtkUNkKSVLWkS_yK0pOltSwHfRy3tbrkxnqD99BuVbCjV1nWSzTAmJLicBtjDaH0NjjD_vMyFiUe83-eZRs-Q_6njWasZNCmTcOq4zlpFoJ_AGeaTbaLIC1OwDV3sNT7pXvo7YI7jmsYEhHAKa8BjZmMjzBPLDRu9TMWtXMnO6nyVYqMxsyQPdNmP-BDNfr_8Rmo_uI5egfE0qRKgAc5MrOGd1fSgtUqeKah3kbLMyCD0_-jWmVInb2Y4LfPcX0iOeTGum2IRKwy6G1tdY8C_gEOnIGAUKOT2sEF98Ythy9auV27BCbcjfCBJlH0rOir_OiQjUIXZqqY0My1kVENBbXj2-VIFIqG-CcxCldFG2Mq0NGo86h1igQIFmLItXLPTS_QnaWADSD9La8JWpg8CuWg-yB3UqYaG5_f2Cl5jRDQdIYavTBvD-lp54y8aEnGA6HksQaCtB7jHX0ZM0pqYu7LvLjeHxAJWsnF4NN0HPz3d307muS0TtrXUxqeZFdTNoqdBOxfuJ2-Ym_LmeubnEzh5wHAguJKZ6S_jcEFM3Jdb1R8Gk9dv2y7BUz1hKSFF7peXc9Ear00JjPHAlR1x0ECqONTSD9Kda80pQlDSh05ITKQ2viOy1jmCsWeSsll6EJPcGEMfAcZ1UMgHYX3sBa5Oz3DS28Ur8yk-I62nUWbcj8n7IsZmZL0CWc2qgCtj2TzFZaVEx8iumUKpU0hmML_kF3JPH2Ie8nB1ko18HZV7A_-n9XGDZzwsfPD9pu4P-fb68KNqU_qQBfe8msYvuFuljC-0kyGrQIQH2X6stwEkyme1TuJfxIZ2t2q2l02gEUVN8LN8qX98hp7DBxXepgdKvqWVOM7icvtW0mPACf1b4izSDqEgqhqx4tNsjixoHcM9M8awzss_y2_jZ3V7gY3pbPgwWKHyyTUzA1ogPfMkjxxUrVLNyHRPmnklUeQdV-vytip3BzNOq4yTUz7jVFrudSDcr_KM6Ie806OkgKF81l-W-40qzx6bGg2DAcZf5hfbTzk-ho51sRBwDp7RJrx2SXSBGXA3ArYzgq-2iat368uDLiQhhbunzKm3_6CFWggpbUO8Kp3FP7-k4Z4CRbHkg8WVT0HhH6w0ysoi-P6_ZH-IKI7XG-GT1kq4yje3qlfRUT0-0_LPsr8LyM6AbOYj4NiWHP3XJ2qa978VVOLJQtY-qG3VX9kMq13C-uU8PDOsOEidYZl2gqFtXhxkXivwACbLMnvzJayXJRev1QkoNxIg1Stl9II4D_ndHfNYeAvMvOnSNafoCOmMzCBp1klovMP_31YvR2B3af1TYanbbHoJt2UR1GRR_Aqr7G6RukNkXAl63LPlDQSYm5BB6zD9iNX9hJ8MSZ1IFIcbM0L32tAWsyKKAEWyr9MGckicDa_hES9adeXuunqqKhUctd94J1dsXLiWCGIet57YIUj5WoF_FQ6D6FY9rB00KhCDlHr1Ot5NCMmn6y-u6TYJUhpl7elEErYGXaGPhUtKUSbAIzOXzBIAKb_MiMVvo6a2VYwsxwZV14X8TYkKw_Y7w5Wt6JA_wTOoen7Cc0eFyc7FZA4NjIMkIUOXymtjzOSkFJz1eMBqp9diET9VYKGsn6GxviD8jWM6-RCWcFurewcn4d6TeTclAt7G_LZrJ9bZtMVlieSJT-3vWr9qVt8OGBUJEJRVOzpr5FBnEceqK8s7D_s8EZwTaGwyAuuaZThy2PNWJhpE4c0UeKgh0ec36Q0ZRN8DF8Khne8Epe1rehOrsfeyFFRuQ5CDGdHimhtOAIbDyg_5PPCp8fgiU4R9xqtizCVTR4ej1VPIClmebUErOl3TN-IyoSc8rv--Vi0ATn69Q8tSPweI07KVEzRJpDtxbnGcbbilPN5_liJcQrLMf5ikaWBoq42s6FXDjr-ASD0h7IlNGHxnN8q__iO6jA9-2PTywI2bbBJsie2L7OaGGehO5zv_rWv_6rbk4HLVcQafi2nC5w1GNeDaXWSz0RjiTfXxjBh98302CQxiM-e1Lvt1Pe6Mqv-pAgXlFrSHDrqw8s4NSS2YpLDTUIOcOx8UutAJOgVpyZm2sQcvtOsGsSUBIyNI_4huseO9EuXF4TUQ-yzQRsimtXaDa6VId0y6qG7dWxTP30SWZkft2iW2_Nz_56MiioY7xACIjzo4s2aGLM352ufd4nEeU-K3UQd5hvhdIWUZn6KTyCUnqgChyIlB0Sto24VwIIj74DYisSiu-d8EYsVr5gZaQ_NaW4T7M_ZB0TJ0ptlU0X_h8uLu0ro2Vc_s7D8nkIKSzhGuuHO4lOjvZ-qLsPxG-pBa6jGvv1hOyng_x99icZ0oM7G7FmDl7SjP1pdLiZAA1hMPPU9b8Uk0j8hb4AFtfoXSfwZBQ8sYlT0_QmcSBgGxfZKXv4RcFSnAEGDNUn1V-P2uNoj06MOwzroZVjTuzVy284Hqe-08Gtt_bvZDmfsHonbEw5DrthsP9SzoC62hc6pcVs_ApQE5LwHgODxT-oejDppixNCr--hJ1IYVj4rRsHsmBv33H5kJP0rwmkdJ-I8rLj66jLf_Qu_OEh02dJqf3XSYsG7io3XCVjA-d-jUhLJSqcPS_3y5thCtWUcG_ucT64ADWdtOH0EkmzN0o7HmOJ48pkGhttNScjXlQUmOdkeBV55dTdXAzAyjKZsxP5ZK1F9m_1TMWDJX6nT4rRFrzv3PQByEyc3Rje7ZUdGa3Qky1-T5uhu1dk4ty_I92CbMDCM-jGZorhg5MX10B_zZ03DFrYTrdcDILS5i_BSOlGT8Du4aSMvwvUC5FLOYQFQdM_ZNIRIGhOSWsvObmVYh0j70YKqitDudSIm1V_Yw6qsW3ZPpLDgBju176FVDJJBn1Wx-DeQ6FrYtOjFHctqJN-2mjWQi_7lAzKbTLsB-9c4iZ4_efWXsHncmAeqvt0gvglQHDhY6cM4yZurpHkrE-lb5-vDLYamv-Du7Cs0pAaynEcbT-f3_F_WOgoXFy2WYOTt2KkSQZnW6ZPHzl3gfVOsHfAkWalMJ6vXa8FuoYfMmgZJpqtee5J6AxJaUea8xQ0VlVwuXmcK8EOPcwF1pWg8w5_SweA9jZ0fh5PaFW-BNlzGDmhRR-8Up0TCUTsdnZN7bABJBlxeQ5GEcwOjgT0UBF0_zXZo5fbk34TSDoEgfdQydVLlOGda8McmvsnNzDSq77a-Vj_8BeVacM1PPG9rp9F_-PQgpM7_7YsNoWMXha4b4_H58q7vPOvMK1zxRzNrq-sm9QhQ1LkzPgt158Gf2IPq8D3rh9YCmJvg1Ju7roShfnVdV_UO73MLnDhoqaUZEdq10723KFpescGNTRpsuWDE8qBiu58rbOzjmpy7nJfuOtfrv_qSjaFRTkShLV5PW2neHjNLlvQlWy-q3yjJXq-2zM-iRehbFIxI3ATcCq-SgThDeQ1qnTg9G0Jtsx3qBNZtCIi8x1oVsyavVJcqvo36UC-IXaXA1vpjuwER1dcZ999sP9MUnXcMTO9ba-GM3dslKvDtuZ5b8x_u7eCJfawzUPItU8iwISYKDWW8wTNOS8Iukujq3-IDOEFqmCOAlkdv6-AWNc7ZVOmyvvgDCpSN5nSkjpJhWI5kP13FJtJNHkNtP4RQkhRhwRh2ei308TvNgT8YSaa4E_BJ-QWQ_9PMNBsfAYSGIl1VaQinZF0qdvNhRIlonuZMV58aEEzsLk6hS7CGlbFwBMwAzZ5Q6PANavXDFiPGeIadxTE4r-iZLQ3CdvJWUiUv3AL3lzYraXX8BGDpEVAAIqoRYZEpR2QgIUui5b3gkCSlG-YdKqJ4HZ_6VCFqpywKsgPCX_c8pVD_6eJhgt9o5Vc0ARsfc1IG_XC-nFWOV4caiARMobX0y4qXDFulrAZInqBZ9Pq5MmbbhBmLLdT-y5fdPpB5UxsIHGqb3pip4ZaKS80IqAt8t7HPXSNza7zb1TwrjNlYcO_KhbLQBB0hMmKULnEJPWLDPKf_9NeAsN3U9AWyj1WpAKjSfpjjbXn37qpTMdgd-Js9-_FDaXDFH_aOYXI0GY1AMpvSSQzx_f6Erq4qyS5TAuAtXbvUm-iVJcHaZTIy7buGJqOUBb7BC1L33KpeQEZuCg6QyAdzn4bZUKvwjXuxNykpZA9LZWaFVdx2QfwCV_yqN2TTvLFmSj5SjldGwbBndjmtHs5kkDcV2mDlm3huEfbEJqf9sdxXaYhIfmUIkFDtYTpE1C0qSol-A6Yagtx_aNfWTL7F2lFI0OusuBwnDfkNow5mPsKqGMIqx5eJA2InLcpV7GTyCxT3BjVsggtSb1-4Zz2TYzBz7iYe8NPe-rxF6XWyHf1N0nyyCY8Y0_CqJS9OPFpsd53a6qY7xlhh1kwBOM8nJWb3OEJjMVspTUfwF90O8D9fDNS293vnG8SArU6d-1L4u0LalQbKXDRzcze8W8R3KWv1N0LXrWwfArPrO1WnpdEkJnbFfc1eUHqThJ39c7RAInK66VtNe5xtUVzuNZDfPKsIfD4Ms5xqMKEOWQt8RIciRapDo9aoWv5l-YCkuTrWp4pWP4b7eu9fizM5ZuzmRCj3Ecc7ZT2uvxe9sP045dqTH6lSeBNW1eW-pmb3oQ-g_mYL6SU60NmDp_mMa5HFuTdGSAAI9jP11k8KQUX6oGGGhx24w9seLaY98N_0v-cWsiNMQSnwR_SsGs6tPYqltHguz_azu0qsQuuXTQK9B06oEDR8tyb6CTqfX8pcumXIXC_DMFYfQ3pBK5R37G_oXTtX9srpw9vSulg4z52GhuvfT09ukMmdNGoIAS0551PjpZRz7-sI_nNTJQKpGgbhiH_zvA3U5hxue7fpAnQYXd6DYxR_y7QXSleoqQhZ2iVQW90Lwqp5MIDJaAx14bn27WBmQSLcuMpgnwpothMYFMmmNMdWYnGcQ0MIjhlOoykau7DRBFsLOKZ88y_9Pke7k9ISeTmArge2IdC1Ma7-GiJ90YVwwXDBSs9ssae8F1kWgyYV9rFxNbpF4uiWdQkVvASmW-QUNWzsHAtfuvrt-TR1SQ1Z-mMP_zF8mVjC14pAP5Z4pYkolLBinwy9V7DjcN0kymIM6fwpLt2h0LgfC1eLK3sutJcJJP9fFd8tTLIskEvUly-TeEct-syQebPxjxpxae7UPmqeDrOtvPi8-JWiHeIoJrUQnnw3ik2ULXvX1VFSnzDcKBAs_xZzdtjRlCGZWD-hgPPRTmG-YWeyovXZDp5Wv06AEL-hJlk4z1ZEt3yA0H6Ni7zE8jQ0_c6zJCWk6YtPhFk0ARZfjjdYSOFwJvx6rIrteH39b5W1yE8X0bm_cdeA1Q6TluBBkwv-9liCSOGT4ctzwaK3-cb0b4ko_apEEtpYkevu2ulqZoFi1S9g1joFZ5ooBLpxYGntuXXbALvq-zZniOJOtTdbpgsFQPS6Ae9kWXddWChNeyv_CEdkwXCkM__ua4GiH_Ce9WlqCzDCEoCYFpr7PyJP3gNg9Q_vkiLQa9V9bc3VtA5z4cjWB3rU5X9fLDZ0xzwO9krtGmsK9r2gkENMMu5Yy5BxGo94n6wRef0eMY6_GTzi7QsRuQSqNQLa98UdN4QGDa2c_-uDpENkMya7_hgM1z_RyUGtqVgpCHrld-jfSIGLPUI6kKWUDZ3USldXuep47KNuO3-BEOP2QEAKgHVlS9g2viG5r6wdeMl7Njs6iMsjs1KnHaqHlfZww8egAuOxAJjFxUYPy5djKn8n5lgPdk9ISeMZfxW5LcP80kPQekLohUbHcJ_JC2rTI76ckZvwuEGDUQTwGHR0B7YonoiTVzrhOWeqndwk3EBp0cr2mIc8vsWANK1WechMxunFVn7RuwV926PZhqFrnoep4ytDP8h4nJ4Z5zr9cXQCDv624H3JdPUYBBoxJ_7-QDM0fpuFXuRArtuezy6PV0a21CHLFtNq3DCWp56o4xgGm8x_8r2NtKTXxSwUY4_5cBHWd80aXF84Z42ldtGkAXayyFsv5er4VvWTzjwfEc39qkUGDQ5feVJb3YhfsT2qFyUnhb167hdJIPkI8rud4vLu3e9eu6xNLcw-LEjHptgEtfxOqiAPrBZLfWgkhfpU-encYtxg9cing8f_bAkf4-sP1tEaczGkdkMD-0orT-aN46m-8Dyn82fQgQdvov6n7KIuQipYomIQ3mJh5mSl2BAGMFlvLY297s3dCkBD3pGbRb6AAqu-5l8yCCVtg7FvzUWoQ3gL8FcP2cK_fYoJf7Z2YbgTNI_5SHiaAb-qxWuIP8ICEsxCHJJWIOfL6UnBXXctp8B_TiSbOFfGFrQPTJDUvKPyN9_mzO4mzXlOXLXu2VRG9J4NMSYTJT6-Q269vzse4SGqnULEUnpm2zQz9b9W97ahoMFYfV8xaVeFZK5ZU8LpyaN6v4mOJuuHm_vZuVircckh7UVVEK67jvRMi5JcKv-hDQhy1EmSRNCiZ4WHmGi7wcLEcJaUVFBRi84nU50Gjjs2kTslgVAnR9MyGqL2N4xvTAjoi4o-SCvDIvgWDnRCHXSD6ghfQagEUVGldGzk3EKEQF7VO5KTdheZ9FDiSXaaJJKit9NnohzmxM651VFC-AW0Ghklj52C5yvHScmJrIpMv4IjFAKj7erMRDjvYJ0v0PZDE0guTvoUFHrZd6umnB68QFINJogoy5GeT1hUs87OjZVQzPrxZqO6rzJK9m3meI2dFvdbgyAdbUx7fJRAu4yf2LC4dh0QaS5z24wuND3y-jHEsOvjUyIklRGeoH8EdGTBI8ZJIYKXJ8Ow797VYFI3FBzKNiPxJH-VFjpw0aqTLXVrAvCxwVK3awVAoWpwWMHN5yT57TOn3kpAbnBdAXG80kwTuOAAagePIVGrzENRGWVPGhvBFi55TDrQFXyymCP6c5q01KY04VU0udmOSe2Bwd-jMk2pjT3CLHb95G4PUVgy-l-occtk0mNRX4k3P9ETjeyOuA05c2rzMDthoHcFUnMqofePnvVK3eliJjh1uoNOrbx1rJuGsDZFEGxUfkjc5z5BW9zVw5YS7mlXjACPSDgMgreTTygsKTL0xhvSPsmu18K-cGz19v8ho7ix5B1WmPDsL75qXEqKsiO0ry1Ka23z8c4omngareIMqyM6OANeslUhQ7M_4o-OSaHUKQ3kAmJ3c_iPpedZUCo8GALcrgifqgd_ckfBRBpYssZhFQkxPNKJZhncuoRkdjxeAzANinaBUCxZ-Bg5DRQI6GCHgzUiUFMIWEqi21FF5UEiq0G2PM7PTE-RRO7wu8qg==',
                    provider_name='openai',
                ),
                next_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='image_generation',
                    tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    provider_name='openai',
                ),
                previous_part_kind='thinking',
            ),
            PartEndEvent(
                index=1,
                part=BuiltinToolCallPart(
                    tool_name='image_generation',
                    tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    provider_name='openai',
                ),
                next_part_kind='file',
            ),
            PartStartEvent(
                index=2,
                part=FilePart(
                    content=BinaryImage(
                        data=IsBytes(),
                        media_type='image/png',
                    ),
                    id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartStartEvent(
                index=2,
                part=FilePart(
                    content=BinaryImage(
                        data=IsBytes(),
                        media_type='image/png',
                        identifier='69eaa4',
                    ),
                    id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                ),
                previous_part_kind='file',
            ),
            PartStartEvent(
                index=3,
                part=BuiltinToolReturnPart(
                    tool_name='image_generation',
                    content={
                        'status': 'completed',
                        'background': 'opaque',
                        'quality': 'high',
                        'size': '1024x1536',
                        'revised_prompt': IsStr(),
                    },
                    tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                ),
                previous_part_kind='file',
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='image_generation',
                    tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    provider_name='openai',
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='image_generation',
                    content={
                        'status': 'completed',
                        'background': 'opaque',
                        'quality': 'high',
                        'size': '1024x1536',
                        'revised_prompt': IsStr(),
                    },
                    tool_call_id='ig_00d13c4dbac420df0068dd91af3070819f86da82a11b9239c2',
                    timestamp=IsDatetime(),
                    provider_name='openai',
                )
            ),
        ]
    )


async def test_openai_responses_image_generation_tool_without_image_output(
    allow_model_requests: None, openai_api_key: str
):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))

    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()])

    with capture_run_messages() as messages:
        with pytest.raises(
            UnexpectedModelBehavior, match=re.escape('Exceeded maximum retries (1) for output validation')
        ):
            await agent.run('Generate an image of an axolotl.')

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
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
                        id='rs_68cdec207364819f94cc61029ed4e1d2079003437d26d0c0',
                        signature='gAAAAABozexg98F3au1443vbazt85gDycVpBeC63gC3bUrFg_jpx3BYCksfaKwRkdFirkKpRHBYG1go0Kbzl_1w7ZqwT-BcPYX1Q7a--X9S-5OvU43ikr2kPbJgGhjWv1W37aLPKfIFOvt0VspdWlN3Diq6u67ktR0R0BXOFuSSv7WQmRAIl2JxiDzquSCb12Uc6i61LxNLpzalYk9lyW4Nm5c4czAsbUNv22AeTsB-nwWzO9NJyPfe6cVybz5gdShl8lKIvBsjHMonyiZqqzuSZakXSlbteA8yvZmOYO9XT1ruXNhCoFnq4pNuMHiZuOyCSIvASPUebLO9Q9uMv-t2aKku_4Xg4VsNUuQQpJfZ_gqYJyZFmANBTAcbMGBVotJS0V0ZkO9kyM9W99udKPuS-IzAkPCiHuBsjMaQzhM9xA2-uNm169O4P6TfqD65iCYAMLvwZbXauC6etI17If_731Oo0k4qcR5G7vruSd_Om-vwAGmb0u3_FNcI73bNw4WkhfGdUucfquaGIBkSm2_H63SOfLHOiLsnLIZT1QkEm2k2LP6O57WHAE3H1Rd1nKkw7GQcY-oQlF9_AGi_okM_nCNf7RXzfEocRYXlkSwoJT67TGV_rp1aSzhxKgzOzntDlcOT4FdyVY_0MMtxnBANZL2mBM8nvmqogWOnrJui-7qmFYP6DJj-GkoqPhi_gXlb9gTHtus_lQtLvJSXFhDlQDT40dPbU6T716Bp1TRQSAtj5s4aIhEupcAyoza4Vzko9WF0lrwyTKeU0Yh2nkrieKNqL9XbhknW9bp8dRMRM5-cG4WZHWgD5nLV4EP-8xcZc0ApwBLebiUhkX3MZ7phpS_ZpwnHxO8GDnvuPNT6pXbvvrUL-Vitwe-qWoodziaUetUcopfpGlBYnZb6_cJOVFUoJDxPxgOkLiUW0WSSe-NsU0XllTG75sVx6cciU7LbDJjkPU3yLOrJcwpE81Khsefm-yyk4BOMzPbhJKg9Lrs9pZmuhXa0-GdMP6fHks4UKYeR5pX7zTU36t4BUYFssq-p8QqSuXah64S50W644yvaJR52PzkjDhNtl_lBbLRsVvlimUz2vGuJfFrATSnw-pkuGEDznc5BfdjqNDxncVWpwu0u2HnfAXbzBBtgNFVvWUegjRhBAwqrlWdjhvTcfzp2xUAoaXcQdRmZUh9Cw1Ib3pvejHk8l56i2Gf5X0VrDVxj2av3LugJ0sDgahri96q_EfYVD6TEF5D1lz-GBFTWJ3CZgdICN1HM2ny8yNg60U7EWFNqN3Rn9wx1LnntWcKxmwLRfTWFr3k3-TTHGam9Nw41T8hSkpGmpDYM2LsOcqdGa6ZYlFEBstzvf8JmPtBxIlD-nFIAVLo0Y4lFmt29x8hi56Ybu3_xdwY63SMj0rydOF0bX1K1QXcIJrSY4joCesFCJ05vB9F39HNByztFA3cBvyZkwzpmif9juYOpM2qfZVtG8oCnaBU90u7lkIvfgRd2H8lH0pnVTLjSq885mjhxRad1pnD-0rIoPY7ppz-quELsMOVqNTZWW8mkCy6uKiWna-3m0THLNz0CNFCnIz5ZTmchoj0x1WewlSamGL6n_SnB0H_lIXq9RUZlwrd1ASQtiOnI94pBRHPXfQ3a4PfNyeZv9GOKHr4pR0uHATkCd-rldmWQ5kG6MbNjNMJChaCMuSwrJiBondTeGIZswsMwOfV5eHgoZwfzkTxD3xo91CLGsE77IDjp05mNYc_KMbEqSmnDWpPJzsHWAX5vahfVXsRFx1Jjw-D4d5PIU_Mx4mwjPLjTrWkd5Hl2uKWvjw0Hb55lHT8D3cGlvIjRnTGM6ndLGndeKjXobs0NBcmgDQNLpHMccHAbb2UVIdvx8cF3zhkc6FdKJ17TZGJQbzWSqkkKwNUCYuRrf6CeHoEZh2ErArmcj57QVdXeDREoSzHhu1XRBfgc-ey4WF2RMzJ47Y0LNxFHpLxEk4Tpg4jcMvfRRKv-8tIu5kZeVSBqZTac90ly9qDeb32mv131YXy9G1A4RCLhjCoxM9M5r8a3AnDIvoMCusPxAM5mHpLEdv8A_5mliza9hztzmDKIQq7sQAow_XafhRJfX0dOeOft6aLkLTaq3KHN1HE0xy9Fupg5ut_AUd8tIvst7Av6vHPiK7gR-rxgWNx2FmKKXhvcWqMeeqFwSyq_18Y4w3ibPc-l-oYXD5gEMAaXGa6WgyaeNBSHNup7HDYlm_WLK3UdH-5IuIMxkhLFgDFED7g9lFl5V0I4bmW8Vqg0AqmNC2hkADsr7mNjuYr3NOzGCTsE2Ed992VG_e3k2PqtHVegZ41jxZTbfqYRM_HYg2rOdnzNN_NV3w0hpKmV-F6UIrb8Sfpscdwa2kz9Zz2huG_QGzI6AyjoepNgOYYZ_Phq4JMtrjge8HcOGRR3z8_hV21N61o5Z9yBfwx5kSnAvPXuAgeuh2MIfB_koOrwbEryE4HJTSA9b2IAs3BS9cvAu36VHVYP7Cj-U3EZxaxRIqusjNHb3WIzswa4AXFfjK65aaCCizHwxTSdn53nlSe_f7y9pMnOV3o467Vtt-oR9dqz92pauT1MxXYdBD8o0q7iOgHsqdcHBDvQmWOlpag9W1sF5uEyzA9bxDGlKLA9xtvfwnC2Y3nv9i-haFaoknx8eQGsM_bzWC54o6VlhJiNeeRUuTNEAsoTXv13usbK6MtSXoGvWPNDkj0CJ4sluSWmOidGG6DW0bEj58HYCedg71EKyNcrxcN7JJFklv1ju6IVY4k6pLMPu6kbYdKzXaxFWkOmB27Qz0dgBbK7hHhHagJWjOKrHWNiOphce9SHf9xQP2U0uJRNKqNnqmKRZLBQ-QcwRUxzaFrDByG7nRV0ybRHpJvrRfCjq-FVF0b8JXudNY3D2gI4p7WJeBqk84bt5a00DzmTZap1cEYi8VYs6zwFRa-LTemxgq8jOaeR33x8I8tDBbnqJdOV_o88Mek2E0mXkKfl_3ZzPzajcjyNHYdfhNHfcB0Oeyk6w_TXnVxOJVHuoYUsXuB4gDDMENk2yubSUFjOUzuZhXXZ4CdE4bymK54tzj9qATV0c7k-YsRlfUk5fX0JRQF6iMMWl0VtVgi3HolN7S5K4Tbs9tB0ByOS5H5N0D2BjlDl-kmd9Yrn6-B7Dxz2rxCtj_TQneCyMzfb6edt8X17c-iLsOV-fZOP3mFTCcnI6HyMxaLQDENgPF12gW3mDZyX2gAd7iRFXCHzmTnVkUEBhlCoIbq5Dc5s0niA3Jq9MHvhtgcXaUusGqCxOtQ22_CWixDntwS3hQx1KxEUzYZxThNSv-f5ji56iiWOB9uVzv6XLPdGYkN5CgdKzqPsim6VWfz52q2TTSZFRuaMGZsJuEpprdcWpHcHiBZsk3Qa-giYazywLHxj3Ph_Kf7kByYuzspVpGX3bct_unkM8jQzg8AWkn0LuFIY8ScyzXAVECJbFiOv4HipzyoyYcpDiPmadV-qAG8ghpw90xwcvgrGnHm5pzHQaqZVPKWXrHnkaJFgQ7OyEpmX-U52R5t4PZCeWJA90zAnvmL7s5MFEDLzSlzXPFjzUo3T8XWz6p6tUksLsGdUUqILT7hYdWwCdDnUJYZvjysEVIkNPNx3lAGW_e82KYUZUjQqiQqsbu1meJkdZdvMPmOlVEGx9oFvTW8cZrZZfJ1EKNOPR_1eyGdh5ZOfxGU2sogpOST3U6dxdeUwpEdhl1dTaZA_chepczFdHI7aMoKSISbpNC7nNhKNfQIpBcwPLtAhlgCOQKuBRMJUPK7VcOpl_nhz3FKexgJ0X_hFqN5qygPtKITZ_Et5ME9k5tzelHVAiosKhWOLngFJzKZkYFavSYUOHemVmUo7licjeUTPAb0d6kKbqqrJ44ZP22gL-Rw0YZa7t0VM7LrD4jD_taPhqmZ28CCchPAfP6Umo6txaiXDusvPG7v-zUGh4XWQlV_5SBSQwvPHUfQHlogHH69jxfJYRD3F-lDIfKR9ta8RuCuCdd2VsBUUAWFaRuulGcLiQ1BFWZ2Fx7W6rXgW8QkufqX9sp0zgTK9fCPN0rPLLU27BUwn0EB3EZe4bDBYrm9xAxYCgoOOE28GMpZsznXXIVaAXvUGVTBnXks9ycNghOw35fsAniH64H8u7Yp45JH8BMb7sk4zHcW_An1TzmUxjq3_JdOo9kkzNEu_qs0Y9btf4M6NajoVWlttXT2RD8XPxIrASjFMv8fxYJG0OsVHJoKTnGFlHqy13dIDMtfOkT658EuGe_pHwp8B1zI1jjpPmMW2MG3TAlumFgy9T0iyw0V3dp6qCZCjavMPIJ2HD9V9TJYueufIptuN1_hZF-xESc1ENH6z_NtMBJmW0hwCU9a6UqZ-K3_ZsJ7EK7q06V53KOSzzaJUY7SgUAFgwfbqFpvqi8emyHMxxIFylw_E1iauYGssBb_tzY1fTWwk9n4h4A9bdKesz_JBNRuKmKyOeYOS2LdJGhp9VOsbTvJtJ3MvwXuiH9rgFTXzXbDgIHxSUnUkMo4mJhnWXiqENnfqc_oOgx6zkicwwX05Bjn_k4nj800fT4RkJTY81dXy54TbvqBgPR--nTkq-WJD0e8_9hjOrtD7c7_p08FhVGWl_d499ylunVccNQ51zfeP0uGt8uYK-S9fNN8ED5PJ5iBycLRU0uvyFdsZFS0DJ4N96xSrdkdPAZ5CjfIn4bawnCs8MrHg47sCgONAqrZ1WZTgyL8j8J0kdfVJTZoLRud4oFmM4t8nQsL1sxH322EWVaXsy9aWAMVWGcc2fesk4jVODZ4mfGtnvm-Q8ifjV7wqRa7s5uXv_-JFUawxNvHqMw2AvG1M0RJTIEnRkCKB_8CXNb4K-GRw12XtZs1KRED6j1yPzLwabcWNl2BpR9JCBmogxFFuPPLdevD2JWptQ3YwTQSIMridFriwY8g31iFMFrNub9Z356Vc9zyMvZ0sRBB4oK6iSy8Y39nU3512YdGpDmMTH-T5Cxy3pi1dphlbYHcar1Yl-n5gSS0_dezGILZbxfvGbxOAIL-AGwSE4JJpUFn6N0xRclaNTPDk9VkKpVktgJeWqkWFI1xdBUp0K9BQdNs6XsJsfMCaIE-ed9SbQpJRdc1S7m6lFKFZFvHujK1zplJemcpOrYIpsuTD8wzWJAxzWwtUoo0DXi0YQOrnFdo9PbcTscUGyNArCcGjOYiL5_oMwhJh5kg5L_OPLH3UhiLI_aHs4gT7nX__HHm_3GVOov7-kK_Rs5lGsZtFQ-yoau74FnzVn8N3YWMqWzCVbwGRBhvAm_fdI53h77CZeJm9yWZkWTzrc0Ua2HqA7KB-eZgCNx9Ox9n0Ilh42hdGLFSwOEF1oEHw63x4ZTENyAP7j7mHWnyVbqRVSZOlP6w8JquYAtS9hUJQb8GZs2MFuFcwTO8SX8gMWWVM3tGtpnz7UwqhcFMZHrJ0ehues8kdtrlqR2BWPYAquFhOQebvqhlYxieuLggwAIxudJr3PFYJrAgAxivKZlhKRSoa1HCFSp_4k_kuVp5AvfBMViLJ39pi_xHX8PnokEmi3_GZiz31hGjj1GgNg8HZCkQtQJcOjMA3YPE3kQHy76YIdxzL59EdWsfuHj9IhJ4J_PoHTg5k683yXkvcXsz_sOYj9j6eMNn3Krp-ZFmWbtmqpWuvKS7vmE2Wtq9RmzvkGPC0jeY6J_ndbyCCljLpFGbvSYC50ejEulcMSUw1ypHhHzoDzO1jFo1xBDfta1H_hAsLkmYKOseTZce6hHIZR5xUwWStmu0y3OoOeVT3cZXEErr7qPLxEzB6nTlSgzLRX-ncRCnnUy0f_pXWwBE9dE6w1Nq4Y9BuS2Ip9oZZvj-gRCxTmmJLL8RYd-G9EeiBSlf09kezq4plUsJwvvPqVlj5CVLk6DkEsTvAM5r8DK3FYnLUgf7hQI_EZ8P9VN_rrPTPa25y19MR1-XIsVGLrVPmyXcj_w6XLBxEc1J8qQ37NZCw_DRfS0_ZUUqSDNlIy-y-DsdLIVtXE-BzpUkgqx5ADmbsiFFj87rS7wE5Yi2-btSpkWdTjaBbsWBgN3sOvcZgFt-AQrCK8BMLENwFrkFicExianGcwm55IlO2iUvCNjz84ogzU0IYhzDbHGCZBzYaLNX4JIVeonOoQuKA6QGueCffX1MKQ8fDKVZQc11Pr9rwjPNR3xxAs0G4ykW0c3809YE60dLbP-AZ8Qm6vxvAmTzd3XlMFaC-KbBxskXtMJRJqxUd5jYx-YQMQiOA8wLytP8INXSHi2_lgMI8QivRHoKx4oAl-qTx_qz0Ut4iw192OVWoJqTjhwIE_NB1Um9GawsbM9gb8VgI69RTPmQi9obao9kkQEUU-NrNtDkumrltR9Q9fXFyRbKn5Hs0PHzC2yiRFa_IqGEqJeqz2CSBkdC5GauukZj2XumrXKKCIYV2vJ-42D6MUqD138clb-prSiOe4dA2-wrKPjaaWdpaMeAbLXbLkHuH4K5h6HOpYIVMu_JWdnzzDnvye0o8A8vrxFZsOSDeiHPcoDyN6-6KYQwOlKpjiswFVdoD_e3Aw9ge6YHDOggsKeyERzv46Hc2GwW_GWrwaWgXSv-lePsj94L1b17WG_y4oEKbNMAmE4A4EBH1aJI6_wEeySONFKhbyOIwUBcRKCoKT40dbGie7kW23rrzSTQHpjdd4NbAbNAGEtbAJbVwZbFC2RMwMWQbz8mmnMSdVw0VN6UVaCdUatYzATx-gk-lg_8ikyOhMwA_lXFI9pfnHIIqGmKHDJdrE4pdDBK5wVLB1H_CYqzUneygueeTUc_rHiMj7QdekTGh4KnA2lurqXemNemePYl621G_FqimReuPlTPmn88h-DvnmcVhTJEMCT2O6huU2_6yOlyulJw-mq6KMS_b3PSB3bgYmJEH7EM7ljv7lu0ZIqZm9xbpVTntP2s9jDBPMb95q8AnllIwH6uZEYdyiIDU9NQj9RIlR4RZ8Oi5b6n2IuKojvP6aK47npwKMvC4IOAMmGaE3_S2X0vFnpT1djiFPAjvklOEA8o0k91YV_z-QbWwVTd5fdRgaqXcqb3dLAuvnFq0bOpJMchlL9lrC02JW07Xi9Lw4EIoDsPBRr5s5DWLG8kLfpEv-YNZatryAbjbwMtJX0O9aNxLzG1bzyXVFwBb_HUgkotdbmlT7cXMMX8D4_xIQcDyjwo925uAVU3va121ddHt2JasKJc1vMyuOKXuDg5F3dYByqEQQH8zywgBi_IRtUx8aXNoNXobUf3LGaLc-3HqVTZby9Xv_Axm-s5jSqcEPL4WJX60rY28qx_lmajcmcVLY7irKovGmfEIlVeHYdMxDxZue2nHHHc7NmF7dGORg870x5iGUcqtmjvOx-NIqVShVQlalGOdIF3_u6r8xlvOL8xWJ6WLXKNaAFVtaoDdObEoR3stpG8iaaaGNcrRqIH7J86ntBUmQ8chIgOOvPz81W0Op5jbSc4d2e_qAAuvt6j_qElh9qsGNMVPoY9DlxqqrlJ3ojG_6yqlX_jP6NGSx31vxc-IgggbyeEm2BVt1v3dJv_Z1Kmt8PGa3--atQsH3cemYMbWcwX2mAdMCHLY7xQJs_X9OgDaPd25bGPHFJQVNM4cypyPSWXEDiR-xIqkB7NzTrSEGf82wSgIPRAo7k-CBpbBVvsuLre4zof8K5u7QH2jYFQA0-9YtDa_Uyc_JcErLU7AroB5ZZr7s_s9qx5w_wJxMNW1NLZJdOKUVryP6DJPqTNmly3PT88GaPUDdrcQa7NplSF1FcqFmzOlZc1Se_fSkgEcFKrX9M7_65bTgLbAvj-8bCTIxZp0-PcmoQO4mrkNkW2Vr1rfSuOH5_vqJCT7s9A-H77RjLseMxJ27YF_dEXOHjsEV0DPM_zfno9_Lb6gL0rqffxhKdM_xRhSsLg47PJlCgoZkL2xVTJfahwydAzbr4ZWbZ84WwprOimGLJBdcQgjHEhxx0_cT9tFpUKkUbpAlUYqm1e23JGUXKrS9kBqM68RjcBAiaAgeKMx_5pNU4RzuJqCARPYgqhsy50LxkN6DgIMiDglLxho4wUYHVVkez3XXPHmO8FJcRuuk85Z-7qDeExR-Dp8vHkAvSGuyuUKHdyaQdhigMm18qFzu5-L_m3-C7xEblO7KJBDtK9mDdaviyLU9bQA4i7lLgUUdkCClPjm7yDt3SgjDZKXpBeD_HEb134vjhc6EVJN0IK0bf-lSswaAPJYcGc9ArU_mch_F1w0X1ISwvFHWKLYBM1bUSCr7vLyX-j-sbwroVSJ53bTedlaUpoVhEjKxcksjSgnndvAeeHAwHV-HCkOH_zqhxp5aTO1vNXeNESG7WoSRLE1_6vF7ZzTGmpG3GIbuBexi0BUe95UwrAJpdtzWMrAiRNQ3LlB8JkPIqX_ZXmHsu1bo84_x1u-VWQZFNTOitRXyLdA9GtDqg2q7RUJLj1Axq6CzjcewoW380K7r95Kimr6RDUUkvi85SjekZuV8NNd328heZWNSdov9lzmJlCokX6N5VDpg-NIPvjJyaQtf6YZfYtOk7pp_koGxatbsxZTSVh_DwXz7K8VfC2mEhr_xdDPe9nKAPrU4r_J9zdtAkkzdJssMWhSCMTj8z3l1bbIzMt05ivwEGLZYqZ5Sy2-duQVd1wA0NfKB2HjfgSyYhX5wN4aDW15copsEtPTrxCLidwc75rLdoi5Ch1Jt74v5UzKh8mxDaGqjdWeHkDrHbVy5hDVk00n1TZ5UAzWCq3lge3717y0ECZX992BrkqjffUYe0dZUUJr_3GWKSS0AZHg5uI-1Tka_DGqF7mWwdz7XpafzU4siolRkGa3QYmB8LWqdbAnpvte-uSwxAiQm2LiFF9h4dOO8U_2gNsniQViwLkD0KTHnuk1_N94P6QsypEL2bcPWCvCw1SgMmgKBaXYpP8FIN1trlpqyk_0abqnq9GoeCI2AjAOkHI0LnNsM8lTwWNqh1b8YVco3or8J4dOPeVQ=',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_68cdec307db4819fbc6af5c42bc6f373079003437d26d0c0',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='c51b7b',
                        ),
                        id='ig_68cdec307db4819fbc6af5c42bc6f373079003437d26d0c0',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1024x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_68cdec307db4819fbc6af5c42bc6f373079003437d26d0c0',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(content='', id='msg_68cdec605234819fab332bfc0ba35a5d079003437d26d0c0'),
                ],
                usage=RequestUsage(
                    input_tokens=2799, cache_read_tokens=2048, output_tokens=1390, details={'reasoning_tokens': 1216}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 19, 23, 49, 51, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68cdec1f3290819f99d9caba8703b251079003437d26d0c0',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please return text or call a tool.',
                        tool_call_id=IsStr(),
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
                        id='rs_68cdec62725c819f92668a905ed1738d079003437d26d0c0',
                        signature='gAAAAABozey1vHMbs_Q2Ou1f-stI__3zJS-qTasRefyc_eyOxqogM8UGbPpL8D6PLFHcypshJpa9SQli-qZRIyG4ioUDLsKpBwFbfjdIhps667-8st03DRTRP0ms2izupS0ae6QqY9qrsPSchrSF2o2PlJOWKZAFJ609S0hGX8VDtrU8nESfp78NQ5HgpgXksXQTxk3_xRmXES2AThlUD0LYykoVKRX-xOyPQsOK7aDEs1CIk3lG1meiXdtJxP1Jm9JQGLWk6kePWUgwnAs818LMVvjcj8GWzFjxKUQlI3S855vYngivkMqYqh4gOcDRGRWej4NRzRmhOK-2yrATl26qnpRwNA1YXkFtn1ojxEkXD99P8RIXNItH4KW19ALs7ZizQmQlKzd96eyPT16OSLqEIfHAXWEKwoB2vTM2ExvHK4il76X9XmgRDy_CI3HAPI-7M3787MJBEY3z9cBe2sIS_GtSk12_GXRBUREhu8wcc4920FxkufYegHd3FzKxBRjyxGpR-jLyI24ahOZRKvoXi4-n1v4umoD5OSMjYpMtr0ykwIBQyyqldi9KqHBpJCzB0wA3JyAn-4JvQsXwIeeAtq3bNSJFaaf9aLJ5OwMO9I6IIWGxoQ1mzqmCs5cVwwjeJLzEc0T5g2qWJdXxdYmjesvMj3pJtgIq3iR2105LydhUiKE-0VLVAQGg-lnjkCtj-muEqlko2_FCHQ7b_hA0VkOUIKOYUDHRtwgtaeNnUpiWk8L7GnBNHtVQ7_kHEGj00UIVC4CKiJqESXS1om73Xt1K1-bglZLfSKfjrAd6E3W51cKQXM7KOfmpRwP-9DThdeOBgjlmMFveru6NYl2ntiu7GF8JJAvjebF3Q6SR4AtFSp8zrZVjlduW3hzQtKaROHLBVgH1KaMST-Nfnkn4AHCbhYNGSZxg4J8M3hh-BLba-lM7o9d0cHsHSORXeuAg40qioVCZNCtIooo6fAdWSAULw-uGdAbRbrJq5nE-w_Lilyeb2mWnMwMbKBHjQO7Kwe92UHvve46vMkiSSX-wZYwthfbO9BM6_ha5BJOtwNggKuBXxqMVizL8WKdvdTVwzP1guDvuVgVoCYKl340jB_EE2V-L-YzbSdaxHi09Gi6E10MgdaSGhNqUJMZWXrezkT6pyRYYRIWhaaImIuQf6JybMUH5hHH8DDEKvofLnQmgPVccU6womuYosIgLOLPOetK6OEFlMsQPBn95hb6jY5vETMyhiVYAVxaeXgk5WA-NdYQ-F2q4kQQS6Ku858AaO9rXMYpkaJ0nIubIbdgQMaXzq6ha5Z0BaxrJhxCDqmHXA5-INBBqWDw0AgJcAlMM_a5ShZA4zVWu4ydzikq8PlnATLbFzOkL8WBhJGRyveKaSyknPfPCBvXu_1l-hwAyvFv2dWWBloD_IIJ3ID_qtjgT7epwC88qmQd5ICPMMdx4DjEu72hU-rTIz30ks970qi_dKVbgpsLAJbHeCCWXBJzJ3UKrC6sc7Kj3rhtelPNEJqFyuB_EJOVX4o7R0AZvkk3wWs3IZ2nE9TeuYckw-hHg35YsC43QSrbIZUbfonaN57ZbwyrnEwnhH3oYXaMiPqqywH0CsZYrO0QLuAeGJTknlpWHgLEVmz39-e3UZFb6WIGIfSFaZHAFpmQBiPjae-qudcbvvfmAgBHqt-Aq4D_06ySnELOtDWlFusZcmHQwG9b0OCDXt6KRTR_-49uuoPCoXlv2nKb2eFhXp1gp6m6WsH21XNaeLU4RF4PR6oRUh-TCLzyBtwCscukF_3gBvcTdwJ4io0Fu6YVtEJux_Ec1vCaQHlUVGtZR7JDVyE5lu1y-aZx0u4s6HheF0bHLYaFgqgOmaNNWwK_jldqp99ZhU7Qat8GcG8YLLEJ04WDIp6_i_Ri7OUf5xgTEkAxS9gOxeJ1EMKR2oB0pf7YJ18XkNqeA7zxXmufJNIbEmXOC3XQKiDY9-2UzTyqjzdZ4V2naUggs7DjAAntcHhVFLGOZgGeQ5FLJ9jfzFlpE8mAg95ZtVvPzYBNFaPoTynqUlukCH4eje_62w_u2TruBMSU3cOV5IqVTLMHu2uwxHWdA1zrVi32LMv8FEYZ8nPyyk_BdapV-SRGQKn1yjGml__I5ksVlqNWVC3BX4hkIxH9K1bO6HjWP2-cdRtTYtRNcOOGOZZv5RtKfpvzjIK5o6d45KDK-jp39_cY9Veyawzc4XwT7jkyL8U0YNsRTEjafcPONK7yWasrOIzNuUppBFdMyER_R8Q1bTMQp1sE-NoAN-0MqupZe1jltzga6i6KLWuOXtMm1_DeHHH3OPNq-kfVs19gbQD13R9kMNjgu8FbAWdoVreG24tKUVSf2nWKReXwxc3WiiODaYTew2ynZ3BUchm3eKebybh4lKoYCw6lLoclnD7smFkP4-72RfaTylVe_npaWU_kWIhxItYosjWGc_ScJhLFdwAOUaNijGNX0TmeMb3uaESw23E3Y0M2pAC_wVkaHvJEYv_nYebwYc6yON5oMFAnOmqaJ14d-LfhuATtTrqD8fGXTPi73rpN5IpA3fVklkyUuu0GWqvsRujNnEcN2nL43LPRwFvwYSeL_tXJoTpTxdgkwjEg4dJr6hImPIwk6Yu_a159LyNKIeZOSZbHi0gao7OeDiSM2_1zb_srjUXgiVPJ26r3GCz07dhcwqUS8lU6nx5q1ncCt9mjTgUG0qmVPzDjfRmeqOU1gPZhh2XQeiXp2AWB-_9M9vu2EiSeOO2gfYKeaFjVhBLbjOeKs9r0xt5JXpdx5IDB1JOfK_MSlKwjl2kOlNo7p3e7RIVB38MQoih95hSz5E6EyOoH2O6KAXLM3qhgcmXe-XJwQKOkA8rPRclgvN-hLSopl6mDGqg7dNKXZB6SQspQPiB4mx91Wwt62aMqWa7Ve-thQ6-oCq9IQTLT34xpCOpHWz4b8kSfr-WDg9cTDdvYIGEDGj_XlE6CK5PuxzezWGYvYggwVJC_65zkGHFBURkBCRHc7otAyxMTKMCTKZnYWUdUEGpTCBonYFPOlh_ApDPnh1h3feF1x8AEIoVkq_ADRKhKbs28y44LQGp9pe49LdsgKN7YT8_azHknz5frvCUQJu8aiP_TaFCAiZhUc0JSATHPv3q6ZsNoUzF8ZfeSWuI3ZpQyoyMq4wxcyH1sFNqtIHd_iS2U9AkEACp6q6FAohK_O7GaDr_T7VfeO3JuRL9icg82WtPOtgJ1o7oqZxa1lfypGkWAgX_KF5aytblSoxlwn-Zk84Mf_YlbJEBO2mUa0Me5uLhM183akeG4y06FR-ANXrsYxrGmhpIWfl90WAKEG4ExdaexgzQOKXedTLOKnIwDW0ZSAu6O-Eiddn2w5GBO93OGEGQwkXDddA9PB9--mdyHgJM1OvXFYIWtziDLZ5qHAUrUFpYUth9B9mV4TXXLeLqx25TO7DL9czT6cGPeYNSOR3HpNdw93Pbro_kDIk-4zrmhlLd-I6w1tSisK7veJE0Y9Svvf6VLky34iV0RlIw-Z976oPP3rVJpSIujNbuFm84V2EvHFkG1oaHoViDxp4QesbToOdhl8GfnHHuwPaT78C1rwucflil_XkLAi0PBv0uGYLWaaAVlhm_lNZc4CYd7qHcdgma8dL27kHwVUJctIzphtV4yAhL-06SY-kwR9snNogfzDSuuNTVHPofu0_B7qA9vjwf3jeE8WJUbq0HxPSIGMSvJ57uh2YzRPWnMJ2PmARWgHAYoWUsZknHyr7DlKqzV1kxrLmr40xk98D9-Qkkq3enbEazhNWXRTIGMuWQbM1FQB3VL8GX3Xa1TjujWQS8_nC9LTId1XzNWsSCNOg8gHXqvt5hAeJVZ9GMTFJREPCBk5ts9odm3N3bjK5KgA3pNahnOS0u_c2auAc1t9C_5xqxEnIRxE6R1lVHz7hDTebYMcpldFI8IEuToExCjUUpoW6FmpeJrv16hwzTqbX69oToXJ2YA4WYCerDob0BAMHKWDC0nqPJTp9wjq2OnC5YKmMKfJ7lKhVprYG14RKnJfOJdChRpWQH2iK1Hc12sCJBJT2Lvm138cQeRApo2uUTs1dpJ-faoPvb6H_Wg-9ys5JsZtrzUXVpj4ttC84M7dZnbdyEh64iOoUINTSG4yryVDfN7PokiuN3DHuuXlWG8hkWAP4uyZg6sXdeSBGfVOjmG7Tc0zNqxZFkCMo9XEaLk_Zz6X7fQPM-MQVkSrb_0-S7tNwcufe3Y0nEnKpM1WohnyojInkRUzPWLUcBukzYpwuXg3MxxayDLwfjfybNb3hO_aNF_5mLz8s40IotW2Oxa1eKfZO5igty32jA5ipXwPjJjBj5yVgmTESDPmYrPkJCdFUtJJpGRTRX30BbLAybWkhJWBGilR-f5wp7yHp1pAcLl8YIKi-Di35kCCG0qtTimMeyb6E9hQww06pdDKAuVuDmC-RhgG9-x6MlmeJgKeNu2VeANq5lIiAKuDEXz4cs-TBrC_PC4j7dnaIbvchVdxx9Regcqfa2XTeXsKJB8uGwHDBQ_qu6zyKJ2ANyqk8x7Vmmz3YNREGNgiiRQK-zDjU2ZRJN0DJnHxdFMYcmywKz0QOab4orE01eqaG_Zy3vr7fDLOMJOl_puAjLnF7xiVmj1UWC2FwoWngvCrGO-2wsHB7fnvJapv-NpIoW146QuSVwhMROuZ-5SmzQz4MlF-ARtdkEvCAO45BZTM-AnTlOiMqMXspKrHGCiwE_S4FStVdHT4D7m6Ha4q_BRD9nBYp9r0vP4QdsfsK0pegYawDfZ3U9Vuk0dRZp3Z2QBXDLuIs0-0ol7_liDeC64KCdL2zrDSQCXeSQGtDpBdbSDf5xkzQ1N_IStAz6dI_FLbLqRPfE9tf3AfA9k7j3BwN_K6eeVfRueVnf0VOXukHURPwMTVE9C8IdPA8OnwLBD-k1YT2aV3Aow6eC4Dwl7cW2FtQ_BuHcpjMfWuJFiUuw9L8bLCWSlJ3rYdX_MYv-9mY--9d3r49p5FHAoK-bJS5yxHg2Q6Eg6DrSC4dORU5p_fKxrPklgOJKbZoQyb96_dp2qkCVgPjN1naoHfhgqHp49MxalqSx9n-vd86NTSXNw22lnboj4qAsLg54P4RSayO-U6jvj09ZuGtLGrCvKPAgHl_xP4dtpdbOC8OJ3nwHNfxAgZr2P5JEaFHN3RuIQGH5gkwNEA2otd40YptNJvD6r4gasyBPNY15jeYYL18yFQobK60gnXQweT_JXTlnGgbvgoGw6rJ2nifXw_o6Q_uBDQE9MOCJiN0FoivBdcwk-NHU-nNg93U30q4zbGbJ96O0E2X2qRyoOaEhBKAqkHrQM-NGjX7aj1YJFCxIFtQ00OQjT0vhGng6-3cB01l44mGdDFV47PZLzcFrGx7iDaR1Oy2Na5rOLphgT1_zvVF2mCw10HYxOYwSDnm6uqo_kADX_HQAGxQx1xDZ9iaZZGD00hdoZp55kRPGt6uZ9ZEEXPWUpAV8mPUlu0DcK6rr948_btFuMWo9KrVneCiF4qxkk_Q56aFoRa_HQwEEuh3sp9PSCT9wIfBYSR3661Ox7C-25rHNkrdv7XC8ArnQ7xT04y9JYOF_XLxtOLlAzscUryz56Qk7lexv7Qqe80fTTSRNFseDNFCxLjWm6wAo_kpaTOOvFiPUxlYwcmW4hy-hW8zH_JRVPaBCCKR_s68yEEAVve-8q0ePzqcElS4Bt8cFyU2kqodDmzzh2HAFcrA_ScDciNbwKWeJublK97rkWmLCuPZwJcvKNT_FfgYU5OxtF6G2TVVhUcbVAhPQFFegVn_8hYkZ3mJFkZa8FYYNMSGn8nssR8F9RInU9PFLfBiI2_sDJKwfhgLZ9Lb4x2Idx1-XNh9pAFkCv4eRKCoQNrfxZ4VgD7L2UeLTiTBOsH117rSSF3Aq5kaLLV9yqXc7bB8ZVdvsQ64ET6Z3sNM6xk78phFYy-Dnl9sM73dX2qpUefxdItLtql4E_jwm-D2qRdGtdkm3FpQhYhTKgfm--H3hj64AfDQP-7WVykl508pb7Ultl6j9ne28UqtsV6LHaXX3Raq78sZZl7pmkgp3dSZlZXuo0zwyh0eR8aDp201EdOqbG6a8Vs7cHo2pjy_31ZrIGA_rZhFRN5uVohUvtPWCwwV3D6qq9XQRFK6GsiIMgO8oH8v2AB_qopEXWkStyn9HgA-UntHqncuVnnVrprWFYfWVuTDkJhTC1-i9ZgdA1eUvZVxl-I4SfmDG540HUVVRw-Kt15vS7K2vc1FDiQA91Iya8eMjVKOs96Cr--cngQ_zZw062ZqKp7avO0EOTamvz8Zi2a0XzdIH5dH0_9_kXE_T4U43Ud_29QMElOxJfxt-9p9lBNBjfEkUvBayPX2XWlePtoQL89QLg-Xwe7RaGu0hvUiROaArk47B_703dSKDll2V5eUKc5f0H7icWyqp8u5rnjQsafBu6RCWHr7YJ1Pk5TBaw1qQCvNA2Z_FVZWN18No61fV16DHyjoiHBBjTROCS7m7_3snv4SDkoiFvIswpEt7pbYtbXLp5t0EQOQkWQOITojrGYIUi9c23kdiXr3EoPsvSMKlJcDyag25wzVNfA-bfiMCkdHPqaaTFBqYqlwA0SI6bwHKhqFKdGkrJ5YIxgaFLsLwwiNMSBR2wWXSFRXz-HCMewHkVV4LpGxkzeihWXJUO3gXeJuVY1EGxB8U15BUtQQcAAe2Om9KZtOsS2kRtO7vZq62E8jl7bUbTmw0XTZ4eh3xg-IRiK1ynur6XqtZH1kNQFq7k60X-6cFwDS7eDGdzrgl-Kbl6VSzxpwxu5Lz-KIjV5gGBUcOjnIpRuwY_s',
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_68cdec701280819fab216c216ff58efe079003437d26d0c0',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='c9d559',
                        ),
                        id='ig_68cdec701280819fab216c216ff58efe079003437d26d0c0',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1024x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_68cdec701280819fab216c216ff58efe079003437d26d0c0',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(content='', id='msg_68cdecb54530819f9e25118291f5d1fe079003437d26d0c0'),
                ],
                usage=RequestUsage(
                    input_tokens=2858, cache_read_tokens=1920, output_tokens=1071, details={'reasoning_tokens': 896}
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 9, 19, 23, 50, 57, tzinfo=timezone.utc),
                },
                provider_response_id='resp_68cdec61d0a0819fac14ed057a9946a1079003437d26d0c0',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_image_or_text_output(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, output_type=str | BinaryImage)

    result = await agent.run('Tell me a two-sentence story about an axolotl.')
    assert result.output == snapshot(IsStr())

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='f77253',
        )
    )


async def test_openai_responses_image_and_text_output(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()])

    result = await agent.run('Tell me a two-sentence story about an axolotl with an illustration.')
    assert result.output == snapshot(IsStr())
    assert result.response.files == snapshot(
        [
            BinaryImage(
                data=IsBytes(),
                media_type='image/png',
                identifier='fbb409',
            )
        ]
    )


async def test_openai_responses_image_generation_with_tool_output(allow_model_requests: None, openai_api_key: str):
    class Animal(BaseModel):
        species: str
        name: str

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()], output_type=Animal)

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(Animal(species='Axolotl', name='Axie'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
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
                        id='rs_0360827931d9421b0068dd832972fc81a0a1d7b8703a3f8f9c',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_0360827931d9421b0068dd833f660c81a09fc92cfc19fb9b13',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='918a98',
                        ),
                        id='ig_0360827931d9421b0068dd833f660c81a09fc92cfc19fb9b13',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1024x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_0360827931d9421b0068dd833f660c81a09fc92cfc19fb9b13',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(content='', id='msg_0360827931d9421b0068dd836f4de881a0ae6d58054d203eb2'),
                ],
                usage=RequestUsage(input_tokens=2253, output_tokens=1755, details={'reasoning_tokens': 1600}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 19, 38, 16, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0360827931d9421b0068dd8328c08c81a0ba854f245883906f',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please return text or include your response in a tool call.',
                        tool_call_id=IsStr(),
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
                        id='rs_0360827931d9421b0068dd8371573081a09265815c4896c60f',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"species":"Axolotl","name":"Axie"}',
                        tool_call_id='call_eE7MHM5WMJnMt5srV69NmBJk',
                        id='fc_0360827931d9421b0068dd83918a8c81a08a765e558fd5e071',
                    ),
                ],
                usage=RequestUsage(input_tokens=587, output_tokens=2587, details={'reasoning_tokens': 2560}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 19, 39, 28, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0360827931d9421b0068dd8370a70081a09d6de822ee43bbc4',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='call_eE7MHM5WMJnMt5srV69NmBJk',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_image_generation_with_native_output(allow_model_requests: None, openai_api_key: str):
    class Animal(BaseModel):
        species: str
        name: str

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()], output_type=NativeOutput(Animal))

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(Animal(species='Ambystoma mexicanum', name='Axolotl'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
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
                        id='rs_09b7ce6df817433c0068dd840825f481a08746132be64b7dbc',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_09b7ce6df817433c0068dd8418e65881a09a80011c41848b07',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='4ed317',
                        ),
                        id='ig_09b7ce6df817433c0068dd8418e65881a09a80011c41848b07',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1024x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_09b7ce6df817433c0068dd8418e65881a09a80011c41848b07',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='{"species":"Ambystoma mexicanum","name":"Axolotl"}',
                        id='msg_09b7ce6df817433c0068dd8455d66481a0a265a59089859b56',
                    ),
                ],
                usage=RequestUsage(input_tokens=1789, output_tokens=1312, details={'reasoning_tokens': 1152}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 19, 41, 59, tzinfo=timezone.utc),
                },
                provider_response_id='resp_09b7ce6df817433c0068dd8407c37881a0ad817ef3cc3a3600',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_image_generation_with_prompted_output(allow_model_requests: None, openai_api_key: str):
    class Animal(BaseModel):
        species: str
        name: str

    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool()], output_type=PromptedOutput(Animal))

    result = await agent.run('Generate an image of an axolotl.')
    assert result.output == snapshot(Animal(species='axolotl', name='Axel'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of an axolotl.',
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
                        id='rs_0d14a5e3c26c21180068dd8721f7e08190964fcca3611acaa8',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_0d14a5e3c26c21180068dd87309a608190ab2d8c7af59983ed',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='958792',
                        ),
                        id='ig_0d14a5e3c26c21180068dd87309a608190ab2d8c7af59983ed',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1024x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_0d14a5e3c26c21180068dd87309a608190ab2d8c7af59983ed',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='{"species":"axolotl","name":"Axel"}',
                        id='msg_0d14a5e3c26c21180068dd8763b4508190bb7487109f73e1f4',
                    ),
                ],
                usage=RequestUsage(input_tokens=1812, output_tokens=1313, details={'reasoning_tokens': 1152}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 19, 55, 9, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0d14a5e3c26c21180068dd871d439081908dc36e63fab0cedf',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_image_generation_with_tools(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, output_type=BinaryImage)

    @agent.tool_plain
    async def get_animal() -> str:
        return 'axolotl'

    result = await agent.run('Generate an image of the animal returned by the get_animal tool.')
    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='160d47',
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of the animal returned by the get_animal tool.',
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
                        id='rs_0481074da98340df0068dd88e41588819180570a0cf50d0e6e',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    ToolCallPart(
                        tool_name='get_animal',
                        args='{}',
                        tool_call_id='call_t76xO1K2zqrJkawkU3tur8vj',
                        id='fc_0481074da98340df0068dd88f000688191afaf54f799b1dfaf',
                    ),
                ],
                usage=RequestUsage(input_tokens=389, output_tokens=721, details={'reasoning_tokens': 704}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 20, 2, 36, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0481074da98340df0068dd88dceb1481918b1d167d99bc51cd',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_animal',
                        content='axolotl',
                        tool_call_id='call_t76xO1K2zqrJkawkU3tur8vj',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_0481074da98340df0068dd88fb39c0819182d36f882ee0904f',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='160d47',
                        ),
                        id='ig_0481074da98340df0068dd88fb39c0819182d36f882ee0904f',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1024x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_0481074da98340df0068dd88fb39c0819182d36f882ee0904f',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(content='', id='msg_0481074da98340df0068dd8934b3f48191920fd2feb9de2332'),
                ],
                usage=RequestUsage(input_tokens=1294, output_tokens=65, details={'reasoning_tokens': 0}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 20, 2, 56, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0481074da98340df0068dd88f0ba04819185a168065ef28040',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_multiple_images(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, output_type=BinaryImage)

    result = await agent.run('Generate two separate images of axolotls.')
    # The first image is used as output
    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/png',
            identifier='2a8c51',
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate two separate images of axolotls.',
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
                        id='rs_0b6169df6e16e9690068dd80d6daec8191ba71651890c0e1e1',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_0b6169df6e16e9690068dd80f7b070819189831dcc01b98a2a',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='2a8c51',
                        ),
                        id='ig_0b6169df6e16e9690068dd80f7b070819189831dcc01b98a2a',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1024x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_0b6169df6e16e9690068dd80f7b070819189831dcc01b98a2a',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_0b6169df6e16e9690068dd8125f4448191bac6818b54114209',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/png',
                            identifier='dd7c41',
                        ),
                        id='ig_0b6169df6e16e9690068dd8125f4448191bac6818b54114209',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1536x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_0b6169df6e16e9690068dd8125f4448191bac6818b54114209',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(content='', id='msg_0b6169df6e16e9690068dd8163a99c8191ae96a95eaa8e6365'),
                ],
                usage=RequestUsage(
                    input_tokens=2675,
                    output_tokens=2157,
                    details={'reasoning_tokens': 1984},
                ),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 19, 28, 22, tzinfo=timezone.utc),
                },
                provider_response_id='resp_0b6169df6e16e9690068dd80d64aec81919c65f238307673bb',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_openai_responses_image_generation_jpeg(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[ImageGenerationTool(output_format='jpeg')], output_type=BinaryImage)

    result = await agent.run('Generate an image of axolotl.')

    assert result.output == snapshot(
        BinaryImage(
            data=IsBytes(),
            media_type='image/jpeg',
            identifier='df8cd2',
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Generate an image of axolotl.',
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
                        id='rs_08acbdf1ae54befc0068dd9cee0698819791dc1b2461291dbe',
                        signature=IsStr(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='image_generation',
                        tool_call_id='ig_08acbdf1ae54befc0068dd9d0347bc8197ad70005495e64e62',
                        provider_name='openai',
                    ),
                    FilePart(
                        content=BinaryImage(
                            data=IsBytes(),
                            media_type='image/jpeg',
                            identifier='df8cd2',
                        ),
                        id='ig_08acbdf1ae54befc0068dd9d0347bc8197ad70005495e64e62',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='image_generation',
                        content={
                            'status': 'completed',
                            'background': 'opaque',
                            'quality': 'high',
                            'size': '1536x1024',
                            'revised_prompt': IsStr(),
                        },
                        tool_call_id='ig_08acbdf1ae54befc0068dd9d0347bc8197ad70005495e64e62',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(content='', id='msg_08acbdf1ae54befc0068dd9d468248819786f55b61db3a9a60'),
                ],
                usage=RequestUsage(input_tokens=1889, output_tokens=1434, details={'reasoning_tokens': 1280}),
                model_name='gpt-5-2025-08-07',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={
                    'finish_reason': 'completed',
                    'timestamp': datetime(2025, 10, 1, 21, 28, 13, tzinfo=timezone.utc),
                },
                provider_response_id='resp_08acbdf1ae54befc0068dd9ced226c8197a2e974b29c565407',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
