from __future__ import annotations

import json
from collections.abc import AsyncIterator, MutableMapping
from typing import Any, cast

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    DocumentUrl,
    FilePart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models.function import (
    AgentInfo,
    BuiltinToolCallsReturns,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.ui.vercel_ai import VercelAIAdapter, VercelAIEventStream
from pydantic_ai.ui.vercel_ai.request_types import (
    DynamicToolOutputAvailablePart,
    FileUIPart,
    ReasoningUIPart,
    SubmitMessage,
    TextUIPart,
    ToolInputAvailablePart,
    ToolOutputAvailablePart,
    ToolOutputErrorPart,
    UIMessage,
)
from pydantic_ai.ui.vercel_ai.response_types import BaseChunk, DataChunk

from .conftest import IsDatetime, IsSameStr, IsStr, try_import

with try_import() as starlette_import_successful:
    from starlette.requests import Request
    from starlette.responses import StreamingResponse

with try_import() as openai_import_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
]


@pytest.mark.skipif(not openai_import_successful(), reason='OpenAI not installed')
async def test_run(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[WebSearchTool()])

    data = SubmitMessage(
        trigger='submit-message',
        id='bvQXcnrJ4OA2iRKU',
        messages=[
            UIMessage(
                id='BeuwNtYIjJuniHbR',
                role='user',
                parts=[
                    TextUIPart(
                        text="""Use a tool

    """,
                    )
                ],
            ),
            UIMessage(
                id='bylfKVeyoR901rax',
                role='assistant',
                parts=[
                    TextUIPart(
                        text='''I\'d be happy to help you use a tool! However, I need more information about what you\'d like to do. I have access to tools for searching and retrieving documentation for two products:

    1. **Pydantic AI** (pydantic-ai) - an open source agent framework library
    2. **Pydantic Logfire** (logfire) - an observability platform

    I can help you with:
    - Searching the documentation for specific topics or questions
    - Getting the table of contents to see what documentation is available
    - Retrieving specific documentation files

    What would you like to learn about or search for? Please let me know:
    - Which product you\'re interested in (Pydantic AI or Logfire)
    - What specific topic, feature, or question you have

    For example, you could ask something like "How do I get started with Pydantic AI?" or "Show me the table of contents for Logfire documentation."''',
                        state='streaming',
                    )
                ],
            ),
            UIMessage(
                id='MTdh4Ie641kDuIRh',
                role='user',
                parts=[TextUIPart(type='text', text='Give me the ToCs', state=None, provider_metadata=None)],
            ),
            UIMessage(
                id='3XlOBgFwaf7GsS4l',
                role='assistant',
                parts=[
                    TextUIPart(
                        text="I'll get the table of contents for both repositories.",
                        state='streaming',
                    ),
                    ToolOutputAvailablePart(
                        type='tool-get_table_of_contents',
                        tool_call_id='toolu_01XX3rjFfG77h3KCbVHoYJMQ',
                        state='output-available',
                        input={'repo': 'pydantic-ai'},
                        output="[Scrubbed due to 'API Key']",
                    ),
                    ToolOutputAvailablePart(
                        type='tool-get_table_of_contents',
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2zZ4sz9g',
                        state='output-available',
                        input={'repo': 'logfire'},
                        output="[Scrubbed due to 'Auth']",
                    ),
                    TextUIPart(
                        text="""Here are the Table of Contents for both repositories:... Both products are designed to work together - Pydantic AI for building AI agents and Logfire for observing and monitoring them in production.""",
                        state='streaming',
                    ),
                ],
            ),
            UIMessage(
                id='QVypsUU4swQ1Loxq',
                role='user',
                parts=[
                    TextUIPart(
                        text='How do I get FastAPI instrumentation to include the HTTP request and response',
                    )
                ],
            ),
        ],
    )

    adapter = VercelAIAdapter(agent, run_input=data)
    assert adapter.messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content="""\
Use a tool

    \
""",
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
I'd be happy to help you use a tool! However, I need more information about what you'd like to do. I have access to tools for searching and retrieving documentation for two products:

    1. **Pydantic AI** (pydantic-ai) - an open source agent framework library
    2. **Pydantic Logfire** (logfire) - an observability platform

    I can help you with:
    - Searching the documentation for specific topics or questions
    - Getting the table of contents to see what documentation is available
    - Retrieving specific documentation files

    What would you like to learn about or search for? Please let me know:
    - Which product you're interested in (Pydantic AI or Logfire)
    - What specific topic, feature, or question you have

    For example, you could ask something like "How do I get started with Pydantic AI?" or "Show me the table of contents for Logfire documentation."\
"""
                    )
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Give me the ToCs',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(content="I'll get the table of contents for both repositories."),
                    ToolCallPart(
                        tool_name='get_table_of_contents',
                        args={'repo': 'pydantic-ai'},
                        tool_call_id='toolu_01XX3rjFfG77h3KCbVHoYJMQ',
                    ),
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_table_of_contents',
                        content="[Scrubbed due to 'API Key']",
                        tool_call_id='toolu_01XX3rjFfG77h3KCbVHoYJMQ',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_table_of_contents',
                        args={'repo': 'logfire'},
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2zZ4sz9g',
                    )
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_table_of_contents',
                        content="[Scrubbed due to 'Auth']",
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2zZ4sz9g',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Here are the Table of Contents for both repositories:... Both products are designed to work together - Pydantic AI for building AI agents and Logfire for observing and monitoring them in production.'
                    )
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I get FastAPI instrumentation to include the HTTP request and response',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in adapter.encode_stream(adapter.run_stream())
    ]
    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'reasoning-start', 'id': IsStr()},
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': 'gAAAAABo5kf5xn6cAv4jZi6LDABpXHtrd1qzU0V68_w3dCOTzg7EVKvB8i56Yah1EC7B4i7Zh3fO1H3Q6sdGS-jULiPLQdtu6TOMlNeZir_mGMVRih89fRk2UBKdwh0YywoHpv7xQLHT0qOtbVxGkzrixkOKe7B-oFGVq7v5Zpz-uD-FsyZCMikyKgVI0MvbbVHHBx8FJULvlVOAKyUQesNtZGnwobMJvUgnOUM_7rPzlC-jpzD1hwURJ93QvSsP1Klpv9Ebuqbxih70HsMmjNqO8PKwRLUzo7IqKGUTarrq0eOZmDird01aNC8ao4paWUE92fEbT8Uzv0QPmWEBjQVUgm_9gL4E0M3XzlWj5hGXrqDBJYNB36ciCJyBNSS4tx2LFkqM0UDg8u8k_7yDdVyL7j-CLz095-jT4IwCRMuKuBjaqIetw42S7xEeTrrIZPGg6oqAgrz715lr1kz8NoO2aDds-RZ780FXcptue64fFy19kRT4tmRlkeUT4oHAlLcJqay-Z9EFtbKBtazKCZyPKDCHiVWQa4tUTlq6SsAvJAkgLMV18qO643EvVhmWsm37Jrb2XaiDydgNQ65olaacS8nH3YuM9xDhGatKuyo8BcQAJdHq6tRHJpSUDG8awLuGckKQ8h2FHPeztkj5ooIwb4EvmHcOTBn1KxpXC4SD6lS24Ob7l9G4VB7udxDrYyFM0d22fh-gdRcd_axy56CTn_wZIyunQwIhQXHAkF-A_tdsrhHLj2V_hQ2ICdPmtZ4ZWr7kGoiGW4RWr75yd3pqzehr64pDXn3HTZ_a7vyMSSvkQFYUZfTRU56cE_09q2MZKfGAd1_X3Tqvh2EJwwXGXi7NwjRt0deK6TUjVmLEFkiRrROGZ1XcFDWoCQs9kmaHUAEWqDKaJJ3Bt6PstG-ZfZlzSouIPpZ_aUzwIuVkYclzT_d6PPY40m304HVHoonODHlNYLYlNaQCeH0th0KAYQmGvt6JE8p5j6UYrYPCT8zXW5YVppuIqMAmT7qOwz1GC1DAFkZNkBVc9QWPeK-38leDqbNzNKUSsBLom1UwyBdJE-6GXX-zPrUr229ak8PT2VrrkiMvKoR9oOqdn0oMlRkCvreOvq7gwUHRy-yhkLtSAEmD2PoK-hPCefS_aDsLs-kscZOY73NrA6bhCAkh5KrUylk3-LKWJLXooZoCvu_ZaG5w8Gi1tBk1F3oa7WdXGeTnl8oxUuDFoiuLF4GASVmGIEMxbrSJhrpkXkZdpWzQAWV6-XK7owSgy8QBkaEy5qu70NwCS90JpvBTSmoe4EDvVwTlgpkCmQrEItynZ-Atma_L1TLHobjJcYQ7muZmFXhforiS7wrga-8oGy9Jch4y93xwLzWwSl6UWBmgqyhbqUiB9wAtituqlulU68WN8iAOaLR-zuPkHJ5TXrGyguReCAwWjEZXFjqtauX7ueoEdb-fKluBVMlivCZS0fXyts47doGzozeKaZg1ahDKxudeoZbGQR3fb6yHeLTM2W2iuk_Tpwp6Rc4Qj6EEFf5k7mTy5W0_cCzVOlv48Qn55OrnhFb8myAqtQ_8-hDt68FVuA-HCfY6KnByNHTBvKLe8Bt9BA9uCVJSXhTytauMucZXvGluISQZFKxEYpwrUkAtI4dWuGUEPaLOGv5oFrtW5-mBLKyinUKhjW0o5FhF8juCWkQ4Tm0IFhQcXasnEUDp3Z--xJl6l-lYurxa9nRrbYco4qbMj6fxHcDcjW4-s_3ZNAHFvgmLJKHvd6dQo8lzwuXw2ZO6I71htJnhPdfo8456tiiH0J7qc5i0jB5CUnttTupJMvGOr5DyrNdGzT9KfoJ4mt970AaIg3NDAsLP0mnZhsVJSUKeLxkoUd64mkhnr8Hjnjbk06QEUmVtyO3ha-jFkCVP2GjwLE2L9gWWQ8pctXuVAtHn2umpoQbcNELKgv9P9rK6eQNTTeVgz0ly8N8-P67SueDSh-Bf5nrUYmc8Yz9hxlYsemremEjftbCpODVpgnpnDINEnH6jELjameMmCJcQDzDIaVLNMM5z71wU_n1RDdWdnBdUKTy0B0Qqds9DHLyP6RGC7NQHO8EirWcpiMtckYDB6M_8QQI_ZQ3ORgmzoWgAOSmiOzlIiT3XNc9G0ts_zZigU4iIBm_BBXYa2PjpO5ijMiZDGybqrpzuNChKAY9f2XFl2YDt11cR-SUFV56dL0Ebc3BESJQ-1CpeODHqqLPd3iRWzUyyT6UVtUo-x9A70HbnJ142QAo0MvZ5Sd-q-TQVrKDMUvXZRQzQ3x3NISPPO_EIOXguwa9EjcLRpV3ldSVpNNTo0O2kQh8HTyH16kGw4RjMT7Qn_DJWpe9lXPwQyI_eCzacR_FwiOOUt5H2cgKgAI-TSKnUm7oFm9lTSPf2FB4LmympN96zX9a1o8YsoYKMnuGypCCn65qblBs0ftTswO2gvZrhjWX9MB7ZlAaA5MgHmrKcLb5ICe5XY3BgGouOWBWkjES-tjAogrUXnBAN7l99g5rkAfAfZvL4RX2H4UGvxHqRUrBME2IIUGKCK5joD-PFZL6x-75S7Kj-vLm37UXglF-RiN3AJbrFce7vzbu2GcLgJMwd4GilTycVUNCIKyJypMs9RbkeUDMeO3HIIyjn2zwNIQGWq1ZSL5yrkEzpBXd8RMCwVdpegZYzZwPhGH673xoj4CV5P42QRZ2qJcjjtGf9E35FCqCWb9ogKV-MWYriwKOFmNN4GKGV39aGQj4NHCw7Mwth5sD6WpoE5BaMbMBEfy2uVTfaqT0KZ7fRcgy1hXOm0A4qDSS3zkfsh4h7Oni2fjPPkwuh_DLx2Y9T01sJ1vYd79Gsmxlmzor-TjED3I24phJUgAanOxxcRD8GIojeiRTI4Tgj95q0SfkCzy_DY-wD6IIHjEyL4D7JO0Y7U_UADJN3ZAoKHWlZl3OSO7nw_uskVM5FB7jqqDfr5P0YBMQMwWjVuAibgMnt-9rO4NPBG1xifL1GwcBvFW30cgxE2SGFrOcPqyc8HS6vc55DgUS4kpPOkN8sTOP8wBT2_jEKzRde3ByBDa7Dyx1hAuet_huuNmYU6mGRcmKCqWc6qW6FJH4sjLSvUW9UoRDMn3ZpVBsHVtdiQAB8RlegtJNyr61gexss0Qla4InT65EbkB1carYnCdeQGyyMUKk-YD79CapD_BtRnMJ5twCFampVIiQn_tANwdMEX9NFGgQgnczJ3MzbrEHHA3iBhapWxWD65FEm2zYrDUaeMNm8yH60RuNKyZ7aZRo-sGKrYWMU6bZ4YwuRbqiTNF1NPizcpGrj_2k_hT-TcbtPnPNAMdXzTK9EB2QHfsQqmsqOImtx8q0vekrqCSWjN4H1xRHrYPZMvlvTdtDdhiWI8WSatsPludVXeYxqXxOy2XulPjzkiVtpgvMUwTGH_Yh-kUP5C94ZuZ3WDM93Gy9AsDdaNScSC_m0Ho4inuGpe5b_ctx7MNCdhdzl3PXwl67Fj_0E3qkdmDS6McKk0S2fOmzbvoJmaL7g2g_QVCCpxA-4vmmecak3J5Me_2EW6sHd5Zn1tvcFezrcmQfcZQbax7yoyZk20egkaQqpkli-AKjSUKBmP0aSR7qrkvAe2z4mgl4yP062GlC2YI3r_8xedjsg3jWRHyyJ4D_7Y0rtmAjVwhi1tfl0XK0JCPKVkwTTCBSbwVQo2yR_zc7zIazj9TDm8qFK2G_Q3QuEqz-Cz9VGQu_BPhrMaH1WaB-2ays6IUf3s7_gUavAAynaEM9kGGHj_t8zkyqDTjalRl0JE5T5PxxYdv1G9s3MU4B0yB3H_LhcDyjVAb1WzPZpcAxxrcBdpLVFa36ewDzKRmVmSONJd3fLtB9_HJhXgtFIIRgAKuhmRf1hMAG6qaJSsJNA3HYRZigy15OMe4RgkFSl1aj3ScrJc1-N3Si44OwqRIJGrVV2ihh_euAE0l_ImdW_vEgpRbgfMqBiDmFbxgdcv2cT_MaOMoFX3n_V1GrAuBzjrVQSBbhgau7nWsynSK4h9781waWYMAZK4LrOK0UQ4iyugcpj0JNO9k4WvB-yRR9E2rtIoEC6Nz0=',
                        'provider_name': 'openai',
                        'id': 'rs_00e767404995b9950068e647f10d8c819187515d1b2517b059',
                    }
                },
            },
            {'type': 'tool-input-start', 'toolCallId': IsStr(), 'toolName': 'web_search', 'providerExecuted': True},
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"query":"OpenTelemetry FastAPI instrumentation capture request and response body","type":"search"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': 'ws_00e767404995b9950068e647f909248191bfe8d05eeed67645',
                'toolName': 'web_search',
                'input': {
                    'query': 'OpenTelemetry FastAPI instrumentation capture request and response body',
                    'type': 'search',
                },
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': IsStr(),
                'output': {'status': 'completed'},
                'providerExecuted': True,
            },
            {'type': 'reasoning-start', 'id': IsStr()},
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': 'gAAAAABo5kf71KqiOXQLdnpn0X2dce5ZXQV3HkgzKX8X-CIy9ax4SrMhk5x7PGHiHFyqGHhXQ9VBYdbmpeQmcXGtngPW7v0BPz1pl4HVfrNGG2MVFgqx3xVgDR8IDDBtoX10qhGFzQOmp_V14WZWPvdwP7irv9LtSXMdDKvJThXM-s4Kp4vdJCPWk6PeLmQ-3bh6eXbxGB9jK5kifSBoJfgicrgpfpUFmdispLXzxdTvyUUMJDjg7p_AgdKwupWPRz0I95d9Fu04GdBZfJn2bcXbUUgKeKKqpOeQjmnxD7Igmce3W0iD5icZq9d7ny_dIfygoel_9JPBNIKJ05TRKVY4C9yiy8VJAvywl55gUdziTWDS5WbEggs99brazkEdWTvqhFcASgqDErzCeppQ0ACWy6a8F2Wgi-g0Iw_MiaB4zvA0VSDC4xOduXhWB4BUJemEp7rE8ztCN7FTKtON98gBixqyV4ueEUDp7SXwkRJ_d-IJMh8w12e3eTTqKs3uljEPDBZXnvPkeCz6GUIuKkEQi5mk4qLi-vZmH3gvhOe-dKJNhhN9CI01PK0bmwNdXr6loXsTmPhsLlp3Mwc9mk1RWvXm0TbDSGDeH1a5UBiqjFI7qjWANOzye5qeFEs-vqIQnC9SBDMMDlFlzLv7LJjsTnz7Q9TTuUx1sdPqSu-lLDM5OBaCdykS4gQzOZqomLJNSb_lMKrzzylpo6bYmV40N_jZkM71gF57n8lBYVmS4t-JidvqsqW3kVlGGDmZ38sA2I7jJmI8v8v3Roio_uNCzzocWCtbcPqgjNsvDzXxTLl_WTjyRfHE9Qdrj-KY565D-ynxlR_iCPFcOx3cwdnprXFf08Jx9WIlvVysra15pjj7WH32t4j_Bp7g3pI8ZVYGwIb4US3Img9D7Plfc3rAWI0d-RGaMFjjbsPzCBZeF9JPDcOwdGLa88ap_vZWkdRScEJffjZocv7FZywA_VgNlGv1S9bP43EuaqEXmF9aNLkMxsoaLLFhRRsgryfxk4jslG38F8BiuHhzxASD8C7f6WNXZVjJL5jW1GkhuYB1qrn9TpXrLR8mJY1Kzkt3dUtNhnIxGKzP',
                        'provider_name': 'openai',
                        'id': 'rs_00e767404995b9950068e647fa69e48191b6f5385a856b2948',
                    }
                },
            },
            {'type': 'tool-input-start', 'toolCallId': IsStr(), 'toolName': 'web_search', 'providerExecuted': True},
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"query":"OTEL_INSTRUMENTATION_HTTP_CAPTURE_BODY Python","type":"search"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': 'ws_00e767404995b9950068e647fb73c48191b0bdb147c3a0d22c',
                'toolName': 'web_search',
                'input': {'query': 'OTEL_INSTRUMENTATION_HTTP_CAPTURE_BODY Python', 'type': 'search'},
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': IsStr(),
                'output': {'status': 'completed'},
                'providerExecuted': True,
            },
            {'type': 'reasoning-start', 'id': IsStr()},
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': 'gAAAAABo5kf-GWE79PWmX5BUJGtdfBs_8-rH3_mU0ojB4l0DH9RZkOBh8uXVIGaJfTwN9yMTzxrSwC8kdSKuYrKAvpfn82Dq0i4KU0q9ZlA4mQuVsekJXqABXyBxSdR0Zt4k_hDaNRQDiSdl5z9qkDhupx3oD0QzTN2vJF0EntJpnQtiOkBlqdiFJG63olxaKLcaYk7w1UpPXHFKHxv_clxSb6Bhs8XJyquYVcuBP_qxal77tkyepJ4HQI46B_bnwS8LaIGjZ516MYzxNdIYPPc2T1TSACumXTsTLseOL2LseiClD_fSAEuCG_g_lnbtkMa95pz6-fTN2dmJaXcg05MSjX8YasVt1lkk8EzF2L_lhqled2ht2Np70R3Ykmlv8TE0kzCx1otr3WEhdi9xarx7pbISF0HuguVGp3V89-Vikge5LYIar1sMOOOFKxjVBhPvXqBB9sw6JCknfXKorDAj2shu9vGrPHX5YVVBvkJwPuSO1c2oV7SlW96Oy0lt3V9subpr6XqFtxO4QR45Kv_jLCs88_0W5ImjXA4vfKzWwP7vYuzWVY3xkaXXydjH2SzRZ3PpJBFzoNF7QjbnlOBlwKaPnPyGnA3gb-m3dNOYCloaa1Z6XIuNS1zlBHlJIHrJrEKMAtMJxBsmmjbxwu_nMbvdri1MzhtqTwySEB5rLTRYvYuU_3tnDbk1iuoB2kcDp_J6AY5eZScwsiyf-zWhTjgAyjdxwYytFKNzL3j1RTHwefI3U8JCmQfDk7kzfrBK3a_1Jm7MW_LnAhD-Nf9dbeNEBaVI-0LnH3A_xFMYLFOcINGe-5SoUYV88CWMIERuaSuPYoXw54S-fqUmDVVT5eNOFEKXWvuiUZDJBWYW0RAZzKXB2uLowcj1qBOeW2dxUR6s5fFudt45OwA7aDi8_ReiyC_RJ1NuOOYQ5YqxsAYZt6-NXALhZKgOapsK3ui6wCeaXGprG7IohQ1jj7rKUcRhs6JWxPSG8SaG9P8D3p_ahdKnSq6rXB9MpWmbtOSVNOLQk51RnMwwj0__6pHcJz-Tk0ZWr3dHBXbNKwxt8j89XF5bbNChGEZ458LcUE6EQQyjwHbqqxQRe325CHqhwwjgh3eXig_en9hHbDRyyYYFFeG_7ysZW5o6gO8QaI3Lo8JW0_PaY6e1um328lICJauuWTDKIWMFOUEahiz_eveoSHWJR-_mi2KBm4OQZ4y9xeffPmFs',
                        'provider_name': 'openai',
                        'id': 'rs_00e767404995b9950068e647fd656081919385a27bd1162fcd',
                    }
                },
            },
            {'type': 'tool-input-start', 'toolCallId': IsStr(), 'toolName': 'web_search', 'providerExecuted': True},
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"query":"OTEL_INSTRUMENTATION_HTTP_CAPTURE_BODY opentelemetry python","type":"search"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': 'ws_00e767404995b9950068e647fee97c8191919865e0c0a78bba',
                'toolName': 'web_search',
                'input': {'query': 'OTEL_INSTRUMENTATION_HTTP_CAPTURE_BODY opentelemetry python', 'type': 'search'},
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': IsStr(),
                'output': {'status': 'completed'},
                'providerExecuted': True,
            },
            {'type': 'reasoning-start', 'id': IsStr()},
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': 'gAAAAABo5kgDMdyWJm_4EBrnLS26HcJ_EvaKlVyP7dihwR4bkE1kXTcQkdj8PYFl5tw9am6ZKst4kTo_eR_ho9j0wrT8lpwlPyOvuF978cGUykoGJXGMlvBIEt7oBgrQGD5cchOr7uNqTfVo5YLY0ywfmTanxgyR3CD6xZiSrSRuG6e4xGfa_5J2FK6EOqQOvXoXI-KRMwXTlrqh3nMUXnUHCAZAQnplTHHg61Muu-tBIbG8dOXvvIxRJEomIJJdNg_g5pRhCvzqfzA7MR5_oBvQtkn_l3U_mPzvjfIITIJT4iCsnxJbYNE-XP8mokgKq6zZNfL4wi_az44_BshnwJA6fGrAmmMSEMgkFli-XYvTEleOqiHBttqB0ESRoztYo0Pxc_rn5AtsvvPtQUYYapaV4qlL3O6_Q95TsXBlnsmSwjTMSIEmSb-X0BMzMtoYV1srbiQiBq_LXwpI1owWH8rJsw4x9C60dWk7xY19_6bdcYE4QuRPnPmyO91pl33iw5hBOGckfKsMjERAmbsjcHSczOO4xa_30EqrDyx6py50KJQMFzFEdZQXdfM7lRIgbUW4ixWw8YP54vfVgSBNz9HsfRfn8V3jvQ0saIMW-M54aISG6hWs45EbtRv041W_J3SlYcDtOcjsusR5c0vd0Nl4VIECHLQc1ULnqkJZSk334B7oUppZNX7hL_Y6Q3m5mbqgbGsNRo40-W4I0lGTDBnSti4-BBEgXCcjZDQBpguyZR7F-sVUbli8kublzIkxzIYxXLpTFKGMNhBtsfyHEKXZB-I_bCy_rxlszOW3NtodoIdavaVbmbpCwXZAzOE4TJPPlQ82F7rEAX-sCtDzzqlvbs6ZW9AS_3r92Y_kx_apxgcl-uQ16pEXPlwy6IQHFhKQ2m0qj-yDc2C25DyJ8oVBK2vGje0Jn2ppsR1m1knfBP4UTogTMCp5MvWhvgsZgJsRRoL_WxQP9HoYajepLrRXft6yVfQKTh1ksaYIWxZeVfatBoFrUEQ7xVcbPVIduEON8tzPqzRTUaYY6z9HHpYennJmk4Iz3psJRHqa68O4WJN3P91FaUgwx600-LwcOimodB_BUsMBKAMkKMpDroLPoLwzAu3ubU_dUqmn2Dj7HEAjjO2WdN0yAn32AgZP1nqUjxed0Yght2LJWdiiPHAKr5DAd6uq0QTUqSm9dOJI3n-HgXMVNDasRYj-DbwsBN14qUm_zzJg8X28U5vXJZ398OhxRcS5x2uZ51xfPL3wwHy3uGpqUR9ZrMr8nEeC-FDYX--QEXVvCCzJsBdGDbW6ufLmMfbcsv9eAXQJEdCk1xvorwQXONydra0Q66ZC38d1rEdfYEa3FvuqK3o8ug9Jkt04RyMqYFjziIC2D2Jf55jSJhM8fGo=',
                        'provider_name': 'openai',
                        'id': 'rs_00e767404995b9950068e648022d288191a6acb6cff99dafba',
                    }
                },
            },
            {'type': 'tool-input-start', 'toolCallId': IsStr(), 'toolName': 'web_search', 'providerExecuted': True},
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"query":"site:github.com open-telemetry/opentelemetry-python-contrib OTEL_INSTRUMENTATION_HTTP_CAPTURE_BODY","type":"search"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': 'ws_00e767404995b9950068e64803f27c81918a39ce50cb8dfbc2',
                'toolName': 'web_search',
                'input': {
                    'query': 'site:github.com open-telemetry/opentelemetry-python-contrib OTEL_INSTRUMENTATION_HTTP_CAPTURE_BODY',
                    'type': 'search',
                },
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': IsStr(),
                'output': {'status': 'completed'},
                'providerExecuted': True,
            },
            {'type': 'reasoning-start', 'id': IsStr()},
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': 'gAAAAABo5kgKZNRzXIfjM9mJNtrOBB9QgXmxr9lqDw4vJJnfnrxtWXUO-0XAPYIAWwBgPImyrwdNXlHHny-QNhswuEa8gbgpPdhkhREfd-OY9s9gOe05wD2oXgLO_2S9kPLA0wnB0HriVNEYXhsxk93jMwj2XmtS6da0O8lO0KvALs3HT7EtVTNTShKqlumzH2HbA3tQFiln7yxVk8C_FIfDhWpzkLeqDGvIXy2GUyUHhQ3-Qmid9KMkj2jwI35TawAMvxckbMWcroYTRXsuVgvULRoGoU7npo_YFTvjQYIHKKKnMKzM0wBef8hPLgsYOKBVwvFvKyFph9fUV6TGDqzX9daq5Jzl3PgMVKm2o43HjN9NQu8nKf2Dg_VSIbfCmxq1Pde95MV4IuF7UEFCAREF4Nc__K5P4HRvroScjXiWur4FJulj6ngOMDnZDVcP1MchELP3KWe6L4Rz2k6itK_xDugXHnq8Ev3WuZfmibDePxi9KpLnjsvQ9sR2yRstKABto_fhbTuTBuH-CjiFV2nP-HuHlIymzmK_OhwX7v6BHvEfTdiCaDQQUD4a7uo9fwCdHK7mK7CFQynRU42fJzW35phx3QJv6XR4BzxzRUavJy7ZqoptixLGI0A9_8QLvh1BluN0ysh17Q5x49Dr5Nsme9kp2gablAXNbz5PWhrHypVba-GAS9K1AFiOIOLYDrSL5o1-QeaJPT0HK8Bvog0fMBuVP3-4UNaxDHKrTLD1eyTvhZpHCctA93zPlaOh1VOsY9k8C6wAFF2BfvUsdRraQZqN3Ec6QxU3WZAGuxqoWlqgzOzxT7OzzGTRfaa25797koiBwy8J7ov2A0Uu2-GsvJKWwv3Ncc9trFH1QaigqB2lHRi-zWAUAz2Zc5fKB0BXP5NoolEHTk9__VEqOBtXMzEbHVzOtVujoX6Xz2JR3AFs_y27UTYbFxpWBEw9zUHw2K9L5pNpLDJxw5Pe6k6scSEajcRTEPXrg7ztGJlhC4SkTuJT989O6MgZDp0U74FnWs0Q89PCgAIKJWTlzXUpqr7WKva11p_MHlLUJBa_zruktgENfp4h0w7WRNdENl-sUJCqZTfdzdcBH7Nvwsg-qq85gLw73JA5-5p7pYkuD-bfk1rOECiq0pufDxDr55hoZC_QeGuGpjgoJpYeYbkPNQR3lKa8GJBL6liu8pmDNbInBkLPU0OIfhvo56shLSdifP2Qc0UnaNARgXGtVIwmgRHWy_IqJKLFB2YUcawSAv6R19IzlLh6tkJVWOAe4gRckuU8W7eBNeY3fPs6ffRE-A7Bx-RYWtjM7L5OFYuB3y2vPVSGUvAcRxmTfLqOwtZao7ZPcldM7ftkZHMiw4wdFGgqGxUTSRzYS7ESMEGLM3sAp61BLB_X_8qMDceC3Kpo2rHiem_TILbOrW_vDnuvfAxB1oWBrr3IsbHb_BDe2fOEnXT5KlJ6XWc7VhCHKi16RJH_-tpN_YHlncSpjjtkPvu15oVB9gBXJ7kvLATMygy4L3_B-PkjNgVkf1-QmOgJ7GPtxk69jAklSKHvVwPN0s55u5RuvKroyZyoNdIhiSY8EvMjZT9MvmcBgUJNGS7ZbKegVeMR-kcsRxSmJS1_png1lv4bX8x7OVONm7kSC_YBFufov8ul45veYCi6ATuGsbPrC_QKp5mWJfYjnopTwKxs33GvZ_q_AXnmRC3zl2dqcu0cwbJVYniVS1qNIi7G9lxUi9Zzlss5bt3jAvVXbAO2GH1Ij0IrNbGB7SlmJLbnochFU7HweHn9aNWTD5khgwfqwZZNli13SS7r1EmCyi8Xt0KB5MAGOz86o_xgeBkIyaxv4ZSb6wY6g2oVu2XPo1229iGh5-_qHhFvpkDXfzIyrAAbs8ZD5ScXlUUgBzK_JOJImSEP4VSlQdgMBl8KLQgQpZtsPhtdCCT8nmhFRx0IYZxs2yLJMukWyY4iUQ00UsLXtVUyufQ08xjcdrVqevdXvJpVA3GIdcktxb9U5wYdMGOPMUFmPejqLZvC8dQwbWOYdGATfGuPdgv9LS3OTpUXkBnfYtTA3P8zk3RxWFCgDCjekxVgHPJ11BCMZFxw2nHjdqFjyAcISvwf9ix2UXCb9DSdmvbjqUTaJBCQD83JZn0lowtdNwTEWG1nUln8Qe9DIn7Ly8Vj6l0Fq-jkllBa7aWUTBS1H929Yx8gpYRApjBhPCvwVX8o6Lp-7qeb9PN6vmgUDfAG-8T825gkMVwyQVhnQvUlAL3D3Mh4YMdPHS5W83YyhawYvYWRgxcVis2oEMeSvr93CpbvmnbtYPRiousGdxu5Lc0KTNop4y0T5hwHGMpgrP4u9H_JaSBUaf9CO0dO3LyZ_x-rdOM8K70m2vmYDxbEHS7I5EBCN83PJnB8_HTV2KgmUsJcE-5OdQk_AEXikVd6ujMum2gA74MbOJCI89NmC6C5Hv9OS8lzDaaJi3KnhCvqePBweQLQrWTT0mc=',
                        'provider_name': 'openai',
                        'id': 'rs_00e767404995b9950068e648060b088191974c790f06b8ea8e',
                    }
                },
            },
            {'type': 'tool-input-start', 'toolCallId': IsStr(), 'toolName': 'web_search', 'providerExecuted': True},
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"type":"search"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'input': {'type': 'search'},
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': IsStr(),
                'output': {'status': 'completed'},
                'providerExecuted': True,
            },
            {'type': 'reasoning-start', 'id': IsStr()},
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': 'gAAAAABo5kgO1HDzgclsJ0nxbd2rPdcO-gp--TPErZWuWzl9b1tNkttOes7PJp-a5m3GggvshsVGOf4Bm5sgp5_xK6z5zpwoyaQVsac6roXrG4QpXIQrXo-2zZUBwnUjP_yV51JDI2Jgjb1kgcgYz65WB0soyx58CMi67hVyOP6dRvR8wfjoBPEHq_rTf3J85Oux3zIH1rQFviHzwKx01RqPeA22bjpNNoYXkZXMDFvBK6xEMox4MzQjPTjwvymDreDTQImB_qVspiqN5EWcRLWEkTSzM4hu8BgI4tW6zaH530pBL95QqkaWF4VdjQ3c6yEBXXJMXIgmvP0avmgQh6fvMAQ9XVeAWe0tqXMmZ_bU1TCqF4wPYau7hJlLaZqxYzN1JH4RHaE9f6oDAc1zy1n1aUL5qtSiCFD7TR5ADPsdX5D3sa6j43uGGmwrUfHCKnxdZcLY5r1FSHpYDah5pKbxGLWdtG3IxT8-c0RadmtIW8j3nrDy0RaZlLjDXaD-IYxsyRyZLIgZU9IvcT5VKJxGmNpWzt5HDVvv_IUAx2FtbTkPRgmz2rRVR-jx1PoAB5Gf0A_DhTt7zlAjjue5GK2GsLqYwv_BoxmfT2132a6dIo5w9JA15W8j4Zlizm5uHL8iLx8lKLwTcHL5YksNUOVp1ELgan9KSXMj87YJabHDq4iio6w9q0EChqL9p4iP7BQvpheHgdgSW-C6H7N3jQZdSq9TTKqIeYjPnP_ZHXubE3jzhd6KWe9VCENUKPXXEuwUHZKcra7NHnmEDJrZt848sw_T3Lar0mnTuuh6-mLjI7rIuxy1VLmLdNyYuWa-hqbQ1aXiqB3E3VKkwhjMOQG8CWGt2jjnJC9OQz9LYrkn-8n-R3jpOKAwhIla4wDvdh9mUFa_1kiHeNSHZjcXnWXXCKDeHs_dZwHkkQ-FL8VO0JW4jueLPZD4EfBPkcEv68kLtI7EcG2yKZ1d2shdCiFrylbAz3bg3MpeN_B7QtusgCaCKDbI5xmwd5hCR6KiUKjaq4nsOQh2y6gfjI_jdWKM6VD1AihhFtNvHDh6wrsv3q5PD5zaoLvMm9maxSLWwJpMr9dfEQ1X43Vs_Lc_PviJIdIHOuo2om-fwRy1FSknvsDSuktqC5g1oNTmcj1F9Vd2uTG48R2-c3USzK5UvJE3qsuIrzzTiM-mTArvtA0R65wni3SzkuppSvR_U7doHWnfHRl5RTzJqZGzZ478r7gWobbVGMDtNM-D1hU09AIq2dqt-IEteNXzSqluyAvdEQwgTgJrznyN-bGA5lbdUlpG70-t0aBiWplVzLVN6GmFuCO4nlsHvz_PVdx5GiahupscwERmH8rJVGP_UAfjnrcEpMt-oL-D50tk-c8hKR92F022b2QSJswabCLzX3kkduQfhWYuWAj_5FTW_8F-Y5SsawarczeT2ajbctFQyCEffxD2jEJEUOZY2wfp1XCcSgkycp6xo29Vnf-1Sk0rpzC6pQE_i_4jX0yzViQcu-YjTzwcp_rMyhEDzYPOVAlDDwdfQscRr9XoZSZ-CNhHYA7C9dn62JHP5tWxVtLxFuLqK0WNaL7l58qHJlAsb_OoWMzgeH40H9CkJuJqhE2SC3HEJBAZyvElvNCneC88YVTNqC-i0GAGyAN9KhAUkt9WDxtel3Q_2W50FEe1FIqMjrmfWwpG9oA6dMk6nrAmN1HhkOSpmEtTVcp3FcaBGQOkm7L51yG1A8FH',
                        'provider_name': 'openai',
                        'id': 'rs_00e767404995b9950068e6480bbd348191b11aa4762de66297',
                    }
                },
            },
            {'type': 'tool-input-start', 'toolCallId': IsStr(), 'toolName': 'web_search', 'providerExecuted': True},
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"type":"search"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': 'ws_00e767404995b9950068e6480e11208191834104e1aaab1148',
                'toolName': 'web_search',
                'input': {'type': 'search'},
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': IsStr(),
                'output': {'status': 'completed'},
                'providerExecuted': True,
            },
            {'type': 'reasoning-start', 'id': IsStr()},
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': 'gAAAAABo5kgRqGx3zwhXFb4elaopbSnsU7HznDhJ6BQvZc6F45lekqR0kJF7nDtrPpt9JH9ZwBBl5xHUN2S4dlWBgWiVvQZYUL_3Ytc3AIB7kyDjQLaT9s6EFFsgIE93aeVHE1rZsX3mZm-VUSWLCNz8PxujflN_143h9JLO7AzxB5D0tkd_1bAP9lVEKuAVxdwkGTmGn7oidTHfoAqFgdFHkpJf77ka4FQSrTHVr_CGptlT7N_tJGdUDaTiNgeBak3aadpFxT5dNS21g_DoHzi46ZcguyCt6pgea_fFNwv1QRkE8Jx7qthNfCzMP6g79RAR008SwHrVq6MjQQvTNH8XI6giX75pnyt47fR4oAn3nQJsp60KlCld5vy9V5lQpvrVn4P1BBrXQX0eXqmVIOp2VQYN6dtVjHApFzJBZRb2tqzCiShK939eDoypVtr6oI84oTLlGBtOyeXSsk-kYY_wUwpqPbmCcvj0OfvzXXFzL0knmP5kzKN8KGE1Ko2sOBkRyRJ9dLUEsVQdyhN1QDrtgSNHo7QAKOD2FkcjMwnIXaNya1Pn_yhCDVl-f_jgdPlWvHf4N6nADs-6YM8kewF4VJNIEyVNrOEUkLm9_uIuJngoY9pxvgxEC1zAi2210NPhLbB8rivvOpItlW2KWk436zezajGyS6AdwDNpW88L3QBNNXr2cx_g9KFAmCmCQ97jXDRu9UYyiYR66YblYrxZ3dfc6gy4FCH4yZP-89Kt-o-g1wM3DVKNAF1RhIwHPFP5yv09Zvu3zJYYFvI6k1mUeDBJM_ipR59ja5zLuJZpVRRCPy3GQ5z4ZbjkDjftmN3-A99Bh_6Uhx8MOMspGhdYvK6x__YGp8_UjpqmIMEmqfxsbrNaBaoisMflxzejTo4tlFGQlw9JFC2QHaYN6OG0-ibNF4VR8JSmDlQ8bqUJzXKQvyPKhrcQUqCRbq9N9TWnB7YZcyC18FBdwVrYNTaWMy8AVQUzFEpvErHRFmANwoQcunIYZFVEkBOnO-nf3Qkb6VD2SpnKRf-NGWTai4H3pdbw4-ZlECKi39BWT3w2Dtrp6erWeNyuYcLPasZ8eQoc-2sbn6ahRglb9ElefIrcdw3IIqEF1sE8qsMvoVlRl6trn_kIFZ3e63dSpgmvjWpxYLJwIhtyoOCR6ddGlr7Vz3sEoiVbmp_I7T046EdyIUjBUXfut29WZ5DWpTlaI-q2YsxwLJI5Z6jEAMOue-oJ0LlG_mfGvysspU8LUL6Ls4GOvR5kuk6eyxJ4axuaXICjCgSUJvEvJCz17gMLHKYYeErlYMUgz-GD1yO1pJvsK6k1NX3ggZGWR5Ra2RkoK1h70KidhBAsXiXEFxFNAMi8E0aB5WCydEVlNl79m86CDB7YTE9LyIxrMc9ZyYiSHzLJaLJHDQ10X0KLRxvjM2Upz68u7aJRtRU7JX135cGL5K1MqRFZMA821b-p1mTuez483R3-Q7fAPH2p59s-BUEEIeHJUfc0ZDg7mAEYhNmAtfWbfG5KB6IiCFMRWCm5jeCAUm1KdokUfyEv2I1Qw-JDmmHLdeUCk43If7wgcN33sLiNuS1TMJ3BBBC4qhHrkFYa3IU6ketzYqrqb9SJcWws5xDYxV2oCV5krYoNGg40BdtMfwUAmCI=',
                        'provider_name': 'openai',
                        'id': 'rs_00e767404995b9950068e6480f16f08191beaad2936e3d3195',
                    }
                },
            },
            {'type': 'tool-input-start', 'toolCallId': IsStr(), 'toolName': 'web_search', 'providerExecuted': True},
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"query":"OTEL_PYTHON_LOG_CORRELATION environment variable","type":"search"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': 'ws_00e767404995b9950068e648118bf88191aa7f804637c45b32',
                'toolName': 'web_search',
                'input': {'query': 'OTEL_PYTHON_LOG_CORRELATION environment variable', 'type': 'search'},
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': IsStr(),
                'output': {'status': 'completed'},
                'providerExecuted': True,
            },
            {'type': 'reasoning-start', 'id': IsStr()},
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': 'gAAAAABo5kgvRqkGOKegaH9grUqGCPO9vfDWr1XwkE-Wtzt6L0t1rany58X7-oMFUxwaIN89pOPaz2k5N6xvW7MZ7NcqhyTtYC03uG2IUnxEN981cD4nfmZq8YJejJK20r0rEEGNWlEzw8g-CAWaeLaEw8Cx3VeleP1vWY3gBurvGnzuhgYn4iZqHB9ShYSI9-nkJw0JNcnI7JFpprPVZlluDcKkTLcfXYfJBgwCTVDVmkqTqNeIxQD7VE6rtzXpgGKiLDoOTEpX-4NzKlFYLw2XN47LbOqlFfW--nNHtfbWh_W0b3cM-sL_PXVSr1jccFaNMR4V7fioRjcdG155tknwzrcWIWkUKHWy_u7kaCQPbG_RKcRVQ0eTziQOdphQESXlPsVrsSnvUYOhsXOEJsbJueHZEEtZZ9wXRk68fEGw0llPcvSEPBj4ThgOrtXKlmOenvvF-WQJbodsIwbXSt3pj7H6yu83IW0sHzmWeESM0f-6SXwZmLevctlxDHAIQnd3KOqRq6uGaYv4bcFIpXH_Rmflacl3R4hlfxUcMdETlywZaSyQhomLvsqxdS4D6BS6uofysgP0YUJO9fnf-18hbyu8OGq5DeeOmdQ8HgULGeyuRCeFeere7jY6blGWtKOzDAhRaaDQbzGZwF7FMbo6EouIYdyI6gyFctN7YlwE71vthJFdlSzHXVQpibNj5u17XXeHbIeY0SI2xciz3javvMxnsgmpTo0PSkcZMYr5OvcA3IqRZl3Q9XfTdySTeFr--kt5c0BLzrdrVamls2fp31jNKO4yVZczUsm3lEQPB-dy8r1KbaGKiEhc45ZoTF6dMoh4eSkpJpwpOb-Y-VOh4MjVAjwFWL4gHgTaTQYD8trLuBBRvf8uCdbF_CS-fA-sDRe843HKHVZM9l-J4rTOGGZ1dTvqOTGv7KV5XJNoGmdJ4rX6m6TAvtPJzQNFQDQcaCuzfYcakxgIPFaqdEoH_NFIKXGpnZLycxtpNm97Ol1qkzoW_K3VNTeEMiWEgy7jC7ATUtyzpZw3o46KsOog-veyfCV0RjZXiMnZLbWo7Syjfxxed7wn3u2jMCRt_8LsOzG8CuwuaMjCTxooxcd7KNBhYxX2LJhKl-F_cFZhPg5U6ciXORcYN9Eusw3P4KDD9A-s534N-xDHSb-cJmg9dmo50IAh2yPCxTMJMK8qIw73o57tHrSab5KQUk-jE3C1ZKRLNig86S49lXwOtZQmHjVVFCKj9V9o7Hx5nPYEb-eT9YwaZBtOB8t5JOvRThlCrX5ygFix48JTF34JJAcwwxDcgB7k4Xg5aeODslV4sggehZreJ7Etaj2m7ymGARjPvnd0-xpy4TE-heP_CX4hnJwL2JinX3FHJwEpDjahNkO7D3x54UmttS9RBCbPTKL7txwpshCm0yhZ3tBZWTN9OFm1HEJbjt5DSHCBKNjcWKWQOHQbfSis8pHjz2ipw_cp_c6qm-n2GsGvMkad56-2BYVZnkN49t5j_yu7nDV0pOifl6LrclQIxB1mRnpvv2bWyeshzgd_VSxIL9x0oLnZ61XQk9ao3FnTp9cUDZqBb67GiJ65agzMd_7vTDAcLFfhBtZX-_3kJ4BLldTxEuilmCtDCHa1PrLxqYKUH3jqSSNY0ya3cdRDNScv8ew4NzEBbYmzWh13IJdwqlgIceleHxKa70zZsudLmXrcoNxBJnQwSk35jxPuYGV-jD9J4pLR3kg9nvwtiTSEYQ8rQBLGiw6c5u9V4-3vhI91n8K1DFA-jQeusftJWUsdQamfONlnnGbDoMA_Da-9nHpNloTFJ6Effeb6RuIYAP-jP9EgXx-6oT8MNoahO7Y20VXf_laHYdNBDeZuiK2EFFyrFsgLEyLyWKSZy_dpsL0qgO1H2LIQ7jSYlfs0s4JC-mhkcfxRrLVsmmVWD1JWdfPXA4rrx1uq3rtNUeZSQQDn9xhgRsDXIbk2Dn6o7tB9jIk_dz3jSwcw76CRSIOelpyc2gH0TffGZ8ieXHhDNcJcjcYpd44J07eQylwC1LxpKtMHV_vlBEgB0hB_gfoVpYXFtx1vikqq_zS7LOegLrl01nypu37YNMPy7UaGxtb0nl64wZpb1B98DKP3VTJT-O3C8UoWqkAfY5a8Bjs8jdO1V-443_GRjOAQQE-qEdrTAM1bon767biPje56J01n0HgYAOPra1RFvj6FWhR81aAaikjfCGdEl7HIfmEdBiAf80kDlgbluiG4daPGs9M6pHU8JVIWqEUQc09H2QFjYMB6Rc4zbFsQezt1x5KvGInI8f5xYsKEHMnoPbp0EfR_6NOZZfRYoEByY_Us7qUN1rZLvf9OaUxbFM19KkRog3g8G_DtC859D4Q60WGmpdzptRmNClzk2r5yOgpUiESfuLyJdya4l-qEhLJK2kAOpSxCc_TI61oeO3I7NBvJZLg2FxOpoACHjMYK7XVKKZkQzo5Z_tY6RjIuMg41ihDvoCukiGxEmpEFL0IGsDXInuviVgwvXeYIYh1rrVSdQYhq8jxTleItj-fia0AXGKNsT_8eW50TS-teK66c_yiSf4ghJU86WrU627LmvgjeNM-JzZd-HVkcMSf2Sew65zhDgzUVte8ObpPHo1W_cXO_uQmcKWeQ8lJ7rD9Jt8G5H2VIA1-HCiZfkJN_3HH71cHcH3zuOh1zjdS9Sp1TUvBEoiIoyw9iZ6J6J8bBYham8TWuilgAmOsqZ7M14eiqr0-Wff3sCAEnpJ7BatjZ0CaQr3bUaRR7zxkRZi2052ivjbFNy_dM2RqFpOl5EdopBjVs6V4eETNd1-4jeWoAT8eMdo28O95MXyXnEwHPTYTKafjtvw6HyBGRLUb6tVNU04CWtswWjNYgrz30Ein3ym8GbgNWa6t9U-T6Utiv3KUwhA4lQSzm91XG7WEEhjzPeXL5MfOOkmbrVLecWrNQzaUgPaBZ04HwIXYBOF0wLuJEC2WG6RA2jxa2G5RS_EwX3Pph5_nAWDpHCv-sk81D-k9xYXdobZlD5dCK-4TbXu0IksOmlYkWTqgAj_GMat1PMLtgNsxACR3e1SgUQnOthvz1UZ7z4q46yMM6yJr-NN4Kxud-3jV1wF2GjIaQBveHIuC7i_we4ObNlt9FL68wC6zPgz6N_Ma4f-Rhu-zQpmIoJxoq8h6i7K0AjEIV5KB4LfSeWsIPh-wJtZihoU8H8MDLq2IRFDYskCj2tmQu56ciHphYWxyS3J2PaNJPJwdPYN8g-UAH_DGQjsXjOL57WW66fabkaMRTWaUh8_S8dj_PY_c51hxFYX5pOKLoZPOshEBFwIGfIAINUK41o39ZKY_tzcqfytYtZ8lUjCVXtNHKDWbluKbEiRWoWdp0IFkViRhvE5J2ZUfc3qNUSi5-opQmFf7FUSqSl33dIcc5nOenxX2cv6bKBYl5fZ8olRGr_YFbb9hiq4bN7QQ7BRlhGEfU-Jn5jkY6ousus5n8ZMDiTo121SLtM7FYHAwlDWFqaTPu0WLg60cAFqXru8eRB2n1HA-2WwoTkdz0SPnCjMoAvqZGW2Y1Fxr9tohOZJpOm-barcwKtFvWC1tOkgf2GnxReTIjYZq6cmVdhz55Wvgy5vduwWoFLGMb2EdeHl2C-y6uJI9t57junYhsxVx84mbHCJttZRB7nq1_ZLrnNOUh8Ot97Z5yMVK2Vfn-88HcT86Uo9O2hFgO0DilmqEIf89EXh-a44NGZvXLToRwZDc-JkmeRB5E9fCjF2uuOv9mPyrFtM8LhMteUunUoETNWE4Fq82svMKq9Mg-k-cNuAbzk34hClf4-c_yyHZw0LKuySRM8jYK9k5FPfDaLPTuDfJ96JeJ5GTb35cF83Ee9r0OVe1TvODMk6t1D6kAHmRlLK3MJkZktHq0ETGlKtNmfHIS7vyn8Ok7PafSWyd6c4Dz50vfNH5INXYm6sruqXlRLTyjL9eg6tKxguFBdk4wsKgVBIgzJ5p1Xs3iDNlwdKekxzPzpHp7blZLUfDtsLSFY0Krf0L1pmzWGSGpEzWFupadiYHG2O7MBe02wibtx01Wi-KyHbeiwO64VuHJplTZ_z0KYSzkrqmWlzy9hXxSu3lBcwbFPLgO0-dlhmJbXHORqBj_dBEbL2N-k9KJU6qqHSM7t1Jj4KaH-Y-8wizK5b5t0ZzU3XEtMY_EAwsdnpqJs4pXBEliZ56KuoemB-lAnd853rwIqcQ4A1Flw88uSMvOHTSTZ0ZirD0pCLyzwW13lBUDpgp7zwZHwDR7DNPUtPpHKDWeK6YkLO5EjsPSLfwhcpysCDdHfrMF1FrcwT_WE6Z7ui5o0LEi-335c95Q8qwlGof9P7UxOna7DjTunXijF-aJNPqmhKE4fY92Auvy4y2wvmAVNuLKdPdS9fVnhMZh19Air0ri8WiXyWvqZOPL1MYL3PElKC5uk_INyK6L3S7aMCmws1zFA6PSb2FvogcDSONeNM03Gd68nJlf4daNQXErKkyt5w5ULSHamEqcT3PZDa5en_X_R4lgL_pUFmy6K70Xxw4LiUvH0gO1-v1jP5IH6aNc6N1cXQs6tt1ajViXcD7aCcUovQ2Ejmvq9EUmgVLWkMaXJB06c7ErXME7-9wqas_bj48FBV9mGwDLxif1lsxoe7pqT8xZ9QwGiKoRVb4t7nJUqLWcx3oGDgEetUMyI0VctLixoDdvuO4CIZliRm6ilMomyfNwlHdBMoU7-AhZThBjp8eA9vMz7LWIl-TFuueueodO9FZEdGOs6uBZffBYU39oImI0di6oeT5upaPkVCBEBLmwHmDEsfiaKR8IPuVjNSo-gX8JKu-GHmECGYb_3ffN4C3IG5w3RACsKzQfL2L_tXBF5a1T0NtDtrjeGI-kYFhwd4mM1HQFNKVYx_475tInEGmwbdtzYKi4OrdOnNUG8QyxnRl5tBXWwOqHvorIJvwzQHgm9jnaNMekwUhb_8fiUhm16G1qhBLQZlz7eofhB9WcfuC4dxrdvMhfGEsXH9b_gCfj8vazaJyfmm6PGnyJIK6b82dXtE6JIJfwmX-m1J7AGCaW0So98-XuOUAvDfxDtiOap7JxiIRbrmrY9rLDAMAkE3BmXY9tX-_tMQm8c2pr3ioc94mWntYbIT0QIiGpentwiHJHComvQkb9Ss3DtOuz5vtP6ImJXXOMzb7FVEg6Qc38qfpSw4XtYfrjI5EaCmPl9P0exeeHN_RrI9-VfuM8KdMrjlHd1LbgwkOOKB-O8YpcuRgUTySEDngxQYYA62jmu03Dfc6eyA4WwptQQZZt5RPiXRD-gxOdhjcNGf6WLMYTmI_pBeo56qSbhbt_tllPYbvP1NzZcKb-ssdYZX9m0eEZTqxWeHT4iay6kq3QIo7WUD6PYXin5jRpGfB6FstSoJcUhcJ5B0ld6A3ij6DsdzOy799EVtxrANZ0VEEg86O1x2ft8wl481wDAcDBObPrTS_i7peacVDWiWxYIRWe3NGXCmq6ZU9lBB1iQeOZ16WRVU1g50_AHxiuwPU_x6DmWXTofqr5VNcpMeIMe7JUxE4OF97r9J7fVomM3JoSUirPXc9Fqk0vqj2aZnqOLyq7AAT-4yqb72QpokQhSixAfWUxg167vijcSHoijL_OOxg8LTBrBnmLKTKwsbKtlSJzY9Pj4iws4tQRpH6LkNNkB0EVZBQNuqEzOy1GAeVfC3fttdV6oQLpq7ZGwt777Fw6HORX8dhqpOsaHyQz-VeWxtnZEfWLfWyIy1gouiDmTLdoNKmON8xy_dAWuEerNf0_Hqm8zbqlBGq2xiictUZZTrPtiX-u5y_nRldhSSIz5lDlDaNcSCNnlElMxjQ-pbFKQg_Zr2jGJeuAY6AuIvW9M7sDROlh8loCGzO9klwYfBjdldE0HsQhyJ8h2sg7qRaxcgGSFdnIVaSnj5tLyVtOiskFQOGj1U7tUD5jE_6nro0Np90_N2X4OquME6lqwofyMOSU3kUo-4CKiq9EF9RksWM9NCpZ59XQcWGTKqujf1cNXlXo4Oz27eRLGzUAhxY8XdTdxEtQ9qiNlAgnWMwFwDuJneZKoKxE3BFf2vXtwUdTMG3mAaLnRfqxRNb8o1QthMVoeLGVDIf6bTGGBVjOHJ5Rsdbq1ZsiQzJcNynvSfAK57BzMIOF53JZ-R8qvrgxapUM7gIqgtdajzTdl9Dj8MWcm1n6DQSWd0au6ZskpDnyfRmF0Vx5qYOwdHcWel0XtDUEh4pBzCxpXnIR2UFF3EbUYfSOZlk2c4RrdGaKApiJhrSveWzbyA5BwbmSj5QRYc8WY1Rjwf8FygQRXQpnFPsgoMuyxcverMLIOGJ2vnAPy3sQ9d8nOYQlOVD9kXv27eRl6ygYXGNm6GRr9lpRU9TFGaoLF4JDFQbQYvQvM_UaGUAM9h_X2Lnnljw4s9AFPV1QR4MgKg8_X6dQt3DKD0Lf76I8jXqdIzIJvBNbY3tAAgvC7uypf5cqUk27ImdMZ2OZoJmz9NcqTiutxBRmJbTIzzWqdwHLKgjeAz3OdTf-eNQO0qNSsMGs8-xC6xm7puBCjKIFkc2xbd3F2HZ4N91sGczTakCJB-5h3G2TcKHFZptvEGfvM06KqRlq6JG_XVgZuG1_O4im0FgDoEgpsR3_wddshPpOavxFpAzOOIY7A344aHAHXyGZtU0SG0Otbq2U2iUTSsmmnvbwTKhnh8OYyytl_zoUPmNSNdkOrQLw0T2K4DNTXyhhdttvff4wSlZ5JNgPWc1vS8Olz_QE1EuZFDrxRFkQo0Tf3mhI1c4LGp4WY-MuMXhS90rQFIsuCUcx_l7KqKHcJMxMFri7M0uY1NwBgvTjlgwsNwZAF1CTUBpnbXMk4dBkxDNM-ha5w3PF3u1JLKNxlGtlqX7QeQQc7zmMym3grPBUM--XV70NvwAKRaGlPSGLnM3ESjO5tXEH1T0ksEP9CSAPfMWk2N4B2Mr6tyyBQTY1GYLDJlam_PbnIVySDjQB9IoPYzfdI4vUWGY-rX8rPF2pMde24r6SqUI9K3hbSgkb9gq1SUNoAsuZRhelLURV_UM3Y777XdSxdCQzoh7sTFu547cwX97Yr3cBZFyzf0BHlt5mNX0WcQK_CHJr_QqYIGrkMilcLulApWXdYTb0P84ntX_LR7rrmFQBfjlVIacsAJ4YvitUS8uRKfVdG1BxlCxaUmVBJjjSYwyPCEUy3g7EyUSn7TSUKj3zBH6o1mZQTXHd36j8AEqeCWNofmrr__j0dMcg7mGlAEE-5h547MDpfNn_bUl6N7S5FzIF_5Y7Cu6c6RtCb63F_XVZm9nMrYOaww-0OAKjPiLiQ59fE3Yb9tACEgwmbcdh_txOmqDu4wIeJmspE6qgxfyadieE9j1gdI0rNWhe7saTKzzo48wGT9ljOrBRUAKmjYR14RwRqfeJqG2hHKAWoZrWAQ2QlxXk3SzVhHXzBTEmCCO_hZFd4YuhSRi65EeAakE_NyDzcsz05Ez4aZSNXonyv7CsNLc-EXrxXgVBReRKxawt8e-NMFy4monSuxBalAr1LArj3f9HHgnb3dERoqG-qfQr6c_0Al4kdm1FFsuX8vtbIPaByUPVI_fQfu3BScJZkWZEMZAoTliy78JhxKPVEQq1JjTVc0rGCcFl5iL-_s6oK9vYbgevfPTl8PFBNUbEr6cfNvikMOeJ564506ZVuzKsvTgXniSQvlk8dgDW29AGSE7dzeLQqq5IdLGqqz_fSc2mQxcAcJgwzRLT1LQr0WkHC17yIbh6MtCfRcLm-vyTuzLiDeTBqoOrH-3WOB3AnHae3nRrlwAR_UFsA1l_fme2vruv2MUY9y8DHX4vwmAhNpj1w29xEuzQpjNEMjKLJiyvf4CwzN7tQU3zUtRnQ8kTyoj2Re5EzETQ8jeF7ZWCZwmMbSQWH4xDyYj5_P96dMaUHGaLBOW-hT5fG56kCyUnV_AcnYAa55FeY-J_j0qHdUV3ZK-bizwcNIlI_kpe0GUGajdNvJZv6r4p9-BxRIDbIHidNjaP8nnyScCYU9uWfJxykcKGgl7jZeWfSAfJASkwsbbfH6Ug7Msv9i2R2HnZNK3WdlAWA4IcpHw19wp79Jq9yE--FTuWaORGD_jcjE1bNPDbQ1AFXc_naxhzV5-XRsFSa_wp4qQznepjGjc8PpczcoCcdfYobR1ZOcrBhhtufEoA_1-ZRjl3f3HPVX3rLFMrIIopS7saLTC_d3e1J3f9CQktA2zGfMKErQmyBPk-UpW1QnTVi3E_qB_IDbQNkV_cOU3doKW9hQXPb9di1An2qhOQSoBaoEnF4xxUrrlfrF7g4WZKM4hcJ-LXNOIk0lhlImnArFnepgH2outlUM1zdn99PXWFbVnnj7VTIvHQ7ZiLV3ej_Tv7oimAme2zB1f83PkrAXPZZ-TzfrFkyFSwOJs32U7VBpKjVhsd3_RxMtrm5QkZiMHKfobGlV5Wd5dmpYgdjSyVv_FkJYVd4itJYgfIh-R9ckUy25Qmy6mlKPCHPB6t1-Ip5e9ihYCJ3jiCCTuw6lGdp2quZc2VWWOm13nPA1Xol0bGIPVMFtuKXYXCRomgSomjqhm-z0NVvQrMsm_YFXl8Yiajq5mXxpDI3rkV5NbHca8mQItDwZ2sfbjtD_UIpk5lGXufs2M5xOzrdk8hSOfy3tLR1q-OkOZaE9U_KWPkcP-FESCoG7sTnVHpT18Ht7s0AneBoo_3q1qEEcX3jmGHwlerOLjpSyZRTaspteUfYXH_0NeiL2HBn3aMW14GP9VsBHAUPlynmFlxiqM8YDpG6bIAt4Yila0cCM9DLn8ja3_-tJt7rIdf7iCJd_753PG24KeKpr13wea0QZ3aBoXwnpqa-C4gbH7hKnExALeEDoUzc8vdENrlaJTYCAI6yM7lg_EZYSRhrTFqDZYFKW53I_i8YU8IfO_7R8s3A2jbTSuhf9WhxD30HvbAlDGk27LSbF0keP88_Vnru_mFuz7z_u59iwhnfDxyEmgNtwPWj7NIOThc5CdEjvTiCwMhHF6u5CxZDN-YNtwQK3ai0dnp2w==',
                        'provider_name': 'openai',
                        'id': 'rs_00e767404995b9950068e648130f0481918dc71103fbd6a486',
                    }
                },
            },
            {'type': 'text-start', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
Short answer:
- Default\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' FastAPI/OpenTelemetry', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': ' instrumentation already records method',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '/route/status', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
.
- To also\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' include HTTP headers', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ', set', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' the capture-', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'headers env', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
 vars.
-\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' To include request', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '/response bodies', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ', use the', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' FastAPI', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '/ASGI', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' request/response', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' hooks and add', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' the', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' payload to', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' the span yourself', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' (with red', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'action/size', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
 limits).

How\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' to do it', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\


1)\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' Enable header capture', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' (server side', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
)
- Choose\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' just the', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' headers you need; avoid', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': ' sensitive ones or sanitize',
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
 them.

export OTEL\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': '_INSTRUMENTATION_HTTP_CAPTURE',
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': '_HEADERS_SERVER_REQUEST="content',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '-type,user', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '-agent"\n', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': 'export OTEL_INSTRUMENTATION',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '_HTTP_CAPTURE_HEADERS', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': '_SERVER_RESPONSE="content-type"\n',
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': 'export OTEL_INSTRUMENTATION_HTTP',
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
_CAPTURE_HEADERS_SANITIZE_FIELDS="authorization,set-cookie"

This makes headers appear on spans as http.request.header.* and http.response.header.*. ([opentelemetry-python-contrib.readthedocs.io](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html))

2)\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': ' Add hooks to capture request',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '/response bodies', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\

Note:\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': IsStr(), 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' a built-in Python', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' env', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' var to', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' auto-capture', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' HTTP bodies for Fast', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'API/AS', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'GI. Use', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' hooks to look at', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' ASGI receive', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '/send events and', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' attach (tr', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'uncated) bodies', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' as span attributes', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
.

from\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' fastapi import', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' FastAPI', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\

from opente\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': 'lemetry.trace', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' import Span', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\

from opente\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': 'lemetry.instrument', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'ation.fastapi import', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' FastAPIInstrument', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
or

MAX\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '_BYTES = ', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '2048 ', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' # keep this', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' small in prod', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\


def client\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '_request_hook(span', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ': Span,', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' scope: dict', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ', message:', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
 dict):
   \
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' if span and', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' span.is_record', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'ing() and', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' message.get("', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'type") ==', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' "http.request', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
":
        body\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' = message.get', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '("body")', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' or b"', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
"
        if\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
 body:
           \
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' span.set_attribute', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
(
                "\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': 'http.request.body', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
",
                body\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '[:MAX_BYTES', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '].decode("', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'utf-8', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '", "replace', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
"),
            )
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\

def client_response\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '_hook(span:', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' Span, scope', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ': dict,', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' message: dict', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
):
    if\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' span and span', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '.is_recording', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '() and message', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '.get("type', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '") == "', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'http.response.body', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
":
        body\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' = message.get', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '("body")', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' or b"', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
"
        if\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
 body:
           \
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' span.set_attribute', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
(
                "\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': 'http.response.body', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
",
                body\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '[:MAX_BYTES', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '].decode("', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'utf-8', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '", "replace', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
"),
            )
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\

app = Fast\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
API()
Fast\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': 'APIInstrumentor', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '.instrument_app(', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\

    app,\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\

    client_request\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '_hook=client', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
_request_hook,
   \
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' client_response_hook', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '=client_response', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
_hook,
)
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\

- The hooks\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' receive the AS', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'GI event dict', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 's: http', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '.request (with', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' body/more', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '_body) and', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' http.response.body', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '. If your', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' bodies can be', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' chunked,', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' you may need', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' to accumulate across', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' calls when message', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '.get("more', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '_body") is', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' True. ', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': '([opentelemetry-python-contrib.readthedocs.io](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html)',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ')', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\


3)\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' Be careful with', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' PII and', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
 size
-\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' Always limit size', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' and consider redaction', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' before putting payloads', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
 on spans.
-\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' Use the sanitize', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' env var above', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' for sensitive headers', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '. ', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': '([opentelemetry-python-contrib.readthedocs.io](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html))\n',
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\

Optional: correlate logs\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
 with traces
-\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' If you also want', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' request/response', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' details in logs with', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' trace IDs, enable', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' Python log correlation:\n', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\

export OTEL_P\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': 'YTHON_LOG_COR', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'RELATION=true', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\


or programmatically\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
:
from opente\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': 'lemetry.instrumentation', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': '.logging import LoggingInstrument',
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
or
LoggingInstrument\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': 'or().instrument(set', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '_logging_format=True)\n', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\

This injects trace\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '_id/span_id into', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' log records so you', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' can line up logs', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' with the span that', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' carries the HTTP payload', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' attributes. ', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': '([opentelemetry-python-contrib.readthedocs.io](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/logging/logging.html?utm_source=openai))\n',
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\

Want me to tailor\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' the hook to only', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' capture JSON bodies,', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' skip binary content,', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' or accumulate chunked', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' bodies safely?', 'id': IsStr()},
            {'type': 'text-end', 'id': IsStr()},
            {'type': 'finish-step'},
            {'type': 'finish', 'finishReason': 'stop'},
            '[DONE]',
        ]
    )


async def test_run_stream_text_and_thinking():
    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaThinkingCalls | str]:
        yield {0: DeltaThinkingPart(content='Half of ')}
        yield {0: DeltaThinkingPart(content='a thought')}
        yield {1: DeltaThinkingPart(content='Another thought')}
        yield {2: DeltaThinkingPart(content='And one more')}
        yield 'Half of '
        yield 'some text'
        yield {5: DeltaThinkingPart(content='More thinking')}

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Tell me about Hello World')],
            ),
        ],
    )

    adapter = VercelAIAdapter(agent, request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in adapter.encode_stream(adapter.run_stream())
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'reasoning-start', 'id': IsStr()},
            {'type': 'reasoning-delta', 'id': IsStr(), 'delta': 'Half of '},
            {'type': 'reasoning-delta', 'id': IsStr(), 'delta': 'a thought'},
            {'type': 'reasoning-end', 'id': IsStr()},
            {'type': 'reasoning-start', 'id': IsStr()},
            {'type': 'reasoning-delta', 'id': IsStr(), 'delta': 'Another thought'},
            {'type': 'reasoning-end', 'id': IsStr()},
            {'type': 'reasoning-start', 'id': IsStr()},
            {'type': 'reasoning-delta', 'id': IsStr(), 'delta': 'And one more'},
            {'type': 'reasoning-end', 'id': IsStr()},
            {'type': 'text-start', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'Half of ', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'some text', 'id': IsStr()},
            {'type': 'text-end', 'id': IsStr()},
            {'type': 'reasoning-start', 'id': IsStr()},
            {'type': 'reasoning-delta', 'id': IsStr(), 'delta': 'More thinking'},
            {'type': 'reasoning-end', 'id': IsStr()},
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_run_stream_thinking_with_signature():
    """Test that thinking parts with signatures include providerMetadata in reasoning-end events."""

    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaThinkingCalls | str]:
        yield {0: DeltaThinkingPart(content='Let me think...')}
        yield {0: DeltaThinkingPart(signature='sig_abc123')}
        yield 'Here is my answer.'

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Think about something')],
            ),
        ],
    )

    adapter = VercelAIAdapter(agent, request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in adapter.encode_stream(adapter.run_stream())
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'reasoning-start', 'id': IsStr()},
            {'type': 'reasoning-delta', 'id': IsStr(), 'delta': 'Let me think...'},
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {'pydantic_ai': {'signature': 'sig_abc123', 'provider_name': 'function'}},
            },
            {'type': 'text-start', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'Here is my answer.', 'id': IsStr()},
            {'type': 'text-end', 'id': IsStr()},
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_event_stream_thinking_end_with_full_metadata():
    """Test handle_thinking_end with all metadata fields (signature, provider_name, provider_details, id)."""

    async def event_generator():
        part = ThinkingPart(
            content='Deep thought...',
            id='thinking_456',
            signature='sig_xyz789',
            provider_name='anthropic',
            provider_details={'model': 'claude-3', 'tokens': 100},
        )
        yield PartStartEvent(index=0, part=part)
        yield PartEndEvent(index=0, part=part)

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Think deeply')],
            ),
        ],
    )
    event_stream = VercelAIEventStream(run_input=request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in event_stream.encode_stream(event_stream.transform_stream(event_generator()))
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'reasoning-start', 'id': IsStr()},
            {'type': 'reasoning-delta', 'id': IsStr(), 'delta': 'Deep thought...'},
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': 'sig_xyz789',
                        'provider_name': 'anthropic',
                        'provider_details': {'model': 'claude-3', 'tokens': 100},
                        'id': 'thinking_456',
                    }
                },
            },
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_event_stream_back_to_back_text():
    async def event_generator():
        yield PartStartEvent(index=0, part=TextPart(content='Hello'))
        yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' world'))
        yield PartEndEvent(index=0, part=TextPart(content='Hello world'), next_part_kind='text')
        yield PartStartEvent(index=1, part=TextPart(content='Goodbye'), previous_part_kind='text')
        yield PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' world'))
        yield PartEndEvent(index=1, part=TextPart(content='Goodbye world'))

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Hello')],
            ),
        ],
    )
    event_stream = VercelAIEventStream(run_input=request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in event_stream.encode_stream(event_stream.transform_stream(event_generator()))
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'text-start', 'id': (message_id := IsSameStr())},
            {'type': 'text-delta', 'delta': 'Hello', 'id': message_id},
            {'type': 'text-delta', 'delta': ' world', 'id': message_id},
            {'type': 'text-delta', 'delta': 'Goodbye', 'id': message_id},
            {'type': 'text-delta', 'delta': ' world', 'id': message_id},
            {'type': 'text-end', 'id': message_id},
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_run_stream_builtin_tool_call():
    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[BuiltinToolCallsReturns | DeltaToolCalls | str]:
        yield {
            0: BuiltinToolCallPart(
                tool_name=WebSearchTool.kind,
                args='{"query":',
                tool_call_id='search_1',
                provider_name='function',
            )
        }
        yield {
            0: DeltaToolCall(
                json_args='"Hello world"}',
                tool_call_id='search_1',
            )
        }
        yield {
            1: BuiltinToolReturnPart(
                tool_name=WebSearchTool.kind,
                content=[
                    {
                        'title': '"Hello, World!" program',
                        'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program',
                    }
                ],
                tool_call_id='search_1',
                provider_name='function',
            )
        }
        yield 'A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". '

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Tell me about Hello World')],
            ),
        ],
    )
    adapter = VercelAIAdapter(agent, request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in adapter.encode_stream(adapter.run_stream())
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'tool-input-start', 'toolCallId': 'search_1', 'toolName': 'web_search', 'providerExecuted': True},
            {'type': 'tool-input-delta', 'toolCallId': 'search_1', 'inputTextDelta': '{"query":'},
            {'type': 'tool-input-delta', 'toolCallId': 'search_1', 'inputTextDelta': '"Hello world"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'search_1',
                'toolName': 'web_search',
                'input': {'query': 'Hello world'},
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'function'}},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': 'search_1',
                'output': [
                    {
                        'title': '"Hello, World!" program',
                        'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program',
                    }
                ],
                'providerExecuted': True,
            },
            {'type': 'text-start', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': 'A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". ',
                'id': IsStr(),
            },
            {'type': 'text-end', 'id': IsStr()},
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_run_stream_tool_call():
    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {
                0: DeltaToolCall(
                    name='web_search',
                    json_args='{"query":',
                    tool_call_id='search_1',
                )
            }
            yield {
                0: DeltaToolCall(
                    json_args='"Hello world"}',
                    tool_call_id='search_1',
                )
            }
        else:
            yield 'A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". '

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    @agent.tool_plain
    async def web_search(query: str) -> dict[str, list[dict[str, str]]]:
        return {
            'results': [
                {
                    'title': '"Hello, World!" program',
                    'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program',
                }
            ]
        }

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Tell me about Hello World')],
            ),
        ],
    )
    adapter = VercelAIAdapter(agent, request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in adapter.encode_stream(adapter.run_stream())
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'tool-input-start', 'toolCallId': 'search_1', 'toolName': 'web_search'},
            {'type': 'tool-input-delta', 'toolCallId': 'search_1', 'inputTextDelta': '{"query":'},
            {'type': 'tool-input-delta', 'toolCallId': 'search_1', 'inputTextDelta': '"Hello world"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'search_1',
                'toolName': 'web_search',
                'input': {'query': 'Hello world'},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': 'search_1',
                'output': {
                    'results': [
                        {
                            'title': '"Hello, World!" program',
                            'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program',
                        }
                    ]
                },
            },
            {'type': 'finish-step'},
            {'type': 'start-step'},
            {'type': 'text-start', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': 'A "Hello, World!" program is usually a simple computer program that emits (or displays) to the screen (often the console) a message similar to "Hello, World!". ',
                'id': IsStr(),
            },
            {'type': 'text-end', 'id': IsStr()},
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_event_stream_file():
    async def event_generator():
        yield PartStartEvent(index=0, part=FilePart(content=BinaryImage(data=b'fake', media_type='image/png')))

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Hello')],
            ),
        ],
    )
    event_stream = VercelAIEventStream(run_input=request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in event_stream.encode_stream(event_stream.transform_stream(event_generator()))
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'file', 'url': 'data:image/png;base64,ZmFrZQ==', 'mediaType': 'image/png'},
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_run_stream_output_tool():
    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        yield {
            0: DeltaToolCall(
                name='final_result',
                json_args='{"query":',
                tool_call_id='search_1',
            )
        }
        yield {
            0: DeltaToolCall(
                json_args='"Hello world"}',
                tool_call_id='search_1',
            )
        }

    def web_search(query: str) -> dict[str, list[dict[str, str]]]:
        return {
            'results': [
                {
                    'title': '"Hello, World!" program',
                    'url': 'https://en.wikipedia.org/wiki/%22Hello,_World!%22_program',
                }
            ]
        }

    agent = Agent(model=FunctionModel(stream_function=stream_function), output_type=web_search)

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Tell me about Hello World')],
            ),
        ],
    )
    adapter = VercelAIAdapter(agent, request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in adapter.encode_stream(adapter.run_stream())
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'tool-input-start', 'toolCallId': 'search_1', 'toolName': 'final_result'},
            {'type': 'tool-input-delta', 'toolCallId': 'search_1', 'inputTextDelta': '{"query":'},
            {'type': 'tool-input-delta', 'toolCallId': 'search_1', 'inputTextDelta': '"Hello world"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'search_1',
                'toolName': 'final_result',
                'input': {'query': 'Hello world'},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': 'search_1',
                'output': 'Final result processed.',
            },
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_run_stream_response_error():
    async def stream_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        yield {
            0: DeltaToolCall(
                name='unknown_tool',
            )
        }

    agent = Agent(model=FunctionModel(stream_function=stream_function))

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Tell me about Hello World')],
            ),
        ],
    )
    adapter = VercelAIAdapter(agent, request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in adapter.encode_stream(adapter.run_stream())
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {
                'type': 'tool-input-start',
                'toolCallId': IsStr(),
                'toolName': 'unknown_tool',
            },
            {'type': 'tool-input-available', 'toolCallId': IsStr(), 'toolName': 'unknown_tool', 'input': {}},
            {
                'type': 'tool-output-error',
                'toolCallId': IsStr(),
                'errorText': """\
Unknown tool name: 'unknown_tool'. No tools available.

Fix the errors and try again.\
""",
            },
            {'type': 'finish-step'},
            {'type': 'start-step'},
            {
                'type': 'tool-input-start',
                'toolCallId': IsStr(),
                'toolName': 'unknown_tool',
            },
            {'type': 'tool-input-available', 'toolCallId': IsStr(), 'toolName': 'unknown_tool', 'input': {}},
            {'type': 'error', 'errorText': 'Exceeded maximum retries (1) for output validation'},
            {'type': 'finish-step'},
            {'type': 'finish', 'finishReason': 'error'},
            '[DONE]',
        ]
    )


async def test_run_stream_request_error():
    agent = Agent(model=TestModel())

    @agent.tool_plain
    async def tool(query: str) -> str:
        raise ValueError('Unknown tool')

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Hello')],
            ),
        ],
    )
    adapter = VercelAIAdapter(agent, request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in adapter.encode_stream(adapter.run_stream())
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'tool-input-start', 'toolCallId': 'pyd_ai_tool_call_id__tool', 'toolName': 'tool'},
            {'type': 'tool-input-delta', 'toolCallId': 'pyd_ai_tool_call_id__tool', 'inputTextDelta': '{"query":"a"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'pyd_ai_tool_call_id__tool',
                'toolName': 'tool',
                'input': {'query': 'a'},
            },
            {'type': 'error', 'errorText': 'Unknown tool'},
            {'type': 'finish-step'},
            {'type': 'finish', 'finishReason': 'error'},
            '[DONE]',
        ]
    )


async def test_run_stream_on_complete_error():
    agent = Agent(model=TestModel())

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Hello')],
            ),
        ],
    )

    def raise_error(run_result: AgentRunResult[Any]) -> None:
        raise ValueError('Faulty on_complete')

    adapter = VercelAIAdapter(agent, request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in adapter.encode_stream(adapter.run_stream(on_complete=raise_error))
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'text-start', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'success ', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '(no ', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'tool ', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'calls)', 'id': IsStr()},
            {'type': 'text-end', 'id': IsStr()},
            {'type': 'error', 'errorText': 'Faulty on_complete'},
            {'type': 'finish-step'},
            {'type': 'finish', 'finishReason': 'error'},
            '[DONE]',
        ]
    )


async def test_run_stream_on_complete():
    agent = Agent(model=TestModel())

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Hello')],
            ),
        ],
    )

    async def on_complete(run_result: AgentRunResult[Any]) -> AsyncIterator[BaseChunk]:
        yield DataChunk(type='data-custom', data={'foo': 'bar'})

    adapter = VercelAIAdapter(agent, request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in adapter.encode_stream(adapter.run_stream(on_complete=on_complete))
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'text-start', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'success ', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '(no ', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'tool ', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'calls)', 'id': IsStr()},
            {'type': 'text-end', 'id': IsStr()},
            {'type': 'data-custom', 'data': {'foo': 'bar'}},
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_data_chunk_with_id_and_transient():
    """Test DataChunk supports optional id and transient fields for AI SDK compatibility."""
    agent = Agent(model=TestModel())

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Hello')],
            ),
        ],
    )

    async def on_complete(run_result: AgentRunResult[Any]) -> AsyncIterator[BaseChunk]:
        # Yield a data chunk with id for reconciliation
        yield DataChunk(type='data-task', id='task-123', data={'status': 'complete'})
        # Yield a transient data chunk (not persisted to history)
        yield DataChunk(type='data-progress', data={'percent': 100}, transient=True)

    adapter = VercelAIAdapter(agent, request)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in adapter.encode_stream(adapter.run_stream(on_complete=on_complete))
    ]

    # Verify the data chunks are present in the events with correct fields
    assert {'type': 'data-task', 'id': 'task-123', 'data': {'status': 'complete'}} in events
    assert {'type': 'data-progress', 'data': {'percent': 100}, 'transient': True} in events


@pytest.mark.skipif(not starlette_import_successful, reason='Starlette is not installed')
async def test_adapter_dispatch_request():
    agent = Agent(model=TestModel())
    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Hello')],
            ),
        ],
    )

    async def receive() -> dict[str, Any]:
        return {'type': 'http.request', 'body': request.model_dump_json().encode('utf-8')}

    starlette_request = Request(
        scope={
            'type': 'http',
            'method': 'POST',
            'headers': [
                (b'content-type', b'application/json'),
            ],
        },
        receive=receive,
    )

    response = await VercelAIAdapter.dispatch_request(starlette_request, agent=agent)

    assert isinstance(response, StreamingResponse)

    chunks: list[str | dict[str, Any]] = []

    async def send(data: MutableMapping[str, Any]) -> None:
        body = cast(bytes, data.get('body', b'')).decode('utf-8').strip().removeprefix('data: ')
        if not body:
            return
        if body == '[DONE]':
            chunks.append('[DONE]')
        else:
            chunks.append(json.loads(body))

    await response.stream_response(send)

    assert chunks == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'text-start', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'success ', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '(no ', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'tool ', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'calls)', 'id': IsStr()},
            {'type': 'text-end', 'id': IsStr()},
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_adapter_load_messages():
    data = SubmitMessage(
        trigger='submit-message',
        id='bvQXcnrJ4OA2iRKU',
        messages=[
            UIMessage(
                id='foobar',
                role='system',
                parts=[
                    TextUIPart(
                        text='You are a helpful assistant.',
                    ),
                ],
            ),
            UIMessage(
                id='BeuwNtYIjJuniHbR',
                role='user',
                parts=[
                    TextUIPart(
                        text='Here are some files:',
                    ),
                    FileUIPart(
                        media_type='image/png',
                        url='data:image/png;base64,ZmFrZQ==',
                    ),
                    FileUIPart(
                        media_type='image/png',
                        url='https://example.com/image.png',
                    ),
                    FileUIPart(
                        media_type='video/mp4',
                        url='https://example.com/video.mp4',
                    ),
                    FileUIPart(
                        media_type='audio/mpeg',
                        url='https://example.com/audio.mp3',
                    ),
                    FileUIPart(
                        media_type='application/pdf',
                        url='https://example.com/document.pdf',
                    ),
                ],
            ),
            UIMessage(
                id='bylfKVeyoR901rax',
                role='assistant',
                parts=[
                    ReasoningUIPart(
                        text='I should tell the user how nice those files are and share another one',
                    ),
                    TextUIPart(
                        text='Nice files, here is another one:',
                        state='streaming',
                    ),
                    FileUIPart(
                        media_type='image/png',
                        url='data:image/png;base64,ZmFrZQ==',
                    ),
                ],
            ),
            UIMessage(
                id='MTdh4Ie641kDuIRh',
                role='user',
                parts=[TextUIPart(type='text', text='Give me the ToCs', state=None, provider_metadata=None)],
            ),
            UIMessage(
                id='3XlOBgFwaf7GsS4l',
                role='assistant',
                parts=[
                    TextUIPart(
                        text="I'll get the table of contents for both repositories.",
                        state='streaming',
                    ),
                    ToolOutputAvailablePart(
                        type='tool-get_table_of_contents',
                        tool_call_id='toolu_01XX3rjFfG77h3KCbVHoYJMQ',
                        input={'repo': 'pydantic'},
                        output="[Scrubbed due to 'API Key']",
                    ),
                    DynamicToolOutputAvailablePart(
                        tool_name='get_table_of_contents',
                        tool_call_id='toolu_01XX3rjFfG77h3KCbVHoY',
                        input={'repo': 'pydantic-ai'},
                        output="[Scrubbed due to 'API Key']",
                    ),
                    ToolOutputErrorPart(
                        type='tool-get_table_of_contents',
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2zZ4sz9g',
                        input={'repo': 'logfire'},
                        error_text="Can't do that",
                    ),
                    ToolOutputAvailablePart(
                        type='tool-web_search',
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2zZ4s',
                        input={'query': 'What is Logfire?'},
                        output="[Scrubbed due to 'Auth']",
                        provider_executed=True,
                        call_provider_metadata={'pydantic_ai': {'provider_name': 'openai'}},
                    ),
                    ToolOutputErrorPart(
                        type='tool-web_search',
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2z',
                        input={'query': 'What is Logfire?'},
                        error_text="Can't do that",
                        provider_executed=True,
                        call_provider_metadata={'pydantic_ai': {'provider_name': 'openai'}},
                    ),
                    TextUIPart(
                        text="""Here are the Table of Contents for both repositories:... Both products are designed to work together - Pydantic AI for building AI agents and Logfire for observing and monitoring them in production.""",
                        state='streaming',
                    ),
                    FileUIPart(
                        media_type='application/pdf',
                        url='data:application/pdf;base64,ZmFrZQ==',
                    ),
                    ToolInputAvailablePart(
                        type='tool-get_table_of_contents',
                        tool_call_id='toolu_01XX3rjFfG77h',
                        input={'repo': 'pydantic'},
                    ),
                    ToolInputAvailablePart(
                        type='tool-web_search',
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2zZ4s',
                        input={'query': 'What is Logfire?'},
                        provider_executed=True,
                    ),
                ],
            ),
        ],
    )

    messages = VercelAIAdapter.load_messages(data.messages)
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='You are a helpful assistant.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content=[
                            'Here are some files:',
                            BinaryImage(data=b'fake', media_type='image/png', _identifier='c053ec'),
                            ImageUrl(url='https://example.com/image.png', _media_type='image/png'),
                            VideoUrl(url='https://example.com/video.mp4', _media_type='video/mp4'),
                            AudioUrl(url='https://example.com/audio.mp3', _media_type='audio/mpeg'),
                            DocumentUrl(url='https://example.com/document.pdf', _media_type='application/pdf'),
                        ],
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='I should tell the user how nice those files are and share another one'),
                    TextPart(content='Nice files, here is another one:'),
                    FilePart(content=BinaryImage(data=b'fake', media_type='image/png', _identifier='c053ec')),
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Give me the ToCs',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(content="I'll get the table of contents for both repositories."),
                    ToolCallPart(
                        tool_name='get_table_of_contents',
                        args={'repo': 'pydantic'},
                        tool_call_id='toolu_01XX3rjFfG77h3KCbVHoYJMQ',
                    ),
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_table_of_contents',
                        content="[Scrubbed due to 'API Key']",
                        tool_call_id='toolu_01XX3rjFfG77h3KCbVHoYJMQ',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_table_of_contents',
                        args={'repo': 'pydantic-ai'},
                        tool_call_id='toolu_01XX3rjFfG77h3KCbVHoY',
                    )
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_table_of_contents',
                        content="[Scrubbed due to 'API Key']",
                        tool_call_id='toolu_01XX3rjFfG77h3KCbVHoY',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_table_of_contents',
                        args={'repo': 'logfire'},
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2zZ4sz9g',
                    )
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content="Can't do that",
                        tool_name='get_table_of_contents',
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2zZ4sz9g',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'What is Logfire?'},
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2zZ4s',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content="[Scrubbed due to 'Auth']",
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2zZ4s',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'What is Logfire?'},
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2z',
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'error_text': "Can't do that", 'is_error': True},
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2z',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                    ),
                    TextPart(
                        content='Here are the Table of Contents for both repositories:... Both products are designed to work together - Pydantic AI for building AI agents and Logfire for observing and monitoring them in production.'
                    ),
                    FilePart(content=BinaryContent(data=b'fake', media_type='application/pdf')),
                    ToolCallPart(
                        tool_name='get_table_of_contents', args={'repo': 'pydantic'}, tool_call_id='toolu_01XX3rjFfG77h'
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'What is Logfire?'},
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2zZ4s',
                    ),
                ],
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_adapter_dump_messages():
    """Test dumping Pydantic AI messages to Vercel AI format."""
    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='You are a helpful assistant.'),
                UserPromptPart(content='Hello, world!'),
            ]
        ),
        ModelResponse(
            parts=[
                TextPart(content='Hi there!'),
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)

    # we need to dump the BaseModels to dicts for `IsStr` to work properly in snapshot
    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'system',
                'metadata': None,
                'parts': [
                    {'type': 'text', 'text': 'You are a helpful assistant.', 'state': 'done', 'provider_metadata': None}
                ],
            },
            {
                'id': IsStr(),
                'role': 'user',
                'metadata': None,
                'parts': [{'type': 'text', 'text': 'Hello, world!', 'state': 'done', 'provider_metadata': None}],
            },
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [{'type': 'text', 'text': 'Hi there!', 'state': 'done', 'provider_metadata': None}],
            },
        ]
    )


async def test_adapter_dump_messages_with_tools():
    """Test dumping messages with tool calls and returns."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='Search for something')]),
        ModelResponse(
            parts=[
                TextPart(content='Let me search for that.'),
                ToolCallPart(
                    tool_name='web_search',
                    args={'query': 'test query'},
                    tool_call_id='tool_123',
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='web_search',
                    content={'results': ['result1', 'result2']},
                    tool_call_id='tool_123',
                )
            ]
        ),
        ModelResponse(parts=[TextPart(content='Here are the results.')]),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)
    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'user',
                'metadata': None,
                'parts': [{'type': 'text', 'text': 'Search for something', 'state': 'done', 'provider_metadata': None}],
            },
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {'type': 'text', 'text': 'Let me search for that.', 'state': 'done', 'provider_metadata': None},
                    {
                        'type': 'dynamic-tool',
                        'tool_name': 'web_search',
                        'tool_call_id': 'tool_123',
                        'state': 'output-available',
                        'input': '{"query":"test query"}',
                        'output': '{"results":["result1","result2"]}',
                        'call_provider_metadata': None,
                        'preliminary': None,
                    },
                ],
            },
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {'type': 'text', 'text': 'Here are the results.', 'state': 'done', 'provider_metadata': None}
                ],
            },
        ]
    )


async def test_adapter_dump_messages_with_builtin_tools():
    """Test dumping messages with builtin tool calls."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='Search for something')]),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'test'},
                    tool_call_id='tool_456',
                    provider_name='openai',
                ),
                BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={'status': 'completed'},
                    tool_call_id='tool_456',
                    provider_name='openai',
                ),
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)
    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'user',
                'metadata': None,
                'parts': [{'type': 'text', 'text': 'Search for something', 'state': 'done', 'provider_metadata': None}],
            },
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {
                        'type': 'tool-web_search',
                        'tool_call_id': 'tool_456',
                        'state': 'output-available',
                        'input': '{"query":"test"}',
                        'output': '{"status":"completed"}',
                        'provider_executed': True,
                        'call_provider_metadata': {'pydantic_ai': {'provider_name': 'openai'}},
                        'preliminary': None,
                    }
                ],
            },
        ]
    )


async def test_adapter_dump_messages_with_builtin_tool_without_return():
    """Test dumping messages with a builtin tool call that has no return in the same message."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='Search for something')]),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'orphan query'},
                    tool_call_id='orphan_tool_id',
                    provider_name='openai',
                ),
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)
    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'user',
                'metadata': None,
                'parts': [{'type': 'text', 'text': 'Search for something', 'state': 'done', 'provider_metadata': None}],
            },
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {
                        'type': 'tool-web_search',
                        'tool_call_id': 'orphan_tool_id',
                        'state': 'input-available',
                        'input': '{"query":"orphan query"}',
                        'provider_executed': True,
                        'call_provider_metadata': {'pydantic_ai': {'provider_name': 'openai'}},
                    }
                ],
            },
        ]
    )


async def test_adapter_dump_messages_with_thinking():
    """Test dumping messages with thinking parts."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='Tell me something')]),
        ModelResponse(
            parts=[
                ThinkingPart(content='Let me think about this...'),
                TextPart(content='Here is my answer.'),
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)
    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'user',
                'metadata': None,
                'parts': [{'type': 'text', 'text': 'Tell me something', 'state': 'done', 'provider_metadata': None}],
            },
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {
                        'type': 'reasoning',
                        'text': 'Let me think about this...',
                        'state': 'done',
                        'provider_metadata': None,
                    },
                    {'type': 'text', 'text': 'Here is my answer.', 'state': 'done', 'provider_metadata': None},
                ],
            },
        ]
    )


async def test_adapter_dump_messages_with_files():
    """Test dumping messages with file parts."""
    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Here is an image:',
                        BinaryImage(data=b'fake_image', media_type='image/png'),
                        ImageUrl(url='https://example.com/image.png', media_type='image/png'),
                    ]
                )
            ]
        ),
        ModelResponse(
            parts=[
                TextPart(content='Nice image!'),
                FilePart(content=BinaryContent(data=b'response_file', media_type='application/pdf')),
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)

    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'user',
                'metadata': None,
                'parts': [
                    {'type': 'text', 'text': 'Here is an image:', 'state': 'done', 'provider_metadata': None},
                    {
                        'type': 'file',
                        'media_type': 'image/png',
                        'filename': None,
                        'url': 'data:image/png;base64,ZmFrZV9pbWFnZQ==',
                        'provider_metadata': None,
                    },
                    {
                        'type': 'file',
                        'media_type': 'image/png',
                        'filename': None,
                        'url': 'https://example.com/image.png',
                        'provider_metadata': None,
                    },
                ],
            },
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {'type': 'text', 'text': 'Nice image!', 'state': 'done', 'provider_metadata': None},
                    {
                        'type': 'file',
                        'media_type': 'application/pdf',
                        'filename': None,
                        'url': 'data:application/pdf;base64,cmVzcG9uc2VfZmlsZQ==',
                        'provider_metadata': None,
                    },
                ],
            },
        ]
    )


async def test_adapter_dump_messages_with_retry():
    """Test dumping messages with retry prompts."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='Do something')]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='my_tool', args={'arg': 'value'}, tool_call_id='tool_789'),
            ]
        ),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    content='Tool failed with error',
                    tool_name='my_tool',
                    tool_call_id='tool_789',
                )
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)

    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'user',
                'metadata': None,
                'parts': [{'type': 'text', 'text': 'Do something', 'state': 'done', 'provider_metadata': None}],
            },
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {
                        'type': 'dynamic-tool',
                        'tool_name': 'my_tool',
                        'tool_call_id': 'tool_789',
                        'state': 'output-error',
                        'input': '{"arg":"value"}',
                        'error_text': """\
Tool failed with error

Fix the errors and try again.\
""",
                        'call_provider_metadata': None,
                    }
                ],
            },
        ]
    )


async def test_adapter_dump_messages_with_retry_no_tool_name():
    """Test dumping messages with retry prompts without tool_name (e.g., output validation errors)."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='Give me a number')]),
        ModelResponse(parts=[TextPart(content='Not a valid number')]),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    content='Output validation failed: expected integer',
                    # No tool_name - this is an output validation error, not a tool error
                )
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)

    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'user',
                'metadata': None,
                'parts': [{'type': 'text', 'text': 'Give me a number', 'state': 'done', 'provider_metadata': None}],
            },
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [{'type': 'text', 'text': 'Not a valid number', 'state': 'done', 'provider_metadata': None}],
            },
            {
                'id': IsStr(),
                'role': 'user',
                'metadata': None,
                'parts': [
                    {
                        'type': 'text',
                        'text': """\
Validation feedback:
Output validation failed: expected integer

Fix the errors and try again.\
""",
                        'state': 'done',
                        'provider_metadata': None,
                    }
                ],
            },
        ]
    )


async def test_adapter_dump_messages_consecutive_text():
    """Test that consecutive text parts are concatenated correctly."""
    messages = [
        ModelResponse(
            parts=[
                TextPart(content='First '),
                TextPart(content='second'),
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)
    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [{'type': 'text', 'text': 'First second', 'state': 'done', 'provider_metadata': None}],
            }
        ]
    )


async def test_adapter_dump_messages_text_with_interruption():
    """Test text concatenation with interruption."""
    messages = [
        ModelResponse(
            parts=[
                TextPart(content='Before tool'),
                BuiltinToolCallPart(
                    tool_name='test',
                    args={},
                    tool_call_id='t1',
                    provider_name='test',
                ),
                BuiltinToolReturnPart(
                    tool_name='test',
                    content='result',
                    tool_call_id='t1',
                    provider_name='test',
                ),
                TextPart(content='After tool'),
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)
    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {'type': 'text', 'text': 'Before tool', 'state': 'done', 'provider_metadata': None},
                    {
                        'type': 'tool-test',
                        'tool_call_id': 't1',
                        'state': 'output-available',
                        'input': '{}',
                        'output': 'result',
                        'provider_executed': True,
                        'call_provider_metadata': {'pydantic_ai': {'provider_name': 'test'}},
                        'preliminary': None,
                    },
                    {
                        'type': 'text',
                        'text': 'After tool',
                        'state': 'done',
                        'provider_metadata': None,
                    },
                ],
            }
        ]
    )


async def test_adapter_dump_load_roundtrip():
    """Test that dump_messages and load_messages are approximately inverse operations."""
    original_messages = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System message'),
                UserPromptPart(content='User message'),
            ]
        ),
        ModelResponse(
            parts=[
                TextPart(content='Response text'),
                ToolCallPart(tool_name='tool1', args={'key': 'value'}, tool_call_id='tc1'),
            ]
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name='tool1', content='tool result', tool_call_id='tc1')]),
        ModelResponse(
            parts=[
                TextPart(content='Final response'),
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(original_messages)

    def sync_timestamps(original: list[ModelRequest | ModelResponse], new: list[ModelRequest | ModelResponse]) -> None:
        for orig_msg, new_msg in zip(original, new):
            for orig_part, new_part in zip(orig_msg.parts, new_msg.parts):
                if hasattr(orig_part, 'timestamp') and hasattr(new_part, 'timestamp'):
                    new_part.timestamp = orig_part.timestamp  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
            if hasattr(orig_msg, 'timestamp') and hasattr(new_msg, 'timestamp'):  # pragma: no branch
                new_msg.timestamp = orig_msg.timestamp  # pyright: ignore[reportAttributeAccessIssue]

    # Load back to Pydantic AI format
    reloaded_messages = VercelAIAdapter.load_messages(ui_messages)
    sync_timestamps(original_messages, reloaded_messages)

    assert reloaded_messages == original_messages


async def test_adapter_dump_load_roundtrip_without_timestamps():
    """Test that dump_messages and load_messages work when messages don't have timestamps."""
    original_messages = [
        ModelRequest(
            parts=[
                UserPromptPart(content='User message'),
            ]
        ),
        ModelResponse(
            parts=[
                TextPart(content='Response text'),
            ]
        ),
    ]

    for msg in original_messages:
        delattr(msg, 'timestamp')

    ui_messages = VercelAIAdapter.dump_messages(original_messages)
    reloaded_messages = VercelAIAdapter.load_messages(ui_messages)

    def sync_timestamps(original: list[ModelRequest | ModelResponse], new: list[ModelRequest | ModelResponse]) -> None:
        for orig_msg, new_msg in zip(original, new):
            for orig_part, new_part in zip(orig_msg.parts, new_msg.parts):
                if hasattr(orig_part, 'timestamp') and hasattr(new_part, 'timestamp'):
                    new_part.timestamp = orig_part.timestamp  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
            if hasattr(orig_msg, 'timestamp') and hasattr(new_msg, 'timestamp'):
                new_msg.timestamp = orig_msg.timestamp  # pyright: ignore[reportAttributeAccessIssue]

    sync_timestamps(original_messages, reloaded_messages)

    for msg in reloaded_messages:
        if hasattr(msg, 'timestamp'):  # pragma: no branch
            delattr(msg, 'timestamp')

    assert len(reloaded_messages) == len(original_messages)


async def test_adapter_dump_messages_text_before_thinking():
    """Test dumping messages where text precedes a thinking part."""
    messages = [
        ModelResponse(
            parts=[
                TextPart(content='Let me check.'),
                ThinkingPart(content='Okay, I am checking now.'),
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)
    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {'type': 'text', 'text': 'Let me check.', 'state': 'done', 'provider_metadata': None},
                    {
                        'type': 'reasoning',
                        'text': 'Okay, I am checking now.',
                        'state': 'done',
                        'provider_metadata': None,
                    },
                ],
            }
        ]
    )


async def test_adapter_dump_messages_tool_call_without_return():
    """Test dumping messages with a tool call that has no corresponding result."""
    messages = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='get_weather',
                    args={'city': 'New York'},
                    tool_call_id='tool_abc',
                ),
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)
    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {
                        'type': 'dynamic-tool',
                        'tool_name': 'get_weather',
                        'tool_call_id': 'tool_abc',
                        'state': 'input-available',
                        'input': '{"city":"New York"}',
                        'call_provider_metadata': None,
                    }
                ],
            }
        ]
    )


async def test_adapter_dump_messages_assistant_starts_with_tool():
    """Test an assistant message that starts with a tool call instead of text."""
    messages = [
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='t', args={}, tool_call_id='tc1'),
                TextPart(content='Some text'),
            ]
        )
    ]
    ui_messages = VercelAIAdapter.dump_messages(messages)

    ui_message_dicts = [msg.model_dump() for msg in ui_messages]
    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {
                        'type': 'dynamic-tool',
                        'tool_name': 't',
                        'tool_call_id': 'tc1',
                        'state': 'input-available',
                        'input': '{}',
                        'call_provider_metadata': None,
                    },
                    {
                        'type': 'text',
                        'text': 'Some text',
                        'state': 'done',
                        'provider_metadata': None,
                    },
                ],
            }
        ]
    )


async def test_convert_user_prompt_part_without_urls():
    """Test converting a user prompt with only text and binary content."""
    from pydantic_ai.ui.vercel_ai._adapter import _convert_user_prompt_part  # pyright: ignore[reportPrivateUsage]

    part = UserPromptPart(content=['text part', BinaryContent(data=b'data', media_type='application/pdf')])
    ui_parts = _convert_user_prompt_part(part)
    assert ui_parts == snapshot(
        [
            TextUIPart(text='text part', state='done'),
            FileUIPart(media_type='application/pdf', url='data:application/pdf;base64,ZGF0YQ=='),
        ]
    )


async def test_adapter_dump_messages_file_without_text():
    """Test a file part appearing without any preceding text."""
    messages = [
        ModelResponse(
            parts=[
                FilePart(content=BinaryContent(data=b'file_data', media_type='image/png')),
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)
    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {
                        'type': 'file',
                        'media_type': 'image/png',
                        'filename': None,
                        'url': 'data:image/png;base64,ZmlsZV9kYXRh',
                        'provider_metadata': None,
                    }
                ],
            }
        ]
    )


async def test_convert_user_prompt_part_only_urls():
    """Test converting a user prompt with only URL content (no binary)."""
    from pydantic_ai.ui.vercel_ai._adapter import _convert_user_prompt_part  # pyright: ignore[reportPrivateUsage]

    part = UserPromptPart(
        content=[
            ImageUrl(url='https://example.com/img.png', media_type='image/png'),
            VideoUrl(url='https://example.com/vid.mp4', media_type='video/mp4'),
        ]
    )
    ui_parts = _convert_user_prompt_part(part)
    assert ui_parts == snapshot(
        [
            FileUIPart(media_type='image/png', url='https://example.com/img.png'),
            FileUIPart(media_type='video/mp4', url='https://example.com/vid.mp4'),
        ]
    )


async def test_adapter_dump_messages_thinking_with_metadata():
    """Test dumping and loading messages with ThinkingPart metadata preservation."""
    original_messages = [
        ModelResponse(
            parts=[
                ThinkingPart(
                    content='Let me think about this...',
                    id='thinking_123',
                    signature='sig_abc',
                    provider_name='anthropic',
                    provider_details={'model': 'claude-3'},
                ),
                TextPart(content='Here is my answer.'),
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(original_messages)
    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {
                        'type': 'reasoning',
                        'text': 'Let me think about this...',
                        'state': 'done',
                        'provider_metadata': {
                            'pydantic_ai': {
                                'id': 'thinking_123',
                                'signature': 'sig_abc',
                                'provider_name': 'anthropic',
                                'provider_details': {'model': 'claude-3'},
                            }
                        },
                    },
                    {'type': 'text', 'text': 'Here is my answer.', 'state': 'done', 'provider_metadata': None},
                ],
            }
        ]
    )

    # Test roundtrip - verify metadata is preserved when loading back
    reloaded_messages = VercelAIAdapter.load_messages(ui_messages)

    # Sync timestamps for comparison (ModelResponse always has timestamp)
    for orig_msg, new_msg in zip(original_messages, reloaded_messages):
        new_msg.timestamp = orig_msg.timestamp

    assert reloaded_messages == original_messages


async def test_adapter_load_messages_json_list_args():
    """Test that JSON list args are kept as strings (not parsed)."""
    ui_messages = [
        UIMessage(
            id='msg1',
            role='assistant',
            parts=[
                DynamicToolOutputAvailablePart(
                    tool_name='my_tool',
                    tool_call_id='tc1',
                    input='[1, 2, 3]',  # JSON list - should stay as string
                    output='result',
                    state='output-available',
                )
            ],
        )
    ]

    messages = VercelAIAdapter.load_messages(ui_messages)

    assert len(messages) == 2  # ToolCall in response + ToolReturn in request
    response = messages[0]
    assert isinstance(response, ModelResponse)
    assert len(response.parts) == 1
    tool_call = response.parts[0]
    assert isinstance(tool_call, ToolCallPart)
    # Args should remain as string since it parses to a list, not a dict
    assert tool_call.args == '[1, 2, 3]'


async def test_adapter_dump_messages_with_cache_point():
    """Test that CachePoint in user content is skipped during conversion."""
    from pydantic_ai.messages import CachePoint

    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Hello',
                        CachePoint(),  # Should be skipped
                        'World',
                    ]
                )
            ]
        ),
    ]

    ui_messages = VercelAIAdapter.dump_messages(messages)
    ui_message_dicts = [msg.model_dump() for msg in ui_messages]

    # CachePoint should be omitted, only text parts remain
    assert ui_message_dicts == snapshot(
        [
            {
                'id': IsStr(),
                'role': 'user',
                'metadata': None,
                'parts': [
                    {'type': 'text', 'text': 'Hello', 'state': 'done', 'provider_metadata': None},
                    {'type': 'text', 'text': 'World', 'state': 'done', 'provider_metadata': None},
                ],
            }
        ]
    )
