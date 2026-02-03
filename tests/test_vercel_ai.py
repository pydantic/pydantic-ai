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
from pydantic_ai.ui.vercel_ai._utils import dump_provider_metadata, load_provider_metadata
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
from pydantic_ai.ui.vercel_ai.response_types import BaseChunk, DataChunk, ToolInputStartChunk

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
                        text='''I'd be happy to help you use a tool! However, I need more information about what you'd like to do. I have access to tools for searching and retrieving documentation for two products:

    1. **Pydantic AI** (pydantic-ai) - an open source agent framework library
    2. **Pydantic Logfire** (logfire) - an observability platform

    I can help you with:
    - Searching the documentation for specific topics or questions
    - Getting the table of contents to see what documentation is available
    - Retrieving specific documentation files

    What would you like to learn about or search for? Please let me know:
    - Which product you're interested in (Pydantic AI or Logfire)
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

    adapter = VercelAIAdapter(agent, run_input=data, sdk_version=6)
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
            {
                'type': 'reasoning-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'tool-input-start',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"query":"Pydantic Logfire FastAPI instrumentation request response body","type":"search","queries":["Pydantic Logfire FastAPI instrumentation request response body","logfire fastapi request body response body","OpenTelemetry FastAPI instrumentation include HTTP request and response body","pydantic logfire fastapi include request response"]}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'input': {
                    'query': 'Pydantic Logfire FastAPI instrumentation request response body',
                    'type': 'search',
                    'queries': [
                        'Pydantic Logfire FastAPI instrumentation request response body',
                        'logfire fastapi request body response body',
                        'OpenTelemetry FastAPI instrumentation include HTTP request and response body',
                        'pydantic logfire fastapi include request response',
                    ],
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
            {
                'type': 'reasoning-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'tool-input-start',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"type":"open_page","url":"https://logfire.pydantic.dev/docs/reference/api/logfire/"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'input': {'type': 'open_page', 'url': 'https://logfire.pydantic.dev/docs/reference/api/logfire/'},
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': IsStr(),
                'output': {'status': 'completed'},
                'providerExecuted': True,
            },
            {
                'type': 'reasoning-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'tool-input-start',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"type":"find_in_page","pattern":"instrument_fastapi","url":"https://logfire.pydantic.dev/docs/reference/api/logfire/"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'input': {
                    'type': 'find_in_page',
                    'pattern': 'instrument_fastapi',
                    'url': 'https://logfire.pydantic.dev/docs/reference/api/logfire/',
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
            {
                'type': 'reasoning-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'tool-input-start',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"type":"open_page","url":"https://logfire.pydantic.dev/docs/reference/api/logfire/"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'input': {'type': 'open_page', 'url': 'https://logfire.pydantic.dev/docs/reference/api/logfire/'},
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': IsStr(),
                'output': {'status': 'completed'},
                'providerExecuted': True,
            },
            {
                'type': 'reasoning-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'tool-input-start',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"type":"find_in_page","pattern":"Instrument a FastAPI app","url":"https://logfire.pydantic.dev/docs/reference/api/logfire/"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'input': {
                    'type': 'find_in_page',
                    'pattern': 'Instrument a FastAPI app',
                    'url': 'https://logfire.pydantic.dev/docs/reference/api/logfire/',
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
            {
                'type': 'reasoning-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'tool-input-start',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"type":"open_page","url":"https://logfire.pydantic.dev/docs/integrations/web-frameworks/"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'input': {'type': 'open_page', 'url': 'https://logfire.pydantic.dev/docs/integrations/web-frameworks/'},
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-output-available',
                'toolCallId': IsStr(),
                'output': {'status': 'completed'},
                'providerExecuted': True,
            },
            {
                'type': 'reasoning-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': IsStr(),
                        'provider_name': 'openai',
                        'id': IsStr(),
                    }
                },
            },
            {
                'type': 'tool-input-start',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"type":"open_page","url":"https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'input': {
                    'type': 'open_page',
                    'url': 'https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/',
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
            {
                'type': 'reasoning-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': IsStr(),
                        'signature': 'gAAAAABpgWJi1r4wu-LMSDcYO-5OHX2tu81pjFz5wNqaTcU9bOq54w9uXOJeFqd2C0HM6rxK1oyNvqvrGWhKHI53MI6Isv-FxMgTPpY5V5y8CX4uNTm8SyB0rAOHJCoyjoA5cDfLNkCcWyMi2tZeU5HEChTw0rMm6DoiOfx4Lqvak5EhTp4IFC9KDs2iu85x1zvPk8NMTLdL4657QVyzVvXppd1DTwhLM7P0DzFjNW2LPvd0fpyOPurY3ZpBXurAT_Uvxll12B1zsghQmbE3Lgr94pUFgq_3xqTUVH2VtxFu28aOIxJzydIbjARfdkJQke21cDBk2gjcg8yhe8UipnpcrT_aGgmRqRxXRqaSLxJKwAsA9-coqP3ICrbp4VQTqaIfnNnYdQIJHAV4PWZdou0LSTVsH-aHwFVrEfH3u_nLkFS6HBmQDRMhrk2bAWOKSkuO8HKGCReU9oO28J2TVtuXMxeBtGYfuvEpQjliJAC66jUroHTJ2Lm6b5cs41dFYn6WTd6m1bIgML5JHjORoZxGP84hM_CmPgeBwwsVAHr6WnlEjbjXF-Pw0LNwGTXSfYZawCp0hiSKfy2-8r_YbVyPm963l4zTQgLuO0-UoKDePxktoYQ1ZkDe73hMfnBNOfhUrCk5csf-6DoEPo6l4-nSokd4plhIWxKWkueZ1WBtazHDLfqiE_q_50ist2MV51VD75TgXr96mY6KtYHxNIoVP4A38Xw-Ldwdy7xZXBFHoSuR-9mKhL7mktKopcbQ97GWC2uz9gKZXWFp2EHkVzDGDZV1k9SuDtpQdwiIGHom_IVyeiMbBINYeJTCXktN4SC3TXQg2dCTPJWCP7dnczVXDfJJr1FCGLlT04jr7Y4culFYW75x4_qeEQTFhWuKm-kCvnsG4N5EcoCGM9thiiGKBwTjZZ_aob4RQsdH9FA-S_ZkeNBNQgEW5iUyK8LUNDvujExYwLWdTGzY3-DB7HprKEgv8augkagOkHxjMYZcKFc88xhxfxtbifQHG5sP_hPwIF55-QtWVBYp6Zq0BoRedjvKRuB7K7EjynnCwgeTBn0wfEJ6vWXkYrncsnpnBWV33fgbEx_Pj3rkXxsO-L5sjm4eWHOXx1QTiEXejJ0EvphTurCBZ-SYVqgQno7sDTbnIWb1rCoL2aNX6hX9DmjRNm3iAamuTW5GtAJ366gHzWjO3eLv3Oiqr76fovZm-p5cFL5Tf9_6WeQzhyu37lEN_N27gdJeDxmZioIM6lnP017C_V5bqR28vjxysn6qvngYzf90zVmI13MYodV-zTEiWG_V338w8FnPZyJjkkXwpBg4X4Yr0SuH2REOcYyG5N8j4kzvPLW2-koNL2Kniu0kz9f36cmQpFXDLNs0g2c5cecluUSxHOLSV9C8MoW_GTh3JCJnMF1RU3xd4SaWzy0GZtiQb8HmDs-_2TMaJ0bSc1HN85g5__DTYJM2dBoRBZuMbszwcesz-kEmxGrZ2TcQq3yJCOduW-5YUupek97mwqDDmbd1kbQfTVfLCVLLitU0DUcAFMmEYW3BOlebC4T9yRU03yx9CWJE24kzsVR_4H55b51N96anWpJySX0Q62cv24kaVXl_1m2QNr0kmYLJ2EK-ZZYQ5djLPU9B8otyLVcM1sicdgdM0IMDJ1rGwlZHbhXZBEgveplpUebtTaRPT6ThHKiYaCMJd48Ufud6vyGmr4-_tglsN7EiyFLUiobcQo4IksaYEr4gP1-Oq2q6LFp40hWtdJf8oNyy_Lih31ALBSACsrv2bn3j9o48VVtLnY7FD8s_i6k3YJU8Is5YisXXMtTrfKXxNUE6JR3H-Mrn7RzBc1OVaqQfgyQkHOmCutww1ksMxFZwAfVGBiqywEplVEo8Lmn8T0xdQ4cslvJOfgeD0FEiWxdW75UGPZ4CZfKQvILoQT7Nie1XShB0FHsL1-Zh1M24x3kSk9S9rRzHQQbBlJ_GtqRE3obeRKjlZj8E5WSxl9kAqKFEtTzZsomQ6x3MqOUsz1hYGf2p-QiJTIqMfbpUV1R3i-JJe3b4xn9Ngq8sh3sPI6rt0bFVL2xwp3KDkQbqLVTpSRwuTg_gXbkIfGKhzz14yd0WZb6W4eTmxR1i8YVc6gEVi-ex5DJTsXw-LYLekQFi8MHoCfaMIWkaQ4RjrXXAsIAqhyNpzfa18H6MHGpNFb98fecDKt9luhoXufls2Ef_7qovAuW53ztaJGL61sbkngTe-TCPHqWkrSMxZ7mraSRC76z7QNRJqRk1OG_plITjRZJSEStfzVjeVvX5rT0WcMEaENQqWcnh6bperF2qXRs6yY9q36lqcrnJUnKWsVGL4OxmtRLpBwH42uI8A0aY1gP07ZH7wvIF8Y1jhXiwJd4a9NhAGHWka9zZTuFntRKzRRzvQFLDLziBjU6KpCPyNi07KI1qISrU7n698AXNItqBV9G4J1saQv613mXnps62rZozYssA2j_3N_glejFb7IV5wWDVSdinC7kAPmFVBpmiPrK9_Fyd-xkBYLujpgZKBI8Ng32dlkz419LFAFSy_rJUx-g2aVROVuPJfKF1PwOPdbXZrizPvaU4_lnPa5wDl7kWIkp0o6LVCf1s3HHRGmxvj5UkGs2aNEIvJoFzsmU20mdi2Y8emPR8QuqzX2Gr3Kf85TPqSFuYj5X2q3p5I4M8KIxdn_1NClkWHblsZmnNWZUL511Pif8dx7NaW1t0mBhaNBa1FVfaeDgoa9Lo6ZWPP3JM3-CkvfsjBH26wn2uK3p2RLNwqOMk1hER-JP9mFMlExPNsetVK0ih7MqI8dcRzmU4Ak6WUWdlVgSPlcIarsufvbg6RF-a7m_zUmm6WdMWLtybpMYrYOm1Al9RC-cAVJVCB1d_hmkDIMQ3px5QSia8gKOEkouT-B43Bm8irRG7vm3o-ksLKxehLLa89nZPqOOv2sbFQ0Rt7_C-DWxQqsYBX9tFpIzIo7cI7Y58XfACvhzBzslS0Xdl9iRIHIe9NFyKdw66PynGs0c7KOmXw89v3h0srlXK4UIa9gpYsAGPTcwLPBug_T0eh8wLQpA0OEpRuDecWOG5FEoRHHK2La5Ou97URAqQp5HqVfrPVvZJDydYRqBWGUS6Bqr6Xt3eDB6-BekYy352WaxrN1TrmB3gRce5c1lywwsw831V-nlmCgTiFwcWbv4cfaMe8klX8gG7h0JlsYztMz0J8FEDSzGV4tUjd7W6zd-SGfZCuMiDYCCJ7neQXNux8J8Gafo84TmRlBCjgSoG4rmcOxNLTz0R6fmxy_FD8TepP1Xr8gObKkZS26209VZFKDhSSEZHPXhLjNh4Xh9N_oXX2mxx926TAd2dp_WVnQor0DHo3SnkZnY0Ozz9g1P6xeYTNVE30ZzH-c2vwUIU1wRX-7Dy_KZueCMe4rMS_mDazIQxWxm161WakvlHxI-9PEafLQUHXE6JIX28efJ-64Bt9KBBa2n0CNav3YcFZ-Kwh7rV4h8YpfrXtsPQRM6y6UtY-KqoWYakyeScHgY7ydgzNRQiE2ZtOeXvLGHgfgejwZoPxgv_l8oNPFb1NPEBuEZ1MBUxiA620sh8G5MZjZgogiSBp9NjJpG6o-apxcX330sXGZuf25f9z2E8qcU3dE8dHBsSJjtO1zKIs2OP168i9c6i1_Px0ggXpwMHqLVfeHbz9AIyhkK7p9wQcojwdKRk0tSGx3d3q4mg3t-S5d1F1MqjfYQ_jSZtu0iCmz4zSH75uZ5GQJYWsWMAvAYsubKn1oouuCxhKJaJ7f-PwUj8_0d2sADUQEteNegTegKp6uEJy3KESoowBzdrykO0wV4vBXObytkzyVm9_FaDO7klFG2E7-r03GfgJFeauiKSp-M9uBB3o3hY9yrvKAsBEkcpP4CkS3eevV2krzWWjmceeqVmJ9p4A0qGLWdNTUEhPNzHBqPsNwTvvy4HFK2mlCEr808J1BnixM2v1sBTQcYE9zTwhmzDpqaJev9qU_WjjUs69LjMsoRXZpAKPMtVFliHIFgtslivwXabzOz_EvtmQdKx86Wi66RHhxK6NS-c2SYgw952TGmfrLjWQCR9s4_WTiKYugHeYVgZksvRBQxIHN7A3NxW8WYxSUAQ3kLdrClaMHPwYQdy3RPgz9Ybj9ikGGe6_q_uk7llMS113dntcJDLZIbhAsZo8epwrhnBPAns57TXy48VwKBDWR1YnSzqi6zerPhzCgRIiVUFoPkvBYb2ct82p0aZXuMjB-ekok6VQP2t_8-dDD2R0HrdhwStB_DB_h-_2RxDu-z30_ObwyHB7kNDPaOKzKGbZdxr9ZmNA2OFlIhZ3jYBe8AMITv-VDxS8d9kvJurpwgrXz7uC0SgFyM_wqBbtvQtWOsDyxBkXDyy1hiI5J0podI1H4cqXxxf31foaHzqR7bJF3wBXwg3_JVNGHnOb1Hrj4GF6UaS7NjjHZES94zvxeIQ_J0B06-wZ9F_92eXWoKe_X74-HyK7XpEwUwyuDzlQR0dLN4BtUyWzFv51dY9EN4=',
                        'provider_name': 'openai',
                    }
                },
            },
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': IsStr(),
                        'signature': 'gAAAAABpgWJi1r4wu-LMSDcYO-5OHX2tu81pjFz5wNqaTcU9bOq54w9uXOJeFqd2C0HM6rxK1oyNvqvrGWhKHI53MI6Isv-FxMgTPpY5V5y8CX4uNTm8SyB0rAOHJCoyjoA5cDfLNkCcWyMi2tZeU5HEChTw0rMm6DoiOfx4Lqvak5EhTp4IFC9KDs2iu85x1zvPk8NMTLdL4657QVyzVvXppd1DTwhLM7P0DzFjNW2LPvd0fpyOPurY3ZpBXurAT_Uvxll12B1zsghQmbE3Lgr94pUFgq_3xqTUVH2VtxFu28aOIxJzydIbjARfdkJQke21cDBk2gjcg8yhe8UipnpcrT_aGgmRqRxXRqaSLxJKwAsA9-coqP3ICrbp4VQTqaIfnNnYdQIJHAV4PWZdou0LSTVsH-aHwFVrEfH3u_nLkFS6HBmQDRMhrk2bAWOKSkuO8HKGCReU9oO28J2TVtuXMxeBtGYfuvEpQjliJAC66jUroHTJ2Lm6b5cs41dFYn6WTd6m1bIgML5JHjORoZxGP84hM_CmPgeBwwsVAHr6WnlEjbjXF-Pw0LNwGTXSfYZawCp0hiSKfy2-8r_YbVyPm963l4zTQgLuO0-UoKDePxktoYQ1ZkDe73hMfnBNOfhUrCk5csf-6DoEPo6l4-nSokd4plhIWxKWkueZ1WBtazHDLfqiE_q_50ist2MV51VD75TgXr96mY6KtYHxNIoVP4A38Xw-Ldwdy7xZXBFHoSuR-9mKhL7mktKopcbQ97GWC2uz9gKZXWFp2EHkVzDGDZV1k9SuDtpQdwiIGHom_IVyeiMbBINYeJTCXktN4SC3TXQg2dCTPJWCP7dnczVXDfJJr1FCGLlT04jr7Y4culFYW75x4_qeEQTFhWuKm-kCvnsG4N5EcoCGM9thiiGKBwTjZZ_aob4RQsdH9FA-S_ZkeNBNQgEW5iUyK8LUNDvujExYwLWdTGzY3-DB7HprKEgv8augkagOkHxjMYZcKFc88xhxfxtbifQHG5sP_hPwIF55-QtWVBYp6Zq0BoRedjvKRuB7K7EjynnCwgeTBn0wfEJ6vWXkYrncsnpnBWV33fgbEx_Pj3rkXxsO-L5sjm4eWHOXx1QTiEXejJ0EvphTurCBZ-SYVqgQno7sDTbnIWb1rCoL2aNX6hX9DmjRNm3iAamuTW5GtAJ366gHzWjO3eLv3Oiqr76fovZm-p5cFL5Tf9_6WeQzhyu37lEN_N27gdJeDxmZioIM6lnP017C_V5bqR28vjxysn6qvngYzf90zVmI13MYodV-zTEiWG_V338w8FnPZyJjkkXwpBg4X4Yr0SuH2REOcYyG5N8j4kzvPLW2-koNL2Kniu0kz9f36cmQpFXDLNs0g2c5cecluUSxHOLSV9C8MoW_GTh3JCJnMF1RU3xd4SaWzy0GZtiQb8HmDs-_2TMaJ0bSc1HN85g5__DTYJM2dBoRBZuMbszwcesz-kEmxGrZ2TcQq3yJCOduW-5YUupek97mwqDDmbd1kbQfTVfLCVLLitU0DUcAFMmEYW3BOlebC4T9yRU03yx9CWJE24kzsVR_4H55b51N96anWpJySX0Q62cv24kaVXl_1m2QNr0kmYLJ2EK-ZZYQ5djLPU9B8otyLVcM1sicdgdM0IMDJ1rGwlZHbhXZBEgveplpUebtTaRPT6ThHKiYaCMJd48Ufud6vyGmr4-_tglsN7EiyFLUiobcQo4IksaYEr4gP1-Oq2q6LFp40hWtdJf8oNyy_Lih31ALBSACsrv2bn3j9o48VVtLnY7FD8s_i6k3YJU8Is5YisXXMtTrfKXxNUE6JR3H-Mrn7RzBc1OVaqQfgyQkHOmCutww1ksMxFZwAfVGBiqywEplVEo8Lmn8T0xdQ4cslvJOfgeD0FEiWxdW75UGPZ4CZfKQvILoQT7Nie1XShB0FHsL1-Zh1M24x3kSk9S9rRzHQQbBlJ_GtqRE3obeRKjlZj8E5WSxl9kAqKFEtTzZsomQ6x3MqOUsz1hYGf2p-QiJTIqMfbpUV1R3i-JJe3b4xn9Ngq8sh3sPI6rt0bFVL2xwp3KDkQbqLVTpSRwuTg_gXbkIfGKhzz14yd0WZb6W4eTmxR1i8YVc6gEVi-ex5DJTsXw-LYLekQFi8MHoCfaMIWkaQ4RjrXXAsIAqhyNpzfa18H6MHGpNFb98fecDKt9luhoXufls2Ef_7qovAuW53ztaJGL61sbkngTe-TCPHqWkrSMxZ7mraSRC76z7QNRJqRk1OG_plITjRZJSEStfzVjeVvX5rT0WcMEaENQqWcnh6bperF2qXRs6yY9q36lqcrnJUnKWsVGL4OxmtRLpBwH42uI8A0aY1gP07ZH7wvIF8Y1jhXiwJd4a9NhAGHWka9zZTuFntRKzRRzvQFLDLziBjU6KpCPyNi07KI1qISrU7n698AXNItqBV9G4J1saQv613mXnps62rZozYssA2j_3N_glejFb7IV5wWDVSdinC7kAPmFVBpmiPrK9_Fyd-xkBYLujpgZKBI8Ng32dlkz419LFAFSy_rJUx-g2aVROVuPJfKF1PwOPdbXZrizPvaU4_lnPa5wDl7kWIkp0o6LVCf1s3HHRGmxvj5UkGs2aNEIvJoFzsmU20mdi2Y8emPR8QuqzX2Gr3Kf85TPqSFuYj5X2q3p5I4M8KIxdn_1NClkWHblsZmnNWZUL511Pif8dx7NaW1t0mBhaNBa1FVfaeDgoa9Lo6ZWPP3JM3-CkvfsjBH26wn2uK3p2RLNwqOMk1hER-JP9mFMlExPNsetVK0ih7MqI8dcRzmU4Ak6WUWdlVgSPlcIarsufvbg6RF-a7m_zUmm6WdMWLtybpMYrYOm1Al9RC-cAVJVCB1d_hmkDIMQ3px5QSia8gKOEkouT-B43Bm8irRG7vm3o-ksLKxehLLa89nZPqOOv2sbFQ0Rt7_C-DWxQqsYBX9tFpIzIo7cI7Y58XfACvhzBzslS0Xdl9iRIHIe9NFyKdw66PynGs0c7KOmXw89v3h0srlXK4UIa9gpYsAGPTcwLPBug_T0eh8wLQpA0OEpRuDecWOG5FEoRHHK2La5Ou97URAqQp5HqVfrPVvZJDydYRqBWGUS6Bqr6Xt3eDB6-BekYy352WaxrN1TrmB3gRce5c1lywwsw831V-nlmCgTiFwcWbv4cfaMe8klX8gG7h0JlsYztMz0J8FEDSzGV4tUjd7W6zd-SGfZCuMiDYCCJ7neQXNux8J8Gafo84TmRlBCjgSoG4rmcOxNLTz0R6fmxy_FD8TepP1Xr8gObKkZS26209VZFKDhSSEZHPXhLjNh4Xh9N_oXX2mxx926TAd2dp_WVnQor0DHo3SnkZnY0Ozz9g1P6xeYTNVE30ZzH-c2vwUIU1wRX-7Dy_KZueCMe4rMS_mDazIQxWxm161WakvlHxI-9PEafLQUHXE6JIX28efJ-64Bt9KBBa2n0CNav3YcFZ-Kwh7rV4h8YpfrXtsPQRM6y6UtY-KqoWYakyeScHgY7ydgzNRQiE2ZtOeXvLGHgfgejwZoPxgv_l8oNPFb1NPEBuEZ1MBUxiA620sh8G5MZjZgogiSBp9NjJpG6o-apxcX330sXGZuf25f9z2E8qcU3dE8dHBsSJjtO1zKIs2OP168i9c6i1_Px0ggXpwMHqLVfeHbz9AIyhkK7p9wQcojwdKRk0tSGx3d3q4mg3t-S5d1F1MqjfYQ_jSZtu0iCmz4zSH75uZ5GQJYWsWMAvAYsubKn1oouuCxhKJaJ7f-PwUj8_0d2sADUQEteNegTegKp6uEJy3KESoowBzdrykO0wV4vBXObytkzyVm9_FaDO7klFG2E7-r03GfgJFeauiKSp-M9uBB3o3hY9yrvKAsBEkcpP4CkS3eevV2krzWWjmceeqVmJ9p4A0qGLWdNTUEhPNzHBqPsNwTvvy4HFK2mlCEr808J1BnixM2v1sBTQcYE9zTwhmzDpqaJev9qU_WjjUs69LjMsoRXZpAKPMtVFliHIFgtslivwXabzOz_EvtmQdKx86Wi66RHhxK6NS-c2SYgw952TGmfrLjWQCR9s4_WTiKYugHeYVgZksvRBQxIHN7A3NxW8WYxSUAQ3kLdrClaMHPwYQdy3RPgz9Ybj9ikGGe6_q_uk7llMS113dntcJDLZIbhAsZo8epwrhnBPAns57TXy48VwKBDWR1YnSzqi6zerPhzCgRIiVUFoPkvBYb2ct82p0aZXuMjB-ekok6VQP2t_8-dDD2R0HrdhwStB_DB_h-_2RxDu-z30_ObwyHB7kNDPaOKzKGbZdxr9ZmNA2OFlIhZ3jYBe8AMITv-VDxS8d9kvJurpwgrXz7uC0SgFyM_wqBbtvQtWOsDyxBkXDyy1hiI5J0podI1H4cqXxxf31foaHzqR7bJF3wBXwg3_JVNGHnOb1Hrj4GF6UaS7NjjHZES94zvxeIQ_J0B06-wZ9F_92eXWoKe_X74-HyK7XpEwUwyuDzlQR0dLN4BtUyWzFv51dY9EN4=',
                        'provider_name': 'openai',
                    }
                },
            },
            {
                'type': 'tool-input-start',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'openai'}},
            },
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"query":"OpenTelemetry ASGI instrumentation capture request body","type":"search","queries":["OpenTelemetry ASGI instrumentation capture request body","opentelemetry-python asgi record request body response body","opentelemetry fastapi capture request body response body server"]}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': IsStr(),
                'toolName': 'web_search',
                'input': {
                    'query': 'OpenTelemetry ASGI instrumentation capture request body',
                    'type': 'search',
                    'queries': [
                        'OpenTelemetry ASGI instrumentation capture request body',
                        'opentelemetry-python asgi record request body response body',
                        'opentelemetry fastapi capture request body response body server',
                    ],
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
            {
                'type': 'reasoning-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': IsStr(),
                        'signature': 'gAAAAABpgWJv4-1BjKoolKEKUXGywODUguprX6RungEeXdgyuPIiNzA0lxhJFkYUS9ouBXoZ6oLgvmLe9HoOQTNzYIlIZS7JIBPem1oLG5434XPg7xxI_jagJUZ5YD3ZRQcvwaYvu6WgRc4p7UrScPU24eHoYwqqb__iwMZCw8P8hRJBaBQ-LDsKxEoAWnqGLvDI4WJbcsgA-Q-EHJzSw_3uYh-zz7Ptx7ebhfGhs8Jm7scUf7aRskf951771_U6XbnOa_ZvyeRQ7lOAJTl0N-rJpg6HKFGcssWhQCS4Q-P3ffctA1J3EV2axaVzj3su7n552d2bHXo7G5MfR3s80-WTa6_rmputW2uk5o5icddPtfCL0l0VWyCtPx8vgazNfK1eKOuF7XgPVgxMQ6EppY6ZKt3n7EDZs8W-z_ns88XPtmmtodIExFApnlV3OSuRShWCHywzz4WSZF7jafw4p8jV2WWBCHnPsK1yOyqVq65cGBdjqx28icjtsegOwfx3jX9xdtk9eax7hMUkz8hYXitPTLvqbz-C3Or6qcTz7X_bElKTaZDIWopjYXMZM6mbBIgKlgENzvhGs4-GYIQuKtTAuIz8_Tv1oB21h9DUJw50QZjriEkCJU7bA3EBKEzpzdSVyG7ZRdAm1rPNp0-k8dZ9XAqEOtYFxXfUwIF606UDZBhhFuIY7AGUG8aJ4vul3CZ7lzOms6ayPMmkZkgDNc4y3Us14Rr66bhTQKYHo-uh4i7hTasbpN_Ioy1YXBXAhEgLXtDtxg5YNZUPfCKdSYi2K8PTdwO_7B9qgeLj76wvGAuC34Tbjv0UHTAbrA-bk8A6LTDFd4KOrSZw8aOoq7wq_i53TN-r0eiic_6PuZZqkHYMktWPXiBzoQUwM3Dr47gH1kK9amMz0VoRNabRp4QTfQSkVYuhAMbBGMI0YeVDvU9ndiYNvuRrrrreHF4U-p0AHONRts9WRv9IeOJxKuMsFp1phUz84pCx-p_MAfglGk_8yp6VU7aEwknWBE6S1rwE2CWNoGSiObDMYHqCJTMjwUgwUyATj441nniubCsQTPsh4mFKMZROunreBTxTCyb1CFhT7Q7ijnb4PRoCBjcSLd1yfu7maju50YEeCUVSANQDnxQkAeNWzIJom8btd77bPWoxGbqDyKvGmbcdDgwf9V5lhorUS6Fo-o7QwECqSdOz-duyc3xcANUe-S9demcsL6U01TQ-ZOeYoJ6D3mEN1_-brVu-n2P3nSC-TPw_yKzGi_QX7-A29hUIOCv0sw3Ojj_rvYRkumsY8rQFzkm7beuTduNhOidKcIM-_zFOH57g60peJ-Y42IHDYLldvSqKeALeI4-mL7wSzpQSr-ui6h7WimXP6PnPe3XFMjuQRc20w5W5KQZ9-rdfVr6F1RUD9dB4JkWJs_q8lPvCDCIC2qZ2pzfnyHx9QlXR-A_B0sbkaw7hxuxNSYWxKJFJ_AwCj9aNloovscBCV6Z1Z-BYkqg26e_3mYvSf0dnkn79EnFKjH8HD30BnAoMKNLQPieiR1ix5xEDihh1UM5gZaFzRiZvGJlhpLU5Btpv2gqjTpwZpBtxRzvMod_EgrdcfKcTfve5T6glAEBoWIrW3KZUKsRdxtkED44ewi6PXEqyJj-Dn3ikvnUdcNkZ8Yv1RSHaEKRbfIfuc9KmW7XWuwx4Fpb6g17OTnQF9Wsb0hSro-XA_HAUNYEN5WtPKkvcbAzLw88eVmuJnakx545F2LbdQUKx-Bq6qAaDgl7zhRogLDRSrJWzaeF_ITESg6FVR9B_efG4QPpUk7NhJNgOuzLj68TdxoYZnAdoyzi326fQYC3u3X0BpJBgLQmBoGCXORNXUe0C77QsTlnX__bLiwiaCu_v6snpjgJLgGb8aoDCGbRCN-yfsUTJ9ShP4Leptgz1jViHr_vBDs8pefnKW79DcakPsGC4Zha-644aIHtb2DhPaTVElPofz3VzA643YBo1hR3fAAYyYa_oEmyPDXeD6jaFNdDmLnCNyg1TAxiZXXm_QWRIwKANWZFSjYLYBHBcMLxDofAxPTsUYVz9eSO1w3Ic7jSH-8g94VAVIF5442hq057lCe8Sr9fGg4vAOiG3Hmxi0jU3tMkz5Jo5r4Pnl4ZEDKv1Y9ZUeiFk--RtEtaLEA3SVeiNBAuyFKjE4nCIXBFksbk3vJZdvY9moStJ4fLMkp2Iyx5_w6S7bFOhXvkzGCls_t8AAXHyR-fNgImu4Bll7uUeUakLd1akoe5WuSot4TpRS90IcYRDTwqhH8BXfyG2FOc3yLTOksEnZ2wKrkIQ_80-FYPRI8L1mkCr3LlUHikbtncfGRyXdxLBkXwNQXu6iYVsQ5SH4m5ajKAe7uIaOR9DnOBSWpw8C3qxpT4gIm8B9bOGnhXCMBDEgSL6VEtGNLJx0gtdMq1mWR0j3xyU4BQdnSlqeP2brChne3WHxRlchf3mA0NZVDzRzJN-hmjU1THChI_M66U79-KgB-_laq0QBXsIqw4J3le_GIXlKDNq9__b08KH7BaBnueu7jrmywsSQV_oupmJ7Ne6XCsv81DKtNBWJ-QFYksoeo2jgU2XXudCi9d_N4LsikjCuWtFfczdHOUna-ppw1JaHmrsKYu_Gqem6ek-AjLyTDkSveQ1rHxiuZoVzlohj3mDe702gXrtoS0SEmSFPrQdZP6NN-w-H6XOsVK7idvniBTS-RkohGvvGA7f9EkXb08LS6YtM3YMvdOwdkDxpF8S8EreuTc8D5L4A5whCoEe8Ynw0aImUiNcSduhr7vqioAwi486eCOQBGPWyM0zjzXM7IiOq2O9q-h_PUxLk_As6ly5rR4yWDLw98Ph4j8l2DxDMXQ_Ds734QDBqsHgYLvcq-urACqlhXbwUFS9HKi8QUhizP0mLBsq4kh1oRZ9Clvi8GgiHKH3W9dhPVZGb0NU_CUo-B_YZS7cGwMwkeuC0Cd5KgnhVsjElAegOFnHx4mq-kTFuI0_9rpZJZd5Xg3gobVkC2gr54lsluXfoHHe3ZRNOs89q8df2w9JyLr6nJVc8trVkC41-TkO3FZ4qqtPkD3HRg6UIZUIDE7LhSi0sAmcIvc6viUC6iAIxTtZ5cPhcqEOXndrAitTia3qEqohBO_86-Kh7EEqDHekqfAvi2JTj08SqtoTksfS5GoATm_vFIrkTy-EwlfkmIvA_WYa5oFbZzIFtSY9JG8ouYDBC88BXyrOp2k-wQYBCfPFrIUURmS_mf8n-osza5Y29VoBDl0ZPndXoqOO1KiENowCyPiu82vB1Vg2pOZ-LQ1BPaQwoAXPJt8G6fBdf--nRZEcY7N5NQxPKMBE4gYvUwJBS-uvAR6ofVqRn9w-6wtstoSRPl6s4Skz2iD-UOw8djmcWqeiqoEpu9TG25zow6uEdnbzLnajCrsTXjDcFsF7E92BhADBIP5_RmS6d3_i0vvFcCYdky0CRHyDelT03hY29FGjbl0El7yKeps_x1gh_8Nt-LFsZcsQV-mbDfjuxp8IMZLs1wjzD7H4Q61p5wdFiGdFGLnzNFKh7bt9mdlLnuvE0Cbf3wpFeRFM6pMjq9LtMB5EGHOp4Sky7nvvGdi0AhYqvYSoOLuy2QJV4gp_qhFwUWYJkKo-RvoW_yr_3qEUtLXKIeUFW_YPaOheqI6eOVBn1BX8AzKwFI-9X7pfsJ1Hc9aaClfM4kYvOMOtz4vQR4bnCSYhwQlUoc5Xcwma3JCL8jo6FLcEqqj7J4fht7ifv9_neUJCThs42RKWtRJuYDbiWiPjc1X5upTJaZv2Bp-LBR2YBq8gEuytNmLEldw_OVEfzl8JYpZfaHH1xaGMcflD0s-B7lqCWsUxd-wK6sWH-yPVvU2lhDRiD6UO0YuVMVRJsK9iJLcl-JvE0TLwWAeEKrFhSUA-YWrSNUGwLRrcHMewlXVyNjLbYDc_duG88-gfzSG-OZmJw1BZ9AxBvA7ogJt_tUd_lvNfcQzMOapZFavAcHCBRBN4DO0D5VKmA0BtYXfXMwRXzrF2kkJKN9F0wS3U0zg0PK3fVLFwb2zn8wOskArz5bpCuai0Mml04GJ2rJkskLT3ElATa8iJ4fMtUS5FcGgOnQXryP57jmgcVzUBF_hZad3uyTlJb-jRZ4kAN5bAVFKCyBzj33w-lzPo7qdALLvC5rqfaisqlDzR9AwxoSDD3zTCpiil-fniVojJ7ZWs1D-JaRzwoL6QCv8IU72NpIg-IwK7XD9ER09SulqlvYfrt3A7YMaWvmrG4kIhVK9BrKgEq_vhPizWre-DC3amAoR0DS5hOmvGkQR7ujfg0e1D6gEdXaC-z3AlMl1Z3U7IVQGGLERNta9rGPLMBDfV-cPGChwZ_Fk0EC-PtkNP1I9rb0eS0R6hCvLiwTcDe0sRkUeBgiTemlUy55IdqfzkdKL0BwCR74qrSSFCa688I-MqCUXKuPkeXvsVUJfyxfwDE_Rhas1xXj3ANddmE0I4a1FHiJJ6ZHkkwGahGnoGGXzRzfeRwKlNUBgxXWYeVe1FhJuLDiCmeaHQfHNqkguG5ZsjElRbhbWVMig31DMntnxWkhBHpyi-mrKoTO9RjReQa-jFP3ZIg0FfN9CbDsJi4jto5gdmb3YVUI1ol6ab2WdX2FkOjIIaDFptMw7NsDX42w-df229NGkYM_Y8BVNNAMJRkhpYrU47B-2I-3kpWef9_369Me6Q7rLj8lcp2vKviOA5_CVNsf-9H-CRwMKPLxufWm-msnPnuxOnK_W2vhLQ8T_-cMv6IlJfLiu3CnS4RgXcp9SUPiYUnIJEpZdOrxTRXLH5M3YjhwmA-5o9mP1XsrWlWvNJvfnYx12piakdSbWblUXz-KVPLtG0htXQBoZR-szK3g-NstfacvD__noCcngtmsZvx-fErxgMp1QtAYk4lZFH1eNM7ENnj3QJ0zVTCQYo3XMgGLRi791QODq_1jkcSRfOrwUcGtT3KhG5xCjxJfEth9nkBSytDHULwwS_0HhriOx4Akwmyyl4PcMiyBZWlNo5dCiayUD0Cr5A4S_xJ6HNlOo5dolcHMlUsVzkAC1AITINoPmaRv89u0HhJ-MJn2L1q7fN4-f3oDCM7_i8lBhNHyQ91N4Wv2W9b1gblSPF0DodcZyj279n6e92Z_VYZw4Pj1N0fGaCSDiGfovXiHO-9DzyoE3ZwwanN3Q_5t3LWnoh-AJJ1SrsuX37TknjFc0uiw0_wYrbPoQX8I5GPEzaIOR9T6Cmnpsz_Vxr4pbJLj8t58la273wCOB-4IjhjY4a_ynFIuWmb4cbJki3vWkfQ47zYcLwFXPc2Gkquy-f4b64Ti1Z7m1C2OPnN1p46nK3zIQcQg-f5u9hcrj5oYMR5Y4n5OErMSXgKtZceVs3ozQj4nNw0KWPghqcBwrASK_3XqTZRakKKhovgNnDucEN5NFZPVbonJHVwrPaDAv-qwe9vZAocbtWH6WVlV26NHxm0d7obIVXGkPTW5fn8PvXx1DYNazRmRQ3ES-bvp79iIIFzqDitTYgbICKf8UJM5xarcb3OL9jTAKgmFV3bDaEhlaCMhohIKECPenFPwLGOLM2-3BBMtErqJtfPFjgyWakOdud5HmQiLaSmvMsupEkbJC7pfNZRdszkDDmK7Rv2zrdd7xnqC5JViUIAj227D30xkfd0gzMTgZm7LCjhx623BjVkfhF7esd9vLwORocWco5al_QlhgWKjqGzuNl_K_Ln50YV6khdHeWs-PQv2HA1DvBqT_mBWPAKndAf2ZFb3IigULfpYZbCimQxshYJJrsTyaEZZ7Nk56xZjrhy41b5ICNPziUIIHPVs7mywpypan4zsVwOGTF5R639wHLfdyLXT3J1PF5Ukz66OlfD-QlFR8RlpOa0i9NN3wAykXShph5ms5xGwtSGrMsVHwkfP2R-QHRWunbtH4183W8EksHwYxm9eVyTSObqZka4tYHcf6vsCDD-Jhv88dIoQwMocln16pVcrHKfjBa8YD4zm-K735g0sMHhJbZzZBS1GjeLdLJR4BwTnSe5O28CrJHO81jWupvUsaQ9p2B2YJ288BYicjd142PafyIe5r6zcMYOD-gDM5z28n6cEKn5UeDrrFSTJ4fUpIxWUiGBgazwlOxtF1SA5WJeqGSGGLDUe9t2jzdmz9A2GMPjRS75CKS8168xyBKl0xD0JhPefKVr_RGahfeU4Q14bKzDX3odtbcYA46JpzA2HNnxL69bvNOZvCMBdQUBubtW6XJUGZy0KqSXgoiYvru5qZayhCpq3x6XcOsbWsjLze2e87UQmzff7scGDm2uEZEWkbxAmF_-BPuY45Y077QWzK5igcKEDYX-KMwZk8K9aK3xdlh0gPjtkfLifrhMbj0Oqh_xGQoPuR1YDtpE_OLW5TUilWCjSR8MTUj69-C27mn_6h-fJvBj9cr9TRlFQl9BurYhd31AFouzvOv3qO3sLMiNROi19m3EJ8iOy-PprI-_T1bAuGqszVZ1eqFU3ACWG6u9VqzfAIFxJuKJ4GgqObayQ_dm8b1MUYvVLRMRYlQyRFDjWj_QrFB6kd-oxSE62PLywkGLaSfiuOG6nBxKJUt6ejxrULoUZaCxgo8YL5nSoBhsIzIqrvPz4VgyOekGNtttij34OIuAKbajhypJxG0',
                        'provider_name': 'openai',
                    }
                },
            },
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': IsStr(),
                        'signature': 'gAAAAABpgWJv4-1BjKoolKEKUXGywODUguprX6RungEeXdgyuPIiNzA0lxhJFkYUS9ouBXoZ6oLgvmLe9HoOQTNzYIlIZS7JIBPem1oLG5434XPg7xxI_jagJUZ5YD3ZRQcvwaYvu6WgRc4p7UrScPU24eHoYwqqb__iwMZCw8P8hRJBaBQ-LDsKxEoAWnqGLvDI4WJbcsgA-Q-EHJzSw_3uYh-zz7Ptx7ebhfGhs8Jm7scUf7aRskf951771_U6XbnOa_ZvyeRQ7lOAJTl0N-rJpg6HKFGcssWhQCS4Q-P3ffctA1J3EV2axaVzj3su7n552d2bHXo7G5MfR3s80-WTa6_rmputW2uk5o5icddPtfCL0l0VWyCtPx8vgazNfK1eKOuF7XgPVgxMQ6EppY6ZKt3n7EDZs8W-z_ns88XPtmmtodIExFApnlV3OSuRShWCHywzz4WSZF7jafw4p8jV2WWBCHnPsK1yOyqVq65cGBdjqx28icjtsegOwfx3jX9xdtk9eax7hMUkz8hYXitPTLvqbz-C3Or6qcTz7X_bElKTaZDIWopjYXMZM6mbBIgKlgENzvhGs4-GYIQuKtTAuIz8_Tv1oB21h9DUJw50QZjriEkCJU7bA3EBKEzpzdSVyG7ZRdAm1rPNp0-k8dZ9XAqEOtYFxXfUwIF606UDZBhhFuIY7AGUG8aJ4vul3CZ7lzOms6ayPMmkZkgDNc4y3Us14Rr66bhTQKYHo-uh4i7hTasbpN_Ioy1YXBXAhEgLXtDtxg5YNZUPfCKdSYi2K8PTdwO_7B9qgeLj76wvGAuC34Tbjv0UHTAbrA-bk8A6LTDFd4KOrSZw8aOoq7wq_i53TN-r0eiic_6PuZZqkHYMktWPXiBzoQUwM3Dr47gH1kK9amMz0VoRNabRp4QTfQSkVYuhAMbBGMI0YeVDvU9ndiYNvuRrrrreHF4U-p0AHONRts9WRv9IeOJxKuMsFp1phUz84pCx-p_MAfglGk_8yp6VU7aEwknWBE6S1rwE2CWNoGSiObDMYHqCJTMjwUgwUyATj441nniubCsQTPsh4mFKMZROunreBTxTCyb1CFhT7Q7ijnb4PRoCBjcSLd1yfu7maju50YEeCUVSANQDnxQkAeNWzIJom8btd77bPWoxGbqDyKvGmbcdDgwf9V5lhorUS6Fo-o7QwECqSdOz-duyc3xcANUe-S9demcsL6U01TQ-ZOeYoJ6D3mEN1_-brVu-n2P3nSC-TPw_yKzGi_QX7-A29hUIOCv0sw3Ojj_rvYRkumsY8rQFzkm7beuTduNhOidKcIM-_zFOH57g60peJ-Y42IHDYLldvSqKeALeI4-mL7wSzpQSr-ui6h7WimXP6PnPe3XFMjuQRc20w5W5KQZ9-rdfVr6F1RUD9dB4JkWJs_q8lPvCDCIC2qZ2pzfnyHx9QlXR-A_B0sbkaw7hxuxNSYWxKJFJ_AwCj9aNloovscBCV6Z1Z-BYkqg26e_3mYvSf0dnkn79EnFKjH8HD30BnAoMKNLQPieiR1ix5xEDihh1UM5gZaFzRiZvGJlhpLU5Btpv2gqjTpwZpBtxRzvMod_EgrdcfKcTfve5T6glAEBoWIrW3KZUKsRdxtkED44ewi6PXEqyJj-Dn3ikvnUdcNkZ8Yv1RSHaEKRbfIfuc9KmW7XWuwx4Fpb6g17OTnQF9Wsb0hSro-XA_HAUNYEN5WtPKkvcbAzLw88eVmuJnakx545F2LbdQUKx-Bq6qAaDgl7zhRogLDRSrJWzaeF_ITESg6FVR9B_efG4QPpUk7NhJNgOuzLj68TdxoYZnAdoyzi326fQYC3u3X0BpJBgLQmBoGCXORNXUe0C77QsTlnX__bLiwiaCu_v6snpjgJLgGb8aoDCGbRCN-yfsUTJ9ShP4Leptgz1jViHr_vBDs8pefnKW79DcakPsGC4Zha-644aIHtb2DhPaTVElPofz3VzA643YBo1hR3fAAYyYa_oEmyPDXeD6jaFNdDmLnCNyg1TAxiZXXm_QWRIwKANWZFSjYLYBHBcMLxDofAxPTsUYVz9eSO1w3Ic7jSH-8g94VAVIF5442hq057lCe8Sr9fGg4vAOiG3Hmxi0jU3tMkz5Jo5r4Pnl4ZEDKv1Y9ZUeiFk--RtEtaLEA3SVeiNBAuyFKjE4nCIXBFksbk3vJZdvY9moStJ4fLMkp2Iyx5_w6S7bFOhXvkzGCls_t8AAXHyR-fNgImu4Bll7uUeUakLd1akoe5WuSot4TpRS90IcYRDTwqhH8BXfyG2FOc3yLTOksEnZ2wKrkIQ_80-FYPRI8L1mkCr3LlUHikbtncfGRyXdxLBkXwNQXu6iYVsQ5SH4m5ajKAe7uIaOR9DnOBSWpw8C3qxpT4gIm8B9bOGnhXCMBDEgSL6VEtGNLJx0gtdMq1mWR0j3xyU4BQdnSlqeP2brChne3WHxRlchf3mA0NZVDzRzJN-hmjU1THChI_M66U79-KgB-_laq0QBXsIqw4J3le_GIXlKDNq9__b08KH7BaBnueu7jrmywsSQV_oupmJ7Ne6XCsv81DKtNBWJ-QFYksoeo2jgU2XXudCi9d_N4LsikjCuWtFfczdHOUna-ppw1JaHmrsKYu_Gqem6ek-AjLyTDkSveQ1rHxiuZoVzlohj3mDe702gXrtoS0SEmSFPrQdZP6NN-w-H6XOsVK7idvniBTS-RkohGvvGA7f9EkXb08LS6YtM3YMvdOwdkDxpF8S8EreuTc8D5L4A5whCoEe8Ynw0aImUiNcSduhr7vqioAwi486eCOQBGPWyM0zjzXM7IiOq2O9q-h_PUxLk_As6ly5rR4yWDLw98Ph4j8l2DxDMXQ_Ds734QDBqsHgYLvcq-urACqlhXbwUFS9HKi8QUhizP0mLBsq4kh1oRZ9Clvi8GgiHKH3W9dhPVZGb0NU_CUo-B_YZS7cGwMwkeuC0Cd5KgnhVsjElAegOFnHx4mq-kTFuI0_9rpZJZd5Xg3gobVkC2gr54lsluXfoHHe3ZRNOs89q8df2w9JyLr6nJVc8trVkC41-TkO3FZ4qqtPkD3HRg6UIZUIDE7LhSi0sAmcIvc6viUC6iAIxTtZ5cPhcqEOXndrAitTia3qEqohBO_86-Kh7EEqDHekqfAvi2JTj08SqtoTksfS5GoATm_vFIrkTy-EwlfkmIvA_WYa5oFbZzIFtSY9JG8ouYDBC88BXyrOp2k-wQYBCfPFrIUURmS_mf8n-osza5Y29VoBDl0ZPndXoqOO1KiENowCyPiu82vB1Vg2pOZ-LQ1BPaQwoAXPJt8G6fBdf--nRZEcY7N5NQxPKMBE4gYvUwJBS-uvAR6ofVqRn9w-6wtstoSRPl6s4Skz2iD-UOw8djmcWqeiqoEpu9TG25zow6uEdnbzLnajCrsTXjDcFsF7E92BhADBIP5_RmS6d3_i0vvFcCYdky0CRHyDelT03hY29FGjbl0El7yKeps_x1gh_8Nt-LFsZcsQV-mbDfjuxp8IMZLs1wjzD7H4Q61p5wdFiGdFGLnzNFKh7bt9mdlLnuvE0Cbf3wpFeRFM6pMjq9LtMB5EGHOp4Sky7nvvGdi0AhYqvYSoOLuy2QJV4gp_qhFwUWYJkKo-RvoW_yr_3qEUtLXKIeUFW_YPaOheqI6eOVBn1BX8AzKwFI-9X7pfsJ1Hc9aaClfM4kYvOMOtz4vQR4bnCSYhwQlUoc5Xcwma3JCL8jo6FLcEqqj7J4fht7ifv9_neUJCThs42RKWtRJuYDbiWiPjc1X5upTJaZv2Bp-LBR2YBq8gEuytNmLEldw_OVEfzl8JYpZfaHH1xaGMcflD0s-B7lqCWsUxd-wK6sWH-yPVvU2lhDRiD6UO0YuVMVRJsK9iJLcl-JvE0TLwWAeEKrFhSUA-YWrSNUGwLRrcHMewlXVyNjLbYDc_duG88-gfzSG-OZmJw1BZ9AxBvA7ogJt_tUd_lvNfcQzMOapZFavAcHCBRBN4DO0D5VKmA0BtYXfXMwRXzrF2kkJKN9F0wS3U0zg0PK3fVLFwb2zn8wOskArz5bpCuai0Mml04GJ2rJkskLT3ElATa8iJ4fMtUS5FcGgOnQXryP57jmgcVzUBF_hZad3uyTlJb-jRZ4kAN5bAVFKCyBzj33w-lzPo7qdALLvC5rqfaisqlDzR9AwxoSDD3zTCpiil-fniVojJ7ZWs1D-JaRzwoL6QCv8IU72NpIg-IwK7XD9ER09SulqlvYfrt3A7YMaWvmrG4kIhVK9BrKgEq_vhPizWre-DC3amAoR0DS5hOmvGkQR7ujfg0e1D6gEdXaC-z3AlMl1Z3U7IVQGGLERNta9rGPLMBDfV-cPGChwZ_Fk0EC-PtkNP1I9rb0eS0R6hCvLiwTcDe0sRkUeBgiTemlUy55IdqfzkdKL0BwCR74qrSSFCa688I-MqCUXKuPkeXvsVUJfyxfwDE_Rhas1xXj3ANddmE0I4a1FHiJJ6ZHkkwGahGnoGGXzRzfeRwKlNUBgxXWYeVe1FhJuLDiCmeaHQfHNqkguG5ZsjElRbhbWVMig31DMntnxWkhBHpyi-mrKoTO9RjReQa-jFP3ZIg0FfN9CbDsJi4jto5gdmb3YVUI1ol6ab2WdX2FkOjIIaDFptMw7NsDX42w-df229NGkYM_Y8BVNNAMJRkhpYrU47B-2I-3kpWef9_369Me6Q7rLj8lcp2vKviOA5_CVNsf-9H-CRwMKPLxufWm-msnPnuxOnK_W2vhLQ8T_-cMv6IlJfLiu3CnS4RgXcp9SUPiYUnIJEpZdOrxTRXLH5M3YjhwmA-5o9mP1XsrWlWvNJvfnYx12piakdSbWblUXz-KVPLtG0htXQBoZR-szK3g-NstfacvD__noCcngtmsZvx-fErxgMp1QtAYk4lZFH1eNM7ENnj3QJ0zVTCQYo3XMgGLRi791QODq_1jkcSRfOrwUcGtT3KhG5xCjxJfEth9nkBSytDHULwwS_0HhriOx4Akwmyyl4PcMiyBZWlNo5dCiayUD0Cr5A4S_xJ6HNlOo5dolcHMlUsVzkAC1AITINoPmaRv89u0HhJ-MJn2L1q7fN4-f3oDCM7_i8lBhNHyQ91N4Wv2W9b1gblSPF0DodcZyj279n6e92Z_VYZw4Pj1N0fGaCSDiGfovXiHO-9DzyoE3ZwwanN3Q_5t3LWnoh-AJJ1SrsuX37TknjFc0uiw0_wYrbPoQX8I5GPEzaIOR9T6Cmnpsz_Vxr4pbJLj8t58la273wCOB-4IjhjY4a_ynFIuWmb4cbJki3vWkfQ47zYcLwFXPc2Gkquy-f4b64Ti1Z7m1C2OPnN1p46nK3zIQcQg-f5u9hcrj5oYMR5Y4n5OErMSXgKtZceVs3ozQj4nNw0KWPghqcBwrASK_3XqTZRakKKhovgNnDucEN5NFZPVbonJHVwrPaDAv-qwe9vZAocbtWH6WVlV26NHxm0d7obIVXGkPTW5fn8PvXx1DYNazRmRQ3ES-bvp79iIIFzqDitTYgbICKf8UJM5xarcb3OL9jTAKgmFV3bDaEhlaCMhohIKECPenFPwLGOLM2-3BBMtErqJtfPFjgyWakOdud5HmQiLaSmvMsupEkbJC7pfNZRdszkDDmK7Rv2zrdd7xnqC5JViUIAj227D30xkfd0gzMTgZm7LCjhx623BjVkfhF7esd9vLwORocWco5al_QlhgWKjqGzuNl_K_Ln50YV6khdHeWs-PQv2HA1DvBqT_mBWPAKndAf2ZFb3IigULfpYZbCimQxshYJJrsTyaEZZ7Nk56xZjrhy41b5ICNPziUIIHPVs7mywpypan4zsVwOGTF5R639wHLfdyLXT3J1PF5Ukz66OlfD-QlFR8RlpOa0i9NN3wAykXShph5ms5xGwtSGrMsVHwkfP2R-QHRWunbtH4183W8EksHwYxm9eVyTSObqZka4tYHcf6vsCDD-Jhv88dIoQwMocln16pVcrHKfjBa8YD4zm-K735g0sMHhJbZzZBS1GjeLdLJR4BwTnSe5O28CrJHO81jWupvUsaQ9p2B2YJ288BYicjd142PafyIe5r6zcMYOD-gDM5z28n6cEKn5UeDrrFSTJ4fUpIxWUiGBgazwlOxtF1SA5WJeqGSGGLDUe9t2jzdmz9A2GMPjRS75CKS8168xyBKl0xD0JhPefKVr_RGahfeU4Q14bKzDX3odtbcYA46JpzA2HNnxL69bvNOZvCMBdQUBubtW6XJUGZy0KqSXgoiYvru5qZayhCpq3x6XcOsbWsjLze2e87UQmzff7scGDm2uEZEWkbxAmF_-BPuY45Y077QWzK5igcKEDYX-KMwZk8K9aK3xdlh0gPjtkfLifrhMbj0Oqh_xGQoPuR1YDtpE_OLW5TUilWCjSR8MTUj69-C27mn_6h-fJvBj9cr9TRlFQl9BurYhd31AFouzvOv3qO3sLMiNROi19m3EJ8iOy-PprI-_T1bAuGqszVZ1eqFU3ACWG6u9VqzfAIFxJuKJ4GgqObayQ_dm8b1MUYvVLRMRYlQyRFDjWj_QrFB6kd-oxSE62PLywkGLaSfiuOG6nBxKJUt6ejxrULoUZaCxgo8YL5nSoBhsIzIqrvPz4VgyOekGNtttij34OIuAKbajhypJxG0',
                        'provider_name': 'openai',
                    }
                },
            },
            {
                'type': 'text-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': IsStr(),
                        'provider_name': 'openai',
                    }
                },
            },
            {
                'type': 'text-delta',
                'delta': """\
Short answer:
- Headers\
""",
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': IsStr(),
                        'provider_name': 'openai',
                    }
                },
            },
            {'type': 'text-delta', 'delta': IsStr(), 'id': IsStr()},
            {'type': 'text-delta', 'delta': IsStr(), 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': ' FastAPI instrumentation, or',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' pass the OpenTelemetry', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' header-capture options', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
.
- Bodies: FastAPI\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': ' server instrumentation does not',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' record request/response', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': ' bodies by default. You can capture validated inputs',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': IsStr(), 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' integration, or add', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' custom middleware. For', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' outbound calls (httpx', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '), Logfire can', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
 capture bodies.

How\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': ' to include HTTP request/response details',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' with Logfire + Fast', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
API

1) Capture\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': ' request and response headers\n',
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
- Easiest:
  log\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': 'fire.instrument_fastapi', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '(app, capture_headers', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
=True)
  This records all\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': ' request and response headers',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' on the span', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '. ', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
([logfire.pydantic.dev](https://logfire.pydantic.dev/docs/reference/api/logfire/))

- Fine\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '‑grained (two', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
 equivalent ways):
  a\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ') Environment variables:', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\

     OTEL_IN\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': 'STRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST=".*',
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
"
     OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
_RESPONSE=".*"
     OTEL_IN\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': 'STRUMENTATION_HTTP_CAPTURE_HEADERS_SANITIZE_FIELDS="Authorization',
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
"
     These control\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': ' which headers are added and',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' which are redacted.', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' ', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
([logfire.pydantic.dev](https://logfire.pydantic.dev/docs/integrations/web-frameworks/))
  b) Keyword args passed\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' through to OpenTelemetry', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' via Logfire:', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\

     logfire.instrument_fastapi(
        \
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
 app,
         http_capture\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
_headers_server_request=".*",
         http_capture_headers_server\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
_response=".*",
     )
     Log\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': 'fire forwards extra kwargs', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' to FastAPIInstrumentor', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '.instrument_app(). ', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': '([logfire.pydantic.dev](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/)',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ')', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\


2) What about the\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
 body?
- Inbound\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': IsStr(), 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' no built‑in flag', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': ' to automatically record HTTP bodies',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' in FastAPI server spans', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '. Logfire does record', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' parsed/validated endpoint', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': ' arguments and any validation',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' errors; you can customize', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' what gets logged via', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': ' request_attributes_mapper.',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' ', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
([logfire.pydantic.dev](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/))
- If you truly need\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' raw bodies, add', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' a small FastAPI/', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': 'Starlette middleware to read',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' and log them (be', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' careful with PII/se', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'crets), or enable AS', 'id': IsStr()},
            {'type': 'text-delta', 'delta': 'GI send/receive spans', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' for low‑level', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' I/O details:', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\

  logfire.instrument\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': '_fastapi(app, record', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
_send_receive=True)
  (creates\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' debug‑level send', 'id': IsStr()},
            {'type': 'text-delta', 'delta': '/receive spans; not the', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' actual body contents).', 'id': IsStr()},
            {'type': 'text-delta', 'delta': ' ', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
([logfire.pydantic.dev](https://logfire.pydantic.dev/docs/reference/api/logfire/))

3) Outbound\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' HTTP bodies (if', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': """\
 that's what you meant)
-\
""",
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' For httpx clients, Log', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': 'fire can capture headers and',
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
 bodies:
  log\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
fire.instrument_httpx(
      capture_headers=True\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
,
      capture_request_body=True,
      capture_response_body=True\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
,
  )
  \
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': """\
([logfire.pydantic.dev](https://logfire.pydantic.dev/docs/reference/api/logfire/))

Would you like a minimal\
""",
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': ' FastAPI middleware example to',
                'id': IsStr(),
            },
            {
                'type': 'text-delta',
                'delta': ' log request/response bodies with',
                'id': IsStr(),
            },
            {'type': 'text-delta', 'delta': ' scrubbing, or are', 'id': IsStr()},
            {
                'type': 'text-delta',
                'delta': ' headers/validated arguments sufficient?',
                'id': IsStr(),
            },
            {
                'type': 'text-end',
                'id': IsStr(),
                'providerMetadata': {'pydantic_ai': {'id': IsStr(), 'provider_name': 'openai'}},
            },
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
            {
                'type': 'reasoning-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': 'sig_xyz789',
                        'provider_name': 'anthropic',
                        'id': 'thinking_456',
                        'provider_details': {'model': 'claude-3', 'tokens': 100},
                    }
                },
            },
            {
                'type': 'reasoning-delta',
                'id': IsStr(),
                'delta': 'Deep thought...',
                'providerMetadata': {
                    'pydantic_ai': {
                        'signature': 'sig_xyz789',
                        'provider_name': 'anthropic',
                        'id': 'thinking_456',
                        'provider_details': {'model': 'claude-3', 'tokens': 100},
                    }
                },
            },
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
    adapter = VercelAIAdapter(agent, request, sdk_version=6)
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
                'toolCallId': 'search_1',
                'toolName': 'web_search',
                'providerExecuted': True,
                'providerMetadata': {'pydantic_ai': {'provider_name': 'function'}},
            },
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
                        call_provider_metadata={
                            'pydantic_ai': {
                                'call_meta': {'provider_name': 'openai'},
                                'return_meta': {'provider_name': 'openai_return'},
                            }
                        },
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
                        provider_name='openai_return',
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
                    provider_details={'tool_type': 'web_search_preview'},
                ),
                BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={'status': 'completed'},
                    tool_call_id='tool_456',
                    provider_name='openai',
                    provider_details={'execution_time_ms': 150},
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
                        'call_provider_metadata': {
                            'pydantic_ai': {
                                'call_meta': {
                                    'provider_name': 'openai',
                                    'provider_details': {'tool_type': 'web_search_preview'},
                                },
                                'return_meta': {
                                    'provider_name': 'openai',
                                    'provider_details': {'execution_time_ms': 150},
                                },
                            }
                        },
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
                        'call_provider_metadata': {
                            'pydantic_ai': {'provider_name': 'openai'}
                        },  # No return part, so defaults to normal call provider name
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

    # Verify roundtrip
    reloaded_messages = VercelAIAdapter.load_messages(ui_messages)
    # Content will have changed for retry prompt part, so we check it's value
    # And then set it back to the original value
    retry_prompt_part = reloaded_messages[2].parts[0]
    assert isinstance(retry_prompt_part, RetryPromptPart)
    assert retry_prompt_part == snapshot(
        RetryPromptPart(
            content='Tool failed with error\n\nFix the errors and try again.',
            tool_name='my_tool',
            tool_call_id='tool_789',
            timestamp=IsDatetime(),
        )
    )
    retry_prompt_part.content = 'Tool failed with error'
    _sync_timestamps(messages, reloaded_messages)
    assert reloaded_messages == messages


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

    # Verify roundtrip
    # Note: This is a lossy conversion - RetryPromptPart without tool_call_id becomes a user text message.
    # When loaded back, it creates a UserPromptPart instead of RetryPromptPart.
    # So we check it's value and then replace it with the original RetryPromptPart to assert equality
    reloaded_messages = VercelAIAdapter.load_messages(ui_messages)
    assert reloaded_messages[2] == snapshot(
        ModelRequest(
            parts=[
                UserPromptPart(
                    content="""\
Validation feedback:
Output validation failed: expected integer

Fix the errors and try again.\
""",
                    timestamp=IsDatetime(),
                )
            ]
        )
    )
    # Get original tool_call_id and replace with original RetryPromptPart
    original_retry = messages[2].parts[0]
    assert isinstance(original_retry, RetryPromptPart)
    reloaded_messages[2] = ModelRequest(
        parts=[
            RetryPromptPart(
                content='Output validation failed: expected integer', tool_call_id=original_retry.tool_call_id
            )
        ]
    )
    _sync_timestamps(messages, reloaded_messages)
    assert reloaded_messages == messages


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
                        'call_provider_metadata': {
                            'pydantic_ai': {
                                'call_meta': {'provider_name': 'test'},
                                'return_meta': {'provider_name': 'test'},
                            }
                        },
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

    # Load back to Pydantic AI format
    reloaded_messages = VercelAIAdapter.load_messages(ui_messages)
    _sync_timestamps(original_messages, reloaded_messages)

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


async def test_adapter_text_part_with_provider_metadata():
    """Test TextPart with provider_name and provider_details preserves metadata and roundtrips."""
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                TextPart(
                    content='Hello with metadata',
                    id='text_123',
                    provider_name='openai',
                    provider_details={'model': 'gpt-4', 'finish_reason': 'stop'},
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
                        'type': 'text',
                        'text': 'Hello with metadata',
                        'state': 'done',
                        'provider_metadata': {
                            'pydantic_ai': {
                                'id': 'text_123',
                                'provider_name': 'openai',
                                'provider_details': {'model': 'gpt-4', 'finish_reason': 'stop'},
                            }
                        },
                    }
                ],
            }
        ]
    )

    # Verify roundtrip
    reloaded_messages = VercelAIAdapter.load_messages(ui_messages)
    _sync_timestamps(messages, reloaded_messages)
    assert reloaded_messages == messages


async def test_adapter_load_messages_text_with_provider_metadata():
    """Test loading TextUIPart with provider_metadata preserves metadata on TextPart."""
    ui_messages = [
        UIMessage(
            id='msg1',
            role='assistant',
            parts=[
                TextUIPart(
                    text='Hello with metadata',
                    state='done',
                    provider_metadata={
                        'pydantic_ai': {
                            'id': 'text_123',
                            'provider_name': 'anthropic',
                            'provider_details': {'model': 'gpt-4', 'tokens': 50},
                        }
                    },
                )
            ],
        )
    ]

    messages = VercelAIAdapter.load_messages(ui_messages)
    assert messages == snapshot(
        [
            ModelResponse(
                parts=[
                    TextPart(
                        content='Hello with metadata',
                        id='text_123',
                        provider_name='anthropic',
                        provider_details={'model': 'gpt-4', 'tokens': 50},
                    )
                ],
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_adapter_tool_call_part_with_provider_metadata():
    """Test ToolCallPart with provider_name and provider_details preserves metadata and roundtrips."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Do something')]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='my_tool',
                    args={'arg': 'value'},
                    tool_call_id='tool_abc',
                    id='call_123',
                    provider_name='openai',
                    provider_details={'index': 0, 'type': 'function'},
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='my_tool',
                    content='result',
                    tool_call_id='tool_abc',
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
                        'tool_call_id': 'tool_abc',
                        'state': 'output-available',
                        'input': '{"arg":"value"}',
                        'output': 'result',
                        'call_provider_metadata': {
                            'pydantic_ai': {
                                'id': 'call_123',
                                'provider_name': 'openai',
                                'provider_details': {'index': 0, 'type': 'function'},
                            }
                        },
                        'preliminary': None,
                    }
                ],
            },
        ]
    )

    # Verify roundtrip
    reloaded_messages = VercelAIAdapter.load_messages(ui_messages)
    _sync_timestamps(messages, reloaded_messages)
    assert reloaded_messages == messages


async def test_adapter_load_messages_tool_call_with_provider_metadata():
    """Test loading dynamic tool part with provider_metadata preserves metadata on ToolCallPart."""
    from pydantic_ai.ui.vercel_ai.request_types import DynamicToolInputAvailablePart

    ui_messages = [
        UIMessage(
            id='msg1',
            role='assistant',
            parts=[
                DynamicToolInputAvailablePart(
                    tool_name='my_tool',
                    tool_call_id='tc_123',
                    input='{"key": "value"}',
                    state='input-available',
                    call_provider_metadata={
                        'pydantic_ai': {
                            'provider_name': 'anthropic',
                            'provider_details': {'index': 0},
                        }
                    },
                )
            ],
        )
    ]

    messages = VercelAIAdapter.load_messages(ui_messages)
    assert messages == snapshot(
        [
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='my_tool',
                        args={'key': 'value'},
                        tool_call_id='tc_123',
                        provider_name='anthropic',
                        provider_details={'index': 0},
                    ),
                ],
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_adapter_file_part_with_provider_metadata():
    """Test FilePart with provider metadata preserves id, provider_name, provider_details and roundtrips."""
    # Use BinaryImage (not BinaryContent) since that's what load_messages produces for images
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                FilePart(
                    content=BinaryImage(data=b'file_data', media_type='image/png'),
                    id='file_123',
                    provider_name='openai',
                    provider_details={'generation_id': 'gen_abc'},
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
                        'type': 'file',
                        'media_type': 'image/png',
                        'filename': None,
                        'url': 'data:image/png;base64,ZmlsZV9kYXRh',
                        'provider_metadata': {
                            'pydantic_ai': {
                                'id': 'file_123',
                                'provider_name': 'openai',
                                'provider_details': {'generation_id': 'gen_abc'},
                            }
                        },
                    }
                ],
            }
        ]
    )

    # Verify roundtrip
    reloaded_messages = VercelAIAdapter.load_messages(ui_messages)
    _sync_timestamps(messages, reloaded_messages)
    assert reloaded_messages == messages


async def test_adapter_load_messages_file_with_provider_metadata():
    """Test loading FileUIPart with provider_metadata preserves id, provider_name, and provider_details."""
    ui_messages = [
        UIMessage(
            id='msg1',
            role='assistant',
            parts=[
                FileUIPart(
                    url='data:image/png;base64,ZmlsZV9kYXRh',
                    media_type='image/png',
                    provider_metadata={
                        'pydantic_ai': {
                            'id': 'file_456',
                            'provider_name': 'anthropic',
                            'provider_details': {'source': 'generated'},
                        }
                    },
                )
            ],
        )
    ]

    messages = VercelAIAdapter.load_messages(ui_messages)
    assert messages == snapshot(
        [
            ModelResponse(
                parts=[
                    FilePart(
                        content=BinaryImage(data=b'file_data', media_type='image/png', _identifier='cdd967'),
                        id='file_456',
                        provider_name='anthropic',
                        provider_details={'source': 'generated'},
                    )
                ],
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_adapter_builtin_tool_part_with_provider_metadata():
    """Test BuiltinToolCallPart with id, provider_name, provider_details and roundtrips."""
    # Use JSON string for content since that's what load_messages produces
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Search')]),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'test'},
                    tool_call_id='bt_123',
                    id='call_456',
                    provider_name='openai',
                    provider_details={'tool_type': 'web_search_preview'},
                ),
                BuiltinToolReturnPart(
                    tool_name='web_search',
                    content='{"results":[]}',  # JSON string for roundtrip compatibility
                    tool_call_id='bt_123',
                    provider_name='openai',
                    provider_details={'execution_time_ms': 150},
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
                'parts': [{'type': 'text', 'text': 'Search', 'state': 'done', 'provider_metadata': None}],
            },
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {
                        'type': 'tool-web_search',
                        'tool_call_id': 'bt_123',
                        'state': 'output-available',
                        'input': '{"query":"test"}',
                        'output': '{"results":[]}',
                        'provider_executed': True,
                        'call_provider_metadata': {
                            'pydantic_ai': {
                                'call_meta': {
                                    'id': 'call_456',
                                    'provider_name': 'openai',
                                    'provider_details': {'tool_type': 'web_search_preview'},
                                },
                                'return_meta': {
                                    'provider_name': 'openai',
                                    'provider_details': {'execution_time_ms': 150},
                                },
                            }
                        },
                        'preliminary': None,
                    }
                ],
            },
        ]
    )

    # Verify roundtrip
    reloaded_messages = VercelAIAdapter.load_messages(ui_messages)
    _sync_timestamps(messages, reloaded_messages)
    assert reloaded_messages == messages


async def test_adapter_builtin_tool_error_part_with_provider_metadata():
    """Test BuiltinToolReturnPart with error content creates ToolOutputErrorPart and roundtrips."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Search')]),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    tool_name='web_search',
                    args={'query': 'test'},
                    tool_call_id='bt_err_123',
                    id='call_err_456',
                    provider_name='openai',
                    provider_details={'tool_type': 'web_search_preview'},
                ),
                BuiltinToolReturnPart(
                    tool_name='web_search',
                    content={'error_text': 'Search failed: rate limit exceeded', 'is_error': True},
                    tool_call_id='bt_err_123',
                    provider_name='openai',
                    provider_details={'error_code': 'RATE_LIMIT'},
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
                'parts': [{'type': 'text', 'text': 'Search', 'state': 'done', 'provider_metadata': None}],
            },
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {
                        'type': 'tool-web_search',
                        'tool_call_id': 'bt_err_123',
                        'state': 'output-error',
                        'input': '{"query":"test"}',
                        'raw_input': None,
                        'error_text': 'Search failed: rate limit exceeded',
                        'provider_executed': True,
                        'call_provider_metadata': {
                            'pydantic_ai': {
                                'call_meta': {
                                    'id': 'call_err_456',
                                    'provider_name': 'openai',
                                    'provider_details': {'tool_type': 'web_search_preview'},
                                },
                                'return_meta': {
                                    'provider_name': 'openai',
                                    'provider_details': {'error_code': 'RATE_LIMIT'},
                                },
                            }
                        },
                    }
                ],
            },
        ]
    )

    # Verify roundtrip
    reloaded_messages = VercelAIAdapter.load_messages(ui_messages)
    _sync_timestamps(messages, reloaded_messages)
    assert reloaded_messages == messages


async def test_adapter_load_messages_builtin_tool_with_provider_details():
    """Test loading builtin tool with provider_details on return part."""
    ui_messages = [
        UIMessage(
            id='msg1',
            role='assistant',
            parts=[
                ToolOutputAvailablePart(
                    type='tool-web_search',
                    tool_call_id='bt_load',
                    input='{"query": "test"}',
                    output='{"results": []}',
                    state='output-available',
                    provider_executed=True,
                    call_provider_metadata={
                        'pydantic_ai': {
                            'call_meta': {
                                'id': 'call_456',
                                'provider_name': 'openai',
                                'provider_details': {'tool_type': 'web_search_preview'},
                            },
                            'return_meta': {
                                'id': 'call_456',
                                'provider_name': 'openai',
                                'provider_details': {'execution_time_ms': 150},
                            },
                        }
                    },
                )
            ],
        )
    ]

    messages = VercelAIAdapter.load_messages(ui_messages)
    assert messages == snapshot(
        [
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'test'},
                        tool_call_id='bt_load',
                        id='call_456',
                        provider_details={'tool_type': 'web_search_preview'},
                        provider_name='openai',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content='{"results": []}',
                        tool_call_id='bt_load',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                        provider_details={'execution_time_ms': 150},
                    ),
                ],
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_adapter_load_messages_builtin_tool_error_with_provider_details():
    """Test loading builtin tool error with provider_details - ensures ToolOutputErrorPart metadata is extracted."""
    ui_messages = [
        UIMessage(
            id='msg1',
            role='assistant',
            parts=[
                ToolOutputErrorPart(
                    type='tool-web_search',
                    tool_call_id='bt_error',
                    input='{"query": "test"}',
                    error_text='Search failed: rate limit exceeded',
                    state='output-error',
                    provider_executed=True,
                    call_provider_metadata={
                        'pydantic_ai': {
                            'call_meta': {
                                'id': 'call_789',
                                'provider_name': 'openai',
                                'provider_details': {'tool_type': 'web_search_preview'},
                            },
                            'return_meta': {
                                'provider_name': 'openai',
                                'provider_details': {'error_code': 'RATE_LIMIT'},
                            },
                        }
                    },
                )
            ],
        )
    ]

    messages = VercelAIAdapter.load_messages(ui_messages)
    assert messages == snapshot(
        [
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'query': 'test'},
                        tool_call_id='bt_error',
                        id='call_789',
                        provider_name='openai',
                        provider_details={'tool_type': 'web_search_preview'},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'error_text': 'Search failed: rate limit exceeded', 'is_error': True},
                        tool_call_id='bt_error',
                        timestamp=IsDatetime(),
                        provider_name='openai',
                        provider_details={'error_code': 'RATE_LIMIT'},
                    ),
                ],
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_adapter_load_messages_tool_input_streaming_part():
    """Test loading ToolInputStreamingPart which doesn't have call_provider_metadata yet."""
    from pydantic_ai.ui.vercel_ai.request_types import ToolInputStreamingPart

    ui_messages = [
        UIMessage(
            id='msg1',
            role='assistant',
            parts=[
                ToolInputStreamingPart(
                    type='tool-my_tool',
                    tool_call_id='tc_streaming',
                    input='{"query": "test"}',
                    state='input-streaming',
                )
            ],
        )
    ]

    messages = VercelAIAdapter.load_messages(ui_messages)
    assert messages == snapshot(
        [
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='my_tool', args={'query': 'test'}, tool_call_id='tc_streaming'),
                ],
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_adapter_load_messages_dynamic_tool_input_streaming_part():
    """Test loading DynamicToolInputStreamingPart which doesn't have call_provider_metadata yet."""
    from pydantic_ai.ui.vercel_ai.request_types import DynamicToolInputStreamingPart

    ui_messages = [
        UIMessage(
            id='msg1',
            role='assistant',
            parts=[
                DynamicToolInputStreamingPart(
                    tool_name='dynamic_tool',
                    tool_call_id='tc_dyn_streaming',
                    input='{"arg": 123}',
                    state='input-streaming',
                )
            ],
        )
    ]

    messages = VercelAIAdapter.load_messages(ui_messages)
    assert messages == snapshot(
        [
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='dynamic_tool', args={'arg': 123}, tool_call_id='tc_dyn_streaming'),
                ],
                timestamp=IsDatetime(),
            )
        ]
    )


async def test_adapter_dump_messages_tool_error_with_provider_metadata():
    """Test dumping ToolCallPart with RetryPromptPart includes provider metadata with provider_name."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content='Do task')]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='failing_tool',
                    args={'x': 1},
                    tool_call_id='tc_fail',
                    id='call_fail_id',
                    provider_name='google',
                    provider_details={'attempt': 1},
                ),
            ]
        ),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    content='Tool execution failed',
                    tool_name='failing_tool',
                    tool_call_id='tc_fail',
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
                'parts': [{'type': 'text', 'text': 'Do task', 'state': 'done', 'provider_metadata': None}],
            },
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': None,
                'parts': [
                    {
                        'type': 'dynamic-tool',
                        'tool_name': 'failing_tool',
                        'tool_call_id': 'tc_fail',
                        'state': 'output-error',
                        'input': '{"x":1}',
                        'error_text': """\
Tool execution failed

Fix the errors and try again.\
""",
                        'call_provider_metadata': {
                            'pydantic_ai': {
                                'id': 'call_fail_id',
                                'provider_name': 'google',
                                'provider_details': {'attempt': 1},
                            }
                        },
                    }
                ],
            },
        ]
    )

    # Verify roundtrip
    reloaded_messages = VercelAIAdapter.load_messages(ui_messages)
    # Content will have changed for retry prompt part, so we set it back to the original value
    retry_prompt_part = reloaded_messages[2].parts[0]
    assert isinstance(retry_prompt_part, RetryPromptPart)
    assert retry_prompt_part.content == 'Tool execution failed\n\nFix the errors and try again.'
    retry_prompt_part.content = 'Tool execution failed'
    _sync_timestamps(messages, reloaded_messages)
    assert reloaded_messages == messages


async def test_event_stream_text_with_provider_metadata():
    """Test that text events include provider_metadata when TextPart has provider_name and provider_details."""

    async def event_generator():
        part = TextPart(
            content='Hello with details',
            id='text_event_id',
            provider_name='openai',
            provider_details={'model': 'gpt-4', 'tokens': 10},
        )
        yield PartStartEvent(index=0, part=part)
        yield PartEndEvent(index=0, part=part)

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Test')],
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
            {
                'type': 'text-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': 'text_event_id',
                        'provider_name': 'openai',
                        'provider_details': {'model': 'gpt-4', 'tokens': 10},
                    }
                },
            },
            {
                'type': 'text-delta',
                'id': IsStr(),
                'delta': 'Hello with details',
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': 'text_event_id',
                        'provider_name': 'openai',
                        'provider_details': {'model': 'gpt-4', 'tokens': 10},
                    }
                },
            },
            {
                'type': 'text-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': 'text_event_id',
                        'provider_name': 'openai',
                        'provider_details': {'model': 'gpt-4', 'tokens': 10},
                    }
                },
            },
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_event_stream_tool_call_end_with_provider_metadata_v5():
    """Test that tool-input-start events exclude provider_metadata for SDK v5."""

    async def event_generator():
        part = ToolCallPart(
            tool_name='my_tool',
            tool_call_id='tc_meta',
            args={'key': 'value'},
            id='tool_call_id_123',
            provider_name='anthropic',
            provider_details={'tool_index': 0},
        )
        yield PartStartEvent(index=0, part=part)
        yield PartEndEvent(index=0, part=part)

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Test')],
            ),
        ],
    )
    event_stream = VercelAIEventStream(run_input=request, sdk_version=5)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in event_stream.encode_stream(event_stream.transform_stream(event_generator()))
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'tool-input-start', 'toolCallId': 'tc_meta', 'toolName': 'my_tool'},
            {'type': 'tool-input-delta', 'toolCallId': 'tc_meta', 'inputTextDelta': '{"key":"value"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'tc_meta',
                'toolName': 'my_tool',
                'input': {'key': 'value'},
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': 'tool_call_id_123',
                        'provider_name': 'anthropic',
                        'provider_details': {'tool_index': 0},
                    }
                },
            },
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_event_stream_tool_call_end_with_provider_metadata_v6():
    """Test that tool-input-available events include provider_metadata with provider_name for SDK v6."""

    async def event_generator():
        part = ToolCallPart(
            tool_name='my_tool',
            tool_call_id='tc_meta',
            args={'key': 'value'},
            id='tool_call_id_123',
            provider_name='anthropic',
            provider_details={'tool_index': 0},
        )
        yield PartStartEvent(index=0, part=part)
        yield PartEndEvent(index=0, part=part)

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Test')],
            ),
        ],
    )
    event_stream = VercelAIEventStream(run_input=request, sdk_version=6)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in event_stream.encode_stream(event_stream.transform_stream(event_generator()))
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {
                'type': 'tool-input-start',
                'toolCallId': 'tc_meta',
                'toolName': 'my_tool',
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': 'tool_call_id_123',
                        'provider_name': 'anthropic',
                        'provider_details': {'tool_index': 0},
                    }
                },
            },
            {'type': 'tool-input-delta', 'toolCallId': 'tc_meta', 'inputTextDelta': '{"key":"value"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'tc_meta',
                'toolName': 'my_tool',
                'input': {'key': 'value'},
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': 'tool_call_id_123',
                        'provider_name': 'anthropic',
                        'provider_details': {'tool_index': 0},
                    }
                },
            },
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_event_stream_builtin_tool_call_end_with_provider_metadata_v5():
    """Test that builtin tool-input-start events exclude provider_metadata for SDK v5."""

    async def event_generator():
        part = BuiltinToolCallPart(
            tool_name='web_search',
            tool_call_id='btc_meta',
            args={'query': 'test'},
            id='builtin_call_id_456',
            provider_name='openai',
            provider_details={'tool_type': 'web_search_preview'},
        )
        yield PartStartEvent(index=0, part=part)
        yield PartEndEvent(index=0, part=part)

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Search')],
            ),
        ],
    )
    event_stream = VercelAIEventStream(run_input=request, sdk_version=5)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in event_stream.encode_stream(event_stream.transform_stream(event_generator()))
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {'type': 'tool-input-start', 'toolCallId': 'btc_meta', 'toolName': 'web_search', 'providerExecuted': True},
            {'type': 'tool-input-delta', 'toolCallId': 'btc_meta', 'inputTextDelta': '{"query":"test"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'btc_meta',
                'toolName': 'web_search',
                'input': {'query': 'test'},
                'providerExecuted': True,
                'providerMetadata': {
                    'pydantic_ai': {
                        'provider_details': {'tool_type': 'web_search_preview'},
                        'provider_name': 'openai',
                        'id': 'builtin_call_id_456',
                    }
                },
            },
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_event_stream_builtin_tool_call_end_with_provider_metadata_v6():
    """Test that builtin tool-input-available events include provider_name in provider_metadata for SDK v6."""

    async def event_generator():
        part = BuiltinToolCallPart(
            tool_name='web_search',
            tool_call_id='btc_meta',
            args={'query': 'test'},
            id='builtin_call_id_456',
            provider_name='openai',
            provider_details={'tool_type': 'web_search_preview'},
        )
        yield PartStartEvent(index=0, part=part)
        yield PartEndEvent(index=0, part=part)

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Search')],
            ),
        ],
    )
    event_stream = VercelAIEventStream(run_input=request, sdk_version=6)
    events = [
        '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
        async for event in event_stream.encode_stream(event_stream.transform_stream(event_generator()))
    ]

    assert events == snapshot(
        [
            {'type': 'start'},
            {'type': 'start-step'},
            {
                'type': 'tool-input-start',
                'toolCallId': 'btc_meta',
                'toolName': 'web_search',
                'providerExecuted': True,
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': 'builtin_call_id_456',
                        'provider_name': 'openai',
                        'provider_details': {'tool_type': 'web_search_preview'},
                    }
                },
            },
            {'type': 'tool-input-delta', 'toolCallId': 'btc_meta', 'inputTextDelta': '{"query":"test"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'btc_meta',
                'toolName': 'web_search',
                'input': {'query': 'test'},
                'providerExecuted': True,
                'providerMetadata': {
                    'pydantic_ai': {
                        'provider_details': {'tool_type': 'web_search_preview'},
                        'provider_name': 'openai',
                        'id': 'builtin_call_id_456',
                    }
                },
            },
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


async def test_event_stream_thinking_delta_with_provider_metadata():
    """Test that thinking delta events include provider_metadata."""
    from pydantic_ai.messages import ThinkingPartDelta

    async def event_generator():
        part = ThinkingPart(
            content='',
            id='think_delta',
            signature='initial_sig',
            provider_name='anthropic',
            provider_details={'model': 'claude'},
        )
        yield PartStartEvent(index=0, part=part)
        yield PartDeltaEvent(
            index=0,
            delta=ThinkingPartDelta(
                content_delta='thinking...',
                signature_delta='updated_sig',
                provider_name='anthropic',
                provider_details={'chunk': 1},
            ),
        )
        yield PartEndEvent(
            index=0,
            part=ThinkingPart(
                content='thinking...',
                id='think_delta',
                signature='updated_sig',
                provider_name='anthropic',
            ),
        )

    request = SubmitMessage(
        id='foo',
        messages=[
            UIMessage(
                id='bar',
                role='user',
                parts=[TextUIPart(text='Think')],
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
            {
                'type': 'reasoning-start',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {
                        'id': 'think_delta',
                        'signature': 'initial_sig',
                        'provider_name': 'anthropic',
                        'provider_details': {'model': 'claude'},
                    }
                },
            },
            {
                'type': 'reasoning-delta',
                'id': IsStr(),
                'delta': 'thinking...',
                'providerMetadata': {
                    'pydantic_ai': {
                        'provider_name': 'anthropic',
                        'signature': 'updated_sig',
                        'provider_details': {'chunk': 1},
                    }
                },
            },
            {
                'type': 'reasoning-end',
                'id': IsStr(),
                'providerMetadata': {
                    'pydantic_ai': {'id': 'think_delta', 'signature': 'updated_sig', 'provider_name': 'anthropic'}
                },
            },
            {'type': 'finish-step'},
            {'type': 'finish'},
            '[DONE]',
        ]
    )


def _sync_timestamps(original: list[ModelMessage], new: list[ModelMessage]) -> None:
    """Utility function to sync timestamps between original and new messages."""
    for orig_msg, new_msg in zip(original, new):
        for orig_part, new_part in zip(orig_msg.parts, new_msg.parts):
            if hasattr(orig_part, 'timestamp') and hasattr(new_part, 'timestamp'):
                new_part.timestamp = orig_part.timestamp  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        if hasattr(orig_msg, 'timestamp') and hasattr(new_msg, 'timestamp'):  # pragma: no branch
            new_msg.timestamp = orig_msg.timestamp  # pyright: ignore[reportAttributeAccessIssue]


class TestDumpProviderMetadata:
    async def test_dump_provider_metadata_filters_none_values(self):
        """Test that dump_provider_metadata only includes non-None values."""

        # All None - should return None
        result = dump_provider_metadata(id=None, provider_name=None, provider_details=None)
        assert result is None

        # Some values
        result = dump_provider_metadata(id='test_id', provider_name=None, provider_details={'key': 'val'})
        assert result == {'pydantic_ai': {'id': 'test_id', 'provider_details': {'key': 'val'}}}

        # All values
        result = dump_provider_metadata(
            id='full_id',
            signature='sig',
            provider_name='provider',
            provider_details={'detail': 1},
        )
        assert result == {
            'pydantic_ai': {
                'id': 'full_id',
                'signature': 'sig',
                'provider_name': 'provider',
                'provider_details': {'detail': 1},
            }
        }

    async def test_dump_provider_metadata_wrapper_key(self):
        """Test that dump_provider_metadata includes the wrapper key."""

        result = dump_provider_metadata(
            wrapper_key='test', id='test_id', provider_name='test_provider', provider_details={'test_detail': 1}
        )
        assert result == {
            'test': {'id': 'test_id', 'provider_name': 'test_provider', 'provider_details': {'test_detail': 1}}
        }

        # Test with None wrapper key
        result = dump_provider_metadata(
            None, id='test_id', provider_name='test_provider', provider_details={'test_detail': 1}
        )
        assert result == {'id': 'test_id', 'provider_name': 'test_provider', 'provider_details': {'test_detail': 1}}


class TestLoadProviderMetadata:
    async def test_load_provider_metadata_loads_provider_metadata(self):
        """Test that load_provider_metadata loads provider metadata."""

        provider_metadata = {
            'pydantic_ai': {'id': 'test_id', 'provider_name': 'test_provider', 'provider_details': {'test_detail': 1}}
        }
        result = load_provider_metadata(provider_metadata)
        assert result == {'id': 'test_id', 'provider_name': 'test_provider', 'provider_details': {'test_detail': 1}}

    async def test_load_provider_metadata_loads_provider_metadata_incorrect_key(self):
        """Test that load_provider_metadata fails to load provider metadata if the wrapper key is not present."""

        provider_metadata = {'test': {'id': 'test_id'}}
        result = load_provider_metadata(provider_metadata)
        assert result == {}


class TestSdkVersion:
    async def test_tool_input_start_chunk_excludes_provider_metadata_for_v5(self):
        chunk = ToolInputStartChunk(
            tool_call_id='tc_1',
            tool_name='my_tool',
            provider_metadata={'pydantic_ai': {'id': 'test_id', 'provider_name': 'openai'}},
        )
        encoded_v5 = json.loads(chunk.encode(sdk_version=5))
        encoded_v6 = json.loads(chunk.encode(sdk_version=6))

        assert 'providerMetadata' not in encoded_v5
        assert encoded_v5 == snapshot({'type': 'tool-input-start', 'toolCallId': 'tc_1', 'toolName': 'my_tool'})

        assert 'providerMetadata' in encoded_v6
        assert encoded_v6 == snapshot(
            {
                'type': 'tool-input-start',
                'toolCallId': 'tc_1',
                'toolName': 'my_tool',
                'providerMetadata': {'pydantic_ai': {'id': 'test_id', 'provider_name': 'openai'}},
            }
        )

    async def test_event_stream_uses_sdk_version(self):
        async def event_generator():
            part = ToolCallPart(
                tool_name='my_tool',
                tool_call_id='tc_ver',
                args={'key': 'value'},
                id='tool_call_id_ver',
                provider_name='anthropic',
            )
            yield PartStartEvent(index=0, part=part)
            yield PartEndEvent(index=0, part=part)

        request = SubmitMessage(
            id='foo',
            messages=[UIMessage(id='bar', role='user', parts=[TextUIPart(text='Test')])],
        )

        event_stream_v5 = VercelAIEventStream(run_input=request, sdk_version=5)
        events_v5: list[str | dict[str, Any]] = [
            '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
            async for event in event_stream_v5.encode_stream(event_stream_v5.transform_stream(event_generator()))
        ]
        tool_input_start_v5: dict[str, Any] = next(
            e for e in events_v5 if isinstance(e, dict) and e.get('type') == 'tool-input-start'
        )
        assert 'providerMetadata' not in tool_input_start_v5

        event_stream_v6 = VercelAIEventStream(run_input=request, sdk_version=6)
        events_v6: list[str | dict[str, Any]] = [
            '[DONE]' if '[DONE]' in event else json.loads(event.removeprefix('data: '))
            async for event in event_stream_v6.encode_stream(event_stream_v6.transform_stream(event_generator()))
        ]
        tool_input_start_v6: dict[str, Any] = next(
            e for e in events_v6 if isinstance(e, dict) and e.get('type') == 'tool-input-start'
        )
        assert 'providerMetadata' in tool_input_start_v6
