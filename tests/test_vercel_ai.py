from __future__ import annotations

import json

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai._request_types import (
    SubmitMessage,
    TextUIPart,
    ToolOutputAvailablePart,
    UIMessage,
)

from .conftest import IsDatetime, IsStr, try_import

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
                metadata=None,
                parts=[
                    TextUIPart(
                        type='text',
                        text="""Use a tool

    """,
                        state=None,
                        provider_metadata=None,
                    )
                ],
            ),
            UIMessage(
                id='bylfKVeyoR901rax',
                role='assistant',
                metadata=None,
                parts=[
                    TextUIPart(
                        type='text',
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
                        provider_metadata=None,
                    )
                ],
            ),
            UIMessage(
                id='MTdh4Ie641kDuIRh',
                role='user',
                metadata=None,
                parts=[TextUIPart(type='text', text='Give me the ToCs', state=None, provider_metadata=None)],
            ),
            UIMessage(
                id='3XlOBgFwaf7GsS4l',
                role='assistant',
                metadata=None,
                parts=[
                    TextUIPart(
                        type='text',
                        text="I'll get the table of contents for both repositories.",
                        state='streaming',
                        provider_metadata=None,
                    ),
                    ToolOutputAvailablePart(
                        type='tool-get_table_of_contents',
                        tool_call_id='toolu_01XX3rjFfG77h3KCbVHoYJMQ',
                        state='output-available',
                        input={'repo': 'pydantic-ai'},
                        output="[Scrubbed due to 'API Key']",
                        provider_executed=None,
                        call_provider_metadata=None,
                        preliminary=None,
                    ),
                    ToolOutputAvailablePart(
                        type='tool-get_table_of_contents',
                        tool_call_id='toolu_01W2yGpGQcMx7pXV2zZ4sz9g',
                        state='output-available',
                        input={'repo': 'logfire'},
                        output="[Scrubbed due to 'Auth']",
                        provider_executed=None,
                        call_provider_metadata=None,
                        preliminary=None,
                    ),
                    TextUIPart(
                        type='text',
                        text="""Here are the Table of Contents for both repositories:... Both products are designed to work together - Pydantic AI for building AI agents and Logfire for observing and monitoring them in production.""",
                        state='streaming',
                        provider_metadata=None,
                    ),
                ],
            ),
            UIMessage(
                id='QVypsUU4swQ1Loxq',
                role='user',
                metadata=None,
                parts=[
                    TextUIPart(
                        type='text',
                        text='How do I get FastAPI instrumentation to include the HTTP request and response',
                        state=None,
                        provider_metadata=None,
                    )
                ],
            ),
        ],
    )

    adapter = VercelAIAdapter(agent, request=data)
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
            {'type': 'reasoning-end', 'id': IsStr()},
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
            {'type': 'reasoning-end', 'id': IsStr()},
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
            {'type': 'reasoning-end', 'id': IsStr()},
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
            {'type': 'reasoning-end', 'id': IsStr()},
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
            {'type': 'reasoning-end', 'id': IsStr()},
            {'type': 'tool-input-start', 'toolCallId': IsStr(), 'toolName': 'web_search', 'providerExecuted': True},
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"query":null,"type":"search"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': 'ws_00e767404995b9950068e6480ac0888191a7897231e6ca9911',
                'toolName': 'web_search',
                'input': {'query': None, 'type': 'search'},
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
            {'type': 'reasoning-end', 'id': IsStr()},
            {'type': 'tool-input-start', 'toolCallId': IsStr(), 'toolName': 'web_search', 'providerExecuted': True},
            {
                'type': 'tool-input-delta',
                'toolCallId': IsStr(),
                'inputTextDelta': '{"query":null,"type":"search"}',
            },
            {
                'type': 'tool-input-available',
                'toolCallId': 'ws_00e767404995b9950068e6480e11208191834104e1aaab1148',
                'toolName': 'web_search',
                'input': {'query': None, 'type': 'search'},
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
            {'type': 'reasoning-end', 'id': IsStr()},
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
            {'type': 'reasoning-end', 'id': IsStr()},
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
            {'type': 'finish'},
            '[DONE]',
        ]
    )
