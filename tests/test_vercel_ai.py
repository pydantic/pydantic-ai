from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.vercel_ai.request_types import (
    SubmitMessage,
    TextUIPart,
    ToolOutputAvailablePart,
    UIMessage,
)
from pydantic_ai.vercel_ai.response_types import (
    FinishChunk,
    ReasoningDeltaChunk,
    ReasoningStartChunk,
    TextDeltaChunk,
    TextStartChunk,
    ToolInputDeltaChunk,
    ToolInputStartChunk,
    ToolOutputAvailableChunk,
)
from pydantic_ai.vercel_ai.starlette import DoneChunk, StarletteChat

from .conftest import IsStr

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


async def test_run(allow_model_requests: None, openai_api_key: str):
    model = OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(model=model, builtin_tools=[WebSearchTool()])
    chat = StarletteChat(agent)

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

    events = [event async for event in chat.run(data, None)]
    assert events == snapshot(
        [
            ReasoningStartChunk(id='d775971d84c848228275a25a097b6409'),
            ReasoningDeltaChunk(id='d775971d84c848228275a25a097b6409', delta=''),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e647f909248191bfe8d05eeed67645', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e647f909248191bfe8d05eeed67645',
                input_text_delta='{"query":"OpenTelemetry FastAPI instrumentation capture request and response body","type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e647f909248191bfe8d05eeed67645', output={'status': 'completed'}
            ),
            ReasoningStartChunk(id='d775971d84c848228275a25a097b6409'),
            ReasoningDeltaChunk(id='d775971d84c848228275a25a097b6409', delta=''),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e647fb73c48191b0bdb147c3a0d22c', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e647fb73c48191b0bdb147c3a0d22c',
                input_text_delta='{"query":"OTEL_INSTRUMENTATION_HTTP_CAPTURE_BODY Python","type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e647fb73c48191b0bdb147c3a0d22c', output={'status': 'completed'}
            ),
            ReasoningStartChunk(id='d775971d84c848228275a25a097b6409'),
            ReasoningDeltaChunk(id='d775971d84c848228275a25a097b6409', delta=''),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e647fee97c8191919865e0c0a78bba', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e647fee97c8191919865e0c0a78bba',
                input_text_delta='{"query":"OTEL_INSTRUMENTATION_HTTP_CAPTURE_BODY opentelemetry python","type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e647fee97c8191919865e0c0a78bba', output={'status': 'completed'}
            ),
            ReasoningStartChunk(id='d775971d84c848228275a25a097b6409'),
            ReasoningDeltaChunk(id='d775971d84c848228275a25a097b6409', delta=''),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e64803f27c81918a39ce50cb8dfbc2', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e64803f27c81918a39ce50cb8dfbc2',
                input_text_delta='{"query":"site:github.com open-telemetry/opentelemetry-python-contrib OTEL_INSTRUMENTATION_HTTP_CAPTURE_BODY","type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e64803f27c81918a39ce50cb8dfbc2', output={'status': 'completed'}
            ),
            ReasoningStartChunk(id='d775971d84c848228275a25a097b6409'),
            ReasoningDeltaChunk(id='d775971d84c848228275a25a097b6409', delta=''),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e6480ac0888191a7897231e6ca9911', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e6480ac0888191a7897231e6ca9911',
                input_text_delta='{"query":null,"type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e6480ac0888191a7897231e6ca9911', output={'status': 'completed'}
            ),
            ReasoningStartChunk(id='d775971d84c848228275a25a097b6409'),
            ReasoningDeltaChunk(id='d775971d84c848228275a25a097b6409', delta=''),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e6480e11208191834104e1aaab1148', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e6480e11208191834104e1aaab1148',
                input_text_delta='{"query":null,"type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e6480e11208191834104e1aaab1148', output={'status': 'completed'}
            ),
            ReasoningStartChunk(id='d775971d84c848228275a25a097b6409'),
            ReasoningDeltaChunk(id='d775971d84c848228275a25a097b6409', delta=''),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e648118bf88191aa7f804637c45b32', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e648118bf88191aa7f804637c45b32',
                input_text_delta='{"query":"OTEL_PYTHON_LOG_CORRELATION environment variable","type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e648118bf88191aa7f804637c45b32', output={'status': 'completed'}
            ),
            ReasoningStartChunk(id='d775971d84c848228275a25a097b6409'),
            ReasoningDeltaChunk(id='d775971d84c848228275a25a097b6409', delta=''),
            TextStartChunk(id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
Short answer:
- Default\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' FastAPI/OpenTelemetry', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' instrumentation already records method', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='/route/status', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
.
- To also\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' include HTTP headers', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=', set', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' the capture-', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='headers env', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
 vars.
-\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' To include request', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='/response bodies', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=', use the', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' FastAPI', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='/ASGI', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' request/response', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' hooks and add', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' the', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' payload to', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' the span yourself', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' (with red', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='action/size', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
 limits).

How\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' to do it', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\


1)\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' Enable header capture', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' (server side', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
)
- Choose\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' just the', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' headers you need; avoid', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' sensitive ones or sanitize', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
 them.

export OTEL\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='_INSTRUMENTATION_HTTP_CAPTURE', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='_HEADERS_SERVER_REQUEST="content', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='-type,user', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='-agent"\n', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='export OTEL_INSTRUMENTATION', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='_HTTP_CAPTURE_HEADERS', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='_SERVER_RESPONSE="content-type"\n', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='export OTEL_INSTRUMENTATION_HTTP', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
_CAPTURE_HEADERS_SANITIZE_FIELDS="authorization,set-cookie"

This makes headers appear on spans as http.request.header.* and http.response.header.*. ([opentelemetry-python-contrib.readthedocs.io](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html))

2)\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' Add hooks to capture request', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='/response bodies', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\

Note:\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=IsStr(), id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' a built-in Python', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' env', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' var to', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' auto-capture', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' HTTP bodies for Fast', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='API/AS', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='GI. Use', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' hooks to look at', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' ASGI receive', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='/send events and', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' attach (tr', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='uncated) bodies', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' as span attributes', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
.

from\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' fastapi import', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' FastAPI', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\

from opente\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='lemetry.trace', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' import Span', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\

from opente\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='lemetry.instrument', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='ation.fastapi import', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' FastAPIInstrument', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
or

MAX\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='_BYTES = ', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='2048 ', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' # keep this', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' small in prod', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\


def client\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='_request_hook(span', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=': Span,', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' scope: dict', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=', message:', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
 dict):
   \
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' if span and', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' span.is_record', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='ing() and', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' message.get("', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='type") ==', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' "http.request', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
":
        body\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' = message.get', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='("body")', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' or b"', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
"
        if\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(
                delta="""\
 body:
           \
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' span.set_attribute', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
(
                "\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='http.request.body', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
",
                body\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='[:MAX_BYTES', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='].decode("', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='utf-8', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='", "replace', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
"),
            )
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(
                delta="""\

def client_response\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='_hook(span:', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' Span, scope', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=': dict,', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' message: dict', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
):
    if\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' span and span', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='.is_recording', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='() and message', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='.get("type', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='") == "', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='http.response.body', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
":
        body\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' = message.get', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='("body")', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' or b"', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
"
        if\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(
                delta="""\
 body:
           \
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' span.set_attribute', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
(
                "\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='http.response.body', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
",
                body\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='[:MAX_BYTES', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='].decode("', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='utf-8', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='", "replace', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
"),
            )
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(
                delta="""\

app = Fast\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(
                delta="""\
API()
Fast\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='APIInstrumentor', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='.instrument_app(', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\

    app,\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(
                delta="""\

    client_request\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='_hook=client', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
_request_hook,
   \
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' client_response_hook', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='=client_response', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
_hook,
)
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(
                delta="""\

- The hooks\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' receive the AS', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='GI event dict', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='s: http', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='.request (with', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' body/more', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='_body) and', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' http.response.body', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='. If your', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' bodies can be', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' chunked,', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' you may need', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' to accumulate across', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' calls when message', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='.get("more', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='_body") is', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' True. ', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta='([opentelemetry-python-contrib.readthedocs.io](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html)',
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=')', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\


3)\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' Be careful with', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' PII and', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
 size
-\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' Always limit size', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' and consider redaction', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' before putting payloads', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
 on spans.
-\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' Use the sanitize', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' env var above', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' for sensitive headers', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='. ', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta='([opentelemetry-python-contrib.readthedocs.io](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html))\n',
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(
                delta="""\

Optional: correlate logs\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(
                delta="""\
 with traces
-\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' If you also want', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' request/response', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' details in logs with', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' trace IDs, enable', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' Python log correlation:\n', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\

export OTEL_P\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='YTHON_LOG_COR', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='RELATION=true', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\


or programmatically\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(
                delta="""\
:
from opente\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='lemetry.instrumentation', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='.logging import LoggingInstrument', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\
or
LoggingInstrument\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='or().instrument(set', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta='_logging_format=True)\n', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta="""\

This injects trace\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta='_id/span_id into', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' log records so you', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' can line up logs', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' with the span that', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' carries the HTTP payload', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' attributes. ', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(
                delta='([opentelemetry-python-contrib.readthedocs.io](https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/logging/logging.html?utm_source=openai))\n',
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(
                delta="""\

Want me to tailor\
""",
                id='d775971d84c848228275a25a097b6409',
            ),
            TextDeltaChunk(delta=' the hook to only', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' capture JSON bodies,', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' skip binary content,', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' or accumulate chunked', id='d775971d84c848228275a25a097b6409'),
            TextDeltaChunk(delta=' bodies safely?', id='d775971d84c848228275a25a097b6409'),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e647f909248191bfe8d05eeed67645', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e647f909248191bfe8d05eeed67645',
                input_text_delta='{"query":"OpenTelemetry FastAPI instrumentation capture request and response body","type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e647f909248191bfe8d05eeed67645', output={'status': 'completed'}
            ),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e647fb73c48191b0bdb147c3a0d22c', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e647fb73c48191b0bdb147c3a0d22c',
                input_text_delta='{"query":"OTEL_INSTRUMENTATION_HTTP_CAPTURE_BODY Python","type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e647fb73c48191b0bdb147c3a0d22c', output={'status': 'completed'}
            ),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e647fee97c8191919865e0c0a78bba', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e647fee97c8191919865e0c0a78bba',
                input_text_delta='{"query":"OTEL_INSTRUMENTATION_HTTP_CAPTURE_BODY opentelemetry python","type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e647fee97c8191919865e0c0a78bba', output={'status': 'completed'}
            ),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e64803f27c81918a39ce50cb8dfbc2', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e64803f27c81918a39ce50cb8dfbc2',
                input_text_delta='{"query":"site:github.com open-telemetry/opentelemetry-python-contrib OTEL_INSTRUMENTATION_HTTP_CAPTURE_BODY","type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e64803f27c81918a39ce50cb8dfbc2', output={'status': 'completed'}
            ),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e6480ac0888191a7897231e6ca9911', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e6480ac0888191a7897231e6ca9911',
                input_text_delta='{"query":null,"type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e6480ac0888191a7897231e6ca9911', output={'status': 'completed'}
            ),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e6480e11208191834104e1aaab1148', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e6480e11208191834104e1aaab1148',
                input_text_delta='{"query":null,"type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e6480e11208191834104e1aaab1148', output={'status': 'completed'}
            ),
            ToolInputStartChunk(
                tool_call_id='ws_00e767404995b9950068e648118bf88191aa7f804637c45b32', tool_name='web_search'
            ),
            ToolInputDeltaChunk(
                tool_call_id='ws_00e767404995b9950068e648118bf88191aa7f804637c45b32',
                input_text_delta='{"query":"OTEL_PYTHON_LOG_CORRELATION environment variable","type":"search"}',
            ),
            ToolOutputAvailableChunk(
                tool_call_id='ws_00e767404995b9950068e648118bf88191aa7f804637c45b32', output={'status': 'completed'}
            ),
            FinishChunk(),
            DoneChunk(),
        ]
    )
