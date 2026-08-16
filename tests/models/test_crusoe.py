"""Tests for `CrusoeModel`.

Crusoe serves open-weight models from many labs behind one OpenAI-compatible endpoint, so the
interesting behavior is what `CrusoeProvider.model_profile()` resolves per model family and what
Crusoe's serving stack does with a standard Chat Completions request: thinking comes back in the
non-standard `reasoning` field, and `response_format` is implemented with guided decoding for every
model, including families whose own profiles don't claim native structured output support.
"""

from __future__ import annotations as _annotations

import re
from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    ModelRequest,
    ModelResponse,
    NativeOutput,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.crusoe import CrusoeModel
    from pydantic_ai.providers.crusoe import CrusoeProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

_INTERNAL_SERVING_ADDRESSES = re.compile(rb'___prefill_addr_[\d.]+:\d+___decode_addr_[\d.]+:\d+_')


@pytest.fixture(scope='module')
def vcr_config(vcr_config: dict[str, Any]) -> dict[str, Any]:
    """Keep Crusoe's internal serving-cluster addresses out of recorded responses.

    Crusoe builds the completion id out of the prefill and decode pod addresses
    (`chatcmpl-___prefill_addr_10.x.x.x:PORT___decode_addr_10.x.x.x:PORT_<id>`). Nothing here
    depends on the value, so there's no reason to publish their internal topology.
    """

    def scrub_response(response: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = response['body']
        body.update(
            {
                key: _INTERNAL_SERVING_ADDRESSES.sub(b'', value)
                for key, value in body.items()
                if isinstance(value, bytes)
            }
        )
        return response

    return {**vcr_config, 'before_record_response': scrub_response}


async def test_crusoe_model_simple(allow_model_requests: None, crusoe_api_key: str):
    """Crusoe returns thinking content in the non-standard `reasoning` field.

    `OpenAIChatModel` falls back to `reasoning`/`reasoning_content` when the profile doesn't name a
    field, so the `ThinkingPart` is recovered without `CrusoeProvider` configuring one — which it
    can't, as Crusoe uses `reasoning` for most models but `reasoning_content` for DeepSeek.
    """
    model = CrusoeModel('zai/GLM-5.2', provider=CrusoeProvider(api_key=crusoe_api_key))
    agent = Agent(model)
    result = await agent.run('What is 2 + 2?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is 2 + 2?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
1.  **Analyze the Input:** The user is asking "What is 2 + 2?".
2.  **Identify the intent:** This is a basic arithmetic question (addition).
3.  **Perform the calculation:** 2 + 2 = 4.
4.  **Formulate the output:** State the answer clearly and concisely. "2 + 2 = 4" or simply "4".
5.  **Final Output Generation:** "2 + 2 = 4."\
""",
                        id='reasoning',
                        provider_name='crusoe',
                    ),
                    TextPart(content='2 + 2 = 4.'),
                ],
                usage=RequestUsage(
                    details={'reasoning_tokens': 108}, input_tokens=20, output_reasoning_tokens=108, output_tokens=118
                ),
                model_name='zai/GLM-5.2',
                timestamp=IsDatetime(),
                provider_name='crusoe',
                provider_url='https://api.inference.crusoecloud.com/v1',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_crusoe_model_streaming(allow_model_requests: None, crusoe_api_key: str):
    model = CrusoeModel('meta-llama/Llama-3.3-70B-Instruct', provider=CrusoeProvider(api_key=crusoe_api_key))
    agent = Agent(model)
    async with agent.run_stream('Count from 1 to 5, comma separated.') as result:
        deltas = [c async for c in result.stream_text(delta=True)]
    assert ''.join(deltas) == snapshot('1, 2, 3, 4, 5')


async def test_crusoe_tool_calling(allow_model_requests: None, crusoe_api_key: str):
    """A tool call round trip, which also sends the model's own thinking back on the second request."""
    model = CrusoeModel('zai/GLM-5.2', provider=CrusoeProvider(api_key=crusoe_api_key))
    agent = Agent(model)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather in a city."""
        return 'sunny, 25C'

    result = await agent.run('What is the weather in Paris?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the weather in Paris?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants to know the weather in Paris. I\'ll call the get_weather function with "Paris" as the city.',
                        id='reasoning',
                        provider_name='crusoe',
                    ),
                    ToolCallPart(tool_name='get_weather', args='{"city": "Paris"}', tool_call_id=IsStr()),
                ],
                usage=RequestUsage(
                    details={'reasoning_tokens': 25},
                    input_tokens=167,
                    output_reasoning_tokens=25,
                    output_tokens=37,
                ),
                model_name='zai/GLM-5.2',
                timestamp=IsDatetime(),
                provider_name='crusoe',
                provider_url='https://api.inference.crusoecloud.com/v1',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='sunny, 25C',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="The weather in Paris is sunny and 25°C. I'll relay this information to the user.",
                        id='reasoning',
                        provider_name='crusoe',
                    ),
                    TextPart(
                        content="The weather in Paris is currently **sunny** with a temperature of **25°C**. It's a great day to enjoy the city! ☀️"
                    ),
                ],
                usage=RequestUsage(
                    details={'reasoning_tokens': 20},
                    input_tokens=214,
                    cache_read_tokens=64,
                    output_reasoning_tokens=20,
                    output_tokens=54,
                ),
                model_name='zai/GLM-5.2',
                timestamp=IsDatetime(),
                provider_name='crusoe',
                provider_url='https://api.inference.crusoecloud.com/v1',
                provider_details={'finish_reason': 'stop', 'timestamp': IsDatetime()},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


class City(BaseModel):
    city: str
    country: str


async def test_crusoe_native_output(allow_model_requests: None, crusoe_api_key: str):
    """`NativeOutput` works on a model family whose own profile doesn't set `supports_json_schema_output`.

    `zai_model_profile` doesn't claim native structured output support, so this would raise
    `UserError: Native structured output is not supported by this model` if `CrusoeProvider` didn't
    set the flag for every model it serves.
    """
    model = CrusoeModel('zai/GLM-5.2', provider=CrusoeProvider(api_key=crusoe_api_key))
    agent = Agent(model, output_type=NativeOutput(City))
    result = await agent.run('Where is the Eiffel Tower?')
    assert result.output == snapshot(City(city='Paris', country='France'))
