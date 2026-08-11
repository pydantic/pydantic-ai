"""Tests for models served by Crusoe Serverless Inference through `OpenAIChatModel`.

Crusoe serves open-weight models from many labs behind one OpenAI-compatible endpoint, so the
interesting behavior is what `CrusoeProvider.model_profile()` resolves per model family and what
Crusoe's serving stack does with a standard Chat Completions request: thinking comes back in the
non-standard `reasoning` field, and `response_format` is implemented with guided decoding for every
model, including families whose own profiles don't claim native structured output support.
"""

from __future__ import annotations as _annotations

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
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.crusoe import CrusoeProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_crusoe_model_simple(allow_model_requests: None, crusoe_api_key: str):
    """Crusoe returns thinking content in the non-standard `reasoning` field.

    `OpenAIChatModel` falls back to `reasoning`/`reasoning_content` when the profile doesn't name a
    field, so the `ThinkingPart` is recovered without `CrusoeProvider` configuring one — which it
    can't, as Crusoe uses `reasoning` for most models but `reasoning_content` for DeepSeek.
    """
    model = OpenAIChatModel('zai/GLM-5.2', provider=CrusoeProvider(api_key=crusoe_api_key))
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
1.  **Analyze the Input:** The user is asking a simple arithmetic question: "What is 2 + 2?"
2.  **Process the Query:**
    *   Identify the operation: Addition (+)
    *   Identify the operands: 2 and 2
    *   Calculate the result: 2 + 2 = 4
3.  **Formulate the Output:** State the answer clearly and concisely. "2 + 2 = 4" or "2 + 2 is 4".
4.  **Final Output Generation:** "2 + 2 = 4."\
""",
                        id='reasoning',
                        provider_name='crusoe',
                    ),
                    TextPart(content='2 + 2 = 4'),
                ],
                usage=RequestUsage(
                    details={'reasoning_tokens': 129}, input_tokens=20, output_reasoning_tokens=129, output_tokens=138
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
    model = OpenAIChatModel('meta-llama/Llama-3.3-70B-Instruct', provider=CrusoeProvider(api_key=crusoe_api_key))
    agent = Agent(model)
    async with agent.run_stream('Count from 1 to 5, comma separated.') as result:
        deltas = [c async for c in result.stream_text(delta=True)]
    assert ''.join(deltas) == snapshot('1, 2, 3, 4, 5')


async def test_crusoe_tool_calling(allow_model_requests: None, crusoe_api_key: str):
    """A tool call round trip, which also sends the model's own thinking back on the second request."""
    model = OpenAIChatModel('zai/GLM-5.2', provider=CrusoeProvider(api_key=crusoe_api_key))
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
                        content='The user wants to know the weather in Paris. I\'ll call the get_weather function with "Paris" as the city parameter.',
                        id='reasoning',
                        provider_name='crusoe',
                    ),
                    ToolCallPart(tool_name='get_weather', args='{"city": "Paris"}', tool_call_id=IsStr()),
                ],
                usage=RequestUsage(
                    details={'reasoning_tokens': 26}, input_tokens=167, output_reasoning_tokens=26, output_tokens=38
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
                        content="The weather in Paris is currently **sunny** with a temperature of **25°C**. It's a beautiful day! ☀️"
                    ),
                ],
                usage=RequestUsage(
                    details={'reasoning_tokens': 20},
                    input_tokens=215,
                    cache_read_tokens=64,
                    output_reasoning_tokens=20,
                    output_tokens=50,
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
    model = OpenAIChatModel('zai/GLM-5.2', provider=CrusoeProvider(api_key=crusoe_api_key))
    agent = Agent(model, output_type=NativeOutput(City))
    result = await agent.run('Where is the Eiffel Tower?')
    assert result.output == snapshot(City(city='Paris', country='France'))
