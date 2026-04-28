"""Tests for Google `NativeOutput` combined with function tools (Gemini 3+).

Gemini 3 lifts the historical restriction that prevented combining `response_schema`
(`NativeOutput`) with `function_declarations` in the same request. These tests cover:

1. The error still fires on Gemini 2.5 (and any other model whose profile does not set
   `google_supports_tool_combination`).
2. The combination works on Gemini 3+ for both sync and streaming requests.
"""

from __future__ import annotations as _annotations

import re

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.output import NativeOutput
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


class CityLocation(BaseModel):
    city: str
    country: str


@pytest.fixture
def google_provider(gemini_api_key: str) -> GoogleProvider:
    return GoogleProvider(api_key=gemini_api_key)


async def test_native_output_with_function_tools_unsupported(
    allow_model_requests: None, google_provider: GoogleProvider
):
    """Models without `google_supports_tool_combination` still raise on `NativeOutput` + function tools."""
    m = GoogleModel('gemini-2.5-flash', provider=google_provider)
    agent = Agent(m, output_type=NativeOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'  # pragma: no cover

    with pytest.raises(
        UserError,
        match=re.escape(
            'This model does not support `NativeOutput` and function tools at the same time. '
            'Use `output_type=ToolOutput(...)` instead.'
        ),
    ):
        await agent.run('What is the largest city in the user country?')


async def test_native_output_with_function_tools(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-3-flash-preview', provider=google_provider)
    agent = Agent(m, output_type=NativeOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country?')
    assert isinstance(result.output, CityLocation)
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the largest city in the user country?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args={},
                        tool_call_id='433twugp',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=29, output_tokens=76, details={'thoughts_tokens': 64, 'text_prompt_tokens': 29}
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='1MnFaZLAD6Ky1MkP8Nrx4QQ',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country', content='Mexico', tool_call_id='433twugp', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "city": "Mexico City",
  "country": "Mexico"
} \
""",
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=123, output_tokens=52, details={'thoughts_tokens': 31, 'text_prompt_tokens': 123}
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='1cnFaY76C47TjMcPkM6k0Qg',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_native_output_with_function_tools_stream(allow_model_requests: None, google_provider: GoogleProvider):
    m = GoogleModel('gemini-3-flash-preview', provider=google_provider)
    agent = Agent(m, output_type=NativeOutput(CityLocation))

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    async with agent.run_stream('What is the largest city in the user country?') as result:
        output = await result.get_output()
    assert isinstance(output, CityLocation)
    assert output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the largest city in the user country?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args={},
                        tool_call_id='zeq8pw5c',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=29, output_tokens=127, details={'thoughts_tokens': 115, 'text_prompt_tokens': 29}
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='1snFaaeXKbjD-sAPtam9qQY',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country', content='Mexico', tool_call_id='zeq8pw5c', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "city": "Mexico City",
  "country": "Mexico"
} \
""",
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=174, output_tokens=57, details={'thoughts_tokens': 36, 'text_prompt_tokens': 174}
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='18nFaaz2H5aQjrEP2ruPIQ',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
