"""Tests for Google structured output + tool combination (Gemini 3).

Tests the three restriction lifts for Gemini 3+:
1. NativeOutput + function tools (response_schema + function_declarations)
2. Function tools + builtin tools (function_declarations + builtin_tools)
3. Output tools + builtin tools (ToolOutput function_declarations + builtin_tools)

Also verifies that older models still raise appropriate errors.
"""

from __future__ import annotations as _annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.output import NativeOutput, ToolOutput
from pydantic_ai.usage import RequestUsage

from ..._inline_snapshot import snapshot
from ...conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel

if TYPE_CHECKING:
    GoogleModelFactory = Callable[..., GoogleModel]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolCallEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead.:DeprecationWarning'
    ),
    pytest.mark.filterwarnings(
        'ignore:`BuiltinToolResultEvent` is deprecated, look for `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead.:DeprecationWarning'
    ),
]


class CityLocation(BaseModel):
    city: str
    country: str


# =============================================================================
# Error tests — older models still block unsupported combinations
# =============================================================================


async def test_native_output_with_function_tools_unsupported(
    allow_model_requests: None, google_model: GoogleModelFactory
):
    m = google_model('gemini-2.5-flash')
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


async def test_function_tools_with_builtin_tools_unsupported(
    allow_model_requests: None, google_model: GoogleModelFactory
):
    m = google_model('gemini-2.5-flash')
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'  # pragma: no cover

    with pytest.raises(
        UserError,
        match=re.escape('This model does not support function tools and built-in tools at the same time.'),
    ):
        await agent.run('What is the largest city in the user country?')


async def test_tool_output_with_builtin_tools_unsupported(allow_model_requests: None, google_model: GoogleModelFactory):
    m = google_model('gemini-2.5-flash')
    agent = Agent(m, output_type=ToolOutput(CityLocation), builtin_tools=[WebSearchTool()])

    with pytest.raises(
        UserError,
        match=re.escape(
            'This model does not support output tools and built-in tools at the same time. '
            'Use `output_type=PromptedOutput(...)` instead.'
        ),
    ):
        await agent.run('What is the largest city in Mexico?')


# =============================================================================
# VCR integration tests — Gemini 3 supports all combinations
# =============================================================================


async def test_native_output_with_function_tools(allow_model_requests: None, google_model: GoogleModelFactory):
    m = google_model('gemini-3-flash-preview')
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


async def test_native_output_with_function_tools_stream(allow_model_requests: None, google_model: GoogleModelFactory):
    m = google_model('gemini-3-flash-preview')
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


async def test_native_output_with_builtin_tools_stream(allow_model_requests: None, google_model: GoogleModelFactory):
    m = google_model('gemini-3-flash-preview')
    agent = Agent(m, output_type=NativeOutput(CityLocation), builtin_tools=[WebSearchTool()])

    async with agent.run_stream('What is the largest city in Mexico?') as result:
        output = await result.get_output()
    assert isinstance(output, CityLocation)
    assert output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the largest city in Mexico?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['largest city in Mexico']},
                        tool_call_id='fxu90zw1',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={
                            'search_suggestions': """\
<style>
.container {
  align-items: center;
  border-radius: 8px;
  display: flex;
  font-family: Google Sans, Roboto, sans-serif;
  font-size: 14px;
  line-height: 20px;
  padding: 8px 12px;
}
.chip {
  display: inline-block;
  border: solid 1px;
  border-radius: 16px;
  min-width: 14px;
  padding: 5px 16px;
  text-align: center;
  user-select: none;
  margin: 0 8px;
  -webkit-tap-highlight-color: transparent;
}
.carousel {
  overflow: auto;
  scrollbar-width: none;
  white-space: nowrap;
  margin-right: -12px;
}
.headline {
  display: flex;
  margin-right: 4px;
}
.gradient-container {
  position: relative;
}
.gradient {
  position: absolute;
  transform: translate(3px, -9px);
  height: 36px;
  width: 9px;
}
@media (prefers-color-scheme: light) {
  .container {
    background-color: #fafafa;
    box-shadow: 0 0 0 1px #0000000f;
  }
  .headline-label {
    color: #1f1f1f;
  }
  .chip {
    background-color: #ffffff;
    border-color: #d2d2d2;
    color: #5e5e5e;
    text-decoration: none;
  }
  .chip:hover {
    background-color: #f2f2f2;
  }
  .chip:focus {
    background-color: #f2f2f2;
  }
  .chip:active {
    background-color: #d8d8d8;
    border-color: #b6b6b6;
  }
  .logo-dark {
    display: none;
  }
  .gradient {
    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);
  }
}
@media (prefers-color-scheme: dark) {
  .container {
    background-color: #1f1f1f;
    box-shadow: 0 0 0 1px #ffffff26;
  }
  .headline-label {
    color: #fff;
  }
  .chip {
    background-color: #2c2c2c;
    border-color: #3c4043;
    color: #fff;
    text-decoration: none;
  }
  .chip:hover {
    background-color: #353536;
  }
  .chip:focus {
    background-color: #353536;
  }
  .chip:active {
    background-color: #464849;
    border-color: #53575b;
  }
  .logo-light {
    display: none;
  }
  .gradient {
    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);
  }
}
</style>
<div class="container">
  <div class="headline">
    <svg class="logo-light" width="18" height="18" viewBox="9 9 35 35" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path fill-rule="evenodd" clip-rule="evenodd" d="M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z" fill="#4285F4"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z" fill="#34A853"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z" fill="#FBBC05"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z" fill="#EA4335"/>
    </svg>
    <svg class="logo-dark" width="18" height="18" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="23" fill="#FFF" r="22"/>
      <path d="M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z" fill="#4285F4"/>
      <path d="M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z" fill="#34A853"/>
      <path d="M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z" fill="#FBBC05"/>
      <path d="M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z" fill="#EA4335"/>
    </svg>
    <div class="gradient-container"><div class="gradient"></div></div>
  </div>
  <div class="carousel">
    <a class="chip" href="https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG2p8CnTMneYx3H5_qpOjNs_2yzsEzG9uAuEoGE4axZrzihin2N8r8bp8V2GbIhfc6xsSiwfxSzloBwUR3poJeegDETRZs5m8f4Y9rtwqSm1FC8brQjZhAXtVkULpwTrMKfvTpwASCdRMC_JfN6CrVJMHoIB9X_-Fv7frZX23dvnbtXonFT-ZYQOPgbeDf0QV0OdCXqE7AvgDU=">largest city in Mexico</a>
  </div>
</div>
"""
                        },
                        tool_call_id='fxu90zw1',
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    TextPart(
                        content='{"city": "Mexico City", "country": "Mexico"} ',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=146, output_tokens=43, details={'thoughts_tokens': 29, 'text_prompt_tokens': 146}
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='0ZvNaav1B9TRjMcPtuLhiQw',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_function_tools_with_builtin_tools(allow_model_requests: None, google_model: GoogleModelFactory):
    m = google_model('gemini-3-flash-preview')
    agent = Agent(m, builtin_tools=[WebSearchTool()])

    @agent.tool_plain
    async def calculator(expression: str) -> str:
        return str(eval(expression))  # pragma: no cover

    result = await agent.run('What is 2+2? Also search for the current weather in Tokyo.')
    assert isinstance(result.output, str)
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is 2+2? Also search for the current weather in Tokyo.', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['', 'current weather in Tokyo']},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'domain': None,
                                'title': 'theweathernetwork.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEnkhgS9dHw7cQ5gIl8G8VtymyxgZYOgxedzQj8YBAzSjbcLbZPY-CjTN3qL8mM1H_iaZo56rw1V-pRT1Lw_9ZxXwhgBssj4H3dUjnEohVZi4rdYFFoY16APOV6PRqYfTjVYARmJ-OuTLWNXB-KhL8iQa77TUo6977ksRI=',
                            },
                            {
                                'domain': None,
                                'title': 'google.com',
                                'uri': 'https://www.google.com/search?q=weather+in+Tokyo,+JP',
                            },
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['current weather in Tokyo']},
                        tool_call_id='xds2qnru',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={
                            'search_suggestions': """\
<style>
.container {
  align-items: center;
  border-radius: 8px;
  display: flex;
  font-family: Google Sans, Roboto, sans-serif;
  font-size: 14px;
  line-height: 20px;
  padding: 8px 12px;
}
.chip {
  display: inline-block;
  border: solid 1px;
  border-radius: 16px;
  min-width: 14px;
  padding: 5px 16px;
  text-align: center;
  user-select: none;
  margin: 0 8px;
  -webkit-tap-highlight-color: transparent;
}
.carousel {
  overflow: auto;
  scrollbar-width: none;
  white-space: nowrap;
  margin-right: -12px;
}
.headline {
  display: flex;
  margin-right: 4px;
}
.gradient-container {
  position: relative;
}
.gradient {
  position: absolute;
  transform: translate(3px, -9px);
  height: 36px;
  width: 9px;
}
@media (prefers-color-scheme: light) {
  .container {
    background-color: #fafafa;
    box-shadow: 0 0 0 1px #0000000f;
  }
  .headline-label {
    color: #1f1f1f;
  }
  .chip {
    background-color: #ffffff;
    border-color: #d2d2d2;
    color: #5e5e5e;
    text-decoration: none;
  }
  .chip:hover {
    background-color: #f2f2f2;
  }
  .chip:focus {
    background-color: #f2f2f2;
  }
  .chip:active {
    background-color: #d8d8d8;
    border-color: #b6b6b6;
  }
  .logo-dark {
    display: none;
  }
  .gradient {
    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);
  }
}
@media (prefers-color-scheme: dark) {
  .container {
    background-color: #1f1f1f;
    box-shadow: 0 0 0 1px #ffffff26;
  }
  .headline-label {
    color: #fff;
  }
  .chip {
    background-color: #2c2c2c;
    border-color: #3c4043;
    color: #fff;
    text-decoration: none;
  }
  .chip:hover {
    background-color: #353536;
  }
  .chip:focus {
    background-color: #353536;
  }
  .chip:active {
    background-color: #464849;
    border-color: #53575b;
  }
  .logo-light {
    display: none;
  }
  .gradient {
    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);
  }
}
</style>
<div class="container">
  <div class="headline">
    <svg class="logo-light" width="18" height="18" viewBox="9 9 35 35" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path fill-rule="evenodd" clip-rule="evenodd" d="M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z" fill="#4285F4"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z" fill="#34A853"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z" fill="#FBBC05"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z" fill="#EA4335"/>
    </svg>
    <svg class="logo-dark" width="18" height="18" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="23" fill="#FFF" r="22"/>
      <path d="M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z" fill="#4285F4"/>
      <path d="M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z" fill="#34A853"/>
      <path d="M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z" fill="#FBBC05"/>
      <path d="M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z" fill="#EA4335"/>
    </svg>
    <div class="gradient-container"><div class="gradient"></div></div>
  </div>
  <div class="carousel">
    <a class="chip" href="https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEGXBokq8opls5ju3iEjlGcCIOFHG5zxuYkoGO_eOsPN1P0Fkz6w057ajxM-7H87VXsCM7XHKG7HPIkPIOfGdj4icvnJ4njtOwoEqi1kNu5RClx4dbd91Q2VRhQ2G4LvjFCoJdFX-94vzivfuv4ihpUveB3hcgZZJzAzt-NuFsUK5ro3ydIzoQimGiBkPNb-9_4chhMY4c1pAgS8w==">current weather in Tokyo</a>
  </div>
</div>
"""
                        },
                        tool_call_id='xds2qnru',
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    TextPart(
                        content="""\
First, **2 + 2 = 4**.

As for the weather in Tokyo, as of Friday, March 27, 2026, it is currently **sunny** with a temperature of **56°F (13°C)**. \n\

The humidity is around 79%, and the forecast for the rest of the day includes periodic rain with a high of 64°F (18°C) and a 60% chance of precipitation.\
""",
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=152, output_tokens=200, details={'thoughts_tokens': 97, 'text_prompt_tokens': 152}
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='28nFaaOaJZKJ1MkPzvDemA0',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_native_output_with_function_and_builtin_tools(
    allow_model_requests: None, google_model: GoogleModelFactory
):
    m = google_model('gemini-3-flash-preview')
    agent = Agent(m, output_type=NativeOutput(CityLocation), builtin_tools=[WebSearchTool()])

    @agent.tool_plain
    async def get_user_country() -> str:
        return 'Mexico'

    result = await agent.run('What is the largest city in the user country? Search the web to confirm.')
    assert isinstance(result.output, CityLocation)
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the largest city in the user country? Search the web to confirm.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_user_country',
                        args={},
                        tool_call_id='6ebuleqr',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=35, output_tokens=72, details={'thoughts_tokens': 60, 'text_prompt_tokens': 35}
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='3MnFad3tOqqqsOIPw7W6oQI',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country', content='Mexico', tool_call_id='6ebuleqr', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={
                            'queries': [
                                '',
                                'largest city in Mexico by population 2026',
                                'population of Mexico City vs other cities in Mexico 2026',
                            ]
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'domain': None,
                                'title': 'economictimes.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFkEWDxvuNxmOuIPk8RDNni40iD_ReP01auJVUWYVnVJSMPpnOM4bpBF6VSbpNugUSSg91JcXZO9IZkg-rMGN_UtTeYeh7BpQiSyLzdwGG43t3C3N3CX_RAJ181NmPApt3MtNg7XFn9SKe3t1J1x29-rk2rvNQxCetg5kjajAVCfpcyJjuDBmRysfZR_xorpPs5OeulZHdSsXy4UY3LXwQJ0r5a7CcJ4JmV2s0QKs4nGK4xnlML4H68EZRRRgycnF7oALfzTmBBGUjf6XBSy86w6ja1q5tebQeKtt1cuWbeCIiHZWg7gQT5HA==',
                            },
                            {
                                'domain': None,
                                'title': 'worldpopulationreview.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE_xEtkMyIQrKJ9r2U3YNfUytdSSNC_gplgPvI1L1WDbR7FXjI64k719XU7Taj-bstfimb71cCUe4qjd30hxz-REdx1wgt59YyGTYfwitq2pPCfGnXkZPtfb0zx4Ma4Y3gDEUiuev4YzCbEaVn3BfzqDAzVKj7F',
                            },
                            {
                                'domain': None,
                                'title': 'chislennost.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGDl3qZhcSRJfSjs5MXnj3vp0t0LMnAW66pZEgt65d-7Cs-QlfxyAHxtdn7vhybgkOzhdAJ3GlrKxs8P31JvYZ2_brvnqcRVxzZ2qiVJ-PCVi0TgtesKlaEldgUybDltvQ7JXJPGJ8ZBO6FCJHEozrBUSRsT8a2rbms3EreW3gRUX5VPj0=',
                            },
                            {
                                'domain': None,
                                'title': 'worldometers.info',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGc2Xi0yzfZZXu5w_01vhz5ZWmyxYOt9QH8e9GL_fl8eVjLUofkqwm8IIQ47Zrv-fzzGEzXEH8yWJnTXchru1wlFZStmfgie8HIj8huzP8cfWbYkjHqx-H-rWG0DDKv9rJNTvwMvJmxbOZeW6SczLp-Vp5t1yDiuDmy0CPn',
                            },
                            {
                                'domain': None,
                                'title': 'macrotrends.net',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGO1NmCwLQa-0c6k0xxLKsFraHnIf3tfduZwUg02PKVXeF4HOuLP1nVc1Qux50KZrE3Z6OBjXq-I42wzg5jdl2REYmoo68pDDEio9byY_9De_-_PDwzGnCyEEIAkDoFD-Vvy5LaZxrvPmYcfbdkvfN1wsQ9X2yozD7o3dZDUl10rOdKPC2YpsEe5g==',
                            },
                            {
                                'domain': None,
                                'title': 'statisticstimes.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFBhlW456B7oa2wHNGZEO1FM-V0rTCVGNly0M5odAdoMK_dijaoHqP_hJT5ZPjHPM12oiFIoZX0hjILWl3tbxgYAoC-A__htIeOMYdVP-lqnCLfq2smD4nig6Jug3tfMtSWgbp6xpQ9CiT3609aLVSC4JVUdsrtHh2kCqTmmfjKbBG3Q9eANvhf',
                            },
                            {
                                'domain': None,
                                'title': 'worldometers.info',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFCqrrAgIPMtDLChV4JjBqjfcYD023xn9ITH8d1dL_-FvBOfLkmEVek2fmBjzw_dWn29kFxY4qpTrTzPyfY4RRO-4OXiDDFiJJCwFYQC4FUmSA3MNqfcqlqg0ajKiLZRZ4InSMp4j1S2vofYwjUVxPfD7wJc2S7wXvsDg==',
                            },
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={
                            'queries': [
                                'largest city in Mexico by population 2026',
                                'population of Mexico City vs other cities in Mexico 2026',
                            ]
                        },
                        tool_call_id='bt412336',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={
                            'search_suggestions': """\
<style>
.container {
  align-items: center;
  border-radius: 8px;
  display: flex;
  font-family: Google Sans, Roboto, sans-serif;
  font-size: 14px;
  line-height: 20px;
  padding: 8px 12px;
}
.chip {
  display: inline-block;
  border: solid 1px;
  border-radius: 16px;
  min-width: 14px;
  padding: 5px 16px;
  text-align: center;
  user-select: none;
  margin: 0 8px;
  -webkit-tap-highlight-color: transparent;
}
.carousel {
  overflow: auto;
  scrollbar-width: none;
  white-space: nowrap;
  margin-right: -12px;
}
.headline {
  display: flex;
  margin-right: 4px;
}
.gradient-container {
  position: relative;
}
.gradient {
  position: absolute;
  transform: translate(3px, -9px);
  height: 36px;
  width: 9px;
}
@media (prefers-color-scheme: light) {
  .container {
    background-color: #fafafa;
    box-shadow: 0 0 0 1px #0000000f;
  }
  .headline-label {
    color: #1f1f1f;
  }
  .chip {
    background-color: #ffffff;
    border-color: #d2d2d2;
    color: #5e5e5e;
    text-decoration: none;
  }
  .chip:hover {
    background-color: #f2f2f2;
  }
  .chip:focus {
    background-color: #f2f2f2;
  }
  .chip:active {
    background-color: #d8d8d8;
    border-color: #b6b6b6;
  }
  .logo-dark {
    display: none;
  }
  .gradient {
    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);
  }
}
@media (prefers-color-scheme: dark) {
  .container {
    background-color: #1f1f1f;
    box-shadow: 0 0 0 1px #ffffff26;
  }
  .headline-label {
    color: #fff;
  }
  .chip {
    background-color: #2c2c2c;
    border-color: #3c4043;
    color: #fff;
    text-decoration: none;
  }
  .chip:hover {
    background-color: #353536;
  }
  .chip:focus {
    background-color: #353536;
  }
  .chip:active {
    background-color: #464849;
    border-color: #53575b;
  }
  .logo-light {
    display: none;
  }
  .gradient {
    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);
  }
}
</style>
<div class="container">
  <div class="headline">
    <svg class="logo-light" width="18" height="18" viewBox="9 9 35 35" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path fill-rule="evenodd" clip-rule="evenodd" d="M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z" fill="#4285F4"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z" fill="#34A853"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z" fill="#FBBC05"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z" fill="#EA4335"/>
    </svg>
    <svg class="logo-dark" width="18" height="18" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="23" fill="#FFF" r="22"/>
      <path d="M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z" fill="#4285F4"/>
      <path d="M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z" fill="#34A853"/>
      <path d="M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z" fill="#FBBC05"/>
      <path d="M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z" fill="#EA4335"/>
    </svg>
    <div class="gradient-container"><div class="gradient"></div></div>
  </div>
  <div class="carousel">
    <a class="chip" href="https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEQ8x_ifgAPgTfrkLZOY7iUTHTgLtB_P_givTQoiXGQiJ7bShRetqhXZfARNh4wuQG3PdW6N6F3gLm9ukLtTFZNJmaHDEHSGI45c5JXmODv-64bF1GQbrQQgeiE3mo5kDLG4Jjrd9a8mM4MMG-FvNOiE8tUj_jg3azSbvxzOZpuRvm9Un37rTmmtooSEpPE7nQQOR0KFNDj13ZspqoAFU6fbl-NITPhpG04s1aXzyO9SbDy4Zn-qb_QkyMVyQ==">population of Mexico City vs other cities in Mexico 2026</a>
    <a class="chip" href="https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHKTEqM-TMBMiSlTurakVPL4zsLA_Obf7PgCZtz9ip3gN3GPrkZVOpCbqXa1-op1Wrcl7uzkBLz8uw0u7c2m92S32pJUJjFt0s--nqKtJUqGayAjJjmBsByYY1Mszyx0z4bsPp-PuQte8NCWBxEDb63b3vuDY31o44Stk0yOqEYuyp_JT_Urll0jS_FSXDUPVX33QexUyZPvZMKvK35cC3Pg4hxFMg-q4pH7JpR">largest city in Mexico by population 2026</a>
  </div>
</div>
"""
                        },
                        tool_call_id='bt412336',
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    TextPart(
                        content="""\
{
  "city": "Mexico City",
  "country": "Mexico"
} \
""",
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=125,
                    output_tokens=284,
                    details={'thoughts_tokens': 175, 'tool_use_prompt_tokens': 88, 'text_prompt_tokens': 125},
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='4snFaeOuCf6b1MkPmf-vyQY',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_native_output_with_builtin_tools(allow_model_requests: None, google_model: GoogleModelFactory):
    m = google_model('gemini-3-flash-preview')
    agent = Agent(m, output_type=NativeOutput(CityLocation), builtin_tools=[WebSearchTool()])

    result = await agent.run('What is the largest city in Mexico?')
    assert isinstance(result.output, CityLocation)
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the largest city in Mexico?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['largest city in Mexico']},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=[
                            {
                                'domain': None,
                                'title': 'sa.gov',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGg97J9aTlqhFd2a0LN4W8tdaPYU32dpl6I-6HUzZx7Zp_aAOTXz6DaUtxWsnA_fJL3bpN4ywfpPrgdjFl9tyU0RH8t5QrR02DyQpO3AzU38ChE-MKpdwM_ZDowoz7NOiXQ56fYA0IYnv1MasIq_1pj4ilwjD74iOcEc4vq4imsvLsqbpp1tMeHnFRz18NpOQSTmNQ9YaiLkBaiBqY0QZr_Z9h6WerAR9c5H90ixDoT3Ds=',
                            },
                            {
                                'domain': None,
                                'title': 'chicagosistercities.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGd2HZezLhKHiISvr8SCWjI71O5DBWM8aK68o_XDohAtCk9mA4vx5BxUSV7BF20-lGztBnl8KdhB6_g7ZqCNY6vrKxxHIrUuDwJHNlzD9pErVlWHlkXH237HlDroHhhBWQst9hOQGWYKpklH8QObeCAWtNE2FHSzVrirTwz3bE=',
                            },
                            {
                                'domain': None,
                                'title': 'worldatlas.com',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQETiMFJxIzb4UmbPxiZMzWUyYlzMpENHv1UUTIb2x7Ils-79uDX4wrxxfykAzecK1hfg725C2ws6RbpQ2qGyxuRug8biZvWUapdIAd9Wzx4irMROUnr_FiQmXj8JvV1_8dwPNL5FOyfI5lvPilNyHCiHtux0IofSnG3skOCZkvp',
                            },
                            {
                                'domain': None,
                                'title': 'wikipedia.org',
                                'uri': 'https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5GGhbtuuevqv6EjMoc7ro02Ih0iEpStmwKBso9zxHboT39uR0nTtn7xAeZy-ttFiEVszJIcuB8_o8elAgSw_SKZ-_fTAJfmnKJ94mLl-EJec3Kl-H5AQJjdk6sxeshAla0lE=',
                            },
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['largest city in Mexico']},
                        tool_call_id='q57vznsj',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={
                            'search_suggestions': """\
<style>
.container {
  align-items: center;
  border-radius: 8px;
  display: flex;
  font-family: Google Sans, Roboto, sans-serif;
  font-size: 14px;
  line-height: 20px;
  padding: 8px 12px;
}
.chip {
  display: inline-block;
  border: solid 1px;
  border-radius: 16px;
  min-width: 14px;
  padding: 5px 16px;
  text-align: center;
  user-select: none;
  margin: 0 8px;
  -webkit-tap-highlight-color: transparent;
}
.carousel {
  overflow: auto;
  scrollbar-width: none;
  white-space: nowrap;
  margin-right: -12px;
}
.headline {
  display: flex;
  margin-right: 4px;
}
.gradient-container {
  position: relative;
}
.gradient {
  position: absolute;
  transform: translate(3px, -9px);
  height: 36px;
  width: 9px;
}
@media (prefers-color-scheme: light) {
  .container {
    background-color: #fafafa;
    box-shadow: 0 0 0 1px #0000000f;
  }
  .headline-label {
    color: #1f1f1f;
  }
  .chip {
    background-color: #ffffff;
    border-color: #d2d2d2;
    color: #5e5e5e;
    text-decoration: none;
  }
  .chip:hover {
    background-color: #f2f2f2;
  }
  .chip:focus {
    background-color: #f2f2f2;
  }
  .chip:active {
    background-color: #d8d8d8;
    border-color: #b6b6b6;
  }
  .logo-dark {
    display: none;
  }
  .gradient {
    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);
  }
}
@media (prefers-color-scheme: dark) {
  .container {
    background-color: #1f1f1f;
    box-shadow: 0 0 0 1px #ffffff26;
  }
  .headline-label {
    color: #fff;
  }
  .chip {
    background-color: #2c2c2c;
    border-color: #3c4043;
    color: #fff;
    text-decoration: none;
  }
  .chip:hover {
    background-color: #353536;
  }
  .chip:focus {
    background-color: #353536;
  }
  .chip:active {
    background-color: #464849;
    border-color: #53575b;
  }
  .logo-light {
    display: none;
  }
  .gradient {
    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);
  }
}
</style>
<div class="container">
  <div class="headline">
    <svg class="logo-light" width="18" height="18" viewBox="9 9 35 35" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path fill-rule="evenodd" clip-rule="evenodd" d="M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z" fill="#4285F4"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z" fill="#34A853"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z" fill="#FBBC05"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z" fill="#EA4335"/>
    </svg>
    <svg class="logo-dark" width="18" height="18" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="23" fill="#FFF" r="22"/>
      <path d="M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z" fill="#4285F4"/>
      <path d="M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z" fill="#34A853"/>
      <path d="M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z" fill="#FBBC05"/>
      <path d="M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z" fill="#EA4335"/>
    </svg>
    <div class="gradient-container"><div class="gradient"></div></div>
  </div>
  <div class="carousel">
    <a class="chip" href="https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHY7KFTHAt2cJVXw_mJeW5D4LQP8EBpE46LAJkWZ7cmaXN2MjilanVPxak9QbQOmT4WfTAwwyGI5-HIctXME_rZAzaq0td_bjz6p0KGcvROYM3aoLY1OryphBo6XcDQg1PgI_-g1bYTQM-9UHBnjqxvvOAccEHmdiG8QTXLk8R6jHAx1JH7rRmazNQ665OZPeGgTAGlZF9r-A==">largest city in Mexico</a>
  </div>
</div>
"""
                        },
                        tool_call_id='q57vznsj',
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    TextPart(
                        content="""\
{
  "city": "Mexico City",
  "country": "Mexico"
}\
""",
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=106, output_tokens=59, details={'thoughts_tokens': 39, 'text_prompt_tokens': 106}
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='5cnFaeD-F-XG-8YPycjEkAs',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


async def test_tool_output_with_builtin_tools(allow_model_requests: None, google_model: GoogleModelFactory):
    m = google_model('gemini-3-flash-preview')
    agent = Agent(m, output_type=ToolOutput(CityLocation), builtin_tools=[WebSearchTool()])

    result = await agent.run('What is the largest city in Mexico?')
    assert isinstance(result.output, CityLocation)
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the largest city in Mexico?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['largest city in Mexico by population']},
                        tool_call_id='wozjlu8c',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={
                            'search_suggestions': """\
<style>
.container {
  align-items: center;
  border-radius: 8px;
  display: flex;
  font-family: Google Sans, Roboto, sans-serif;
  font-size: 14px;
  line-height: 20px;
  padding: 8px 12px;
}
.chip {
  display: inline-block;
  border: solid 1px;
  border-radius: 16px;
  min-width: 14px;
  padding: 5px 16px;
  text-align: center;
  user-select: none;
  margin: 0 8px;
  -webkit-tap-highlight-color: transparent;
}
.carousel {
  overflow: auto;
  scrollbar-width: none;
  white-space: nowrap;
  margin-right: -12px;
}
.headline {
  display: flex;
  margin-right: 4px;
}
.gradient-container {
  position: relative;
}
.gradient {
  position: absolute;
  transform: translate(3px, -9px);
  height: 36px;
  width: 9px;
}
@media (prefers-color-scheme: light) {
  .container {
    background-color: #fafafa;
    box-shadow: 0 0 0 1px #0000000f;
  }
  .headline-label {
    color: #1f1f1f;
  }
  .chip {
    background-color: #ffffff;
    border-color: #d2d2d2;
    color: #5e5e5e;
    text-decoration: none;
  }
  .chip:hover {
    background-color: #f2f2f2;
  }
  .chip:focus {
    background-color: #f2f2f2;
  }
  .chip:active {
    background-color: #d8d8d8;
    border-color: #b6b6b6;
  }
  .logo-dark {
    display: none;
  }
  .gradient {
    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);
  }
}
@media (prefers-color-scheme: dark) {
  .container {
    background-color: #1f1f1f;
    box-shadow: 0 0 0 1px #ffffff26;
  }
  .headline-label {
    color: #fff;
  }
  .chip {
    background-color: #2c2c2c;
    border-color: #3c4043;
    color: #fff;
    text-decoration: none;
  }
  .chip:hover {
    background-color: #353536;
  }
  .chip:focus {
    background-color: #353536;
  }
  .chip:active {
    background-color: #464849;
    border-color: #53575b;
  }
  .logo-light {
    display: none;
  }
  .gradient {
    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);
  }
}
</style>
<div class="container">
  <div class="headline">
    <svg class="logo-light" width="18" height="18" viewBox="9 9 35 35" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path fill-rule="evenodd" clip-rule="evenodd" d="M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z" fill="#4285F4"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z" fill="#34A853"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z" fill="#FBBC05"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z" fill="#EA4335"/>
    </svg>
    <svg class="logo-dark" width="18" height="18" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="23" fill="#FFF" r="22"/>
      <path d="M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z" fill="#4285F4"/>
      <path d="M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z" fill="#34A853"/>
      <path d="M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z" fill="#FBBC05"/>
      <path d="M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z" fill="#EA4335"/>
    </svg>
    <div class="gradient-container"><div class="gradient"></div></div>
  </div>
  <div class="carousel">
    <a class="chip" href="https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFOHZ57Ithg_-fDPkAYB2qW7ghSiWy1s2epO9WlV6TGvrCbn9JogGQq57Ea36ZxfYPtJW2s4AbW3k9nuLfJJEPs339TMX0XberwteQkgSZKO7NKJuAwNGLerivi1YjIj9lRJ4UVbp15qOI1u2cQW_lbEKaKZtbeg1Ogrf53rgx2QIutDJogEYFM9E7zszDbDNDLXdpjtT_jiU5TadbCW2L8VtpoXgWNYQ==">largest city in Mexico by population</a>
  </div>
</div>
"""
                        },
                        tool_call_id='wozjlu8c',
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args={'city': 'Mexico City', 'country': 'Mexico'},
                        tool_call_id='lxh4aeg2',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=98, output_tokens=88, details={'thoughts_tokens': 65, 'text_prompt_tokens': 98}
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='6MnFaajXBvuu-8YPwf6ciQs',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='lxh4aeg2',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


async def test_auto_mode_with_function_and_builtin_tools(allow_model_requests: None, google_model: GoogleModelFactory):
    m = google_model('gemini-3-flash-preview')
    agent = Agent(m, output_type=CityLocation, builtin_tools=[WebSearchTool()])

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
                        tool_call_id='7choxrkt',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    )
                ],
                usage=RequestUsage(
                    input_tokens=85, output_tokens=84, details={'thoughts_tokens': 72, 'text_prompt_tokens': 85}
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='6cnFab3WIfuj1MkP-svwmQI',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_user_country', content='Mexico', tool_call_id='7choxrkt', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['largest city in Mexico by population']},
                        tool_call_id='p59fjsgt',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={
                            'search_suggestions': """\
<style>
.container {
  align-items: center;
  border-radius: 8px;
  display: flex;
  font-family: Google Sans, Roboto, sans-serif;
  font-size: 14px;
  line-height: 20px;
  padding: 8px 12px;
}
.chip {
  display: inline-block;
  border: solid 1px;
  border-radius: 16px;
  min-width: 14px;
  padding: 5px 16px;
  text-align: center;
  user-select: none;
  margin: 0 8px;
  -webkit-tap-highlight-color: transparent;
}
.carousel {
  overflow: auto;
  scrollbar-width: none;
  white-space: nowrap;
  margin-right: -12px;
}
.headline {
  display: flex;
  margin-right: 4px;
}
.gradient-container {
  position: relative;
}
.gradient {
  position: absolute;
  transform: translate(3px, -9px);
  height: 36px;
  width: 9px;
}
@media (prefers-color-scheme: light) {
  .container {
    background-color: #fafafa;
    box-shadow: 0 0 0 1px #0000000f;
  }
  .headline-label {
    color: #1f1f1f;
  }
  .chip {
    background-color: #ffffff;
    border-color: #d2d2d2;
    color: #5e5e5e;
    text-decoration: none;
  }
  .chip:hover {
    background-color: #f2f2f2;
  }
  .chip:focus {
    background-color: #f2f2f2;
  }
  .chip:active {
    background-color: #d8d8d8;
    border-color: #b6b6b6;
  }
  .logo-dark {
    display: none;
  }
  .gradient {
    background: linear-gradient(90deg, #fafafa 15%, #fafafa00 100%);
  }
}
@media (prefers-color-scheme: dark) {
  .container {
    background-color: #1f1f1f;
    box-shadow: 0 0 0 1px #ffffff26;
  }
  .headline-label {
    color: #fff;
  }
  .chip {
    background-color: #2c2c2c;
    border-color: #3c4043;
    color: #fff;
    text-decoration: none;
  }
  .chip:hover {
    background-color: #353536;
  }
  .chip:focus {
    background-color: #353536;
  }
  .chip:active {
    background-color: #464849;
    border-color: #53575b;
  }
  .logo-light {
    display: none;
  }
  .gradient {
    background: linear-gradient(90deg, #1f1f1f 15%, #1f1f1f00 100%);
  }
}
</style>
<div class="container">
  <div class="headline">
    <svg class="logo-light" width="18" height="18" viewBox="9 9 35 35" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path fill-rule="evenodd" clip-rule="evenodd" d="M42.8622 27.0064C42.8622 25.7839 42.7525 24.6084 42.5487 23.4799H26.3109V30.1568H35.5897C35.1821 32.3041 33.9596 34.1222 32.1258 35.3448V39.6864H37.7213C40.9814 36.677 42.8622 32.2571 42.8622 27.0064V27.0064Z" fill="#4285F4"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 43.8555C30.9659 43.8555 34.8687 42.3195 37.7213 39.6863L32.1258 35.3447C30.5898 36.3792 28.6306 37.0061 26.3109 37.0061C21.8282 37.0061 18.0195 33.9811 16.6559 29.906H10.9194V34.3573C13.7563 39.9841 19.5712 43.8555 26.3109 43.8555V43.8555Z" fill="#34A853"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M16.6559 29.8904C16.3111 28.8559 16.1074 27.7588 16.1074 26.6146C16.1074 25.4704 16.3111 24.3733 16.6559 23.3388V18.8875H10.9194C9.74388 21.2072 9.06992 23.8247 9.06992 26.6146C9.06992 29.4045 9.74388 32.022 10.9194 34.3417L15.3864 30.8621L16.6559 29.8904V29.8904Z" fill="#FBBC05"/>
      <path fill-rule="evenodd" clip-rule="evenodd" d="M26.3109 16.2386C28.85 16.2386 31.107 17.1164 32.9095 18.8091L37.8466 13.8719C34.853 11.082 30.9659 9.3736 26.3109 9.3736C19.5712 9.3736 13.7563 13.245 10.9194 18.8875L16.6559 23.3388C18.0195 19.2636 21.8282 16.2386 26.3109 16.2386V16.2386Z" fill="#EA4335"/>
    </svg>
    <svg class="logo-dark" width="18" height="18" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="23" fill="#FFF" r="22"/>
      <path d="M33.76 34.26c2.75-2.56 4.49-6.37 4.49-11.26 0-.89-.08-1.84-.29-3H24.01v5.99h8.03c-.4 2.02-1.5 3.56-3.07 4.56v.75l3.91 2.97h.88z" fill="#4285F4"/>
      <path d="M15.58 25.77A8.845 8.845 0 0 0 24 31.86c1.92 0 3.62-.46 4.97-1.31l4.79 3.71C31.14 36.7 27.65 38 24 38c-5.93 0-11.01-3.4-13.45-8.36l.17-1.01 4.06-2.85h.8z" fill="#34A853"/>
      <path d="M15.59 20.21a8.864 8.864 0 0 0 0 5.58l-5.03 3.86c-.98-2-1.53-4.25-1.53-6.64 0-2.39.55-4.64 1.53-6.64l1-.22 3.81 2.98.22 1.08z" fill="#FBBC05"/>
      <path d="M24 14.14c2.11 0 4.02.75 5.52 1.98l4.36-4.36C31.22 9.43 27.81 8 24 8c-5.93 0-11.01 3.4-13.45 8.36l5.03 3.85A8.86 8.86 0 0 1 24 14.14z" fill="#EA4335"/>
    </svg>
    <div class="gradient-container"><div class="gradient"></div></div>
  </div>
  <div class="carousel">
    <a class="chip" href="https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQELEDCmkw9Vk5zkoWAZwYslEAGSOuJNO2kC-6RCF4Xn--832qhrdWhsMZirnGDHoxUY3EZlICCbKgwHY6fHBiU_IEmtlgwmQdGhq3WU5ZBkkJ6MxfKnNrjXn8qZJEsrFND13Sl-AKMYWDjufv6x0GGd8EZk7MLOC4KyVxAMfUuYCnJ2d9fTEUZ6nKN0cTJDKUsnSW-z4V80N7YLAFkxRTe3X_o1QlqcJw==">largest city in Mexico by population</a>
  </div>
</div>
"""
                        },
                        tool_call_id='p59fjsgt',
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args={'city': 'Mexico City', 'country': 'Mexico'},
                        tool_call_id='dsxlh118',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=187,
                    output_tokens=186,
                    details={'thoughts_tokens': 73, 'tool_use_prompt_tokens': 90, 'text_prompt_tokens': 187},
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='68nFacGdN9mY-8YP5rHl0As',
                finish_reason='stop',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='dsxlh118',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


# =============================================================================
# Gemini 2 fallback behavior
# =============================================================================


async def test_auto_output_mode_with_builtin_tools_falls_back(
    allow_model_requests: None, google_model: GoogleModelFactory
):
    """Gemini 2.5 with auto output mode + builtin tools silently converts to prompted output."""
    m = google_model('gemini-2.5-flash')
    agent = Agent(m, output_type=CityLocation, builtin_tools=[WebSearchTool()])
    result = await agent.run('What is the largest city in Mexico?')
    assert isinstance(result.output, CityLocation)
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What is the largest city in Mexico?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='web_search',
                        args={'queries': ['largest city in Mexico']},
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content=None,
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(content='{"city": "Mexico City", "country": "Mexico"}'),
                ],
                usage=RequestUsage(
                    input_tokens=85,
                    output_tokens=192,
                    details={
                        'thoughts_tokens': 32,
                        'tool_use_prompt_tokens': 132,
                        'text_prompt_tokens': 85,
                        'text_tool_use_prompt_tokens': 132,
                    },
                ),
                model_name='gemini-2.5-flash',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id='lTrNaf2-IoqojMcPnYv3kQI',
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
