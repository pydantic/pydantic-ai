"""DeepSeek's OpenAI-compatible Responses API, driven by `OpenAIResponsesModel` + `DeepSeekProvider`.

DeepSeek exposes a Responses endpoint at `https://api.deepseek.com/responses` for both V4 models.
It is stateless — every response reports `store: false`, and there is no server-side conversation to
resume — so `previous_response_id` continuation is not available and is not exercised here.

Each case snapshots both the resulting messages and the bodies that actually went out, captured by an
`httpx2` event hook so the wire assertions run against what the client built rather than what the
cassette happens to hold. The request bodies are what pin the facts this pairing rests on:

- the `openai_reasoning_effort` setting reaching DeepSeek as `reasoning.effort`
- DeepSeek's raw chain-of-thought going back as `reasoning_text` content on a follow-up request
- the `phase` DeepSeek labels its output with being surfaced in `provider_details` but never sent
  back, since `openai_supports_phase` stays off for a provider that doesn't document accepting it
- `NativeOutput` working here while it is refused on Chat Completions, the one capability DeepSeek
  splits between its two endpoints
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import httpx2
import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeOutput,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsDecimal, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.deepseek import DeepSeekProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


_request_body_adapter = TypeAdapter(dict[str, JsonValue])


class City(BaseModel):
    city: str
    country: str


def get_temperature(city: str) -> float:
    """Get the current temperature in a city."""
    return 21.0


# DeepSeek V4 prices vary by request time: genai-prices doubles the rate during 01:00-04:00 and
# 06:00-10:00 UTC, and a cost is priced from `ModelResponse.timestamp`, which the model stamps with
# the wall clock rather than replaying it off the cassette. So these response-shape cases assert
# that a cost was calculated without duplicating genai-prices' time-dependent pricing tests — a
# literal pin here holds only for the pricing tier and window it was recorded in.
@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    follow_up_prompt: str | None = None
    stream: bool = False
    output_type: type[str] | type[City] | NativeOutput[City] = str
    tools: Sequence[Callable[..., Any]] = ()
    model_settings: OpenAIResponsesModelSettings | None = None
    expected_output: str | City = ''
    expected_messages: list[ModelMessage] = field(default_factory=list[ModelMessage])
    expected_request_bodies: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


CASES = [
    Case(
        id='text',
        prompt='What is the capital of France?',
        expected_output=snapshot('The capital of France is Paris.'),
        expected_messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='1dfb7474-dbd9-4b8d-a852-ebe49345dd32',
                            provider_name='deepseek',
                            provider_details={'raw_content': ['We need answer capital of France.']},
                        ),
                        TextPart(
                            content='The capital of France is Paris.',
                            id='8050c022-c55f-4377-b009-1d6341eacc57',
                            provider_name='deepseek',
                            provider_details={'phase': 'final_answer'},
                        ),
                    ],
                    usage=RequestUsage(
                        details={'reasoning_tokens': 7},
                        input_tokens=90,
                        output_reasoning_tokens=7,
                        output_tokens=15,
                        cost=IsDecimal(),
                    ),
                    model_name='deepseek-v4-flash',
                    timestamp=IsDatetime(),
                    provider_name='deepseek',
                    provider_url='https://api.deepseek.com',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id='2e532992-cac2-41fd-80b5-c98ad79215f3',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        expected_request_bodies=snapshot(
            [
                {
                    'input': [{'role': 'user', 'content': 'What is the capital of France?'}],
                    'model': 'deepseek-v4-flash',
                    'stream': False,
                }
            ]
        ),
    ),
    Case(
        id='text_stream',
        prompt='What is the capital of France?',
        stream=True,
        expected_output=snapshot('The capital of France is Paris.'),
        expected_messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='What is the capital of France?', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='b594b7e1-3dbb-4b65-b8c2-f4f5aae4ee80',
                            provider_name='deepseek',
                            provider_details={'raw_content': ['We need answer capital of France.']},
                        ),
                        TextPart(
                            content='The capital of France is Paris.',
                            id='f9be6778-cb0b-4264-8d37-24af0349d7ef',
                            provider_name='deepseek',
                            provider_details={'phase': 'final_answer'},
                        ),
                    ],
                    usage=RequestUsage(
                        details={'reasoning_tokens': 7},
                        output_tokens=15,
                        output_reasoning_tokens=7,
                        input_tokens=90,
                        cost=IsDecimal(),
                    ),
                    model_name='deepseek-v4-flash',
                    timestamp=IsDatetime(),
                    provider_name='deepseek',
                    provider_url='https://api.deepseek.com',
                    provider_details={'timestamp': IsDatetime(), 'finish_reason': 'completed'},
                    provider_response_id='bf5e7791-6c05-44ca-b7e0-56aa217150b1',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        expected_request_bodies=snapshot(
            [
                {
                    'input': [{'role': 'user', 'content': 'What is the capital of France?'}],
                    'model': 'deepseek-v4-flash',
                    'stream': True,
                }
            ]
        ),
    ),
    # The follow-up request body is where the `openai_supports_phase` off side is pinned: DeepSeek
    # labels its answer `final_answer` on the way in, and the assistant turn goes back out as a
    # bare `{'role': 'assistant', 'content': ...}` with no `phase` key. Flipping the flag on would
    # add one and fail this snapshot.
    Case(
        id='text_multi_turn',
        prompt='Say exactly: hello',
        follow_up_prompt='Now say exactly: goodbye',
        expected_output=snapshot('goodbye'),
        expected_messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Say exactly: hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='1e97d346-aa16-46e5-866d-b34e91c5d499',
                            provider_name='deepseek',
                            provider_details={'raw_content': ['We need answer exactly hello.']},
                        ),
                        TextPart(
                            content='hello',
                            id='c431c0bc-edd8-4d1d-9bab-46ddd59d15e1',
                            provider_name='deepseek',
                            provider_details={'phase': 'final_answer'},
                        ),
                    ],
                    usage=RequestUsage(
                        details={'reasoning_tokens': 6},
                        input_tokens=87,
                        output_reasoning_tokens=6,
                        output_tokens=8,
                        cost=IsDecimal(),
                    ),
                    model_name='deepseek-v4-flash',
                    timestamp=IsDatetime(),
                    provider_name='deepseek',
                    provider_url='https://api.deepseek.com',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id='49d254fd-deb4-4207-a85a-038a3893db25',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[UserPromptPart(content='Now say exactly: goodbye', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='3e320fb9-da94-4b4f-8ef8-08d750d3cdf9',
                            provider_name='deepseek',
                            provider_details={
                                'raw_content': [
                                    'We need to respond to the user. The user said "Now say exactly: goodbye". The instruction is to say exactly "goodbye". We already responded "hello" to the previous "Say exactly: hello". Now we just output "goodbye". Ensure no extra text.'
                                ]
                            },
                        ),
                        TextPart(
                            content='goodbye',
                            id='2d999a8e-8086-4e87-bf7e-051cab5b9181',
                            provider_name='deepseek',
                            provider_details={'phase': 'final_answer'},
                        ),
                    ],
                    usage=RequestUsage(
                        details={'reasoning_tokens': 56},
                        input_tokens=97,
                        output_reasoning_tokens=56,
                        output_tokens=59,
                        cost=IsDecimal(),
                    ),
                    model_name='deepseek-v4-flash',
                    timestamp=IsDatetime(),
                    provider_name='deepseek',
                    provider_url='https://api.deepseek.com',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id='cad18dc8-ce7a-4247-8cd7-d9574926a924',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        expected_request_bodies=snapshot(
            [
                {
                    'input': [{'role': 'user', 'content': 'Say exactly: hello'}],
                    'model': 'deepseek-v4-flash',
                    'stream': False,
                },
                {
                    'input': [
                        {'role': 'user', 'content': 'Say exactly: hello'},
                        {
                            'id': '1e97d346-aa16-46e5-866d-b34e91c5d499',
                            'summary': [],
                            'encrypted_content': None,
                            'type': 'reasoning',
                            'content': [{'text': 'We need answer exactly hello.', 'type': 'reasoning_text'}],
                        },
                        {'role': 'assistant', 'content': 'hello'},
                        {'role': 'user', 'content': 'Now say exactly: goodbye'},
                    ],
                    'model': 'deepseek-v4-flash',
                    'stream': False,
                },
            ]
        ),
    ),
    Case(
        id='function_tool',
        prompt='What is the temperature in Tokyo?',
        tools=[get_temperature],
        expected_output=snapshot('The current temperature in Tokyo is 21.0°C.'),
        expected_messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='What is the temperature in Tokyo?', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='76f8b89e-4a41-46fb-86ae-546cc1e4ba6c',
                            provider_name='deepseek',
                            provider_details={
                                'raw_content': [
                                    'The user asks for the temperature in Tokyo. I should call the get_temperature tool.'
                                ]
                            },
                        ),
                        ToolCallPart(
                            tool_name=get_temperature.__qualname__,
                            args='{"city": "Tokyo"}',
                            tool_call_id='call_00_iD0U8IMtyIljI0ET7GLz1318',
                            id='8fc1af85-3010-42b2-bcb8-1a6d5003ad3b',
                            provider_name='deepseek',
                        ),
                    ],
                    usage=RequestUsage(
                        details={'reasoning_tokens': 18},
                        input_tokens=366,
                        cache_read_tokens=256,
                        output_reasoning_tokens=18,
                        output_tokens=63,
                        cost=IsDecimal(),
                    ),
                    model_name='deepseek-v4-flash',
                    timestamp=IsDatetime(),
                    provider_name='deepseek',
                    provider_url='https://api.deepseek.com',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id='92471b7c-94ad-452f-a3f5-c29aa74a95e1',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=get_temperature.__qualname__,
                            content=21.0,
                            tool_call_id='call_00_iD0U8IMtyIljI0ET7GLz1318',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The current temperature in Tokyo is 21.0°C.',
                            id='f00869b9-665c-4653-b2ee-3857ef1413fd',
                            provider_name='deepseek',
                            provider_details={'phase': 'final_answer'},
                        )
                    ],
                    usage=RequestUsage(
                        details={'reasoning_tokens': 0},
                        input_tokens=444,
                        cache_read_tokens=384,
                        output_reasoning_tokens=0,
                        output_tokens=14,
                        cost=IsDecimal(),
                    ),
                    model_name='deepseek-v4-flash',
                    timestamp=IsDatetime(),
                    provider_name='deepseek',
                    provider_url='https://api.deepseek.com',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id='7f3d6c65-b8c2-410e-87ce-987bc467aef6',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        expected_request_bodies=snapshot(
            [
                {
                    'input': [{'role': 'user', 'content': 'What is the temperature in Tokyo?'}],
                    'model': 'deepseek-v4-flash',
                    'stream': False,
                    'tool_choice': 'auto',
                    'tools': [
                        {
                            'name': get_temperature.__qualname__,
                            'parameters': {
                                'additionalProperties': False,
                                'properties': {'city': {'type': 'string'}},
                                'required': ['city'],
                                'type': 'object',
                            },
                            'type': 'function',
                            'description': 'Get the current temperature in a city.',
                            'strict': True,
                        }
                    ],
                },
                {
                    'input': [
                        {'role': 'user', 'content': 'What is the temperature in Tokyo?'},
                        {
                            'id': '76f8b89e-4a41-46fb-86ae-546cc1e4ba6c',
                            'summary': [],
                            'encrypted_content': None,
                            'type': 'reasoning',
                            'content': [
                                {
                                    'text': 'The user asks for the temperature in Tokyo. I should call the get_temperature tool.',
                                    'type': 'reasoning_text',
                                }
                            ],
                        },
                        {
                            'name': get_temperature.__qualname__,
                            'arguments': '{"city": "Tokyo"}',
                            'call_id': 'call_00_iD0U8IMtyIljI0ET7GLz1318',
                            'type': 'function_call',
                        },
                        {
                            'type': 'function_call_output',
                            'call_id': 'call_00_iD0U8IMtyIljI0ET7GLz1318',
                            'output': '21.0',
                        },
                    ],
                    'model': 'deepseek-v4-flash',
                    'stream': False,
                    'tool_choice': 'auto',
                    'tools': [
                        {
                            'name': get_temperature.__qualname__,
                            'parameters': {
                                'additionalProperties': False,
                                'properties': {'city': {'type': 'string'}},
                                'required': ['city'],
                                'type': 'object',
                            },
                            'type': 'function',
                            'description': 'Get the current temperature in a city.',
                            'strict': True,
                        }
                    ],
                },
            ]
        ),
    ),
    Case(
        id='function_tool_stream',
        prompt='What is the temperature in Tokyo?',
        stream=True,
        tools=[get_temperature],
        expected_output=snapshot('The current temperature in Tokyo is **21.0°C**.'),
        expected_messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='What is the temperature in Tokyo?', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='fa6f3a83-5d25-46e8-9d03-1a89ce5cf2ba',
                            provider_name='deepseek',
                            provider_details={
                                'raw_content': ["The user asks about temperature in Tokyo. I'll call the tool."]
                            },
                        ),
                        ToolCallPart(
                            tool_name=get_temperature.__qualname__,
                            args='{"city": "Tokyo"}',
                            tool_call_id='call_00_xjY8Z2BvSlzgEmmw0DtH0464',
                            id='62bf2bb7-56af-4e3a-883b-83d4aad54da1',
                            provider_name='deepseek',
                        ),
                    ],
                    usage=RequestUsage(
                        details={'reasoning_tokens': 14},
                        output_tokens=59,
                        output_reasoning_tokens=14,
                        cache_read_tokens=256,
                        input_tokens=366,
                        cost=IsDecimal(),
                    ),
                    model_name='deepseek-v4-flash',
                    timestamp=IsDatetime(),
                    provider_name='deepseek',
                    provider_url='https://api.deepseek.com',
                    provider_details={'timestamp': IsDatetime(), 'finish_reason': 'completed'},
                    provider_response_id='1235b7ba-fdc9-4a1c-bfe4-6137c207baf3',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=get_temperature.__qualname__,
                            content=21.0,
                            tool_call_id='call_00_xjY8Z2BvSlzgEmmw0DtH0464',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The current temperature in Tokyo is **21.0°C**.',
                            id='305ae5c3-aed7-4986-88b5-4bcbb950ca6b',
                            provider_name='deepseek',
                            provider_details={'phase': 'final_answer'},
                        )
                    ],
                    usage=RequestUsage(
                        details={'reasoning_tokens': 0},
                        output_tokens=14,
                        output_reasoning_tokens=0,
                        cache_read_tokens=384,
                        input_tokens=440,
                        cost=IsDecimal(),
                    ),
                    model_name='deepseek-v4-flash',
                    timestamp=IsDatetime(),
                    provider_name='deepseek',
                    provider_url='https://api.deepseek.com',
                    provider_details={'timestamp': IsDatetime(), 'finish_reason': 'completed'},
                    provider_response_id='33df88f0-9f36-4616-95b0-ead91a37f7f1',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        expected_request_bodies=snapshot(
            [
                {
                    'input': [{'role': 'user', 'content': 'What is the temperature in Tokyo?'}],
                    'model': 'deepseek-v4-flash',
                    'stream': True,
                    'tool_choice': 'auto',
                    'tools': [
                        {
                            'name': get_temperature.__qualname__,
                            'parameters': {
                                'additionalProperties': False,
                                'properties': {'city': {'type': 'string'}},
                                'required': ['city'],
                                'type': 'object',
                            },
                            'type': 'function',
                            'description': 'Get the current temperature in a city.',
                            'strict': True,
                        }
                    ],
                },
                {
                    'input': [
                        {'role': 'user', 'content': 'What is the temperature in Tokyo?'},
                        {
                            'id': 'fa6f3a83-5d25-46e8-9d03-1a89ce5cf2ba',
                            'summary': [],
                            'encrypted_content': None,
                            'type': 'reasoning',
                            'content': [
                                {
                                    'text': "The user asks about temperature in Tokyo. I'll call the tool.",
                                    'type': 'reasoning_text',
                                }
                            ],
                        },
                        {
                            'name': get_temperature.__qualname__,
                            'arguments': '{"city": "Tokyo"}',
                            'call_id': 'call_00_xjY8Z2BvSlzgEmmw0DtH0464',
                            'type': 'function_call',
                        },
                        {
                            'type': 'function_call_output',
                            'call_id': 'call_00_xjY8Z2BvSlzgEmmw0DtH0464',
                            'output': '21.0',
                        },
                    ],
                    'model': 'deepseek-v4-flash',
                    'stream': True,
                    'tool_choice': 'auto',
                    'tools': [
                        {
                            'name': get_temperature.__qualname__,
                            'parameters': {
                                'additionalProperties': False,
                                'properties': {'city': {'type': 'string'}},
                                'required': ['city'],
                                'type': 'object',
                            },
                            'type': 'function',
                            'description': 'Get the current temperature in a city.',
                            'strict': True,
                        }
                    ],
                },
            ]
        ),
    ),
    Case(
        id='structured_output',
        prompt='What is the capital of Japan?',
        output_type=City,
        expected_output=snapshot(City(city='Tokyo', country='Japan')),
        expected_messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='What is the capital of Japan?', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='eaa32d41-e515-4399-adf5-9ef8c58fc62d',
                            provider_name='deepseek',
                            provider_details={
                                'raw_content': [
                                    'The user asked for the capital of Japan, which is Tokyo. I should use the final_result tool.'
                                ]
                            },
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"city": "Tokyo", "country": "Japan"}',
                            tool_call_id='call_00_fHeIT46fj9tK0YhyHu7Q9878',
                            id='c830d5e3-c87c-4e9c-9150-b157d70e5f9f',
                            provider_name='deepseek',
                        ),
                    ],
                    usage=RequestUsage(
                        details={'reasoning_tokens': 21},
                        input_tokens=376,
                        cache_read_tokens=256,
                        output_reasoning_tokens=21,
                        output_tokens=81,
                        cost=IsDecimal(),
                    ),
                    model_name='deepseek-v4-flash',
                    timestamp=IsDatetime(),
                    provider_name='deepseek',
                    provider_url='https://api.deepseek.com',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id='4333ebf8-e16c-4969-b5ac-fde1294e642a',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call_00_fHeIT46fj9tK0YhyHu7Q9878',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        expected_request_bodies=snapshot(
            [
                {
                    'input': [{'role': 'user', 'content': 'What is the capital of Japan?'}],
                    'model': 'deepseek-v4-flash',
                    'stream': False,
                    'tool_choice': 'auto',
                    'tools': [
                        {
                            'name': 'final_result',
                            'parameters': {
                                'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                                'required': ['city', 'country'],
                                'type': 'object',
                                'additionalProperties': False,
                            },
                            'type': 'function',
                            'description': 'The final response which ends this conversation',
                            'strict': True,
                        }
                    ],
                }
            ]
        ),
    ),
    Case(
        id='reasoning_effort',
        prompt='What is 17 * 23?',
        model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort='high'),
        expected_output=snapshot('391'),
        expected_messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='What is 17 * 23?', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='99b90f85-3f3e-400c-b826-147f81c4672f',
                            provider_name='deepseek',
                            provider_details={
                                'raw_content': ['We need answer simple. Need comply. 17*23 = 391. final.']
                            },
                        ),
                        TextPart(
                            content='391',
                            id='d7c69e03-b418-418f-98ab-fe09ad528cc0',
                            provider_name='deepseek',
                            provider_details={'phase': 'final_answer'},
                        ),
                    ],
                    usage=RequestUsage(
                        details={'reasoning_tokens': 18},
                        input_tokens=91,
                        output_reasoning_tokens=18,
                        output_tokens=20,
                        cost=IsDecimal(),
                    ),
                    model_name='deepseek-v4-flash',
                    timestamp=IsDatetime(),
                    provider_name='deepseek',
                    provider_url='https://api.deepseek.com',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id='7a293ad0-b030-4055-a857-894eec88f488',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        expected_request_bodies=snapshot(
            [
                {
                    'input': [{'role': 'user', 'content': 'What is 17 * 23?'}],
                    'model': 'deepseek-v4-flash',
                    'reasoning': {'effort': 'high'},
                    'stream': False,
                }
            ]
        ),
    ),
    Case(
        id='native_output',
        prompt='What is the capital of Mexico?',
        output_type=NativeOutput(City),
        expected_output=snapshot(City(city='Mexico City', country='Mexico')),
        expected_messages=snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='What is the capital of Mexico?', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ThinkingPart(
                            content='',
                            id='45058da2-f6c2-482f-a8ef-40ce3701e03d',
                            provider_name='deepseek',
                            provider_details={
                                'raw_content': [
                                    'We need to answer the capital of Mexico. The schema requires city and country. So city: Mexico City, country: Mexico.'
                                ]
                            },
                        ),
                        TextPart(
                            content='{"city": "Mexico City", "country": "Mexico"}',
                            id='153730a1-af42-4a99-b704-f406054ac191',
                            provider_name='deepseek',
                            provider_details={'phase': 'final_answer'},
                        ),
                    ],
                    usage=RequestUsage(
                        details={'reasoning_tokens': 26},
                        input_tokens=148,
                        output_reasoning_tokens=26,
                        output_tokens=40,
                        cost=IsDecimal(),
                    ),
                    model_name='deepseek-v4-flash',
                    timestamp=IsDatetime(),
                    provider_name='deepseek',
                    provider_url='https://api.deepseek.com',
                    provider_details={'finish_reason': 'completed', 'timestamp': IsDatetime()},
                    provider_response_id='3dd7f44c-de93-47d7-b2cb-d2269c3e29c4',
                    finish_reason='stop',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        ),
        expected_request_bodies=snapshot(
            [
                {
                    'input': [{'role': 'user', 'content': 'What is the capital of Mexico?'}],
                    'model': 'deepseek-v4-flash',
                    'stream': False,
                    'text': {
                        'format': {
                            'type': 'json_schema',
                            'name': City.__qualname__,
                            'schema': {
                                'properties': {'city': {'type': 'string'}, 'country': {'type': 'string'}},
                                'required': ['city', 'country'],
                                'type': 'object',
                                'additionalProperties': False,
                            },
                            'strict': True,
                        }
                    },
                }
            ]
        ),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
async def test_deepseek_responses(case: Case, allow_model_requests: None, deepseek_api_key: str):
    """`OpenAIResponsesModel('deepseek-v4-flash', provider=DeepSeekProvider())` against the live API.

    Every case snapshots the request bodies as the httpx hook saw them, so the recorded facts the
    pairing depends on stay pinned to what the client builds rather than to the cassette: the
    `tool_choice: 'auto'` that DeepSeek's thinking-conditional forcing restriction produces while
    thinking is on (its default) even where the run would otherwise force a tool, and the `phase`
    label DeepSeek puts on its output being surfaced in `provider_details` while never going back
    out (the off side of `openai_supports_phase`, which stays off because DeepSeek doesn't document
    accepting it).
    """
    sent_bodies: list[dict[str, Any]] = []

    async def capture_request(request: httpx2.Request) -> None:
        sent_bodies.append(json.loads(request.content))

    http_client = httpx2.AsyncClient(event_hooks={'request': [capture_request]})
    model = OpenAIResponsesModel(
        'deepseek-v4-flash', provider=DeepSeekProvider(api_key=deepseek_api_key, http_client=http_client)
    )
    agent = Agent(model, output_type=case.output_type, tools=case.tools, model_settings=case.model_settings)

    async def run(
        prompt: str, message_history: list[ModelMessage] | None = None
    ) -> tuple[str | City, list[ModelMessage]]:
        if case.stream:
            async with agent.run_stream(prompt, message_history=message_history) as streamed:
                output = await streamed.get_output()
            return output, streamed.all_messages()
        result = await agent.run(prompt, message_history=message_history)
        return result.output, result.all_messages()

    output, messages = await run(case.prompt)
    if case.follow_up_prompt is not None:
        output, messages = await run(case.follow_up_prompt, message_history=messages)

    assert output == case.expected_output
    assert messages == case.expected_messages
    assert sent_bodies == case.expected_request_bodies


async def test_deepseek_responses_replay_unsent_item_ids(allow_model_requests: None, deepseek_api_key: str) -> None:
    """Item IDs another provider minted never reach DeepSeek, so they don't pin the interleaved order."""
    sent_bodies: list[dict[str, JsonValue]] = []

    async def capture_request(request: httpx2.Request) -> None:
        sent_bodies.append(_request_body_adapter.validate_json(request.read()))

    history = [
        ModelResponse(
            parts=[
                ThinkingPart(content='inspect inputs', id='rs_1', provider_name='openai'),
                ToolCallPart('read', {'path': 'a'}, tool_call_id='call-a', id='fc_1', provider_name='openai'),
                ThinkingPart(content='inspect views', id='rs_2', provider_name='openai'),
                ToolCallPart('view', {'path': 'b'}, tool_call_id='call-b', id='fc_2', provider_name='openai'),
            ],
            provider_name='openai',
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('read', 'contents-a', tool_call_id='call-a'),
                ToolReturnPart('view', 'contents-b', tool_call_id='call-b'),
            ]
        ),
    ]
    original_history = deepcopy(history)

    async with httpx2.AsyncClient(event_hooks={'request': [capture_request]}) as http_client:
        model = OpenAIResponsesModel(
            'deepseek-v4-flash', provider=DeepSeekProvider(api_key=deepseek_api_key, http_client=http_client)
        )
        result = await Agent(model).run('Reply exactly: done', message_history=history)

    assert result.output == 'done'
    assert history == original_history
    assert sent_bodies == snapshot(
        [
            {
                'input': [
                    {
                        'role': 'assistant',
                        'content': """\
<think>
inspect inputs
</think>\
""",
                    },
                    {
                        'role': 'assistant',
                        'content': """\
<think>
inspect views
</think>\
""",
                    },
                    {'name': 'read', 'arguments': '{"path":"a"}', 'call_id': 'call-a', 'type': 'function_call'},
                    {'name': 'view', 'arguments': '{"path":"b"}', 'call_id': 'call-b', 'type': 'function_call'},
                    {'type': 'function_call_output', 'call_id': 'call-a', 'output': 'contents-a'},
                    {'type': 'function_call_output', 'call_id': 'call-b', 'output': 'contents-b'},
                    {'role': 'user', 'content': 'Reply exactly: done'},
                ],
                'model': 'deepseek-v4-flash',
                'stream': False,
            }
        ]
    )


async def test_deepseek_responses_replay_interleaved_settled_function_calls(
    allow_model_requests: None, deepseek_api_key: str
) -> None:
    """DeepSeek accepts the grouped wire projection of complete portable history."""
    sent_bodies: list[dict[str, JsonValue]] = []

    async def capture_request(request: httpx2.Request) -> None:
        sent_bodies.append(_request_body_adapter.validate_json(request.read()))

    history = [
        ModelResponse(
            parts=[
                ThinkingPart(content='inspect inputs'),
                ToolCallPart('read', {'path': 'a'}, tool_call_id='call-a'),
                ThinkingPart(content='inspect views'),
                ToolCallPart('view', {'path': 'b'}, tool_call_id='call-b'),
            ]
        ),
        ModelRequest(
            parts=[
                RetryPromptPart('read failed', tool_name='read', tool_call_id='call-a'),
                ToolReturnPart('view', 'contents', tool_call_id='call-b'),
            ]
        ),
    ]
    original_history = deepcopy(history)

    async with httpx2.AsyncClient(event_hooks={'request': [capture_request]}) as http_client:
        model = OpenAIResponsesModel(
            'deepseek-v4-flash', provider=DeepSeekProvider(api_key=deepseek_api_key, http_client=http_client)
        )
        result = await Agent(model).run('Reply exactly: done', message_history=history)

    assert result.output == 'done'
    assert history == original_history
    assert sent_bodies == snapshot(
        [
            {
                'input': [
                    {'role': 'assistant', 'content': '<think>\ninspect inputs\n</think>'},
                    {'role': 'assistant', 'content': '<think>\ninspect views\n</think>'},
                    {'name': 'read', 'arguments': '{"path":"a"}', 'call_id': 'call-a', 'type': 'function_call'},
                    {'name': 'view', 'arguments': '{"path":"b"}', 'call_id': 'call-b', 'type': 'function_call'},
                    {
                        'type': 'function_call_output',
                        'call_id': 'call-a',
                        'output': 'read failed\n\nFix the errors and try again.',
                    },
                    {'type': 'function_call_output', 'call_id': 'call-b', 'output': 'contents'},
                    {'role': 'user', 'content': 'Reply exactly: done'},
                ],
                'model': 'deepseek-v4-flash',
                'stream': False,
            }
        ]
    )


async def test_deepseek_responses_replay_own_reasoning_history(
    allow_model_requests: None, deepseek_api_key: str
) -> None:
    """A turn DeepSeek itself produced replays, reasoning items and their IDs intact.

    `_process_response` gives every DeepSeek reasoning item a `ThinkingPart` carrying the provider's
    own `id` and its raw CoT, so this is the shape a DeepSeek-origin history actually has. It is the
    interesting case because those IDs go out on the wire: the request below is rejected when the
    calls stay interleaved, which is what makes reordering the load-bearing step rather than the
    portable-history case where nothing is emitted to preserve.

    Two turns run over the same history so the recording carries a consecutive request pair for the
    suite-wide cache-prefix check: reordering has to place the calls identically on every turn, or it
    would rewrite the cacheable prefix and silently cost a full re-read of the conversation.
    """
    sent_bodies: list[dict[str, JsonValue]] = []

    async def capture_request(request: httpx2.Request) -> None:
        sent_bodies.append(_request_body_adapter.validate_json(request.read()))

    history = [
        ModelResponse(
            parts=[
                ThinkingPart(
                    content='inspect inputs',
                    id='rs-a',
                    provider_name='deepseek',
                    provider_details={'raw_content': ['inspect inputs']},
                ),
                ToolCallPart('read', {'path': 'a'}, tool_call_id='call-a'),
                ThinkingPart(
                    content='inspect views',
                    id='rs-b',
                    provider_name='deepseek',
                    provider_details={'raw_content': ['inspect views']},
                ),
                ToolCallPart('view', {'path': 'b'}, tool_call_id='call-b'),
            ],
            provider_name='deepseek',
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('read', 'contents', tool_call_id='call-a'),
                ToolReturnPart('view', 'rendered', tool_call_id='call-b'),
            ]
        ),
    ]
    original_history = deepcopy(history)

    async with httpx2.AsyncClient(event_hooks={'request': [capture_request]}) as http_client:
        model = OpenAIResponsesModel(
            'deepseek-v4-flash', provider=DeepSeekProvider(api_key=deepseek_api_key, http_client=http_client)
        )
        agent = Agent(model)
        result = await agent.run('Reply exactly: done', message_history=history)
        follow_up = await agent.run('Reply exactly: again', message_history=result.all_messages())

    assert result.output == 'done'
    assert follow_up.output == 'again'
    assert history == original_history

    first_input, second_input = (body['input'] for body in sent_bodies)
    assert isinstance(first_input, list)
    assert isinstance(second_input, list)
    assert second_input[: len(first_input) - 1] == first_input[:-1], (
        'the replayed turn must render identically on both turns, or the cacheable prefix moves'
    )
    assert sent_bodies == snapshot(
        [
            {
                'input': [
                    {
                        'id': 'rs-a',
                        'summary': [{'text': 'inspect inputs', 'type': 'summary_text'}],
                        'encrypted_content': None,
                        'type': 'reasoning',
                        'content': [{'text': 'inspect inputs', 'type': 'reasoning_text'}],
                    },
                    {
                        'id': 'rs-b',
                        'summary': [{'text': 'inspect views', 'type': 'summary_text'}],
                        'encrypted_content': None,
                        'type': 'reasoning',
                        'content': [{'text': 'inspect views', 'type': 'reasoning_text'}],
                    },
                    {'name': 'read', 'arguments': '{"path":"a"}', 'call_id': 'call-a', 'type': 'function_call'},
                    {'name': 'view', 'arguments': '{"path":"b"}', 'call_id': 'call-b', 'type': 'function_call'},
                    {'type': 'function_call_output', 'call_id': 'call-a', 'output': 'contents'},
                    {'type': 'function_call_output', 'call_id': 'call-b', 'output': 'rendered'},
                    {'role': 'user', 'content': 'Reply exactly: done'},
                ],
                'model': 'deepseek-v4-flash',
                'stream': False,
            },
            {
                'input': [
                    {
                        'id': 'rs-a',
                        'summary': [{'text': 'inspect inputs', 'type': 'summary_text'}],
                        'encrypted_content': None,
                        'type': 'reasoning',
                        'content': [{'text': 'inspect inputs', 'type': 'reasoning_text'}],
                    },
                    {
                        'id': 'rs-b',
                        'summary': [{'text': 'inspect views', 'type': 'summary_text'}],
                        'encrypted_content': None,
                        'type': 'reasoning',
                        'content': [{'text': 'inspect views', 'type': 'reasoning_text'}],
                    },
                    {'name': 'read', 'arguments': '{"path":"a"}', 'call_id': 'call-a', 'type': 'function_call'},
                    {'name': 'view', 'arguments': '{"path":"b"}', 'call_id': 'call-b', 'type': 'function_call'},
                    {'type': 'function_call_output', 'call_id': 'call-a', 'output': 'contents'},
                    {'type': 'function_call_output', 'call_id': 'call-b', 'output': 'rendered'},
                    {'role': 'user', 'content': 'Reply exactly: done'},
                    {
                        'id': '84cf774d-60a1-4e20-85e3-4cf73ee18cd5',
                        'summary': [],
                        'encrypted_content': None,
                        'type': 'reasoning',
                        'content': [
                            {
                                'text': 'The user requested reading "a" and viewing "b", then asked to reply exactly "done". The outputs confirm both operations succeeded. The next turn should be the exact reply "done".',
                                'type': 'reasoning_text',
                            }
                        ],
                    },
                    {'role': 'assistant', 'content': 'done'},
                    {'role': 'user', 'content': 'Reply exactly: again'},
                ],
                'model': 'deepseek-v4-flash',
                'stream': False,
            },
        ]
    )


async def test_deepseek_responses_rejects_interleaved_function_calls(
    allow_model_requests: None, deepseek_api_key: str
) -> None:
    """DeepSeek rejects the interleaved order — the provider fact the grouping rests on.

    Recorded by turning the capability fact back on, which is the only way to reach the pre-fix wire
    order now. Without this the rejection would live only in prose, so nobody could tell when
    DeepSeek fixes its endpoint and the normalization becomes dead weight.
    """
    history = [
        ModelResponse(
            parts=[
                ThinkingPart(content='inspect inputs'),
                ToolCallPart('read', {'path': 'a'}, tool_call_id='call-a'),
                ThinkingPart(content='inspect views'),
                ToolCallPart('view', {'path': 'b'}, tool_call_id='call-b'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('read', 'contents', tool_call_id='call-a'),
                ToolReturnPart('view', 'rendered', tool_call_id='call-b'),
            ]
        ),
    ]
    model = OpenAIResponsesModel(
        'deepseek-v4-flash',
        provider=DeepSeekProvider(api_key=deepseek_api_key),
        profile=OpenAIModelProfile(openai_responses_supports_interleaved_function_calls=True),
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model).run('Reply exactly: done', message_history=history)

    assert exc_info.value.status_code == 400
    assert 'No tool output found for tool call call-a' in str(exc_info.value)
