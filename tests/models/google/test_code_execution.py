"""Tests for Google CodeExecutionTool (uses executableCode/codeExecutionResult parts, not toolCall/toolResponse)."""

from __future__ import annotations as _annotations

from datetime import timezone
from typing import TYPE_CHECKING

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FinalResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    UserPromptPart,
)
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.messages import (
    BuiltinToolCallEvent,  # pyright: ignore[reportDeprecated]
    BuiltinToolResultEvent,  # pyright: ignore[reportDeprecated]
)
from pydantic_ai.usage import RequestUsage

from ...conftest import IsDatetime, IsNow, IsStr, try_import
from ...parts_from_messages import part_types_from_messages

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

if TYPE_CHECKING:
    from collections.abc import Callable

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
    pytest.mark.filterwarnings('ignore:.*is deprecated and will reach end-of-life.*:DeprecationWarning'),
]


async def test_code_execution_stream(
    allow_model_requests: None,
    google_model: GoogleModelFactory,
):
    """Test Gemini streaming only code execution result or executable_code."""
    m = google_model('gemini-3-flash-preview')
    agent = Agent(
        model=m,
        instructions='Be concise and always use Python to do calculations no matter how small.',
        builtin_tools=[CodeExecutionTool()],
    )

    event_parts: list[AgentStreamEvent] = []
    async with agent.iter(user_prompt='what is 65465-6544 * 65464-6+1.02255') as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        event_parts.append(event)

    assert agent_run.result is not None
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='what is 65465-6544 * 65464-6+1.02255',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='Be concise and always use Python to do calculations no matter how small.',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
result = 65465 - 6544 * 65464 - 6 + 1.02255
print(result)\
""",
                            'language': 'PYTHON',
                            'id': '8xju7mua',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n', 'id': '8xju7mua'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(
                        content='The result of $65465 - 6544 \\times 65464 - 6 + 1.02255$ is **-428,330,955.97745**.',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=147,
                    output_tokens=636,
                    details={
                        'thoughts_tokens': 168,
                        'tool_use_prompt_tokens': 360,
                        'text_prompt_tokens': 147,
                        'text_tool_use_prompt_tokens': 360,
                    },
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert event_parts == snapshot(
        [
            PartStartEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={
                        'code': """\
result = 65465 - 6544 * 65464 - 6 + 1.02255
print(result)\
""",
                        'language': 'PYTHON',
                        'id': '8xju7mua',
                    },
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                    provider_details={'thought_signature': IsStr()},
                ),
            ),
            PartEndEvent(
                index=0,
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={
                        'code': """\
result = 65465 - 6544 * 65464 - 6 + 1.02255
print(result)\
""",
                        'language': 'PYTHON',
                        'id': '8xju7mua',
                    },
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                    provider_details={'thought_signature': IsStr()},
                ),
                next_part_kind='builtin-tool-return',
            ),
            PartStartEvent(
                index=1,
                part=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n', 'id': '8xju7mua'},
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=2,
                part=TextPart(content='The result of $65465 - 6544 \\times 6546'),
                previous_part_kind='builtin-tool-return',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='4 - 6 + 1.02255$ is **-428,330')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=',955.97745**.')),
            PartDeltaEvent(
                index=2,
                delta=TextPartDelta(
                    content_delta='',
                    provider_name='google-gla',
                    provider_details={'thought_signature': IsStr()},
                ),
            ),
            PartEndEvent(
                index=2,
                part=TextPart(
                    content='The result of $65465 - 6544 \\times 65464 - 6 + 1.02255$ is **-428,330,955.97745**.',
                    provider_name='google-gla',
                    provider_details={'thought_signature': IsStr()},
                ),
            ),
            BuiltinToolCallEvent(  # pyright: ignore[reportDeprecated]
                part=BuiltinToolCallPart(
                    tool_name='code_execution',
                    args={
                        'code': """\
result = 65465 - 6544 * 65464 - 6 + 1.02255
print(result)\
""",
                        'language': 'PYTHON',
                        'id': '8xju7mua',
                    },
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                    provider_details={'thought_signature': IsStr()},
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n', 'id': '8xju7mua'},
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                )
            ),
        ]
    )


async def test_code_execution(allow_model_requests: None, google_model: GoogleModelFactory):
    m = google_model('gemini-3-flash-preview')
    agent = Agent(m, instructions='You are a helpful chatbot.', builtin_tools=[CodeExecutionTool()])

    result = await agent.run('What day is today in Utrecht?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What day is today in Utrecht?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful chatbot.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
from datetime import datetime
import pytz

# Get the current time in Utrecht, Netherlands (Europe/Amsterdam timezone)
utrecht_timezone = pytz.timezone('Europe/Amsterdam')
utrecht_time = datetime.now(utrecht_timezone)

# Format the date
today_date = utrecht_time.strftime("%A, %B %d, %Y")
print(f"Current day in Utrecht: {today_date}")
""",
                            'language': 'PYTHON',
                            'id': 'h0mwtrhs',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'outcome': 'OUTCOME_OK',
                            'output': 'Current day in Utrecht: Tuesday, May 05, 2026\n',
                            'id': 'h0mwtrhs',
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
import datetime
print(datetime.datetime.now())
""",
                            'language': 'PYTHON',
                            'id': '7lr99y60',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'outcome': 'OUTCOME_OK', 'output': '2026-05-05 20:40:33.367937\n', 'id': '7lr99y60'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(
                        content=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=257,
                    output_tokens=2675,
                    details={
                        'thoughts_tokens': 773,
                        'tool_use_prompt_tokens': 1732,
                        'text_prompt_tokens': 196,
                        'text_tool_use_prompt_tokens': 1732,
                    },
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    result = await agent.run('What day is tomorrow?', message_history=result.all_messages())
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What day is tomorrow?', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful chatbot.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
import datetime
print(datetime.datetime.now())
""",
                            'language': 'PYTHON',
                            'id': 'l5m4dm9r',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'outcome': 'OUTCOME_OK',
                            'output': IsStr(),
                            'id': 'l5m4dm9r',
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
import datetime
import pytz
# Utrecht is Europe/Amsterdam
tz = pytz.timezone('Europe/Amsterdam')
now = datetime.datetime.now(tz)
print(f"Time in Utrecht: {now}")
""",
                            'language': 'PYTHON',
                            'id': 'tu0hnkbw',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={
                            'outcome': 'OUTCOME_OK',
                            'output': 'Time in Utrecht: 2026-05-05 22:40:41.913103+02:00\n',
                            'id': 'tu0hnkbw',
                        },
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(
                        content='Tomorrow in Utrecht will be **Friday, May 24, 2024**.',
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=893,
                    output_tokens=4474,
                    details={
                        'thoughts_tokens': 1312,
                        'tool_use_prompt_tokens': 3056,
                        'text_prompt_tokens': 786,
                        'text_tool_use_prompt_tokens': 3056,
                    },
                ),
                model_name='gemini-3-flash-preview',
                timestamp=IsDatetime(),
                provider_name='google-gla',
                provider_url='https://generativelanguage.googleapis.com/',
                provider_details={'finish_reason': 'STOP'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_receive_history_from_another_provider(
    allow_model_requests: None, anthropic_api_key: str, gemini_api_key: str
):
    anthropic_model = AnthropicModel('claude-sonnet-4-0', provider=AnthropicProvider(api_key=anthropic_api_key))
    google_model = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key=gemini_api_key))
    agent = Agent(builtin_tools=[CodeExecutionTool()])

    result = await agent.run('How much is 3 * 12390?', model=anthropic_model)
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [[UserPromptPart], [TextPart, BuiltinToolCallPart, BuiltinToolReturnPart, TextPart]]
    )

    result = await agent.run('Multiplied by 12390', model=google_model, message_history=result.all_messages())
    assert part_types_from_messages(result.all_messages()) == snapshot(
        [
            [UserPromptPart],
            [TextPart, BuiltinToolCallPart, BuiltinToolReturnPart, TextPart],
            [UserPromptPart],
            [BuiltinToolCallPart, BuiltinToolReturnPart, BuiltinToolCallPart, BuiltinToolReturnPart, TextPart],
        ]
    )
