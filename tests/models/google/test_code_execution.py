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

from ...conftest import IsDatetime, IsNow, IsStr
from ...parts_from_messages import part_types_from_messages

if TYPE_CHECKING:
    from .conftest import GoogleModelFactory

pytestmark = [
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
                            'id': 'gzl5c0n7',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n', 'id': 'gzl5c0n7'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                        provider_name='google-gla',
                    ),
                    TextPart(
                        content='The result of $65465 - 6544 \\times 65464 - 6 + 1.02255$ is **-428,330,955.97745**.'
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=213,
                    output_tokens=483,
                    details={
                        'tool_use_prompt_tokens': 426,
                        'text_prompt_tokens': 213,
                        'text_tool_use_prompt_tokens': 426,
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
                        'id': 'gzl5c0n7',
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
                        'id': 'gzl5c0n7',
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
                    content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n', 'id': 'gzl5c0n7'},
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                    provider_name='google-gla',
                ),
                previous_part_kind='builtin-tool-call',
            ),
            PartStartEvent(
                index=2,
                part=TextPart(content='The result of'),
                previous_part_kind='builtin-tool-return',
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta=' $65465 - 6544 \\times 654')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='64 - 6 + 1.02255$ is **-428,33')),
            PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='0,955.97745**.')),
            PartEndEvent(
                index=2,
                part=TextPart(
                    content='The result of $65465 - 6544 \\times 65464 - 6 + 1.02255$ is **-428,330,955.97745**.'
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
                        'id': 'gzl5c0n7',
                    },
                    tool_call_id=IsStr(),
                    provider_name='google-gla',
                    provider_details={'thought_signature': IsStr()},
                )
            ),
            BuiltinToolResultEvent(  # pyright: ignore[reportDeprecated]
                result=BuiltinToolReturnPart(
                    tool_name='code_execution',
                    content={'outcome': 'OUTCOME_OK', 'output': '-428330955.97745\n', 'id': 'gzl5c0n7'},
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
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
from datetime import datetime
import pytz

# Get the timezone for Utrecht
utrecht_tz = pytz.timezone('Europe/Amsterdam')

# Get the current time in Utrecht
utrecht_now = datetime.now(utrecht_tz)

# Format the date
print(utrecht_now.strftime('%A, %B %d, %Y'))
""",
                            'language': 'PYTHON',
                            'id': '2biwo6yl',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'outcome': 'OUTCOME_OK', 'output': 'Friday, April 17, 2026\n', 'id': '2biwo6yl'},
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
                            'id': 'yskzpivu',
                        },
                        tool_call_id=IsStr(),
                        provider_name='google-gla',
                        provider_details={'thought_signature': IsStr()},
                    ),
                    BuiltinToolReturnPart(
                        tool_name='code_execution',
                        content={'outcome': 'OUTCOME_OK', 'output': '2026-04-17 19:38:50.433087\n', 'id': 'yskzpivu'},
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
                    input_tokens=94,
                    output_tokens=3559,
                    details={
                        'thoughts_tokens': 1761,
                        'tool_use_prompt_tokens': 1760,
                        'text_prompt_tokens': 94,
                        'text_tool_use_prompt_tokens': 1760,
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
            ),
            ModelResponse(
                parts=[
                    BuiltinToolCallPart(
                        tool_name='code_execution',
                        args={
                            'code': """\
from datetime import datetime, timedelta
import pytz

utrecht_tz = pytz.timezone('Europe/Amsterdam')
utrecht_now = datetime.now(utrecht_tz)
tomorrow = utrecht_now + timedelta(days=1)

print(f"Today: {utrecht_now.strftime('%A, %B %d, %Y')}")
print(f"Tomorrow: {tomorrow.strftime('%A, %B %d, %Y')}")
""",
                            'language': 'PYTHON',
                            'id': 'x9aranmq',
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
                            'id': 'x9aranmq',
                        },
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
                    input_tokens=728,
                    output_tokens=2178,
                    details={
                        'thoughts_tokens': 986,
                        'tool_use_prompt_tokens': 1173,
                        'text_prompt_tokens': 728,
                        'text_tool_use_prompt_tokens': 1173,
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
            ),
        ]
    )


async def test_receive_history_from_another_provider(
    allow_model_requests: None, anthropic_api_key: str, gemini_api_key: str
):
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.google import GoogleProvider

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
            [BuiltinToolCallPart, BuiltinToolReturnPart, TextPart],
        ]
    )
