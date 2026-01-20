"""VCR test for CodeModeToolset with Logfire MCP server."""

from __future__ import annotations

import os
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.messages import RetryPromptPart
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsInstance, IsInt, IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai.mcp import MCPServerStdio
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.toolsets.code_mode import CodeModeToolset
    from pydantic_ai.toolsets.combined import CombinedToolset
    from pydantic_ai.toolsets.function import FunctionToolset

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mcp and openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.fixture
def logfire_read_token() -> str:
    return os.getenv('LOGFIRE_READ_TOKEN', 'mock-logfire-token')


@pytest.fixture
def logfire_server(logfire_read_token: str) -> MCPServerStdio:
    return MCPServerStdio('uvx', args=['logfire-mcp@latest', '--read-token', logfire_read_token])


async def test_code_mode_with_logfire_mcp(
    allow_model_requests: None, openai_api_key: str, logfire_server: MCPServerStdio
):
    """CodeModeToolset + Logfire MCP + user-defined tool."""
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    agent: Agent[None, str] = Agent(model)

    user_toolset: FunctionToolset[None] = FunctionToolset()

    def format_as_csv(data: list[dict[str, Any]]) -> str:
        """Convert data to CSV format. Each dict becomes a row with its values."""
        if not data:
            return ''
        headers = list(data[0].keys())
        lines = [','.join(headers)]
        for row in data:
            lines.append(','.join(str(row.get(h, '')) for h in headers))
        return '\n'.join(lines)

    user_toolset.add_function(format_as_csv, takes_ctx=False)

    combined = CombinedToolset([logfire_server, user_toolset])
    code_mode = CodeModeToolset(wrapped=combined)

    async with code_mode:
        result = await agent.run(
            'Use arbitrary_query to get the 5 most recent traces with columns: span_name, duration_ms. '
            'Then format the results as CSV using format_as_csv. Return the CSV.',
            toolsets=[code_mode],
        )

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Use arbitrary_query to get the 5 most recent traces with columns: span_name, duration_ms. Then format the results as CSV using format_as_csv. Return the CSV.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='run_code',
                        args='{"code": "arbitrary_query(\'SELECT span_name, duration_ms FROM records ORDER BY start_timestamp DESC LIMIT 5\', 0)"}',
                        tool_call_id='call_kZbAWU3Wda57VwmNGaX9dq89',
                    ),
                    ToolCallPart(
                        tool_name='run_code',
                        args='{"code": "schema_reference()"}',
                        tool_call_id='call_dhj91eYWX1rWUD2S7sy9M0wB',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=505,
                    output_tokens=69,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-CzsedMwYl5TGRESvye7y43pFJG2Q5',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Error executing tool arbitrary_query: b\'{"detail":"Invalid token"}\'',
                        tool_name='run_code',
                        tool_call_id='call_kZbAWU3Wda57VwmNGaX9dq89',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='run_code',
                        content='',
                        tool_call_id='call_dhj91eYWX1rWUD2S7sy9M0wB',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="The query to retrieve the 5 most recent traces failed because the column `duration_ms` does not exist. Instead, you should use the `duration` column, which is of type `DOUBLE PRECISION`. Let's correct that and try again to get the results in CSV format."
                    ),
                    ToolCallPart(
                        tool_name='run_code',
                        args='{"code": "arbitrary_query(\'SELECT span_name, duration FROM records ORDER BY start_timestamp DESC LIMIT 5\', 0)"}',
                        tool_call_id='call_9CyU1XEZisIWlqRqQZR9b8lU',
                    ),
                    ToolCallPart(
                        tool_name='run_code',
                        args='{"code": "schema_reference()"}',
                        tool_call_id='call_SyCvqNJDTOTBAqILcnlVRIpQ',
                    ),
                ],
                usage=RequestUsage(
                    input_tokens=1670,
                    output_tokens=125,
                    details={
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 0,
                        'rejected_prediction_tokens': 0,
                    },
                ),
                model_name='gpt-4o-2024-08-06',
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details={'finish_reason': 'tool_calls', 'timestamp': IsDatetime()},
                provider_response_id='chatcmpl-CzsefQQUdHqT9Lb7zw8x4RcetxApD',
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Error executing tool arbitrary_query: b\'{"detail":"Invalid token"}\'',
                        tool_name='run_code',
                        tool_call_id='call_9CyU1XEZisIWlqRqQZR9b8lU',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='run_code',
                        content='',
                        tool_call_id='call_SyCvqNJDTOTBAqILcnlVRIpQ',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr())],
                usage=RequestUsage(
                    input_tokens=IsInt(),
                    cache_read_tokens=1792,
                    output_tokens=IsInt(),
                    details=IsInstance(dict),  # pyright: ignore[reportUnknownArgumentType]
                ),
                model_name=IsStr(),
                timestamp=IsDatetime(),
                provider_name='openai',
                provider_url='https://api.openai.com/v1/',
                provider_details=IsInstance(dict),  # pyright: ignore[reportUnknownArgumentType]
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )
