"""Spills retain a dedicated bounded reader even when file tools are registered."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.tool_output_limits import READ_TOOL_NAME, Band, Spill, ToolOutputLimits


async def test_read_tool_result_offered_with_file_system(tmp_path: Path) -> None:
    payload = '\n'.join(f'line {i}' for i in range(500))
    offered: list[set[str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.append({tool.name for tool in info.function_tools})
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if read := [part for part in returns if part.tool_name == READ_TOOL_NAME]:
            return ModelResponse(parts=[TextPart(str(read[0].content))])
        if spilled := [part for part in returns if part.tool_name == 'big_tool']:
            assert spilled[0].metadata is not None
            return ModelResponse(
                parts=[ToolCallPart(READ_TOOL_NAME, {'handle': spilled[0].metadata['overflow_handle']})]
            )
        return ModelResponse(parts=[ToolCallPart('big_tool', {})])

    agent = Agent(
        FunctionModel(respond),
        capabilities=[ToolOutputLimits(bands=[Band(over=100, action=Spill())]), LocalWorkspace(tmp_path), FileSystem()],
    )

    @agent.tool_plain
    def big_tool() -> str:
        return payload

    result = await agent.run('go')
    assert 'line 100' in result.output
    assert all(READ_TOOL_NAME in tools for tools in offered)
    spilled = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'big_tool'
    ]
    assert len(spilled) == 1
    assert f'Read it with {READ_TOOL_NAME}(' in str(spilled[0].content)
