"""Run a `ToolOutputLimits` spill query the way a model would, through an agent."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.tool_output_limits import (
    GREP_TOOL_NAME,
    GREP_TOOL_RESULT_KIND,
    READ_TOOL_NAME,
    READ_TOOL_RESULT_KIND,
    Band,
    LocalFileOverflowStore,
    Spill,
    ToolOutputLimits,
)


async def query_spill(
    tmp_path: Path, payload: str, tool_name: str, args: dict[str, object], handle: str | None = None
) -> str:
    """Spill `payload`, call `tool_name` on its handle with `args`, and return what the model got back."""
    offered: dict[str, str | None] = {}
    called: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.update({tool.name: tool.tool_kind for tool in info.function_tools})
        parts = [part for message in messages for part in message.parts]
        if answer := [part for part in parts if isinstance(part, ToolReturnPart) and part.tool_name == tool_name]:
            retried = 'retry: ' if answer[0].outcome == 'retried' else ''
            return ModelResponse(parts=[TextPart(f'{retried}{answer[0].content}')])
        if spilled := [part for part in parts if isinstance(part, ToolReturnPart)]:
            assert spilled[0].metadata is not None
            stored = handle if handle is not None else spilled[0].metadata['overflow_handle']
            called.append(stored)
            return ModelResponse(parts=[ToolCallPart(tool_name, {'handle': stored, **args})])
        return ModelResponse(parts=[ToolCallPart('big_tool', {})])

    limits: ToolOutputLimits[None] = ToolOutputLimits(
        bands=[Band(over=50, action=Spill())], store=LocalFileOverflowStore(base_dir=tmp_path)
    )
    agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[limits])

    @agent.tool_plain
    def big_tool() -> str:
        return payload

    result = await agent.run('go')
    assert offered[READ_TOOL_NAME] == READ_TOOL_RESULT_KIND.id
    assert offered[GREP_TOOL_NAME] == GREP_TOOL_RESULT_KIND.id
    return result.output.replace(f'handle={called[0]!r}', "handle='<h>'")
