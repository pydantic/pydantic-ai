"""Drive tool calls through `Agent(capabilities=[...])` and return what the model saw back."""

from collections.abc import AsyncIterator, Sequence

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.workspaces import WorkspaceBackend


async def call_tools(
    capabilities: Sequence[AbstractCapability[None]],
    calls: Sequence[tuple[str, dict[str, object]]],
    *,
    workspace: WorkspaceBackend | None = None,
) -> list[str]:
    """Make each call in turn within one run, then finish; return each tool result or retry prompt text."""
    turn = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal turn
        turn += 1
        if turn <= len(calls):
            name, arguments = calls[turn - 1]
            return ModelResponse(parts=[ToolCallPart(name, arguments)])
        return ModelResponse(parts=[TextPart('done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        for part in respond(messages, info).parts:
            if isinstance(part, ToolCallPart):
                yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}
            else:
                assert isinstance(part, TextPart)
                yield part.content

    agent = Agent(FunctionModel(respond, stream_function=stream), deps_type=type(None), capabilities=capabilities)
    result = await agent.run('Use the tools', workspace=workspace)
    return [
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, (ToolReturnPart, RetryPromptPart))
    ]


async def call_tool(
    capabilities: Sequence[AbstractCapability[None]],
    name: str,
    arguments: dict[str, object],
    *,
    workspace: WorkspaceBackend | None = None,
) -> str:
    """Call `name` with `arguments` once, then finish; return the tool result or retry prompt text."""
    return '\n'.join(await call_tools(capabilities, [(name, arguments)], workspace=workspace))
