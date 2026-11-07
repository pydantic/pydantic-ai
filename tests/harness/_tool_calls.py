"""Drive tool calls through `Agent(capabilities=[...])` and return what the model saw back."""

from collections.abc import AsyncIterator, Sequence

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.workspaces import WorkspaceBackend

WORKSPACE_TOOL_CALLS: list[tuple[str, dict[str, object]]] = [
    ('write_file', {'path': 'note.txt', 'content': 'one'}),
    ('edit_file', {'path': 'note.txt', 'old_text': 'one', 'new_text': 'two'}),
    ('read_file', {'path': 'note.txt'}),
    ('shell', {'command': 'echo shell-$(cat note.txt)'}),
]
"""FileSystem and Shell tools that emit capability events, ending with the shell printing `shell-two`."""


def scripted_model(calls: Sequence[tuple[str, dict[str, object]]]) -> FunctionModel:
    """Make each call in turn, then answer with every tool result."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        turn = sum(isinstance(message, ModelResponse) for message in messages)
        if turn < len(calls):
            name, arguments = calls[turn]
            return ModelResponse(parts=[ToolCallPart(name, arguments)])
        returns = [str(part.content) for m in messages for part in m.parts if isinstance(part, ToolReturnPart)]
        return ModelResponse(parts=[TextPart('\n'.join(returns) or 'done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        for part in respond(messages, info).parts:
            if isinstance(part, ToolCallPart):
                yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}
            else:
                assert isinstance(part, TextPart)
                yield part.content

    return FunctionModel(respond, stream_function=stream)


async def call_tools(
    capabilities: Sequence[AbstractCapability[None]],
    calls: Sequence[tuple[str, dict[str, object]]],
    *,
    workspace: WorkspaceBackend | None = None,
) -> list[str]:
    """Make each call in turn within one run, then finish; return each tool result or retry prompt text."""
    agent = Agent(scripted_model(calls), deps_type=type(None), capabilities=capabilities)
    result = await agent.run('Use the tools', workspace=workspace)
    return [
        str(part.content)
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
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
