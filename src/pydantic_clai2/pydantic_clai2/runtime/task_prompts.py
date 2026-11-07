"""Prompt a finished delegated task on its own, with the user's message, apart from the conversation.

A task runs from a run of the session's agent: its `SubAgents` delegate tool picks the sub-agent, the tools
it inherits, its model and workspace, and the harness owner saves the task's history. So a prompt the user
sends a task is a run of the session's agent whose model requests are scripted: the first calls the
delegate tool to resume the task with the user's message, the next ends the run. The task runs in the
foreground of that run, so its report comes back to the run rather than to the conversation, and the run
is kept out of the conversation, as a `/monty` cell's is. The model the session runs on is the run's, so a
task that inherits its parent's model gets the one it ran on before.
"""

from dataclasses import dataclass

from pydantic_ai import RunContext, SkipModelRequest
from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.subagents import SubAgents

TASK_PROMPT = 'clai2_task_prompt'
"""Run metadata marking a run that prompts a task for the user, so observers can tell it from a turn."""


def is_task_prompt(ctx: RunContext[object]) -> bool:
    """Whether `ctx` is a run that prompts a task for the user rather than a turn of the conversation."""
    return bool(ctx.metadata and ctx.metadata.get(TASK_PROMPT))


def is_delegate_tool(ctx: RunContext[object], tool_def: ToolDefinition) -> bool:
    """Whether `tool_def` is the run's `SubAgents` delegate tool, whatever it is configured to be called.

    Not one of the task-control tools the same `SubAgents` adds, which show as tools rather than as a task's rows.
    """
    owner = ctx.capabilities.get(tool_def.capability_id or '')
    return isinstance(owner, SubAgents) and owner.tool_name == tool_def.name


def delegate_tool(ctx: RunContext[object]) -> str | None:
    """The name of the run's `SubAgents` delegate tool, if it has one."""
    for name, definition in ctx.tools.items():
        if is_delegate_tool(ctx, definition):
            return name
    return None


@dataclass(kw_only=True)
class TaskPrompt(AbstractCapability[object]):
    """Scripts a run's model requests: resume `task_id` on `agent_name` with `text`, then end the run.

    No model is asked. When the delegate call does not succeed, its result is kept as `refused`.
    """

    agent_name: str
    task_id: str
    text: str
    refused: str | None = None

    async def wrap_model_request(
        self, ctx: RunContext[object], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        request = request_context.messages[-1]
        assert isinstance(request, ModelRequest)
        raise SkipModelRequest(self._respond(ctx, request))

    def _respond(self, ctx: RunContext[object], request: ModelRequest) -> ModelResponse:
        for part in request.parts:
            if isinstance(part, ToolReturnPart):
                # Retried, failed, denied or interrupted, the call resumed nothing.
                if part.outcome != 'success':
                    self.refused = str(part.content)
                return ModelResponse(parts=[TextPart('done')])
        tool = delegate_tool(ctx)
        if tool is None:
            self.refused = 'This agent delegates no tasks now, so its tasks take no prompts. Turn sub-agents on first.'
            return ModelResponse(parts=[TextPart('done')])
        args = {'agent_name': self.agent_name, 'task': self.text, 'resume': self.task_id}
        return ModelResponse(parts=[ToolCallPart(tool, args)])
