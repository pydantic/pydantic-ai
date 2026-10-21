"""A sub-agent whose run stops on a tool call that needs approval, shared by the delegation tests."""

from collections.abc import Sequence

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def deferring_agent(
    name: str = 'approver', *, capabilities: Sequence[AbstractCapability[object]] = ()
) -> Agent[object, str | DeferredToolRequests]:
    """A sub-agent that calls a tool needing approval, then finishes once the tool has returned.

    Without a capability that resolves the call, its run ends with `DeferredToolRequests`.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart('approved and done')])
        return ModelResponse(parts=[ToolCallPart('dangerous', {}, tool_call_id='d1')])

    agent = Agent(
        FunctionModel(model_fn), name=name, output_type=[str, DeferredToolRequests], capabilities=list(capabilities)
    )

    @agent.tool_plain(requires_approval=True)
    def dangerous() -> str:
        return 'done'

    return agent
