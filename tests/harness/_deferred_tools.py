"""Sub-agents whose runs end early for their caller to resume, shared by the delegation tests."""

from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent, DeferredToolRequests, PendingModelResponse
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


class BackgroundModel(FunctionModel):
    """A model whose responses the provider keeps working on, checked on every second, recording cancellations."""

    def __init__(self, *args: Any, fail_cancel: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fail_cancel = fail_cancel
        self.cancelled: list[str | None] = []

    def continuation_delay(self, response: ModelResponse) -> float | None:
        return 1.0

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        self.cancelled.append(response.provider_response_id)
        if self.fail_cancel:
            raise RuntimeError('cancel failed')


def pending_agent(
    name: str = 'background', *, fail_cancel: bool = False
) -> tuple[Agent[object, str | PendingModelResponse], BackgroundModel]:
    """A sub-agent whose model request is left waiting on the provider, so its run ends with `PendingModelResponse`."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[], provider_response_id='job-1', state='suspended')

    model = BackgroundModel(model_fn, fail_cancel=fail_cancel)
    return Agent(model, name=name, output_type=[str, PendingModelResponse]), model
