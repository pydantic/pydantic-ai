from __future__ import annotations

from collections.abc import Callable

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import EscalateOnOutputRetry, EscalationLevel


def make_models(executions: list[str]) -> dict[str, Model]:
    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        executions.append('primary')
        if not any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[ToolCallPart('lookup', {})])
        return ModelResponse(parts=[TextPart('draft')])

    def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        executions.append('backup')
        return ModelResponse(parts=[TextPart('ready')])

    return {
        'primary': FunctionModel(primary, model_name='primary'),
        'backup': FunctionModel(backup, model_name='backup'),
    }


def make_agent(
    models: dict[str, Model],
    durability: AbstractCapability[object],
    executions: list[str],
    lookup_tool: Callable[[], str] | None = None,
) -> Agent[object, str]:
    def lookup() -> str:
        executions.append('lookup')
        return 'evidence'

    agent = Agent(
        name='output_escalation',
        deps_type=object,
        tools=[lookup_tool or lookup],
        capabilities=[
            EscalateOnOutputRetry(
                levels=[EscalationLevel(models['primary'], after_failures=1), EscalationLevel(models['backup'])]
            ),
            durability,
        ],
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    return agent
