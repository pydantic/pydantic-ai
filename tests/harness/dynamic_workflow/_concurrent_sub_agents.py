"""Two concurrent sub-agents whose steps start in a different order on the first run than on replay."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

SCRIPT = "import asyncio\nawait asyncio.gather(summarize(task='alpha'), summarize(task='beta'))"
EXPECTED = "done: ['summary of alpha', 'summary of beta']"


async def _summarize_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    (task,) = [part.content for part in messages[0].parts if isinstance(part, UserPromptPart)]
    if len(messages) == 1:
        if task == 'alpha':
            # Let beta's second step start before alpha's, unlike the replay, where no model runs.
            await asyncio.sleep(0.05)
        return ModelResponse(parts=[ToolCallPart('noop')])
    return ModelResponse(parts=[TextPart(f'summary of {task}')])


def _orchestrator_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
    if not returns:
        return ModelResponse(parts=[ToolCallPart('run_workflow', {'code': SCRIPT})])
    return ModelResponse(parts=[TextPart(f'done: {returns[0].content}')])


def noop() -> str:
    return 'ok'


def build_orchestrator(
    durability: Callable[[], AbstractCapability[Any]], *, durable_orchestrator: bool = True
) -> Agent[None, str]:
    """An orchestrator whose script gathers two `summarize` sub-agents, which carry `durability()`."""
    summarize = Agent(
        FunctionModel(_summarize_model),
        name='summarize',
        description='Summarize a text.',
        tools=[noop],
        capabilities=[durability()],
    )
    return Agent(
        FunctionModel(_orchestrator_model),
        name='orchestrator',
        capabilities=[DynamicWorkflow(agents=[summarize]), *([durability()] if durable_orchestrator else [])],
    )
