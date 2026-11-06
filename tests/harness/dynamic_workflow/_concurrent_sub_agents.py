"""Two concurrent sub-agents whose steps start in a different order on the first run than on replay."""

from __future__ import annotations

import asyncio

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo

SCRIPT = "import asyncio\nawait asyncio.gather(summarize(task='alpha'), summarize(task='beta'))"
EXPECTED = "done: ['summary of alpha', 'summary of beta']"


async def summarize_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    (task,) = [part.content for part in messages[0].parts if isinstance(part, UserPromptPart)]
    if len(messages) == 1:
        if task == 'alpha':
            # Let beta's second step start before alpha's, unlike the replay, where no model runs.
            await asyncio.sleep(0.05)
        return ModelResponse(parts=[ToolCallPart('noop')])
    return ModelResponse(parts=[TextPart(f'summary of {task}')])


def orchestrator_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
    if not returns:
        return ModelResponse(parts=[ToolCallPart('run_workflow', {'code': SCRIPT})])
    return ModelResponse(parts=[TextPart(f'done: {returns[0].content}')])


def noop() -> str:
    return 'ok'
