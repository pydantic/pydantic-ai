"""Two sub-agents gathered by a script, where the one dispatched first finishes last."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

SCRIPT = "import asyncio\nawait asyncio.gather(summarize(task='alpha'), summarize(task='beta'))"
EXPECTED = "done: ['summary of alpha', 'summary of beta']"


def summarize_model(calls: list[str], on_call: Callable[[str], None] | None = None) -> FunctionModel:
    """A sub-agent model that records its requests in `calls` as `'<task> <message count>'`, then calls `on_call`.

    Alpha's first request waits until beta has finished, so the two must run concurrently, and
    alpha, dispatched first, finishes last. A dispatch that runs them one at a time fails on the
    timeout rather than hanging.
    """
    beta_done = asyncio.Event()

    async def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        (task,) = [part.content for part in messages[0].parts if isinstance(part, UserPromptPart)]
        calls.append(f'{task} {len(messages)}')
        if on_call is not None:
            on_call(calls[-1])
        if len(messages) > 1:
            if task == 'beta':
                beta_done.set()
            return ModelResponse(parts=[TextPart(f'summary of {task}')])
        if task == 'alpha':
            await asyncio.wait_for(beta_done.wait(), timeout=5)
        return ModelResponse(parts=[ToolCallPart('noop')])

    return FunctionModel(summarize)


def _orchestrator_model(script: str) -> FunctionModel:
    def orchestrate(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart('run_workflow', {'code': script})])
        return ModelResponse(parts=[TextPart(f'done: {returns[0].content}')])

    return FunctionModel(orchestrate)


def noop() -> str:
    return 'ok'


def build_orchestrator(
    durability: Callable[[], AbstractCapability[Any]],
    summarize: FunctionModel,
    *,
    durable_orchestrator: bool = True,
    durable_sub_agents: bool = True,
    script: str = SCRIPT,
    tools: Sequence[Tool[None]] | None = None,
) -> Agent[None, str]:
    """An orchestrator that runs `script`, by default gathering two `summarize` sub-agents, which carry `durability()`.

    The sub-agents get `tools`, by default a `noop` tool.
    """
    summarize_agent = Agent(
        summarize,
        name='summarize',
        description='Summarize a text.',
        tools=list(tools) if tools is not None else [noop],
        capabilities=[durability()] if durable_sub_agents else [],
    )
    return Agent(
        _orchestrator_model(script),
        name='orchestrator',
        capabilities=[DynamicWorkflow(agents=[summarize_agent]), *([durability()] if durable_orchestrator else [])],
    )
