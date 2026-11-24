"""AWS Lambda integration tests for `BackgroundTools`."""

from __future__ import annotations

import pytest

pytest.importorskip('aws_durable_execution_sdk_python')

from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import BackgroundTools
from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, run_durable
from tests.harness.aws_lambda.conftest import FakeDurableContext


def test_sequential_durable_run_keeps_tools_in_the_foreground() -> None:
    """AWS Lambda runs tools sequentially in a durable execution, so a background tool runs in the foreground."""
    instructions: list[str | None] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        instructions.append(info.instructions)
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('research', '{}')])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn), name='a', capabilities=[BackgroundTools(), AWSLambdaDurability()])

    @agent.tool_plain(metadata={'background': True})
    async def research() -> str:
        return 'findings'

    ctx = FakeDurableContext()
    with pytest.warns(UserWarning) as warned:
        result = run_durable(lambda: agent.run('go'), context=ctx)

    assert [str(warning.message) for warning in warned] == snapshot(
        [
            "`BackgroundTools` runs 'research' in the foreground because this run executes tools sequentially (parallel execution mode 'sequential', which a durable engine such as AWS Lambda, Absurd or pgtask sets by default), and does not tell the model about background calls."
        ]
    )
    assert result.output == 'done'
    assert instructions == [None, None]
    assert [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ] == ['findings']
    assert ctx.step_names == snapshot(
        ['a__model.request', 'a__function_toolset__<agent>.call_tool:research', 'a__model.request']
    )
