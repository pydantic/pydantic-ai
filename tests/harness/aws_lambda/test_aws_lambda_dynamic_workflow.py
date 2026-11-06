"""`DynamicWorkflow` under `AWSLambdaDurability`: concurrent sub-agents replay their own checkpoints."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, run_durable
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow
from tests.harness.dynamic_workflow._concurrent_sub_agents import EXPECTED, noop, orchestrator_model, summarize_model

from .conftest import FakeDurableContext


def test_concurrent_sub_agents_replay_their_own_checkpoints() -> None:
    summarize = Agent(
        FunctionModel(summarize_model),
        name='summarize',
        description='Summarize a text.',
        tools=[noop],
        capabilities=[AWSLambdaDurability()],
    )
    orchestrator = Agent(
        FunctionModel(orchestrator_model),
        name='orchestrator',
        capabilities=[DynamicWorkflow(agents=[summarize]), AWSLambdaDurability()],
    )
    first = FakeDurableContext()
    assert run_durable(lambda: orchestrator.run('go'), context=first).output == EXPECTED

    resumed = FakeDurableContext(journal=first.operations)
    assert run_durable(lambda: orchestrator.run('go'), context=resumed).output == EXPECTED
    assert resumed.invoked == []
