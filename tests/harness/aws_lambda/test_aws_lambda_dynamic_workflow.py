"""`DynamicWorkflow` under `AWSLambdaDurability`: concurrent sub-agents replay their own checkpoints."""

from __future__ import annotations

from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, run_durable
from tests.harness.dynamic_workflow._concurrent_sub_agents import EXPECTED, build_orchestrator

from .conftest import FakeDurableContext


def test_concurrent_sub_agents_replay_their_own_checkpoints() -> None:
    orchestrator = build_orchestrator(AWSLambdaDurability)
    first = FakeDurableContext()
    assert run_durable(lambda: orchestrator.run('go'), context=first).output == EXPECTED

    resumed = FakeDurableContext(journal=first.operations)
    assert run_durable(lambda: orchestrator.run('go'), context=resumed).output == EXPECTED
    assert resumed.invoked == []
