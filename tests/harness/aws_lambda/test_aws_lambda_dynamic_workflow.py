"""`DynamicWorkflow` under `AWSLambdaDurability`: gathered sub-agents run one at a time, so replay gives each call its own checkpoints."""

from __future__ import annotations

import pytest

from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, run_durable
from tests.harness.dynamic_workflow._concurrent_sub_agents import EXPECTED, build_orchestrator

from .conftest import FakeDurableContext


@pytest.mark.parametrize('durable_orchestrator', [True, False])
def test_gathered_sub_agents_replay_their_own_checkpoints(durable_orchestrator: bool) -> None:
    orchestrator = build_orchestrator(AWSLambdaDurability, durable_orchestrator=durable_orchestrator)
    first = FakeDurableContext()
    assert run_durable(lambda: orchestrator.run('go'), context=first).output == EXPECTED

    resumed = FakeDurableContext(journal=first.operations)
    assert run_durable(lambda: orchestrator.run('go'), context=resumed).output == EXPECTED
    assert resumed.invoked == []
