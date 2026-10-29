import pytest

pytest.importorskip('aws_durable_execution_sdk_python')

from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, run_durable
from tests.harness.escalate_on_output_retry._scenario import make_agent, make_models

from .conftest import FakeDurableContext


def test_replay_reconstructs_selection() -> None:
    executions: list[str] = []
    models = make_models(executions)
    first = FakeDurableContext()
    agent = make_agent(models, AWSLambdaDurability(models=models), executions)
    assert run_durable(lambda: agent.run('Answer.'), context=first).output == 'ready'
    recovered = make_agent(models, AWSLambdaDurability(models=models), executions)
    replay = FakeDurableContext(journal=first.operations)
    assert run_durable(lambda: recovered.run('Answer.'), context=replay).output == 'ready'
    assert replay.invoked == []
    assert executions == ['primary', 'lookup', 'primary', 'backup']
