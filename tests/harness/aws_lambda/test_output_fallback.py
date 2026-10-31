import pytest

from tests.conftest import try_import
from tests.durable_exec._output_fallback import make_agent, make_models

with try_import() as imports_successful:
    from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, run_durable

    from .conftest import FakeDurableContext

pytestmark = pytest.mark.skipif(not imports_successful(), reason='aws-durable-execution-sdk-python not installed')


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
