from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai.messages import ModelResponse
from tests.conftest import try_import
from tests.durable_exec._output_fallback import make_agent, make_models

with try_import() as imports_successful:
    from prefect import flow
    from prefect.settings import PREFECT_LOCAL_STORAGE_PATH, temporary_settings

    from pydantic_ai.durable_exec.prefect import PrefectDurability

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='prefect not installed'),
    pytest.mark.xdist_group(name='prefect'),
]


@pytest.mark.parametrize('blockbuster_enabled', [False])
async def test_flow_retry_reconstructs_selection(
    prefect_test_server: None, tmp_path: Path, blockbuster_enabled: bool
) -> None:
    """Prefect's flow-source discovery and result persistence perform synchronous filesystem I/O."""
    assert blockbuster_enabled is False
    executions: list[str] = []
    models = make_models(executions)
    agent = make_agent(models, PrefectDurability(models=models), executions)
    attempts: list[list[str | None]] = []

    @flow(retries=1)
    async def run() -> str:
        result = await agent.run('Answer.')
        attempts.append([m.model_name for m in result.all_messages() if isinstance(m, ModelResponse)])
        if len(attempts) == 1:
            raise RuntimeError('Retry flow.')
        return result.output

    with temporary_settings({PREFECT_LOCAL_STORAGE_PATH: tmp_path}):
        assert await run() == 'ready'
    assert attempts == [['primary', 'primary', 'backup']] * 2
    assert executions == ['primary', 'lookup', 'primary', 'backup']
