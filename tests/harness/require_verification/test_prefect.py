"""`RequireVerification` under `PrefectDurability`: the redirect runs in flow code."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.require_verification import RequireVerification
from tests.conftest import try_import

from ._editing_agent import attempts, editing_model, tools

with try_import() as imports_successful:
    from prefect import flow
    from prefect.settings import PREFECT_LOCAL_STORAGE_PATH, temporary_settings

    from pydantic_ai.durable_exec.prefect import PrefectDurability

pytestmark = [
    pytest.mark.xdist_group(name='prefect'),
    pytest.mark.skipif(not imports_successful(), reason='prefect not installed'),
]


async def test_flow_redirects_an_unverified_edit(prefect_test_server: None, tmp_path: Path) -> None:
    agent = Agent(
        FunctionModel(editing_model),
        name='require_verification_prefect',
        tools=tools,
        capabilities=[RequireVerification(), PrefectDurability()],
    )

    @flow
    async def run() -> list[ModelMessage]:
        return (await agent.run('Fix the bug')).all_messages()

    with temporary_settings({PREFECT_LOCAL_STORAGE_PATH: tmp_path}):
        messages = await run()

    assert attempts(messages) == [1]
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].text == 'Verified.'
