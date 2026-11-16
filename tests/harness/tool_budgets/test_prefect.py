"""`ToolBudgets` under `PrefectDurability`: budgets are counted in flow code."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness import ToolBudgets
from tests.conftest import try_import

from ._search_agent import refusal, refusals, search, searches, searching_model

with try_import() as imports_successful:
    from prefect import flow
    from prefect.settings import PREFECT_LOCAL_STORAGE_PATH, temporary_settings

    from pydantic_ai.durable_exec.prefect import PrefectDurability

pytestmark = [
    pytest.mark.xdist_group(name='prefect'),
    pytest.mark.skipif(not imports_successful(), reason='prefect not installed'),
]


async def test_flow_refuses_the_call_over_budget(prefect_test_server: None, tmp_path: Path) -> None:
    agent = Agent(
        FunctionModel(searching_model),
        name='tool_budgets_prefect',
        tools=[search],
        capabilities=[ToolBudgets(per_tool={'search': 2}), PrefectDurability()],
    )

    @flow
    async def run() -> list[ModelMessage]:
        return (await agent.run('Research it')).all_messages()

    searches.clear()
    with temporary_settings({PREFECT_LOCAL_STORAGE_PATH: tmp_path}):
        messages = await run()

    assert refusals(messages) == [refusal('per_tool')]
    assert searches == ['q0', 'q1']
