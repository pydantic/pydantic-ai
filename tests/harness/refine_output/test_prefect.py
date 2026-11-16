"""`RefineOutput` under `PrefectDurability`: the review runs in flow code."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.refine_output import DEFAULT_REFINE_OUTPUT_INSTRUCTIONS, RefineOutput
from tests.conftest import try_import

from ._drafting_agent import drafting_model, reviews

with try_import() as imports_successful:
    from prefect import flow
    from prefect.settings import PREFECT_LOCAL_STORAGE_PATH, temporary_settings

    from pydantic_ai.durable_exec.prefect import PrefectDurability

pytestmark = [
    pytest.mark.xdist_group(name='prefect'),
    pytest.mark.skipif(not imports_successful(), reason='prefect not installed'),
]


async def test_flow_reviews_the_output(prefect_test_server: None, tmp_path: Path) -> None:
    agent = Agent(
        FunctionModel(drafting_model), name='refine_output_prefect', capabilities=[RefineOutput(), PrefectDurability()]
    )

    @flow
    async def run() -> list[ModelMessage]:
        return (await agent.run('Explain the change')).all_messages()

    with temporary_settings({PREFECT_LOCAL_STORAGE_PATH: tmp_path}):
        messages = await run()

    assert reviews(messages) == [(DEFAULT_REFINE_OUTPUT_INSTRUCTIONS, 1)]
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].text == 'revised'
