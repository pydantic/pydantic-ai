"""`DetectLoops` under `PrefectDurability`: detection runs in flow code."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.detect_loops import DetectLoops
from tests.conftest import try_import

from ._stuck_agent import NUDGE_METADATA, nudges, read_file, stuck_model

with try_import() as imports_successful:
    from prefect import flow
    from prefect.settings import PREFECT_LOCAL_STORAGE_PATH, temporary_settings

    from pydantic_ai.durable_exec.prefect import PrefectDurability

pytestmark = [
    pytest.mark.xdist_group(name='prefect'),
    pytest.mark.skipif(not imports_successful(), reason='prefect not installed'),
]


async def test_flow_delivers_the_nudge(prefect_test_server: None, tmp_path: Path) -> None:
    agent = Agent(
        FunctionModel(stuck_model),
        name='detect_loops_prefect',
        tools=[read_file],
        capabilities=[DetectLoops(), PrefectDurability()],
    )

    @flow
    async def run() -> list[ModelMessage]:
        return (await agent.run('Summarize missing.txt')).all_messages()

    with temporary_settings({PREFECT_LOCAL_STORAGE_PATH: tmp_path}):
        messages = await run()

    assert nudges(messages) == [NUDGE_METADATA]
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].text == 'missing.txt does not exist.'
