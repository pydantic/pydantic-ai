"""`DetectLoops` under `PrefectDurability`: detection runs in flow code, and a loop judge is never asked there."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.detect_loops import DetectLoops, LoopJudge
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


async def test_a_judge_is_not_asked_in_flow_code(prefect_test_server: None, tmp_path: Path) -> None:
    responses = iter([ModelResponse(parts=[TextPart('I should check the config file again. ' * 80)])])

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return next(responses, ModelResponse(parts=[TextPart('The config is fine.')]))

    def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # A judge asked here would overrule the heuristic, so the nudge below proves it wasn't asked.
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'looping': False})])  # pragma: no cover

    agent = Agent(
        FunctionModel(model),
        name='detect_loops_prefect_judge',
        capabilities=[DetectLoops(stream_detector=LoopJudge(model=FunctionModel(judge))), PrefectDurability()],
    )

    @flow
    async def run() -> list[ModelMessage]:
        return (await agent.run('Is the config fine?')).all_messages()

    with temporary_settings({PREFECT_LOCAL_STORAGE_PATH: tmp_path}):
        messages = await run()

    # The heuristic alone found the loop in the response, checked whole in flow code.
    assert [nudge['pattern'] for nudge in nudges(messages) if isinstance(nudge, dict)] == ['repeated_stream_text']
    assert isinstance(messages[-1], ModelResponse)
    assert messages[-1].text == 'The config is fine.'
