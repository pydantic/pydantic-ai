"""Guardrails under a real durable engine: a verdict is recorded, not re-evaluated on replay."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec._codec import JSON_CODEC
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.guardrails import (
    GuardrailResult,
    InputGuardrail,
    OutputGuardrail,
    ToolCallInfo,
    ToolGuardrail,
    ToolResultInfo,
)
from tests.conftest import detach_dbos_logging

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_guardrails',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
        'run_admin_server': False,
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()
        detach_dbos_logging()


_calls: list[str] = []


def _input_guard(prompt: str) -> GuardrailResult:
    _calls.append('input')
    return GuardrailResult.replace(prompt.replace('secret', '[redacted]'))


def _output_guard(output: object) -> GuardrailResult:
    _calls.append('output')
    return GuardrailResult.replace(f'{output}!')


def _args_guard(call: ToolCallInfo) -> bool:
    _calls.append('args')
    return True


def _result_guard(result: ToolResultInfo) -> GuardrailResult:
    _calls.append('result')
    return GuardrailResult.replace(f'{result.result} (screened)')


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    returns = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    if returns:
        return ModelResponse(parts=[TextPart(str(returns[0].content))])
    return ModelResponse(parts=[ToolCallPart('lookup', {'key': 'a'}, tool_call_id='call-1')])


_agent: Agent[None, str] = Agent(
    FunctionModel(_respond),
    name='durable_guardrails',
    deps_type=type(None),
    capabilities=[
        InputGuardrail(guard=_input_guard, id='input'),
        OutputGuardrail(guard=_output_guard, id='output'),
        ToolGuardrail(guard=_args_guard, result_guard=_result_guard, id='tool'),
        DBOSDurability[None](),
    ],
)


@_agent.tool_plain
def lookup(key: str) -> str:
    return f'value of {key}'


@DBOS.workflow(name='durable_guardrails')
async def _workflow() -> str:
    return (await _agent.run('my secret question')).output


async def test_dbos_replays_the_recorded_verdicts(dbos: DBOS) -> None:
    """A replay reuses each recorded verdict: no guard is called again, and the outcome is the same."""
    _calls.clear()
    workflow_id = str(uuid.uuid4())

    with SetWorkflowID(workflow_id):
        assert await _workflow() == 'value of a (screened)!'
    assert _calls == ['input', 'args', 'result', 'output']
    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert [step['function_name'] for step in steps] == [
        'durable_guardrails__capability__input.guard',
        'durable_guardrails__model.request',
        'durable_guardrails__capability__tool.guard',
        'durable_guardrails__capability__tool.result_guard',
        'durable_guardrails__model.request',
        'durable_guardrails__capability__output.guard',
    ]

    # Re-execute the workflow function past its last step, the way recovery does.
    handle = await DBOS.fork_workflow_async(workflow_id, len(steps) + 1)
    assert await handle.get_result(polling_interval_sec=0.01) == 'value of a (screened)!'

    assert _calls == ['input', 'args', 'result', 'output'], 'the replay called the guards again'


@pytest.mark.parametrize(
    'capability',
    [InputGuardrail(guard=_input_guard), OutputGuardrail(guard=_output_guard), ToolGuardrail(guard=_args_guard)],
    ids=['input', 'output', 'tool'],
)
def test_a_guardrail_without_an_id_cannot_bind_to_a_durable_agent(capability: AbstractCapability[None]) -> None:
    with pytest.raises(UserError, match='needs an explicit `id`'):
        Agent(
            FunctionModel(_respond),
            name='unnamed',
            deps_type=type(None),
            capabilities=[capability, DBOSDurability[None]()],
        )


@pytest.mark.parametrize(
    'verdict',
    [GuardrailResult.allow(), GuardrailResult.block(), GuardrailResult.replace(None), GuardrailResult.retry('again')],
)
def test_a_verdict_round_trips_through_a_json_journal(verdict: GuardrailResult) -> None:
    assert JSON_CODEC.load(GuardrailResult, JSON_CODEC.dump(GuardrailResult, verdict)) == verdict
