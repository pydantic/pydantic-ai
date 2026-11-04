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

from pydantic import BaseModel

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec import JSON_CODEC
from pydantic_ai.exceptions import ApprovalRequired, UserError
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
from pydantic_ai_harness.guardrails._exceptions import InputBlocked, ToolBlocked
from tests.conftest import detach_dbos_logging
from tests.harness._recording_durability import RecordingDurability

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
    [
        GuardrailResult.allow(),
        GuardrailResult.block(),
        GuardrailResult.replace(None),
        GuardrailResult.replace('x'),
        GuardrailResult.retry('again'),
        GuardrailResult.approve(),
    ],
)
def test_a_verdict_round_trips_through_a_json_journal(verdict: GuardrailResult) -> None:
    assert JSON_CODEC.load(GuardrailResult, JSON_CODEC.dump(GuardrailResult, verdict)) == verdict


class Record(BaseModel):
    name: str
    ssn: str | None


async def test_a_replacement_keeps_its_type_through_a_json_journal() -> None:
    """A replacement comes back from a JSON journal as plain data and is rebuilt as the value it replaces."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if returns:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'name': 'a', 'ssn': '123'})])
        return ModelResponse(parts=[ToolCallPart('lookup', {'key': 'a'})])

    agent = Agent(
        FunctionModel(respond),
        name='json_guardrails',
        output_type=Record,
        capabilities=[
            OutputGuardrail(guard=lambda output: GuardrailResult.replace(Record(name='a', ssn=None)), id='output'),
            ToolGuardrail(result_guard=lambda result: GuardrailResult.replace(Record(name='b', ssn=None)), id='tool'),
            RecordingDurability(),
        ],
    )

    @agent.tool_plain
    def lookup(key: str) -> Record:
        return Record(name=key, ssn='123')

    result = await agent.run('look it up')

    assert result.output == Record(name='a', ssn=None)
    tool_return = next(
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'lookup'
    )
    assert tool_return.content == Record(name='b', ssn=None)


def _call_lookup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) > 1:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'name': 'a', 'ssn': '123'})])
    return ModelResponse(parts=[ToolCallPart('lookup', {'key': 'a'}, tool_call_id='call-1')])


def _json_agent(*capabilities: AbstractCapability[None], output_type: type[object] = str) -> Agent[None, object]:
    agent = Agent(
        FunctionModel(_call_lookup),
        name='json_guardrails',
        deps_type=type(None),
        output_type=[output_type, DeferredToolRequests],
        capabilities=[*capabilities, RecordingDurability()],
    )
    agent.tool_plain(lookup)
    return agent


async def test_a_raised_guard_error_reaches_the_run_through_a_json_journal() -> None:
    def refuse(prompt: str) -> bool:
        raise InputBlocked('no')

    def refuse_call(call: ToolCallInfo) -> bool:
        raise ToolBlocked(call.name, 'not today')

    with pytest.raises(InputBlocked, match='no'):
        await _json_agent(InputGuardrail(guard=refuse, id='input')).run('go')
    with pytest.raises(ToolBlocked) as blocked:
        await _json_agent(ToolGuardrail(guard=refuse_call, id='tool')).run('go')
    assert (blocked.value.tool_name, blocked.value.reason) == ('lookup', 'not today')


async def test_approval_metadata_from_a_guard_survives_a_json_journal() -> None:
    def ask(call: ToolCallInfo) -> bool:
        raise ApprovalRequired(metadata={'reason': 'refund'})

    result = await _json_agent(ToolGuardrail(guard=ask, id='tool')).run('go')

    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.metadata == {'call-1': {'reason': 'refund'}}


async def test_a_deliberate_plain_data_replacement_is_not_rebuilt() -> None:
    agent = _json_agent(
        OutputGuardrail(guard=lambda output: GuardrailResult.replace({'redacted': True}), id='output'),
        output_type=Record,
    )

    result = await agent.run('go')

    assert result.output == {'redacted': True}
