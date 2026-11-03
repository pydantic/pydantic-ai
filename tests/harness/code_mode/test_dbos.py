"""DBOS integration tests for CodeMode.

Verifies that the snapshot-based execution loop works inside a DBOS
durable workflow. DBOS uses SQLite locally -- no external services needed.

Durability is attached via the `DBOSDurability` capability; pydantic-ai 2.14
deprecated the `DBOSAgent` wrapper in its favor (pydantic/pydantic-ai#4977).
With the capability, the caller owns the workflow: `agent.run_sync()` is
called inside an explicit `@DBOS.workflow()` rather than the wrapper starting
one. The agent and workflow are defined at module scope so DBOS registers the
capability's steps and the workflow before `DBOS.launch()`.

DBOS defaults to `parallel_ordered_events` execution mode, under which
gathered sandbox calls still run concurrently.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Generator
from typing import Any

import pytest
from pydantic import BaseModel

from tests.conftest import detach_dbos_logging

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai import Agent, ToolDefinition
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import CodeMode


class Total(BaseModel):
    total: int


pytestmark = pytest.mark.xdist_group(name='harness-dbos')

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def dbos_instance(tmp_path_factory: pytest.TempPathFactory) -> Generator[DBOS, Any, None]:
    dbos_sqlite_file = tmp_path_factory.mktemp('dbos') / 'dbostest.sqlite'
    dbos_config: DBOSConfig = {
        'name': 'pydantic_ai_harness_dbos_tests',
        'system_database_url': f'sqlite:///{dbos_sqlite_file}',
        'run_admin_server': False,
        'enable_otlp': False,
    }
    dbos = DBOS(config=dbos_config)
    DBOS.launch()
    try:
        yield dbos
    finally:
        DBOS.destroy()
        detach_dbos_logging()


# ---------------------------------------------------------------------------
# Tools and model
# ---------------------------------------------------------------------------


_additions: list[tuple[int, int]] = []


@DBOS.step()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    _additions.append((a, b))
    return a + b


_captured_tool_defs: list[list[ToolDefinition]] = []
_captured_instructions: list[str | None] = []


def _code_mode_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    _captured_tool_defs.append(info.function_tools)
    _captured_instructions.append(info.instructions)

    for msg in messages:
        if isinstance(msg, ModelResponse):
            continue
        for part in msg.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                return ModelResponse(parts=[TextPart(content=f'done: {part.content}')])

    code = 'result = await add(a=3, b=4)\nresult'
    if any(
        isinstance(part, UserPromptPart) and part.content == 'Gather two additions'
        for message in messages
        for part in message.parts
    ):
        code = 'import asyncio\nawait asyncio.gather(add(a=3, b=4), add(a=5, b=6))'
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='run_code',
                args={'code': code},
                tool_call_id='test_tc_1',
            )
        ]
    )


code_mode_agent = Agent(
    FunctionModel(_code_mode_model),
    name='code_mode_dbos_agent',
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[CodeMode(), DBOSDurability()],
)


@DBOS.workflow()
def run_code_mode_agent(prompt: str) -> dict[str, Any]:
    result = code_mode_agent.run_sync(prompt)
    return {'output': str(result.output), 'messages': result.all_messages_json().decode()}


class _LimitAbortCrash(BaseException):
    """Leave the failed attempt pending so DBOS recovery replays its recorded model request."""


_limit_errors: list[str] = []


@DBOS.workflow()
def run_limited_code_mode_agent(prompt: str, tool_calls_limit: int) -> None:
    try:
        code_mode_agent.run_sync(prompt, usage_limits=UsageLimits(tool_calls_limit=tool_calls_limit))
    except UsageLimitExceeded as error:
        _limit_errors.append(str(error))
        if len(_limit_errors) == 1:
            raise _LimitAbortCrash from error
        raise


_GATHERED_CALLS = 3
_gather_started = 0
_all_gathered_started = asyncio.Event()


async def fetch(label: str) -> str:
    """Return `label` once every gathered call has started."""
    global _gather_started
    _gather_started += 1
    if _gather_started == _GATHERED_CALLS:
        _all_gathered_started.set()
    # Bounded so a regression to serialized dispatch fails the test instead of hanging the workflow.
    await asyncio.wait_for(_all_gathered_started.wait(), timeout=10)
    return label


def _gather_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    last = messages[-1]
    assert isinstance(last, ModelRequest)
    for part in last.parts:
        if isinstance(part, ToolReturnPart):
            return ModelResponse(parts=[TextPart(content=f'done: {part.content}')])
    code = "import asyncio\nawait asyncio.gather(fetch(label='a'), fetch(label='b'), fetch(label='c'))"
    return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='gather')])


gather_agent = Agent(
    FunctionModel(_gather_model),
    name='code_mode_dbos_gather_agent',
    toolsets=[FunctionToolset(tools=[fetch], id='fetch')],
    capabilities=[CodeMode(), DBOSDurability()],
)


@DBOS.workflow()
def run_gather_agent(prompt: str) -> str:
    return gather_agent.run_sync(prompt).output


def _submit_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    code = 'total = await add(a=40, b=2)\nfinal_result(total=total)'
    return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='submit')])


submit_agent = Agent(
    FunctionModel(_submit_model),
    name='code_mode_dbos_submit_agent',
    output_type=Total,
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[CodeMode(allow_final_output=True), DBOSDurability()],
)


@DBOS.workflow()
def run_submit_agent(prompt: str) -> dict[str, Any]:
    result = submit_agent.run_sync(prompt)
    return {'total': result.output.total, 'requests': result.usage.requests}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_code_mode_runs_in_dbos_workflow(dbos_instance: DBOS) -> None:
    """CodeMode's snapshot-based execution loop works inside a DBOS durable
    workflow under DBOS's default `parallel_ordered_events` mode.

    The step-log assertion at the end proves the capability actually routed the
    model requests through DBOS steps: `DBOSDurability` is transparent outside
    a workflow, so a passing run alone would not show durability engaged."""
    _captured_tool_defs.clear()
    _captured_instructions.clear()

    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id):
        payload = run_code_mode_agent('Calculate 3 + 4')

    assert payload['output'] == 'done: 7'

    messages = json.loads(payload['messages'])
    assert len(messages) == 4

    # 1. User prompt
    assert messages[0]['kind'] == 'request'
    assert messages[0]['parts'][0]['part_kind'] == 'user-prompt'
    assert messages[0]['parts'][0]['content'] == 'Calculate 3 + 4'

    # 2. Model response -- run_code tool call
    assert messages[1]['kind'] == 'response'
    tc = messages[1]['parts'][0]
    assert tc['part_kind'] == 'tool-call'
    assert tc['tool_name'] == 'run_code'
    assert tc['args'] == {'code': 'result = await add(a=3, b=4)\nresult'}
    assert tc['tool_call_id'] == 'test_tc_1'

    # 3. Tool return with nested tool call metadata
    assert messages[2]['kind'] == 'request'
    tr = messages[2]['parts'][0]
    assert tr['part_kind'] == 'tool-return'
    assert tr['tool_name'] == 'run_code'
    assert tr['content'] == 7
    assert tr['tool_call_id'] == 'test_tc_1'

    metadata = tr['metadata']
    assert metadata is not None
    assert metadata['code_mode'] is True
    nested_calls = metadata['tool_calls']
    nested_returns = metadata['tool_returns']
    assert len(nested_calls) == 1
    assert len(nested_returns) == 1

    nested_call = next(iter(nested_calls.values()))
    assert nested_call['tool_name'] == 'add'
    assert nested_call['args'] == {'a': 3, 'b': 4}

    nested_return = next(iter(nested_returns.values()))
    assert nested_return['tool_name'] == 'add'
    assert nested_return['content'] == 7
    assert nested_return['tool_call_id'] == nested_call['tool_call_id']

    # 4. Final text response
    assert messages[3]['kind'] == 'response'
    assert messages[3]['parts'][0]['part_kind'] == 'text'
    assert messages[3]['parts'][0]['content'] == 'done: 7'

    # 5. Verify tool definitions sent to the model
    # The model was called twice (first request + after tool return), both
    # should see the same tool definitions.
    assert len(_captured_tool_defs) == 2
    for tool_defs, instructions in zip(_captured_tool_defs, _captured_instructions, strict=True):
        tool_names = [td.name for td in tool_defs]
        # CodeMode wraps `add` into `run_code` -- the model should only see `run_code`
        assert 'run_code' in tool_names
        assert 'add' not in tool_names

        # The catalog crosses the model request in the instructions, not in `run_code`'s description.
        assert instructions is not None
        assert 'async def add' in instructions
        run_code_td = next(td for td in tool_defs if td.name == 'run_code')
        assert run_code_td.description is not None
        assert 'async def add' not in run_code_td.description
        assert run_code_td.parameters_json_schema['properties']['code']['type'] == 'string'

    # 6. Durability engaged: model requests were recorded as DBOS steps
    steps = dbos_instance.list_workflow_steps(workflow_id)
    step_names = [step['function_name'] for step in steps]
    assert step_names.count('code_mode_dbos_agent__model.request') == 2


@pytest.mark.parametrize(
    'prompt,limit,expected_additions',
    [('Calculate 3 + 4', 1, []), ('Gather two additions', 2, [(3, 4)])],
    ids=['sequential', 'gather'],
)
def test_tool_calls_limit_refuses_during_recovery(
    dbos_instance: DBOS, prompt: str, limit: int, expected_additions: list[tuple[int, int]]
) -> None:
    _limit_errors.clear()
    _captured_tool_defs.clear()
    _captured_instructions.clear()
    _additions.clear()
    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_LimitAbortCrash):
        run_limited_code_mode_agent(prompt, limit)

    recorded_additions: list[tuple[int, object, object]] = [
        (step['function_id'], step['output'], step['error'])
        for step in dbos_instance.list_workflow_steps(workflow_id)
        if step['function_name'] == 'add'
    ]
    assert len(recorded_additions) == len(expected_additions)
    if expected_additions:
        assert recorded_additions[0][1:] == (7, None)

    handle = DBOS.resume_workflow(workflow_id)
    with pytest.raises(UsageLimitExceeded, match=rf'tool_calls_limit of {limit} \(tool_calls={limit + 1}\)'):
        handle.get_result()

    assert len(_limit_errors) == 2
    assert _limit_errors[0] == _limit_errors[1]
    assert len(_captured_tool_defs) == 1
    assert _additions == expected_additions
    steps = dbos_instance.list_workflow_steps(workflow_id)
    assert [step['function_name'] for step in steps].count('code_mode_dbos_agent__model.request') == 1
    assert [
        (step['function_id'], step['output'], step['error']) for step in steps if step['function_name'] == 'add'
    ] == recorded_additions


def test_gathered_calls_overlap_in_dbos_workflow(dbos_instance: DBOS) -> None:
    """`DBOSDurability`'s default `parallel_ordered_events` mode keeps gathered sandbox calls concurrent.

    Each call waits until all three are in flight, so serialized dispatch times out (issue #9248).
    """
    with SetWorkflowID(str(uuid.uuid4())):
        output = run_gather_agent('Gather three fetches.')

    assert output == "done: ['a', 'b', 'c']"


def test_code_mode_submits_final_output_in_dbos_workflow(dbos_instance: DBOS) -> None:
    """`run_code` runs in the workflow under DBOS, so the code can submit the run's output."""
    with SetWorkflowID(str(uuid.uuid4())):
        result = run_submit_agent('Add 40 and 2')
    assert result == {'total': 42, 'requests': 1}
