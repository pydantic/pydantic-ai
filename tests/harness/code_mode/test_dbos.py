"""DBOS integration tests for CodeMode.

Verifies that the snapshot-based execution loop works inside a DBOS
durable workflow. DBOS uses SQLite locally -- no external services needed.

Durability is attached via the `DBOSDurability` capability; pydantic-ai 2.14
deprecated the `DBOSAgent` wrapper in its favor (pydantic/pydantic-ai#4977).
With the capability, the caller owns the workflow: `agent.run_sync()` is
called inside an explicit `@DBOS.workflow()` rather than the wrapper starting
one. The agent and workflow are defined at module scope so DBOS registers the
capability's steps and the workflow before `DBOS.launch()`.

DBOS defaults to `parallel_ordered_events` execution mode, which triggers
the sequential FutureSnapshot resolution path in the execution loop.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from typing import Any

import pytest

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID
    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai import Agent, ToolDefinition
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset

from pydantic_ai_harness import CodeMode

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


# ---------------------------------------------------------------------------
# Tools and model
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


_captured_tool_defs: list[list[ToolDefinition]] = []


def _code_mode_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    _captured_tool_defs.append(info.function_tools)

    for msg in messages:
        if isinstance(msg, ModelResponse):
            continue
        for part in msg.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                return ModelResponse(parts=[TextPart(content=f'done: {part.content}')])

    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='run_code',
                args={'code': 'result = await add(a=3, b=4)\nresult'},
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_code_mode_runs_in_dbos_workflow(dbos_instance: DBOS) -> None:
    """CodeMode's snapshot-based execution loop works inside a DBOS durable
    workflow. DBOS defaults to `parallel_ordered_events` mode, which triggers
    the sequential FutureSnapshot resolution path.

    The step-log assertion at the end proves the capability actually routed the
    model requests through DBOS steps: `DBOSDurability` is transparent outside
    a workflow, so a passing run alone would not show durability engaged."""
    _captured_tool_defs.clear()

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
    for tool_defs in _captured_tool_defs:
        tool_names = [td.name for td in tool_defs]
        # CodeMode wraps `add` into `run_code` -- the model should only see `run_code`
        assert 'run_code' in tool_names
        assert 'add' not in tool_names

        run_code_td = next(td for td in tool_defs if td.name == 'run_code')
        assert run_code_td.description is not None
        assert 'async def add' in run_code_td.description
        assert run_code_td.parameters_json_schema['properties']['code']['type'] == 'string'

    # 6. Durability engaged: model requests were recorded as DBOS steps
    steps = dbos_instance.list_workflow_steps(workflow_id)
    step_names = [step['function_name'] for step in steps]
    assert step_names.count('code_mode_dbos_agent__model.request') == 2
