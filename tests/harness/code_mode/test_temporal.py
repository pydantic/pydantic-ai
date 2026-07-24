"""Temporal integration tests for CodeMode.

Verifies that the snapshot-based execution loop (`feed_start`/`resume`)
works with Temporal's workflow sandbox and history replay.

Monty executes snippets in subprocess workers. The first `run_code` call lazily creates
the run's worker pool and initial checked-out REPL session. That session is reused across
calls until it is reset or invalidated.

Durability is attached via the `TemporalDurability` capability; pydantic-ai
2.14 deprecated the `TemporalAgent` wrapper in its favor
(pydantic/pydantic-ai#4977). The workflow calls the plain `Agent` directly and
`AgentPlugin` finds the bound capability to register its activities on the
worker. Pydantic AI resolves CodeMode outside TemporalDurability through
CodeMode's capability ordering metadata.

These tests start a local Temporal dev server via
`WorkflowEnvironment.start_local()` -- the Temporal SDK downloads and
runs `temporalite` automatically.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

import pytest

try:
    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        TemporalDurability,
    )
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Replayer, Worker
    from temporalio.worker.workflow_sandbox import (
        RestrictedWorkflowAccessError,
        SandboxedWorkflowRunner,
        SandboxRestrictions,
    )
    from temporalio.workflow import ActivityConfig
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent, ToolDefinition
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset

from pydantic_ai_harness import CodeMode

pytestmark = pytest.mark.anyio

TEMPORAL_PORT = 7244  # avoid conflict with other test suites
TASK_QUEUE = 'pydantic-ai-harness-code-mode-queue'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)


def _workflow_runner() -> SandboxedWorkflowRunner:
    return SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(
            # Coverage imports parser modules lazily while tracing workflow code.
            'coverage',
        )
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Temporal's Python SDK runs on asyncio."""
    return 'asyncio'


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        dev_server_extra_args=[
            '--dynamic-config-value',
            'frontend.enableServerVersionCheck=false',
        ],
    ) as env:
        yield env


@pytest.fixture
async def client(temporal_env: WorkflowEnvironment) -> Client:
    return await Client.connect(
        f'localhost:{TEMPORAL_PORT}',
        plugins=[PydanticAIPlugin()],
    )


# ---------------------------------------------------------------------------
# Tools and agents (module-level -- Temporal requirement)
# ---------------------------------------------------------------------------


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


_captured_tool_defs: list[list[ToolDefinition]] = []


# FunctionModel that emits a run_code tool call for the given code snippet.
def _code_mode_model(messages: list[ModelRequest | ModelResponse], info: AgentInfo) -> ModelResponse:
    """Model that uses two REPL feeds, then returns the second result as text."""
    _captured_tool_defs.append(info.function_tools)

    returns = [
        part
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]
    if len(returns) == 1:
        return ModelResponse(
            parts=[ToolCallPart(tool_name='run_code', args={'code': 'result * 10'}, tool_call_id='test_tc_2')]
        )
    if len(returns) == 2:
        return ModelResponse(parts=[TextPart(content=f'done: {returns[-1].content}')])

    # First call -- emit run_code.
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
    name='code_mode_temporal_agent',
    toolsets=[FunctionToolset(tools=[add], id='math')],
    capabilities=[CodeMode(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class CodeModeWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> dict[str, Any]:
        result = await code_mode_agent.run(prompt)
        return {
            'output': str(result.output),
            'messages': result.all_messages_json().decode(),
        }


@workflow.defn
class SandboxRestrictionWorkflow:
    """Probe that passing Monty through does not allow Python subprocess calls."""

    @workflow.run
    async def run(self) -> str:
        try:
            subprocess.run([sys.executable, '-c', 'pass'], check=True)
        except RestrictedWorkflowAccessError as e:
            return e.qualified_name
        return 'subprocess was allowed'  # pragma: no cover


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_code_mode_runs_in_temporal_workflow(client: Client) -> None:
    """CodeMode runs workflow-side, nested tools use activities, and the history replays."""
    _captured_tool_defs.clear()
    workflow_id = 'test_code_mode_temporal_1'
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CodeModeWorkflow, SandboxRestrictionWorkflow],
        plugins=[AgentPlugin(code_mode_agent)],
        workflow_runner=_workflow_runner(),
    ):
        result = await client.execute_workflow(
            CodeModeWorkflow.run,
            args=['Calculate 3 + 4'],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        sandbox_result = await client.execute_workflow(
            SandboxRestrictionWorkflow.run,
            id='test_code_mode_temporal_sandbox_restrictions',
            task_queue=TASK_QUEUE,
        )

    assert result['output'] == 'done: 70'
    assert sandbox_result == 'subprocess.run.__call__'

    messages = json.loads(result['messages'])
    assert len(messages) == 6

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

    # Verify nested tool call/return metadata
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

    # 4-5. A second feed consumes the variable assigned by the first feed.
    assert messages[3]['kind'] == 'response'
    second_call = messages[3]['parts'][0]
    assert second_call['part_kind'] == 'tool-call'
    assert second_call['args'] == {'code': 'result * 10'}
    assert messages[4]['kind'] == 'request'
    second_return = messages[4]['parts'][0]
    assert second_return['part_kind'] == 'tool-return'
    assert second_return['content'] == 70

    # 6. Final text response
    assert messages[5]['kind'] == 'response'
    assert messages[5]['parts'][0]['part_kind'] == 'text'
    assert messages[5]['parts'][0]['content'] == 'done: 70'

    # 5. Verify tool definitions sent to the model
    assert len(_captured_tool_defs) == 3
    for tool_defs in _captured_tool_defs:
        tool_names = [td.name for td in tool_defs]
        # CodeMode wraps `add` into `run_code` -- the model should only see `run_code`
        assert 'run_code' in tool_names
        assert 'add' not in tool_names

        run_code_td = next(td for td in tool_defs if td.name == 'run_code')
        assert run_code_td.description is not None
        assert 'async def add' in run_code_td.description
        assert run_code_td.parameters_json_schema['properties']['code']['type'] == 'string'

    history = await client.get_workflow_handle(workflow_id).fetch_history()
    replay_result = await Replayer(
        workflows=[CodeModeWorkflow],
        plugins=[PydanticAIPlugin()],
        workflow_runner=_workflow_runner(),
    ).replay_workflow(history)
    assert replay_result.replay_failure is None
