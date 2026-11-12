"""`Permissions` under `DBOSDurability`: remembered approvals hold on replay and in the next run."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.permissions import PermissionDecision, PermissionRequest, Permissions, ask_permissions
from tests.conftest import detach_dbos_logging, try_import, without_admin_server

with try_import() as imports_successful:
    from dbos import DBOS, DBOSConfig, SetWorkflowID

    from pydantic_ai.durable_exec.dbos import DBOSDurability

pytestmark = [
    pytest.mark.xdist_group(name='harness-dbos'),
    pytest.mark.skipif(not imports_successful(), reason='dbos not installed'),
]


class _Crash(BaseException):
    """Leave the first attempt pending, so resuming it replays the journaled steps."""


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_permissions',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
    }
    instance = DBOS(config=without_admin_server(config))
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()
        detach_dbos_logging()


def _two_commands(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    step = sum(isinstance(message, ModelResponse) for message in messages if message.run_id == messages[-1].run_id)
    if step < 2:
        return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'echo hi'})])
    return ModelResponse(parts=[TextPart('done')])


def _returns(messages: list[ModelMessage]) -> list[object]:
    return [part.content for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


def test_remembered_approval_holds_on_replay_and_in_the_next_run(dbos: DBOS) -> None:
    asked: list[PermissionRequest] = []

    async def person(request: PermissionRequest) -> PermissionDecision:
        asked.append(request)
        return PermissionDecision(approved=True, remember=True)

    agent = Agent(
        FunctionModel(_two_commands),
        name='permissions_dbos',
        capabilities=[
            Permissions('read-only'),
            HandleDeferredToolCalls(handler=lambda ctx, requests: ask_permissions(requests, person)),
            DBOSDurability(),
        ],
    )

    @agent.tool_plain(metadata={'code_arg_language': 'shell', 'code_arg_name': 'command'})
    def run_command(command: str) -> str:
        return command

    attempts: list[list[ModelMessage]] = []

    @DBOS.workflow(name='permissions_dbos')
    def workflow(history: list[ModelMessage] | None) -> None:
        attempts.append(agent.run_sync('go', message_history=history).all_messages())
        if len(attempts) == 1:
            raise _Crash
        if len(attempts) == 2:
            # Recovery runs on a DBOS executor thread, where `run_sync` sets a new event loop that nothing else closes.
            asyncio.get_event_loop().close()

    workflow_id = str(uuid.uuid4())
    with SetWorkflowID(workflow_id), pytest.raises(_Crash):
        workflow(None)
    assert len(asked) == 1

    DBOS.resume_workflow(workflow_id).get_result()
    # The inline handler is not journaled, so the replay asks about the first call again, but not the second.
    assert len(asked) == 2

    workflow(attempts[-1])
    assert len(asked) == 2
    assert _returns(attempts[-1]) == ['echo hi'] * 4
