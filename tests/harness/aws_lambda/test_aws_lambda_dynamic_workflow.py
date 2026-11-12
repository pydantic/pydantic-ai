"""`DynamicWorkflow` under `AWSLambdaDurability`: gathered sub-agents run in parallel, each in a branch of its own.

These run the real SDK's `@durable_execution` handler against an in-memory service, so branch identity
and replay are the SDK's own.
"""

from __future__ import annotations

import base64
import itertools
import json
from dataclasses import replace
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import RequestInput
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, durable_agent_handler
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow
from tests.harness.dynamic_workflow._concurrent_sub_agents import (
    EXPECTED,
    SCRIPT,
    build_orchestrator,
    noop,
    orchestrator_model,
    summarize_model,
)

pytest.importorskip('aws_durable_execution_sdk_python')

from aws_durable_execution_sdk_python.execution import (
    DurableExecutionInvocationInputWithClient,
    InitialExecutionState,
    durable_execution,  # pyright: ignore[reportUnknownVariableType]
)
from aws_durable_execution_sdk_python.lambda_service import (
    CallbackDetails,
    CheckpointOutput,
    CheckpointUpdatedExecutionState,
    ContextDetails,
    ExecutionDetails,
    Operation,
    OperationAction,
    OperationStatus,
    OperationType,
    OperationUpdate,
    StateOutput,
    StepDetails,
)
from aws_durable_execution_sdk_python.retries import RetryPresets
from aws_durable_execution_sdk_python.serdes import DEFAULT_JSON_SERDES, SerDes, SerDesContext

_ARN = 'arn:aws:lambda:us-east-1:000000000000:function:test/durable-execution/run/invocation'
_STATUS = {
    OperationAction.START: OperationStatus.STARTED,
    OperationAction.SUCCEED: OperationStatus.SUCCEEDED,
    OperationAction.FAIL: OperationStatus.FAILED,
}


class _Service:
    """An in-memory durable-execution service: it keeps the checkpoints a later invocation replays."""

    def __init__(self) -> None:
        self.operations: dict[str, Operation] = {}
        self._tokens = itertools.count()

    def checkpoint(
        self,
        durable_execution_arn: str,
        checkpoint_token: str,
        updates: list[OperationUpdate],
        client_token: str | None,
    ) -> CheckpointOutput:
        created: list[Operation] = []
        for update in updates:
            step = StepDetails(result=update.payload, error=update.error)
            context = ContextDetails(result=update.payload, error=update.error)
            callback = CallbackDetails(callback_id=f'callback-{update.operation_id[:8]}')
            operation = self.operations[update.operation_id] = Operation(
                operation_id=update.operation_id,
                operation_type=update.operation_type,
                status=_STATUS[update.action],
                parent_id=update.parent_id,
                name=update.name,
                sub_type=update.sub_type,
                step_details=step if update.operation_type is OperationType.STEP else None,
                context_details=context if update.operation_type is OperationType.CONTEXT else None,
                callback_details=callback if update.operation_type is OperationType.CALLBACK else None,
            )
            created.append(operation)
        # The service creates a callback's ID, which the SDK reads back from the new state.
        return CheckpointOutput(
            checkpoint_token=str(next(self._tokens)),
            new_execution_state=CheckpointUpdatedExecutionState(operations=created),
        )

    def get_execution_state(  # pragma: no cover - every invocation is handed its whole state
        self, durable_execution_arn: str, checkpoint_token: str, next_marker: str, max_items: int = 1000
    ) -> StateOutput:
        return StateOutput()

    def invoke(self, orchestrator: Agent[None, str]) -> str:
        output = self._invoke(orchestrator)
        assert output['Status'] == 'SUCCEEDED', output
        return json.loads(output['Result'])

    def suspend(self, orchestrator: Agent[None, str]) -> None:
        assert self._invoke(orchestrator)['Status'] == 'PENDING'

    def complete_callback(self, callback_id: str, result: str) -> None:
        """Resolve a callback the way `SendDurableExecutionCallbackSuccess` does."""
        (operation,) = [
            operation
            for operation in self.operations.values()
            if operation.callback_details is not None and operation.callback_details.callback_id == callback_id
        ]
        self.operations[operation.operation_id] = replace(
            operation,
            status=OperationStatus.SUCCEEDED,
            callback_details=CallbackDetails(callback_id=callback_id, result=result),
        )

    def _invoke(self, orchestrator: Agent[None, str]) -> dict[str, Any]:
        @durable_execution
        @durable_agent_handler
        async def handler(event: Any, context: Any) -> str:
            return (await orchestrator.run('go')).output

        execution = Operation(
            operation_id='invocation',
            operation_type=OperationType.EXECUTION,
            status=OperationStatus.STARTED,
            execution_details=ExecutionDetails(input_payload='{}'),
        )
        event = DurableExecutionInvocationInputWithClient(
            durable_execution_arn=_ARN,
            checkpoint_token='start',
            initial_execution_state=InitialExecutionState(
                operations=[execution, *self.operations.values()], next_marker=''
            ),
            service_client=self,
        )
        return handler(event, None)  # pyright: ignore[reportArgumentType]

    def crash_during_last_step(self, branch_name: str) -> None:
        """Leave the checkpoints a process that died in the branch's last step would have left."""
        (branch,) = [operation for operation in self.operations.values() if operation.name == branch_name]
        parents: list[Operation] = []
        parent_id = branch.parent_id
        while parent_id is not None:
            parent = self.operations[parent_id]
            parents.append(parent)
            parent_id = parent.parent_id
        # The branch and the operations enclosing it never finished, and nothing after them started.
        top = parents[-1]
        operations = list(self.operations)
        for operation_id in operations[operations.index(top.operation_id) + 1 :]:
            if self.operations[operation_id].parent_id is None:
                del self.operations[operation_id]
        del self.operations[self.children(branch_name)[-1].operation_id]
        for unfinished in [branch, *parents]:
            self.operations[unfinished.operation_id] = replace(
                unfinished, status=OperationStatus.STARTED, context_details=None
            )

    def children(self, name: str) -> list[Operation]:
        (parent,) = [operation for operation in self.operations.values() if operation.name == name]
        return [operation for operation in self.operations.values() if operation.parent_id == parent.operation_id]


def _branch_steps(service: _Service) -> list[list[str | None]]:
    branches = sorted(service.children('run_workflow'), key=lambda branch: branch.name or '')
    return [[step.name for step in service.children(branch.name or '')] for branch in branches]


@pytest.mark.parametrize('durable_orchestrator', [True, False])
def test_parallel_sub_agents_get_their_own_branches_and_replay(durable_orchestrator: bool) -> None:
    calls: list[str] = []
    # The sleep's future is resolved before any sub-agent is dispatched, so it starts no `parallel` operation.
    orchestrator = build_orchestrator(
        AWSLambdaDurability,
        summarize_model(calls),
        durable_orchestrator=durable_orchestrator,
        script=SCRIPT.replace('import asyncio\n', 'import asyncio\nawait asyncio.sleep(0)\n'),
    )
    service = _Service()
    assert service.invoke(orchestrator) == EXPECTED
    # Alpha, dispatched first, finished last: the sub-agents ran concurrently.
    assert sorted(calls) == ['alpha 1', 'alpha 3', 'beta 1', 'beta 3']
    assert calls[-1] == 'alpha 3'
    # One `parallel` operation, with a branch per sub-agent run, each holding its own steps.
    alpha, beta = _branch_steps(service)
    assert alpha == beta
    assert len(alpha) > 1

    assert service.invoke(orchestrator) == EXPECTED
    assert len(calls) == 4


def test_parallel_sub_agents_recover_from_a_crash_mid_branch() -> None:
    calls: list[str] = []
    orchestrator = build_orchestrator(AWSLambdaDurability, summarize_model(calls))
    service = _Service()
    assert service.invoke(orchestrator) == EXPECTED
    service.crash_during_last_step('parallel-branch-0')

    # Beta replays its whole branch, and alpha its checkpoints up to the step that was in flight.
    assert service.invoke(orchestrator) == EXPECTED
    assert calls[4:] == ['alpha 3']


def test_a_failed_sub_agent_is_recorded_in_its_branch() -> None:
    calls: list[str] = []

    def fail(call: str) -> None:
        raise ValueError(call)

    script = "try:\n    await summarize(task='alpha')\nexcept RuntimeError as error:\n    result = str(error)\nresult"
    # Without retries, the failed model request fails the sub-agent run.
    orchestrator = build_orchestrator(
        lambda: AWSLambdaDurability(step_config={'retry_strategy': RetryPresets.none()}),
        summarize_model(calls, fail),
        script=script,
    )
    service = _Service()
    output = service.invoke(orchestrator)
    # The error type is the SDK's wrapper for a failed step, whose name differs between SDK versions.
    assert output.startswith("done: sub-agent 'summarize' raised ")
    (branch,) = service.children('run_workflow')
    assert branch.status is OperationStatus.SUCCEEDED

    assert service.invoke(orchestrator) == output
    assert calls == ['alpha 1']


def test_sub_agents_without_durability_run_outside_any_branch() -> None:
    calls: list[str] = []
    orchestrator = build_orchestrator(AWSLambdaDurability, summarize_model(calls), durable_sub_agents=False)
    service = _Service()
    assert service.invoke(orchestrator) == EXPECTED
    assert [operation.name for operation in service.operations.values()] == ['orchestrator__model.request'] * 2


class _Sealed(SerDes[Any]):
    """Stands in for a serializer that encrypts what it checkpoints."""

    def serialize(self, value: Any, serdes_context: SerDesContext) -> str:
        return 'sealed:' + base64.b64encode(DEFAULT_JSON_SERDES.serialize(value, serdes_context).encode()).decode()

    def deserialize(self, data: str, serdes_context: SerDesContext) -> Any:
        return DEFAULT_JSON_SERDES.deserialize(base64.b64decode(data.removeprefix('sealed:')).decode(), serdes_context)


def test_parallel_checkpoints_use_the_sub_agents_serializer() -> None:
    calls: list[str] = []
    orchestrator = build_orchestrator(
        lambda: AWSLambdaDurability(step_config={'serdes': _Sealed()}), summarize_model(calls)
    )
    service = _Service()
    assert service.invoke(orchestrator) == EXPECTED
    payloads = [
        (operation.step_details or operation.context_details or StepDetails()).result
        for operation in service.operations.values()
    ]
    assert not [payload for payload in payloads if payload and 'summary of' in payload]

    assert service.invoke(orchestrator) == EXPECTED
    assert len(calls) == 4


def test_a_callback_in_a_branch_suspends_the_invocation_and_resumes_its_branch() -> None:
    calls: list[str] = []
    asked: list[str] = []
    deleted: list[str] = []

    def clean_up(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(f'clean {len(messages)}')
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('delete', {'path': 'a.txt'}, tool_call_id='call-1')])
        return ModelResponse(parts=[TextPart('cleaned up')])

    def ask(ctx: RunContext[Any], requests: DeferredToolRequests, token: str) -> None:
        asked.append(token)

    cleaner = Agent(FunctionModel(clean_up), name='clean', capabilities=[AWSLambdaDurability(), RequestInput(ask)])

    @cleaner.tool_plain(requires_approval=True)
    def delete(path: str) -> str:
        deleted.append(path)
        return f'deleted {path}'

    summarize = Agent(
        summarize_model(calls, alpha_waits_for_beta=False),
        name='summarize',
        tools=[noop],
        capabilities=[AWSLambdaDurability()],
    )
    script = "import asyncio\nawait asyncio.gather(clean(task='a.txt'), summarize(task='beta'))"
    orchestrator = Agent(
        orchestrator_model(script),
        name='orchestrator',
        capabilities=[DynamicWorkflow(agents=[cleaner, summarize]), AWSLambdaDurability()],
    )
    service = _Service()
    service.suspend(orchestrator)

    # The callback belongs to the first branch's child context, and the sibling branch finished.
    ((callback_id, branch),) = [
        (operation.callback_details.callback_id, service.operations[operation.parent_id or ''])
        for operation in service.operations.values()
        if operation.callback_details is not None
    ]
    assert branch.name == 'parallel-branch-0'
    (sibling,) = [operation for operation in service.children('run_workflow') if operation.name == 'parallel-branch-1']
    assert sibling.status is OperationStatus.SUCCEEDED
    assert asked == [callback_id]
    assert sorted(calls) == ['beta 1', 'beta 3', 'clean 1']

    service.complete_callback(callback_id, '{"approvals": {"call-1": true}}')
    assert service.invoke(orchestrator) == "done: ['cleaned up', 'summary of beta']"
    assert deleted == ['a.txt']
    assert asked == [callback_id]
    assert sorted(calls) == ['beta 1', 'beta 3', 'clean 1', 'clean 3']


def test_sub_agents_sharing_a_serializer_share_a_parallel_operation() -> None:
    calls: list[str] = []
    sealed = _Sealed()
    model = summarize_model(calls)
    alpha, beta = (
        Agent(model, name=name, tools=[noop], capabilities=[AWSLambdaDurability(step_config={'serdes': sealed})])
        for name in ('alpha', 'beta')
    )
    script = "import asyncio\nawait asyncio.gather(alpha(task='alpha'), beta(task='beta'))"
    orchestrator = Agent(
        orchestrator_model(script), name='orchestrator', capabilities=[DynamicWorkflow(agents=[alpha, beta])]
    )
    service = _Service()
    assert service.invoke(orchestrator) == EXPECTED
    assert [operation.name for operation in service.operations.values() if operation.parent_id is None].count(
        'run_workflow'
    ) == 1
