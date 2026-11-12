"""`AskUser` under `TemporalDurability`: the workflow waits for the answer, and the answerer isn't asked again on replay.

These tests start Temporal's time-skipping test server via `WorkflowEnvironment.start_time_skipping()`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta

import anyio
import pytest
from pydantic import TypeAdapter

try:
    from temporalio import workflow
    from temporalio.common import RetryPolicy
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        TemporalDurability,
        resolve_external_event,
    )
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import AskUser
from pydantic_ai_harness.ask_user import TOOL_NAME, AskUserAnswer, AskUserRequest, AskUserResponse
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal')]

TASK_QUEUE = 'pydantic-ai-harness-ask-user-queue'
_QUESTIONS = [{'header': 'Scope', 'question': 'Which files?', 'options': [{'label': 'All'}, {'label': 'Some'}]}]

_tokens: list[str] = []
_handed_on: list[anyio.Event] = []


def _ask_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart(TOOL_NAME, {'questions': _QUESTIONS}, tool_call_id='q-1')])
    returned = next(part for part in messages[-1].parts if isinstance(part, ToolReturnPart))
    return ModelResponse(parts=[TextPart(returned.model_response_str())])


async def _hand_on(ctx: RunContext[None], request: AskUserRequest, token: str) -> AskUserResponse | None:
    _tokens.append(token)
    _handed_on[-1].set()
    return None


ask_user_agent = Agent(
    FunctionModel(_ask_then_answer),
    name='temporal_ask_user',
    deps_type=type(None),
    capabilities=[
        TemporalDurability[None](
            activity_config=ActivityConfig(
                start_to_close_timeout=timedelta(seconds=60), retry_policy=RetryPolicy(maximum_attempts=1)
            )
        ),
        AskUser[None](answerer=_hand_on, timeout=600),
    ],
)


@workflow.defn
class AskUserWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await ask_user_agent.run(prompt)).output


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Temporal's Python SDK runs on asyncio."""
    return 'asyncio'


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_time_skipping(
        plugins=[PydanticAIPlugin()], download_dest_dir=temporal_dev_server_cache_dir()
    ) as env:
        yield env


async def test_a_question_waits_for_the_answer_and_its_answerer_is_not_asked_again(
    temporal_env: WorkflowEnvironment,
) -> None:
    client = temporal_env.client
    _handed_on.append(anyio.Event())
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[AskUserWorkflow],
        plugins=[AgentPlugin(ask_user_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            AskUserWorkflow.run, args=['tidy up'], id=f'ask-user-{uuid.uuid4()}', task_queue=TASK_QUEUE
        )
        with anyio.fail_after(30):
            await _handed_on[-1].wait()
        [token] = _tokens

        response = AskUserResponse(answers=(AskUserAnswer(header='Scope', selected=('All',)),))
        await resolve_external_event(client, token, TypeAdapter(AskUserResponse).dump_python(response, mode='json'))
        assert json.loads(await handle.result()) == {'Scope': ['All']}

    await Replayer(
        workflows=[AskUserWorkflow],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(await handle.fetch_history())
    assert _tokens == [token], 'the replay does not hand the question on again'
