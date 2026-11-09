"""Function tool arguments crossing the activity boundary.

The workflow validates a tool call's arguments, and the activity that runs the tool validates them
again after they arrive as JSON. A tool whose single parameter is a Pydantic model has validated
arguments of the form `{name: model}`; when that model also has a field named after the parameter,
the transported `{name: {...fields}}` reads like the model's own input, so the activity must not
guess which one it holds.
"""

from __future__ import annotations

import sys
import uuid
import warnings
from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.models.function import AgentInfo, FunctionModel

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.worker import Worker

    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        TemporalAgent,  # pyright: ignore[reportDeprecated]
        TemporalDurability,
    )
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

with workflow.unsafe.imports_passed_through():
    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE

pytestmark = [
    pytest.mark.xdist_group(name='temporal-durability'),
    pytest.mark.filterwarnings(
        'ignore:`TemporalAgent` is deprecated:pydantic_ai._warnings.PydanticAIDeprecationWarning'
    ),
]


class Job(BaseModel):
    data: Any = None
    dry_run: bool = False


_received: list[Job] = []
"""The arguments the tool ran with, recorded in the activity."""


def run_job(data: Job) -> str:
    """Run a job."""
    _received.append(data)
    return 'Job finished.'


def _model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        args = {'data': {'path': '/reports'}, 'dry_run': True}
        return ModelResponse(parts=[ToolCallPart('run_job', args, tool_call_id='c1')])
    return ModelResponse(parts=[TextPart('Done.')])


_durable_agent = Agent(
    FunctionModel(_model_fn),
    name='tool_args_durable',
    tools=[run_job],
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)

with warnings.catch_warnings():
    warnings.simplefilter('ignore', PydanticAIDeprecationWarning)
    _wrapped_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
        Agent(FunctionModel(_model_fn), name='tool_args_wrapped', tools=[run_job]),
        activity_config=BASE_ACTIVITY_CONFIG,
    )


@workflow.defn
class ToolArgsWorkflow:
    @workflow.run
    async def run(self, path: str) -> str:
        agent = _durable_agent if path == 'durability' else _wrapped_agent
        result = await agent.run('Run the reports job as a dry run.')
        return result.output


@pytest.mark.parametrize('path', ['durability', 'temporal_agent'])
async def test_single_model_arg_named_like_its_field_keeps_its_arguments(client: Client, path: str):
    """The tool runs with the arguments the workflow validated, not the model's defaults."""
    _received.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ToolArgsWorkflow],
        plugins=[AgentPlugin(_durable_agent if path == 'durability' else _wrapped_agent)],
    ):
        output = await client.execute_workflow(
            ToolArgsWorkflow.run,
            args=[path],
            id=f'{ToolArgsWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )

    assert output == 'Done.'
    assert _received == [Job(data={'path': '/reports'}, dry_run=True)]
