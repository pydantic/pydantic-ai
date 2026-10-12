"""`Skills` under `TemporalDurability`: the catalog is read through the durable workspace in `for_run`.

These tests start a local Temporal dev server via `WorkflowEnvironment.start_local()`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path

import pytest

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.common import RetryPolicy
    from temporalio.testing import WorkflowEnvironment
    from temporalio.worker import Worker
    from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporalio not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import LoadCapabilityReturnPart, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.skills import Skills
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal')]

TEMPORAL_PORT = 7257  # avoid conflict with the other Temporal suites
TASK_QUEUE = 'pydantic-ai-harness-skills-queue'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)
# See tests/spend/test_temporal.py for why these modules pass through.
_RESTRICTIONS = SandboxRestrictions.default.with_passthrough_modules('coverage', 'annotated_types')

# Module level, as Temporal requires, so the workspace is a checked-in directory: the workflow sandbox
# imports this module again and forbids file I/O there.
WORK = Path(__file__).parent / 'temporal_workspace'


def _load_reviewer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Load the `reviewer` skill, then answer with the instructions `load_capability` returned."""
    loaded = [part for message in messages for part in message.parts if isinstance(part, LoadCapabilityReturnPart)]
    if loaded:
        return ModelResponse(parts=[TextPart(loaded[0].content.get('instructions', ''))])
    return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'reviewer'})])


skills_agent = Agent(
    FunctionModel(_load_reviewer),
    name='skills_agent',
    deps_type=type(None),
    capabilities=[
        LocalWorkspace[None](WORK),
        Skills[None]('skills'),
        TemporalDurability[None](activity_config=BASE_ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class SkillsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await skills_agent.run(prompt)).output


@pytest.fixture(scope='module')
def anyio_backend() -> str:
    """Temporal's Python SDK runs on asyncio."""
    return 'asyncio'


@pytest.fixture(scope='module')
async def client() -> AsyncIterator[Client]:
    async with await WorkflowEnvironment.start_local(  # pyright: ignore[reportUnknownMemberType]
        port=TEMPORAL_PORT,
        dev_server_extra_args=['--dynamic-config-value', 'frontend.enableServerVersionCheck=false'],
        download_dest_dir=temporal_dev_server_cache_dir(),
    ):
        yield await Client.connect(f'localhost:{TEMPORAL_PORT}', plugins=[PydanticAIPlugin()])


async def test_skills_are_read_through_the_durable_workspace(client: Client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SkillsWorkflow],
        plugins=[AgentPlugin(skills_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
    ):
        output = await client.execute_workflow(
            SkillsWorkflow.run,
            'review this',
            id='test_skills_temporal',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=25),
        )

    assert output == '# Skill: reviewer\n\nCheck the tests.'
