"""`Memory` under `TemporalDurability`: every memory tool call runs inside a Temporal activity.

These tests start a local Temporal dev server via `WorkflowEnvironment.start_local()`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from typing import Any

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
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.memory import InMemoryStore, Memory
from tests.harness._temporal import ignore_source_reads_left_open
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal'), ignore_source_reads_left_open]

TEMPORAL_PORT = 7263  # avoid conflict with the other Temporal suites
TASK_QUEUE = 'pydantic-ai-harness-memory-queue'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)
# See tests/spend/test_temporal.py for why these modules pass through.
_RESTRICTIONS = SandboxRestrictions.default.with_passthrough_modules('coverage', 'annotated_types')

# Every memory tool, in an order where each one has something to act on.
_TOOL_CALLS: list[tuple[str, dict[str, str]]] = [
    ('write_memory', {'content': 'The user prefers tabs.', 'file': 'style.md'}),
    ('read_memory', {'file': 'style.md'}),
    ('search_memory', {'query': 'tabs'}),
    ('delete_memory', {'file': 'style.md'}),
]


def _call_memory_tools(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call the next memory tool, then answer once all of them have returned.

    The model runs in its own activity on the worker, outside the workflow sandbox, so the returns it
    records here are visible to the test.
    """
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if len(returns) < len(_TOOL_CALLS):
        name, args = _TOOL_CALLS[len(returns)]
        return ModelResponse(parts=[ToolCallPart(name, args)])
    tool_returns.extend((part.tool_name, part.content) for part in returns)
    return ModelResponse(parts=[TextPart('done')])


tool_returns: list[tuple[str, Any]] = []
memory_store = InMemoryStore()

memory_agent = Agent(
    FunctionModel(_call_memory_tools),
    name='memory_agent',
    deps_type=type(None),
    capabilities=[
        Memory[None](store=memory_store, inject_memory=False),
        TemporalDurability[None](activity_config=BASE_ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class MemoryWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await memory_agent.run(prompt)).output


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


async def test_memory_tools_run_inside_activities(client: Client) -> None:
    """The memory tools open spans on `ctx.tracer`, which an activity resolves from the worker's agent."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[MemoryWorkflow],
        plugins=[AgentPlugin(memory_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
    ):
        output = await client.execute_workflow(
            MemoryWorkflow.run,
            'remember my style',
            id='test_memory_temporal',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=25),
        )

    assert output == 'done'
    returns = dict(tool_returns)
    assert returns['write_memory']['status'] == 'created'
    assert returns['read_memory'] == 'The user prefers tabs.\n'
    assert [match['file'] for match in returns['search_memory']['matches']] == ['style.md']
    assert returns['delete_memory']['status'] == 'deleted'
    assert await memory_store.read('memory_agent/style.md', max_chars=100) is None
