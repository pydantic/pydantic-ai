"""`Knowledge` under `TemporalDurability`: the search, its sync and its workspace reads run in an activity.

These tests start a local Temporal dev server via `WorkflowEnvironment.start_local()`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
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
from pydantic_ai.embeddings import EmbeddingModel, EmbeddingResult, EmbeddingSettings
from pydantic_ai.embeddings.result import EmbedInputType
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.knowledge import Knowledge
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal')]

TEMPORAL_PORT = 7262  # avoid conflict with the other Temporal suites
TASK_QUEUE = 'pydantic-ai-harness-knowledge-queue'
BASE_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)
# See tests/spend/test_temporal.py for why these modules pass through.
_RESTRICTIONS = SandboxRestrictions.default.with_passthrough_modules('coverage', 'annotated_types')

# Module level, as Temporal requires, so the workspace is a checked-in directory: the workflow sandbox
# imports this module again and forbids file I/O there.
WORK = Path(__file__).parent / 'temporal_workspace'


class KeywordEmbeddingModel(EmbeddingModel):
    """Embeds text as the count of each of a few words in it."""

    @property
    def model_name(self) -> str:
        return 'keywords'

    @property
    def system(self) -> str:
        return 'test'

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        texts, _ = self.prepare_embed(inputs, settings)
        return EmbeddingResult(
            embeddings=[[float(text.lower().count(word)) for word in ('refund', 'shipping')] for text in texts],
            inputs=texts,
            input_type=input_type,
            model_name='keywords',
            provider_name='test',
        )


def _search_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Search for refunds, then answer with the id of the top document."""
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(str(part.content))])
    return ModelResponse(parts=[ToolCallPart('search_knowledge', {'query': 'refund'})])


knowledge_agent = Agent(
    FunctionModel(_search_then_answer),
    name='knowledge_agent',
    deps_type=type(None),
    capabilities=[
        LocalWorkspace[None](WORK),
        Knowledge[None](['docs'], embedder=KeywordEmbeddingModel(), max_results=1),
        TemporalDurability[None](activity_config=BASE_ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class KnowledgeWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await knowledge_agent.run(prompt)).output


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


async def test_search_reads_and_embeds_in_an_activity(client: Client) -> None:
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[KnowledgeWorkflow],
        plugins=[AgentPlugin(knowledge_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
    ):
        output = await client.execute_workflow(
            KnowledgeWorkflow.run,
            'how do refunds work?',
            id='test_knowledge_temporal',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=25),
        )

    assert output == (
        "[{'document_id': 'docs/refunds.md', 'text': 'Refunds are issued within 14 days of a refund request.', "
        "'score': 1.0}]"
    )
