"""`Coder` under `TemporalDurability`: its file and shell tools run as activities against the run's workspace.

These tests start a local Temporal dev server via `WorkflowEnvironment.start_local()`.
"""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator, Iterator
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
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import Shell
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal')]

TEMPORAL_PORT = 7259  # avoid conflict with the other Temporal suites
TASK_QUEUE = 'pydantic-ai-harness-coder-queue'
ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=1),
)
# See tests/spend/test_temporal.py for why these modules pass through.
_SANDBOXED = SandboxRestrictions.default.with_passthrough_modules('coverage', 'annotated_types')
_PASSTHROUGH = _SANDBOXED.with_passthrough_modules('pydantic_ai_harness')

# Module level, as Temporal requires, so the workspace is a checked-in directory: the workflow sandbox
# imports this module again and forbids file I/O there. Its `.gitignore` keeps what the runs write out of git.
WORK = Path(__file__).parent / 'temporal_workspace'

STEPS: list[tuple[str, dict[str, object]]] = [
    ('write_file', {'path': 'notes.txt', 'content': 'hello\n'}),
    ('edit_file', {'path': 'notes.txt', 'replacements': [{'old_text': 'hello', 'new_text': 'hello from temporal'}]}),
    ('read_file', {'path': 'notes.txt'}),
    ('shell', {'command': 'cat notes.txt && echo done'}),
]


def _script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call the file and shell tools in turn, then answer with every tool result, one per line."""
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if len(returns) < len(STEPS):
        name, args = STEPS[len(returns)]
        return ModelResponse(parts=[ToolCallPart(name, args)])
    return ModelResponse(parts=[TextPart('\n---\n'.join(part.model_response_str() for part in returns))])


async def _stream_script(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    """`_script`, streamed: the durable model request streams."""
    for part in _script(messages, info).parts:
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}


coder_agent = Agent(
    FunctionModel(_script, stream_function=_stream_script),
    name='coder_agent',
    deps_type=type(None),
    capabilities=[
        LocalWorkspace[None](WORK),
        Coder[None](),
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class CoderWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await coder_agent.run(prompt)).output


def _delegate(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """The parent delegates a write to `self`; the delegate writes the file; each then answers with its tool result."""
    first = messages[0].parts[-1]
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if returns:
        return ModelResponse(parts=[TextPart(returns[-1].model_response_str())])
    if isinstance(first, UserPromptPart) and first.content == 'Delegate it':
        return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'self', 'task': 'Write it'})])
    return ModelResponse(
        parts=[ToolCallPart('write_file', {'path': 'delegated.txt', 'content': 'from the delegate\n'})]
    )


async def _stream_delegate(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    for part in _delegate(messages, info).parts:
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}


delegating_agent = Agent(
    FunctionModel(_delegate, stream_function=_stream_delegate),
    name='delegating_coder_agent',
    deps_type=type(None),
    capabilities=[
        LocalWorkspace[None](WORK),
        Coder[None](),
        TemporalDurability[None](activity_config=ACTIVITY_CONFIG),
    ],
)


@workflow.defn
class DelegatingWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await delegating_agent.run(prompt)).output


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


@pytest.fixture
def workspace() -> Iterator[Path]:
    """Empty the checked-in workspace around each run, keeping its `.gitignore`."""

    def clear() -> None:
        for entry in WORK.iterdir():
            if entry.name == '.gitignore':
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    clear()
    yield WORK
    clear()


@pytest.mark.parametrize('runner', ['sandboxed', 'passthrough'])
async def test_file_and_shell_tools_run_as_activities(client: Client, workspace: Path, runner: str) -> None:
    restrictions = _PASSTHROUGH if runner == 'passthrough' else _SANDBOXED
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CoderWorkflow],
        plugins=[AgentPlugin(coder_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=restrictions),
    ):
        output = await client.execute_workflow(
            CoderWorkflow.run,
            'edit the notes',
            id=f'test_coder_temporal_{runner}',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=60),
        )

    assert (workspace / 'notes.txt').read_text() == 'hello from temporal\n'
    written, edited, read, shell = output.split('\n---\n')
    assert written == 'Wrote 6 chars (1 lines) to notes.txt.'
    assert edited == 'Edited notes.txt.'
    assert read.endswith('1\thello from temporal\n')
    assert shell.startswith('hello from temporal\ndone\n')


async def test_delegation_to_self_runs_in_the_workflow(client: Client, workspace: Path) -> None:
    """An activity has no `ctx.model` to run an agent on, so the delegate to `self` runs in the workflow."""
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[DelegatingWorkflow],
        plugins=[AgentPlugin(delegating_agent)],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_SANDBOXED),
    ):
        output = await client.execute_workflow(
            DelegatingWorkflow.run,
            'Delegate it',
            id='test_coder_temporal_delegation',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=60),
        )

    assert (workspace / 'delegated.txt').read_text() == 'from the delegate\n'
    assert 'Wrote 18 chars (1 lines) to delegated.txt.' in output


def test_workspace_capabilities_register_their_toolsets() -> None:
    """Each leaf toolset has a stable id, which Temporal needs to name its activities."""
    agent = Agent(
        FunctionModel(_script, stream_function=_stream_script),
        name='workspace_capabilities_agent',
        capabilities=[
            LocalWorkspace(WORK),
            FileSystem(),
            Shell(),
            Shell(id='background_shell', tools=['start_command', 'check_command']),
            RepoContext(expose_inventory_tool=True),
            ToolOutputLimits(),
            TemporalDurability(activity_config=ACTIVITY_CONFIG),
        ],
    )
    ids: set[str | None] = set()
    for toolset in agent.toolsets:
        toolset.apply(lambda leaf: ids.add(leaf.id))
    assert {'file_system', 'shell', 'background_shell', 'repo_context', 'tool_output_limits'} <= ids
