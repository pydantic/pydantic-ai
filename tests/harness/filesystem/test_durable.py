"""Durable execution of file-change approvals on a local workspace."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileSystem
from tests.harness._temporal import skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal'), skip_temporal_sandbox_on_314]


@workflow.defn
class FileWorkflow(PydanticAIWorkflow):
    agent: Agent[None, str]

    @workflow.run
    async def run(self) -> str:
        result = await self.agent.run('try to change a file')
        return result.output


def _default_model(messages: object, info: object) -> ModelResponse:
    if not any(p.part_kind == 'tool-return' for m in messages for p in m.parts):  # type: ignore[attr-defined]
        path = messages[0].parts[0].content  # type: ignore[attr-defined]
        return ModelResponse(parts=[ToolCallPart('write_file', {'path': path, 'content': 'changed'})])
    return ModelResponse(parts=[TextPart('done')])


async def _default_stream(messages: object, info: object):
    for index, part in enumerate(_default_model(messages, info).parts):
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {index: DeltaToolCall(name=part.tool_name, json_args=json.dumps(part.args))}


_default_agent = Agent(
    FunctionModel(_default_model, stream_function=_default_stream),
    name='default_runner_file_veto',
    capabilities=[LocalWorkspace('/'), FileSystem(), TemporalDurability()],
)


@_default_agent.on_event(FileChangeRequestEvent)
async def _default_veto(ctx: RunContext[None], event: FileChangeRequestEvent) -> None:
    event.cancel('denied')


@workflow.defn
class DefaultFileWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [_default_agent]

    @workflow.run
    async def run(self, root: str) -> list[str]:
        result = await _default_agent.run(root + '/protected.txt')
        return [str(p.content) for m in result.all_messages() for p in m.parts if p.part_kind == 'tool-return']


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'  # Temporal's test server requires an asyncio event loop.


async def test_temporal_default_runner_veto(tmp_path: Path) -> None:
    (tmp_path / 'protected.txt').write_text('original')
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        async with Worker(client, task_queue='default-veto', workflows=[DefaultFileWorkflow]):
            returns = await client.execute_workflow(
                DefaultFileWorkflow.run,
                str(tmp_path),
                id=uuid4().hex,
                task_queue='default-veto',
                execution_timeout=timedelta(seconds=120),
            )
    assert 'denied' in str(returns)
    assert (tmp_path / 'protected.txt').read_text() == 'original'


@pytest.mark.parametrize(
    'capability,tool_name,args',
    [
        pytest.param(capability, tool_name, args, id=f'{tool_name}-{name}')
        for name, capability in [('filesystem', FileSystem()), ('coder', Coder())]
        for tool_name, args in [
            ('write_file', {'path': 'note.txt', 'content': 'changed'}),
            ('edit_file', {'path': 'note.txt', 'old_text': 'original', 'new_text': 'changed'}),
            ('create_directory', {'path': 'newdir'}),
        ]
        # Coder does not expose create_directory.
        if not (name == 'coder' and tool_name == 'create_directory')
    ],
)
async def test_temporal_vetoes_before_mutation(
    tmp_path: Path, capability: FileSystem | Coder, tool_name: str, args: dict[str, str]
) -> None:
    (tmp_path / 'note.txt').write_text('original')
    requests: list[str] = []

    def model(messages: object, info: object) -> ModelResponse:
        if not any(p.part_kind == 'tool-return' for m in messages for p in m.parts):  # type: ignore[attr-defined]
            return ModelResponse(parts=[ToolCallPart(tool_name, args)])
        return ModelResponse(parts=[TextPart('done')])

    async def stream(messages: object, info: object):
        for index, part in enumerate(model(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {index: DeltaToolCall(name=part.tool_name, json_args=json.dumps(part.args))}

    agent = Agent(
        FunctionModel(model, stream_function=stream),
        name='durable_file_veto',
        capabilities=[LocalWorkspace(tmp_path), capability, TemporalDurability()],
    )

    @agent.on_event(FileChangeRequestEvent)
    async def veto(ctx: RunContext[None], event: FileChangeRequestEvent) -> None:
        requests.append(event.operation)
        event.cancel('denied')

    FileWorkflow.agent = agent
    FileWorkflow.__pydantic_ai_agents__ = [agent]
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        # The test's dynamically configured agent lives in this test module; production
        # agents are normally constructed at module scope instead of injected by the test.
        runner = SandboxedWorkflowRunner(
            restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types')
        )
        async with Worker(client, task_queue='file-veto', workflows=[FileWorkflow], workflow_runner=runner):
            assert (
                await client.execute_workflow(
                    FileWorkflow.run, id=uuid4().hex, task_queue='file-veto', execution_timeout=timedelta(seconds=120)
                )
                == 'done'
            )
    assert requests == [tool_name.removesuffix('_file') if tool_name != 'create_directory' else 'create_directory']
    assert (tmp_path / 'note.txt').read_text() == 'original'
    assert not (tmp_path / 'newdir').exists()
