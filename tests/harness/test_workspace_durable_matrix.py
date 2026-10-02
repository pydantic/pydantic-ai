"""LocalWorkspace capability matrix on real Prefect and DBOS engines."""

from __future__ import annotations

import gc
import json
import re
import warnings
from collections.abc import Generator
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

pytest.importorskip('dbos')
pytest.importorskip('temporalio')

from dbos import DBOS, DBOSConfig
from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.dbos import DBOSDurability
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileSystem
from pydantic_ai_harness.shell import Shell
from tests.harness._temporal import skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir


@pytest.fixture
def prefect_server() -> Generator[None, None, None]:
    # Prefect is an optional core extra; its absence must not hide the DBOS cells.
    pytest.importorskip('prefect')
    from prefect.settings import PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED, temporary_settings
    from prefect.testing.utilities import prefect_test_harness

    with temporary_settings({PREFECT_SERVER_SERVICES_TASK_RUN_RECORDER_ENABLED: False}):
        with prefect_test_harness(server_startup_timeout=120):
            yield
    # Prefect's test server leaves client sockets for GC on Python 3.14.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='unclosed.*socket', category=ResourceWarning)
        gc.collect()


def _agent(root: Path, engine: str, capability: str, vetoes: list[str]) -> Agent[None, str]:
    scripts: dict[str, list[tuple[str, dict[str, str]]]] = {
        'filesystem': [
            ('write_file', {'path': 'file.txt', 'content': 'first'}),
            ('edit_file', {'path': 'file.txt', 'old_text': 'first', 'new_text': 'second'}),
            ('write_file', {'path': 'blocked.txt', 'content': 'denied'}),
        ],
        'shell': [
            ('run_command', {'command': 'cd dir && pwd'}),
            ('run_command', {'command': 'pwd'}),
            ('start_command', {'command': 'echo job-done'}),
            ('check_command', {'command_id': '__ID__'}),
        ],
        'coder': [
            ('write_file', {'path': 'code.py', 'content': 'print("first")\n'}),
            ('edit_file', {'path': 'code.py', 'old_text': 'first', 'new_text': 'second'}),
            ('shell', {'command': 'python3 code.py'}),
        ],
    }
    script = scripts[capability]

    def model(messages: list[ModelMessage], info: object) -> ModelResponse:
        returns = [p for m in messages for p in m.parts if isinstance(p, (ToolReturnPart, RetryPromptPart))]
        # Past the script's end, a shell job still running gets checked again.
        if len(returns) >= len(script) and '[status: running]' not in str(returns[-1].content):
            return ModelResponse(parts=[TextPart('done')])
        name, original = script[min(len(returns), len(script) - 1)]
        args = dict(original)
        if args.get('command_id') == '__ID__':
            ids = re.findall(r'ID: ([0-9a-f]{32})', ' '.join(str(p.content) for p in returns))
            args['command_id'] = ids[-1] if ids else 'missing'
        return ModelResponse(parts=[ToolCallPart(name, args)])

    async def stream(messages: list[ModelMessage], info: object):
        for index, part in enumerate(model(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {index: DeltaToolCall(name=part.tool_name, json_args=json.dumps(part.args))}

    cap = {'filesystem': FileSystem, 'shell': lambda: Shell(persist_cwd=True), 'coder': Coder}[capability]()
    if engine == 'prefect':
        from pydantic_ai.durable_exec.prefect import PrefectDurability

        durability = PrefectDurability()
    else:
        # A run with a workspace is sequential on DBOS; saying so keeps it from warning.
        durability = DBOSDurability(parallel_execution_mode='sequential')
    agent = Agent(
        FunctionModel(model, stream_function=stream),
        name=f'matrix_{engine}_{capability}_{uuid4().hex}',
        capabilities=[LocalWorkspace(root), cap, durability],
    )

    @agent.on_event(FileChangeRequestEvent)
    async def veto(ctx: RunContext[None], event: FileChangeRequestEvent) -> None:
        if event.path.endswith('blocked.txt'):
            vetoes.append(event.path)
            event.cancel('denied')

    return agent


@pytest.mark.parametrize('capability', ['coder', 'shell', 'filesystem'])
async def test_prefect_workspace_capabilities(tmp_path: Path, prefect_server: None, capability: str) -> None:
    from prefect import flow

    (tmp_path / 'dir').mkdir()
    vetoes: list[str] = []
    agent = _agent(tmp_path, 'prefect', capability, vetoes)

    @flow
    async def run() -> list[str]:
        result = await agent.run('go')
        assert result.output == 'done'
        return [str(p.content) for m in result.all_messages() for p in m.parts if p.part_kind == 'tool-return']

    outputs = await run()
    _assert_results(tmp_path, capability, outputs, vetoes)


@pytest.mark.parametrize('capability', ['coder', 'shell', 'filesystem'])
async def test_dbos_workspace_capabilities(tmp_path: Path, capability: str) -> None:
    (tmp_path / 'dir').mkdir()
    vetoes: list[str] = []
    agent = _agent(tmp_path, 'dbos', capability, vetoes)

    @DBOS.workflow(name=f'matrix_{capability}_{uuid4().hex}')
    async def run() -> list[str]:
        result = await agent.run('go')
        assert result.output == 'done'
        return [str(p.content) for m in result.all_messages() for p in m.parts if p.part_kind == 'tool-return']

    DBOS(
        config=DBOSConfig(
            name='harness_matrix', system_database_url=f'sqlite:///{tmp_path / "db.sqlite"}', run_admin_server=False
        )
    )
    DBOS.launch()
    try:
        outputs = await run()
    finally:
        DBOS.destroy()
    _assert_results(tmp_path, capability, outputs, vetoes)


def _assert_results(root: Path, capability: str, outputs: list[str], vetoes: list[str]) -> None:
    if capability == 'filesystem':
        assert (root / 'file.txt').read_text() == 'second'
        assert not (root / 'blocked.txt').exists()
        assert len(vetoes) == 1
        assert 'denied' in outputs[-1]
    elif capability == 'coder':
        assert (root / 'code.py').read_text() == 'print("second")\n'
        assert 'second' in outputs[-1]
    else:
        assert str(root / 'dir') in outputs[0] and str(root / 'dir') in outputs[1]
        assert 'ID: ' in outputs[2]
        assert '[status: finished]' in outputs[-1]


_restart_ready: anyio.Event | None = None
_restart_continue: anyio.Event | None = None


async def _restart_model(messages: list[ModelMessage], info: object) -> ModelResponse:
    returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
    if not returns:
        return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'cd dir && pwd'})])
    if len(returns) == 1:
        assert _restart_ready is not None and _restart_continue is not None
        _restart_ready.set()
        # The worker stops here, after the first durable command and before the next model turn.
        await _restart_continue.wait()
        return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'pwd'})])
    return ModelResponse(parts=[TextPart('done')])


@workflow.defn
class ShellRestartWorkflow(PydanticAIWorkflow):
    agent: Agent[None, str]

    @workflow.run
    async def run(self) -> list[str]:
        result = await self.agent.run('go')
        return [str(p.content) for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]


@pytest.mark.temporal
@pytest.mark.xdist_group(name='harness-temporal')
@skip_temporal_sandbox_on_314
async def test_temporal_new_worker_keeps_shell_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    global _restart_ready, _restart_continue
    # Debug mode runs workflow activations inline, but this test needs the first activation to outlive its worker.
    monkeypatch.delenv('TEMPORAL_DEBUG', raising=False)
    (tmp_path / 'dir').mkdir()
    _restart_ready = anyio.Event()
    _restart_continue = anyio.Event()
    agent = Agent(
        FunctionModel(_restart_model),
        name='restart_shell',
        capabilities=[LocalWorkspace(tmp_path), Shell(persist_cwd=True), TemporalDurability()],
    )
    ShellRestartWorkflow.agent = agent
    ShellRestartWorkflow.__pydantic_ai_agents__ = [agent]
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types')
    )
    queue = f'restart-{uuid4().hex}'
    try:
        async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
            client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
            async with Worker(client, task_queue=queue, workflows=[ShellRestartWorkflow], workflow_runner=runner):
                handle = await client.start_workflow(
                    ShellRestartWorkflow.run, id=queue, task_queue=queue, execution_timeout=timedelta(seconds=120)
                )
                with anyio.fail_after(20):
                    await _restart_ready.wait()
            # Unblock the model activity only after the old worker is gone; the new worker executes call 2.
            assert _restart_continue is not None
            _restart_continue.set()
            async with Worker(client, task_queue=queue, workflows=[ShellRestartWorkflow], workflow_runner=runner):
                with anyio.fail_after(20):
                    outputs = await handle.result()
        assert str(tmp_path / 'dir') in outputs[0]
        assert str(tmp_path / 'dir') in outputs[1]
    finally:
        _restart_ready = None
        _restart_continue = None
