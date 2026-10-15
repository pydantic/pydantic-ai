"""The `orchestrator` plugin's tools, driving a workspace's CLAI: a real web app served in this process.

The workspace's `clai2 --web` is not spawned: the launcher's process stand-in serves an in-process shell on
the port and with the token the launcher chose, so every tool talks to the real web UI over loopback.
"""

import asyncio
import io
import os
import signal
import subprocess
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import anyio
import httpx
import pytest
from pydantic import TypeAdapter
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.ask_user import AskUser, AskUserAnswer, AskUserRequest, AskUserResponse
from pydantic_clai2._app import _Shell, create_shell  # pyright: ignore[reportPrivateUsage]
from pydantic_clai2.builtin_plugins import orchestrator as orchestrator_module
from pydantic_clai2.builtin_plugins.orchestrator import Orchestrator, OrchestratorPlugin, SessionAnswer, summarize
from pydantic_clai2.builtin_plugins.web import WebPlugin
from pydantic_clai2.builtin_plugins.web.app import HEADER
from pydantic_clai2.builtin_plugins.web.protocol import Item, WorkspacesInfo
from pydantic_clai2.builtin_plugins.web.workspaces import listing, open_workspace
from pydantic_clai2.builtin_plugins.workspaces import manager
from pydantic_clai2.builtin_plugins.workspaces.sessions import Launcher
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, SessionEnd, SessionStart, load_plugin
from pydantic_clai2.plugins.loader import PluginLoader

pytestmark = pytest.mark.anyio

QUESTION = {
    'questions': [{'question': 'Which colour?', 'header': 'Colour', 'options': [{'label': 'Red'}, {'label': 'Blue'}]}]
}


def git(directory: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(directory), *args], check=True, capture_output=True, text=True, timeout=15
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    repo = tmp_path / 'project'
    repo.mkdir()
    git(repo, 'init', '-b', 'main')
    git(repo, 'config', 'user.name', 'CLAI Test')
    git(repo, 'config', 'user.email', 'clai@example.com')
    (repo / 'app.py').write_text('print("hi")\n')
    git(repo, 'add', '.')
    git(repo, 'commit', '-m', 'First')
    return repo


def prompt_of(messages: Sequence[ModelMessage]) -> str:
    return next(
        part.content
        for message in reversed(messages)
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )


def child_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """The workspace's model: answers, asks a question, blocks in a tool, or fails, as the prompt says."""
    last = messages[-1]
    if isinstance(last, ModelRequest) and not any(isinstance(part, UserPromptPart) for part in last.parts):
        return ModelResponse(parts=[TextPart('Done after the tool.')])
    prompt = prompt_of(messages)
    if 'ask' in prompt:
        return ModelResponse(parts=[ToolCallPart('ask_user_question', QUESTION)])
    if 'block' in prompt:
        return ModelResponse(parts=[ToolCallPart('block', {})])
    if 'fail' in prompt:
        raise RuntimeError('the model broke')
    return ModelResponse(parts=[TextPart(f'Hello from the workspace: {prompt}. ' + 'x' * 5000)])


async def child_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    """`child_model`, streamed, as CLAI requests it."""
    for part in child_model(messages, info).parts:
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}


@dataclass
class Child:
    """The workspace's CLAI, served in this process when the launcher starts it."""

    shell: _Shell[None, str]
    web: WebPlugin
    blocked: anyio.Event = field(default_factory=anyio.Event)


@dataclass
class InProcess:
    """`Processes` that serve `child` instead of running `clai2 --web`."""

    child: Child
    running: set[int] = field(default_factory=set[int])
    signals: list[int] = field(default_factory=list[int])

    def spawn(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> int:
        web = self.child.web
        web.port = int(argv[argv.index('--port') + 1])
        web.token = Path(argv[argv.index('--token-file') + 1]).read_text(encoding='utf-8').strip()
        asyncio.get_running_loop().create_task(web.start())
        self.running.add(4242)
        return 4242

    def alive(self, pid: int) -> bool:
        return pid in self.running

    def signal(self, pid: int, signum: int) -> None:
        self.signals.append(signum)
        self.running.discard(pid)
        asyncio.get_running_loop().create_task(self.child.web.stop())


@asynccontextmanager
async def child_serving(tmp_path: Path) -> AsyncGenerator[Child]:
    tools = Capability[None]()
    child = None

    @tools.tool_plain
    async def block() -> str:
        """Wait forever."""
        assert child is not None
        child.blocked.set()
        await anyio.sleep_forever()
        return 'never'  # pragma: no cover

    agent = Agent(FunctionModel(child_model, stream_function=child_stream), deps_type=type(None), capabilities=[tools])
    shell = create_shell(
        agent,
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO(), width=100),
        settings=None,
        store=SettingsStore(tmp_path / 'child.db'),
        builtin_plugins=[
            PluginSettings(id='web', factory='pydantic_clai2.builtin_plugins.web', settings={'open_browser': False}),
            PluginSettings(id='workspaces', factory='pydantic_clai2.builtin_plugins.workspaces'),
        ],
        project=ProjectSettings(),
        headless=True,
    )
    await shell.loader.load_all()
    [web] = [entry.loaded.plugin for entry in shell.loader.entries() if entry.name == 'web' and entry.loaded]
    assert isinstance(web, WebPlugin)
    child = Child(shell=shell, web=web)
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(shell.run)
            yield child
            group.cancel_scope.cancel()
    finally:
        await shell.loader.close('exit')


@dataclass
class Parent:
    loader: PluginLoader[None]
    plugin: OrchestratorPlugin
    tools: Orchestrator
    processes: InProcess


@asynccontextmanager
async def parent(
    tmp_path: Path, child: Child, monkeypatch: pytest.MonkeyPatch, *, workspaces: bool = True
) -> AsyncGenerator[Parent]:
    """A loader with `orchestrator`, and `workspaces` unless told otherwise, whose launcher serves `child`."""
    processes = InProcess(child)
    launcher = Launcher(processes=processes, python='python', interval=0.01, startup_timeout=10, stop_timeout=5)
    monkeypatch.setattr(manager, 'Launcher', lambda: launcher)
    store = SettingsStore(tmp_path / 'parent.db')
    builtin = [PluginSettings(id='orchestrator', factory='pydantic_clai2.builtin_plugins.orchestrator')]
    if workspaces:
        builtin.append(PluginSettings(id='workspaces', factory='pydantic_clai2.builtin_plugins.workspaces'))
    loader = PluginLoader[None](
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(FunctionModel(child_model)), settings=store.load()),
        builtin=builtin,
    )
    await loader.load_all()
    [plugin] = [entry.loaded.plugin for entry in loader.entries() if entry.name == 'orchestrator' and entry.loaded]
    assert isinstance(plugin, OrchestratorPlugin)
    try:
        yield Parent(loader=loader, plugin=plugin, tools=plugin.orchestrator, processes=processes)
    finally:
        await loader.close('exit')


async def created(tools: Orchestrator, repository: Path) -> str:
    made = await tools.workspace_create(kind='local-worktree', repo=str(repository), ref='main')
    return made.split()[2].removesuffix('.')


async def test_a_workspace_session_from_start_to_answer(
    tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make a workspace, start its CLAI with a prompt, wait, read, steer, answer its question, and interrupt."""
    async with child_serving(tmp_path) as child, parent(tmp_path, child, monkeypatch) as made:
        tools = made.tools
        assert await tools.workspace_list() == 'No workspaces yet.\nKinds: local-worktree.'
        workspace = await created(tools, repository)
        assert workspace.startswith('project-')
        listed = await tools.workspace_list()
        assert listed.startswith(f'{workspace}  ready  local-worktree  clai-{workspace}  ')

        started = await tools.session_start(workspace, 'say hi')
        port = child.web.port
        assert started == (
            f'Workspace {workspace} runs at http://127.0.0.1:{port}/; the prompt is sent. Wait with `session_wait`.'
        )
        waited = await tools.session_wait(workspace, timeout=10)
        assert waited.startswith(f'Workspace {workspace} is idle; last turn: completed. Its final answer:\n')
        assert 'Hello from the workspace: say hi.' in waited and waited.endswith('… (1,034 more characters)')
        assert child.web.token not in started + waited + listed

        read = await tools.session_read(workspace, last=1)
        assert read.startswith(
            f"Workspace {workspace}: idle; session 'say hi'.\nLast turn: completed.\nLast 1 of 2 items:\n"
        )
        assert (await tools.session_read(workspace, last=0)).endswith('Last 0 of 2 items:')

        assert await tools.session_send(workspace, 'ask me') == 'Sent. Wait with `session_wait`.'
        asking = await tools.session_wait(workspace, timeout=10)
        [request] = (await tools.view(await tools.workspaces().get(workspace))).questions
        assert asking == (
            f'Workspace {workspace} asks:\nQuestion request {request}:\n  - Colour: Which colour? Options: Red, Blue\n'
            'Answer with `session_answer`.'
        )
        assert f'Question request {request}' in await tools.session_read(workspace)
        assert (
            await tools.session_answer(workspace, 'nope')
            == f"Workspace {workspace} has no open question or dialog 'nope'."
        )
        answered = await tools.session_answer(workspace, request, [SessionAnswer(header='Colour', selected=['Blue'])])
        assert answered == 'Answered. Wait with `session_wait`.'
        assert 'Done after the tool.' in await tools.session_wait(workspace, timeout=10)

        await tools.session_send(workspace, 'ask me again')
        await tools.session_wait(workspace, timeout=10)
        [request] = (await tools.view(await tools.workspaces().get(workspace))).questions
        await tools.session_answer(workspace, request, [SessionAnswer(header='Colour', custom='Teal')])
        assert 'Done after the tool.' in await tools.session_wait(workspace, timeout=10)

        await tools.session_send(workspace, 'block')
        await child.blocked.wait()
        assert 'is still running' in await tools.session_wait(workspace, timeout=0.05)
        assert 'a turn is running' in await tools.session_read(workspace)
        assert await tools.session_interrupt(workspace) == 'Interrupted.'
        assert 'last turn: cancelled' in await tools.session_wait(workspace, timeout=10)

        await tools.session_send(workspace, 'fail')
        await tools.session_wait(workspace, timeout=10)
        assert 'Last turn: failed (' in await tools.session_read(workspace)


async def test_answering_a_dialog(tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with child_serving(tmp_path) as child, parent(tmp_path, child, monkeypatch) as made:
        tools = made.tools
        workspace = await created(tools, repository)
        await tools.session_start(workspace, 'hi')
        await tools.session_wait(workspace, timeout=10)
        view = await tools.view(await tools.workspaces().get(workspace))
        results: list[int] = []
        async with (
            httpx.AsyncClient(base_url=view.client.url, trust_env=False) as page,
            anyio.create_task_group() as group,
        ):

            async def act() -> None:
                answer = await page.post('/api/plugins/workspaces/fields/add/action', headers={HEADER: child.web.token})
                results.append(answer.status_code)

            group.start_soon(act)
            with anyio.fail_after(10):
                while not view.dialogs:
                    await anyio.sleep(0.01)
            [dialog] = view.dialogs
            assert f'Dialog {dialog} (text): New workspace kind' in await tools.session_wait(workspace, timeout=1)
            assert await tools.session_answer(workspace, dialog, value='docs') == 'Answered the dialog.'
        assert results == [200]


async def test_destroying_needs_the_users_pick(
    tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`workspace_destroy` refuses until the user picked `Destroy ID` in `ask_user_question`; then the CLAI stops."""
    async with child_serving(tmp_path) as child, parent(tmp_path, child, monkeypatch) as made:
        tools = made.tools
        workspace = await created(tools, repository)
        await tools.session_start(workspace, 'hi')
        await tools.session_wait(workspace, timeout=10)
        path = (await tools.workspaces().get(workspace)).path
        asked: list[AskUserRequest] = []

        async def user(request: AskUserRequest) -> AskUserResponse:
            asked.append(request)
            if len(asked) == 1:
                return AskUserResponse(cancelled=True)
            picked = f'Keep {workspace}' if len(asked) == 2 else f'Destroy {workspace}'
            return AskUserResponse(answers=(AskUserAnswer(header='Destroy', selected=(picked,)),))

        question = {
            'questions': [
                {
                    'question': f'Destroy workspace {workspace}?',
                    'header': 'Destroy',
                    'options': [{'label': f'Destroy {workspace}'}, {'label': f'Keep {workspace}'}],
                }
            ]
        }
        script = [
            ToolCallPart('workspace_destroy', {'workspace': workspace}),
            ToolCallPart('ask_user_question', question),
            ToolCallPart('workspace_destroy', {'workspace': workspace}),
            ToolCallPart('ask_user_question', question),
            ToolCallPart('workspace_destroy', {'workspace': workspace}),
            ToolCallPart('ask_user_question', question),
            ToolCallPart('workspace_destroy', {'workspace': workspace}),
        ]
        results: list[str] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            last = messages[-1]
            assert isinstance(last, ModelRequest)
            results.extend(
                str(part.content)
                for part in last.parts
                if isinstance(part, ToolReturnPart) and part.tool_name != 'ask_user_question'
            )
            return ModelResponse(parts=[script.pop(0)] if script else [TextPart('Done.')])

        capabilities = [*made.loader.capabilities(), AskUser[None](answerer=user)]
        agent = Agent(FunctionModel(model), deps_type=type(None), capabilities=capabilities)
        await agent.run('destroy it')
        assert results[0].startswith(
            f'Not destroyed. Ask the user first with `ask_user_question`, offering `Destroy {workspace}`'
        )
        assert results[1].startswith('Not destroyed.'), 'declining is no confirmation'
        assert results[2].startswith('Not destroyed.'), 'keeping it is no confirmation'
        assert results[3] == (f'Destroyed workspace {workspace}; its checkout at {path} was removed, its branch kept.')
        assert not path.exists() and made.processes.signals == [signal.SIGINT]
        assert tools.links == {}
        assert await tools.workspace_list() == 'No workspaces yet.\nKinds: local-worktree.'


async def test_problems_come_back_as_errors(tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with child_serving(tmp_path) as child, parent(tmp_path, child, monkeypatch) as made:
        [capability] = made.plugin.get_capabilities()
        assert isinstance(capability, Capability)
        tools = made.tools
        assert (await tools.workspace_list()).startswith('No workspaces')
        workspace = await created(tools, repository)
        reading = orchestrator_module.reported(tools.session_read)
        assert await reading(workspace) == (
            f'Error: Workspace {workspace} has no CLAI running. Start it with `session_start`.'
        )
        assert await orchestrator_module.reported(tools.workspace_create)(kind='docker') == (
            "Error: No workspace kind 'docker'. Kinds: local-worktree."
        )

        # A CLAI that stopped: the connection ends, and the next use connects again once it runs.
        await tools.session_start(workspace, 'block')
        await child.blocked.wait()
        stopped: list[str] = []
        async with anyio.create_task_group() as group:

            async def waiting() -> None:
                stopped.append(await tools.session_wait(workspace, timeout=10))

            group.start_soon(waiting)
            await anyio.wait_all_tasks_blocked()
            await child.web.stop()
        assert stopped == [f'The CLAI of workspace {workspace} stopped.']
        await child.web.start()
        assert await tools.session_interrupt(workspace) == 'Interrupted.'
        assert 'last turn: cancelled' in await tools.session_wait(workspace, timeout=10)
        await tools.disconnect(workspace)
        assert 'Last turn' not in await tools.session_read(workspace)
        waited = await tools.session_wait(workspace, timeout=10)
        assert waited.startswith(f'Workspace {workspace} is idle; last turn: no turn ran.')

        # A CLAI that does not answer cannot be connected to.
        current = await tools.workspaces().get(workspace)
        assert current.session is not None
        await tools.disconnect(workspace)
        await child.web.stop()
        monkeypatch.setattr(orchestrator_module, 'CONNECT_TIMEOUT', 5)
        assert await reading(workspace) == f'Error: Could not connect to the CLAI of workspace {workspace}.'

        await made.plugin.on_session_end(SessionEnd(reason='exit'))
        assert tools.links == {}


async def test_without_the_workspaces_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with child_serving(tmp_path) as child, parent(tmp_path, child, monkeypatch, workspaces=False) as made:
        listing = orchestrator_module.reported(made.tools.workspace_list)
        assert (
            await listing()
            == 'Error: The `workspaces` plugin is off. Ask the user to run `/plugins enable workspaces`.'
        )
    host = PluginHost(name='orchestrator', console=Console(file=io.StringIO()), settings={})
    alone = load_plugin(OrchestratorPlugin, host)
    assert isinstance(alone.plugin, OrchestratorPlugin)
    assert 'is off' in await orchestrator_module.reported(alone.plugin.orchestrator.workspace_list)()


def test_items_in_short() -> None:
    tool = Item(id='t', kind='tool', name='run_code', args='{"code": "print(1)"}', text='1', state='done')
    assert summarize(tool) == '- tool `run_code` (done): {"code": "print(1)"} -> 1'
    assert summarize(Item(id='a', kind='assistant', text=' short ')) == '- assistant: short'


async def test_tools_never_see_a_token(tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The record keeps the token in a file; nothing a tool returns, nor the workspace's listing, carries it."""
    async with child_serving(tmp_path) as child, parent(tmp_path, child, monkeypatch) as made:
        tools = made.tools
        workspace = await created(tools, repository)
        said = [await tools.session_start(workspace, 'hi'), await tools.session_wait(workspace, timeout=10)]
        said += [await tools.session_read(workspace, last=10), await tools.workspace_list()]
        record = await tools.workspaces().get(workspace)
        assert record.session is not None and replace(record).session == record.session
        assert all(child.web.token not in text for text in said)


async def test_the_sidebar_lists_and_opens_workspaces(
    tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The web UI's Workspaces section reads the registry, and only its own page gets a workspace's token."""
    async with child_serving(tmp_path) as child, parent(tmp_path, child, monkeypatch) as made:
        workspace = await created(made.tools, repository)
        # The workspace's own CLAI has `workspaces` on too and shares the registry, so its page serves the section.
        await made.tools.session_start(workspace, 'hi')
        headers = {HEADER: child.web.token}
        async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{child.web.port}', trust_env=False) as page:
            shown = await page.get('/api/workspaces', headers=headers)
            info = TypeAdapter(WorkspacesInfo).validate_json(shown.content)
            [row] = info.workspaces
            assert info.enabled
            assert (row.id, row.status, row.url) == (workspace, 'running', f'http://127.0.0.1:{child.web.port}/')
            opened = await page.post(f'/api/workspaces/{workspace}/open', headers=headers)
            assert opened.json() == {'url': f'http://127.0.0.1:{child.web.port}/?token={child.web.token}'}
            assert (await page.post('/api/workspaces/nope-1/open', headers=headers)).status_code == 404
        await made.tools.session_wait(workspace, timeout=10)
    assert await listing(None) == WorkspacesInfo(enabled=False, workspaces=[])
    with pytest.raises(LookupError, match='The workspaces plugin is off'):
        await open_workspace(None, 'any')
