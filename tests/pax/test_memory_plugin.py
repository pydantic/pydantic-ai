"""The built-in `memory` plugin keeps the model's notes per project, for later sessions in that project to recall."""

import json
import os
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    TextContent,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_pax import DEFAULT_PLUGINS
from pydantic_ai_pax.app import create_stock_agent
from pydantic_ai_pax.builtin_plugins import memory
from pydantic_ai_pax.builtin_plugins.memory import memory_root, project_folder
from pydantic_ai_pax.config import PluginSettings, Settings
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import TurnStart
from pydantic_ai_pax.runtime.session_runtime import SessionRuntime, create_shell

[DECLARATION] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'memory']
MEMORY_TOOLS = {'write_memory', 'read_memory', 'delete_memory', 'search_memory'}
NOTE = '- The parser lives in src/parse.py'
VOLATILE = {'timestamp', 'run_id', 'conversation_id'}


def git(directory: Path, *args: str) -> None:
    subprocess.run(['git', '-C', str(directory), *args], check=True, capture_output=True, timeout=15)


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', os.devnull)
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    repo = tmp_path / 'repo'
    repo.mkdir()
    git(repo, 'init', '--quiet', '--initial-branch=main')
    git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '--quiet', '--allow-empty', '-m', 'first')
    return repo.resolve()


@dataclass
class Requests:
    """Stands in for every model Pax resolves: saves a note when asked to remember, and keeps what each request sent."""

    sent: list[str] = field(default_factory=list[str])
    tools: list[set[str]] = field(default_factory=list[set[str]])
    snapshots: list[list[str]] = field(default_factory=list[list[str]])

    def resolve(self, name: str) -> Model:
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            self.tools.append({tool.name for tool in info.function_tools})
            # A snapshot reaches the model as `<memory>`-framed user text, its notebook under a `MEMORY.md` heading.
            self.snapshots.append(
                [
                    item.content.removeprefix('### MEMORY.md\n\n')
                    for message in messages
                    if isinstance(message, ModelRequest)
                    for part in message.parts
                    if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
                    for item in part.content
                    if isinstance(item, TextContent) and item.content.startswith('### MEMORY.md')
                ]
            )
            dumped = json.loads(ModelMessagesTypeAdapter.dump_json(messages))
            for message in dumped:
                for key in VOLATILE:
                    message.pop(key, None)
                for part in message['parts']:
                    part.pop('timestamp', None)
            self.sent.append(json.dumps(dumped))
            last = messages[-1]
            asked = any(isinstance(part, UserPromptPart) and part.content == 'Remember this' for part in last.parts)
            if asked and not any(isinstance(part, ToolReturnPart) for part in last.parts):
                yield {0: DeltaToolCall(name='write_memory', json_args=json.dumps({'content': NOTE}), tool_call_id='m')}
                return
            yield 'Done'

        return FunctionModel(stream_function=stream, model_name=name)


async def session(
    tmp_path: Path, workspace: Path, *, declarations: list[PluginSettings] | None = None
) -> tuple[SessionRuntime[None, str], Requests]:
    shell = create_shell(
        create_stock_agent(),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=Settings(model='openai:gpt-6'),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=[DECLARATION.model_copy(update={'enabled': True})] if declarations is None else declarations,
        project=ProjectSettings(),
        workspace=workspace,
        headless=True,
    )
    requests = Requests()
    shell.session.resolve_model = requests.resolve
    await shell.loader.load_all()
    return shell, requests


async def turns(tmp_path: Path, workspace: Path, *texts: str, **options: list[PluginSettings]) -> Requests:
    shell, requests = await session(tmp_path, workspace, **options)
    try:
        for text in texts:
            ended = await shell.run_turn(TurnStart(text=text), headless=True)
            assert ended.outcome == 'completed', ended.error
    finally:
        await shell.loader.close('exit')
    return requests


def recalled(requests: Requests) -> bool:
    return any(NOTE in snapshot for snapshot in requests.snapshots[-1])


async def test_off_by_default_and_a_note_is_recalled_in_the_next_session(tmp_path: Path, repository: Path) -> None:
    assert not DECLARATION.enabled
    first = await turns(tmp_path, repository, 'Remember this', 'Thanks')
    assert MEMORY_TOOLS <= first.tools[0]
    # Nothing saved yet, so the first run records no snapshot; the next run of the session records the note once,
    # appended to the history, so the request before it is still the prefix of the request after it.
    assert first.snapshots == [[], [], [NOTE]]
    assert first.sent[2].startswith(first.sent[1][:-1])

    saved = memory_root() / str(project_folder(repository)) / 'main' / 'MEMORY.md'
    assert saved.read_text() == NOTE + '\n'
    assert saved.parent.parent.parent.stat().st_mode & 0o777 == 0o700

    # A subfolder of the repository is the same project.
    (repository / 'src').mkdir()
    assert recalled(await turns(tmp_path, repository / 'src', 'Hi'))


async def test_another_project_does_not_see_the_note(tmp_path: Path, repository: Path) -> None:
    await turns(tmp_path, repository, 'Remember this')
    other = tmp_path / 'other'
    other.mkdir()
    requests = await turns(tmp_path, other, 'Hi')
    assert MEMORY_TOOLS <= requests.tools[0]
    assert requests.snapshots == [[]]
    assert project_folder(other) != project_folder(repository)


async def test_worktrees_of_one_repository_share_it(tmp_path: Path, repository: Path) -> None:
    worktree = tmp_path / 'linked'
    git(repository, 'worktree', 'add', '--quiet', '-b', 'side', str(worktree))
    await turns(tmp_path, worktree, 'Remember this')
    assert recalled(await turns(tmp_path, repository, 'Hi'))


async def test_switched_off_it_adds_nothing(tmp_path: Path, repository: Path) -> None:
    await turns(tmp_path, repository, 'Remember this')
    off = await turns(tmp_path / 'off', repository, 'Hi', declarations=[DECLARATION])
    absent = await turns(tmp_path / 'absent', repository, 'Hi', declarations=[])
    assert off.sent == absent.sent
    assert off.tools == absent.tools
    assert not MEMORY_TOOLS & off.tools[0]


async def test_no_memory_in_the_home_folder(tmp_path: Path) -> None:
    home = Path.home()
    home.mkdir(parents=True, exist_ok=True)
    shell, requests = await session(tmp_path, home)
    try:
        assert 'home folder' in await shell.commands.execute_async('/memory')
        ended = await shell.run_turn(TurnStart(text='Hi'), headless=True)
        assert ended.outcome == 'completed', ended.error
    finally:
        await shell.loader.close('exit')
    assert not MEMORY_TOOLS & requests.tools[0]
    assert not memory_root().exists()


async def test_memory_command_names_the_folder(tmp_path: Path, repository: Path) -> None:
    shell, _ = await session(tmp_path, repository)
    try:
        shown = await shell.commands.execute_async('/memory')
    finally:
        await shell.loader.close('exit')
    assert str(memory_root() / str(project_folder(repository)) / 'main') in shown


async def test_not_on_windows(tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, 'supports_local_workspace', lambda: False)
    shell, requests = await session(tmp_path, repository)
    try:
        assert 'Windows' in await shell.commands.execute_async('/memory')
        ended = await shell.run_turn(TurnStart(text='Hi'), headless=True)
        assert ended.outcome == 'completed', ended.error
    finally:
        await shell.loader.close('exit')
    assert not MEMORY_TOOLS & requests.tools[0]


def test_folder_names_read_as_the_project_and_stay_apart(tmp_path: Path) -> None:
    named = tmp_path / 'My Repo!'
    unnamed = tmp_path / '+++'
    elsewhere = tmp_path / 'b' / 'My Repo!'
    for folder in (named, unnamed, elsewhere):
        folder.mkdir(parents=True)
    assert str(project_folder(named)).startswith('my-repo-')
    assert str(project_folder(unnamed)).startswith('project-')
    assert project_folder(named) != project_folder(elsewhere)
