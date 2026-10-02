"""Named folder selection and compatibility for CLAI's Coder wrapper."""

import io
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue, ValidationError
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.subagents import SubAgents
from pydantic_clai2.builtin_plugins import coder as coder_plugin
from pydantic_clai2.builtin_plugins.coder import CoderPlugin, CoderSettings, CoderSource
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, SessionStart, load_plugin
from pydantic_clai2.plugins.loader import PluginLoader


def test_named_folders_and_explicit_paths_preserve_precedence() -> None:
    settings = CoderSettings(agent_folders=['agents', 'global', './custom', '~/other', '/shared', 'agents'])
    assert settings.folders(home=Path('/home/tester')) == [
        '.agents/agents',
        '.claude/agents',
        '.codex/agents',
        '.agents/global',
        '.claude/global',
        '.codex/global',
        './custom',
        '/home/tester/other',
        '/shared',
        '/home/tester/.agents/agents',
        '/home/tester/.claude/agents',
        '/home/tester/.codex/agents',
        '/home/tester/.agents/global',
        '/home/tester/.claude/global',
        '/home/tester/.codex/global',
    ]
    assert CoderSettings().folders(home=Path('/home/tester')) == []


@pytest.mark.parametrize('value', ['', ' ', ' global', 'global ', 'bad\x00path'])
def test_reject_invalid_folder_selections(value: str) -> None:
    with pytest.raises(ValidationError, match='Agent folders'):
        CoderSettings(agent_folders=[value])


def coder_host(*, terminal: bool = False, **settings: JsonValue) -> PluginHost[object]:
    console = Console(file=io.StringIO(), force_terminal=terminal)
    return PluginHost(name='coder', console=console, settings=dict(settings))


def test_settings_source_validates_saves_and_resets() -> None:
    host = coder_host(sub_agents=False, instructions='Keep this guidance.')
    source = CoderSource(host)
    unrestricted, sub_agents, folders = source.rows()
    assert source.current(unrestricted) == 'false'
    assert source.apply(unrestricted, 'true') == 'Saved Unrestricted filesystem.'
    assert host.settings(CoderSettings).unrestricted_filesystem is True
    assert source.current(sub_agents) == 'false'
    assert source.current(folders) == '[]'
    assert source.problem(folders, '["agents", " bad"]') is not None
    assert source.problem(folders, 'not json') is not None
    assert source.problem(folders, '["agents", "global"]') is None
    assert source.apply(folders, '["agents", "global"]') == 'Saved Agent folders.'
    assert source.apply(sub_agents, 'true') == 'Saved Sub-agents.'
    saved = host.settings(CoderSettings)
    assert saved.agent_folders == ['agents', 'global']
    assert saved.sub_agents is True
    assert saved.instructions == 'Keep this guidance.'
    assert source.reset(folders) == 'Reset Agent folders.'
    assert host.settings(CoderSettings).agent_folders == []
    assert source.reset(folders) == 'Reset Agent folders.'


def test_menu_edits_save_only_chosen_settings() -> None:
    saved: list[dict[str, JsonValue]] = []
    host = PluginHost[object](
        name='coder',
        console=Console(file=io.StringIO()),
        settings={'unrestricted_filesystem': True, 'sub_agents': True},
        save_settings=saved.append,
    )
    source = CoderSource(host)
    source.apply(source.rows()[1], 'false')
    assert saved[-1] == {'unrestricted_filesystem': True, 'sub_agents': False}
    source.reset(source.rows()[1])
    assert saved[-1] == {'unrestricted_filesystem': True}


async def test_configure_outside_a_terminal_explains_how(monkeypatch: pytest.MonkeyPatch) -> None:
    assert 'from a terminal' in await CoderPlugin[object].from_host(coder_host()).configure()


@pytest.mark.parametrize('messages', [['Saved Agent folders.'], []])
async def test_configure_runs_the_field_menu(monkeypatch: pytest.MonkeyPatch, messages: list[str]) -> None:
    plugin = CoderPlugin[object].from_host(coder_host(terminal=True))

    async def run_worker(work: Callable[[], list[str]]) -> list[str]:
        return messages

    monkeypatch.setattr(coder_plugin, 'run_worker', run_worker)
    assert await plugin.configure() == ('\n'.join(messages) or 'No Coder settings changed.')


@pytest.mark.parametrize('unrestricted', [False, True])
async def test_unrestricted_filesystem_setting_controls_paths_outside_the_project(
    tmp_path: Path, unrestricted: bool
) -> None:
    project, outside = tmp_path / 'project', tmp_path / 'outside.txt'
    project.mkdir()
    outside.write_text('reachable-content')
    settings: dict[str, JsonValue] = {
        'unrestricted_filesystem': unrestricted,
        'repo_context': False,
        'sub_agents': False,
    }
    host = PluginHost[None](name='coder', console=Console(file=io.StringIO()), settings=settings)
    loaded = load_plugin(CoderPlugin, host)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('read_file', {'path': str(outside)})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), deps_type=type(None))
    result = await agent.run('read it', capabilities=[*loaded.capabilities, LocalWorkspace(project)])
    returns = [
        str(part.content)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, (ToolReturnPart, RetryPromptPart))
    ]
    assert any('reachable-content' in content for content in returns) is unrestricted


@pytest.mark.parametrize('factory', ['pydantic_ai_harness:Coder', 'pydantic_ai_harness.coder:Coder'])
@pytest.mark.parametrize('folders', [[], ['agents', 'global']])
async def test_old_coder_declarations_load_without_rewriting_preferences(
    tmp_path: Path, factory: str, folders: list[str]
) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    settings: dict[str, JsonValue] = {'repo_context': False, 'sub_agents': True, 'instructions': 'Keep this guidance.'}
    if folders:
        settings['agent_folders'] = cast(JsonValue, folders)
    declaration = PluginSettings(id='coder', factory=factory, settings=settings)
    store.save_plugin(declaration)
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
    )
    await loader.load_all()
    (coder,) = loader.capabilities()
    assert isinstance(coder, Coder)
    (delegation,) = [
        capability for capability in cast(Coder[None], coder).capabilities if isinstance(capability, SubAgents)
    ]
    assert delegation.agent_folders == (CoderSettings(agent_folders=folders).folders(home=Path.home()) or None)
    assert store.plugins() == [declaration]
    await loader.unload('coder')
