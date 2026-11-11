"""Named folder selection and compatibility for CLAI's Coder wrapper."""

import io
from pathlib import Path
from typing import cast

import pytest
from pydantic import JsonValue, ValidationError
from rich.console import Console
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.subagents import SubAgents
from pydantic_clai2.builtin_plugins.coder import CoderPlugin, CoderSettings, CoderSource
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, SessionStart, collect
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.ui.menus import dialogs
from pydantic_clai2.ui.menus.field_menu import FieldMenu, run_flow
from tests.clai2.menu_script import Script, pick, typed


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
    unrestricted, sub_agents, folders, _skill_folders, skills, _delegate_models, _review_agent = source.rows()
    assert (source.current(skills), skills.display(source.current(skills))) == ('[]', 'All on')
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


def test_enter_on_skill_folders_turned_off_keeps_them_off() -> None:
    """The editor opens holding `[]`, so Enter keeps skills off."""
    saved: list[dict[str, JsonValue]] = []
    host = PluginHost[object](
        name='coder', console=Console(file=io.StringIO()), settings={'skill_folders': []}, save_settings=saved.append
    )
    menu = FieldMenu(CoderSource(host))
    skill_folders = menu.row_for('skill_folders')
    assert skill_folders is not None
    editor = menu.build_editor(skill_folders)
    assert (editor.text, editor._placeholder) == (  # pyright: ignore[reportPrivateUsage]
        '[]',
        'empty resets to [".agents/skills", ".claude/skills", "~/.agents/skills", "~/.claude/skills"]',
    )
    script = Script(
        lists=[pick('skill_folders'), pick('skill_folders'), MenuResult(cancelled=True)],
        choices=[],
        texts=[typed('[]'), typed('')],
    )
    assert run_flow(menu, script.runners) == ['Reset Skill folders.']
    assert saved == [{}]
    assert menu.initial(skill_folders) == '', 'a setting at its default opens empty'


async def test_configure_outside_a_terminal_explains_how(monkeypatch: pytest.MonkeyPatch) -> None:
    assert 'from a terminal' in await CoderPlugin[object].from_host(coder_host()).configure()


@pytest.mark.parametrize(
    ('script', 'shown', 'saves'),
    [
        (
            Script(
                [pick('agent_folders'), pick('sub_agents'), MenuResult(cancelled=True)],
                [pick('name'), pick('close'), pick('false')],
                [typed('agents')],
            ),
            'Agent folder saved.\nSaved Sub-agents.',
            [{'agent_folders': ['agents']}, {'agent_folders': ['agents'], 'sub_agents': False}],
        ),
        (Script([MenuResult(cancelled=True)], [], []), 'Coder settings: nothing changed.', []),
    ],
)
async def test_configure_shows_the_declared_menu_in_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: Script, shown: str, saves: list[dict[str, JsonValue]]
) -> None:
    """The **Agent folders** row opens the folder list through the terminal's dialogs; other rows edit a value."""
    monkeypatch.chdir(tmp_path)
    saved: list[dict[str, JsonValue]] = []
    host = PluginHost[object](
        name='coder', console=Console(file=io.StringIO(), force_terminal=True), settings={}, save_settings=saved.append
    )
    monkeypatch.setattr(dialogs, 'RUNNERS', script.runners)
    assert await CoderPlugin[object].from_host(host).configure() == shown
    assert saved == saves


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
    plugin = CoderPlugin[None].from_host(host)
    await plugin.prepare()
    loaded = collect(plugin)

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
        if isinstance(part, ToolReturnPart)
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
    coder, *_skills = loader.capabilities()
    assert isinstance(coder, Coder)
    (delegation,) = [
        capability for capability in cast(Coder[None], coder).capabilities if isinstance(capability, SubAgents)
    ]
    assert delegation.agent_folders == (CoderSettings(agent_folders=folders).folders(home=Path.home()) or None)
    assert store.plugins() == [declaration]
    await loader.unload('coder')
