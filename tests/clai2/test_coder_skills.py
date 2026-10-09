"""Agent Skills in the built-in `coder` plugin: folders, the model's catalog, `/skill-name` commands, and notices."""

import io
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

import pytest
from pydantic import JsonValue, ValidationError
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.skills import Skills
from pydantic_clai2 import chat
from pydantic_clai2._app import create_shell
from pydantic_clai2.builtin_plugins.coder import DEFAULT_SKILL_FOLDERS, CoderPlugin, CoderSettings, CoderSource
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, SessionStart, collect, load_plugin
from pydantic_clai2.plugins.loader import PluginLoader

PromptT = TypeVar('PromptT')

CODER = 'pydantic_clai2.builtin_plugins.coder'


def write_skill(library: Path, name: str, *, description: str = 'Help with the task.', body: str = 'Do it.') -> Path:
    directory = library / name
    directory.mkdir(parents=True)
    (directory / 'SKILL.md').write_text(f'---\ndescription: {description}\n---\n\n{body}\n', encoding='utf-8')
    return directory


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The launch directory, where project skills live; `HOME` is already `tmp_path / 'home'`."""
    directory = tmp_path / 'project'
    directory.mkdir()
    (tmp_path / 'home').mkdir()
    monkeypatch.chdir(directory)
    return directory


@pytest.fixture
def home(tmp_path: Path, project: Path) -> Path:
    return tmp_path / 'home'


async def load_coder(
    settings: Mapping[str, JsonValue] | None = None, *, submitted: list[str] | None = None
) -> tuple[LoadedPlugin[None], io.StringIO]:
    """Load `coder` as the loader does: build it, `prepare`, collect, and start its session."""
    output = io.StringIO()
    host = PluginHost[None](
        name='coder',
        console=Console(file=output, width=1000),
        settings={'repo_context': False, 'sub_agents': False, **(settings or {})},
        submit_prompt=None if submitted is None else submitted.append,
    )
    plugin = CoderPlugin[None].from_host(host)
    await plugin.prepare()
    loaded = collect(plugin)
    await loaded.dispatch(SessionStart(agent=Agent(TestModel()), settings=Settings(model='test')))
    return loaded, output


@dataclass
class FirstRequest:
    """What the model saw on its first request."""

    instructions: str = ''
    tools: list[str] = field(default_factory=list[str])


async def run_once(loaded: LoadedPlugin[None], workspace: Path) -> FirstRequest:
    seen = FirstRequest()

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        seen.instructions = request.instructions or ''
        seen.tools = [tool.name for tool in info.function_tools]
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), deps_type=type(None))
    await agent.run('go', capabilities=[*loaded.capabilities, LocalWorkspace[None](workspace)])
    return seen


def has_skills(loaded: LoadedPlugin[None]) -> bool:
    return any(isinstance(capability, Skills) for capability in loaded.capabilities)


class TestSkillFolders:
    def test_default_folders_are_project_then_home_agents_then_claude(self) -> None:
        assert CoderSettings().skill_folders == list(DEFAULT_SKILL_FOLDERS)
        assert CoderSettings().skill_libraries(home=Path('/home/tester')) == [
            '.agents/skills',
            '.claude/skills',
            '/home/tester/.agents/skills',
            '/home/tester/.claude/skills',
        ]

    def test_folders_expand_home_and_keep_order_once(self) -> None:
        settings = CoderSettings(skill_folders=['~/team', 'skills', '/abs', 'skills'])
        assert settings.skill_libraries(home=Path('/home/tester')) == ['/home/tester/team', 'skills', '/abs']

    @pytest.mark.parametrize('value', ['', ' skills', 'bad\x00path'])
    def test_invalid_folders_are_rejected(self, value: str) -> None:
        with pytest.raises(ValidationError, match='Skill folders must be nonempty'):
            CoderSettings(skill_folders=[value])

    def test_settings_menu_edits_skill_folders(self) -> None:
        host = PluginHost[object](name='coder', console=Console(file=io.StringIO()), settings={})
        source = CoderSource(host)
        row = next(row for row in source.rows() if row.key == 'skill_folders')
        assert source.current(row) == '[".agents/skills", ".claude/skills", "~/.agents/skills", "~/.claude/skills"]'
        assert source.apply(row, '[]') == 'Saved Skill folders.'
        assert host.settings(CoderSettings).skill_folders == []
        assert source.reset(row) == 'Reset Skill folders.'
        assert host.settings(CoderSettings).skill_folders == list(DEFAULT_SKILL_FOLDERS)

    async def test_saved_skill_folders_are_tagged_for_older_builds(self, tmp_path: Path, project: Path) -> None:
        store = SettingsStore(tmp_path / 'settings.db')
        loader: PluginLoader[None] = PluginLoader(
            store=store,
            console=Console(file=io.StringIO()),
            commands=Commands(),
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
            builtin=(PluginSettings(id='coder', factory=CODER, settings={'repo_context': False}),),
        )
        await loader.load_all()
        loaded = loader.entries()[0].loaded
        assert loaded is not None
        loaded.host.save_settings(CoderSettings(repo_context=False, skill_folders=['skills']))

        assert store.plugin_requirements('coder') == {'skill_folders': ['coder-skill-folders']}
        await loader.close('exit')


class TestSkillCatalog:
    async def test_project_and_personal_skills_reach_the_model_once_each(self, project: Path, home: Path) -> None:
        write_skill(project / '.claude' / 'skills', 'review', description='Project review.')
        (project / '.agents').mkdir()
        # This repository's layout: `.agents/skills` links into `.claude/skills`.
        (project / '.agents' / 'skills').symlink_to(project / '.claude' / 'skills')
        write_skill(home / '.claude' / 'skills', 'release', description='Personal release.')
        loaded, output = await load_coder()

        seen = await run_once(loaded, project)

        assert '- review: Project review.\n- release: Personal release.' in seen.instructions
        assert 'load_capability' in seen.tools
        assert [command.name for command in loaded.commands] == ['review', 'release']
        assert output.getvalue() == ''

    @pytest.mark.parametrize('settings', [{}, {'skill_folders': []}], ids=['no-skills-found', 'skills-off'])
    async def test_without_skills_coder_adds_no_catalog_tool_or_commands(
        self, project: Path, settings: dict[str, JsonValue]
    ) -> None:
        loaded, output = await load_coder(settings)

        seen = await run_once(loaded, project)

        # A `Skills` that found nothing stays bound, so a skill added mid-session reaches the next turn's catalog.
        assert [type(capability) for capability in loaded.capabilities] == ([Coder] if settings else [Coder, Skills])
        assert list(loaded.commands) == []
        assert 'load_capability' not in seen.tools
        assert 'deferred' not in seen.instructions
        assert output.getvalue() == ''

    async def test_clashes_and_malformed_skills_are_reported_once_at_load(self, project: Path, home: Path) -> None:
        write_skill(project / '.agents' / 'skills', 'review', description='Project review.')
        write_skill(home / '.agents' / 'skills', 'review', description='Personal review.')
        broken = home / '.claude' / 'skills' / 'broken'
        broken.mkdir(parents=True)
        (broken / 'SKILL.md').write_text('no frontmatter', encoding='utf-8')
        loaded, output = await load_coder()

        assert has_skills(loaded)
        assert [command.name for command in loaded.commands] == ['review']
        resolved_home = home.resolve()
        assert output.getvalue() == (
            f'Skipping {resolved_home}/.agents/skills/review/SKILL.md: skill name '
            f"'review' is already taken by {project.resolve()}/.agents/skills/review/SKILL.md.\n"
            f'Skipping {resolved_home}/.claude/skills/broken/SKILL.md: {resolved_home}/.claude/skills/broken/SKILL.md '
            'must start with YAML frontmatter delimited by `---`.\n'
        )

    async def test_control_characters_from_skill_files_are_made_inert(self, project: Path) -> None:
        write_skill(project / '.agents' / 'skills', 'review', description='"Review \\e[31m red."')
        broken = project / '.agents' / 'skills' / 'bad\x1bc'
        broken.mkdir()
        (broken / 'SKILL.md').write_text('no frontmatter', encoding='utf-8')
        loaded, output = await load_coder()

        (command,) = loaded.commands
        assert command.description == 'Skill: Review \\x1b[31m red.'
        assert '\x1b' not in output.getvalue()
        assert 'bad\\x1bc' in output.getvalue()

    async def test_an_unreadable_folder_turns_skills_off_but_keeps_coder(self, project: Path) -> None:
        library = project / '.agents' / 'skills'
        write_skill(library, 'review')
        library.chmod(0)
        try:
            loaded, output = await load_coder()
        finally:
            library.chmod(0o755)

        assert [type(capability) for capability in loaded.capabilities] == [Coder]
        assert output.getvalue().startswith('Skills are off: [Errno 13] Permission denied')

    async def test_skills_are_off_where_clai_has_no_local_workspace(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_skill(project / '.agents' / 'skills', 'review')
        monkeypatch.setattr('pydantic_clai2.runtime._session.sys.platform', 'win32')

        loaded, output = await load_coder()

        assert [type(capability) for capability in loaded.capabilities] == [Coder]
        assert (list(loaded.commands), output.getvalue()) == ([], '')

    async def test_a_misconfigured_folder_turns_skills_off_with_a_notice(self, project: Path) -> None:
        write_skill(project / 'skills', 'review')
        loaded, output = await load_coder({'skill_folders': ['skills/review']})

        assert not has_skills(loaded)
        assert list(loaded.commands) == []
        assert output.getvalue().startswith('Skills are off: Skill library path points to a skill package')


def _inputs(monkeypatch: pytest.MonkeyPatch, values: list[str]) -> None:
    class Prompt(Generic[PromptT]):
        def __init__(self, **kwargs: object) -> None:
            pass

        async def prompt_async(self, label: str, **kwargs: object) -> str:
            return values.pop(0)

    monkeypatch.setattr('pydantic_clai2._app.PromptSession', Prompt)


class TestSkillCommands:
    async def test_skill_command_submits_the_rendered_skill_as_a_turn(
        self, tmp_path: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        skill = write_skill(project / '.agents' / 'skills', 'review', body='Review $ARGUMENTS carefully.')
        prompts: list[str] = []

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            prompts.extend(
                part.content
                for part in messages[-1].parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            )
            yield 'reviewed'

        output = io.StringIO()
        _inputs(monkeypatch, ['/review src/app.py "quoted"', '/exit'])
        await chat(
            Agent(FunctionModel(stream_function=stream), deps_type=type(None)),
            deps=None,
            console=Console(file=output, width=200),
            settings=Settings(model=None),
            store=SettingsStore(tmp_path / 'config.db'),
            builtin_plugins=(
                PluginSettings(id='coder', factory=CODER, settings={'repo_context': False, 'sub_agents': False}),
            ),
        )

        assert prompts == [
            f'# Skill: review\n\nSkill directory: `{skill.resolve()}`. Relative paths in this skill resolve '
            'against it.\n\nReview src/app.py "quoted" carefully.'
        ]
        assert 'reviewed' in output.getvalue()

    def test_the_shell_takes_a_prompt_only_from_a_running_command(self, tmp_path: Path) -> None:
        shell = create_shell(
            Agent(TestModel(), deps_type=type(None)),
            deps=None,
            plugins=(),
            usage_limits=None,
            console=Console(file=io.StringIO()),
            settings=None,
            store=SettingsStore(tmp_path / 'config.db'),
            builtin_plugins=(),
            project=ProjectSettings(),
        )
        with pytest.raises(RuntimeError, match='works only in a /command handler run between turns'):
            shell.submit_prompt('hello')

    def test_load_plugin_refuses_a_plugin_it_cannot_prepare(self) -> None:
        host = PluginHost[None](name='coder', console=Console(file=io.StringIO()), settings={})
        with pytest.raises(TypeError, match=r'CoderPlugin overrides `prepare`.*await plugin.prepare\(\)'):
            load_plugin(CoderPlugin, host)

    def test_submit_prompt_needs_a_shell(self) -> None:
        host = PluginHost[None](name='coder', console=Console(file=io.StringIO()), settings={})
        with pytest.raises(RuntimeError, match="Plugin 'coder' has no shell to run a prompt in"):
            host.submit_prompt('hello')

    async def test_reload_picks_up_skills_added_mid_session(self, tmp_path: Path, project: Path) -> None:
        store = SettingsStore(tmp_path / 'settings.db')
        commands = Commands()
        loader: PluginLoader[None] = PluginLoader(
            store=store,
            console=Console(file=io.StringIO()),
            commands=commands,
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
            builtin=(PluginSettings(id='coder', factory=CODER, settings={'repo_context': False}),),
        )
        await loader.load_all()
        assert 'review' not in commands
        write_skill(project / '.agents' / 'skills', 'review')

        # What `/plugins reload coder` does, without re-importing the module for the rest of the test process.
        await loader.unload('coder')
        await loader.load('coder')

        assert 'review' in commands
        await loader.close('exit')

    async def test_a_skill_named_like_a_command_gives_way_and_is_reported(self, tmp_path: Path, project: Path) -> None:
        write_skill(project / '.agents' / 'skills', 'model')
        write_skill(project / '.agents' / 'skills', 'review')
        store = SettingsStore(tmp_path / 'settings.db')
        commands = Commands()
        builtin = Command(name='model', description='Select a model', handler=lambda _: 'picked')
        commands.register(builtin)
        output = io.StringIO()
        loader: PluginLoader[None] = PluginLoader(
            store=store,
            console=Console(file=output, width=200),
            commands=commands,
            session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
            builtin=(PluginSettings(id='coder', factory=CODER, settings={'repo_context': False}),),
        )
        await loader.load_all()

        assert commands.execute('/model') == 'picked'
        assert 'review' in commands
        assert output.getvalue() == '/model (Skill: Help with the task.) is hidden by another /model command.\n'
        await loader.unload('coder')
        assert commands.execute('/model') == 'picked'
        assert 'review' not in commands


class TestOverridableCommands:
    def test_a_later_command_takes_the_name_of_an_overridable_one(self) -> None:
        commands = Commands()
        skill = Command(name='code-review', description='Skill', handler=lambda _: 'skill', overridable=True)
        assert commands.register_many([skill]) == []
        real = Command(name='code-review', description='Real', handler=lambda _: 'real')

        assert commands.register_many([real]) == [skill]
        assert commands.execute('/code-review') == 'real'

        commands.unregister([skill])
        assert commands.execute('/code-review') == 'real'
        commands.unregister([real])
        assert 'code-review' not in commands

    def test_two_overridable_commands_keep_the_first(self) -> None:
        commands = Commands()
        first = Command(name='review', description='First', handler=lambda _: 'first', overridable=True)
        second = Command(name='review', description='Second', handler=lambda _: 'second', overridable=True)

        assert commands.register_many([first, second]) == [second]
        assert commands.execute('/review') == 'first'

    @pytest.mark.parametrize('name', ['-leading', 'has space', 'dot.name', ''])
    def test_command_names_are_words_and_hyphens(self, name: str) -> None:
        with pytest.raises(ValueError, match='Invalid or duplicate command'):
            Commands().register(Command(name=name, description='Bad', handler=lambda _: ''))
