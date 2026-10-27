"""Agent Skills in the built-in `coder` plugin: folders, the model's catalog, `/skill-name` commands, notices, and turning skills off."""

import io
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

import pytest
from pydantic import BaseModel, JsonValue, ValidationError
from rich.console import Console

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import (
    AgentCapability,
    CombinedCapability,
    LocalWorkspace,
    WrapperCapability,
)
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.skills import Skills
from pydantic_clai2 import Session, chat
from pydantic_clai2.app import create_stock_agent
from pydantic_clai2.builtin_plugins.coder import DEFAULT_SKILL_FOLDERS, CoderPlugin, CoderSettings, CoderSource
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import (
    LoadedPlugin,
    NoSettings,
    Plugin,
    PluginHost,
    SessionStart,
    collect,
    load_plugin,
)
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.runtime.session_runtime import create_shell
from tests.clai2.dialog_script import Back, Pick, ScriptedDialogs

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


def bound_types(loaded: LoadedPlugin[None]) -> list[type[object]]:
    """The capability classes `loaded` binds, with the `Skills` `coder` wraps in a group shown as `Skills`."""
    types: list[type[object]] = []
    for capability in loaded.capabilities:
        match capability:
            case WrapperCapability(wrapped=CombinedCapability(capabilities=[Skills()])):
                types.append(Skills)
            case _:
                types.append(type(capability))
    return types


def commands_of(loaded: LoadedPlugin[None]) -> Commands:
    commands = Commands()
    commands.register_many(loaded.commands)
    return commands


def has_skills(loaded: LoadedPlugin[None]) -> bool:
    return Skills in bound_types(loaded)


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
        loaded.host.save_settings(
            CoderSettings(repo_context=False, task_control=False, skill_folders=['skills'], disabled_skills=['review'])
        )

        assert store.plugin_requirements('coder') == {
            'task_control': ['coder-task-control'],
            'skill_folders': ['coder-skill-folders'],
            'disabled_skills': ['coder-disabled-skills'],
        }
        await loader.close('exit')


class TestDisabledSkills:
    @pytest.mark.parametrize(
        ('disabled', 'offered'),
        [
            ([], ['deploy', 'review']),
            (['review'], ['deploy']),
            (['review', 'deploy'], []),
            # The setting is the same in every project, so a name this one lacks is not an error.
            (['elsewhere'], ['deploy', 'review']),
            # Compared after NFKC normalization, as harness compares names: a fullwidth `v` is `v`.
            (['re\uff56iew'], ['deploy']),
        ],
        ids=['none', 'one', 'all', 'unknown', 'normalized'],
    )
    async def test_a_disabled_skill_is_hidden_from_the_model_and_has_no_command(
        self, project: Path, disabled: list[str], offered: list[str]
    ) -> None:
        for name in ('deploy', 'review'):
            write_skill(project / '.agents' / 'skills', name, description=f'{name.title()} help.')
        loaded, output = await load_coder({'disabled_skills': list[JsonValue](disabled)})

        seen = await run_once(loaded, project)

        assert sorted(command.name for command in loaded.commands) == offered
        assert [line for line in seen.instructions.splitlines() if line.startswith('- ')] == [
            f'- {name}: {name.title()} help.' for name in offered
        ]
        assert ('load_capability' in seen.tools) == bool(offered)
        assert output.getvalue() == ''

    @pytest.mark.parametrize('value', ['', ' review', 'bad\x1bname'])
    def test_invalid_names_are_rejected(self, value: str) -> None:
        with pytest.raises(ValidationError, match='Skill names must be nonempty'):
            CoderSettings(disabled_skills=[value])

    def test_names_are_saved_normalized(self) -> None:
        assert CoderSettings(disabled_skills=['re\uff56iew']).disabled_skills == ['review']

    async def test_the_skills_row_toggles_each_skill_and_saves_it(self, project: Path) -> None:
        for name in ('deploy', 'review'):
            write_skill(project / '.agents' / 'skills', name, description=f'{name.title()} help.')
        loaded, _ = await load_coder({'disabled_skills': ['elsewhere']})
        plugin = loaded.plugin
        assert isinstance(plugin, CoderPlugin)
        source = CoderSource(plugin.host)
        row = next(row for row in source.rows() if row.key == 'disabled_skills')
        assert (row.label, row.display(source.current(row))) == ('Skills', '1 off')
        script = ScriptedDialogs(
            [
                Pick(value='review'),
                Pick(value='review'),
                Pick(value='deploy'),
                Pick(value='elsewhere'),
                Pick(value=':close'),
            ]
        )

        messages = await plugin.settings_menu().actions['disabled_skills'](script)

        assert messages == ['Turned review off.', 'Turned review on.', 'Turned deploy off.', 'Turned elsewhere on.']
        assert plugin.host.settings(CoderSettings).disabled_skills == ['deploy']
        assert [option.label for option in script.asked[0].options] == [
            '[x] deploy',
            '[x] review',
            '[ ] elsewhere (not found here)',
            'Save & close',
        ]
        assert [option.label for option in script.asked[1].options][:2] == ['[x] deploy', '[ ] review']
        deploy = project.resolve() / '.agents' / 'skills' / 'deploy' / 'SKILL.md'
        assert script.asked[0].options[0].description == f'Deploy help.\n\n{deploy}'
        assert ' '.join(script.asked[1].description.split()).startswith('Turned review off. Each pick turns a skill')
        assert source.current(row) == '["deploy"]'

    async def test_the_skill_list_says_when_no_skills_were_found_and_reports_save_errors(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loaded, _ = await load_coder({'disabled_skills': ['review']})
        plugin = loaded.plugin
        assert isinstance(plugin, CoderPlugin)

        def refuse(settings: BaseModel, /) -> None:
            raise ValueError('disk full')

        # On the host: an earlier `loader.reload('coder')` in this process may have replaced the module's classes.
        monkeypatch.setattr(plugin.host, 'save_settings', refuse)
        script = ScriptedDialogs([Pick(value='review'), Back()])

        assert await plugin.settings_menu().actions['disabled_skills'](script) == []
        assert ' '.join(script.asked[0].description.split()).startswith(
            'No skills were found in the skill folders when Coder loaded. Each pick'
        )
        assert script.asked[1].description.startswith('Could not save: disk full\n\n')
        assert plugin.host.settings(CoderSettings).disabled_skills == ['review']


class TestSkillFoldersAboveTheLaunchDirectory:
    async def test_launching_in_a_subdirectory_finds_the_repositorys_skills_nearest_first(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (project / '.git').mkdir()
        write_skill(project / '.agents' / 'skills', 'review', description='Root review.')
        write_skill(project / '.agents' / 'skills', 'release', description='Root release.')
        package = project / 'packages' / 'app'
        write_skill(package / '.agents' / 'skills', 'review', description='Package review.')
        src = package / 'src'
        src.mkdir()
        monkeypatch.chdir(src)

        with pytest.warns(UserWarning, match="skill name 'review' is already taken"):
            loaded, output = await load_coder({'skill_folders': ['.agents/skills']})
            seen = await run_once(loaded, src)

        assert '- review: Package review.\n- release: Root release.' in seen.instructions
        assert [command.name for command in loaded.commands] == ['review', 'release']
        assert 'Root review' not in output.getvalue()

    async def test_the_sessions_directory_is_searched_not_the_processs(
        self, tmp_path: Path, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skills come from `host.workspace` and the repository around it, wherever the process is."""
        (project / '.git').mkdir()
        write_skill(project / '.agents' / 'skills', 'release')
        (project / 'src').mkdir()
        write_skill(tmp_path / 'elsewhere' / '.agents' / 'skills', 'stray')
        monkeypatch.chdir(tmp_path / 'elsewhere')
        host = PluginHost[None](
            name='coder',
            console=Console(file=io.StringIO()),
            settings={'repo_context': False, 'sub_agents': False, 'skill_folders': ['.agents/skills']},
            workspace=lambda: project / 'src',
        )
        plugin = CoderPlugin[None].from_host(host)
        await plugin.prepare()

        assert [command.name for command in collect(plugin).commands] == ['release']

    async def test_the_walk_stops_at_the_git_root(self, tmp_path: Path, project: Path) -> None:
        write_skill(tmp_path / '.agents' / 'skills', 'outside')
        (project / '.git').mkdir()

        loaded, _ = await load_coder({'skill_folders': ['.agents/skills']})

        assert list(loaded.commands) == []

    async def test_outside_a_repository_only_the_launch_directory_is_searched(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_skill(project / '.agents' / 'skills', 'review')
        (project / 'src').mkdir()
        monkeypatch.chdir(project / 'src')

        loaded, _ = await load_coder({'skill_folders': ['.agents/skills']})

        assert list(loaded.commands) == []

    def test_relative_folders_expand_across_parents_in_place(self) -> None:
        settings = CoderSettings(skill_folders=['.agents/skills', '~/skills', '/abs', '.claude/skills'])

        assert settings.skill_libraries(home=Path('/home/tester'), parents=[Path('/repo/pkg'), Path('/repo')]) == [
            '.agents/skills',
            '/repo/pkg/.agents/skills',
            '/repo/.agents/skills',
            '/home/tester/skills',
            '/abs',
            '.claude/skills',
            '/repo/pkg/.claude/skills',
            '/repo/.claude/skills',
        ]


class TestSkillInvocationFrontmatter:
    async def test_a_model_disabled_skill_is_a_command_only(self, project: Path) -> None:
        library = project / '.agents' / 'skills'
        (library / 'deploy').mkdir(parents=True)
        (library / 'deploy' / 'SKILL.md').write_text(
            '---\ndescription: Deploy.\ndisable-model-invocation: true\n---\n\nShip it.\n', encoding='utf-8'
        )
        loaded, output = await load_coder()

        seen = await run_once(loaded, project)

        assert [command.name for command in loaded.commands] == ['deploy']
        assert 'load_capability' not in seen.tools
        assert 'deploy' not in seen.instructions
        assert output.getvalue() == ''

    async def test_a_skill_that_is_not_user_invocable_is_catalog_only(self, project: Path) -> None:
        library = project / '.agents' / 'skills'
        (library / 'background').mkdir(parents=True)
        (library / 'background' / 'SKILL.md').write_text(
            '---\ndescription: Background.\nuser-invocable: false\n---\n\nKnow this.\n', encoding='utf-8'
        )
        loaded, _ = await load_coder()

        seen = await run_once(loaded, project)

        assert list(loaded.commands) == []
        assert '- background: Background.' in seen.instructions

    async def test_frontmatter_for_other_clients_loads_without_a_notice(self, project: Path) -> None:
        library = project / '.agents' / 'skills'
        (library / 'review').mkdir(parents=True)
        (library / 'review' / 'SKILL.md').write_text(
            '---\ndescription: Review.\nallowed-tools: Read\ncontext: fork\n---\n', encoding='utf-8'
        )

        loaded, output = await load_coder()

        assert [command.name for command in loaded.commands] == ['review']
        assert output.getvalue() == ''


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
        assert bound_types(loaded) == ([Coder] if settings else [Coder, Skills])
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

    # Mode bits do not bind root, and Windows cannot make a directory unreadable with them.
    @pytest.mark.skipif(
        os.name == 'nt' or getattr(os, 'geteuid', lambda: 1)() == 0,
        reason='POSIX mode bits must apply to this process.',
    )
    async def test_an_unreadable_folder_turns_skills_off_but_keeps_coder(self, project: Path) -> None:
        library = project / '.agents' / 'skills'
        write_skill(library, 'review')
        library.chmod(0)
        try:
            loaded, output = await load_coder()
        finally:
            library.chmod(0o755)

        assert bound_types(loaded) == [Coder]
        assert output.getvalue().startswith('Skills are off: [Errno 13] Permission denied')

    async def test_skills_are_off_where_clai_has_no_local_workspace(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_skill(project / '.agents' / 'skills', 'review')
        monkeypatch.setattr('pydantic_clai2.runtime.session.sys.platform', 'win32')

        loaded, output = await load_coder()

        assert bound_types(loaded) == [Coder]
        assert (list(loaded.commands), output.getvalue()) == ([], '')

    @pytest.mark.parametrize('supplies_workspace', [False, True], ids=['no-workspace', 'sandbox'])
    async def test_skills_load_beside_a_plugin_capability_function(
        self, tmp_path: Path, project: Path, supplies_workspace: bool
    ) -> None:
        # Such a plugin may supply a sandbox, so CLAI attaches the project directory only once the run's
        # capability functions resolve: after `Skills` reads its catalog.
        write_skill(project / '.agents' / 'skills', 'review', description='Project review.')
        sandbox = tmp_path / 'sandbox'
        sandbox.mkdir()

        class PerRun(Plugin[NoSettings, None]):
            def get_capabilities(self) -> Sequence[AgentCapability[None]]:
                return (self.for_run,)

            def for_run(self, ctx: RunContext[None]) -> LocalWorkspace[None] | None:
                return LocalWorkspace[None](sandbox) if supplies_workspace else None

        per_run = load_plugin(
            PerRun, PluginHost[None](name='per_run', console=Console(file=io.StringIO()), settings={})
        )
        coder, _ = await load_coder()
        session = Session(
            create_stock_agent(TestModel(call_tools=[], custom_output_text='done')),
            deps=None,
            workspace=project,
            plugins=[*coder.capabilities, *per_run.capabilities],
        )
        result = await session.prompt('go')

        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        assert '- review: Project review.' in (request.instructions or '')
        # A sandbox the capability function supplies still wins over the project directory.
        expected = (sandbox if supplies_workspace else project).resolve()
        assert result.response.workspace_ref == WorkspaceRef(provider='local', id=str(expected))

    @pytest.mark.parametrize('first', ['coder', 'own'])
    async def test_another_skills_on_the_agent_combines_with_coders(self, project: Path, first: str) -> None:
        write_skill(project / '.agents' / 'skills', 'review', description='Project review.')
        write_skill(project / 'team', 'deploy', description='Team deploy.')
        coder, _ = await load_coder()
        # It names `.agents/skills` too; combined with `coder`'s, that `review` is listed once.
        own = Skills[None](['team', '.agents/skills'])
        plugins = [*coder.capabilities, own] if first == 'coder' else [own, *coder.capabilities]
        session = Session(
            create_stock_agent(TestModel(call_tools=[], custom_output_text='done')),
            deps=None,
            workspace=project,
            plugins=plugins,
        )
        result = await session.prompt('go')

        request = result.all_messages()[0]
        assert isinstance(request, ModelRequest)
        instructions = request.instructions or ''
        assert instructions.count('- review: Project review.') == 1
        assert '- deploy: Team deploy.' in instructions

    async def test_a_folder_that_is_one_skill_is_skipped_with_a_notice(self, project: Path, home: Path) -> None:
        # A stray `.agents/skills/SKILL.md` in a repository must not turn off the user's personal skills.
        write_skill(project / '.agents', 'skills')
        (project / '.claude').mkdir()
        (project / '.claude' / 'skills').write_text('not a folder', encoding='utf-8')
        write_skill(home / '.agents' / 'skills', 'release')
        loaded, output = await load_coder()

        assert has_skills(loaded)
        assert [command.name for command in loaded.commands] == ['release']
        resolved = project.resolve()
        assert output.getvalue() == (
            f'Skipping {resolved}/.agents/skills: it is a skill package, not a library.\n'
            f'Skipping {resolved}/.claude/skills: it is not a directory.\n'
        )

    async def test_a_skill_package_created_mid_session_does_not_fail_turns(self, project: Path, home: Path) -> None:
        write_skill(home / '.agents' / 'skills', 'release', description='Personal release.')
        loaded, _ = await load_coder()
        # What a model asked to "write a skill" might create.
        write_skill(project / '.agents', 'skills')

        with pytest.warns(UserWarning, match=r'\.agents/skills: it is a skill package, not a library\.'):
            seen = await run_once(loaded, project)

        assert '- release: Personal release.' in seen.instructions


def _inputs(monkeypatch: pytest.MonkeyPatch, values: list[str]) -> None:
    class Prompt(Generic[PromptT]):
        def __init__(self, **kwargs: object) -> None:
            pass

        async def prompt_async(self, label: str, **kwargs: object) -> str:
            return values.pop(0)

    monkeypatch.setattr('pydantic_clai2.ui.prompt.terminal.PromptSession', Prompt)


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

    async def test_a_skill_command_sends_the_skill_as_it_is_now(self, project: Path) -> None:
        skill = write_skill(project / '.agents' / 'skills', 'review', body='Old steps.')
        submitted: list[str] = []
        loaded, _ = await load_coder(submitted=submitted)
        (skill / 'SKILL.md').write_text('---\ndescription: Help.\n---\n\nNew steps.\n', encoding='utf-8')

        assert await commands_of(loaded).execute_async('/review app.py') == ''

        assert submitted == [
            f'# Skill: review\n\nSkill directory: `{skill.resolve()}`. Relative paths in this skill resolve '
            'against it.\n\nNew steps.\n\nARGUMENTS: app.py'
        ]

    async def test_a_skill_command_says_so_when_the_skill_is_gone(self, project: Path) -> None:
        skill = write_skill(project / '.agents' / 'skills', 'review')
        write_skill(project / '.agents' / 'skills', 'deploy')
        background = write_skill(project / '.agents' / 'skills', 'background')
        submitted: list[str] = []
        loaded, _ = await load_coder(submitted=submitted)
        commands = commands_of(loaded)
        (skill / 'SKILL.md').unlink()
        (project / '.agents' / 'skills' / 'deploy' / 'SKILL.md').write_text('no frontmatter', encoding='utf-8')
        (background / 'SKILL.md').write_text('---\ndescription: B.\nuser-invocable: false\n---\n', encoding='utf-8')

        with pytest.raises(ValueError) as gone:
            await commands.execute_async('/review')
        with pytest.raises(ValueError) as broken:
            await commands.execute_async('/deploy')
        with pytest.raises(ValueError, match=r"Skill 'background' is now `user-invocable: false`\. /reload"):
            await commands.execute_async('/background')

        assert str(gone.value) == (
            f"Skill 'review' is gone: no skill folder has it any more; it was in {skill.resolve()}/SKILL.md. "
            '/reload updates the commands.'
        )
        assert str(broken.value).startswith(
            f"Skill 'deploy' is gone: Skipping {project.resolve()}/.agents/skills/deploy/SKILL.md: "
        )
        assert submitted == []

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
