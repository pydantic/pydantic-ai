"""Coding tools, Agent Skills, and opt-in named disk-agent folders for the terminal shell."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Generic, Self

import anyio
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_serializer,
)

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, CombinedCapability, WrapperCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, ModelResolutionContext
from pydantic_ai.tools import RunContext
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.skills import SkillDefinition, Skills
from pydantic_ai_harness.subagents import ModelOption, SubAgent, SubAgents, TaskControl
from pydantic_clai2.builtin_plugins.coder_folders import edit_agent_folders, is_folder_name
from pydantic_clai2.builtin_plugins.coder_skills import edit_skills
from pydantic_clai2.commands import Command
from pydantic_clai2.config.project_settings import repository_directories
from pydantic_clai2.models.profiles import parse_model
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, SessionStart
from pydantic_clai2.plugins.menus import FieldRow, SettingsMenu
from pydantic_clai2.runtime.session import ModelDefaults, SessionModels, resolve_model_name, supports_local_workspace
from pydantic_clai2.runtime.tasks import REVIEW_AGENT_NAME, review_workspace
from pydantic_clai2.ui.menus.field_menu import first_error
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.tool_output import terminal_text

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)

DEFAULT_SKILL_FOLDERS = ('.agents/skills', '.claude/skills', '~/.agents/skills', '~/.claude/skills')
"""Project skills before personal ones, `.agents` before `.claude`: the first skill with a name wins."""

SKILL_FOLDERS_FEATURE = 'coder-skill-folders'
"""Builds without it would reject a saved `skill_folders`, so they drop the setting instead."""

DISABLED_SKILLS_FEATURE = 'coder-disabled-skills'
"""Builds without it would reject a saved `disabled_skills`, so they drop the setting and offer every skill."""

TASK_CONTROL_FEATURE = 'coder-task-control'
"""Builds without it would reject a saved `task_control`, so they drop the setting."""

DELEGATE_MODELS_FEATURE = 'coder-delegate-models'
"""Builds without the delegate model menu ignore its saved configuration."""

REVIEW_AGENT_FEATURE = 'coder-review-agent'
"""Builds without the Review specialist ignore its saved opt-in."""


class DelegateModel(BaseModel):
    """One named option in the delegate model menu."""

    model_config = ConfigDict(extra='forbid', strict=True, frozen=True)

    model: str
    description: str | None = None
    settings: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator('settings')
    @classmethod
    def valid_settings(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        # The model settings menu stays unloaded until delegate overrides are configured.
        from pydantic_clai2.models.model_settings import ModelSettingsForm

        ModelSettingsForm.model_validate(value)
        return value

    @model_serializer(mode='plain')
    def _only_chosen(self) -> dict[str, JsonValue]:
        values: dict[str, JsonValue] = {'model': self.model}
        if 'description' in self.model_fields_set:
            values['description'] = self.description
        if 'settings' in self.model_fields_set:
            values['settings'] = self.settings
        return values

    @field_validator('model')
    @classmethod
    def valid_model(cls, value: str) -> str:
        ref = parse_model(value)
        if not ref.provider or not ref.name or value != value.strip() or not value.isprintable():
            raise ValueError('Use a nonempty PROVIDER[@ACCOUNT]:MODEL or chain:NAME.')
        return value


def _expand_home(value: str, *, home: Path) -> str:
    return (home / value[2:]).as_posix() if value.startswith('~/') else value


class CoderSettings(BaseModel):
    """Validated coding preferences, including explicitly selected agent folders."""

    model_config = ConfigDict(extra='forbid', strict=True)

    instructions: str | None = Field(default=None, description='Replace the default coding instructions.')
    unrestricted_filesystem: bool = Field(
        default=False, description='Let file tools reach any path on this machine, not only the project directory.'
    )
    workspace: str | None = Field(default=None, description='Deprecated workspace option, retained for saved settings.')
    repo_context: bool = Field(default=True, description='Include repository instructions with coding tools.')
    sub_agents: bool = Field(default=True, description='Enable delegation to sub-agents.')
    task_control: bool = Field(
        default=True,
        description='With sub-agents, let the model run tasks in the background and steer, answer, or cancel them.',
    )
    delegate_models: dict[str, DelegateModel] = Field(
        default_factory=dict,
        description='With Sub-agents enabled, offer named models or saved chains to delegate_task.',
    )
    review_agent: bool = Field(
        default=False,
        description='With Sub-agents enabled, add a read-only Review specialist that can use another directory.',
    )
    agent_folders: list[str] = Field(
        default_factory=list,
        description=(
            'Agent folder names or explicit paths. Names search .agents, .claude and .codex in the project and home. '
            'Use ["agents"] for standard folders, ["agents", "global"] to include global folders, or [] to disable.'
        ),
    )
    skill_folders: list[str] = Field(
        default_factory=lambda: list(DEFAULT_SKILL_FOLDERS),
        description=(
            'Agent Skills folders, in precedence order: relative paths are in the project, ~/ is your home. '
            'Missing folders are skipped. Use [] to turn skills off.'
        ),
    )
    disabled_skills: list[str] = Field(
        default_factory=list,
        description='Skill names hidden from the model and from the /commands, in every project.',
    )

    @field_validator('agent_folders', 'skill_folders')
    @classmethod
    def valid_folders(cls, values: list[str], info: ValidationInfo) -> list[str]:
        for value in values:
            if not value.strip() or value != value.strip() or '\x00' in value:
                kind = 'Agent' if info.field_name == 'agent_folders' else 'Skill'
                raise ValueError(
                    f'{kind} folders must be nonempty names or paths without surrounding whitespace or NUL.'
                )
        return values

    @field_validator('disabled_skills')
    @classmethod
    def valid_skill_names(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value or value != value.strip() or not value.isprintable():
                raise ValueError('Skill names must be nonempty, without surrounding whitespace or control characters.')
        # Harness compares skill names after NFKC normalization, so a name typed another way still matches.
        return [unicodedata.normalize('NFKC', value) for value in values]

    @field_validator('delegate_models')
    @classmethod
    def valid_delegate_keys(cls, values: dict[str, DelegateModel]) -> dict[str, DelegateModel]:
        for key in values:
            if not key or key != key.strip() or not key.isprintable():
                raise ValueError('Delegate model keys must be nonempty, without surrounding whitespace or controls.')
        return values

    @model_serializer(mode='wrap')
    def _only_chosen(self, handler: SerializerFunctionWrapHandler) -> dict[str, JsonValue]:
        """Save only the settings someone chose, so menu edits never pin the other defaults."""
        dumped: dict[str, JsonValue] = handler(self)
        return {key: value for key, value in dumped.items() if key in self.model_fields_set}

    def folders(self, *, home: Path) -> list[str]:
        """Explicit selections and project names precede automatic personal counterparts."""
        project: list[str] = []
        personal: list[str] = []
        for value in self.agent_folders:
            if is_folder_name(value):
                for prefix in ('.agents', '.claude', '.codex'):
                    folder = f'{prefix}/{value}'
                    project.append(folder)
                    personal.append((home / folder).as_posix())
            else:
                project.append(_expand_home(value, home=home))
        return list(dict.fromkeys([*project, *personal]))

    def skill_libraries(self, *, home: Path, parents: Sequence[Path] = ()) -> list[str]:
        """`skill_folders` with `~/` expanded, each once, in precedence order.

        Each relative folder is followed by the same folder in each of `parents`: the session directory's
        parents up to the git root, nearest first, so launching in a subdirectory still finds the
        repository's skills.
        """
        libraries: list[str] = []
        for value in self.skill_folders:
            library = _expand_home(value, home=home)
            libraries.append(library)
            if not Path(library).is_absolute():
                libraries.extend((parent / library).as_posix() for parent in parents)
        return list(dict.fromkeys(libraries))


class CoderSource(Generic[DepsT]):
    """File access and delegation settings, saved through the same validated model."""

    title = 'Coder settings'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        self.host = host

    def rows(self) -> tuple[FieldRow, ...]:
        settings = self.host.settings(CoderSettings)
        folders, disabled = settings.agent_folders, settings.disabled_skills
        return (
            FieldRow(
                key='unrestricted_filesystem',
                label='Unrestricted filesystem',
                description=CoderSettings.model_fields['unrestricted_filesystem'].description or '',
                default='false',
                choices=('true', 'false'),
                allow_custom=False,
            ),
            FieldRow(
                key='sub_agents',
                label='Sub-agents',
                description='Enable delegation.',
                default='true',
                choices=('true', 'false'),
                allow_custom=False,
            ),
            FieldRow(
                key='agent_folders',
                label='Agent folders',
                description='Manage sub-agent definition folders.\nChoosing it opens the folder list.\nAn empty list turns disk agents off.',
                default='[]',
                choice_labels={
                    '[]': 'None (off)',
                    **({json.dumps(folders): f'{len(folders)} selected'} if folders else {}),
                },
            ),
            FieldRow(
                key='skill_folders',
                label='Skill folders',
                description=CoderSettings.model_fields['skill_folders'].description or '',
                default=json.dumps(DEFAULT_SKILL_FOLDERS),
            ),
            FieldRow(
                key='disabled_skills',
                label='Skills',
                description='Turn each skill on or off.\nChoosing it opens the skill list.',
                default='[]',
                choice_labels={
                    '[]': 'All on',
                    **({json.dumps(disabled): f'{len(disabled)} off'} if disabled else {}),
                },
            ),
            FieldRow(
                key='delegate_models',
                label='Delegate models',
                description=CoderSettings.model_fields['delegate_models'].description or '',
                default='{}',
            ),
            FieldRow(
                key='review_agent',
                label='Review specialist',
                description=CoderSettings.model_fields['review_agent'].description or '',
                default='false',
                choices=('true', 'false'),
                allow_custom=False,
            ),
        )

    def current(self, row: FieldRow) -> str:
        settings = self.host.settings(CoderSettings)
        values: dict[str, JsonValue] = {
            'unrestricted_filesystem': settings.unrestricted_filesystem,
            'sub_agents': settings.sub_agents,
            'review_agent': settings.review_agent,
            'agent_folders': list[JsonValue](settings.agent_folders),
            'skill_folders': list[JsonValue](settings.skill_folders),
            'disabled_skills': list[JsonValue](settings.disabled_skills),
            'delegate_models': {
                key: option.model_dump(mode='json', exclude_unset=True)
                for key, option in settings.delegate_models.items()
            },
        }
        return json.dumps(values[row.key])

    def _with(self, key: str, value: JsonValue) -> CoderSettings:
        data: dict[str, JsonValue] = self.host.settings(CoderSettings).model_dump(mode='json')
        data[key] = value
        return CoderSettings.model_validate(data)

    def _updated(self, row: FieldRow, raw: str) -> CoderSettings:
        return self._with(row.key, _JSON.validate_json(raw))

    def agent_folders(self) -> list[str]:
        return self.host.settings(CoderSettings).agent_folders

    def sub_agents_enabled(self) -> bool:
        return self.host.settings(CoderSettings).sub_agents

    def expand_folder(self, value: str) -> list[str]:
        """The locations one entry searches, in the order the Coder plugin passes them on."""
        return CoderSettings(agent_folders=[value]).folders(home=Path.home())

    def save_agent_folders(self, folders: list[str]) -> None:
        """Persist the folder list, keeping every other Coder preference."""
        self.host.save_settings(self._with('agent_folders', list[JsonValue](folders)))

    def disabled_skills(self) -> list[str]:
        return self.host.settings(CoderSettings).disabled_skills

    def save_disabled_skills(self, names: list[str]) -> None:
        """Persist the skills that are off, keeping every other Coder preference."""
        self.host.save_settings(self._with('disabled_skills', list[JsonValue](names)))

    def problem(self, row: FieldRow, text: str) -> str | None:
        try:
            self._updated(row, text)
        except ValueError as exc:
            return first_error(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        self.host.save_settings(self._updated(row, raw))
        return f'Saved {row.label}.'

    def reset(self, row: FieldRow) -> str:
        data: dict[str, JsonValue] = self.host.settings(CoderSettings).model_dump(mode='json')
        data.pop(row.key, None)
        self.host.save_settings(CoderSettings.model_validate(data))
        return f'Reset {row.label}.'


@dataclass
class _ProjectSkills(WrapperCapability[DepsT]):
    """Harness `Skills` that read the project directory while the run's workspace is not chosen yet.

    When a plugin adds a capability function, CLAI attaches the project directory only once the run's
    capability functions resolve, so a sandbox one supplies wins (see `local_workspace`). `Skills` reads
    its catalog in `for_run`, before then, and would find no workspace. It reads the project directory
    instead, where the run works unless a capability function supplies a workspace of its own.

    The `Skills` sits in a group, so core still sees it and another `Skills` on the agent combines with it.
    Core keeps the combined one where the last of them was, so one combined outside this wrapper reads
    the run's workspace as before.
    """

    project: LocalWorkspaceBackend = field(kw_only=True)
    """The session's directory (`PluginHost.workspace`), where the commands' skills are read."""

    async def for_run(self, ctx: RunContext[DepsT]) -> AbstractCapability[DepsT]:
        if not ctx.workspace.attached:
            ctx = replace(ctx, workspace=Workspace(self.project))
        return await super().for_run(ctx)


class CoderPlugin(Plugin[CoderSettings, DepsT]):
    """Harness `Coder` and `Skills`, a `/command` per skill, and a field editor for the preferences."""

    @classmethod
    def from_host(cls, host: PluginHost[DepsT]) -> Self:
        # Tags the setting, so a build that cannot read it uses the default instead of failing to load `coder`.
        requires = {
            'task_control': [TASK_CONTROL_FEATURE],
            'skill_folders': [SKILL_FOLDERS_FEATURE],
            'disabled_skills': [DISABLED_SKILLS_FEATURE],
            'delegate_models': [DELEGATE_MODELS_FEATURE],
            'review_agent': [REVIEW_AGENT_FEATURE],
        }
        return cls(host, host.settings(CoderSettings, requires=requires))

    def __init__(self, host: PluginHost[DepsT], settings: CoderSettings) -> None:
        super().__init__(host, settings)
        self._skills: Skills[DepsT] | None = None
        self._skill_workspace: LocalWorkspaceBackend | None = None
        self._skill_definitions: tuple[SkillDefinition, ...] = ()
        self._skill_notices: tuple[str, ...] = ()

    async def prepare(self) -> None:
        """Read the skill folders once, for the `/skill-name` commands and the notices shown at load.

        The model's catalog is read again at the start of every run, so it lists a skill added since;
        its command arrives when the plugin loads again.
        """
        directory = self.host.workspace
        # The session's directory itself stays relative, resolved by the workspace as before.
        _, *parents = await anyio.to_thread.run_sync(repository_directories, directory)
        libraries = self.settings.skill_libraries(home=Path(await anyio.Path.home()), parents=parents)
        # Without a local workspace (Windows), the folders cannot be read here or by the run.
        if not libraries or not supports_local_workspace():
            return
        # The model's skills leave out the ones turned off. The setting is the same in every project, so a
        # name this project lacks is skipped. The commands and the settings menu read every skill.
        skills = Skills[DepsT](
            libraries,
            exclude=self.settings.disabled_skills,
            missing_directories='skip',
            duplicate_names='keep_first',
            unknown_names='skip',
        )
        listed = Skills[DepsT](libraries, missing_directories='skip', duplicate_names='keep_first')
        workspace = LocalWorkspaceBackend(directory)
        try:
            catalog = await listed.load(workspace)
        except (UserError, ValueError, OSError) as exc:
            # A misconfigured or unreadable folder would otherwise fail every turn, or the whole plugin, so
            # skills are left out and the file and shell tools still load.
            self._skill_notices = (f'Skills are off: {exc}',)
            return
        self._skills = skills
        self._skill_workspace = workspace
        self._skill_definitions = catalog.skills
        # Only skipped skills are reported. Fields such as `allowed-tools` belong to other clients that share
        # the skill folders, so naming them at every launch would be a warning nobody can act on.
        self._skill_notices = tuple(skipped.message for skipped in catalog.skipped)

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        settings = self.settings
        custom_delegation = bool(settings.delegate_models) or settings.review_agent
        coder = Coder[DepsT](
            instructions=settings.instructions,
            unrestricted_filesystem=settings.unrestricted_filesystem,
            workspace=settings.workspace,
            repo_context=settings.repo_context,
            sub_agents=settings.sub_agents and not custom_delegation,
            agent_folders=settings.folders(home=Path.home()) or None,
            task_control=TaskControl()
            if settings.sub_agents and settings.task_control and not custom_delegation
            else None,
        )
        capabilities: list[AgentCapability[DepsT]] = [coder]
        if settings.sub_agents and custom_delegation:
            from pydantic_clai2.models.model_settings import ModelSettingsForm

            agents: list[SubAgent[object]] = []
            if settings.review_agent:
                reviewer = Agent(
                    deps_type=object,
                    name=REVIEW_AGENT_NAME,
                    description='Review code and proposed changes with read-only file tools.',
                    instructions=(
                        'Read the repository instructions in your workspace. '
                        'Review the code or changes described in the task. '
                        'Return actionable findings with file paths.'
                    ),
                    capabilities=[FileSystem[object](read_only=True), RepoContext[object](expose_inventory_tool=False)],
                )
                agents.append(SubAgent(reviewer, read_only=True, resumable=False, can_ask_questions=False))
            models: dict[str, ModelOption] = {
                key: ModelOption(
                    option.model,
                    description=option.description,
                    settings=ModelSettingsForm.model_validate(option.settings).to_model_settings(),
                )
                for key, option in settings.delegate_models.items()
            }
            capabilities.append(
                SubAgents[DepsT](
                    agents=agents,
                    models=models,
                    include_self=True,
                    agent_folders=settings.folders(home=Path.home()) or None,
                    task_control=TaskControl() if settings.task_control else None,
                    workspace_resolver=review_workspace if settings.review_agent else None,
                    shared_capabilities=[
                        SessionModels[DepsT](self._resolve_delegate_model),
                        ModelDefaults[DepsT](self.host.model_settings),
                    ],
                )
            )
        if self._skills is not None and self._skill_workspace is not None:
            capabilities.append(
                _ProjectSkills[DepsT](CombinedCapability([self._skills]), project=self._skill_workspace)
            )
        return capabilities

    async def _resolve_delegate_model(self, ctx: ModelResolutionContext[DepsT], name: str) -> Model | None:
        return await resolve_model_name(self.host.resolve_model, name)

    def get_commands(self) -> Sequence[Command]:
        return tuple(
            Command(
                name=skill.name,
                # Skill files come from the repository, so control characters must not reach the terminal.
                description=f'Skill: {terminal_text(skill.description.splitlines()[0])}',
                handler=partial(self._invoke, skill),
                raw=True,
                overridable=True,
            )
            for skill in self._skill_definitions
            if skill.user_invocable and skill.name not in self.settings.disabled_skills
        )

    async def _invoke(self, loaded: SkillDefinition, args: list[str]) -> str:
        """Send the skill as the folders hold it now, as the model would load it, not as it was at load."""
        assert self._skills is not None and self._skill_workspace is not None, 'skill commands need loaded skills'
        catalog = await self._skills.load(self._skill_workspace)
        skill = next((skill for skill in catalog.skills if skill.name == loaded.name), None)
        if skill is None:
            problem = next((skipped.message for skipped in catalog.skipped if skipped.path == loaded.path), None)
            detail = problem or f'no skill folder has it any more; it was in {loaded.path}.'
            # Skill files come from the repository, so control characters must not reach the terminal.
            raise ValueError(terminal_text(f'Skill {loaded.name!r} is gone: {detail} /reload updates the commands.'))
        if not skill.user_invocable:
            raise ValueError(f'Skill {loaded.name!r} is now `user-invocable: false`. /reload updates the commands.')
        self.host.submit_prompt(skill.render(args[0] if args else ''))
        return ''

    async def on_session_start(self, event: SessionStart) -> None:
        for notice in self._skill_notices:
            self.host.console.print(terminal_text(notice), style=theme.color(theme.WARNING), markup=False)

    def settings_menu(self) -> SettingsMenu:
        source = CoderSource(self.host)
        return SettingsMenu(
            source=source,
            actions={
                'agent_folders': partial(edit_agent_folders, source),
                'disabled_skills': partial(edit_skills, source, self._skill_definitions),
            },
        )
