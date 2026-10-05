"""Coding tools, Agent Skills, and opt-in named disk-agent folders for the terminal shell."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Generic

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
from typing_extensions import Self

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.skills import SkillDefinition, Skills
from pydantic_clai2.commands import Command
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, SessionStart
from pydantic_clai2.ui.menus.field_menu import FieldMenu, FieldRow, first_error, run_flow
from pydantic_clai2.ui.menus.menu_worker import run_worker
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.tool_output import terminal_text

_FOLDER_NAME = re.compile(r'[A-Za-z0-9_-]+')

DEFAULT_SKILL_FOLDERS = ('.agents/skills', '.claude/skills', '~/.agents/skills', '~/.claude/skills')
"""Project skills before personal ones, `.agents` before `.claude`: the first skill with a name wins."""

SKILL_FOLDERS_FEATURE = 'coder-skill-folders'
"""Builds without it would reject a saved `skill_folders`, so they drop the setting instead."""


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
            if _FOLDER_NAME.fullmatch(value):
                for prefix in ('.agents', '.claude', '.codex'):
                    folder = f'{prefix}/{value}'
                    project.append(folder)
                    personal.append((home / folder).as_posix())
            else:
                project.append(_expand_home(value, home=home))
        return list(dict.fromkeys([*project, *personal]))

    def skill_libraries(self, *, home: Path) -> list[str]:
        """`skill_folders` with `~/` expanded, each once, in precedence order."""
        return list(dict.fromkeys(_expand_home(value, home=home) for value in self.skill_folders))


class CoderSource(Generic[DepsT]):
    """File access and delegation settings, saved through the same validated model."""

    title = 'Coder settings'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        self.host = host

    def rows(self) -> tuple[FieldRow, ...]:
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
                description=CoderSettings.model_fields['agent_folders'].description or '',
                default='[]',
            ),
            FieldRow(
                key='skill_folders',
                label='Skill folders',
                description=CoderSettings.model_fields['skill_folders'].description or '',
                default=json.dumps(DEFAULT_SKILL_FOLDERS),
            ),
        )

    def current(self, row: FieldRow) -> str:
        settings = self.host.settings(CoderSettings)
        values: dict[str, JsonValue] = {
            'unrestricted_filesystem': settings.unrestricted_filesystem,
            'sub_agents': settings.sub_agents,
            'agent_folders': list[JsonValue](settings.agent_folders),
            'skill_folders': list[JsonValue](settings.skill_folders),
        }
        return json.dumps(values[row.key])

    def _updated(self, row: FieldRow, raw: str) -> CoderSettings:
        data: dict[str, JsonValue] = self.host.settings(CoderSettings).model_dump(mode='json')
        data[row.key] = TypeAdapter(JsonValue).validate_json(raw)
        return CoderSettings.model_validate(data)

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


class CoderPlugin(Plugin[CoderSettings, DepsT]):
    """Harness `Coder` and `Skills`, a `/command` per skill, and a field editor for the preferences."""

    @classmethod
    def from_host(cls, host: PluginHost[DepsT]) -> Self:
        # Tags the setting, so a build that cannot read it uses the default instead of failing to load `coder`.
        return cls(host, host.settings(CoderSettings, requires={'skill_folders': [SKILL_FOLDERS_FEATURE]}))

    def __init__(self, host: PluginHost[DepsT], settings: CoderSettings) -> None:
        super().__init__(host, settings)
        self._skills: Skills[DepsT] | None = None
        self._skill_definitions: tuple[SkillDefinition, ...] = ()
        self._skill_notices: tuple[str, ...] = ()

    async def prepare(self) -> None:
        """Read the skill folders once, for the `/skill-name` commands and the notices shown at load.

        The model's catalog is read again at the start of every run, so it lists a skill added since;
        its command arrives when the plugin loads again.
        """
        libraries = self.settings.skill_libraries(home=Path(await anyio.Path.home()))
        if not libraries:
            return
        skills = Skills[DepsT](libraries, missing_directories='skip', duplicate_names='keep_first')
        try:
            catalog = await skills.load(LocalWorkspaceBackend(Path(await anyio.Path.cwd())))
        except (UserError, ValueError) as exc:
            # A misconfigured folder would otherwise fail every turn; leave skills out instead.
            self._skill_notices = (f'Skills are off: {exc}',)
            return
        self._skills = skills
        self._skill_definitions = catalog.skills
        self._skill_notices = catalog.skipped

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        settings = self.settings
        coder = Coder[DepsT](
            instructions=settings.instructions,
            unrestricted_filesystem=settings.unrestricted_filesystem,
            workspace=settings.workspace,
            repo_context=settings.repo_context,
            sub_agents=settings.sub_agents,
            agent_folders=settings.folders(home=Path.home()) or None,
        )
        return (coder,) if self._skills is None else (coder, self._skills)

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
        )

    def _invoke(self, skill: SkillDefinition, args: list[str]) -> str:
        self.host.submit_prompt(skill.render(args[0] if args else ''))
        return ''

    async def on_session_start(self, event: SessionStart) -> None:
        for notice in self._skill_notices:
            self.host.console.print(terminal_text(notice), style=theme.color(theme.WARNING), markup=False)

    async def configure(self) -> str:
        if not self.host.console.is_terminal:
            return 'Configure Coder from a terminal: /plugins configure coder'
        messages = await run_worker(lambda: run_flow(FieldMenu(CoderSource(self.host))))
        return '\n'.join(messages) or 'No Coder settings changed.'
