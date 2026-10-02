"""Coding tools and opt-in named disk-agent folders for the terminal shell."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Generic

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    field_validator,
    model_serializer,
)

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.coder import Coder
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost
from pydantic_clai2.ui.menus.field_menu import FieldMenu, FieldRow, first_error, run_flow
from pydantic_clai2.ui.menus.menu_worker import run_worker

_FOLDER_NAME = re.compile(r'[A-Za-z0-9_-]+')


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

    @field_validator('agent_folders')
    @classmethod
    def valid_folders(cls, values: list[str]) -> list[str]:
        for value in values:
            if not value.strip() or value != value.strip() or '\x00' in value:
                raise ValueError('Agent folders must be nonempty names or paths without surrounding whitespace or NUL.')
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
                project.append((home / value[2:]).as_posix() if value.startswith('~/') else value)
        return list(dict.fromkeys([*project, *personal]))


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
        )

    def current(self, row: FieldRow) -> str:
        settings = self.host.settings(CoderSettings)
        values: dict[str, JsonValue] = {
            'unrestricted_filesystem': settings.unrestricted_filesystem,
            'sub_agents': settings.sub_agents,
            'agent_folders': list[JsonValue](settings.agent_folders),
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
    """Harness `Coder`, plus a field editor for the delegation preferences."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        settings = self.settings
        return (
            Coder[DepsT](
                instructions=settings.instructions,
                unrestricted_filesystem=settings.unrestricted_filesystem,
                workspace=settings.workspace,
                repo_context=settings.repo_context,
                sub_agents=settings.sub_agents,
                agent_folders=settings.folders(home=Path.home()) or None,
            ),
        )

    async def configure(self) -> str:
        if not self.host.console.is_terminal:
            return 'Configure Coder from a terminal: /plugins configure coder'
        messages = await run_worker(lambda: run_flow(FieldMenu(CoderSource(self.host))))
        return '\n'.join(messages) or 'No Coder settings changed.'
