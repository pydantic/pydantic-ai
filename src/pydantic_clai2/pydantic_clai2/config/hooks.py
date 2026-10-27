"""The `hooks` table: commands CLAI runs before and after tool calls, and as a turn starts and ends.

Validated here, apart from the plugin that runs them, so `.clai/settings.json` is checked at startup
without importing it.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

HOOKS_FACTORY = 'pydantic_clai2.builtin_plugins.hooks'
"""The built-in plugin that runs a `hooks` table, from your settings or from the project file."""
PROJECT_HOOKS = 'project_hooks'
"""The id of the declaration a project file's `hooks` table becomes, which `/plugins enable` approves."""

HookEvent = Literal['before_tool', 'after_tool', 'turn_start', 'turn_end']


class HookCommand(BaseModel):
    """One command to run for an event."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    command: (
        Annotated[str, Field(min_length=1)] | Annotated[list[Annotated[str, Field(min_length=1)]], Field(min_length=1)]
    )
    """An argv list, run without a shell, whose `{name}` placeholders are filled from the tool's arguments; or a
    string, run by the shell as written."""
    tools: list[str] | None = Field(default=None, min_length=1)
    """Run only for these tool names; every tool when left out. Tool events only."""
    timeout: float = Field(default=60, gt=0, le=3600, allow_inf_nan=False)
    """Seconds before the command is stopped and reported as failed."""


class HookTable(BaseModel):
    """The commands for each event, run in order. A project file's `hooks` key takes this shape."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    before_tool: list[HookCommand] = Field(default_factory=list[HookCommand])
    after_tool: list[HookCommand] = Field(default_factory=list[HookCommand])
    turn_start: list[HookCommand] = Field(default_factory=list[HookCommand])
    turn_end: list[HookCommand] = Field(default_factory=list[HookCommand])

    @model_validator(mode='after')
    def _tools_only_on_tool_events(self) -> 'HookTable':
        for event, commands in (('turn_start', self.turn_start), ('turn_end', self.turn_end)):
            if any(command.tools is not None for command in commands):
                raise ValueError(f'{event} hooks run once per turn, not per tool: remove `tools`')
        return self

    def commands(self, event: HookEvent) -> list[HookCommand]:
        """The commands for `event`, in the order they run."""
        tables = {
            'before_tool': self.before_tool,
            'after_tool': self.after_tool,
            'turn_start': self.turn_start,
            'turn_end': self.turn_end,
        }
        return tables[event]


class HooksSettings(HookTable):
    """The `hooks` plugin's settings: a table, plus the project it came from when a project file declared it."""

    project: str | None = None
    """The folder holding `.clai` for a project's table. It runs only in sessions under that folder, since
    the approval is saved with your settings, which every repository shares."""
