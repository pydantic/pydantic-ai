"""Validated settings, independent of persistence and terminal code."""

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from pydantic_clai2.config.theme_names import names

UpdateChannel = Literal['stable', 'main']
UPDATE_CHANNELS: tuple[UpdateChannel, ...] = get_args(UpdateChannel)
STORED_MAIN_CHANNEL = 'bleeding'
"""`main`'s former name, still what is saved for it: older builds accept only this name."""
ToolCallDisplay = Literal['detailed', 'grouped']
TOOL_CALL_DISPLAYS: tuple[ToolCallDisplay, ...] = get_args(ToolCallDisplay)


class Settings(BaseModel):
    """An immutable snapshot; storage contains only explicit overrides."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    model: str | None = Field(
        default='openai-codex:gpt-6-astra',
        min_length=1,
        description='Provider-qualified model for every turn. Overrides the model baked into the agent.',
    )
    request_limit: int = Field(
        default=10000, gt=0, description='Most model requests one prompt may make before the turn stops.'
    )
    tool_retries: int = Field(
        default=3, ge=0, description='Default retries per tool call. Explicit tool retry limits take precedence.'
    )
    pool_accounts: bool = Field(
        default=True,
        description=(
            'A model without @PROFILE runs on every signed-in account of its provider, in /accounts order, '
            'as @* does. PROVIDER@default:MODEL runs the default account alone.'
        ),
    )
    speculative_code_mode: bool = Field(
        default=False,
        description=(
            'Fold tools into a sandboxed run_code that executes and speculates while the model writes. '
            'Ctrl+X Ctrl+S toggles it.'
        ),
    )
    instructions: str = Field(
        default='',
        description=(
            "Your own instructions, sent after CLAI's built-in, AGENTS.md, and plugin instructions. "
            '/system_prompt edits them.'
        ),
    )
    session_namer: bool = Field(default=True, description='Name saved sessions in the background using a model.')
    session_namer_model: str | None = Field(
        default=None, description='Naming model override; null uses the current model.'
    )
    theme: str = Field(
        default='default', description='Keep CLAI colours, or select a bundled Termflow palette with /theme.'
    )
    spinner: str = Field(
        default='working',
        min_length=1,
        description='Working animation; /spinner previews the choices. An unknown name shows working.',
    )
    thinking: bool = Field(default=True, description="Show the model's thinking as it streams.")
    splash: bool = Field(default=True, description='Animate the startup splash. Takes effect next start.')
    tool_calls: ToolCallDisplay = Field(
        default='detailed',
        description=(
            'How tool calls print: detailed gives each call its own line with arguments; '
            'grouped counts consecutive calls by tool on one live line. Grouped ignores display.tool_output '
            'and display.tool_arg_chars; file diffs still print.'
        ),
    )
    tool_output: bool = Field(
        default=False, description='Show shell and grep output below tool summaries. File diffs are always shown.'
    )
    shell_lines: int = Field(
        default=20, ge=0, le=1000, description='Shell preview lines when display.tool_output is enabled.'
    )
    grep_lines: int = Field(
        default=20, ge=0, le=1000, description='Grep preview lines when display.tool_output is enabled.'
    )
    tool_arg_chars: int = Field(
        default=40, ge=0, le=1000, description='Characters shown per tool argument value; 0 hides arguments.'
    )
    smooth_seconds: float = Field(
        default=0.5,
        ge=0.1,
        le=5,
        allow_inf_nan=False,
        description='Catch-up window for smoothed response streaming, 0.1 to 5 seconds.',
    )
    update_channel: UpdateChannel = Field(
        default='stable',
        description='Where /update looks: stable PyPI releases, or main for the newest CLAI commit on main.',
    )

    @field_validator('update_channel', mode='before')
    @classmethod
    def read_former_channel_name(cls, value: object) -> object:
        """Read `bleeding`, saved by older builds and by this one, as `main`."""
        return 'main' if value == STORED_MAIN_CHANNEL else value

    @field_validator('theme')
    @classmethod
    def validate_theme(cls, value: str) -> str:
        """Keep picker, project files, and saved preferences on the same registry."""
        if value not in names():
            raise ValueError(f'Unknown theme: {value}. Choose from: {", ".join(names())}')
        return value


SETTING_FIELDS = {
    'model': 'model',
    'run.request_limit': 'request_limit',
    'display.thinking': 'thinking',
    'display.theme': 'theme',
    'display.spinner': 'spinner',
    'display.splash': 'splash',
    'display.tool_calls': 'tool_calls',
    'display.tool_output': 'tool_output',
    'display.shell_lines': 'shell_lines',
    'display.grep_lines': 'grep_lines',
    'display.tool_arg_chars': 'tool_arg_chars',
    'display.smooth_seconds': 'smooth_seconds',
    'run.tool_retries': 'tool_retries',
    'run.speculative_code_mode': 'speculative_code_mode',
    'run.instructions': 'instructions',
    'accounts.pool': 'pool_accounts',
    'sessions.naming': 'session_namer',
    'sessions.naming_model': 'session_namer_model',
    'updates.channel': 'update_channel',
}

STRING_SETTINGS = frozenset(
    {'model', 'run.instructions', 'display.theme', 'display.spinner', 'display.tool_calls', 'updates.channel'}
)
"""Keys whose typed value is taken as text rather than parsed as JSON."""


def resolve_settings(overrides: dict[str, JsonValue]) -> Settings:
    """Reject unknown setting names and validate stored or supplied values."""
    unknown = overrides.keys() - SETTING_FIELDS.keys()
    if unknown:
        raise ValueError(f'Unknown settings: {", ".join(sorted(unknown))}')
    return Settings.model_validate({SETTING_FIELDS[key]: value for key, value in overrides.items()})


_CODER_FACTORY = 'pydantic_ai_harness.coder:Coder'


class PluginSettings(BaseModel):
    """Declaration for a trusted plugin: a module defining one `Plugin` subclass, `module:PluginClass`, or `module:Capability`."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    id: str = Field(min_length=1)
    factory: str = Field(pattern=r'^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?::[A-Za-z_]\w*)?$')
    path: str | None = None
    enabled: bool = True
    settings: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode='after')
    def _coder_without_delegation(self) -> 'PluginSettings':
        """Preserve the delegation opt-out of declarations saved before `sub_agents` existed.

        The stock declaration explicitly opts in. Supplied agents still receive plugins per run,
        where self-delegation requires capabilities bound to the agent instead.
        """
        if self.factory != _CODER_FACTORY or 'sub_agents' in self.settings:
            return self
        return self.model_copy(update={'settings': {**self.settings, 'sub_agents': False}})
