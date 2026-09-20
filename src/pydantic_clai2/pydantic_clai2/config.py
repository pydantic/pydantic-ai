"""Validated settings, independent of persistence and terminal code."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from .theme import names


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
    session_namer: bool = Field(default=True, description='Name saved sessions in the background using a model.')
    session_namer_model: str | None = Field(
        default=None, description='Naming model override; null uses the current model.'
    )
    theme: str = Field(
        default='default', description='Keep CLAI colours, or select a bundled Termflow palette with /theme.'
    )
    thinking: bool = Field(default=True, description="Show the model's thinking as it streams.")
    splash: bool = Field(default=True, description='Animate the startup splash. Takes effect next start.')
    tool_output: bool = Field(
        default=False, description='Show tool output previews and file diffs below tool summaries.'
    )
    shell_lines: int = Field(
        default=20, ge=0, le=1000, description='Shell preview lines when display.tool_output is enabled.'
    )
    grep_lines: int = Field(
        default=20, ge=0, le=1000, description='Grep preview lines when display.tool_output is enabled.'
    )
    smooth_seconds: float = Field(
        default=0.5,
        ge=0.1,
        le=5,
        allow_inf_nan=False,
        description='Catch-up window for smoothed response streaming, 0.1 to 5 seconds.',
    )

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
    'display.splash': 'splash',
    'display.tool_output': 'tool_output',
    'display.shell_lines': 'shell_lines',
    'display.grep_lines': 'grep_lines',
    'display.smooth_seconds': 'smooth_seconds',
    'run.tool_retries': 'tool_retries',
    'sessions.naming': 'session_namer',
    'sessions.naming_model': 'session_namer_model',
}


def resolve_settings(overrides: dict[str, JsonValue]) -> Settings:
    """Reject unknown setting names and validate stored or supplied values."""
    unknown = overrides.keys() - SETTING_FIELDS.keys()
    if unknown:
        raise ValueError(f'Unknown settings: {", ".join(sorted(unknown))}')
    return Settings.model_validate({SETTING_FIELDS[key]: value for key, value in overrides.items()})


class PluginSettings(BaseModel):
    """Declaration for a trusted plugin: a module with `activate`, or `module:Capability`."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    id: str = Field(min_length=1)
    factory: str = Field(pattern=r'^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?::[A-Za-z_]\w*)?$')
    path: str | None = None
    enabled: bool = True
    settings: dict[str, JsonValue] = Field(default_factory=dict)
