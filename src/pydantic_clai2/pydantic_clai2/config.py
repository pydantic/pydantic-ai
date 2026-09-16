"""Validated settings, independent of persistence and terminal code."""

from pydantic import BaseModel, ConfigDict, Field, JsonValue


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
    thinking: bool = Field(default=True, description="Show the model's thinking as it streams.")
    splash: bool = Field(default=True, description='Animate the startup splash. Takes effect next start.')
    shell_lines: int = Field(
        default=20, ge=0, le=1000, description='Lines of shell output to preview before truncating.'
    )
    grep_lines: int = Field(default=20, ge=0, le=1000, description='Grep result lines to preview before truncating.')
    smooth_seconds: float = Field(
        default=0.5,
        ge=0.1,
        le=5,
        allow_inf_nan=False,
        description='Catch-up window for smoothed response streaming, 0.1 to 5 seconds.',
    )


SETTING_FIELDS = {
    'model': 'model',
    'run.request_limit': 'request_limit',
    'display.thinking': 'thinking',
    'display.splash': 'splash',
    'display.shell_lines': 'shell_lines',
    'display.grep_lines': 'grep_lines',
    'display.smooth_seconds': 'smooth_seconds',
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
