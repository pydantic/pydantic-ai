"""The per-model settings a user can edit, validated before they reach `ModelSettings`."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pydantic_ai.settings import ModelSettings


class ModelSettingsForm(BaseModel):
    """Overrides for one model. Unset fields leave the provider default in place."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    max_tokens: int | None = Field(default=None, gt=0, description='Cap on tokens the model may generate per request.')
    temperature: float | None = Field(
        default=None, ge=0, le=2, allow_inf_nan=False, description='Sampling randomness; 0 is deterministic.'
    )
    top_p: float | None = Field(
        default=None, gt=0, le=1, allow_inf_nan=False, description='Nucleus sampling cutoff; lower is narrower.'
    )
    top_k: int | None = Field(default=None, gt=0, description='Sample from the top K tokens only (where supported).')
    seed: int | None = Field(default=None, description='Fixed seed for repeatable sampling (where supported).')
    timeout: float | None = Field(default=None, gt=0, allow_inf_nan=False, description='Request timeout in seconds.')
    presence_penalty: float | None = Field(
        default=None, ge=-2, le=2, allow_inf_nan=False, description='Push the model toward new topics.'
    )
    frequency_penalty: float | None = Field(
        default=None, ge=-2, le=2, allow_inf_nan=False, description='Push the model away from repeating itself.'
    )
    parallel_tool_calls: bool | None = Field(default=None, description='Let the model call several tools at once.')
    thinking: bool | Literal['minimal', 'low', 'medium', 'high', 'xhigh'] | None = Field(
        default=None, description='Extended thinking: on, off, or an effort level (where supported).'
    )
    service_tier: Literal['auto', 'default', 'flex', 'priority'] | None = Field(
        default=None, description='Provider service tier (OpenAI).'
    )

    def to_model_settings(self) -> ModelSettings | None:
        """What `agent.run(model_settings=...)` receives; `None` when nothing is set.

        Built field by field: `ModelSettings.timeout` admits `httpx.Timeout`, so Pydantic
        cannot validate the TypedDict as a whole.
        """
        settings = ModelSettings()
        if self.max_tokens is not None:
            settings['max_tokens'] = self.max_tokens
        if self.temperature is not None:
            settings['temperature'] = self.temperature
        if self.top_p is not None:
            settings['top_p'] = self.top_p
        if self.top_k is not None:
            settings['top_k'] = self.top_k
        if self.seed is not None:
            settings['seed'] = self.seed
        if self.timeout is not None:
            settings['timeout'] = self.timeout
        if self.presence_penalty is not None:
            settings['presence_penalty'] = self.presence_penalty
        if self.frequency_penalty is not None:
            settings['frequency_penalty'] = self.frequency_penalty
        if self.parallel_tool_calls is not None:
            settings['parallel_tool_calls'] = self.parallel_tool_calls
        if self.thinking is not None:
            settings['thinking'] = self.thinking
        if self.service_tier is not None:
            settings['service_tier'] = self.service_tier
        return settings or None


def model_settings_from_json(values: dict[str, JsonValue]) -> ModelSettingsForm:
    """Validate what the store holds for one model."""
    return ModelSettingsForm.model_validate(values)
