"""The per-model settings a user can edit, validated before they reach `ModelSettings`."""

import re
from typing import Literal

from anthropic.types.beta import (
    BetaThinkingBlockBindingParam,
    BetaThinkingConfigAdaptiveParam,
    BetaThinkingConfigEnabledParam,
    BetaThinkingConfigParam,
)
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_ai.settings import ModelSettings

from .custom_params import expand_params


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

    openai_reasoning_effort: Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'] | None = Field(
        default=None, description='OpenAI reasoning effort. Overrides generic thinking.'
    )
    openai_reasoning_context: Literal['auto', 'current_turn', 'all_turns'] | None = Field(
        default=None, description='Responses reasoning retained across turns.'
    )
    openai_reasoning_mode: Literal['standard', 'pro'] | None = Field(
        default=None, description='Responses reasoning mode.'
    )
    openai_reasoning_summary: Literal['auto', 'concise', 'detailed'] | None = Field(
        default=None, description='Responses reasoning summary display.'
    )
    openai_text_verbosity: Literal['low', 'medium', 'high'] | None = Field(
        default=None, description='Responses answer verbosity.'
    )
    anthropic_thinking_mode: Literal['enabled', 'adaptive', 'disabled'] | None = Field(
        default=None, description='Claude thinking mode. Overrides generic thinking.'
    )
    anthropic_thinking_budget: int | None = Field(
        default=None, ge=1024, description='Classic thinking token budget, below max_tokens. Default: 10000.'
    )
    anthropic_effort: Literal['low', 'medium', 'high', 'xhigh', 'max'] | None = Field(
        default=None, description='Claude response effort.'
    )
    anthropic_thinking_display: Literal['updates', 'summarized'] | None = Field(
        default=None,
        description=(
            "How Fable 5.1 reports reasoning between tool calls. 'updates' shows status lines while "
            "reasoning stays hidden; 'summarized' folds them into a condensed trace."
        ),
    )
    anthropic_preserved_thinking: Literal['error', 'drop_block'] | None = Field(
        default=None,
        description=(
            'Preserved thinking. How Anthropic handles a replayed thinking block whose conversation '
            "prefix changed: 'error' rejects the request, 'drop_block' continues without that turn's reasoning."
        ),
    )
    anthropic_interleaved_thinking: bool | None = Field(
        default=None, description='Let Claude 4 think between tool calls (adds the interleaved-thinking beta).'
    )

    glm_thinking: Literal['enabled', 'disabled'] | None = Field(
        default=None, description='GLM deep-thinking mode. Newer GLM models also decide this themselves.'
    )
    glm_clear_thinking: bool | None = Field(
        default=None, description='GLM: true clears earlier reasoning; false preserves it (GLM-4.5 and newer).'
    )
    glm_reasoning_effort: Literal['max', 'xhigh', 'high', 'medium', 'low', 'minimal', 'none'] | None = Field(
        default=None, description='GLM chain-of-thought effort (GLM-5.2 and newer).'
    )

    custom_params: dict[str, JsonValue] | None = Field(
        default=None, description='Custom request body parameters. Dotted keys nest; custom values win.'
    )

    @field_validator('custom_params')
    @classmethod
    def valid_custom_params(cls, value: dict[str, JsonValue] | None) -> dict[str, JsonValue] | None:
        """Reject malformed persisted keys before request conversion can raise outside validation."""
        if value is not None:
            expand_params(pairs=value)
        return value

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
        settings.update(self._openai_settings())
        settings.update(self._anthropic_settings())
        body = self._glm_body()
        if self.custom_params:
            body.update(expand_params(pairs=self.custom_params))
        if body:
            settings['extra_body'] = body
        return settings or None

    def _openai_settings(self) -> OpenAIResponsesModelSettings:
        openai = OpenAIResponsesModelSettings()
        if self.openai_reasoning_effort is not None:
            openai['openai_reasoning_effort'] = self.openai_reasoning_effort
        if self.openai_reasoning_context is not None:
            openai['openai_reasoning_context'] = self.openai_reasoning_context
        if self.openai_reasoning_mode is not None:
            openai['openai_reasoning_mode'] = self.openai_reasoning_mode
        if self.openai_reasoning_summary is not None:
            openai['openai_reasoning_summary'] = self.openai_reasoning_summary
        if self.openai_text_verbosity is not None:
            openai['openai_text_verbosity'] = self.openai_text_verbosity
        return openai

    def _anthropic_settings(self) -> AnthropicModelSettings:
        anthropic = AnthropicModelSettings()
        if self.anthropic_effort is not None:
            anthropic['anthropic_effort'] = self.anthropic_effort
        thinking = self._anthropic_thinking()
        if thinking is not None:
            anthropic['anthropic_thinking'] = thinking
            if thinking['type'] == 'enabled' and self.max_tokens is None:
                anthropic['max_tokens'] = (self.anthropic_thinking_budget or 10000) + 4096
        betas = self._anthropic_betas(thinking=thinking)
        if betas:
            anthropic['extra_headers'] = {'anthropic-beta': ','.join(betas)}
        return anthropic

    def _anthropic_thinking(self) -> BetaThinkingConfigParam | None:
        """The `thinking` object, carrying Fable 5.1's `display` and `block_binding` keys.

        `display` and `block_binding` only exist on the enabled and adaptive shapes, so asking
        for one picks adaptive unless a mode was chosen. Core adds the block-binding beta when
        `block_binding` is present.
        """
        mode = self.anthropic_thinking_mode
        if mode is None:
            if self.anthropic_thinking_display is None and self.anthropic_preserved_thinking is None:
                return None
            mode = 'adaptive'
        if mode == 'disabled':
            return {'type': 'disabled'}
        thinking: BetaThinkingConfigEnabledParam | BetaThinkingConfigAdaptiveParam
        if mode == 'enabled':
            thinking = {'type': 'enabled', 'budget_tokens': self.anthropic_thinking_budget or 10000}
        else:
            thinking = {'type': 'adaptive'}
        if self.anthropic_thinking_display is not None:
            thinking['display'] = self.anthropic_thinking_display
        if self.anthropic_preserved_thinking is not None:
            thinking['block_binding'] = BetaThinkingBlockBindingParam(
                prefix_mismatch_behavior=self.anthropic_preserved_thinking
            )
        return thinking

    def _anthropic_betas(self, *, thinking: BetaThinkingConfigParam | None) -> list[str]:
        """Betas these controls need. Core attaches the thinking-binding beta itself.

        The display beta follows the `display` value that actually reaches the body, so a
        disabled mode does not ask for it.
        """
        betas: list[str] = []
        if self.anthropic_interleaved_thinking:
            betas.append('interleaved-thinking-2025-05-14')
        if thinking is not None and thinking['type'] != 'disabled' and thinking.get('display') == 'updates':
            betas.append('thinking-display-updates-2026-08-18')
        return betas

    def _glm_body(self) -> dict[str, JsonValue]:
        """GLM's native `thinking` and `reasoning_effort` fields; proxies adjust via custom params.

        Only keys the user set are sent: GLM-5 decides whether to think on its own, so forcing
        `type` would override that.
        """
        thinking: dict[str, JsonValue] = {}
        if self.glm_thinking is not None:
            thinking['type'] = self.glm_thinking
        if self.glm_clear_thinking is not None:
            thinking['clear_thinking'] = self.glm_clear_thinking
        body: dict[str, JsonValue] = {'thinking': thinking} if thinking else {}
        if self.glm_reasoning_effort is not None and self.glm_thinking != 'disabled':
            body['reasoning_effort'] = self.glm_reasoning_effort
        return body


def model_defaults(*, model: str) -> dict[str, JsonValue]:
    """CLAI defaults for GPT-6 and GPT-5.6 families, independent of provider."""
    name = model.partition(':')[2] if ':' in model else model
    name = name.rsplit('/', 1)[-1]
    if not re.match(r'^gpt-(?:6(?:\.\d+)?|5\.6)(?:$|[-:])', name):
        return {}
    return {
        'thinking': True,
        'service_tier': 'default',
        'openai_reasoning_effort': 'medium',
        'openai_reasoning_context': 'all_turns',
        'openai_reasoning_mode': 'standard',
        'openai_reasoning_summary': 'detailed',
        'openai_text_verbosity': 'low',
    }


def model_settings_from_json(values: dict[str, JsonValue], *, model: str = '') -> ModelSettingsForm:
    """Read shared preferences, ignoring keys from newer versions without changing the store.

    Known fields still validate normally. New edits use the strict form directly.
    """
    return ModelSettingsForm.model_validate({**model_defaults(model=model), **values}, extra='ignore')
