"""The per-model settings a user can edit, validated before they reach `ModelSettings`."""

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from pydantic_ai.settings import CacheRetention, ModelSettings
from pydantic_ai_pax.models.custom_params import expand_params
from pydantic_ai_pax.models.profiles import KEY_NAME, base_model, provider_of

if TYPE_CHECKING:
    # Typing only: the provider modules load the Anthropic and OpenAI SDKs, and every turn loads this module.
    from anthropic.types.beta import (
        BetaThinkingBlockBindingParam,
        BetaThinkingConfigAdaptiveParam,
        BetaThinkingConfigEnabledParam,
        BetaThinkingConfigParam,
    )

    from pydantic_ai.models.anthropic import AnthropicModelSettings
    from pydantic_ai.models.openai import OpenAIResponsesModelSettings

ULTRAFAST_CODEX_MODELS = frozenset(
    f'openai-codex:{name}' for name in ('gpt-6.2-astra', 'gpt-6.2-sol', 'gpt-6.1-sol', 'gpt-6-astra')
)
"""Codex models that offer the Ultrafast speed, per https://developers.openai.com/codex/speed."""

CodexSpeed = Literal['standard', 'fast', 'ultrafast']
"""A Codex speed, as the Codex CLI's `/fast` and `/ultrafast` select it."""

_CODEX_SERVICE_TIERS: dict[CodexSpeed, Literal['default', 'priority', 'ultrafast']] = {
    'standard': 'default',
    'fast': 'priority',
    'ultrafast': 'ultrafast',
}
"""The `openai_service_tier` each Codex speed sends, as the Codex CLI's `/fast` and `/ultrafast` do."""

_TIER_SPEEDS: dict[str, CodexSpeed] = {'default': 'standard', 'priority': 'fast'}
"""The Codex speed a saved `service_tier` stands for."""

_SPEED_TIERS: dict[CodexSpeed, Literal['default', 'priority']] = {
    'standard': 'default',
    'fast': 'priority',
    'ultrafast': 'priority',
}
"""The `service_tier` saved beside each Codex speed for older Pax builds, its nearest for ultrafast."""


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
    cache: bool | CacheRetention | None = Field(
        default=None,
        description='Prompt caching: off, on, or a retention (where the provider supports configuring it).',
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
        default=None, description='OpenAI answer verbosity.'
    )
    codex_speed: CodexSpeed | None = Field(
        default=None,
        description=(
            'Codex speed. Fast and ultrafast use more ChatGPT credits; availability depends on your model and '
            'plan. Overrides service_tier.'
        ),
    )
    anthropic_cache: bool | Literal['5m', '1h'] | None = Field(
        default=None, description='Claude conversation caching: off, on (5 minutes), or a TTL.'
    )
    anthropic_cache_instructions: bool | Literal['5m', '1h'] | None = Field(
        default=None, description='Cache static Claude instructions.'
    )
    anthropic_cache_tool_definitions: bool | Literal['5m', '1h'] | None = Field(
        default=None, description='Cache Claude tool descriptions and schemas.'
    )
    anthropic_thinking_mode: Literal['enabled', 'adaptive', 'disabled', 'between_tools'] | None = Field(
        default=None,
        description="Claude thinking mode. Overrides generic thinking. 'between_tools' is Sonnet 5.5's off switch.",
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
            "How Claude reports reasoning between tool calls. 'updates' (Fable 5.1) shows status lines while "
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
    base_url: str | None = Field(
        default=None,
        min_length=1,
        description="Endpoint this model's provider is built with, such as a gateway. Wins over its environment variable.",
    )
    api_key: str | None = Field(
        default=None,
        pattern=KEY_NAME,
        description='The /keys entry this model uses instead of the API key in its environment variable.',
    )

    @field_validator('base_url')
    @classmethod
    def absolute_url(cls, value: str | None) -> str | None:
        """Reject a value no provider could send requests to, such as one without `https://`."""
        if value is not None:
            url = urlsplit(value)
            if url.scheme not in ('http', 'https') or not url.netloc:
                raise ValueError('must be an http:// or https:// URL, such as https://gateway.example/v1')
        return value

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
        if self.cache is not None:
            settings['cache'] = self.cache
        settings.update(self._openai_settings())
        settings.update(self._anthropic_settings())
        body = self._glm_body()
        if self.custom_params:
            body.update(expand_params(pairs=self.custom_params))
        if body:
            settings['extra_body'] = body
        return settings or None

    def _openai_settings(self) -> 'OpenAIResponsesModelSettings':
        openai: OpenAIResponsesModelSettings = {}
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
        if self.codex_speed is not None:
            openai['openai_service_tier'] = _CODEX_SERVICE_TIERS[self.codex_speed]
        return openai

    def _anthropic_settings(self) -> 'AnthropicModelSettings':
        anthropic: AnthropicModelSettings = {}
        if self.anthropic_cache is not None:
            anthropic['anthropic_cache'] = self.anthropic_cache
        if self.anthropic_cache_instructions is not None:
            anthropic['anthropic_cache_instructions'] = self.anthropic_cache_instructions
        if self.anthropic_cache_tool_definitions is not None:
            anthropic['anthropic_cache_tool_definitions'] = self.anthropic_cache_tool_definitions
        if self.anthropic_effort is not None:
            anthropic['anthropic_effort'] = self.anthropic_effort
        thinking = self._anthropic_thinking()
        if thinking is not None:
            anthropic['anthropic_thinking'] = thinking
        betas = self._anthropic_betas(thinking=thinking)
        if betas:
            anthropic['extra_headers'] = {'anthropic-beta': ','.join(betas)}
        return anthropic

    def _anthropic_thinking(self) -> 'BetaThinkingConfigParam | None':
        """The `thinking` object, carrying the `display` and `block_binding` keys.

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
        if mode == 'between_tools':
            return {'type': 'between_tools'}
        thinking: BetaThinkingConfigEnabledParam | BetaThinkingConfigAdaptiveParam
        if mode == 'enabled':
            thinking = {'type': 'enabled', 'budget_tokens': self.anthropic_thinking_budget or 10000}
        else:
            thinking = {'type': 'adaptive'}
        if self.anthropic_thinking_display is not None:
            thinking['display'] = self.anthropic_thinking_display
        if self.anthropic_preserved_thinking is not None:
            binding: BetaThinkingBlockBindingParam = {'prefix_mismatch_behavior': self.anthropic_preserved_thinking}
            thinking['block_binding'] = binding
        return thinking

    def _anthropic_betas(self, *, thinking: 'BetaThinkingConfigParam | None') -> list[str]:
        """Betas these controls need. Core attaches the thinking-binding beta itself.

        The display beta follows the `display` value that actually reaches the body, so a
        disabled mode does not ask for it.
        """
        betas: list[str] = []
        if self.anthropic_interleaved_thinking:
            betas.append('interleaved-thinking-2025-05-14')
        if (
            thinking is not None
            and thinking['type'] in ('enabled', 'adaptive')
            and thinking.get('display') == 'updates'
        ):
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


def with_codex_speed(saved: Mapping[str, JsonValue], speed: CodexSpeed | None) -> dict[str, JsonValue]:
    """`saved` with `speed` and the `service_tier` older Pax builds read for it, its nearest for ultrafast.

    `None` drops both, so the model's default speed applies again.
    """
    row = {key: value for key, value in saved.items() if key != 'codex_speed'}
    if speed is not None:
        return {**row, 'codex_speed': speed, 'service_tier': _SPEED_TIERS[speed]}
    if isinstance(tier := row.get('service_tier'), str) and tier in _TIER_SPEEDS:
        del row['service_tier']
    return row


def chosen_codex_speed(saved: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """`saved` with the Codex speed last chosen: a tier an older Pax build changed since outranks `codex_speed`."""
    speed, tier = saved.get('codex_speed'), saved.get('service_tier')
    if speed in _SPEED_TIERS and isinstance(tier, str) and tier in _TIER_SPEEDS and _SPEED_TIERS[speed] != tier:
        return {**saved, 'codex_speed': _TIER_SPEEDS[tier]}
    return dict(saved)


def model_defaults(*, model: str, saved: Mapping[str, JsonValue] | None = None) -> dict[str, JsonValue]:
    """Pax's prompt caching default for every model, reasoning defaults for GPT families, and Codex speeds.

    Caching uses the unified `cache` setting, which providers without configurable caching ignore. A
    saved `anthropic_cache*` override still applies, since provider-specific cache settings take
    precedence over it. A Codex model's speed follows a `service_tier` in `saved` that `/fast` writes,
    as older Pax builds also read it; without a saved tier, the Codex models that offer ultrafast use it.
    """
    defaults: dict[str, JsonValue] = {'cache': True, **gpt_defaults(model=model)}
    if provider_of(model) == 'openai-codex':
        tier = (saved or {}).get('service_tier')
        if isinstance(tier, str) and tier in _TIER_SPEEDS:
            defaults['codex_speed'] = _TIER_SPEEDS[tier]
        elif tier is None and base_model(model) in ULTRAFAST_CODEX_MODELS:
            defaults['codex_speed'] = 'ultrafast'
    return defaults


def gpt_defaults(*, model: str) -> dict[str, JsonValue]:
    """Pax's reasoning defaults for GPT-6 and GPT-5.6 families, whichever provider serves them."""
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


def default_model_settings(
    *, model: str, saved: Mapping[str, JsonValue], selected: Mapping[str, JsonValue] | None = None
) -> ModelSettings | None:
    """The family defaults for `model` that `saved` leaves unset, as `ModelSettings`.

    Every default fills exactly one `ModelSettings` key, so merging the saved settings over these
    gives the same settings as `model_settings_from_json(saved, model=model)`. `selected` is the saved
    row of the CLI selection `model` runs under, such as its chain, whose tier picks a Codex speed.
    Without it, as for `open_stock_agent`, Codex gets no default speed, so a tier set above these applies.
    """
    defaults = model_defaults(model=model, saved=selected)
    if selected is None:
        defaults.pop('codex_speed', None)
    defaults = {key: value for key, value in defaults.items() if key not in saved}
    return ModelSettingsForm.model_validate(defaults).to_model_settings()


def model_settings_from_json(values: dict[str, JsonValue], *, model: str | None = None) -> ModelSettingsForm:
    """Read shared preferences, ignoring keys from newer versions without changing the store.

    Known fields still validate normally. New edits use the strict form directly. With `model`, its
    family defaults fill the fields `values` leaves unset; without it, only `values` are read.
    """
    defaults = model_defaults(model=model, saved=values) if model is not None else {}
    return ModelSettingsForm.model_validate({**defaults, **chosen_codex_speed(values)}, extra='ignore')
