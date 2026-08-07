"""Z.AI (Zhipu AI) model implementation using OpenAI-compatible API."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from typing_extensions import override

from ..profiles import ModelProfileSpec
from ..profiles.zai import ZaiModelProfile
from ..providers import Provider
from ..settings import ModelSettings
from . import ModelRequestParameters

try:
    from openai import AsyncOpenAI, Omit, omit

    from .openai import OpenAIChatModel, OpenAIChatModelSettings
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Z.AI model, '
        'you can use the `zai` optional group — `pip install "pydantic-ai-slim[zai]"`'
    ) from _import_error

__all__ = ('ZaiModel', 'ZaiModelName', 'ZaiModelSettings')

LatestZaiModelNames = Literal[
    'glm-5.2',
    'glm-5.1',
    'glm-5',
    'glm-5-turbo',
    'glm-5v-turbo',
    'glm-4.7',
    'glm-4.7-flash',
    'glm-4.7-flashx',
    'glm-4.6',
    'glm-4.6v',
    'glm-4.6v-flash',
    'glm-4.6v-flashx',
    'glm-4.5',
    'glm-4.5v',
    'glm-4.5-air',
    'glm-4.5-airx',
    'glm-4.5-x',
    'glm-4.5-flash',
    'glm-4-32b-0414-128k',
    'autoglm-phone-multilingual',
]

ZaiModelName = str | LatestZaiModelNames
"""Possible Z.AI model names.

Since Z.AI supports a variety of models and the list changes frequently, we explicitly list known models
but allow any name in the type hints.

See <https://docs.z.ai/> for an up to date list of models.
"""


class ZaiModelSettings(ModelSettings, total=False):
    """Settings used for a Z.AI model request.

    ALL FIELDS MUST BE `zai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    zai_clear_thinking: bool
    """Whether to clear historical thinking content from prior turns.

    Defaults to `False` (preserved thinking) on thinking-capable models, retaining reasoning content
    from prior assistant responses for improved multi-turn coherence and consistency with other
    providers. Set to `True` to clear it instead.

    Only affects cross-turn historical thinking blocks; it does not change whether the model
    generates thinking in the current turn (controlled by the unified `thinking` setting).

    When using preserved thinking, you must return the complete, unmodified `reasoning_content`
    back to the API. All consecutive `reasoning_content` blocks must exactly match the original sequence.

    See [the Z.AI docs](https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking) for more details.
    """


@dataclass(init=False)
class ZaiModel(OpenAIChatModel):
    """A model that uses Z.AI's OpenAI-compatible API.

    Z.AI (Zhipu AI) provides GLM models with support for thinking/reasoning mode
    and preserved thinking across turns.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    def __init__(
        self,
        model_name: ZaiModelName,
        *,
        provider: Literal['zai'] | Provider[AsyncOpenAI] = 'zai',
        profile: ModelProfileSpec | None = None,
        settings: ZaiModelSettings | None = None,
    ):
        """Initialize a Z.AI model.

        Args:
            model_name: The name of the Z.AI model to use.
            provider: The provider to use. Defaults to 'zai'.
            profile: The model profile to use. Defaults to a profile based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @override
    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        merged_settings, customized_parameters = super().prepare_request(model_settings, model_request_parameters)
        profile = cast(ZaiModelProfile, self.profile)
        new_settings = _zai_settings_to_openai_settings(
            cast(ZaiModelSettings, merged_settings or {}),
            customized_parameters,
            supports_thinking=profile.get('supports_thinking', False),
            supports_reasoning_effort=profile.get('zai_supports_reasoning_effort', False),
        )
        return new_settings, customized_parameters

    @override
    def _translate_thinking(
        self,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Omit:
        # Z.AI uses `extra_body.thinking.type`, not the OpenAI `reasoning_effort` parameter,
        # which `prepare_request` translates the unified `thinking` setting into.
        del model_settings, model_request_parameters
        return omit


def _zai_settings_to_openai_settings(
    model_settings: ZaiModelSettings,
    model_request_parameters: ModelRequestParameters,
    *,
    supports_thinking: bool,
    supports_reasoning_effort: bool,
) -> OpenAIChatModelSettings:
    """Transforms a 'ZaiModelSettings' object into an 'OpenAIChatModelSettings' object.

    Maps the unified `thinking` setting and Z.AI-specific `zai_clear_thinking` into the
    `extra_body.thinking` payload expected by the Z.AI API's OpenAI-compatible endpoint. When the model
    supports it, an explicit effort level is additionally forwarded as `extra_body.reasoning_effort`.

    Args:
        model_settings: The 'ZaiModelSettings' object to transform.
        model_request_parameters: The request parameters carrying the resolved unified `thinking` value.
        supports_thinking: Whether the model supports thinking, gating the default `clear_thinking`.
        supports_reasoning_effort: Whether the model accepts a per-request `reasoning_effort` (GLM-5.2).

    Returns:
        An 'OpenAIChatModelSettings' object with equivalent settings.
    """
    extra_body = dict(cast(dict[str, Any], model_settings.get('extra_body', {})))

    thinking_payload: dict[str, Any] = {}
    thinking_level = model_request_parameters.thinking
    if thinking_level is False:
        thinking_payload['type'] = 'disabled'
    elif thinking_level is not None:
        # `True` and any effort level (`'minimal'`/`'low'`/`'medium'`/`'high'`/`'xhigh'`)
        # enable thinking. Effort granularity is forwarded separately via `reasoning_effort`
        # on models that support it.
        thinking_payload['type'] = 'enabled'

    clear_thinking = model_settings.get('zai_clear_thinking')
    if clear_thinking is None and supports_thinking:
        # Preserve cross-turn reasoning by default on thinking-capable models, for consistency with other
        # providers (e.g. Cerebras for GLM). Independent of this turn's `type`; set `zai_clear_thinking=True`
        # to clear. Left unset on non-thinking models so they receive no thinking payload.
        clear_thinking = False
    if clear_thinking is not None:
        thinking_payload['clear_thinking'] = clear_thinking

    if thinking_payload:
        extra_body['thinking'] = thinking_payload

    # An explicit effort level (not a bare `True`/`False`) maps to Z.AI's `reasoning_effort`; a plain
    # `thinking=True` leaves it unset so Z.AI applies its own default.
    if supports_reasoning_effort and isinstance(thinking_level, str):
        extra_body['reasoning_effort'] = thinking_level

    filtered = {k: v for k, v in model_settings.items() if not k.startswith('zai_')}
    if extra_body:
        filtered['extra_body'] = extra_body

    return cast(OpenAIChatModelSettings, filtered)
