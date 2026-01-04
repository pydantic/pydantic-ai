"""Cerebras model implementation using OpenAI-compatible API."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from typing_extensions import override

from ..exceptions import UserError
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings, ThinkingConfig
from . import ModelRequestParameters

try:
    from openai import AsyncOpenAI

    from .openai import OpenAIChatModel, OpenAIChatModelSettings
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Cerebras model, '
        'you can use the `cerebras` optional group — `pip install "pydantic-ai-slim[cerebras]"'
    ) from _import_error

__all__ = ('CerebrasModel', 'CerebrasModelName', 'CerebrasModelSettings')

LatestCerebrasModelNames = Literal[
    'gpt-oss-120b',
    'llama-3.3-70b',
    'llama3.1-8b',
    'qwen-3-235b-a22b-instruct-2507',
    'qwen-3-32b',
    'zai-glm-4.6',
]

CerebrasModelName = str | LatestCerebrasModelNames
"""Possible Cerebras model names.

Since Cerebras supports a variety of models and the list changes frequently, we explicitly list known models
but allow any name in the type hints.

See <https://inference-docs.cerebras.ai/models/overview> for an up to date list of models.
"""


class CerebrasModelSettings(ModelSettings, total=False):
    """Settings used for a Cerebras model request.

    ALL FIELDS MUST BE `cerebras_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    cerebras_disable_reasoning: bool
    """Disable reasoning for the model.

    This setting is only supported on reasoning models: `zai-glm-4.6` and `gpt-oss-120b`.

    See [the Cerebras docs](https://inference-docs.cerebras.ai/resources/openai#passing-non-standard-parameters) for more details.
    """


@dataclass(init=False)
class CerebrasModel(OpenAIChatModel):
    """A model that uses Cerebras's OpenAI-compatible API.

    Cerebras provides ultra-fast inference powered by the Wafer-Scale Engine (WSE).

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    def __init__(
        self,
        model_name: CerebrasModelName,
        *,
        provider: Literal['cerebras'] | Provider[AsyncOpenAI] = 'cerebras',
        profile: ModelProfileSpec | None = None,
        settings: CerebrasModelSettings | None = None,
    ):
        """Initialize a Cerebras model.

        Args:
            model_name: The name of the Cerebras model to use.
            provider: The provider to use. Defaults to 'cerebras'.
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
        merged_settings = cast(CerebrasModelSettings, merged_settings or {})

        # Apply unified thinking config if no provider-specific setting
        if 'cerebras_disable_reasoning' not in merged_settings:
            disable_reasoning = self._resolve_reasoning_config(merged_settings)
            if disable_reasoning is not None:
                merged_settings['cerebras_disable_reasoning'] = disable_reasoning

        new_settings = _cerebras_settings_to_openai_settings(merged_settings)
        return new_settings, customized_parameters

    def _resolve_reasoning_config(self, model_settings: CerebrasModelSettings) -> bool | None:
        """Resolve unified thinking settings to Cerebras disable_reasoning config.

        Args:
            model_settings: The model settings to check.

        Returns:
            True to disable reasoning, None to leave default behavior.

        Raises:
            UserError: If thinking is requested but the model doesn't support it.
        """
        thinking = model_settings.get('thinking')
        if thinking is None:
            return None

        # Check model support
        if not self.profile.supports_thinking:
            raise UserError(
                f'Model {self._model_name!r} does not support reasoning. '
                'Use a reasoning model (e.g., zai-glm-4.6, gpt-oss-120b) or remove the thinking setting.'
            )

        # Handle boolean shorthand
        if thinking is True:
            # Enable thinking (default behavior, no need to set disable_reasoning)
            return None
        elif thinking is False:
            if self.profile.thinking_always_enabled:
                raise UserError(f'Model {self._model_name!r} has reasoning always enabled and cannot be disabled.')
            # Disable reasoning
            return True

        # Handle ThinkingConfig dict
        config: ThinkingConfig = thinking

        # Warn about ignored settings
        ignored: list[str] = []
        for setting in ('budget_tokens', 'effort', 'include_in_response', 'summary'):
            if config.get(setting) is not None:
                ignored.append(setting)
        if ignored:
            from ._warnings import warn_settings_ignored_batch

            warn_settings_ignored_batch(
                setting_names=ignored,
                provider_name=self.system,
                model_name=self._model_name,
                reason='Cerebras reasoning can only be enabled or disabled. Reasoning will be enabled with default behavior',
            )

        # Check enabled=False
        if config.get('enabled') is False:
            if self.profile.thinking_always_enabled:
                raise UserError(f'Model {self._model_name!r} has reasoning always enabled and cannot be disabled.')
            return True

        # Any other config enables thinking (no need to disable)
        return None


def _cerebras_settings_to_openai_settings(model_settings: CerebrasModelSettings) -> OpenAIChatModelSettings:
    """Transforms a 'CerebrasModelSettings' object into an 'OpenAIChatModelSettings' object.

    Args:
        model_settings: The 'CerebrasModelSettings' object to transform.

    Returns:
        An 'OpenAIChatModelSettings' object with equivalent settings.
    """
    extra_body = cast(dict[str, Any], model_settings.get('extra_body', {}))

    if (disable_reasoning := model_settings.pop('cerebras_disable_reasoning', None)) is not None:
        extra_body['disable_reasoning'] = disable_reasoning

    if extra_body:
        model_settings['extra_body'] = extra_body

    return OpenAIChatModelSettings(**model_settings)  # type: ignore[reportCallIssue]
