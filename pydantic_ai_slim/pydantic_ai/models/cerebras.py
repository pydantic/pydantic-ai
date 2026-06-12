"""Cerebras model implementation using OpenAI-compatible API."""

from __future__ import annotations as _annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, cast

from typing_extensions import override

from .._warnings import PydanticAIDeprecationWarning
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings
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
    'zai-glm-4.7',
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

    Deprecated: use the unified `thinking=False` setting instead.
    """

    cerebras_clear_thinking: bool
    """Set Cerebras's `clear_thinking` flag via `extra_body`."""


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
    def _translate_thinking(
        self,
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        """Pass through an explicit `openai_reasoning_effort` (including the `'none'` injected to disable reasoning)."""
        from openai import omit

        # Disabling is injected as `openai_reasoning_effort='none'` in `_cerebras_settings_to_openai_settings`;
        # other unified thinking levels are omitted because Cerebras reasons by default.
        if effort := model_settings.get('openai_reasoning_effort'):
            return effort
        return omit

    @override
    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        merged_settings, customized_parameters = super().prepare_request(model_settings, model_request_parameters)
        new_settings = _cerebras_settings_to_openai_settings(
            cast(CerebrasModelSettings, merged_settings or {}), customized_parameters
        )
        return new_settings, customized_parameters


def _cerebras_settings_to_openai_settings(
    model_settings: CerebrasModelSettings, model_request_parameters: ModelRequestParameters
) -> OpenAIChatModelSettings:
    """Transforms a 'CerebrasModelSettings' object into an 'OpenAIChatModelSettings' object.

    Args:
        model_settings: The 'CerebrasModelSettings' object to transform.
        model_request_parameters: The 'ModelRequestParameters' object to use for the transformation.

    Returns:
        An 'OpenAIChatModelSettings' object with equivalent settings.
    """
    extra_body = cast(dict[str, Any], model_settings.get('extra_body', {}))

    disable_reasoning = model_settings.pop(
        'cerebras_disable_reasoning', None
    )  # TODO(v3): remove cerebras_disable_reasoning
    if disable_reasoning is not None:
        warnings.warn(
            '`cerebras_disable_reasoning` is deprecated, use the unified `thinking=False` setting instead.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
    else:
        disable_reasoning = model_request_parameters.thinking is False

    if (clear_thinking := model_settings.pop('cerebras_clear_thinking', None)) is not None:
        extra_body['clear_thinking'] = clear_thinking

    if extra_body:
        model_settings['extra_body'] = extra_body

    openai_settings = OpenAIChatModelSettings(**model_settings)  # type: ignore[reportCallIssue]
    if disable_reasoning:
        # Cerebras deprecated `extra_body['disable_reasoning']` on 2026-03-24 in favor of the standard
        # `reasoning_effort='none'`. https://inference-docs.cerebras.ai/resources/glm-47-migration
        openai_settings['openai_reasoning_effort'] = 'none'
    return openai_settings
