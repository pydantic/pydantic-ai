"""Z.AI (Zhipu AI) model implementation using OpenAI-compatible API."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from typing_extensions import override

from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..settings import ModelSettings
from . import ModelRequestParameters

try:
    from openai import AsyncOpenAI

    from .openai import OpenAIChatModel, OpenAIChatModelSettings
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Z.AI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = ('ZaiModel', 'ZaiModelName', 'ZaiModelSettings')

LatestZaiModelNames = Literal[
    'glm-4.7',
    'glm-4.6',
    'glm-4.6v',
    'glm-4.5-air',
    'glm-4.5-air-250723',
]

ZaiModelName = str | LatestZaiModelNames


class ZaiModelSettings(ModelSettings, total=False):
    """Settings used for a Z.AI model request.

    ALL FIELDS MUST BE `zai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    zai_thinking: bool
    """Enable thinking/reasoning mode for the model.

    When enabled, the model will produce reasoning content before the final response.
    Supported on `glm-4.7`.

    See [the Z.AI docs](https://docs.z.ai/guides/capabilities/thinking-mode) for more details.
    """

    zai_clear_thinking: bool
    """Whether to clear thinking content between turns.

    Set to `False` for preserved thinking, which retains reasoning content from prior
    assistant responses for improved multi-turn coherence. Defaults to `True` when not set.

    When using preserved thinking, you must return the complete, unmodified reasoning_content
    back to the API. All consecutive reasoning_content blocks must exactly match the original sequence.

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
        new_settings = _zai_settings_to_openai_settings(cast(ZaiModelSettings, merged_settings or {}))
        return new_settings, customized_parameters


def _zai_settings_to_openai_settings(model_settings: ZaiModelSettings) -> OpenAIChatModelSettings:
    """Transforms a 'ZaiModelSettings' object into an 'OpenAIChatModelSettings' object.

    Converts Z.AI-specific thinking settings into the `extra_body` format expected
    by the Z.AI API's OpenAI-compatible endpoint.

    Args:
        model_settings: The 'ZaiModelSettings' object to transform.

    Returns:
        An 'OpenAIChatModelSettings' object with equivalent settings.
    """
    extra_body = cast(dict[str, Any], model_settings.get('extra_body', {}))

    thinking_enabled = model_settings.pop('zai_thinking', None)
    clear_thinking = model_settings.pop('zai_clear_thinking', None)

    if thinking_enabled is not None:
        thinking: dict[str, Any] = {
            'type': 'enabled' if thinking_enabled else 'disabled',
        }
        if clear_thinking is not None:
            thinking['clear_thinking'] = clear_thinking
        extra_body['thinking'] = thinking

    if extra_body:
        model_settings['extra_body'] = extra_body

    return OpenAIChatModelSettings(**model_settings)  # type: ignore[reportCallIssue]
