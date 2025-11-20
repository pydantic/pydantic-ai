"""Cerebras model implementation using OpenAI-compatible API."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Literal

try:
    from cerebras.cloud.sdk import AsyncCerebras  # noqa: F401
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `cerebras-cloud-sdk` package to use the Cerebras model, '
        'you can use the `cerebras` optional group — `pip install "pydantic-ai-slim[cerebras]"`'
    ) from _import_error

from ..profiles import ModelProfile, ModelProfileSpec
from ..profiles.harmony import harmony_model_profile
from ..profiles.meta import meta_model_profile
from ..profiles.qwen import qwen_model_profile
from ..providers import Provider
from ..settings import ModelSettings
from .openai import OpenAIChatModel, OpenAIModelProfile  # type: ignore[attr-defined]

__all__ = ('CerebrasModel', 'CerebrasModelName')

CerebrasModelName = Literal[
    'gpt-oss-120b',
    'llama-3.3-70b',
    'llama3.1-8b',
    'qwen-3-235b-a22b-instruct-2507',
    'qwen-3-32b',
    'zai-glm-4.6',
]


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
        provider: Literal['cerebras'] | Provider[Any] = 'cerebras',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Cerebras model.

        Args:
            model_name: The name of the Cerebras model to use.
            provider: The provider to use. Can be 'cerebras' or a Provider instance.
            profile: The model profile to use. Defaults to a profile based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        if provider == 'cerebras':
            from ..providers.cerebras import CerebrasProvider

            # Extract api_key from settings if provided
            api_key = settings.get('api_key') if settings else None
            provider = CerebrasProvider(api_key=api_key) if api_key else CerebrasProvider()  # type: ignore[call-overload]

        # Use our custom model_profile method if no profile is provided
        if profile is None:
            profile = self._cerebras_model_profile

        super().__init__(model_name, provider=provider, profile=profile, settings=settings)  # type: ignore[arg-type]

    def _cerebras_model_profile(self, model_name: str) -> ModelProfile:
        """Get the model profile for this Cerebras model.

        Returns a profile with web search disabled since Cerebras doesn't support it.
        """
        model_name_lower = model_name.lower()

        # Get base profile based on model family
        if model_name_lower.startswith('llama'):
            base_profile = meta_model_profile(model_name)
        elif model_name_lower.startswith('qwen'):
            base_profile = qwen_model_profile(model_name)
        elif model_name_lower.startswith('gpt-oss'):
            base_profile = harmony_model_profile(model_name)
        else:
            # Default profile for unknown models
            base_profile = ModelProfile()

        # Wrap in OpenAIModelProfile with web search disabled
        return OpenAIModelProfile(
            openai_chat_supports_web_search=False,
        ).update(base_profile)

    async def _completions_create(
        self,
        messages: list[Any],
        stream: bool,
        model_settings: dict[str, Any],
        model_request_parameters: Any,
    ) -> Any:
        """Override to remove web_search_options parameter and convert Cerebras response to OpenAI format."""
        from openai._types import NOT_GIVEN
        from openai.types.chat import ChatCompletion

        # Get the original client method
        original_create = self.client.chat.completions.create

        # Create a wrapper that removes web_search_options and filters OMIT values
        async def create_without_web_search(**kwargs):
            # Remove web_search_options if present
            kwargs.pop('web_search_options', None)

            # Remove all keys with OMIT or NOT_GIVEN values
            keys_to_remove = []
            for key, value in kwargs.items():
                # Check if it's OMIT by checking the type name
                if hasattr(value, '__class__') and value.__class__.__name__ == 'Omit':
                    keys_to_remove.append(key)
                elif value is NOT_GIVEN:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del kwargs[key]

            # Call Cerebras SDK
            cerebras_response = await original_create(**kwargs)

            # Convert Cerebras response to OpenAI ChatCompletion
            # The Cerebras SDK returns a compatible structure, we just need to convert the type
            response_dict = (
                cerebras_response.model_dump() if hasattr(cerebras_response, 'model_dump') else cerebras_response
            )
            return ChatCompletion.model_validate(response_dict)

        # Temporarily replace the method
        self.client.chat.completions.create = create_without_web_search  # type: ignore

        try:
            # Call the parent implementation
            return await super()._completions_create(messages, stream, model_settings, model_request_parameters)  # type: ignore
        finally:
            # Restore the original method
            self.client.chat.completions.create = original_create  # type: ignore
