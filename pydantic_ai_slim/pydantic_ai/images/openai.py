from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import BinaryImage
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.usage import RequestUsage

from .base import ImageGenerationInput, ImageGenerationModel
from .result import GeneratedImage, ImageGenerationResult
from .settings import ImageGenerationSettings

try:
    from openai import APIConnectionError, APIStatusError, AsyncOpenAI
    from openai.types.image_model import ImageModel as LatestOpenAIImageModelNames
    from openai.types.images_response import ImagesResponse, Usage

    from pydantic_ai.models.openai import OMIT
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI image generation model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

OpenAIImageGenerationModelName = str | LatestOpenAIImageModelNames
"""Possible OpenAI image generation model names."""


class OpenAIImageGenerationSettings(ImageGenerationSettings, total=False):
    """Settings used for an OpenAI image generation request.

    All fields from [`ImageGenerationSettings`][pydantic_ai.images.ImageGenerationSettings]
    are supported, plus OpenAI-specific settings prefixed with `openai_`.
    """

    # ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    openai_size: str
    """OpenAI image size setting.

    This is provider-specific because OpenAI, Gemini, xAI, and other image APIs use
    different concepts for pixel sizes, aspect ratios, and resolution tiers.
    """

    openai_quality: Literal['low', 'medium', 'high', 'auto']
    """GPT Image quality setting."""

    openai_background: Literal['transparent', 'opaque', 'auto']
    """OpenAI image background setting."""

    openai_moderation: Literal['low', 'auto']
    """OpenAI moderation strictness."""

    openai_output_compression: int
    """OpenAI output compression setting."""

    openai_user: str
    """OpenAI end-user identifier."""


@dataclass(init=False)
class OpenAIImageGenerationModel(ImageGenerationModel):
    """OpenAI image generation model implementation."""

    _model_name: OpenAIImageGenerationModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: OpenAIImageGenerationModelName,
        *,
        provider: Literal['openai'] | Provider[AsyncOpenAI] = 'openai',
        settings: ImageGenerationSettings | None = None,
    ):
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider

        super().__init__(settings=settings)

    @property
    def _client(self) -> AsyncOpenAI:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def model_name(self) -> OpenAIImageGenerationModelName:
        """The image generation model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The image generation model provider."""
        return self._provider.name

    async def generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageGenerationInput] | None = None,
        settings: ImageGenerationSettings | None = None,
    ) -> ImageGenerationResult:
        prompt, images, settings = self.prepare_generate(prompt, images=images, settings=settings)
        if images:
            raise UserError('Reference images are not supported by the OpenAI image generation adapter')

        settings = cast(OpenAIImageGenerationSettings, settings)
        output_compression = settings.get('openai_output_compression', OMIT)

        try:
            response = await self._client.images.generate(
                prompt=prompt,
                model=self.model_name,
                n=settings.get('n') or OMIT,
                # The OpenAI SDK type can lag behind newer GPT Image size constraints.
                size=cast(Any, settings.get('openai_size')) or OMIT,
                output_format=settings.get('output_format') or OMIT,
                quality=settings.get('openai_quality') or OMIT,
                background=settings.get('openai_background') or OMIT,
                moderation=settings.get('openai_moderation') or OMIT,
                output_compression=output_compression,
                user=settings.get('openai_user') or OMIT,
                extra_headers=settings.get('extra_headers'),
                extra_body=settings.get('extra_body'),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover
        except APIConnectionError as e:  # pragma: no cover
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

        return self._map_response(prompt, settings, response)

    def _map_response(
        self, prompt: str, settings: ImageGenerationSettings, response: ImagesResponse
    ) -> ImageGenerationResult:
        response_data = response.data
        if not response_data:
            raise UnexpectedModelBehavior('OpenAI image generation response did not contain any images')

        output_format = response.output_format or settings.get('output_format') or 'png'
        media_type = _media_type_from_output_format(output_format)
        images: list[GeneratedImage] = []
        for image in response_data:
            if not image.b64_json:
                raise UnexpectedModelBehavior(
                    'OpenAI image generation response did not contain base64 image data',
                    response.model_dump_json(exclude_none=True),
                )
            images.append(
                GeneratedImage(
                    content=BinaryImage(data=base64.b64decode(image.b64_json), media_type=media_type),
                    revised_prompt=image.revised_prompt,
                    size=response.size,
                    quality=response.quality,
                    output_format=output_format,
                    background=response.background,
                )
            )

        return ImageGenerationResult(
            images=images,
            prompt=prompt,
            usage=_map_usage(response.usage, self.system, self.base_url, self.model_name),
            model_name=self.model_name,
            provider_name=self.system,
            provider_url=self.base_url,
            settings=settings,
            provider_details=_response_provider_details(response),
        )


def _media_type_from_output_format(output_format: str) -> str:
    if output_format == 'jpeg':
        return 'image/jpeg'
    return f'image/{output_format}'


def _response_provider_details(response: ImagesResponse) -> dict[str, object]:
    provider_details: dict[str, object] = {}
    if response.created:
        provider_details['created'] = response.created
    return provider_details


def _map_usage(
    usage: Usage | None,
    provider: str,
    provider_url: str,
    model: str,
) -> RequestUsage:
    if usage is None:
        return RequestUsage()

    details: dict[str, int] = {}
    usage_data = usage.model_dump(exclude_none=True)
    input_tokens_details = usage.input_tokens_details
    output_tokens_details = usage.output_tokens_details
    details['input_text_tokens'] = input_tokens_details.text_tokens
    details['input_image_tokens'] = input_tokens_details.image_tokens
    if output_tokens_details is not None:
        details['output_text_tokens'] = output_tokens_details.text_tokens
        details['output_image_tokens'] = output_tokens_details.image_tokens

    extracted_usage = RequestUsage.extract(
        {'model': model, 'usage': usage_data},
        provider=provider,
        provider_url=provider_url,
        provider_fallback='openai',
        api_flavor='images',
        details=details,
    )
    if extracted_usage.input_tokens or extracted_usage.output_tokens:
        return extracted_usage

    return RequestUsage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, details=details)
