from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import BinaryImage, ImageUrl, UploadedFile
from pydantic_ai.models import download_item
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.usage import RequestUsage

from ._openai_geometry import (
    resolve_openai_aspect_ratio,
    resolve_openai_compatibility_size,
    resolve_openai_dimensions,
    size_matches_aspect_ratio,
)
from .base import ImageGenerationInput, ImageGenerationModel
from .result import GeneratedImage, ImageGenerationResult
from .settings import ImageGenerationSettings, warn_image_generation_settings

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

    openai_input_fidelity: Literal['high', 'low']
    """OpenAI input fidelity setting for image editing."""

    openai_moderation: Literal['low', 'auto']
    """OpenAI moderation strictness for image generation."""

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
        openai_settings = cast(OpenAIImageGenerationSettings, settings)
        resolved = _resolve_openai_settings(openai_settings, is_edit=bool(images), model_name=self.model_name)
        warn_image_generation_settings(self.system, ignored=resolved.ignored, conflicts=resolved.conflicts)

        try:
            if images:
                response = await self._client.images.edit(
                    image=await self._map_input_images(images),
                    prompt=prompt,
                    model=self.model_name,
                    n=openai_settings.get('n') or OMIT,
                    size=resolved.size or OMIT,
                    output_format=openai_settings.get('output_format') or OMIT,
                    quality=resolved.quality or OMIT,
                    background=resolved.background or OMIT,
                    input_fidelity=resolved.input_fidelity or OMIT,
                    output_compression=(
                        resolved.output_compression if resolved.output_compression is not None else OMIT
                    ),
                    user=openai_settings.get('openai_user') or OMIT,
                    extra_headers=openai_settings.get('extra_headers'),
                    extra_body=openai_settings.get('extra_body'),
                )
            else:
                response = await self._client.images.generate(
                    prompt=prompt,
                    model=self.model_name,
                    n=openai_settings.get('n') or OMIT,
                    size=resolved.size or OMIT,
                    output_format=openai_settings.get('output_format') or OMIT,
                    quality=resolved.quality or OMIT,
                    background=resolved.background or OMIT,
                    moderation=resolved.moderation or OMIT,
                    output_compression=(
                        resolved.output_compression if resolved.output_compression is not None else OMIT
                    ),
                    user=openai_settings.get('openai_user') or OMIT,
                    extra_headers=openai_settings.get('extra_headers'),
                    extra_body=openai_settings.get('extra_body'),
                )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover
        except APIConnectionError as e:  # pragma: no cover
            raise ModelAPIError(model_name=self.model_name, message=e.message) from e

        return self._map_response(prompt, settings, response)

    async def _map_input_images(self, images: Sequence[ImageGenerationInput]) -> list[tuple[str, bytes, str]]:
        mapped_images: list[tuple[str, bytes, str]] = []
        for index, image in enumerate(images):
            if isinstance(image, UploadedFile):
                if image.provider_name != self.system:
                    raise UserError(
                        f'UploadedFile with `provider_name={image.provider_name!r}` cannot be used with '
                        f'{type(self).__name__}. Expected `provider_name` to be `{self.system!r}`.'
                    )
                raise UserError(
                    'OpenAI image editing requires file content and does not accept `UploadedFile.file_id`; '
                    'use `BinaryImage` or `ImageUrl` instead'
                )

            if isinstance(image, ImageUrl):
                downloaded_image = await download_item(image, data_format='bytes')
                data = downloaded_image['data']
                media_type = downloaded_image['data_type']
            else:
                data = image.data
                media_type = image.media_type

            extension = _openai_input_extension(media_type)
            mapped_images.append((f'image-{index}.{extension}', data, media_type))

        return mapped_images

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


def _openai_input_extension(media_type: str) -> str:
    if media_type == 'image/png':
        return 'png'
    if media_type == 'image/jpeg':
        return 'jpg'
    if media_type == 'image/webp':
        return 'webp'
    raise UserError(
        f'OpenAI image editing only supports PNG, JPEG, or WebP input images, got media type {media_type!r}'
    )


@dataclass
class _OpenAIResolvedSettings:
    size: str | None
    quality: Literal['low', 'medium', 'high', 'auto'] | None
    background: Literal['transparent', 'opaque', 'auto'] | None
    input_fidelity: Literal['high', 'low'] | None
    moderation: Literal['low', 'auto'] | None
    output_compression: int | None
    ignored: list[str]
    conflicts: list[str]


def _resolve_openai_settings(
    settings: OpenAIImageGenerationSettings, *, is_edit: bool, model_name: str
) -> _OpenAIResolvedSettings:
    ignored: list[str] = []
    conflicts: list[str] = []

    quality = settings.get('openai_quality')
    if quality is None:
        quality = settings.get('quality')
    elif (common_quality := settings.get('quality')) is not None and common_quality != quality:
        conflicts.append('quality')

    background = settings.get('openai_background')
    if background is None:
        background = settings.get('background')
    elif (common_background := settings.get('background')) is not None and common_background != background:
        conflicts.append('background')

    input_fidelity = settings.get('openai_input_fidelity')
    if input_fidelity is None:
        input_fidelity = settings.get('input_fidelity')
    elif (common_input_fidelity := settings.get('input_fidelity')) is not None and (
        common_input_fidelity != input_fidelity
    ):
        conflicts.append('input_fidelity')

    moderation = settings.get('openai_moderation')
    if moderation is None:
        moderation = settings.get('moderation')
    elif (common_moderation := settings.get('moderation')) is not None and common_moderation != moderation:
        conflicts.append('moderation')

    output_compression = settings.get('openai_output_compression')
    if output_compression is None:
        output_compression = settings.get('output_compression')
    elif (common_compression := settings.get('output_compression')) is not None and (
        common_compression != output_compression
    ):
        conflicts.append('output_compression')

    if is_edit and moderation is not None:
        ignored.append('moderation')
    elif not is_edit and input_fidelity is not None:
        ignored.append('input_fidelity')

    return _OpenAIResolvedSettings(
        size=_resolve_openai_size(settings, ignored, conflicts, model_name=model_name),
        quality=quality,
        background=background,
        input_fidelity=input_fidelity,
        moderation=moderation,
        output_compression=output_compression,
        ignored=ignored,
        conflicts=conflicts,
    )


def _resolve_openai_size(
    settings: OpenAIImageGenerationSettings,
    ignored: list[str],
    conflicts: list[str],
    *,
    model_name: str,
) -> str | None:
    provider_size = settings.get('openai_size')
    size = settings.get('size')
    dimensions = settings.get('dimensions')
    aspect_ratio = settings.get('aspect_ratio')
    resolved_dimensions = resolve_openai_dimensions(model_name, dimensions) if dimensions is not None else None

    if provider_size is not None:
        if size is not None and size != provider_size:
            conflicts.append('size')
        if resolved_dimensions is not None and resolved_dimensions != provider_size:
            conflicts.append('dimensions')
        if aspect_ratio is not None and not size_matches_aspect_ratio(provider_size, aspect_ratio):
            conflicts.append('aspect_ratio')
        return provider_size

    if resolved_dimensions is not None:
        return resolved_dimensions

    resolved_size: str | None = None
    if size is not None:
        resolved_size = resolve_openai_compatibility_size(model_name, size)
        if resolved_size is None:
            ignored.append('size')

    if aspect_ratio is not None:
        mapped_size = resolve_openai_aspect_ratio(model_name, aspect_ratio)
        if mapped_size is None:
            ignored.append('aspect_ratio')
        elif resolved_size in (None, 'auto'):
            resolved_size = mapped_size
        elif not size_matches_aspect_ratio(resolved_size, aspect_ratio):
            ignored.append('aspect_ratio')

    return resolved_size


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
