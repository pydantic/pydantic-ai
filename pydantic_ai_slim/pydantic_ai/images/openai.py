from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from pydantic_ai.exceptions import ContentFilterError, ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import BinaryImage, ImageUrl, UploadedFile
from pydantic_ai.models import check_allow_model_requests, download_item
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.usage import RequestUsage

from ._media_type import image_media_type_from_bytes
from ._openai_geometry import resolve_openai_geometry
from ._validation import validate_image_count, warn_image_generation_settings
from .base import ImageGenerationInput, ImageGenerationModel
from .result import GeneratedImage, ImageGenerationResult
from .settings import ImageGenerationSettings, ImageOutputFormat

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

_UNSUPPORTED_MODEL_NAMES = frozenset(('dall-e-2', 'dall-e-3'))
"""DALL·E models the OpenAI SDK's `ImageModel` literal admits but this adapter does not implement.

They diverge from the GPT Image contract in every dimension this adapter encodes: they default to
`response_format='url'` (this adapter requires base64 bytes), have their own size sets, cap `n` at 1
for `dall-e-3`, and use a `standard`/`hd` quality vocabulary. Rejecting them by name keeps the error
actionable while leaving unrecognized future models to fall through to the provider.
"""


class OpenAIImageGenerationSettings(ImageGenerationSettings, total=False):
    """Settings used for an OpenAI image generation request.

    All fields from [`ImageGenerationSettings`][pydantic_ai.images.ImageGenerationSettings]
    are supported, plus OpenAI-specific settings prefixed with `openai_`.
    """

    # ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    openai_n: int
    """The number of images to generate."""

    openai_output_format: ImageOutputFormat
    """The generated image format."""

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

    openai_moderation: Literal['auto', 'low']
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
        if model_name in _UNSUPPORTED_MODEL_NAMES:
            raise UserError(
                f'OpenAI image generation model {model_name!r} is not supported. '
                'Use a GPT Image model such as `gpt-image-2` or `gpt-image-1`.'
            )
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
        check_allow_model_requests()
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
                    n=openai_settings.get('openai_n') or OMIT,
                    size=resolved.size or OMIT,
                    output_format=openai_settings.get('openai_output_format') or OMIT,
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
                    n=openai_settings.get('openai_n') or OMIT,
                    size=resolved.size or OMIT,
                    output_format=openai_settings.get('openai_output_format') or OMIT,
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
                match e.body:
                    case {'error': {'code': 'moderation_blocked'}}:
                        raise ContentFilterError(
                            'OpenAI image generation was blocked for content moderation',
                            json.dumps(e.body),
                        ) from e
                    case _:
                        pass
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise  # pragma: lax no cover
        except APIConnectionError as e:
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

        images: list[GeneratedImage] = []
        for image in response_data:
            if not image.b64_json:
                raise UnexpectedModelBehavior(
                    'OpenAI image generation response did not contain base64 image data',
                    response.model_dump_json(exclude_none=True),
                )
            try:
                image_data = base64.b64decode(image.b64_json, validate=True)
            except binascii.Error as e:
                raise UnexpectedModelBehavior(
                    'OpenAI image generation response did not contain valid base64 image data',
                    response.model_dump_json(exclude_none=True),
                ) from e

            # OpenAI echoes the requested `output_format` even when it returns bytes in a different
            # format (openai-node#1850), so trust the actual bytes rather than attach an unverified
            # media type to arbitrary data.
            if (sniffed_media_type := image_media_type_from_bytes(image_data)) is None:
                raise UnexpectedModelBehavior(
                    'OpenAI image generation response did not contain a recognized image format',
                    response.model_dump_json(exclude_none=True),
                )
            media_type = sniffed_media_type
            output_format = sniffed_media_type.removeprefix('image/')

            images.append(
                GeneratedImage(
                    content=BinaryImage(data=image_data, media_type=media_type),
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
    moderation: Literal['auto', 'low'] | None
    output_compression: int | None
    ignored: list[str]
    conflicts: list[str]


def _resolve_openai_settings(
    settings: OpenAIImageGenerationSettings, *, is_edit: bool, model_name: str
) -> _OpenAIResolvedSettings:
    ignored: list[str] = []
    conflicts: list[str] = []

    validate_image_count('OpenAI', settings.get('openai_n'), maximum=10)
    quality = settings.get('openai_quality')
    background = settings.get('openai_background')
    input_fidelity = settings.get('openai_input_fidelity')
    moderation = settings.get('openai_moderation')
    output_compression = settings.get('openai_output_compression')

    if is_edit:
        if moderation is not None:
            ignored.append('moderation')
    elif input_fidelity is not None:
        ignored.append('input_fidelity')

    geometry = resolve_openai_geometry(model_name, settings, provider_size=settings.get('openai_size'))

    return _OpenAIResolvedSettings(
        size=geometry.size,
        quality=quality,
        background=background,
        input_fidelity=input_fidelity,
        moderation=moderation,
        output_compression=output_compression,
        ignored=ignored + geometry.ignored,
        conflicts=conflicts + geometry.conflicts,
    )


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
