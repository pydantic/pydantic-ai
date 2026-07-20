from __future__ import annotations

import base64
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, cast

from typing_extensions import assert_never

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import BinaryImage, ImageUrl, UploadedFile
from pydantic_ai.models import download_item
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.usage import RequestUsage

from .base import ImageGenerationInput, ImageGenerationModel
from .result import GeneratedImage, ImageGenerationResult
from .settings import ImageGenerationSettings, warn_image_generation_settings

try:
    import grpc
    from xai_sdk import AsyncClient
    from xai_sdk.aio.image import ImageResponse
    from xai_sdk.proto import usage_pb2
    from xai_sdk.types import (
        ImageAspectRatio,
        ImageGenerationModel as LatestXaiImageGenerationModelNames,
        ImageResolution,
    )

    from ._xai_geometry import resolve_xai_aspect_ratio, resolve_xai_dimensions, resolve_xai_size
except ImportError as _import_error:
    raise ImportError(
        'Please install `xai-sdk` to use the xAI image generation model, '
        'you can use the `xai` optional group — `pip install "pydantic-ai-slim[xai]"`'
    ) from _import_error


XaiImageGenerationModelName = str | LatestXaiImageGenerationModelNames
"""Possible xAI image generation model names."""


class XaiImageGenerationSettings(ImageGenerationSettings, total=False):
    """Settings used for an xAI image generation request.

    All fields from [`ImageGenerationSettings`][pydantic_ai.images.ImageGenerationSettings]
    are supported on a best-effort basis, plus xAI-specific settings prefixed with `xai_`.
    """

    # ALL FIELDS MUST BE `xai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    xai_user: str
    """A unique identifier representing your end-user."""

    xai_aspect_ratio: ImageAspectRatio
    """The aspect ratio of the generated image."""

    xai_resolution: ImageResolution
    """The resolution tier of the generated image."""


@dataclass(init=False)
class XaiImageGenerationModel(ImageGenerationModel):
    """xAI image generation model implementation."""

    _model_name: XaiImageGenerationModelName = field(repr=False)
    _provider: Provider[AsyncClient] = field(repr=False)

    def __init__(
        self,
        model_name: XaiImageGenerationModelName,
        *,
        provider: Literal['xai'] | Provider[AsyncClient] = 'xai',
        settings: ImageGenerationSettings | None = None,
    ):
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider

        super().__init__(settings=settings)

    @property
    def _client(self) -> AsyncClient:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> XaiImageGenerationModelName:
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
        xai_settings = cast(XaiImageGenerationSettings, settings)
        resolved = _resolve_xai_settings(xai_settings, model_name=self.model_name)
        warn_image_generation_settings(self.system, ignored=resolved.ignored, conflicts=resolved.conflicts)
        image_url, image_file_id, image_urls, image_file_ids = await self._map_input_images(images)
        n = xai_settings.get('n') or 1

        with _map_api_errors(self.model_name):
            if n == 1:
                response = await self._client.image.sample(
                    prompt,
                    self.model_name,
                    image_url=image_url,
                    image_file_id=image_file_id,
                    image_urls=image_urls,
                    image_file_ids=image_file_ids,
                    user=xai_settings.get('xai_user'),
                    image_format='base64',
                    aspect_ratio=resolved.aspect_ratio,
                    resolution=resolved.resolution,
                )
                responses = [response]
            else:
                responses = list(
                    await self._client.image.sample_batch(
                        prompt,
                        self.model_name,
                        n,
                        image_url=image_url,
                        image_file_id=image_file_id,
                        image_urls=image_urls,
                        image_file_ids=image_file_ids,
                        user=xai_settings.get('xai_user'),
                        image_format='base64',
                        aspect_ratio=resolved.aspect_ratio,
                        resolution=resolved.resolution,
                    )
                )

        return self._map_response(prompt, settings, responses)

    async def _map_input_images(
        self, images: Sequence[ImageGenerationInput]
    ) -> tuple[str | None, str | None, list[str] | None, list[str] | None]:
        image_references: list[str] = []
        file_ids: list[str] = []
        input_kinds: list[Literal['reference', 'file_id']] = []

        for image in images:
            if isinstance(image, UploadedFile):
                if image.provider_name != self.system:
                    raise UserError(
                        f'UploadedFile with `provider_name={image.provider_name!r}` cannot be used with '
                        f'{type(self).__name__}. Expected `provider_name` to be `{self.system!r}`.'
                    )
                file_ids.append(image.file_id)
                input_kinds.append('file_id')
            elif isinstance(image, BinaryImage):
                image_references.append(_binary_image_data_url(image.data, image.media_type))
                input_kinds.append('reference')
            elif isinstance(image, ImageUrl):
                if image.force_download:
                    downloaded_image = await download_item(image, data_format='bytes')
                    image_references.append(
                        _binary_image_data_url(downloaded_image['data'], downloaded_image['data_type'])
                    )
                else:
                    image_references.append(image.url)
                input_kinds.append('reference')
            else:
                assert_never(image)

        if len(input_kinds) == 1:
            if input_kinds[0] == 'file_id':
                return None, file_ids[0], None, None
            return image_references[0], None, None, None

        if 'file_id' in input_kinds and 'reference' in input_kinds:
            provider_order = sorted(input_kinds, key=lambda kind: kind == 'reference')
            if input_kinds != provider_order:
                raise UserError(
                    'xAI sends file-ID image inputs before URL or binary inputs. '
                    'Place all `UploadedFile` inputs first to preserve reference-image order.'
                )

        return None, None, image_references or None, file_ids or None

    def _map_response(
        self,
        prompt: str,
        settings: ImageGenerationSettings,
        responses: Sequence[ImageResponse],
    ) -> ImageGenerationResult:
        if not responses:
            raise UnexpectedModelBehavior('xAI image generation response did not contain any images')

        images: list[GeneratedImage] = []
        for response in responses:
            try:
                content = _decode_data_url(response.base64)
            except (ValueError, TypeError) as e:
                raise UnexpectedModelBehavior(
                    'xAI image generation response did not contain valid base64 image data',
                    str({'respect_moderation': response.respect_moderation}),
                ) from e
            images.append(
                GeneratedImage(
                    content=content,
                    output_format=content.media_type.removeprefix('image/'),
                    provider_details={'respect_moderation': response.respect_moderation},
                )
            )

        first_response = responses[0]
        return ImageGenerationResult(
            images=images,
            prompt=prompt,
            usage=_map_usage(first_response.usage),
            model_name=first_response.model or self.model_name,
            provider_name=self.system,
            provider_url=self.base_url,
            settings=settings,
            provider_details=_response_provider_details(first_response),
        )


def _binary_image_data_url(data: bytes, media_type: str) -> str:
    encoded = base64.b64encode(data).decode()
    return f'data:{media_type};base64,{encoded}'


def _decode_data_url(value: str) -> BinaryImage:
    header, encoded = value.split(',', maxsplit=1)
    if not header.startswith('data:image/') or not header.endswith(';base64'):
        raise ValueError('Not a base64 image data URL')
    media_type = header.removeprefix('data:').removesuffix(';base64')
    return BinaryImage(data=base64.b64decode(encoded, validate=True), media_type=media_type)


@dataclass
class _XaiResolvedSettings:
    aspect_ratio: ImageAspectRatio | None
    resolution: ImageResolution | None
    ignored: list[str]
    conflicts: list[str]


def _resolve_xai_settings(
    settings: XaiImageGenerationSettings, *, model_name: XaiImageGenerationModelName
) -> _XaiResolvedSettings:
    ignored = [
        name
        for name in ('background', 'input_fidelity', 'moderation', 'output_compression', 'output_format', 'quality')
        if name in settings
    ]
    conflicts: list[str] = []

    if dimensions := settings.get('dimensions'):
        mapped_aspect_ratio, mapped_resolution = resolve_xai_dimensions(model_name, dimensions)
        aspect_ratio = settings.get('xai_aspect_ratio')
        if aspect_ratio is None:
            aspect_ratio = mapped_aspect_ratio
        elif aspect_ratio != mapped_aspect_ratio:
            conflicts.append('dimensions')
        resolution = settings.get('xai_resolution')
        if resolution is None:
            resolution = mapped_resolution
        elif resolution != mapped_resolution:
            conflicts.append('dimensions')
        return _XaiResolvedSettings(
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            ignored=ignored,
            conflicts=conflicts,
        )

    return _XaiResolvedSettings(
        aspect_ratio=_resolve_xai_aspect_ratio(settings, ignored, conflicts),
        resolution=_resolve_xai_resolution(settings, ignored, conflicts),
        ignored=ignored,
        conflicts=conflicts,
    )


def _resolve_xai_aspect_ratio(
    settings: XaiImageGenerationSettings,
    ignored: list[str],
    conflicts: list[str],
) -> ImageAspectRatio | None:
    provider_aspect_ratio = settings.get('xai_aspect_ratio')
    common_aspect_ratio = settings.get('aspect_ratio')
    mapped_aspect_ratio = resolve_xai_aspect_ratio(common_aspect_ratio) if common_aspect_ratio else None
    if common_aspect_ratio is not None and mapped_aspect_ratio is None:
        ignored.append('aspect_ratio')
    if provider_aspect_ratio is not None:
        if mapped_aspect_ratio is not None and mapped_aspect_ratio != provider_aspect_ratio:
            conflicts.append('aspect_ratio')
        return provider_aspect_ratio
    return mapped_aspect_ratio


def _resolve_xai_resolution(
    settings: XaiImageGenerationSettings,
    ignored: list[str],
    conflicts: list[str],
) -> ImageResolution | None:
    provider_resolution = settings.get('xai_resolution')
    common_size = settings.get('size')
    mapped_resolution = resolve_xai_size(common_size) if common_size else None
    if common_size is not None and mapped_resolution is None:
        ignored.append('size')
    if provider_resolution is not None:
        if mapped_resolution is not None and mapped_resolution != provider_resolution:
            conflicts.append('size')
        return provider_resolution
    if mapped_resolution is not None:
        return mapped_resolution

    # A common ratio promises one canonical model geometry. Pin xAI's documented default tier
    # instead of relying on a provider default that could change independently.
    common_aspect_ratio = settings.get('aspect_ratio')
    if common_aspect_ratio is not None and resolve_xai_aspect_ratio(common_aspect_ratio) is not None:
        return '1k'
    return None


def _map_usage(usage: usage_pb2.SamplingUsage) -> RequestUsage:
    details: dict[str, int] = {}
    for field_name, detail_name in (
        ('reasoning_tokens', 'reasoning_tokens'),
        ('cached_prompt_text_tokens', 'cached_prompt_text_tokens'),
        ('prompt_text_tokens', 'input_text_tokens'),
        ('prompt_image_tokens', 'input_image_tokens'),
    ):
        if value := cast(int, getattr(usage, field_name)):
            details[detail_name] = value

    return RequestUsage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        details=details,
    )


def _response_provider_details(response: ImageResponse) -> dict[str, object]:
    provider_details: dict[str, object] = {}
    usage = response.usage
    if usage.HasField('cost_in_usd_ticks'):
        provider_details['cost_in_usd_ticks'] = usage.cost_in_usd_ticks
    if (cost_usd := response.cost_usd) is not None:
        provider_details['cost_usd'] = cost_usd
    return provider_details


@contextmanager
def _map_api_errors(model_name: str) -> Generator[None]:
    try:
        yield
    except grpc.RpcError as e:
        status_code = _GRPC_STATUS_TO_HTTP.get(e.code())
        details = e.details() or str(e)
        if status_code is not None:
            raise ModelHTTPError(status_code=status_code, model_name=model_name, body=details) from e
        raise ModelAPIError(model_name=model_name, message=details) from e


_GRPC_STATUS_TO_HTTP: dict[grpc.StatusCode, int] = {
    grpc.StatusCode.UNAUTHENTICATED: 401,
    grpc.StatusCode.PERMISSION_DENIED: 403,
    grpc.StatusCode.NOT_FOUND: 404,
    grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
    grpc.StatusCode.INTERNAL: 500,
    grpc.StatusCode.UNAVAILABLE: 503,
    grpc.StatusCode.DEADLINE_EXCEEDED: 504,
}
