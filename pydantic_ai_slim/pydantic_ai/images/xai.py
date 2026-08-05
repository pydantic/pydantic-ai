from __future__ import annotations

import base64
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, cast

from typing_extensions import assert_never

from pydantic_ai.exceptions import (
    ContentFilterError,
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import BinaryImage, ImageUrl, UploadedFile
from pydantic_ai.models import check_allow_model_requests, download_item
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.usage import RequestUsage

from ._validation import validate_image_count, warn_image_generation_settings
from .base import ImageGenerationInput, ImageGenerationModel
from .result import GeneratedImage, ImageGenerationResult
from .settings import ImageGenerationSettings

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

    from ._xai_geometry import resolve_xai_geometry
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

    xai_n: int
    """The number of images to generate."""

    xai_user: str
    """A unique identifier representing your end-user."""

    xai_aspect_ratio: ImageAspectRatio
    """The aspect ratio of the generated image."""

    xai_resolution: ImageResolution
    """The resolution tier of the generated image."""


@dataclass(init=False)
class XaiImageGenerationModel(ImageGenerationModel):
    """xAI image generation model implementation.

    This model works with the Grok Imagine models, such as `grok-imagine-image` and
    `grok-imagine-image-quality`, through the official xAI SDK, which connects over gRPC.

    xAI moderates silently: a flagged image in a batch comes back empty rather than as an error, so the
    clean images are returned and the flagged positions are reported through
    `provider_details['moderated_image_indices']`. A
    [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] is raised only when every image
    was flagged. See the [xAI model page](../models/xai.md#image-generation) for details.

    Example:
    ```python
    from pydantic_ai.images.xai import XaiImageGenerationModel
    from pydantic_ai.providers.xai import XaiProvider

    # Using xAI directly (requires XAI_API_KEY env var)
    model = XaiImageGenerationModel('grok-imagine-image')

    # Or with explicit provider configuration
    model = XaiImageGenerationModel(
        'grok-imagine-image',
        provider=XaiProvider(api_key='your-api-key'),
    )
    ```
    """

    _model_name: XaiImageGenerationModelName = field(repr=False)
    _provider: Provider[AsyncClient] = field(repr=False)

    def __init__(
        self,
        model_name: XaiImageGenerationModelName,
        *,
        provider: Literal['xai'] | Provider[AsyncClient] = 'xai',
        settings: ImageGenerationSettings | None = None,
    ):
        """Initialize an xAI image generation model.

        Args:
            model_name: The name of the Grok Imagine model to use.
                See [xAI's image generation documentation](https://docs.x.ai/developers/model-capabilities/images/generation)
                for available models.
            provider: The provider to use for authentication and API access. Can be:

                - `'xai'` (default): Uses the standard xAI API
                - An [`XaiProvider`][pydantic_ai.providers.xai.XaiProvider] instance for custom
                  configuration, such as a custom `api_host` or `xai_client`
            settings: Model-specific
                [`ImageGenerationSettings`][pydantic_ai.images.ImageGenerationSettings]
                to use as defaults for this model.
        """
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
        check_allow_model_requests()
        prompt, images, settings = self.prepare_generate(prompt, images=images, settings=settings)
        xai_settings = cast(XaiImageGenerationSettings, settings)
        resolved = _resolve_xai_settings(xai_settings, model_name=self.model_name)
        warn_image_generation_settings(self.system, ignored=resolved.ignored, conflicts=resolved.conflicts)
        image_url, image_file_id, image_urls, image_file_ids = await self._map_input_images(images)
        n = xai_settings.get('xai_n') or 1

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
        if len(images) > 3:
            raise UserError('xAI image editing accepts at most three reference images')

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
        moderated_indices: list[int] = []
        for index, response in enumerate(responses):
            # xAI moderation is silent: a flagged slot comes back with `respect_moderation=False` and an
            # empty payload, and reading its `.base64` raises client-side. Skip it so one flagged slot
            # doesn't discard the rest of a paid batch.
            if not response.respect_moderation:
                moderated_indices.append(index)
                continue
            try:
                content = _decode_data_url(response.base64)
            except (ValueError, TypeError) as e:
                raise UnexpectedModelBehavior(
                    'xAI image generation response did not contain valid base64 image data'
                ) from e
            images.append(
                GeneratedImage(
                    content=content,
                    output_format=content.media_type.removeprefix('image/'),
                    provider_details={'respect_moderation': response.respect_moderation},
                )
            )

        if not images:
            raise ContentFilterError('xAI flagged all generated images for content moderation')

        first_response = responses[0]
        provider_details = _response_provider_details(first_response)
        if moderated_indices:
            provider_details['moderated_image_indices'] = moderated_indices
        return ImageGenerationResult(
            images=images,
            prompt=prompt,
            usage=_map_usage(first_response.usage, self.system, self.base_url, self.model_name),
            model_name=first_response.model or self.model_name,
            provider_name=self.system,
            provider_url=self.base_url,
            settings=settings,
            provider_details=provider_details,
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
    validate_image_count('xAI', settings.get('xai_n'), maximum=10)
    geometry = resolve_xai_geometry(
        model_name,
        settings,
        provider_aspect_ratio=settings.get('xai_aspect_ratio'),
        provider_resolution=settings.get('xai_resolution'),
    )

    # xAI is reached over gRPC, which has no per-request body or header escape hatch, so these
    # portable settings cannot be honored here as they are on the HTTP-based providers.
    ignored = list(geometry.ignored)
    if settings.get('extra_headers'):
        ignored.append('extra_headers')
    if settings.get('extra_body'):
        ignored.append('extra_body')

    return _XaiResolvedSettings(
        aspect_ratio=geometry.aspect_ratio,
        resolution=geometry.resolution,
        ignored=ignored,
        conflicts=geometry.conflicts,
    )


def _map_usage(
    usage: usage_pb2.SamplingUsage,
    provider: str,
    provider_url: str,
    model: str,
) -> RequestUsage:
    details: dict[str, int] = {}
    for field_name, detail_name in (
        ('reasoning_tokens', 'reasoning_tokens'),
        ('prompt_text_tokens', 'input_text_tokens'),
        ('prompt_image_tokens', 'input_image_tokens'),
    ):
        if value := cast(int, getattr(usage, field_name)):
            details[detail_name] = value

    # `cached_prompt_text_tokens` is fed to `extract` rather than `details` so genai-prices maps it
    # onto the typed `cache_read_tokens` and prices it at the cached rate, matching `models/xai.py`.
    usage_data: dict[str, int] = {
        'prompt_tokens': usage.prompt_tokens,
        'completion_tokens': usage.completion_tokens,
    }
    if cached_tokens := usage.cached_prompt_text_tokens:
        usage_data['cached_prompt_text_tokens'] = cached_tokens

    extracted_usage = RequestUsage.extract(
        {'model': model, 'usage': usage_data},
        provider=provider,
        provider_url=provider_url,
        provider_fallback='x-ai',
        details=details,
    )
    if extracted_usage.input_tokens or extracted_usage.output_tokens:
        return extracted_usage

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
    grpc.StatusCode.INVALID_ARGUMENT: 400,
    grpc.StatusCode.UNAUTHENTICATED: 401,
    grpc.StatusCode.PERMISSION_DENIED: 403,
    grpc.StatusCode.NOT_FOUND: 404,
    grpc.StatusCode.RESOURCE_EXHAUSTED: 429,
    grpc.StatusCode.INTERNAL: 500,
    grpc.StatusCode.UNAVAILABLE: 503,
    grpc.StatusCode.DEADLINE_EXCEEDED: 504,
}
