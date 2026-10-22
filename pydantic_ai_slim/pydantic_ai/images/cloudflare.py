"""Native image generation and editing on Cloudflare Workers AI."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal, assert_never

import httpx2
from pydantic import TypeAdapter, ValidationError

from ..exceptions import ContentFilterError, ModelHTTPError, UnexpectedModelBehavior, UserError
from ..messages import BinaryImage, ImageUrl, UploadedFile
from ..models import check_allow_model_requests, download_item
from ..providers._cloudflare_http import send_request
from ..providers.cloudflare import CloudflareProvider
from ._media_type import image_media_type_from_bytes, output_format_from_media_type
from .base import ImageGenerationInput, ImageGenerationModel
from .result import GeneratedImage, ImageGenerationResult
from .settings import ImageGenerationSettings

__all__ = ('CloudflareImageGenerationModel', 'CloudflareImageGenerationModelName', 'CloudflareImageGenerationSettings')

CloudflareImageGenerationModelName = (
    Literal[
        '@cf/black-forest-labs/flux-1-schnell',
        '@cf/black-forest-labs/flux-2-dev',
        '@cf/black-forest-labs/flux-2-klein-4b',
        '@cf/black-forest-labs/flux-2-klein-9b',
        '@cf/leonardo/lucid-origin',
        '@cf/leonardo/phoenix-1.0',
        '@cf/stabilityai/stable-diffusion-xl-base-1.0',
    ]
    | str
)
"""Workers AI image model IDs. Unlisted models use the native JSON request format."""


class CloudflareImageGenerationSettings(ImageGenerationSettings, total=False):
    """Native image controls; availability and limits depend on the selected model.

    `dimensions` maps to width/height except on FLUX.1 Schnell, which has no size control.
    `aspect_ratio` maps to integral pixel dimensions with the longest edge at most 1024 pixels.
    FLUX.1 Schnell rejects either geometry control because it has no size parameter.
    """

    cloudflare_steps: int
    """Diffusion steps. FLUX models use `steps`; other models use `num_steps`."""

    cloudflare_seed: int
    """Random seed for reproducible generation."""

    cloudflare_guidance: float
    """How closely the generation follows the prompt."""

    cloudflare_negative_prompt: str
    """What to exclude from the generated image."""


@dataclass(init=False, eq=False)
class CloudflareImageGenerationModel(ImageGenerationModel):
    """An image model hosted on the native Workers AI run endpoint.

    FLUX.2 uses multipart requests and supports up to four reference images.
    Other JSON image models support text prompts.
    Image URLs are downloaded before being sent. Uploaded provider file IDs are unsupported.
    See [Cloudflare](../../models/cloudflare.md#image-generation) for model-specific limits.
    """

    _model_name: str = field(repr=False)
    _provider: CloudflareProvider = field(repr=False)

    def __init__(
        self,
        model_name: CloudflareImageGenerationModelName,
        *,
        provider: Literal['cloudflare'] | CloudflareProvider = 'cloudflare',
        settings: CloudflareImageGenerationSettings | None = None,
    ) -> None:
        """Initialize an image model.

        Args:
            model_name: The full Workers AI image model ID.
            provider: Account configuration and HTTP client.
            settings: Default generation settings.
        """
        self._model_name = model_name
        self._provider = CloudflareProvider() if isinstance(provider, str) else provider
        super().__init__(settings=settings)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return self._provider.name

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    async def generate(  # noqa: C901
        self,
        prompt: str,
        *,
        images: Sequence[ImageGenerationInput] | None = None,
        settings: ImageGenerationSettings | None = None,
    ) -> ImageGenerationResult:
        check_allow_model_requests()
        prompt, prepared_images, merged_settings = self.prepare_generate(prompt, images=images, settings=settings)
        native_settings = _settings_adapter.validate_python(merged_settings)
        multipart = self.model_name.startswith('@cf/black-forest-labs/flux-2-')
        limit = 4 if multipart else 0
        if len(prepared_images) > limit:
            raise UserError(f'Cloudflare model {self.model_name!r} accepts at most {limit} reference images.')

        body: dict[str, object] = {'prompt': prompt}
        dimensions = merged_settings.get('dimensions')
        aspect_ratio = merged_settings.get('aspect_ratio')
        if self.model_name == '@cf/black-forest-labs/flux-1-schnell' and (dimensions or aspect_ratio):
            raise UserError('Cloudflare FLUX.1 Schnell does not support dimensions or aspect_ratio.')
        if dimensions is None and aspect_ratio is not None:
            width, height = aspect_ratio.split(':')
            ratio = Fraction(width) / Fraction(height)
            unit = 1024 // max(ratio.numerator, ratio.denominator)
            dimensions = ratio.numerator * unit, ratio.denominator * unit
        if dimensions is not None:
            body.update(width=dimensions[0], height=dimensions[1])
        if (steps := native_settings.get('cloudflare_steps')) is not None:
            body['steps' if self.model_name.startswith('@cf/black-forest-labs/flux-') else 'num_steps'] = steps
        if (seed := native_settings.get('cloudflare_seed')) is not None:
            body['seed'] = seed
        if (guidance := native_settings.get('cloudflare_guidance')) is not None:
            body['guidance'] = guidance
        if (negative_prompt := native_settings.get('cloudflare_negative_prompt')) is not None:
            body['negative_prompt'] = negative_prompt
        if (extra_body := merged_settings.get('extra_body')) is not None:
            try:
                body.update(_body_adapter.validate_python(extra_body))
            except ValidationError as e:
                raise UserError(f'Invalid Cloudflare image `extra_body`: {e}') from e

        reference_images: list[BinaryImage] = []
        for image in prepared_images:
            if isinstance(image, BinaryImage):
                reference_images.append(image)
            elif isinstance(image, ImageUrl):
                downloaded = await download_item(image, data_format='bytes', type_format='mime')
                reference_images.append(BinaryImage(data=downloaded['data'], media_type=downloaded['data_type']))
            elif isinstance(image, UploadedFile):
                raise UserError('Cloudflare image generation requires image bytes or URLs, not uploaded file IDs.')
            else:
                assert_never(image)

        self._provider._refresh_http_client()  # pyright: ignore[reportPrivateUsage]
        client = self._provider.client
        headers = httpx2.Headers(merged_settings.get('extra_headers', {}))
        if 'Authorization' not in headers:
            headers['Authorization'] = f'Bearer {self._provider.api_key}'
        url = f'{self.base_url}/ai/run/{self.model_name}'
        if multipart:
            # Using file parts for scalar fields forces multipart even for a text-only request.
            parts: list[tuple[str, tuple[str | None, bytes, str | None]]] = []
            for name, value in body.items():
                if not isinstance(value, (str, int, float, bool)):
                    raise UserError(f'Cloudflare multipart field {name!r} must be a scalar value.')
                parts.append((name, (None, str(value).encode(), None)))
            for index, image in enumerate(reference_images):
                parts.append((f'input_image_{index}', (image.identifier, image.data, image.media_type)))
            request = client.build_request('POST', url, files=parts, headers=headers)
        else:
            request = client.build_request('POST', url, json=body, headers=headers)
        try:
            response = await send_request(client, request, model_name=self.model_name)
        except ModelHTTPError as e:
            if e.status_code == 400 and 'NSFW' in str(e.body):
                raise ContentFilterError(f'Cloudflare blocked image generation: {e.body}') from e
            raise
        if response.headers.get('content-type', '').startswith('image/'):
            data = response.content
        else:
            try:
                parsed = _response_adapter.validate_json(response.content)
                data = base64.b64decode(parsed.result.image, validate=True)
            except (ValidationError, binascii.Error) as e:
                raise UnexpectedModelBehavior(f'Invalid Cloudflare image response: {e}', response.text) from e
        media_type = image_media_type_from_bytes(data)
        if media_type is None:
            raise UnexpectedModelBehavior('Cloudflare returned unrecognized or empty image data.')
        return ImageGenerationResult(
            images=[
                GeneratedImage(
                    content=BinaryImage(data=data, media_type=media_type),
                    output_format=output_format_from_media_type(media_type),
                )
            ],
            prompt=prompt,
            model_name=self.model_name,
            provider_name=self.system,
            provider_response_id=response.headers.get('cf-ai-req-id'),
            provider_url=self.base_url,
        )


@dataclass
class _ImageResult:
    image: str


@dataclass
class _ImageResponse:
    result: _ImageResult


_response_adapter = TypeAdapter(_ImageResponse)
_body_adapter = TypeAdapter(dict[str, object])
_settings_adapter = TypeAdapter(CloudflareImageGenerationSettings)
