from __future__ import annotations

from collections.abc import Sequence
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
from .settings import ImageGenerationSettings

try:
    from google.genai import Client, errors
    from google.genai.types import (
        BlobDict,
        ContentDict,
        ContentUnionDict,
        FileDataDict,
        GenerateContentConfigDict,
        GenerateContentResponse,
        HttpOptionsDict,
        ImageConfigDict,
        PartDict,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `google-genai` to use the Google image generation model, '
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error


GoogleImageGenerationModelName = str
"""Possible Google image generation model names."""


class GoogleImageGenerationSettings(ImageGenerationSettings, total=False):
    """Settings used for a Google image generation request.

    All fields from [`ImageGenerationSettings`][pydantic_ai.images.ImageGenerationSettings]
    are supported on a best-effort basis, plus Google-specific settings prefixed with `google_`.
    """

    # ALL FIELDS MUST BE `google_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    google_image_config: ImageConfigDict
    """Google image generation configuration, including aspect ratio and image size."""


@dataclass(init=False)
class GoogleImageGenerationModel(ImageGenerationModel):
    """Google Gemini image generation model implementation."""

    _model_name: GoogleImageGenerationModelName = field(repr=False)
    _provider: Provider[Client] = field(repr=False)

    def __init__(
        self,
        model_name: GoogleImageGenerationModelName,
        *,
        provider: Literal['google'] | Provider[Client] = 'google',
        settings: ImageGenerationSettings | None = None,
    ):
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider

        super().__init__(settings=settings)

    @property
    def _client(self) -> Client:
        return self._provider.client

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> GoogleImageGenerationModelName:
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
        google_settings = cast(GoogleImageGenerationSettings, settings)
        contents = await self._map_contents(prompt, images)
        config = self._map_config(google_settings)

        try:
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
        except errors.APIError as e:
            if (status_code := e.code) >= 400:
                body = cast(object, e.details)  # pyright: ignore[reportUnknownMemberType]
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=body) from e
            raise ModelAPIError(model_name=self.model_name, message=str(e)) from e

        return self._map_response(prompt, settings, response)

    async def _map_contents(self, prompt: str, images: Sequence[ImageGenerationInput]) -> list[ContentUnionDict]:
        parts: list[PartDict] = [{'text': prompt}]
        for image in images:
            parts.append(await self._map_input_image(image))
        return [ContentDict(role='user', parts=parts)]

    async def _map_input_image(self, image: ImageGenerationInput) -> PartDict:
        if isinstance(image, BinaryImage):
            part = PartDict(inline_data=BlobDict(data=image.data, mime_type=image.media_type))
        elif isinstance(image, UploadedFile):
            if image.provider_name != self.system:
                raise UserError(
                    f'UploadedFile with `provider_name={image.provider_name!r}` cannot be used with '
                    f'{type(self).__name__}. Expected `provider_name` to be `{self.system!r}`.'
                )
            if not image.file_id.startswith('https://'):
                raise UserError(
                    'Google image generation requires `UploadedFile.file_id` to be a Google Files API URI '
                    'starting with `https://`'
                )
            part = PartDict(file_data=FileDataDict(file_uri=image.file_id, mime_type=image.media_type))
        elif isinstance(image, ImageUrl):
            if not image.force_download and image.url.startswith(
                'https://generativelanguage.googleapis.com/v1beta/files'
            ):
                part = PartDict(file_data=FileDataDict(file_uri=image.url, mime_type=image.media_type))
            else:
                downloaded_image = await download_item(image, data_format='bytes')
                part = PartDict(
                    inline_data=BlobDict(
                        data=downloaded_image['data'],
                        mime_type=downloaded_image['data_type'],
                    )
                )
        else:
            assert_never(image)

        if image.vendor_metadata and (media_resolution := image.vendor_metadata.get('media_resolution')) is not None:
            part['media_resolution'] = media_resolution
        return part

    def _map_config(self, settings: GoogleImageGenerationSettings) -> GenerateContentConfigDict:
        image_config = settings.get('google_image_config')
        http_options = None
        if extra_headers := settings.get('extra_headers'):
            http_options = HttpOptionsDict(headers=dict(extra_headers))

        return GenerateContentConfigDict(
            response_modalities=['TEXT', 'IMAGE'],
            image_config=ImageConfigDict(**image_config) if image_config is not None else None,
            http_options=http_options,
        )

    def _map_response(
        self,
        prompt: str,
        settings: ImageGenerationSettings,
        response: GenerateContentResponse,
    ) -> ImageGenerationResult:
        images: list[GeneratedImage] = []
        for candidate in response.candidates or []:
            if candidate.content is None:
                continue
            for part in candidate.content.parts or []:
                if part.thought or part.inline_data is None or part.inline_data.data is None:
                    continue
                media_type = part.inline_data.mime_type or 'image/png'
                provider_details: dict[str, object] | None = (
                    {'has_thought_signature': True} if part.thought_signature else None
                )
                images.append(
                    GeneratedImage(
                        content=BinaryImage(data=part.inline_data.data, media_type=media_type),
                        output_format=_output_format_from_media_type(media_type),
                        provider_details=provider_details,
                    )
                )

        provider_details = _response_provider_details(response)
        if not images:
            raise UnexpectedModelBehavior(
                'Google image generation response did not contain any images',
                str(provider_details) if provider_details else None,
            )

        return ImageGenerationResult(
            images=images,
            prompt=prompt,
            usage=_map_usage(response, self.system, self.base_url),
            model_name=response.model_version or self.model_name,
            provider_name=self.system,
            provider_url=self.base_url,
            settings=settings,
            provider_details=provider_details,
            provider_response_id=response.response_id,
        )


def _output_format_from_media_type(media_type: str) -> str | None:
    if media_type.startswith('image/'):
        return media_type.removeprefix('image/')
    return None


def _response_provider_details(response: GenerateContentResponse) -> dict[str, object]:
    provider_details: dict[str, object] = {}
    candidate = response.candidates[0] if response.candidates else None

    if candidate and candidate.finish_reason:
        provider_details['finish_reason'] = candidate.finish_reason.value
    if candidate and candidate.safety_ratings:
        provider_details['safety_ratings'] = [rating.model_dump(by_alias=True) for rating in candidate.safety_ratings]
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        provider_details['block_reason'] = response.prompt_feedback.block_reason.value
        if response.prompt_feedback.block_reason_message:
            provider_details['block_reason_message'] = response.prompt_feedback.block_reason_message
        if response.prompt_feedback.safety_ratings:
            provider_details['safety_ratings'] = [
                rating.model_dump(by_alias=True) for rating in response.prompt_feedback.safety_ratings
            ]
    if response.create_time is not None:
        provider_details['timestamp'] = response.create_time
    if response.usage_metadata and response.usage_metadata.traffic_type:
        provider_details['traffic_type'] = response.usage_metadata.traffic_type.value

    return provider_details


def _map_usage(response: GenerateContentResponse, provider: str, provider_url: str) -> RequestUsage:
    metadata = response.usage_metadata
    if metadata is None:
        return RequestUsage()

    details: dict[str, int] = {}
    if metadata.cached_content_token_count:
        details['cached_content_tokens'] = metadata.cached_content_token_count
    if metadata.thoughts_token_count:
        details['thoughts_tokens'] = metadata.thoughts_token_count
    if metadata.tool_use_prompt_token_count:
        details['tool_use_prompt_tokens'] = metadata.tool_use_prompt_token_count

    for prefix, metadata_details in (
        ('prompt', metadata.prompt_tokens_details),
        ('cache', metadata.cache_tokens_details),
        ('candidates', metadata.candidates_tokens_details),
        ('tool_use_prompt', metadata.tool_use_prompt_tokens_details),
    ):
        if not metadata_details:
            continue
        for detail in metadata_details:
            if not detail.modality or not detail.token_count:
                continue
            details[f'{detail.modality.lower()}_{prefix}_tokens'] = detail.token_count

    return RequestUsage.extract(
        response.model_dump(include={'model_version', 'usage_metadata'}, by_alias=True),
        provider=provider,
        provider_url=provider_url,
        provider_fallback='google',
        details=details,
    )
