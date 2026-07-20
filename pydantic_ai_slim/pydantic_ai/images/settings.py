import warnings
from collections.abc import Sequence
from typing import Literal, TypeAlias

from typing_extensions import TypedDict

from pydantic_ai.native_tools import ImageAspectRatio

ImageOutputFormat = Literal['png', 'jpeg', 'webp']
"""Common generated image output formats."""

ImageGenerationSize: TypeAlias = Literal['auto', '1024x1024', '1024x1536', '1536x1024', '512', '1K', '2K', '4K']
"""Provider-dependent image size accepted for compatibility with `ImageGenerationTool`.

OpenAI interprets pixel dimensions, while Google and xAI interpret resolution tiers.
Prefer provider-specific settings when exact provider behavior matters.
"""


class ImageGenerationSettings(TypedDict, total=False):
    """Normalized settings for configuring image generation models.

    Settings are applied on a best-effort basis. Each provider adapter maps the
    settings it supports and warns when an explicitly configured setting cannot
    affect the selected request.

    Provider-specific settings classes extend this with provider-prefixed
    options. A provider-specific setting takes precedence over its normalized
    equivalent when both are configured.
    """

    n: int
    """The number of images to generate.

    Supported by: OpenAI and xAI. Google currently supports one image per request.
    """

    output_format: ImageOutputFormat
    """The requested output format.

    Supported by: OpenAI. Google and xAI determine the format themselves.
    """

    background: Literal['transparent', 'opaque', 'auto']
    """The requested image background.

    Supported by: OpenAI. Transparent backgrounds require PNG or WebP output.
    """

    input_fidelity: Literal['high', 'low']
    """How closely an edit should preserve features of its reference images.

    Supported by: OpenAI image editing.
    """

    moderation: Literal['auto', 'low']
    """The requested moderation level.

    Supported by: OpenAI image generation.
    """

    output_compression: int
    """The requested output compression percentage.

    Supported by: OpenAI for JPEG and WebP output.
    """

    quality: Literal['low', 'medium', 'high', 'auto']
    """The requested generation quality.

    Supported by: OpenAI.
    """

    size: ImageGenerationSize
    """The provider-dependent output size.

    OpenAI accepts pixel dimensions; Google accepts `512`, `1K`, `2K`, or `4K`;
    xAI maps `1K` and `2K` to its resolution tiers. This field preserves the
    existing `ImageGenerationTool` vocabulary and is not an exact cross-provider
    resolution abstraction.
    """

    aspect_ratio: ImageAspectRatio
    """The requested aspect ratio.

    Supported directly by Google and xAI. OpenAI maps `1:1`, `2:3`, and `3:2`
    to its corresponding pixel sizes.
    """

    extra_headers: dict[str, str]
    """Extra headers to send to the model.

    This follows the existing `ModelSettings` and `EmbeddingSettings` escape-hatch pattern.
    Prefer provider-prefixed typed settings when a setting is part of the supported public API.
    """

    extra_body: object
    """Extra body to send to the model.

    This follows the existing `ModelSettings` and `EmbeddingSettings` escape-hatch pattern.
    Prefer provider-prefixed typed settings when a setting is part of the supported public API.
    """


def merge_image_generation_settings(
    base: ImageGenerationSettings | None, overrides: ImageGenerationSettings | None
) -> ImageGenerationSettings | None:
    """Merge two sets of image generation settings, with overrides taking precedence."""
    # Note: we may want merge recursively if/when we add non-primitive values.
    if base and overrides:
        return base | overrides
    else:
        return base or overrides


def warn_image_generation_settings(
    provider: str,
    *,
    ignored: Sequence[str] = (),
    conflicts: Sequence[str] = (),
) -> None:
    """Emit one warning for settings ignored or overridden by a provider adapter."""
    warning_parts: list[str] = []
    if ignored:
        names = ', '.join(f'`{name}`' for name in dict.fromkeys(ignored))
        warning_parts.append(f'ignored unsupported settings: {names}')
    if conflicts:
        names = ', '.join(f'`{name}`' for name in dict.fromkeys(conflicts))
        warning_parts.append(f'used provider-specific settings instead of: {names}')
    if warning_parts:
        warnings.warn(f'{provider} image generation {"; ".join(warning_parts)}', UserWarning, stacklevel=3)
