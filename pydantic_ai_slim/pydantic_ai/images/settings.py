import warnings
from collections.abc import Sequence
from typing import Literal, TypeAlias

from typing_extensions import TypedDict

from pydantic_ai.exceptions import UserError

ImageOutputFormat = Literal['png', 'jpeg', 'webp']
"""Common generated image output formats."""

ImageDimensions: TypeAlias = tuple[int, int]
"""Exact output image dimensions as `(width, height)` in pixels.

Supported values are model-specific. GPT Image 1.x accepts three fixed shapes;
GPT Image 2 accepts any shape satisfying its edge, area, multiple-of-16, and 3:1
limits; Gemini and Grok Imagine accept the documented or verified shapes for
their aspect-ratio and resolution tiers. See the
[Image Generation guide](../image-generation.md#supported-exact-dimensions)
for the complete matrix.
"""

ImageGenerationAspectRatio: TypeAlias = Literal[
    '1:1',
    '1:2',
    '1:4',
    '1:8',
    '2:1',
    '2:3',
    '3:2',
    '3:4',
    '4:1',
    '4:3',
    '4:5',
    '5:4',
    '8:1',
    '9:16',
    '9:19.5',
    '9:20',
    '16:9',
    '19.5:9',
    '20:9',
    '21:9',
]
"""Portable aspect ratios understood by at least one direct image model adapter.

Each adapter maps a supported ratio to one canonical exact output shape. Model
families support different subsets: GPT Image 1.x supports `1:1`, `2:3`, and
`3:2`; GPT Image 2 supports sixteen ratios; current Gemini families support ten
or fourteen; Grok Imagine supports thirteen. See the
[Image Generation guide](../image-generation.md#canonical-dimensions-for-aspect_ratio)
for the ratio-to-dimensions matrix.
"""

ImageGenerationSize: TypeAlias = str
"""Provider-dependent image size accepted by direct image model adapters.

OpenAI interprets pixel dimensions, while Google and xAI interpret model-specific resolution tiers.
Prefer provider-specific settings when exact provider behavior matters.
"""

_LEGACY_IMAGE_SIZES = frozenset({'auto', '1024x1024', '1024x1536', '1536x1024', '512', '1K', '2K', '4K'})
_LEGACY_IMAGE_ASPECT_RATIOS = frozenset({'21:9', '16:9', '4:3', '3:2', '1:1', '9:16', '3:4', '2:3', '5:4', '4:5'})


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

    Must be a positive integer. Boolean values are not accepted.

    Supported by: OpenAI and xAI. Google currently supports one image per request.
    """

    output_format: ImageOutputFormat
    """The requested output format.

    Supported by: OpenAI. Google and xAI determine the format themselves.
    """

    background: Literal['transparent', 'opaque', 'auto']
    """The requested image background.

    Supported by: OpenAI. Transparent backgrounds require PNG or WebP output and are not supported by GPT Image 2.
    """

    input_fidelity: Literal['high', 'low']
    """How closely an edit should preserve features of its reference images.

    Supported by: OpenAI image editing before GPT Image 2. GPT Image 2 always
    processes input images at high fidelity, so this setting is ignored for that model.
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

    OpenAI accepts pixel dimensions; known Google models accept a subset of `512`,
    `1K`, `2K`, and `4K`; xAI maps `1K` and `2K` to its resolution tiers. This
    direct-API compatibility field is not an exact cross-provider resolution abstraction.
    """

    dimensions: ImageDimensions
    """The exact output dimensions as `(width, height)` in pixels.

    This is mutually exclusive with `aspect_ratio` and the compatibility `size`
    setting. The selected provider and model must support the exact dimensions;
    no rounding or nearest-shape fallback is applied. GPT Image 1.x accepts its
    three fixed shapes, GPT Image 2 validates a continuous constrained range, and
    Gemini/Grok Imagine accept their model-specific aspect-ratio and resolution
    table entries. See the `ImageDimensions` documentation for the full matrix.
    """

    aspect_ratio: ImageGenerationAspectRatio
    """The requested aspect ratio.

    Provider adapters map this to the canonical model-specific exact dimensions
    documented by `ImageGenerationAspectRatio`. Not every ratio is supported by
    every model; an unsupported explicit value is ignored with a `UserWarning`.
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


def validate_image_generation_settings(settings: ImageGenerationSettings) -> None:
    """Validate provider-independent image generation setting invariants."""
    if (n := settings.get('n')) is not None and (not isinstance(n, int) or isinstance(n, bool) or n <= 0):
        raise UserError('Image generation `n` must be a positive integer')

    dimensions = settings.get('dimensions')
    if dimensions is None:
        return

    if settings.get('aspect_ratio') is not None:
        raise UserError('Image generation `dimensions` and `aspect_ratio` are mutually exclusive')
    if settings.get('size') is not None:
        raise UserError('Image generation `dimensions` and `size` are mutually exclusive')

    validate_image_dimensions(dimensions)


def validate_image_dimensions(dimensions: ImageDimensions) -> None:
    """Validate the common exact-dimensions value before provider-specific mapping."""
    if not isinstance(dimensions, tuple):
        raise UserError('Image generation `dimensions` must be a `(width, height)` tuple of positive integers')
    if len(dimensions) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions
    ):
        raise UserError('Image generation `dimensions` must be a `(width, height)` tuple of positive integers')


def image_generation_tool_settings(
    settings: ImageGenerationSettings,
) -> tuple[ImageGenerationSettings, list[str]]:
    """Return the subset supported by the legacy `ImageGenerationTool` surface."""
    legacy_settings = settings.copy()
    ignored: list[str] = []
    if legacy_settings.pop('dimensions', None) is not None:
        ignored.append('dimensions')
    if (size := legacy_settings.get('size')) is not None and size not in _LEGACY_IMAGE_SIZES:
        legacy_settings.pop('size')
        ignored.append('size')
    if (
        aspect_ratio := legacy_settings.get('aspect_ratio')
    ) is not None and aspect_ratio not in _LEGACY_IMAGE_ASPECT_RATIOS:
        legacy_settings.pop('aspect_ratio')
        ignored.append('aspect_ratio')
    return legacy_settings, ignored


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
