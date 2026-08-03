from typing import Literal, TypeAlias

from typing_extensions import TypedDict

ImageOutputFormat = Literal['png', 'webp', 'jpeg']
"""Generated image output formats used by providers that support format selection."""

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


class ImageGenerationSettings(TypedDict, total=False):
    """Normalized settings for configuring image generation models.

    This type contains only settings with the same semantics across every direct
    image provider. Provider-specific settings classes extend it with prefixed
    options for controls that are not portable.
    """

    dimensions: ImageDimensions
    """The exact output dimensions as `(width, height)` in pixels.

    This is mutually exclusive with `aspect_ratio`. The selected provider and
    model must support the exact dimensions;
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
