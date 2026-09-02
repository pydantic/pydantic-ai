from typing import Literal, TypeAlias

from typing_extensions import TypedDict

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
"""Portable aspect ratios accepted by at least one direct image model adapter.

The canonical exact shape a ratio produces is model-family specific, and the
families name different subsets: GPT Image 1.x three, GPT Image 2 sixteen,
Gemini 2.5 Flash and Gemini 3 Pro ten, Gemini 3.1 Flash and Flash Lite fourteen,
and Grok Imagine thirteen. A ratio outside a family's set still reaches Gemini,
which validates it itself; OpenAI and xAI have no way to carry it and raise
`UserError`. See the
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

    Providers with a native aspect-ratio field receive the ratio as given, and reject
    an unsupported one themselves. OpenAI has no such field, so Pydantic AI maps the
    ratio to one of the model family's enumerated sizes and raises `UserError` for a
    ratio outside that set; xAI takes an enum with no member for some portable values
    and raises `UserError` for those. See the
    [Image Generation guide](../image-generation.md#canonical-dimensions-for-aspect_ratio)
    for the per-family shapes.
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
