from typing import Literal

from typing_extensions import TypedDict

ImageOutputFormat = Literal['png', 'jpeg', 'webp']
"""Common generated image output formats."""


class ImageGenerationSettings(TypedDict, total=False):
    """Common settings for configuring image generation models.

    These settings apply across multiple image generation providers. Not all
    settings are supported by all models - check the specific model's
    documentation for details.

    Provider-specific settings classes extend this with provider-prefixed
    options.
    """

    n: int
    """The number of images to generate."""

    output_format: ImageOutputFormat
    """The requested output format."""

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
