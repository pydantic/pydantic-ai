from dataclasses import dataclass

from xai_sdk.types import ImageAspectRatio, ImageResolution

from pydantic_ai.exceptions import UserError

from ._geometry import prefer_provider_value
from .settings import (
    ImageDimensions,
    ImageGenerationAspectRatio,
    ImageGenerationSettings,
)

# Keyed by the members of the gRPC `ImageAspectRatio` enum, the only ratio vocabulary the image RPC
# accepts. A portable ratio the enum omits — `21:9` is the one `ImageGenerationAspectRatio` carries —
# has no wire representation at all, and bumping the `xai-sdk` floor does not add it: 1.19.0, the
# newest published release, generates the same 13 members as the locked 1.18.0.
_XAI_GEOMETRIES: dict[ImageAspectRatio, dict[ImageResolution, ImageDimensions]] = {
    '1:1': {'1k': (1024, 1024), '2k': (2048, 2048)},
    '3:4': {'1k': (864, 1152), '2k': (1776, 2368)},
    '4:3': {'1k': (1152, 864), '2k': (2368, 1776)},
    '9:16': {'1k': (720, 1280), '2k': (1584, 2816)},
    '16:9': {'1k': (1280, 720), '2k': (2816, 1584)},
    '2:3': {'1k': (832, 1248), '2k': (1664, 2496)},
    '3:2': {'1k': (1248, 832), '2k': (2496, 1664)},
    '9:19.5': {'1k': (576, 1248), '2k': (1344, 2912)},
    '19.5:9': {'1k': (1248, 576), '2k': (2912, 1344)},
    '9:20': {'1k': (576, 1280), '2k': (1440, 3200)},
    '20:9': {'1k': (1280, 576), '2k': (3200, 1440)},
    '1:2': {'1k': (704, 1408), '2k': (1456, 2912)},
    '2:1': {'1k': (1408, 704), '2k': (2912, 1456)},
}
# Both canonical models share the single `_XAI_GEOMETRIES` table, so an alias needs no geometry data
# of its own — it only has to be recognized here. Enumerated rather than matched by prefix so an
# unknown future model still falls through to the error, the way `grok-imagine-image-9` should.
# `grok-imagine-image-2.0` is a third model rather than an alias of either, and xAI publishes no
# ratio-to-pixel mapping for it, so `dimensions` keeps raising there until someone probes it; its
# `aspect_ratio` and provider-prefixed settings work regardless, since neither consults this set.
# https://docs.x.ai/developers/models/grok-imagine-image
# https://docs.x.ai/developers/models/grok-imagine-image-quality
_XAI_GEOMETRY_MODELS = frozenset(
    {
        'grok-imagine-image',
        'grok-imagine-image-2026-03-02',
        'grok-imagine-image-quality',
        'grok-imagine-image-quality-20260403',
        'grok-imagine-image-quality-latest',
        'grok-imagine-image-pro',
    }
)
_XAI_ASPECT_RATIOS: dict[str, ImageAspectRatio] = {value: value for value in _XAI_GEOMETRIES}


@dataclass
class _XaiGeometry:
    aspect_ratio: ImageAspectRatio | None
    resolution: ImageResolution | None
    conflicts: list[str]


def resolve_xai_geometry(
    model_name: str,
    settings: ImageGenerationSettings,
    *,
    provider_aspect_ratio: ImageAspectRatio | None,
    provider_resolution: ImageResolution | None,
) -> _XaiGeometry:
    """Resolve common and xAI-specific geometry to native SDK fields."""
    conflicts: list[str] = []

    if dimensions := settings.get('dimensions'):
        mapped_aspect_ratio, mapped_resolution = resolve_xai_dimensions(model_name, dimensions)
        return _XaiGeometry(
            aspect_ratio=prefer_provider_value(
                provider_aspect_ratio, mapped_aspect_ratio, setting_name='dimensions', conflicts=conflicts
            ),
            resolution=prefer_provider_value(
                provider_resolution, mapped_resolution, setting_name='dimensions', conflicts=conflicts
            ),
            conflicts=conflicts,
        )

    common_aspect_ratio = settings.get('aspect_ratio')
    if provider_aspect_ratio is not None:
        # The provider-specific value wins, so the common one never reaches the wire and is not
        # mapped: a portable ratio xAI's enum cannot express is a conflict to warn about here, not an
        # error. `_XAI_ASPECT_RATIOS` is an identity map, so the raw strings compare exactly.
        if common_aspect_ratio is not None and common_aspect_ratio != provider_aspect_ratio:
            conflicts.append('aspect_ratio')
        aspect_ratio = provider_aspect_ratio
    else:
        aspect_ratio = resolve_xai_aspect_ratio(common_aspect_ratio) if common_aspect_ratio else None

    if provider_resolution is not None:
        resolution = provider_resolution
    elif common_aspect_ratio is not None:
        # A common ratio promises one canonical model geometry. Pin xAI's documented default tier
        # instead of relying on a provider default that could change independently.
        resolution = '1k'
    else:
        resolution = None

    return _XaiGeometry(aspect_ratio=aspect_ratio, resolution=resolution, conflicts=conflicts)


def resolve_xai_dimensions(model_name: str, dimensions: ImageDimensions) -> tuple[ImageAspectRatio, ImageResolution]:
    """Map exact dimensions to the verified xAI aspect-ratio and resolution pair."""
    if model_name not in _XAI_GEOMETRY_MODELS:
        raise UserError(f'xAI model {model_name!r} does not have a known exact-dimensions mapping')
    for aspect_ratio, resolutions in _XAI_GEOMETRIES.items():
        for resolution, supported_dimensions in resolutions.items():
            if dimensions == supported_dimensions:
                return aspect_ratio, resolution
    raise UserError(f'xAI model {model_name!r} does not support `dimensions={dimensions!r}`')


def resolve_xai_aspect_ratio(aspect_ratio: ImageGenerationAspectRatio) -> ImageAspectRatio:
    """Map a portable aspect ratio onto the `ImageAspectRatio` proto enum the request travels in.

    xAI takes the ratio as an enum rather than free text, so a portable value with no enum member has
    no wire representation at all and is rejected here instead of being dropped from the request.
    """
    mapped = _XAI_ASPECT_RATIOS.get(aspect_ratio)
    if mapped is None:
        supported = ', '.join(f'`{value}`' for value in _XAI_ASPECT_RATIOS)
        raise UserError(
            f'The `xai_sdk` `ImageAspectRatio` enum has no member for `aspect_ratio={aspect_ratio!r}`, '
            f'so the gRPC image request cannot carry it. Supported aspect ratios are: {supported}.'
        )
    return mapped
