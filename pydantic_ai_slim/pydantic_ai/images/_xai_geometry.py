from dataclasses import dataclass

from xai_sdk.types import ImageAspectRatio, ImageResolution

from pydantic_ai.exceptions import UserError

from .settings import (
    ImageDimensions,
    ImageGenerationAspectRatio,
    ImageGenerationSettings,
    validate_image_dimensions,
)

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
_XAI_GEOMETRY_MODELS = frozenset({'grok-imagine-image', 'grok-imagine-image-quality'})
_XAI_ASPECT_RATIOS: dict[str, ImageAspectRatio] = {value: value for value in _XAI_GEOMETRIES}
_XAI_RESOLUTIONS: dict[str, ImageResolution] = {'1K': '1k', '2K': '2k'}


@dataclass
class _XaiGeometry:
    aspect_ratio: ImageAspectRatio | None
    resolution: ImageResolution | None
    ignored: list[str]
    conflicts: list[str]


def resolve_xai_geometry(
    model_name: str,
    settings: ImageGenerationSettings,
    *,
    provider_aspect_ratio: ImageAspectRatio | None,
    provider_resolution: ImageResolution | None,
) -> _XaiGeometry:
    """Resolve common and xAI-specific geometry to native SDK fields."""
    ignored: list[str] = []
    conflicts: list[str] = []

    if dimensions := settings.get('dimensions'):
        mapped_aspect_ratio, mapped_resolution = resolve_xai_dimensions(model_name, dimensions)
        aspect_ratio = provider_aspect_ratio
        if aspect_ratio is None:
            aspect_ratio = mapped_aspect_ratio
        elif aspect_ratio != mapped_aspect_ratio:
            conflicts.append('dimensions')
        resolution = provider_resolution
        if resolution is None:
            resolution = mapped_resolution
        elif resolution != mapped_resolution:
            conflicts.append('dimensions')
        return _XaiGeometry(
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            ignored=ignored,
            conflicts=conflicts,
        )

    common_aspect_ratio = settings.get('aspect_ratio')
    mapped_aspect_ratio = resolve_xai_aspect_ratio(common_aspect_ratio) if common_aspect_ratio else None
    if common_aspect_ratio is not None and mapped_aspect_ratio is None:
        ignored.append('aspect_ratio')
    if provider_aspect_ratio is not None:
        if mapped_aspect_ratio is not None and mapped_aspect_ratio != provider_aspect_ratio:
            conflicts.append('aspect_ratio')
        aspect_ratio = provider_aspect_ratio
    else:
        aspect_ratio = mapped_aspect_ratio

    common_size = settings.get('size')
    mapped_resolution = resolve_xai_size(common_size) if common_size else None
    if common_size is not None and mapped_resolution is None:
        ignored.append('size')
    if provider_resolution is not None:
        if mapped_resolution is not None and mapped_resolution != provider_resolution:
            conflicts.append('size')
        resolution = provider_resolution
    elif mapped_resolution is not None:
        resolution = mapped_resolution
    elif mapped_aspect_ratio is not None:
        # A common ratio promises one canonical model geometry. Pin xAI's documented default tier
        # instead of relying on a provider default that could change independently.
        resolution = '1k'
    else:
        resolution = None

    return _XaiGeometry(
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        ignored=ignored,
        conflicts=conflicts,
    )


def resolve_xai_dimensions(model_name: str, dimensions: ImageDimensions) -> tuple[ImageAspectRatio, ImageResolution]:
    """Map exact dimensions to the verified xAI aspect-ratio and resolution pair."""
    validate_image_dimensions(dimensions)
    if model_name not in _XAI_GEOMETRY_MODELS:
        raise UserError(f'xAI model {model_name!r} does not have a known exact-dimensions mapping')
    for aspect_ratio, resolutions in _XAI_GEOMETRIES.items():
        for resolution, supported_dimensions in resolutions.items():
            if dimensions == supported_dimensions:
                return aspect_ratio, resolution
    raise UserError(f'xAI model {model_name!r} does not support `dimensions={dimensions!r}`')


def resolve_xai_aspect_ratio(aspect_ratio: ImageGenerationAspectRatio) -> ImageAspectRatio | None:
    """Map a portable aspect ratio to the xAI SDK type when supported."""
    return _XAI_ASPECT_RATIOS.get(aspect_ratio)


def resolve_xai_size(size: str) -> ImageResolution | None:
    """Map the compatibility size vocabulary to an xAI resolution tier."""
    return _XAI_RESOLUTIONS.get(size)
