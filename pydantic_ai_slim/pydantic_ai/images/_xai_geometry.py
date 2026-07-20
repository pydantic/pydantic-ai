from xai_sdk.types import ImageAspectRatio, ImageResolution

from pydantic_ai.exceptions import UserError

from .settings import ImageDimensions, ImageGenerationAspectRatio, validate_image_dimensions

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
