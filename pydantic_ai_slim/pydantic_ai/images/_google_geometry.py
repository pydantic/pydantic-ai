from dataclasses import dataclass

from pydantic_ai.exceptions import UserError

from .settings import (
    ImageDimensions,
    ImageGenerationAspectRatio,
    ImageGenerationSettings,
    validate_image_dimensions,
)

_GEMINI_31_512_DIMENSIONS: dict[ImageGenerationAspectRatio, ImageDimensions] = {
    '1:1': (512, 512),
    '1:4': (256, 1024),
    '1:8': (192, 1536),
    '2:3': (424, 632),
    '3:2': (632, 424),
    '3:4': (448, 600),
    '4:1': (1024, 256),
    '4:3': (600, 448),
    '4:5': (464, 576),
    '5:4': (576, 464),
    '8:1': (1536, 192),
    '9:16': (384, 688),
    '16:9': (688, 384),
    '21:9': (792, 168),
}
_GEMINI_31_STANDARD_RATIOS: tuple[ImageGenerationAspectRatio, ...] = (
    '1:1',
    '2:3',
    '3:2',
    '3:4',
    '4:3',
    '4:5',
    '5:4',
    '9:16',
    '16:9',
    '21:9',
)
_GEMINI_25_DIMENSIONS: dict[ImageGenerationAspectRatio, ImageDimensions] = {
    '1:1': (1024, 1024),
    '2:3': (832, 1248),
    '3:2': (1248, 832),
    '3:4': (864, 1184),
    '4:3': (1184, 864),
    '4:5': (896, 1152),
    '5:4': (1152, 896),
    '9:16': (768, 1344),
    '16:9': (1344, 768),
    '21:9': (1536, 672),
}


@dataclass(frozen=True)
class _GoogleImageGeometryProfile:
    dimensions: dict[ImageGenerationAspectRatio, dict[str | None, ImageDimensions]]
    default_size: str | None


def _scaled_dimensions(
    base: dict[ImageGenerationAspectRatio, ImageDimensions], scales: dict[str, int]
) -> dict[ImageGenerationAspectRatio, dict[str | None, ImageDimensions]]:
    return {
        ratio: {size: (width * scale, height * scale) for size, scale in scales.items()}
        for ratio, (width, height) in base.items()
    }


_GEMINI_31_FLASH_DIMENSIONS = _scaled_dimensions(_GEMINI_31_512_DIMENSIONS, {'512': 1, '1K': 2, '2K': 4, '4K': 8})
# Google's published 21:9 row does not scale uniformly from the 512 tier, so keep
# its documented dimensions explicit rather than deriving them arithmetically.
_GEMINI_31_FLASH_DIMENSIONS['21:9'] = {
    '512': (792, 168),
    '1K': (1584, 672),
    '2K': (3168, 1344),
    '4K': (6336, 2688),
}
_GEMINI_31_FLASH_PROFILE = _GoogleImageGeometryProfile(
    dimensions=_GEMINI_31_FLASH_DIMENSIONS,
    default_size='1K',
)
_GEMINI_31_PRO_PROFILE = _GoogleImageGeometryProfile(
    dimensions={
        ratio: {
            size: dimensions for size, dimensions in _GEMINI_31_FLASH_PROFILE.dimensions[ratio].items() if size != '512'
        }
        for ratio in _GEMINI_31_STANDARD_RATIOS
    },
    default_size='1K',
)
_GEMINI_31_FLASH_LITE_PROFILE = _GoogleImageGeometryProfile(
    dimensions={
        ratio: {'1K': _GEMINI_31_FLASH_PROFILE.dimensions[ratio]['1K']} for ratio in _GEMINI_31_STANDARD_RATIOS
    },
    default_size='1K',
)
_GEMINI_25_FLASH_PROFILE = _GoogleImageGeometryProfile(
    dimensions={ratio: {None: dimensions} for ratio, dimensions in _GEMINI_25_DIMENSIONS.items()},
    default_size=None,
)


@dataclass
class _GoogleGeometry:
    aspect_ratio: str | None
    image_size: str | None
    ignored: list[str]
    conflicts: list[str]


def _prefer_google_value(
    provider_value: str | None,
    mapped_value: str | None,
    *,
    setting_name: str,
    conflicts: list[str],
) -> str | None:
    if provider_value is None:
        return mapped_value
    if mapped_value is not None and provider_value != mapped_value:
        conflicts.append(setting_name)
    return provider_value


def resolve_google_geometry(
    model_name: str,
    settings: ImageGenerationSettings,
    *,
    provider_aspect_ratio: str | None,
    provider_size: str | None,
    provider_size_is_set: bool,
) -> _GoogleGeometry:
    """Resolve common and Google-specific geometry to native image config fields."""
    aspect_ratio = provider_aspect_ratio
    image_size = provider_size
    ignored: list[str] = []
    conflicts: list[str] = []

    if dimensions := settings.get('dimensions'):
        mapped_aspect_ratio, mapped_size = resolve_google_dimensions(model_name, dimensions)
        aspect_ratio = _prefer_google_value(
            aspect_ratio, mapped_aspect_ratio, setting_name='dimensions', conflicts=conflicts
        )
        image_size = _prefer_google_value(image_size, mapped_size, setting_name='dimensions', conflicts=conflicts)
    elif common_aspect_ratio := settings.get('aspect_ratio'):
        mapped_geometry = resolve_google_aspect_ratio(model_name, common_aspect_ratio)
        if mapped_geometry is None:
            ignored.append('aspect_ratio')
        else:
            mapped_aspect_ratio, default_size = mapped_geometry
            aspect_ratio = _prefer_google_value(
                aspect_ratio, mapped_aspect_ratio, setting_name='aspect_ratio', conflicts=conflicts
            )
            if default_size is not None and not provider_size_is_set and 'size' not in settings:
                image_size = default_size

    if common_size := settings.get('size'):
        if not google_supports_image_size(model_name, common_size):
            ignored.append('size')
        else:
            image_size = _prefer_google_value(image_size, common_size, setting_name='size', conflicts=conflicts)

    return _GoogleGeometry(
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        ignored=ignored,
        conflicts=conflicts,
    )


def resolve_google_dimensions(
    model_name: str, dimensions: ImageDimensions
) -> tuple[ImageGenerationAspectRatio, str | None]:
    """Map exact dimensions to Google-native aspect-ratio and image-size fields."""
    validate_image_dimensions(dimensions)
    profile = _google_image_geometry_profile(model_name)
    if profile is not None:
        for aspect_ratio, sizes in profile.dimensions.items():
            for image_size, supported_dimensions in sizes.items():
                if supported_dimensions == dimensions:
                    return aspect_ratio, image_size

    raise UserError(f'Google model {model_name!r} does not support `dimensions={dimensions!r}`')


def resolve_google_aspect_ratio(
    model_name: str, aspect_ratio: ImageGenerationAspectRatio
) -> tuple[ImageGenerationAspectRatio, str | None] | None:
    """Map a normalized aspect ratio to the model's canonical Google geometry."""
    profile = _google_image_geometry_profile(model_name)
    if profile is None:
        return aspect_ratio, None
    if aspect_ratio not in profile.dimensions:
        return None
    return aspect_ratio, profile.default_size


def google_supports_image_size(model_name: str, image_size: str) -> bool:
    # `size` is the compatibility path and historically forwards every Google tier.
    # Only the new exact `dimensions` mapping is restricted by model profile.
    return image_size in ('512', '1K', '2K', '4K')


def _google_image_geometry_profile(model_name: str) -> _GoogleImageGeometryProfile | None:
    if 'gemini-3.1-flash-lite-image' in model_name:
        return _GEMINI_31_FLASH_LITE_PROFILE
    if 'gemini-3.1-flash-image' in model_name:
        return _GEMINI_31_FLASH_PROFILE
    if 'gemini-3-pro-image' in model_name or 'gemini-3.1-pro-image' in model_name:
        return _GEMINI_31_PRO_PROFILE
    if 'gemini-2.5-flash-image' in model_name:
        return _GEMINI_25_FLASH_PROFILE
    return None
