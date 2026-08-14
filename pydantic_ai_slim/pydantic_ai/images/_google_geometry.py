from dataclasses import dataclass

from pydantic_ai.exceptions import UserError

from .settings import (
    ImageDimensions,
    ImageGenerationAspectRatio,
    ImageGenerationSettings,
)

_GEMINI_31_512_DIMENSIONS: dict[ImageGenerationAspectRatio, ImageDimensions] = {
    '1:1': (512, 512),
    '2:3': (424, 632),
    '3:2': (632, 424),
    '3:4': (448, 600),
    '4:3': (600, 448),
    '4:5': (464, 576),
    '5:4': (576, 464),
    '9:16': (384, 688),
    '16:9': (688, 384),
    '21:9': (792, 168),
}
_GEMINI_31_STANDARD_RATIOS: tuple[ImageGenerationAspectRatio, ...] = tuple(_GEMINI_31_512_DIMENSIONS)
# The extended ratios and 21:9 neither scale uniformly from the 512 tier nor match Google's published
# table, so every tier is listed as observed against the live API.
_GEMINI_31_EXPLICIT_DIMENSIONS: dict[ImageGenerationAspectRatio, dict[str | None, ImageDimensions]] = {
    '21:9': {'512': (792, 168), '1K': (1584, 672), '2K': (3168, 1344), '4K': (6336, 2688)},
    '1:4': {'512': (256, 1024), '1K': (512, 2064), '2K': (1024, 4128), '4K': (2048, 8256)},
    '1:8': {'512': (176, 1456), '1K': (352, 2928), '2K': (704, 5856), '4K': (1408, 11712)},
    '4:1': {'512': (1024, 256), '1K': (2064, 512), '2K': (4128, 1024), '4K': (8256, 2048)},
    '8:1': {'512': (1456, 176), '1K': (2928, 352), '2K': (5856, 704), '4K': (11712, 1408)},
}
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
_GEMINI_31_FLASH_DIMENSIONS.update(_GEMINI_31_EXPLICIT_DIMENSIONS)
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
# Flash Lite serves the same 1K shapes as Flash, and only that tier: Google's documented 512 column
# is rejected by the live API.
_GEMINI_31_FLASH_LITE_PROFILE = _GoogleImageGeometryProfile(
    dimensions={ratio: {'1K': sizes['1K']} for ratio, sizes in _GEMINI_31_FLASH_DIMENSIONS.items()},
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
    conflicts: list[str] = []

    if dimensions := settings.get('dimensions'):
        mapped_aspect_ratio, mapped_size = resolve_google_dimensions(model_name, dimensions)
        aspect_ratio = _prefer_google_value(
            aspect_ratio, mapped_aspect_ratio, setting_name='dimensions', conflicts=conflicts
        )
        image_size = _prefer_google_value(image_size, mapped_size, setting_name='dimensions', conflicts=conflicts)
    elif common_aspect_ratio := settings.get('aspect_ratio'):
        # Forwarded whether or not the model's geometry profile lists the ratio: the profile records
        # the shapes we can name for `dimensions`, and the API is the authority on what it accepts.
        aspect_ratio = _prefer_google_value(
            aspect_ratio, common_aspect_ratio, setting_name='aspect_ratio', conflicts=conflicts
        )
        profile = _google_image_geometry_profile(model_name)
        if profile is not None and profile.default_size is not None and not provider_size_is_set:
            image_size = profile.default_size

    return _GoogleGeometry(aspect_ratio=aspect_ratio, image_size=image_size, conflicts=conflicts)


def resolve_google_dimensions(
    model_name: str, dimensions: ImageDimensions
) -> tuple[ImageGenerationAspectRatio, str | None]:
    """Map exact dimensions to Google-native aspect-ratio and image-size fields."""
    profile = _google_image_geometry_profile(model_name)
    if profile is not None:
        for aspect_ratio, sizes in profile.dimensions.items():
            for image_size, supported_dimensions in sizes.items():
                if supported_dimensions == dimensions:
                    return aspect_ratio, image_size

    raise UserError(f'Google model {model_name!r} does not support `dimensions={dimensions!r}`')


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
