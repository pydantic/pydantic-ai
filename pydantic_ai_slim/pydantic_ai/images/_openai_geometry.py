from dataclasses import dataclass
from fractions import Fraction

from pydantic_ai.exceptions import UserError

from .settings import (
    ImageDimensions,
    ImageGenerationAspectRatio,
    ImageGenerationSettings,
)

_LEGACY_SIZES = ('auto', '1024x1024', '1024x1536', '1536x1024')
_LEGACY_ASPECT_RATIO_TO_SIZE = {
    '1:1': '1024x1024',
    '2:3': '1024x1536',
    '3:2': '1536x1024',
}

# Canonical approximately-one-megapixel outputs for GPT Image 2. Each value has
# an exact requested aspect ratio and satisfies the model's documented limits.
_GPT_IMAGE_2_ASPECT_RATIO_TO_SIZE = {
    '1:1': '1024x1024',
    '1:2': '704x1408',
    '2:1': '1408x704',
    '2:3': '832x1248',
    '3:2': '1248x832',
    '3:4': '864x1152',
    '4:3': '1152x864',
    '4:5': '896x1120',
    '5:4': '1120x896',
    '9:16': '720x1280',
    '9:19.5': '672x1456',
    '9:20': '720x1600',
    '16:9': '1280x720',
    '19.5:9': '1456x672',
    '20:9': '1600x720',
    '21:9': '1568x672',
}


_GPT_IMAGE_1_MODELS = ('gpt-image-1', 'gpt-image-1.5')
"""GPT Image 1.x models whose geometry is the three-size legacy set.

`gpt-image-1-mini` and dated snapshots are matched through the dash suffix. `chatgpt-image-latest` is
deliberately absent: it tracks whichever model ChatGPT currently serves, so its geometry can change
without a name change and Pydantic AI cannot pin a size table to it.
"""


def is_gpt_image_1(model_name: str | None) -> bool:
    return model_name is not None and any(
        model_name == name or model_name.startswith(f'{name}-') for name in _GPT_IMAGE_1_MODELS
    )


def is_gpt_image_2(model_name: str | None) -> bool:
    return model_name == 'gpt-image-2' or (model_name is not None and model_name.startswith('gpt-image-2-'))


@dataclass
class _OpenAIGeometry:
    size: str | None
    conflicts: list[str]


def resolve_openai_geometry(
    model_name: str,
    settings: ImageGenerationSettings,
    *,
    provider_size: str | None,
) -> _OpenAIGeometry:
    """Resolve common and OpenAI-specific geometry to the native `size` field."""
    conflicts: list[str] = []
    dimensions = settings.get('dimensions')
    aspect_ratio = settings.get('aspect_ratio')
    resolved_dimensions = resolve_openai_dimensions(model_name, dimensions) if dimensions is not None else None

    if provider_size is not None:
        if resolved_dimensions is not None and resolved_dimensions != provider_size:
            conflicts.append('dimensions')
        if aspect_ratio is not None and not size_matches_aspect_ratio(provider_size, aspect_ratio):
            conflicts.append('aspect_ratio')
        return _OpenAIGeometry(size=provider_size, conflicts=conflicts)

    if resolved_dimensions is not None:
        return _OpenAIGeometry(size=resolved_dimensions, conflicts=conflicts)

    if aspect_ratio is not None:
        return _OpenAIGeometry(size=resolve_openai_aspect_ratio(model_name, aspect_ratio), conflicts=conflicts)

    return _OpenAIGeometry(size=None, conflicts=conflicts)


def resolve_openai_dimensions(model_name: str | None, dimensions: ImageDimensions) -> str:
    """Validate exact dimensions for an OpenAI image model and return its native size.

    A model outside the known families is forwarded unchecked: `size` is a plain string on the wire,
    so OpenAI can judge a shape a newly released model supports, where a frozen table would reject it
    until Pydantic AI catches up.
    """
    width, height = dimensions
    size = f'{width}x{height}'

    if is_gpt_image_2(model_name):
        problems: list[str] = []
        if width % 16 or height % 16:
            problems.append('width and height must be multiples of 16')
        if max(width, height) > 3840:
            problems.append('the longest edge must be at most 3840 pixels')
        if max(width, height) > 3 * min(width, height):
            problems.append('the aspect ratio must not exceed 3:1')
        pixels = width * height
        if pixels < 655_360 or pixels > 8_294_400:
            problems.append('the total pixel count must be between 655360 and 8294400')
        if problems:
            raise UserError(
                f'OpenAI model {model_name!r} does not support `dimensions={dimensions!r}`: {"; ".join(problems)}'
            )
        return size

    if is_gpt_image_1(model_name) and size not in _LEGACY_SIZES:
        supported = ', '.join(value for value in _LEGACY_SIZES if value != 'auto')
        raise UserError(
            f'OpenAI model {model_name!r} does not support `dimensions={dimensions!r}`. '
            f'Supported exact dimensions are: {supported}.'
        )
    return size


def resolve_openai_aspect_ratio(model_name: str | None, aspect_ratio: ImageGenerationAspectRatio) -> str:
    """Map a normalized aspect ratio to the model's canonical native size.

    OpenAI has no aspect-ratio field, so Pydantic AI performs the mapping itself and the request can
    only carry a ratio the model family enumerates. A ratio outside that set is rejected rather than
    dropped, because dropping it bills the caller for the model's default square instead. A model
    outside the known families has no canonical shapes to map onto, so it is rejected too — unlike
    `dimensions`, which needs no mapping and is forwarded for OpenAI to judge.
    """
    if is_gpt_image_2(model_name):
        mapping = _GPT_IMAGE_2_ASPECT_RATIO_TO_SIZE
    elif is_gpt_image_1(model_name):
        mapping = _LEGACY_ASPECT_RATIO_TO_SIZE
    else:
        raise UserError(
            f'Pydantic AI has no `aspect_ratio` mapping for OpenAI model {model_name!r}. '
            'Use `openai_size` or `dimensions` to set the output geometry.'
        )

    size = mapping.get(aspect_ratio)
    if size is None:
        supported = ', '.join(f'`{ratio}`' for ratio in mapping)
        raise UserError(
            f'OpenAI model {model_name!r} does not support `aspect_ratio={aspect_ratio!r}`. '
            f'Supported aspect ratios are: {supported}.'
        )
    return size


def size_matches_aspect_ratio(size: str, aspect_ratio: ImageGenerationAspectRatio) -> bool:
    dimensions = parse_dimensions(size)
    if dimensions is None:
        return False
    width, height = dimensions
    ratio_width, ratio_height = aspect_ratio.split(':', maxsplit=1)
    return Fraction(width, height) == Fraction(ratio_width) / Fraction(ratio_height)


def parse_dimensions(size: str) -> ImageDimensions | None:
    try:
        width_string, height_string = size.split('x', maxsplit=1)
        width = int(width_string)
        height = int(height_string)
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height
