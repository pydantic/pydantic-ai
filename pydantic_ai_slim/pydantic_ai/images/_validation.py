import warnings
from collections.abc import Sequence

from pydantic_ai.exceptions import UserError

from .settings import ImageDimensions, ImageGenerationSettings


def validate_image_generation_settings(settings: ImageGenerationSettings) -> None:
    """Validate provider-independent image generation setting invariants."""
    dimensions = settings.get('dimensions')
    if dimensions is None:
        return

    if settings.get('aspect_ratio') is not None:
        raise UserError('Image generation `dimensions` and `aspect_ratio` are mutually exclusive')

    validate_image_dimensions(dimensions)


def validate_image_dimensions(dimensions: ImageDimensions) -> None:
    """Validate the common exact-dimensions value before provider-specific mapping."""
    if not isinstance(dimensions, tuple):
        raise UserError('Image generation `dimensions` must be a `(width, height)` tuple of positive integers')
    if len(dimensions) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in dimensions
    ):
        raise UserError('Image generation `dimensions` must be a `(width, height)` tuple of positive integers')


def validate_image_count(provider: str, n: int | None, *, maximum: int) -> None:
    """Validate a provider-specific requested image count."""
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n <= 0 or n > maximum):
        raise UserError(f'{provider} image generation count must be an integer between 1 and {maximum}')


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
