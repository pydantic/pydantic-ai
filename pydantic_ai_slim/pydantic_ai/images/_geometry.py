from __future__ import annotations

from typing import TypeVar

_GeometryValueT = TypeVar('_GeometryValueT', bound=str)


def prefer_provider_value(
    provider_value: _GeometryValueT | None,
    mapped_value: _GeometryValueT | None,
    *,
    setting_name: str,
    conflicts: list[str],
) -> _GeometryValueT | None:
    """Pick the provider-specific geometry value over the one mapped from a portable setting.

    A provider-prefixed setting always wins over its portable equivalent, so the mapped value is used
    only when the request carries no native one. A disagreement between the two is recorded in
    `conflicts`, which `warn_image_generation_settings` reports to the caller.
    """
    if provider_value is None:
        return mapped_value
    if mapped_value is not None and provider_value != mapped_value:
        conflicts.append(setting_name)
    return provider_value
