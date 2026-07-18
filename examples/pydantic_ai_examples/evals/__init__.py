from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import TimeRangeResponse as TimeRangeResponse

if TYPE_CHECKING:
    from .agent import infer_time_range as infer_time_range

__all__ = ('TimeRangeResponse', 'infer_time_range')


def __getattr__(name: str) -> Any:
    if name == 'infer_time_range':
        from .agent import infer_time_range

        return infer_time_range
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
