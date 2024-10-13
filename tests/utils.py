from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:

    def IsNow(*args: Any, **kwargs: Any) -> datetime: ...
else:
    from dirty_equals import IsNow


__all__ = ('IsNow',)
