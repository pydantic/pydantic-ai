from __future__ import annotations

import warnings
from typing import TypeVar

_T = TypeVar('_T', bound=type)


class PydanticEvalsDeprecationWarning(UserWarning):
    """Warning emitted when a deprecated Pydantic Evals API is used.

    Inherits from `UserWarning` instead of `DeprecationWarning` so that
    deprecations are visible by default at runtime, following the approach
    described in https://sethmlarson.dev/deprecations-via-warnings-dont-work-for-python-libraries.
    """


def warn_positional_dataclass_init(cls: _T) -> _T:
    """Wrap a dataclass `__init__` so positional construction emits a deprecation warning.

    The typed `__init__` signature pyright infers from `@dataclass` is unchanged, so existing
    positional callers still type-check; only the runtime check is new. Intended as a
    stepping-stone to `@dataclass(kw_only=True)` in pydantic-evals v2.
    """
    original_init = cls.__init__
    cls_name = cls.__name__

    def __init__(self: object, *args: object, **kwargs: object) -> None:
        if args:
            warnings.warn(
                f'Constructing `{cls_name}` with positional arguments is deprecated; '
                f'use keyword arguments instead. Positional construction will be removed in pydantic-evals v2.',
                PydanticEvalsDeprecationWarning,
                stacklevel=2,
            )
        original_init(self, *args, **kwargs)

    __init__.__qualname__ = f'{cls_name}.__init__'
    __init__.__doc__ = original_init.__doc__
    cls.__init__ = __init__
    return cls
