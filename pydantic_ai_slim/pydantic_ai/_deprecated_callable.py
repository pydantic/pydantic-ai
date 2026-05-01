"""Staging machinery for v2 method-to-property migrations.

In 1.x, methods like `AgentRunResult.usage()` are converted to properties via
`@deprecated_callable_property`. The property returns a wrapper that *is* an
instance of the underlying type (so `isinstance` is preserved), but is also
callable: calling it (parentheses) emits a `DeprecationWarning` and returns
the same value.

In v2, the decorator is replaced with `@property` (or the method is removed
entirely, in which case users follow the deprecation hint).
"""

from __future__ import annotations as _annotations

import dataclasses
import warnings
from collections.abc import Callable
from datetime import datetime
from typing import Any

from .messages import ModelResponse
from .usage import RequestUsage, RunUsage


class _DeprecatedCallableRunUsage(RunUsage):
    """A `RunUsage` whose `()` call emits `DeprecationWarning` and returns itself."""

    _deprecation_message: str

    def __init__(self, base: RunUsage, message: str) -> None:
        for f in dataclasses.fields(RunUsage):
            object.__setattr__(self, f.name, getattr(base, f.name))
        object.__setattr__(self, '_deprecation_message', message)

    def __call__(self) -> RunUsage:
        warnings.warn(self._deprecation_message, DeprecationWarning, stacklevel=2)
        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RunUsage):
            return all(getattr(self, f.name) == getattr(other, f.name) for f in dataclasses.fields(RunUsage))
        return NotImplemented

    def __repr__(self) -> str:
        kv_pairs = (f'{f.name}={value!r}' for f in dataclasses.fields(RunUsage) if (value := getattr(self, f.name)))
        return f'RunUsage({", ".join(kv_pairs)})'


class _DeprecatedCallableRequestUsage(RequestUsage):
    """A `RequestUsage` whose `()` call emits `DeprecationWarning` and returns itself."""

    _deprecation_message: str

    def __init__(self, base: RequestUsage, message: str) -> None:
        for f in dataclasses.fields(RequestUsage):
            object.__setattr__(self, f.name, getattr(base, f.name))
        object.__setattr__(self, '_deprecation_message', message)

    def __call__(self) -> RequestUsage:
        warnings.warn(self._deprecation_message, DeprecationWarning, stacklevel=2)
        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RequestUsage):
            return all(getattr(self, f.name) == getattr(other, f.name) for f in dataclasses.fields(RequestUsage))
        return NotImplemented

    def __repr__(self) -> str:
        kv_pairs = (f'{f.name}={value!r}' for f in dataclasses.fields(RequestUsage) if (value := getattr(self, f.name)))
        return f'RequestUsage({", ".join(kv_pairs)})'


class _DeprecatedCallableDatetime(datetime):
    """A `datetime` whose `()` call emits `DeprecationWarning` and returns itself."""

    _deprecation_message: str

    def __new__(cls, base: datetime, message: str) -> _DeprecatedCallableDatetime:
        instance = datetime.__new__(
            cls,
            base.year,
            base.month,
            base.day,
            base.hour,
            base.minute,
            base.second,
            base.microsecond,
            base.tzinfo,
            fold=base.fold,
        )
        instance._deprecation_message = message
        return instance

    def __call__(self) -> datetime:
        warnings.warn(self._deprecation_message, DeprecationWarning, stacklevel=2)
        return self

    def __repr__(self) -> str:
        return datetime(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
            self.tzinfo,
            fold=self.fold,
        ).__repr__()


class _DeprecatedCallableResponse(ModelResponse):
    """A `ModelResponse` whose `()` call emits `DeprecationWarning` and returns itself."""

    _deprecation_message: str

    def __init__(self, base: ModelResponse, message: str) -> None:
        for f in dataclasses.fields(ModelResponse):
            object.__setattr__(self, f.name, getattr(base, f.name))
        object.__setattr__(self, '_deprecation_message', message)

    def __call__(self) -> ModelResponse:
        warnings.warn(self._deprecation_message, DeprecationWarning, stacklevel=2)
        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ModelResponse):
            return all(getattr(self, f.name) == getattr(other, f.name) for f in dataclasses.fields(ModelResponse))
        return NotImplemented

    def __repr__(self) -> str:
        kv_pairs = (
            f'{f.name}={getattr(self, f.name)!r}'
            for f in dataclasses.fields(ModelResponse)
            if f.repr and getattr(self, f.name) != f.default
        )
        return f'ModelResponse({", ".join(kv_pairs)})'


def _wrap(value: Any, message: str) -> Any:
    if isinstance(value, RunUsage):
        return _DeprecatedCallableRunUsage(value, message)
    if isinstance(value, RequestUsage):
        return _DeprecatedCallableRequestUsage(value, message)
    if isinstance(value, ModelResponse):
        return _DeprecatedCallableResponse(value, message)
    if isinstance(value, datetime):
        return _DeprecatedCallableDatetime(value, message)
    raise TypeError(f'No deprecation wrapper registered for type {type(value).__name__!r}')  # pragma: no cover


class _DeprecatedCallableProperty:
    """Descriptor presenting a method-style accessor as a property.

    The accessor returns a wrapper that is `isinstance`-compatible with the
    underlying value type, but calling it emits `DeprecationWarning`.
    """

    def __init__(self, fget: Callable[[Any], Any], message: str) -> None:
        self._fget = fget
        self._message = message
        self.__doc__ = fget.__doc__

    def __get__(self, instance: Any, owner: type | None = None) -> Any:
        if instance is None:  # pragma: no cover
            return self
        return _wrap(self._fget(instance), self._message)


def deprecated_callable_property(message: str) -> Callable[[Callable[[Any], Any]], _DeprecatedCallableProperty]:
    """Decorator for staging a v2 method-to-property migration.

    Args:
        message: deprecation warning emitted when the accessor is called like a method.

    The decorated function takes `(self)` and returns one of `RunUsage`, `RequestUsage`,
    `datetime`, or `ModelResponse`. In v2 this decorator is replaced with `@property`
    (or the method is removed entirely, depending on the card).
    """

    def decorator(fget: Callable[[Any], Any]) -> _DeprecatedCallableProperty:
        return _DeprecatedCallableProperty(fget, message)

    return decorator
