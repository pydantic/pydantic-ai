"""Wrapper for inline_snapshot that uses lightweight stubs by default.

inline_snapshot is expensive to import and has heavy startup overhead (AST rewriting, etc.)
that significantly slows pytest-xdist worker initialization, especially on Python 3.10.

When no --inline-snapshot flag is passed to pytest, we use lightweight stubs:
- snapshot(value) returns a proxy that compares using the value, warning on mismatch
- snapshot() with no args raises an error directing you to use --inline-snapshot=create
- Is(value) compares using the underlying value
- customize_repr is a no-op decorator

Pass --inline-snapshot=<mode> or --snap/--snap-fix to use the real library.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from warnings import catch_warnings, simplefilter

if TYPE_CHECKING:
    from inline_snapshot import (
        Is as Is,
        customize_repr as customize_repr,  # pyright: ignore[reportUnknownVariableType]
        snapshot as snapshot,
    )
    from inline_snapshot.extra import raises as raises, warns as warns
elif any(arg.startswith('--inline-snapshot') or arg.startswith('--snap') for arg in sys.argv):
    from inline_snapshot import (
        Is as Is,
        customize_repr as customize_repr,  # pyright: ignore[reportUnknownVariableType]
        snapshot as snapshot,
    )
    from inline_snapshot.extra import raises as raises, warns as warns
else:

    class _SnapshotProxy:
        """Proxy that compares using the snapshot value, warning on mismatch."""

        def __init__(self, value: Any) -> None:
            self._value = value

        def __repr__(self) -> str:
            return repr(self._value)

        def __eq__(self, other: object) -> bool:
            result = other == self._value
            if not result:
                warnings.warn(
                    f'Snapshot mismatch: {other!r} != {self._value!r}\n'
                    'Re-run with --inline-snapshot=fix to update snapshots.',
                    stacklevel=2,
                )
            return result  # type: ignore[return-value]

    _MISSING = object()

    def snapshot(value: Any = _MISSING) -> Any:
        if value is _MISSING:
            raise RuntimeError(
                'snapshot() called without a value. Run with --inline-snapshot=create to generate initial snapshots.'
            )
        return _SnapshotProxy(value)

    class Is:
        def __init__(self, value: Any) -> None:
            self.value = value

        def __repr__(self) -> str:
            return f'Is({self.value!r})'

        def __eq__(self, other: object) -> bool:
            return other == self.value

    def customize_repr(func: Any) -> Any:
        return func

    @contextmanager
    def warns(expected_warnings: Any, /, include_line: bool = False, include_file: bool = False) -> Iterator[None]:
        with catch_warnings(record=True) as caught:
            simplefilter('always')
            yield
        formatted: list[str] = []
        for w in caught:
            parts: list[str] = []
            if include_file:
                parts.append(w.filename)
            if include_line:
                parts.append(str(w.lineno))
            parts.append(f'{w.category.__name__}: {w.message}')
            formatted.append(', '.join(parts) if len(parts) > 1 else parts[0])
        assert formatted == expected_warnings

    @contextmanager
    def raises(exception: Any) -> Iterator[None]:
        try:
            yield
        except Exception as e:
            actual = f'{type(e).__name__}: {e}'
        else:
            actual = '<no exception>'
        assert actual == exception
