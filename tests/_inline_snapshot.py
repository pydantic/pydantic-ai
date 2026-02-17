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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from inline_snapshot import (
        Is as Is,
        customize_repr as customize_repr,  # pyright: ignore[reportUnknownVariableType]
        snapshot as snapshot,
    )
elif any(arg.startswith('--inline-snapshot') or arg.startswith('--snap') for arg in sys.argv):
    from inline_snapshot import (
        Is as Is,
        customize_repr as customize_repr,  # pyright: ignore[reportUnknownVariableType]
        snapshot as snapshot,
    )
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
