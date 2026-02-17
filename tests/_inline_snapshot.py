"""Wrapper for inline_snapshot that uses no-ops in CI to speed up test collection.

inline_snapshot is expensive to import and has heavy startup overhead (AST rewriting, etc.)
that significantly slows pytest-xdist worker initialization, especially on Python 3.10.
In CI, snapshots are already frozen in the source code, so we only need `snapshot` to
return its argument for equality checks and `Is` to do the same.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Let pyright always see the real inline_snapshot types
    from inline_snapshot import (
        Is as Is,
        customize_repr as customize_repr,  # pyright: ignore[reportUnknownVariableType]
        snapshot as snapshot,
    )
elif os.environ.get('CI'):

    def snapshot(value: Any = None) -> Any:
        return value

    class Is:
        def __init__(self, value: Any) -> None:
            self.value = value

        def __repr__(self) -> str:
            return f'Is({self.value!r})'

        def __eq__(self, other: object) -> bool:
            return other == self.value

    def customize_repr(func: Any) -> Any:
        return func

else:
    from inline_snapshot import (
        Is as Is,
        customize_repr as customize_repr,  # pyright: ignore[reportUnknownVariableType]
        snapshot as snapshot,
    )
