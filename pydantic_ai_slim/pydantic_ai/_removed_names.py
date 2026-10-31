"""Name the V2 replacement when V1 code imports something Pydantic AI V2 removed.

Without this, `from pydantic_ai.mcp import MCPServerStdio` fails with a bare `cannot import name`, which reads
like a broken install. Code that pins `pydantic-ai>=1` with no upper bound gets V2 on a fresh install, and its
users then see our error rather than the library's. See `docs/migration.md` for the full map.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import NoReturn

MIGRATION_URL = 'https://pydantic.dev/docs/ai/overview/migration/'


def removed_names_getattr(module: str, replacements: Mapping[str, str]) -> Callable[[str], NoReturn]:
    """A module-level `__getattr__` for `module` that names the replacement of each name V2 removed from it.

    A removed name raises an `ImportError`, since that is what a `from ... import` surfaces with its message intact:
    an `AttributeError` subclass would be swallowed into Python's generic `cannot import name`. So `hasattr()` on a
    removed name raises rather than returning `False`. Any other name gets the usual `AttributeError`.

    Assign it under `if not TYPE_CHECKING:`, so type checkers keep reporting names the module doesn't have.
    """

    def __getattr__(name: str) -> NoReturn:
        if (replacement := replacements.get(name)) is None:
            raise AttributeError(f'module {module!r} has no attribute {name!r}')
        raise ImportError(
            f'`{module}.{name}` was removed in Pydantic AI V2: {replacement} See {MIGRATION_URL}', name=module
        )

    return __getattr__
