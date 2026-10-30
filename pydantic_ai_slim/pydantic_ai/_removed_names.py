"""Name the V2 replacement when V1 code imports something Pydantic AI V2 removed.

Without this, `from pydantic_ai.mcp import MCPServerStdio` fails with a bare `cannot import name`, which reads
like a broken install. Code that pins `pydantic-ai>=1` with no upper bound gets V2 on a fresh install, and its
users then see our error rather than the library's. See `docs/migration.md` for the full map.
"""

from __future__ import annotations

from collections.abc import Mapping

MIGRATION_URL = 'https://pydantic.dev/docs/ai/overview/migration/'


def removed_name_error(module: str, name: str, replacements: Mapping[str, str]) -> Exception:
    """The error a module-level `__getattr__` raises for `name`.

    An `ImportError` naming the replacement when V2 removed `name`, since that is what a `from ... import`
    surfaces with its message intact. An `AttributeError` subclass would be swallowed into Python's generic
    `cannot import name`, so `hasattr()` on a removed name raises rather than returning `False`. Any other name
    gets the usual `AttributeError`.
    """
    if (replacement := replacements.get(name)) is None:
        return AttributeError(f'module {module!r} has no attribute {name!r}')
    return ImportError(
        f'`{module}.{name}` was removed in Pydantic AI V2: {replacement} See {MIGRATION_URL}', name=module
    )
