# pyright: reportWildcardImportFromLibrary=false
"""Deprecated alias for [`pydantic_graph.basenode`][pydantic_graph.basenode].

The contents of this module moved to [`pydantic_graph.basenode`][pydantic_graph.basenode].
Importing from `pydantic_graph.nodes` still works but emits a
[`PydanticGraphDeprecationWarning`][pydantic_graph.PydanticGraphDeprecationWarning].
"""

from __future__ import annotations as _annotations

import warnings as _warnings

from pydantic_graph._warnings import PydanticGraphDeprecationWarning as _DeprecationWarning

_warnings.warn(
    '`pydantic_graph.nodes` is deprecated, import from `pydantic_graph.basenode` (or `pydantic_graph`) instead.',
    _DeprecationWarning,
    stacklevel=2,
)

from pydantic_graph.basenode import (  # noqa: E402
    BaseNode,
    DepsT,
    Edge,
    End,
    GraphRunContext,
    NodeDef,
    NodeRunEndT,
    RunEndT,
    StateT,
    generate_snapshot_id,
)

__all__ = (
    'GraphRunContext',
    'BaseNode',
    'End',
    'Edge',
    'NodeDef',
    'DepsT',
    'StateT',
    'RunEndT',
    'NodeRunEndT',
    'generate_snapshot_id',
)
