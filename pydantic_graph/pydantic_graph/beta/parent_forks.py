# pyright: reportWildcardImportFromLibrary=false
"""Deprecated alias for [`pydantic_graph.parent_forks`][pydantic_graph.parent_forks]."""

from __future__ import annotations as _annotations

import warnings as _warnings

from pydantic_graph._warnings import PydanticGraphDeprecationWarning as _DeprecationWarning

_warnings.warn(
    '`pydantic_graph.beta.parent_forks` is deprecated, import from `pydantic_graph.parent_forks` instead.',
    _DeprecationWarning,
    stacklevel=2,
)

from pydantic_graph.parent_forks import *  # noqa: E402, F403
