# pyright: reportWildcardImportFromLibrary=false
"""Deprecated alias for [`pydantic_graph.paths`][pydantic_graph.paths]."""

from __future__ import annotations as _annotations

import warnings as _warnings

from pydantic_graph._warnings import PydanticGraphDeprecationWarning as _DeprecationWarning

_warnings.warn(
    '`pydantic_graph.beta.paths` is deprecated, import from `pydantic_graph.paths` instead.',
    _DeprecationWarning,
    stacklevel=2,
)

from pydantic_graph.paths import *  # noqa: E402, F403
