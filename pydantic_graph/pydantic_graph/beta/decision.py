# pyright: reportWildcardImportFromLibrary=false
"""Deprecated alias for [`pydantic_graph.decision`][pydantic_graph.decision]."""

from __future__ import annotations as _annotations

import warnings as _warnings

from pydantic_graph._warnings import PydanticGraphDeprecationWarning as _DeprecationWarning

_warnings.warn(
    '`pydantic_graph.beta.decision` is deprecated, import from `pydantic_graph.decision` instead.',
    _DeprecationWarning,
    stacklevel=2,
)

from pydantic_graph.decision import *  # noqa: E402, F403
