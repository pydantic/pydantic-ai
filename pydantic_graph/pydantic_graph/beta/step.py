# pyright: reportWildcardImportFromLibrary=false
"""Deprecated alias for [`pydantic_graph.step`][pydantic_graph.step]."""

from __future__ import annotations as _annotations

import warnings as _warnings

from pydantic_graph._warnings import PydanticGraphDeprecationWarning as _DeprecationWarning

_warnings.warn(
    '`pydantic_graph.beta.step` is deprecated, import from `pydantic_graph.step` instead.',
    _DeprecationWarning,
    stacklevel=2,
)

from pydantic_graph.step import *  # noqa: E402, F403
