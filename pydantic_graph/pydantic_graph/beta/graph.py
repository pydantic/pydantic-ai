# pyright: reportWildcardImportFromLibrary=false
"""Deprecated alias for [`pydantic_graph.graph_builder`][pydantic_graph.graph_builder]."""

from __future__ import annotations as _annotations

import warnings as _warnings

from pydantic_graph._warnings import PydanticGraphDeprecationWarning as _DeprecationWarning

_warnings.warn(
    '`pydantic_graph.beta.graph` is deprecated, import from `pydantic_graph.graph_builder` instead.',
    _DeprecationWarning,
    stacklevel=2,
)

from pydantic_graph.graph_builder import *  # noqa: E402, F403
