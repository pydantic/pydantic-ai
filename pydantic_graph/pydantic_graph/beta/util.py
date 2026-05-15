# pyright: reportWildcardImportFromLibrary=false
"""Deprecated alias for [`pydantic_graph.graph_builder.util`][pydantic_graph.graph_builder.util]."""

from __future__ import annotations as _annotations

import warnings as _warnings

from pydantic_graph._warnings import PydanticGraphDeprecationWarning as _DeprecationWarning

_warnings.warn(
    '`pydantic_graph.beta.util` is deprecated, import from `pydantic_graph.graph_builder.util` instead.',
    _DeprecationWarning,
    stacklevel=2,
)

from pydantic_graph.graph_builder.util import *  # noqa: E402, F403
