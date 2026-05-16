# pyright: reportWildcardImportFromLibrary=false
"""Deprecated alias for [`pydantic_graph.node_types`][pydantic_graph.node_types]."""

from __future__ import annotations as _annotations

import warnings as _warnings

from pydantic_graph._warnings import PydanticGraphDeprecationWarning as _DeprecationWarning

_warnings.warn(
    '`pydantic_graph.beta.node_types` is deprecated, import from `pydantic_graph.node_types` instead.',
    _DeprecationWarning,
    stacklevel=2,
)

from pydantic_graph.node_types import *  # noqa: E402, F403
