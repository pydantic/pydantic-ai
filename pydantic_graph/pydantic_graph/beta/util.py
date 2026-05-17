# pyright: reportWildcardImportFromLibrary=false
"""Deprecated alias for [`pydantic_graph.util`][pydantic_graph.util]."""

from __future__ import annotations as _annotations

import warnings as _warnings

from pydantic_graph._warnings import PydanticGraphDeprecationWarning as _DeprecationWarning

_warnings.warn(
    '`pydantic_graph.beta.util` is deprecated, import from `pydantic_graph.util` instead.',
    _DeprecationWarning,
    stacklevel=2,
)

from pydantic_graph.util import *  # noqa: E402, F403
