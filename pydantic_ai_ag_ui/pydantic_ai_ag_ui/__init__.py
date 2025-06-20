"""Pydantic AI integration for ag-ui protocol.

This package provides seamless integration between pydantic-ai agents and ag-ui
for building interactive AI applications with streaming event-based communication.
"""

from __future__ import annotations

from .adapter import Adapter
from .consts import SSE_CONTENT_TYPE
from .deps import StateDeps
from .protocols import StateHandler

__all__ = [
    'Adapter',
    'SSE_CONTENT_TYPE',
    'StateDeps',
    'StateHandler',
]
