"""Pydantic AI integration for ag-ui protocol.

This package provides seamless integration between pydantic-ai agents and ag-ui
for building interactive AI applications with streaming event-based communication.
"""

from __future__ import annotations

from .adapter import AdapterAGUI
from .consts import SSE_ACCEPT
from .deps import StateDeps

__all__ = [
    'AdapterAGUI',
    'SSE_ACCEPT',
    'StateDeps',
]
