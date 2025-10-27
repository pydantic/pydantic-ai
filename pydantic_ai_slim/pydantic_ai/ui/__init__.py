"""Base classes for UI event stream protocols.

This module provides abstract base classes for implementing UI event stream adapters
that transform Pydantic AI agent events into protocol-specific events (e.g., AG-UI, Vercel AI).
"""

from __future__ import annotations

from .adapter import BaseAdapter, OnCompleteFunc, StateDeps, StateHandler
from .event_stream import SSE_CONTENT_TYPE, BaseEventStream

__all__ = [
    'BaseAdapter',
    'BaseEventStream',
    'SSE_CONTENT_TYPE',
    'StateDeps',
    'StateHandler',
    'OnCompleteFunc',
]
