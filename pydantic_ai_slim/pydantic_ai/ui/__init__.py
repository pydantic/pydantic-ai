"""Base classes for UI event stream protocols.

This module provides abstract base classes for implementing UI event stream adapters
that transform Pydantic AI agent events into protocol-specific events (e.g., AG-UI, Vercel AI).
"""

from __future__ import annotations

from ._adapter import StateDeps, StateHandler, UIAdapter
from ._event_stream import SSE_CONTENT_TYPE, NativeEvent, OnCompleteFunc, UIEventStream
from ._messages_builder import MessagesBuilder

__all__ = [
    'UIAdapter',
    'UIEventStream',
    'SSE_CONTENT_TYPE',
    'StateDeps',
    'StateHandler',
    'NativeEvent',
    'OnCompleteFunc',
    'MessagesBuilder',
]
