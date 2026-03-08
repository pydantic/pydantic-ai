"""Vercel AI protocol adapter for Pydantic AI agents.

This module provides classes for integrating Pydantic AI agents with the Vercel AI protocol,
enabling streaming event-based communication for interactive AI applications.

Converted to Python from:
https://github.com/vercel/ai/blob/ai%405.0.34/packages/ai/src/ui/ui-messages.ts
"""

from ._adapter import VercelAIAdapter
from ._event_stream import VercelAIEventStream

__all__ = [
    'VercelAIEventStream',
    'VercelAIAdapter',
]
