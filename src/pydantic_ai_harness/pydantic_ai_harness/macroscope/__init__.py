"""Macroscope CLI code-review capability: run `macroscope codereview` from an agent."""

from pydantic_ai_harness.macroscope._capability import Macroscope
from pydantic_ai_harness.macroscope._toolset import (
    MacroscopeIssue,
    MacroscopeReview,
    MacroscopeToolset,
    parse_macroscope_stream,
)

__all__ = [
    'Macroscope',
    'MacroscopeIssue',
    'MacroscopeReview',
    'MacroscopeToolset',
    'parse_macroscope_stream',
]
