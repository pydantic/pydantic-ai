"""Testing utilities for pydantic-ai.

This module provides utilities for VCR cassette serialization used in testing.
"""

from .json_body_serializer import deserialize, serialize

__all__ = ['deserialize', 'serialize']
