"""Detect repeated-action loops in a run and interrupt them."""

from pydantic_ai_harness.detect_loops._capability import (
    DetectedLoop,
    DetectLoops,
    LoopDetectedError,
    LoopHandler,
    LoopPattern,
)

__all__ = [
    'DetectLoops',
    'DetectedLoop',
    'LoopDetectedError',
    'LoopHandler',
    'LoopPattern',
]
