"""Detect repeated-action loops in a run, and repeating text within a response, and interrupt them."""

from pydantic_ai_harness.detect_loops._capability import (
    DetectedLoop,
    DetectLoops,
    LoopDetectedError,
    LoopHandler,
    LoopNudgePart,
    LoopPattern,
)
from pydantic_ai_harness.detect_loops._detectors import (
    LoopDetector,
    LoopJudge,
    Repetition,
    RepetitionDetector,
    StreamChannel,
    StreamSample,
)

__all__ = [
    'DetectLoops',
    'DetectedLoop',
    'LoopDetectedError',
    'LoopDetector',
    'LoopHandler',
    'LoopJudge',
    'LoopNudgePart',
    'LoopPattern',
    'Repetition',
    'RepetitionDetector',
    'StreamChannel',
    'StreamSample',
]
