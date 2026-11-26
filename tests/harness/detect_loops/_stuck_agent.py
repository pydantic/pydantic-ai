"""A model stuck calling one tool until `DetectLoops` nudges it, shared by the durable-engine tests."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai_harness.detect_loops import LoopNudgePart

NUDGE = ('repeated_result', ('read_file',))
"""The `pattern` and `tool_names` of the one nudge the stuck model gets."""

model_calls: list[None] = []
"""One entry per model call, so a test can tell a replayed response from a new one."""


def stuck_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    model_calls.append(None)
    if 'Change your approach' in str(messages[-1]):
        return ModelResponse(parts=[TextPart('missing.txt does not exist.')])
    return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'missing.txt'})])


def read_file(path: str) -> str:
    return 'Error: file not found'


def nudges(messages: list[ModelMessage]) -> list[tuple[str, tuple[str, ...]]]:
    return [
        (part.pattern, part.tool_names)
        for message in messages
        for part in message.parts
        if isinstance(part, LoopNudgePart)
    ]
