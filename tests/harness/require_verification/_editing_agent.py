"""A model and tools that edit, try to finish, then verify when `RequireVerification` asks, for the durable tests."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo
from pydantic_ai.tools import Tool
from pydantic_ai_harness.file_system import WRITE_FILE_KIND
from pydantic_ai_harness.require_verification import VerificationRequiredPart

model_calls: list[None] = []
"""One entry per `editing_model` call, so a test can tell a replayed response from a new one."""

_TURNS: list[ModelResponse] = [
    ModelResponse(parts=[ToolCallPart('write_file', {'path': 'app.py', 'content': 'x = 1\n'})]),
    ModelResponse(parts=[TextPart('Done.')]),
    ModelResponse(parts=[ToolCallPart('run_command', {'command': 'pytest -q'})]),
    ModelResponse(parts=[TextPart('Verified.')]),
]


def editing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Writes a file, answers `'Done.'`, runs `pytest -q` when sent back, then answers `'Verified.'`."""
    model_calls.append(None)
    return _TURNS[sum(isinstance(message, ModelResponse) for message in messages)]


def write_file(path: str, content: str) -> str:
    """Write a file."""
    return f'Wrote {path}.'


def run_command(command: str) -> str:
    """Run a shell command."""
    return '(no output)'


tools = [
    Tool(write_file, tool_kind=WRITE_FILE_KIND),
    Tool(run_command, metadata={'code_arg_language': 'shell', 'code_arg_name': 'command'}),
]


def attempts(messages: Sequence[ModelMessage]) -> list[int]:
    """The attempt number of each redirect in `messages`."""
    return [
        part.attempt
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, VerificationRequiredPart)
    ]
