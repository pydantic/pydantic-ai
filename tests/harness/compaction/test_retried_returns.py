"""Compaction leaves retried tool returns alone: they ask for the call again rather than hold a result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.compaction import ClearToolResults, DeduplicateFileReads
from pydantic_ai_harness.filesystem import FileSystem


def _file_key(call: ToolCallPart) -> str | None:
    # By the raw args, so a retried call whose args failed validation still names its file.
    path = call.args_as_dict().get('path') if call.tool_name == 'read_file' else None
    return path if isinstance(path, str) else None


def _returns(messages: list[ModelMessage]) -> dict[str, ToolReturnPart]:
    return {
        part.tool_call_id: part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }


async def _run(
    tmp_path: Path, capability: AbstractCapability[object], reads: list[tuple[str, dict[str, Any]]]
) -> list[list[ModelMessage]]:
    """Make each `(tool_call_id, args)` `read_file` call in turn, and return the history each request sent."""
    (tmp_path / 'a.py').write_text('a\n')
    (tmp_path / 'b.py').write_text('b\n')
    requests: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(list(messages))
        if len(requests) <= len(reads):
            tool_call_id, args = reads[len(requests) - 1]
            return ModelResponse(parts=[ToolCallPart('read_file', args, tool_call_id=tool_call_id)])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[FileSystem[object](content_hashes=False), capability])
    await agent.run('Read', workspace=LocalWorkspaceBackend(tmp_path))
    return requests


async def test_a_retried_read_does_not_supersede_a_successful_one(tmp_path: Path) -> None:
    requests = await _run(
        tmp_path,
        DeduplicateFileReads[object](file_key=_file_key),
        [('read', {'path': 'a.py'}), ('retry', {'path': 'a.py', 'offset': 'not a number'})],
    )

    content = _returns(requests[1])['read'].content
    returns = _returns(requests[2])
    assert returns['retry'].outcome == 'retried'
    # The retry holds no file content, so the successful read keeps its own.
    assert returns['read'].content == content
    assert returns['read'].content != '[superseded file read]'


READS: list[tuple[str, dict[str, Any]]] = [
    ('read_a', {'path': 'a.py'}),
    ('read_b', {'path': 'b.py'}),
    ('retry', {'path': 'a.py', 'offset': 'not a number'}),
]


async def test_clear_tool_results_does_not_count_a_retried_return_toward_keep_pairs(tmp_path: Path) -> None:
    requests = await _run(tmp_path, ClearToolResults[object](max_messages=3, keep_pairs=1), READS)

    read_b = _returns(requests[2])['read_b'].content
    assert read_b != '[tool result cleared]'

    # The last request clears again. The retry isn't a result, so `read_b` is still the newest one and is kept.
    returns = _returns(requests[3])
    assert returns['retry'].outcome == 'retried'
    assert [returns[key].content for key in ('read_a', 'read_b')] == ['[tool result cleared]', read_b]


async def test_clear_tool_results_does_not_clear_a_retried_return(tmp_path: Path) -> None:
    requests = await _run(tmp_path, ClearToolResults[object](max_messages=1, keep_pairs=0), READS)

    # Each request clears every result so far, but the model still sees why it has to make the call again.
    returns = _returns(requests[3])
    assert [returns[key].content for key in ('read_a', 'read_b')] == ['[tool result cleared]'] * 2
    assert returns['retry'].outcome == 'retried'
    assert returns['retry'].content != '[tool result cleared]'
