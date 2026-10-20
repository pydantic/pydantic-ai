"""Compaction clears typed `ToolReturnPart` subclasses, like `FileSystem`'s `ReadFileReturnPart`."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.compaction import ClearToolResults, DeduplicateFileReads
from pydantic_ai_harness.filesystem import FileSystem, ReadFileCallPart, ReadFileReturnPart

READ = '[notes.txt | 1 lines]\n     1\thello\n'


def _file_key(call: ToolCallPart) -> str | None:
    args = call.typed_args if isinstance(call, ReadFileCallPart) else None
    return args['path'] if args is not None else None


def _returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


async def _read_repeatedly(
    tmp_path: Path, capability: AbstractCapability[object], reads: int
) -> list[list[ModelMessage]]:
    """Read `notes.txt` `reads` times, one read per step, and return the history each request sent."""
    (tmp_path / 'notes.txt').write_text('hello\n')
    requests: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(list(messages))
        if len(requests) <= reads:
            return ModelResponse(
                parts=[ToolCallPart('read_file', {'path': 'notes.txt'}, tool_call_id=f'read_{len(requests)}')]
            )
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[FileSystem[object](content_hashes=False), capability])
    await agent.run('Read it', workspace=LocalWorkspaceBackend(tmp_path))
    return requests


async def test_deduplicate_file_reads_clears_superseded_read_file(tmp_path: Path) -> None:
    requests = await _read_repeatedly(tmp_path, DeduplicateFileReads[object](file_key=_file_key), reads=2)

    returns = _returns(requests[-1])
    assert [type(part) for part in returns] == [ReadFileReturnPart, ReadFileReturnPart]
    assert [part.content for part in returns] == ['[superseded file read]', READ]
    assert [part.tool_call_id for part in returns] == ['read_1', 'read_2']

    # The cleared part still round-trips as its typed subclass.
    loaded = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(requests[-1]))
    assert _returns(loaded) == returns


async def test_clear_tool_results_clears_read_file_and_counts_from_it(tmp_path: Path) -> None:
    requests = await _read_repeatedly(tmp_path, ClearToolResults[object](max_messages=4, keep_pairs=1), reads=3)

    # The third request has two read pairs over five messages, so the older read is cleared.
    third = _returns(requests[2])
    assert [type(part) for part in third] == [ReadFileReturnPart, ReadFileReturnPart]
    assert [part.content for part in third] == ['[tool result cleared]', READ]

    # Counting from that cleared read, the fourth request is still under `max_messages`, so the second
    # read is kept rather than cleared one step later.
    fourth = _returns(requests[3])
    assert [part.content for part in fourth] == ['[tool result cleared]', READ, READ]
