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
    ToolSearchReturnContent,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.compaction import ClearToolResults, DeduplicateFileReads
from pydantic_ai_harness.filesystem import READ_FILE_KIND, FileSystem, ReadFileCallPart, ReadFileReturnPart

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


async def _first_request(capability: AbstractCapability[object], history: list[ModelMessage]) -> list[ModelMessage]:
    """Continue *history* for one plain-text step and return the history that request sent."""
    requests: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(list(messages))
        return ModelResponse(parts=[TextPart('done')])

    await Agent(FunctionModel(respond), capabilities=[capability]).run('Go on', message_history=history)
    return requests[0]


async def test_clear_tool_results_keeps_a_typed_result_even_when_the_placeholder_fits_its_shape() -> None:
    search_content: ToolSearchReturnContent = {'discovered_tools': [{'name': 'alpha'}]}
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Find a tool')]),
        ModelResponse(parts=[ToolCallPart('search_tools', {'queries': ['alpha']}, tool_call_id='search')]),
        ModelRequest(parts=[ToolSearchReturnPart(content=search_content, tool_call_id='search')]),
        ModelResponse(parts=[TextPart('Found it')]),
    ]
    # A placeholder that is valid `ToolSearchReturnContent` must still not replace the discovered tools.
    capability = ClearToolResults[object](max_messages=1, keep_pairs=0, placeholder='{"discovered_tools": []}')

    [result] = _returns(await _first_request(capability, history))
    assert type(result) is ToolSearchReturnPart
    assert result.content == search_content


async def test_clear_tool_results_does_not_count_from_a_typed_result_that_reads_like_the_placeholder() -> None:
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Read them')]),
        ModelResponse(parts=[ToolCallPart('read_file', {'path': 'a.txt'}, tool_call_id='read_a')]),
        # This file's content happens to be the placeholder text, but compaction never cleared it.
        ModelRequest(
            parts=[
                ToolReturnPart('read_file', '[tool result cleared]', tool_call_id='read_a', tool_kind=READ_FILE_KIND.id)
            ]
        ),
        ModelResponse(parts=[ToolCallPart('read_file', {'path': 'b.txt'}, tool_call_id='read_b')]),
        ModelRequest(parts=[ToolReturnPart('read_file', READ, tool_call_id='read_b', tool_kind=READ_FILE_KIND.id)]),
        ModelResponse(parts=[TextPart('Read both')]),
    ]
    assert isinstance(history[2].parts[0], ReadFileReturnPart)

    # Seven messages with no clear so far exceed `max_messages=5`, so `read_b` is cleared.
    returns = _returns(await _first_request(ClearToolResults[object](max_messages=5, keep_pairs=0), history))
    assert [part.content for part in returns] == ['[tool result cleared]', '[tool result cleared]']
    assert [part.metadata for part in returns] == [
        None,
        {'pydantic-ai-harness.compaction.cleared-tool-result.v1': True},
    ]


async def test_clear_tool_results_marks_cleared_results_alongside_their_metadata() -> None:
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Look things up')]),
        ModelResponse(
            parts=[ToolCallPart('lookup', {}, tool_call_id='dict'), ToolCallPart('lookup', {}, tool_call_id='str')]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('lookup', 'one', tool_call_id='dict', metadata={'source': 'cache'}),
                ToolReturnPart('lookup', 'two', tool_call_id='str', metadata='note'),
            ]
        ),
        ModelResponse(parts=[TextPart('Looked up')]),
    ]

    returns = _returns(await _first_request(ClearToolResults[object](max_messages=1, keep_pairs=0), history))
    assert [part.content for part in returns] == ['[tool result cleared]', '[tool result cleared]']
    # A `dict` gains the mark; other metadata is kept as it is.
    assert [part.metadata for part in returns] == [
        {'source': 'cache', 'pydantic-ai-harness.compaction.cleared-tool-result.v1': True},
        'note',
    ]
