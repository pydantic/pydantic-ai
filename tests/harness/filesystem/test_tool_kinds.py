"""`FileSystem`'s tools declare tool kinds, so their parts can be recognized whatever the tools are named."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent, ToolKind
from pydantic_ai.capabilities import AbstractCapability, PrefixTools
from pydantic_ai.messages import (
    BinaryContent,
    BinaryImage,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.filesystem import (
    CREATE_DIRECTORY_KIND,
    EDIT_FILE_KIND,
    FILE_INFO_KIND,
    FILE_SYSTEM_TOOL_NAMES,
    FIND_FILES_KIND,
    GREP_KIND,
    LIST_DIRECTORY_KIND,
    LIST_FILES_KIND,
    READ_FILE_KIND,
    SEARCH_FILES_KIND,
    WRITE_FILE_KIND,
    CreateDirectoryCallPart,
    EditFileCallPart,
    FileInfoCallPart,
    FileSystem,
    FindFilesCallPart,
    GrepCallPart,
    ListDirectoryCallPart,
    ListFilesCallPart,
    ReadFileCallPart,
    ReadFileReturnPart,
    SearchFilesCallPart,
    WriteFileCallPart,
)

KINDS: dict[str, ToolKind] = {
    'read_file': READ_FILE_KIND,
    'write_file': WRITE_FILE_KIND,
    'edit_file': EDIT_FILE_KIND,
    'list_directory': LIST_DIRECTORY_KIND,
    'search_files': SEARCH_FILES_KIND,
    'find_files': FIND_FILES_KIND,
    'create_directory': CREATE_DIRECTORY_KIND,
    'file_info': FILE_INFO_KIND,
    'list_files': LIST_FILES_KIND,
    'grep': GREP_KIND,
}

CALLS: dict[str, tuple[type[ToolCallPart], dict[str, object]]] = {
    'read_file': (ReadFileCallPart, {'path': 'a.txt', 'offset': 1, 'limit': None}),
    'write_file': (WriteFileCallPart, {'path': 'a.txt', 'content': 'x', 'expected_hash': None}),
    'edit_file': (
        EditFileCallPart,
        {'path': 'a.txt', 'replacements': [{'old_text': 'x', 'new_text': 'y'}], 'expected_hash': 'abc'},
    ),
    'list_directory': (ListDirectoryCallPart, {}),
    'search_files': (SearchFilesCallPart, {'pattern': 'x', 'include_glob': '*.py'}),
    'find_files': (FindFilesCallPart, {'pattern': '*.py', 'path': 'src'}),
    'create_directory': (CreateDirectoryCallPart, {'path': 'new'}),
    'file_info': (FileInfoCallPart, {'path': 'a.txt'}),
    'list_files': (ListFilesCallPart, {'glob': '*.py'}),
    'grep': (GrepCallPart, {'pattern': 'x', 'ignore_case': True, 'literal': False, 'context': 2}),
}


async def _tool_defs(tmp_path: Path, capability: AbstractCapability[object]) -> dict[str, str | None]:
    model = TestModel(call_tools=[])
    await Agent(model, capabilities=[capability]).run('Inspect', workspace=LocalWorkspaceBackend(tmp_path))
    assert model.last_model_request_parameters is not None
    return {tool.name: tool.tool_kind for tool in model.last_model_request_parameters.function_tools}


class TestFileSystemToolKinds:
    def test_every_tool_has_a_kind(self) -> None:
        assert set(KINDS) == set(FILE_SYSTEM_TOOL_NAMES)
        assert {name: kind.id for name, kind in KINDS.items()} == {
            name: f'file_system.{name}' for name in FILE_SYSTEM_TOOL_NAMES
        }

    @pytest.mark.parametrize('content_hashes', [True, False])
    async def test_definitions_carry_kinds(self, tmp_path: Path, content_hashes: bool) -> None:
        capability = FileSystem[object](tools=FILE_SYSTEM_TOOL_NAMES, content_hashes=content_hashes)
        assert await _tool_defs(tmp_path, capability) == {name: kind.id for name, kind in KINDS.items()}

    async def test_kinds_survive_a_prefix(self, tmp_path: Path) -> None:
        capability = PrefixTools[object](FileSystem[object](tools=['read_file', 'grep']), prefix='fs')
        assert await _tool_defs(tmp_path, capability) == {
            'fs_read_file': READ_FILE_KIND.id,
            'fs_grep': GREP_KIND.id,
        }

    @pytest.mark.parametrize('name', FILE_SYSTEM_TOOL_NAMES)
    def test_call_part_round_trips(self, name: str) -> None:
        part_type, args = CALLS[name]
        response = ModelResponse(parts=[ToolCallPart(f'renamed_{name}', args, tool_kind=KINDS[name].id)])
        part = response.parts[0]
        assert type(part) is part_type
        assert isinstance(part, ToolCallPart)
        assert part.typed_args == args

        [loaded] = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json([response]))
        assert isinstance(loaded, ModelResponse)
        assert type(loaded.parts[0]) is part_type
        assert loaded.parts[0] == part

    def test_call_part_without_its_required_args_stays_a_plain_call(self) -> None:
        part = ModelResponse(parts=[ToolCallPart('read_file', {'offset': 3}, tool_kind=READ_FILE_KIND.id)]).parts[0]
        assert type(part) is ToolCallPart
        assert part.tool_kind == READ_FILE_KIND.id

    @pytest.mark.parametrize(
        'content',
        [
            '[a.txt | 1 lines]\n     1\thello\n',
            # An image loads back as a `BinaryImage`, the `BinaryContent` subclass core narrows images to.
            ['[logo.png | image/png]', BinaryImage(data=b'\x89PNG\r\n\x1a\n', media_type='image/png')],
        ],
        ids=['text', 'image'],
    )
    def test_read_return_round_trips(self, content: str | list[str | BinaryContent]) -> None:
        request = ModelRequest(
            parts=[ToolReturnPart('fs_read_file', content, tool_call_id='call_1', tool_kind=READ_FILE_KIND.id)]
        )
        part = request.parts[0]
        assert isinstance(part, ReadFileReturnPart)
        assert part.typed_content == content

        [loaded] = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json([request]))
        assert isinstance(loaded, ModelRequest)
        assert isinstance(loaded.parts[0], ReadFileReturnPart)
        assert loaded.parts[0] == part

    def test_failed_read_return_stays_a_plain_return(self) -> None:
        request = ModelRequest(
            parts=[
                ToolReturnPart(
                    'read_file', 'denied', tool_call_id='call_1', tool_kind=READ_FILE_KIND.id, outcome='denied'
                )
            ]
        )
        part = request.parts[0]
        assert type(part) is ToolReturnPart
        assert part.tool_kind == READ_FILE_KIND.id

    async def test_run_messages_narrow_to_typed_parts(self, tmp_path: Path) -> None:
        (tmp_path / 'notes.txt').write_text('hello\n')

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart('read_file', {'path': 'notes.txt'}, tool_call_id='read'),
                        ToolCallPart('write_file', {'path': 'out.txt', 'content': 'bye\n'}, tool_call_id='write'),
                    ]
                )
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(respond), capabilities=[FileSystem(content_hashes=False)])
        result = await agent.run('Copy', workspace=LocalWorkspaceBackend(tmp_path))
        messages = result.all_messages()
        assert isinstance(messages[0], ModelRequest)
        assert isinstance(messages[0].parts[0], UserPromptPart)

        reads = [part for message in messages for part in message.parts if isinstance(part, ReadFileCallPart)]
        assert [read.typed_args for read in reads] == [{'path': 'notes.txt'}]
        writes = [part for message in messages for part in message.parts if isinstance(part, WriteFileCallPart)]
        assert [write.typed_args for write in writes] == [{'path': 'out.txt', 'content': 'bye\n'}]

        returns = {
            part.tool_call_id: part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        assert isinstance(returns['read'], ReadFileReturnPart)
        assert returns['read'].content == '[notes.txt | 1 lines]\n     1\thello\n'
        assert type(returns['write']) is ToolReturnPart
        assert returns['write'].tool_kind == WRITE_FILE_KIND.id
        assert (tmp_path / 'out.txt').read_text() == 'bye\n'
