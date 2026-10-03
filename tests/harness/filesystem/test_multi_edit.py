"""The opt-in `multi_edit` tool: ordered first-occurrence or replace-all edits to one file, written all at once."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path

import anyio.to_thread
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import AgentStreamEvent, ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace, WorkspaceBackend
from pydantic_ai_harness.filesystem import (
    DEFAULT_TOOL_NAMES,
    FILE_SYSTEM_TOOL_NAMES,
    EditOp,
    FileChangeRequestEvent,
    FileEditedEvent,
    FileSystem,
    FileSystemToolset,
)

from .._tool_calls import call_tool

TOOLS = (*DEFAULT_TOOL_NAMES, 'multi_edit')


@dataclass
class Listener(AbstractCapability[None]):
    """Records every change request, cancels it when `cancel` is set, and runs `act` while it is held."""

    cancel: bool = False
    act: Callable[[], object] | None = None
    requests: list[FileChangeRequestEvent] = field(default_factory=list[FileChangeRequestEvent])

    @on_event(FileChangeRequestEvent)
    async def _on_request(self, ctx: RunContext[None], event: FileChangeRequestEvent) -> None:
        self.requests.append(event)
        if self.act is not None:
            await anyio.to_thread.run_sync(self.act)
        if self.cancel:
            event.cancel('not today')


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _toolset(root: Path) -> FileSystemToolset[None]:
    toolset = FileSystem[None](root_dir=root, tools=TOOLS).get_toolset()
    assert isinstance(toolset, FileSystemToolset)
    return toolset


async def _call(
    root: Path,
    arguments: dict[str, object],
    *,
    listeners: tuple[AbstractCapability[None], ...] = (),
    **settings: object,
) -> str:
    capability = FileSystem[None](root_dir=root, tools=TOOLS, **settings)  # pyright: ignore[reportArgumentType]
    return await call_tool([capability, *listeners], 'multi_edit', arguments, workspace=LocalWorkspaceBackend(root))


async def _tool_names(workspace: WorkspaceBackend, **settings: object) -> list[str]:
    model = TestModel(call_tools=[])
    capability = FileSystem[None](**settings)  # pyright: ignore[reportArgumentType]
    await Agent(model, deps_type=type(None), capabilities=[capability]).run('Inspect', workspace=workspace)
    assert model.last_model_request_parameters is not None
    return [tool.name for tool in model.last_model_request_parameters.function_tools]


class TestMultiEdit:
    async def test_edits_apply_in_order(self, tmp_path: Path) -> None:
        """Each edit sees the result of the previous one."""
        edited = 'Bye, universe!\n'
        (tmp_path / 'f.txt').write_text('Hello, world!\n')
        result = await _call(
            tmp_path,
            {
                'path': 'f.txt',
                'edits': [
                    {'old_string': 'world', 'new_string': 'universe'},
                    {'old_string': 'Hello, universe', 'new_string': 'Bye, universe'},
                ],
            },
        )
        assert result == f'Applied 2 edits (2 replacements) to f.txt. [hash:{_hash(edited)}]'
        assert (tmp_path / 'f.txt').read_text() == edited

    async def test_first_occurrence_need_not_be_unique(self, tmp_path: Path) -> None:
        """A repeated match replaces the first occurrence, where `edit_file` would ask for more context."""
        (tmp_path / 'f.txt').write_text('foo bar foo\n')
        await _call(tmp_path, {'path': 'f.txt', 'edits': [{'old_string': 'foo', 'new_string': 'baz'}]})
        assert (tmp_path / 'f.txt').read_text() == 'baz bar foo\n'

    async def test_replace_all(self, tmp_path: Path) -> None:
        (tmp_path / 'f.txt').write_text('foo bar foo\n')
        result = await _call(
            tmp_path, {'path': 'f.txt', 'edits': [{'old_string': 'foo', 'new_string': 'x', 'replace_all': True}]}
        )
        assert result.startswith('Applied 1 edits (2 replacements) to f.txt.')
        assert (tmp_path / 'f.txt').read_text() == 'x bar x\n'

    @pytest.mark.parametrize(
        'edits,message',
        [
            ([], 'edits is empty'),
            ([{'old_string': '', 'new_string': 'x'}], 'edits[0]: old_string is empty. No changes were written.'),
            (
                [{'old_string': 'one', 'new_string': 'uno'}, {'old_string': 'missing', 'new_string': 'x'}],
                'edits[1]: old_string not found in f.txt. No changes were written.',
            ),
        ],
    )
    async def test_failed_batch_leaves_file_unchanged(
        self, tmp_path: Path, edits: list[dict[str, object]], message: str
    ) -> None:
        """All or nothing: a failing edit discards the ones before it and asks the model to retry."""
        (tmp_path / 'f.txt').write_text('one two\n')
        assert message in await _call(tmp_path, {'path': 'f.txt', 'edits': edits})
        assert (tmp_path / 'f.txt').read_text() == 'one two\n'

    async def test_crlf_and_hash_handshake(self, tmp_path: Path) -> None:
        """The hash `read_file` reports is accepted, and `\\r\\n` survives the rewrite."""
        (tmp_path / 'f.txt').write_bytes(b'a\r\nb\r\n')
        read = await call_tool(
            [FileSystem[None](root_dir=tmp_path)],
            'read_file',
            {'path': 'f.txt'},
            workspace=LocalWorkspaceBackend(tmp_path),
        )
        reported = _hash('a\r\nb\r\n')
        assert f'hash:{reported}' in read
        await _call(
            tmp_path, {'path': 'f.txt', 'edits': [{'old_string': 'a', 'new_string': 'A'}], 'expected_hash': reported}
        )
        assert (tmp_path / 'f.txt').read_bytes() == b'A\r\nb\r\n'

    @pytest.mark.parametrize(
        'path,content,arguments,message',
        [
            ('f.txt', b'old\n', {'expected_hash': '000000000000'}, 'Conflict'),
            ('.env', b'old\n', {}, 'protected'),
            ('blob.bin', b'old\0', {}, 'blob.bin is a binary file; multi_edit only edits text files.'),
            ('ghost.txt', None, {}, 'File not found: ghost.txt'),
            ('../outside.txt', None, {}, 'outside root_dir'),
        ],
    )
    async def test_refused_edit_writes_nothing(
        self, tmp_path: Path, path: str, content: bytes | None, arguments: dict[str, object], message: str
    ) -> None:
        if content is not None:
            (tmp_path / path).write_bytes(content)
        result = await _call(
            tmp_path, {'path': path, 'edits': [{'old_string': 'old', 'new_string': 'new'}], **arguments}
        )
        assert message in result
        if content is not None:
            assert (tmp_path / path).read_bytes() == content

    async def test_content_hashes_off(self, tmp_path: Path) -> None:
        (tmp_path / 'f.txt').write_text('one\n')
        result = await _call(
            tmp_path, {'path': 'f.txt', 'edits': [{'old_string': 'one', 'new_string': 'uno'}]}, content_hashes=False
        )
        assert result == 'Applied 1 edits (1 replacements) to f.txt.'
        assert (tmp_path / 'f.txt').read_text() == 'uno\n'

    async def test_direct_method(self, tmp_path: Path) -> None:
        edited = 'b c\n'
        (tmp_path / 'f.txt').write_text('a a\n')
        workspace = LocalWorkspaceBackend(tmp_path)
        toolset = _toolset(tmp_path)
        result = await toolset.multi_edit(
            'f.txt',
            [EditOp(old_string='a', new_string='b'), EditOp(old_string='a', new_string='c', replace_all=True)],
            expected_hash=_hash('a a\n'),
            workspace=workspace,
        )
        assert result == f'Applied 2 edits (2 replacements) to f.txt. [hash:{_hash(edited)}]'
        with pytest.raises(ModelRetry, match=r'edits\[0\]: old_string not found'):
            await toolset.multi_edit('f.txt', [EditOp(old_string='zzz', new_string='y')], workspace=workspace)
        assert (tmp_path / 'f.txt').read_text() == edited


class TestMultiEditRegistration:
    async def test_opt_in(self, tmp_path: Path) -> None:
        """Not in the default tool set, so existing `FileSystem` users see the same tools as before."""
        workspace = LocalWorkspaceBackend(tmp_path)
        assert 'multi_edit' not in await _tool_names(workspace)
        assert 'multi_edit' in FILE_SYSTEM_TOOL_NAMES
        assert 'multi_edit' in await _tool_names(workspace, tools=TOOLS)

    async def test_hidden_from_read_only(self, tmp_path: Path) -> None:
        workspace = LocalWorkspaceBackend(tmp_path)
        assert 'multi_edit' not in await _tool_names(workspace, tools=TOOLS, read_only=True)
        assert 'multi_edit' not in await _tool_names(ReadOnlyWorkspace(Workspace(workspace)), tools=TOOLS)

    @pytest.mark.parametrize('content_hashes', [True, False])
    async def test_schema(self, tmp_path: Path, content_hashes: bool) -> None:
        model = TestModel(call_tools=[])
        capability = FileSystem[None](tools=TOOLS, content_hashes=content_hashes)
        await Agent(model, deps_type=type(None), capabilities=[capability]).run(
            'Inspect', workspace=LocalWorkspaceBackend(tmp_path)
        )
        assert model.last_model_request_parameters is not None
        (tool,) = [t for t in model.last_model_request_parameters.function_tools if t.name == 'multi_edit']
        assert ('expected_hash' in tool.parameters_json_schema['properties']) is content_hashes
        edit_schema = tool.parameters_json_schema['$defs']['EditOp']
        assert edit_schema['required'] == ['old_string', 'new_string']
        assert edit_schema['properties']['replace_all'] == {'default': False, 'type': 'boolean'}


class TestMultiEditEvents:
    async def test_announced_and_reported_as_an_edit(self, tmp_path: Path) -> None:
        (tmp_path / 'f.txt').write_text('old\nold\n')
        listener = Listener()
        events: list[AgentStreamEvent] = []

        async def handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                events.append(event)

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) > 1:
                yield 'done'
                return
            args = {'path': 'f.txt', 'edits': [{'old_string': 'old', 'new_string': 'new', 'replace_all': True}]}
            yield {0: DeltaToolCall(name='multi_edit', json_args=json.dumps(args), tool_call_id='call_1')}

        capability = FileSystem[None](root_dir=tmp_path, tools=TOOLS, id='file_system')
        agent = Agent(FunctionModel(stream_function=stream), deps_type=type(None), capabilities=[capability, listener])
        await agent.run('go', event_stream_handler=handler, workspace=LocalWorkspaceBackend(tmp_path))

        diff = '--- a/f.txt\n+++ b/f.txt\n@@ -1,2 +1,2 @@\n-old\n-old\n+new\n+new'
        assert listener.requests == [
            FileChangeRequestEvent(
                path='f.txt',
                root_dir=str(tmp_path),
                operation='edit',
                diff=diff,
                truncated=False,
                capability_id='file_system',
                tool_call_id='call_1',
                tool_name='multi_edit',
            )
        ]
        assert [event for event in events if isinstance(event, FileEditedEvent)] == [
            FileEditedEvent(
                path='f.txt',
                root_dir=str(tmp_path),
                content_hash=_hash('new\nnew\n'),
                diff=diff,
                truncated=False,
                capability_id='file_system',
                tool_call_id='call_1',
                tool_name='multi_edit',
            )
        ]

    async def test_cancelled_request_leaves_the_file_alone(self, tmp_path: Path) -> None:
        (tmp_path / 'f.txt').write_text('old\n')
        result = await _call(
            tmp_path,
            {'path': 'f.txt', 'edits': [{'old_string': 'old', 'new_string': 'new'}]},
            listeners=(Listener(cancel=True),),
        )
        assert result == "['f.txt' was not edited: not today]"
        assert (tmp_path / 'f.txt').read_text() == 'old\n'

    async def test_file_changed_while_announced_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / 'f.txt'
        target.write_text('old\n')
        result = await _call(
            tmp_path,
            {'path': 'f.txt', 'edits': [{'old_string': 'old', 'new_string': 'new'}]},
            listeners=(Listener(act=lambda: target.write_text('other\n')),),
        )
        assert 'Conflict' in result
        assert target.read_text() == 'other\n'
