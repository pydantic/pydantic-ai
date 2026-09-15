"""Events emitted by the FileSystem capability."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from collections.abc import AsyncIterable, AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolResultEvent,
    ModelMessage,
    RetryPromptPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from pydantic_ai_harness.filesystem import (
    MAX_DIFF_SOURCE_CHARS,
    MAX_EVENT_DIFF_CHARS,
    DirectoryCreatedEvent,
    DirectoryListedEvent,
    FileChangeRequestEvent,
    FileEditedEvent,
    FileOperation,
    FileReadEvent,
    FilesSearchedEvent,
    FileSystem,
    FileSystemToolset,
    FileWrittenEvent,
    SearchKind,
)

pytestmark = pytest.mark.anyio

# Mode bits do not bind root, and Windows has no write-only mode.
needs_mode_bits = pytest.mark.skipif(
    os.name == 'nt' or getattr(os, 'geteuid', lambda: 1)() == 0, reason='POSIX mode bits must apply to this process.'
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _has_tool_result(messages: list[ModelMessage]) -> bool:
    return any(isinstance(part, (RetryPromptPart, ToolReturnPart)) for message in messages for part in message.parts)


def _tool_model(tool_name: str, json_args: str) -> FunctionModel:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if not _has_tool_result(messages):
            yield {0: DeltaToolCall(name=tool_name, json_args=json_args, tool_call_id='call_1')}
        else:
            yield 'done'

    return FunctionModel(stream_function=stream)


async def _run_and_collect(
    root: Path,
    tool_name: str,
    json_args: str,
    *,
    denied_patterns: list[str] | None = None,
    listeners: Sequence[AbstractCapability[None]] = (),
    max_results: int = 1000,
) -> list[AgentStreamEvent]:
    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    # Named explicitly: an anonymous capability gets a run-local synthetic id
    # on newer pydantic-ai, which the event assertions could not pin down.
    capability = FileSystem[None](
        root_dir=root,
        denied_patterns=denied_patterns or [],
        max_search_results=max_results,
        max_find_results=max_results,
        id='file_system',
    )
    agent = Agent(_tool_model(tool_name, json_args), deps_type=type(None), capabilities=[capability, *listeners])
    await agent.run('go', event_stream_handler=handler)
    return events


def _tool_result(events: list[AgentStreamEvent]) -> str:
    """What the model was told by the one tool call the run made."""
    results = [event.part for event in events if isinstance(event, FunctionToolResultEvent)]
    assert len(results) == 1
    assert isinstance(results[0], ToolReturnPart)
    return results[0].model_response_str()


def _retry_reason(events: list[AgentStreamEvent]) -> str:
    """What the model was told when its one tool call was rejected."""
    results = [event.part for event in events if isinstance(event, FunctionToolResultEvent)]
    assert len(results) == 1
    assert isinstance(results[0], RetryPromptPart)
    return results[0].model_response()


@dataclass
class WrittenListener(AbstractCapability[None]):
    """Subscribes to writes the way a capability such as `RepoContext` would.

    It reads the file as the event arrives, so a test can pin that the write
    has landed before the notification fires.
    """

    root: Path
    written: list[FileWrittenEvent] = field(default_factory=list[FileWrittenEvent])
    on_disk: list[str] = field(default_factory=list[str])

    @on_event(FileWrittenEvent)
    async def _on_written(self, ctx: RunContext[None], event: FileWrittenEvent) -> None:
        self.written.append(event)
        self.on_disk.append((self.root / event.path).read_text())


@dataclass
class Listener(AbstractCapability[None]):
    """Records every change request and cancels it when `cancel` is set."""

    cancel: bool = False
    reason: str | None = None
    requests: list[FileChangeRequestEvent] = field(default_factory=list[FileChangeRequestEvent])

    @on_event(FileChangeRequestEvent)
    async def _on_request(self, ctx: RunContext[None], event: FileChangeRequestEvent) -> None:
        self.requests.append(event)
        if self.cancel:
            event.cancel(self.reason)


@dataclass(kw_only=True)
class MeddlingListener(AbstractCapability[None]):
    """Runs `act` on the workspace while a change is announced, as another writer could during a slow approval."""

    act: Callable[[], object]

    @on_event(FileChangeRequestEvent)
    async def _on_request(self, ctx: RunContext[None], event: FileChangeRequestEvent) -> None:
        self.act()


@dataclass
class RaisingListener(AbstractCapability[None]):
    """Raises out of the announcement, as a listener with an error in it does."""

    @on_event(FileChangeRequestEvent)
    async def _on_request(self, ctx: RunContext[None], event: FileChangeRequestEvent) -> None:
        raise RuntimeError('no writes today')


def _file_at_leaf(root: Path) -> None:
    (root / 'made').write_text('x')


def _file_at_parent(root: Path) -> None:
    (root / 'sub').rmdir()
    (root / 'sub').write_text('x')


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _root(path: Path) -> str:
    return os.path.realpath(path)


class TestFileSystemEvents:
    async def test_read_emits_normalized_path_and_hash(self, tmp_path: Path) -> None:
        content = 'hello\n'
        (tmp_path / 'target.txt').write_text(content)

        events = await _run_and_collect(tmp_path, 'read_file', '{"path":"sub/../target.txt"}')

        assert [event for event in events if isinstance(event, FileReadEvent)] == [
            FileReadEvent(
                path='target.txt',
                root_dir=_root(tmp_path),
                content_hash=_hash(content),
                capability_id='file_system',
                tool_call_id='call_1',
                tool_name='read_file',
            )
        ]

    async def test_binary_read_emits_raw_content_hash(self, tmp_path: Path) -> None:
        content = b'hello\x00world'
        (tmp_path / 'binary.bin').write_bytes(content)

        events = await _run_and_collect(tmp_path, 'read_file', '{"path":"binary.bin"}')

        read_events = [event for event in events if isinstance(event, FileReadEvent)]
        assert len(read_events) == 1
        assert read_events[0].path == 'binary.bin'
        assert read_events[0].content_hash == hashlib.sha256(content).hexdigest()[:12]

    async def test_list_emits_normalized_path_and_entry_count(self, tmp_path: Path) -> None:
        sub = tmp_path / 'sub'
        sub.mkdir()
        (sub / 'one.txt').write_text('one')
        (sub / 'two.txt').write_text('two')

        events = await _run_and_collect(tmp_path, 'list_directory', '{"path":"other/../sub"}')

        assert [event for event in events if isinstance(event, DirectoryListedEvent)] == [
            DirectoryListedEvent(
                path='sub',
                root_dir=_root(tmp_path),
                entry_count=2,
                capability_id='file_system',
                tool_call_id='call_1',
                tool_name='list_directory',
            )
        ]

    async def test_write_emits_one_written_event(self, tmp_path: Path) -> None:
        (tmp_path / 'target.txt').write_text('old\n')

        events = await _run_and_collect(tmp_path, 'write_file', '{"path":"sub/../target.txt","content":"new\\n"}')

        assert [event for event in events if isinstance(event, FileWrittenEvent)] == [
            FileWrittenEvent(
                path='target.txt',
                root_dir=_root(tmp_path),
                content_hash=_hash('new\n'),
                capability_id='file_system',
                tool_call_id='call_1',
                tool_name='write_file',
            )
        ]

    async def test_edit_emits_one_edited_event_with_the_diff(self, tmp_path: Path) -> None:
        (tmp_path / 'target.txt').write_text('old\n')

        events = await _run_and_collect(
            tmp_path, 'edit_file', '{"path":"sub/../target.txt","old_text":"old","new_text":"new"}'
        )

        # A `FileEditedEvent` is a `FileWrittenEvent`, so a listener for writes sees the edit too.
        assert [event for event in events if isinstance(event, FileWrittenEvent)] == [
            FileEditedEvent(
                path='target.txt',
                root_dir=_root(tmp_path),
                content_hash=_hash('new\n'),
                diff='--- a/target.txt\n+++ b/target.txt\n@@ -1 +1 @@\n-old\n+new',
                truncated=False,
                capability_id='file_system',
                tool_call_id='call_1',
                tool_name='edit_file',
            )
        ]

    async def test_create_directory_emits_one_created_event(self, tmp_path: Path) -> None:
        events = await _run_and_collect(tmp_path, 'create_directory', '{"path":"sub/../new/deep"}')

        assert [event for event in events if isinstance(event, DirectoryCreatedEvent)] == [
            DirectoryCreatedEvent(
                path='new/deep',
                root_dir=_root(tmp_path),
                capability_id='file_system',
                tool_call_id='call_1',
                tool_name='create_directory',
            )
        ]

    async def test_existing_directory_emits_nothing(self, tmp_path: Path) -> None:
        (tmp_path / 'existing').mkdir()

        events = await _run_and_collect(tmp_path, 'create_directory', '{"path":"existing"}')

        assert not any(isinstance(event, (DirectoryCreatedEvent, FileChangeRequestEvent)) for event in events)

    @pytest.mark.parametrize(
        ('tool_name', 'json_args', 'search'),
        [
            ('search_files', '{"pattern":"x","path":"sub"}', 'grep'),
            ('find_files', '{"pattern":"*.py","path":"sub"}', 'find'),
        ],
    )
    async def test_searches_emit_one_searched_event(
        self, tmp_path: Path, tool_name: str, json_args: str, search: SearchKind
    ) -> None:
        sub = tmp_path / 'sub'
        sub.mkdir()
        (sub / 'one.py').write_text('x = 1\nx = 2\n')
        (sub / 'two.py').write_text('y = 1\n')

        events = await _run_and_collect(tmp_path, tool_name, json_args)

        assert [event for event in events if isinstance(event, FilesSearchedEvent)] == [
            FilesSearchedEvent(
                path='sub',
                root_dir=_root(tmp_path),
                pattern='x' if search == 'grep' else '*.py',
                search=search,
                match_count=2,
                truncated=False,
                capability_id='file_system',
                tool_call_id='call_1',
                tool_name=tool_name,
            )
        ]

    async def test_write_crlf_emits_hash_of_written_bytes(self, tmp_path: Path) -> None:
        """A `\\r\\n` write event mirrors the on-disk text, not a translated view."""
        content = 'alpha\r\nbeta\r\n'
        events = await _run_and_collect(tmp_path, 'write_file', '{"path":"crlf.txt","content":"alpha\\r\\nbeta\\r\\n"}')

        assert (tmp_path / 'crlf.txt').read_bytes() == content.encode()
        written = [event for event in events if isinstance(event, FileWrittenEvent)]
        assert len(written) == 1
        assert written[0].content_hash == _hash(content)

    async def test_edit_crlf_preserves_bytes_and_event_hash(self, tmp_path: Path) -> None:
        """Editing a CRLF file leaves `\\r\\n` intact and reports its canonical hash."""
        (tmp_path / 'target.txt').write_bytes(b'old\r\n')

        events = await _run_and_collect(
            tmp_path, 'edit_file', '{"path":"target.txt","old_text":"old","new_text":"new"}'
        )

        assert (tmp_path / 'target.txt').read_bytes() == b'new\r\n'
        written = [event for event in events if isinstance(event, FileWrittenEvent)]
        assert len(written) == 1
        assert written[0].content_hash == _hash('new\r\n')

    @pytest.mark.parametrize(
        ('tool_name', 'json_args'),
        [('search_files', '{"pattern":"x"}'), ('find_files', '{"pattern":"*.py"}')],
    )
    async def test_capped_search_is_marked_truncated(self, tmp_path: Path, tool_name: str, json_args: str) -> None:
        for name in ('a', 'b', 'c'):
            (tmp_path / f'{name}.py').write_text('x = 1\n')

        events = await _run_and_collect(tmp_path, tool_name, json_args, max_results=2)

        searched = [event for event in events if isinstance(event, FilesSearchedEvent)]
        assert [(event.match_count, event.truncated) for event in searched] == [(2, True)]
        assert 'truncated at 2 matches' in _tool_result(events)

    @pytest.mark.parametrize(
        ('tool_name', 'json_args'),
        [('search_files', '{"pattern":"x"}'), ('find_files', '{"pattern":"*.py"}')],
    )
    async def test_exact_fill_is_not_marked_truncated(self, tmp_path: Path, tool_name: str, json_args: str) -> None:
        """Filling the cap exactly is not truncation: nothing was dropped."""
        for name in ('a', 'b'):
            (tmp_path / f'{name}.py').write_text('x = 1\n')

        events = await _run_and_collect(tmp_path, tool_name, json_args, max_results=2)

        searched = [event for event in events if isinstance(event, FilesSearchedEvent)]
        assert [(event.match_count, event.truncated) for event in searched] == [(2, False)]
        assert 'truncated at' not in _tool_result(events)

    async def test_subdirectory_root_is_carried_on_the_event(self, tmp_path: Path) -> None:
        project = tmp_path / 'project'
        project.mkdir()
        (project / 'code.py').write_text('x = 1\n')

        events = await _run_and_collect(project, 'read_file', '{"path":"code.py"}')

        read_events = [event for event in events if isinstance(event, FileReadEvent)]
        assert len(read_events) == 1
        assert read_events[0].path == 'code.py'
        assert read_events[0].root_dir == _root(project)
        assert Path(read_events[0].root_dir, read_events[0].path).read_text() == 'x = 1\n'

    async def test_read_rejected_for_out_of_range_offset_emits_no_event(self, tmp_path: Path) -> None:
        (tmp_path / 'short.txt').write_text('one\ntwo\n')

        events = await _run_and_collect(tmp_path, 'read_file', '{"path":"short.txt","offset":2}')

        assert not any(isinstance(event, FileReadEvent) for event in events)

    async def test_denied_operation_emits_no_capability_event(self, tmp_path: Path) -> None:
        (tmp_path / 'secret.txt').write_text('hidden')

        events = await _run_and_collect(tmp_path, 'read_file', '{"path":"secret.txt"}', denied_patterns=['secret.txt'])

        assert not any(isinstance(event, (FileReadEvent, DirectoryListedEvent, FileWrittenEvent)) for event in events)


class TestFileChangeRequests:
    @pytest.mark.parametrize(
        ('tool_name', 'json_args', 'operation', 'diff'),
        [
            (
                'write_file',
                '{"path":"target.txt","content":"new\\n"}',
                'write',
                '--- a/target.txt\n+++ b/target.txt\n@@ -1 +1 @@\n-old\n+new',
            ),
            (
                'write_file',
                '{"path":"fresh.txt","content":"one\\ntwo\\n"}',
                'write',
                '--- a/fresh.txt\n+++ b/fresh.txt\n@@ -0,0 +1,2 @@\n+one\n+two',
            ),
            (
                'edit_file',
                '{"path":"target.txt","old_text":"old","new_text":"new"}',
                'edit',
                '--- a/target.txt\n+++ b/target.txt\n@@ -1 +1 @@\n-old\n+new',
            ),
            ('create_directory', '{"path":"made"}', 'create_directory', ''),
        ],
    )
    async def test_request_carries_the_proposed_diff(
        self, tmp_path: Path, tool_name: str, json_args: str, operation: FileOperation, diff: str
    ) -> None:
        (tmp_path / 'target.txt').write_text('old\n')
        listener = Listener()

        await _run_and_collect(tmp_path, tool_name, json_args, listeners=[listener])

        path = json.loads(json_args)['path']
        assert listener.requests == [
            FileChangeRequestEvent(
                path=path,
                root_dir=_root(tmp_path),
                operation=operation,
                diff=diff,
                truncated=False,
                capability_id='file_system',
                tool_call_id='call_1',
                tool_name=tool_name,
            )
        ]

    @pytest.mark.parametrize(
        ('tool_name', 'json_args', 'refusal'),
        [
            ('write_file', '{"path":"target.txt","content":"new\\n"}', "['target.txt' was not written: not today]"),
            ('write_file', '{"path":"fresh.txt","content":"new\\n"}', "['fresh.txt' was not written: not today]"),
            (
                'edit_file',
                '{"path":"target.txt","old_text":"old","new_text":"new"}',
                "['target.txt' was not edited: not today]",
            ),
            ('create_directory', '{"path":"made"}', "['made' was not created: not today]"),
        ],
    )
    async def test_cancelled_request_leaves_the_workspace_alone(
        self, tmp_path: Path, tool_name: str, json_args: str, refusal: str
    ) -> None:
        (tmp_path / 'target.txt').write_text('old\n')

        events = await _run_and_collect(
            tmp_path, tool_name, json_args, listeners=[Listener(cancel=True, reason='not today')]
        )

        assert _tool_result(events) == refusal
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ['target.txt']
        assert (tmp_path / 'target.txt').read_text() == 'old\n'
        assert not any(isinstance(event, (FileWrittenEvent, DirectoryCreatedEvent)) for event in events)

    async def test_cancel_without_a_reason_names_the_listener(self, tmp_path: Path) -> None:
        events = await _run_and_collect(
            tmp_path, 'write_file', '{"path":"fresh.txt","content":"new\\n"}', listeners=[Listener(cancel=True)]
        )

        assert _tool_result(events) == "['fresh.txt' was not written: cancelled by a listener]"

    async def test_a_later_bare_cancel_keeps_the_reason(self, tmp_path: Path) -> None:
        """Listeners run in turn; one that cancels without a reason does not erase the reason another gave."""
        events = await _run_and_collect(
            tmp_path,
            'write_file',
            '{"path":"fresh.txt","content":"new\\n"}',
            listeners=[Listener(cancel=True, reason='not today'), Listener(cancel=True)],
        )

        assert _tool_result(events) == "['fresh.txt' was not written: not today]"

    @pytest.mark.parametrize(
        ('tool_name', 'json_args'),
        [
            ('write_file', '{"path":"target.txt","content":"new\\\\n"}'),
            ('write_file', '{"path":"fresh.txt","content":"new\\\\n"}'),
            ('edit_file', '{"path":"target.txt","old_text":"old","new_text":"new"}'),
            ('create_directory', '{"path":"made"}'),
        ],
    )
    async def test_raising_listener_aborts_the_run_and_applies_nothing(
        self, tmp_path: Path, tool_name: str, json_args: str
    ) -> None:
        """A listener that raises aborts the run, as the docs promise, and the change is not applied."""
        (tmp_path / 'target.txt').write_text('old\n')

        written = WrittenListener(root=tmp_path)
        with pytest.raises(RuntimeError, match='no writes today'):
            await _run_and_collect(tmp_path, tool_name, json_args, listeners=[written, RaisingListener()])

        assert written.written == []

        # Nothing was written, created or replaced: only the file the test made.
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ['target.txt']
        assert (tmp_path / 'target.txt').read_text() == 'old\n'

    async def test_denied_write_emits_no_request(self, tmp_path: Path) -> None:
        listener = Listener()

        await _run_and_collect(
            tmp_path,
            'write_file',
            '{"path":"secret.txt","content":"x"}',
            denied_patterns=['secret.txt'],
            listeners=[listener],
        )

        assert listener.requests == []

    @pytest.mark.parametrize(
        ('tool_name', 'json_args'),
        [
            ('edit_file', '{"path":"target.txt","old_text":"old","new_text":"new","expected_hash":"000000000000"}'),
            ('write_file', '{"path":"target.txt","content":"new\\n","expected_hash":"000000000000"}'),
        ],
    )
    async def test_stale_change_emits_no_request(self, tmp_path: Path, tool_name: str, json_args: str) -> None:
        (tmp_path / 'target.txt').write_text('old\n')
        listener = Listener()

        events = await _run_and_collect(tmp_path, tool_name, json_args, listeners=[listener])

        assert listener.requests == []
        assert 'Conflict' in _retry_reason(events)
        assert (tmp_path / 'target.txt').read_text() == 'old\n'

    @pytest.mark.parametrize(
        ('tool_name', 'json_args', 'reason'),
        [
            ('write_file', '{"path":"missing/target.txt","content":"x"}', 'does not exist'),
            ('write_file', '{"path":"file.txt/child.txt","content":"x"}', 'parent that is not a directory'),
            ('create_directory', '{"path":"file.txt"}', 'exists and is not a directory'),
            ('create_directory', '{"path":"file.txt/deeper/still"}', 'parent that is not a directory'),
        ],
    )
    async def test_change_that_cannot_happen_emits_no_request(
        self, tmp_path: Path, tool_name: str, json_args: str, reason: str
    ) -> None:
        (tmp_path / 'file.txt').write_text('x')
        listener = Listener()

        events = await _run_and_collect(tmp_path, tool_name, json_args, listeners=[listener])

        assert listener.requests == []
        assert reason in _retry_reason(events)

    async def test_a_listener_for_writes_receives_the_edit(self, tmp_path: Path) -> None:
        (tmp_path / 'target.txt').write_text('old\n')
        listener = WrittenListener(root=tmp_path)

        await _run_and_collect(
            tmp_path, 'edit_file', '{"path":"target.txt","old_text":"old","new_text":"new"}', listeners=[listener]
        )

        (event,) = listener.written
        assert isinstance(event, FileEditedEvent)
        assert event.path == 'target.txt'
        assert event.diff.endswith('-old\n+new')
        # The serialized kind the docs promise, where `main` emitted `file_system.file_written`.
        assert event.kind == 'file_system.file_edited'
        # The event fires only after the edit has landed on disk.
        assert listener.on_disk == ['new\n']

    @pytest.mark.parametrize('newlines', [254, 255, 256])
    async def test_diff_line_work_bound(self, tmp_path: Path, newlines: int) -> None:
        old = 'a\n' * 255
        new = 'b\n' * newlines
        (tmp_path / 'target.txt').write_text(old)
        listener = Listener()
        events = await _run_and_collect(
            tmp_path,
            'edit_file',
            json.dumps({'path': 'target.txt', 'old_text': old, 'new_text': new}),
            listeners=[listener],
        )
        (request,) = listener.requests
        (edited,) = [event for event in events if isinstance(event, FileEditedEvent)]
        assert request.truncated is (newlines == 256)
        assert edited.truncated is request.truncated
        assert edited.diff == request.diff
        if newlines == 256:
            assert request.diff == '--- a/target.txt\n+++ b/target.txt'
        else:
            assert '@@' in request.diff
        assert (tmp_path / 'target.txt').read_text() == new

    async def test_large_edit_marks_the_edited_event_truncated(self, tmp_path: Path) -> None:
        """An edit whose diff exceeds the bound reports the edited event as cut, not only the request that announced it."""
        (tmp_path / 'big.txt').write_text('a\n')
        listener = WrittenListener(root=tmp_path)

        await _run_and_collect(
            tmp_path,
            'edit_file',
            json.dumps({'path': 'big.txt', 'old_text': 'a', 'new_text': 'z\n' * 4000}),
            listeners=[listener],
        )

        (event,) = listener.written
        assert isinstance(event, FileEditedEvent)
        assert event.truncated is True
        assert len(event.diff) <= MAX_EVENT_DIFF_CHARS
        # Cut on a line boundary, as the request's diff is: whole diff lines only.
        assert event.diff.splitlines()[-1] == '+z'

    async def test_written_event_fires_after_the_content_lands(self, tmp_path: Path) -> None:
        listener = WrittenListener(root=tmp_path)

        await _run_and_collect(tmp_path, 'write_file', '{"path":"target.txt","content":"new\\n"}', listeners=[listener])

        (event,) = listener.written
        assert not isinstance(event, FileEditedEvent)
        assert event.kind == 'file_system.file_written'
        # The event fires only after the write has landed on disk.
        assert listener.on_disk == ['new\n']

    async def test_large_diff_is_cut_and_marked(self, tmp_path: Path) -> None:
        content = ''.join(f'line {i}\n' for i in range(2000))
        listener = Listener()

        events = await _run_and_collect(
            tmp_path, 'write_file', json.dumps({'path': 'big.txt', 'content': content}), listeners=[listener]
        )

        (request,) = listener.requests
        assert request.truncated
        assert len(request.diff) <= MAX_EVENT_DIFF_CHARS
        # Cut on a line boundary: the last kept line is a whole diff line.
        assert request.diff.splitlines()[-1].startswith('+line ')
        assert not request.diff.endswith('\n')
        assert (tmp_path / 'big.txt').read_text() == content
        assert any(isinstance(event, FileWrittenEvent) for event in events)

    @pytest.mark.parametrize('big_side', ['old', 'new'])
    async def test_oversized_change_is_not_diffed(self, tmp_path: Path, big_side: str) -> None:
        """Past `MAX_DIFF_SOURCE_CHARS` on either side nothing is diffed: the event carries the headers, marked as cut."""
        # Well past the bound in bytes too, so the existing file is hashed in more than one chunk.
        huge = 'x\u00e9\u00e9\u00e9' * MAX_DIFF_SOURCE_CHARS
        content = 'small\n' if big_side == 'old' else huge
        if big_side == 'old':
            (tmp_path / 'huge.txt').write_text(huge, encoding='utf-8')
        listener = Listener()

        events = await _run_and_collect(
            tmp_path,
            'write_file',
            json.dumps(
                {'path': 'huge.txt', 'content': content, 'expected_hash': _hash(huge) if big_side == 'old' else None}
            ),
            listeners=[listener],
        )

        (request,) = listener.requests
        assert (request.diff, request.truncated) == ('--- a/huge.txt\n+++ b/huge.txt', True)
        assert (tmp_path / 'huge.txt').read_text(encoding='utf-8') == content
        assert any(isinstance(event, FileWrittenEvent) for event in events)

    @pytest.mark.skipif(os.name == 'nt', reason='Windows rejects the name before the request.')
    async def test_headers_of_an_oversized_name_are_cut_at_the_bound(self, tmp_path: Path) -> None:
        """A quoted name that passes the cap on its own leaves the headers cut at `MAX_EVENT_DIFF_CHARS`."""
        # Each control byte becomes a four-character escape in the quoted name;
        # five components of 220 of them push the two headers past the cap.
        chunk = '\x01' * 220
        parts = [chunk] * 5 + ['f.txt']
        directory = tmp_path
        for part in parts[:-1]:
            directory = directory / part
            try:
                directory.mkdir()
            except OSError:  # pragma: no cover - unreachable in CI; macOS PATH_MAX is 1024 bytes
                pytest.skip('the OS path limit is too short to carry a name this wide (macOS)')
        listener = Listener(cancel=True)

        await _run_and_collect(
            tmp_path,
            'write_file',
            json.dumps({'path': '/'.join(parts), 'content': 'y\n' * 20000}),
            listeners=[listener],
        )

        (request,) = listener.requests
        assert request.truncated
        assert len(request.diff) == MAX_EVENT_DIFF_CHARS
        assert request.diff.startswith('--- "a/')

    @pytest.mark.parametrize(
        ('lines', 'cut'),
        [
            # `-aa` against 2716 `+z` lines makes the diff exactly `MAX_EVENT_DIFF_CHARS`: kept whole.
            (2716, False),
            (2717, True),
        ],
    )
    async def test_event_diff_bound_is_inclusive(self, tmp_path: Path, lines: int, cut: bool) -> None:
        (tmp_path / 'f.txt').write_text('aa\n')
        listener = Listener()

        await _run_and_collect(
            tmp_path, 'write_file', json.dumps({'path': 'f.txt', 'content': 'z\n' * lines}), listeners=[listener]
        )

        (request,) = listener.requests
        assert request.truncated is cut
        assert len(request.diff) == MAX_EVENT_DIFF_CHARS
        assert request.diff.splitlines()[-1] == '+z'

    @pytest.mark.parametrize(
        ('extra', 'diffed'),
        [(0, True), (1, False)],
    )
    async def test_diff_source_bound_is_inclusive(self, tmp_path: Path, extra: int, diffed: bool) -> None:
        """Exactly `MAX_DIFF_SOURCE_CHARS` on a side is still diffed; one more character is not."""
        old = ('x' * 1023 + '\n') * 31 + 'x' * 1021 + '\na' + 'y' * extra + '\n'
        assert len(old) == MAX_DIFF_SOURCE_CHARS + extra
        (tmp_path / 'f.txt').write_text(old)
        listener = Listener()

        await _run_and_collect(
            tmp_path,
            'write_file',
            json.dumps({'path': 'f.txt', 'content': old.replace('\na', '\nb')}),
            listeners=[listener],
        )

        (request,) = listener.requests
        assert request.truncated is not diffed
        assert ('\n+b' in request.diff) is diffed

    async def test_multibyte_text_within_the_bound_is_diffed(self, tmp_path: Path) -> None:
        """The bound is in characters: a file of multibyte characters under it is diffed, whatever its byte size."""
        old = '\u00e9\u00e9\n' * (MAX_DIFF_SOURCE_CHARS // 3)
        (tmp_path / 'target.txt').write_text(old, encoding='utf-8')
        listener = Listener()

        await _run_and_collect(
            tmp_path, 'write_file', '{"path":"target.txt","content":"small\\n"}', listeners=[listener]
        )

        (request,) = listener.requests
        assert request.diff.startswith('--- a/target.txt\n+++ b/target.txt\n@@ -1,')
        assert '\n-\u00e9\u00e9\n' in request.diff
        assert request.truncated

    @pytest.mark.parametrize(
        ('old', 'new', 'hunk'),
        [
            ('text\n', 'text', '@@ -1 +1 @@\n-text\n+text\n\\ No newline at end of file'),
            ('text', 'text\n', '@@ -1 +1 @@\n-text\n\\ No newline at end of file\n+text'),
        ],
    )
    async def test_final_newline_change_has_a_diff(self, tmp_path: Path, old: str, new: str, hunk: str) -> None:
        """A change to the final newline alone is still a change, marked the way `git diff` marks it."""
        (tmp_path / 'target.txt').write_bytes(old.encode())

        events = await _run_and_collect(
            tmp_path, 'edit_file', json.dumps({'path': 'target.txt', 'old_text': old, 'new_text': new})
        )

        (edited,) = [event for event in events if isinstance(event, FileEditedEvent)]
        assert edited.diff == f'--- a/target.txt\n+++ b/target.txt\n{hunk}'
        assert (tmp_path / 'target.txt').read_bytes() == new.encode()

    @pytest.mark.skipif(os.name == 'nt', reason='Windows rejects the name before the request.')
    async def test_control_character_in_the_name_cannot_forge_the_diff(self, tmp_path: Path) -> None:
        """A name with a newline in it is quoted in the headers, as `git diff` quotes it."""
        listener = Listener(cancel=True)

        await _run_and_collect(
            tmp_path,
            'write_file',
            json.dumps({'path': 'we\nird.txt', 'content': 'x\n'}),
            listeners=[listener],
        )

        (request,) = listener.requests
        assert request.path == 'we\nird.txt'
        assert request.diff == '--- "a/we\\nird.txt"\n+++ "b/we\\nird.txt"\n@@ -0,0 +1 @@\n+x'

    @needs_mode_bits
    async def test_unreadable_target_is_announced_as_cut(self, tmp_path: Path) -> None:
        """A file the process can write but not read is still written; the listener is told the diff is not shown."""
        target = tmp_path / 'target.txt'
        target.write_text('old\n')
        target.chmod(0o222)
        listener = Listener()

        events = await _run_and_collect(
            tmp_path, 'write_file', '{"path":"target.txt","content":"new\\n"}', listeners=[listener]
        )

        (request,) = listener.requests
        assert (request.diff, request.truncated) == ('--- a/target.txt\n+++ b/target.txt', True)
        target.chmod(0o644)
        assert target.read_text() == 'new\n'
        assert any(isinstance(event, FileWrittenEvent) for event in events)

    @needs_mode_bits
    async def test_unreadable_target_with_expected_hash_is_refused(self, tmp_path: Path) -> None:
        """An `expected_hash` the process cannot check is an error, not a write that skips the check."""
        target = tmp_path / 'target.txt'
        target.write_text('old\n')
        target.chmod(0o222)
        listener = Listener()

        events = await _run_and_collect(
            tmp_path,
            'write_file',
            json.dumps({'path': 'target.txt', 'content': 'new\n', 'expected_hash': _hash('old\n')}),
            listeners=[listener],
        )

        assert listener.requests == []
        assert 'Permission denied' in _retry_reason(events)
        target.chmod(0o644)
        assert target.read_text() == 'old\n'

    @pytest.mark.skipif(os.name == 'nt', reason='FIFOs require POSIX.')
    async def test_fifo_swapped_before_the_announcement_read_does_not_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The announcement reads the target non-blocking, so a FIFO swapped onto it cannot stall the run."""
        target = tmp_path / 'target.txt'
        target.write_text('old\n')
        original_open = os.open
        swapped = False

        def swap_then_open(path: str | os.PathLike[str], flags: int, mode: int = 0o777) -> int:
            nonlocal swapped
            if not swapped and os.path.realpath(path) == os.path.realpath(target):
                swapped = True
                target.unlink()
                os.mkfifo(target)
                assert flags & os.O_NONBLOCK
            return original_open(path, flags, mode)

        monkeypatch.setattr(os, 'open', swap_then_open)

        events = await _run_and_collect(tmp_path, 'write_file', '{"path":"target.txt","content":"new\\n"}')

        assert swapped
        assert "Path 'target.txt' exists and is not a regular file" in _retry_reason(events)
        assert not any(isinstance(event, FileWrittenEvent) for event in events)

    @pytest.mark.parametrize(
        ('tool_name', 'json_args'),
        [
            ('write_file', '{"path":"target.txt","content":"new\\n"}'),
            ('edit_file', '{"path":"target.txt","old_text":"old","new_text":"new"}'),
        ],
    )
    async def test_file_changed_while_announced_is_refused(
        self, tmp_path: Path, tool_name: str, json_args: str
    ) -> None:
        """The diff a listener approved describes the change applied, however long the listener took."""
        target = tmp_path / 'target.txt'
        target.write_text('old\n')

        events = await _run_and_collect(
            tmp_path, tool_name, json_args, listeners=[MeddlingListener(act=lambda: target.write_text('other\n'))]
        )

        assert 'Conflict' in _retry_reason(events)
        assert target.read_text() == 'other\n'
        assert not any(isinstance(event, FileWrittenEvent) for event in events)

    async def test_invalid_utf8_text_keeps_the_hash_handshake(self, tmp_path: Path) -> None:
        """The guard hashes a text file with an invalid byte the way `read_file` reported it."""
        (tmp_path / 'target.txt').write_bytes(b'a\xffb\n')
        reported = _hash('a\ufffdb\n')

        read_events = await _run_and_collect(tmp_path, 'read_file', '{"path":"target.txt"}')
        (read,) = [event for event in read_events if isinstance(event, FileReadEvent)]
        assert read.content_hash == reported

        events = await _run_and_collect(
            tmp_path, 'write_file', json.dumps({'path': 'target.txt', 'content': 'new\n', 'expected_hash': reported})
        )

        assert (tmp_path / 'target.txt').read_text() == 'new\n'
        assert any(isinstance(event, FileWrittenEvent) for event in events)

    @pytest.mark.parametrize('appeared', ['other\n', ''])
    async def test_file_that_appeared_while_announced_is_refused(self, tmp_path: Path, appeared: str) -> None:
        """A write announced as creating the file does not overwrite one that appeared in the meantime.

        The empty file is the case worth pinning: it hashes to a reachable
        value, so a guard of the empty file's own hash would let it through.
        """
        target = tmp_path / 'fresh.txt'

        events = await _run_and_collect(
            tmp_path,
            'write_file',
            '{"path":"fresh.txt","content":"new\\n"}',
            listeners=[MeddlingListener(act=lambda: target.write_text(appeared))],
        )

        assert 'Conflict' in _retry_reason(events)
        assert target.read_text() == appeared
        assert not any(isinstance(event, FileWrittenEvent) for event in events)

    @pytest.mark.parametrize(
        ('tool_name', 'json_args'),
        [
            ('write_file', '{"path":"sub/target.txt","content":"new\\n"}'),
            ('edit_file', '{"path":"sub/target.txt","old_text":"old","new_text":"new"}'),
            ('create_directory', '{"path":"sub/made"}'),
        ],
    )
    @pytest.mark.parametrize('outside', [True, False])
    async def test_path_replaced_while_announced_is_refused(
        self, tmp_path: Path, tool_name: str, json_args: str, outside: bool
    ) -> None:
        """Swapping a directory on the path for a symlink during the request does not redirect the change."""
        root = tmp_path / 'root'
        sub = root / 'sub'
        sub.mkdir(parents=True)
        (sub / 'target.txt').write_text('old\n')
        elsewhere = tmp_path / 'outside' if outside else root / 'elsewhere'
        elsewhere.mkdir()

        def swap() -> None:
            sub.rename(root / 'moved')
            sub.symlink_to(elsewhere, target_is_directory=True)

        events = await _run_and_collect(root, tool_name, json_args, listeners=[MeddlingListener(act=swap)])

        reason = _retry_reason(events)
        assert (
            'resolves outside the root directory' if outside else 'was replaced while the change was announced'
        ) in reason
        assert list(elsewhere.iterdir()) == []
        assert not any(isinstance(event, (FileWrittenEvent, DirectoryCreatedEvent)) for event in events)

    async def test_directory_that_appeared_while_announced_is_not_reported(self, tmp_path: Path) -> None:
        made = tmp_path / 'made'

        events = await _run_and_collect(
            tmp_path, 'create_directory', '{"path":"made"}', listeners=[MeddlingListener(act=made.mkdir)]
        )

        assert _tool_result(events) == 'Created directory: made'
        assert not any(isinstance(event, DirectoryCreatedEvent) for event in events)

    @pytest.mark.parametrize(
        ('json_args', 'collide', 'reason'),
        [
            ('{"path":"made"}', _file_at_leaf, 'exists and is not a directory'),
            ('{"path":"sub/made"}', _file_at_parent, 'parent that is not a directory'),
        ],
    )
    async def test_collision_that_appeared_while_announced_is_refused(
        self, tmp_path: Path, json_args: str, collide: Callable[[Path], None], reason: str
    ) -> None:
        (tmp_path / 'sub').mkdir()

        events = await _run_and_collect(
            tmp_path, 'create_directory', json_args, listeners=[MeddlingListener(act=lambda: collide(tmp_path))]
        )

        assert reason in _retry_reason(events)
        assert not any(isinstance(event, DirectoryCreatedEvent) for event in events)

    async def test_edit_does_not_recreate_a_file_deleted_while_announced(self, tmp_path: Path) -> None:
        target = tmp_path / 'target.txt'
        target.write_text('old\n')

        events = await _run_and_collect(
            tmp_path,
            'edit_file',
            '{"path":"target.txt","old_text":"old","new_text":"new"}',
            listeners=[MeddlingListener(act=target.unlink)],
        )

        reason = _retry_reason(events)
        assert f'[Errno {errno.ENOENT}]' in reason
        assert "'target.txt'" in reason
        assert _root(tmp_path) not in reason
        assert not target.exists()
        assert not any(isinstance(event, FileWrittenEvent) for event in events)

    async def test_direct_call_asks_nobody(self, tmp_path: Path) -> None:
        toolset = FileSystem[None](root_dir=tmp_path).get_toolset()
        assert isinstance(toolset, FileSystemToolset)

        await toolset.write_file('direct.txt', 'hi\n')
        await toolset.edit_file('direct.txt', 'hi', 'bye')
        await toolset.create_directory('made')

        assert (tmp_path / 'direct.txt').read_text() == 'bye\n'
        assert (tmp_path / 'made').is_dir()
