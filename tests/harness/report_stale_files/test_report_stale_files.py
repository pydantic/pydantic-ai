"""Tests for the `ReportStaleFiles` capability."""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path

import anyio.to_thread
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    Authority,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import FileEntry, LocalWorkspaceBackend, Workspace, WorkspaceError
from pydantic_ai_harness import FileSystem, ReportStaleFiles
from pydantic_ai_harness.report_stale_files import StaleFilesPart
from tests.function_model_prefix import REQUEST_SCOPED_PROMPT_DROPPED
from tests.harness._recording_durability import RecordingDurability
from tests.harness.file_system.test_windows_paths import DriveBackend

Step = tuple[Callable[[], object] | None, ToolCallPart | str]
"""What happens in the workspace while the model answers a request, and the model's answer."""

DELETED = 'These files you read or wrote were deleted from the workspace:'
CHANGED = (
    'These files changed in the workspace after you last read or wrote them, so what you saw of them is out of date. '
    'Read them again before relying on them or editing them:'
)


def _notices(messages: list[ModelMessage]) -> list[list[str]]:
    """The stale-file notices each model request carried, one list per request."""
    requests: list[list[str]] = [[]]
    for message in messages:
        if isinstance(message, ModelResponse):
            requests.append([])
            continue
        requests[-1].extend(part.content for part in message.parts if isinstance(part, StaleFilesPart))
    return requests[:-1]


def _agent(
    tmp_path: Path,
    *steps: Step,
    capability: ReportStaleFiles[None] | None = None,
    file_system: FileSystem[None] | None = None,
    authority: Authority = 'operator',
) -> Agent[None, str]:
    """An agent whose model takes the next step, after the step's action changes the workspace."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        action, reply = steps[sum(isinstance(message, ModelResponse) for message in messages)]
        if action is not None:
            await anyio.to_thread.run_sync(action)
        if isinstance(reply, str):
            yield reply
        else:
            yield {0: DeltaToolCall(name=reply.tool_name, json_args=reply.args_as_json_str())}

    return Agent(
        FunctionModel(stream_function=respond),
        deps_type=type(None),
        capabilities=[
            LocalWorkspace(tmp_path, authority=authority),
            file_system or FileSystem(),
            capability or ReportStaleFiles(),
        ],
    )


class FlakyBackend(LocalWorkspaceBackend):
    """A local workspace whose reads fail while `failing` is set."""

    failing = False

    async def read_bytes(self, path: str) -> bytes:
        if self.failing:
            raise WorkspaceError('The workspace is unavailable.')
        return await super().read_bytes(path)


class RealpathOnceBackend(LocalWorkspaceBackend):
    """A local workspace that fails to resolve `data.txt` the second time it's asked, after `FileSystem` resolved it."""

    calls = 0

    async def realpath(self, path: str) -> str:
        if path.endswith('data.txt'):
            self.calls += 1
            if self.calls == 2:
                raise WorkspaceError('The workspace is unavailable.')
        return await super().realpath(path)


class SizeBackend(LocalWorkspaceBackend):
    """A local workspace whose `stat` reports `size` for every file, as a backend that can't measure it might."""

    def __init__(self, *, working_dir: Path, size: int | None) -> None:
        super().__init__(working_dir=working_dir)
        self.size = size

    async def stat(self, path: str) -> FileEntry:
        return replace(await super().stat(path), size=self.size)


def _read(path: str) -> ToolCallPart:
    return ToolCallPart('read_file', {'path': path})


_LIST = ToolCallPart('list_directory', {'path': '.'})


class TestReportStaleFiles:
    @pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
    async def test_changed_file_is_reported_until_read_again(self, tmp_path: Path) -> None:
        config = tmp_path / 'config.py'
        config.write_text('A = 1\n')
        agent = _agent(
            tmp_path,
            (None, _read('config.py')),
            (lambda: config.write_text('A = 2\n'), _LIST),
            (None, _LIST),
            (None, _read('config.py')),
            (None, 'done'),
        )
        result = await agent.run('go')
        notice = f'{CHANGED}\n- `config.py`'
        assert _notices(result.all_messages()) == [[], [], [notice], [notice], []]

    async def test_own_writes_and_edits_are_not_reported(self, tmp_path: Path) -> None:
        (tmp_path / 'notes.md').write_text('Sunny.\n')
        agent = _agent(
            tmp_path,
            (None, _read('notes.md')),
            (None, ToolCallPart('edit_file', {'path': 'notes.md', 'old_text': 'Sunny.', 'new_text': 'Rainy.'})),
            (None, ToolCallPart('write_file', {'path': 'new.md', 'content': 'New.\n'})),
            (lambda: (tmp_path / 'new.md').write_text('Changed.\n'), _LIST),
            (None, 'done'),
        )
        result = await agent.run('go')
        assert _notices(result.all_messages()) == [[], [], [], [], [f'{CHANGED}\n- `new.md`']]

    @pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
    async def test_deleted_file_is_reported_once(self, tmp_path: Path) -> None:
        (tmp_path / 'old.py').write_text('x = 1\n')
        agent = _agent(
            tmp_path,
            (None, _read('old.py')),
            (lambda: (tmp_path / 'old.py').unlink(), _LIST),
            (None, _LIST),
            (None, 'done'),
        )
        result = await agent.run('go')
        assert _notices(result.all_messages()) == snapshot(
            [[], [], ['These files you read or wrote were deleted from the workspace:\n- `old.py`'], []]
        )

    @pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
    async def test_file_replaced_by_a_directory_is_reported_deleted_once(self, tmp_path: Path) -> None:
        target = tmp_path / 'data.txt'
        target.write_text('one\n')

        def replace_with_directory() -> None:
            target.unlink()
            target.mkdir()

        def restore_changed() -> None:
            shutil.rmtree(target)
            target.write_text('two\n')

        agent = _agent(
            tmp_path,
            (None, _read('data.txt')),
            (replace_with_directory, _LIST),
            (restore_changed, _LIST),
            (None, 'done'),
        )
        result = await agent.run('go')
        assert _notices(result.all_messages()) == [[], [], [f'{DELETED}\n- `data.txt`'], []]

    async def test_paths_are_named_as_the_file_tools_take_them(self, tmp_path: Path) -> None:
        """The working directory is inside the `FileSystem` root, so event paths are relative to a different root."""
        app = tmp_path / 'app'
        app.mkdir()
        (app / 'config.py').write_text('A = 1\n')
        (tmp_path / 'shared.txt').write_text('one\n')

        def change_both() -> None:
            (app / 'config.py').write_text('A = 2\n')
            (tmp_path / 'shared.txt').write_text('two\n')

        agent = _agent(
            app,
            (None, _read('config.py')),
            (None, _read('../shared.txt')),
            (change_both, _LIST),
            (None, 'done'),
            file_system=FileSystem(root_dir=tmp_path),
        )
        result = await agent.run('go')
        assert _notices(result.all_messages())[-1] == [
            f'{CHANGED}\n- `{(tmp_path / "shared.txt").as_posix()}`\n- `config.py`'
        ]

    async def test_windows_workspace_paths_are_named_relative_to_the_working_directory(self, tmp_path: Path) -> None:
        (tmp_path / 'project' / 'src').mkdir(parents=True)
        app = tmp_path / 'project' / 'src' / 'app.py'
        app.write_text('x = 1\n')
        agent = _agent(
            tmp_path,
            (None, _read('src\\app.py')),
            (lambda: app.write_text('x = 2\n'), _LIST),
            (None, 'done'),
        )
        result = await agent.run('go', workspace=DriveBackend(tmp_path))
        assert _notices(result.all_messages())[-1] == [f'{CHANGED}\n- `src/app.py`']

    async def test_files_past_the_byte_budget_are_not_checked(self, tmp_path: Path) -> None:
        """The most recently read file is checked first; the older one no longer fits in what is left."""
        names = ('older.txt', 'newer.txt')
        for name in names:
            (tmp_path / name).write_text('one\n')

        def change_both() -> None:
            for name in names:
                (tmp_path / name).write_text('two\n')

        agent = _agent(
            tmp_path,
            *((None, _read(name)) for name in names),
            (change_both, _LIST),
            (None, 'done'),
            capability=ReportStaleFiles(max_checked_bytes=6),
        )
        result = await agent.run('go')
        assert _notices(result.all_messages())[-1] == [f'{CHANGED}\n- `newer.txt`']

    async def test_files_that_exactly_fill_the_byte_budget_are_all_checked(self, tmp_path: Path) -> None:
        """The two most recently read files fill the budget, so the oldest isn't checked on that request."""
        names = ('oldest.txt', 'older.txt', 'newer.txt')
        for name in names:
            (tmp_path / name).write_text('one\n')

        def change_both() -> None:
            for name in names:
                (tmp_path / name).write_text('two\n')

        agent = _agent(
            tmp_path,
            *((None, _read(name)) for name in names),
            (change_both, _LIST),
            (None, 'done'),
            capability=ReportStaleFiles(max_checked_bytes=8),
        )
        result = await agent.run('go')
        assert _notices(result.all_messages())[-1] == [f'{CHANGED}\n- `newer.txt`\n- `older.txt`']

    async def test_file_of_unknown_size_is_not_read(self, tmp_path: Path) -> None:
        """Without a size up front, the read couldn't be kept to the budget."""
        target = tmp_path / 'data.txt'
        target.write_text('one\n')
        agent = _agent(tmp_path, (None, _read('data.txt')), (lambda: target.write_text('two\n'), _LIST), (None, 'done'))
        result = await agent.run('go', workspace=SizeBackend(working_dir=tmp_path, size=None))
        assert _notices(result.all_messages()) == [[], [], []]

    async def test_file_that_reads_past_the_byte_budget_is_dropped(self, tmp_path: Path) -> None:
        """A file that grew past the budget after its `stat` isn't hashed, and isn't read again on later requests."""
        target = tmp_path / 'data.txt'
        target.write_text('one\n')
        agent = _agent(
            tmp_path,
            (None, _read('data.txt')),
            (lambda: target.write_text('grown well past the budget\n'), _LIST),
            (lambda: target.write_text('two\n'), _LIST),
            (None, 'done'),
            capability=ReportStaleFiles(max_checked_bytes=10),
        )
        result = await agent.run('go', workspace=SizeBackend(working_dir=tmp_path, size=1))
        assert _notices(result.all_messages()) == [[], [], [], []]

    async def test_one_notice_names_changed_then_deleted_files_within_one_budget(self, tmp_path: Path) -> None:
        for name in ('changed.txt', 'deleted.txt', 'gone.txt'):
            (tmp_path / name).write_text('one\n')

        def change_and_delete() -> None:
            (tmp_path / 'changed.txt').write_text('two\n')
            (tmp_path / 'deleted.txt').unlink()
            (tmp_path / 'gone.txt').unlink()

        agent = _agent(
            tmp_path,
            (None, _read('gone.txt')),
            (None, _read('deleted.txt')),
            (None, _read('changed.txt')),
            (change_and_delete, _LIST),
            (None, 'done'),
            capability=ReportStaleFiles(max_reported_files=2),
        )
        result = await agent.run('go')
        assert _notices(result.all_messages())[-1] == snapshot(
            [
                """\
These files changed in the workspace after you last read or wrote them, so what you saw of them is out of date. Read them again before relying on them or editing them:
- `changed.txt`

These files you read or wrote were deleted from the workspace:
- `deleted.txt`
- and 1 more\
"""
            ]
        )

    async def test_file_the_workspace_fails_to_read_is_checked_again(self, tmp_path: Path) -> None:
        target = tmp_path / 'data.txt'
        target.write_text('one\n')
        backend = FlakyBackend(working_dir=tmp_path)

        def change_and_fail() -> None:
            target.write_text('two\n')
            backend.failing = True

        def recover() -> None:
            backend.failing = False

        agent = _agent(
            tmp_path,
            (None, _read('data.txt')),
            (change_and_fail, _LIST),
            (recover, _LIST),
            (None, 'done'),
        )
        result = await agent.run('go', workspace=backend)
        assert _notices(result.all_messages()) == [[], [], [], [f'{CHANGED}\n- `data.txt`']]

    async def test_file_replaced_by_a_link_out_of_the_root_is_not_read(self, tmp_path: Path) -> None:
        """A link the file tools would refuse to follow isn't followed to check the file either."""
        project = tmp_path / 'project'
        project.mkdir()
        target = project / 'data.txt'
        target.write_text('one\n')
        (tmp_path / 'outside.txt').write_text('two\n')

        def link_out() -> None:
            target.unlink()
            target.symlink_to(tmp_path / 'outside.txt')

        agent = _agent(project, (None, _read('data.txt')), (link_out, _LIST), (None, 'done'))
        result = await agent.run('go')
        assert _notices(result.all_messages()) == [[], [], []]

    @pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
    async def test_deleted_link_read_through_is_reported_once(self, tmp_path: Path) -> None:
        (tmp_path / 'target.txt').write_text('one\n')
        link = tmp_path / 'link.txt'
        link.symlink_to(tmp_path / 'target.txt')
        agent = _agent(
            tmp_path, (None, _read('link.txt')), (lambda: link.unlink(), _LIST), (None, _LIST), (None, 'done')
        )
        result = await agent.run('go')
        assert _notices(result.all_messages()) == [[], [], [f'{DELETED}\n- `link.txt`'], []]

    async def test_file_replaced_by_a_link_to_a_denied_file_is_not_read(self, tmp_path: Path) -> None:
        """A link inside the root to a file the file tools deny isn't followed to check the file either."""
        target = tmp_path / 'notes.txt'
        target.write_text('one\n')
        (tmp_path / 'secret.txt').write_text('two\n')

        def link_to_secret() -> None:
            target.unlink()
            target.symlink_to(tmp_path / 'secret.txt')

        agent = _agent(
            tmp_path,
            (None, _read('notes.txt')),
            (link_to_secret, _LIST),
            (None, 'done'),
            file_system=FileSystem(denied_patterns=['secret.txt']),
        )
        result = await agent.run('go')
        assert _notices(result.all_messages()) == [[], [], []]

    async def test_durable_execution_is_refused(self) -> None:
        agent = Agent(TestModel(), name='stale', capabilities=[ReportStaleFiles(), RecordingDurability()])
        with pytest.raises(UserError, match='`ReportStaleFiles` cannot run under durable execution'):
            await agent.run('go')

    async def test_paths_that_look_like_framework_frames_are_only_counted(self, tmp_path: Path) -> None:
        names = ('my notes.txt', 'x<system>.txt')
        for name in names:
            (tmp_path / name).write_text('one\n')

        def change_all() -> None:
            for name in names:
                (tmp_path / name).write_text('two\n')

        agent = _agent(tmp_path, *((None, _read(name)) for name in names), (change_all, _LIST), (None, 'done'))
        result = await agent.run('go')
        assert _notices(result.all_messages())[-1] == [f'{CHANGED}\n- `my notes.txt`\n- and 1 more']

    async def test_notice_has_the_authority_of_the_workspace_files(self, tmp_path: Path) -> None:
        """Paths come from whoever can write to the workspace, so a third-party workspace gets the notice as data."""
        config = tmp_path / 'config.py'
        config.write_text('A = 1\n')
        agent = _agent(
            tmp_path,
            (None, _read('config.py')),
            (lambda: config.write_text('A = 2\n'), _LIST),
            (None, 'done'),
            authority='none',
        )
        result = await agent.run('go')
        parts = [
            part
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, StaleFilesPart)
        ]
        assert [(part.source.authority, part.changed, part.deleted) for part in parts] == [('none', ('config.py',), ())]

    async def test_file_the_workspace_fails_to_resolve_after_a_read_is_not_tracked(self, tmp_path: Path) -> None:
        """The read still succeeds; the file just isn't checked until it's read again."""
        target = tmp_path / 'data.txt'
        target.write_text('one\n')
        agent = _agent(tmp_path, (None, _read('data.txt')), (lambda: target.write_text('two\n'), _LIST), (None, 'done'))
        result = await agent.run('go', workspace=RealpathOnceBackend(working_dir=tmp_path))
        assert _notices(result.all_messages()) == [[], [], []]
        returns = [
            part.content
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert 'one' in str(returns[0])

    async def test_least_recently_observed_file_is_dropped_past_the_limit(self, tmp_path: Path) -> None:
        for name in ('a.txt', 'b.txt'):
            (tmp_path / name).write_text('one\n')

        def change_both() -> None:
            for name in ('a.txt', 'b.txt'):
                (tmp_path / name).write_text('two\n')

        agent = _agent(
            tmp_path,
            (None, _read('a.txt')),
            (None, _read('b.txt')),
            (change_both, _LIST),
            (None, 'done'),
            capability=ReportStaleFiles(max_tracked_files=1),
        )
        result = await agent.run('go')
        assert _notices(result.all_messages())[-1] == [f'{CHANGED}\n- `b.txt`']

    async def test_notice_names_at_most_max_reported_files(self, tmp_path: Path) -> None:
        names = ('my notes.txt', 'a.txt', 'b.txt')
        for name in names:
            (tmp_path / name).write_text('one\n')

        def change_all() -> None:
            for name in names:
                (tmp_path / name).write_text('two\n')

        agent = _agent(
            tmp_path,
            *((None, _read(name)) for name in names),
            (change_all, _LIST),
            (None, 'done'),
            capability=ReportStaleFiles(max_reported_files=1),
        )
        result = await agent.run('go')
        assert _notices(result.all_messages())[-1] == [f'{CHANGED}\n- `b.txt`\n- and 2 more']

    async def test_change_between_runs_of_a_conversation_is_reported(self, tmp_path: Path) -> None:
        config = tmp_path / 'config.py'
        config.write_text('A = 1\n')
        agent = _agent(tmp_path, (None, _read('config.py')), (None, 'done'), (None, 'done again'))
        first = await agent.run('go')
        config.write_text('A = 2\n')
        result = await agent.run('again', message_history=first.all_messages())
        assert _notices(result.all_messages()) == [[], [], [f'{CHANGED}\n- `config.py`']]

    @pytest.mark.moves_cache_prefix(
        reason='the conversation moves to another workspace, so the working-directory line names a new directory'
    )
    async def test_conversation_continued_in_another_workspace_is_not_checked(self, tmp_path: Path) -> None:
        """A file recorded in the first workspace isn't checked from a run in the second, even after it changes."""
        first_dir, second_dir = tmp_path / 'first', tmp_path / 'second'
        for directory, content in ((first_dir, 'A = 1\n'), (second_dir, 'A = 2\n')):
            directory.mkdir()
            (directory / 'config.py').write_text(content)
        agent = _agent(first_dir, (None, _read('config.py')), (None, 'done'), (None, 'done again'))
        first = await agent.run('go')
        (first_dir / 'config.py').write_text('A = 3\n')
        result = await agent.run(
            'again', message_history=first.all_messages(), workspace=LocalWorkspaceBackend(working_dir=second_dir)
        )
        assert _notices(result.all_messages()) == [[], [], []]

    async def test_other_conversations_do_not_share_observations(self, tmp_path: Path) -> None:
        config = tmp_path / 'config.py'
        config.write_text('A = 1\n')
        agent = _agent(tmp_path, (None, _read('config.py')), (None, 'done'), (None, 'done again'))
        await agent.run('go', conversation_id='first')
        config.write_text('A = 2\n')
        result = await agent.run('go', conversation_id='second')
        assert _notices(result.all_messages()) == [[], []]

    async def test_least_recently_used_conversation_is_forgotten(self, tmp_path: Path) -> None:
        config = tmp_path / 'config.py'
        config.write_text('A = 1\n')
        capability = ReportStaleFiles[None]()
        agent = _agent(tmp_path, (None, _read('config.py')), (None, 'done'), (None, 'done'), capability=capability)
        first = await agent.run('go')
        for index in range(1024):
            await capability.for_run(_ctx(tmp_path, f'other-{index}'))
        config.write_text('A = 2\n')
        result = await agent.run('again', message_history=first.all_messages())
        assert _notices(result.all_messages()) == [[], [], []]

    async def test_run_without_a_conversation_id_keeps_its_own_observations(self, tmp_path: Path) -> None:
        capability = await _observed(tmp_path)
        request: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('go')])]
        assert await _enqueued(capability, tmp_path, None, request) == []
        assert await _enqueued(capability, tmp_path, 'observed', request) == [f'{CHANGED}\n- `config.py`']

    async def test_provider_resume_turn_gets_no_notice(self, tmp_path: Path) -> None:
        capability = await _observed(tmp_path)
        resume: list[ModelMessage] = [ModelResponse(parts=[TextPart('prior')])]
        assert await _enqueued(capability, tmp_path, 'observed', resume) == []
        request: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('go')])]
        assert await _enqueued(capability, tmp_path, 'observed', request) == [f'{CHANGED}\n- `config.py`']

    def test_limits_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match=r'`max_tracked_files` must be at least 1\.'):
            ReportStaleFiles[None](max_tracked_files=0)
        with pytest.raises(ValueError, match=r'`max_reported_files` must be at least 1\.'):
            ReportStaleFiles[None](max_reported_files=0)
        with pytest.raises(ValueError, match=r'`max_checked_bytes` must be at least 1\.'):
            ReportStaleFiles[None](max_checked_bytes=0)


def _ctx(tmp_path: Path, conversation_id: str | None) -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        conversation_id=conversation_id,
        workspace=Workspace(LocalWorkspaceBackend(working_dir=tmp_path)),
    )


async def _observed(tmp_path: Path) -> ReportStaleFiles[None]:
    """A capability that recorded `config.py` in the conversation `'observed'`, after which the file changed."""
    config = tmp_path / 'config.py'
    config.write_text('A = 1\n')
    capability = ReportStaleFiles[None]()
    await _agent(tmp_path, (None, _read('config.py')), (None, 'done'), capability=capability).run(
        'go', conversation_id='observed'
    )
    config.write_text('A = 2\n')
    return capability


async def _enqueued(
    capability: ReportStaleFiles[None], tmp_path: Path, conversation_id: str | None, messages: list[ModelMessage]
) -> list[str]:
    """The notices a run of `conversation_id` enqueues before a request whose history is `messages`."""
    ctx = _ctx(tmp_path, conversation_id)
    run = await capability.for_run(ctx)
    enqueued: list[object] = []
    ctx.enqueue = lambda *content, priority='asap': enqueued.extend(content)
    request_context = ModelRequestContext(
        model=TestModel(), messages=messages, model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    assert await run.before_model_request(ctx, request_context) is request_context
    return [part.content for part in enqueued if isinstance(part, StaleFilesPart)]
