"""Tests for the `ReportStaleFiles` capability."""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import anyio.to_thread
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai._utils import is_str_dict
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, SystemPromptPart, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import FileSystem, ReportStaleFiles
from pydantic_ai_harness.report_stale_files import STALE_FILES_ORIGIN
from tests.function_model_prefix import REQUEST_SCOPED_PROMPT_DROPPED

Step = tuple[Callable[[], object] | None, ToolCallPart | str]
"""What happens in the workspace while the model answers a request, and the model's answer."""

CHANGED = (
    'These files changed in the workspace after you last read them, so what you saw of them is out of date. '
    'Read them again before relying on them or editing them:'
)


def _notices(messages: list[ModelMessage]) -> list[list[str]]:
    """The stale-file notices each model request carried, one list per request."""
    requests: list[list[str]] = [[]]
    for message in messages:
        if isinstance(message, ModelResponse):
            requests.append([])
            continue
        requests[-1].extend(
            part.content
            for part in message.parts
            if isinstance(part, SystemPromptPart)
            and is_str_dict(part.metadata)
            and part.metadata.get('origin') == STALE_FILES_ORIGIN
        )
    return requests[:-1]


def _agent(
    tmp_path: Path,
    *steps: Step,
    capability: ReportStaleFiles[None] | None = None,
    file_system: FileSystem[None] | None = None,
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
        capabilities=[LocalWorkspace(tmp_path), file_system or FileSystem(), capability or ReportStaleFiles()],
    )


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
            [[], [], ['These files you read were deleted from the workspace:\n- `old.py`'], []]
        )

    async def test_unreadable_file_is_skipped_and_checked_again(self, tmp_path: Path) -> None:
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
        assert _notices(result.all_messages()) == [[], [], [], [f'{CHANGED}\n- `data.txt`']]

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
            f'{CHANGED}\n- `config.py`\n- `{(tmp_path / "shared.txt").as_posix()}`'
        ]

    async def test_file_too_large_to_read_is_skipped(self, tmp_path: Path) -> None:
        target = tmp_path / 'big.bin'
        target.write_bytes(b'small')

        def grow() -> None:
            with target.open('r+b') as file:
                file.truncate(65 << 20)

        agent = _agent(tmp_path, (None, _read('big.bin')), (grow, _LIST), (None, 'done'))
        result = await agent.run('go')
        assert _notices(result.all_messages()) == [[], [], []]

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

    async def test_notice_names_at_most_max_reported_files_and_only_plain_paths(self, tmp_path: Path) -> None:
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
        assert _notices(result.all_messages())[-1] == [f'{CHANGED}\n- `a.txt`\n- and 2 more']

    async def test_change_between_runs_of_a_conversation_is_reported(self, tmp_path: Path) -> None:
        config = tmp_path / 'config.py'
        config.write_text('A = 1\n')
        agent = _agent(tmp_path, (None, _read('config.py')), (None, 'done'), (None, 'done again'))
        first = await agent.run('go')
        config.write_text('A = 2\n')
        result = await agent.run('again', message_history=first.all_messages())
        assert _notices(result.all_messages()) == [[], [], [f'{CHANGED}\n- `config.py`']]

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
            await capability.for_run(_ctx(conversation_id=f'other-{index}'))
        config.write_text('A = 2\n')
        result = await agent.run('again', message_history=first.all_messages())
        assert _notices(result.all_messages()) == [[], [], []]

    async def test_run_without_a_conversation_id_keeps_its_own_observations(self) -> None:
        capability = ReportStaleFiles[None]()
        first = await capability.for_run(_ctx(conversation_id=None))
        second = await capability.for_run(_ctx(conversation_id=None))
        assert first is not second
        assert first == second == capability

    async def test_provider_resume_turn_gets_no_notice(self) -> None:
        enqueued: list[object] = []
        ctx = _ctx(conversation_id=None)
        ctx.enqueue = lambda *content, priority='asap': enqueued.extend(content)
        request_context = ModelRequestContext(
            model=TestModel(),
            messages=[ModelResponse(parts=[TextPart('prior')])],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )
        assert await ReportStaleFiles[None]().before_model_request(ctx, request_context) is request_context
        assert enqueued == []

    def test_limits_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match=r'`max_tracked_files` must be at least 1\.'):
            ReportStaleFiles[None](max_tracked_files=0)
        with pytest.raises(ValueError, match=r'`max_reported_files` must be at least 1\.'):
            ReportStaleFiles[None](max_reported_files=0)


def _ctx(*, conversation_id: str | None) -> RunContext[None]:
    return RunContext[None](
        deps=None, model=TestModel(), usage=RunUsage(), prompt=None, conversation_id=conversation_id
    )
