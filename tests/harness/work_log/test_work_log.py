"""Tests for `WorkLog`: what it records, the tools that read it back, and its stores."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import anyio
import pytest
from pydantic import TypeAdapter

import pydantic_ai_harness.work_log
from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, ValidatedToolArgs
from pydantic_ai.exceptions import CallDeferred, ModelRetry, ToolFailed, UserError
from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    OperatorSource,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace, WorkspaceError
from pydantic_ai_harness import CodeMode, ToolBridge, ToolOutputLimits, WorkLog
from pydantic_ai_harness.compaction import ResetNoticePart
from pydantic_ai_harness.tool_output_limits import Band, LocalFileOverflowStore, Truncate
from pydantic_ai_harness.work_log import (
    WORK_LOG_READ_KIND,
    WORK_LOG_SEARCH_KIND,
    FileWorkLogStore,
    InMemoryWorkLogStore,
    WorkLogEntry,
    WorkLogRead,
    WorkLogRecord,
    WorkLogSearchResult,
    WorkLogStore,
    WorkspaceWorkLogStore,
)
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.harness.conftest import IsStr

from .._recording_durability import RecordingDurability

_CONVERSATION = 'conv'
_SEARCH = TypeAdapter(WorkLogSearchResult)
_READ = TypeAdapter(WorkLogRead)


def _record(
    key: str,
    content: str = 'result',
    *,
    tool_name: str | None = 'lookup',
    args: str | None = '{}',
    run_id: str | None = 'run-1',
) -> WorkLogRecord:
    return WorkLogRecord(
        key=key,
        run_id=run_id,
        step=1,
        context=0,
        kind='tool',
        tool_name=tool_name,
        tool_call_id=key,
        parent_tool_call_id=None,
        args=args,
        outcome='success',
        content=content,
        chars=len(content),
        timestamp='2026-11-13T00:00:00+00:00',
    )


def _scripted(*turns: Sequence[ModelResponsePart]) -> FunctionModel:
    """A model that sends each turn's parts in order, then answers `done`."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        turn = sum(isinstance(message, ModelResponse) for message in messages)
        return ModelResponse(parts=list(turns[turn]) if turn < len(turns) else [TextPart('done')])

    return FunctionModel(respond)


def _tool_entries(entries: list[WorkLogEntry]) -> list[tuple[str | None, str | None, str]]:
    return [(entry['tool_name'], entry['outcome'], entry['content']) for entry in entries if entry['kind'] == 'tool']


def _returned(messages: Sequence[ModelMessage], tool_call_id: str) -> object:
    """What the model was sent back for one call: its return value, or its retry message."""
    (content,) = [
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_call_id == tool_call_id
    ]
    return content


async def _call_tools(
    store: WorkLogStore, *calls: tuple[str, dict[str, object]], capabilities: Sequence[AbstractCapability[None]] = ()
) -> list[object]:
    """Make each work log tool call in its own turn, over `store`'s log; return what the model saw for each."""
    turns = [[ToolCallPart(name, args, tool_call_id=f'call-{index}')] for index, (name, args) in enumerate(calls)]
    agent = Agent(_scripted(*turns), deps_type=type(None), capabilities=[WorkLog(store), *capabilities])
    result = await agent.run('Look it up', conversation_id=_CONVERSATION)
    return [_returned(result.all_messages(), f'call-{index}') for index in range(len(calls))]


def lookup(city: str) -> str:
    return f'sunny in {city}'


class TestWorkLog:
    async def test_an_entry_records_the_call_and_its_result(self) -> None:
        store = InMemoryWorkLogStore()
        agent = Agent(
            _scripted([ToolCallPart('lookup', {'city': 'Paris'}, tool_call_id='c1')]),
            capabilities=[WorkLog(store)],
            tools=[lookup],
        )

        result = await agent.run('Weather?', conversation_id=_CONVERSATION)

        run_id = result.run_id
        assert await store.read(_CONVERSATION) == [
            {
                'key': f'{run_id}:1:c1:lookup:0:success',
                'run_id': run_id,
                'step': 1,
                'context': 0,
                'kind': 'tool',
                'tool_name': 'lookup',
                'tool_call_id': 'c1',
                'parent_tool_call_id': None,
                'args': '{"city":"Paris"}',
                'outcome': 'success',
                'content': 'sunny in Paris',
                'chars': 14,
                'timestamp': IsStr(),
                'seq': 1,
            },
            {
                'key': f'{run_id}:2:response:0',
                'run_id': run_id,
                'step': 2,
                'context': 0,
                'kind': 'text',
                'tool_name': None,
                'tool_call_id': None,
                'parent_tool_call_id': None,
                'args': None,
                'outcome': None,
                'content': 'done',
                'chars': 4,
                'timestamp': IsStr(),
                'seq': 2,
            },
        ]

    @pytest.mark.parametrize('work_log_first', [True, False])
    async def test_a_result_is_logged_whole_while_tool_output_limits_spills_it(
        self, tmp_path: Path, work_log_first: bool
    ) -> None:
        big = '\n'.join(f'line {index}' for index in range(3_000))

        def dump() -> str:
            return big

        store = InMemoryWorkLogStore()
        capabilities: list[AbstractCapability[None]] = [
            WorkLog(store),
            ToolOutputLimits(store=LocalFileOverflowStore(tmp_path)),
        ]
        agent = Agent(
            _scripted([ToolCallPart('dump', {}, tool_call_id='c1')]),
            deps_type=type(None),
            capabilities=capabilities if work_log_first else capabilities[::-1],
            tools=[dump],
        )

        result = await agent.run('Dump it', conversation_id=_CONVERSATION)

        assert str(_returned(result.all_messages(), 'c1')).startswith('[Tool output too large')
        assert _tool_entries(await store.read(_CONVERSATION)) == [('dump', 'success', big)]

    async def test_a_retry_a_failure_and_a_handled_failure_are_logged(self) -> None:
        attempts: list[str] = []

        def flaky() -> str:
            attempts.append('flaky')
            if len(attempts) == 1:
                raise ModelRetry('try again')
            return 'ok'

        def gone() -> str:
            raise ToolFailed('gone for good')

        store = InMemoryWorkLogStore()
        agent = Agent(
            _scripted(
                [ToolCallPart('flaky', {}, tool_call_id='c1')],
                [ToolCallPart('flaky', {}, tool_call_id='c2')],
                [ToolCallPart('gone', {}, tool_call_id='c3')],
            ),
            capabilities=[WorkLog(store)],
            tools=[flaky, gone],
        )

        await agent.run('Go', conversation_id=_CONVERSATION)

        assert _tool_entries(await store.read(_CONVERSATION)) == [
            ('flaky', 'retried', 'try again'),
            ('flaky', 'success', 'ok'),
            ('gone', 'failed', 'ToolFailed: gone for good'),
        ]

    async def test_a_result_an_outer_hook_rejects_is_logged_as_retried_too(self) -> None:
        class RejectFirst(AbstractCapability[None]):
            async def after_tool_execute(
                self,
                ctx: RunContext[None],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: ValidatedToolArgs,
                result: object,
            ) -> object:
                if call.tool_call_id == 'c1':
                    raise ModelRetry('not good enough')
                return result

        def lookup() -> str:
            return 'ok'

        store = InMemoryWorkLogStore()
        agent = Agent(
            _scripted([ToolCallPart('lookup', {}, tool_call_id='c1')], [ToolCallPart('lookup', {}, tool_call_id='c2')]),
            deps_type=type(None),
            capabilities=[RejectFirst(), WorkLog(store)],
            tools=[lookup],
        )

        await agent.run('Go', conversation_id=_CONVERSATION)

        # The outer hook runs after `WorkLog` recorded the result, and the model is told to retry.
        assert _tool_entries(await store.read(_CONVERSATION)) == [
            ('lookup', 'success', 'ok'),
            ('lookup', 'retried', 'not good enough'),
            ('lookup', 'success', 'ok'),
        ]

    async def test_an_exception_is_logged_as_failed_by_type_and_still_raised(self) -> None:
        def broken() -> str:
            raise ValueError('postgres://user:secret@db')

        store = InMemoryWorkLogStore()
        agent = Agent(
            _scripted([ToolCallPart('broken', {}, tool_call_id='c1')]), capabilities=[WorkLog(store)], tools=[broken]
        )

        with pytest.raises(ValueError, match='secret'):
            await agent.run('Go', conversation_id=_CONVERSATION)

        assert _tool_entries(await store.read(_CONVERSATION)) == [('broken', 'failed', 'ValueError')]

    async def test_an_error_an_outer_capability_recovers_is_logged_once_as_success(self) -> None:
        class Recover(AbstractCapability[None]):
            async def on_tool_execute_error(
                self,
                ctx: RunContext[None],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: ValidatedToolArgs,
                error: Exception,
            ) -> object:
                return 'recovered'

        def broken() -> str:
            raise ValueError('boom')

        store = InMemoryWorkLogStore()
        agent = Agent(
            _scripted([ToolCallPart('broken', {}, tool_call_id='c1')]),
            deps_type=type(None),
            capabilities=[Recover(), WorkLog(store)],
            tools=[broken],
        )

        await agent.run('Go', conversation_id=_CONVERSATION)

        assert _tool_entries(await store.read(_CONVERSATION)) == [('broken', 'success', 'recovered')]

    async def test_a_deferred_call_is_not_logged(self) -> None:
        def later() -> str:
            raise CallDeferred

        store = InMemoryWorkLogStore()
        agent = Agent(
            _scripted([ToolCallPart('later', {}, tool_call_id='c1')]),
            capabilities=[WorkLog(store)],
            tools=[later],
            output_type=[str, DeferredToolRequests],
        )

        result = await agent.run('Go', conversation_id=_CONVERSATION)

        assert isinstance(result.output, DeferredToolRequests)
        assert await store.read(_CONVERSATION) == []

    @pytest.mark.parametrize(
        ('record_text', 'record_thinking', 'expected'),
        [
            (True, False, [('text', 'hello')]),
            (True, True, [('thinking', 'pondering'), ('text', 'hello')]),
            (False, False, []),
            (False, True, [('thinking', 'pondering')]),
        ],
    )
    async def test_text_is_logged_and_thinking_only_when_asked(
        self, record_text: bool, record_thinking: bool, expected: list[tuple[str, str]]
    ) -> None:
        store = InMemoryWorkLogStore()
        agent = Agent(
            _scripted([ThinkingPart('pondering'), TextPart('hello')]),
            capabilities=[WorkLog(store, record_text=record_text, record_thinking=record_thinking)],
        )

        await agent.run('Hi', conversation_id=_CONVERSATION)

        assert [(entry['kind'], entry['content']) for entry in await store.read(_CONVERSATION)] == expected

    async def test_the_work_logs_own_tools_are_not_logged(self) -> None:
        store = InMemoryWorkLogStore()
        await store.append(_CONVERSATION, _record('earlier'))

        await _call_tools(store, ('search_work_log', {}), ('read_work_log', {'seq': 1}))

        assert [(entry['kind'], entry['content']) for entry in await store.read(_CONVERSATION)] == [
            ('tool', 'result'),
            ('text', 'done'),
        ]

    async def test_files_are_named_by_media_type_rather_than_logged(self) -> None:
        png = BinaryContent(data=b'\x89PNG', media_type='image/png')

        def captioned() -> list[object]:
            return ['a caption', png]

        def rich() -> ToolReturn[str]:
            return ToolReturn(return_value='value', content=[TextContent('first'), 'second', png, CachePoint()])

        def visible() -> ToolReturn[str]:
            return ToolReturn(return_value='value', content='what the model sees')

        def structured() -> ToolReturn[dict[str, int]]:
            return ToolReturn(return_value={'count': 1})

        def linked() -> ImageUrl:
            return ImageUrl('https://example.com/picture')

        store = InMemoryWorkLogStore()
        names = ['captioned', 'rich', 'visible', 'structured', 'linked']
        agent = Agent(
            _scripted(*([ToolCallPart(name, {}, tool_call_id=name)] for name in names)),
            capabilities=[WorkLog(store)],
            tools=[captioned, rich, visible, structured, linked],
        )

        await agent.run('Go', conversation_id=_CONVERSATION)

        assert _tool_entries(await store.read(_CONVERSATION)) == [
            ('captioned', 'success', 'a caption\n[image/png content not logged]'),
            ('rich', 'success', 'value\nfirst\nsecond\n[image/png content not logged]'),
            ('visible', 'success', 'value\nwhat the model sees'),
            ('structured', 'success', '{\n  "count": 1\n}'),
            ('linked', 'success', '[image-url content not logged]'),
        ]

    async def test_a_structured_result_is_logged_one_field_per_line(self) -> None:
        def rows() -> object:
            return {'ids': [1, 2]}

        def typed() -> object:
            return {'kind': int}

        store = InMemoryWorkLogStore()
        agent = Agent(
            _scripted(
                [ToolCallPart('rows', {}, tool_call_id='rows')], [ToolCallPart('typed', {}, tool_call_id='typed')]
            ),
            # Core can't send a `type` to the model, so the result reaches it truncated to text.
            capabilities=[WorkLog(store), ToolOutputLimits(bands=[Band(over=20, action=Truncate(max_chars=10))])],
            tools=[rows, typed],
        )

        await agent.run('Go', conversation_id=_CONVERSATION)

        assert _tool_entries(await store.read(_CONVERSATION)) == [
            ('rows', 'success', '{\n  "ids": [\n    1,\n    2\n  ]\n}'),
            # A leaf with no JSON form is rendered as its `repr` instead, in compact JSON.
            ('typed', 'success', '{"kind":"<class \'int\'>"}'),
        ]

    async def test_a_tool_call_id_reused_across_steps_gives_two_entries(self) -> None:
        store = InMemoryWorkLogStore()
        agent = Agent(
            _scripted(
                [ToolCallPart('lookup', {'city': 'Paris'}, tool_call_id='same')],
                [ToolCallPart('lookup', {'city': 'Rome'}, tool_call_id='same')],
            ),
            capabilities=[WorkLog(store)],
            tools=[lookup],
        )

        await agent.run('Go', conversation_id=_CONVERSATION)

        assert _tool_entries(await store.read(_CONVERSATION)) == [
            ('lookup', 'success', 'sunny in Paris'),
            ('lookup', 'success', 'sunny in Rome'),
        ]

    async def test_a_continued_run_appends_to_the_same_log(self) -> None:
        store = InMemoryWorkLogStore()
        agent = Agent(
            _scripted(
                [ToolCallPart('lookup', {'city': 'Paris'})],
                [TextPart('done')],
                [ToolCallPart('lookup', {'city': 'Rome'})],
            ),
            capabilities=[WorkLog(store)],
            tools=[lookup],
        )

        first = await agent.run('Paris?')
        second = await agent.run('Rome?', message_history=first.all_messages())

        assert second.conversation_id == first.conversation_id
        entries = await store.read(first.conversation_id)
        assert [(entry['seq'], entry['run_id'], entry['content']) for entry in entries] == [
            (1, first.run_id, 'sunny in Paris'),
            (2, first.run_id, 'done'),
            (3, second.run_id, 'sunny in Rome'),
            (4, second.run_id, 'done'),
        ]
        assert await store.read('another conversation') == []

    async def test_context_counts_the_latest_reset_notice(self) -> None:
        notice = ResetNoticePart(
            content='The conversation before this point was cleared.',
            source=OperatorSource(),
            number=2,
            work_log=None,
            kept_responses=0,
        )
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Start')]),
            ModelResponse(parts=[TextPart('Started')]),
            ModelRequest(parts=[notice]),
        ]
        store = InMemoryWorkLogStore()
        agent = Agent(
            _scripted([TextPart('done')], [ToolCallPart('lookup', {'city': 'Paris'})]),
            capabilities=[WorkLog(store)],
            tools=[lookup],
        )

        before = await agent.run('Paris?', conversation_id='before')
        after = await agent.run('Paris?', message_history=history, conversation_id=_CONVERSATION)

        assert [entry['context'] for entry in await store.read(before.conversation_id)] == [0]
        assert [entry['context'] for entry in await store.read(after.conversation_id)] == [2, 2]

    async def test_two_work_logs_that_agree_are_one_and_two_that_differ_raise(self) -> None:
        kept, dropped = InMemoryWorkLogStore(), InMemoryWorkLogStore()
        agent = Agent(
            _scripted([ToolCallPart('lookup', {'city': 'Paris'})]),
            capabilities=[WorkLog(kept), WorkLog(dropped)],
            tools=[lookup],
        )

        await agent.run('Paris?', conversation_id=_CONVERSATION)

        assert _tool_entries(await kept.read(_CONVERSATION)) == [('lookup', 'success', 'sunny in Paris')]
        assert await dropped.read(_CONVERSATION) == []
        with pytest.raises(UserError, match="disagree on 'record_text'"):
            Agent(_scripted(), capabilities=[WorkLog(), WorkLog(record_text=False)])

    async def test_exports(self) -> None:
        assert WorkLog is pydantic_ai_harness.work_log.WorkLog
        assert (WORK_LOG_SEARCH_KIND.id, WORK_LOG_READ_KIND.id) == (
            'work_log.search_work_log',
            'work_log.read_work_log',
        )

    async def test_operations_are_durable(self) -> None:
        store = InMemoryWorkLogStore()
        durability = RecordingDurability()
        agent = Agent(
            _scripted(
                [ToolCallPart('lookup', {'city': 'Paris'})],
                [ToolCallPart('search_work_log', {})],
                [ToolCallPart('read_work_log', {'seq': 1})],
            ),
            name='agent',
            capabilities=[WorkLog(store), durability],
            tools=[lookup],
        )

        await agent.run('Go', conversation_id=_CONVERSATION)

        assert [name for name, _ in durability.calls if '__capability__' in name] == [
            'agent__capability__work_log.append',
            'agent__capability__work_log.search',
            'agent__capability__work_log.read',
            'agent__capability__work_log.append',
        ]
        assert len(await store.read(_CONVERSATION)) == 2


class TestWorkLogCodeMode:
    async def test_nested_calls_are_logged_under_run_code(self) -> None:
        code = "a = await lookup(city='Paris')\nb = await lookup(city='Rome')\na + ', ' + b"
        store = InMemoryWorkLogStore()
        agent = Agent(
            _scripted([ToolCallPart('run_code', {'code': code}, tool_call_id='rc')]),
            capabilities=[CodeMode(), WorkLog(store)],
            tools=[lookup],
        )

        await agent.run('Go', conversation_id=_CONVERSATION)

        entries = await store.read(_CONVERSATION)
        assert [(e['seq'], e['tool_name'], e['tool_call_id'], e['parent_tool_call_id']) for e in entries] == [
            (1, 'lookup', 'rc__1', 'rc'),
            (2, 'lookup', 'rc__2', 'rc'),
            (3, 'run_code', 'rc', None),
            (4, None, None, None),
        ]
        assert entries[2]['args'] == json.dumps({'code': code}, separators=(',', ':'))

    async def test_a_tool_bridge_call_and_the_tool_it_runs_are_both_logged(self) -> None:
        """`ToolBridge` runs the bridged tool under its `tool_call`'s id, so both entries share it."""
        store = InMemoryWorkLogStore()
        args = {'name': 'lookup', 'args_json': json.dumps({'city': 'Paris'})}
        agent = Agent(
            _scripted([ToolCallPart('tool_call', args, tool_call_id='tc')]),
            capabilities=[ToolBridge(), WorkLog(store)],
            tools=[lookup],
        )

        await agent.run('Go', conversation_id=_CONVERSATION)

        entries = await store.read(_CONVERSATION)
        assert [(e['tool_name'], e['tool_call_id'], e['parent_tool_call_id'], e['content']) for e in entries] == [
            ('lookup', 'tc', 'tc', 'sunny in Paris'),
            ('tool_call', 'tc', None, 'sunny in Paris'),
            (None, None, None, 'done'),
        ]

    async def test_a_speculated_call_is_logged_under_its_provisional_id(self) -> None:
        """A call `CodeMode(speculate=...)` launches while the script streams keeps its provisional ID once adopted.

        The script also holds a literal call in a branch it never takes: speculation launches it and then cancels
        it, and it stays in the log only if it finished first.
        """
        code = 'if False:\n    a = await search(query="never")\nb = await search(query="alpha")\nprint(b)'
        # Filler statements leave stream to go after the calls close, so the launched calls can run meanwhile.
        args = json.dumps({'code': code + ''.join(f'\npad_{index} = {index}' for index in range(20)) + '\n"ok"'})

        alpha_started = anyio.Event()
        # The alpha call's statement has closed once the stream is past `print(b)`.
        closed_at = args.index('print(b)') + len('print(b)')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) > 1:
                yield 'done'
                return
            yield {1: DeltaToolCall(name='run_code', tool_call_id='rc')}
            waited = False
            for offset in range(0, len(args), 16):
                yield {1: DeltaToolCall(json_args=args[offset : offset + 16])}
                if not waited and offset + 16 >= closed_at:
                    # Hold the rest of the stream until speculation has launched the call: a hang guard, not a timing
                    # assertion, since without speculation `search` would not start before the stream ends.
                    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                        await alpha_started.wait()
                    waited = True

        async def search(query: str) -> str:
            if query == 'alpha':
                alpha_started.set()
            return f'result:{query}'

        store = InMemoryWorkLogStore()
        agent = Agent(
            FunctionModel(stream_function=stream),
            deps_type=type(None),
            tools=[search],
            capabilities=[CodeMode[None](speculate=['search']), WorkLog(store)],
        )

        await agent.run('Go', conversation_id=_CONVERSATION)

        searches = [entry for entry in await store.read(_CONVERSATION) if entry['tool_name'] == 'search']
        [adopted] = [entry for entry in searches if '"alpha"' in (entry['args'] or '')]
        assert adopted['tool_call_id'] == IsStr(regex=r'rc__spec_\d+')
        assert adopted['parent_tool_call_id'] == 'rc'
        assert all(entry['outcome'] == 'success' for entry in searches)
        assert len(searches) <= 2

    async def test_eager_nested_calls_are_logged_before_run_code(self) -> None:
        searched = asyncio.Event()

        async def search(query: str) -> str:
            searched.set()
            return query

        code = 'value = await search(query="alpha")\nx = 1\ny = 2\nvalue'
        arguments = json.dumps({'code': code})

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if any(isinstance(message, ModelResponse) for message in messages):
                yield 'done'
                return
            chunks = [arguments[offset : offset + 4] for offset in range(0, len(arguments), 4)]
            yield {1: DeltaToolCall(name='run_code', tool_call_id='rc')}
            for chunk in chunks[:-1]:
                yield {1: DeltaToolCall(json_args=chunk)}
            # A hang guard, not a timing assertion: without eager execution `search` would never start.
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await searched.wait()
            yield {1: DeltaToolCall(json_args=chunks[-1])}

        store = InMemoryWorkLogStore()
        agent = Agent(
            FunctionModel(stream_function=stream),
            capabilities=[CodeMode(eager=True), WorkLog(store)],
            tools=[search],
        )

        await agent.run('Go', conversation_id=_CONVERSATION)

        entries = await store.read(_CONVERSATION)
        assert [(e['seq'], e['tool_name'], e['parent_tool_call_id']) for e in entries if e['kind'] == 'tool'] == [
            (1, 'search', 'rc'),
            (2, 'run_code', None),
        ]

    async def test_the_tools_work_from_code_and_render_their_return_types(self) -> None:
        store = InMemoryWorkLogStore()
        await store.append(_CONVERSATION, _record('k1', 'paris is sunny'))
        seen: list[AgentInfo] = []
        code = (
            "found = await search_work_log(query='paris')\n"
            "entry = await read_work_log(seq=found['matches'][0]['seq'])\n"
            "entry['text']"
        )

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info)
            if len(seen) == 1:
                return ModelResponse(parts=[ToolCallPart('run_code', {'code': code}, tool_call_id='rc')])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(respond), capabilities=[CodeMode(), WorkLog(store)])

        result = await agent.run('Go', conversation_id=_CONVERSATION)

        assert _returned(result.all_messages(), 'rc') == 'paris is sunny'
        # CodeMode's catalog of callable functions goes into the instructions.
        catalog = seen[0].instructions or ''
        assert 'class WorkLogSearchResult(TypedDict):' in catalog
        assert 'class WorkLogRead(TypedDict):' in catalog


class TestWorkLogTools:
    async def test_tools_are_redacted_for_ui_clients(self) -> None:
        """The log holds every tool's result, a `'private'` tool's included, so a UI client gets a placeholder."""
        seen: dict[str, str | None] = {}

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.update((tool.name, tool.ui_visibility) for tool in info.function_tools)
            return ModelResponse(parts=[TextPart('done')])

        await Agent(FunctionModel(respond), capabilities=[WorkLog()]).run('Go', conversation_id=_CONVERSATION)

        assert seen == {'search_work_log': 'redacted', 'read_work_log': 'redacted'}

    async def test_search_ranks_by_bm25(self) -> None:
        store = InMemoryWorkLogStore()
        await store.append(_CONVERSATION, _record('k1', 'the weather in paris is sunny'))
        await store.append(_CONVERSATION, _record('k2', 'rome is cloudy'))
        await store.append(_CONVERSATION, _record('k3', 'paris paris paris'))

        (found,) = await _call_tools(store, ('search_work_log', {'query': 'paris'}))

        assert _SEARCH.validate_python(found) == {
            'total_entries': 3,
            'matches': [
                {
                    'seq': 3,
                    'context': 0,
                    'kind': 'tool',
                    'tool_name': 'lookup',
                    'args': '{}',
                    'excerpt': 'paris paris paris',
                    'chars': 17,
                },
                {
                    'seq': 1,
                    'context': 0,
                    'kind': 'tool',
                    'tool_name': 'lookup',
                    'args': '{}',
                    'excerpt': 'the weather in paris is sunny',
                    'chars': 29,
                },
            ],
            'capped': False,
        }

    async def test_search_filters_and_lists_the_newest_first_without_a_query(self) -> None:
        store = InMemoryWorkLogStore()
        await store.append(_CONVERSATION, _record('k1', tool_name='lookup', run_id='run-1'))
        await store.append(_CONVERSATION, _record('k2', tool_name='fetch', run_id='run-1'))
        await store.append(_CONVERSATION, _record('k3', tool_name='lookup', run_id='run-2'))
        await store.append(_CONVERSATION, _record('k4', tool_name='lookup', run_id='run-2'))

        found = await _call_tools(
            store,
            ('search_work_log', {'limit': 2}),
            ('search_work_log', {'query': '  '}),
            ('search_work_log', {'tool_name': 'lookup', 'run_id': 'run-1'}),
            ('search_work_log', {'tool_name': 'fetch', 'query': 'result'}),
            ('search_work_log', {'limit': 0}),
        )

        assert [[match['seq'] for match in _SEARCH.validate_python(result)['matches']] for result in found[:4]] == [
            [4, 3],
            [4, 3, 2, 1],
            [1],
            [2],
        ]
        assert found[4] == '`limit` must be >= 1.'

    async def test_search_clips_arguments_and_centres_the_excerpt_on_the_query(self) -> None:
        store = InMemoryWorkLogStore()
        content = 'x ' * 400 + 'needle' + ' y' * 400
        await store.append(_CONVERSATION, _record('k1', content, args=json.dumps({'q': 'a' * 300})))

        (found,) = await _call_tools(store, ('search_work_log', {'query': 'needle'}))

        (match,) = _SEARCH.validate_python(found)['matches']
        assert match['args'] == '{"q": "' + 'a' * 193 + '...'
        assert match['excerpt'].startswith('...') and match['excerpt'].endswith('...')
        assert 'needle' in match['excerpt'] and len(match['excerpt']) == 306
        assert match['chars'] == len(content)

    async def test_search_stops_before_the_read_size_cap(self) -> None:
        store = InMemoryWorkLogStore()
        for index in range(120):
            await store.append(_CONVERSATION, _record(f'k{index}', 'needle ' * 100, args='a' * 300))

        (found,) = await _call_tools(store, ('search_work_log', {'query': 'needle', 'limit': 1_000}))

        result = _SEARCH.validate_python(found)
        assert result['capped'] is True
        assert 0 < len(result['matches']) < 120
        assert len(json.dumps(result['matches'], ensure_ascii=False)) <= 50_000

    async def test_read_pages_through_an_entry(self) -> None:
        store = InMemoryWorkLogStore()
        await store.append(_CONVERSATION, _record('k1', '\n'.join(f'line {index}' for index in range(10))))

        found = await _call_tools(
            store,
            ('read_work_log', {'seq': 1, 'limit': 3}),
            ('read_work_log', {'seq': 1, 'offset': 8, 'limit': 5}),
            ('read_work_log', {'seq': 1, 'limit': 2, 'from_end': True}),
            ('read_work_log', {'seq': 1, 'offset': 8, 'limit': 5, 'from_end': True}),
            ('read_work_log', {'seq': 1, 'pattern': 'line 1'}),
        )

        assert [
            (read['text'], read['offset'], read['next_offset'], read['total'])
            for read in (_READ.validate_python(result) for result in found)
        ] == [
            ('line 0\nline 1\nline 2', 0, 3, 10),
            ('line 8\nline 9', 8, None, 10),
            ('line 8\nline 9', 0, 2, 10),
            ('line 0\nline 1', 8, None, 10),
            ('line 1', 0, None, 1),
        ]
        assert _READ.validate_python(found[0]) == {
            'seq': 1,
            'context': 0,
            'kind': 'tool',
            'tool_name': 'lookup',
            'tool_call_id': 'k1',
            'parent_tool_call_id': None,
            'args': '{}',
            'outcome': 'success',
            'text': 'line 0\nline 1\nline 2',
            'unit': 'lines',
            'offset': 0,
            'next_offset': 3,
            'total': 10,
            'capped': False,
            'cut_at': None,
        }

    async def test_read_is_capped_in_lines_and_characters(self) -> None:
        store = InMemoryWorkLogStore()
        await store.append(_CONVERSATION, _record('k1', '\n'.join(['x'] * 1_500)))
        await store.append(_CONVERSATION, _record('k2', '\n'.join(['y' * 30_000] * 3)))

        found = await _call_tools(
            store, ('read_work_log', {'seq': 1, 'limit': 5_000}), ('read_work_log', {'seq': 2, 'limit': 3})
        )

        lines, chars = (_READ.validate_python(result) for result in found)
        assert (len(lines['text'].split('\n')), lines['next_offset'], lines['capped']) == (1_000, 1_000, False)
        assert (len(chars['text']), chars['capped']) == (50_000, True)
        # The cap cut the second line, so reading on starts at it rather than past the window.
        assert chars['next_offset'] == 1

    async def test_a_capped_read_reads_on_from_the_first_line_not_returned_whole(self) -> None:
        # Each line is 20,000 characters, so a window of all four holds two whole and cuts the third.
        content = '\n'.join(letter * 20_000 for letter in 'abcd')
        store = InMemoryWorkLogStore()
        await store.append(_CONVERSATION, _record('k1', content))
        await store.append(_CONVERSATION, _record('k2', 'z' * 60_000 + '\nshort'))

        found = await _call_tools(
            store,
            ('read_work_log', {'seq': 1, 'limit': 4}),
            ('read_work_log', {'seq': 1, 'offset': 2, 'limit': 4}),
            ('read_work_log', {'seq': 1, 'limit': 4, 'from_end': True}),
            ('read_work_log', {'seq': 1, 'offset': 2, 'limit': 4, 'from_end': True}),
            ('read_work_log', {'seq': 2}),
            ('read_work_log', {'seq': 2, 'from_end': True}),
        )

        reads = [_READ.validate_python(result) for result in found]
        assert [(read['offset'], read['next_offset'], read['capped']) for read in reads] == [
            (0, 2, True),
            (2, None, False),
            # From the end the cap keeps the lines nearest the end, and reading on goes back from the cut one.
            (0, 2, True),
            (2, None, False),
            # A line longer than the cap on its own is cut, and reading on skips it: no read returns more of it.
            (0, 1, True),
            (0, 1, True),
        ]
        first, rest, last, earlier, long, long_from_end = (read['text'] for read in reads)
        assert first == f'{"a" * 20_000}\n{"b" * 20_000}\n{"c" * 9_998}'
        assert rest == f'{"c" * 20_000}\n{"d" * 20_000}'
        assert last == f'{"b" * 9_998}\n{"c" * 20_000}\n{"d" * 20_000}'
        assert earlier == f'{"a" * 20_000}\n{"b" * 20_000}'
        assert long == 'z' * 50_000
        assert long_from_end == f'{"z" * 49_994}\nshort'

    async def test_a_cap_on_a_line_break_cuts_no_line(self) -> None:
        store = InMemoryWorkLogStore()
        await store.append(_CONVERSATION, _record('k1', 'a' * 50_000 + '\nb'))
        await store.append(_CONVERSATION, _record('k2', 'a\n' + 'b' * 50_000))

        found = await _call_tools(store, ('read_work_log', {'seq': 1}), ('read_work_log', {'seq': 2, 'from_end': True}))

        reads = [_READ.validate_python(result) for result in found]
        assert [(read['text'], read['next_offset'], read['capped'], read['cut_at']) for read in reads] == [
            ('a' * 50_000, 1, True, None),
            ('b' * 50_000, 1, True, None),
        ]

    async def test_a_cut_line_continues_in_a_bytes_read(self) -> None:
        content = '\n'.join(letter * 20_000 for letter in 'abcd')
        store = InMemoryWorkLogStore()
        await store.append(_CONVERSATION, _record('k1', content))
        await store.append(_CONVERSATION, _record('k2', 'z' * 60_000 + '\nshort'))

        found = await _call_tools(
            store,
            ('read_work_log', {'seq': 1, 'limit': 4}),
            ('read_work_log', {'seq': 1, 'limit': 4, 'from_end': True}),
            ('read_work_log', {'seq': 2}),
            ('read_work_log', {'seq': 2, 'from_end': True}),
            ('read_work_log', {'seq': 2, 'unit': 'bytes', 'offset': 50_000, 'limit': 50_000}),
            ('read_work_log', {'seq': 2, 'unit': 'bytes', 'limit': 3, 'from_end': True}),
        )

        reads = [_READ.validate_python(result) for result in found]
        # Forward, the cut line goes on past what was shown; from the end, its start was left out.
        assert [read['cut_at'] for read in reads[:4]] == [40_002 + 9_998, 20_001, 50_000, 0]
        rest, tail = reads[4:]
        assert (rest['text'], rest['unit'], rest['next_offset'], rest['total']) == (
            'z' * 10_000 + '\nshort',
            'bytes',
            None,
            60_006,
        )
        assert (tail['text'], tail['next_offset']) == ('ort', 3)

    async def test_bad_read_arguments_ask_for_a_retry(self) -> None:
        store = InMemoryWorkLogStore()
        await store.append(_CONVERSATION, _record('k1'))

        calls: list[tuple[str, dict[str, object]]] = [
            ('read_work_log', {'seq': 5}),
            ('read_work_log', {'seq': 1, 'offset': -1}),
            ('read_work_log', {'seq': 1, 'limit': 0}),
            ('read_work_log', {'seq': 1, 'unit': 'bytes', 'pattern': 'x'}),
        ]
        # One run each: a tool's retry budget allows one retry per run.
        found = [await _call_tools(store, call) for call in calls]
        empty = await _call_tools(InMemoryWorkLogStore(), ('read_work_log', {'seq': 1}))

        assert found == [
            ['No work log entry has `seq` 5: entries run from 1 to 1.'],
            ['`offset` must be >= 0.'],
            ['`limit` must be >= 1.'],
            ["`pattern` filters lines, so it cannot be combined with `unit='bytes'`."],
        ]
        assert empty == ['No work log entry has `seq` 1: the work log is empty.']

    async def test_tool_output_limits_reduces_neither_search_nor_read(self, tmp_path: Path) -> None:
        store = InMemoryWorkLogStore()
        for index in range(120):
            await store.append(_CONVERSATION, _record(f'k{index}', 'needle ' * 100, args='a' * 300))
        await store.append(_CONVERSATION, _record('long', 'z' * 30_000))

        searched, read = await _call_tools(
            store,
            ('search_work_log', {'query': 'needle', 'limit': 1_000}),
            ('read_work_log', {'seq': 121}),
            capabilities=[ToolOutputLimits(store=LocalFileOverflowStore(tmp_path))],
        )

        assert _SEARCH.validate_python(searched)['capped'] is True
        assert _READ.validate_python(read)['text'] == 'z' * 30_000


class TestWorkLogStores:
    async def test_in_memory_store_appends_once_per_key_and_reads_ranges(self) -> None:
        store = InMemoryWorkLogStore()

        seqs = [await store.append('log', _record(key)) for key in ('k1', 'k2', 'k1', 'k3')]

        assert seqs == [1, 2, 1, 3]
        assert [entry['key'] for entry in await store.read('log')] == ['k1', 'k2', 'k3']
        assert [entry['seq'] for entry in await store.read('log', 2)] == [2, 3]
        assert [entry['seq'] for entry in await store.read('log', 1, 3)] == [1, 2]
        assert await store.read('log', 0, 0) == []
        assert store.locate('log') is None
        assert InMemoryWorkLogStore() == store

        store.discard('log')

        assert await store.read('log') == []

    async def test_file_store_survives_a_new_instance(self, tmp_path: Path) -> None:
        store = FileWorkLogStore(tmp_path / 'logs')
        assert [await store.append('log', _record(key)) for key in ('k1', 'k2', 'k1')] == [1, 2, 1]

        reopened = FileWorkLogStore(tmp_path / 'logs')

        assert await reopened.append('log', _record('k2')) == 2
        assert await reopened.append('log', _record('k3', 'line\u2028separated')) == 3
        assert [(entry['seq'], entry['content']) for entry in await reopened.read('log', 2)] == [
            (2, 'result'),
            (3, 'line\u2028separated'),
        ]
        assert (tmp_path / 'logs' / 'log.jsonl').read_text().count('\n') == 3
        assert reopened.locate('log') == str(tmp_path / 'logs' / 'log.jsonl')
        assert await reopened.read('missing') == []

    async def test_file_store_drops_a_write_a_crash_cut_short(self, tmp_path: Path) -> None:
        store = FileWorkLogStore(tmp_path)
        await store.append('log', _record('k1'))
        path = tmp_path / 'log.jsonl'
        # Cut inside a UTF-8 sequence, as a crash can leave it.
        path.write_bytes(path.read_bytes() + '{"seq": 2, "content": "\u00e9'.encode()[:-1])

        reopened = FileWorkLogStore(tmp_path)

        assert [entry['key'] for entry in await reopened.read('log')] == ['k1']
        assert await reopened.append('log', _record('k2')) == 2
        assert [entry['key'] for entry in await FileWorkLogStore(tmp_path).read('log')] == ['k1', 'k2']

    @pytest.mark.parametrize('log_id', ['.', '..', 'a/b', 'x' * 129, 'sha256-abc'])
    async def test_file_store_hashes_an_unsafe_log_id(self, tmp_path: Path, log_id: str) -> None:
        store = FileWorkLogStore(tmp_path)

        await store.append(log_id, _record('k1'))

        name = f'sha256-{hashlib.sha256(log_id.encode()).hexdigest()}.jsonl'
        assert store.locate(log_id) == str(tmp_path / name)
        assert [path.name for path in tmp_path.iterdir()] == [name]

    async def test_file_store_keeps_an_id_apart_from_the_name_of_a_hashed_one(self, tmp_path: Path) -> None:
        store = FileWorkLogStore(tmp_path)
        hashed_name = f'sha256-{hashlib.sha256(b"tenant:alice").hexdigest()}'

        await store.append('tenant:alice', _record('k1', 'alice'))
        await store.append(hashed_name, _record('k1', 'mallory'))

        assert [entry['content'] for entry in await FileWorkLogStore(tmp_path).read('tenant:alice')] == ['alice']
        assert len(list(tmp_path.iterdir())) == 2

    @pytest.mark.skipif(not hasattr(os, 'geteuid'), reason='POSIX ownership check')
    async def test_file_store_keeps_its_directory_owner_only(self, tmp_path: Path) -> None:
        old_umask = os.umask(0)
        try:
            await FileWorkLogStore(tmp_path / 'new').append('log', _record('k1'))
        finally:
            os.umask(old_umask)
        assert oct((tmp_path / 'new').stat().st_mode & 0o777) == '0o700'
        assert oct((tmp_path / 'new' / 'log.jsonl').stat().st_mode & 0o777) == '0o600'

        existing = tmp_path / 'existing'
        existing.mkdir()
        existing.chmod(0o755)
        await FileWorkLogStore(existing).append('log', _record('k1'))
        assert oct(existing.stat().st_mode & 0o777) == '0o700'

    @pytest.mark.skipif(not hasattr(os, 'geteuid'), reason='POSIX ownership check')
    async def test_file_store_refuses_a_directory_another_user_owns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        await FileWorkLogStore(tmp_path).append('log', _record('k1'))
        other_uid = tmp_path.stat().st_uid + 1
        monkeypatch.setattr(os, 'geteuid', lambda: other_uid)

        with pytest.raises(PermissionError, match='not owned by the current user'):
            await FileWorkLogStore(tmp_path).append('log', _record('k2'))
        with pytest.raises(PermissionError, match='not owned by the current user'):
            await FileWorkLogStore(tmp_path).read('log')

    async def test_file_store_refuses_a_symlinked_directory(self, tmp_path: Path) -> None:
        target = tmp_path / 'target'
        target.mkdir()
        (tmp_path / 'logs').symlink_to(target, target_is_directory=True)

        with pytest.raises(PermissionError, match='is a symbolic link'):
            await FileWorkLogStore(tmp_path / 'logs').append('log', _record('k1'))
        assert list(target.iterdir()) == []

    async def test_file_store_skips_ownership_without_posix_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delattr(os, 'geteuid', raising=False)
        store = FileWorkLogStore(tmp_path)

        assert await store.append('log', _record('k1')) == 1
        assert [entry['key'] for entry in await store.read('log')] == ['k1']

    async def test_file_store_appends_concurrently_without_gaps(self, tmp_path: Path) -> None:
        store = FileWorkLogStore(tmp_path)

        async with anyio.create_task_group() as group:
            for index in range(20):
                group.start_soon(store.append, 'log', _record(f'k{index}'))

        entries = await FileWorkLogStore(tmp_path).read('log')
        assert sorted(entry['seq'] for entry in entries) == list(range(1, 21))
        assert len({entry['key'] for entry in entries}) == 20

    async def test_workspace_store_keeps_the_log_in_the_run_workspace(self, tmp_path: Path) -> None:
        store = WorkspaceWorkLogStore()
        agent = Agent(
            _scripted([ToolCallPart('lookup', {'city': 'Paris'})]),
            capabilities=[LocalWorkspace(tmp_path), WorkLog(store)],
            tools=[lookup],
        )

        await agent.run('Paris?', conversation_id=_CONVERSATION)

        directory = tmp_path / '.pydantic-ai-harness' / 'work-log' / _CONVERSATION
        assert len(list(directory.iterdir())) == 2
        assert (tmp_path / '.pydantic-ai-harness' / '.gitignore').read_text() == '*\n'
        assert store.locate(_CONVERSATION) == '.pydantic-ai-harness/work-log/conv/'
        cold = WorkspaceWorkLogStore().bind(Workspace(LocalWorkspaceBackend(tmp_path)))
        entries = await cold.read(_CONVERSATION)
        assert [entry['content'] for entry in entries] == ['sunny in Paris', 'done']
        assert await cold.append(_CONVERSATION, _record(entries[1]['key'])) == 2
        assert len(list(directory.iterdir())) == 2
        assert cold.locate(_CONVERSATION) == store.locate(_CONVERSATION)

    async def test_workspace_store_names_entries_by_seq_and_key(self, tmp_path: Path) -> None:
        store = WorkspaceWorkLogStore()
        workspace = Workspace(LocalWorkspaceBackend(tmp_path))

        assert [await store.append(workspace, 'log', _record(key)) for key in ('k1', 'k2', 'k1')] == [1, 2, 1]

        directory = tmp_path / '.pydantic-ai-harness' / 'work-log' / 'log'
        assert sorted(path.name for path in directory.iterdir()) == [
            f'00000001-{hashlib.sha256(b"k1").hexdigest()[:16]}.json',
            f'00000002-{hashlib.sha256(b"k2").hexdigest()[:16]}.json',
        ]
        (directory / 'notes.txt').write_text('not an entry')
        cold = WorkspaceWorkLogStore()
        assert [entry['key'] for entry in await cold.read(workspace, 'log', 1, 2)] == ['k1']
        assert await cold.append(workspace, 'log', _record('k3')) == 3

    async def test_workspace_store_with_its_own_workspace_needs_none_on_the_run(self, tmp_path: Path) -> None:
        (tmp_path / 'logs').mkdir()
        store = WorkspaceWorkLogStore(LocalWorkspaceBackend(tmp_path / 'logs'))
        agent = Agent(
            _scripted([ToolCallPart('lookup', {'city': 'Paris'})]), capabilities=[WorkLog(store)], tools=[lookup]
        )

        await agent.run('Paris?', conversation_id=_CONVERSATION)

        assert len(list((tmp_path / 'logs' / '.pydantic-ai-harness' / 'work-log' / _CONVERSATION).iterdir())) == 2
        # The agent's tools reach only the run's workspace, so a reset notice names no handle for this log.
        assert store.locate(_CONVERSATION) is None

    async def test_workspace_store_needs_a_workspace(self) -> None:
        agent = Agent(_scripted(), capabilities=[WorkLog(WorkspaceWorkLogStore())])

        with pytest.raises(UserError, match='`WorkLog` needs a workspace'):
            await agent.run('Go')

    async def test_workspace_store_refuses_symlinks(self, tmp_path: Path) -> None:
        workspace = Workspace(LocalWorkspaceBackend(tmp_path))
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        logs = tmp_path / '.pydantic-ai-harness' / 'work-log'
        logs.mkdir(parents=True)
        (logs / 'linked').symlink_to(elsewhere)

        with pytest.raises(WorkspaceError, match='is a symlink'):
            await WorkspaceWorkLogStore().append(workspace, 'linked', _record('k1'))

        store = WorkspaceWorkLogStore()
        await store.append(workspace, 'log', _record('k1'))
        (logs / 'log' / f'00000002-{hashlib.sha256(b"k2").hexdigest()[:16]}.json').symlink_to(elsewhere / 'target.json')
        with pytest.raises(WorkspaceError, match='is a symlink'):
            await store.append(workspace, 'log', _record('k2'))
        assert not (elsewhere / 'target.json').exists()

        (logs / 'log' / f'00000002-{hashlib.sha256(b"k2").hexdigest()[:16]}.json').unlink()
        await store.append(workspace, 'other', _record('k1'))
        (other_entry,) = (logs / 'other').glob('*.json')
        (logs / 'log' / f'00000003-{"0" * 16}.json').symlink_to(other_entry)
        with pytest.raises(WorkspaceError, match=r'00000003-0{16}\.json` in the workspace is a symlink'):
            await store.read(workspace, 'log')
        (logs / 'log' / f'00000003-{"0" * 16}.json').unlink()

        (logs / 'log').rename(tmp_path / 'moved')
        (logs / 'log').symlink_to(tmp_path / 'moved')
        with pytest.raises(WorkspaceError, match='is a symlink'):
            await store.read(workspace, 'log')
