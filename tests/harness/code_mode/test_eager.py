"""Behavioral tests for `CodeMode(eager=True)`."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager

import pytest
from pydantic_ai import Agent, RunContext, Tool, ToolReturn
from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.code_mode import CodeMode, CodeModeToolset

from .._recording_durability import RecordingDurability

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def build_run_context() -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        pending_messages=[],
    )


async def event_stream(events: Sequence[AgentStreamEvent]) -> AsyncIterator[AgentStreamEvent]:
    for event in events:
        yield event


def prior_run_code_calls(messages: list[ModelMessage]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_name == 'run_code'
    )


def stream_json_args(code: str, *, extra: str = '', chunk_size: int = 16) -> list[str]:
    args = json.dumps({'code': code})[:-1] + extra + '}'
    return [args[offset : offset + chunk_size] for offset in range(0, len(args), chunk_size)]


def run_code_return_content(messages: list[ModelMessage]) -> object:
    contents = [
        part.content
        for message in messages
        for part in getattr(message, 'parts', [])
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]
    assert contents
    return contents[-1]


@asynccontextmanager
async def prepared_eager_toolset(
    tools: Sequence[Tool[None]],
    capability: CodeMode[None] | None = None,
) -> AsyncGenerator[tuple[CodeMode[None], CodeModeToolset[None], RunContext[None], ToolsetTool[None]]]:
    capability = capability or CodeMode[None](eager=True)
    root = capability.get_wrapper_toolset(FunctionToolset[None](tools=tools))
    assert isinstance(root, CodeModeToolset)
    ctx = build_run_context()
    async with root:
        manager = await ToolManager(toolset=root).for_run_step(ctx)
        ctx.tool_manager = manager
        assert manager.tools is not None
        run_code = manager.tools['run_code']
        active = run_code.toolset
        assert isinstance(active, CodeModeToolset)
        yield capability, active, ctx, run_code


async def observe(capability: CodeMode[None], ctx: RunContext[None], events: Sequence[AgentStreamEvent]) -> None:
    async for _ in capability.wrap_run_event_stream(ctx, stream=event_stream(events)):
        pass


class TestEagerCodeMode:
    async def test_executes_before_model_stream_finishes(self):
        first_call = asyncio.Event()
        calls: list[str] = []

        async def search(query: str) -> str:
            calls.append(query)
            first_call.set()
            return f'result:{query}'

        code = 'a = await search(query="alpha")\nb = await search(query="beta")\nprint(a)\nprint(b)\n"ok"'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if prior_run_code_calls(messages):
                yield 'done'
                return
            chunks = stream_json_args(code, chunk_size=4)
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in chunks[:-1]:
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)
            await asyncio.wait_for(first_call.wait(), timeout=5)
            yield {1: DeltaToolCall(json_args=chunks[-1])}

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[CodeMode(eager=True)],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        assert calls == ['alpha', 'beta']
        assert run_code_return_content(result.all_messages()) == {
            'output': 'result:alpha\nresult:beta\n',
            'result': 'ok',
        }

    async def test_nested_tool_hooks_apply_to_eager_fragments(self):
        seen: list[str] = []
        stream_finished = asyncio.Event()
        search_ran_early = False

        class RecordTools(AbstractCapability[None]):
            async def before_tool_execute(
                self,
                ctx: RunContext[None],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: ValidatedToolArgs,
            ) -> ValidatedToolArgs:
                seen.append(tool_def.name)
                return args

        async def search(query: str) -> str:
            nonlocal search_ran_early
            search_ran_early = not stream_finished.is_set()
            return query

        code = 'value = await search(query="alpha")\nx = 1\ny = 2\nz = 3\nvalue'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if prior_run_code_calls(messages):
                yield 'done'
                return
            chunks = stream_json_args(code, chunk_size=4)
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in chunks:
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)
            stream_finished.set()

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[CodeMode(eager=True), RecordTools()],
        )
        agent.tool_plain(search)

        await agent.run('go')

        assert search_ran_early
        assert seen == ['search', 'run_code']

    async def test_failure_preserves_state_and_reports_prior_output(self):
        bad = 'saved = 7\nprint("diagnostic")\nboom = 1 // 0\n"unused"'
        async with prepared_eager_toolset([]) as (capability, toolset, ctx, run_code):
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0, part=ToolCallPart(tool_name='run_code', args={'code': bad}, tool_call_id='c1')
                    )
                ],
            )
            exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')

            with pytest.raises(ModelRetry) as exc_info:
                await toolset.call_tool('run_code', {'code': bad}, exec_ctx, run_code)

            recovered = await toolset.call_tool('run_code', {'code': 'print(saved)\n"recovered"'}, exec_ctx, run_code)

        assert '[stdout before error]' in exc_info.value.message
        assert 'diagnostic' in exc_info.value.message
        assert recovered.return_value == {'output': '7\n', 'result': 'recovered'}

    async def test_failure_in_a_streamed_fragment_stops_later_fragments(self):
        calls: list[str] = []

        def stale() -> None:
            calls.append('stale')  # pragma: no cover - reaching this line fails the assertion below

        bad = 'print("diagnostic")\nboom = 1 // 0\nawait stale()\n"unused"'
        async with prepared_eager_toolset([Tool(stale)]) as (capability, toolset, ctx, run_code):
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0, part=ToolCallPart(tool_name='run_code', args={'code': bad}, tool_call_id='c1')
                    )
                ],
            )
            exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')

            with pytest.raises(ModelRetry) as exc_info:
                await toolset.call_tool('run_code', {'code': bad}, exec_ctx, run_code)

        assert calls == []
        assert 'diagnostic' in exc_info.value.message

    async def test_call_budget_spans_streamed_fragments_and_tail(self):
        calls: list[str] = []

        def search(query: str) -> str:
            calls.append(query)
            return query

        greedy = 'a = await search(query="alpha")\nb = await search(query="beta")\n"unused"'

        async def stream_attempts(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            prior = prior_run_code_calls(messages)
            if prior >= 2:
                yield 'done'
                return
            code = greedy if prior == 0 else '"recovered"'
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in stream_json_args(code):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_attempts),
            deps_type=type(None),
            capabilities=[CodeMode(eager=True, max_tool_calls=1)],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        assert calls == ['alpha']

    async def test_monty_limits_apply_to_eager_fragments(self):
        capability = CodeMode[None](eager=True, resource_limits={'max_duration_secs': 0.3})
        code = 'y = 0\nfor i in range(100_000_000):\n    y += i\nz = 1\nz'

        async with prepared_eager_toolset([], capability) as (capability, toolset, ctx, run_code):
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0, part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='c1')
                    )
                ],
            )
            exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')

            with pytest.raises(ModelRetry, match='time limit exceeded'):
                await toolset.call_tool('run_code', {'code': code}, exec_ctx, run_code)

    @pytest.mark.parametrize('escaped_key', [False, True])
    async def test_restart_seen_before_code_does_not_execute_a_prefix(self, escaped_key: bool):
        calls: list[str] = []

        def search(query: str) -> str:
            calls.append(query)
            return query

        code = 'value = await search(query="alpha")\nx = 1\nvalue'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if prior_run_code_calls(messages):
                yield 'done'
                return
            key = '\\u0072estart' if escaped_key else 'restart'
            args = '{"' + key + '": true, "code": ' + json.dumps(code) + '}'
            yield {1: DeltaToolCall(name='run_code')}
            for offset in range(0, len(args), 16):
                yield {1: DeltaToolCall(json_args=args[offset : offset + 16])}
                await asyncio.sleep(0)

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[CodeMode(eager=True)],
        )
        agent.tool_plain(search)

        await agent.run('go')

        assert calls == ['alpha']

    async def test_late_restart_reexecutes_an_already_run_prefix(self):
        first_call = asyncio.Event()
        calls: list[str] = []

        def search(query: str) -> str:
            calls.append(query)
            first_call.set()
            return query

        code = 'value = await search(query="alpha")\nx = 1\nvalue'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if prior_run_code_calls(messages):
                yield 'done'
                return
            code_args = json.dumps({'code': code})[:-1]
            yield {1: DeltaToolCall(name='run_code', json_args=code_args)}
            await asyncio.wait_for(first_call.wait(), timeout=5)
            yield {1: DeltaToolCall(json_args=', "restart": true}')}

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[CodeMode(eager=True)],
        )
        agent.tool_plain(search)

        await agent.run('go')

        assert calls == ['alpha', 'alpha']

    async def test_statements_on_one_line_execute_once_and_keep_the_result(self):
        calls: list[int] = []

        def charge(value: int) -> int:
            calls.append(value)
            return value

        code = 'q = 1; paid = await charge(value=1)\nfinal = 2\nprint(final)\n"ok"'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if prior_run_code_calls(messages):
                yield 'done'
                return
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in stream_json_args(code):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[CodeMode(eager=True)],
        )
        agent.tool_plain(charge)

        result = await agent.run('go')

        assert calls == [1]
        assert run_code_return_content(result.all_messages()) == {'output': '2\n', 'result': 'ok'}

    async def test_later_streamed_calls_cannot_overtake_the_first(self):
        calls: list[str] = []
        first_ran = asyncio.Event()

        def probe(value: str) -> str:
            calls.append(value)
            if value == 'one':
                first_ran.set()
            return value

        async with prepared_eager_toolset([Tool(probe)]) as (capability, toolset, ctx, run_code):
            first = 'answer = await probe(value="one")\nx = 1\nanswer'
            second = 'answer = await probe(value="two")\nx = 1\nanswer'
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0,
                        part=ToolCallPart(tool_name='run_code', args={'code': 'answer = ('}, tool_call_id='c1'),
                    ),
                    PartStartEvent(
                        index=1,
                        part=ToolCallPart(tool_name='run_code', args={'code': second}, tool_call_id='c2'),
                    ),
                    PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': first}, tool_call_id='c1')),
                ],
            )
            await asyncio.wait_for(first_ran.wait(), timeout=5)
            assert calls == ['one']

            first_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
            second_ctx = dataclasses.replace(ctx, tool_call_id='c2', tool_name='run_code')
            await toolset.call_tool('run_code', {'code': first}, first_ctx, run_code)
            await toolset.call_tool('run_code', {'code': second}, second_ctx, run_code)

        assert calls == ['one', 'two']

    async def test_run_code_does_not_overtake_an_earlier_native_tool(self):
        calls: list[str] = []
        stream_finished = asyncio.Event()

        def authorize() -> str:
            calls.append('authorize')
            return 'authorized'

        def act() -> str:
            assert stream_finished.is_set()
            calls.append('act')
            return 'done'

        code = 'result = await act()\nx = 1\nresult'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if prior_run_code_calls(messages):
                yield 'done'
                return
            yield {
                0: DeltaToolCall(name='authorize', json_args='{}'),
                1: DeltaToolCall(name='run_code'),
            }
            for chunk in stream_json_args(code):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)
            stream_finished.set()

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[CodeMode(eager=True, tools=['act'])],
        )
        agent.tool_plain(sequential=True)(authorize)
        agent.tool_plain(act)

        result = await agent.run('go')

        assert result.output == 'done'
        assert calls == ['authorize', 'act']

    async def test_deltas_are_matched_by_part_index(self):
        calls: list[str] = []

        def probe() -> None:
            calls.append('wrong')  # pragma: no cover - reaching this line fails the assertion below

        async with prepared_eager_toolset([Tool(probe)]) as (capability, _, ctx, _):
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0,
                        part=ToolCallPart(tool_name='run_code', args='', tool_call_id='code-call'),
                    ),
                    PartStartEvent(
                        index=1,
                        part=ToolCallPart(tool_name='native', args='', tool_call_id='native-call'),
                    ),
                    PartDeltaEvent(
                        index=1,
                        delta=ToolCallPartDelta(args_delta={'code': 'await probe()\nx = 1\nx'}),
                    ),
                ],
            )
            await asyncio.sleep(0)

        assert calls == []

    async def test_diverged_prefix_restarts_the_session(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        stale_calls: list[str] = []

        async def slow() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        def stale() -> None:
            stale_calls.append('stale')  # pragma: no cover - reaching this line fails the assertion below

        async with prepared_eager_toolset([Tool(slow), Tool(stale)]) as (capability, toolset, ctx, run_code):
            original = 'x = 1\nawait slow()\nawait stale()\nx'
            rewritten = 'y = 2\ny'
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0,
                        part=ToolCallPart(tool_name='run_code', args={'code': original}, tool_call_id='c1'),
                    ),
                ],
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            await observe(
                capability,
                ctx,
                [PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': rewritten}, tool_call_id='c1'))],
            )
            assert cancelled.is_set()
            assert stale_calls == []
            exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')

            with pytest.raises(ModelRetry, match='no longer matches'):
                await toolset.call_tool('run_code', {'code': rewritten}, exec_ctx, run_code)
            with pytest.raises(ModelRetry, match='not defined'):
                await toolset.call_tool('run_code', {'code': 'x'}, exec_ctx, run_code)

        assert cancelled.is_set()
        assert stale_calls == []

    async def test_late_restart_cancels_queued_statements(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        stale_calls: list[str] = []

        async def slow() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        def stale() -> None:
            stale_calls.append('stale')  # pragma: no cover - reaching this line fails the assertion below

        async with prepared_eager_toolset([Tool(slow), Tool(stale)]) as (capability, _, ctx, _):
            code = 'await slow()\nawait stale()\n"done"'
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0, part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='c1')
                    )
                ],
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            await observe(
                capability,
                ctx,
                [PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'restart': True}, tool_call_id='c1'))],
            )

            assert cancelled.is_set()
            assert stale_calls == []

    async def test_incomplete_compound_statement_waits_until_it_is_closed(self):
        called = asyncio.Event()
        calls: list[str] = []

        def probe() -> None:
            calls.append('probe')
            called.set()

        async with prepared_eager_toolset([Tool(probe)]) as (capability, toolset, ctx, run_code):
            partial = 'if True:\n    await probe()\n'
            complete = partial + 'x = 1\nx'
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0, part=ToolCallPart(tool_name='run_code', args={'code': partial}, tool_call_id='c1')
                    )
                ],
            )
            await asyncio.sleep(0)
            assert calls == []

            await observe(
                capability,
                ctx,
                [PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': complete}, tool_call_id='c1'))],
            )
            await asyncio.wait_for(called.wait(), timeout=5)
            exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
            result = await toolset.call_tool('run_code', {'code': complete}, exec_ctx, run_code)

        assert calls == ['probe']
        assert result.return_value == 1

    async def test_rewritten_tool_call_id_follows_the_streamed_call(self):
        calls: list[str] = []

        def probe(value: str) -> str:
            calls.append(value)
            return value

        async with prepared_eager_toolset([Tool(probe)]) as (capability, toolset, ctx, run_code):
            code = 'first = await probe(value="prefix")\nx = 1\nawait probe(value="tail")'
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0,
                        part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='stream-id'),
                    ),
                    PartDeltaEvent(index=0, delta=ToolCallPartDelta(tool_call_id='final-id')),
                ],
            )
            exec_ctx = dataclasses.replace(ctx, tool_call_id='final-id', tool_name='run_code')

            result = await toolset.call_tool('run_code', {'code': code}, exec_ctx, run_code)

        assert calls == ['prefix', 'tail']
        assert result.return_value == 'tail'
        assert result.metadata is not None
        assert list(result.metadata['tool_calls']) == ['stream-id__1', 'stream-id__2']
        assert list(result.metadata['tool_returns']) == ['stream-id__1', 'stream-id__2']

    async def test_dict_replacement_is_rescanned(self):
        calls: list[str] = []
        called = asyncio.Event()

        def probe(value: str) -> str:
            calls.append(value)
            called.set()
            return value

        async with prepared_eager_toolset([Tool(probe)]) as (capability, toolset, ctx, run_code):
            fixed = 'answer = await probe(value="fixed")\nx = 1\nanswer'
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0,
                        part=ToolCallPart(
                            tool_name='run_code', args={'code': 'answer = (\nx = 1\n'}, tool_call_id='c1'
                        ),
                    ),
                    PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': fixed}, tool_call_id='c1')),
                ],
            )
            await asyncio.wait_for(called.wait(), timeout=5)
            assert calls == ['fixed']
            exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
            result = await toolset.call_tool('run_code', {'code': fixed}, exec_ctx, run_code)

        assert calls == ['fixed']
        assert result.return_value == 'fixed'

    async def test_step_specific_toolset_handles_streamed_fragments(self):
        calls: list[str] = []
        called = asyncio.Event()

        def probe(value: str) -> str:
            calls.append(value)
            called.set()
            return value

        class StepSwap(FunctionToolset[None]):
            replacement: FunctionToolset[None] | None = None

            async def for_run_step(self, ctx: RunContext[None]) -> FunctionToolset[None]:
                return self.replacement or self

        wrapped = StepSwap(tools=[])
        wrapped.replacement = FunctionToolset[None](tools=[Tool(probe)])
        capability = CodeMode[None](eager=True)
        root = capability.get_wrapper_toolset(wrapped)
        assert isinstance(root, CodeModeToolset)
        ctx = build_run_context()

        async with root:
            manager = await ToolManager(toolset=root).for_run_step(ctx)
            ctx.tool_manager = manager
            assert manager.tools is not None
            run_code = manager.tools['run_code']
            toolset = run_code.toolset
            assert isinstance(toolset, CodeModeToolset)
            code = 'answer = await probe(value="step")\nx = 1\nanswer'
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0, part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='c1')
                    )
                ],
            )
            await asyncio.wait_for(called.wait(), timeout=5)
            assert calls == ['step']
            exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
            result = await toolset.call_tool('run_code', {'code': code}, exec_ctx, run_code)

        assert calls == ['step']
        assert result.return_value == 'step'

    async def test_malformed_and_irrelevant_stream_events_fall_back_to_dispatch(self):
        calls: list[str] = []

        def probe() -> str:
            calls.append('probe')
            return 'done'

        async with prepared_eager_toolset([Tool(probe)]) as (capability, toolset, ctx, run_code):
            oversized = '{"code": "' + 'x' * 300_000
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(index=1, part=ToolCallPart(tool_name='run_code', args='[\\n', tool_call_id='one')),
                    PartStartEvent(index=1, part=ToolCallPart(tool_name='run_code', args='', tool_call_id='one')),
                    PartStartEvent(index=5, part=ToolCallPart(tool_name='other', args={}, tool_call_id='other')),
                    PartDeltaEvent(
                        index=1,
                        delta=ToolCallPartDelta(args_delta={'restart': False}, tool_call_id='one'),
                    ),
                    PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta=None, tool_call_id='one')),
                    PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta=oversized, tool_call_id='one')),
                    PartStartEvent(index=2, part=ToolCallPart(tool_name='run_code', args='', tool_call_id='two')),
                    PartDeltaEvent(index=3, delta=ToolCallPartDelta(args_delta='ignored')),
                    PartDeltaEvent(index=4, delta=ToolCallPartDelta(args_delta='ignored', tool_call_id='missing')),
                    PartEndEvent(index=5, part=ToolCallPart(tool_name='other', args={}, tool_call_id='other')),
                ],
            )
            assert calls == []

            code = 'answer = await probe()\nanswer'
            exec_ctx = dataclasses.replace(ctx, tool_call_id='one', tool_name='run_code')
            result = await toolset.call_tool('run_code', {'code': code}, exec_ctx, run_code)

        assert calls == ['probe']
        assert result.return_value == 'done'

    async def test_validation_retry_does_not_disable_later_eager_execution(self):
        valid_call_started = asyncio.Event()

        def probe() -> str:
            valid_call_started.set()
            return 'ok'

        valid = 'answer = await probe()\nx = 1\nanswer'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            prior = prior_run_code_calls(messages)
            if prior == 0:
                yield {1: DeltaToolCall(name='run_code', json_args='{"wrong": true}')}
            elif prior == 1:
                chunks = stream_json_args(valid)
                yield {1: DeltaToolCall(name='run_code')}
                for chunk in chunks[:-1]:
                    yield {1: DeltaToolCall(json_args=chunk)}
                    await asyncio.sleep(0)
                await asyncio.wait_for(valid_call_started.wait(), timeout=5)
                yield {1: DeltaToolCall(json_args=chunks[-1])}
            else:
                yield 'done'

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            deps_type=type(None),
            capabilities=[CodeMode(eager=True)],
        )
        agent.tool_plain(probe)

        result = await agent.run('go')

        assert result.output == 'done'

    async def test_oversized_dict_replacement_keeps_queued_statements(self):
        started = asyncio.Event()
        release = asyncio.Event()
        calls: list[str] = []

        async def slow() -> None:
            started.set()
            await release.wait()

        def probe(value: str) -> str:
            calls.append(value)
            return value

        async with prepared_eager_toolset([Tool(slow), Tool(probe)]) as (capability, toolset, ctx, run_code):
            code = 'await slow()\nawait probe(value="queued")\nawait probe(value="tail")'
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0, part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='c1')
                    )
                ],
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            oversized = code + '\n' + '# padding\n' * 30_000
            await observe(
                capability,
                ctx,
                [PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': oversized}, tool_call_id='c1'))],
            )
            release.set()
            exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
            await toolset.call_tool('run_code', {'code': oversized}, exec_ctx, run_code)

        assert calls == ['queued', 'tail']

    async def test_toolset_exit_cancels_an_unfinished_fragment(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def slow() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async with prepared_eager_toolset([Tool(slow)]) as (capability, _, ctx, _):
            code = 'await slow()\nx = 1\nx'
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0,
                        part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='c1'),
                    )
                ],
            )
            await asyncio.wait_for(started.wait(), timeout=5)

        assert cancelled.is_set()

    async def test_late_restart_abandons_a_pump_held_by_a_non_cooperative_tool(self, monkeypatch: pytest.MonkeyPatch):
        """Discarding an in-flight call waits for the cancelled pump only within the budget.
        A nested tool that swallows the cancellation is abandoned, the statements queued
        behind it never run, and the tool is allowed to finish in the background."""
        monkeypatch.setattr('pydantic_ai_harness.code_mode._eager.PUMP_CANCEL_TIMEOUT_SECONDS', 0.25)
        started = asyncio.Event()
        done = asyncio.Event()
        calls: list[str] = []

        async def stubborn() -> str:
            calls.append('stubborn')
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Swallow the cancellation and keep working past the cleanup budget.
                await asyncio.sleep(1.2)
            calls.append('done')
            done.set()
            return 'late'

        async with prepared_eager_toolset([Tool(stubborn)]) as (capability, _, ctx, _):
            code = 'await stubborn()\nawait stubborn()\n"ok"'
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0, part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='c1')
                    )
                ],
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            # Without the budget, the restart would wait out the 1.2 second non-cooperative linger.
            discard_start = time.monotonic()
            await observe(
                capability,
                ctx,
                [PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'restart': True}, tool_call_id='c1'))],
            )

            assert time.monotonic() - discard_start < 0.75
            assert calls == ['stubborn']
            # The abandoned tool is allowed to finish in the background.
            await asyncio.wait_for(done.wait(), timeout=5)
            assert calls == ['stubborn', 'done']

    async def test_output_matches_normal_code_mode_across_fragments(self):
        def blob(size: int) -> str:
            return 'x' * size

        code = 'print(await blob(size=1048575))\nprint("y")\n"ok"'
        tools = [Tool(blob)]

        async with prepared_eager_toolset(tools) as (capability, toolset, ctx, run_code):
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0, part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='c1')
                    )
                ],
            )
            exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
            eager_result = await toolset.call_tool('run_code', {'code': code}, exec_ctx, run_code)

        normal_root = CodeMode[None]().get_wrapper_toolset(FunctionToolset[None](tools=tools))
        assert isinstance(normal_root, CodeModeToolset)
        normal_ctx = build_run_context()
        async with normal_root:
            manager = await ToolManager(toolset=normal_root).for_run_step(normal_ctx)
            normal_ctx.tool_manager = manager
            assert manager.tools is not None
            normal_tool = manager.tools['run_code']
            normal_ctx = dataclasses.replace(normal_ctx, tool_call_id='c1', tool_name='run_code')
            normal_result = await normal_root.call_tool('run_code', {'code': code}, normal_ctx, normal_tool)

        assert isinstance(eager_result, ToolReturn)
        assert isinstance(normal_result, ToolReturn)
        assert eager_result.return_value == normal_result.return_value

    @pytest.mark.parametrize(
        ('args_format', 'restart'),
        [('dict', False), ('string', False), ('string', True)],
    )
    async def test_scan_work_limit_returns_queued_statements_to_dispatch(self, args_format: str, restart: bool):
        started = asyncio.Event()
        release = asyncio.Event()
        slow_calls = 0
        stale_calls: list[str] = []

        async def slow() -> None:
            nonlocal slow_calls
            slow_calls += 1
            started.set()
            await release.wait()

        def stale() -> None:
            stale_calls.append('stale')  # pragma: no cover - reaching this line fails the assertion below

        code = 'await slow()\n\nif True:\n    await stale()\nvalue = 1\n"done"'
        async with prepared_eager_toolset([Tool(slow), Tool(stale)]) as (capability, toolset, ctx, run_code):
            if args_format == 'string':
                start_args: str | dict[str, object] = '{"code":"' + json.dumps(code)[1:-1]
                padding = '\\n#' * 10_000
                deltas = [
                    PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta=padding, tool_call_id='c1'))
                    for _ in range(10)
                ]
                deltas.append(
                    PartDeltaEvent(
                        index=0,
                        delta=ToolCallPartDelta(args_delta='","restart":true}' if restart else '"}', tool_call_id='c1'),
                    )
                )
            else:
                start_args = {'code': code}
                padded_code = code + '\n' + '# padding\n' * 10_000
                deltas = [
                    PartDeltaEvent(
                        index=0,
                        delta=ToolCallPartDelta(
                            args_delta={'code': padded_code + '# more\n' * step}, tool_call_id='c1'
                        ),
                    )
                    for step in range(1, 12)
                ]
            await observe(
                capability,
                ctx,
                [PartStartEvent(index=0, part=ToolCallPart(tool_name='run_code', args=start_args, tool_call_id='c1'))],
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            await observe(capability, ctx, deltas)
            release.set()

            final_code = code if restart else 'await slow()\n"done"'
            exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
            await toolset.call_tool('run_code', {'code': final_code, 'restart': restart}, exec_ctx, run_code)
            assert slow_calls == (2 if restart else 1)
            assert stale_calls == (['stale'] if restart else [])

    @pytest.mark.parametrize(
        ('invalid_args', 'halt_before_invalid'),
        [
            ('diverged', True),
            ('restart', True),
            ('null_code', True),
            ('missing_code', True),
            ('oversized_diverged', False),
            ('null_code', False),
            ('missing_code', False),
        ],
    )
    async def test_invalidated_queued_statements_are_cancelled(self, invalid_args: str, halt_before_invalid: bool):
        started = asyncio.Event()
        cancelled = asyncio.Event()
        stale_calls: list[str] = []

        async def slow() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        def stale() -> None:
            stale_calls.append('stale')  # pragma: no cover - reaching this line fails the assertion below

        initial = 'await slow()\nawait stale()\n"done"'
        same_prefix = initial + '\n' + '# padding\n' * 10_000
        async with prepared_eager_toolset([Tool(slow), Tool(stale)]) as (capability, _, ctx, _):
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0, part=ToolCallPart(tool_name='run_code', args={'code': initial}, tool_call_id='c1')
                    )
                ],
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            if halt_before_invalid:
                await observe(
                    capability,
                    ctx,
                    [
                        PartDeltaEvent(
                            index=0,
                            delta=ToolCallPartDelta(args_delta={'code': same_prefix}, tool_call_id='c1'),
                        )
                        for _ in range(11)
                    ],
                )
            if invalid_args == 'diverged':
                event: AgentStreamEvent = PartDeltaEvent(
                    index=0,
                    delta=ToolCallPartDelta(args_delta={'code': 'replacement = 1\nreplacement'}, tool_call_id='c1'),
                )
            elif invalid_args == 'restart':
                event = PartDeltaEvent(
                    index=0,
                    delta=ToolCallPartDelta(args_delta={'restart': True}, tool_call_id='c1'),
                )
            elif invalid_args == 'oversized_diverged':
                event = PartDeltaEvent(
                    index=0,
                    delta=ToolCallPartDelta(args_delta={'code': 'replacement\n' * 30_000}, tool_call_id='c1'),
                )
            elif invalid_args == 'null_code':
                if halt_before_invalid:
                    event = PartDeltaEvent(
                        index=0,
                        delta=ToolCallPartDelta(args_delta={'code': None}, tool_call_id='c1'),
                    )
                else:
                    event = PartStartEvent(
                        index=0,
                        part=ToolCallPart(tool_name='run_code', args={'code': None}, tool_call_id='c1'),
                    )
            else:
                event = PartStartEvent(
                    index=0,
                    part=ToolCallPart(tool_name='run_code', args={'wrong': True}, tool_call_id='c1'),
                )
            await observe(capability, ctx, [event])

            assert cancelled.is_set()
            assert stale_calls == []

    async def test_interrupted_statement_restarts_the_session(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def slow() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async with prepared_eager_toolset([Tool(slow)]) as (capability, toolset, ctx, run_code):
            code = 'saved = 7\nawait slow()\nx = 1\nx'
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0, part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='c1')
                    )
                ],
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            # The call never reaches dispatch (for example, argument validation rejected it), so the
            # next step retires it while `slow()` is still running.
            next_step = dataclasses.replace(ctx, run_step=1)
            await observe(
                capability,
                next_step,
                [PartStartEvent(index=0, part=ToolCallPart(tool_name='other', args={}, tool_call_id='o1'))],
            )
            assert cancelled.is_set()

            exec_ctx = dataclasses.replace(next_step, tool_call_id='c2', tool_name='run_code')
            with pytest.raises(ModelRetry, match='not defined'):
                await toolset.call_tool('run_code', {'code': 'saved'}, exec_ctx, run_code)

    async def test_inactive_during_durable_execution(self):
        stream_finished = asyncio.Event()
        calls: list[str] = []

        def search(query: str) -> str:
            assert stream_finished.is_set()
            calls.append(query)
            return query

        code = 'answer = await search(query="alpha")\nx = 1\nanswer'

        async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if prior_run_code_calls(messages):
                yield 'done'
                return
            yield {1: DeltaToolCall(name='run_code')}
            for chunk in stream_json_args(code):
                yield {1: DeltaToolCall(json_args=chunk)}
                await asyncio.sleep(0)
            stream_finished.set()

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream_code),
            name='eager',
            deps_type=type(None),
            capabilities=[CodeMode(eager=True), RecordingDurability()],
        )
        agent.tool_plain(search)

        result = await agent.run('go')

        assert result.output == 'done'
        assert calls == ['alpha']

    @pytest.mark.parametrize(
        ('code', 'value'),
        [
            ('answer = await probe(value="cr")\rx = 1\ranswer', 'cr'),
            ('answer = await probe(value="large")\n' + '# padding\n' * 30_000 + 'answer', 'large'),
        ],
    )
    async def test_unscannable_input_falls_back_to_normal_execution(self, code: str, value: str):
        calls: list[str] = []

        def probe(value: str) -> str:
            calls.append(value)
            return value

        async with prepared_eager_toolset([Tool(probe)]) as (capability, toolset, ctx, run_code):
            await observe(
                capability,
                ctx,
                [
                    PartStartEvent(
                        index=0, part=ToolCallPart(tool_name='run_code', args={'code': code}, tool_call_id='c1')
                    )
                ],
            )
            assert calls == []
            exec_ctx = dataclasses.replace(ctx, tool_call_id='c1', tool_name='run_code')
            result = await toolset.call_tool('run_code', {'code': code}, exec_ctx, run_code)

        assert calls == [value]
        assert result.return_value == value
