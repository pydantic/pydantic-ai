"""Tests for `CodeMode(repl_store=...)`, which keeps the `run_code` REPL across agent runs."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.code_mode import CodeMode, CodeModeReplStore, CodeModeToolset

from .._recording_durability import RecordingDurability

READINESS_WAIT_TIMEOUT = 10
"""Seconds to wait for a background tool to reach a step; generous so loaded xdist workers do not flake."""


@dataclass
class MemoryReplStore:
    """REPL states keyed by conversation, recording every load and save."""

    states: dict[str, bytes] = field(default_factory=dict[str, bytes])
    loads: list[str] = field(default_factory=list[str])
    saves: list[tuple[str, bool]] = field(default_factory=list[tuple[str, bool]])

    async def load(self, ctx: RunContext[None]) -> bytes | None:
        assert ctx.conversation_id is not None
        self.loads.append(ctx.conversation_id)
        return self.states.get(ctx.conversation_id)

    async def save(self, ctx: RunContext[None], state: bytes | None) -> None:
        assert ctx.conversation_id is not None
        self.saves.append((ctx.conversation_id, state is not None))
        if state is None:
            self.states.pop(ctx.conversation_id, None)
        else:
            self.states[ctx.conversation_id] = state


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _reply(messages: list[ModelMessage], scripts: Mapping[str, Mapping[str, object]]) -> ModelResponse:
    """Answer a scripted prompt with its `run_code` arguments, then report what `run_code` returned."""
    request = messages[-1]
    assert isinstance(request, ModelRequest)
    part = request.parts[-1]
    if isinstance(part, UserPromptPart):
        assert isinstance(part.content, str)
        args = scripts.get(part.content)
        if args is None:
            return ModelResponse(parts=[TextPart('no code')])
        return ModelResponse(parts=[ToolCallPart('run_code', dict(args))])
    if isinstance(part, RetryPromptPart):
        return ModelResponse(parts=[TextPart(f'retry: {part.content}')])
    assert isinstance(part, ToolReturnPart)
    return ModelResponse(parts=[TextPart(str(part.content))])


def scripted(scripts: Mapping[str, Mapping[str, object]]) -> FunctionModel:
    """A model that runs `scripts[prompt]` and replies with the result, streaming its arguments in small chunks."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return _reply(messages, scripts)

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        [part] = _reply(messages, scripts).parts
        if isinstance(part, TextPart):
            yield part.content
            return
        assert isinstance(part, ToolCallPart)
        yield {0: DeltaToolCall(name='run_code')}
        args = json.dumps(part.args)
        for offset in range(0, len(args), 8):
            yield {0: DeltaToolCall(json_args=args[offset : offset + 8])}
            await asyncio.sleep(0)

    return FunctionModel(respond, stream_function=stream)


SCRIPTS: Mapping[str, Mapping[str, object]] = {
    'set': {'code': 'x = await add(a=40, b=1)'},
    'get': {'code': 'x + 1'},
}


def truncated(state: bytes) -> bytes:
    return state[:-10]


def foreign(state: bytes) -> bytes:
    return b'not a monty dump'


def agent_with(
    store: CodeModeReplStore[None], scripts: Mapping[str, Mapping[str, object]] = SCRIPTS, *, eager: bool = False
) -> Agent[None, str]:
    agent = Agent(scripted(scripts), deps_type=type(None), capabilities=[CodeMode[None](repl_store=store, eager=eager)])
    agent.tool_plain(add)
    return agent


class TestCodeModeReplStore:
    @pytest.mark.parametrize('eager', [False, True])
    async def test_state_carries_over_to_the_next_run(self, eager: bool) -> None:
        store = MemoryReplStore()
        agent = agent_with(store, eager=eager)

        await agent.run('set', conversation_id='c1')
        # The restored names would fail the type check a fresh REPL's first snippet gets.
        assert (await agent.run('get', conversation_id='c1')).output == '42'

        fresh = (await agent.run('get', conversation_id='c2')).output
        assert fresh.startswith('retry: Type error') and '`x` used when not defined' in fresh
        assert store.loads == ['c1', 'c1', 'c2']

    async def test_streamed_statements_run_in_the_restored_repl(self) -> None:
        store = MemoryReplStore()
        scripts = {**SCRIPTS, 'stream': {'code': 'y = x + 1\nz = await add(a=y, b=0)\nz'}}
        agent = agent_with(store, scripts, eager=True)

        await agent.run('set', conversation_id='c1')
        assert (await agent.run('stream', conversation_id='c1')).output == '42'

    async def test_run_without_run_code_leaves_the_saved_state(self) -> None:
        store = MemoryReplStore()
        agent = agent_with(store)

        await agent.run('set', conversation_id='c1')
        saved = dict(store.states)
        assert (await agent.run('chat', conversation_id='c1')).output == 'no code'

        assert store.states == saved
        assert store.saves == [('c1', True)]
        assert (await agent.run('get', conversation_id='c1')).output == '42'

    async def test_restart_replaces_the_saved_state(self) -> None:
        store = MemoryReplStore()
        scripts = {**SCRIPTS, 'restart': {'code': 'y = 1', 'restart': True}}
        agent = agent_with(store, scripts)

        await agent.run('set', conversation_id='c1')
        await agent.run('restart', conversation_id='c1')

        assert store.loads == ['c1']
        missing = (await agent.run('get', conversation_id='c1')).output
        assert missing.startswith('retry: Runtime error') and "name 'x' is not defined" in missing

    async def test_restart_without_new_state_forgets_the_saved_state(self) -> None:
        store = MemoryReplStore()
        scripts = {**SCRIPTS, 'restart': {'code': 'await add(a="bad", b=1)', 'restart': True}}
        agent = agent_with(store, scripts)

        await agent.run('set', conversation_id='c1')
        assert (await agent.run('restart', conversation_id='c1')).output.startswith('retry: Type error')

        assert store.states == {}
        assert store.saves == [('c1', True), ('c1', False)]

    @pytest.mark.parametrize(
        ('damage', 'warning'),
        [
            pytest.param(truncated, 'could not restore the saved REPL state', id='truncated'),
            pytest.param(foreign, 'saved in another format', id='foreign'),
        ],
    )
    async def test_unloadable_state_starts_a_fresh_repl(
        self, damage: Callable[[bytes], bytes], warning: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        store = MemoryReplStore()
        scripts = {**SCRIPTS, 'bad': {'code': 'await add(a="bad", b=1)'}, 'fresh': {'code': 'x = 1\nx'}}
        agent = agent_with(store, scripts)
        await agent.run('set', conversation_id='c1')
        store.states['c1'] = damage(store.states['c1'])

        with caplog.at_level(logging.WARNING):
            # The fresh REPL's first snippet is type-checked, as in a run without saved state.
            assert (await agent.run('bad', conversation_id='c1')).output.startswith('retry: Type error')
            assert (await agent.run('fresh', conversation_id='c1')).output == '1'

        assert warning in caplog.text
        assert (await agent.run('get', conversation_id='c1')).output == '2'

    async def test_state_saved_under_other_limits_starts_fresh(self, caplog: pytest.LogCaptureFixture) -> None:
        """A restored REPL would keep the limits it was saved under, so changed limits start a fresh one."""
        store = MemoryReplStore()
        await agent_with(store).run('set', conversation_id='c1')
        stricter = Agent(
            scripted(SCRIPTS),
            deps_type=type(None),
            capabilities=[CodeMode[None](repl_store=store, resource_limits={'max_suspensions': 10})],
        )
        stricter.tool_plain(add)

        with caplog.at_level(logging.WARNING):
            output = (await stricter.run('get', conversation_id='c1')).output

        assert output.startswith('retry: Type error') and '`x` used when not defined' in output
        assert 'under other resource limits' in caplog.text

    async def test_cancelled_run_still_saves(self) -> None:
        store = MemoryReplStore()
        returned = anyio.Event()

        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if isinstance(messages[-1].parts[-1], ToolReturnPart):
                returned.set()
                await anyio.sleep_forever()
            return _reply(messages, SCRIPTS)

        agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[CodeMode[None](repl_store=store)])
        agent.tool_plain(add)

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(lambda: agent.run('set', conversation_id='c1'))
            await returned.wait()
            tasks.cancel_scope.cancel()

        assert store.saves == [('c1', True)]

    async def test_durable_execution_does_not_use_the_store(self) -> None:
        store = MemoryReplStore()
        agent = Agent(
            scripted(SCRIPTS),
            name='durable',
            deps_type=type(None),
            capabilities=[CodeMode[None](repl_store=store), RecordingDurability()],
        )
        agent.tool_plain(add)

        await agent.run('set', conversation_id='c1')

        assert (store.loads, store.saves) == ([], [])

    async def test_store_errors_do_not_fail_the_run(self, caplog: pytest.LogCaptureFixture) -> None:
        @dataclass
        class BrokenStore:
            async def load(self, ctx: RunContext[None]) -> bytes | None:
                raise OSError('disk unavailable')

            async def save(self, ctx: RunContext[None], state: bytes | None) -> None:
                raise OSError('disk full')

        agent = agent_with(BrokenStore(), {'fresh': {'code': 'x = 1\nx'}})

        with caplog.at_level(logging.WARNING):
            assert (await agent.run('fresh', conversation_id='c1')).output == '1'

        assert 'could not load the saved REPL state' in caplog.text and 'disk unavailable' in caplog.text
        assert 'could not save the REPL state' in caplog.text and 'disk full' in caplog.text

    async def test_a_save_past_the_timeout_is_abandoned(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A timeout already spent stops the save before it reaches the store, as a hung dump or store would be."""
        monkeypatch.setattr('pydantic_ai_harness.code_mode._repl_store.SAVE_TIMEOUT_SECONDS', 0)
        store = MemoryReplStore()
        agent = agent_with(store)

        with caplog.at_level(logging.WARNING):
            await agent.run('set', conversation_id='c1')

        assert (store.saves, store.states) == ([], {})
        assert 'stopped saving the REPL state' in caplog.text

    async def test_a_repl_still_feeding_at_exit_is_not_saved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A nested tool that ignores cancellation keeps its eager statement running past teardown."""
        monkeypatch.setattr('pydantic_ai_harness.code_mode._eager.CANCEL_TIMEOUT_SECONDS', 0)
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def stubborn() -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
            finished.set()
            return 'late'

        class ReleasingToolset(FunctionToolset[None]):
            """Lets the stubborn tool go on exit, which teardown reaches only after giving up on its pump."""

            async def __aexit__(self, *args: Any) -> bool | None:
                release.set()
                return await super().__aexit__(*args)

        store = MemoryReplStore()
        capability = CodeMode[None](eager=True, repl_store=store)
        root = capability.get_wrapper_toolset(ReleasingToolset(tools=[Tool(stubborn)]))
        assert isinstance(root, CodeModeToolset)
        ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), conversation_id='c1')
        async with root:
            ctx.tool_manager = await ToolManager(toolset=root).for_run_step(ctx)
            code = 'x = await stubborn()\nx\n'
            start = PartStartEvent(index=0, part=ToolCallPart('run_code', {'code': code}, tool_call_id='c1'))

            async def events() -> AsyncIterator[PartStartEvent]:
                yield start

            async for _ in capability.wrap_run_event_stream(ctx, stream=events()):
                pass
            await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)

        await asyncio.wait_for(finished.wait(), timeout=READINESS_WAIT_TIMEOUT)
        assert store.loads == ['c1']
        assert store.saves == []
