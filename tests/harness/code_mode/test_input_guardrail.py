"""Streamed `CodeMode` execution composed with `InputGuardrail`.

Eager and speculative execution run nested tool calls while the model is still streaming the
`run_code` call. A parallel `InputGuardrail` decides during that same stream, so those calls wait
for its verdict, the way normal tool dispatch waits for the model request to finish.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness import CodeMode, GuardrailResult, InputGuardrail

CODE = 'a = await ping(n=1)\nb = await ping(n=2)\n[a, b]'
_ARGS = json.dumps({'code': CODE})
_CUT = _ARGS.index('[a, b]')
FIRST_CHUNK, LAST_CHUNK = _ARGS[:_CUT], _ARGS[_CUT:]
"""The first chunk closes the first statement for eager execution and completes both calls for speculation."""

BLOCKED_TIERS = [
    pytest.param(lambda: CodeMode[None](eager=True), 1.0, id='eager'),
    pytest.param(lambda: CodeMode[None](speculate=['ping']), 0.1, id='speculate'),
    pytest.param(lambda: CodeMode[None](eager=True, speculate=['ping']), 0.1, id='eager-speculate'),
]
"""Each tier with how long a blocked run gives it to reach `ping` before the guard decides.

A call that is not held runs within this and ends the wait early; a held call has nothing to
signal, so the wait elapses. Eager statements reach `ping` through Monty, which runs off the event
loop and takes longest on its first start in a process. Speculation starts `ping` from the event
loop as soon as the call text has streamed.
"""

STREAMED_TIERS = [
    pytest.param(lambda: CodeMode[None](eager=True), id='eager'),
    pytest.param(lambda: CodeMode[None](speculate=['ping']), id='speculate'),
    pytest.param(lambda: CodeMode[None](eager=True, speculate=['ping']), id='eager-speculate'),
]


class GuardedRun:
    """A streamed `run_code` call whose last chunk waits for `ping(n=1)` to run.

    After the first chunk, the stream gives streamed execution `reach_grace` seconds to reach `ping`,
    records which calls ran by then, and lets a parallel guard decide. Holding the last chunk until
    `ping(n=1)` ran makes any call that runs at all run before the stream finished.
    """

    def __init__(self, reach_grace: float) -> None:
        self.reach_grace = reach_grace
        self.calls: list[int] = []
        self.calls_at_verdict: list[int] | None = None
        self.verdict_due = asyncio.Event()
        self.first_call_ran = asyncio.Event()

    async def ping(self, n: int) -> int:
        self.calls.append(n)
        if n == 1:
            self.first_call_ran.set()
        return n

    async def stream(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='run_code', json_args=FIRST_CHUNK, tool_call_id='run_code_1')}
        with anyio.move_on_after(self.reach_grace):
            await self.first_call_ran.wait()
        self.calls_at_verdict = list(self.calls)
        self.verdict_due.set()
        # A blocking verdict abandons the stream here, before the last chunk.
        await asyncio.wait_for(self.first_call_ran.wait(), timeout=5)
        yield {0: DeltaToolCall(json_args=LAST_CHUNK)}

    def agent(self, code_mode: CodeMode[None], guardrail: InputGuardrail[None] | None) -> Agent[None, str]:
        capabilities = [code_mode] if guardrail is None else [code_mode, guardrail]
        agent = Agent(FunctionModel(stream_function=self.stream), deps_type=type(None), capabilities=capabilities)
        agent.tool_plain(self.ping)
        return agent


class TestCodeModeInputGuardrail:
    @pytest.mark.parametrize(('code_mode', 'reach_grace'), BLOCKED_TIERS)
    async def test_parallel_guard_that_blocks_runs_no_nested_call(
        self, code_mode: Callable[[], CodeMode[None]], reach_grace: float
    ):
        run = GuardedRun(reach_grace=reach_grace)

        async def guard(prompt: str) -> GuardrailResult:
            await run.verdict_due.wait()
            return GuardrailResult.block('blocked')

        result = await run.agent(code_mode(), InputGuardrail(guard=guard, parallel=True)).run('go')

        assert result.output == 'blocked'
        assert run.calls == []

    @pytest.mark.parametrize('code_mode', STREAMED_TIERS)
    async def test_parallel_guard_that_allows_releases_nested_calls_during_stream(
        self, code_mode: Callable[[], CodeMode[None]]
    ):
        run = GuardedRun(reach_grace=0)

        async def guard(prompt: str) -> GuardrailResult:
            await run.verdict_due.wait()
            return GuardrailResult.allow()

        result = await run.agent(code_mode(), InputGuardrail(guard=guard, parallel=True)).run('go')

        assert result.output == 'done'
        assert run.calls_at_verdict == []
        assert sorted(run.calls) == [1, 2]

    @pytest.mark.parametrize('with_sequential_guard', [False, True], ids=['no-guard', 'sequential-guard'])
    @pytest.mark.parametrize('code_mode', STREAMED_TIERS)
    async def test_without_parallel_guard_runs_nested_calls_without_waiting(
        self, code_mode: Callable[[], CodeMode[None]], with_sequential_guard: bool
    ):
        run = GuardedRun(reach_grace=5)
        guardrail = InputGuardrail[None](guard=lambda prompt: True) if with_sequential_guard else None

        result = await run.agent(code_mode(), guardrail).run('go')

        assert result.output == 'done'
        assert run.calls_at_verdict is not None and 1 in run.calls_at_verdict
        assert sorted(run.calls) == [1, 2]

    @pytest.mark.parametrize('code_mode', STREAMED_TIERS)
    async def test_sequential_guard_that_blocks_runs_no_code(self, code_mode: Callable[[], CodeMode[None]]):
        run = GuardedRun(reach_grace=0)

        result = await run.agent(
            code_mode(), InputGuardrail(guard=lambda prompt: GuardrailResult.block('blocked'))
        ).run('go')

        assert result.output == 'blocked'
        assert run.calls_at_verdict is None
        assert run.calls == []
