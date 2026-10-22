"""Streamed `CodeMode` execution composed with `InputGuardrail`.

Eager and speculative execution run nested tool calls while the model is still streaming the
`run_code` call. A parallel `InputGuardrail` decides during that same stream, so those calls wait
for its verdict, the way normal tool dispatch waits for the model request to finish.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable

import pytest

from pydantic_ai import Agent, ModelRequestContext, ModelRetry, RunContext
from pydantic_ai.capabilities import AgentCapability, Hooks
from pydantic_ai.exceptions import SkipModelRequest, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import CodeMode, GuardrailResult, InputGuardrail
from pydantic_ai_harness.guardrails._capability import input_guardrails_allow

from .._recording_durability import RecordingDurability

READINESS_WAIT_TIMEOUT = 5
"""Bounds a wait on a test's own events, which only times out when the expected ordering broke."""

CODE = 'a = await ping(n=1)\nb = await ping(n=2)\n[a, b]'
_ARGS = json.dumps({'code': CODE})
_CUT = _ARGS.index('[a, b]')
FIRST_CHUNK, LAST_CHUNK = _ARGS[:_CUT], _ARGS[_CUT:]
"""The first chunk closes the first statement for eager execution and completes both calls for speculation."""

STREAMED_TIERS = [
    pytest.param(lambda: CodeMode[None](eager=True), id='eager'),
    pytest.param(lambda: CodeMode[None](speculate=['ping']), id='speculate'),
    pytest.param(lambda: CodeMode[None](eager=True, speculate=['ping']), id='eager-speculate'),
]


def guard_deciding_after_a_retried_first_response(
    result: GuardrailResult,
) -> tuple[InputGuardrail[None], Hooks[None]]:
    """A parallel guard returning `result`, and hooks that reject the first model response with `ModelRetry`.

    The guard decides once the rejected response's task has finished raising, which gives
    `wrap_model_request` the first chance to act on the retry.
    """
    response_task: asyncio.Future[asyncio.Task[object] | None] = asyncio.get_running_loop().create_future()
    hooks = Hooks[None]()

    @hooks.on.after_model_request
    async def retry_first_response(
        ctx: RunContext[None], request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        if not response_task.done():
            response_task.set_result(asyncio.current_task())
            raise ModelRetry('try again')
        return response

    async def guard(prompt: str) -> GuardrailResult:
        task = await response_task
        assert task is not None
        await asyncio.wait({task})
        return result

    return InputGuardrail[None](guard=guard, parallel=True), hooks


class GuardedRun:
    """A streamed `run_code` call whose last chunk waits for `ping(n=1)` to run.

    The stream sends its first chunk once `first_chunk_released` is set, which it is by default.
    It then waits for a nested call to start, records which calls ran by then, and lets a parallel
    guard decide. Holding the last chunk until `ping(n=1)` ran makes any call that runs at all run
    before the stream finished.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.calls_at_verdict: list[int] | None = None
        self.call_started = asyncio.Event()
        self.verdict_due = asyncio.Event()
        self.first_call_ran = asyncio.Event()
        self.requested = asyncio.Event()
        self.first_chunk_released = asyncio.Event()
        self.first_chunk_released.set()

    def watch_verdict_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Count a nested call as started once it waits for the input guardrails' verdict, before `ping` runs."""

        async def allow(ctx: RunContext[object]) -> bool:
            self.call_started.set()
            return await input_guardrails_allow(ctx)

        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset.input_guardrails_allow', allow)

    async def ping(self, n: int) -> int:
        self.calls.append(n)
        if n == 1:
            self.call_started.set()
            self.first_call_ran.set()
        return n

    async def stream(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last.parts):
            yield 'done'
            return
        self.requested.set()
        await self.first_chunk_released.wait()
        yield {0: DeltaToolCall(name='run_code', json_args=FIRST_CHUNK, tool_call_id='run_code_1')}
        await asyncio.wait_for(self.call_started.wait(), timeout=READINESS_WAIT_TIMEOUT)
        self.calls_at_verdict = list(self.calls)
        self.verdict_due.set()
        # A blocking verdict abandons the stream here, before the last chunk.
        await asyncio.wait_for(self.first_call_ran.wait(), timeout=READINESS_WAIT_TIMEOUT)
        yield {0: DeltaToolCall(json_args=LAST_CHUNK)}

    def agent(self, *capabilities: AgentCapability[None]) -> Agent[None, str]:
        agent = Agent(FunctionModel(stream_function=self.stream), deps_type=type(None), capabilities=capabilities)
        agent.tool_plain(self.ping)
        return agent


class TestCodeModeInputGuardrail:
    @pytest.mark.parametrize('code_mode', STREAMED_TIERS)
    async def test_parallel_guard_that_blocks_runs_no_nested_call(
        self, code_mode: Callable[[], CodeMode[None]], monkeypatch: pytest.MonkeyPatch
    ):
        run = GuardedRun()
        run.watch_verdict_wait(monkeypatch)

        async def guard(prompt: str) -> GuardrailResult:
            await run.verdict_due.wait()
            return GuardrailResult.block('blocked')

        result = await run.agent(code_mode(), InputGuardrail(guard=guard, parallel=True)).run('go')

        assert result.output == 'blocked'
        assert run.calls == []

    @pytest.mark.parametrize('wrapper', ['capability-func', 'prefix-tools'])
    async def test_parallel_guard_behind_a_wrapper_blocks_nested_calls(
        self, wrapper: str, monkeypatch: pytest.MonkeyPatch
    ):
        run = GuardedRun()
        run.watch_verdict_wait(monkeypatch)

        async def guard(prompt: str) -> GuardrailResult:
            await run.verdict_due.wait()
            return GuardrailResult.block('blocked')

        guardrail = InputGuardrail[None](guard=guard, parallel=True)
        wrapped: AgentCapability[None] = (
            guardrail.prefix_tools('guard') if wrapper == 'prefix-tools' else (lambda ctx: guardrail)
        )

        result = await run.agent(CodeMode[None](speculate=['ping']), wrapped).run('go')

        assert result.output == 'blocked'
        assert run.calls == []

    async def test_concurrent_run_allowed_first_does_not_release_another_runs_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        run = GuardedRun()
        run.watch_verdict_wait(monkeypatch)
        run.first_chunk_released.clear()

        async def guard(prompt: str) -> GuardrailResult:
            if prompt == 'allowed':
                return GuardrailResult.allow()
            await run.verdict_due.wait()
            return GuardrailResult.block('blocked')

        async def reply(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield 'done'

        agent = run.agent(CodeMode[None](speculate=['ping']), InputGuardrail(guard=guard, parallel=True))
        blocked_run = asyncio.create_task(agent.run('blocked'))
        try:
            await asyncio.wait_for(run.requested.wait(), timeout=READINESS_WAIT_TIMEOUT)
            allowed = await agent.run('allowed', model=FunctionModel(stream_function=reply))
        finally:
            run.first_chunk_released.set()
        blocked = await blocked_run

        assert allowed.output == 'done'
        assert blocked.output == 'blocked'
        assert run.calls == []

    @pytest.mark.parametrize('code_mode', STREAMED_TIERS)
    async def test_parallel_guard_that_allows_releases_nested_calls_during_stream(
        self, code_mode: Callable[[], CodeMode[None]], monkeypatch: pytest.MonkeyPatch
    ):
        run = GuardedRun()
        run.watch_verdict_wait(monkeypatch)

        async def guard(prompt: str) -> GuardrailResult:
            await run.verdict_due.wait()
            return GuardrailResult.allow()

        result = await run.agent(code_mode(), InputGuardrail(guard=guard, parallel=True)).run('go')

        assert result.output == 'done'
        assert run.calls_at_verdict == []
        assert sorted(run.calls) == [1, 2]

    @pytest.mark.parametrize('code_mode', STREAMED_TIERS)
    async def test_call_over_tool_calls_limit_waits_for_a_parallel_guard_that_blocks(
        self, code_mode: Callable[[], CodeMode[None]], monkeypatch: pytest.MonkeyPatch
    ):
        """The verdict comes before the `tool_calls_limit` check, so the block ends the run, not the limit."""
        run = GuardedRun()
        run.watch_verdict_wait(monkeypatch)

        async def guard(prompt: str) -> GuardrailResult:
            await run.verdict_due.wait()
            return GuardrailResult.block('blocked')

        agent = run.agent(code_mode(), InputGuardrail(guard=guard, parallel=True))
        # `run_code` and `ping(n=1)` together already exceed the limit.
        result = await agent.run('go', usage_limits=UsageLimits(tool_calls_limit=1))

        assert result.output == 'blocked'
        assert run.calls == []

    @pytest.mark.parametrize('code_mode', STREAMED_TIERS)
    async def test_tool_calls_limit_counts_calls_a_parallel_guard_released(
        self, code_mode: Callable[[], CodeMode[None]], monkeypatch: pytest.MonkeyPatch
    ):
        """Calls that waited for the verdict together are checked against `tool_calls_limit` one at a time."""
        run = GuardedRun()
        run.watch_verdict_wait(monkeypatch)

        async def guard(prompt: str) -> GuardrailResult:
            await run.verdict_due.wait()
            return GuardrailResult.allow()

        agent = run.agent(code_mode(), InputGuardrail(guard=guard, parallel=True))
        # Room for `run_code` and one nested call.
        with pytest.raises(UsageLimitExceeded, match=re.escape('tool_calls_limit of 2 (tool_calls=3)')):
            await agent.run('go', usage_limits=UsageLimits(tool_calls_limit=2))

        assert run.calls_at_verdict == []
        assert run.calls == [1]

    @pytest.mark.parametrize('with_sequential_guard', [False, True], ids=['no-guard', 'sequential-guard'])
    @pytest.mark.parametrize('code_mode', STREAMED_TIERS)
    async def test_without_parallel_guard_runs_nested_calls_without_waiting(
        self, code_mode: Callable[[], CodeMode[None]], with_sequential_guard: bool
    ):
        run = GuardedRun()
        capabilities: list[AgentCapability[None]] = [code_mode()]
        if with_sequential_guard:
            capabilities.append(InputGuardrail[None](guard=lambda prompt: True))

        result = await run.agent(*capabilities).run('go')

        assert result.output == 'done'
        assert run.calls_at_verdict is not None and 1 in run.calls_at_verdict
        assert sorted(run.calls) == [1, 2]

    @pytest.mark.parametrize('code_mode', STREAMED_TIERS)
    async def test_stream_error_before_the_verdict_runs_no_nested_call(
        self, code_mode: Callable[[], CodeMode[None]], monkeypatch: pytest.MonkeyPatch
    ):
        run = GuardedRun()
        run.watch_verdict_wait(monkeypatch)

        async def undecided(prompt: str) -> GuardrailResult:
            return await asyncio.Future[GuardrailResult]()

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            yield {0: DeltaToolCall(name='run_code', json_args=FIRST_CHUNK, tool_call_id='run_code_1')}
            await asyncio.wait_for(run.call_started.wait(), timeout=READINESS_WAIT_TIMEOUT)
            # The stream fails while a nested call waits, which cancels the undecided guard.
            raise RuntimeError('stream dropped')

        agent = Agent(
            FunctionModel(stream_function=stream),
            deps_type=type(None),
            capabilities=[code_mode(), InputGuardrail[None](guard=undecided, parallel=True)],
        )
        agent.tool_plain(run.ping)

        with pytest.raises(RuntimeError, match='stream dropped'):
            await agent.run('go')
        assert run.calls == []

    @pytest.mark.parametrize('code_mode', STREAMED_TIERS)
    async def test_sequential_guard_that_blocks_runs_no_code(self, code_mode: Callable[[], CodeMode[None]]):
        run = GuardedRun()

        result = await run.agent(
            code_mode(), InputGuardrail(guard=lambda prompt: GuardrailResult.block('blocked'))
        ).run('go')

        assert result.output == 'blocked'
        assert run.calls_at_verdict is None
        assert run.calls == []

    @pytest.mark.parametrize('blocks', [False, True], ids=['allow', 'block'])
    async def test_parallel_guard_deciding_after_a_retried_first_request(self, blocks: bool):
        run = GuardedRun()
        guard, retry_hooks = guard_deciding_after_a_retried_first_response(
            GuardrailResult.block('blocked') if blocks else GuardrailResult.allow()
        )

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Request 1 is rejected by the hook; its retry (request 2) calls `run_code`.
            if len(messages) == 1:
                return ModelResponse(parts=[TextPart('first')])
            if len(messages) == 3:
                return ModelResponse(parts=[ToolCallPart('run_code', {'code': 'await ping(n=1)'})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[CodeMode[None](), guard, retry_hooks],
        )
        agent.tool_plain(run.ping)

        result = await agent.run('go')

        assert (result.output, run.calls) == (('blocked', []) if blocks else ('done', [1]))

    @pytest.mark.parametrize('blocks', [False, True], ids=['allow', 'block'])
    async def test_launches_held_for_a_parallel_guard_across_a_retried_first_request(
        self, blocks: bool, monkeypatch: pytest.MonkeyPatch
    ):
        run = GuardedRun()
        run.watch_verdict_wait(monkeypatch)
        code_mode = CodeMode[None](speculate=['ping'])
        guard, retry_hooks = guard_deciding_after_a_retried_first_response(
            GuardrailResult.block('blocked') if blocks else GuardrailResult.allow()
        )

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            # Request 1 launches both calls and is retried while they wait; request 2 repeats the snippet.
            if len(messages) not in (1, 3):
                yield 'done'
                return
            yield {0: DeltaToolCall(name='run_code', json_args=FIRST_CHUNK, tool_call_id=f'run_code_{len(messages)}')}
            if len(messages) == 1:
                await asyncio.wait_for(run.call_started.wait(), timeout=READINESS_WAIT_TIMEOUT)
                return
            yield {0: DeltaToolCall(json_args=LAST_CHUNK)}

        agent = Agent(
            FunctionModel(stream_function=stream),
            deps_type=type(None),
            capabilities=[code_mode, guard, retry_hooks],
        )
        agent.tool_plain(run.ping)

        if blocks:
            # A streamed run whose guard blocks after the whole response streamed raises the block.
            with pytest.raises(SkipModelRequest):
                await agent.run('go')
            assert run.calls == []
        else:
            result = await agent.run('go')
            assert result.output == 'done'
            assert sorted(run.calls) == [1, 2]
            assert (code_mode.speculation_stats.launched, code_mode.speculation_stats.adopted) == (2, 2)

    @pytest.mark.parametrize('blocks', [False, True], ids=['allow', 'block'])
    async def test_parallel_guard_under_durable_execution(self, blocks: bool):
        run = GuardedRun()

        async def guard(prompt: str) -> GuardrailResult:
            return GuardrailResult.block('blocked') if blocks else GuardrailResult.allow()

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('run_code', {'code': 'await ping(n=1)'})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(respond),
            name='guarded',
            deps_type=type(None),
            capabilities=[CodeMode[None](), InputGuardrail[None](guard=guard, parallel=True), RecordingDurability()],
        )
        agent.tool_plain(run.ping)

        result = await agent.run('go')

        assert (result.output, run.calls) == (('blocked', []) if blocks else ('done', [1]))
