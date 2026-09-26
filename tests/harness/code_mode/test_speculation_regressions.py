"""Regressions for speculation under run cancellation and malformed stream arguments.

These pin two failure classes the suite's behavioral tests do not reach:

1. A run cancelled through an anyio scope leaked speculative launches. The scope's
   level-triggered re-cancellation interrupted ``close()`` at its first cleanup await,
   abandoning every later watch; the abandoned launches kept running past the run's end.
   A failed step keeps its launches for the retry, which is what lets two watches hold
   unclaimed launches at the same time.
2. Raw surrogate code points in streamed ``run_code`` arguments made pydantic-core's JSON
   decoder raise ``TypeError`` inside the speculative watcher. The watcher is a robustness
   feature: an undecodable prefix must mean no speculation, never a crashed run.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import anyio
import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from pydantic_ai_harness.code_mode import CodeMode, SpeculativeCallLaunchedEvent
from pydantic_ai_harness.code_mode._streaming import decode_partial_args

from .test_speculation import observe, prepared_toolset

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _branch_code(taken: str, other: str) -> str:
    """A snippet with literal calls on both branches, so the watcher launches both."""
    return f'if True:\n    await search(query="{taken}")\nelse:\n    await search(query="{other}")\n'


class TestRunCancellation:
    async def test_cancellation_cancels_launches_in_every_part(self):
        """A run cancelled mid-stream cancels the speculative launches of every live part.

        Step one fails into a retry before dispatching anything, so its two branch launches
        stay queued. Step two (the retry) streams two more. When the run is cancelled,
        cleanup must cancel all four; the pre-fix ``close()`` aborted after the first
        watch's wait and leaked the rest, so this test used to end with tools still
        sleeping. The scope's cancellation is swallowed by the scope itself, so the run
        simply ends without a result.
        """
        started: list[str] = []
        cancelled: list[str] = []
        first_started = asyncio.Event()
        all_started = asyncio.Event()

        async def search(query: str) -> str:
            """Return a canned result."""
            started.append(query)
            if len(started) == 2:
                first_started.set()
            if len(started) == 4:
                all_started.set()
            try:
                return await asyncio.Future[str]()
            except asyncio.CancelledError:
                cancelled.append(query)
                raise

        step_one = _branch_code('one', 'two') + 'bad syntax here'
        step_two = _branch_code('three', 'four')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            prior_calls = sum(
                1 for m in messages if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart)
            )
            code = step_one if prior_calls == 0 else step_two
            args = json.dumps({'code': code})
            yield {0: DeltaToolCall(name='run_code')}
            yield {0: DeltaToolCall(json_args=args)}
            await first_started.wait()

        model = FunctionModel(stream_function=stream)
        agent = Agent(model, deps_type=type(None), capabilities=[CodeMode[None](speculate=['search'])], tools=[search])

        saw_all_four = False
        scope = anyio.CancelScope()
        with scope:
            async with anyio.create_task_group() as tg:

                async def do_run() -> None:
                    await agent.run('go')

                async def canceller() -> None:
                    nonlocal saw_all_four
                    await all_started.wait()
                    saw_all_four = len(started) == 4
                    scope.cancel()

                tg.start_soon(do_run)
                tg.start_soon(canceller)

        assert saw_all_four, f'not all four launches started before cancellation: {sorted(started)}'
        assert sorted(started) == ['four', 'one', 'three', 'two']
        # The regression: every unclaimed launch was cancelled at run end, in every part.
        assert sorted(cancelled) == ['four', 'one', 'three', 'two']
        leftovers = [t for t in asyncio.all_tasks() if 'search' in repr(t.get_coro())]
        assert not leftovers, f'leaked speculative launches still running: {leftovers}'


class TestMalformedStreamArgs:
    async def test_raw_surrogates_in_streamed_args_launch_nothing(self):
        """Raw surrogate code points in streamed args degrade to no speculation, not a crash.

        A lone high surrogate in the JSON text is legal Python and legal UTF-16-in-progress,
        but pydantic-core's decoder rejects it with ``TypeError`` instead of a parse error.
        The watcher must treat that like any other undecodable prefix.
        """

        async def search(query: str) -> str:
            """Return a canned result."""
            raise AssertionError('malformed arguments must not launch tools')  # pragma: no cover

        args_text = '{"code": "a = await search(query=\u0022al\ud800pha\u0022)"'
        assert '\ud800' in args_text  # a raw (unescaped) high surrogate in the streamed text
        part = ToolCallPart(tool_name='run_code', args='', tool_call_id='surrogate_part')
        async with prepared_toolset([Tool(search)], CodeMode[None](speculate=['search'])) as (
            run_capability,
            _toolset,
            ctx,
            _run_code,
        ):
            await observe(
                run_capability,
                ctx,
                [
                    PartStartEvent(index=0, part=part),
                    PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta=args_text)),
                    PartEndEvent(index=0, part=ToolCallPart(tool_name='run_code', args={'code': 'x'})),
                ],
            )
            assert run_capability.speculation_stats.launched == 0
            assert ctx._event_stream_buffer is not None
            launches = [e for e in ctx._event_stream_buffer if isinstance(e, SpeculativeCallLaunchedEvent)]
            assert not launches


class TestPartialArgsDecoding:
    @pytest.mark.parametrize(
        'args_text',
        [
            '{"code": "a\ud800b"}',  # raw high surrogate
            '{"code": "a\udcb9b"}',  # raw low surrogate
            '{"code": \ud800',  # raw surrogate outside any string value
            '{"code": "a\\ud800b"}',  # unpaired surrogate escape
        ],
    )
    def test_undecodable_prefixes_return_none(self, args_text: str) -> None:
        """Undecodable prefixes -- including raw surrogates -- mean no partial args."""
        assert decode_partial_args(args_text) is None

    def test_decodable_prefix_still_decodes(self) -> None:
        """The robustness fix must not eat prefixes that do decode."""
        # JSON text carries escaped newlines, exactly as a provider would stream them.
        assert decode_partial_args('{"code": "a = 1\\nb = 2"') == {'code': 'a = 1\nb = 2'}
        assert decode_partial_args('{"code": "a\\ud83d\\udc36b"}') == {'code': 'a\U0001f436b'}
