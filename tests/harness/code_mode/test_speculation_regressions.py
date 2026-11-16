"""Regressions for speculation under run cancellation, malformed stream arguments, and scan cost.

These pin three failure classes the suite's behavioral tests do not reach:

1. A run cancelled through an anyio scope leaked speculative launches. The scope's
   level-triggered re-cancellation interrupted `close()` at its first cleanup await,
   abandoning every later watch; the abandoned launches kept running past the run's end.
   A failed step keeps its launches for the retry, which is what lets two watches hold
   unclaimed launches at the same time.
2. Raw surrogate code points in streamed `run_code` arguments made pydantic-core's JSON
   decoder raise `TypeError` inside the speculative watcher. The watcher is a robustness
   feature: an undecodable prefix must mean no speculation, never a crashed run.
3. The text scan for launchable calls sliced and counted the whole prefix before every match,
   so one scan of the streamed code was quadratic in its length, run on the event loop.
"""

from __future__ import annotations

import ast
import asyncio
import json
import threading
import warnings
from collections.abc import AsyncIterator
from typing import SupportsIndex

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
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.code_mode import CodeMode, SpeculativeCallLaunchedEvent
from pydantic_ai_harness.code_mode._streaming import closed_statements, decode_partial_args, parse_code

from .test_speculation import emitted, observe, prepared_toolset


def _branch_code(taken: str, other: str) -> str:
    """A snippet with literal calls on both branches, so the watcher launches both."""
    return f'if True:\n    await search(query="{taken}")\nelse:\n    await search(query="{other}")\n'


class TestRunCancellation:
    async def test_cancellation_cancels_launches_in_every_part(self):
        """A run cancelled mid-stream cancels the speculative launches of every live part.

        Step one fails into a retry before dispatching anything, so its two branch launches
        stay queued. Step two (the retry) streams two more. When the run is cancelled,
        cleanup must cancel all four; the pre-fix `close()` aborted after the first
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
        but pydantic-core's decoder rejects it with `TypeError` instead of a parse error.
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
            assert ctx._event_stream_buffer is not None  # pyright: ignore[reportPrivateUsage]
            launches = [e for e in ctx._event_stream_buffer if isinstance(e, SpeculativeCallLaunchedEvent)]  # pyright: ignore[reportPrivateUsage]
            assert not launches


@pytest.mark.parametrize('eager', [False, True])
@pytest.mark.parametrize('literal', [r'"\("', r'r"\("', r'"\\("'])
async def test_streamed_backslashes_do_not_warn(eager: bool, literal: str) -> None:
    """Repeated analysis of regex literals stays quiet and preserves the tool argument."""
    calls: list[str] = []
    returns: list[object] = []

    async def search(query: str) -> str:
        """Return the query unchanged."""
        calls.append(query)
        return query

    code = f'first = await search(query={literal})\nsecond = await search(query={literal})\n[first, second]\n'

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        for message in messages:
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                    returns.append(part.content)
        if returns:
            yield 'done'
            return
        yield {0: DeltaToolCall(name='run_code')}
        for char in json.dumps({'code': code}):
            yield {0: DeltaToolCall(json_args=char)}

    agent = Agent(
        FunctionModel(stream_function=stream),
        deps_type=type(None),
        capabilities=[CodeMode[None](speculate=['search'], eager=eager)],
        tools=[search],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        # What the suite already ignores: a socket or coroutine another test leaked can warn here when collected.
        warnings.filterwarnings('ignore', category=ResourceWarning)
        warnings.filterwarnings('ignore', "coroutine '.*' was never awaited", RuntimeWarning)
        result = await agent.run('go')
    assert not caught
    assert result.output == 'done'
    assert calls == [r'\(', r'\(']
    assert returns == [[r'\(', r'\(']]


def test_analysis_warning_filter_is_narrow_and_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Analysis ignores invalid escapes only; unrelated diagnostics and normal Python remain intact."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert len(closed_statements('value = "\\("\nnext_value = 1\n')) == 1
        assert closed_statements('value = "\\("\nnext_value = (\n') == []
        assert isinstance(parse_code(r'"\("', mode='eval'), ast.Expression)
        with pytest.raises(SyntaxError):
            parse_code('value = (')
        with pytest.warns((SyntaxWarning, DeprecationWarning), match='invalid escape sequence'):
            ast.parse(r'"\("')

        def diagnostic_parse(code: str, *, filename: str = '<unknown>', mode: str) -> ast.Module:
            warnings.warn('unrelated parser diagnostic', SyntaxWarning)
            return ast.Module(body=[], type_ignores=[])

        with pytest.warns(SyntaxWarning, match='unrelated parser diagnostic'):
            with monkeypatch.context() as patch:
                patch.setattr(ast, 'parse', diagnostic_parse)
                parse_code('value = 1')


def test_analysis_does_not_hide_other_threads_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Older Python warning filters are process-global, so restrict them to the analysis filename."""

    def warn_elsewhere() -> None:
        warnings.warn_explicit('invalid escape sequence', SyntaxWarning, filename='user_code.py', lineno=1)

    real_parse = ast.parse

    def concurrent_parse(source: str, filename: str = '<unknown>', mode: str = 'exec') -> ast.AST:
        # Coverage's 3.14 `sysmon` core also calls `ast.parse(source)`, without a filename, the first time it
        # sees a branch in a file -- before `parse_code` installs its filter. Only the analysis parse, which
        # always passes a filename, may emit the other thread's warning.
        if filename == '<unknown>':  # pragma: lax no cover
            return real_parse(source, filename=filename, mode=mode)
        thread = threading.Thread(target=warn_elsewhere)
        thread.start()
        thread.join()
        return real_parse(source, filename=filename, mode=mode)

    with pytest.warns(SyntaxWarning, match='invalid escape sequence') as caught:
        with monkeypatch.context() as patch:
            patch.setattr(ast, 'parse', concurrent_parse)
            parse_code('value = 1')
    assert caught[0].filename == 'user_code.py'


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


class _CountingCode(str):
    """Streamed code that tallies the characters host-side scanning slices or counts out of it.

    Dict-shaped argument deltas reach the scanner without a JSON round-trip, so the scanner reads
    this very instance; C-level readers (`re`, `ast`) are not counted, Python-level ones are.
    """

    touched: int = 0

    def __getitem__(self, key: SupportsIndex | slice, /) -> str:
        piece = super().__getitem__(key)
        self.touched += len(piece)
        return piece

    def count(self, sub: str, start: SupportsIndex | None = None, end: SupportsIndex | None = None, /) -> int:
        self.touched += len(range(*slice(start, end).indices(len(self))))
        return super().count(sub, start, end)


_BLOCK = (
    "a = await lookup(key='k{i:02}')\n"
    "obj.lookup(key='attribute')\n"
    "def lookup(key='definition'): ...\n"
    'b = await lookup(\n'
    "    key='m{i:02}',\n"
    ')\n'
)
"""Six lines with one call on one line, one spanning three, and the two shapes the scan skips."""


async def _stream_whole(code: str) -> list[SpeculativeCallLaunchedEvent]:
    """Stream `code` as one dict-shaped delta and return the launches the text scan made."""

    async def lookup(key: str) -> str:
        """Return a canned result."""
        return key  # pragma: no cover - the launches are evicted before they run

    async with prepared_toolset([Tool(lookup)], CodeMode[None](speculate=['lookup'])) as (
        run_capability,
        _toolset,
        ctx,
        _run_code,
    ):
        part = ToolCallPart(tool_name='run_code', args={}, tool_call_id='scan_part')
        await observe(
            run_capability,
            ctx,
            [
                PartStartEvent(index=0, part=part),
                PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': code})),
                PartEndEvent(index=0, part=part),
            ],
        )
        return [event for event in emitted(ctx) if isinstance(event, SpeculativeCallLaunchedEvent)]


class TestTextScanCost:
    async def test_repeated_calls_launch_with_their_own_line_spans(self):
        """Every repetition launches its two calls with their own lines; the skipped shapes stay skipped."""
        launches = await _stream_whole(''.join(_BLOCK.format(i=i) for i in range(8)))

        assert [(event.arguments, event.line_start, event.line_end) for event in launches] == [
            launch
            for i in range(8)
            for launch in (({'key': f'k{i:02}'}, 6 * i + 1, 6 * i + 1), ({'key': f'm{i:02}'}, 6 * i + 4, 6 * i + 6))
        ]

    async def test_scan_work_grows_linearly_with_the_code(self):
        """Each extra batch of repetitions adds the same scan work, however much code precedes it."""
        touched: list[int] = []
        for blocks in (4, 8, 12):
            code = _CountingCode(''.join(_BLOCK.format(i=i) for i in range(blocks)))
            launches = await _stream_whole(code)
            assert len(launches) == 2 * blocks
            touched.append(code.touched)

        assert touched[2] - touched[1] == touched[1] - touched[0]

    async def test_nested_call_walks_stop_at_the_scan_budget(self, monkeypatch: pytest.MonkeyPatch):
        """Overlapping paren walks of nested calls stop at the budget instead of finishing first."""
        budget = 10_000
        monkeypatch.setattr('pydantic_ai_harness.code_mode._speculation.MAX_SCAN_WORK_CHARS', budget)
        for depth in (200, 400, 800):
            code = _CountingCode('x = ' + 'lookup(' * depth + "key='k'" + ')' * depth)
            assert not await _stream_whole(code)
            # Unbounded, the walks alone cover about `4 * depth ** 2` characters.
            assert code.touched <= 2 * budget
