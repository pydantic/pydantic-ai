"""Speculative execution's sandbox wiring: the tool fold, eager timing, counting, and sandbox call display."""

import pytest

pytest.importorskip('pydantic_monty')

import io
import json
import re
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import anyio
from pydantic_ai import Agent, AgentRunResultEvent, ModelRetry, PartStartEvent, RunContext, Tool
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.code_mode import (
    SpeculativeCallClaimedEvent,
    SpeculativeCallEvictedEvent,
    SpeculativeCallMissedEvent,
)
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.filesystem import FileSystem
from rich.console import Console

from pydantic_clai2 import StreamRenderer
from pydantic_clai2.command_context import CommandContext
from pydantic_clai2.customization import customization_guide
from pydantic_clai2.eager_timing import EagerExecutionCompletedEvent
from pydantic_clai2.sandbox_calls import SandboxCallFinishedEvent, SandboxCallStartedEvent
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.speculation import Speculation, SpeculationCounters
from pydantic_clai2.speculative_mode import (
    GUIDANCE,
    NATIVE_TOOLS,
    SPECULATIVE_TOOLS,
    ShowSandboxCalls,
    SpeculativeExecution,
    speculative_capabilities,
)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def streamed(respond: Callable[[list[ModelMessage], AgentInfo], ModelResponse]) -> FunctionModel:
    """Speculative runs stream, so replay each response as one delta per part."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        for index, part in enumerate(respond(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {index: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    return FunctionModel(stream_function=stream)


def fold_agent(model: FunctionModel, counters: SpeculationCounters, root: Path) -> Agent[None, str]:
    """Coder's own `FileSystem` provides the read tools, as in CLAI, so they may speculate."""
    for name in ('a.py', 'b.py'):
        (root / name).write_text(f'contents of {name}\n')
    return Agent(model, capabilities=[FileSystem(root_dir=root), *speculative_capabilities(counters)])


def test_switch_supplies_the_sandbox_capabilities(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    context = CommandContext(
        settings=store.load(), store=store, clear_history=lambda: None, apply_setting=lambda key, settings: None
    )
    switch = Speculation(context=context, console=Console(file=io.StringIO()))
    switch.toggle()
    assert [type(capability).__name__ for capability in switch.capabilities()] == [
        'CodeMode',
        'EagerTiming',
        'SpeculativeExecution',
        'ShowSandboxCalls',
    ]


class TestFold:
    async def test_shipped_tool_names_match_the_allowlists(self) -> None:
        seen: list[str] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.extend(tool.name for tool in info.function_tools)
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[Coder[None](repo_context=False), customization_guide()],
        )
        await agent.run('hi')
        assert {*SPECULATIVE_TOOLS, *NATIVE_TOOLS} <= set(seen)

    async def test_shell_folds_in_but_other_code_tools_stay_native(self) -> None:
        seen: list[AgentInfo] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info)
            return ModelResponse(parts=[TextPart('done')])

        def run_workflow(script: str) -> str:
            """Run a workflow script."""
            return script  # pragma: no cover -- only the tool surface is inspected.

        workflow = Tool(run_workflow, metadata={'code_arg_name': 'script'})
        agent: Agent[None, str] = Agent(
            streamed(respond),
            tools=[workflow],
            capabilities=[Coder(repo_context=False), *speculative_capabilities(SpeculationCounters())],
        )
        await agent.run('hi')
        [info] = seen
        tools = {tool.name: tool for tool in info.function_tools}
        assert sorted(tools) == ['edit_file', 'run_code', 'run_workflow', 'write_file']
        assert 'async def shell(' in (tools['run_code'].description or '')

    async def test_writes_stay_native_and_guidance_rides_along(self, tmp_path: Path) -> None:
        seen: list[AgentInfo] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info)
            return ModelResponse(parts=[TextPart('done')])

        await fold_agent(streamed(respond), SpeculationCounters(), tmp_path).run('hi')
        [info] = seen
        assert sorted(tool.name for tool in info.function_tools) == ['edit_file', 'run_code', 'write_file']
        assert GUIDANCE.strip() in (info.instructions or '')
        assert (info.model_settings or {}).get('anthropic_eager_input_streaming') is True

    @pytest.mark.parametrize(
        'metadata',
        [
            pytest.param(None, id='lookalike-name'),
            pytest.param({'read_only': True}, id='self-declared'),
            pytest.param({'annotations': {'readOnlyHint': True}}, id='mcp-hint'),
        ],
    )
    async def test_only_the_expected_capability_speculates(self, metadata: dict[str, object] | None) -> None:
        counters = SpeculationCounters()
        ran: list[str] = []
        snippet = 'if False:\n    await read_file(path="a.py")\n'

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) > 1:
                yield 'done'
                return
            args = json.dumps({'code': snippet})
            yield {0: DeltaToolCall(name='run_code')}
            for offset in range(0, len(args), 8):
                yield {0: DeltaToolCall(json_args=args[offset : offset + 8])}
                await anyio.sleep(0)

        def read_file(path: str) -> str:
            """A plugin's side-effecting lookalike of Coder's `read_file`."""
            ran.append(path)  # pragma: no cover -- the untaken branch must not launch it.
            return path  # pragma: no cover

        agent: Agent[None, str] = Agent(
            FunctionModel(stream_function=stream),
            tools=[Tool(read_file, metadata=metadata)],
            capabilities=speculative_capabilities(counters),
        )
        with anyio.fail_after(10):
            await agent.run('go')
        assert ran == []
        assert (counters.hits, counters.wasted) == (0, 0)


class TestEagerTiming:
    @pytest.mark.parametrize('rekey', [False, True])
    async def test_counts_speculative_hits_and_eager_overlap(self, rekey: bool, tmp_path: Path) -> None:
        counters = SpeculationCounters()
        probe_started, stream_finished = anyio.Event(), anyio.Event()
        head = 'text = await read_file(path="a.py")\nwaited = await probe()\npad = 0\n'
        tail = 'done = 1\n'

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) > 1:
                yield 'done'
                return
            args = json.dumps({'code': head + tail})
            yield {1: DeltaToolCall(name='run_code', tool_call_id='streamed')}
            if rekey:
                # Some providers only settle the call id after the part has started.
                yield {1: DeltaToolCall(tool_call_id='final')}
            split = args.index('done = 1')
            for offset in range(0, split, 8):
                yield {1: DeltaToolCall(json_args=args[offset : min(offset + 8, split)])}
                await anyio.sleep(0)
            await probe_started.wait()
            yield {1: DeltaToolCall(json_args=args[split:])}
            stream_finished.set()

        agent = fold_agent(FunctionModel(stream_function=stream), counters, tmp_path)

        @agent.tool_plain
        async def probe() -> str:
            """Hold the stream open until this call has started."""
            probe_started.set()
            await stream_finished.wait()
            return 'probed'

        with anyio.fail_after(10):
            async with agent.run_stream_events('go') as events:
                async for _ in events:
                    pass
        assert (counters.hits, counters.misses, counters.wasted) == (1, 0, 0)
        assert counters.eager_ms > 0

    async def test_unstreamed_and_restarted_snippets_report_nothing(self, tmp_path: Path) -> None:
        counters = SpeculationCounters()
        calls = iter(
            [
                ToolCallPart('run_code', {'code': 'name = "a"\nx = await read_file(path=name)\nx'}),
                ToolCallPart('run_code', {'code': 'y = 1', 'restart': True}),
            ]
        )

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[next(calls, TextPart('done'))])

        await fold_agent(streamed(respond), counters, tmp_path).run('go')
        assert counters.eager_ms == 0
        assert counters.misses == 1


def claimed(*, ready: bool, elapsed_ms: float, launch_id: str = 'l') -> SpeculativeCallClaimedEvent:
    return SpeculativeCallClaimedEvent(
        launch_id=launch_id,
        nested_tool_call_id='c__1',
        wrapped_tool_name='read_file',
        ready_at_claim=ready,
        elapsed_ms=elapsed_ms,
    )


def context() -> RunContext[None]:
    return RunContext[None](deps=None, model=TestModel(), usage=RunUsage())


class TestSpeculativeExecution:
    async def test_partial_hits_count_without_time(self) -> None:
        counters = SpeculationCounters()
        report = SpeculativeExecution[None](counters)
        for event in (
            claimed(ready=True, elapsed_ms=250),
            claimed(ready=False, elapsed_ms=900),
            SpeculativeCallMissedEvent(sandbox_function='grep', wrapped_tool_name='grep', nested_tool_call_id='c__2'),
            SpeculativeCallEvictedEvent(launch_id='l2', wrapped_tool_name='grep', state='ready'),
            EagerExecutionCompletedEvent(saved_ms=1_000),
            PartStartEvent(index=0, part=TextPart('ignored')),
        ):
            await report.on_event(context(), event=event)
        assert counters == SpeculationCounters(hits=2, misses=1, wasted=1, speculative_ms=250, eager_ms=1_000)


SNIPPET = """\
text = await read_file(path="a.py")
name = "b.py"
other = await read_file(path=name)
try:
    await read_file(path="missing.py")
except Exception:
    pass
if text == "nope":
    skipped = await read_file(path="never.py")
try:
    await flaky()
except Exception:
    pass
try:
    await broken()
except Exception:
    pass
"""


class TestSandboxCallDisplay:
    async def test_calls_inside_run_code_render_like_direct_calls(self, tmp_path: Path) -> None:
        counters = SpeculationCounters()

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            if len(messages) > 1:
                yield 'done'
                return
            args = json.dumps({'code': SNIPPET})
            yield {1: DeltaToolCall(name='run_code')}
            for offset in range(0, len(args), 8):
                yield {1: DeltaToolCall(json_args=args[offset : offset + 8])}
                await anyio.sleep(0)

        agent = fold_agent(FunctionModel(stream_function=stream), counters, tmp_path)

        @agent.tool_plain
        def flaky() -> str:
            """Ask for a retry."""
            raise ModelRetry('try again')

        @agent.tool_plain
        def broken() -> str:
            """Fail outright."""
            raise RuntimeError('kaboom')

        output = io.StringIO()
        renderer = StreamRenderer(Console(file=output, width=120), stop_loading=lambda: None)
        reports: list[SandboxCallStartedEvent | SandboxCallFinishedEvent] = []
        with anyio.fail_after(10):
            async with agent.run_stream_events('go') as events:
                async for event in events:
                    if isinstance(event, AgentRunResultEvent):
                        continue
                    if isinstance(event, (SandboxCallStartedEvent, SandboxCallFinishedEvent)):
                        reports.append(event)
                    await renderer.on_stream_event(event)
        await renderer.finish()

        assert (counters.hits, counters.misses, counters.wasted) == (2, 1, 1)
        started = [event.call for event in reports if isinstance(event, SandboxCallStartedEvent)]
        assert [(call.tool_name, call.args) for call in started] == [
            ('read_file', {'path': 'a.py'}),
            ('read_file', {'path': 'b.py'}),
            ('read_file', {'path': 'missing.py'}),
            ('flaky', {}),
            ('broken', {}),
        ]
        finished = [event.result for event in reports if isinstance(event, SandboxCallFinishedEvent)]
        assert [call.tool_call_id for call in started] == [result.tool_call_id for result in finished]
        assert all(re.fullmatch(r'.+__\d+', call.tool_call_id) for call in started)
        # A failed speculative launch is still claimed and shown, like a cold failure.
        assert [type(result).__name__ for result in finished] == [
            'ToolReturnPart',
            'ToolReturnPart',
            'RetryPromptPart',
            'RetryPromptPart',
            'RetryPromptPart',
        ]
        headers = [line for line in output.getvalue().splitlines() if line.startswith('\u25cf')]
        assert headers == [
            '\u25cf run_code',
            "\u25cf read_file 'a.py' offset=0 limit=2000 lines",
            "\u25cf read_file 'b.py' offset=0 limit=2000 lines",
            "\u25cf read_file 'missing.py' offset=0 limit=2000 lines",
            '\u25cf flaky',
            '\u25cf broken',
        ]

    async def test_launch_evicted_while_running_is_not_held(self) -> None:
        show = ShowSandboxCalls[None]()
        call = ToolCallPart(tool_name='read_file', args={'path': 'a.py'}, tool_call_id='parent__spec_1')

        async def late(args: dict[str, object]) -> str:
            await show.on_event(
                context(),
                event=SpeculativeCallEvictedEvent(
                    launch_id=call.tool_call_id, wrapped_tool_name='read_file', state='pending'
                ),
            )
            return 'late'

        tool_def = ToolDefinition(name='read_file')
        assert await show.wrap_tool_execute(context(), call=call, tool_def=tool_def, args={}, handler=late) == 'late'
        # Nothing was held, so the claim reports nothing; emitting would fail, as this context has no stream.
        await show.on_event(context(), event=claimed(ready=True, elapsed_ms=1, launch_id=call.tool_call_id))
