"""Footer accounting and terminal restoration without provider calls."""

import asyncio
import io
import re

import pytest
from pydantic_ai import FunctionToolCallEvent, FunctionToolResultEvent, PartDeltaEvent, PartStartEvent
from pydantic_ai.messages import NativeToolCallPart, TextPart, ToolCallPart, ToolCallPartDelta, ToolReturnPart
from rich.console import Console

from pydantic_clai2.status import Status, StatusLine


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def test_estimate_includes_tool_argument_deltas() -> None:
    status = Status(model='test')
    status.observe(PartStartEvent(index=0, part=TextPart(content='abcd')))
    status.observe(PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta='12345678')))
    assert '~3 streamed tokens' in status.text()
    assert 'context: ?' in status.text()
    status.context_tokens = 1000
    status.output_tokens = 20
    assert 'context: 1,000 tokens' in status.text()
    assert '20 output tokens' in status.text()


def test_tool_status_transitions() -> None:
    status = Status()
    status.observe(PartStartEvent(index=0, part=NativeToolCallPart('web_search', {})))
    call = ToolCallPart('shell', {})
    status.observe(PartStartEvent(index=0, part=call))
    assert status.activity == 'tool: shell'
    status.observe(PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={})))
    status.observe(FunctionToolCallEvent(part=call))
    assert status.activity == 'running: shell'
    status.observe(FunctionToolResultEvent(part=ToolReturnPart('shell', 'done')))
    assert status.activity == 'working'


@pytest.mark.parametrize('truecolor', [False, True])
async def test_shimmer_without_spinner(monkeypatch: pytest.MonkeyPatch, truecolor: bool) -> None:
    monkeypatch.setenv('COLORTERM', 'truecolor' if truecolor else '')
    output = io.StringIO()
    frames: list[str] = []
    original_sleep = asyncio.sleep

    async def tick(delay: float) -> None:
        frames.append(output.getvalue().split('\x1b[2K')[-1])
        if len(frames) == 11:
            raise asyncio.CancelledError
        await original_sleep(0)

    monkeypatch.setattr('pydantic_clai2.status.asyncio.sleep', tick)
    async with StatusLine(Console(file=output, force_terminal=True, width=40, height=24), Status(model='test\x1b\n')):
        while len(frames) < 11:
            await original_sleep(0)
    plain = [re.sub(r'\x1b\[[0-9;]*m|\x1b8', '', frame) for frame in frames]
    assert plain[0].startswith('test?? | context:')
    assert all(frame == plain[0] for frame in plain)
    assert all(len(frame) == 39 for frame in plain)
    assert frames[0] != frames[10]
    assert ('38;2;' in frames[0]) == truecolor
    assert ('\x1b[38;2;155;119;255m' if truecolor else '\x1b[35m') in frames[0]
    assert ('\x1b[38;2;0;255;235m' if truecolor else '\x1b[96m') not in output.getvalue()
    assert '\n' not in output.getvalue()


async def test_tiny_terminal() -> None:
    async with StatusLine(Console(file=io.StringIO(), force_terminal=True, height=2), Status()):
        await asyncio.sleep(0)


async def test_redirected_output_has_no_footer() -> None:
    output = io.StringIO()
    async with StatusLine(Console(file=output, force_terminal=False), Status()):
        pass
    assert output.getvalue() == ''


@pytest.mark.parametrize('fail', [False, True])
async def test_cursor_restored_after_run(fail: bool) -> None:
    output = io.StringIO()
    try:
        async with StatusLine(Console(file=output, force_terminal=True, width=80, height=24), Status()):
            assert '\x1b[?25l' in output.getvalue()
            assert '\x1b[?25h' not in output.getvalue()
            if fail:
                raise ValueError('run failed')
    except ValueError:
        assert fail
    assert output.getvalue().endswith('\x1b[?25h')


async def test_cancellation_restores_scroll_region() -> None:
    output = io.StringIO()
    entered = asyncio.Event()

    async def run() -> None:
        async with StatusLine(Console(file=output, force_terminal=True, width=80, height=24), Status()):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert output.getvalue().startswith('\x1b[?25l\x1b7\x1b[1;23r')
    assert '\x1b[23;1H' not in output.getvalue()
    assert '\n' not in output.getvalue()
    assert '\x1b[r' in output.getvalue()
    assert output.getvalue().endswith('\x1b8\x1b[?25h')
