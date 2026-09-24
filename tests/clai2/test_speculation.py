"""Speculative execution without the sandbox: the switch, the row, sandbox call display, and the chord.

These run without `pydantic-monty`; `test_speculative_mode.py` covers the sandbox wiring.
"""

import io
import sys
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import PartStartEvent
from pydantic_ai.messages import AgentStreamEvent, FunctionToolCallEvent, TextPart, ToolCallPart
from rich.console import Console
from rich.text import Text

from pydantic_clai2 import StreamRenderer, theme
from pydantic_clai2.command_context import CommandContext
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import Settings, resolve_settings
from pydantic_clai2.image_input import ImageInput
from pydantic_clai2.interrupts import Interrupts
from pydantic_clai2.live_prompt import LivePrompt
from pydantic_clai2.sandbox_calls import SandboxCallOrder, SandboxCallStartedEvent
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.speculation import Speculation, SpeculationCounters


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def plain(row: str) -> str:
    return Text.from_ansi(row).plain


def speculation(tmp_path: Path, console: Console | None = None) -> Speculation:
    store = SettingsStore(tmp_path / 'config.db')
    context = CommandContext(
        settings=store.load(),
        store=store,
        clear_history=lambda: None,
        apply_setting=lambda key, settings: None,
    )
    return Speculation(context=context, console=console or Console(file=io.StringIO()))


class TestSwitch:
    def test_off_by_default_and_older_settings_load(self) -> None:
        assert Settings().speculative_code_mode is False
        assert resolve_settings({'display.thinking': False}).speculative_code_mode is False
        assert resolve_settings({'run.speculative_code_mode': True}).speculative_code_mode is True

    def test_toggle_persists_and_shows_row_only_while_on(self, tmp_path: Path) -> None:
        switch = speculation(tmp_path)
        assert switch.row() == ''
        assert switch.capabilities([]) == []

        assert switch.toggle() == 'Speculative execution on from the next turn. Ctrl+X Ctrl+S toggles it.'
        assert SettingsStore(tmp_path / 'config.db').overrides() == {'run.speculative_code_mode': True}
        assert plain(switch.row()).startswith('Speculative Execution  0 hits')

        switch.counters.hits = 2
        assert switch.toggle().startswith('Speculative execution off')
        assert switch.row() == ''
        assert switch.capabilities([]) == []
        switch.toggle()
        assert plain(switch.row()).startswith('Speculative Execution  2 hits')

    def test_missing_sandbox_dependency_warns_and_runs_natively(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = io.StringIO()
        switch = speculation(tmp_path, Console(file=output, width=200))
        switch.toggle()
        monkeypatch.setitem(sys.modules, 'pydantic_clai2.speculative_mode', None)
        assert switch.capabilities([]) == []
        assert 'Speculative execution is unavailable' in output.getvalue()


class TestRow:
    def test_matches_code_puppy_layout(self) -> None:
        counters = SpeculationCounters(hits=29, misses=1, wasted=0, speculative_ms=520, eager_ms=6_549)
        assert plain(counters.row()) == (
            'Speculative Execution  29 hits \u00b7 1 miss \u00b7 0 wasted    saved \u2265 7.0s'
        )

    def test_counts_light_up_only_when_nonzero(self) -> None:
        idle = SpeculationCounters().row()
        assert theme.sgr(theme.SUCCESS, bold=True) not in idle
        busy = SpeculationCounters(hits=1, wasted=1, eager_ms=100).row()
        assert f'{theme.sgr(theme.SUCCESS, bold=True)}1 hit' in busy
        assert f'{theme.sgr(theme.ERROR, bold=True)}1 wasted' in busy
        assert f'{theme.sgr(theme.MUTED)}0 misses' in busy
        assert f'{theme.sgr(theme.SUCCESS, bold=True)}saved \u2265 0.1s' in busy


class TestSandboxCallOrder:
    async def test_plugin_renderers_see_sandbox_calls(self) -> None:
        seen: list[str] = []

        def plugin(event: AgentStreamEvent) -> str | None:
            if isinstance(event, FunctionToolCallEvent):
                seen.append(event.part.tool_name)
                return 'drawn by plugin'
            return None

        output = io.StringIO()
        renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None, renderers=[plugin])
        call = ToolCallPart(tool_name='read_file', args={'path': 'a.py'}, tool_call_id='parent__1')
        await renderer.on_stream_event(SandboxCallStartedEvent(tool_call_id='parent__1', call=call))
        assert seen == ['run_code', 'read_file']
        assert output.getvalue().count('drawn by plugin') == 2

    def test_each_run_code_header_renders_once(self) -> None:
        order = SandboxCallOrder()
        call = ToolCallPart(tool_name='read_file', tool_call_id='eager__1')
        assert len(order.tool_events(SandboxCallStartedEvent(call=call)) or []) == 2
        assert len(order.tool_events(SandboxCallStartedEvent(call=call)) or []) == 1
        late = FunctionToolCallEvent(ToolCallPart(tool_name='run_code', tool_call_id='eager'))
        assert order.tool_events(late) == []

        after_stream = FunctionToolCallEvent(ToolCallPart(tool_name='run_code', tool_call_id='parent'))
        assert order.tool_events(after_stream) is None
        call = ToolCallPart(tool_name='read_file', tool_call_id='parent__1')
        assert order.tool_events(SandboxCallStartedEvent(call=call)) == [FunctionToolCallEvent(call)]
        direct = FunctionToolCallEvent(ToolCallPart(tool_name='read_file', tool_call_id='direct'))
        assert order.tool_events(direct) is None
        assert order.tool_events(direct) is None
        assert order.tool_events(SandboxCallStartedEvent(call=replace(call, tool_call_id='parent__2'))) == [
            FunctionToolCallEvent(replace(call, tool_call_id='parent__2'))
        ]
        assert order.tool_events(PartStartEvent(index=0, part=TextPart('hi'))) is None


@contextmanager
def live_prompt(
    *, height: int, pinned: Callable[[], str], chords: Mapping[str, Callable[[], str]] | None = None
) -> Generator[LivePrompt]:
    console = Console(file=io.StringIO(), force_terminal=True, width=100, height=height)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        yield LivePrompt(
            console=console,
            commands=Commands(),
            history=InMemoryHistory(),
            images=ImageInput(),
            interrupts=Interrupts(),
            toolbar=lambda: [('', 'ready')],
            clock=lambda: 0,
            chords=chords,
            pinned=pinned,
        )


class TestChord:
    def test_ctrl_x_ctrl_s_toggles_and_pins_the_row(self) -> None:
        toggles: list[str] = []

        def toggle() -> str:
            toggles.append('toggled')
            return 'Speculative execution on'

        with live_prompt(height=24, pinned=lambda: 'PINNED ROW', chords={'ctrl-x ctrl-s': toggle}) as live:
            live.feed('ctrl-x')
            live.feed('ctrl-s')
            frame = [plain(row) for row in live.frame()]
            assert toggles == ['toggled']
            assert frame[-2:] == ['PINNED ROW', 'Speculative execution on']

            live.feed('ctrl-x')
            live.feed('a')
            assert toggles == ['toggled']
            assert live.buffer.text == 'a'
            assert plain(live.frame()[-1]).startswith('ready')

    @pytest.mark.parametrize(('height', 'shown'), [(6, False), (7, True)])
    def test_pinned_row_only_takes_a_spare_row(self, height: int, shown: bool) -> None:
        with live_prompt(height=height, pinned=lambda: 'PINNED ROW') as live:
            frame = [plain(row) for row in live.frame()]
        # `PromptSurface.paint` keeps the last `height - 2` rows; the title rule must survive.
        assert len(frame) <= height - 2
        assert frame[0].startswith('\u2500')
        assert ('PINNED ROW' in frame) is shown
