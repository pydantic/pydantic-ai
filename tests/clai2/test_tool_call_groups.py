"""`display.tool_calls = grouped` counts consecutive calls by tool on one live line."""

import io
import itertools
from dataclasses import dataclass
from pathlib import Path

import pytest
from rich.console import Console, RenderableType
from rich.text import Text

from pydantic_ai import (
    AgentStreamEvent,
    CapabilityEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
)
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai_harness.filesystem import FileEditedEvent, FileWrittenEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandOutputEvent, CommandStartedEvent
from pydantic_clai2 import StreamRenderer
from pydantic_clai2.commands import set_completions
from pydantic_clai2.config import Settings, resolve_settings
from pydantic_clai2.ui.menus.field_menu import CUSTOM, FieldMenu
from pydantic_clai2.ui.menus.set_menu import SettingsSource
from pydantic_clai2.ui.menus.tool_calls_preview import tool_calls_preview
from pydantic_clai2.ui.prompt.prompt_surface import PromptSurface
from pydantic_clai2.ui.rendering.tool_group import ToolCallGroup
from tests.clai2.menu_script import make_context


@dataclass(kw_only=True)
class Notice(CapabilityEvent, namespace='test', name='tool_group_notice'):
    pass


def call(name: str, index: int = 0, **args: object) -> FunctionToolCallEvent:
    return FunctionToolCallEvent(part=ToolCallPart(name, args, tool_call_id=f'{name}-{index}'))


async def feed(renderer: StreamRenderer, names: str) -> None:
    """One call per whitespace-separated name, e.g. `shell shell grep`."""
    for index, name in enumerate(names.split()):
        await renderer.on_stream_event(call(name, index))


def grouped(
    output: io.StringIO,
    *,
    terminal: bool = False,
    width: int = 80,
    show_thinking: bool = True,
    show_tool_output: bool = False,
) -> StreamRenderer:
    console = Console(file=output, width=width, force_terminal=terminal, color_system=None)
    return StreamRenderer(
        console,
        stop_loading=lambda: None,
        tool_calls='grouped',
        show_thinking=show_thinking,
        show_tool_output=show_tool_output,
    )


async def test_consecutive_calls_share_one_line_until_another_tool_arrives() -> None:
    output = io.StringIO()
    renderer = grouped(output)
    await feed(renderer, 'shell shell shell shell grep grep shell shell shell')
    assert output.getvalue() == ''  # Not a terminal: nothing prints until the line is final.
    await renderer.finish()
    assert output.getvalue() == '● shell 4, grep 2, shell 3\n\n'


async def test_terminal_redraws_the_line_in_place_as_the_count_grows() -> None:
    output = io.StringIO()
    renderer = grouped(output, terminal=True)
    await feed(renderer, 'shell shell grep')
    assert output.getvalue() == '\r● shell 1\r● shell 2\r● shell 2, grep 1'
    await renderer.finish()
    assert output.getvalue().endswith('grep 1\n\n')
    await renderer.finish()  # Nothing left to close.
    assert output.getvalue().count('\n') == 2


async def test_finish_without_calls_prints_nothing() -> None:
    output = io.StringIO()
    await grouped(output).finish()
    assert output.getvalue() == ''


async def test_text_ends_the_group_and_later_calls_start_a_new_one() -> None:
    output = io.StringIO()
    renderer = grouped(output)
    await feed(renderer, 'shell shell')
    await renderer.on_stream_event(PartStartEvent(index=1, part=TextPart(content='Done.\n')))
    await feed(renderer, 'shell')
    await renderer.finish()
    assert output.getvalue() == '● shell 2\n\nDone.\n\n● shell 1\n\n'


async def test_hidden_thinking_does_not_end_the_group() -> None:
    output = io.StringIO()
    renderer = grouped(output, show_thinking=False)
    await feed(renderer, 'shell')
    await renderer.on_stream_event(PartStartEvent(index=1, part=ThinkingPart(content='hmm')))
    await feed(renderer, 'shell')
    await renderer.finish()
    assert output.getvalue() == '● shell 2\n\n'


async def test_a_part_that_shows_nothing_does_not_end_the_group() -> None:
    output = io.StringIO()
    renderer = grouped(output)
    await feed(renderer, 'shell')
    reasoning = ThinkingPart(content='', signature='encrypted')  # Reasoning returned without a summary.
    await renderer.on_stream_event(PartStartEvent(index=1, part=reasoning))
    await renderer.on_stream_event(PartEndEvent(index=1, part=reasoning))
    await feed(renderer, 'shell')
    await renderer.on_stream_event(PartStartEvent(index=2, part=TextPart(content='')))
    await renderer.on_stream_event(PartDeltaEvent(index=2, delta=TextPartDelta(content_delta='Done.\n')))
    await renderer.finish()
    assert output.getvalue() == '● shell 2\n\nDone.\n\n'


def test_a_new_tool_leaves_room_for_its_count_to_grow() -> None:
    output = io.StringIO()
    group = ToolCallGroup(Console(file=output, width=20), colors=None)
    # `● shell 2, search 1` fits in 20 cells, but `search 100` would not.
    for name in ['shell'] * 2 + ['search'] * 100:
        group.add(name)
    group.close()
    assert output.getvalue() == '● shell 2\n● search 100\n\n'


def test_a_tool_too_wide_for_its_row_keeps_its_count() -> None:
    output = io.StringIO()
    group = ToolCallGroup(Console(file=output, width=24), colors=None)
    for _ in range(12):
        group.add('mcp_github__list_review_comments')
    group.close()
    assert output.getvalue() == '● mcp_github__list_r… 12\n\n'


def test_a_count_past_the_room_left_for_it_cuts_the_name_instead() -> None:
    output = io.StringIO()
    group = ToolCallGroup(Console(file=output, width=20), colors=None)
    for name in ['shell'] * 2 + ['grep'] * 10_000:
        group.add(name)
    group.close()
    assert output.getvalue() == '● shell 2, gr… 10000\n\n'


async def test_tool_part_starts_and_results_do_not_end_or_count() -> None:
    output = io.StringIO()
    renderer = grouped(output)
    for index in range(2):
        await renderer.on_stream_event(
            PartStartEvent(index=index, part=ToolCallPart('shell', {}, tool_call_id=f'{index}'))
        )
        await renderer.on_stream_event(call('shell', index))
        await renderer.on_stream_event(
            FunctionToolResultEvent(part=ToolReturnPart('shell', 'ok', tool_call_id=f'shell-{index}'))
        )
    await renderer.finish()
    assert output.getvalue() == '● shell 2\n\n'


async def test_plugin_renderings_end_the_group() -> None:
    output = io.StringIO()

    def renders_notes(event: AgentStreamEvent) -> RenderableType | None:
        return Text('note') if isinstance(event, FunctionToolCallEvent) and event.part.tool_name == 'note' else None

    renderer = StreamRenderer(
        Console(file=output, width=80), stop_loading=lambda: None, tool_calls='grouped', renderers=[renders_notes]
    )
    await feed(renderer, 'shell shell note shell')
    await renderer.finish()
    assert output.getvalue() == '● shell 2\n\nnote\n\n● shell 1\n\n'


def edited(index: int) -> FileEditedEvent:
    return FileEditedEvent(
        path='a.py',
        root_dir='/tmp',
        content_hash='h',
        diff='-old\n+new',
        truncated=False,
        tool_call_id=f'edit_file-{index}',
    )


async def test_shell_and_arguments_are_ignored_but_edits_and_writes_print_as_in_detailed() -> None:
    output = io.StringIO()
    renderer = grouped(output, show_tool_output=True)
    await renderer.on_stream_event(call('shell', command='ls'))
    for event in (
        CommandStartedEvent(tool_call_id='shell-0', command='ls', pid=1),
        CommandOutputEvent(tool_call_id='shell-0', text='a.py\n'),
        CommandFinishedEvent(
            tool_call_id='shell-0',
            exit_code=0,
            pid=1,
            output_path='/o',
            status_path='/s',
            total_lines=1,
            truncated=False,
        ),
    ):
        await renderer.on_stream_event(event)
    await renderer.on_stream_event(Notice())  # An unrelated capability event leaves the group alone.
    await renderer.on_stream_event(call('edit_file', 1, path='a.py'))
    await renderer.on_stream_event(edited(1))
    await renderer.on_stream_event(call('write_file', 2, path='b.py'))
    await renderer.on_stream_event(
        FileWrittenEvent(path='b.py', root_dir='/tmp', content_hash='h', tool_call_id='write_file-2')
    )
    await renderer.finish()
    assert output.getvalue() == '● shell 1\n\n● edit_file a.py\n\n-old\n+new\n\n● write_file b.py\n\n\n'


async def test_a_diff_ends_a_group_opened_after_its_call() -> None:
    output = io.StringIO()
    renderer = grouped(output)
    await renderer.on_stream_event(call('edit_file', 1, path='a.py'))
    await renderer.on_stream_event(call('read_file', 2, path='b.py'))
    await renderer.on_stream_event(edited(1))
    await renderer.finish()
    assert output.getvalue() == '● edit_file a.py\n\n● read_file 1\n\n-old\n+new\n\n'


async def test_a_tool_that_no_longer_fits_starts_the_next_line() -> None:
    output = io.StringIO()
    renderer = grouped(output, width=20)
    await feed(renderer, 'shell shell grep read_file grep')
    await renderer.finish()
    assert output.getvalue() == '● shell 2, grep 1\n● read_file 1\n● grep 1\n\n'


async def test_wrapped_lines_on_a_terminal_each_end_where_they_were_drawn() -> None:
    output = io.StringIO()
    renderer = grouped(output, terminal=True, width=20)
    await feed(renderer, 'shell shell grep read_file')
    await renderer.finish()
    assert output.getvalue() == ('\r● shell 1\r● shell 2\r● shell 2, grep 1\n\r● read_file 1\n\n')


def _rows(surface: PromptSurface, *, width: int) -> list[str]:
    return [Text.from_ansi(row).plain for row in surface.transcript.frame(width=width, height=50).rows]


async def test_live_prompt_output_printed_during_a_group_lands_after_its_line() -> None:
    surface = PromptSurface(output=io.StringIO(), size=lambda: (80, 24))
    console = Console(file=surface, force_terminal=True, width=80)
    renderer = StreamRenderer(console, stop_loading=lambda: None, tool_calls='grouped')
    await feed(renderer, 'shell shell')
    console.print('> /set')  # A command typed mid-turn echoes while the group is open.
    await feed(renderer, 'shell')
    await renderer.finish()
    assert _rows(surface, width=80)[:3] == ['● shell 3', '', '> /set']


async def test_live_prompt_wraps_the_group_again_for_a_new_width() -> None:
    surface = PromptSurface(output=io.StringIO(), size=lambda: (20, 24))
    renderer = StreamRenderer(
        Console(file=surface, force_terminal=True, width=20), stop_loading=lambda: None, tool_calls='grouped'
    )
    await feed(renderer, 'shell shell grep read_file')
    await renderer.finish()
    assert _rows(surface, width=20)[:3] == ['● shell 2, grep 1', '● read_file 1', '']
    assert _rows(surface, width=80)[:3] == ['● shell 2, grep 1, read_file 1', '', '']


async def test_live_prompt_replay_inside_another_print_keeps_that_print_out_of_the_group() -> None:
    size = [80, 24]
    ticks = itertools.count(0, 10.0)
    surface = PromptSurface(output=io.StringIO(), size=lambda: (size[0], size[1]), clock=lambda: next(ticks))
    console = Console(file=surface, force_terminal=True)
    surface.paint(())  # An open editor: every write paints a frame.
    renderer = StreamRenderer(console, stop_loading=lambda: None, tool_calls='grouped')
    await feed(renderer, 'shell shell grep')
    await renderer.finish()
    size[0] = 60  # The frame this print paints replays the group at the new width.
    console.print('next')
    assert _rows(surface, width=60)[:3] == ['● shell 2, grep 1', '', 'next']


def test_live_prompt_replays_a_call_with_an_empty_name() -> None:
    surface = PromptSurface(output=io.StringIO(), size=lambda: (80, 24))
    group = ToolCallGroup(Console(file=surface, force_terminal=True, width=80), colors=None)
    group.add('')  # A model can call a tool it was never given, by any name.
    group.close()
    assert _rows(surface, width=40)[:2] == ['●  1', '']


async def test_abort_ends_the_line() -> None:
    output = io.StringIO()
    renderer = grouped(output)
    await feed(renderer, 'shell')
    await renderer.abort()
    assert output.getvalue() == '● shell 1\n\n'


def test_control_characters_in_a_name_are_inert() -> None:
    output = io.StringIO()
    group = ToolCallGroup(Console(file=output, width=80), colors=None)
    group.add('sh\x1b[2Jell')
    group.close()
    assert '\x1b' not in output.getvalue()


async def test_detailed_remains_the_default() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output, width=80), stop_loading=lambda: None)
    await feed(renderer, 'shell shell')
    assert output.getvalue() == '● shell\n\n● shell\n\n'


def test_setting_is_validated_and_defaults_to_detailed() -> None:
    assert Settings().tool_calls == 'detailed'
    assert resolve_settings({'display.tool_calls': 'grouped'}).tool_calls == 'grouped'
    with pytest.raises(ValueError):
        resolve_settings({'display.tool_calls': 'compact'})
    assert tuple(set_completions(['display.tool_calls', ''])) == ('detailed', 'grouped')


def test_preview_shows_the_same_calls_in_each_style() -> None:
    detailed = Text.from_ansi(tool_calls_preview('detailed', width=60)).plain
    assert detailed.splitlines()[0] == '● shell git status'
    assert detailed.count('● shell') == 4 and detailed.count('● read_file') == 2
    assert Text.from_ansi(tool_calls_preview('grouped', width=60)).plain == '● shell 3, read_file 2, shell 1'


def test_set_menu_previews_each_style_in_the_choice_picker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)
    menu = FieldMenu(SettingsSource(context))
    row = menu.row_for('display.tool_calls')
    assert row is not None and row.choices == ('detailed', 'grouped')
    output = io.StringIO()
    monkeypatch.setattr('sys.stdout', output)
    monkeypatch.setenv('COLUMNS', '100')
    monkeypatch.setenv('LINES', '30')
    keys = iter(['down', 'enter'])
    monkeypatch.setattr('pydantic_clai2.ui.menus.field_menu.menu_key', lambda: next(keys))
    result = menu.build_choices(row).run()
    assert result.item is not None and result.item.value == 'grouped'
    plain = Text.from_ansi(output.getvalue()).plain
    assert '● shell git status' in plain
    assert '● shell 3, read_file 2, shell 1' in plain
    # The typed-value row has no sample to preview.
    keys = iter(['down', 'down', 'enter'])
    result = menu.build_choices(row).run()
    assert result.item is not None and result.item.value == CUSTOM
