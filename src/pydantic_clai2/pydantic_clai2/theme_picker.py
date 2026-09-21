"""Select a bundled Termflow palette without changing colours while browsing."""

from io import StringIO

from rich.console import Console
from rich.padding import Padding
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text
from termflow import Parser, Renderer  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui import MenuBuilder, MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import Menu  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.terminal import terminal_size  # pyright: ignore[reportMissingTypeStubs]

from . import theme
from ._rendering import markdown_style
from .command_context import CommandContext
from .field_menu import TERMINAL, Runners
from .menu_worker import menu_key, run_worker
from .tool_output import print_tool_header


def theme_preview(name: str, *, width: int) -> str:
    """Render a sample conversation on the candidate background, without terminal palette changes."""
    output = StringIO()
    console = Console(file=output, width=width - 2, force_terminal=True, color_system='truecolor', highlight=False)
    with theme.use(lambda: name):
        console.print('CLAI 2.0', style=theme.color(theme.ACCENT))
        console.print('> Summarize this change')
        console.print('Thinking Checking the files...', style=theme.color(theme.THINKING))
        print_tool_header(console, name='read_file', argument='src/app.py')
        parser = Parser()
        renderer = Renderer(output=output, width=width - 2, style=markdown_style())
        for line in ('## Summary', 'Adds a searchable theme picker.'):
            for event in parser.parse_line(line):
                renderer.render(event)
        palette = theme.current()
        console.print(
            Syntax('return "ready"', 'python', theme='monokai', background_color=palette.bg if palette else 'default')
        )
        console.print('Warning: output truncated', style=theme.color(theme.WARNING))
        console.print('Error: example.py not found', style=theme.color(theme.ERROR))
        console.print('─' * (width - 2), style=theme.color(theme.MUTED))
        console.print('> Ask a follow-up')
        console.print('─' * (width - 2), style=theme.color(theme.MUTED))
        console.print('test | context: 2.4k | ready', style=theme.color(theme.MUTED))
        background = Style(color=palette.fg, bgcolor=palette.bg) if palette is not None else Style()
    sample = Text.from_ansi(output.getvalue().rstrip('\n'))
    preview = Console(file=StringIO(), width=width, force_terminal=True, color_system='truecolor')
    with preview.capture() as capture:
        preview.print(Padding(sample, (0, 1)), style=background)
    return capture.get()


def build_theme_picker(context: CommandContext) -> Menu:
    """Offer the existing default and bundled palettes with a conversation preview."""
    names = theme.names()
    list_width = 30
    return (
        MenuBuilder('Select theme')
        .style(markdown_style())
        .items(
            [MenuItem(f'{name}{" (current)" if name == context.settings.theme else ""}', value=name) for name in names]
        )
        .searchable()
        .initial_index(names.index(context.settings.theme))
        .list_width(list_width)
        .preview(lambda item: theme_preview(str(item.value), width=max(20, terminal_size()[0] - list_width - 4)))
        .footer_hint('type to filter - Enter apply - Esc close')
        .key_source(menu_key)
        .build()
    )


async def theme_command(context: CommandContext, args: list[str], *, runners: Runners = TERMINAL) -> str:
    """Persist a palette by name or picker; cancellation leaves the preference unchanged."""
    if len(args) > 1:
        raise ValueError('Usage: /theme [NAME]. Use /theme to browse Termflow palettes.')
    if args:
        name = args[0]
    else:
        result = await run_worker(lambda: runners.run_list(build_theme_picker(context)))
        if result.cancelled or result.item is None or not isinstance(result.item.value, str):
            return ''
        name = result.item.value
    context.set_setting(['display.theme', name])
    return ''
