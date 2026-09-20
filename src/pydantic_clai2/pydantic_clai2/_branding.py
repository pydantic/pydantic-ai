"""Persistent CLAI banner, separate from the stdlib startup splash."""

from pyfiglet import Figlet
from rich.console import Console
from rich.text import Text

from . import theme


def print_banner(console: Console) -> None:
    """Print CLAI 2.0 in the same `ansi_shadow` font as Code Puppy, in brand colours."""
    banner = Figlet(font='ansi_shadow', width=200).renderText('CLAI 2.0')
    if console.width < max(map(len, banner.splitlines())):
        console.print('CLAI 2.0', style=theme.color(theme.ACCENT))
        return
    for index, line in enumerate(banner.splitlines()):
        console.print(Text(line, style=theme.color(theme.BANNER[min(index // 2, 2)])))
