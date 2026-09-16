"""Persistent CLAI banner, separate from the stdlib startup splash."""

from pyfiglet import Figlet
from rich.console import Console
from rich.text import Text


def print_banner(console: Console) -> None:
    """Print CLAI 2.0 in the same `ansi_shadow` font as Code Puppy."""
    banner = Figlet(font='ansi_shadow', width=200).renderText('CLAI 2.0')
    if console.width < max(map(len, banner.splitlines())):
        console.print('CLAI 2.0', style='bold cyan')
        return
    for index, line in enumerate(banner.splitlines()):
        console.print(Text(line, style=('bright_blue', 'bright_cyan', 'bright_green')[min(index // 2, 2)]))
