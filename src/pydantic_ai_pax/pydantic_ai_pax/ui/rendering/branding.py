"""Persistent Pax banner, separate from the stdlib startup splash."""

from pyfiglet import Figlet
from rich.console import Console
from rich.text import Text

from pydantic_ai_pax.ui.rendering import theme


def print_banner(console: Console) -> None:
    """Print Pax in the same `ansi_shadow` font as Code Puppy, in Pydantic brand colours.

    The logo is branding, not UI chrome, so it skips `theme.color` and keeps the same colours under every palette,
    including when a theme change repaints the transcript.
    """
    banner = Figlet(font='ansi_shadow', width=200).renderText('Pax')
    with theme.branded():
        if console.width < max(map(len, banner.splitlines())):
            console.print('Pax', style=theme.LOGO)
            return
        for index, line in enumerate(banner.splitlines()):
            console.print(Text(line, style=theme.BANNER[min(index // 2, 2)]))
