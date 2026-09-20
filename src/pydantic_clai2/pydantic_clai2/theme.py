"""Pydantic brand palette and the roles CLAI paints with. Stdlib only, so the splash can import it.

Source: pydantic.dev `.agents/skills/pydantic-visual-identity/references/brand-identity.md`.
"""

import os

LITHIUM = '#E520E9'
"""Primary brand accent. Logo, interactive highlights."""
CALCIUM = '#FF6550'
"""Secondary accent. Warm counterpoint; errors."""
PURPLE = '#9B77FF'
"""Tertiary accent. Code syntax, decorative elements."""
AQUA = '#77FFD8'
"""Dark-mode accent. Code syntax, links."""
SUGAR = '#FBFFEA'
"""Headline text on dark surfaces."""
LIGHT_PURPLE = '#F0E0FD'
"""Highlight."""
DARK_PURPLE = '#36182D'
"""Dark background surface."""
ELEMENT_PURPLE = '#49353F'
"""Icons, outlines, secondary elements on dark surfaces."""
GREY = '#8F888E'
"""Muted text and code on dark surfaces."""
AI_CYAN = '#00FFEB'
"""Pydantic AI sub-brand accent."""
AI_YELLOW = '#D0FF71'
"""Pydantic AI sub-brand accent."""

ACCENT = f'bold {LITHIUM}'
"""Names, arguments, the thing to look at."""
INFO = AI_CYAN
"""Guidance the user asked for or needs next."""
WARNING = AI_YELLOW
"""Something stopped early but nothing broke."""
ERROR = CALCIUM
"""Something broke."""
MUTED = GREY
"""Housekeeping: tool markers, previews, hints."""
THINKING = PURPLE
"""The model's reasoning heading."""
BANNER = (LITHIUM, PURPLE, AI_CYAN)
"""Top-to-bottom gradient for the CLAI banner."""
DIFF_ADDITION = '#465258'
"""Aqua over Dark Purple, one quarter strength."""
DIFF_DELETION = '#682B36'
"""Calcium over Dark Purple, one quarter strength."""

_BASIC = {
    LITHIUM: 95,
    CALCIUM: 91,
    PURPLE: 35,
    AQUA: 96,
    SUGAR: 97,
    LIGHT_PURPLE: 97,
    GREY: 90,
    AI_CYAN: 96,
    AI_YELLOW: 93,
}


def truecolor() -> bool:
    """Whether the terminal advertises 24-bit colour."""
    return os.getenv('COLORTERM', '').lower() in ('truecolor', '24bit')


def sgr(color: str, *, bold: bool = False) -> str:
    """Raw escape for surfaces that bypass Rich, with a 16-colour fallback."""
    if color.startswith('bold '):
        color = color.removeprefix('bold ')
        bold = True
    prefix = '1;' if bold else ''
    if truecolor():
        red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
        return f'\x1b[{prefix}38;2;{red};{green};{blue}m'
    return f'\x1b[{prefix}{_BASIC[color]}m'
