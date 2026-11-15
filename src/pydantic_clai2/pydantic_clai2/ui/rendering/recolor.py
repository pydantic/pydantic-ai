"""Repaint styled output in the theme now selected, translating the colours of the theme that painted it."""

from functools import cache

from rich.color import Color
from rich.style import Style
from rich.text import Span, Text

from pydantic_clai2.ui.rendering import theme


@cache
def _translation(source: str, target: str) -> dict[str, Color]:
    """Each hex colour `source` paints a role with, to the colour `target` paints that role with."""
    painted = dict(theme.roles(target))
    table: dict[str, Color] = {}
    for role, colour in theme.roles(source):
        # Named and numbered colours are the terminal's own, which a palette change already repaints.
        if colour.startswith('#'):
            table.setdefault(colour.lower(), Color.parse(painted[role]))
    return table


def recolor(text: Text, *, source: str, target: str) -> Text:
    """`text`, painted in theme `source`, as theme `target` paints it; colours no role uses stay."""
    table = _translation(source, target)

    def swap(colour: Color | None) -> Color | None:
        if colour is None or colour.triplet is None:
            return colour
        return table.get(colour.triplet.hex, colour)

    def restyle(span: Span) -> Span:
        style = span.style if isinstance(span.style, Style) else Style.parse(span.style)
        return Span(span.start, span.end, style + Style.from_color(swap(style.color), swap(style.bgcolor)))

    themed = text.copy()
    themed.spans = [restyle(span) for span in themed.spans]
    return themed
