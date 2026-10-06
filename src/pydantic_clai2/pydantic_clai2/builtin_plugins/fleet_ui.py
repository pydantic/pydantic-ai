"""How fleet control looks in the terminal: change notices, the `/catalog` picker, and policy blocks (hackathon)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text
from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu

from pydantic_clai2.builtin_plugins.fleet import Change, Provenance
from pydantic_clai2.ui.menus.menu_worker import menu_key
from pydantic_clai2.ui.menus.slash_search import slash_search
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import markdown_style

NOUNS = {'skill': 'skill', 'mcp_server': 'MCP server', 'plugin': 'plugin', 'instruction': 'instruction'}
BLOCKED_PREFIX = 'Blocked by your organization'


def _mark(action: str) -> str:
    return {'added': '+', 'updated': '~', 'removed': '-'}[action]


def notice_panel(changes: Sequence[Change], *, version: str | None, link: str | None) -> RenderableType:
    """One compact panel per batch: what changed, why, and where to look."""
    lines: list[RenderableType] = []
    for change in changes:
        line = Text()
        line.append(f'{_mark(change.action)} ', style=theme.color(theme.ACCENT))
        if change.kind == 'instructions':
            line.append('company instructions')
        else:
            line.append(f'{NOUNS.get(change.kind, change.kind)} ', style=theme.color(theme.MUTED))
            line.append(change.name, style='bold')
        if change.description and change.action != 'removed':
            line.append(f': {change.description}')
        lines.append(line)
        why = change.provenance.describe()
        if why:
            lines.append(Text(f'    {why}', style=theme.color(theme.MUTED)))
    footer = Text('/catalog to browse', style=theme.color(theme.MUTED))
    if link:
        footer.append(f'  ·  {link}', style=theme.color(theme.MUTED))
    lines.append(footer)
    title = f'◆ From your organization{f" (config v{version})" if version else ""}'
    return Panel(
        Group(*lines),
        title=Text(title, style=theme.color(theme.ACCENT)),
        title_align='left',
        border_style=theme.color(theme.MUTED),
        expand=False,
        padding=(0, 1),
    )


def blocked_panel(message: str, *, link: str | None) -> RenderableType:
    """The user-facing side of a policy block; the model gets the plain message as the tool result."""
    body = Text(message.split('\n', 1)[0])
    if link:
        body.append(f'\nLearn more: {link}', style=theme.color(theme.MUTED))
    return Panel(
        body,
        title=Text('◆ Policy', style=theme.color(theme.WARNING)),
        title_align='left',
        border_style=theme.color(theme.WARNING),
        expand=False,
        padding=(0, 1),
    )


@dataclass(frozen=True)
class CatalogRow:
    """One line of the `/catalog` picker."""

    key: str
    kind: str
    name: str
    description: str
    delivery: str
    """`organization`, `default on` or `optional`."""
    on: bool
    locked: bool
    new: bool
    adoption: int | None
    provenance: Provenance

    @property
    def toggleable(self) -> bool:
        return self.delivery != 'organization' and not (self.locked and self.on)


def row_label(row: CatalogRow) -> str:
    """`● skill pr-shepherd 🔒 • new`: state, kind, name, lock and the new dot."""
    state = '●' if row.on else '○'
    lock = ' 🔒' if row.locked else ''
    new = ' • new' if row.new else ''
    return f'{state} {NOUNS.get(row.kind, row.kind):<10} {row.name}{lock}{new}'


def row_preview(row: CatalogRow, *, link: str | None) -> str:
    """The picker's side panel: what it is, who uses it, why it's there, and what Enter does."""
    parts = [f'**{row.name}** ({NOUNS.get(row.kind, row.kind)}, {row.delivery})', '', row.description or '']
    if row.adoption:
        parts.append(f'Used by {row.adoption} teammates.')
    why = row.provenance.describe()
    if why:
        parts.append(f'Why: {why}.')
    if row.locked:
        parts.append('Locked by your organization: it stays on.')
    elif row.delivery == 'organization':
        parts.append('Pushed to everyone by your organization.')
    else:
        parts.append(f'Enter turns it {"off" if row.on else "on"}.')
    if link:
        parts.extend(['', link])
    return '\n'.join(parts)


def catalog_menu(rows: Sequence[CatalogRow], *, version: str | None, link: str | None, index: int = 0) -> Menu:
    """The `/catalog` picker: organization items, then optional add-ons; Enter toggles an add-on."""
    title = f'From your organization{f" (v{version})" if version else ""} and optional add-ons'
    builder = (
        MenuBuilder(title)
        .style(markdown_style())
        .items([MenuItem(row_label(row), value=row) for row in rows])
        .initial_index(min(index, max(len(rows) - 1, 0)))
        .preview(lambda item: row_preview(item.value, link=link) if isinstance(item.value, CatalogRow) else '')
    )
    return slash_search(builder, footer='enter toggle · / search · esc close', key_source=menu_key)


def why_text(row: CatalogRow, *, link: str | None) -> str:
    """`/catalog why NAME`: where an item came from."""
    lines = [f'{row.name} ({NOUNS.get(row.kind, row.kind)}, {row.delivery}): {row.description}']
    why = row.provenance.describe()
    lines.append(f'Why: {why}.' if why else 'Your organization added it; no reason was recorded.')
    if row.provenance.source:
        lines.append(f'Source: {row.provenance.source}.')
    if row.adoption:
        lines.append(f'Used by {row.adoption} teammates.')
    if link:
        lines.append(f'See it in Logfire: {link}')
    return '\n'.join(lines)
