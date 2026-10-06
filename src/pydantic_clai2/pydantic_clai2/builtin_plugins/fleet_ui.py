"""How fleet control looks in the terminal: change notices, the `/catalog` picker, and policy blocks (hackathon)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.text import Text
from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu

from pydantic_clai2.builtin_plugins.fleet import Change, Provenance, Snapshot
from pydantic_clai2.ui.menus.menu_worker import menu_key
from pydantic_clai2.ui.menus.slash_search import KeyHandler, slash_search
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import markdown_style

NOUNS = {'skill': 'skill', 'mcp_server': 'MCP server', 'plugin': 'plugin', 'instruction': 'instruction'}
BLOCKED_PREFIX = 'Blocked by policy'


def _mark(action: str) -> str:
    return {'added': '+', 'updated': '~', 'removed': '-'}[action]


PLURALS = {'skill': 'skills', 'mcp_server': 'MCP servers', 'plugin': 'plugins', 'instruction': 'instructions'}


def notice_panel(changes: Sequence[Change], *, snapshot: Snapshot, source: str, link: str | None) -> RenderableType:
    """One compact panel per batch: names only, grouped, then one line with versions and where to look.

    `source` is who it is from: a pushed `display_name` ("Pydantic"), else the Logfire project ("logfire/clai2").
    `/catalog why NAME` has the details.
    """
    lines: list[RenderableType] = []
    for action in ('added', 'updated', 'removed'):
        groups: dict[str, list[str]] = {}
        for change in changes:
            if change.action != action:
                continue
            if change.kind == 'instructions':
                groups.setdefault('instructions', []).append('company instructions')
            elif change.tier == 'catalog':
                groups.setdefault('catalog', []).append(change.name)
            else:
                groups.setdefault(PLURALS.get(change.kind, change.kind), []).append(change.name)
        if not groups:
            continue
        line = Text(f'{_mark(action)} ', style=theme.color(theme.ACCENT))
        for index, (group, names) in enumerate(groups.items()):
            if index:
                line.append(' · ', style=theme.color(theme.MUTED))
            line.append(f'{group}: ', style=theme.color(theme.MUTED))
            line.append(', '.join(names), style='bold')
        lines.append(line)
    tiers = {change.tier for change in changes}
    unknown = '' in tiers  # A removal no longer says where it came from.
    versions = snapshot.versions(config='company' in tiers or unknown, catalog='catalog' in tiers or unknown)
    footer = Text(' · '.join(part for part in (versions, '/catalog to browse', link or '') if part))
    footer.stylize(theme.color(theme.MUTED))
    lines.append(footer)
    return Panel(
        Group(*lines),
        title=Text(f'◆ Updated from Logfire · {source}', style=theme.color(theme.ACCENT)),
        title_align='left',
        border_style=theme.color(theme.MUTED),
        expand=False,
        padding=(0, 1),
    )


def blocked_panel(message: str, *, link: str | None) -> RenderableType:
    """The user-facing side of a policy block; the model gets the plain message as the tool result."""
    # The user doesn't need the model's instructions ("Do not retry ..."), only what happened and why.
    body = Text(message.split(' Do not retry', 1)[0])
    body.append('\nEnforced by clai2 on this machine.', style=theme.color(theme.MUTED))
    if link:
        body.append(f' Learn more: {link}', style=theme.color(theme.MUTED))
    return Panel(
        body,
        title=Text('◆ Policy', style=theme.color(theme.WARNING)),
        title_align='left',
        border_style=theme.color(theme.WARNING),
        expand=False,
        padding=(0, 1),
    )


@dataclass(frozen=True, kw_only=True)
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
    declined: bool = False
    """The user declined the consent prompt for it, so it stays off until they turn it on again."""
    elsewhere: bool = False
    """Scoped (`applies_to`) to other teams or repos, so it does not apply here."""
    full_text: str = ''
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
    declined = ' · declined' if row.declined else ''
    elsewhere = ' · not for this repo/team' if row.elsewhere else ''
    return f'{state} {NOUNS.get(row.kind, row.kind):<10} {row.name}{lock}{new}{declined}{elsewhere}'


def row_preview(row: CatalogRow, *, link: str | None) -> str:
    """The picker's side panel: what it is, who uses it, why it's there, and what Enter does."""
    parts = [f'**{row.name}** ({NOUNS.get(row.kind, row.kind)}, {row.delivery})', '', row.description or '']
    if row.adoption:
        parts.append(f'Used by {row.adoption} teammates.')
    why = row.provenance.describe()
    if why:
        parts.append(f'Why: {why}.')
    if row.locked:
        parts.append('Required by your organization: it stays on.')
    else:
        parts.append(f'Enter turns it {"off" if row.on else "on"}.')
    if row.elsewhere:
        parts.append('Scoped to other teams or repos, so it does not apply here.')
    parts.extend(['', 'p shows everything it would add.'])
    if link:
        parts.extend(['', link])
    return '\n'.join(parts)


def catalog_menu(
    rows: Sequence[CatalogRow],
    *,
    snapshot: Snapshot,
    link: str | None,
    index: int = 0,
    hotkeys: Mapping[str, KeyHandler] | None = None,
) -> Menu:
    """The `/catalog` picker: organization items, then optional add-ons; Enter toggles an add-on."""
    versions = snapshot.versions()
    title = f'From your organization and optional add-ons{f" ({versions})" if versions else ""}'
    builder = (
        MenuBuilder(title)
        .style(markdown_style())
        .items([MenuItem(row_label(row), value=row) for row in rows])
        .initial_index(min(index, max(len(rows) - 1, 0)))
        .preview(lambda item: row_preview(item.value, link=link) if isinstance(item.value, CatalogRow) else '')
    )
    return slash_search(builder, footer='enter toggle · p preview · esc close', key_source=menu_key, hotkeys=hotkeys)


def preview_panel(row: CatalogRow) -> RenderableType:
    """Everything an item would add: the full skill or instruction text, or the server URL and env it sends."""
    return Panel(
        Text(row.full_text or '(nothing to show)'),
        title=Text(f'{NOUNS.get(row.kind, row.kind)} {row.name}', style=theme.color(theme.ACCENT)),
        title_align='left',
        border_style=theme.color(theme.MUTED),
        expand=False,
        padding=(0, 1),
    )


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
