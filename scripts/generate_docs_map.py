#!/usr/bin/env python
"""Generate the committed Pydantic AI docs atlas from `docs/navigation.yml`."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import unified_diff
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

import yaml
from pydantic import RootModel, ValidationError

if TYPE_CHECKING:
    from tiktoken.core import Encoding

TOKEN_ENCODING = 'cl100k_base'
PAGE_SLURP_TOKENS = 8000
REGION_SUBAGENT_TOKENS = 40_000
HIGHWAY_LIMIT = 15
API_REGION = 'API Reference'
LOCAL_SOURCE = 'pydantic-ai'
NAV_RELATIVE_PATH = Path('docs/navigation.yml')
ATLAS_RELATIVE_PATH = Path('agent_docs/docs-atlas.md')
HTML_RELATIVE_PATH = Path('docs/map.html')
GRAPH_PLACEHOLDER = '__GRAPH_JSON__'
_SKIP_SCHEMES = ('http://', 'https://', 'mailto:', 'ftp://', 'javascript:')
_VIEWER_TEMPLATE = Path(__file__).resolve().parent / 'docs_map' / 'viewer.html'

_INLINE_LINK = re.compile(r'(?<!!)\[(?:[^\]]+)\]\(([^)\s]+)(?:\s+(?:"[^"]*"|\'[^\']*\'))?\)')
_REF_DEF = re.compile(r'^ {0,3}\[([^\]]+)\]:\s+<?([^\s>]+)>?', re.MULTILINE)
_REF_USE = re.compile(r'(?<!!)\[([^\]]+)\]\[([^\]]*)\]')


@dataclass(frozen=True)
class _RawPage:
    title: str
    path: str
    slug: str
    aliases: tuple[str, ...]
    section: str
    top: str


@dataclass(frozen=True)
class _Node:
    title: str
    path: str
    slug: str
    section: str
    top: str
    tokens: int
    inbound: int
    outbound: int
    outbound_weighted: int


@dataclass(frozen=True)
class _Edge:
    src: str
    dst: str
    n: int


@dataclass(frozen=True)
class _DocsMap:
    nodes: tuple[_Node, ...]
    edges: tuple[_Edge, ...]
    section_edges: tuple[_Edge, ...]
    region_order: tuple[str, ...]


class _ObjectMap(RootModel[dict[str, object]]):
    pass


class _ObjectList(RootModel[list[object]]):
    pass


def main(argv: list[str] | None = None) -> int:
    """Regenerate the atlas, check it, and/or write the HTML viewer."""
    parser = argparse.ArgumentParser(description='Generate the committed Pydantic AI docs atlas.')
    parser.add_argument(
        '--check',
        action='store_true',
        help='exit 0 if the atlas on disk matches; otherwise print a short diff and exit nonzero',
    )
    parser.add_argument(
        '--html',
        type=Path,
        metavar='PATH',
        help='write an HTML viewer with the graph inlined (D3 is loaded from a CDN). Defaults to docs/map.html when regenerating',
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    docs_map = build_docs_map(root)
    markdown = render_atlas(docs_map)
    atlas_path = root / ATLAS_RELATIVE_PATH

    html = _render_html(docs_map)
    default_html_path = root / HTML_RELATIVE_PATH

    if args.check:
        status = _check_file(atlas_path, markdown, 'Docs atlas')
        if status == 0:
            status = _check_file(default_html_path, html, 'Docs map HTML')
    else:
        atlas_path.write_text(markdown, encoding='utf-8')
        default_html_path.write_text(html, encoding='utf-8')
        status = 0

    if args.html is not None:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(html, encoding='utf-8')

    return status


def build_docs_map(root: Path) -> _DocsMap:
    """Parse navigation, count tokens, and build the cross-link graph."""
    raw_pages = _collect_pages(root / NAV_RELATIVE_PATH)
    by_path = {page.path: page for page in raw_pages}
    if len(by_path) != len(raw_pages):
        raise SystemExit('docs/navigation.yml lists the same path more than once among local pages')
    by_slug = _slug_index(raw_pages)
    docs_dir = root / 'docs'

    tokens: dict[str, int] = {}
    edge_counts: Counter[tuple[str, str]] = Counter()
    for page in raw_pages:
        file_path = docs_dir / page.path
        if not file_path.is_file():
            raise SystemExit(f'docs page listed in navigation.yml is missing: {page.path}')
        text = file_path.read_text(encoding='utf-8')
        tokens[page.path] = _token_count(text)
        for href in _extract_hrefs(text):
            target = _resolve_href(page.path, href, by_path, by_slug)
            if target is None or target == page.path:
                continue
            edge_counts[(page.path, target)] += 1

    inbound: dict[str, int] = defaultdict(int)
    outbound: dict[str, int] = defaultdict(int)
    outbound_weighted: dict[str, int] = defaultdict(int)
    for (src, dst), n in edge_counts.items():
        inbound[dst] += n
        outbound[src] += 1
        outbound_weighted[src] += n

    nodes = tuple(
        _Node(
            title=page.title,
            path=page.path,
            slug=page.slug,
            section=page.section,
            top=page.top,
            tokens=tokens[page.path],
            inbound=inbound[page.path],
            outbound=outbound[page.path],
            outbound_weighted=outbound_weighted[page.path],
        )
        for page in raw_pages
    )
    edges = tuple(_Edge(src=src, dst=dst, n=n) for (src, dst), n in sorted(edge_counts.items()))
    section_counts: Counter[tuple[str, str]] = Counter()
    top_by_path = {page.path: page.top for page in raw_pages}
    for (src, dst), n in edge_counts.items():
        src_top = top_by_path[src]
        dst_top = top_by_path[dst]
        if src_top != dst_top:
            section_counts[(src_top, dst_top)] += n
    section_edges = tuple(_Edge(src=src, dst=dst, n=n) for (src, dst), n in sorted(section_counts.items()))
    region_order = tuple(dict.fromkeys(page.top for page in raw_pages))
    return _DocsMap(nodes=nodes, edges=edges, section_edges=section_edges, region_order=region_order)


def render_atlas(docs_map: _DocsMap) -> str:
    """Render the compact agent-facing markdown atlas."""
    nodes = docs_map.nodes
    total_pages = len(nodes)
    total_tokens = sum(node.tokens for node in nodes)
    api_nodes = [node for node in nodes if node.top == API_REGION]
    listed_regions = [top for top in docs_map.region_order if top != API_REGION]
    by_region: dict[str, list[_Node]] = defaultdict(list)
    for node in nodes:
        by_region[node.top].append(node)

    lines = [
        '# Docs atlas (generated — do not hand-edit)',
        '',
        'Generated by `scripts/generate_docs_map.py`. Regenerate with `make docs-map`. Check with `make docs-map-check`.',
        '',
        f'Source: `{NAV_RELATIVE_PATH}` + markdown links. Tokens: `{TOKEN_ENCODING}`.',
        (
            f'Published local pages: {total_pages} ({_fmt(total_tokens)} tok). '
            f'API reference omitted below ({_fmt(sum(n.tokens for n in api_nodes))} tok across '
            f'{len(api_nodes)} stub pages).'
        ),
        '',
        '## Dispatch rules',
        '',
        f'- Region ≥ {_fmt(REGION_SUBAGENT_TOKENS)} tok → spawn a subagent; do not load the region into this session.',
        f'- Page ≥ {_fmt(PAGE_SLURP_TOKENS)} tok → grep / read by section. Do not slurp the file.',
        '- Start at the hub page of the region (overview, else the first sidebar page).',
        '- API reference: open the one symbol page, never the section.',
        '',
        '## Cross-region highways',
        '',
    ]

    highways = [edge for edge in docs_map.section_edges if edge.src != API_REGION and edge.dst != API_REGION]
    highways.sort(key=lambda edge: (-edge.n, edge.src, edge.dst))
    for edge in highways[:HIGHWAY_LIMIT]:
        lines.append(f'- {edge.src} → {edge.dst} ×{edge.n}')

    lines.extend(['', '## Regions', ''])
    for top in listed_regions:
        region_nodes = by_region[top]
        region_tokens = sum(node.tokens for node in region_nodes)
        flag = 'SUBAGENT' if region_tokens >= REGION_SUBAGENT_TOKENS else 'inline'
        hub = _region_hub(region_nodes)
        lines.append(f'### {top} — {_fmt(region_tokens)} tok, {len(region_nodes)} pages — {flag}')
        lines.append('')
        lines.append(f'Hub: `{hub.path}` ({_fmt(hub.tokens)} tok, in {hub.inbound}, out {hub.outbound})')
        lines.append('')
        for node in region_nodes:
            mark = ' **do not slurp**' if node.tokens >= PAGE_SLURP_TOKENS else ''
            lines.append(f'- `{node.path}` — {_fmt(node.tokens)} tok, in {node.inbound}, out {node.outbound}{mark}')
        lines.append('')

    return '\n'.join(lines).rstrip() + '\n'


def _collect_pages(path: Path) -> list[_RawPage]:
    if not path.is_file():
        raise SystemExit(f'navigation file is missing: {path}')
    loaded: object = yaml.safe_load(path.read_text(encoding='utf-8'))
    data = _as_dict(loaded, str(path))
    pages: list[_RawPage] = []
    for item in _as_list(data.get('navigation'), 'navigation'):
        pages.extend(_walk_nav(item, top='', section_path=''))
    return pages


def _walk_nav(item: object, top: str, section_path: str) -> list[_RawPage]:
    data = _as_dict(item, 'navigation entry')
    if 'page' in data:
        if data.get('source') is not None:
            return []
        title = _as_str(data.get('page'), 'page')
        path = _as_str(data.get('path'), 'path')
        slug_value = data.get('slug')
        slug = _as_str(slug_value, 'slug') if slug_value is not None else ''
        aliases = _as_str_list(data.get('aliases'), 'aliases')
        return [
            _RawPage(
                title=title,
                path=path,
                slug=slug,
                aliases=tuple(aliases),
                section=section_path or top,
                top=top,
            )
        ]
    section = _as_str(data.get('section'), 'section')
    new_top = top or section
    new_path = section if not section_path else f'{section_path} / {section}'
    pages: list[_RawPage] = []
    for child in _as_list(data.get('contents'), f'{section} contents'):
        pages.extend(_walk_nav(child, new_top, new_path))
    return pages


def _as_dict(value: object, where: str) -> dict[str, object]:
    try:
        return _ObjectMap.model_validate(value).root
    except ValidationError as e:
        raise SystemExit(f'{where}: expected a mapping, got {type(value).__name__}') from e


def _as_list(value: object, where: str) -> list[object]:
    if value is None:
        return []
    try:
        return _ObjectList.model_validate(value).root
    except ValidationError as e:
        raise SystemExit(f'{where}: expected a list, got {type(value).__name__}') from e


def _as_str(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise SystemExit(f'{where}: expected a string {where}, got {type(value).__name__}')
    return value


def _as_str_list(value: object, where: str) -> list[str]:
    if value is None:
        return []
    items = _as_list(value, where)
    return [_as_str(item, where) for item in items]


def _slug_index(pages: list[_RawPage]) -> dict[str, str]:
    index: dict[str, str] = {}
    for page in pages:
        if page.slug:
            index[page.slug] = page.path
        for alias in page.aliases:
            index[alias] = page.path
    return index


def _extract_hrefs(markdown: str) -> list[str]:
    hrefs = [match.group(1) for match in _INLINE_LINK.finditer(markdown)]
    defs = {_normalize_ref(match.group(1)): match.group(2) for match in _REF_DEF.finditer(markdown)}
    for match in _REF_USE.finditer(markdown):
        key = _normalize_ref(match.group(2) or match.group(1))
        target = defs.get(key)
        if target is not None:
            hrefs.append(target)
    return hrefs


def _normalize_ref(value: str) -> str:
    return ' '.join(value.lower().split())


def _resolve_href(
    source_path: str,
    href: str,
    by_path: dict[str, _RawPage],
    by_slug: dict[str, str],
) -> str | None:
    target = href.strip().strip('<>')
    if not target:
        return None
    lowered = target.lower()
    if lowered.startswith(_SKIP_SCHEMES):
        return None
    path_part = unquote(target.split('#', 1)[0]).strip()
    if not path_part:
        return None
    relative = _posix_join(source_path, path_part)
    if relative is None:
        return None
    if relative in by_path:
        return relative
    slug_key = relative[:-3] if relative.endswith('.md') else relative
    return by_slug.get(slug_key) or by_slug.get(relative)


def _posix_join(source_path: str, href_path: str) -> str | None:
    if href_path.startswith('/'):
        parts = [part for part in href_path.split('/') if part]
        out: list[str] = []
    else:
        parent = source_path.rsplit('/', 1)[0] if '/' in source_path else ''
        out = parent.split('/') if parent else []
        parts = href_path.split('/')
    for part in parts:
        if part in ('', '.'):
            continue
        if part == '..':
            if not out:
                return None
            out.pop()
            continue
        out.append(part)
    return '/'.join(out)


def _token_count(text: str) -> int:
    return len(_encoding().encode_ordinary(text))


@cache
def _encoding() -> Encoding:
    try:
        import tiktoken
    except ImportError as e:
        raise SystemExit(
            'tiktoken is required to generate the docs atlas (`cl100k_base`). '
            'Install the openai extra (`uv sync --extra openai`). '
            'Refusing to fall back to a character heuristic because committed output must be deterministic.'
        ) from e
    return tiktoken.get_encoding(TOKEN_ENCODING)


def _region_hub(nodes: list[_Node]) -> _Node:
    """Pick the sidebar entry point, not the most-linked page.

    Pages are already in `docs/navigation.yml` order. Prefer an overview/index
    page when one exists in the region; otherwise the first sidebar page.
    """
    for node in nodes:
        name = node.path.rsplit('/', 1)[-1]
        if name == 'index.md' or node.title.casefold() == 'overview':
            return node
    return nodes[0]


def _fmt(n: int) -> str:
    return f'{n:,}'


def _check_file(path: Path, expected: str, label: str) -> int:
    if not path.is_file():
        print(f'{path} is missing. Run `make docs-map` to generate it.', file=sys.stderr)
        return 1
    on_disk = path.read_text(encoding='utf-8')
    if on_disk == expected:
        return 0
    diff = unified_diff(
        on_disk.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile=str(path),
        tofile='generated',
        n=3,
    )
    preview = ''.join(list(diff)[:80])
    print(f'{label} is stale. Run `make docs-map` to regenerate.', file=sys.stderr)
    sys.stderr.write(preview)
    if not preview.endswith('\n'):
        print(file=sys.stderr)
    return 1


def _graph_payload(docs_map: _DocsMap) -> dict[str, object]:
    nodes_json = [
        {
            'title': node.title,
            'path': node.path,
            'slug': node.slug,
            'source': LOCAL_SOURCE,
            'section': node.section,
            'top': node.top,
            'tokens': node.tokens,
            'out_unique': node.outbound,
            'in': node.inbound,
            'out_w': node.outbound_weighted,
        }
        for node in docs_map.nodes
    ]
    edges_json = [{'src': edge.src, 'dst': edge.dst, 'n': edge.n} for edge in docs_map.edges]
    section_json = [{'src': edge.src, 'dst': edge.dst, 'n': edge.n} for edge in docs_map.section_edges]
    return {
        'generated_from': 'docs/navigation.yml + markdown links',
        'token_encoding': TOKEN_ENCODING,
        'page_slurp_tokens': PAGE_SLURP_TOKENS,
        'region_subagent_tokens': REGION_SUBAGENT_TOKENS,
        'totals': {
            'pages': len(docs_map.nodes),
            'tokens': sum(node.tokens for node in docs_map.nodes),
            'edges': len(docs_map.edges),
            'link_mentions': sum(edge.n for edge in docs_map.edges),
        },
        'nodes': nodes_json,
        'edges': edges_json,
        'section_edges': section_json,
    }


def _render_html(docs_map: _DocsMap) -> str:
    if not _VIEWER_TEMPLATE.is_file():
        raise SystemExit(f'viewer template is missing: {_VIEWER_TEMPLATE}')
    template = _VIEWER_TEMPLATE.read_text(encoding='utf-8')
    if GRAPH_PLACEHOLDER not in template:
        raise SystemExit(f'{_VIEWER_TEMPLATE} is missing {GRAPH_PLACEHOLDER}')
    payload = json.dumps(_graph_payload(docs_map), indent=2, sort_keys=True)
    payload = payload.replace('<', '\\u003c')
    return template.replace(GRAPH_PLACEHOLDER, payload)


if __name__ == '__main__':
    raise SystemExit(main())
