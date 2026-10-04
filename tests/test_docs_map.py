"""Tests for the committed docs atlas generator.

This is a unit/integration test of `scripts/generate_docs_map.py`. It does not
use VCR: the source of truth is `docs/navigation.yml` plus markdown files already
in the tree. On a cold cache, `tiktoken` downloads `cl100k_base` once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_docs_map import DocsMap, build_docs_map, render_atlas, render_html

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def docs_map() -> DocsMap:
    return build_docs_map(ROOT)


@pytest.fixture(scope='module')
def atlas(docs_map: DocsMap) -> str:
    return render_atlas(docs_map)


def test_atlas_core_concepts_and_agent(atlas: str):
    assert '### Core Concepts' in atlas
    assert '`agent.md`' in atlas


def test_api_reference_omitted_from_region_listing(atlas: str):
    assert '### API Reference' not in atlas
    assert 'API reference omitted below' in atlas
    assert 'open the one symbol page, never the section' in atlas


def test_harness_pages_are_listed(atlas: str):
    for path in ('harness/coder.md', 'harness/researcher.md', 'harness/filesystem.md', 'harness/pydantic-ai-docs.md'):
        assert f'`{path}`' in atlas


def test_hubs_follow_sidebar_entry_points(atlas: str):
    """Hubs are overview/index pages, not the highest-inbound page."""
    assert 'Hub: `index.md`' in atlas
    assert 'Hub: `agent.md`' in atlas
    assert 'Hub: `models/overview.md`' in atlas
    assert 'Hub: `tools.md`' in atlas
    assert 'Hub: `evals.md`' in atlas
    assert 'Hub: `mcp/overview.md`' in atlas
    core = atlas.split('### Core Concepts', 1)[1].split('### ', 1)[0]
    assert 'Hub: `agent.md`' in core
    assert 'Hub: `capabilities/overview.md`' not in core
    overview = atlas.split('### Overview', 1)[1].split('### ', 1)[0]
    assert overview.index('`index.md`') < overview.index('`install.md`')
    tools = atlas.split('### Tools & Toolsets', 1)[1].split('### ', 1)[0]
    assert tools.index('`tools.md`') < tools.index('`tools-advanced.md`')


def test_html_viewer_inlines_graph(docs_map: DocsMap):
    html = render_html(docs_map)
    assert '__GRAPH_JSON__' not in html
    assert '"path": "agent.md"' in html
    assert 'Force graph' in html
    assert 'Fortress rooms' in html
    assert 'Agent atlas' in html
    assert "searchParams.set('variant'" in html or 'searchParams.set("variant"' in html
    assert 'https://pydantic.dev/docs/ai/' in html
    assert 'cdn.jsdelivr.net' not in html
    assert 'src="./d3.min.js"' not in html
    assert (ROOT / 'scripts' / 'docs_map' / 'd3.min.js').read_text(encoding='utf-8') in html
    assert "raw !== '" not in html
    assert 'raw.startsWith' in html
    assert 'id="legend-body"' in html
    assert '<details id="legend"' in html


def _write_docs(root: Path, navigation: str) -> None:
    (root / 'docs').mkdir()
    (root / 'docs' / 'navigation.yml').write_text(navigation, encoding='utf-8')
    (root / 'docs' / 'index.md').write_text('# Home\n', encoding='utf-8')


def test_link_entries_are_skipped(tmp_path: Path):
    _write_docs(
        tmp_path,
        """
navigation:
  - section: Overview
    contents:
      - page: Home
        path: index.md
      - link: GitHub
        href: https://github.com/pydantic/pydantic-ai
""",
    )
    assert [node.path for node in build_docs_map(tmp_path).nodes] == ['index.md']


def test_folder_entries_are_rejected(tmp_path: Path):
    _write_docs(
        tmp_path,
        """
navigation:
  - section: Overview
    contents:
      - folder: guides
""",
    )
    with pytest.raises(SystemExit, match='`folder` and `include` entries are not supported yet'):
        build_docs_map(tmp_path)
