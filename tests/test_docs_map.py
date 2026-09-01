"""Tests for the committed docs atlas generator.

This is a unit/integration test of `scripts/generate_docs_map.py`. It does not
hit the network and does not use VCR: the source of truth is `docs/navigation.yml`
plus markdown files already in the tree.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'generate_docs_map.py'
ATLAS = ROOT / 'agent_docs' / 'docs-atlas.md'


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding='utf-8',
        check=False,
    )


def test_check_passes_against_committed_atlas():
    """`--check` must match the committed atlas, or the generator drifted."""
    result = _run('--check')
    assert result.returncode == 0, result.stderr + result.stdout


def test_atlas_core_concepts_and_agent():
    text = ATLAS.read_text(encoding='utf-8')
    assert '### Core Concepts' in text
    assert '`agent.md`' in text


def test_api_reference_omitted_from_region_listing():
    text = ATLAS.read_text(encoding='utf-8')
    assert '### API Reference' not in text
    assert 'API reference omitted below' in text
    assert 'open the one symbol page, never the section' in text


def test_harness_sourced_pages_are_absent():
    text = ATLAS.read_text(encoding='utf-8')
    for path in ('coder.md', 'researcher.md', 'filesystem.md', 'pydantic-ai-docs.md'):
        assert f'`{path}`' not in text


def test_html_viewer_is_self_contained(tmp_path: Path):
    html_path = tmp_path / 'index.html'
    result = _run('--check', '--html', str(html_path))
    assert result.returncode == 0, result.stderr + result.stdout
    html = html_path.read_text(encoding='utf-8')
    assert '__GRAPH_JSON__' not in html
    assert '"path": "agent.md"' in html
    assert 'Force graph' in html
    assert 'Fortress rooms' in html
    assert 'Agent atlas' in html
    assert "searchParams.set('variant'" in html or 'searchParams.set("variant"' in html
    assert html_path.with_name('graph.json').is_file()
