from __future__ import annotations

import sys
from pathlib import Path

import pytest
from inline_snapshot import snapshot

sys.path.insert(0, str(Path(__file__).parent))

import check_glossary_spellings

_REPO = Path(__file__).parents[1]

_GLOSSARY = """\
# Glossary

## Spellings

| Write | Not |
|---|---|
| policy base | `policy-base` |
| trivial-S | `trivial S` |
"""


def _run(monkeypatch: pytest.MonkeyPatch, *args: str | Path) -> int:
    monkeypatch.setattr(sys, 'argv', ['check_glossary_spellings.py', *map(str, args)])
    return check_glossary_spellings.main()


@pytest.fixture
def glossary(tmp_path: Path) -> Path:
    path = tmp_path / 'GLOSSARY.md'
    path.write_text(_GLOSSARY)
    return path


def test_rejects_a_spelling_in_prose(
    tmp_path: Path, glossary: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    doc = tmp_path / 'AGENTS.md'
    doc.write_text(
        'Use the Policy-base checkout.\nAn obviously trivial S PR; a trivial Something.\nThe merge bar exempts a trivial\n  S PR from douwebot.\n'
    )
    assert _run(monkeypatch, '--glossary', glossary, doc) == 1
    assert capsys.readouterr().out.replace(str(tmp_path), '<tmp>') == snapshot(
        """\
<tmp>/AGENTS.md:1: write "policy base", not "Policy-base" (<tmp>/GLOSSARY.md → Spellings)
<tmp>/AGENTS.md:2: write "trivial-S", not "trivial S" (<tmp>/GLOSSARY.md → Spellings)
<tmp>/AGENTS.md:3: write "trivial-S", not "trivial S" (<tmp>/GLOSSARY.md → Spellings)
"""
    )


def test_skips_code_links_file_names_and_paths(tmp_path: Path, glossary: Path, monkeypatch: pytest.MonkeyPatch):
    doc = tmp_path / 'SKILL.md'
    doc.write_text(
        """\
Keep `policy-base` and `a span that
wraps policy-base`.
See [the gate](#policy-base), `claims.md`, policy-base.md and skills/policy-base.

```bash
cd policy-base
```
"""
    )
    assert _run(monkeypatch, '--glossary', glossary, doc) == 0


def test_fails_without_a_spellings_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    empty = tmp_path / 'GLOSSARY.md'
    empty.write_text('# Glossary\n')
    assert _run(monkeypatch, '--glossary', empty) == 1
    assert capsys.readouterr().err.replace(str(tmp_path), '<tmp>') == snapshot(
        '<tmp>/GLOSSARY.md: no "## Spellings" table rows found\n'
    )


def test_repo_glossary_passes_its_own_check(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(_REPO)
    assert _run(monkeypatch, 'agent_docs/GLOSSARY.md') == 0
