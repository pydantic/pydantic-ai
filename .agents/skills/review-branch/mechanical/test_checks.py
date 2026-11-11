"""Self-checks for the mechanical review package.

Run: uv run pytest .agents/skills/review-branch/mechanical/test_checks.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

# The package imports itself as `mechanical`, so its parent directory goes on the path.
_PKG_PARENT = Path(__file__).resolve().parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from mechanical.checks import cassettes, denylist, docs_refs, missing_tests, patterns  # noqa: E402
from mechanical.models import ScanContext, Severity  # noqa: E402
from mechanical.runner import main  # noqa: E402

TESTDATA = Path(__file__).resolve().parent / 'testdata'


def _ctx(repo: Path) -> ScanContext:
    return ScanContext(repo=repo, mode_all=True)


def _place(repo: Path, fixture: str, relative: str) -> None:
    target = repo / relative
    target.parent.mkdir(parents=True)
    shutil.copy(TESTDATA / fixture, target)


def test_denylist_flags_removed_apis_in_slim_layout(tmp_path: Path) -> None:
    _place(tmp_path, 'bad_denylist.py', 'pydantic_ai_slim/pydantic_ai/bad.py')
    findings = denylist.run(_ctx(tmp_path))
    rules = {f.rule_id for f in findings}
    assert {'denylist.load_mcp_servers', 'denylist.BuiltinToolCallPart', 'denylist.ag_ui_import'} <= rules
    assert any(f.severity == Severity.ERROR for f in findings), 'expected error severity for production use'


def test_denylist_flags_names_removed_in_v2_outside_quotes(tmp_path: Path) -> None:
    _place(tmp_path, 'bad_denylist.py', 'pydantic_ai_slim/pydantic_ai/bad.py')
    rules = {f.rule_id for f in denylist.run(_ctx(tmp_path))}
    removed = {
        'MCPServerStdio',
        'MCPServerSSE',
        'MCPServerStreamableHTTP',
        'cached_async_http_client',
        'OpenAIModel',
        'OpenAIModelSettings',
        'DeferredToolCalls',
        'Usage',
    }
    assert {f'denylist.{name}' for name in removed} <= rules
    # The test data names `MCPServerHTTP` only as a quoted `_REMOVED_IN_V2` key.
    assert 'denylist.MCPServerHTTP' not in rules


def test_denylist_flags_harness_code_use_as_error(tmp_path: Path) -> None:
    _place(tmp_path, 'bad_denylist.py', 'src/pydantic_ai_harness/pydantic_ai_harness/cap/_capability.py')
    assert any(f.severity == Severity.ERROR for f in denylist.run(_ctx(tmp_path)))


def test_patterns_scans_clai2(tmp_path: Path) -> None:
    _place(tmp_path, 'bad_patterns.py', 'src/pydantic_clai2/pydantic_clai2/bad.py')
    assert 'patterns.bare_type_ignore' in {f.rule_id for f in patterns.run(_ctx(tmp_path))}


def test_missing_tests_reads_harness_reexports(tmp_path: Path) -> None:
    package = tmp_path / 'src/pydantic_ai_harness/pydantic_ai_harness/cap'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text("from ._capability import Cap\n\n__all__ = ['Cap']\n")
    (package / '_capability.py').write_text('class Cap:\n    pass\n\n\ndef helper():\n    pass\n')
    (tmp_path / 'tests').mkdir()
    rel = 'src/pydantic_ai_harness/pydantic_ai_harness/cap/_capability.py'
    ctx = ScanContext(
        repo=tmp_path,
        mode_all=False,
        diff_base='base',
        file_filter=frozenset({rel}),
        added_lines={rel: {1, 5}},
    )
    assert {f.message.split('`')[1] for f in missing_tests.run(ctx)} == {'Cap'}


def test_patterns_flags_high_signal_patterns(tmp_path: Path) -> None:
    _place(tmp_path, 'bad_patterns.py', 'tests/test_bad.py')
    rules = {f.rule_id for f in patterns.run(_ctx(tmp_path))}
    assert {
        'patterns.bare_type_ignore',
        'patterns.line_number_ref',
        'patterns.empty_snapshot',
        'patterns.importorskip_in_function',
    } <= rules


def test_cassettes_flag_secrets(tmp_path: Path) -> None:
    _place(tmp_path, 'cassette_secret.yaml', 'tests/cassettes/leak.yaml')
    assert any(f.severity == Severity.ERROR for f in cassettes.run(_ctx(tmp_path)))


def test_docs_refs_flag_removed_api_mentions(tmp_path: Path) -> None:
    _place(tmp_path, 'docs_removed.md', 'docs/mcp.md')
    rules = {f.rule_id for f in docs_refs.run(_ctx(tmp_path))}
    assert {'docs_refs.load_mcp_servers', 'docs_refs.to_ag_ui'} <= rules


def test_runner_help() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(['--help'])
    assert exc_info.value.code == 0
