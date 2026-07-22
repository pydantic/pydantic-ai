"""Self-checks for mechanical review package (stdlib unittest)."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Allow `python3 -m unittest test_checks` from this directory
_PKG_PARENT = Path(__file__).resolve().parent.parent
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from mechanical_review.checks import denylist, patterns, cassettes, docs_refs  # noqa: E402
from mechanical_review.models import ScanContext, Severity  # noqa: E402

TESTDATA = Path(__file__).resolve().parent / 'testdata'


def _ctx(repo: Path) -> ScanContext:
    return ScanContext(repo=repo, mode_all=True)


class TestDenylist(unittest.TestCase):
    def test_flags_removed_apis_in_slim_layout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            target = repo / 'pydantic_ai_slim' / 'pydantic_ai' / 'bad.py'
            target.parent.mkdir(parents=True)
            shutil.copy(TESTDATA / 'bad_denylist.py', target)
            findings = denylist.run(_ctx(repo))
            rules = {f.rule_id for f in findings}
            self.assertIn('denylist.load_mcp_servers', rules)
            self.assertIn('denylist.BuiltinToolCallPart', rules)
            self.assertIn('denylist.ag_ui_import', rules)
            errors = [f for f in findings if f.severity == Severity.ERROR]
            self.assertTrue(errors, 'expected error severity for production use')


class TestPatterns(unittest.TestCase):
    def test_flags_high_signal_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            target = repo / 'tests' / 'test_bad.py'
            target.parent.mkdir(parents=True)
            shutil.copy(TESTDATA / 'bad_patterns.py', target)
            findings = patterns.run(_ctx(repo))
            rules = {f.rule_id for f in findings}
            self.assertIn('patterns.bare_type_ignore', rules)
            self.assertIn('patterns.line_number_ref', rules)
            self.assertIn('patterns.empty_snapshot', rules)
            self.assertIn('patterns.importorskip_in_function', rules)


class TestCassettes(unittest.TestCase):
    def test_flags_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            target = repo / 'tests' / 'cassettes' / 'leak.yaml'
            target.parent.mkdir(parents=True)
            shutil.copy(TESTDATA / 'cassette_secret.yaml', target)
            findings = cassettes.run(_ctx(repo))
            self.assertTrue(findings)
            self.assertTrue(any(f.severity == Severity.ERROR for f in findings))


class TestDocsRefs(unittest.TestCase):
    def test_flags_docs_mentions(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            target = repo / 'docs' / 'mcp.md'
            target.parent.mkdir(parents=True)
            shutil.copy(TESTDATA / 'docs_removed.md', target)
            findings = docs_refs.run(_ctx(repo))
            rules = {f.rule_id for f in findings}
            self.assertIn('docs_refs.load_mcp_servers', rules)
            self.assertIn('docs_refs.to_ag_ui', rules)


class TestRunnerSmoke(unittest.TestCase):
    def test_cli_help(self) -> None:
        from mechanical_review.runner import main

        with self.assertRaises(SystemExit) as cm:
            main(['--help'])
        self.assertEqual(cm.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
