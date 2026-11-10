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

from mechanical.checks import cassettes, denylist, docs_refs, missing_tests, patterns  # noqa: E402
from mechanical.models import ScanContext, Severity  # noqa: E402

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

    def test_flags_names_removed_in_v2_outside_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            target = repo / 'pydantic_ai_slim' / 'pydantic_ai' / 'bad.py'
            target.parent.mkdir(parents=True)
            shutil.copy(TESTDATA / 'bad_denylist.py', target)
            rules = {f.rule_id for f in denylist.run(_ctx(repo))}
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
            self.assertLessEqual({f'denylist.{name}' for name in removed}, rules)
            # The test data names `MCPServerHTTP` only as a quoted `_REMOVED_IN_V2` key.
            self.assertNotIn('denylist.MCPServerHTTP', rules)


class TestPackageRoots(unittest.TestCase):
    def test_denylist_flags_harness_code_use_as_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            target = repo / 'src' / 'pydantic_ai_harness' / 'pydantic_ai_harness' / 'cap' / '_capability.py'
            target.parent.mkdir(parents=True)
            shutil.copy(TESTDATA / 'bad_denylist.py', target)
            findings = denylist.run(_ctx(repo))
            self.assertTrue(any(f.severity == Severity.ERROR for f in findings))

    def test_patterns_scans_clai2(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            target = repo / 'src' / 'pydantic_clai2' / 'pydantic_clai2' / 'bad.py'
            target.parent.mkdir(parents=True)
            target.write_text('x = 1  # type: ignore\n')
            rules = {f.rule_id for f in patterns.run(_ctx(repo))}
            self.assertIn('patterns.bare_type_ignore', rules)

    def test_missing_tests_reads_harness_reexports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            package = repo / 'src' / 'pydantic_ai_harness' / 'pydantic_ai_harness' / 'cap'
            package.mkdir(parents=True)
            (package / '__init__.py').write_text("from ._capability import Cap\n\n__all__ = ['Cap']\n")
            (package / '_capability.py').write_text('class Cap:\n    pass\n\n\ndef helper():\n    pass\n')
            (repo / 'tests').mkdir()
            rel = 'src/pydantic_ai_harness/pydantic_ai_harness/cap/_capability.py'
            ctx = ScanContext(
                repo=repo,
                mode_all=False,
                diff_base='base',
                file_filter=frozenset({rel}),
                added_lines={rel: {1, 5}},
            )
            names = {f.message.split('`')[1] for f in missing_tests.run(ctx)}
            self.assertEqual(names, {'Cap'})


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
        from mechanical.runner import main

        with self.assertRaises(SystemExit) as cm:
            main(['--help'])
        self.assertEqual(cm.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
