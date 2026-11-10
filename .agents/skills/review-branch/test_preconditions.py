"""Tests for preconditions.py against temporary Git histories (stdlib unittest).

Run: uv run pytest .agents/skills/review-branch/test_preconditions.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import preconditions  # noqa: E402
import prepare  # noqa: E402

HARNESS = 'src/pydantic_ai_harness/pydantic_ai_harness/'
SLIM = 'pydantic_ai_slim/pydantic_ai/'
REPO = HERE.parents[2]
CORE_FILE = SLIM + '_run_context.py'
CORE_SOURCE = (
    'from dataclasses import dataclass\n'
    '\n'
    '\n'
    '@dataclass\n'
    'class RunContext:\n'
    '    usage: int = 0\n'
    '\n'
    '    def step(self) -> int:\n'
    '        """Advance the run."""\n'
    '        return self.usage + 1\n'
    '\n'
    '\n'
    'def describe(ctx: RunContext) -> str:\n'
    "    return 'run'\n"
)
# `describe` is `annotation_only`, so no change to it gates review.
CONTRACT_MANIFEST: dict[str, dict[str, dict[str, list[str]]]] = {
    CORE_FILE: {
        'RunContext.step': {
            'engines': ['temporal'],
            'kinds': ['override'],
            'sites': [SLIM + 'durable_exec/temporal/_run_context.py::TemporalRunContext.step'],
        },
        'RunContext.usage': {
            'engines': ['dbos', 'temporal'],
            'kinds': ['attribute_read'],
            'sites': [SLIM + 'durable_exec/dbos/_model.py::DBOSModel.request'],
        },
        'describe': {
            'engines': ['prefect'],
            'kinds': ['annotation_only'],
            'sites': [SLIM + 'durable_exec/prefect/_model.py::PrefectModel.request'],
        },
    },
}


class PreconditionTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # The output sits outside the repo, so a commit never picks up a run's artifacts.
        self.repo = Path(directory.name) / 'repo'
        self.repo.mkdir()
        self.output = Path(directory.name) / 'review'
        for args in (
            ('init', '-q', '--initial-branch=main'),
            ('config', 'user.name', 'Review Test'),
            ('config', 'user.email', 'review@example.invalid'),
            ('config', 'commit.gpgSign', 'false'),
            ('config', 'core.hooksPath', '/dev/null'),
        ):
            self.git(*args)
        self.write(SLIM + 'models/openai.py', 'def request():\n    return 1\n')
        self.write(SLIM + 'agent/__init__.py', 'class Agent:\n    pass\n')
        self.write(HARNESS + 'memory/_capability.py', 'class Memory:\n    pass\n')
        self.write(HARNESS + 'memory/__init__.py', "from ._capability import Memory\n\n__all__ = ['Memory']\n")
        for agent in preconditions.REVIEWER_AGENTS:
            self.write(f'.agents/agents/{agent}.md', f'---\nname: {agent}\n---\n')
        self.commit('base')
        self.git('checkout', '-qb', 'feature')

    def git(self, *args: str) -> str:
        return subprocess.run(
            ['git', '-C', str(self.repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def write(self, name: str, text: str) -> None:
        target = self.repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, message: str) -> None:
        self.git('add', '--all')
        self.git('commit', '-qm', message)

    def run_preconditions(self, pr: dict[str, object] | None = None) -> dict[str, object]:
        prepare.prepare(self.repo, 'main', 'HEAD', self.output)
        pr_json = self.output / 'pr.json'
        pr_json.write_text(json.dumps(pr if pr is not None else {'body': '', 'isDraft': False}))
        subprocess.run(
            [
                sys.executable,
                str(HERE / 'preconditions.py'),
                '--repo',
                str(self.repo),
                '--output',
                str(self.output),
                '--pr-json',
                str(pr_json),
            ],
            check=True,
            capture_output=True,
        )
        return json.loads((self.output / 'preconditions.json').read_text())

    @staticmethod
    def dispatched(report: dict[str, object]) -> set[str]:
        reviewers = report['reviewers']
        assert isinstance(reviewers, dict)
        return {name for name, entry in reviewers.items() if entry['dispatch']}

    def test_harness_diff_opens_package_lanes(self) -> None:
        self.write(
            HARNESS + 'memory/_capability.py',
            'class Memory:\n'
            '    def run(self, profile, store):\n'
            '        if profile.supports_tools:\n'
            '            return 1\n'
            '        if store and self.store:\n'
            '            return 2\n',
        )
        self.write('tests/harness/test_memory.py', 'def test_memory():\n    pass\n')
        self.commit('harness change')
        report = self.run_preconditions()
        pre = report['preconditions']
        assert isinstance(pre, dict)
        self.assertEqual(pre['packages_touched'], ['harness'])
        self.assertEqual(pre['public_modules'], [HARNESS + 'memory/_capability.py'])
        # Arm A reads a profile anywhere in a package; arm B's weak tokens stay slim-only.
        self.assertEqual(len(pre['ambient_branch_lines']), 1)
        self.assertIn('profile.supports_tools', pre['ambient_branch_lines'][0])
        self.assertTrue(
            {'review-public-api', 'review-runtime-behavior', 'review-tests', 'review-ambient-matrix'}
            <= self.dispatched(report)
        )
        self.assertNotIn('review-cassettes', self.dispatched(report))
        self.assertEqual((self.output / 'pr-ready-for-docs.txt').read_text(), 'yes\n')

    def test_graph_evals_cli_and_realtime_open_runtime_review(self) -> None:
        for package, path in (
            ('graph', 'pydantic_graph/pydantic_graph/__init__.py'),
            ('evals', 'pydantic_evals/pydantic_evals/dataset.py'),
            ('clai', 'clai/clai/__init__.py'),
            ('slim', SLIM + 'realtime/_session.py'),
            ('clai2', 'src/pydantic_clai2/pydantic_clai2/app.py'),
        ):
            with self.subTest(package=package):
                self.git('reset', '--hard', 'main')
                self.write(path, 'def run():\n    return 1\n')
                self.commit(package)
                report = self.run_preconditions()
                self.assertEqual(report['preconditions']['packages_touched'], [package])
                self.assertEqual(report['preconditions']['package_python_files'], [path])
                self.assertIn('review-runtime-behavior', self.dispatched(report))
                if package != 'slim':
                    self.assertIn('review-integration-impact', self.dispatched(report))

    def test_cross_package_change_keeps_integration_review(self) -> None:
        self.write('pydantic_graph/pydantic_graph/__init__.py', 'class Graph:\n    pass\n')
        self.write(SLIM + 'agent/__init__.py', 'from pydantic_graph import Graph\n\nclass Agent(Graph):\n    pass\n')
        self.commit('graph consumer')
        report = self.run_preconditions()
        self.assertEqual(report['preconditions']['packages_touched'], ['slim', 'graph'])
        self.assertIn('review-runtime-behavior', self.dispatched(report))
        self.assertIn('review-integration-impact', self.dispatched(report))

    def test_top_level_slim_module_and_agent_package(self) -> None:
        self.write(SLIM + 'messages.py', 'def new_part():\n    return 1\n')
        self.write(SLIM + 'agent/__init__.py', 'class Agent:\n    def new_method(self):\n        pass\n')
        self.commit('slim change')
        pre = self.run_preconditions()['preconditions']
        assert isinstance(pre, dict)
        self.assertEqual(pre['new_py_files'], [SLIM + 'messages.py'])
        self.assertEqual(pre['wrapped_class_files'], [SLIM + 'agent/__init__.py'])
        self.assertIn(SLIM + 'agent/__init__.py', pre['public_modules'])
        self.assertEqual((self.output / 'new-py-files.txt').read_text(), SLIM + 'messages.py\n')

    def test_slim_weak_ambient_token_needs_a_condition(self) -> None:
        self.write(
            SLIM + 'models/openai.py',
            'def request(model_name, background):\n'
            "    if model_name == 'gpt':\n"
            '        return background\n'
            '    # if background and store: comment only\n'
            '    return 1\n',
        )
        self.commit('ambient')
        pre = self.run_preconditions()['preconditions']
        assert isinstance(pre, dict)
        lines = pre['ambient_branch_lines']
        self.assertEqual(len(lines), 1)
        self.assertIn("if model_name == 'gpt'", lines[0])
        self.assertEqual(pre['provider_mapping_changed'], [SLIM + 'models/openai.py'])

    def test_realtime_profile_read_opens_ambient_review(self) -> None:
        self.write(
            SLIM + 'realtime/_session.py',
            'class RealtimeSession:\n'
            '    def seed(self):\n'
            "        if self._profile.get('supports_session_seeding', False):\n"
            '            return 1\n',
        )
        self.commit('realtime profile branch')
        report = self.run_preconditions()
        lines = report['preconditions']['ambient_branch_lines']
        self.assertEqual(len(lines), 1)
        self.assertIn("self._profile.get('supports_session_seeding'", lines[0])
        self.assertIn('review-ambient-matrix', self.dispatched(report))

    def test_concurrency_lifecycle_dispatch(self) -> None:
        for name, path, text, expected in (
            (
                'realtime path',
                SLIM + 'realtime/openai.py',
                'def connect():\n    return 1\n',
                SLIM + 'realtime/openai.py',
            ),
            (
                'task group in harness',
                HARNESS + 'memory/_capability.py',
                'import anyio\n\nasync def run():\n    async with anyio.create_task_group() as tg:\n        tg.start_soon(run)\n',
                'lifecycle-token-match',
            ),
            (
                'teardown in slim',
                SLIM + 'agent/__init__.py',
                'class Agent:\n    async def __aexit__(self, *args):\n        return None\n',
                'lifecycle-token-match',
            ),
            ('no lifecycle primitive', SLIM + 'models/openai.py', 'def request():\n    return 2\n', None),
            (
                'comment only',
                SLIM + 'models/openai.py',
                'def request():\n    # create_task( is mentioned here\n    return 1\n',
                None,
            ),
        ):
            with self.subTest(name):
                self.git('reset', '--hard', 'main')
                self.write(path, text)
                self.commit(name)
                report = self.run_preconditions()
                signals = report['preconditions']['concurrency_lifecycle']
                if expected is None:
                    self.assertEqual(signals, [])
                    self.assertNotIn('review-concurrency-lifecycle', self.dispatched(report))
                else:
                    self.assertIn(expected, signals)
                    self.assertIn('review-concurrency-lifecycle', self.dispatched(report))
                    self.assertEqual(report['reviewers']['review-concurrency-lifecycle']['wave'], 1)

    def test_trust_boundary_dispatch(self) -> None:
        for name, path, text, expected in (
            ('ui adapter', SLIM + 'ui/_adapter.py', 'def sanitize():\n    return 1\n', SLIM + 'ui/_adapter.py'),
            (
                'workspace backend',
                SLIM + 'workspaces/local.py',
                'def run():\n    return 1\n',
                SLIM + 'workspaces/local.py',
            ),
            (
                'harness sandbox',
                HARNESS + 'bubblewrap_sandbox/_workspace.py',
                'def launch():\n    return 1\n',
                HARNESS + 'bubblewrap_sandbox/_workspace.py',
            ),
            (
                'ssh workspace',
                HARNESS + 'ssh_workspace/_backend.py',
                'def stop():\n    return 1\n',
                HARNESS + 'ssh_workspace/_backend.py',
            ),
            (
                'digit in sandbox name',
                HARNESS + 'e2b_sandbox/_backend.py',
                'def stop():\n    return 1\n',
                HARNESS + 'e2b_sandbox/_backend.py',
            ),
            (
                'clai2 browser server',
                'src/pydantic_clai2/pydantic_clai2/builtin_plugins/web/server.py',
                'def serve():\n    return 1\n',
                'src/pydantic_clai2/pydantic_clai2/builtin_plugins/web/server.py',
            ),
            ('workflow', '.github/workflows/ci.yml', 'on: pull_request_target\n', '.github/workflows/ci.yml'),
            (
                'framing tag in harness',
                HARNESS + 'memory/_capability.py',
                "def frame(text):\n    return f'<memory>{text}</memory>'\n",
                'framing-token-match',
            ),
            ('plain model change', SLIM + 'models/openai.py', 'def request():\n    return 2\n', None),
        ):
            with self.subTest(name):
                self.git('reset', '--hard', 'main')
                self.write(path, text)
                self.commit(name)
                report = self.run_preconditions()
                signals = report['preconditions']['trust_boundary']
                if expected is None:
                    self.assertEqual(signals, [])
                    self.assertNotIn('review-trust-boundary', self.dispatched(report))
                else:
                    self.assertIn(expected, signals)
                    self.assertIn('review-trust-boundary', self.dispatched(report))
        self.assertNotIn('review-ui-security', report['reviewers'])

    def test_test_shape_needs_tests_that_reach_private_symbols(self) -> None:
        private = ''.join(f'def test_case_{i}():\n    assert adapter._load_message({i})\n' for i in range(10))
        public = ''.join(f'def test_case_{i}():\n    assert agent.run_sync({i})\n' for i in range(10))
        for name, text, expected in (('private', private, True), ('public', public, False)):
            with self.subTest(name):
                self.git('reset', '--hard', 'main')
                self.write('tests/test_shape.py', text)
                self.commit(name)
                report = self.run_preconditions()
                pre = report['preconditions']
                self.assertEqual(pre['new_test_fn_count'], 10)
                self.assertEqual(pre['private_test_reach_lines'], 10 if expected else 0)
                self.assertEqual('review-test-shape' in self.dispatched(report), expected)

    def test_realtime_and_harness_family_members(self) -> None:
        self.write(SLIM + 'realtime/openai.py', 'def connect():\n    return 1\n')
        self.write(HARNESS + 'docker_sandbox/_workspace.py', 'def launch():\n    return 1\n')
        self.commit('family base')
        self.git('branch', '-f', 'main')
        self.write(SLIM + 'realtime/openai.py', 'def connect():\n    return 2\n')
        self.write(HARNESS + 'docker_sandbox/_workspace.py', 'def launch():\n    return 2\n')
        self.write(HARNESS + 'pgtask/_operation_backend.py', 'def run():\n    return 1\n')
        self.commit('family change')
        report = self.run_preconditions()
        pre = report['preconditions']
        self.assertEqual(
            sorted(pre['sibling_family_files']),
            [SLIM + 'realtime/openai.py', HARNESS + 'docker_sandbox/_workspace.py'],
        )
        self.assertEqual(pre['sibling_family_added'], [HARNESS + 'pgtask/_operation_backend.py'])
        self.assertIn('review-sibling-consistency', self.dispatched(report))
        # A realtime codec maps the provider wire, so cassette Check 7 applies to it.
        self.assertEqual(pre['provider_mapping_changed'], [SLIM + 'realtime/openai.py'])
        self.assertIn('review-cassettes', self.dispatched(report))

    def test_missing_agent_file_stops_before_dispatch(self) -> None:
        self.write('README.md', 'docs\n')
        self.commit('docs')
        prepare.prepare(self.repo, 'main', 'HEAD', self.output)
        (self.repo / '.agents/agents/review-trust-boundary.md').unlink()
        result = subprocess.run(
            [sys.executable, str(HERE / 'preconditions.py'), '--repo', str(self.repo), '--output', str(self.output)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn('review-trust-boundary', result.stderr)
        self.assertIn('Restore the review agents', result.stderr)
        self.assertFalse((self.output / 'preconditions.json').exists())

    def test_base_without_durable_exec_contract(self) -> None:
        for name, contract in (
            ('no contract', {}),
            (
                'classifier without touched_entries',
                {
                    'scripts/durable_exec_contract.py': 'def build_manifest():\n    return {}\n',
                    'scripts/durable_exec_contract.json': '{}\n',
                },
            ),
        ):
            with self.subTest(name):
                self.git('reset', '--hard', 'main')
                if contract:
                    for path, text in contract.items():
                        self.write(path, text)
                    self.commit(f'{name} base')
                    self.git('branch', '-f', 'main')
                self.write(SLIM + 'models/openai.py', 'def request():\n    return 2\n')
                self.commit('model change')
                report = self.run_preconditions()
                self.assertIsNone(report['preconditions']['durable_exec_entries'])
                self.assertIn(
                    'durable-exec contract: absent at base', report['reviewers']['review-integration-impact']['reason']
                )
                self.assertEqual((self.output / 'durable-exec-entries.txt').read_text(), '')

    def write_contract(self) -> None:
        """Write this repo's classifier and `CONTRACT_MANIFEST`."""
        self.write('scripts/durable_exec_contract.py', (REPO / 'scripts/durable_exec_contract.py').read_text())
        self.write('scripts/durable_exec_contract.json', json.dumps(CONTRACT_MANIFEST, indent=2) + '\n')

    def test_durable_exec_contract_entries(self) -> None:
        self.write_contract()
        self.write(CORE_FILE, CORE_SOURCE)
        self.commit('durable-exec contract')
        self.git('branch', '-f', 'main')
        for name, source, expected in (
            (
                'gated body',
                CORE_SOURCE.replace('self.usage + 1', 'self.usage + 2'),
                [f'{CORE_FILE}::RunContext.step (temporal)'],
            ),
            ('docstring only', CORE_SOURCE.replace('Advance the run.', 'Advance the run by one step.'), []),
            ('annotation-only entry', CORE_SOURCE.replace("'run'", "'step'"), []),
            (
                'core file removed',
                None,
                [f'{CORE_FILE}::RunContext.step (temporal)', f'{CORE_FILE}::RunContext.usage (dbos, temporal)'],
            ),
        ):
            with self.subTest(name):
                self.git('reset', '--hard', 'main')
                if source is None:
                    (self.repo / CORE_FILE).unlink()
                else:
                    self.write(CORE_FILE, source)
                self.commit(name)
                report = self.run_preconditions()
                self.assertEqual(report['preconditions']['durable_exec_entries'], expected)
                self.assertEqual(
                    (self.output / 'durable-exec-entries.txt').read_text(), ''.join(f'{line}\n' for line in expected)
                )
                self.assertIn(
                    f'durable-exec entries: {len(expected)}', report['reviewers']['review-integration-impact']['reason']
                )
                # No other trigger of the lane fires on this core file.
                self.assertEqual('review-integration-impact' in self.dispatched(report), bool(expected))

    def test_durable_exec_contract_read_at_base_tip(self) -> None:
        self.git('checkout', '-q', 'main')
        self.write(CORE_FILE, CORE_SOURCE)
        self.commit('core file')
        self.git('checkout', '-qB', 'feature')
        # The base branch gains the contract after the feature branch was cut, so the merge base has none.
        self.git('checkout', '-q', 'main')
        self.write_contract()
        self.commit('durable-exec contract')
        self.git('checkout', '-q', 'feature')
        self.write(CORE_FILE, CORE_SOURCE.replace('self.usage + 1', 'self.usage + 2'))
        self.commit('gated body')
        report = self.run_preconditions()
        entry = f'{CORE_FILE}::RunContext.step (temporal)'
        self.assertEqual(report['preconditions']['durable_exec_entries'], [entry])
        self.assertEqual((self.output / 'durable-exec-entries.txt').read_text(), f'{entry}\n')
        self.assertIn('review-integration-impact', self.dispatched(report))

    def test_pr_metadata(self) -> None:
        self.write('README.md', 'docs\n')
        self.commit('docs')
        report = self.run_preconditions(
            {'body': 'Intro\nSpec: https://example.com/spec\n', 'isDraft': True, 'reviewDecision': ''}
        )
        pre = report['preconditions']
        assert isinstance(pre, dict)
        self.assertEqual(pre['spec_urls'], ['https://example.com/spec'])
        self.assertEqual(pre['pr_ready_for_docs'], 'no')
        self.assertIn('review-spec-coverage', self.dispatched(report))
        self.assertIn('review-docs', self.dispatched(report))
        self.assertNotIn('review-runtime-behavior', self.dispatched(report))


class ChangedLinesTests(unittest.TestCase):
    def test_body_lines_that_look_like_headers(self) -> None:
        diff = (
            b'diff --git a/x.py b/x.py\n'
            b'--- a/x.py\n'
            b'+++ b/x.py\n'
            b'@@ -1,2 +1,2 @@\n'
            b'--- old\n'
            b'+++ new\n'
            b' same\n'
            b'\\ No newline at end of file\n'
        )
        self.assertEqual(preconditions.changed_lines(diff), {'x.py': ['--- old', '+++ new']})

    def test_deleted_file_uses_old_path(self) -> None:
        diff = b'diff --git a/y.py b/y.py\n--- a/y.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-gone\n'
        self.assertEqual(preconditions.changed_lines(diff), {'y.py': ['-gone']})


if __name__ == '__main__':
    unittest.main()
