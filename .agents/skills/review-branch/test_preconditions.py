"""Tests for preconditions.py against temporary Git histories.

Run: uv run pytest .agents/skills/review-branch/test_preconditions.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import preconditions  # noqa: E402
import prepare  # noqa: E402

HARNESS = 'src/pydantic_ai_harness/pydantic_ai_harness/'
SLIM = 'pydantic_ai_slim/pydantic_ai/'
CONTRACT_CLASSIFIER = HERE.parents[2] / 'scripts/durable_exec_contract.py'
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


@dataclass
class Report:
    """`preconditions.json`, read back into the types `preconditions.main` writes it from."""

    pre: preconditions.Preconditions
    reviewers: dict[str, preconditions.ReviewerDispatch]

    @property
    def dispatched(self) -> set[str]:
        return {name for name, entry in self.reviewers.items() if entry.dispatch}


@dataclass
class GitRepo:
    """A pydantic-ai-shaped repository on a `feature` branch cut from `main`."""

    path: Path
    output: Path

    def git(self, *args: str) -> str:
        return subprocess.run(
            ['git', '-C', str(self.path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    def write(self, name: str, text: str) -> None:
        target = self.path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def commit(self, message: str) -> None:
        self.git('add', '--all')
        self.git('commit', '-qm', message)

    def write_contract(self) -> None:
        """Write this repo's classifier and `CONTRACT_MANIFEST`."""
        self.write('scripts/durable_exec_contract.py', CONTRACT_CLASSIFIER.read_text())
        self.write('scripts/durable_exec_contract.json', json.dumps(CONTRACT_MANIFEST, indent=2) + '\n')

    def run_preconditions(self, pr: dict[str, object] | None = None) -> Report:
        prepare.prepare(self.path, 'main', 'HEAD', self.output)
        pr_json = self.output / 'pr.json'
        pr_json.write_text(json.dumps(pr if pr is not None else {'body': '', 'isDraft': False}))
        args = ['--repo', str(self.path), '--output', str(self.output), '--pr-json', str(pr_json)]
        assert preconditions.main(args) == 0
        report = json.loads((self.output / 'preconditions.json').read_text())
        return Report(
            preconditions.Preconditions(**report['preconditions']),
            {name: preconditions.ReviewerDispatch(**entry) for name, entry in report['reviewers'].items()},
        )


@pytest.fixture
def repo(tmp_path: Path) -> GitRepo:
    # The output sits outside the repo, so a commit never picks up a run's artifacts.
    repo = GitRepo(tmp_path / 'repo', tmp_path / 'review')
    repo.path.mkdir()
    for args in (
        ('init', '-q', '--initial-branch=main'),
        ('config', 'user.name', 'Review Test'),
        ('config', 'user.email', 'review@example.invalid'),
        ('config', 'commit.gpgSign', 'false'),
        ('config', 'core.hooksPath', '/dev/null'),
    ):
        repo.git(*args)
    repo.write(SLIM + 'models/openai.py', 'def request():\n    return 1\n')
    repo.write(SLIM + 'agent/__init__.py', 'class Agent:\n    pass\n')
    repo.write(HARNESS + 'memory/_capability.py', 'class Memory:\n    pass\n')
    repo.write(HARNESS + 'memory/__init__.py', "from ._capability import Memory\n\n__all__ = ['Memory']\n")
    for agent in preconditions.REVIEWER_AGENTS:
        repo.write(f'.agents/agents/{agent}.md', f'---\nname: {agent}\n---\n')
    repo.commit('base')
    repo.git('checkout', '-qb', 'feature')
    return repo


def test_harness_diff_opens_package_lanes(repo: GitRepo) -> None:
    repo.write(
        HARNESS + 'memory/_capability.py',
        'class Memory:\n'
        '    def run(self, profile, store):\n'
        '        if profile.supports_tools:\n'
        '            return 1\n'
        '        if store and self.store:\n'
        '            return 2\n',
    )
    repo.write('tests/harness/test_memory.py', 'def test_memory():\n    pass\n')
    repo.commit('harness change')
    report = repo.run_preconditions()
    assert report.pre.packages_touched == ['harness']
    assert report.pre.public_modules == [HARNESS + 'memory/_capability.py']
    # Arm A reads a profile anywhere in a package; arm B's weak tokens stay slim-only.
    assert len(report.pre.ambient_branch_lines) == 1
    assert 'profile.supports_tools' in report.pre.ambient_branch_lines[0]
    assert {'review-public-api', 'review-runtime-behavior', 'review-tests', 'review-ambient-matrix'} <= (
        report.dispatched
    )
    assert 'review-cassettes' not in report.dispatched
    assert (repo.output / 'pr-ready-for-docs.txt').read_text() == 'yes\n'


@pytest.mark.parametrize(
    ('package', 'path'),
    [
        ('graph', 'pydantic_graph/pydantic_graph/__init__.py'),
        ('evals', 'pydantic_evals/pydantic_evals/dataset.py'),
        ('clai', 'clai/clai/__init__.py'),
        ('slim', SLIM + 'realtime/_session.py'),
        ('clai2', 'src/pydantic_clai2/pydantic_clai2/app.py'),
    ],
)
def test_graph_evals_cli_and_realtime_open_runtime_review(repo: GitRepo, package: str, path: str) -> None:
    repo.write(path, 'def run():\n    return 1\n')
    repo.commit(package)
    report = repo.run_preconditions()
    assert report.pre.packages_touched == [package]
    assert report.pre.package_python_files == [path]
    assert 'review-runtime-behavior' in report.dispatched
    if package != 'slim':
        assert 'review-integration-impact' in report.dispatched


def test_cross_package_change_keeps_integration_review(repo: GitRepo) -> None:
    repo.write('pydantic_graph/pydantic_graph/__init__.py', 'class Graph:\n    pass\n')
    repo.write(SLIM + 'agent/__init__.py', 'from pydantic_graph import Graph\n\nclass Agent(Graph):\n    pass\n')
    repo.commit('graph consumer')
    report = repo.run_preconditions()
    assert report.pre.packages_touched == ['slim', 'graph']
    assert {'review-runtime-behavior', 'review-integration-impact'} <= report.dispatched


def test_top_level_slim_module_and_agent_package(repo: GitRepo) -> None:
    repo.write(SLIM + 'messages.py', 'def new_part():\n    return 1\n')
    repo.write(SLIM + 'agent/__init__.py', 'class Agent:\n    def new_method(self):\n        pass\n')
    repo.commit('slim change')
    pre = repo.run_preconditions().pre
    assert pre.new_py_files == [SLIM + 'messages.py']
    assert pre.wrapped_class_files == [SLIM + 'agent/__init__.py']
    assert SLIM + 'agent/__init__.py' in pre.public_modules
    assert (repo.output / 'new-py-files.txt').read_text() == SLIM + 'messages.py\n'


def test_slim_weak_ambient_token_needs_a_condition(repo: GitRepo) -> None:
    repo.write(
        SLIM + 'models/openai.py',
        'def request(model_name, background):\n'
        "    if model_name == 'gpt':\n"
        '        return background\n'
        '    # if background and store: comment only\n'
        '    return 1\n',
    )
    repo.commit('ambient')
    pre = repo.run_preconditions().pre
    assert len(pre.ambient_branch_lines) == 1
    assert "if model_name == 'gpt'" in pre.ambient_branch_lines[0]
    assert pre.provider_mapping_changed == [SLIM + 'models/openai.py']


def test_realtime_profile_read_opens_ambient_review(repo: GitRepo) -> None:
    repo.write(
        SLIM + 'realtime/_session.py',
        'class RealtimeSession:\n'
        '    def seed(self):\n'
        "        if self._profile.get('supports_session_seeding', False):\n"
        '            return 1\n',
    )
    repo.commit('realtime profile branch')
    report = repo.run_preconditions()
    assert len(report.pre.ambient_branch_lines) == 1
    assert "self._profile.get('supports_session_seeding'" in report.pre.ambient_branch_lines[0]
    assert 'review-ambient-matrix' in report.dispatched


@pytest.mark.parametrize(
    ('path', 'text', 'expected'),
    [
        pytest.param(
            SLIM + 'realtime/openai.py',
            'def connect():\n    return 1\n',
            SLIM + 'realtime/openai.py',
            id='realtime path',
        ),
        pytest.param(
            HARNESS + 'memory/_capability.py',
            'import anyio\n\nasync def run():\n    async with anyio.create_task_group() as tg:\n        tg.start_soon(run)\n',
            'lifecycle-token-match',
            id='task group in harness',
        ),
        pytest.param(
            SLIM + 'agent/__init__.py',
            'class Agent:\n    async def __aexit__(self, *args):\n        return None\n',
            'lifecycle-token-match',
            id='teardown in slim',
        ),
        pytest.param(SLIM + 'models/openai.py', 'def request():\n    return 2\n', None, id='no lifecycle primitive'),
        pytest.param(
            SLIM + 'models/openai.py',
            'def request():\n    # create_task( is mentioned here\n    return 1\n',
            None,
            id='comment only',
        ),
    ],
)
def test_concurrency_lifecycle_dispatch(repo: GitRepo, path: str, text: str, expected: str | None) -> None:
    repo.write(path, text)
    repo.commit('change')
    report = repo.run_preconditions()
    signals = report.pre.concurrency_lifecycle
    if expected is None:
        assert signals == []
        assert 'review-concurrency-lifecycle' not in report.dispatched
    else:
        assert expected in signals
        assert 'review-concurrency-lifecycle' in report.dispatched
        assert report.reviewers['review-concurrency-lifecycle'].wave == 1


@pytest.mark.parametrize(
    ('path', 'text', 'expected'),
    [
        pytest.param(
            SLIM + 'ui/_adapter.py', 'def sanitize():\n    return 1\n', SLIM + 'ui/_adapter.py', id='ui adapter'
        ),
        pytest.param(
            SLIM + 'workspaces/local.py',
            'def run():\n    return 1\n',
            SLIM + 'workspaces/local.py',
            id='workspace backend',
        ),
        pytest.param(
            HARNESS + 'bubblewrap_sandbox/_workspace.py',
            'def launch():\n    return 1\n',
            HARNESS + 'bubblewrap_sandbox/_workspace.py',
            id='harness sandbox',
        ),
        pytest.param(
            HARNESS + 'ssh_workspace/_backend.py',
            'def stop():\n    return 1\n',
            HARNESS + 'ssh_workspace/_backend.py',
            id='ssh workspace',
        ),
        pytest.param(
            HARNESS + 'e2b_sandbox/_backend.py',
            'def stop():\n    return 1\n',
            HARNESS + 'e2b_sandbox/_backend.py',
            id='digit in sandbox name',
        ),
        pytest.param(
            'src/pydantic_clai2/pydantic_clai2/builtin_plugins/web/server.py',
            'def serve():\n    return 1\n',
            'src/pydantic_clai2/pydantic_clai2/builtin_plugins/web/server.py',
            id='clai2 browser server',
        ),
        pytest.param(
            '.github/workflows/ci.yml', 'on: pull_request_target\n', '.github/workflows/ci.yml', id='workflow'
        ),
        pytest.param(
            HARNESS + 'memory/_capability.py',
            "def frame(text):\n    return f'<memory>{text}</memory>'\n",
            'framing-token-match',
            id='framing tag in harness',
        ),
        pytest.param(SLIM + 'models/openai.py', 'def request():\n    return 2\n', None, id='plain model change'),
    ],
)
def test_trust_boundary_dispatch(repo: GitRepo, path: str, text: str, expected: str | None) -> None:
    repo.write(path, text)
    repo.commit('change')
    report = repo.run_preconditions()
    signals = report.pre.trust_boundary
    if expected is None:
        assert signals == []
        assert 'review-trust-boundary' not in report.dispatched
    else:
        assert expected in signals
        assert 'review-trust-boundary' in report.dispatched
    assert 'review-ui-security' not in report.reviewers


@pytest.mark.parametrize('private', [True, False])
def test_test_shape_needs_tests_that_reach_private_symbols(repo: GitRepo, private: bool) -> None:
    call = 'adapter._load_message' if private else 'agent.run_sync'
    repo.write('tests/test_shape.py', ''.join(f'def test_case_{i}():\n    assert {call}({i})\n' for i in range(10)))
    repo.commit('tests')
    report = repo.run_preconditions()
    assert report.pre.new_test_fn_count == 10
    assert report.pre.private_test_reach_lines == (10 if private else 0)
    assert ('review-test-shape' in report.dispatched) == private


def test_realtime_and_harness_family_members(repo: GitRepo) -> None:
    repo.write(SLIM + 'realtime/openai.py', 'def connect():\n    return 1\n')
    repo.write(HARNESS + 'docker_sandbox/_workspace.py', 'def launch():\n    return 1\n')
    repo.commit('family base')
    repo.git('branch', '-f', 'main')
    repo.write(SLIM + 'realtime/openai.py', 'def connect():\n    return 2\n')
    repo.write(HARNESS + 'docker_sandbox/_workspace.py', 'def launch():\n    return 2\n')
    repo.write(HARNESS + 'pgtask/_operation_backend.py', 'def run():\n    return 1\n')
    repo.commit('family change')
    report = repo.run_preconditions()
    assert sorted(report.pre.sibling_family_files) == [
        SLIM + 'realtime/openai.py',
        HARNESS + 'docker_sandbox/_workspace.py',
    ]
    assert report.pre.sibling_family_added == [HARNESS + 'pgtask/_operation_backend.py']
    assert 'review-sibling-consistency' in report.dispatched
    # A realtime codec maps the provider wire, so cassette Check 7 applies to it.
    assert report.pre.provider_mapping_changed == [SLIM + 'realtime/openai.py']
    assert 'review-cassettes' in report.dispatched


def test_provider_package_module_counts_as_a_model_module(repo: GitRepo) -> None:
    package = SLIM + 'models/anthropic/'
    repo.write(package + '__init__.py', 'class AnthropicModel:\n    pass\n')
    repo.write(package + '_streaming.py', 'def accumulate(response):\n    return response\n')
    repo.commit('provider package')
    repo.git('branch', '-f', 'main')
    repo.write(package + '_streaming.py', 'def accumulate(response):\n    return response._parts_manager\n')
    repo.commit('streaming change')
    report = repo.run_preconditions()
    assert report.pre.sibling_family_files == [package + '_streaming.py']
    assert report.pre.wrapped_class_files == [package + '_streaming.py']
    assert 'models-streaming-state-match' in report.pre.suspend_resume
    assert {'review-sibling-consistency', 'review-integration-impact', 'review-suspend-resume'} <= report.dispatched


def test_missing_agent_file_stops_before_dispatch(repo: GitRepo, capsys: pytest.CaptureFixture[str]) -> None:
    repo.write('README.md', 'docs\n')
    repo.commit('docs')
    prepare.prepare(repo.path, 'main', 'HEAD', repo.output)
    (repo.path / '.agents/agents/review-trust-boundary.md').unlink()
    assert preconditions.main(['--repo', str(repo.path), '--output', str(repo.output)]) == 1
    stderr = capsys.readouterr().err
    assert 'review-trust-boundary' in stderr
    assert 'Restore the review agents' in stderr
    assert not (repo.output / 'preconditions.json').exists()


@pytest.mark.parametrize(
    'contract',
    [
        pytest.param({}, id='no contract'),
        pytest.param(
            {
                'scripts/durable_exec_contract.py': 'def build_manifest():\n    return {}\n',
                'scripts/durable_exec_contract.json': '{}\n',
            },
            id='classifier without touched_entries',
        ),
    ],
)
def test_base_without_durable_exec_contract(repo: GitRepo, contract: dict[str, str]) -> None:
    if contract:
        for path, text in contract.items():
            repo.write(path, text)
        repo.commit('contract base')
        repo.git('branch', '-f', 'main')
    repo.write(SLIM + 'models/openai.py', 'def request():\n    return 2\n')
    repo.commit('model change')
    report = repo.run_preconditions()
    assert report.pre.durable_exec_entries is None
    assert 'durable-exec contract: absent at base' in report.reviewers['review-integration-impact'].reason
    assert (repo.output / 'durable-exec-entries.txt').read_text() == ''


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        pytest.param(
            CORE_SOURCE.replace('self.usage + 1', 'self.usage + 2'),
            [f'{CORE_FILE}::RunContext.step (temporal)'],
            id='gated body',
        ),
        pytest.param(CORE_SOURCE.replace('Advance the run.', 'Advance the run by one step.'), [], id='docstring only'),
        pytest.param(CORE_SOURCE.replace("'run'", "'step'"), [], id='annotation-only entry'),
        pytest.param(
            None,
            [f'{CORE_FILE}::RunContext.step (temporal)', f'{CORE_FILE}::RunContext.usage (dbos, temporal)'],
            id='core file removed',
        ),
    ],
)
def test_durable_exec_contract_entries(repo: GitRepo, source: str | None, expected: list[str]) -> None:
    repo.write_contract()
    repo.write(CORE_FILE, CORE_SOURCE)
    repo.commit('durable-exec contract')
    repo.git('branch', '-f', 'main')
    if source is None:
        (repo.path / CORE_FILE).unlink()
    else:
        repo.write(CORE_FILE, source)
    repo.commit('change')
    report = repo.run_preconditions()
    assert report.pre.durable_exec_entries == expected
    assert (repo.output / 'durable-exec-entries.txt').read_text() == ''.join(f'{line}\n' for line in expected)
    assert f'durable-exec entries: {len(expected)}' in report.reviewers['review-integration-impact'].reason
    # No other trigger of the lane fires on this core file.
    assert ('review-integration-impact' in report.dispatched) == bool(expected)


def test_durable_exec_contract_read_at_base_tip(repo: GitRepo) -> None:
    repo.git('checkout', '-q', 'main')
    repo.write(CORE_FILE, CORE_SOURCE)
    repo.commit('core file')
    repo.git('checkout', '-qB', 'feature')
    # The base branch gains the contract after the feature branch was cut, so the merge base has none.
    repo.git('checkout', '-q', 'main')
    repo.write_contract()
    repo.commit('durable-exec contract')
    repo.git('checkout', '-q', 'feature')
    repo.write(CORE_FILE, CORE_SOURCE.replace('self.usage + 1', 'self.usage + 2'))
    repo.commit('gated body')
    report = repo.run_preconditions()
    entry = f'{CORE_FILE}::RunContext.step (temporal)'
    assert report.pre.durable_exec_entries == [entry]
    assert (repo.output / 'durable-exec-entries.txt').read_text() == f'{entry}\n'
    assert 'review-integration-impact' in report.dispatched


def test_pr_metadata(repo: GitRepo) -> None:
    repo.write('README.md', 'docs\n')
    repo.commit('docs')
    report = repo.run_preconditions(
        {'body': 'Intro\nSpec: https://example.com/spec\n', 'isDraft': True, 'reviewDecision': ''}
    )
    assert report.pre.spec_urls == ['https://example.com/spec']
    assert report.pre.pr_ready_for_docs == 'no'
    assert {'review-spec-coverage', 'review-docs'} <= report.dispatched
    assert 'review-runtime-behavior' not in report.dispatched


def test_body_lines_that_look_like_headers() -> None:
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
    assert preconditions.changed_lines(diff) == {'x.py': ['--- old', '+++ new']}


def test_deleted_file_uses_old_path() -> None:
    diff = b'diff --git a/y.py b/y.py\n--- a/y.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-gone\n'
    assert preconditions.changed_lines(diff) == {'y.py': ['-gone']}
