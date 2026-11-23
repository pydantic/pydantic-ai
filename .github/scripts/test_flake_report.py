from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import flake_report as report

REPO = 'pydantic/pydantic-ai'
LINK = f'[run 7](https://github.com/{REPO}/actions/runs/7)'
PASSED = ''
FAILED = '<failure message="AssertionError: assert 1 == 2">long traceback</failure>'
ERRORED = '<error message="failed on teardown with &quot;RuntimeError: boom&quot;">long traceback</error>'
SKIPPED = '<skipped type="pytest.skip" message="needs a thing">skip</skipped>'


def write_report(artifact: Path, name: str, *cases: tuple[str, str]) -> None:
    """A JUnit report, as pytest writes one, of `(node id, result)` cases."""
    body = ''
    for node_id, result in cases:
        file, *names = node_id.split('::')
        classname = '.'.join([file.removesuffix('.py').replace('/', '.'), *names[:-1]])
        body += f'<testcase classname="{classname}" name="{names[-1]}" file="tests/base.py">{result}</testcase>'
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / name).write_text(f'<testsuites><testsuite>{body}</testsuite></testsuites>', encoding='utf-8')


@pytest.fixture
def root(tmp_path: Path) -> Path:
    checkout = tmp_path / 'checkout'
    for file in ('tests/test_a.py', 'tests/models/test_b.py'):
        (checkout / file).parent.mkdir(parents=True, exist_ok=True)
        (checkout / file).touch()
    return checkout


class FakeClient(report.attention.GitHubClient):
    """Open GitHub issues held in memory; every write is recorded."""

    def __init__(self, *issues: Mapping[str, object], listed_all: bool = True) -> None:
        super().__init__('token')
        self.issues = [dict(issue) for issue in issues]
        self.listed_all = listed_all
        self.created: list[dict[str, object]] = []
        self.comments: list[tuple[str, str]] = []

    def first_pages(self, path: str, *, count: int) -> tuple[list[dict[str, object]], bool]:
        assert path == f'/repos/{REPO}/issues?state=open&labels=flake'
        return self.issues, self.listed_all

    def post(self, path: str, payload: Mapping[str, object]) -> object:
        if path == f'/repos/{REPO}/issues':
            issue = {**payload, 'number': 200 + len(self.created)}
            self.created.append(issue)
            return issue
        self.comments.append((path, str(payload['body'])))
        return {}


def run(seed: int | None = None) -> report.Run:
    return report.Run(REPO, 7, 'https://github.com', seed)


def test_a_test_that_failed_and_then_passed_in_one_leg_is_flaky(tmp_path: Path, root: Path):
    reports = tmp_path / 'reports'
    write_report(
        reports / 'junit-test-0',
        'first.xml',
        ('tests/test_a.py::test_flaky', FAILED),
        ('tests/test_a.py::test_broken', FAILED),
        ('tests/test_a.py::test_teardown', ERRORED),
        ('tests/test_a.py::test_skipped_on_rerun', FAILED),
        ('tests/models/test_b.py::TestB::test_flaky[openai-True]', FAILED),
    )
    write_report(
        reports / 'junit-test-0',
        'rerun.xml',
        ('tests/test_a.py::test_flaky', PASSED),
        ('tests/test_a.py::test_broken', FAILED),
        ('tests/test_a.py::test_teardown', PASSED),
        ('tests/test_a.py::test_skipped_on_rerun', SKIPPED),
        ('tests/models/test_b.py::TestB::test_flaky[openai-True]', PASSED),
    )
    # Legs differ in Python and install, so a failure in one and a pass in another is no flake.
    write_report(reports / 'junit-test-all-extras-3', 'a.xml', ('tests/test_a.py::test_per_leg', FAILED))
    write_report(reports / 'junit-test-all-extras-4', 'b.xml', ('tests/test_a.py::test_per_leg', PASSED))
    # Only directories named like a test job's artifact are read.
    write_report(
        reports / 'other', 'a.xml', ('tests/test_a.py::test_other', FAILED), ('tests/test_a.py::test_other', PASSED)
    )

    assert report.collect(reports, root) == {
        'tests/models/test_b.py::TestB::test_flaky[openai-True]': report.Flake(
            {'test leg 0': 'AssertionError: assert 1 == 2'}
        ),
        'tests/test_a.py::test_flaky': report.Flake({'test leg 0': 'AssertionError: assert 1 == 2'}),
        'tests/test_a.py::test_teardown': report.Flake({'test leg 0': 'failed on teardown with "RuntimeError: boom"'}),
    }


def test_with_fails_on_rerun_a_test_that_failed_its_rerun_too_is_filed(tmp_path: Path, root: Path):
    reports = tmp_path / 'reports'
    write_report(
        reports / 'junit-shuffled-0',
        'first.xml',
        ('tests/test_a.py::test_broken', FAILED),
        ('tests/test_a.py::test_flaky', FAILED),
        ('tests/test_a.py::test_not_rerun', FAILED),
        # Failed in call and in teardown: pytest lists the test twice in one report.
        ('tests/test_a.py::test_twice_in_one_report', FAILED),
        ('tests/test_a.py::test_twice_in_one_report', ERRORED),
        ('tests/test_a.py::test_mixed', FAILED),
    )
    write_report(
        reports / 'junit-shuffled-0',
        'rerun.xml',
        ('tests/test_a.py::test_broken', ERRORED),
        ('tests/test_a.py::test_flaky', PASSED),
        ('tests/test_a.py::test_mixed', FAILED),
    )
    # Flaky in one leg, so filed as flaky, with only the legs it flaked in.
    write_report(reports / 'junit-shuffled-1', 'first.xml', ('tests/test_a.py::test_mixed', FAILED))
    write_report(reports / 'junit-shuffled-1', 'rerun.xml', ('tests/test_a.py::test_mixed', PASSED))

    assert report.collect(reports, root, fails_on_rerun=True) == {
        'tests/test_a.py::test_broken': report.Flake(
            {'shuffled leg 0': 'AssertionError: assert 1 == 2'}, fails_on_rerun=True
        ),
        'tests/test_a.py::test_flaky': report.Flake({'shuffled leg 0': 'AssertionError: assert 1 == 2'}),
        'tests/test_a.py::test_mixed': report.Flake({'shuffled leg 1': 'AssertionError: assert 1 == 2'}),
    }
    # The merge queue files only the tests that passed their rerun.
    assert list(report.collect(reports, root)) == ['tests/test_a.py::test_flaky', 'tests/test_a.py::test_mixed']


def test_a_report_cut_short_still_counts_and_is_named(tmp_path: Path, root: Path, capsys: pytest.CaptureFixture[str]):
    reports = tmp_path / 'reports'
    write_report(reports / 'junit-test-0', 'a.xml', ('tests/test_a.py::test_flaky', FAILED))
    (reports / 'junit-test-0' / 'b.xml').write_text(
        '<testsuites><testsuite><testcase classname="tests.test_a" name="test_flaky"/><testcase', encoding='utf-8'
    )

    assert report.collect(reports, root) == {
        'tests/test_a.py::test_flaky': report.Flake({'test leg 0': 'AssertionError: assert 1 == 2'})
    }
    assert capsys.readouterr().out.startswith('::warning::junit-test-0/b.xml read only in part: ')


@pytest.mark.parametrize(
    ('classname', 'name', 'expected'),
    [
        ('tests.test_a', 'test_x', 'tests/test_a.py::test_x'),
        (
            'tests.models.test_b.TestB',
            'test_x[gpt-4o, anthropic:claude/3 =1.5+]',
            'tests/models/test_b.py::TestB::test_x[gpt-4o, anthropic:claude/3 =1.5+]',
        ),
        # A parametrization id that could mention, reference or mark up is dropped, and the cases share one issue.
        ('tests.test_a', 'test_x[@someone]', 'tests/test_a.py::test_x'),
        ('tests.test_a', 'test_x[#123]', 'tests/test_a.py::test_x'),
        ('tests.test_a', 'test_x[`code`]', 'tests/test_a.py::test_x'),
        ('tests.test_a', 'test_x[é]', 'tests/test_a.py::test_x'),
        ('tests.test_a', f'test_x[{"a" * 101}]', 'tests/test_a.py::test_x'),
        # Paths outside the checkout's `tests/` and names that aren't identifiers make no node id.
        ('tests.test_missing', 'test_x', None),
        ('scripts.test_a', 'test_x', None),
        ('tests.test_a.Test B', 'test_x', None),
        ('tests.test_a', 'test x', None),
        ('tests.test_a', '[x]', None),
    ],
)
def test_a_node_id_holds_only_checkout_paths_and_identifier_shaped_names(
    root: Path, classname: str, name: str, expected: str | None
):
    assert report.issue_node_id(classname, name, root) == expected


def test_a_node_id_whose_path_is_not_the_files_own_makes_none(root: Path):
    # An empty part in a report's class name still finds the file, but the path it spells is not the file's.
    assert report.junit.collected_node_id('tests..test_a', 'test_x', root) == 'tests//test_a.py::test_x'

    assert report.issue_node_id('tests..test_a', 'test_x', root) is None


def test_a_flaky_test_without_an_issue_gets_one_labeled_flake_with_fenced_evidence():
    client = FakeClient()
    flakes = {'tests/test_a.py::test_x': report.Flake({'test leg 0': 'AssertionError: ``` `@someone` #123'})}

    assert report.reconcile(client, run(), flakes) == ['#200: filed `tests/test_a.py::test_x`']
    assert client.created == [
        {
            'title': 'Flaky tests/test_a.py::test_x',
            'body': (
                f'`tests/test_a.py::test_x` failed and then passed when its job reran it, in {LINK}.\n\n'
                '````text\ntest leg 0: AssertionError: ``` `@someone` #123\n````\n\n'
                'Claim this issue before you fix the flake, as "Flake claims" in '
                '`.agents/skills/pushing-commits-to-the-repo/SKILL.md` says.\n'
            ),
            'labels': ['flake'],
            'number': 200,
        }
    ]
    assert client.comments == []


def test_a_flaky_test_with_an_open_issue_of_its_exact_title_gets_a_comment():
    client = FakeClient(
        {'number': 101, 'title': 'Flaky tests/test_a.py::test_x'},
        {'number': 102, 'title': 'Flaky tests/test_a.py::test_x[1]'},
        {'number': 103, 'title': 'Flaky tests/test_a.py::test_y', 'pull_request': {}},
    )
    flakes = {
        'tests/test_a.py::test_x': report.Flake({'test leg 0': 'boom', 'test-all-extras leg 3': 'bang'}),
        'tests/test_a.py::test_y': report.Flake({'test leg 1': 'boom'}),
    }

    assert report.reconcile(client, run(seed=42), flakes) == [
        '#101: noted that `tests/test_a.py::test_x` flaked again',
        '#200: filed `tests/test_a.py::test_y`',
    ]
    assert client.comments == [
        (
            f'/repos/{REPO}/issues/101/comments',
            f'Failed and then passed again in {LINK}.\n\n'
            '```text\ntest leg 0: boom\ntest-all-extras leg 3: bang\n```\n\n'
            'The first run shuffled the tests with `-p randomly --randomly-seed=42` and the rerun kept their default '
            'order, so the failure can depend on the order.\n',
        )
    ]
    assert [issue['title'] for issue in client.created] == ['Flaky tests/test_a.py::test_y']


def test_a_test_that_fails_on_rerun_too_gets_an_issue_or_a_comment_that_says_so():
    client = FakeClient({'number': 101, 'title': 'Flaky tests/test_a.py::test_x'})
    flakes = {
        'tests/test_a.py::test_x': report.Flake({'shuffled leg 0': 'boom'}, fails_on_rerun=True),
        'tests/test_a.py::test_y': report.Flake({'shuffled leg 0': 'bang'}, fails_on_rerun=True),
    }

    assert report.reconcile(client, run(seed=42), flakes) == [
        '#101: noted that `tests/test_a.py::test_x` flaked again',
        '#200: filed `tests/test_a.py::test_y`',
    ]
    order = (
        'The first run shuffled the tests with `-p randomly --randomly-seed=42` and the rerun kept their default order.'
    )
    assert client.comments == [
        (
            f'/repos/{REPO}/issues/101/comments',
            f'Failed in {LINK}, and fails on rerun too.\n\n```text\nshuffled leg 0: boom\n```\n\n{order}\n',
        )
    ]
    assert client.created == [
        {
            'title': 'Flaky tests/test_a.py::test_y',
            'body': (
                f'`tests/test_a.py::test_y` failed in {LINK}, and fails on rerun too.\n\n'
                f'```text\nshuffled leg 0: bang\n```\n\n{order}\n\n'
                'Claim this issue before you fix the flake, as "Flake claims" in '
                '`.agents/skills/pushing-commits-to-the-repo/SKILL.md` says.\n'
            ),
            'labels': ['flake'],
            'number': 200,
        }
    ]


def test_new_issues_are_capped_per_run_and_the_rest_are_logged():
    client = FakeClient()
    flakes = {f'tests/test_a.py::test_{index}': report.Flake({'test leg 0': 'boom'}) for index in range(5)}

    log = report.reconcile(client, run(), flakes)

    assert [issue['title'] for issue in client.created] == [
        f'Flaky tests/test_a.py::test_{index}' for index in range(3)
    ]
    assert log[-1] == (
        '2 flaky tests wait for a later run (limit 3): `tests/test_a.py::test_3`, `tests/test_a.py::test_4`'
    )


def test_too_many_open_flake_issues_files_nothing():
    client = FakeClient(listed_all=False)

    with pytest.raises(RuntimeError, match='Too many open flake issues'):
        report.reconcile(client, run(), {'tests/test_a.py::test_x': report.Flake({'test leg 0': 'boom'})})
    assert client.created == []


def test_main_files_with_a_token_and_lists_without_one(
    tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    reports = tmp_path / 'reports'
    write_report(reports / 'junit-test-0', 'a.xml', ('tests/test_a.py::test_x', FAILED))
    write_report(reports / 'junit-test-0', 'b.xml', ('tests/test_a.py::test_x', PASSED))
    write_report(reports / 'junit-shuffled-0', 'a.xml', ('tests/test_a.py::test_y', FAILED))
    write_report(reports / 'junit-shuffled-0', 'b.xml', ('tests/test_a.py::test_y', FAILED))
    client = FakeClient()
    summary = tmp_path / 'summary.md'

    def fake_client(token: str) -> FakeClient:
        assert token == 'token'
        return client

    monkeypatch.setattr(report.attention, 'GitHubClient', fake_client)
    monkeypatch.setenv('GITHUB_WORKSPACE', str(root))
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)

    assert report.main(['--reports', str(reports), '--dry-run']) == 0
    assert capsys.readouterr().out == '`tests/test_a.py::test_x` flaked in test leg 0\n'
    assert report.main(['--reports', str(reports), '--dry-run', '--fails-on-rerun']) == 0
    assert capsys.readouterr().out == (
        '`tests/test_a.py::test_x` flaked in test leg 0\n`tests/test_a.py::test_y` failed its rerun too in shuffled leg 0\n'
    )
    assert report.main(['--reports', str(reports)]) == 1
    assert client.created == []

    for name, value in {
        'GITHUB_TOKEN': 'token',
        'GITHUB_REPOSITORY': REPO,
        'GITHUB_RUN_ID': '7',
        'GITHUB_STEP_SUMMARY': str(summary),
    }.items():
        monkeypatch.setenv(name, value)
    assert report.main(['--reports', str(reports), '--seed', '42']) == 0
    assert summary.read_text(encoding='utf-8') == '- #200: filed `tests/test_a.py::test_x`\n'
    assert '--randomly-seed=42' in str(client.created[0]['body'])
