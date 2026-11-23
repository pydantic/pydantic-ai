from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import never_ran

from tests.manifests import Exempt

_RAN = ''
_SKIPPED = '<skipped type="pytest.skip" message="needs a thing">skip</skipped>'
_XFAIL = '<skipped type="pytest.xfail" message="known bug"/>'


def _report(directory: Path, name: str, *cases: tuple[str, str, str]) -> None:
    """A JUnit report, as pytest writes one, of `(test file, class path and test name, result)` cases."""
    body = ''.join(
        f'<testcase classname="{".".join([file.removesuffix(".py").replace("/", "."), *test.split("::")[:-1]])}" '
        f'name="{test.split("::")[-1]}" file="tests/base.py">{result}</testcase>'
        for file, test, result in cases
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(f'<testsuites><testsuite>{body}</testsuite></testsuites>', encoding='utf-8')


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for file in ('tests/test_a.py', 'tests/test_b.py', 'tests/gated/test_c.py', 'scripts/test_d.py'):
        (tmp_path / file).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / file).touch()
    monkeypatch.setattr(never_ran, 'ROOT', tmp_path)
    monkeypatch.setattr(never_ran, 'NEVER_RAN_EXEMPT', {})
    monkeypatch.setattr(never_ran, 'PATH_GATED_JOBS', {'gated-job': ('tests/gated/',)})
    monkeypatch.setenv('NEEDS', json.dumps({'gated-job': {'result': 'success'}}))
    return tmp_path


def test_a_test_every_cell_skips_fails(root: Path, capsys: pytest.CaptureFixture[str]):
    reports = root / 'reports'
    _report(
        reports / 'junit-test-0',
        'a.xml',
        ('tests/test_a.py', 'test_one', _SKIPPED),
        ('tests/test_a.py', 'test_two', _SKIPPED),
    )
    _report(
        reports / 'junit-test-1',
        'b.xml',
        ('tests/test_a.py', 'test_one', _RAN),
        ('tests/test_a.py', 'test_two', _SKIPPED),
    )

    assert never_ran.main([str(reports)]) == 1
    out = capsys.readouterr().out
    assert 'tests/test_a.py::test_two' in out
    assert 'test_one' not in out


def test_a_test_a_class_inherits_is_named_by_the_module_that_collected_it(
    root: Path, capsys: pytest.CaptureFixture[str]
):
    reports = root / 'reports'
    _report(reports, 'a.xml', ('tests/gated/test_c.py', 'TestOuter::TestInner::test_one[x]', _SKIPPED))
    (root / 'tests/base.py').touch()

    assert never_ran.main([str(reports)]) == 1
    assert 'tests/gated/test_c.py::TestOuter::TestInner::test_one[x]' in capsys.readouterr().out


def test_an_expected_failure_ran_and_files_outside_tests_are_not_judged(root: Path):
    reports = root / 'reports'
    _report(reports, 'a.xml', ('tests/test_a.py', 'test_one', _XFAIL), ('scripts/test_d.py', 'test_three', _SKIPPED))

    assert never_ran.main([str(reports)]) == 0


@pytest.mark.parametrize(
    'key', ['tests/test_a.py::test_one[x]', 'tests/test_a.py::test_one', 'tests/test_a.py', 'tests/']
)
def test_an_entry_exempts_a_test_its_file_or_a_directory_holding_it(
    root: Path, monkeypatch: pytest.MonkeyPatch, key: str
):
    monkeypatch.setattr(never_ran, 'NEVER_RAN_EXEMPT', {key: Exempt('no cell installs the thing')})
    reports = root / 'reports'
    _report(reports, 'a.xml', ('tests/test_a.py', 'test_one[x]', _SKIPPED))

    assert never_ran.main([str(reports)]) == 0


def test_an_entry_that_names_no_skipped_test_is_stale(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(never_ran, 'NEVER_RAN_EXEMPT', {'tests/test_b.py': Exempt('no cell installs the thing')})
    reports = root / 'reports'
    _report(reports, 'a.xml', ('tests/test_b.py', 'test_one', _RAN))

    assert never_ran.main([str(reports)]) == 1
    assert 'no longer name a member' in capsys.readouterr().out


def test_an_entry_needs_a_reason(root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(never_ran, 'NEVER_RAN_EXEMPT', {'tests/test_a.py': Exempt(' ')})
    reports = root / 'reports'
    _report(reports, 'a.xml', ('tests/test_a.py', 'test_one', _SKIPPED))

    assert never_ran.main([str(reports)]) == 1
    assert 'have no reason' in capsys.readouterr().out


@pytest.mark.parametrize(('result', 'exit_code'), [('skipped', 0), ('success', 1)])
def test_a_path_gated_jobs_tests_are_exempt_only_when_it_was_skipped(
    root: Path, monkeypatch: pytest.MonkeyPatch, result: str, exit_code: int
):
    monkeypatch.setenv('NEEDS', json.dumps({'gated-job': {'result': result}}))
    reports = root / 'reports'
    _report(reports, 'a.xml', ('tests/gated/test_c.py', 'test_one', _SKIPPED))

    assert never_ran.main([str(reports)]) == exit_code


def test_no_reports_fails(root: Path):
    (root / 'reports').mkdir()

    assert never_ran.main([str(root / 'reports')]) == 1
