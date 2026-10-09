"""The test-suite cost guards in `tests/cost_guards.py`, run against small in-process pytest sessions."""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import pytest

from .cost_guards import BUDGET_ENV_VAR, _popen_argv, launches_python  # pyright: ignore[reportPrivateUsage]

pytest_plugins = ['pytester']

RunPytest: TypeAlias = Callable[..., pytest.RunResult]

SPAWN = 'subprocess.run([sys.executable, "-I", "-S", "-c", "pass"], check=True)'


@pytest.fixture
def run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> RunPytest:
    """Run an in-process pytest session over `source` with only the cost guards loaded."""
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    monkeypatch.delenv(BUDGET_ENV_VAR, raising=False)
    monkeypatch.delenv('GITHUB_STEP_SUMMARY', raising=False)
    pytester.makeini(
        """
        [pytest]
        markers =
            subprocess: launches Python
            slow: exempt from the budget
        """
    )

    def run(source: str, *args: str) -> pytest.RunResult:
        pytester.makepyfile(test_guarded=f'import subprocess, sys, time\nimport pytest\n{textwrap.dedent(source)}')
        return pytester.runpytest_inprocess('-p', 'tests.cost_guards', '-p', 'no:cacheprovider', *args)

    return run


@pytest.mark.parametrize(
    ('argv', 'expected'),
    [
        ([sys.executable, '-c', 'pass'], True),
        (['python3.13', '-m', 'pydantic_clai2'], True),
        (['/usr/bin/python3'], True),
        (['pytest', '--version'], True),
        (['sh', '-c', 'python -c pass'], True),
        (['/bin/bash', '-c', 'env -i FOO=1 python -m x'], True),
        (['uv', 'run', '--frozen', 'python', '-V'], True),
        (['uv', 'run', 'ruff', 'check'], False),
        (['uv', 'sync'], False),
        (['git', 'status'], False),
        (['sh', '-c', 'git status'], False),
        (['sh', '-c', "echo 'unbalanced"], False),
        (['sh', 'script.sh'], False),
        ([], False),
    ],
)
def test_launches_python(argv: list[str], expected: bool):
    assert launches_python(argv) is expected


def test_popen_argv_mirrors_how_popen_builds_the_command():
    assert _popen_argv('python -c pass', None, True) == ['/bin/sh', '-c', 'python -c pass']
    assert _popen_argv(b'python', None, False) == ['python']
    assert _popen_argv(['fake', '-c', 'pass'], Path(sys.executable), False) == [sys.executable, '-c', 'pass']


def test_an_unmarked_python_launch_fails_with_guidance(run: RunPytest):
    result = run(
        f"""
        def test_spawns():
            {SPAWN}

        def test_runs_git():
            subprocess.run(['git', '--version'], check=True, capture_output=True)
        """
    )
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(
        [
            '*`test_guarded.py::test_spawns` launched a Python interpreter during call:',
            '*-I -S -c pass',
            '*Call the entry point in-process instead; see the "Test cost" section of tests/AGENTS.md.',
            "*mark it `@pytest.mark.subprocess(reason='...')`.",
        ]
    )


def test_every_launch_path_is_seen(run: RunPytest):
    result = run(
        """
        import asyncio
        import multiprocessing.util
        import os

        def test_asyncio():
            async def main():
                process = await asyncio.create_subprocess_exec(sys.executable, '-I', '-S', '-c', 'pass')
                await process.wait()
            asyncio.run(main())

        def test_shell():
            subprocess.run(f'{sys.executable} -I -S -c pass', shell=True, check=True)

        def test_thread():
            import threading

            thread = threading.Thread(target=subprocess.run, args=([sys.executable, '-I', '-S', '-c', 'pass'],))
            thread.start()
            thread.join()

        def test_multiprocessing_spawn():
            pid = multiprocessing.util.spawnv_passfds(sys.executable, [sys.executable, '-I', '-S', '-c', 'pass'], ())
            os.waitpid(pid, 0)
        """
    )
    result.assert_outcomes(failed=4)


def test_a_marked_launch_passes_and_a_marker_needs_a_reason(run: RunPytest):
    result = run(
        f"""
        @pytest.mark.subprocess(reason='checks interpreter startup')
        def test_marked():
            {SPAWN}

        @pytest.mark.subprocess
        def test_no_reason():
            pass

        @pytest.mark.slow(reason='  ')
        def test_blank_reason():
            pass
        """
    )
    result.assert_outcomes(passed=1, errors=2)
    result.stdout.fnmatch_lines(['*`@pytest.mark.subprocess` requires `reason=...`*'])
    result.stdout.fnmatch_lines(['*`@pytest.mark.slow` requires `reason=...`*'])


def test_a_launch_in_a_shared_fixture_needs_no_marker(run: RunPytest):
    result = run(
        f"""
        @pytest.fixture(scope='session')
        def server():
            {SPAWN}

        def test_first(server):
            pass

        def test_second(server):
            pass
        """
    )
    result.assert_outcomes(passed=2)


def test_an_unmarked_launch_in_an_already_failing_test_adds_a_section(run: RunPytest):
    result = run(
        f"""
        def test_fails():
            {SPAWN}
            assert False
        """
    )
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(['*cost guard*', '*launched a Python interpreter during call*'])


STALE_SOURCE = f"""
@pytest.mark.subprocess(reason='used to launch Python')
def test_stale():
    pass

@pytest.mark.subprocess(reason='launches Python for one parameter only')
@pytest.mark.parametrize('launch', [False, True])
def test_partly(launch):
    if launch:
        {SPAWN}

@pytest.mark.subprocess(reason='only launches Python on some platforms', conditional=True)
def test_conditional():
    pass
"""


def test_a_marker_that_covers_no_launch_fails_the_session(run: RunPytest):
    # Deselecting a test whose marker is never judged leaves the verdict on the others intact.
    result = run(STALE_SOURCE, '-k', 'not conditional')
    result.assert_outcomes(passed=3)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        [
            '*stale subprocess markers*',
            'test_guarded.py::test_stale: no test covered by its `@pytest.mark.subprocess` launched a Python '
            'interpreter; remove the stale marker',
        ]
    )
    assert 'test_partly' not in result.stdout.str().split('stale subprocess markers')[1]


MODULE_MARKED_SOURCE = """
pytestmark = pytest.mark.subprocess(reason='used to launch Python')

def test_one():
    pass

def test_two():
    pass
"""


def test_a_module_marker_that_covers_no_launch_fails_the_session(run: RunPytest):
    result = run(MODULE_MARKED_SOURCE)
    result.stdout.fnmatch_lines(['test_guarded.py: no test covered by its `@pytest.mark.subprocess`*'])


@pytest.mark.parametrize(
    ('source', 'args'),
    [
        pytest.param(MODULE_MARKED_SOURCE, ('-k', 'one'), id='deselected'),
        pytest.param(MODULE_MARKED_SOURCE, ('test_guarded.py::test_one',), id='node-id'),
        pytest.param(MODULE_MARKED_SOURCE + 'def test_fails():\n    assert False\n', ('-x',), id='stopped-early'),
    ],
)
def test_a_marker_is_not_judged_on_a_partial_run(run: RunPytest, source: str, args: tuple[str, ...]):
    result = run(source, *args)
    assert 'stale subprocess markers' not in result.stdout.str()


def test_a_failing_marked_test_is_not_judged(run: RunPytest):
    result = run(
        """
        @pytest.mark.subprocess(reason='launches Python before failing')
        def test_fails_first():
            assert False
        """
    )
    result.assert_outcomes(failed=1)
    assert 'stale subprocess markers' not in result.stdout.str()


@pytest.mark.subprocess(reason='checks the guard and the stale-marker verdict across pytest-xdist workers')
def test_guards_work_under_xdist(run: RunPytest, monkeypatch: pytest.MonkeyPatch):
    # Workers import `tests.cost_guards` from a fresh interpreter started in the session's temporary directory.
    monkeypatch.setenv('PYTHONPATH', str(Path(__file__).parent.parent))
    result = run(
        f"""
        @pytest.mark.subprocess(reason='used to launch Python')
        def test_stale():
            pass

        def test_unmarked():
            {SPAWN}
        """,
        '-p',
        'xdist',
        '-n',
        '2',
    )
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(
        [
            '*`test_guarded.py::test_unmarked` launched a Python interpreter during call:*',
            '*test_guarded.py::test_stale: no test covered by its `@pytest.mark.subprocess`*',
            '*worker startup and collection of 2 tests took *s (slowest worker collected in *s)',
        ]
    )


def test_the_time_budget_fails_slow_tests_unless_marked(run: RunPytest, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(BUDGET_ENV_VAR, '0.05')
    result = run(
        """
        @pytest.fixture(scope='module')
        def shared_server():
            time.sleep(0.1)

        def test_fast(shared_server):
            pass

        def test_slow():
            time.sleep(0.1)

        @pytest.mark.slow(reason='waits on purpose')
        def test_marked_slow():
            time.sleep(0.1)
        """
    )
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(
        [
            '*`test_guarded.py::test_slow` took 0.1*s in setup and call, over the 0.05s per-test budget '
            f'({BUDGET_ENV_VAR}).',
            '*inject or monkeypatch timeouts*',
            "*mark it `@pytest.mark.slow(reason='...')`.",
        ]
    )


def test_the_budget_message_names_excluded_shared_setup(run: RunPytest, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(BUDGET_ENV_VAR, '0.05')
    result = run(
        """
        @pytest.fixture(scope='module')
        def shared_server():
            time.sleep(0.01)

        def test_slow(shared_server):
            time.sleep(0.1)
        """
    )
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(['*not counting 0.0*s of shared fixture setup*'])


def test_a_run_that_collects_nothing_is_not_judged(run: RunPytest):
    result = run('')
    assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED


def test_collection_time_is_reported(run: RunPytest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    step_summary = tmp_path / 'summary.md'
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(step_summary))
    result = run('def test_nothing():\n    pass\n')
    result.stdout.fnmatch_lines(
        ['collection of 1 tests took *s', 'slowest test modules to collect: test_guarded.py (*s)']
    )
    assert step_summary.read_text().startswith('pytest collection of 1 tests took ')
    assert '- `test_guarded.py`: ' in step_summary.read_text()
