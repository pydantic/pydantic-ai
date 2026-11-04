"""Tests for `tests/gc_isolation.py`.

Each runs a real pytest subprocess: the collector's state is per-process, and a session in this process would thaw
what the outer session froze. Each test's inner session checks one scenario, and the test asserts the outcome of every
inner test, so a leak blamed on the wrong test fails the assertion.
"""

from __future__ import annotations as _annotations

import os
import re
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias

import pytest

from . import gc_isolation

pytest_plugins = ['pytester']

pytestmark = pytest.mark.subprocess(reason='garbage collector state is per-process, so each session needs its own')

RunSession: TypeAlias = Callable[..., pytest.RunResult]

WITH_PLUGIN = ('-p', 'gc_isolation')

LEAK = """
class Leak:
    '''Finalized only by the collector, and fails loudly whichever warning filters are active.'''

    def __init__(self):
        self.cycle = self

    def __del__(self):
        raise RuntimeError('leak finalized')
"""


@pytest.fixture
def run_session(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> RunSession:
    """Run `source` as a test module in a pytest subprocess, under the filters the suite applies to leaks.

    Only explicit collections run, so nothing depends on when allocations happen to trigger an automatic one.
    """
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    for name in list(os.environ):
        if name.startswith('COVERAGE_'):
            monkeypatch.delenv(name)
    pytester.makepyfile(gc_isolation=Path(gc_isolation.__file__).read_text(encoding='utf-8'))
    pytester.makeini(
        """
        [pytest]
        filterwarnings =
            error
            ignore:unclosed <socket:ResourceWarning
            ignore:unclosed event loop:ResourceWarning
        """
    )

    def run(source: str, *args: str) -> pytest.RunResult:
        pytester.makepyfile(test_session='import gc\ngc.disable()\n' + textwrap.dedent(source))
        return pytester.runpytest_subprocess('-rA', *args)

    return run


def outcomes(result: pytest.RunResult) -> list[str]:
    """Every entry in the `-rA` summary, such as `ERROR test_leaks`: a test erroring in teardown also reports `PASSED`."""
    found = (re.match(r'(PASSED|FAILED|ERROR) \S+::(\w+)', line) for line in result.outlines)
    return sorted(f'{match[1]} {match[2]}' for match in found if match)


@pytest.mark.parametrize(
    'body',
    [
        pytest.param(
            """
            with pytest.warns(UserWarning) as record:
                gc.collect()
                warnings.warn('expected', UserWarning)
            assert [str(w.message) for w in record] == ['expected']
            """,
            id='recording-warnings',
        ),
        pytest.param(
            """
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                gc.collect()
            """,
            id='rejecting-warnings',
        ),
    ],
)
@pytest.mark.parametrize(
    ('args', 'later_test_outcome'),
    [
        pytest.param((), 'FAILED', id='without-plugin'),
        pytest.param(WITH_PLUGIN, 'PASSED', id='with-plugin'),
    ],
)
def test_an_ignored_leak_reaches_a_later_tests_warning_scope_only_without_the_plugin(
    run_session: RunSession, body: str, args: tuple[str, ...], later_test_outcome: str
) -> None:
    source = f"""
    import asyncio, gc, warnings
    import pytest

    def test_leaks():
        asyncio.new_event_loop()

    def test_later():
{textwrap.indent(textwrap.dedent(body), ' ' * 8)}
    """
    assert outcomes(run_session(source, *args)) == [f'{later_test_outcome} test_later', 'PASSED test_leaks']


@pytest.mark.parametrize(
    ('leak', 'error'),
    [
        pytest.param('Holder(tempfile.TemporaryFile())', 'unclosed file', id='file'),
        # The collector can finalize a leaked loop's self-pipe before the loop, whose `__del__` then raises this whatever
        # the warning filters are. Closing the socket here picks that finalization order.
        pytest.param(
            'asyncio.new_event_loop()._ssock.close()', 'Invalid file descriptor: -1', id='loop-with-finalized-self-pipe'
        ),
    ],
)
def test_a_real_resource_leak_fails_only_the_test_that_caused_it(
    run_session: RunSession, leak: str, error: str
) -> None:
    result = run_session(
        f"""
        import asyncio, tempfile

        class Holder:
            def __init__(self, resource):
                self.resource = resource
                self.cycle = self

        def test_innocent_before():
            pass

        def test_leaks():
            {leak}

        def test_innocent_after():
            pass
        """,
        *WITH_PLUGIN,
    )
    assert outcomes(result) == [
        'ERROR test_leaks',
        'PASSED test_innocent_after',
        'PASSED test_innocent_before',
        'PASSED test_leaks',
    ]
    assert error in result.stdout.str()


def test_a_leaked_fixture_value_fails_the_test_that_used_it(run_session: RunSession) -> None:
    result = run_session(
        LEAK
        + """
import pytest

@pytest.fixture
def leaked():
    yield Leak()

def test_leaks(leaked):
    pass

def test_innocent_after():
    pass
""",
        *WITH_PLUGIN,
    )
    assert outcomes(result) == ['ERROR test_leaks', 'PASSED test_innocent_after', 'PASSED test_leaks']


def test_the_suite_runs_with_the_plugin(request: pytest.FixtureRequest) -> None:
    assert request.config.pluginmanager.has_plugin(gc_isolation.PLUGIN_NAME)


def test_the_first_test_is_isolated_from_garbage_made_during_collection(run_session: RunSession) -> None:
    result = run_session(
        LEAK
        + """
import gc

Leak()

def test_first():
    gc.collect()
""",
        *WITH_PLUGIN,
    )
    assert outcomes(result) == ['PASSED test_first']


@pytest.mark.parametrize(
    ('args', 'collecting_test_outcome'),
    [
        pytest.param((), 'FAILED', id='without-plugin'),
        pytest.param(WITH_PLUGIN, 'PASSED', id='with-plugin'),
    ],
)
def test_a_collection_only_reaches_what_the_running_test_created(
    run_session: RunSession, args: tuple[str, ...], collecting_test_outcome: str
) -> None:
    source = (
        LEAK
        + """
import gc

held = []

def test_fills_a_cache():
    held.append(Leak())

def test_collects():
    held.clear()
    gc.collect()
"""
    )
    assert outcomes(run_session(source, *args)) == [
        f'{collecting_test_outcome} test_collects',
        'PASSED test_fills_a_cache',
    ]


@pytest.mark.parametrize(
    ('held', 'reported'),
    [
        pytest.param('Leak()', "RuntimeError('leak finalized')", id='raising-in-del'),
        pytest.param('[socket.socket()]', 'unclosed <socket', id='resource-warning'),
    ],
)
def test_what_outlives_its_test_is_reported_after_the_last_test_and_fails_nothing(
    run_session: RunSession, held: str, reported: str
) -> None:
    result = run_session(
        LEAK
        + f"""
import socket

held = []

def test_fills_a_cache():
    cycle = [{held}]
    cycle.append(cycle)
    held.append(cycle)

def test_clears_it():
    held.clear()
""",
        *WITH_PLUGIN,
    )
    assert outcomes(result) == ['PASSED test_clears_it', 'PASSED test_fills_a_cache']
    assert result.ret == pytest.ExitCode.OK
    result.stderr.fnmatch_lines([f'Leaked by a test, and dropped after it ended: {reported}*'])


def test_what_outlives_its_test_is_reported_when_the_run_stops_early(run_session: RunSession) -> None:
    result = run_session(
        LEAK
        + """
held = []

def test_fills_a_cache():
    held.append(Leak())

def test_stops_the_run():
    held.clear()
    assert False
""",
        '-x',
        *WITH_PLUGIN,
    )
    assert outcomes(result) == ['FAILED test_stops_the_run', 'PASSED test_fills_a_cache']
    result.stderr.fnmatch_lines(["Leaked by a test, and dropped after it ended: RuntimeError('leak finalized')*"])


def test_a_hypothesis_test_can_register_its_random_with_its_holder_frozen(run_session: RunSession) -> None:
    result = run_session(
        """
        from hypothesis import given, settings, strategies as st

        # Only a derandomized run registers Hypothesis's random, and so reaches the check in `register_random()`.
        @settings(derandomize=True)
        @given(st.none())
        def test_property(value):
            pass
        """,
        *WITH_PLUGIN,
    )
    assert outcomes(result) == ['PASSED test_property']


def test_nothing_stays_frozen_once_the_tests_are_done(run_session: RunSession, pytester: pytest.Pytester) -> None:
    # Some Python versions start with frozen objects of their own, which unfreezing leaves frozen.
    pytester.makeconftest(
        """
        import gc

        frozen_at_start = gc.get_freeze_count()

        def pytest_terminal_summary(terminalreporter):
            terminalreporter.write_line(f'frozen since the start: {gc.get_freeze_count() - frozen_at_start}')
        """
    )
    result = run_session('def test_passes():\n    pass\n', *WITH_PLUGIN)
    assert 'frozen since the start: 0' in result.stdout.str()
