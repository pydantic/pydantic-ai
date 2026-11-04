"""Tests for `tests/gc_isolation.py`.

Each runs a real pytest subprocess: the collector's state is per-process, and a session in this process would thaw
what the outer session froze. Each test's inner session checks one scenario, and the test asserts the outcome of every
inner test, so a leak blamed on the wrong test fails the assertion.
"""

from __future__ import annotations as _annotations

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
    """Run `source` as a test module in a pytest subprocess, under the filters the suite applies to leaks."""
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    pytester.makepyfile(gc_isolation=Path(gc_isolation.__file__).read_text(encoding='utf-8'))
    pytester.makeini(
        """
        [pytest]
        filterwarnings =
            error
            ignore:unclosed event loop:ResourceWarning
        """
    )

    def run(source: str, *args: str) -> pytest.RunResult:
        pytester.makepyfile(test_session=textwrap.dedent(source))
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
        pytest.param('Holder(socket.socket())', 'unclosed <socket', id='socket'),
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
        import asyncio, socket, tempfile

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


def test_a_test_can_still_collect_its_own_garbage(run_session: RunSession) -> None:
    result = run_session(
        """
        import gc, weakref

        class Node:
            pass

        def test_own_cycle_is_collected():
            node = Node()
            node.cycle = node
            ref = weakref.ref(node)
            del node
            gc.collect()
            assert ref() is None
        """,
        *WITH_PLUGIN,
    )
    assert outcomes(result) == ['PASSED test_own_cycle_is_collected']


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


OUTLIVES_ITS_TEST = (
    LEAK
    + """
held = []

def test_fills_a_cache():
    held.append(Leak())

def test_clears_it():
    held.clear()
"""
)


def test_a_sweep_collects_what_outlived_the_test_that_made_it(
    run_session: RunSession, pytester: pytest.Pytester
) -> None:
    pytester.makeconftest('import gc_isolation\ngc_isolation.SWEEP_INTERVAL = 2')
    result = run_session(OUTLIVES_ITS_TEST + '\ndef test_swept_before():\n    pass\n', *WITH_PLUGIN)
    assert outcomes(result) == ['ERROR test_swept_before', 'PASSED test_clears_it', 'PASSED test_fills_a_cache']


def test_the_end_of_session_collection_reports_what_no_sweep_reached(run_session: RunSession) -> None:
    result = run_session(OUTLIVES_ITS_TEST, *WITH_PLUGIN)
    assert 'leak finalized' in result.stderr.str()


def test_nothing_is_frozen_once_the_tests_are_done(run_session: RunSession, pytester: pytest.Pytester) -> None:
    # `gc.get_referrers()` can't see frozen objects, and Hypothesis's terminal summary imports it, whose `register_random()`
    # check then warns that a live object has no referrers.
    pytester.makeconftest(
        """
        import gc

        def pytest_terminal_summary(terminalreporter):
            terminalreporter.write_line(f'frozen objects at session end: {gc.get_freeze_count()}')
        """
    )
    result = run_session('def test_passes():\n    pass\n', *WITH_PLUGIN)
    assert 'frozen objects at session end: 0' in result.stdout.str()
