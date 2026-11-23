"""The leak guards in `tests/leak_guards.py`, run against small in-process pytest sessions."""

from __future__ import annotations as _annotations

import textwrap

import pytest

pytest_plugins = ['pytester']

WORKER_LOOP = """
import threading

from pydantic_ai._utils import get_event_loop


def on_worker_thread(close: bool) -> None:
    def run() -> None:
        loop = get_event_loop()
        if close:
            loop.close()

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
"""


def run_session(pytester: pytest.Pytester, source: str) -> pytest.RunResult:
    pytester.makepyfile(test_session=WORKER_LOOP + textwrap.dedent(source))
    # The session inherits this suite's warning filters, under which a collected loop's `ResourceWarning` fails the
    # test too: ignored here, so each outcome is the guard's own.
    return pytester.runpytest(
        '-p', 'tests.leak_guards', '-p', 'no:cacheprovider', '-p', 'no:pretty', '-W', 'ignore::ResourceWarning'
    )


def test_a_loop_left_open_on_a_worker_thread_fails_the_test_that_created_it(pytester: pytest.Pytester):
    result = run_session(
        pytester,
        """
        def test_leaks():
            on_worker_thread(close=False)

        def test_closes():
            on_worker_thread(close=True)

        def test_leaks_until_collected():
            import gc

            on_worker_thread(close=False)
            gc.collect()
        """,
    )
    result.assert_outcomes(passed=3, errors=2)
    created_at = r'\s+pydantic_ai_slim/pydantic_ai/_utils\.py:\d+ in get_event_loop'
    result.stdout.re_match_lines(
        [
            r'.*::test_leaks left 1 event loop\(s\) open, created at:',
            created_at,
            r'.*::test_leaks_until_collected left 1 event loop\(s\) open, created at:',
            created_at,
        ]
    )


def test_a_loop_on_the_main_thread_or_in_a_shared_fixture_is_left_to_its_owner(pytester: pytest.Pytester):
    result = run_session(
        pytester,
        """
        import pytest

        @pytest.fixture(scope='module')
        def module_loop():
            on_worker_thread(close=False)

        def test_main_thread():
            get_event_loop()

        def test_shared_fixture(module_loop):
            pass
        """,
    )
    result.assert_outcomes(passed=2)
