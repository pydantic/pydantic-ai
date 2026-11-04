"""Keeps garbage from earlier tests out of each test's collections.

Registered by `tests/conftest.py`; see "Garbage from earlier tests" in `tests/AGENTS.md`.

A leaked object held by a reference cycle (an unclosed event loop, socket or `zmq` context) is finalized whenever the
collector next runs, which is usually during an unrelated later test. Its `ResourceWarning` then lands in that test's
`pytest.warns` record, or is raised inside `__del__` under its `simplefilter('error')`, and that test fails.

Before each test, this plugin freezes every existing object, which takes microseconds whatever the heap's size and
limits the test's collections, including explicit `gc.collect()` calls, to objects created since. As the test tears
down, it collects those objects. That walks only what the test created, so it's cheap, and it finalizes the test's own
leaks in its own teardown under the suite's `filterwarnings`: a leak the suite doesn't ignore, and that's garbage by then,
fails the test that caused it.

An object that outlives its test and is dropped later, such as one in a cache that a later test clears, stays frozen. It's
collected once, after the last test, where no test can be blamed for it, and reported on stderr without failing anything.
Left to pytest's own end-of-session collection, it would fail the session with no test to point at.

The freeze has one visible side effect: `gc.get_referrers()` and `gc.get_objects()` don't see frozen objects. So it ends
with the test loop, before session-end hooks. Hypothesis's terminal summary is what imports it in a session that didn't
already, and that import's `register_random()` check warns if a live object appears to have no referrers.
"""

from __future__ import annotations as _annotations

import gc
import sys
import warnings
from collections.abc import Generator, Iterator
from typing import cast

import pytest

PLUGIN_NAME = 'pydantic_ai_gc_isolation'


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol() -> Generator[None, object, object]:
    gc.freeze()
    return (yield)


@pytest.fixture(autouse=True)
def _finalize_this_tests_garbage(request: pytest.FixtureRequest) -> Iterator[None]:
    """Set up before the test's other function-scoped fixtures, so it collects after they are torn down.

    pytest holds fixture values in `funcargs` until after teardown, which would keep a leaked one out of this collection.
    """
    yield
    cast(pytest.Function, request.node).funcargs.clear()  # pyright: ignore[reportUnknownMemberType]
    gc.collect()


@pytest.hookimpl(wrapper=True)
def pytest_runtestloop() -> Generator[None, object, object]:
    try:
        return (yield)
    finally:
        _collect_what_outlived_its_test()


def _collect_what_outlived_its_test() -> None:
    leaks: list[object] = []
    previous_hook = sys.unraisablehook
    sys.unraisablehook = lambda unraisable: leaks.append(repr(unraisable.exc_value))
    try:
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter('always')
            gc.unfreeze()
            gc.collect()
    finally:
        sys.unraisablehook = previous_hook
    for leak in (*leaks, *(warning.message for warning in warned)):  # pragma: lax no cover
        print(f'Leaked by a test, and dropped after it ended: {leak}', file=sys.stderr)
