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

An object in a reference cycle that outlives its test and is dropped later, such as one in a cache that a later test
clears, stays frozen, keeping whatever it holds, because collecting it sooner would finalize it in a test. One collection
after the last test finalizes it, where no test can be blamed, and reports it on stderr instead of failing. Left to
pytest's own end-of-session collection, it would fail the session with no test to point at. The freeze has no effect on
an object in no cycle, which is freed as it's dropped, even in a later test.

Frozen objects are invisible to `gc.get_referrers()` and `gc.get_objects()`. Hypothesis's `register_random()` notices, on
Python 3.14 and later: the frozen thread-local holding its own random leaves `gc.get_referrers()` finding nothing, so the
first derandomized Hypothesis test in a worker would fail on the warning this plugin ignores. That warning means something
references the random that `gc.get_referrers()` can't see; a random that nothing references still raises `ReferenceError`.
"""

from __future__ import annotations as _annotations

import gc
import sys
import warnings
from collections.abc import Generator, Iterator
from typing import cast

import pytest

PLUGIN_NAME = 'pydantic_ai_gc_isolation'


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line('filterwarnings', 'ignore:It looks like `register_random` was passed')


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol() -> Generator[None, object, object]:
    gc.freeze()
    return (yield)


@pytest.fixture(autouse=True)
def _finalize_this_tests_garbage(request: pytest.FixtureRequest) -> Iterator[None]:
    """Set up before the function-scoped fixtures of `tests/` and the test module, so it collects after they are torn down.

    Garbage from an autouse fixture of a plugin registered earlier, such as `cassetter`'s `cassette`, stays frozen.
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

    def record(unraisable: sys.UnraisableHookArgs) -> None:
        leaks.append(repr(unraisable.exc_value))  # pragma: lax no cover

    with pytest.MonkeyPatch.context() as patch, warnings.catch_warnings(record=True) as warned:
        patch.setattr(sys, 'unraisablehook', record)
        warnings.simplefilter('always')
        gc.unfreeze()
        gc.collect()
    for leak in (*leaks, *(warning.message for warning in warned)):  # pragma: lax no cover
        print(f'Leaked by a test, and dropped after it ended: {leak}', file=sys.stderr)
