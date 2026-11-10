"""Synthetic patterns for unit tests."""

from typing import Any


def foo() -> Any:  # Any return
    # see line 42 for details — L99 is also bad
    warnings.warn('deprecated')  # type: ignore
    return 1


def test_something():
    pytest.importorskip('nonexistent_pkg')
    assert snapshot() == 1
