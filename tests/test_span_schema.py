"""The stale-exemption report of `tests/span_schema.py`, run against small in-process pytest sessions.

The report only judges a session that selected every test, which no CI shard is, so these sessions are the only place
it runs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import pytest

from . import span_schema

pytest_plugins = ['pytester']

RunPytest: TypeAlias = Callable[..., pytest.RunResult]


@pytest.fixture
def run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> RunPytest:
    """Run an in-process session of one test that exports no span, with only the span-schema check loaded."""
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    # The check keeps its session state in module globals, which the outer session is using too.
    monkeypatch.setattr(span_schema, '_used', set[tuple[str, str]]())
    monkeypatch.setattr(span_schema, '_walks', list[bool]())
    monkeypatch.setattr(span_schema, '_session_problems', list[str]())
    pytester.makeini('[pytest]\ntestpaths = .\n')
    pytester.makepyfile(test_no_spans='def test_no_spans():\n    pass\n')

    def run(*args: str) -> pytest.RunResult:
        return pytester.runpytest_inprocess('-p', 'tests.span_schema', '-p', 'no:cacheprovider', *args)

    return run


def test_a_session_that_selected_every_test_fails_on_an_exemption_no_span_used(run: RunPytest):
    result = run()
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        ['*span schema exemptions*', '`SPAN_SCHEMA_EXEMPTIONS` entries * no longer name a member: remove them.']
    )


def test_a_session_given_paths_leaves_unused_exemptions_alone(run: RunPytest):
    """A session given paths may not produce every span, so an exemption it didn't use isn't stale."""
    assert run('test_no_spans.py').ret == pytest.ExitCode.OK
