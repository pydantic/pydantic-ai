"""`tests/span_schema.py`, run against small in-process pytest sessions with only its check loaded.

Its failures are the part that matters, and a passing suite never reaches them: no span in it declares a property it
leaves out unexempted, and the stale-exemption report only judges a session that selected every test, which no CI shard
is.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import TypeAlias

import pytest

from . import span_schema

pytest_plugins = ['pytester']

RunPytest: TypeAlias = Callable[..., pytest.RunResult]

_SPANS = """
import json

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def test_span():
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    schema = json.dumps({{'type': 'object', 'properties': {{'{name}': {{}}}}}})
    attributes = {{'gen_ai.operation.name': '{operation}', 'logfire.json_schema': schema}}
    with provider.get_tracer('pydantic-ai').start_as_current_span('{operation} model', attributes=attributes):
        pass
"""
"""A test that exports one successful Pydantic AI span declaring a property it doesn't set."""

needs_sdk = pytest.mark.skipif(
    importlib.util.find_spec('opentelemetry.sdk') is None, reason='opentelemetry-sdk not installed'
)


@pytest.fixture
def run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> RunPytest:
    """Run an in-process session with only the span-schema check loaded."""
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    # The check keeps its session state in module globals, which the outer session is using too.
    monkeypatch.setattr(span_schema, '_used', set[tuple[str, str]]())
    monkeypatch.setattr(span_schema, '_walks', list[bool]())
    monkeypatch.setattr(span_schema, '_session_problems', list[str]())
    pytester.makeini('[pytest]\ntestpaths = .\n')

    def run(*args: str, **files: str) -> pytest.RunResult:
        pytester.makepyfile(**files)
        return pytester.runpytest_inprocess('-p', 'tests.span_schema', '-p', 'no:cacheprovider', *args)

    return run


@needs_sdk
def test_a_declared_property_the_span_leaves_out_fails_its_test(run: RunPytest):
    result = run(test_spans=_SPANS.format(operation='chat', name='surprise'))
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        [
            '*span `chat model` declares `surprise` in its `logfire.json_schema` but has no such attribute',
            # `[[]` is a literal `[` in a glob.
            "*classify [[]('chat', 'surprise')] in `SPAN_SCHEMA_EXEMPTIONS`*",
        ]
    )


@needs_sdk
def test_a_set_on_success_property_must_be_set_on_a_span_that_succeeded(run: RunPytest):
    """`('embeddings', 'embeddings')` is `SetOnSuccess`: exempt on a failed span, so not on this one."""
    # Without the entry this would pass as an unexempted property, testing nothing about `SetOnSuccess`.
    assert isinstance(span_schema.SPAN_SCHEMA_EXEMPTIONS[('embeddings', 'embeddings')], span_schema.SetOnSuccess)
    result = run(test_spans=_SPANS.format(operation='embeddings', name='embeddings'))
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(['*span `embeddings model` declares `embeddings` in its `logfire.json_schema`*'])


def test_a_session_that_selected_every_test_fails_on_an_exemption_no_span_used(run: RunPytest):
    result = run(test_no_spans='def test_no_spans():\n    pass\n')
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        ['*span schema exemptions*', '`SPAN_SCHEMA_EXEMPTIONS` entries * no longer name a member: remove them.']
    )


def test_a_session_given_paths_leaves_unused_exemptions_alone(run: RunPytest):
    """A session given paths may not produce every span, so an exemption it didn't use isn't stale."""
    assert run('test_no_spans.py', test_no_spans='def test_no_spans():\n    pass\n').ret == pytest.ExitCode.OK


def test_a_failed_session_leaves_unused_exemptions_alone(run: RunPytest):
    """A failed test may have stopped before exporting the span an exemption covers."""
    result = run(test_fails='def test_fails():\n    assert False\n')
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.no_fnmatch_line('*no longer name a member*')
