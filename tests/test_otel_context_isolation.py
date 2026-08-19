"""Tests in a worker process share the main-thread OTel context, so an `attach` without a matching
`detach` in one test is visible to every later test in that worker. The `fresh_logfire` fixture
guards against this by resetting the context to its pre-test snapshot at teardown, so no leak
outlives the test that made it; these two tests pin that behavior.

They must run in order on the same worker (hence the shared `xdist_group`): the first deliberately
leaks a non-sampled active span, the second asserts spans are still recorded. Without that reset,
parent-based sampling marks the second test's spans unsampled and exporters silently drop them,
which surfaced as `context_subtree()` returning an empty tree in `tests/evals/test_otel.py`.
"""

from __future__ import annotations as _annotations

import pytest
from opentelemetry import context as otel_context, trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from .conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.otel._context_subtree import context_subtree
    from pydantic_evals.otel.span_tree import SpanTree

with try_import() as logfire_import_successful:
    import logfire
    from logfire.testing import CaptureLogfire

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'),
    pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed'),
    pytest.mark.xdist_group('otel_context_isolation'),
]


def test_leak_non_sampled_span_into_ambient_context():
    """Simulate a test that leaks: attach a non-sampled active span and never detach it."""
    ctx = trace.set_span_in_context(
        NonRecordingSpan(SpanContext(trace_id=0x1234, span_id=0x5678, is_remote=True, trace_flags=TraceFlags(0)))
    )
    otel_context.attach(ctx)
    assert not trace.get_current_span().get_span_context().trace_flags.sampled


@pytest.mark.anyio
async def test_spans_recorded_despite_leak_in_previous_test(capfire: CaptureLogfire):
    """The previous test's leaked non-sampled span must not make this test's spans invisible."""
    assert capfire
    with context_subtree() as tree:
        with logfire.span('root'):
            pass
    assert isinstance(tree, SpanTree)
    assert len(tree.roots) == 1
    assert tree.roots[0].name == 'root'
