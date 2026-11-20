"""Fail a test whose spans declare a `logfire.json_schema` property they don't set.

A span's `logfire.json_schema` attribute tells Logfire how to render its other attributes, so each property it
declares is one the span promises to carry. A declared property with no attribute is usually telemetry the
instrumentation meant to record and lost: the embeddings span declared `embeddings` but wrote it into an attribute
dict the span had already read, so it was never exported (https://github.com/pydantic/pydantic-ai/pull/8295).

Every span Pydantic AI's tracer (scope `pydantic-ai`) emits and a test exports, through logfire's `capfire` fixture or
a raw `InMemorySpanExporter` the test creates, is checked when the test's teardown ends, including spans a test drops
with `capfire.exporter.clear()`. Spans other libraries emit follow their own schema conventions and are left out.

A property a span leaves out on purpose has an entry in `SPAN_SCHEMA_EXEMPTIONS`, keyed by the span's
`gen_ai.operation.name` and the property. A span that lost the attributes it started with, as one does under a sampler
that returns none, is keyed by the first word of its name, which is its operation. An `Exempt` entry covers every
span; a `SetOnSuccess` entry covers only spans that end in an error, so the check still sees the property go missing
from a span that succeeded. An entry no span used fails as stale, but only on a session that selected every test and
passed: a CI shard or a targeted run doesn't produce every span, and a failed test may not have exported its span.
"""

from __future__ import annotations

import json
import weakref
from collections.abc import Callable, Generator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar

import pytest

from .manifests import Exempt, assert_classified

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from xdist.workermanage import WorkerController

PLUGIN_NAME = 'pydantic_ai_span_schema'


@dataclass(frozen=True)
class SetOnSuccess:
    """A property a span sets from its operation's result, so only a span that ends in an error leaves it out."""

    reason: str


SPAN_SCHEMA_EXEMPTIONS: dict[tuple[str, str], Exempt | SetOnSuccess] = {
    ('invoke_agent', 'final_result'): Exempt(
        '`Instrumentation` declares it on every agent run span, and sets it only when the run ends with output and '
        '`include_content` is on.'
    ),
    ('execute_tool', 'gen_ai.tool.call.result'): Exempt(
        'Set when the call returns, is skipped with a replacement, or, with `include_content`, retries or fails into '
        'feedback the model sees. A deferred call has no result yet and records the deferral instead; a call that '
        'raises anything else or is cancelled records the exception; an output function records its retry prompt on '
        'the request and agent spans.'
    ),
    ('execute_tool', 'tool_response'): Exempt(
        '`gen_ai.tool.call.result` under its instrumentation version 2 name, left out in the same cases.'
    ),
    ('chat', 'model_request_parameters'): Exempt(
        'Declared whenever `include_model_request_parameters` is on, and left out when there is nothing to record: a '
        'realtime session started without `model_request_parameters`, or parameters `include_content=False` redacts '
        'to nothing.'
    ),
    ('embeddings', 'embedding_settings'): Exempt(
        'Declared on every span, and set only when the call passes settings other than `extra_headers` and '
        '`extra_body`.'
    ),
    ('embeddings', 'embeddings'): SetOnSuccess('Set from the result, which a failed call does not have.'),
    ('image_generation', 'image_count'): Exempt(
        'Set from the result when generation finishes, so a failed generation leaves it out, as does the span '
        '`test_instrumentation_records_complete_response_metrics` opens and never finishes.'
    ),
    ('decide', 'pydantic_ai.decision.answers'): Exempt(
        'Set from the response, which a failed request does not have, and left out when the answers cannot be '
        'serialized.'
    ),
    ('decide', 'pydantic_ai.decision.confidence'): Exempt(
        'Declared when the request asks field questions, and set only when the step uses one of their answers: a '
        'picked route handed off before any fill uses none, and a failed request has none.'
    ),
}

_TABLE_NAME = 'SPAN_SCHEMA_EXEMPTIONS'
_WORKER_OUTPUT_KEY = 'span_schema'
_PATCH_KEY = pytest.StashKey[pytest.MonkeyPatch]()

ExporterT = TypeVar('ExporterT', bound='SpanExporter')
InitP = ParamSpec('InitP')

_exported: list[ReadableSpan] = []
"""The spans exported since the running test's setup began, to an exporter created since then."""
_fresh: weakref.WeakSet[SpanExporter] = weakref.WeakSet()
"""The exporters created since the running test's setup began.

A span exported to an older one, such as the exporter an earlier test's `capfire` left on logfire's global provider, is
not checked, so whether a test's spans are checked never depends on which tests ran before it on the same worker.
"""
_used: set[tuple[str, str]] = set()
"""The exemptions a span in this session needed."""
_walks: list[bool] = []
"""Whether each collection, or each `pytest-xdist` worker's, selected every test."""
_session_problems: list[str] = []


def _recording(
    export: Callable[[ExporterT, Sequence[ReadableSpan]], SpanExportResult],
) -> Callable[[ExporterT, Sequence[ReadableSpan]], SpanExportResult]:
    def export_and_record(exporter: ExporterT, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if exporter in _fresh:
            _exported.extend(spans)
        return export(exporter, spans)

    return export_and_record


def _tracking(
    init: Callable[Concatenate[ExporterT, InitP], None],
) -> Callable[Concatenate[ExporterT, InitP], None]:
    def init_and_track(exporter: ExporterT, *args: InitP.args, **kwargs: InitP.kwargs) -> None:
        init(exporter, *args, **kwargs)
        _fresh.add(exporter)

    return init_and_track


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_PATCH_KEY] = patch = pytest.MonkeyPatch()
    # Both need `opentelemetry-sdk`, which not every install shape the suite runs in has.
    with suppress(ImportError):
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        patch.setattr(InMemorySpanExporter, '__init__', _tracking(InMemorySpanExporter.__init__))
        patch.setattr(InMemorySpanExporter, 'export', _recording(InMemorySpanExporter.export))
    with suppress(ImportError):
        from logfire.testing import TestExporter

        patch.setattr(TestExporter, '__init__', _tracking(TestExporter.__init__))
        patch.setattr(TestExporter, 'export', _recording(TestExporter.export))


def pytest_unconfigure(config: pytest.Config) -> None:
    config.stash[_PATCH_KEY].undo()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup() -> None:
    _exported.clear()
    _fresh.clear()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown() -> Generator[None, None, None]:
    yield
    missing: dict[tuple[str, str], str] = {}
    for span in _exported:
        attributes = span.attributes or {}
        schema = attributes.get('logfire.json_schema')
        if (
            span.instrumentation_scope is None
            or span.instrumentation_scope.name != 'pydantic-ai'
            or not isinstance(schema, str)
            # Logfire exports a pending copy of each span when it starts, before the span sets what it declares.
            or attributes.get('logfire.span_type') == 'pending_span'
        ):
            continue
        operation = str(attributes.get('gen_ai.operation.name', span.name.partition(' ')[0]))
        properties: dict[str, object] = json.loads(schema).get('properties', {})
        for name in properties.keys() - attributes.keys():
            entry = SPAN_SCHEMA_EXEMPTIONS.get(key := (operation, name))
            if entry is not None and (isinstance(entry, Exempt) or not span.status.is_ok):
                _used.add(key)
            else:
                missing[key] = span.name
    _exported.clear()
    if missing:
        lines = [
            f'span `{span_name}` declares `{name}` in its `logfire.json_schema` but has no such attribute'
            for (_, name), span_name in sorted(missing.items())
        ]
        lines.append(
            f'Set the attribute or stop declaring it. If the span leaves it out on purpose, classify {sorted(missing)} '
            f'in `{_TABLE_NAME}` in `tests/span_schema.py` with the reason: a `SetOnSuccess` entry covers only spans '
            'that end in an error.'
        )
        pytest.fail('\n'.join(lines), pytrace=False)


# --- Stale exemptions ----------------------------------------------------------------------------------------------
#
# A session selected every test when it was given no paths and ignored, deselected and narrowed nothing. Under
# `pytest-xdist` each worker collects every test, judges its own collection, and hands the controller the exemptions
# its tests used.


def pytest_deselected(items: Sequence[pytest.Item]) -> None:
    # pytest calls this only when it deselected something, so the walk is incomplete.
    _walks.append(False)


def pytest_collection_finish(session: pytest.Session) -> None:
    config = session.config
    narrowing_options = ('ignore', 'ignore_glob', 'lf', 'collectonly', 'setuponly', 'setupplan')
    _walks.append(
        config.args_source is pytest.Config.ArgsSource.TESTPATHS
        and not any(config.getoption(name, None) for name in narrowing_options)
    )


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: WorkerController, error: object | None) -> None:
    complete, used = getattr(node, 'workeroutput', {}).get(_WORKER_OUTPUT_KEY, (False, []))
    _walks.append(complete and error is None)
    _used.update((operation, name) for operation, name in used)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    if hasattr(config, 'workerinput'):
        workeroutput: dict[str, object] = getattr(config, 'workeroutput')  # set by `pytest-xdist` on workers
        workeroutput[_WORKER_OUTPUT_KEY] = (all(_walks), sorted(_used))
        return
    # A failed test may have stopped before exporting the span an entry covers, which would report the entry stale.
    ran_every_test = exitstatus == pytest.ExitCode.OK and not (session.shouldstop or session.shouldfail)
    try:
        assert_classified(
            _used,
            SPAN_SCHEMA_EXEMPTIONS,
            complete_walk=ran_every_test and bool(_walks) and all(_walks),
            table_name=_TABLE_NAME,
        )
    except AssertionError as error:
        _session_problems.append(str(error))
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if _session_problems:
        terminalreporter.section('span schema exemptions', red=True)
        for problem in _session_problems:
            terminalreporter.write_line(problem)
