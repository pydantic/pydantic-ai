"""One Logfire configuration for the suite, and the spans the process actually exported.

`service_name`, `environment` and `service_version` are set on purpose: they are three of the hint
span's attributes, and they are what tells two services that each define an agent of the same name
apart in the Agent Control UI. A variable is derived from the agent's name alone, so without them
the span cannot say which deployment reported it.

`SpanCapture` takes the real `agent_control_config_hint` off the OpenTelemetry pipeline rather than
reconstructing it, which is the local half of the hint-span evidence -- and the half that stays
useful when the platform's own ingest is down.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta

import logfire
from logfire import AdvancedOptions, VariablesOptions
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from ._platform import Platform

SERVICE_NAME = 'pydantic-ai-harness-integration'
ENVIRONMENT = 'integration'
SERVICE_VERSION = '0.0.0-integration'

HINT_SPAN = 'agent_control_config_hint'
"""The span an agent describes itself to Logfire on, once per process per agent."""


@dataclass
class _Recorder(SpanExporter):
    """An exporter that keeps every span it is handed."""

    spans: list[ReadableSpan] = field(default_factory=list[ReadableSpan])

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Keep the batch and report success."""
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Nothing to release."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Nothing is buffered, so a flush is always complete."""
        return True


class SpanCapture:
    """Every span this process emits, as it is handed to the exporters."""

    def __init__(self) -> None:
        self._recorder = _Recorder()

    def new_processor(self) -> SimpleSpanProcessor:
        """A processor that records into this capture, for one `logfire.configure()` call.

        A fresh one each time rather than a single processor re-registered, because `configure()`
        replaces the tracer provider and shuts down the processors the old one held. Nothing here
        stops recording when that happens today -- the exporter's `shutdown` is a no-op and
        `SimpleSpanProcessor.on_end` has no shutdown guard -- but that is two implementation details
        to depend on for a suite whose evidence is the spans it collected.
        """
        return SimpleSpanProcessor(self._recorder)

    def clear(self) -> None:
        """Forget the spans recorded so far, so a test sees only its own."""
        self._recorder.spans.clear()

    def named(self, span_name: str) -> list[ReadableSpan]:
        """Every recorded span with this name."""
        return [span for span in self._recorder.spans if span.name == span_name]

    def hint_attributes(self) -> dict[str, object]:
        """The attributes of the first hint span, or an empty mapping if none was emitted."""
        spans = self.named(HINT_SPAN)
        return dict(spans[0].attributes or {}) if spans else {}

    def hint_baseline_json(self) -> str:
        """The code baseline JSON the hint span carried, as the variable's `example` would hold it."""
        baseline = self.hint_attributes().get('agent_control.baseline')
        assert isinstance(baseline, str), f'the hint span carried no baseline: {baseline!r}'
        return baseline


def configure(platform: Platform, capture: SpanCapture, *, base_url: str | None = None) -> None:
    """Configure Logfire against `platform`, recording its spans into `capture`.

    `send_to_logfire='if-token-present'` rather than the default: without
    `LOGFIRE_PLATFORM_WRITE_TOKEN` the spans stay local, every scenario that reads the hint span
    locally still works, and only the read-back tests skip.

    One capture is passed in rather than made here, because `logfire.configure()` replaces the span
    processors it was given: a test that reconfigures would otherwise detach the capture every later
    test reads its evidence from.

    Args:
        platform: The platform to resolve variables against and send spans to.
        capture: Where the exported spans are recorded.
        base_url: Override the platform origin, so a test can point the SDK at a dead port without
            reaching for a different platform.
    """
    logfire.configure(
        service_name=SERVICE_NAME,
        service_version=SERVICE_VERSION,
        environment=ENVIRONMENT,
        token=platform.write_token,
        api_key=platform.api_key,
        send_to_logfire='if-token-present',
        console=False,
        metrics=False,
        additional_span_processors=[capture.new_processor()],
        advanced=AdvancedOptions(base_url=base_url or platform.base_url),
        # Polling is the fallback; the platform pushes updates over SSE. 10s is the floor, and a
        # test that has just published calls `refresh_variables` rather than waiting for either.
        variables=VariablesOptions(polling_interval=timedelta(seconds=10)),
    )


@contextmanager
def configured_against(platform: Platform, capture: SpanCapture, *, base_url: str) -> Generator[None]:
    """Point Logfire somewhere else for the duration of the block, then put it back.

    `logfire.configure()` is process-global, so a test that needs a different destination has to
    reconfigure and restore rather than build a second instance. Restoring matters more than the
    reconfiguration: without it every later test in the process would resolve against the dead port
    this is used to reach.
    """
    configure(platform, capture, base_url=base_url)
    try:
        yield
    finally:
        configure(platform, capture)


def refresh_variables() -> None:
    """Make the SDK's variable cache current, so a test reads the value it just published.

    The remote provider polls and listens on SSE; forcing the fetch takes the race out of the test
    without changing what is being tested. `refresh(force=True)` blocks until the fetch completes.
    """
    logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider().refresh(force=True)


def flush() -> None:
    """Push every span to the platform, so a query can find it."""
    logfire.force_flush(15_000)
