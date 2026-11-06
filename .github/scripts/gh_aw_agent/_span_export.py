"""OTLP/JSON span file exporter for the host-side forwarder."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


class OtlpJsonFileExporter(SpanExporter):
    """Append one OTLP/JSON span per line to a file the host can read after the run."""

    def __init__(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open('a', encoding='utf-8', buffering=1)
        os.chmod(path, 0o644)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            for span in spans:
                self._handle.write(span.to_json() + '\n')
        except Exception:
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        self._handle.flush()
        return True

    def shutdown(self) -> None:
        self._handle.close()
