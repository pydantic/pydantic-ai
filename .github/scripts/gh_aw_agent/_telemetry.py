"""In-process telemetry for the workflow agent.

gh-aw excludes `OTEL_EXPORTER_OTLP_ENDPOINT` and its headers from the agent
sandbox, so the SDK cannot post directly. The agent writes OTLP/JSON spans to a
path under the shared `/tmp/gh-aw` mount and a host post-step
(`forward_agent_traces.py`) posts them with the workflow's own credential. The
path and the file format are repository configuration, not engine surface.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib

logger = logging.getLogger(__name__)

EXPORT_PATH_ENV = 'PYDANTIC_AI_OTEL_EXPORT_PATH'


def _module_digest() -> str:
    """Name the code that actually ran, not the commit a workflow_run checked out."""
    digest = hashlib.sha256()
    for source in sorted(pathlib.Path(__file__).parent.glob('*.py')):
        digest.update(source.name.encode() + b'\0' + source.read_bytes())
    return digest.hexdigest()[:12]


def configure_observability() -> None:
    """Instrument pydantic-ai, httpx and MCP, and mirror spans to the export file."""
    export_path = os.environ.get(EXPORT_PATH_ENV)
    write_token = os.environ.get('LOGFIRE_WRITE_TOKEN') or os.environ.get('LOGFIRE_TOKEN')
    if not export_path and not write_token:
        return
    try:
        import logfire
        from opentelemetry.sdk.trace import SpanProcessor
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        processors: list[SpanProcessor] = []
        if export_path:
            from ._span_export import OtlpJsonFileExporter

            processors.append(BatchSpanProcessor(OtlpJsonFileExporter(pathlib.Path(export_path))))
        logfire.configure(
            service_name=os.environ.get('OTEL_SERVICE_NAME', 'gh-aw'),
            service_version=_module_digest(),
            send_to_logfire='if-token-present',
            console=False,
            token=write_token or None,
            additional_span_processors=processors,
        )
        logfire.instrument_pydantic_ai(include_content=True, include_binary_content=True)
        logfire.instrument_httpx(capture_headers=True, capture_response_body=True)
        logfire.instrument_mcp()
    except Exception as exc:  # pragma: no cover -- instrumentation is best effort
        logger.warning('observability disabled: %r', exc)


def service_version() -> str:
    return _module_digest()


__all__ = ['configure_observability', 'service_version']
