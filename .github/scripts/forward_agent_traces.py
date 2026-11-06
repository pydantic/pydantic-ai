"""Forward the workflow agent's OTLP/JSON span file to the configured endpoint.

gh-aw excludes `OTEL_EXPORTER_OTLP_ENDPOINT` and its headers from the agent
sandbox, so the agent cannot export directly. The composition writes one
OTLP/JSON span per line to `PYDANTIC_AI_OTEL_EXPORT_PATH` (under the shared
`/tmp/gh-aw` mount); this host step posts them to the endpoint the workflow's
own `observability.otlp` block configures.

Telemetry failures never fail the workflow: a missing endpoint or file is a
no-op, and delivery errors are reported on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence

DEFAULT_EXPORT_PATH = '/tmp/gh-aw/agent/otel-spans.jsonl'
BATCH_SIZE = 200

_KIND = {
    'SpanKind.INTERNAL': 1,
    'SpanKind.SERVER': 2,
    'SpanKind.CLIENT': 3,
    'SpanKind.PRODUCER': 4,
    'SpanKind.CONSUMER': 5,
}
_STATUS = {'UNSET': 0, 'OK': 1, 'ERROR': 2}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}  # pyright: ignore[reportReturnType]


def _attribute_value(value: object) -> dict[str, object]:
    if isinstance(value, bool):
        return {'boolValue': value}
    if isinstance(value, int):
        return {'intValue': str(value)}
    if isinstance(value, float):
        return {'doubleValue': value}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {'arrayValue': {'values': [_attribute_value(item) for item in value]}}
    return {'stringValue': str(value)}


def _attributes(values: object) -> list[dict[str, object]]:
    return [{'key': str(key), 'value': _attribute_value(value)} for key, value in _mapping(values).items()]


def _hex(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.removeprefix('0x')


def _nanos(value: object) -> str:
    """Convert the exporter's RFC 3339 timestamp to Unix nanoseconds."""
    if not isinstance(value, str) or not value:
        return '0'
    from datetime import datetime

    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return str(int(parsed.timestamp() * 1_000_000_000))


def _span(record: Mapping[str, object]) -> dict[str, object]:
    context = _mapping(record.get('context'))
    parent = _hex(record.get('parent_id'))
    status = _mapping(record.get('status'))
    span: dict[str, object] = {
        'traceId': _hex(context.get('trace_id')) or '',
        'spanId': _hex(context.get('span_id')) or '',
        'name': str(record.get('name', '')),
        'kind': _KIND.get(str(record.get('kind')), 1),
        'startTimeUnixNano': _nanos(record.get('start_time')),
        'endTimeUnixNano': _nanos(record.get('end_time')),
        'attributes': _attributes(record.get('attributes')),
        'status': {
            'code': _STATUS.get(str(status.get('status_code')), 0),
            **({'message': str(status['description'])} if status.get('description') else {}),
        },
    }
    if parent:
        span['parentSpanId'] = parent
    events = record.get('events')
    if isinstance(events, Sequence) and not isinstance(events, (str, bytes)) and events:
        span['events'] = [
            {
                'name': str(_mapping(event).get('name', '')),
                'timeUnixNano': _nanos(_mapping(event).get('timestamp')),
                'attributes': _attributes(_mapping(event).get('attributes')),
            }
            for event in events
        ]
    return span


def _resource(record: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    attributes = dict(_mapping(_mapping(record.get('resource')).get('attributes')))
    service_name = os.environ.get('OTEL_SERVICE_NAME')
    if service_name:
        attributes['service.name'] = service_name
    for pair in os.environ.get('OTEL_RESOURCE_ATTRIBUTES', '').split(','):
        key, _, value = pair.partition('=')
        if key and value:
            attributes[key] = value
    return json.dumps(attributes, sort_keys=True), {'attributes': _attributes(attributes)}


def build_payload(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Group spans by resource and render the OTLP/HTTP JSON request body."""
    grouped: dict[str, dict[str, object]] = {}
    for record in records:
        key, resource = _resource(record)
        entry = grouped.setdefault(key, {'resource': resource, 'spans': []})
        spans = entry['spans']
        assert isinstance(spans, list)
        spans.append(_span(record))
    return {
        'resourceSpans': [
            {'resource': entry['resource'], 'scopeSpans': [{'scope': {'name': 'pydantic-ai'}, 'spans': entry['spans']}]}
            for entry in grouped.values()
        ]
    }


def _headers() -> dict[str, str]:
    headers = {'Content-Type': 'application/json'}
    for pair in os.environ.get('OTEL_EXPORTER_OTLP_HEADERS', '').split(','):
        key, _, value = pair.partition('=')
        if key and value:
            headers[key.strip()] = value.strip()
    return headers


def _post(endpoint: str, payload: dict[str, object]) -> None:
    request = urllib.request.Request(
        endpoint.rstrip('/') + '/v1/traces',
        data=json.dumps(payload).encode(),
        headers=_headers(),
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=30):
        return


def forward(path: pathlib.Path, endpoint: str) -> int:
    """Post every span in `path` and remove the file once delivery succeeds."""
    if not path.is_file():
        return 0
    records: list[Mapping[str, object]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        try:
            value: object = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(_mapping(value))
    exported = 0
    for start in range(0, len(records), BATCH_SIZE):
        try:
            _post(endpoint, build_payload(records[start : start + BATCH_SIZE]))
        except (OSError, urllib.error.URLError) as error:
            print(f'agent span forwarder: delivery failed: {error}', file=sys.stderr)
            return 1
        exported += len(records[start : start + BATCH_SIZE])
    if path.exists():
        path.unlink()
    if exported:
        print(f'agent span forwarder: exported {exported} span(s)')
    return 0


def main() -> int:
    """Run the forwarder from the workflow post-step environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=pathlib.Path, default=pathlib.Path(DEFAULT_EXPORT_PATH))
    args = parser.parse_args()
    endpoint = os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT', '')
    if not endpoint:
        return 0
    return forward(args.path, endpoint)


if __name__ == '__main__':
    raise SystemExit(main())
