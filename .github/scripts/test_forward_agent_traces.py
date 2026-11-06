"""Tests for forwarding the agent's OTLP/JSON span file."""

from __future__ import annotations

import json
import pathlib

import pytest
from forward_agent_traces import build_payload, forward


def _span(name: str = 'demo') -> dict[str, object]:
    return {
        'name': name,
        'context': {'trace_id': '0x' + 'a' * 32, 'span_id': '0x' + 'b' * 16},
        'parent_id': None,
        'kind': 'SpanKind.CLIENT',
        'start_time': '2026-11-06T00:00:00.000000Z',
        'end_time': '2026-11-06T00:00:01.000000Z',
        'status': {'status_code': 'ERROR', 'description': 'timeout'},
        'attributes': {'gen_ai.operation.name': 'chat', 'retries': 2, 'ok': False},
        'events': [{'name': 'event', 'timestamp': '2026-11-06T00:00:00.500000Z', 'attributes': {}}],
        'resource': {'attributes': {'service.name': 'gh-aw.workflow', 'service.version': 'abc'}},
    }


def test_payload_groups_spans_and_types_attributes() -> None:
    payload = build_payload([_span(), _span('other')])
    (resource_spans,) = payload['resourceSpans']
    (scope_spans,) = resource_spans['scopeSpans']
    assert len(scope_spans['spans']) == 2
    span = scope_spans['spans'][0]
    assert span['traceId'] == 'a' * 32
    assert span['spanId'] == 'b' * 16
    assert span['kind'] == 3
    assert span['status'] == {'code': 2, 'message': 'timeout'}
    attributes = {item['key']: item['value'] for item in span['attributes']}
    assert attributes['retries'] == {'intValue': '2'}
    assert attributes['ok'] == {'boolValue': False}
    assert attributes['gen_ai.operation.name'] == {'stringValue': 'chat'}
    assert span['events'][0]['timeUnixNano'] != '0'


def test_missing_file_is_a_no_op(tmp_path: pathlib.Path) -> None:
    assert forward(tmp_path / 'missing.jsonl', 'https://example.invalid') == 0


def test_forward_posts_and_removes_the_file(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'spans.jsonl'
    path.write_text(json.dumps(_span()) + '\n')
    posted: list[dict[str, object]] = []

    def fake_post(endpoint: str, payload: dict[str, object]) -> None:
        assert endpoint == 'https://example.invalid'
        posted.append(payload)

    monkeypatch.setattr('forward_agent_traces._post', fake_post)
    assert forward(path, 'https://example.invalid') == 0
    assert len(posted) == 1
    assert not path.exists()


def test_delivery_failure_keeps_the_file(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'spans.jsonl'
    path.write_text(json.dumps(_span()) + '\n')

    def failing_post(*args: object) -> None:
        raise OSError('refused')

    monkeypatch.setattr('forward_agent_traces._post', failing_post)
    assert forward(path, 'https://example.invalid') == 1
    assert path.exists()
