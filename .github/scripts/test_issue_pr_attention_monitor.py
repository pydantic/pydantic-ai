from __future__ import annotations

import datetime as dt
import io
import json
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import issue_pr_attention_monitor as monitor

NOW = dt.datetime(2026, 7, 20, tzinfo=dt.timezone.utc)
OLD = '2026-07-16T00:00:00Z'


def item(number: int, *, labels: list[str] | None = None, updated_at: str = OLD) -> dict[str, Any]:
    return {
        'number': number,
        'state': 'open',
        'updated_at': updated_at,
        'title': f'Item {number}',
        'body': 'Please decide the project direction.',
        'comments': 0,
        'labels': [{'name': label} for label in labels or []],
        'assignees': [],
    }


class FakeClient:
    def __init__(self, items: dict[int, dict[str, Any]] | None = None) -> None:
        self.items = items or {}
        self.calls: list[tuple[str, str, object | None]] = []
        self.fail_get: set[int] = set()
        self.assignment_succeeds = True
        self.timelines: dict[int, list[dict[str, Any]]] = {}

    def get(self, path: str) -> Any:
        self.calls.append(('GET', path, None))
        if '/labels/' in path:
            return {'name': path.rsplit('/', 1)[-1]}
        if '/issues?state=open&labels=' in path:
            return list(self.items.values())
        if '/issues/' in path and '/comments?' not in path:
            number = int(path.split('/issues/')[1].split('/')[0])
            if number in self.fail_get:
                raise urllib.error.HTTPError(path, 500, 'boom', {}, None)
            return self.items[number]
        if '/comments?' in path:
            return []
        raise AssertionError(path)

    def post(self, path: str, payload: object) -> Any:
        self.calls.append(('POST', path, payload))
        if path.endswith('/assignees'):
            return {'assignees': [{'login': monitor._OWNER}]} if self.assignment_succeeds else {'assignees': []}
        return {}

    def delete(self, path: str) -> None:
        self.calls.append(('DELETE', path, None))

    def last_page(self, path: str) -> list[dict[str, Any]]:
        self.calls.append(('LAST', path, None))
        number = int(path.split('/issues/')[1].split('/')[0])
        if number in self.timelines:
            return self.timelines[number]
        labels = {label['name'] for label in self.items[number]['labels']}
        stage = monitor._stage(labels)
        label = monitor._ACTION_LABEL if stage == 0 else monitor._STAGE_LABELS[stage - 1]
        events = [
            {
                'event': 'labeled',
                'created_at': OLD,
                'actor': {'login': 'github-actions[bot]'},
                'label': {'name': label},
            }
        ]
        if stage > 0 and path.endswith('/timeline'):
            events.append(
                {
                    'event': 'commented',
                    'created_at': OLD,
                    'actor': {'login': 'github-actions[bot]'},
                    'body': monitor._reminder(stage),
                }
            )
        return events

    def last_pages(self, path: str, *, count: int = 1) -> list[dict[str, Any]]:
        return self.last_page(path)


class SnapshotClient(FakeClient):
    def __init__(self, values: dict[int, dict[str, Any]]) -> None:
        super().__init__(values)
        self.search_results = list(values.values())

    def get(self, path: str) -> Any:
        if path.startswith('/search/issues?'):
            self.calls.append(('GET', path, None))
            if 'per_page=1&' in path or path.endswith('per_page=1'):
                return {'total_count': len(self.search_results), 'items': self.search_results[:1]}
            return {'total_count': len(self.search_results), 'items': self.search_results}
        if '/check-runs?' in path:
            self.calls.append(('GET', path, None))
            return {'check_runs': [{'name': 'CI', 'status': 'completed', 'conclusion': 'success'}]}
        if '/pulls/' in path and '/comments?' not in path:
            self.calls.append(('GET', path, None))
            number = int(path.split('/pulls/')[1])
            return {
                **self.items[number],
                'review_comments': 0,
                'draft': False,
                'mergeable_state': 'clean',
                'requested_reviewers': [],
                'head': {'sha': f'sha-{number}'},
            }
        return super().get(path)

    def last_page(self, path: str) -> list[dict[str, Any]]:
        if '/pulls/' in path and path.endswith('/reviews'):
            self.calls.append(('LAST', path, None))
            return []
        return super().last_page(path)


def write_snapshot(path: Path, values: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({'generated_at': NOW.isoformat(), 'candidates': values}), encoding='utf-8')


def write_output(
    path: Path,
    numbers: list[str],
    *,
    next_actor: str = 'maintainer',
    confidence: str = 'high',
) -> None:
    path.write_text(
        json.dumps(
            {
                'items': [
                    {
                        'type': 'record_attention_decision',
                        'item_number': n,
                        'next_actor': next_actor,
                        'confidence': confidence,
                    }
                    for n in numbers
                ]
            }
        ),
        encoding='utf-8',
    )


def test_last_page_uses_the_page_containing_the_newest_activity():
    assert monitor._last_page(0, 8) == 1
    assert monitor._last_page(8, 8) == 1
    assert monitor._last_page(9, 8) == 2


def test_build_and_write_snapshot_are_bounded_and_agent_readable(tmp_path: Path):
    pull = {**item(8), 'pull_request': {'url': 'https://api.github.test/pulls/8'}}
    client = SnapshotClient({7: item(7), 8: pull})

    snapshot = monitor.build_snapshot(client, 'pydantic/pydantic-ai', now=NOW)
    assert [value['number'] for value in snapshot['candidates']] == [7, 8]
    assert [value['kind'] for value in snapshot['candidates']] == ['issue', 'pull_request']

    path = tmp_path / 'attention-candidates.json'
    assert monitor.write_snapshot(client, 'pydantic/pydantic-ai', str(path), now=NOW) == [
        'wrote 2 attention candidate(s)'
    ]
    assert json.loads(path.read_text(encoding='utf-8'))['candidates'][1]['kind'] == 'pull_request'


def test_pull_request_context_includes_newest_review_state():
    pull = {**item(8), 'pull_request': {'url': 'https://api.github.test/pulls/8'}}
    client = SnapshotClient({8: pull})

    def reviews(path: str) -> list[dict[str, Any]]:
        assert path.endswith('/pulls/8/reviews')
        return [
            {
                'submitted_at': '2026-07-16T01:00:00Z',
                'user': {'login': 'maintainer'},
                'author_association': 'MEMBER',
                'state': 'CHANGES_REQUESTED',
                'body': '',
            }
        ]

    client.last_page = reviews  # type: ignore[method-assign]
    snapshot = monitor.build_snapshot(client, 'pydantic/pydantic-ai', now=NOW)

    review = snapshot['candidates'][0]['recent_activity'][0]
    assert review['kind'] == 'review'
    assert review['state'] == 'CHANGES_REQUESTED'


def test_candidate_discovery_returns_empty_without_stale_items():
    client = SnapshotClient({})
    assert monitor._candidate_page(client, 'pydantic/pydantic-ai', now=NOW) == []


def test_snapshot_skips_active_and_recent_items_but_reconsiders_terminal_state():
    client = SnapshotClient(
        {
            1: item(1, labels=[monitor._ACTION_LABEL]),
            2: item(2, labels=[monitor._PINGED_LABEL]),
            3: item(3, updated_at='2026-07-19T00:00:00Z'),
        }
    )
    candidates = monitor.build_snapshot(client, 'pydantic/pydantic-ai', now=NOW)['candidates']
    assert [candidate['number'] for candidate in candidates] == [2]


def test_snapshot_rejects_aggregate_oversize(monkeypatch: pytest.MonkeyPatch):
    client = SnapshotClient({7: item(7)})
    monkeypatch.setattr(monitor, '_SNAPSHOT_LIMIT', 1)
    with pytest.raises(RuntimeError, match='snapshot exceeds'):
        monitor.build_snapshot(client, 'pydantic/pydantic-ai', now=NOW)


def test_snapshot_uses_utf8_without_ascii_escape_inflation(tmp_path: Path):
    value = item(7)
    value['body'] = '🤖' * 100
    client = SnapshotClient({7: value})
    path = tmp_path / 'snapshot.json'

    monitor.write_snapshot(client, 'pydantic/pydantic-ai', str(path), now=NOW)

    assert '🤖' in path.read_text(encoding='utf-8')
    assert path.stat().st_size <= monitor._SNAPSHOT_LIMIT


def test_parse_decisions_rejects_injection_and_duplicates(tmp_path: Path):
    output = tmp_path / 'output.json'
    write_output(output, ['1; echo pwned'])
    with pytest.raises(ValueError, match='positive decimal'):
        monitor._parse_decisions(str(output))

    write_output(output, ['1', '1'])
    with pytest.raises(ValueError, match='duplicate'):
        monitor._parse_decisions(str(output))


@pytest.mark.parametrize(
    ('contents', 'message'),
    [
        ([], 'Snapshot must contain'),
        ({}, 'Snapshot must contain'),
        ({'candidates': [None]}, 'candidate must be'),
        ({'candidates': [{'number': 0, 'updated_at': OLD}]}, 'unique positive'),
    ],
)
def test_snapshot_validation_rejects_invalid_shapes(tmp_path: Path, contents: object, message: str):
    path = tmp_path / 'snapshot.json'
    path.write_text(json.dumps(contents), encoding='utf-8')
    with pytest.raises(ValueError, match=message):
        monitor._snapshot_candidates(str(path))


def test_agent_output_requires_items_but_ignores_other_safe_outputs(tmp_path: Path):
    path = tmp_path / 'output.json'
    path.write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='items list'):
        monitor._parse_decisions(str(path))
    path.write_text(json.dumps({'items': [None, {'type': 'noop'}]}), encoding='utf-8')
    assert monitor._parse_decisions(str(path)) == []


def test_apply_revalidates_then_assigns_and_labels(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 7, 'updated_at': OLD}])
    write_output(output, ['7'])
    client = FakeClient({7: item(7)})

    lines = monitor.apply_decisions(client, 'pydantic/pydantic-ai', str(output), str(snapshot))

    assert lines == ['#7: assigned @adtyavrdhn and requested maintainer attention']
    assert (
        'POST',
        '/repos/pydantic/pydantic-ai/issues/7/assignees',
        {'assignees': ['adtyavrdhn']},
    ) in client.calls


def test_apply_restarts_a_prior_terminal_escalation(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 7, 'updated_at': OLD}])
    write_output(output, ['7'])
    client = FakeClient({7: item(7, labels=[monitor._ESCALATED_LABEL])})

    monitor.apply_decisions(client, 'r', str(output), str(snapshot))

    assert any(call[0] == 'DELETE' and monitor._ESCALATED_LABEL in call[1] for call in client.calls)
    assert any(call[0] == 'POST' and call[2] == {'labels': [monitor._ACTION_LABEL]} for call in client.calls)


def test_apply_records_settled_negative_without_requesting_attention(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 7, 'updated_at': OLD}])
    write_output(output, ['7'], next_actor='contributor')
    client = FakeClient({7: item(7)})

    assert monitor.apply_decisions(client, 'pydantic/pydantic-ai', str(output), str(snapshot)) == [
        '#7: did not request maintainer attention'
    ]
    assert not any(call[0] == 'POST' and call[1].endswith('/labels') for call in client.calls)
    assert not any(call[1].endswith('/assignees') for call in client.calls)


def test_apply_leaves_uncertain_or_low_confidence_item_for_reconsideration(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 7, 'updated_at': OLD}])
    write_output(output, ['7'], next_actor='uncertain', confidence='high')
    client = FakeClient({7: item(7)})
    assert monitor.apply_decisions(client, 'r', str(output), str(snapshot)) == [
        '#7: left unclassified for a future run'
    ]

    write_output(output, ['7'], confidence='medium')
    assert monitor.apply_decisions(client, 'r', str(output), str(snapshot)) == [
        '#7: left unclassified for a future run'
    ]


def test_apply_rejects_numbers_outside_the_immutable_snapshot(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 7, 'updated_at': OLD}])
    write_output(output, ['8'])
    client = FakeClient()

    with pytest.raises(ValueError, match='outside the snapshot'):
        monitor.apply_decisions(client, 'pydantic/pydantic-ai', str(output), str(snapshot))
    assert client.calls == []


def test_apply_requires_one_decision_per_candidate(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 7, 'updated_at': OLD}, {'number': 8, 'updated_at': OLD}])
    write_output(output, ['7'])
    with pytest.raises(ValueError, match='classify every'):
        monitor.apply_decisions(FakeClient(), 'r', str(output), str(snapshot))


def test_apply_abstains_when_item_changed_after_classification(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 7, 'updated_at': OLD}])
    write_output(output, ['7'])
    client = FakeClient({7: item(7, updated_at='2026-07-19T00:00:00Z')})

    lines = monitor.apply_decisions(client, 'pydantic/pydantic-ai', str(output), str(snapshot))

    assert lines == ['#7: skipped because the item changed after classification']
    assert not any(call[0] == 'POST' and '/issues/7/' in call[1] for call in client.calls)


def test_apply_fails_if_github_silently_ignores_assignment(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 7, 'updated_at': OLD}])
    write_output(output, ['7'])
    client = FakeClient({7: item(7)})
    client.assignment_succeeds = False

    with pytest.raises(RuntimeError, match=r'#7: RuntimeError: GitHub did not assign'):
        monitor.apply_decisions(client, 'pydantic/pydantic-ai', str(output), str(snapshot))
    assert any(call[0] == 'POST' and call[1].endswith('/labels') for call in client.calls)


def test_apply_keeps_processing_after_one_item_fails(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 1, 'updated_at': OLD}, {'number': 2, 'updated_at': OLD}])
    write_output(output, ['1', '2'])
    client = FakeClient({1: item(1), 2: item(2)})
    client.fail_get.add(1)

    with pytest.raises(RuntimeError, match=r'#1: HTTPError'):
        monitor.apply_decisions(client, 'pydantic/pydantic-ai', str(output), str(snapshot))
    assert any(call[0] == 'POST' and call[1].endswith('/issues/2/labels') for call in client.calls)


@pytest.mark.parametrize(
    ('labels', 'expected_stage', 'mention'),
    [
        ([monitor._ACTION_LABEL], 1, '@adtyavrdhn'),
        ([monitor._ACTION_LABEL, monitor._PINGED_LABEL], 2, '@DouweM'),
    ],
)
def test_reconcile_advances_visible_stage_labels(labels: list[str], expected_stage: int, mention: str):
    client = FakeClient({7: item(7, labels=labels)})

    assert monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW) == [f'#7: posted reminder {expected_stage}']

    comment = next(call for call in client.calls if call[0] == 'POST' and call[1].endswith('/comments'))
    assert mention in str(comment[2])
    next_label = monitor._STAGE_LABELS[expected_stage - 1]
    assert any(call[0] == 'POST' and call[2] == {'labels': [next_label]} for call in client.calls)


def test_reconcile_stops_after_escalation():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._ESCALATED_LABEL])})

    assert monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW) == ['#7: completed terminal escalation']
    deletes = [call for call in client.calls if call[0] == 'DELETE']
    assert monitor._ACTION_LABEL in deletes[0][1]


def test_reconcile_cleans_an_obsolete_lower_stage_label():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL, monitor._ESCALATED_LABEL])})

    assert monitor.reconcile(client, 'r', now=NOW) == ['#7: completed terminal escalation']
    assert any(call[0] == 'DELETE' and monitor._PINGED_LABEL in call[1] for call in client.calls)


def test_reconcile_rejects_a_foreign_stage_label():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._ESCALATED_LABEL])})
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'outside-collaborator'},
            'label': {'name': monitor._ESCALATED_LABEL},
        }
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == ['#7: removed a foreign attention transition']
    assert any(call[0] == 'DELETE' and monitor._ACTION_LABEL in call[1] for call in client.calls)


def test_recent_activity_delays_the_next_reminder():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL])})
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': '2026-07-19T00:00:00Z',
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._ACTION_LABEL},
        }
    ]

    assert monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW) == []


def test_maintainer_comment_completes_the_request():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL])})
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._PINGED_LABEL},
        },
        {
            'event': 'commented',
            'created_at': '2026-07-17T00:00:00Z',
            'actor': {'login': monitor._OWNER},
            'body': 'Decision made.',
        },
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == ['#7: maintainer acknowledged the request']
    assert sum(call[0] == 'DELETE' for call in client.calls) == 2


def test_member_acknowledgement_in_the_same_second_completes_the_request():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL])})
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._ACTION_LABEL},
        },
        {
            'event': 'reviewed',
            'submitted_at': OLD,
            'actor': {'login': 'another-maintainer'},
            'author_association': 'MEMBER',
        },
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == ['#7: maintainer acknowledged the request']


def test_reconcile_does_not_trust_a_shared_bot_comment_as_state():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL])})
    body = monitor._reminder(1)
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._ACTION_LABEL},
        },
        {
            'event': 'commented',
            'created_at': '2026-07-17T00:00:00Z',
            'actor': {'login': 'github-actions[bot]'},
            'body': body,
        },
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == ['#7: posted reminder 1']
    assert sum(call[0] == 'POST' and call[1].endswith('/comments') for call in client.calls) == 1


def test_reconcile_restores_a_missing_comment_after_label_success():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL])})
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._PINGED_LABEL},
        }
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == ['#7: restored reminder 1']
    assert not any(call[0] == 'POST' and call[1].endswith('/labels') for call in client.calls)
    assert sum(call[0] == 'POST' and call[1].endswith('/comments') for call in client.calls) == 1


def test_reconcile_ignores_closed_or_opted_out_items():
    active = item(1, labels=[monitor._ACTION_LABEL])
    active['state'] = 'closed'
    opted_out = item(2)
    client = FakeClient({1: active, 2: opted_out})
    assert monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW) == []


def test_full_page_processes_a_bounded_batch_instead_of_aborting():
    client = FakeClient(
        {number: item(number, labels=[monitor._ACTION_LABEL]) for number in range(1, monitor._RECONCILE_LIMIT + 1)}
    )

    lines = monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW)

    assert sum('posted reminder' in line for line in lines) == monitor._RECONCILE_LIMIT
    assert lines[-1] == 'additional attention items remain for a later rotated batch'


def test_one_item_failure_does_not_block_later_items():
    client = FakeClient(
        {
            1: item(1, labels=[monitor._ACTION_LABEL]),
            2: item(2, labels=[monitor._ACTION_LABEL]),
        }
    )
    client.fail_get.add(1)

    with pytest.raises(RuntimeError, match=r'#1: HTTPError'):
        monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW)
    assert any(call[0] == 'POST' and call[1].endswith('/issues/2/comments') for call in client.calls)


def test_snapshot_is_inside_harness_workspace_and_writer_has_only_fixed_output():
    workflow = Path(__file__).parent.parent / 'workflows' / 'pydantic-ai-attention-triage.md'
    text = workflow.read_text()

    assert 'Read `attention-candidates.json`' in text
    assert 'path: attention-candidates.json' in text
    assert 'record-attention-decision:' in text
    assert 'issues: write' in text
    assert 'Slack' not in text
    assert 'PYDANTIC_AI_TRIAGE_SLACK_WEBHOOK_URL' not in text
    assert 'github: false' in text


class StubResponse(io.BytesIO):
    status = 200
    headers: dict[str, str] = {}

    def __enter__(self) -> StubResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_github_client_bounds_response_parsing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(monitor.urllib.request, 'urlopen', lambda request, timeout: StubResponse(b'{"ok": true}'))
    assert monitor.GitHubClient('token').get('/test') == {'ok': True}

    monkeypatch.setattr(monitor, '_RESPONSE_LIMIT', 2)
    monkeypatch.setattr(monitor.urllib.request, 'urlopen', lambda request, timeout: StubResponse(b'{}\n'))
    with pytest.raises(RuntimeError, match='response exceeds'):
        monitor.GitHubClient('token').get('/test')
