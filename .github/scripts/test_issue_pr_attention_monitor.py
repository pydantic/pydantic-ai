from __future__ import annotations

import datetime as dt
import io
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import issue_pr_attention_monitor as monitor

NOW = dt.datetime(2026, 7, 20, tzinfo=dt.timezone.utc)
OLD = '2026-07-16T00:00:00Z'


def item(
    number: int,
    *,
    labels: list[str] | None = None,
    assignees: list[str] | None = None,
    updated_at: str = OLD,
) -> dict[str, Any]:
    return {
        'number': number,
        'state': 'open',
        'updated_at': updated_at,
        'title': f'Item {number}',
        'body': 'Please decide the project direction.',
        'comments': 0,
        'labels': [{'name': label} for label in labels or []],
        'assignees': [{'login': login} for login in assignees or []],
    }


class FakeClient:
    def __init__(self, items: dict[int, dict[str, Any]] | None = None) -> None:
        self.items = items or {}
        self.calls: list[tuple[str, str, object | None]] = []
        self.fail_get: set[int] = set()
        self.assignment_succeeds = True
        self.permissions: dict[str, str] = {}
        self.timelines: dict[int, list[dict[str, Any]]] = {}

    def get(self, path: str) -> Any:
        self.calls.append(('GET', path, None))
        if '/labels/' in path:
            return {'name': path.rsplit('/', 1)[-1]}
        if '/issues?state=' in path and 'labels=' in path:
            requested = urllib.parse.unquote(path.split('labels=')[1].split('&')[0])
            return [
                value for value in self.items.values() if requested in {str(label['name']) for label in value['labels']}
            ]
        if '/collaborators/' in path and path.endswith('/permission'):
            login = path.split('/collaborators/')[1].split('/')[0]
            return {'permission': self.permissions.get(login, 'read')}
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
            return (
                {'assignees': [{'login': monitor._FALLBACK_OWNER}]} if self.assignment_succeeds else {'assignees': []}
            )
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
        if stage == 1 and path.endswith('/timeline'):
            events.append(
                {
                    'event': 'commented',
                    'created_at': OLD,
                    'actor': {'login': 'github-actions[bot]'},
                    'body': monitor._reminder([monitor._FALLBACK_OWNER]),
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


def test_snapshot_skips_active_recent_and_escalated_items():
    client = SnapshotClient(
        {
            1: item(1, labels=[monitor._ACTION_LABEL]),
            2: item(2, labels=[monitor._PINGED_LABEL]),
            3: item(3, updated_at='2026-07-19T00:00:00Z'),
            4: item(4, labels=[monitor._ESCALATED_LABEL]),
        }
    )
    candidates = monitor.build_snapshot(client, 'pydantic/pydantic-ai', now=NOW)['candidates']
    assert [candidate['number'] for candidate in candidates] == [2]


def test_candidate_search_excludes_active_and_escalated_labels():
    client = SnapshotClient({})
    monitor._candidate_page(client, 'pydantic/pydantic-ai', now=NOW)

    query = client.calls[0][1]
    assert urllib.parse.quote_plus(f'-label:"{monitor._ACTION_LABEL}"') in query
    assert urllib.parse.quote_plus(f'-label:"{monitor._ESCALATED_LABEL}"') in query


def test_snapshot_recheck_skips_items_closed_after_search():
    closed = item(7)
    closed['state'] = 'closed'
    client = SnapshotClient({7: closed})

    assert monitor.build_snapshot(client, 'pydantic/pydantic-ai', now=NOW)['candidates'] == []


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

    assert lines == ['#7: requested maintainer attention from @adtyavrdhn']
    assert (
        'POST',
        '/repos/pydantic/pydantic-ai/issues/7/assignees',
        {'assignees': ['adtyavrdhn']},
    ) in client.calls


def test_apply_pings_all_assigned_maintainers_without_reassigning(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 7, 'updated_at': OLD}])
    write_output(output, ['7'])
    client = FakeClient({7: item(7, assignees=['alice', 'bob', 'reader'])})
    # `admin`/`write`/`read`/`none` are the only values the legacy permission
    # field returns; `maintain` appears only in role_name, never here.
    client.permissions = {'alice': 'admin', 'bob': 'write', 'reader': 'read'}

    assert monitor.apply_decisions(client, 'r', str(output), str(snapshot)) == [
        '#7: requested maintainer attention from @alice @bob'
    ]
    assert not any(call[1].endswith('/assignees') for call in client.calls)


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


def test_apply_rejects_unknown_actor_or_confidence(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 7, 'updated_at': OLD}])

    write_output(output, ['7'], next_actor='attacker')
    with pytest.raises(ValueError, match='Invalid next_actor'):
        monitor.apply_decisions(FakeClient({7: item(7)}), 'r', str(output), str(snapshot))

    write_output(output, ['7'], confidence='certain')
    with pytest.raises(ValueError, match='Invalid confidence'):
        monitor.apply_decisions(FakeClient({7: item(7)}), 'r', str(output), str(snapshot))


def test_apply_assigns_fallback_when_no_assignee_is_a_maintainer(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 7, 'updated_at': OLD}])
    write_output(output, ['7'])
    client = FakeClient({7: item(7, assignees=['reader'])})
    client.permissions = {'reader': 'read'}

    assert monitor.apply_decisions(client, 'r', str(output), str(snapshot)) == [
        '#7: requested maintainer attention from @adtyavrdhn'
    ]
    assert ('POST', '/repos/r/issues/7/assignees', {'assignees': ['adtyavrdhn']}) in client.calls


def test_apply_skips_closed_or_already_actioned_items(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_snapshot(snapshot, [{'number': 7, 'updated_at': OLD}])
    write_output(output, ['7'])
    closed = item(7)
    closed['state'] = 'closed'

    for changed in (closed, item(7, labels=[monitor._ACTION_LABEL])):
        client = FakeClient({7: changed})
        assert monitor.apply_decisions(client, 'r', str(output), str(snapshot)) == [
            '#7: skipped because the item changed after classification'
        ]
        assert not any(call[0] == 'POST' and '/issues/7/' in call[1] for call in client.calls)


def test_reconcile_reminds_assigned_maintainers():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL], assignees=['bob', 'alice'])})
    client.permissions = {'alice': 'admin', 'bob': 'write'}

    assert monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW) == (['#7: reminded assigned maintainer'], [])

    comment = next(call for call in client.calls if call[0] == 'POST' and call[1].endswith('/comments'))
    assert comment[2] == {'body': '@alice @bob this still needs a maintainer decision. Could you take a look?'}
    assert any(call[0] == 'POST' and call[2] == {'labels': [monitor._PINGED_LABEL]} for call in client.calls)


def test_reconcile_queues_private_escalation_without_public_comment():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL])})
    escalations: list[int] = []

    assert monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW, escalations=escalations) == (
        ['#7: queued private Slack escalation'],
        [],
    )
    assert escalations == [7]
    assert not any(call[1].endswith('/comments') for call in client.calls)
    assert not any(call[0] == 'DELETE' and monitor._ACTION_LABEL in call[1] for call in client.calls)


def test_reconcile_retries_private_escalation_until_delivery():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL, monitor._ESCALATED_LABEL])})
    escalations: list[int] = []

    assert monitor.reconcile(client, 'r', now=NOW, escalations=escalations) == (
        ['#7: queued private Slack escalation'],
        [],
    )
    assert escalations == [7]
    assert any(call[0] == 'DELETE' and monitor._PINGED_LABEL in call[1] for call in client.calls)
    assert not any(call[0] == 'DELETE' and monitor._ACTION_LABEL in call[1] for call in client.calls)


def test_terminal_stage_preserves_the_reminder_acknowledgement_boundary():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._ESCALATED_LABEL])})
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._PINGED_LABEL},
        },
        {
            'event': 'commented',
            'created_at': '2026-07-18T00:00:00Z',
            'actor': {'login': monitor._FALLBACK_OWNER},
            'body': 'I will handle this.',
        },
        {
            'event': 'labeled',
            'created_at': '2026-07-19T00:00:00Z',
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._ESCALATED_LABEL},
        },
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: maintainer acknowledged the request'], [])


def test_reconcile_rechecks_acknowledgement_before_queueing_slack():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL])})
    event = {
        'event': 'labeled',
        'created_at': OLD,
        'actor': {'login': 'github-actions[bot]'},
        'label': {'name': monitor._PINGED_LABEL},
    }
    reminder = {
        'event': 'commented',
        'created_at': OLD,
        'actor': {'login': 'github-actions[bot]'},
        'body': monitor._reminder([monitor._FALLBACK_OWNER]),
    }
    calls = 0

    def pages(path: str, *, count: int = 1) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        if path.endswith('/events'):
            return [event]
        if calls == 2:
            return [event, reminder]
        return [
            event,
            reminder,
            {
                'event': 'commented',
                'created_at': '2026-07-18T00:00:00Z',
                'actor': {'login': monitor._FALLBACK_OWNER},
            },
        ]

    client.last_pages = pages  # type: ignore[method-assign]

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: maintainer acknowledged the request'], [])


def test_finalize_clears_active_state_only_after_slack_delivery():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._ESCALATED_LABEL])})

    assert monitor.finalize_escalations(client, 'r', monitor._escalation_numbers({'items': [7]})) == [
        '#7: delivered private Slack escalation'
    ]
    assert any(call[0] == 'DELETE' and monitor._ACTION_LABEL in call[1] for call in client.calls)


def test_finalize_continues_after_one_delivered_item_fails():
    client = FakeClient(
        {
            7: item(7, labels=[monitor._ACTION_LABEL, monitor._ESCALATED_LABEL]),
            8: item(8, labels=[monitor._ACTION_LABEL, monitor._ESCALATED_LABEL]),
        }
    )
    client.fail_get.add(7)

    with pytest.raises(RuntimeError, match=r'#7: HTTPError'):
        monitor.finalize_escalations(client, 'r', [7, 8])
    assert any(call[0] == 'DELETE' and '/issues/8/' in call[1] for call in client.calls)


def test_finalize_clears_delivered_state_if_item_closed_during_delivery():
    closed = item(7, labels=[monitor._ACTION_LABEL, monitor._ESCALATED_LABEL])
    closed['state'] = 'closed'
    client = FakeClient({7: closed})

    assert monitor.finalize_escalations(client, 'r', [7]) == ['#7: delivered private Slack escalation']


def test_finalize_skips_items_whose_state_was_already_cleared():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL]), 8: item(8)})

    assert monitor.finalize_escalations(client, 'r', [7, 8]) == []
    assert not any(call[0] == 'DELETE' for call in client.calls)


@pytest.mark.parametrize(
    'contents',
    [
        [7],
        {'items': [7], 'extra': 1},
        {'items': 7},
        {'items': [True]},
        {'items': ['7']},
        {'items': [0]},
        {'items': [7, 7]},
        {'items': list(range(1, monitor._RECONCILE_LIMIT + 2))},
    ],
)
def test_escalation_input_rejects_invalid_shapes(contents: object):
    with pytest.raises(ValueError, match='Escalations must'):
        monitor._escalation_numbers(contents)


def test_snapshot_and_decision_batch_limits_are_enforced(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    write_snapshot(snapshot, [{'number': n, 'updated_at': OLD} for n in range(1, monitor._CANDIDATE_LIMIT + 2)])
    with pytest.raises(ValueError, match='candidate limit'):
        monitor._snapshot_candidates(str(snapshot))

    output = tmp_path / 'output.json'
    write_output(output, [str(n) for n in range(1, monitor._CANDIDATE_LIMIT + 2)])
    with pytest.raises(ValueError, match='too many or duplicate'):
        monitor._parse_decisions(str(output))


def test_escalation_output_is_fixed_and_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output = tmp_path / 'github-output'
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))

    monitor._write_escalations('pydantic/pydantic-ai', [8, 7, 7])

    values = dict(line.split('=', 1) for line in output.read_text(encoding='utf-8').splitlines())
    assert values['has_escalations'] == 'true'
    assert values['escalation_items'] == '[7,8]'
    assert json.loads(values['slack_payload']) == {
        'text': ':warning: Maintainer attention needs your view: '
        '<https://github.com/pydantic/pydantic-ai/issues/7|#7>, '
        '<https://github.com/pydantic/pydantic-ai/issues/8|#8>'
    }


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

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: removed a foreign attention transition'], [])
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

    assert monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW) == ([], [])


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
            'actor': {'login': monitor._FALLBACK_OWNER},
            'body': 'Decision made.',
        },
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: maintainer acknowledged the request'], [])
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
            # The real timeline API puts a review's author under `user`, not
            # `actor` — exercising the `or event.get('user')` fallback in `_actor`.
            'event': 'reviewed',
            'submitted_at': OLD,
            'user': {'login': 'another-maintainer'},
            'author_association': 'MEMBER',
        },
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: maintainer acknowledged the request'], [])


def test_recipient_non_comment_event_completes_the_request():
    # A recipient who labels, milestones, self-assigns, or closes while being
    # reminded is engaging: any non-denylisted event by a recipient acknowledges.
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL], assignees=['alice'])})
    client.permissions = {'alice': 'admin'}
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._ACTION_LABEL},
        },
        {
            'event': 'labeled',
            'created_at': '2026-07-17T00:00:00Z',
            'actor': {'login': 'alice'},
            'label': {'name': 'question'},
        },
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: maintainer acknowledged the request'], [])


def test_collaborator_comment_by_non_recipient_completes_the_request():
    # An outside collaborator with repo access can acknowledge via a comment even
    # when they are not one of the assigned recipients (COLLABORATOR association).
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL])})
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
            'actor': {'login': 'outside-collaborator'},
            'author_association': 'COLLABORATOR',
            'body': 'I can take this.',
        },
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: maintainer acknowledged the request'], [])


def test_reconcile_does_not_trust_a_shared_bot_comment_as_state():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL])})
    body = monitor._reminder([monitor._FALLBACK_OWNER])
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

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: reminded assigned maintainer'], [])
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

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: restored maintainer reminder'], [])
    assert any(call[0] == 'POST' and call[2] == {'labels': [monitor._PINGED_LABEL]} for call in client.calls)
    assert sum(call[0] == 'POST' and call[1].endswith('/comments') for call in client.calls) == 1


def test_reassignment_restarts_the_reminder_sla():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL], assignees=['bob'])})
    client.permissions = {'bob': 'write'}
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._PINGED_LABEL},
        },
        {
            'event': 'commented',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'body': '@alice this still needs a maintainer decision. Could you take a look?',
        },
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: restored maintainer reminder'], [])
    assert any(call[0] == 'DELETE' and monitor._PINGED_LABEL in call[1] for call in client.calls)
    assert any(call[0] == 'POST' and call[2] == {'labels': [monitor._PINGED_LABEL]} for call in client.calls)


def test_closed_item_completes_and_strips_lifecycle_labels():
    # Closing an item is the ultimate resolution: the action and stage labels
    # are removed so a later reopen can't wake an ancient SLA clock.
    closed = item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL])
    closed['state'] = 'closed'
    client = FakeClient({7: closed})

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: completed after the item was closed'], [])
    assert any(call[0] == 'DELETE' and monitor._ACTION_LABEL in call[1] for call in client.calls)
    assert any(call[0] == 'DELETE' and monitor._PINGED_LABEL in call[1] for call in client.calls)
    # No public reminder or private escalation is produced for a closed item.
    assert not any(call[1].endswith('/comments') for call in client.calls)


def test_reopened_item_without_action_label_fires_no_reminder():
    # After the closed-item completion strips the labels, a reopen leaves no
    # action label, so no stage transition can fire an instant reminder.
    reopened = item(2)
    client = FakeClient({2: reopened})
    assert monitor.reconcile(client, 'r', now=NOW) == ([], [])
    assert not any(call[1].endswith('/comments') for call in client.calls)


def test_full_page_processes_a_bounded_batch_instead_of_aborting():
    client = FakeClient(
        {number: item(number, labels=[monitor._ACTION_LABEL]) for number in range(1, monitor._RECONCILE_LIMIT + 1)}
    )

    lines, failures = monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW)

    assert failures == []
    assert sum('reminded assigned maintainer' in line for line in lines) == monitor._RECONCILE_LIMIT
    assert lines[-1] == 'additional attention items remain for a later rotated batch'


def test_one_item_failure_does_not_block_later_items():
    client = FakeClient(
        {
            1: item(1, labels=[monitor._ACTION_LABEL]),
            2: item(2, labels=[monitor._ACTION_LABEL]),
        }
    )
    client.fail_get.add(1)

    lines, failures = monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW)

    assert lines == ['#2: reminded assigned maintainer']
    assert failures and failures[0].startswith('#1: HTTPError')
    assert any(call[0] == 'POST' and call[1].endswith('/issues/2/comments') for call in client.calls)


def test_invalid_event_timestamp_does_not_block_later_items():
    client = FakeClient(
        {
            1: item(1, labels=[monitor._ACTION_LABEL]),
            2: item(2, labels=[monitor._ACTION_LABEL]),
        }
    )
    client.timelines[1] = [
        {
            'event': 'labeled',
            'created_at': 'invalid',
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._ACTION_LABEL},
        }
    ]

    lines, failures = monitor.reconcile(client, 'r', now=NOW)

    assert lines == ['#2: reminded assigned maintainer']
    assert failures and failures[0].startswith('#1: ValueError')
    assert any(call[0] == 'POST' and call[1].endswith('/issues/2/comments') for call in client.calls)


def test_one_item_failure_still_queues_other_escalations():
    client = FakeClient(
        {
            1: item(1, labels=[monitor._ACTION_LABEL]),
            2: item(2, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL]),
        }
    )
    client.fail_get.add(1)
    escalations: list[int] = []

    lines, failures = monitor.reconcile(client, 'r', now=NOW, escalations=escalations)

    assert lines == ['#2: queued private Slack escalation']
    assert escalations == [2]
    assert failures and failures[0].startswith('#1: HTTPError')


def test_bot_triggered_mention_event_is_not_an_acknowledgement():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL])})
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._ACTION_LABEL},
        },
        {'event': 'mentioned', 'created_at': '2026-07-17T00:00:00Z', 'actor': {'login': monitor._FALLBACK_OWNER}},
        {'event': 'subscribed', 'created_at': '2026-07-17T00:00:00Z', 'actor': {'login': monitor._FALLBACK_OWNER}},
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: reminded assigned maintainer'], [])


def test_latest_stage_transition_restarts_the_sla_clock():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL])})
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._PINGED_LABEL},
        },
        {
            'event': 'labeled',
            'created_at': '2026-07-19T00:00:00Z',
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._PINGED_LABEL},
        },
        {
            'event': 'commented',
            'created_at': '2026-07-19T00:00:00Z',
            'actor': {'login': 'github-actions[bot]'},
            'body': monitor._reminder([monitor._FALLBACK_OWNER]),
        },
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == ([], [])


def test_sweep_restores_eligibility_after_new_activity():
    client = FakeClient({7: item(7, labels=[monitor._ESCALATED_LABEL])})
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._ESCALATED_LABEL},
        },
        {
            'event': 'unlabeled',
            'created_at': '2026-07-17T00:00:00Z',
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._ACTION_LABEL},
        },
        {'event': 'commented', 'created_at': '2026-07-18T00:00:00Z', 'actor': {'login': 'contributor'}},
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == (
        ['#7: restored attention eligibility after new activity'],
        [],
    )
    assert any(call[0] == 'DELETE' and monitor._ESCALATED_LABEL in call[1] for call in client.calls)


def test_sweep_keeps_untouched_escalated_item_dormant():
    client = FakeClient({7: item(7, labels=[monitor._ESCALATED_LABEL])})
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._ESCALATED_LABEL},
        },
        {
            'event': 'unlabeled',
            'created_at': '2026-07-17T00:00:00Z',
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._ACTION_LABEL},
        },
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == ([], [])
    assert not any(call[0] == 'DELETE' for call in client.calls)


def test_sweep_removes_a_foreign_escalation_marker():
    client = FakeClient({7: item(7, labels=[monitor._ESCALATED_LABEL])})
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'outside-collaborator'},
            'label': {'name': monitor._ESCALATED_LABEL},
        }
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: removed a foreign attention transition'], [])
    assert any(call[0] == 'DELETE' and monitor._ESCALATED_LABEL in call[1] for call in client.calls)


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


def test_compiled_lock_pins_failure_report_and_stable_artifact_name():
    # Actions runs the compiled .lock.yml, not the .md; nothing else pins the
    # two together, so guard the load-bearing strings against a bad recompile.
    lock = Path(__file__).parent.parent / 'workflows' / 'pydantic-ai-attention-triage.lock.yml'
    text = lock.read_text()

    assert 'GH_AW_FAILURE_REPORT_AS_ISSUE: "true"' in text
    assert 'name: attention-candidates-${{ github.run_id }}' in text
    # The run_attempt suffix must stay gone: "Re-run failed jobs" bumps the
    # attempt number, but only the original run_id upload exists.
    assert 'name: attention-candidates-${{ github.run_id }}-' not in text


def test_operations_workflow_routes_only_terminal_escalation_to_slack():
    workflow = Path(__file__).parent.parent / 'workflows' / 'issue-pr-attention-monitor.yml'
    text = workflow.read_text()

    assert 'PYDANTIC_AI_TRIAGE_SLACK_WEBHOOK_URL' in text
    assert 'steps.reconcile.outputs.slack_payload' in text
    assert 'issue_pr_attention_monitor.py finalize' in text
    assert 'permissions: {}' in text
    assert 'ATTENTION_ESCALATIONS' in text


def test_monitor_imports_with_stdlib_only():
    # Production invokes the script with the runner's bare `python` (no venv,
    # no third-party packages); `-S` blocks site-packages to reproduce that.
    result = subprocess.run(
        [sys.executable, '-S', '-c', 'import issue_pr_attention_monitor'],
        env={**os.environ, 'PYTHONPATH': str(Path(__file__).parent)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


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
