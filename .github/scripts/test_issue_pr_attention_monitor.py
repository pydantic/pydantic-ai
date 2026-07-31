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
import yaml

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
        'user': {'login': 'contributor'},
        'author_association': 'NONE',
        'labels': [{'name': label} for label in labels or []],
        'assignees': [{'login': login} for login in assignees or []],
    }


def label_event(
    label: str, *, actor: str = 'github-actions[bot]', created_at: str = OLD, event_id: str | None = None
) -> dict[str, Any]:
    event = {'event': 'labeled', 'created_at': created_at, 'actor': {'login': actor}, 'label': {'name': label}}
    return {**event, 'id': event_id} if event_id else event


class FakeClient:
    def __init__(self, items: dict[int, dict[str, Any]] | None = None) -> None:
        self.items = items or {}
        self.calls: list[tuple[str, str, object | None]] = []
        self.fail_get: set[int] = set()
        self.fail_delete_labels: set[str] = set()
        self.assignment_succeeds = True
        self.permissions: dict[str, str] = {}
        self.comments: dict[int, list[dict[str, Any]]] = {}
        self.timelines: dict[int, list[dict[str, Any]]] = {}
        self.roster_reads = 0

    def get(self, path: str) -> Any:
        self.calls.append(('GET', path, None))
        if path.startswith('/search/issues?'):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
            requested_state = 'closed' if 'is:closed' in query.get('q', [''])[0] else 'open'
            values = [
                value
                for value in self.items.values()
                if value['state'] == requested_state
                and monitor._ACTION_LABEL in {str(label['name']) for label in value['labels']}
            ]
            per_page = int(query.get('per_page', ['30'])[0])
            page = int(query.get('page', ['1'])[0])
            start = (page - 1) * per_page
            return {'total_count': len(values), 'items': values[start : start + per_page]}
        if '/labels/' in path:
            return {'name': path.rsplit('/', 1)[-1]}
        if '/issues?state=' in path and 'labels=' in path:
            requested = urllib.parse.unquote(path.split('labels=')[1].split('&')[0])
            state = path.split('/issues?state=')[1].split('&')[0]
            return [
                value
                for value in self.items.values()
                if requested in {str(label['name']) for label in value['labels']}
                and (state == 'all' or value['state'] == state)
            ]
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
            assert isinstance(payload, dict)
            number = int(path.split('/issues/')[1].split('/')[0])
            requested = [str(login) for login in payload['assignees']] if self.assignment_succeeds else []
            existing = [str(value['login']) for value in self.items[number]['assignees']]
            response = {
                **self.items[number],
                'assignees': [{'login': login} for login in dict.fromkeys([*existing, *requested])],
            }
            self.items[number] = response
            return response
        if path.endswith('/labels'):
            assert isinstance(payload, dict)
            number = int(path.split('/issues/')[1].split('/')[0])
            existing = {str(value['name']) for value in self.items[number]['labels']}
            labels = [str(label) for label in payload['labels']]
            self.items[number]['labels'].extend({'name': label} for label in labels if label not in existing)
        return {}

    def delete(self, path: str, payload: object | None = None) -> None:
        self.calls.append(('DELETE', path, payload))
        if path.endswith('/assignees'):
            assert isinstance(payload, dict)
            number = int(path.split('/issues/')[1].split('/')[0])
            removed = {str(login).casefold() for login in payload['assignees']}
            self.items[number]['assignees'] = [
                value for value in self.items[number]['assignees'] if str(value['login']).casefold() not in removed
            ]
            return
        if '/labels/' in path:
            number = int(path.split('/issues/')[1].split('/')[0])
            removed = urllib.parse.unquote(path.rsplit('/', 1)[-1])
            if removed in self.fail_delete_labels:
                raise urllib.error.HTTPError(path, 500, 'boom', {}, None)
            self.items[number]['labels'] = [
                value for value in self.items[number]['labels'] if str(value['name']) != removed
            ]

    def last_page(self, path: str) -> list[dict[str, Any]]:
        self.calls.append(('LAST', path, None))
        number = int(path.split('/issues/')[1].split('/')[0])
        if number in self.timelines:
            return [
                {**event, 'id': event.get('id', f'event-{index}')} for index, event in enumerate(self.timelines[number])
            ]
        labels = {label['name'] for label in self.items[number]['labels']}
        stage = monitor._stage(labels)
        label = monitor._ACTION_LABEL if stage == 0 else monitor._STAGE_LABELS[stage - 1]
        events = [label_event(label, event_id=f'default-stage-{stage}')]
        return events

    def last_pages(self, path: str, *, count: int = 1) -> list[dict[str, Any]]:
        return self.last_page(path)

    def pages(self, path: str, *, count: int):
        number = int(path.split('/issues/')[1].split('/')[0])
        yield self.comments.get(number, [])

    def maintainer_logins(self, repo: str) -> dict[str, str]:
        self.roster_reads += 1
        values = {monitor._FALLBACK_OWNER: 'write', **self.permissions}
        return {
            login.casefold(): login
            for login, permission in values.items()
            if permission in {'write', 'maintain', 'admin'}
        }


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


def test_candidate_search_covers_recent_activity_and_the_backlog():
    client = SnapshotClient({})
    monitor._candidate_page(client, 'pydantic/pydantic-ai', now=NOW)

    searches = [path for method, path, _ in client.calls if method == 'GET' and path.startswith('/search/issues?')]
    assert any('updated%3A%3E%3D' in path and 'order=desc' in path for path in searches)
    assert any('order=asc' in path and f'-label%3A%22{monitor._ACTION_LABEL}%22' in path for path in searches)
    assert all(f'-label%3A%22{monitor._ESCALATED_LABEL}%22' in path for path in searches)


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


def test_owner_selection_reloads_discussion_after_concurrent_activity():
    stale = item(7, labels=[monitor._ACTION_LABEL])
    current = item(
        7,
        labels=[monitor._ACTION_LABEL],
        updated_at='2026-07-17T00:00:00Z',
    )
    current['comments'] = 1
    client = FakeClient({7: current})
    client.permissions = {'DouweM': 'write'}
    client.comments[7] = [{'user': {'login': 'DouweM'}, 'created_at': '2026-07-17T00:00:00Z'}]

    assert monitor._ensure_recipients(client, 'r', stale) == ['DouweM']
    assert ('POST', '/repos/r/issues/7/assignees', {'assignees': ['DouweM']}) in client.calls


def test_owner_selection_uses_one_roster_for_a_large_discussion():
    issue = item(7, labels=[monitor._ACTION_LABEL])
    issue['comments'] = 51
    client = FakeClient({7: issue})
    client.permissions = {'DouweM': 'admin'}
    client.comments[7] = [
        *[{'user': {'login': f'contributor-{number}'}} for number in range(50)],
        {'user': {'login': 'DouweM'}},
    ]

    assert monitor._first_maintainer_in_discussion(client, 'r', issue) == 'DouweM'
    assert client.roster_reads == 1
    assert not any('/permission' in path for method, path, _ in client.calls if method == 'GET')


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


def notice_ref(
    number: int,
    stage: int,
    *,
    transition_id: int | str | None = None,
    recipients: list[str] | None = None,
) -> dict[str, object]:
    return {
        'number': number,
        'expected_stage': stage,
        'transition_id': transition_id if transition_id is not None else f'default-stage-{stage}',
        'recipients': recipients or [monitor._FALLBACK_OWNER],
    }


def test_reconcile_queues_channel_reminder_for_assigned_maintainers():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL], assignees=['bob', 'alice'])})
    client.permissions = {'alice': 'admin', 'bob': 'write'}
    notices: list[monitor.Notice] = []

    assert monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW, notices=notices) == (
        ['#7: queued channel reminder'],
        [],
    )
    assert notices == [
        {
            'number': 7,
            'kind': 'reminder',
            'expected_stage': 0,
            'transition_id': 'default-stage-0',
            'title': 'Item 7',
            'recipients': ['alice', 'bob'],
        }
    ]
    assert monitor._PINGED_LABEL not in {label['name'] for label in client.items[7]['labels']}
    assert monitor.finalize_notices(
        client, 'pydantic/pydantic-ai', monitor._notice_refs({'items': [notice_ref(7, 0, recipients=['alice', 'bob'])]})
    ) == ['#7: recorded channel reminder']
    assert monitor._PINGED_LABEL in {label['name'] for label in client.items[7]['labels']}


def test_reconcile_routes_existing_action_to_first_maintainer_participant():
    issue = item(4261, labels=[monitor._ACTION_LABEL], assignees=['dsfaccini'])
    issue['comments'] = 2
    client = FakeClient({4261: issue})
    client.permissions = {'DouweM': 'admin', 'dsfaccini': 'write'}
    client.comments[4261] = [
        {'user': {'login': 'DouweM'}, 'created_at': '2026-02-09T16:48:57Z'},
        {'user': {'login': 'dsfaccini'}, 'created_at': '2026-07-01T19:00:34Z'},
    ]
    notices: list[monitor.Notice] = []

    assert monitor.reconcile(client, 'r', now=NOW, notices=notices) == (
        ['#4261: queued channel reminder'],
        [],
    )
    assert notices[0]['recipients'] == ['DouweM']
    assert ('POST', '/repos/r/issues/4261/assignees', {'assignees': ['DouweM']}) in client.calls
    assert [assignee['login'] for assignee in client.items[4261]['assignees']] == ['DouweM']


def test_reconcile_drops_a_notice_if_the_owner_changes_before_queueing():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL], assignees=[monitor._FALLBACK_OWNER])})
    original_get = client.get
    item_reads = 0

    def get(path: str) -> Any:
        nonlocal item_reads
        if path.endswith('/issues/7'):
            item_reads += 1
            if item_reads == 3:
                client.items[7]['assignees'] = []
        return original_get(path)

    client.get = get  # type: ignore[method-assign]
    notices: list[monitor.Notice] = []

    assert monitor.reconcile(client, 'r', now=NOW, notices=notices) == ([], [])
    assert notices == []


def test_reconcile_queues_channel_escalation_without_advancing_before_delivery():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL])})
    notices: list[monitor.Notice] = []

    assert monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW, notices=notices) == (
        ['#7: queued channel escalation'],
        [],
    )
    assert notices[0]['kind'] == 'escalation'
    assert monitor._ESCALATED_LABEL not in {label['name'] for label in client.items[7]['labels']}
    assert monitor.finalize_notices(client, 'pydantic/pydantic-ai', [notices[0]]) == ['#7: recorded channel escalation']
    assert monitor._ESCALATED_LABEL in {label['name'] for label in client.items[7]['labels']}


def test_reconcile_retries_preexisting_pending_escalation():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, *monitor._STAGE_LABELS])})
    notices: list[monitor.Notice] = []

    assert monitor.reconcile(client, 'r', now=NOW, notices=notices) == (
        ['#7: queued channel escalation'],
        [],
    )
    assert notices[0]['expected_stage'] == 2
    assert monitor.finalize_notices(client, 'r', [notices[0]]) == ['#7: recorded channel escalation']
    assert monitor._ACTION_LABEL not in {label['name'] for label in client.items[7]['labels']}


def test_reconcile_finishes_a_delivered_escalation_receipt_without_reposting():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL, monitor._DELIVERED_LABEL])})
    client.timelines[7] = [
        label_event(monitor._PINGED_LABEL),
        label_event(monitor._DELIVERED_LABEL),
    ]
    assert monitor.reconcile(client, 'r', now=NOW) == (
        ['#7: finished delivered channel escalation'],
        [],
    )
    assert {label['name'] for label in client.items[7]['labels']} == {monitor._ESCALATED_LABEL}


def test_reconcile_ignores_a_foreign_delivery_receipt():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._DELIVERED_LABEL])})
    client.timelines[7] = [
        label_event(monitor._ACTION_LABEL),
        label_event(monitor._DELIVERED_LABEL, actor='maintainer'),
    ]
    assert monitor.reconcile(client, 'r', now=NOW)[0] == ['#7: queued channel reminder']


def test_terminal_stage_preserves_the_reminder_acknowledgement_boundary():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._ESCALATED_LABEL])})
    client.timelines[7] = [
        label_event(monitor._PINGED_LABEL),
        {
            'event': 'commented',
            'created_at': '2026-07-18T00:00:00Z',
            'actor': {'login': monitor._FALLBACK_OWNER},
            'body': 'I will handle this.',
        },
        label_event(monitor._ESCALATED_LABEL, created_at='2026-07-19T00:00:00Z'),
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: maintainer acknowledged the request'], [])


def test_terminal_stage_rechecks_acknowledgement_after_owner_selection():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._ESCALATED_LABEL])})
    initial = client.last_pages
    timeline_reads = 0

    def last_pages(path: str, *, count: int = 1) -> list[dict[str, Any]]:
        nonlocal timeline_reads
        values = initial(path, count=count)
        if '/timeline' in path:
            timeline_reads += 1
            if timeline_reads == 2:
                return [
                    *values,
                    {
                        'event': 'commented',
                        'created_at': '2026-07-18T00:00:00Z',
                        'actor': {'login': monitor._FALLBACK_OWNER},
                    },
                ]
        return values

    client.last_pages = last_pages  # type: ignore[method-assign]
    notices: list[monitor.Notice] = []

    assert monitor.reconcile(client, 'r', now=NOW, notices=notices) == (
        ['#7: maintainer acknowledged the request'],
        [],
    )
    assert notices == []


def test_terminal_finalize_retry_does_not_repost_the_delivered_escalation():
    client = FakeClient(
        {7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL], assignees=[monitor._FALLBACK_OWNER])}
    )
    client.fail_delete_labels.add(monitor._ACTION_LABEL)

    with pytest.raises(RuntimeError, match='Failed to finalize attention'):
        monitor.finalize_notices(client, 'r', monitor._notice_refs({'items': [notice_ref(7, 1)]}))

    assert {'labels': [monitor._ESCALATED_LABEL, monitor._DELIVERED_LABEL]} in [call[2] for call in client.calls]
    assert {label['name'] for label in client.items[7]['labels']} == {
        monitor._ACTION_LABEL,
        *monitor._STAGE_LABELS,
        monitor._DELIVERED_LABEL,
    }
    client.fail_delete_labels.clear()
    client.timelines[7] = [
        label_event(monitor._ESCALATED_LABEL),
        label_event(monitor._DELIVERED_LABEL),
    ]
    notices: list[monitor.Notice] = []

    assert monitor.reconcile(client, 'r', now=NOW, notices=notices) == (
        ['#7: finished delivered channel escalation'],
        [],
    )
    assert notices == []


@pytest.mark.parametrize(
    ('labels', 'ref'),
    [
        ([monitor._ACTION_LABEL, monitor._PINGED_LABEL], notice_ref(7, 0)),
        ([monitor._ACTION_LABEL], notice_ref(7, 0, transition_id='replacement-transition')),
        ([monitor._ACTION_LABEL], notice_ref(7, 0, recipients=['different-owner'])),
    ],
)
def test_finalize_skips_a_stale_notice(labels: list[str], ref: dict[str, object]):
    client = FakeClient({7: item(7, labels=labels, assignees=[monitor._FALLBACK_OWNER])})

    assert monitor.finalize_notices(client, 'r', monitor._notice_refs({'items': [ref]})) == []
    assert {label['name'] for label in client.items[7]['labels']} == set(labels)


def test_prepare_notices_filters_stale_owners_immediately_before_delivery():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL], assignees=[monitor._FALLBACK_OWNER])})
    refs = monitor._notice_refs({'items': [notice_ref(7, 0)]})

    assert [notice['number'] for notice in monitor.prepare_notices(client, 'r', refs)] == [7]

    client.permissions['bob'] = 'write'
    client.items[7]['assignees'].append({'login': 'bob'})
    assert monitor.prepare_notices(client, 'r', refs) == []

    client.items[7]['assignees'] = []
    assert monitor.prepare_notices(client, 'r', refs) == []


@pytest.mark.parametrize(
    'contents',
    [
        [7],
        {'items': [notice_ref(7, 0)], 'extra': 1},
        {'items': 7},
        {'items': [True]},
        {'items': ['7']},
        {'items': [notice_ref(0, 0)]},
        {'items': [notice_ref(7, 0), notice_ref(7, 0)]},
        {'items': [notice_ref(number, 0) for number in range(1, monitor._RECONCILE_LIMIT + 2)]},
        {'items': [notice_ref(7, 3)]},
        {'items': [notice_ref(7, 0, transition_id='')]},
        {'items': [notice_ref(7, 0, recipients=['bad login'])]},
    ],
)
def test_notice_input_rejects_invalid_shapes(contents: object):
    with pytest.raises(ValueError, match='Notice'):
        monitor._notice_refs(contents)


def test_snapshot_and_decision_batch_limits_are_enforced(tmp_path: Path):
    snapshot = tmp_path / 'snapshot.json'
    write_snapshot(snapshot, [{'number': n, 'updated_at': OLD} for n in range(1, monitor._CANDIDATE_LIMIT + 2)])
    with pytest.raises(ValueError, match='candidate limit'):
        monitor._snapshot_candidates(str(snapshot))

    output = tmp_path / 'output.json'
    write_output(output, [str(n) for n in range(1, monitor._CANDIDATE_LIMIT + 2)])
    with pytest.raises(ValueError, match='too many or duplicate'):
        monitor._parse_decisions(str(output))


def test_notice_output_is_actionable_and_escapes_untrusted_titles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output = tmp_path / 'github-output'
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))

    monitor._write_notices(
        'pydantic/pydantic-ai',
        [
            {
                'number': 7,
                'kind': 'reminder',
                'expected_stage': 0,
                'transition_id': 'event-7',
                'title': 'Handle <unsafe>\n*fake owner* | <!channel>',
                'recipients': ['DouweM'],
            }
        ],
    )

    values = dict(line.split('=', 1) for line in output.read_text(encoding='utf-8').splitlines())
    assert values['has_notices'] == 'true'
    assert json.loads(values['notice_items']) == [notice_ref(7, 0, transition_id='event-7', recipients=['DouweM'])]
    text = json.loads(values['slack_payload'])['text']
    assert text.count('<!channel>') == 1
    assert '#7 Handle &lt;unsafe&gt; fake owner &lt;!channel&gt;' in text
    assert 'owner @DouweM; why: no maintainer has acted for three days' in text
    assert '*Expected action:*' in text
    assert 'If no work is needed, say so briefly' in text
    assert 'Do not remove the attention labels' in text


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


def test_closed_item_completes_and_strips_lifecycle_labels():
    closed = item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL])
    closed['state'] = 'closed'
    client = FakeClient({7: closed})

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: completed after the item was closed'], [])
    assert any(call[0] == 'DELETE' and monitor._ACTION_LABEL in call[1] for call in client.calls)
    assert any(call[0] == 'DELETE' and monitor._PINGED_LABEL in call[1] for call in client.calls)
    assert not any(call[1].endswith('/comments') for call in client.calls)


def test_close_and_reopen_between_runs_retires_the_old_lifecycle():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL])})
    client.timelines[7] = [
        {
            'event': 'labeled',
            'created_at': OLD,
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._PINGED_LABEL},
        },
        {'event': 'closed', 'created_at': '2026-07-18T00:00:00Z', 'actor': {'login': 'contributor'}},
        {'event': 'reopened', 'created_at': '2026-07-18T00:01:00Z', 'actor': {'login': 'contributor'}},
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == (
        ['#7: completed after the item was closed'],
        [],
    )
    assert not {monitor._ACTION_LABEL, monitor._PINGED_LABEL}.intersection(
        {label['name'] for label in client.items[7]['labels']}
    )


def test_cleanup_keeps_active_retry_state_if_stage_cleanup_fails():
    client = FakeClient({7: item(7, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL])})
    client.fail_delete_labels.add(monitor._PINGED_LABEL)

    with pytest.raises(urllib.error.HTTPError):
        monitor._complete(client, 'r', 7, {monitor._ACTION_LABEL, monitor._PINGED_LABEL})

    assert monitor._ACTION_LABEL in {label['name'] for label in client.items[7]['labels']}
    assert not any(
        call[0] == 'DELETE' and urllib.parse.unquote(call[1]).endswith(f'/{monitor._ACTION_LABEL}')
        for call in client.calls
    )


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
    assert sum('queued channel reminder' in line for line in lines) == monitor._ACTIVE_OPEN_LIMIT
    assert lines[-1] == 'additional attention items remain for a later rotated batch'


def test_active_attention_pages_rotate_between_runs():
    client = FakeClient(
        {
            number: item(number, labels=[monitor._ACTION_LABEL])
            for number in range(1, monitor._ACTIVE_OPEN_LIMIT * 2 + 2)
        }
    )

    monitor.reconcile(client, 'r', now=NOW)
    monitor.reconcile(client, 'r', now=NOW + dt.timedelta(hours=6))

    searches = [
        path
        for method, path, _ in client.calls
        if method == 'GET' and path.startswith('/search/issues?') and f'per_page={monitor._ACTIVE_OPEN_LIMIT}&' in path
    ]
    assert len(searches) == 2
    assert searches[0] != searches[1]


def test_one_item_failure_does_not_block_later_items():
    client = FakeClient(
        {
            1: item(1, labels=[monitor._ACTION_LABEL]),
            2: item(2, labels=[monitor._ACTION_LABEL]),
        }
    )
    client.fail_get.add(1)

    lines, failures = monitor.reconcile(client, 'pydantic/pydantic-ai', now=NOW)

    assert lines == ['#2: queued channel reminder']
    assert failures and failures[0].startswith('#1: HTTPError')
    assert not any(call[0] == 'POST' and call[1].endswith('/issues/2/comments') for call in client.calls)


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

    assert lines == ['#2: queued channel reminder']
    assert failures and failures[0].startswith('#1: ValueError')
    assert not any(call[0] == 'POST' and call[1].endswith('/issues/2/comments') for call in client.calls)


def test_one_item_failure_still_queues_other_notices():
    client = FakeClient(
        {
            1: item(1, labels=[monitor._ACTION_LABEL]),
            2: item(2, labels=[monitor._ACTION_LABEL, monitor._PINGED_LABEL]),
        }
    )
    client.fail_get.add(1)
    notices: list[monitor.Notice] = []

    lines, failures = monitor.reconcile(client, 'r', now=NOW, notices=notices)

    assert lines == ['#2: queued channel escalation']
    assert [notice['number'] for notice in notices] == [2]
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

    assert monitor.reconcile(client, 'r', now=NOW) == (['#7: queued channel reminder'], [])


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
            'event': 'labeled',
            'created_at': '2026-07-19T00:00:00Z',
            'actor': {'login': 'github-actions[bot]'},
            'label': {'name': monitor._PINGED_LABEL},
        },
    ]

    assert monitor._transition(client.last_pages('/repos/r/issues/7/events'), 1)[1]['id'] == 'event-2'
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
        {'event': 'commented', 'created_at': OLD, 'actor': {'login': 'contributor'}},
    ]

    assert monitor.reconcile(client, 'r', now=NOW) == (
        ['#7: restored attention eligibility after new activity'],
        [],
    )
    assert any(call[0] == 'DELETE' and monitor._ESCALATED_LABEL in call[1] for call in client.calls)
    assert any(
        call[0] == 'GET' and monitor._ESCALATED_LABEL in urllib.parse.unquote(call[1]) and 'direction=desc' in call[1]
        for call in client.calls
    )


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


def test_sweep_clears_escalation_marker_from_closed_items():
    closed = item(7, labels=[monitor._ESCALATED_LABEL])
    closed['state'] = 'closed'
    client = FakeClient({7: closed})

    assert monitor.reconcile(client, 'r', now=NOW) == (
        ['#7: cleared escalation marker after the item was closed'],
        [],
    )
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


def test_compiled_lock_keeps_agent_read_only_and_stable_artifact_name():
    # Actions runs the compiled .lock.yml, not the .md; nothing else pins the
    # two together, so guard the load-bearing strings against a bad recompile.
    lock = Path(__file__).parent.parent / 'workflows' / 'pydantic-ai-attention-triage.lock.yml'
    text = lock.read_text()
    jobs = yaml.safe_load(text)['jobs']
    agent_permissions = jobs['agent']['permissions']
    decision_permissions = jobs['record_attention_decision']['permissions']

    assert 'GH_AW_FAILURE_REPORT_AS_ISSUE: "false"' in text
    assert agent_permissions['pull-requests'] == 'read'
    assert set(agent_permissions.values()) == {'read'}
    assert decision_permissions['pull-requests'] == 'write'
    assert 'name: attention-candidates-${{ github.run_id }}' in text
    # The run_attempt suffix must stay gone: "Re-run failed jobs" bumps the
    # attempt number, but only the original run_id upload exists.
    assert 'name: attention-candidates-${{ github.run_id }}-' not in text


def test_operations_workflow_routes_all_notices_to_the_triage_channel():
    workflow = Path(__file__).parent.parent / 'workflows' / 'issue-pr-attention-monitor.yml'
    text = workflow.read_text()

    assert 'PYDANTIC_AI_TRIAGE_SLACK_WEBHOOK_URL' in text
    assert 'issue_pr_attention_monitor.py finalize' in text
    assert 'permissions: {}' in text
    assert 'ATTENTION_NOTICES' in text
    assert 'issue_pr_attention_monitor.py prepare' in text
    assert 'needs.notify.outputs.notice_items' in text
    assert 'steps.prepare.outputs.slack_payload' in text
    assert 'Post actionable attention digest to the triage channel' in text


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
