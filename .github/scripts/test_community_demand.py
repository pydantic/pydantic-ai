from __future__ import annotations

import json
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import community_demand

REPO = 'pydantic/pydantic-ai'


def issue(number: int, **overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        'number': number,
        'state': 'open',
        'title': 'Support streaming for provider X',
        'body': 'Reported behavior and a reproduction.',
        'created_at': '2026-01-01T00:00:00Z',
        'updated_at': '2026-08-20T00:00:00Z',
        'labels': [{'name': 'feature'}],
        'assignees': [],
        'comments': 4,
        'reactions': {'total_count': 2},
    }
    value.update(overrides)
    return value


class FakeClient(community_demand.attention.GitHubClient):
    def __init__(self, issues: dict[int, dict[str, Any]], search: list[int]) -> None:
        super().__init__('token')
        self.issues = issues
        self.search = search
        self.comments: dict[int, list[dict[str, Any]]] = {}
        self.labeled: list[tuple[int, list[str]]] = []
        self.queries: list[str] = []

    def get(self, path: str) -> Any:
        if path.startswith('/search/issues'):
            self.queries.append(path)
            query = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
            per_page = int(query.get('per_page', ['30'])[0])
            page = int(query.get('page', ['1'])[0])
            start = (page - 1) * per_page
            return {
                'total_count': len(self.search),
                'items': [{'number': number} for number in self.search[start : start + per_page]],
            }
        if '/labels/' in path:
            return {}
        number = int(path.rsplit('/', 1)[1])
        return self.issues[number]

    def last_pages(self, path: str, *, count: int = 1) -> list[dict[str, Any]]:
        number = int(path.split('/issues/')[1].split('/')[0])
        return self.comments.get(number, [])

    def post(self, path: str, payload: Any) -> Any:
        if path.endswith('/labels') and '/issues/' in path:
            number = int(path.split('/issues/')[1].split('/')[0])
            names = list(payload['labels'])
            self.labeled.append((number, names))
            self.issues[number]['labels'].extend({'name': name} for name in names)
            return self.issues[number]
        return {}


def write_verdicts(path: Path, entries: list[dict[str, str]]) -> str:
    payload = {'items': [{'type': 'record_community_verdict', **entry} for entry in entries]}
    path.write_text(json.dumps(payload), encoding='utf-8')
    return str(path)


def test_snapshot_search_excludes_already_decided_issues(tmp_path: Path):
    # The candidate pool must never contain issues a human or the triage
    # pipeline already decided on: prioritized, labeled, assigned, or closed.
    client = FakeClient({7: issue(7)}, search=[7])

    now = community_demand.dt.datetime.now(community_demand.dt.timezone.utc)
    community_demand.write_snapshot(client, REPO, str(tmp_path / 's.json'), now=now)

    query = urllib.parse.parse_qs(urllib.parse.urlparse(client.queries[-1]).query)['q'][0]
    assert 'no:assignee' in query
    for excluded in ('community-backed', 'p:1-highest', 'p:2-high', 'unplanned', 'duplicate'):
        assert f'-label:"{excluded}"' in query


def test_snapshot_sheds_trailing_candidates_instead_of_failing(tmp_path: Path):
    # Without shedding, a fat backlog would fail the sweep every week on the
    # same size limit, with no self-healing.
    now = community_demand.dt.datetime.now(community_demand.dt.timezone.utc)
    numbers = list(range(1, 9))
    client = FakeClient({number: issue(number) for number in numbers}, search=numbers)
    fat_comment = {'user': {'login': 'reporter'}, 'author_association': 'NONE', 'created_at': '', 'body': 'x' * 700}
    for number in numbers:
        client.comments[number] = [dict(fat_comment) for _ in range(20)]

    community_demand.write_snapshot(client, REPO, str(tmp_path / 's.json'), now=now)

    written = json.loads((tmp_path / 's.json').read_text(encoding='utf-8'))['candidates']
    assert 0 < len(written) < len(numbers)


def test_snapshot_revalidates_each_candidate_against_live_state(tmp_path: Path):
    now = community_demand.dt.datetime.now(community_demand.dt.timezone.utc)
    recent = (now - community_demand.dt.timedelta(days=3)).strftime('%Y-%m-%dT%H:%M:%SZ')
    client = FakeClient(
        {
            7: issue(7),
            8: issue(8, assignees=[{'login': 'alice'}]),
            9: issue(9, labels=[{'name': 'p:2-high'}]),
            10: issue(10, created_at=recent),
            11: issue(11, comments=1, reactions={'total_count': 0}),
        },
        search=[7, 8, 9, 10, 11],
    )
    client.comments[7] = [
        {
            'user': {'login': 'reporter'},
            'author_association': 'NONE',
            'created_at': recent,
            'body': 'Still hitting this.',
        }
    ]

    community_demand.write_snapshot(client, REPO, str(tmp_path / 's.json'), now=now)

    snapshot = json.loads((tmp_path / 's.json').read_text(encoding='utf-8'))
    assert [candidate['number'] for candidate in snapshot['candidates']] == [7]
    assert snapshot['candidates'][0]['recent_comments'][0]['body'] == 'Still hitting this.'


def test_apply_labels_only_high_confidence_genuine_verdicts(tmp_path: Path):
    now = community_demand.dt.datetime.now(community_demand.dt.timezone.utc)
    client = FakeClient({7: issue(7), 8: issue(8), 9: issue(9)}, search=[7, 8, 9])
    snapshot_path = tmp_path / 's.json'
    community_demand.write_snapshot(client, REPO, str(snapshot_path), now=now)
    output = write_verdicts(
        tmp_path / 'out.json',
        [
            {'item_number': '7', 'verdict': 'genuine', 'confidence': 'high'},
            {'item_number': '8', 'verdict': 'genuine', 'confidence': 'medium'},
            {'item_number': '9', 'verdict': 'artificial', 'confidence': 'high'},
        ],
    )

    lines = community_demand.apply_verdicts(client, REPO, output, str(snapshot_path))

    assert client.labeled == [(7, ['community-backed'])]
    assert '#7: marked as genuine community demand' in lines


def test_apply_rejects_output_that_does_not_cover_every_candidate(tmp_path: Path):
    now = community_demand.dt.datetime.now(community_demand.dt.timezone.utc)
    client = FakeClient({7: issue(7), 8: issue(8)}, search=[7, 8])
    snapshot_path = tmp_path / 's.json'
    community_demand.write_snapshot(client, REPO, str(snapshot_path), now=now)
    output = write_verdicts(tmp_path / 'out.json', [{'item_number': '7', 'verdict': 'genuine', 'confidence': 'high'}])

    with pytest.raises(ValueError, match='every snapshot candidate exactly once'):
        community_demand.apply_verdicts(client, REPO, output, str(snapshot_path))
    assert client.labeled == []


@pytest.mark.parametrize(
    ('entry', 'message'),
    [
        ({'item_number': '7; echo pwned', 'verdict': 'genuine', 'confidence': 'high'}, 'positive decimal'),
        # The boundary is ASCII-only by design: '７' (a fullwidth digit) passes
        # str.isdecimal() but is not something the agent is ever asked to write.
        ({'item_number': '７', 'verdict': 'genuine', 'confidence': 'high'}, 'positive decimal'),
        ({'item_number': '7', 'verdict': 'attacker', 'confidence': 'high'}, r"verdict\s+Input should be 'genuine'"),
        ({'item_number': '7', 'verdict': 'genuine', 'confidence': 'certain'}, r"confidence\s+Input should be 'high'"),
    ],
)
def test_apply_rejects_hostile_verdict_values_before_any_label_write(
    tmp_path: Path, entry: dict[str, str], message: str
):
    now = community_demand.dt.datetime.now(community_demand.dt.timezone.utc)
    client = FakeClient({7: issue(7)}, search=[7])
    snapshot_path = tmp_path / 's.json'
    community_demand.write_snapshot(client, REPO, str(snapshot_path), now=now)
    output = write_verdicts(tmp_path / 'out.json', [entry])

    with pytest.raises(ValueError, match=message):
        community_demand.apply_verdicts(client, REPO, output, str(snapshot_path))
    assert client.labeled == []


def test_apply_skips_an_issue_that_changed_after_classification(tmp_path: Path):
    now = community_demand.dt.datetime.now(community_demand.dt.timezone.utc)
    client = FakeClient({7: issue(7)}, search=[7])
    snapshot_path = tmp_path / 's.json'
    community_demand.write_snapshot(client, REPO, str(snapshot_path), now=now)
    # A maintainer assigned it between classification and apply.
    client.issues[7]['assignees'] = [{'login': 'alice'}]
    output = write_verdicts(tmp_path / 'out.json', [{'item_number': '7', 'verdict': 'genuine', 'confidence': 'high'}])

    lines = community_demand.apply_verdicts(client, REPO, output, str(snapshot_path))

    assert client.labeled == []
    assert lines == ['#7: skipped because the issue changed after classification']


def test_workflow_verdict_capacity_covers_the_candidate_limit():
    # A safe-output `max` below the snapshot limit silently drops verdicts and
    # fails the exactly-once check on every busy run (the attention workflow
    # shipped that bug once).
    source = Path(__file__).parent.parent / 'workflows' / 'pydantic-ai-community-demand.md'
    frontmatter = yaml.safe_load(re.split(r'^---$', source.read_text(encoding='utf-8'), flags=re.M)[1])

    job = frontmatter['safe-outputs']['jobs']['record-community-verdict']
    assert job['max'] >= community_demand._CANDIDATE_LIMIT  # pyright: ignore[reportPrivateUsage]
