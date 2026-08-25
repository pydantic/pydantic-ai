from __future__ import annotations

import json
import sys
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))
import semantic_owner_router as router

UPDATED = '2026-08-25T00:00:00Z'
MENTIONS = {
    'adtyavrdhn': '<@UADITYA>',
    'dsfaccini': '<@UDAVID>',
    'DouweM': '<@UDOUWE>',
    'mpfaffenberger': '<@UMIKE>',
}
STR_LIST = TypeAdapter(list[str])


def item(
    number: int,
    *,
    assignees: list[str] | None = None,
    pull_request: bool = False,
    title: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        'number': number,
        'state': 'open',
        'updated_at': UPDATED,
        'title': title or f'Item {number}',
        'body': 'Please route this work.',
        'user': {'login': 'contributor'},
        'labels': [{'name': 'bug'}],
        'assignees': [{'login': login} for login in assignees or []],
    }
    if pull_request:
        value['pull_request'] = {'url': f'https://api.github.com/pulls/{number}'}
    return value


class FakeClient(router.attention.GitHubClient):
    def __init__(self, values: dict[int, dict[str, Any]]) -> None:
        super().__init__('token')
        self.items = values
        self.calls: list[tuple[str, str, object | None]] = []
        self.files: dict[int, list[str]] = {}

    def get(self, path: str) -> Any:
        self.calls.append(('GET', path, None))
        if path.endswith('/permission'):
            login = urllib.parse.unquote(path.split('/collaborators/')[1].removesuffix('/permission'))
            permission = 'write' if login in {'adtyavrdhn', 'dsfaccini', 'DouweM', 'mpfaffenberger'} else 'none'
            return {'permission': permission}
        if path.startswith('/search/issues?'):
            return {'total_count': len(self.items), 'items': list(self.items.values())[:10]}
        if '/pulls/' in path and '/files?' in path:
            number = int(path.split('/pulls/')[1].split('/')[0])
            return [{'filename': filename} for filename in self.files.get(number, [])]
        if '/issues/' in path:
            return self.items[int(path.split('/issues/')[1].split('/')[0])]
        raise AssertionError(path)

    def post(self, path: str, payload: Mapping[str, object]) -> Any:
        self.calls.append(('POST', path, payload))
        assert path.endswith('/assignees')
        number = int(path.split('/issues/')[1].split('/')[0])
        existing = [str(value['login']) for value in self.items[number]['assignees']]
        requested = STR_LIST.validate_python(payload.get('assignees'))
        self.items[number]['assignees'] = [{'login': login} for login in dict.fromkeys([*existing, *requested])]
        return self.items[number]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding='utf-8')


def test_event_snapshot_routes_only_the_triggering_pull_request(tmp_path: Path):
    client = FakeClient({7: item(7, pull_request=True), 8: item(8)})
    client.files[7] = ['pydantic_ai_harness/compaction/_summarizing_compaction.py']
    event = tmp_path / 'event.json'
    write_json(event, {'action': 'opened', 'pull_request': {'number': 7}})

    snapshot = router.build_snapshot(client, 'pydantic/pydantic-ai-harness', event_path=str(event))

    assert snapshot['candidates'] == [
        {
            'number': 7,
            'kind': 'pull_request',
            'title': 'Item 7',
            'body': 'Please route this work.',
            'author': 'contributor',
            'updated_at': UPDATED,
            'labels': ['bug'],
            'files': ['pydantic_ai_harness/compaction/_summarizing_compaction.py'],
        }
    ]


def test_safety_snapshot_uses_oldest_unassigned_query_and_preserves_existing_maintainer():
    client = FakeClient({7: item(7), 8: item(8, assignees=['dsfaccini'])})

    snapshot = router.build_snapshot(client, 'pydantic/pydantic-ai')

    assert snapshot == {
        'candidates': [
            {
                'number': 7,
                'kind': 'issue',
                'title': 'Item 7',
                'body': 'Please route this work.',
                'author': 'contributor',
                'updated_at': UPDATED,
                'labels': ['bug'],
                'files': [],
            }
        ]
    }
    search_path = next(path for method, path, _ in client.calls if method == 'GET' and path.startswith('/search/'))
    query = urllib.parse.parse_qs(urllib.parse.urlparse(search_path).query)['q'][0]
    assert query == 'repo:pydantic/pydantic-ai is:open no:assignee'
    assert 'sort=created&order=asc' in search_path


def test_apply_routes_assigns_fixed_owner_and_builds_fixed_notice(tmp_path: Path):
    client = FakeClient({7: item(7, title='Stream <events>')})
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_json(snapshot, {'candidates': [{'number': 7, 'updated_at': UPDATED}]})
    write_json(
        output,
        {'items': [{'type': 'route_maintainer_owner', 'item_number': '7', 'route': 'aditya-streaming-runtime'}]},
    )

    lines, notices = router.apply_routes(client, 'pydantic/pydantic-ai', str(output), str(snapshot))

    assert lines == ['#7: routed to @adtyavrdhn for streaming, cancellation, UI protocols, or CodeMode runtime']
    assert [value['login'] for value in client.items[7]['assignees']] == ['adtyavrdhn']
    assert notices == [
        {
            'number': 7,
            'title': 'Stream <events>',
            'owner': 'adtyavrdhn',
            'basis': 'streaming, cancellation, UI protocols, or CodeMode runtime',
        }
    ]


def test_apply_routes_preserves_a_maintainer_assigned_after_snapshot(tmp_path: Path):
    client = FakeClient({7: item(7, assignees=['DouweM'])})
    snapshot = tmp_path / 'snapshot.json'
    output = tmp_path / 'output.json'
    write_json(snapshot, {'candidates': [{'number': 7, 'updated_at': UPDATED}]})
    write_json(
        output,
        {'items': [{'type': 'route_maintainer_owner', 'item_number': '7', 'route': 'mike-tools-harness'}]},
    )

    lines, notices = router.apply_routes(client, 'r', str(output), str(snapshot))

    assert lines == ['#7: kept existing maintainer owner @DouweM']
    assert notices == []
    assert not any(method == 'POST' for method, _, _ in client.calls)


@pytest.mark.parametrize(
    ('value', 'message'),
    [
        (
            {'items': [{'type': 'route_maintainer_owner', 'item_number': '7', 'route': 'invented'}]},
            'Invalid semantic route',
        ),
        (
            {
                'items': [
                    {'type': 'route_maintainer_owner', 'item_number': '7', 'route': 'mike-tools-harness'},
                    {'type': 'route_maintainer_owner', 'item_number': '7', 'route': 'mike-tools-harness'},
                ]
            },
            'duplicate',
        ),
    ],
)
def test_parse_decisions_rejects_routes_outside_the_fixed_policy(tmp_path: Path, value: object, message: str):
    path = tmp_path / 'output.json'
    write_json(path, value)

    with pytest.raises(ValueError, match=message):
        router.parse_decisions(str(path))


def test_slack_mentions_require_every_owner_and_real_member_syntax():
    assert router.parse_slack_mentions(json.dumps(MENTIONS)) == MENTIONS

    with pytest.raises(ValueError, match='every semantic owner'):
        router.parse_slack_mentions(json.dumps({**MENTIONS, 'DouweM': '@DouweM'}))


def test_notification_is_one_escaped_assignment_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output = tmp_path / 'github-output'
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))

    router.write_notifications(
        'pydantic/pydantic-ai',
        [
            {
                'number': 7,
                'title': 'Stream <events>',
                'owner': 'adtyavrdhn',
                'basis': 'streaming, cancellation, UI protocols, or CodeMode runtime',
            }
        ],
        MENTIONS,
    )

    values = dict(line.split('=', 1) for line in output.read_text().splitlines())
    assert values['has_assignments'] == 'true'
    assert json.loads(values['slack_payload']) == {
        'text': ':label: Semantic owner routing for pydantic/pydantic-ai\n'
        '• <https://github.com/pydantic/pydantic-ai/issues/7|#7 Stream &lt;events&gt;> → <@UADITYA>\n'
        '      why: streaming, cancellation, UI protocols, or CodeMode runtime'
    }


def test_workflow_keeps_events_safety_sweep_and_writes_outside_the_agent():
    root = Path(__file__).parents[1]
    source = root / 'workflows/pydantic-ai-owner-routing.md'
    frontmatter = yaml.safe_load(source.read_text().split('---', 2)[1])
    text = source.read_text()

    assert frontmatter[True]['issues']['types'] == ['opened', 'reopened']
    assert frontmatter[True]['pull_request_target']['types'] == ['opened', 'reopened', 'ready_for_review']
    assert frontmatter[True]['schedule'] == [{'cron': '25 */6 * * *'}]
    assert frontmatter['tools'] == {'bash': [], 'github': False}
    job = frontmatter['safe-outputs']['jobs']['route-maintainer-owner']
    assert job['permissions'] == {'contents': 'read', 'issues': 'write', 'pull-requests': 'write'}
    assert '${{ vars.PYDANTIC_AI_TRIAGE_SLACK_MENTIONS }}' in text
    assert '${{ secrets.PYDANTIC_AI_TRIAGE_SLACK_WEBHOOK_URL }}' in text
    assert 'repository: ${{ job.workflow_repository }}' in text
    assert 'ref: ${{ job.workflow_sha }}' in text


def test_compiled_owner_routing_lock_exists_and_matches_security_contract():
    lock = Path(__file__).parents[1] / 'workflows/pydantic-ai-owner-routing.lock.yml'
    if not lock.exists():
        pytest.skip('workflow lock is generated after the source tests pass')
    text = lock.read_text()
    assert 'pull_request_target:' in text
    assert 'workflow_call:' in text
    assert 'route_maintainer_owner' in text
    assert 'semantic_owner_router.py apply' in text
    assert 'slackapi/slack-github-action@45a88b9581bfab2566dc881e2cd66d334e621e2c' in text
